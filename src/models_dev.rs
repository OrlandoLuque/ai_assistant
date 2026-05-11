//! models.dev model registry (V104.9)
//!
//! Loads, caches, and queries a typed model catalog. Source format is
//! the [models.dev](https://models.dev) `api.json` schema (or any
//! compatible JSON), but the actual HTTP fetch is left to the caller —
//! this module only handles parse + cache + lookup so it stays unit-testable
//! offline.
//!
//! Typical wiring:
//!
//! ```ignore
//! let cfg = ModelsDevConfig {
//!     cache_path: Some(default_cache_path()),
//!     ..Default::default()
//! };
//! let json: String = my_http_client.get("https://models.dev/api.json")?;
//! let reg = ModelRegistry::from_json(&json)?;
//! save_cache(&reg, &cfg)?;
//! let opus = reg.lookup("claude-opus-4-7").unwrap();
//! ```
//!
//! `load_cache` returns `None` if the cached file is older than
//! `cache_ttl` so callers can decide to refresh.
//!
//! ## Schema tolerance
//!
//! models.dev evolves; we deserialize into our own typed struct using
//! `#[serde(default)]` and ignore unknown fields. Missing required fields
//! degrade gracefully (e.g. unknown context window → `None`, not error).

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelsDevConfig {
    /// Where to store / read the cache. None = no on-disk cache.
    pub cache_path: Option<PathBuf>,
    /// How long a cached file is considered fresh.
    pub cache_ttl: Duration,
    /// Maximum bytes accepted from any source (parse + cache load both
    /// guard with this). Default 4 MiB.
    pub max_payload_bytes: u64,
}

impl Default for ModelsDevConfig {
    fn default() -> Self {
        Self {
            cache_path: None,
            cache_ttl: Duration::from_secs(24 * 60 * 60),
            max_payload_bytes: 4 * 1024 * 1024,
        }
    }
}

// ============================================================================
// Schema
// ============================================================================

/// A model capability flag — orthogonal to provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Streaming,
    ToolUse,
    Vision,
    Audio,
    Reasoning,
    Embeddings,
    JsonMode,
    PromptCaching,
}

/// Pricing in USD per million tokens (mirrors models.dev's units).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
    #[serde(default)]
    pub input_per_mtok: Option<f64>,
    #[serde(default)]
    pub output_per_mtok: Option<f64>,
    #[serde(default)]
    pub cache_read_per_mtok: Option<f64>,
    #[serde(default)]
    pub cache_write_per_mtok: Option<f64>,
}

/// A single model entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Canonical id used to invoke the model (e.g. `"claude-opus-4-7"`).
    pub id: String,
    /// Optional display name.
    #[serde(default)]
    pub name: Option<String>,
    /// Provider key (e.g. `"anthropic"`, `"openai"`, `"ollama"`).
    pub provider: String,
    /// Max context window, in tokens.
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Max single-response output, in tokens.
    #[serde(default)]
    pub max_output: Option<u32>,
    /// Capability flags this model supports.
    #[serde(default)]
    pub capabilities: Vec<ModelCapability>,
    /// Pricing.
    #[serde(default)]
    pub pricing: ModelPricing,
    /// Optional "knowledge cutoff" date, ISO-8601 (`YYYY-MM-DD`).
    #[serde(default)]
    pub knowledge_cutoff: Option<String>,
    /// Optional release date, ISO-8601.
    #[serde(default)]
    pub release_date: Option<String>,
    /// Optional aliases (alternate ids that resolve to this entry).
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// In-memory registry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<ModelMetadata>,
    /// When this snapshot was fetched (monotonic across cache reloads).
    #[serde(default)]
    pub fetched_at: Option<SystemTime>,
    /// Free-form source URL (for trace/diagnostics).
    #[serde(default)]
    pub source: Option<String>,
}

impl ModelRegistry {
    /// Parse from a JSON document. Accepts both `{"models": [...]}` and a
    /// bare array `[...]`. Sets `fetched_at` to `now`.
    pub fn from_json(text: &str) -> Result<Self, ModelsDevError> {
        // Bare-array tolerance.
        let trimmed = text.trim_start();
        let value: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| ModelsDevError::Parse(e.to_string()))?;

        let models_val = match value {
            serde_json::Value::Array(_) => value,
            serde_json::Value::Object(ref obj) => obj
                .get("models")
                .cloned()
                .unwrap_or(serde_json::Value::Array(Vec::new())),
            other => {
                return Err(ModelsDevError::Parse(format!(
                    "unexpected top-level JSON kind: {}",
                    type_name_of(&other)
                )));
            }
        };

        let models: Vec<ModelMetadata> =
            serde_json::from_value(models_val).map_err(|e| ModelsDevError::Parse(e.to_string()))?;
        // Reject empty id.
        for m in &models {
            if m.id.trim().is_empty() {
                return Err(ModelsDevError::Parse("model with empty id".into()));
            }
        }

        Ok(Self {
            models,
            fetched_at: Some(SystemTime::now()),
            source: None,
        })
    }

    /// Look up by canonical id or alias (case-insensitive).
    pub fn lookup(&self, id: &str) -> Option<&ModelMetadata> {
        let lower = id.to_lowercase();
        self.models.iter().find(|m| {
            m.id.to_lowercase() == lower || m.aliases.iter().any(|a| a.to_lowercase() == lower)
        })
    }

    /// All models from a given provider.
    pub fn by_provider(&self, provider: &str) -> Vec<&ModelMetadata> {
        let lower = provider.to_lowercase();
        self.models
            .iter()
            .filter(|m| m.provider.to_lowercase() == lower)
            .collect()
    }

    /// All models supporting *every* listed capability.
    pub fn supporting(&self, caps: &[ModelCapability]) -> Vec<&ModelMetadata> {
        self.models
            .iter()
            .filter(|m| caps.iter().all(|c| m.capabilities.contains(c)))
            .collect()
    }

    /// Group models by provider, sorted by id.
    pub fn grouped_by_provider(&self) -> BTreeMap<String, Vec<&ModelMetadata>> {
        let mut map: BTreeMap<String, Vec<&ModelMetadata>> = BTreeMap::new();
        for m in &self.models {
            map.entry(m.provider.clone()).or_default().push(m);
        }
        for v in map.values_mut() {
            v.sort_by(|a, b| a.id.cmp(&b.id));
        }
        map
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

// ============================================================================
// Cache I/O
// ============================================================================

/// Load the cached registry if it exists and is fresh enough per
/// `cfg.cache_ttl`. Returns Ok(None) for missing/stale; Err for I/O or
/// parse problems on a present file.
pub fn load_cache(cfg: &ModelsDevConfig) -> Result<Option<ModelRegistry>, ModelsDevError> {
    let Some(path) = cfg.cache_path.as_ref() else {
        return Ok(None);
    };
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ModelsDevError::Io(e.to_string())),
    };
    if meta.len() > cfg.max_payload_bytes {
        return Err(ModelsDevError::TooLarge {
            size: meta.len(),
            limit: cfg.max_payload_bytes,
        });
    }
    let mtime = meta
        .modified()
        .map_err(|e| ModelsDevError::Io(e.to_string()))?;
    if let Ok(age) = SystemTime::now().duration_since(mtime) {
        if age > cfg.cache_ttl {
            return Ok(None);
        }
    }
    let text = fs::read_to_string(path).map_err(|e| ModelsDevError::Io(e.to_string()))?;
    let mut reg: ModelRegistry =
        serde_json::from_str(&text).map_err(|e| ModelsDevError::Parse(e.to_string()))?;
    if reg.fetched_at.is_none() {
        reg.fetched_at = Some(mtime);
    }
    Ok(Some(reg))
}

/// Atomically save the registry to `cfg.cache_path` (if any). Writes to
/// a temp file then renames. No-op when no cache_path is configured.
pub fn save_cache(reg: &ModelRegistry, cfg: &ModelsDevConfig) -> Result<(), ModelsDevError> {
    let Some(path) = cfg.cache_path.as_ref() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ModelsDevError::Io(e.to_string()))?;
    }
    let body =
        serde_json::to_string_pretty(reg).map_err(|e| ModelsDevError::Parse(e.to_string()))?;
    if body.len() as u64 > cfg.max_payload_bytes {
        return Err(ModelsDevError::TooLarge {
            size: body.len() as u64,
            limit: cfg.max_payload_bytes,
        });
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, body.as_bytes()).map_err(|e| ModelsDevError::Io(e.to_string()))?;
    fs::rename(&tmp, path).map_err(|e| ModelsDevError::Io(e.to_string()))?;
    Ok(())
}

/// Default cache location: `<config-dir>/ai_assistant/models_dev_cache.json`.
pub fn default_cache_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("ai_assistant").join("models_dev_cache.json"))
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum ModelsDevError {
    Io(String),
    Parse(String),
    TooLarge { size: u64, limit: u64 },
}

impl std::fmt::Display for ModelsDevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "I/O error: {}", s),
            Self::Parse(s) => write!(f, "parse error: {}", s),
            Self::TooLarge { size, limit } => {
                write!(f, "payload too large: {} > {} bytes", size, limit)
            }
        }
    }
}

impl std::error::Error for ModelsDevError {}

// ============================================================================
// Helpers
// ============================================================================

fn type_name_of(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA").ok().map(PathBuf::from)
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Application Support"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".config"))
            })
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

// ============================================================================
// Bridge to `crate::models` (models.rs)
// ============================================================================
//
// `models_dev` mirrors the public models.dev schema, while
// `crate::models::ModelInfo` is the in-crate type used by the rest of the
// pipeline (cost, routing, capability gating, etc.). The bridge below lets
// callers fetch + cache a models.dev catalog and feed it into the
// in-crate registry without writing the glue themselves.

/// Map a provider key (as it appears in `ModelMetadata::provider`) to
/// `AiProvider`. Case-insensitive; unknown keys fall through to
/// `OpenAICompatible { base_url: "" }`.
pub fn provider_from_key(key: &str) -> crate::config::AiProvider {
    use crate::config::AiProvider;
    match key.to_lowercase().as_str() {
        "ollama" => AiProvider::Ollama,
        "lmstudio" | "lm_studio" | "lm-studio" => AiProvider::LMStudio,
        "textgenwebui" | "text-gen-webui" | "oobabooga" => AiProvider::TextGenWebUI,
        "koboldcpp" | "kobold" => AiProvider::KoboldCpp,
        "localai" => AiProvider::LocalAI,
        "llamacpp" | "llama.cpp" | "llama-cpp" | "llama_cpp" => AiProvider::LlamaCpp,
        "vllm" => AiProvider::VLLM,
        "openai" => AiProvider::OpenAI,
        "anthropic" => AiProvider::Anthropic,
        "gemini" | "google" => AiProvider::Gemini,
        "groq" => AiProvider::Groq,
        "together" | "together_ai" | "togetherai" => AiProvider::Together,
        "fireworks" | "fireworks_ai" => AiProvider::Fireworks,
        "deepseek" => AiProvider::DeepSeek,
        "mistral" => AiProvider::Mistral,
        "perplexity" => AiProvider::Perplexity,
        "openrouter" => AiProvider::OpenRouter,
        _ => AiProvider::OpenAICompatible {
            base_url: String::new(),
        },
    }
}

impl ModelMetadata {
    /// Convert this entry to the in-crate `models::ModelInfo` type with full
    /// capability mapping (vision / tool_use / json_mode / streaming,
    /// pricing in USD/Mtok, context window, max output, knowledge cutoff).
    pub fn to_model_info(&self) -> crate::models::ModelInfo {
        use crate::models::{ModelCapabilityInfo, ModelInfo};
        let caps = ModelCapabilityInfo {
            context_window: self.context_window.map(|v| v as usize),
            supports_vision: self.capabilities.contains(&ModelCapability::Vision),
            supports_tool_calling: self.capabilities.contains(&ModelCapability::ToolUse),
            supports_json_mode: self.capabilities.contains(&ModelCapability::JsonMode),
            supports_streaming: self.capabilities.contains(&ModelCapability::Streaming),
            input_cost_per_million: self.pricing.input_per_mtok,
            output_cost_per_million: self.pricing.output_per_mtok,
            max_output_tokens: self.max_output.map(|v| v as usize),
            knowledge_cutoff: self.knowledge_cutoff.clone(),
        };
        ModelInfo::new(self.id.clone(), provider_from_key(&self.provider)).with_capabilities(caps)
    }
}

impl ModelRegistry {
    /// Convert every entry to `models::ModelInfo`. Order follows the
    /// catalog as-loaded.
    pub fn to_model_infos(&self) -> Vec<crate::models::ModelInfo> {
        self.models.iter().map(|m| m.to_model_info()).collect()
    }
}

impl crate::models::ModelRegistry {
    /// Register every model from a `models_dev::ModelRegistry` into this
    /// in-crate registry. Existing entries with the same name are
    /// overwritten (last-write-wins, matching `register()` semantics).
    pub fn extend_from_models_dev(&mut self, src: &ModelRegistry) {
        for m in &src.models {
            self.register(m.to_model_info());
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
          "models": [
            {
              "id": "claude-opus-4-7",
              "name": "Claude Opus 4.7",
              "provider": "anthropic",
              "context_window": 200000,
              "max_output": 8192,
              "capabilities": ["streaming", "tool_use", "vision", "prompt_caching"],
              "pricing": {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
              "knowledge_cutoff": "2026-01-01",
              "aliases": ["opus-4.7"]
            },
            {
              "id": "gpt-4o",
              "provider": "openai",
              "context_window": 128000,
              "capabilities": ["streaming", "tool_use", "vision", "json_mode"]
            },
            {
              "id": "llama3.3:70b",
              "provider": "ollama",
              "capabilities": ["streaming"]
            }
          ]
        }"#
    }

    #[test]
    fn parse_sample() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        assert_eq!(reg.len(), 3);
        let opus = reg.lookup("claude-opus-4-7").unwrap();
        assert_eq!(opus.context_window, Some(200000));
        assert!(opus.capabilities.contains(&ModelCapability::PromptCaching));
        assert_eq!(opus.pricing.input_per_mtok, Some(15.0));
    }

    #[test]
    fn lookup_resolves_alias_case_insensitive() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        let by_alias = reg.lookup("OPUS-4.7").unwrap();
        assert_eq!(by_alias.id, "claude-opus-4-7");
    }

    #[test]
    fn by_provider_returns_all() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        assert_eq!(reg.by_provider("Anthropic").len(), 1);
        assert_eq!(reg.by_provider("openai").len(), 1);
        assert_eq!(reg.by_provider("ollama").len(), 1);
        assert_eq!(reg.by_provider("nope").len(), 0);
    }

    #[test]
    fn supporting_filters_by_capability_intersection() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        let vision = reg.supporting(&[ModelCapability::Vision]);
        assert_eq!(vision.len(), 2);
        let json = reg.supporting(&[ModelCapability::JsonMode]);
        assert_eq!(json.len(), 1);
        assert_eq!(json[0].id, "gpt-4o");
        // AND, not OR:
        let both = reg.supporting(&[ModelCapability::Vision, ModelCapability::JsonMode]);
        assert_eq!(both.len(), 1);
    }

    #[test]
    fn grouped_by_provider_sorts_within_group() {
        let json = r#"{"models": [
            {"id":"b","provider":"x"},{"id":"a","provider":"x"},{"id":"c","provider":"y"}
        ]}"#;
        let reg = ModelRegistry::from_json(json).unwrap();
        let g = reg.grouped_by_provider();
        let xs: Vec<_> = g.get("x").unwrap().iter().map(|m| m.id.as_str()).collect();
        assert_eq!(xs, vec!["a", "b"]);
    }

    #[test]
    fn parse_accepts_bare_array() {
        let arr = r#"[{"id":"m","provider":"p"}]"#;
        let reg = ModelRegistry::from_json(arr).unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn parse_rejects_empty_id() {
        let err = ModelRegistry::from_json(r#"[{"id":"","provider":"p"}]"#).unwrap_err();
        match err {
            ModelsDevError::Parse(_) => {}
            other => panic!("expected Parse, got {:?}", other),
        }
    }

    #[test]
    fn parse_ignores_unknown_fields() {
        let json = r#"{"models":[{"id":"a","provider":"p","_extra":42}]}"#;
        let reg = ModelRegistry::from_json(json).unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn parse_rejects_garbage() {
        let err = ModelRegistry::from_json("not json").unwrap_err();
        match err {
            ModelsDevError::Parse(_) => {}
            other => panic!("expected Parse, got {:?}", other),
        }
    }

    // ---------- cache I/O ----------

    fn cache_cfg(name: &str) -> (ModelsDevConfig, PathBuf) {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ai_assistant_modelsdev_{}_{}.json",
            name,
            std::process::id()
        ));
        let _ = fs::remove_file(&p);
        (
            ModelsDevConfig {
                cache_path: Some(p.clone()),
                ..Default::default()
            },
            p,
        )
    }

    #[test]
    fn save_and_load_cache_roundtrip() {
        let (cfg, path) = cache_cfg("roundtrip");
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        save_cache(&reg, &cfg).unwrap();
        let loaded = load_cache(&cfg).unwrap().unwrap();
        assert_eq!(loaded.len(), 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_cache_returns_none_when_missing() {
        let cfg = ModelsDevConfig {
            cache_path: Some(PathBuf::from("/no/such/dir/zzz/cache.json")),
            ..Default::default()
        };
        assert!(load_cache(&cfg).unwrap().is_none());
    }

    #[test]
    fn load_cache_returns_none_when_stale() {
        let (mut cfg, path) = cache_cfg("stale");
        cfg.cache_ttl = Duration::from_nanos(1);
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        save_cache(&reg, &cfg).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(load_cache(&cfg).unwrap().is_none());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_rejects_oversized_payload() {
        let (mut cfg, path) = cache_cfg("oversized");
        cfg.max_payload_bytes = 50;
        let mut models = Vec::new();
        for i in 0..100 {
            models.push(ModelMetadata {
                id: format!("model-{}", i),
                provider: "p".into(),
                ..Default::default()
            });
        }
        let reg = ModelRegistry {
            models,
            fetched_at: None,
            source: None,
        };
        let err = save_cache(&reg, &cfg).unwrap_err();
        match err {
            ModelsDevError::TooLarge { .. } => {}
            other => panic!("expected TooLarge, got {:?}", other),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_oversized_cache_file() {
        let (mut cfg, path) = cache_cfg("loadbig");
        let big = serde_json::json!({"models": (0..2000).map(|i| {
            serde_json::json!({"id": format!("m-{}", i), "provider":"p"})
        }).collect::<Vec<_>>()});
        fs::write(&path, serde_json::to_string(&big).unwrap()).unwrap();
        cfg.max_payload_bytes = 100;
        let err = load_cache(&cfg).unwrap_err();
        match err {
            ModelsDevError::TooLarge { .. } => {}
            other => panic!("expected TooLarge, got {:?}", other),
        }
        let _ = fs::remove_file(&path);
    }

    // ---------- bridge to crate::models ----------

    #[test]
    fn provider_from_key_known_providers() {
        use crate::config::AiProvider;
        assert_eq!(provider_from_key("openai"), AiProvider::OpenAI);
        assert_eq!(provider_from_key("OpenAI"), AiProvider::OpenAI);
        assert_eq!(provider_from_key("anthropic"), AiProvider::Anthropic);
        assert_eq!(provider_from_key("ollama"), AiProvider::Ollama);
        assert_eq!(provider_from_key("lm_studio"), AiProvider::LMStudio);
        assert_eq!(provider_from_key("lm-studio"), AiProvider::LMStudio);
        assert_eq!(provider_from_key("groq"), AiProvider::Groq);
    }

    #[test]
    fn provider_from_key_unknown_falls_through_to_openai_compatible() {
        use crate::config::AiProvider;
        match provider_from_key("totally-new-provider-9000") {
            AiProvider::OpenAICompatible { base_url } => assert!(base_url.is_empty()),
            other => panic!("expected OpenAICompatible, got {:?}", other),
        }
    }

    #[test]
    fn model_metadata_to_model_info_maps_full_capabilities() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        let opus = reg.lookup("claude-opus-4-7").unwrap();
        let info = opus.to_model_info();
        assert_eq!(info.name, "claude-opus-4-7");
        let caps = info.capabilities.as_ref().expect("caps present");
        assert_eq!(caps.context_window, Some(200_000));
        assert_eq!(caps.max_output_tokens, Some(8_192));
        assert!(caps.supports_vision);
        assert!(caps.supports_tool_calling);
        assert!(caps.supports_streaming);
        assert!(!caps.supports_json_mode);
        assert_eq!(caps.input_cost_per_million, Some(15.0));
        assert_eq!(caps.output_cost_per_million, Some(75.0));
        assert_eq!(caps.knowledge_cutoff.as_deref(), Some("2026-01-01"));
    }

    #[test]
    fn extend_in_crate_registry_from_models_dev() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        let mut in_crate = crate::models::ModelRegistry::new();
        in_crate.extend_from_models_dev(&reg);
        assert_eq!(in_crate.model_count(), 3);
        let opus = in_crate.get("claude-opus-4-7").unwrap();
        assert!(opus
            .capabilities
            .as_ref()
            .map(|c| c.supports_vision)
            .unwrap_or(false));
    }

    #[test]
    fn to_model_infos_preserves_order() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        let infos = reg.to_model_infos();
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].name, "claude-opus-4-7");
        assert_eq!(infos[1].name, "gpt-4o");
        assert_eq!(infos[2].name, "llama3.3:70b");
    }
}
