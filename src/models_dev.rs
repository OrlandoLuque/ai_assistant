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

// ============================================================================
// Extended schema (V137) — open-weights universe
// ============================================================================
//
// `ModelMetadata` above models the cloud catalog (one row per
// addressable model). Open-weights modelling needs richer structure:
// a family (e.g. Llama-3.1-8B) ships in many quantizations (Q4_K_M,
// Q5_K_S, Q8_0…) and may be paired with LoRA adapters that specialise
// the base for a task. Each variant has its own VRAM/RAM footprint
// and backend support matrix.
//
// The new types live alongside `ModelMetadata`. `ModelRegistry` gains
// an optional `families` field; callers that only consume the legacy
// cloud schema (the only consumers as of V136) keep working unchanged.

/// Modality the family operates on. Default `Text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Modality {
    /// Plain text in/out.
    Text,
    /// Accepts image + text input, produces text.
    VisionText,
    /// Audio in, text out (ASR-like) or text in, audio out (TTS).
    TextAudio,
    /// Audio only (e.g. speech recogniser).
    AudioOnly,
    /// Any-to-any multimodal.
    Multimodal,
    /// Vector embeddings only.
    Embedding,
}

impl Default for Modality {
    fn default() -> Self {
        Self::Text
    }
}

/// Coarse family tag for routing/recommendation. Orthogonal to
/// `ModelCapability` which lists individual feature flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FamilyTag {
    GeneralChat,
    Reasoning,
    Coding,
    Math,
    Multilingual,
    Roleplay,
    Vision,
    LongContext,
    Embedding,
    Instruct,
    Base,
}

/// Quantization scheme. Open enum: unknown schemes round-trip through
/// `Other(String)` so new GGUF variants don't break parsing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Quantization {
    Fp32,
    Fp16,
    Bf16,
    Q8_0,
    Q6K,
    Q5KM,
    Q5KS,
    Q5_0,
    Q5_1,
    Q4KM,
    Q4KS,
    Q4_0,
    Q4_1,
    Q3KL,
    Q3KM,
    Q3KS,
    Q2K,
    /// PrismML fork only.
    Q1_0,
    Iq4NL,
    Iq3S,
    Iq2XS,
    Other(String),
}

impl Quantization {
    /// Parse a quantization tag. Case-insensitive; trims surrounding
    /// whitespace. Unknown schemes round-trip through `Other`.
    pub fn parse(s: &str) -> Self {
        let upper = s.trim().to_ascii_uppercase();
        match upper.as_str() {
            "FP32" | "F32" => Self::Fp32,
            "FP16" | "F16" => Self::Fp16,
            "BF16" => Self::Bf16,
            "Q8_0" => Self::Q8_0,
            "Q6_K" => Self::Q6K,
            "Q5_K_M" => Self::Q5KM,
            "Q5_K_S" => Self::Q5KS,
            "Q5_0" => Self::Q5_0,
            "Q5_1" => Self::Q5_1,
            "Q4_K_M" => Self::Q4KM,
            "Q4_K_S" => Self::Q4KS,
            "Q4_0" => Self::Q4_0,
            "Q4_1" => Self::Q4_1,
            "Q3_K_L" => Self::Q3KL,
            "Q3_K_M" => Self::Q3KM,
            "Q3_K_S" => Self::Q3KS,
            "Q2_K" => Self::Q2K,
            "Q1_0" => Self::Q1_0,
            "IQ4_NL" => Self::Iq4NL,
            "IQ3_S" => Self::Iq3S,
            "IQ2_XS" => Self::Iq2XS,
            _ => Self::Other(s.trim().to_string()),
        }
    }

    /// Canonical string form (GGUF naming, uppercase).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Fp32 => "FP32",
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q6K => "Q6_K",
            Self::Q5KM => "Q5_K_M",
            Self::Q5KS => "Q5_K_S",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q4KM => "Q4_K_M",
            Self::Q4KS => "Q4_K_S",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q3KL => "Q3_K_L",
            Self::Q3KM => "Q3_K_M",
            Self::Q3KS => "Q3_K_S",
            Self::Q2K => "Q2_K",
            Self::Q1_0 => "Q1_0",
            Self::Iq4NL => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2XS => "IQ2_XS",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Coarse quality tier — used by the recommender to pick a
    /// fallback when the sweet-spot quantization doesn't fit VRAM.
    pub fn quality_rank(&self) -> u8 {
        match self {
            Self::Fp32 | Self::Fp16 | Self::Bf16 => 100,
            Self::Q8_0 => 90,
            Self::Q6K => 80,
            Self::Q5KM => 72,
            Self::Q5KS | Self::Q5_0 | Self::Q5_1 => 68,
            Self::Q4KM => 60,
            Self::Q4KS | Self::Q4_0 | Self::Q4_1 | Self::Iq4NL => 55,
            Self::Q3KL => 45,
            Self::Q3KM => 40,
            Self::Q3KS | Self::Iq3S => 35,
            Self::Q2K | Self::Iq2XS => 25,
            Self::Q1_0 => 15,
            Self::Other(_) => 50,
        }
    }
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Quantization {
    fn from(s: &str) -> Self {
        Self::parse(s)
    }
}

impl From<String> for Quantization {
    fn from(s: String) -> Self {
        Self::parse(&s)
    }
}

impl From<Quantization> for String {
    fn from(q: Quantization) -> Self {
        q.as_str().to_string()
    }
}

impl Serialize for Quantization {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Quantization {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(Self::parse(&s))
    }
}

/// Variant kind — what the file is, structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VariantKind {
    /// Full weights (the base model).
    Base,
    /// Mixture-of-experts variant (e.g. Mixtral).
    MoE,
    /// Distilled smaller version of a larger family.
    Distilled,
    /// Fine-tuned variant (instruct, code, etc.).
    FineTune,
    /// Merge of several fine-tunes.
    Merge,
}

impl Default for VariantKind {
    fn default() -> Self {
        Self::Base
    }
}

/// Optional content/behaviour modifier independent of `VariantKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VariantModifier {
    /// Refusal layers removed via orthogonalisation. Safety guardrails
    /// largely gone — caller responsibility to enforce policy.
    Abliterated,
    /// Trained without alignment / fine-tuned past it.
    Uncensored,
    /// Quantised by a community member (not the original creator).
    CommunityQuant,
}

/// Where a variant lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelSource {
    HuggingFace {
        repo: String,
        file: Option<String>,
    },
    Ollama {
        tag: String,
    },
    Url {
        url: String,
    },
    /// Curated entry shipped in-crate (`curated_models.rs`).
    Curated {
        key: String,
    },
}

impl ModelSource {
    /// Stable identifier for dedup / caching.
    pub fn key(&self) -> String {
        match self {
            Self::HuggingFace { repo, file } => match file {
                Some(f) => format!("hf:{}#{}", repo, f),
                None => format!("hf:{}", repo),
            },
            Self::Ollama { tag } => format!("ollama:{}", tag),
            Self::Url { url } => format!("url:{}", url),
            Self::Curated { key } => format!("curated:{}", key),
        }
    }
}

/// Provenance / publisher classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Provenance {
    /// Published by the model's creator (Meta, Mistral, etc.).
    Official,
    /// Community fork or fine-tune.
    CommunityFork {
        author: String,
    },
    /// Mainstream community quant (TheBloke, bartowski, …).
    CommunityQuant {
        author: String,
    },
    /// Curated entry — vetted manually in-crate.
    Curated,
    Unknown,
}

impl Default for Provenance {
    fn default() -> Self {
        Self::Unknown
    }
}

/// GPU architecture / API target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GpuArch {
    /// NVIDIA CUDA compute capability (e.g. `"7.5"`, `"8.9"`).
    Cuda { compute: String },
    /// AMD ROCm / HIP.
    Rocm,
    /// Apple Metal.
    Metal,
    /// Vulkan compute (cross-vendor fallback).
    Vulkan,
    /// CPU-only — no GPU required.
    Cpu,
}

/// Backend known to be able to load this variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Backend {
    LlamaCppMainline,
    LlamaCppPrismML,
    Ollama,
    Vllm,
    LmStudio,
    TextGenWebUi,
    KoboldCpp,
    Candle,
    Mlx,
}

/// Sweet-spot tag — multiple may apply to the same variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SweetSpot {
    /// Best output quality for this family.
    Quality,
    /// Best quality / VRAM tradeoff.
    VramEfficiency,
    /// Fastest tok/s on consumer GPU.
    Speed,
    /// Smallest viable for the family.
    Lowest,
}

/// LoRA adapter purpose.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AdapterPurpose {
    Coding,
    Writing,
    Reasoning,
    Math,
    Translation,
    Roleplay,
    Medical,
    Legal,
    /// Anything not covered by the closed list — keep the original
    /// label for surface in UI.
    Other(String),
}

/// Hardware required to run a specific variant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HardwareRequirements {
    /// Minimum VRAM in bytes; `None` means CPU-only is viable.
    #[serde(default)]
    pub min_vram_bytes: Option<u64>,
    /// Minimum system RAM in bytes (off-GPU residency).
    #[serde(default)]
    pub min_ram_bytes: u64,
    /// GPU architectures known to work (empty = unknown / any).
    #[serde(default)]
    pub gpu_archs: Vec<GpuArch>,
    /// Backends known to load this variant.
    #[serde(default)]
    pub backends: Vec<Backend>,
}

impl HardwareRequirements {
    /// True if the variant has no GPU requirement and a runnable RAM
    /// figure.
    pub fn is_cpu_viable(&self) -> bool {
        self.min_vram_bytes.is_none() || self.gpu_archs.iter().any(|a| matches!(a, GpuArch::Cpu))
    }
}

/// A single weight file / variant of a family.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelVariant {
    /// Stable id, e.g. `"llama-3.1-8b-Q4_K_M"`.
    pub id: String,
    /// Optional display label override.
    #[serde(default)]
    pub display_name: Option<String>,
    /// Structural kind.
    #[serde(default)]
    pub variant_kind: VariantKind,
    /// Quantization scheme (if applicable).
    #[serde(default)]
    pub quantization: Option<Quantization>,
    /// Optional content/behaviour modifier.
    #[serde(default)]
    pub modifier: Option<VariantModifier>,
    /// On-disk size in bytes.
    #[serde(default)]
    pub size_bytes: u64,
    /// Hardware constraints.
    #[serde(default)]
    pub requirements: HardwareRequirements,
    /// Where the file lives.
    pub source: ModelSource,
    /// Sweet-spot tags (may be empty).
    #[serde(default)]
    pub sweet_spot_for: Vec<SweetSpot>,
    /// Who published it.
    #[serde(default)]
    pub provenance: Provenance,
    /// SPDX-ish license string. Free-form because the open-weights
    /// world is full of bespoke licenses (Llama Community, Gemma…).
    #[serde(default)]
    pub license: String,
}

/// A LoRA adapter — a low-rank patch applied on top of a base model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LoraAdapter {
    /// Stable id of the adapter.
    pub id: String,
    /// Optional display label.
    #[serde(default)]
    pub display_name: Option<String>,
    /// `ModelFamily::id` this adapter targets.
    pub base_family: String,
    /// What the adapter specialises the base for.
    pub purpose: AdapterPurpose,
    /// On-disk size in bytes.
    #[serde(default)]
    pub size_bytes: u64,
    /// Where the file lives.
    pub source: ModelSource,
    /// Adapter license.
    #[serde(default)]
    pub license: String,
}

/// A model family — base weights + N variants (quantizations / fine-tunes)
/// + optional LoRA adapters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelFamily {
    /// Stable family id, e.g. `"llama-3.1-8b"`.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Who created the base model.
    #[serde(default)]
    pub creator: String,
    /// One-line summary.
    #[serde(default)]
    pub description: String,
    /// Primary modality.
    #[serde(default)]
    pub modality: Modality,
    /// Max context window in tokens (None = unknown / varies).
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Knowledge cutoff as ISO-8601 string.
    #[serde(default)]
    pub training_cutoff: Option<String>,
    /// Coarse family tags for routing/recommendation.
    #[serde(default)]
    pub family_tags: Vec<FamilyTag>,
    /// Concrete variants (quantizations, fine-tunes…).
    #[serde(default)]
    pub variants: Vec<ModelVariant>,
    /// LoRA adapters known for this family.
    #[serde(default)]
    pub lora_adapters: Vec<LoraAdapter>,
}

impl ModelFamily {
    /// Lookup a variant by id within this family.
    pub fn lookup_variant(&self, id: &str) -> Option<&ModelVariant> {
        let lower = id.to_lowercase();
        self.variants.iter().find(|v| v.id.to_lowercase() == lower)
    }

    /// Lookup a LoRA adapter by id within this family.
    pub fn lookup_adapter(&self, id: &str) -> Option<&LoraAdapter> {
        let lower = id.to_lowercase();
        self.lora_adapters
            .iter()
            .find(|a| a.id.to_lowercase() == lower)
    }

    /// True if any of the family's tags match the requested tag.
    pub fn has_tag(&self, tag: FamilyTag) -> bool {
        self.family_tags.contains(&tag)
    }

    /// Variants that fit inside `available_vram_bytes` of GPU memory.
    /// CPU-viable variants are always included (since they don't need
    /// VRAM).
    pub fn variants_fitting_vram(&self, available_vram_bytes: u64) -> Vec<&ModelVariant> {
        self.variants
            .iter()
            .filter(|v| match v.requirements.min_vram_bytes {
                None => true,
                Some(need) => need <= available_vram_bytes,
            })
            .collect()
    }
}

/// In-memory registry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<ModelMetadata>,
    /// Open-weights families (V137+). Empty for legacy models.dev
    /// payloads.
    #[serde(default)]
    pub families: Vec<ModelFamily>,
    /// When this snapshot was fetched (monotonic across cache reloads).
    #[serde(default)]
    pub fetched_at: Option<SystemTime>,
    /// Free-form source URL (for trace/diagnostics).
    #[serde(default)]
    pub source: Option<String>,
}

impl ModelRegistry {
    /// Parse from a JSON document. Accepts both `{"models": [...]}` and a
    /// bare array `[...]`. The object form may also carry a `families: [...]`
    /// array (V137+) holding open-weights families with their variants and
    /// LoRA adapters; the field is optional and falls back to empty for
    /// legacy payloads. Sets `fetched_at` to `now`.
    pub fn from_json(text: &str) -> Result<Self, ModelsDevError> {
        // Bare-array tolerance.
        let trimmed = text.trim_start();
        let value: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| ModelsDevError::Parse(e.to_string()))?;

        let (models_val, families_val) = match value {
            serde_json::Value::Array(_) => (value, serde_json::Value::Array(Vec::new())),
            serde_json::Value::Object(mut obj) => {
                let models_val = obj
                    .remove("models")
                    .unwrap_or(serde_json::Value::Array(Vec::new()));
                let families_val = obj
                    .remove("families")
                    .unwrap_or(serde_json::Value::Array(Vec::new()));
                (models_val, families_val)
            }
            other => {
                return Err(ModelsDevError::Parse(format!(
                    "unexpected top-level JSON kind: {}",
                    type_name_of(&other)
                )));
            }
        };

        let models: Vec<ModelMetadata> =
            serde_json::from_value(models_val).map_err(|e| ModelsDevError::Parse(e.to_string()))?;
        let families: Vec<ModelFamily> = serde_json::from_value(families_val)
            .map_err(|e| ModelsDevError::Parse(e.to_string()))?;
        // Reject empty ids in either section.
        for m in &models {
            if m.id.trim().is_empty() {
                return Err(ModelsDevError::Parse("model with empty id".into()));
            }
        }
        for f in &families {
            if f.id.trim().is_empty() {
                return Err(ModelsDevError::Parse("family with empty id".into()));
            }
            for v in &f.variants {
                if v.id.trim().is_empty() {
                    return Err(ModelsDevError::Parse(format!(
                        "variant with empty id in family {}",
                        f.id
                    )));
                }
            }
            for a in &f.lora_adapters {
                if a.id.trim().is_empty() {
                    return Err(ModelsDevError::Parse(format!(
                        "adapter with empty id in family {}",
                        f.id
                    )));
                }
            }
        }

        Ok(Self {
            models,
            families,
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

    // ---------- open-weights families (V137) ----------

    /// Number of families loaded.
    pub fn family_count(&self) -> usize {
        self.families.len()
    }

    /// Look up a family by id (case-insensitive).
    pub fn lookup_family(&self, id: &str) -> Option<&ModelFamily> {
        let lower = id.to_lowercase();
        self.families.iter().find(|f| f.id.to_lowercase() == lower)
    }

    /// Find a variant across all families by variant id (case-insensitive).
    /// Returns the owning family alongside the variant.
    pub fn find_variant(&self, id: &str) -> Option<(&ModelFamily, &ModelVariant)> {
        let lower = id.to_lowercase();
        for fam in &self.families {
            if let Some(v) = fam.variants.iter().find(|v| v.id.to_lowercase() == lower) {
                return Some((fam, v));
            }
        }
        None
    }

    /// Find a LoRA adapter across all families by adapter id
    /// (case-insensitive). Returns the owning family alongside the adapter.
    pub fn find_adapter(&self, id: &str) -> Option<(&ModelFamily, &LoraAdapter)> {
        let lower = id.to_lowercase();
        for fam in &self.families {
            if let Some(a) = fam
                .lora_adapters
                .iter()
                .find(|a| a.id.to_lowercase() == lower)
            {
                return Some((fam, a));
            }
        }
        None
    }

    /// All families carrying `tag`.
    pub fn families_by_tag(&self, tag: FamilyTag) -> Vec<&ModelFamily> {
        self.families.iter().filter(|f| f.has_tag(tag)).collect()
    }

    /// All families whose primary modality matches.
    pub fn families_by_modality(&self, modality: Modality) -> Vec<&ModelFamily> {
        self.families
            .iter()
            .filter(|f| f.modality == modality)
            .collect()
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
// HTTP fetcher (V138) — gated behind `models-dev-fetcher`
// ============================================================================
//
// Closes the docstring's "actual HTTP fetch is left to the caller" by
// bundling an async fetcher in-crate. Reuses `ModelsDevConfig` and the
// `save_cache`/`load_cache` helpers from V104.9; the fetcher only adds
// the network half plus a `RefreshPolicy`. Off by default — callers that
// only want the parser/cache from V137 don't pull in tokio + reqwest.

#[cfg(feature = "models-dev-fetcher")]
mod fetcher {
    use super::{load_cache, save_cache, ModelRegistry, ModelsDevConfig, ModelsDevError};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    /// HTTP client abstracted for the catalog fetcher. The default impl
    /// (`ReqwestCatalogClient`) uses `reqwest`. A mock impl is used in
    /// tests to canned-respond without touching the network.
    pub trait CatalogFetchClient: Send + Sync {
        /// GET `url` and return the raw body. Implementations MUST refuse
        /// to read more than `max_bytes` (return
        /// `ModelsDevError::TooLarge`); responses without
        /// `Content-Length` should be streamed and aborted once the cap
        /// is exceeded.
        fn get_bytes_capped<'a>(
            &'a self,
            url: &'a str,
            timeout: Duration,
            max_bytes: u64,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ModelsDevError>> + Send + 'a>>;
    }

    /// `reqwest`-backed implementation of `CatalogFetchClient`.
    #[derive(Clone, Default)]
    pub struct ReqwestCatalogClient {
        client: reqwest::Client,
    }

    impl ReqwestCatalogClient {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_client(client: reqwest::Client) -> Self {
            Self { client }
        }
    }

    impl CatalogFetchClient for ReqwestCatalogClient {
        fn get_bytes_capped<'a>(
            &'a self,
            url: &'a str,
            timeout: Duration,
            max_bytes: u64,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ModelsDevError>> + Send + 'a>> {
            Box::pin(async move {
                use futures::StreamExt;
                let resp = self
                    .client
                    .get(url)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| ModelsDevError::Io(format!("GET {}: {}", url, e)))?;

                let status = resp.status();
                if !status.is_success() {
                    return Err(ModelsDevError::Io(format!("HTTP {} from {}", status, url)));
                }

                // Pre-flight check via Content-Length when present.
                if let Some(declared) = resp.content_length() {
                    if declared > max_bytes {
                        return Err(ModelsDevError::TooLarge {
                            size: declared,
                            limit: max_bytes,
                        });
                    }
                }

                let mut buf: Vec<u8> = Vec::new();
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk =
                        chunk.map_err(|e| ModelsDevError::Io(format!("stream {}: {}", url, e)))?;
                    if (buf.len() as u64).saturating_add(chunk.len() as u64) > max_bytes {
                        return Err(ModelsDevError::TooLarge {
                            size: (buf.len() + chunk.len()) as u64,
                            limit: max_bytes,
                        });
                    }
                    buf.extend_from_slice(&chunk);
                }
                Ok(buf)
            })
        }
    }

    /// Exponential backoff with jitter applied between failed refresh
    /// attempts in `Background` mode.
    #[derive(Debug, Clone)]
    pub struct BackoffPolicy {
        pub initial_delay: Duration,
        pub max_delay: Duration,
        /// Stop / mark "degraded" after this many consecutive failures.
        pub max_consecutive_failures: u32,
    }

    impl Default for BackoffPolicy {
        fn default() -> Self {
            Self {
                initial_delay: Duration::from_secs(30),
                max_delay: Duration::from_secs(60 * 60),
                max_consecutive_failures: 5,
            }
        }
    }

    /// When to talk to the network.
    #[derive(Debug, Clone)]
    pub enum RefreshPolicy {
        /// Never refresh; only ever serve cached data.
        Never,
        /// Fetch only when no cache exists or the load failed.
        OnMiss,
        /// Fetch when the cache is older than `cfg.cache_ttl` (default).
        OnStale,
        /// Spawn a background task that refreshes on `interval`.
        Background {
            interval: Duration,
            on_error: BackoffPolicy,
        },
    }

    impl Default for RefreshPolicy {
        fn default() -> Self {
            Self::OnStale
        }
    }

    /// Async fetcher for `ModelRegistry`. Combines `ModelsDevConfig`
    /// (cache TTL, max payload bytes, cache path) with a network client
    /// and a refresh policy.
    pub struct ModelsDevFetcher {
        endpoint: String,
        cfg: ModelsDevConfig,
        client: Arc<dyn CatalogFetchClient>,
        policy: RefreshPolicy,
        request_timeout: Duration,
        /// Cached in-memory copy + write lock — coalesces concurrent
        /// `registry()` callers onto a single inflight fetch.
        inner: Mutex<Option<ModelRegistry>>,
        /// Set when N consecutive background failures cross the
        /// configured threshold. Callers can read via
        /// `is_degraded()`.
        degraded: AtomicBool,
        /// Monotonic counter incremented on every successful fetch —
        /// observable for testing / metrics.
        refresh_count: AtomicU64,
    }

    impl ModelsDevFetcher {
        /// Default endpoint.
        pub const DEFAULT_ENDPOINT: &'static str = "https://models.dev/api.json";

        pub fn new(cfg: ModelsDevConfig, client: Arc<dyn CatalogFetchClient>) -> Self {
            Self {
                endpoint: Self::DEFAULT_ENDPOINT.to_string(),
                cfg,
                client,
                policy: RefreshPolicy::default(),
                request_timeout: Duration::from_secs(30),
                inner: Mutex::new(None),
                degraded: AtomicBool::new(false),
                refresh_count: AtomicU64::new(0),
            }
        }

        pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.endpoint = endpoint.into();
            self
        }

        pub fn with_policy(mut self, policy: RefreshPolicy) -> Self {
            self.policy = policy;
            self
        }

        pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
            self.request_timeout = timeout;
            self
        }

        pub fn endpoint(&self) -> &str {
            &self.endpoint
        }

        pub fn is_degraded(&self) -> bool {
            self.degraded.load(Ordering::Acquire)
        }

        pub fn refresh_count(&self) -> u64 {
            self.refresh_count.load(Ordering::Acquire)
        }

        /// Return the registry. Behaviour depends on `policy`:
        /// - `Never`: returns the in-memory copy or `Err` if unset and
        ///   no cache.
        /// - `OnMiss`: fetches only if no in-memory copy and no cache.
        /// - `OnStale`: re-fetches if cache is older than `cache_ttl`.
        /// - `Background`: serves the cached copy; refresh runs out of
        ///   band.
        pub async fn registry(&self) -> Result<ModelRegistry, ModelsDevError> {
            // Serialise concurrent callers onto one fetch.
            let mut guard = self.inner.lock().await;

            if guard.is_none() {
                // Bootstrap from on-disk cache when present.
                if let Some(reg) = load_cache(&self.cfg)? {
                    *guard = Some(reg);
                }
            }

            let needs_fetch = match (&self.policy, &*guard) {
                (RefreshPolicy::Never, Some(_)) => false,
                (RefreshPolicy::Never, None) => {
                    return Err(ModelsDevError::Io(
                        "RefreshPolicy::Never and no cache available".into(),
                    ));
                }
                (RefreshPolicy::OnMiss, Some(_)) => false,
                (RefreshPolicy::OnMiss, None) => true,
                (RefreshPolicy::OnStale, _) => {
                    // load_cache already filtered stale entries.
                    guard.is_none()
                }
                (RefreshPolicy::Background { .. }, Some(_)) => false,
                (RefreshPolicy::Background { .. }, None) => true,
            };

            if needs_fetch {
                let reg = self.fetch_once().await?;
                *guard = Some(reg);
            }

            guard
                .as_ref()
                .cloned()
                .ok_or_else(|| ModelsDevError::Io("registry unavailable".into()))
        }

        /// Force a refresh now, ignoring policy. Updates the in-memory
        /// copy and the on-disk cache (if configured).
        pub async fn force_refresh(&self) -> Result<(), ModelsDevError> {
            let mut guard = self.inner.lock().await;
            let reg = self.fetch_once().await?;
            *guard = Some(reg);
            Ok(())
        }

        /// One unguarded fetch + cache write.
        async fn fetch_once(&self) -> Result<ModelRegistry, ModelsDevError> {
            let bytes = self
                .client
                .get_bytes_capped(
                    &self.endpoint,
                    self.request_timeout,
                    self.cfg.max_payload_bytes,
                )
                .await?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| ModelsDevError::Parse(format!("response not utf-8: {}", e)))?;
            let mut reg = ModelRegistry::from_json(text)?;
            reg.source = Some(self.endpoint.clone());
            save_cache(&reg, &self.cfg)?;
            self.refresh_count.fetch_add(1, Ordering::AcqRel);
            self.degraded.store(false, Ordering::Release);
            Ok(reg)
        }

        /// Spawn a background refresh task. Returns a handle that
        /// cancels the task on drop. Only meaningful when policy is
        /// `Background`; for other policies the task exits immediately.
        pub fn start_background(self: &Arc<Self>) -> BackgroundHandle {
            let (interval, backoff) = match &self.policy {
                RefreshPolicy::Background { interval, on_error } => (*interval, on_error.clone()),
                _ => {
                    let cancel = Arc::new(AtomicBool::new(true));
                    return BackgroundHandle { cancel, join: None };
                }
            };

            let cancel = Arc::new(AtomicBool::new(false));
            let cancel_for_task = cancel.clone();
            let fetcher = Arc::clone(self);

            let join = tokio::spawn(async move {
                let mut consecutive_failures: u32 = 0;
                let mut next_delay = interval;

                loop {
                    if cancel_for_task.load(Ordering::Acquire) {
                        break;
                    }
                    tokio::time::sleep(next_delay).await;
                    if cancel_for_task.load(Ordering::Acquire) {
                        break;
                    }

                    match fetcher.force_refresh().await {
                        Ok(()) => {
                            consecutive_failures = 0;
                            next_delay = interval;
                        }
                        Err(_e) => {
                            consecutive_failures = consecutive_failures.saturating_add(1);
                            if consecutive_failures >= backoff.max_consecutive_failures {
                                fetcher.degraded.store(true, Ordering::Release);
                            }
                            // Exponential backoff with cap.
                            let exp = backoff
                                .initial_delay
                                .saturating_mul(1u32 << consecutive_failures.min(10));
                            next_delay = exp.min(backoff.max_delay);
                        }
                    }
                }
            });

            BackgroundHandle {
                cancel,
                join: Some(join),
            }
        }
    }

    /// Cancellable handle for the background refresh task. Dropping
    /// signals cancellation; the task observes the flag at its next
    /// wake-up.
    pub struct BackgroundHandle {
        cancel: Arc<AtomicBool>,
        join: Option<tokio::task::JoinHandle<()>>,
    }

    impl BackgroundHandle {
        pub fn cancel(&self) {
            self.cancel.store(true, Ordering::Release);
        }

        pub fn is_cancelled(&self) -> bool {
            self.cancel.load(Ordering::Acquire)
        }
    }

    impl Drop for BackgroundHandle {
        fn drop(&mut self) {
            self.cancel();
            if let Some(j) = self.join.take() {
                j.abort();
            }
        }
    }
}

#[cfg(feature = "models-dev-fetcher")]
pub use fetcher::{
    BackgroundHandle, BackoffPolicy, CatalogFetchClient, ModelsDevFetcher, RefreshPolicy,
    ReqwestCatalogClient,
};

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
            ..Default::default()
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

    // ---------- open-weights families (V137) ----------
    //
    // Inline fixture for Llama-3.1-8B with four GGUF quantizations and one
    // LoRA adapter. Mirrors the shape we expect a real catalog payload to
    // take: variants carry sweet-spot tags, hardware requirements, source
    // pointers and a provenance.

    fn families_json() -> &'static str {
        r#"{
          "models": [],
          "families": [
            {
              "id": "llama-3.1-8b",
              "display_name": "Llama 3.1 8B",
              "creator": "Meta",
              "description": "Open-weights 8B base from Meta.",
              "modality": "text",
              "context_window": 131072,
              "training_cutoff": "2024-12-01",
              "family_tags": ["general_chat", "instruct", "long_context"],
              "variants": [
                {
                  "id": "llama-3.1-8b-Q8_0",
                  "variant_kind": "base",
                  "quantization": "Q8_0",
                  "size_bytes": 8500000000,
                  "requirements": {
                    "min_vram_bytes": 10000000000,
                    "min_ram_bytes": 4000000000,
                    "backends": ["llama_cpp_mainline", "ollama"]
                  },
                  "source": {"hugging_face": {"repo": "bartowski/Llama-3.1-8B-GGUF",
                                              "file": "Llama-3.1-8B-Q8_0.gguf"}},
                  "sweet_spot_for": ["quality"],
                  "provenance": {"community_quant": {"author": "bartowski"}},
                  "license": "Llama-3.1-Community"
                },
                {
                  "id": "llama-3.1-8b-Q5_K_M",
                  "variant_kind": "base",
                  "quantization": "Q5_K_M",
                  "size_bytes": 5700000000,
                  "requirements": {
                    "min_vram_bytes": 7000000000,
                    "min_ram_bytes": 3000000000,
                    "backends": ["llama_cpp_mainline", "ollama"]
                  },
                  "source": {"hugging_face": {"repo": "bartowski/Llama-3.1-8B-GGUF",
                                              "file": "Llama-3.1-8B-Q5_K_M.gguf"}},
                  "sweet_spot_for": [],
                  "provenance": {"community_quant": {"author": "bartowski"}},
                  "license": "Llama-3.1-Community"
                },
                {
                  "id": "llama-3.1-8b-Q4_K_M",
                  "variant_kind": "base",
                  "quantization": "Q4_K_M",
                  "size_bytes": 4900000000,
                  "requirements": {
                    "min_vram_bytes": 6000000000,
                    "min_ram_bytes": 3000000000,
                    "backends": ["llama_cpp_mainline", "ollama", "lm_studio"]
                  },
                  "source": {"hugging_face": {"repo": "bartowski/Llama-3.1-8B-GGUF",
                                              "file": "Llama-3.1-8B-Q4_K_M.gguf"}},
                  "sweet_spot_for": ["vram_efficiency"],
                  "provenance": {"community_quant": {"author": "bartowski"}},
                  "license": "Llama-3.1-Community"
                },
                {
                  "id": "llama-3.1-8b-Q3_K_M",
                  "variant_kind": "base",
                  "quantization": "Q3_K_M",
                  "size_bytes": 4000000000,
                  "requirements": {
                    "min_vram_bytes": 5000000000,
                    "min_ram_bytes": 3000000000,
                    "backends": ["llama_cpp_mainline"]
                  },
                  "source": {"hugging_face": {"repo": "bartowski/Llama-3.1-8B-GGUF",
                                              "file": "Llama-3.1-8B-Q3_K_M.gguf"}},
                  "sweet_spot_for": ["lowest"],
                  "provenance": {"community_quant": {"author": "bartowski"}},
                  "license": "Llama-3.1-Community"
                }
              ],
              "lora_adapters": [
                {
                  "id": "llama-3.1-8b-coding-lora",
                  "display_name": "Llama-3.1-8B Coding LoRA",
                  "base_family": "llama-3.1-8b",
                  "purpose": "coding",
                  "size_bytes": 150000000,
                  "source": {"hugging_face": {"repo": "example/coding-lora", "file": null}},
                  "license": "Apache-2.0"
                }
              ]
            }
          ]
        }"#
    }

    #[test]
    fn family_parse_round_trip() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        assert_eq!(reg.family_count(), 1);
        let fam = reg.lookup_family("llama-3.1-8b").expect("family");
        assert_eq!(fam.display_name, "Llama 3.1 8B");
        assert_eq!(fam.modality, Modality::Text);
        assert_eq!(fam.context_window, Some(131_072));
        assert_eq!(fam.variants.len(), 4);
        assert_eq!(fam.lora_adapters.len(), 1);
    }

    #[test]
    fn family_lookup_variant_finds_owning_family() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        let (fam, var) = reg.find_variant("llama-3.1-8b-Q4_K_M").expect("variant");
        assert_eq!(fam.id, "llama-3.1-8b");
        assert_eq!(var.quantization, Some(Quantization::Q4KM));
        assert!(var.sweet_spot_for.contains(&SweetSpot::VramEfficiency));
    }

    #[test]
    fn family_lookup_variant_case_insensitive() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        assert!(reg.find_variant("LLAMA-3.1-8B-q4_k_m").is_some());
    }

    #[test]
    fn family_lookup_adapter_returns_owning_family() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        let (fam, ad) = reg
            .find_adapter("llama-3.1-8b-coding-lora")
            .expect("adapter");
        assert_eq!(fam.id, "llama-3.1-8b");
        assert_eq!(ad.purpose, AdapterPurpose::Coding);
        assert_eq!(ad.base_family, "llama-3.1-8b");
    }

    #[test]
    fn family_filter_by_tag() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        assert_eq!(reg.families_by_tag(FamilyTag::Instruct).len(), 1);
        assert_eq!(reg.families_by_tag(FamilyTag::Coding).len(), 0);
    }

    #[test]
    fn family_filter_by_modality() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        assert_eq!(reg.families_by_modality(Modality::Text).len(), 1);
        assert_eq!(reg.families_by_modality(Modality::VisionText).len(), 0);
    }

    #[test]
    fn variants_fitting_vram_falls_back_to_smaller_quants() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        let fam = reg.lookup_family("llama-3.1-8b").unwrap();
        // 8 GiB: drops Q8_0 (needs 10 GB), keeps Q5_K_M, Q4_K_M, Q3_K_M.
        let fits_8gb = fam.variants_fitting_vram(8_000_000_000);
        assert_eq!(fits_8gb.len(), 3);
        // 5.5 GiB: only Q3_K_M (5 GB) fits.
        let fits_5_5gb = fam.variants_fitting_vram(5_500_000_000);
        assert_eq!(fits_5_5gb.len(), 1);
        assert_eq!(fits_5_5gb[0].id, "llama-3.1-8b-Q3_K_M");
        // 12 GiB: everything fits.
        assert_eq!(fam.variants_fitting_vram(12_000_000_000).len(), 4);
    }

    #[test]
    fn quantization_parse_round_trip() {
        for raw in [
            "Q4_K_M", "Q8_0", "Q5_K_M", "Q3_K_S", "FP16", "BF16", "IQ4_NL", "Q2_K",
        ] {
            let q = Quantization::parse(raw);
            assert_eq!(q.as_str(), raw, "round-trip failed for {}", raw);
        }
    }

    #[test]
    fn quantization_parse_unknown_keeps_original_label() {
        let q = Quantization::parse("Q4_K_M_NEW");
        match q {
            Quantization::Other(ref s) => assert_eq!(s, "Q4_K_M_NEW"),
            _ => panic!("expected Other"),
        }
        assert_eq!(q.as_str(), "Q4_K_M_NEW");
    }

    #[test]
    fn quantization_parse_is_case_insensitive() {
        assert_eq!(Quantization::parse("q4_k_m"), Quantization::Q4KM);
        assert_eq!(Quantization::parse(" Q8_0 "), Quantization::Q8_0);
    }

    #[test]
    fn quantization_quality_rank_orders_correctly() {
        assert!(Quantization::Fp16.quality_rank() > Quantization::Q8_0.quality_rank());
        assert!(Quantization::Q8_0.quality_rank() > Quantization::Q5KM.quality_rank());
        assert!(Quantization::Q5KM.quality_rank() > Quantization::Q4KM.quality_rank());
        assert!(Quantization::Q4KM.quality_rank() > Quantization::Q3KM.quality_rank());
        assert!(Quantization::Q3KM.quality_rank() > Quantization::Q2K.quality_rank());
    }

    #[test]
    fn quantization_serde_round_trip() {
        let q = Quantization::Q4KM;
        let s = serde_json::to_string(&q).unwrap();
        assert_eq!(s, "\"Q4_K_M\"");
        let back: Quantization = serde_json::from_str(&s).unwrap();
        assert_eq!(back, Quantization::Q4KM);
        // Unknown round-trips through Other.
        let parsed: Quantization = serde_json::from_str("\"ZZZ_4\"").unwrap();
        assert_eq!(parsed.as_str(), "ZZZ_4");
    }

    #[test]
    fn model_source_key_is_stable() {
        let s1 = ModelSource::HuggingFace {
            repo: "a/b".into(),
            file: Some("c.gguf".into()),
        };
        let s2 = ModelSource::HuggingFace {
            repo: "a/b".into(),
            file: None,
        };
        assert_eq!(s1.key(), "hf:a/b#c.gguf");
        assert_eq!(s2.key(), "hf:a/b");
        assert_eq!(
            ModelSource::Ollama {
                tag: "llama3".into()
            }
            .key(),
            "ollama:llama3"
        );
        assert_eq!(
            ModelSource::Curated { key: "k1".into() }.key(),
            "curated:k1"
        );
    }

    #[test]
    fn hardware_requirements_is_cpu_viable() {
        let cpu = HardwareRequirements {
            min_vram_bytes: None,
            ..Default::default()
        };
        assert!(cpu.is_cpu_viable());

        let gpu_only = HardwareRequirements {
            min_vram_bytes: Some(8_000_000_000),
            gpu_archs: vec![GpuArch::Cuda {
                compute: "8.9".into(),
            }],
            ..Default::default()
        };
        assert!(!gpu_only.is_cpu_viable());

        let mixed = HardwareRequirements {
            min_vram_bytes: Some(8_000_000_000),
            gpu_archs: vec![
                GpuArch::Cuda {
                    compute: "8.9".into(),
                },
                GpuArch::Cpu,
            ],
            ..Default::default()
        };
        assert!(mixed.is_cpu_viable());
    }

    #[test]
    fn family_round_trip_via_save_and_load() {
        let reg = ModelRegistry::from_json(families_json()).unwrap();
        let (cfg, path) = cache_cfg("families");
        save_cache(&reg, &cfg).unwrap();
        let loaded = load_cache(&cfg).unwrap().unwrap();
        assert_eq!(loaded.family_count(), 1);
        let (_, v) = loaded.find_variant("llama-3.1-8b-Q4_K_M").expect("variant");
        assert_eq!(v.quantization, Some(Quantization::Q4KM));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn parse_rejects_family_with_empty_id() {
        let bad = r#"{"families":[{"id":"","display_name":"X"}]}"#;
        let err = ModelRegistry::from_json(bad).unwrap_err();
        assert!(matches!(err, ModelsDevError::Parse(_)));
    }

    #[test]
    fn parse_rejects_variant_with_empty_id() {
        let bad = r#"{"families":[{
            "id":"f","display_name":"F",
            "variants":[{"id":"","source":{"ollama":{"tag":"x"}}}]
        }]}"#;
        let err = ModelRegistry::from_json(bad).unwrap_err();
        assert!(matches!(err, ModelsDevError::Parse(_)));
    }

    #[test]
    fn legacy_models_only_payload_still_parses_with_empty_families() {
        let reg = ModelRegistry::from_json(sample_json()).unwrap();
        assert_eq!(reg.family_count(), 0);
        assert_eq!(reg.len(), 3);
    }

    // ---------- HTTP fetcher (V138) — gated behind models-dev-fetcher ----------

    #[cfg(feature = "models-dev-fetcher")]
    mod fetcher_tests {
        use super::super::{
            load_cache, save_cache, CatalogFetchClient, ModelRegistry, ModelsDevConfig,
            ModelsDevError, ModelsDevFetcher, RefreshPolicy,
        };
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::Duration;

        fn fetcher_payload() -> &'static str {
            r#"{"models":[{"id":"m-fetch","provider":"openai"}]}"#
        }

        /// Mock client. Returns canned bodies in order; cycles when
        /// the queue is exhausted. Records the call count so tests
        /// can assert "exactly N requests went out".
        #[derive(Clone, Default)]
        struct MockClient {
            responses: Arc<Mutex<Vec<Result<Vec<u8>, &'static str>>>>,
            calls: Arc<AtomicUsize>,
        }

        impl MockClient {
            fn with_ok(body: &str) -> Self {
                Self {
                    responses: Arc::new(Mutex::new(vec![Ok(body.as_bytes().to_vec())])),
                    calls: Arc::new(AtomicUsize::new(0)),
                }
            }

            fn with_responses(items: Vec<Result<Vec<u8>, &'static str>>) -> Self {
                Self {
                    responses: Arc::new(Mutex::new(items)),
                    calls: Arc::new(AtomicUsize::new(0)),
                }
            }

            fn call_count(&self) -> usize {
                self.calls.load(Ordering::Acquire)
            }
        }

        impl CatalogFetchClient for MockClient {
            fn get_bytes_capped<'a>(
                &'a self,
                _url: &'a str,
                _timeout: Duration,
                max_bytes: u64,
            ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ModelsDevError>> + Send + 'a>>
            {
                self.calls.fetch_add(1, Ordering::AcqRel);
                let mut responses = self.responses.lock().unwrap();
                let resp = if responses.is_empty() {
                    Err("queue exhausted")
                } else if responses.len() == 1 {
                    responses[0].clone()
                } else {
                    responses.remove(0)
                };
                Box::pin(async move {
                    match resp {
                        Ok(bytes) => {
                            if (bytes.len() as u64) > max_bytes {
                                Err(ModelsDevError::TooLarge {
                                    size: bytes.len() as u64,
                                    limit: max_bytes,
                                })
                            } else {
                                Ok(bytes)
                            }
                        }
                        Err(msg) => Err(ModelsDevError::Io(msg.into())),
                    }
                })
            }
        }

        fn fetcher_cfg(name: &str) -> (ModelsDevConfig, std::path::PathBuf) {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "ai_assistant_fetcher_{}_{}.json",
                name,
                std::process::id()
            ));
            let _ = std::fs::remove_file(&p);
            (
                ModelsDevConfig {
                    cache_path: Some(p.clone()),
                    ..Default::default()
                },
                p,
            )
        }

        #[tokio::test]
        async fn registry_fetches_when_cache_absent() {
            let (cfg, path) = fetcher_cfg("absent");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client.clone()));
            let reg = fetcher.registry().await.unwrap();
            assert_eq!(reg.len(), 1);
            assert_eq!(reg.models[0].id, "m-fetch");
            assert_eq!(client.call_count(), 1);
            // Cache was written.
            assert!(path.exists());
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn registry_coalesces_repeat_calls_within_ttl() {
            let (cfg, path) = fetcher_cfg("coalesce");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client.clone()));
            let _r1 = fetcher.registry().await.unwrap();
            let _r2 = fetcher.registry().await.unwrap();
            let _r3 = fetcher.registry().await.unwrap();
            // OnStale + fresh cache → no extra network calls.
            assert_eq!(client.call_count(), 1);
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn registry_never_policy_refuses_when_no_cache() {
            let (cfg, _path) = fetcher_cfg("never_empty");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client.clone()))
                .with_policy(RefreshPolicy::Never);
            let err = fetcher.registry().await.unwrap_err();
            assert!(matches!(err, ModelsDevError::Io(_)));
            assert_eq!(client.call_count(), 0);
        }

        #[tokio::test]
        async fn registry_never_policy_serves_existing_cache() {
            let (cfg, path) = fetcher_cfg("never_cached");
            // Pre-seed the cache from disk.
            let seeded = ModelRegistry::from_json(fetcher_payload()).unwrap();
            save_cache(&seeded, &cfg).unwrap();

            let client = MockClient::with_ok("{}"); // would parse fine
            let fetcher = ModelsDevFetcher::new(cfg.clone(), Arc::new(client.clone()))
                .with_policy(RefreshPolicy::Never);
            let reg = fetcher.registry().await.unwrap();
            assert_eq!(reg.len(), 1);
            assert_eq!(client.call_count(), 0);
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn registry_on_miss_skips_fetch_when_cache_present() {
            let (cfg, path) = fetcher_cfg("onmiss_hit");
            let seeded = ModelRegistry::from_json(fetcher_payload()).unwrap();
            save_cache(&seeded, &cfg).unwrap();

            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg.clone(), Arc::new(client.clone()))
                .with_policy(RefreshPolicy::OnMiss);
            let reg = fetcher.registry().await.unwrap();
            assert_eq!(reg.len(), 1);
            assert_eq!(client.call_count(), 0);
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn force_refresh_always_fetches() {
            let (cfg, path) = fetcher_cfg("force");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client.clone()));
            let _r = fetcher.registry().await.unwrap();
            assert_eq!(client.call_count(), 1);
            fetcher.force_refresh().await.unwrap();
            fetcher.force_refresh().await.unwrap();
            assert_eq!(client.call_count(), 3);
            assert_eq!(fetcher.refresh_count(), 3);
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn payload_bomb_rejected_by_cap() {
            let (mut cfg, _path) = fetcher_cfg("bomb");
            cfg.max_payload_bytes = 20; // smaller than payload
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client));
            let err = fetcher.registry().await.unwrap_err();
            assert!(matches!(err, ModelsDevError::TooLarge { .. }));
        }

        #[tokio::test]
        async fn fetch_error_propagates() {
            let (cfg, _path) = fetcher_cfg("err");
            let client = MockClient::with_responses(vec![Err("boom")]);
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client));
            let err = fetcher.registry().await.unwrap_err();
            match err {
                ModelsDevError::Io(s) => assert!(s.contains("boom")),
                other => panic!("expected Io, got {:?}", other),
            }
        }

        #[tokio::test]
        async fn parse_error_on_garbage_response() {
            let (cfg, _path) = fetcher_cfg("garbage");
            let client = MockClient::with_responses(vec![Ok(b"not json".to_vec())]);
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client));
            let err = fetcher.registry().await.unwrap_err();
            assert!(matches!(err, ModelsDevError::Parse(_)));
        }

        #[tokio::test]
        async fn fetched_registry_round_trips_through_cache() {
            let (cfg, path) = fetcher_cfg("rt");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg.clone(), Arc::new(client));
            let _ = fetcher.registry().await.unwrap();
            let from_disk = load_cache(&cfg).unwrap().expect("cache present");
            assert_eq!(from_disk.len(), 1);
            assert_eq!(
                from_disk.source.as_deref(),
                Some(ModelsDevFetcher::DEFAULT_ENDPOINT)
            );
            let _ = std::fs::remove_file(&path);
        }

        #[tokio::test]
        async fn refresh_count_increments_only_on_success() {
            let (cfg, _path) = fetcher_cfg("refresh_count");
            let client = MockClient::with_responses(vec![
                Err("transient"),
                Ok(fetcher_payload().as_bytes().to_vec()),
            ]);
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client));
            assert!(fetcher.force_refresh().await.is_err());
            assert_eq!(fetcher.refresh_count(), 0);
            fetcher.force_refresh().await.unwrap();
            assert_eq!(fetcher.refresh_count(), 1);
        }

        #[tokio::test]
        async fn background_handle_cancel_is_idempotent() {
            let (cfg, _path) = fetcher_cfg("bg_cancel");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = Arc::new(ModelsDevFetcher::new(cfg, Arc::new(client)).with_policy(
                RefreshPolicy::Background {
                    interval: Duration::from_millis(50),
                    on_error: Default::default(),
                },
            ));
            let handle = fetcher.start_background();
            assert!(!handle.is_cancelled());
            handle.cancel();
            handle.cancel();
            assert!(handle.is_cancelled());
            // Dropping handle aborts the task; nothing else to assert.
            drop(handle);
        }

        #[tokio::test]
        async fn endpoint_override_is_propagated() {
            let (cfg, _path) = fetcher_cfg("endpoint");
            let client = MockClient::with_ok(fetcher_payload());
            let fetcher = ModelsDevFetcher::new(cfg, Arc::new(client))
                .with_endpoint("https://example.test/api.json");
            assert_eq!(fetcher.endpoint(), "https://example.test/api.json");
            let reg = fetcher.registry().await.unwrap();
            assert_eq!(reg.source.as_deref(), Some("https://example.test/api.json"));
        }
    }
}
