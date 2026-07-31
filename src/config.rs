//! Configuration types for AI providers

use serde::{Deserialize, Serialize};

use crate::error::{AiError, AiResult, ConfigError};
use crate::retry::RetryConfig;

/// Default URL for `llama.cpp` `llama-server` (matches upstream default).
fn default_llamacpp_url() -> String {
    "http://127.0.0.1:8080".to_string()
}

/// Default URL for vLLM OpenAI-compatible server (matches upstream default).
fn default_vllm_url() -> String {
    "http://127.0.0.1:8000".to_string()
}

/// Available AI provider types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum AiProvider {
    /// Ollama (native API)
    Ollama,
    /// LM Studio (OpenAI-compatible)
    LMStudio,
    /// oobabooga's text-generation-webui (OpenAI-compatible)
    TextGenWebUI,
    /// Kobold.cpp / KoboldAI
    KoboldCpp,
    /// LocalAI (OpenAI-compatible)
    LocalAI,
    /// llama.cpp `llama-server` (OpenAI-compatible).
    ///
    /// Works with both upstream llama.cpp and forks such as PrismML's
    /// `PrismML-Eng/llama.cpp` (which adds the custom `Q1_0` quantization
    /// type used by the Bonsai 1-bit models).
    LlamaCpp,
    /// vLLM OpenAI-compatible high-throughput server.
    ///
    /// vLLM (<https://github.com/vllm-project/vllm>) is a GPU-optimized LLM
    /// serving engine using PagedAttention + continuous batching for
    /// high-concurrency workloads (multi-agent, eval batches, research
    /// pipelines). Loads any HuggingFace transformer repo by ID and exposes
    /// `/v1/chat/completions`, `/v1/models`, `/version`, `/health`,
    /// `/metrics`, and `/v1/load_lora_adapter` (LoRA hot-swap).
    VLLM,
    /// Custom OpenAI-compatible endpoint
    OpenAICompatible { base_url: String },
    /// OpenAI cloud API (requires API key)
    OpenAI,
    /// Anthropic cloud API (requires API key)
    Anthropic,
    /// Google Gemini API (requires API key)
    Gemini,
    /// AWS Bedrock (requires AWS credentials)
    Bedrock { region: String },
    /// Groq cloud API (OpenAI-compatible, requires API key)
    Groq,
    /// Together AI cloud API (OpenAI-compatible, requires API key)
    Together,
    /// Fireworks AI cloud API (OpenAI-compatible, requires API key)
    Fireworks,
    /// DeepSeek cloud API (OpenAI-compatible, requires API key)
    DeepSeek,
    /// Mistral AI cloud API (OpenAI-compatible, requires API key)
    Mistral,
    /// Perplexity cloud API (OpenAI-compatible, requires API key)
    Perplexity,
    /// OpenRouter cloud API (OpenAI-compatible, requires API key)
    OpenRouter,
    /// Azure OpenAI Service (requires endpoint, deployment name, and API key)
    AzureOpenAI {
        endpoint: String,
        deployment: String,
    },
}

impl Default for AiProvider {
    fn default() -> Self {
        AiProvider::Ollama
    }
}

impl AiProvider {
    /// Get a human-readable name for the provider
    pub fn display_name(&self) -> &str {
        match self {
            AiProvider::Ollama => "Ollama",
            AiProvider::LMStudio => "LM Studio",
            AiProvider::TextGenWebUI => "text-generation-webui",
            AiProvider::KoboldCpp => "Kobold.cpp",
            AiProvider::LocalAI => "LocalAI",
            AiProvider::LlamaCpp => "llama.cpp",
            AiProvider::VLLM => "vLLM",
            AiProvider::OpenAICompatible { .. } => "OpenAI Compatible",
            AiProvider::OpenAI => "OpenAI",
            AiProvider::Anthropic => "Anthropic",
            AiProvider::Gemini => "Google Gemini",
            AiProvider::Bedrock { .. } => "AWS Bedrock",
            AiProvider::Groq => "Groq",
            AiProvider::Together => "Together AI",
            AiProvider::Fireworks => "Fireworks AI",
            AiProvider::DeepSeek => "DeepSeek",
            AiProvider::Mistral => "Mistral AI",
            AiProvider::Perplexity => "Perplexity",
            AiProvider::OpenRouter => "OpenRouter",
            AiProvider::AzureOpenAI { .. } => "Azure OpenAI",
        }
    }

    /// Get an icon for the provider (emoji)
    pub fn icon(&self) -> &str {
        match self {
            AiProvider::Ollama => "🦙",
            AiProvider::LMStudio => "🎬",
            AiProvider::TextGenWebUI => "🌐",
            AiProvider::KoboldCpp => "🐉",
            AiProvider::LocalAI => "🤖",
            AiProvider::LlamaCpp => "🦫",
            AiProvider::VLLM => "🚀",
            AiProvider::OpenAICompatible { .. } => "🔌",
            AiProvider::OpenAI => "🧠",
            AiProvider::Anthropic => "🏛️",
            AiProvider::Gemini => "💎",
            AiProvider::Bedrock { .. } => "☁️",
            AiProvider::Groq => "⚡",
            AiProvider::Together => "🤝",
            AiProvider::Fireworks => "🎆",
            AiProvider::DeepSeek => "🔍",
            AiProvider::Mistral => "🌬️",
            AiProvider::Perplexity => "🔮",
            AiProvider::OpenRouter => "🔀",
            AiProvider::AzureOpenAI { .. } => "☁️",
        }
    }

    /// Check if this provider uses OpenAI-compatible API
    pub fn is_openai_compatible(&self) -> bool {
        matches!(
            self,
            AiProvider::LMStudio
                | AiProvider::TextGenWebUI
                | AiProvider::LocalAI
                | AiProvider::LlamaCpp
                | AiProvider::VLLM
                | AiProvider::OpenAICompatible { .. }
                | AiProvider::OpenAI
                | AiProvider::Groq
                | AiProvider::Together
                | AiProvider::Fireworks
                | AiProvider::DeepSeek
                | AiProvider::Mistral
                | AiProvider::Perplexity
                | AiProvider::OpenRouter
                | AiProvider::AzureOpenAI { .. }
        )
    }

    /// Check if this is a cloud provider requiring an API key.
    pub fn is_cloud(&self) -> bool {
        matches!(
            self,
            AiProvider::OpenAI
                | AiProvider::Anthropic
                | AiProvider::Gemini
                | AiProvider::Bedrock { .. }
                | AiProvider::Groq
                | AiProvider::Together
                | AiProvider::Fireworks
                | AiProvider::DeepSeek
                | AiProvider::Mistral
                | AiProvider::Perplexity
                | AiProvider::OpenRouter
                | AiProvider::AzureOpenAI { .. }
        )
    }
}

/// AI Assistant configuration
#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AiConfig {
    /// Current provider
    pub provider: AiProvider,
    /// Currently selected model name
    pub selected_model: String,
    /// Ollama API URL
    pub ollama_url: String,
    /// LM Studio API URL
    pub lm_studio_url: String,
    /// text-generation-webui API URL
    pub text_gen_webui_url: String,
    /// Kobold.cpp API URL
    pub kobold_url: String,
    /// LocalAI API URL
    pub local_ai_url: String,
    /// llama.cpp `llama-server` URL (and PrismML fork).
    /// Default port for upstream llama-server is 8080.
    #[serde(default = "default_llamacpp_url")]
    pub llamacpp_url: String,
    /// vLLM OpenAI-compatible server URL.
    /// Default port for vLLM is 8000.
    #[serde(default = "default_vllm_url")]
    pub vllm_url: String,
    /// Custom OpenAI-compatible URL
    pub custom_url: String,
    /// API key for cloud providers (OpenAI, Anthropic).
    /// Falls back to env vars OPENAI_API_KEY / ANTHROPIC_API_KEY if empty.
    #[serde(default)]
    pub api_key: String,
    /// Maximum number of history messages to include in context
    pub max_history_messages: usize,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: f32,
    /// Retry configuration for network operations.
    /// Skipped during serialization; defaults to `RetryConfig::default()` on deserialization.
    #[serde(skip)]
    pub retry_config: RetryConfig,
    /// Optional path to a multimodal projector (`mmproj.gguf`) that pairs
    /// with the selected base model when running `llama-server` /
    /// `koboldcpp`. The library does not load it itself — the server
    /// must be started with `--mmproj <path>`. The field exists so the
    /// CLI / GUI can persist the user's choice and so future embedded
    /// launchers know which projector to pass through. Validated via
    /// [`crate::mmproj::MultimodalProjector::from_path`] at config load
    /// time when the `vision` feature is enabled.
    #[serde(default)]
    pub mmproj_path: Option<std::path::PathBuf>,
    /// Override for the Ollama context window (`num_ctx`) in tokens.
    ///
    /// `None` (default) auto-sizes `num_ctx` to fit the prompt, capped at a
    /// VRAM-safe ceiling and at the model's real window. Set an explicit value
    /// to request a larger window when you have the VRAM (e.g. to keep a big
    /// injected knowledge document fully in context) — but note that an
    /// over-large `num_ctx` allocates a proportionally large KV cache and can
    /// OOM/crash the Ollama server. Capped at the model's real context size.
    #[serde(default)]
    pub ollama_num_ctx: Option<usize>,
    /// Ollama embedding model for **semantic** knowledge retrieval (e.g.
    /// `"nomic-embed-text"`). When set and reachable, large-knowledge / fresh-
    /// context retrieval ranks passages by embedding similarity (handles
    /// paraphrase / synonyms); when `None` or unreachable it falls back to the
    /// always-available lexical term-overlap ranker.
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Sampling seed, for **reproducible** generation.
    ///
    /// `None` (default) lets the backend pick a fresh seed per request, so the
    /// same prompt can yield different completions. Setting a seed makes
    /// sampling deterministic: the same prompt, model and options produce
    /// byte-identical output across runs *at any temperature*.
    ///
    /// This is the right way to get reproducibility — reaching for
    /// `temperature = 0.0` instead is both weaker (it only removes the sampling
    /// randomness, not the seed's effect on tie-breaks) and, on some
    /// Ollama/llama.cpp builds, actively dangerous: near-greedy sampling can
    /// abort the runner mid-request (`Assertion failed: found` in
    /// `llama-sampling.cpp`), which surfaces to the caller as a connection
    /// failure. A fixed seed at a normal temperature avoids that path.
    ///
    /// Currently sent to Ollama; providers that do not accept a seed ignore it.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl std::fmt::Debug for AiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AiConfig")
            .field("provider", &self.provider)
            .field("selected_model", &self.selected_model)
            .field("ollama_url", &self.ollama_url)
            .field("lm_studio_url", &self.lm_studio_url)
            .field("text_gen_webui_url", &self.text_gen_webui_url)
            .field("kobold_url", &self.kobold_url)
            .field("local_ai_url", &self.local_ai_url)
            .field("llamacpp_url", &self.llamacpp_url)
            .field("vllm_url", &self.vllm_url)
            .field("custom_url", &self.custom_url)
            .field(
                "api_key",
                &if self.api_key.is_empty() {
                    "<empty>"
                } else {
                    "<REDACTED>"
                },
            )
            .field("max_history_messages", &self.max_history_messages)
            .field("temperature", &self.temperature)
            .field(
                "mmproj_path",
                &self
                    .mmproj_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().into_owned()),
            )
            .field("ollama_num_ctx", &self.ollama_num_ctx)
            .field("embedding_model", &self.embedding_model)
            .field("seed", &self.seed)
            .finish()
    }
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            provider: AiProvider::Ollama,
            selected_model: String::new(),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            lm_studio_url: "http://127.0.0.1:1234".to_string(),
            text_gen_webui_url: "http://127.0.0.1:5000".to_string(),
            kobold_url: "http://127.0.0.1:5001".to_string(),
            local_ai_url: "http://127.0.0.1:8080".to_string(),
            llamacpp_url: default_llamacpp_url(),
            vllm_url: default_vllm_url(),
            custom_url: String::new(),
            api_key: String::new(),
            max_history_messages: 20,
            temperature: 0.7,
            retry_config: RetryConfig::default(),
            mmproj_path: None,
            ollama_num_ctx: None,
            embedding_model: None,
            seed: None,
        }
    }
}

impl AiConfig {
    /// Get the base URL for the current provider
    pub fn get_base_url(&self) -> String {
        self.get_provider_url(&self.provider)
    }

    /// Validate the configured `mmproj_path`, if any. Returns:
    ///
    /// * `None` — no path configured.
    /// * `Some(Ok(_))` — path validated successfully.
    /// * `Some(Err(_))` — path was set but failed validation. Caller
    ///   decides whether to warn-log, surface to UI, or refuse to start.
    ///
    /// Validation is intentionally non-fatal at config load: a stale
    /// path in a config file should not stop the assistant from running
    /// text-only requests. See [`crate::mmproj::MultimodalProjector::from_path`]
    /// for the full check pipeline.
    #[cfg(feature = "vision")]
    pub fn validated_mmproj(
        &self,
    ) -> Option<Result<crate::mmproj::MultimodalProjector, crate::mmproj::MmprojValidationError>>
    {
        self.mmproj_path
            .as_ref()
            .map(crate::mmproj::MultimodalProjector::from_path)
    }

    /// Get URL for a specific provider
    pub fn get_provider_url(&self, provider: &AiProvider) -> String {
        match provider {
            AiProvider::Ollama => self.ollama_url.clone(),
            AiProvider::LMStudio => self.lm_studio_url.clone(),
            AiProvider::TextGenWebUI => self.text_gen_webui_url.clone(),
            AiProvider::KoboldCpp => self.kobold_url.clone(),
            AiProvider::LocalAI => self.local_ai_url.clone(),
            AiProvider::LlamaCpp => self.llamacpp_url.clone(),
            AiProvider::VLLM => self.vllm_url.clone(),
            AiProvider::OpenAICompatible { base_url } => base_url.clone(),
            AiProvider::OpenAI => "https://api.openai.com".to_string(),
            AiProvider::Anthropic => "https://api.anthropic.com".to_string(),
            AiProvider::Gemini => "https://generativelanguage.googleapis.com".to_string(),
            AiProvider::Bedrock { ref region } => {
                format!("https://bedrock-runtime.{}.amazonaws.com", region)
            }
            AiProvider::Groq => "https://api.groq.com/openai".to_string(),
            AiProvider::Together => "https://api.together.xyz".to_string(),
            AiProvider::Fireworks => "https://api.fireworks.ai/inference".to_string(),
            AiProvider::DeepSeek => "https://api.deepseek.com".to_string(),
            AiProvider::Mistral => "https://api.mistral.ai".to_string(),
            AiProvider::Perplexity => "https://api.perplexity.ai".to_string(),
            AiProvider::OpenRouter => "https://openrouter.ai/api".to_string(),
            AiProvider::AzureOpenAI { ref endpoint, .. } => endpoint.clone(),
        }
    }

    /// Get the API key for the current cloud provider.
    ///
    /// Returns the configured `api_key` if non-empty, otherwise falls back
    /// to the appropriate environment variable:
    /// - `OPENAI_API_KEY` for OpenAI
    /// - `ANTHROPIC_API_KEY` for Anthropic
    ///
    /// Returns `None` for local providers or if no key is found.
    pub fn get_api_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        match &self.provider {
            AiProvider::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
            AiProvider::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
            AiProvider::Gemini => std::env::var("GOOGLE_API_KEY")
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .ok(),
            AiProvider::Bedrock { .. } => {
                // AWS Bedrock uses AWS credentials (access key + secret), not a single API key.
                // Return access key if available.
                std::env::var("AWS_ACCESS_KEY_ID").ok()
            }
            AiProvider::Groq => std::env::var("GROQ_API_KEY").ok(),
            AiProvider::Together => std::env::var("TOGETHER_API_KEY").ok(),
            AiProvider::Fireworks => std::env::var("FIREWORKS_API_KEY").ok(),
            AiProvider::DeepSeek => std::env::var("DEEPSEEK_API_KEY").ok(),
            AiProvider::Mistral => std::env::var("MISTRAL_API_KEY").ok(),
            AiProvider::Perplexity => std::env::var("PERPLEXITY_API_KEY").ok(),
            AiProvider::OpenRouter => std::env::var("OPENROUTER_API_KEY").ok(),
            AiProvider::AzureOpenAI { .. } => std::env::var("AZURE_OPENAI_API_KEY").ok(),
            _ => None,
        }
    }

    // -------------------------------------------------------------------------
    // Fluent builder helpers (V161)
    //
    // `AiConfig` keeps all fields `pub` and its `Default`, so direct field
    // assignment still works exactly as before. These additive, chainable
    // setters make the common "start from default, tweak a few things" path
    // read nicely and pair with `validate()` for a fail-fast check.
    // -------------------------------------------------------------------------

    /// Choose the provider (chainable). Equivalent to assigning `.provider`.
    pub fn with_provider(mut self, provider: AiProvider) -> Self {
        self.provider = provider;
        self
    }

    /// Choose the model name (chainable).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.selected_model = model.into();
        self
    }

    /// Set the cloud API key (chainable). Overrides the env-var fallback used
    /// by [`get_api_key`](Self::get_api_key).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set the generation temperature (chainable). Range is checked by
    /// [`validate`](Self::validate), not here.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set how many history messages to keep in context (chainable).
    pub fn with_max_history_messages(mut self, max_history_messages: usize) -> Self {
        self.max_history_messages = max_history_messages;
        self
    }

    /// Set the retry policy for network operations (chainable).
    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    /// Pin the sampling seed for reproducible generation (chainable).
    ///
    /// Prefer this over `temperature = 0.0` when you need repeatable output —
    /// see [`seed`](Self::seed) for why.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validate the configuration, catching mistakes that would otherwise only
    /// surface as confusing runtime failures:
    ///
    /// * `temperature` outside the documented `0.0..=2.0` range,
    /// * a cloud provider selected with no API key (neither configured nor
    ///   reachable via the provider's `*_API_KEY` environment variable),
    /// * a provider that resolves to an empty base URL.
    ///
    /// This is opt-in — call it after building a config if you want a
    /// fail-fast check. Returns `Ok(())` when the config is usable as-is.
    pub fn validate(&self) -> AiResult<()> {
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(AiError::Config(ConfigError::InvalidValue {
                field: "temperature".to_string(),
                value: self.temperature.to_string(),
                expected: "a value in 0.0..=2.0".to_string(),
            }));
        }
        if self.provider.is_cloud() && self.get_api_key().is_none() {
            return Err(AiError::Config(ConfigError::MissingValue {
                field: "api_key".to_string(),
                description: format!(
                    "provider '{}' is a cloud provider and needs an API key \
                     (set AiConfig.api_key or the provider's *_API_KEY env var)",
                    self.provider.display_name()
                ),
            }));
        }
        if self.get_base_url().trim().is_empty() {
            return Err(AiError::Config(ConfigError::MissingValue {
                field: "base_url".to_string(),
                description: format!(
                    "provider '{}' resolves to an empty base URL",
                    self.provider.display_name()
                ),
            }));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_provider_defaults() {
        let provider = AiProvider::default();
        assert_eq!(provider, AiProvider::Ollama);
    }

    #[test]
    fn test_config_builder_fluent() {
        let cfg = AiConfig::default()
            .with_provider(AiProvider::Ollama)
            .with_model("llama3")
            .with_temperature(0.5)
            .with_max_history_messages(10);
        assert_eq!(cfg.selected_model, "llama3");
        assert_eq!(cfg.temperature, 0.5);
        assert_eq!(cfg.max_history_messages, 10);
        // A local provider with a default URL validates cleanly.
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_seed_defaults_to_unset() {
        // Default must stay random-per-request: pinning a seed by accident would
        // make every caller's output silently repeat.
        assert_eq!(AiConfig::default().seed, None);

        let cfg = AiConfig::default().with_seed(42);
        assert_eq!(cfg.seed, Some(42));
        // A seed is orthogonal to temperature — reproducibility does not require
        // giving up sampling.
        assert!(cfg.with_temperature(0.5).validate().is_ok());
    }

    #[test]
    fn test_config_validate_rejects_bad_temperature() {
        assert!(AiConfig::default()
            .with_temperature(5.0)
            .validate()
            .is_err());
        assert!(AiConfig::default()
            .with_temperature(-0.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_validate_cloud_requires_key() {
        let cfg = AiConfig::default()
            .with_provider(AiProvider::OpenAI)
            .with_api_key("");
        // Only assert the failure when the developer's env doesn't already
        // provide a key (otherwise the env fallback legitimately satisfies it).
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert!(cfg.validate().is_err());
        }
        // With a key set, it must validate regardless of env.
        let cfg_keyed = AiConfig::default()
            .with_provider(AiProvider::OpenAI)
            .with_api_key("sk-test");
        assert!(cfg_keyed.validate().is_ok());
    }

    #[test]
    fn test_ai_provider_display_names() {
        assert_eq!(AiProvider::Ollama.display_name(), "Ollama");
        assert_eq!(AiProvider::LMStudio.display_name(), "LM Studio");
        assert_eq!(
            AiProvider::TextGenWebUI.display_name(),
            "text-generation-webui"
        );
        assert_eq!(AiProvider::KoboldCpp.display_name(), "Kobold.cpp");
        assert_eq!(AiProvider::LocalAI.display_name(), "LocalAI");
        assert_eq!(AiProvider::LlamaCpp.display_name(), "llama.cpp");
        let custom = AiProvider::OpenAICompatible {
            base_url: "http://custom".to_string(),
        };
        assert_eq!(custom.display_name(), "OpenAI Compatible");
    }

    #[test]
    fn test_ai_provider_openai_compatibility() {
        assert!(!AiProvider::Ollama.is_openai_compatible());
        assert!(AiProvider::LMStudio.is_openai_compatible());
        assert!(AiProvider::TextGenWebUI.is_openai_compatible());
        assert!(!AiProvider::KoboldCpp.is_openai_compatible());
        assert!(AiProvider::LocalAI.is_openai_compatible());
        assert!(AiProvider::LlamaCpp.is_openai_compatible());
        let custom = AiProvider::OpenAICompatible {
            base_url: "http://x".to_string(),
        };
        assert!(custom.is_openai_compatible());
    }

    #[test]
    fn test_llamacpp_default_url() {
        let config = AiConfig::default();
        assert_eq!(config.llamacpp_url, "http://127.0.0.1:8080");
    }

    #[test]
    fn test_llamacpp_get_provider_url() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::LlamaCpp;
        assert_eq!(config.get_base_url(), "http://127.0.0.1:8080");
        config.llamacpp_url = "http://127.0.0.1:9999".to_string();
        assert_eq!(config.get_base_url(), "http://127.0.0.1:9999");
    }

    #[test]
    fn test_llamacpp_not_cloud() {
        assert!(!AiProvider::LlamaCpp.is_cloud());
    }

    #[test]
    fn test_ai_config_defaults() {
        let config = AiConfig::default();
        assert_eq!(config.provider, AiProvider::Ollama);
        assert_eq!(config.ollama_url, "http://127.0.0.1:11434");
        assert_eq!(config.lm_studio_url, "http://127.0.0.1:1234");
        assert_eq!(config.max_history_messages, 20);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ai_config_retry_config_default() {
        let config = AiConfig::default();
        // Default retry: 3 max retries, exponential backoff
        assert_eq!(config.retry_config.max_retries, 3);
        assert!(config.retry_config.add_jitter);
    }

    #[test]
    fn test_ai_config_get_base_url() {
        let config = AiConfig::default();
        assert_eq!(config.get_base_url(), "http://127.0.0.1:11434");

        let mut config2 = AiConfig::default();
        config2.provider = AiProvider::LMStudio;
        assert_eq!(config2.get_base_url(), "http://127.0.0.1:1234");

        let mut config3 = AiConfig::default();
        config3.provider = AiProvider::OpenAICompatible {
            base_url: "http://my-api:9000".to_string(),
        };
        assert_eq!(config3.get_base_url(), "http://my-api:9000");
    }

    #[test]
    fn test_cloud_providers() {
        assert!(AiProvider::OpenAI.is_cloud());
        assert!(AiProvider::Anthropic.is_cloud());
        assert!(!AiProvider::Ollama.is_cloud());
        assert!(!AiProvider::LMStudio.is_cloud());
    }

    #[test]
    fn test_provider_icons() {
        assert!(!AiProvider::Ollama.icon().is_empty());
        assert!(!AiProvider::OpenAI.icon().is_empty());
    }

    #[test]
    fn test_all_providers_have_display_names() {
        let providers = [
            AiProvider::Ollama,
            AiProvider::LMStudio,
            AiProvider::OpenAI,
            AiProvider::Anthropic,
            AiProvider::Gemini,
        ];
        for p in &providers {
            assert!(!p.display_name().is_empty());
        }
    }

    #[test]
    fn test_config_temperature_default() {
        let config = AiConfig::default();
        assert!(config.temperature >= 0.0 && config.temperature <= 2.0);
    }

    #[test]
    fn test_azure_openai_display_name() {
        let p = AiProvider::AzureOpenAI {
            endpoint: "https://x.openai.azure.com".into(),
            deployment: "gpt-4o".into(),
        };
        assert_eq!(p.display_name(), "Azure OpenAI");
    }

    #[test]
    fn test_azure_openai_is_cloud() {
        let p = AiProvider::AzureOpenAI {
            endpoint: "https://x.openai.azure.com".into(),
            deployment: "gpt-4o".into(),
        };
        assert!(p.is_cloud());
    }

    #[test]
    fn test_azure_openai_is_openai_compatible() {
        let p = AiProvider::AzureOpenAI {
            endpoint: "https://x.openai.azure.com".into(),
            deployment: "gpt-4o".into(),
        };
        assert!(p.is_openai_compatible());
    }

    #[test]
    fn test_vllm_default_url() {
        let config = AiConfig::default();
        assert_eq!(config.vllm_url, "http://127.0.0.1:8000");
    }

    #[test]
    fn test_vllm_get_provider_url() {
        let mut config = AiConfig::default();
        config.provider = AiProvider::VLLM;
        assert_eq!(config.get_base_url(), "http://127.0.0.1:8000");
        config.vllm_url = "http://gpu-box:8000".to_string();
        assert_eq!(config.get_base_url(), "http://gpu-box:8000");
    }

    #[test]
    fn test_vllm_is_openai_compatible() {
        assert!(AiProvider::VLLM.is_openai_compatible());
    }

    #[test]
    fn test_vllm_is_not_cloud() {
        assert!(!AiProvider::VLLM.is_cloud());
    }

    #[test]
    fn test_vllm_display_name_and_icon() {
        assert_eq!(AiProvider::VLLM.display_name(), "vLLM");
        assert!(!AiProvider::VLLM.icon().is_empty());
    }

    #[test]
    fn test_azure_openai_get_api_key_env_fallback() {
        let config = AiConfig {
            provider: AiProvider::AzureOpenAI {
                endpoint: "https://x.openai.azure.com".into(),
                deployment: "gpt-4o".into(),
            },
            api_key: "my-azure-key".to_string(),
            ..Default::default()
        };
        assert_eq!(config.get_api_key(), Some("my-azure-key".to_string()));
    }
}
