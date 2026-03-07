//! Configuration file support for loading/saving AI assistant settings
//!
//! Supports TOML and JSON configuration files with automatic format detection.
//!
//! # Example TOML configuration
//!
//! ```toml
//! [provider]
//! type = "ollama"
//! model = "llama2"
//!
//! [urls]
//! ollama = "http://localhost:11434"
//! lm_studio = "http://localhost:1234"
//!
//! [generation]
//! temperature = 0.7
//! max_history = 20
//!
//! [rag]
//! enabled = true
//! knowledge_tokens = 2000
//! conversation_tokens = 1500
//! ```
//!
//! # Example JSON configuration
//!
//! ```json
//! {
//!   "provider": {
//!     "type": "ollama",
//!     "model": "llama2"
//!   },
//!   "urls": {
//!     "ollama": "http://localhost:11434"
//!   },
//!   "generation": {
//!     "temperature": 0.7
//!   }
//! }
//! ```

use crate::config::{AiConfig, AiProvider};
use crate::error::{AiError, ConfigError, IoError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Complete configuration file structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigFile {
    /// Provider configuration
    #[serde(default)]
    pub provider: ProviderConfig,

    /// URL configurations for all providers
    #[serde(default)]
    pub urls: UrlConfig,

    /// Generation settings
    #[serde(default)]
    pub generation: GenerationConfig,

    /// RAG configuration
    #[serde(default)]
    pub rag: RagFileConfig,

    /// Security settings
    #[serde(default)]
    pub security: SecurityConfig,

    /// Caching settings
    #[serde(default)]
    pub cache: CacheConfig,

    /// Logging settings
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Container/Docker management settings
    #[serde(default)]
    pub containers: ContainersConfig,
}

/// Provider configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type: "ollama", "lmstudio", "textgenwebui", "kobold", "localai", "openai_compatible"
    #[serde(rename = "type", default)]
    pub provider_type: String,

    /// Selected model name
    #[serde(default)]
    pub model: String,

    /// Custom base URL for OpenAI-compatible providers
    #[serde(default)]
    pub custom_url: Option<String>,

    /// API key (for providers that require it)
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: "ollama".to_string(),
            model: String::new(),
            custom_url: None,
            api_key: None,
        }
    }
}

/// URL configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlConfig {
    /// Ollama API URL
    #[serde(default = "default_ollama_url")]
    pub ollama: String,

    /// LM Studio API URL
    #[serde(default = "default_lm_studio_url")]
    pub lm_studio: String,

    /// text-generation-webui API URL
    #[serde(default = "default_text_gen_url")]
    pub text_gen_webui: String,

    /// Kobold.cpp API URL
    #[serde(default = "default_kobold_url")]
    pub kobold: String,

    /// LocalAI API URL
    #[serde(default = "default_local_ai_url")]
    pub local_ai: String,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_lm_studio_url() -> String {
    "http://localhost:1234".to_string()
}
fn default_text_gen_url() -> String {
    "http://localhost:5000".to_string()
}
fn default_kobold_url() -> String {
    "http://localhost:5001".to_string()
}
fn default_local_ai_url() -> String {
    "http://localhost:8080".to_string()
}

impl Default for UrlConfig {
    fn default() -> Self {
        Self {
            ollama: default_ollama_url(),
            lm_studio: default_lm_studio_url(),
            text_gen_webui: default_text_gen_url(),
            kobold: default_kobold_url(),
            local_ai: default_local_ai_url(),
        }
    }
}

/// Generation configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum history messages to include
    #[serde(default = "default_max_history")]
    pub max_history: usize,

    /// Top-p sampling parameter
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    #[serde(default)]
    pub top_k: Option<i32>,

    /// Repeat penalty
    #[serde(default)]
    pub repeat_penalty: Option<f32>,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_history() -> usize {
    20
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_history: default_max_history(),
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
        }
    }
}

/// RAG configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagFileConfig {
    /// Enable knowledge RAG
    #[serde(default)]
    pub knowledge_enabled: bool,

    /// Enable conversation RAG
    #[serde(default)]
    pub conversation_enabled: bool,

    /// Maximum tokens for knowledge context
    #[serde(default = "default_knowledge_tokens")]
    pub knowledge_tokens: usize,

    /// Maximum tokens for conversation context
    #[serde(default = "default_conversation_tokens")]
    pub conversation_tokens: usize,

    /// Number of top chunks to retrieve
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum relevance score
    #[serde(default = "default_min_relevance")]
    pub min_relevance: f32,

    /// Enable append-only mode for knowledge
    #[serde(default)]
    pub append_only: bool,

    /// Database path (relative to config file or absolute)
    #[serde(default)]
    pub database_path: Option<String>,

    /// Hybrid search configuration
    #[serde(default)]
    pub hybrid: Option<HybridConfig>,
}

fn default_knowledge_tokens() -> usize {
    2000
}
fn default_conversation_tokens() -> usize {
    1500
}
fn default_top_k() -> usize {
    5
}
fn default_min_relevance() -> f32 {
    0.1
}

impl Default for RagFileConfig {
    fn default() -> Self {
        Self {
            knowledge_enabled: false,
            conversation_enabled: false,
            knowledge_tokens: default_knowledge_tokens(),
            conversation_tokens: default_conversation_tokens(),
            top_k: default_top_k(),
            min_relevance: default_min_relevance(),
            append_only: false,
            database_path: None,
            hybrid: None,
        }
    }
}

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Enable semantic search
    #[serde(default)]
    pub semantic_enabled: bool,

    /// BM25 weight (0.0 - 1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,

    /// Semantic weight (0.0 - 1.0)
    #[serde(default = "default_semantic_weight")]
    pub semantic_weight: f32,
}

fn default_bm25_weight() -> f32 {
    0.6
}
fn default_semantic_weight() -> f32 {
    0.4
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            semantic_enabled: false,
            bm25_weight: default_bm25_weight(),
            semantic_weight: default_semantic_weight(),
        }
    }
}

/// Security configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable rate limiting
    #[serde(default)]
    pub rate_limit_enabled: bool,

    /// Requests per minute limit
    #[serde(default = "default_rate_limit")]
    pub requests_per_minute: u32,

    /// Enable input sanitization
    #[serde(default = "default_true")]
    pub sanitize_input: bool,

    /// Enable audit logging
    #[serde(default)]
    pub audit_logging: bool,
}

fn default_rate_limit() -> u32 {
    60
}
fn default_true() -> bool {
    true
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            rate_limit_enabled: false,
            requests_per_minute: default_rate_limit(),
            sanitize_input: default_true(),
            audit_logging: false,
        }
    }
}

/// Cache configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable response caching
    #[serde(default)]
    pub enabled: bool,

    /// Maximum cache entries
    #[serde(default = "default_cache_size")]
    pub max_entries: usize,

    /// TTL in seconds
    #[serde(default = "default_cache_ttl")]
    pub ttl_seconds: u64,

    /// Enable semantic caching
    #[serde(default)]
    pub semantic_enabled: bool,

    /// Similarity threshold for cache hits (0.0 - 1.0)
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
}

fn default_cache_size() -> usize {
    100
}
fn default_cache_ttl() -> u64 {
    3600
}
fn default_similarity_threshold() -> f32 {
    0.95
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: default_cache_size(),
            ttl_seconds: default_cache_ttl(),
            semantic_enabled: false,
            similarity_threshold: default_similarity_threshold(),
        }
    }
}

/// Logging configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level: "trace", "debug", "info", "warn", "error"
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log file path (optional)
    #[serde(default)]
    pub file: Option<String>,

    /// Log metrics
    #[serde(default)]
    pub metrics: bool,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            metrics: false,
        }
    }
}

/// Container/Docker management configuration.
///
/// Controls whether Docker container management is available at runtime,
/// both in the CLI REPL (`enabled`) and via the MCP endpoint (`mcp_enabled`).
/// Both default to `false` — Docker features must be explicitly opted into.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainersConfig {
    /// Whether container management commands are enabled (default: false).
    #[serde(default)]
    pub enabled: bool,

    /// Default timeout for Docker operations in seconds (default: 60).
    #[serde(default = "default_container_timeout")]
    pub default_timeout_secs: u64,

    /// Allowed Docker images (empty = all images allowed).
    #[serde(default)]
    pub allowed_images: Vec<String>,

    /// Whether to expose Docker tools via the MCP endpoint (default: false).
    #[serde(default)]
    pub mcp_enabled: bool,
}

fn default_container_timeout() -> u64 {
    60
}

impl Default for ContainersConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_timeout_secs: default_container_timeout(),
            allowed_images: Vec::new(),
            mcp_enabled: false,
        }
    }
}

/// Configuration file format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfigFormat {
    /// TOML format
    Toml,
    /// JSON format
    Json,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("toml") => Some(ConfigFormat::Toml),
            Some("json") => Some(ConfigFormat::Json),
            _ => None,
        }
    }

    /// Detect format from content
    ///
    /// Looks for JSON indicators ({, [) at the start after trimming.
    /// TOML section headers like `[section]` are distinguished from JSON arrays.
    pub fn from_content(content: &str) -> Self {
        let trimmed = content.trim();
        if trimmed.starts_with('{') {
            // Definitely JSON object
            ConfigFormat::Json
        } else if trimmed.starts_with('[') {
            // Could be JSON array or TOML section header
            // TOML sections have format [name] followed by newline or end
            // JSON arrays have format [value, ...]
            if let Some(end_bracket) = trimmed.find(']') {
                let between = &trimmed[1..end_bracket];
                // TOML section names are simple identifiers (letters, numbers, underscore, dash, dot)
                // JSON arrays contain values, quotes, or commas
                let is_toml_section = between
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ' ');
                if is_toml_section && !between.contains(',') && !between.contains('"') {
                    return ConfigFormat::Toml;
                }
            }
            ConfigFormat::Json
        } else {
            ConfigFormat::Toml
        }
    }
}

impl ConfigFile {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a file (auto-detect format)
    pub fn load(path: &Path) -> Result<Self, AiError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| IoError::with_path("read", path.display().to_string(), e.to_string()))?;

        let format =
            ConfigFormat::from_path(path).unwrap_or_else(|| ConfigFormat::from_content(&content));

        Self::parse(&content, format)
    }

    /// Parse configuration from a string
    pub fn parse(content: &str, format: ConfigFormat) -> Result<Self, AiError> {
        match format {
            ConfigFormat::Toml => {
                // Simple TOML parser (basic implementation without toml crate)
                Self::parse_toml(content)
            }
            ConfigFormat::Json => serde_json::from_str(content).map_err(|e| {
                ConfigError::LoadFailed {
                    path: "<string>".to_string(),
                    reason: e.to_string(),
                }
                .into()
            }),
        }
    }

    /// Save configuration to a file
    pub fn save(&self, path: &Path) -> Result<(), AiError> {
        let format = ConfigFormat::from_path(path).unwrap_or(ConfigFormat::Json);
        let content = self.serialize(format)?;

        std::fs::write(path, content)
            .map_err(|e| IoError::with_path("write", path.display().to_string(), e.to_string()))?;

        Ok(())
    }

    /// Serialize configuration to string
    pub fn serialize(&self, format: ConfigFormat) -> Result<String, AiError> {
        match format {
            ConfigFormat::Toml => Ok(self.to_toml()),
            ConfigFormat::Json => serde_json::to_string_pretty(self).map_err(|e| {
                ConfigError::SaveFailed {
                    path: "<string>".to_string(),
                    reason: e.to_string(),
                }
                .into()
            }),
        }
    }

    /// Convert to AiConfig
    pub fn to_ai_config(&self) -> AiConfig {
        let provider = match self.provider.provider_type.to_lowercase().as_str() {
            "ollama" => AiProvider::Ollama,
            "lmstudio" | "lm_studio" => AiProvider::LMStudio,
            "textgenwebui" | "text_gen_webui" => AiProvider::TextGenWebUI,
            "kobold" | "koboldcpp" => AiProvider::KoboldCpp,
            "localai" | "local_ai" => AiProvider::LocalAI,
            "openai_compatible" => AiProvider::OpenAICompatible {
                base_url: self.provider.custom_url.clone().unwrap_or_default(),
            },
            "openai" => AiProvider::OpenAI,
            "anthropic" => AiProvider::Anthropic,
            "gemini" | "google" => AiProvider::Gemini,
            s if s.starts_with("bedrock") => {
                let region = self
                    .provider
                    .custom_url
                    .as_deref()
                    .unwrap_or("us-east-1")
                    .to_string();
                AiProvider::Bedrock { region }
            }
            "groq" => AiProvider::Groq,
            "together" => AiProvider::Together,
            "fireworks" => AiProvider::Fireworks,
            "deepseek" => AiProvider::DeepSeek,
            "mistral" => AiProvider::Mistral,
            "perplexity" => AiProvider::Perplexity,
            "openrouter" => AiProvider::OpenRouter,
            _ => AiProvider::Ollama,
        };

        AiConfig {
            provider,
            selected_model: self.provider.model.clone(),
            ollama_url: self.urls.ollama.clone(),
            lm_studio_url: self.urls.lm_studio.clone(),
            text_gen_webui_url: self.urls.text_gen_webui.clone(),
            kobold_url: self.urls.kobold.clone(),
            local_ai_url: self.urls.local_ai.clone(),
            custom_url: self.provider.custom_url.clone().unwrap_or_default(),
            api_key: self.provider.api_key.clone().unwrap_or_default(),
            max_history_messages: self.generation.max_history,
            temperature: self.generation.temperature,
            retry_config: crate::retry::RetryConfig::default(),
        }
    }

    /// Create from AiConfig
    pub fn from_ai_config(config: &AiConfig) -> Self {
        let (provider_type, custom_url) = match &config.provider {
            AiProvider::Ollama => ("ollama".to_string(), None),
            AiProvider::LMStudio => ("lmstudio".to_string(), None),
            AiProvider::TextGenWebUI => ("textgenwebui".to_string(), None),
            AiProvider::KoboldCpp => ("kobold".to_string(), None),
            AiProvider::LocalAI => ("localai".to_string(), None),
            AiProvider::OpenAICompatible { base_url } => {
                ("openai_compatible".to_string(), Some(base_url.clone()))
            }
            AiProvider::OpenAI => ("openai".to_string(), None),
            AiProvider::Anthropic => ("anthropic".to_string(), None),
            AiProvider::Gemini => ("gemini".to_string(), None),
            AiProvider::Bedrock { ref region } => ("bedrock".to_string(), Some(region.clone())),
            AiProvider::Groq => ("groq".to_string(), None),
            AiProvider::Together => ("together".to_string(), None),
            AiProvider::Fireworks => ("fireworks".to_string(), None),
            AiProvider::DeepSeek => ("deepseek".to_string(), None),
            AiProvider::Mistral => ("mistral".to_string(), None),
            AiProvider::Perplexity => ("perplexity".to_string(), None),
            AiProvider::OpenRouter => ("openrouter".to_string(), None),
        };

        Self {
            provider: ProviderConfig {
                provider_type,
                model: config.selected_model.clone(),
                custom_url,
                api_key: None,
            },
            urls: UrlConfig {
                ollama: config.ollama_url.clone(),
                lm_studio: config.lm_studio_url.clone(),
                text_gen_webui: config.text_gen_webui_url.clone(),
                kobold: config.kobold_url.clone(),
                local_ai: config.local_ai_url.clone(),
            },
            generation: GenerationConfig {
                temperature: config.temperature,
                max_history: config.max_history_messages,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Parse TOML content (basic implementation)
    fn parse_toml(content: &str) -> Result<Self, AiError> {
        let mut config = ConfigFile::default();
        let mut current_section = String::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Section header
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len() - 1].to_string();
                continue;
            }

            // Key-value pair
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let value = line[eq_pos + 1..].trim();
                let value = Self::parse_toml_value(value);

                match current_section.as_str() {
                    "provider" => match key {
                        "type" => config.provider.provider_type = value,
                        "model" => config.provider.model = value,
                        "custom_url" => config.provider.custom_url = Some(value),
                        "api_key" => config.provider.api_key = Some(value),
                        _ => {}
                    },
                    "urls" => match key {
                        "ollama" => config.urls.ollama = value,
                        "lm_studio" => config.urls.lm_studio = value,
                        "text_gen_webui" => config.urls.text_gen_webui = value,
                        "kobold" => config.urls.kobold = value,
                        "local_ai" => config.urls.local_ai = value,
                        _ => {}
                    },
                    "generation" => match key {
                        "temperature" => {
                            config.generation.temperature = value.parse().unwrap_or(0.7)
                        }
                        "max_history" => {
                            config.generation.max_history = value.parse().unwrap_or(20)
                        }
                        "top_p" => config.generation.top_p = value.parse().ok(),
                        "top_k" => config.generation.top_k = value.parse().ok(),
                        "repeat_penalty" => config.generation.repeat_penalty = value.parse().ok(),
                        "max_tokens" => config.generation.max_tokens = value.parse().ok(),
                        _ => {}
                    },
                    "rag" => match key {
                        "knowledge_enabled" | "enabled" => {
                            config.rag.knowledge_enabled = value == "true"
                        }
                        "conversation_enabled" => config.rag.conversation_enabled = value == "true",
                        "knowledge_tokens" => {
                            config.rag.knowledge_tokens = value.parse().unwrap_or(2000)
                        }
                        "conversation_tokens" => {
                            config.rag.conversation_tokens = value.parse().unwrap_or(1500)
                        }
                        "top_k" => config.rag.top_k = value.parse().unwrap_or(5),
                        "min_relevance" => config.rag.min_relevance = value.parse().unwrap_or(0.1),
                        "append_only" => config.rag.append_only = value == "true",
                        "database_path" => config.rag.database_path = Some(value),
                        _ => {}
                    },
                    "security" => match key {
                        "rate_limit_enabled" => {
                            config.security.rate_limit_enabled = value == "true"
                        }
                        "requests_per_minute" => {
                            config.security.requests_per_minute = value.parse().unwrap_or(60)
                        }
                        "sanitize_input" => config.security.sanitize_input = value == "true",
                        "audit_logging" => config.security.audit_logging = value == "true",
                        _ => {}
                    },
                    "cache" => match key {
                        "enabled" => config.cache.enabled = value == "true",
                        "max_entries" => config.cache.max_entries = value.parse().unwrap_or(100),
                        "ttl_seconds" => config.cache.ttl_seconds = value.parse().unwrap_or(3600),
                        "semantic_enabled" => config.cache.semantic_enabled = value == "true",
                        "similarity_threshold" => {
                            config.cache.similarity_threshold = value.parse().unwrap_or(0.95)
                        }
                        _ => {}
                    },
                    "logging" => match key {
                        "level" => config.logging.level = value,
                        "file" => config.logging.file = Some(value),
                        "metrics" => config.logging.metrics = value == "true",
                        _ => {}
                    },
                    "rag.hybrid" | "hybrid" => {
                        let hybrid = config.rag.hybrid.get_or_insert_with(HybridConfig::default);
                        match key {
                            "semantic_enabled" => hybrid.semantic_enabled = value == "true",
                            "bm25_weight" => hybrid.bm25_weight = value.parse().unwrap_or(0.6),
                            "semantic_weight" => {
                                hybrid.semantic_weight = value.parse().unwrap_or(0.4)
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(config)
    }

    /// Parse a TOML value (remove quotes, handle basic types)
    fn parse_toml_value(value: &str) -> String {
        let value = value.trim();
        if (value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\''))
        {
            value[1..value.len() - 1].to_string()
        } else {
            value.to_string()
        }
    }

    /// Generate TOML string
    fn to_toml(&self) -> String {
        let mut out = String::new();

        out.push_str("# AI Assistant Configuration\n\n");

        out.push_str("[provider]\n");
        out.push_str(&format!("type = \"{}\"\n", self.provider.provider_type));
        if !self.provider.model.is_empty() {
            out.push_str(&format!("model = \"{}\"\n", self.provider.model));
        }
        if let Some(ref url) = self.provider.custom_url {
            out.push_str(&format!("custom_url = \"{}\"\n", url));
        }
        out.push('\n');

        out.push_str("[urls]\n");
        out.push_str(&format!("ollama = \"{}\"\n", self.urls.ollama));
        out.push_str(&format!("lm_studio = \"{}\"\n", self.urls.lm_studio));
        out.push_str(&format!(
            "text_gen_webui = \"{}\"\n",
            self.urls.text_gen_webui
        ));
        out.push_str(&format!("kobold = \"{}\"\n", self.urls.kobold));
        out.push_str(&format!("local_ai = \"{}\"\n", self.urls.local_ai));
        out.push('\n');

        out.push_str("[generation]\n");
        out.push_str(&format!("temperature = {}\n", self.generation.temperature));
        out.push_str(&format!("max_history = {}\n", self.generation.max_history));
        if let Some(top_p) = self.generation.top_p {
            out.push_str(&format!("top_p = {}\n", top_p));
        }
        if let Some(top_k) = self.generation.top_k {
            out.push_str(&format!("top_k = {}\n", top_k));
        }
        out.push('\n');

        out.push_str("[rag]\n");
        out.push_str(&format!(
            "knowledge_enabled = {}\n",
            self.rag.knowledge_enabled
        ));
        out.push_str(&format!(
            "conversation_enabled = {}\n",
            self.rag.conversation_enabled
        ));
        out.push_str(&format!(
            "knowledge_tokens = {}\n",
            self.rag.knowledge_tokens
        ));
        out.push_str(&format!(
            "conversation_tokens = {}\n",
            self.rag.conversation_tokens
        ));
        out.push_str(&format!("top_k = {}\n", self.rag.top_k));
        out.push_str(&format!("min_relevance = {}\n", self.rag.min_relevance));
        out.push_str(&format!("append_only = {}\n", self.rag.append_only));
        if let Some(ref path) = self.rag.database_path {
            out.push_str(&format!("database_path = \"{}\"\n", path));
        }
        out.push('\n');

        if let Some(ref hybrid) = self.rag.hybrid {
            out.push_str("[rag.hybrid]\n");
            out.push_str(&format!("semantic_enabled = {}\n", hybrid.semantic_enabled));
            out.push_str(&format!("bm25_weight = {}\n", hybrid.bm25_weight));
            out.push_str(&format!("semantic_weight = {}\n", hybrid.semantic_weight));
            out.push('\n');
        }

        out.push_str("[security]\n");
        out.push_str(&format!(
            "rate_limit_enabled = {}\n",
            self.security.rate_limit_enabled
        ));
        out.push_str(&format!(
            "requests_per_minute = {}\n",
            self.security.requests_per_minute
        ));
        out.push_str(&format!(
            "sanitize_input = {}\n",
            self.security.sanitize_input
        ));
        out.push_str(&format!(
            "audit_logging = {}\n",
            self.security.audit_logging
        ));
        out.push('\n');

        out.push_str("[cache]\n");
        out.push_str(&format!("enabled = {}\n", self.cache.enabled));
        out.push_str(&format!("max_entries = {}\n", self.cache.max_entries));
        out.push_str(&format!("ttl_seconds = {}\n", self.cache.ttl_seconds));
        out.push_str(&format!(
            "semantic_enabled = {}\n",
            self.cache.semantic_enabled
        ));
        out.push_str(&format!(
            "similarity_threshold = {}\n",
            self.cache.similarity_threshold
        ));
        out.push('\n');

        out.push_str("[logging]\n");
        out.push_str(&format!("level = \"{}\"\n", self.logging.level));
        if let Some(ref file) = self.logging.file {
            out.push_str(&format!("file = \"{}\"\n", file));
        }
        out.push_str(&format!("metrics = {}\n", self.logging.metrics));

        out
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), AiError> {
        // Validate temperature
        if self.generation.temperature < 0.0 || self.generation.temperature > 2.0 {
            return Err(ConfigError::InvalidValue {
                field: "generation.temperature".to_string(),
                value: self.generation.temperature.to_string(),
                expected: "0.0 to 2.0".to_string(),
            }
            .into());
        }

        // Validate RAG weights
        if let Some(ref hybrid) = self.rag.hybrid {
            let sum = hybrid.bm25_weight + hybrid.semantic_weight;
            if (sum - 1.0).abs() > 0.01 {
                return Err(ConfigError::InvalidValue {
                    field: "rag.hybrid.bm25_weight + semantic_weight".to_string(),
                    value: sum.to_string(),
                    expected: "weights should sum to 1.0".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }
}

/// Convenience function to load config from a path
pub fn load_config(path: &Path) -> Result<AiConfig, AiError> {
    let config_file = ConfigFile::load(path)?;
    config_file.validate()?;
    Ok(config_file.to_ai_config())
}

/// Convenience function to save config to a path
pub fn save_config(config: &AiConfig, path: &Path) -> Result<(), AiError> {
    let config_file = ConfigFile::from_ai_config(config);
    config_file.save(path)
}

/// Get the default config file path for the current platform
pub fn default_config_path() -> std::path::PathBuf {
    if let Some(config_dir) = platform_dirs::config_dir() {
        config_dir.join("ai_assistant").join("config.toml")
    } else {
        std::path::PathBuf::from("ai_assistant_config.toml")
    }
}

/// Platform-specific config directory resolution (replaces dirs_next crate)
mod platform_dirs {
    use std::path::PathBuf;

    pub fn config_dir() -> Option<PathBuf> {
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
}

// ============================================================================
// Configuration Validation (Item 4.4)
// ============================================================================

/// A validation error for a specific configuration field.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigValidationError {
    pub field: String,
    pub message: String,
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " (suggestion: {})", suggestion)?;
        }
        Ok(())
    }
}

impl ConfigFile {
    /// Validate all configuration fields with detailed error reporting.
    /// Unlike `validate()` which returns on the first error, this returns
    /// ALL validation errors found so they can be displayed at once.
    pub fn validate_detailed(&self) -> Result<(), Vec<ConfigValidationError>> {
        let mut errors = Vec::new();

        // Temperature range
        if self.generation.temperature < 0.0 || self.generation.temperature > 2.0 {
            errors.push(ConfigValidationError {
                field: "generation.temperature".to_string(),
                message: format!(
                    "temperature {} is out of range [0.0, 2.0]",
                    self.generation.temperature
                ),
                suggestion: Some("Use a value between 0.0 and 2.0".to_string()),
            });
        }

        // max_tokens > 0 if set
        if let Some(max_tokens) = self.generation.max_tokens {
            if max_tokens == 0 {
                errors.push(ConfigValidationError {
                    field: "generation.max_tokens".to_string(),
                    message: "max_tokens must be > 0".to_string(),
                    suggestion: Some("Set to at least 1, or remove to use default".to_string()),
                });
            }
        }

        // max_history > 0
        if self.generation.max_history == 0 {
            errors.push(ConfigValidationError {
                field: "generation.max_history".to_string(),
                message: "max_history must be > 0".to_string(),
                suggestion: Some("Set to at least 1".to_string()),
            });
        }

        // URL validation helper
        let url_fields = [
            ("urls.ollama", &self.urls.ollama),
            ("urls.lm_studio", &self.urls.lm_studio),
            ("urls.text_gen_webui", &self.urls.text_gen_webui),
            ("urls.kobold", &self.urls.kobold),
            ("urls.local_ai", &self.urls.local_ai),
        ];
        for (field_name, url) in &url_fields {
            if !url.starts_with("http://") && !url.starts_with("https://") {
                errors.push(ConfigValidationError {
                    field: field_name.to_string(),
                    message: format!("URL '{}' must start with http:// or https://", url),
                    suggestion: Some("Prefix with http:// or https://".to_string()),
                });
            }
        }

        // Custom URL validation
        if let Some(ref custom_url) = self.provider.custom_url {
            if !custom_url.starts_with("http://") && !custom_url.starts_with("https://") {
                errors.push(ConfigValidationError {
                    field: "provider.custom_url".to_string(),
                    message: format!("Custom URL '{}' must start with http:// or https://", custom_url),
                    suggestion: Some("Prefix with http:// or https://".to_string()),
                });
            }
        }

        // Cache TTL > 0 when caching is enabled
        if self.cache.enabled && self.cache.ttl_seconds == 0 {
            errors.push(ConfigValidationError {
                field: "cache.ttl_seconds".to_string(),
                message: "TTL must be > 0 when caching is enabled".to_string(),
                suggestion: Some("Set to at least 1 second, or disable caching".to_string()),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// =============================================================================
// CONFIG HOT-RELOAD (v8 item 8.2)
// =============================================================================

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Tracks which config fields can be hot-reloaded vs require restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadScope {
    /// Can be applied without restart (model, temperature, log level)
    HotReload,
    /// Requires server restart (host, port, TLS settings)
    RequiresRestart,
}

/// Result of a config reload check.
#[derive(Debug, Clone)]
pub struct ReloadResult {
    /// Whether the config file was modified since last check
    pub changed: bool,
    /// Fields that were hot-reloaded
    pub reloaded_fields: Vec<String>,
    /// Fields that changed but require restart
    pub restart_required_fields: Vec<String>,
}

/// Watches a configuration file for changes and applies hot-reloadable settings.
#[derive(Debug)]
pub struct ConfigWatcher {
    /// Path to the config file being watched
    path: PathBuf,
    /// Last modification time observed
    last_modified: Arc<Mutex<Option<SystemTime>>>,
    /// Current loaded config
    current_config: Arc<Mutex<ConfigFile>>,
    /// Poll interval
    poll_interval: Duration,
}

impl ConfigWatcher {
    /// Create a new config watcher for the given file path.
    pub fn new(path: impl Into<PathBuf>, poll_interval_secs: u64) -> Result<Self, String> {
        let path = path.into();
        let config = ConfigFile::load(&path).map_err(|e| e.to_string())?;
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok();

        Ok(Self {
            path,
            last_modified: Arc::new(Mutex::new(mtime)),
            current_config: Arc::new(Mutex::new(config)),
            poll_interval: Duration::from_secs(poll_interval_secs),
        })
    }

    /// Get the poll interval.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Get a clone of the current config.
    pub fn current_config(&self) -> ConfigFile {
        self.current_config
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default()
    }

    /// Check if the config file has been modified and reload if so.
    /// Returns which fields were hot-reloaded vs which require restart.
    pub fn check_and_reload(&self) -> Result<ReloadResult, String> {
        let current_mtime = std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .map_err(|e| format!("Failed to stat config file: {}", e))?;

        let mut last = self.last_modified.lock().map_err(|e| format!("Lock poisoned: {}", e))?;
        let changed = match *last {
            Some(prev) => current_mtime > prev,
            None => true,
        };

        if !changed {
            return Ok(ReloadResult {
                changed: false,
                reloaded_fields: vec![],
                restart_required_fields: vec![],
            });
        }

        // Reload config
        let new_config = ConfigFile::load(&self.path).map_err(|e| e.to_string())?;

        let mut reloaded = Vec::new();
        let mut restart_required = Vec::new();

        let mut current = self.current_config.lock().map_err(|e| format!("Lock poisoned: {}", e))?;

        // Check hot-reloadable fields
        if new_config.provider.model != current.provider.model {
            reloaded.push("provider.model".to_string());
        }
        if (new_config.generation.temperature - current.generation.temperature).abs() > f32::EPSILON {
            reloaded.push("generation.temperature".to_string());
        }
        if new_config.generation.max_history != current.generation.max_history {
            reloaded.push("generation.max_history".to_string());
        }
        if new_config.generation.max_tokens != current.generation.max_tokens {
            reloaded.push("generation.max_tokens".to_string());
        }
        if new_config.logging.level != current.logging.level {
            reloaded.push("logging.level".to_string());
        }
        if new_config.cache.enabled != current.cache.enabled {
            reloaded.push("cache.enabled".to_string());
        }
        if new_config.cache.ttl_seconds != current.cache.ttl_seconds {
            reloaded.push("cache.ttl_seconds".to_string());
        }

        // Check restart-required fields
        if new_config.provider.provider_type != current.provider.provider_type {
            restart_required.push("provider.type".to_string());
        }
        if new_config.provider.custom_url != current.provider.custom_url {
            restart_required.push("provider.custom_url".to_string());
        }
        if new_config.urls.ollama != current.urls.ollama {
            restart_required.push("urls.ollama".to_string());
        }
        if new_config.urls.lm_studio != current.urls.lm_studio {
            restart_required.push("urls.lm_studio".to_string());
        }

        // Apply the new config
        *current = new_config;
        *last = Some(current_mtime);

        log::info!(
            "Config reloaded from {:?}: {} hot-reloaded, {} require restart",
            self.path,
            reloaded.len(),
            restart_required.len()
        );

        Ok(ReloadResult {
            changed: true,
            reloaded_fields: reloaded,
            restart_required_fields: restart_required,
        })
    }

    /// Classify a config field by its reload scope.
    pub fn field_scope(field: &str) -> ReloadScope {
        match field {
            "provider.model" | "generation.temperature" | "generation.max_history"
            | "generation.max_tokens" | "logging.level" | "cache.enabled"
            | "cache.ttl_seconds" => ReloadScope::HotReload,
            _ => ReloadScope::RequiresRestart,
        }
    }
}

/// Register MCP tools for runtime configuration management.
///
/// Provides 6 MCP tools for reading and optionally modifying the AI assistant's
/// configuration at runtime. When `allow_writes` is `false`, all mutation tools
/// return an error, enabling safe read-only introspection.
///
/// # Tools registered
///
/// - `config.get` — read current configuration (never exposes API keys)
/// - `config.set` — update a configuration field (gated by `allow_writes`)
/// - `config.list_providers` — enumerate all available providers
/// - `config.validate` — check current configuration for errors
/// - `config.get_provider_url` — get the URL for a specific provider
/// - `config.can_write` — check whether write access is enabled
pub fn register_config_tools(
    server: &mut crate::mcp_protocol::McpServer,
    config: Arc<Mutex<AiConfig>>,
    allow_writes: bool,
) {
    use crate::mcp_protocol::McpTool;

    // --- config.get ---
    let c = config.clone();
    server.register_tool(
        McpTool::new("config.get", "Get current AI configuration (safe: no API keys exposed)")
            .with_property("field", "string", "Specific field to get (model, temperature, provider, max_history, ollama_url, lm_studio_url). Omit for all.", false),
        move |args| {
            let cfg = c.lock().map_err(|e| e.to_string())?;
            let field = args.get("field").and_then(|v| v.as_str());

            match field {
                Some("model") => Ok(serde_json::json!({ "model": cfg.selected_model })),
                Some("temperature") => Ok(serde_json::json!({ "temperature": cfg.temperature })),
                Some("provider") => Ok(serde_json::json!({
                    "provider": cfg.provider.display_name(),
                    "is_cloud": cfg.provider.is_cloud(),
                })),
                Some("max_history") => Ok(serde_json::json!({ "max_history_messages": cfg.max_history_messages })),
                Some("ollama_url") => Ok(serde_json::json!({ "ollama_url": cfg.ollama_url })),
                Some("lm_studio_url") => Ok(serde_json::json!({ "lm_studio_url": cfg.lm_studio_url })),
                Some(unknown) => Err(format!("Unknown field: {}. Available: model, temperature, provider, max_history, ollama_url, lm_studio_url", unknown)),
                None => Ok(serde_json::json!({
                    "provider": cfg.provider.display_name(),
                    "is_cloud": cfg.provider.is_cloud(),
                    "model": cfg.selected_model,
                    "temperature": cfg.temperature,
                    "max_history_messages": cfg.max_history_messages,
                    "base_url": cfg.get_base_url(),
                    "has_api_key": !cfg.api_key.is_empty(),
                })),
            }
        },
    );

    // --- config.set ---
    let c = config.clone();
    server.register_tool(
        McpTool::new("config.set", "Update a configuration field at runtime (requires write access)")
            .with_property("field", "string", "Field to set: model, temperature, max_history, provider, ollama_url, lm_studio_url, text_gen_webui_url, kobold_url, local_ai_url, custom_url", true)
            .with_property("value", "string", "New value (strings as-is, numbers as string e.g. \"0.9\")", true),
        move |args| {
            if !allow_writes {
                return Err("Write access is disabled. Set allow_writes=true to enable configuration changes.".to_string());
            }

            let field = args.get("field").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: field")?;
            let value = args.get("value").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: value")?;

            let mut cfg = c.lock().map_err(|e| e.to_string())?;

            match field {
                "model" => {
                    let old = cfg.selected_model.clone();
                    cfg.selected_model = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "model", "old": old, "new": value }))
                }
                "temperature" => {
                    let temp: f32 = value.parse().map_err(|_| format!("Invalid temperature value: {}", value))?;
                    if temp < 0.0 || temp > 2.0 {
                        return Err(format!("Temperature {} out of range [0.0, 2.0]", temp));
                    }
                    let old = cfg.temperature;
                    cfg.temperature = temp;
                    Ok(serde_json::json!({ "status": "updated", "field": "temperature", "old": old, "new": temp }))
                }
                "max_history" => {
                    let val: usize = value.parse().map_err(|_| format!("Invalid max_history value: {}", value))?;
                    if val == 0 {
                        return Err("max_history must be > 0".to_string());
                    }
                    let old = cfg.max_history_messages;
                    cfg.max_history_messages = val;
                    Ok(serde_json::json!({ "status": "updated", "field": "max_history", "old": old, "new": val }))
                }
                "provider" => {
                    let provider = match value.to_lowercase().as_str() {
                        "ollama" => AiProvider::Ollama,
                        "lmstudio" | "lm_studio" => AiProvider::LMStudio,
                        "textgenwebui" | "text_gen_webui" => AiProvider::TextGenWebUI,
                        "koboldcpp" | "kobold_cpp" => AiProvider::KoboldCpp,
                        "localai" | "local_ai" => AiProvider::LocalAI,
                        "openai" => AiProvider::OpenAI,
                        "anthropic" => AiProvider::Anthropic,
                        "gemini" => AiProvider::Gemini,
                        "groq" => AiProvider::Groq,
                        "together" => AiProvider::Together,
                        "fireworks" => AiProvider::Fireworks,
                        "deepseek" => AiProvider::DeepSeek,
                        "mistral" => AiProvider::Mistral,
                        "perplexity" => AiProvider::Perplexity,
                        "openrouter" => AiProvider::OpenRouter,
                        other => return Err(format!("Unknown provider: {}. Use: ollama, lmstudio, openai, anthropic, gemini, groq, together, fireworks, deepseek, mistral, perplexity, openrouter", other)),
                    };
                    let old = cfg.provider.display_name().to_string();
                    cfg.provider = provider;
                    Ok(serde_json::json!({ "status": "updated", "field": "provider", "old": old, "new": cfg.provider.display_name() }))
                }
                "ollama_url" => {
                    let old = cfg.ollama_url.clone();
                    cfg.ollama_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "ollama_url", "old": old, "new": value }))
                }
                "lm_studio_url" => {
                    let old = cfg.lm_studio_url.clone();
                    cfg.lm_studio_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "lm_studio_url", "old": old, "new": value }))
                }
                "text_gen_webui_url" => {
                    let old = cfg.text_gen_webui_url.clone();
                    cfg.text_gen_webui_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "text_gen_webui_url", "old": old, "new": value }))
                }
                "kobold_url" => {
                    let old = cfg.kobold_url.clone();
                    cfg.kobold_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "kobold_url", "old": old, "new": value }))
                }
                "local_ai_url" => {
                    let old = cfg.local_ai_url.clone();
                    cfg.local_ai_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "local_ai_url", "old": old, "new": value }))
                }
                "custom_url" => {
                    let old = cfg.custom_url.clone();
                    cfg.custom_url = value.to_string();
                    Ok(serde_json::json!({ "status": "updated", "field": "custom_url", "old": old, "new": value }))
                }
                unknown => Err(format!("Unknown field: {}. Settable: model, temperature, max_history, provider, ollama_url, lm_studio_url, text_gen_webui_url, kobold_url, local_ai_url, custom_url", unknown)),
            }
        },
    );

    // --- config.list_providers ---
    let c = config.clone();
    server.register_tool(
        McpTool::new("config.list_providers", "List all available AI providers with details"),
        move |_args| {
            let cfg = c.lock().map_err(|e| e.to_string())?;
            let providers: Vec<serde_json::Value> = [
                AiProvider::Ollama, AiProvider::LMStudio, AiProvider::TextGenWebUI,
                AiProvider::KoboldCpp, AiProvider::LocalAI, AiProvider::OpenAI,
                AiProvider::Anthropic, AiProvider::Gemini, AiProvider::Groq,
                AiProvider::Together, AiProvider::Fireworks, AiProvider::DeepSeek,
                AiProvider::Mistral, AiProvider::Perplexity, AiProvider::OpenRouter,
            ].iter().map(|p| {
                serde_json::json!({
                    "name": p.display_name(),
                    "is_cloud": p.is_cloud(),
                    "is_openai_compatible": p.is_openai_compatible(),
                    "url": cfg.get_provider_url(p),
                })
            }).collect();

            Ok(serde_json::json!({
                "providers": providers,
                "current": cfg.provider.display_name(),
                "count": providers.len(),
            }))
        },
    );

    // --- config.validate ---
    let c = config.clone();
    server.register_tool(
        McpTool::new("config.validate", "Validate current configuration for errors"),
        move |_args| {
            let cfg = c.lock().map_err(|e| e.to_string())?;
            let mut errors = Vec::new();

            if cfg.temperature < 0.0 || cfg.temperature > 2.0 {
                errors.push(serde_json::json!({
                    "field": "temperature",
                    "message": format!("temperature {} out of range [0.0, 2.0]", cfg.temperature),
                }));
            }
            if cfg.max_history_messages == 0 {
                errors.push(serde_json::json!({
                    "field": "max_history_messages",
                    "message": "max_history_messages must be > 0",
                }));
            }
            if cfg.selected_model.is_empty() {
                errors.push(serde_json::json!({
                    "field": "model",
                    "message": "no model selected",
                    "severity": "warning",
                }));
            }
            if cfg.provider.is_cloud() && cfg.api_key.is_empty() {
                errors.push(serde_json::json!({
                    "field": "api_key",
                    "message": format!("cloud provider {} requires an API key", cfg.provider.display_name()),
                    "severity": "warning",
                }));
            }

            Ok(serde_json::json!({
                "valid": errors.is_empty(),
                "errors": errors,
            }))
        },
    );

    // --- config.get_provider_url ---
    let c = config.clone();
    server.register_tool(
        McpTool::new("config.get_provider_url", "Get the URL for a specific provider")
            .with_property("provider", "string", "Provider name (ollama, lmstudio, openai, anthropic, etc.)", true),
        move |args| {
            let cfg = c.lock().map_err(|e| e.to_string())?;
            let name = args.get("provider").and_then(|v| v.as_str())
                .ok_or("Missing required parameter: provider")?;

            let provider = match name.to_lowercase().as_str() {
                "ollama" => AiProvider::Ollama,
                "lmstudio" | "lm_studio" => AiProvider::LMStudio,
                "textgenwebui" | "text_gen_webui" => AiProvider::TextGenWebUI,
                "koboldcpp" | "kobold_cpp" => AiProvider::KoboldCpp,
                "localai" | "local_ai" => AiProvider::LocalAI,
                "openai" => AiProvider::OpenAI,
                "anthropic" => AiProvider::Anthropic,
                "gemini" => AiProvider::Gemini,
                "groq" => AiProvider::Groq,
                "together" => AiProvider::Together,
                "fireworks" => AiProvider::Fireworks,
                "deepseek" => AiProvider::DeepSeek,
                "mistral" => AiProvider::Mistral,
                "perplexity" => AiProvider::Perplexity,
                "openrouter" => AiProvider::OpenRouter,
                other => return Err(format!("Unknown provider: {}", other)),
            };

            Ok(serde_json::json!({
                "provider": provider.display_name(),
                "url": cfg.get_provider_url(&provider),
                "is_cloud": provider.is_cloud(),
            }))
        },
    );

    // --- config.can_write ---
    server.register_tool(
        McpTool::new("config.can_write", "Check if write access to configuration is enabled"),
        move |_args| {
            Ok(serde_json::json!({
                "allow_writes": allow_writes,
            }))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_toml() {
        let toml = r#"
[provider]
type = "ollama"
model = "llama2"

[generation]
temperature = 0.8
max_history = 30

[rag]
knowledge_enabled = true
"#;
        let config = ConfigFile::parse(toml, ConfigFormat::Toml).unwrap();
        assert_eq!(config.provider.provider_type, "ollama");
        assert_eq!(config.provider.model, "llama2");
        assert_eq!(config.generation.temperature, 0.8);
        assert_eq!(config.generation.max_history, 30);
        assert!(config.rag.knowledge_enabled);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "provider": {
                "type": "lmstudio",
                "model": "mistral"
            },
            "generation": {
                "temperature": 0.5
            }
        }"#;
        let config = ConfigFile::parse(json, ConfigFormat::Json).unwrap();
        assert_eq!(config.provider.provider_type, "lmstudio");
        assert_eq!(config.provider.model, "mistral");
        assert_eq!(config.generation.temperature, 0.5);
    }

    #[test]
    fn test_to_ai_config() {
        let mut config = ConfigFile::default();
        config.provider.provider_type = "ollama".to_string();
        config.provider.model = "phi3".to_string();
        config.generation.temperature = 0.9;

        let ai_config = config.to_ai_config();
        assert!(matches!(ai_config.provider, AiProvider::Ollama));
        assert_eq!(ai_config.selected_model, "phi3");
        assert_eq!(ai_config.temperature, 0.9);
    }

    #[test]
    fn test_roundtrip_toml() {
        let config = ConfigFile::default();
        let toml = config.to_toml();
        let parsed = ConfigFile::parse(&toml, ConfigFormat::Toml).unwrap();

        assert_eq!(config.provider.provider_type, parsed.provider.provider_type);
        assert_eq!(config.generation.temperature, parsed.generation.temperature);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ConfigFormat::from_content("{\"key\": \"value\"}"),
            ConfigFormat::Json
        );
        assert_eq!(
            ConfigFormat::from_content("[section]\nkey = \"value\""),
            ConfigFormat::Toml
        );
    }

    // ========================================================================
    // ConfigFile::validate() tests (Item 4.4)
    // ========================================================================

    #[test]
    fn test_valid_config() {
        let config = ConfigFile::default();
        assert!(config.validate_detailed().is_ok());
    }

    #[test]
    fn test_invalid_temperature_high() {
        let mut config = ConfigFile::default();
        config.generation.temperature = 3.0;
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field == "generation.temperature"));
    }

    #[test]
    fn test_invalid_temperature_negative() {
        let mut config = ConfigFile::default();
        config.generation.temperature = -0.5;
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field == "generation.temperature"));
    }

    #[test]
    fn test_invalid_url() {
        let mut config = ConfigFile::default();
        config.urls.ollama = "not-a-url".to_string();
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field.contains("urls.")));
    }

    #[test]
    fn test_multiple_errors() {
        let mut config = ConfigFile::default();
        config.generation.temperature = 5.0;
        config.urls.ollama = "bad".to_string();
        config.urls.lm_studio = "also bad".to_string();
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.len() >= 3);
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let mut config = ConfigFile::default();
        config.generation.max_tokens = Some(0);
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field == "generation.max_tokens"));
    }

    #[test]
    fn test_validate_defaults() {
        let config = ConfigFile::default();
        assert!(config.validate_detailed().is_ok());
    }

    #[test]
    fn test_validation_error_display() {
        let err = ConfigValidationError {
            field: "temperature".to_string(),
            message: "out of range".to_string(),
            suggestion: Some("use 0.0 to 2.0".to_string()),
        };
        let s = format!("{}", err);
        assert!(s.contains("temperature"));
        assert!(s.contains("out of range"));
    }

    #[test]
    fn test_validation_suggestions() {
        let mut config = ConfigFile::default();
        config.generation.temperature = 10.0;
        let errors = config.validate_detailed().unwrap_err();
        let temp_err = errors.iter().find(|e| e.field == "generation.temperature").unwrap();
        assert!(temp_err.suggestion.is_some());
    }

    #[test]
    fn test_validate_cache_config() {
        let mut config = ConfigFile::default();
        config.cache.enabled = true;
        config.cache.ttl_seconds = 0;
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field == "cache.ttl_seconds"));
    }

    #[test]
    fn test_validate_max_history_zero() {
        let mut config = ConfigFile::default();
        config.generation.max_history = 0;
        let errors = config.validate_detailed().unwrap_err();
        assert!(errors.iter().any(|e| e.field == "generation.max_history"));
    }

    #[test]
    fn test_validate_custom_url_valid() {
        let mut config = ConfigFile::default();
        config.provider.custom_url = Some("http://custom.server:9999".to_string());
        assert!(config.validate_detailed().is_ok());
    }

    // --- ConfigWatcher tests (v8 item 8.2) ---

    #[test]
    fn test_config_watcher_from_file() {
        let dir = std::env::temp_dir().join("ai_test_watcher");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");
        std::fs::write(
            &path,
            "[provider]\ntype = \"ollama\"\nmodel = \"llama2\"\n",
        )
        .unwrap();

        let watcher = ConfigWatcher::new(&path, 5);
        assert!(watcher.is_ok());
        let watcher = watcher.unwrap();
        assert_eq!(watcher.poll_interval(), Duration::from_secs(5));
        let config = watcher.current_config();
        assert_eq!(config.provider.model, "llama2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_watcher_no_change() {
        let dir = std::env::temp_dir().join("ai_test_watcher_nochange");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");
        std::fs::write(&path, "[provider]\ntype = \"ollama\"\nmodel = \"llama2\"\n").unwrap();

        let watcher = ConfigWatcher::new(&path, 1).unwrap();
        // First check — marks the mtime
        let _ = watcher.check_and_reload();
        // Second check — no changes
        let result = watcher.check_and_reload().unwrap();
        assert!(!result.changed);
        assert!(result.reloaded_fields.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_watcher_detects_change() {
        let dir = std::env::temp_dir().join("ai_test_watcher_change");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");
        std::fs::write(&path, "[provider]\ntype = \"ollama\"\nmodel = \"llama2\"\n").unwrap();

        let watcher = ConfigWatcher::new(&path, 1).unwrap();
        let _ = watcher.check_and_reload();

        // Modify the file (need a small sleep so mtime changes)
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(
            &path,
            "[provider]\ntype = \"ollama\"\nmodel = \"mistral\"\n\n[generation]\ntemperature = 0.9\n",
        )
        .unwrap();

        let result = watcher.check_and_reload().unwrap();
        assert!(result.changed);
        assert!(result.reloaded_fields.contains(&"provider.model".to_string()));
        assert_eq!(watcher.current_config().provider.model, "mistral");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_watcher_restart_required_fields() {
        let dir = std::env::temp_dir().join("ai_test_watcher_restart");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");
        std::fs::write(&path, "[provider]\ntype = \"ollama\"\nmodel = \"llama2\"\n").unwrap();

        let watcher = ConfigWatcher::new(&path, 1).unwrap();
        let _ = watcher.check_and_reload();

        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&path, "[provider]\ntype = \"openai\"\nmodel = \"llama2\"\n").unwrap();

        let result = watcher.check_and_reload().unwrap();
        assert!(result.changed);
        assert!(result.restart_required_fields.contains(&"provider.type".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_watcher_nonexistent_file() {
        let result = ConfigWatcher::new("/tmp/nonexistent_ai_config_test_12345.toml", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_field_scope_classification() {
        assert_eq!(ConfigWatcher::field_scope("provider.model"), ReloadScope::HotReload);
        assert_eq!(ConfigWatcher::field_scope("generation.temperature"), ReloadScope::HotReload);
        assert_eq!(ConfigWatcher::field_scope("logging.level"), ReloadScope::HotReload);
        assert_eq!(ConfigWatcher::field_scope("cache.enabled"), ReloadScope::HotReload);
        assert_eq!(ConfigWatcher::field_scope("urls.ollama"), ReloadScope::RequiresRestart);
        assert_eq!(ConfigWatcher::field_scope("provider.custom_url"), ReloadScope::RequiresRestart);
    }

    #[test]
    fn test_reload_scope_eq() {
        assert_eq!(ReloadScope::HotReload, ReloadScope::HotReload);
        assert_ne!(ReloadScope::HotReload, ReloadScope::RequiresRestart);
    }

    #[test]
    fn test_reload_result_debug() {
        let result = ReloadResult {
            changed: true,
            reloaded_fields: vec!["provider.model".to_string()],
            restart_required_fields: vec![],
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("provider.model"));
    }

    // =========================================================================
    // Config MCP tools tests
    // =========================================================================

    fn make_test_server_with_config(allow_writes: bool) -> (crate::mcp_protocol::McpServer, Arc<Mutex<AiConfig>>) {
        let mut server = crate::mcp_protocol::McpServer::new("test-config", "1.0.0");
        let mut config = AiConfig::default();
        config.selected_model = "llama2".to_string();
        config.temperature = 0.7;
        let shared = Arc::new(Mutex::new(config));
        register_config_tools(&mut server, shared.clone(), allow_writes);
        (server, shared)
    }

    fn call_tool(server: &crate::mcp_protocol::McpServer, name: &str, args: serde_json::Value) -> serde_json::Value {
        use crate::mcp_protocol::McpRequest;
        // Initialize first
        let init = McpRequest::new("initialize")
            .with_id(0u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }));
        server.handle_request(init);

        let req = McpRequest::new("tools/call")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "name": name,
                "arguments": args,
            }));
        let resp = server.handle_request(req);
        resp.result.unwrap_or_default()
    }

    fn extract_text(result: &serde_json::Value) -> serde_json::Value {
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            if let Some(first) = content.first() {
                if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                    return serde_json::from_str(text).unwrap_or_default();
                }
            }
        }
        serde_json::Value::Null
    }

    #[test]
    fn test_config_mcp_get_all() {
        let (server, _cfg) = make_test_server_with_config(false);
        let result = call_tool(&server, "config.get", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["model"], "llama2");
        assert_eq!(data["provider"], "Ollama");
        assert_eq!(data["has_api_key"], false);
    }

    #[test]
    fn test_config_mcp_get_field() {
        let (server, _cfg) = make_test_server_with_config(false);
        let result = call_tool(&server, "config.get", serde_json::json!({"field": "temperature"}));
        let data = extract_text(&result);
        let temp = data["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_config_mcp_set_temperature() {
        let (server, cfg) = make_test_server_with_config(true);
        let result = call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "1.2"}));
        let data = extract_text(&result);
        assert_eq!(data["status"], "updated");
        let locked = cfg.lock().unwrap();
        assert!((locked.temperature - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_config_mcp_set_model() {
        let (server, cfg) = make_test_server_with_config(true);
        let result = call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "mistral"}));
        let data = extract_text(&result);
        assert_eq!(data["status"], "updated");
        assert_eq!(data["old"], "llama2");
        assert_eq!(data["new"], "mistral");
        assert_eq!(cfg.lock().unwrap().selected_model, "mistral");
    }

    #[test]
    fn test_config_mcp_set_blocked_when_readonly() {
        let (server, cfg) = make_test_server_with_config(false);
        let result = call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "mistral"}));
        // Should return error — check that the config didn't change
        assert_eq!(cfg.lock().unwrap().selected_model, "llama2");
        // The response should have an error, not content with "updated"
        let data = extract_text(&result);
        assert_ne!(data.get("status").and_then(|s| s.as_str()), Some("updated"));
    }

    #[test]
    fn test_config_mcp_set_invalid_temperature() {
        let (server, cfg) = make_test_server_with_config(true);
        let result = call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "3.5"}));
        // Should fail — temperature remains unchanged
        let locked = cfg.lock().unwrap();
        assert!((locked.temperature - 0.7).abs() < 0.01);
        let data = extract_text(&result);
        assert_ne!(data.get("status").and_then(|s| s.as_str()), Some("updated"));
    }

    #[test]
    fn test_config_mcp_list_providers() {
        let (server, _cfg) = make_test_server_with_config(false);
        let result = call_tool(&server, "config.list_providers", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["current"], "Ollama");
        let providers = data["providers"].as_array().unwrap();
        assert!(providers.len() >= 15);
        // Check that Ollama is in the list
        assert!(providers.iter().any(|p| p["name"] == "Ollama"));
    }

    #[test]
    fn test_config_mcp_validate() {
        let (server, _cfg) = make_test_server_with_config(false);
        let result = call_tool(&server, "config.validate", serde_json::json!({}));
        let data = extract_text(&result);
        // Default config with empty model triggers a warning but is still "valid" (warnings != errors)
        assert!(data.get("valid").is_some());
    }

    #[test]
    fn test_config_mcp_can_write() {
        let (server_ro, _) = make_test_server_with_config(false);
        let result = call_tool(&server_ro, "config.can_write", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["allow_writes"], false);

        let (server_rw, _) = make_test_server_with_config(true);
        let result = call_tool(&server_rw, "config.can_write", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["allow_writes"], true);
    }

    #[test]
    fn test_config_mcp_change_midway() {
        // Simulates AI changing its own config during execution
        let (server, cfg) = make_test_server_with_config(true);

        // Step 1: Read initial config
        let result = call_tool(&server, "config.get", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["model"], "llama2");
        assert_eq!(data["provider"], "Ollama");

        // Step 2: AI decides to switch model mid-conversation
        let _ = call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "codellama"}));
        let result = call_tool(&server, "config.get", serde_json::json!({"field": "model"}));
        let data = extract_text(&result);
        assert_eq!(data["model"], "codellama");

        // Step 3: AI raises temperature for creative task
        let _ = call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "1.5"}));
        let result = call_tool(&server, "config.get", serde_json::json!({"field": "temperature"}));
        let data = extract_text(&result);
        assert!((data["temperature"].as_f64().unwrap() - 1.5).abs() < 0.01);

        // Step 4: AI switches provider entirely
        let _ = call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "openai"}));
        let result = call_tool(&server, "config.get", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["provider"], "OpenAI");
        assert!(data["is_cloud"].as_bool().unwrap());

        // Step 5: AI lowers temperature back for analytical task
        let _ = call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "0.2"}));

        // Step 6: Verify all changes persisted through Arc<Mutex<>>
        let locked = cfg.lock().unwrap();
        assert_eq!(locked.selected_model, "codellama");
        assert!((locked.temperature - 0.2).abs() < 0.01);
        assert!(locked.provider.is_cloud());
        assert_eq!(locked.max_history_messages, 20); // Unchanged
    }

    // =========================================================================
    // Behavioral integration tests: config changes have real effects
    // =========================================================================

    #[test]
    fn test_config_change_provider_affects_base_url() {
        // Changing provider via MCP should change the base URL used for API calls
        let (server, cfg) = make_test_server_with_config(true);

        // Initially Ollama — local URL
        let url1 = cfg.lock().unwrap().get_base_url();
        assert_eq!(url1, "http://localhost:11434");

        // Switch to OpenAI via MCP
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "openai"}));
        let url2 = cfg.lock().unwrap().get_base_url();
        assert_eq!(url2, "https://api.openai.com");

        // Switch to Anthropic via MCP
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "anthropic"}));
        let url3 = cfg.lock().unwrap().get_base_url();
        assert_eq!(url3, "https://api.anthropic.com");

        // Switch to Groq via MCP
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "groq"}));
        let url4 = cfg.lock().unwrap().get_base_url();
        assert_eq!(url4, "https://api.groq.com/openai");

        // Switch back to local — URL should change back
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "lmstudio"}));
        let url5 = cfg.lock().unwrap().get_base_url();
        assert_eq!(url5, "http://localhost:1234");
    }

    #[test]
    fn test_config_change_provider_affects_cloud_detection() {
        // is_cloud() should reflect the current provider
        let (server, cfg) = make_test_server_with_config(true);

        // Ollama is local
        assert!(!cfg.lock().unwrap().provider.is_cloud());

        // Switch to cloud providers, verify is_cloud changes each time
        for (provider, expected_cloud) in [
            ("openai", true), ("anthropic", true), ("gemini", true),
            ("ollama", false), ("lmstudio", false),
            ("groq", true), ("together", true),
            ("ollama", false),
        ] {
            call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": provider}));
            let locked = cfg.lock().unwrap();
            assert_eq!(
                locked.provider.is_cloud(), expected_cloud,
                "Provider {} should have is_cloud={}", provider, expected_cloud
            );
        }
    }

    #[test]
    fn test_config_change_url_affects_provider_routing() {
        // Changing a URL via MCP should affect where requests would be routed
        let (server, cfg) = make_test_server_with_config(true);

        // Change Ollama URL to a custom address
        call_tool(&server, "config.set", serde_json::json!({"field": "ollama_url", "value": "http://gpu-server:11434"}));
        let locked = cfg.lock().unwrap();
        assert_eq!(locked.get_base_url(), "http://gpu-server:11434");
        assert_eq!(locked.get_provider_url(&AiProvider::Ollama), "http://gpu-server:11434");
        drop(locked);

        // LM Studio URL change
        call_tool(&server, "config.set", serde_json::json!({"field": "lm_studio_url", "value": "http://192.168.1.100:1234"}));
        let locked = cfg.lock().unwrap();
        assert_eq!(locked.get_provider_url(&AiProvider::LMStudio), "http://192.168.1.100:1234");
    }

    #[test]
    fn test_config_change_max_history_affects_context_window() {
        // max_history_messages controls how many conversation messages are included
        let (server, cfg) = make_test_server_with_config(true);

        // Default is 20
        assert_eq!(cfg.lock().unwrap().max_history_messages, 20);

        // AI decides to use shorter context for speed
        call_tool(&server, "config.set", serde_json::json!({"field": "max_history", "value": "5"}));
        assert_eq!(cfg.lock().unwrap().max_history_messages, 5);

        // AI expands context for complex task
        call_tool(&server, "config.set", serde_json::json!({"field": "max_history", "value": "100"}));
        assert_eq!(cfg.lock().unwrap().max_history_messages, 100);

        // Verify 0 is rejected
        let result = call_tool(&server, "config.set", serde_json::json!({"field": "max_history", "value": "0"}));
        let data = extract_text(&result);
        assert_ne!(data.get("status").and_then(|s| s.as_str()), Some("updated"));
        // Still 100 — rejected change didn't apply
        assert_eq!(cfg.lock().unwrap().max_history_messages, 100);
    }

    #[test]
    fn test_config_change_temperature_validation_boundaries() {
        // Temperature changes should be validated at boundaries
        let (server, cfg) = make_test_server_with_config(true);

        // Valid boundaries
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "0.0"}));
        assert!((cfg.lock().unwrap().temperature - 0.0).abs() < 0.001);

        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "2.0"}));
        assert!((cfg.lock().unwrap().temperature - 2.0).abs() < 0.001);

        // Invalid: negative
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "-0.1"}));
        assert!((cfg.lock().unwrap().temperature - 2.0).abs() < 0.001); // Unchanged

        // Invalid: too high
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "2.1"}));
        assert!((cfg.lock().unwrap().temperature - 2.0).abs() < 0.001); // Unchanged
    }

    #[test]
    fn test_config_validate_detects_cloud_without_key() {
        // Switching to a cloud provider without API key should trigger validation warning
        let (server, _cfg) = make_test_server_with_config(true);

        // Ollama — no API key needed
        let result = call_tool(&server, "config.validate", serde_json::json!({}));
        let data = extract_text(&result);
        let errors = data["errors"].as_array().unwrap();
        assert!(!errors.iter().any(|e| e["field"] == "api_key"));

        // Switch to OpenAI without API key
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "openai"}));
        let result = call_tool(&server, "config.validate", serde_json::json!({}));
        let data = extract_text(&result);
        let errors = data["errors"].as_array().unwrap();
        assert!(errors.iter().any(|e| e["field"] == "api_key"),
            "Validation should warn about missing API key for cloud provider");
    }

    #[test]
    fn test_config_full_session_scenario() {
        // Simulate a complete AI session where it adapts its config for different tasks
        let (server, cfg) = make_test_server_with_config(true);

        // Phase 1: Fast local coding task
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "ollama"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "codellama:13b"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "0.1"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "max_history", "value": "5"}));
        {
            let c = cfg.lock().unwrap();
            assert!(!c.provider.is_cloud());
            assert_eq!(c.selected_model, "codellama:13b");
            assert!((c.temperature - 0.1).abs() < 0.01);
            assert_eq!(c.max_history_messages, 5);
            assert_eq!(c.get_base_url(), "http://localhost:11434");
        }

        // Phase 2: Complex reasoning task — switch to cloud, high context
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "anthropic"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "claude-3-opus"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "0.3"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "max_history", "value": "50"}));
        {
            let c = cfg.lock().unwrap();
            assert!(c.provider.is_cloud());
            assert_eq!(c.selected_model, "claude-3-opus");
            assert!((c.temperature - 0.3).abs() < 0.01);
            assert_eq!(c.max_history_messages, 50);
            assert_eq!(c.get_base_url(), "https://api.anthropic.com");
        }

        // Phase 3: Creative writing — high temperature
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "openai"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "gpt-4o"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "1.8"}));
        {
            let c = cfg.lock().unwrap();
            assert!(c.provider.is_cloud());
            assert_eq!(c.get_base_url(), "https://api.openai.com");
            assert!((c.temperature - 1.8).abs() < 0.01);
        }

        // Phase 4: Back to local for privacy
        call_tool(&server, "config.set", serde_json::json!({"field": "provider", "value": "lmstudio"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "model", "value": "mistral-7b"}));
        call_tool(&server, "config.set", serde_json::json!({"field": "temperature", "value": "0.7"}));
        {
            let c = cfg.lock().unwrap();
            assert!(!c.provider.is_cloud());
            assert_eq!(c.get_base_url(), "http://localhost:1234");
            assert_eq!(c.selected_model, "mistral-7b");
        }

        // Verify: final state via config.get reflects all cumulative changes
        let result = call_tool(&server, "config.get", serde_json::json!({}));
        let data = extract_text(&result);
        assert_eq!(data["provider"], "LM Studio");
        assert_eq!(data["model"], "mistral-7b");
        assert!(!data["is_cloud"].as_bool().unwrap());
    }

    #[test]
    fn test_containers_config_default() {
        let config = ContainersConfig::default();
        assert!(!config.enabled);
        assert!(!config.mcp_enabled);
        assert_eq!(config.default_timeout_secs, 60);
        assert!(config.allowed_images.is_empty());
    }

    #[test]
    fn test_containers_config_json_parse() {
        let json = r#"{
            "containers": {
                "enabled": true,
                "default_timeout_secs": 120,
                "allowed_images": ["busybox:latest", "alpine:latest"],
                "mcp_enabled": true
            }
        }"#;
        let config = ConfigFile::parse(json, ConfigFormat::Json).unwrap();
        assert!(config.containers.enabled);
        assert!(config.containers.mcp_enabled);
        assert_eq!(config.containers.default_timeout_secs, 120);
        assert_eq!(config.containers.allowed_images.len(), 2);
        assert_eq!(config.containers.allowed_images[0], "busybox:latest");
    }

    #[test]
    fn test_config_round_trip_with_containers() {
        let mut config = ConfigFile::default();
        config.containers.enabled = true;
        config.containers.mcp_enabled = true;
        config.containers.default_timeout_secs = 90;
        config.containers.allowed_images = vec!["python:3.12-slim".to_string()];

        let json = config.serialize(ConfigFormat::Json).unwrap();
        let parsed = ConfigFile::parse(&json, ConfigFormat::Json).unwrap();

        assert!(parsed.containers.enabled);
        assert!(parsed.containers.mcp_enabled);
        assert_eq!(parsed.containers.default_timeout_secs, 90);
        assert_eq!(parsed.containers.allowed_images, vec!["python:3.12-slim"]);
    }
}
