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
}
