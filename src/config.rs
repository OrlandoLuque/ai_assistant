//! Configuration types for AI providers

use serde::{Deserialize, Serialize};

use crate::retry::RetryConfig;

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
        }
    }

    /// Check if this provider uses OpenAI-compatible API
    pub fn is_openai_compatible(&self) -> bool {
        matches!(
            self,
            AiProvider::LMStudio
                | AiProvider::TextGenWebUI
                | AiProvider::LocalAI
                | AiProvider::OpenAICompatible { .. }
                | AiProvider::OpenAI
                | AiProvider::Groq
                | AiProvider::Together
                | AiProvider::Fireworks
                | AiProvider::DeepSeek
                | AiProvider::Mistral
                | AiProvider::Perplexity
                | AiProvider::OpenRouter
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
            .field("custom_url", &self.custom_url)
            .field("api_key", &if self.api_key.is_empty() { "<empty>" } else { "<REDACTED>" })
            .field("max_history_messages", &self.max_history_messages)
            .field("temperature", &self.temperature)
            .finish()
    }
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            provider: AiProvider::Ollama,
            selected_model: String::new(),
            ollama_url: "http://localhost:11434".to_string(),
            lm_studio_url: "http://localhost:1234".to_string(),
            text_gen_webui_url: "http://localhost:5000".to_string(),
            kobold_url: "http://localhost:5001".to_string(),
            local_ai_url: "http://localhost:8080".to_string(),
            custom_url: String::new(),
            api_key: String::new(),
            max_history_messages: 20,
            temperature: 0.7,
            retry_config: RetryConfig::default(),
        }
    }
}

impl AiConfig {
    /// Get the base URL for the current provider
    pub fn get_base_url(&self) -> String {
        self.get_provider_url(&self.provider)
    }

    /// Get URL for a specific provider
    pub fn get_provider_url(&self, provider: &AiProvider) -> String {
        match provider {
            AiProvider::Ollama => self.ollama_url.clone(),
            AiProvider::LMStudio => self.lm_studio_url.clone(),
            AiProvider::TextGenWebUI => self.text_gen_webui_url.clone(),
            AiProvider::KoboldCpp => self.kobold_url.clone(),
            AiProvider::LocalAI => self.local_ai_url.clone(),
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
            _ => None,
        }
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
    fn test_ai_provider_display_names() {
        assert_eq!(AiProvider::Ollama.display_name(), "Ollama");
        assert_eq!(AiProvider::LMStudio.display_name(), "LM Studio");
        assert_eq!(
            AiProvider::TextGenWebUI.display_name(),
            "text-generation-webui"
        );
        assert_eq!(AiProvider::KoboldCpp.display_name(), "Kobold.cpp");
        assert_eq!(AiProvider::LocalAI.display_name(), "LocalAI");
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
        let custom = AiProvider::OpenAICompatible {
            base_url: "http://x".to_string(),
        };
        assert!(custom.is_openai_compatible());
    }

    #[test]
    fn test_ai_config_defaults() {
        let config = AiConfig::default();
        assert_eq!(config.provider, AiProvider::Ollama);
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.lm_studio_url, "http://localhost:1234");
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
        assert_eq!(config.get_base_url(), "http://localhost:11434");

        let mut config2 = AiConfig::default();
        config2.provider = AiProvider::LMStudio;
        assert_eq!(config2.get_base_url(), "http://localhost:1234");

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
            AiProvider::Ollama, AiProvider::LMStudio, AiProvider::OpenAI,
            AiProvider::Anthropic, AiProvider::Gemini,
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
}
