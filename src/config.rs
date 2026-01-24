//! Configuration types for AI providers

use serde::{Deserialize, Serialize};

/// Available AI provider types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
        )
    }
}

/// AI Assistant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Maximum number of history messages to include in context
    pub max_history_messages: usize,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: f32,
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
            max_history_messages: 20,
            temperature: 0.7,
        }
    }
}

impl AiConfig {
    /// Get the base URL for the current provider
    pub fn get_base_url(&self) -> String {
        match &self.provider {
            AiProvider::Ollama => self.ollama_url.clone(),
            AiProvider::LMStudio => self.lm_studio_url.clone(),
            AiProvider::TextGenWebUI => self.text_gen_webui_url.clone(),
            AiProvider::KoboldCpp => self.kobold_url.clone(),
            AiProvider::LocalAI => self.local_ai_url.clone(),
            AiProvider::OpenAICompatible { base_url } => base_url.clone(),
        }
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
        }
    }
}
