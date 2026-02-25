//! Model information types

use crate::config::AiProvider;
use serde::{Deserialize, Serialize};

/// Capability details for a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilityInfo {
    /// Maximum context window in tokens
    pub context_window: Option<usize>,
    /// Whether the model supports vision/image inputs
    pub supports_vision: bool,
    /// Whether the model supports tool/function calling
    pub supports_tool_calling: bool,
    /// Whether the model supports structured JSON output mode
    pub supports_json_mode: bool,
    /// Whether the model supports streaming responses
    pub supports_streaming: bool,
    /// Cost per million input tokens in USD
    pub input_cost_per_million: Option<f64>,
    /// Cost per million output tokens in USD
    pub output_cost_per_million: Option<f64>,
    /// Maximum output tokens the model can generate
    pub max_output_tokens: Option<usize>,
    /// Knowledge cutoff date (e.g., "2024-04")
    pub knowledge_cutoff: Option<String>,
}

impl Default for ModelCapabilityInfo {
    fn default() -> Self {
        Self {
            context_window: None,
            supports_vision: false,
            supports_tool_calling: false,
            supports_json_mode: false,
            supports_streaming: false,
            input_cost_per_million: None,
            output_cost_per_million: None,
            max_output_tokens: None,
            knowledge_cutoff: None,
        }
    }
}

/// Information about an available AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name/identifier
    pub name: String,
    /// Provider this model is from
    pub provider: AiProvider,
    /// Model size (e.g., "7.0 GB")
    pub size: Option<String>,
    /// Last modified timestamp
    pub modified_at: Option<String>,
    /// Detailed capability information
    pub capabilities: Option<ModelCapabilityInfo>,
}

impl ModelInfo {
    /// Create a new ModelInfo
    pub fn new(name: impl Into<String>, provider: AiProvider) -> Self {
        Self {
            name: name.into(),
            provider,
            size: None,
            modified_at: None,
            capabilities: None,
        }
    }

    /// Set the size
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Set the modified_at timestamp
    pub fn with_modified_at(mut self, modified_at: impl Into<String>) -> Self {
        self.modified_at = Some(modified_at.into());
        self
    }

    /// Set the capability information
    pub fn with_capabilities(mut self, capabilities: ModelCapabilityInfo) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Get display name with size info
    pub fn display_name(&self) -> String {
        if let Some(ref size) = self.size {
            format!("{} ({})", self.name, size)
        } else {
            self.name.clone()
        }
    }

    /// Get display name with provider icon
    pub fn display_name_with_icon(&self) -> String {
        format!("{} {}", self.provider.icon(), self.display_name())
    }
}

/// Format bytes as human-readable size
pub fn format_size(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    }
}

/// Registry of known models with their capabilities
pub struct ModelRegistry {
    models: std::collections::HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
        }
    }

    /// Register a model in the registry
    pub fn register(&mut self, model: ModelInfo) {
        self.models.insert(model.name.clone(), model);
    }

    /// Look up a model by name
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// Number of models in the registry
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Create a registry pre-populated with well-known models
    pub fn with_known_models() -> Self {
        let mut registry = Self::new();

        // GPT-4o
        registry.register(
            ModelInfo::new("gpt-4o", AiProvider::OpenAI).with_capabilities(ModelCapabilityInfo {
                context_window: Some(128_000),
                supports_vision: true,
                supports_tool_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                input_cost_per_million: Some(2.50),
                output_cost_per_million: Some(10.0),
                max_output_tokens: Some(16_384),
                knowledge_cutoff: Some("2024-04".to_string()),
            }),
        );

        // GPT-4o-mini
        registry.register(
            ModelInfo::new("gpt-4o-mini", AiProvider::OpenAI).with_capabilities(
                ModelCapabilityInfo {
                    context_window: Some(128_000),
                    supports_vision: true,
                    supports_tool_calling: true,
                    supports_json_mode: true,
                    supports_streaming: true,
                    input_cost_per_million: Some(0.15),
                    output_cost_per_million: Some(0.60),
                    max_output_tokens: Some(16_384),
                    knowledge_cutoff: Some("2024-04".to_string()),
                },
            ),
        );

        // Claude 3.5 Sonnet
        registry.register(
            ModelInfo::new("claude-3.5-sonnet", AiProvider::Anthropic).with_capabilities(
                ModelCapabilityInfo {
                    context_window: Some(200_000),
                    supports_vision: true,
                    supports_tool_calling: true,
                    supports_json_mode: true,
                    supports_streaming: true,
                    input_cost_per_million: Some(3.0),
                    output_cost_per_million: Some(15.0),
                    max_output_tokens: Some(8_192),
                    knowledge_cutoff: Some("2024-04".to_string()),
                },
            ),
        );

        // Claude 3 Haiku
        registry.register(
            ModelInfo::new("claude-3-haiku", AiProvider::Anthropic).with_capabilities(
                ModelCapabilityInfo {
                    context_window: Some(200_000),
                    supports_vision: true,
                    supports_tool_calling: true,
                    supports_json_mode: true,
                    supports_streaming: true,
                    input_cost_per_million: Some(0.25),
                    output_cost_per_million: Some(1.25),
                    max_output_tokens: Some(4_096),
                    knowledge_cutoff: Some("2024-04".to_string()),
                },
            ),
        );

        // Gemini 1.5 Pro
        registry.register(
            ModelInfo::new("gemini-1.5-pro", AiProvider::Gemini).with_capabilities(
                ModelCapabilityInfo {
                    context_window: Some(2_000_000),
                    supports_vision: true,
                    supports_tool_calling: true,
                    supports_json_mode: true,
                    supports_streaming: true,
                    input_cost_per_million: Some(1.25),
                    output_cost_per_million: Some(5.0),
                    max_output_tokens: Some(8_192),
                    knowledge_cutoff: Some("2024-04".to_string()),
                },
            ),
        );

        // Llama 3 (local via Ollama)
        registry.register(
            ModelInfo::new("llama3", AiProvider::Ollama).with_capabilities(ModelCapabilityInfo {
                context_window: Some(8_000),
                supports_vision: false,
                supports_tool_calling: false,
                supports_json_mode: false,
                supports_streaming: true,
                input_cost_per_million: None,
                output_cost_per_million: None,
                max_output_tokens: Some(4_096),
                knowledge_cutoff: None,
            }),
        );

        // Mistral (local via Ollama)
        registry.register(
            ModelInfo::new("mistral", AiProvider::Ollama).with_capabilities(ModelCapabilityInfo {
                context_window: Some(32_000),
                supports_vision: false,
                supports_tool_calling: true,
                supports_json_mode: true,
                supports_streaming: true,
                input_cost_per_million: Some(0.25),
                output_cost_per_million: Some(0.25),
                max_output_tokens: Some(4_096),
                knowledge_cutoff: None,
            }),
        );

        registry
    }

    /// Return all models that support vision/image inputs
    pub fn models_with_vision(&self) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.capabilities.as_ref().map_or(false, |c| c.supports_vision))
            .collect()
    }

    /// Return all models that support tool/function calling
    pub fn models_with_tools(&self) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| {
                m.capabilities
                    .as_ref()
                    .map_or(false, |c| c.supports_tool_calling)
            })
            .collect()
    }

    /// Return all models with at least `min_tokens` context window
    pub fn models_within_context(&self, min_tokens: usize) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| {
                m.capabilities.as_ref().map_or(false, |c| {
                    c.context_window.map_or(false, |w| w >= min_tokens)
                })
            })
            .collect()
    }

    /// Return the model with the lowest input cost per million tokens
    pub fn cheapest_model(&self) -> Option<&ModelInfo> {
        self.models
            .values()
            .filter(|m| {
                m.capabilities
                    .as_ref()
                    .and_then(|c| c.input_cost_per_million)
                    .is_some()
            })
            .min_by(|a, b| {
                let cost_a = a
                    .capabilities
                    .as_ref()
                    .expect("guaranteed by filter: capabilities is Some")
                    .input_cost_per_million
                    .expect("guaranteed by filter: input_cost_per_million is Some");
                let cost_b = b
                    .capabilities
                    .as_ref()
                    .expect("guaranteed by filter: capabilities is Some")
                    .input_cost_per_million
                    .expect("guaranteed by filter: input_cost_per_million is Some");
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_new() {
        let model = ModelInfo::new("llama3:7b", AiProvider::Ollama);
        assert_eq!(model.name, "llama3:7b");
        assert_eq!(model.provider, AiProvider::Ollama);
        assert!(model.size.is_none());
        assert!(model.modified_at.is_none());
    }

    #[test]
    fn test_model_info_with_size() {
        let model = ModelInfo::new("mistral:7b", AiProvider::Ollama).with_size("4.1 GB");
        assert_eq!(model.size, Some("4.1 GB".to_string()));
    }

    #[test]
    fn test_model_info_display_name() {
        let model = ModelInfo::new("llama3:7b", AiProvider::Ollama);
        assert_eq!(model.display_name(), "llama3:7b");

        let model_with_size = ModelInfo::new("llama3:7b", AiProvider::Ollama).with_size("3.8 GB");
        assert_eq!(model_with_size.display_name(), "llama3:7b (3.8 GB)");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.0 GB");
        assert_eq!(format_size(512 * 1024 * 1024), "512 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100 MB");
    }

    #[test]
    fn test_model_capability_info_default() {
        let cap = ModelCapabilityInfo::default();
        assert!(cap.context_window.is_none());
        assert!(!cap.supports_vision);
        assert!(!cap.supports_tool_calling);
        assert!(!cap.supports_json_mode);
        assert!(!cap.supports_streaming);
        assert!(cap.input_cost_per_million.is_none());
        assert!(cap.output_cost_per_million.is_none());
        assert!(cap.max_output_tokens.is_none());
        assert!(cap.knowledge_cutoff.is_none());
    }

    #[test]
    fn test_model_info_with_capabilities() {
        let cap = ModelCapabilityInfo {
            context_window: Some(128_000),
            supports_vision: true,
            supports_tool_calling: true,
            ..Default::default()
        };
        let model = ModelInfo::new("gpt-4o", AiProvider::OpenAI).with_capabilities(cap);
        assert!(model.capabilities.is_some());
        let c = model.capabilities.unwrap();
        assert_eq!(c.context_window, Some(128_000));
        assert!(c.supports_vision);
        assert!(c.supports_tool_calling);
        assert!(!c.supports_json_mode);
    }

    #[test]
    fn test_model_registry_crud() {
        let mut registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);
        assert!(registry.get("gpt-4o").is_none());

        registry.register(ModelInfo::new("gpt-4o", AiProvider::OpenAI));
        assert_eq!(registry.model_count(), 1);

        let model = registry.get("gpt-4o");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "gpt-4o");
        assert_eq!(model.unwrap().provider, AiProvider::OpenAI);

        // Overwrite with same name
        registry.register(ModelInfo::new("gpt-4o", AiProvider::OpenAI).with_size("large"));
        assert_eq!(registry.model_count(), 1);
    }

    #[test]
    fn test_known_models_populated() {
        let registry = ModelRegistry::with_known_models();
        assert_eq!(registry.model_count(), 7);

        assert!(registry.get("gpt-4o").is_some());
        assert!(registry.get("gpt-4o-mini").is_some());
        assert!(registry.get("claude-3.5-sonnet").is_some());
        assert!(registry.get("claude-3-haiku").is_some());
        assert!(registry.get("gemini-1.5-pro").is_some());
        assert!(registry.get("llama3").is_some());
        assert!(registry.get("mistral").is_some());

        // Verify a specific model's capabilities
        let gpt4o = registry.get("gpt-4o").unwrap();
        let cap = gpt4o.capabilities.as_ref().unwrap();
        assert_eq!(cap.context_window, Some(128_000));
        assert!(cap.supports_vision);
        assert!(cap.supports_tool_calling);
    }

    #[test]
    fn test_filter_vision_models() {
        let registry = ModelRegistry::with_known_models();
        let vision_models = registry.models_with_vision();

        // gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-3-haiku, gemini-1.5-pro
        assert_eq!(vision_models.len(), 5);

        let names: Vec<&str> = vision_models.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"gpt-4o"));
        assert!(names.contains(&"claude-3.5-sonnet"));
        assert!(names.contains(&"gemini-1.5-pro"));
        // llama3 and mistral do NOT have vision
        assert!(!names.contains(&"llama3"));
        assert!(!names.contains(&"mistral"));
    }

    #[test]
    fn test_filter_by_context() {
        let registry = ModelRegistry::with_known_models();

        // 100k+ context: gpt-4o (128k), gpt-4o-mini (128k), claude-3.5-sonnet (200k),
        // claude-3-haiku (200k), gemini-1.5-pro (2M)
        let large_ctx = registry.models_within_context(100_000);
        assert_eq!(large_ctx.len(), 5);

        let names: Vec<&str> = large_ctx.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"gpt-4o"));
        assert!(names.contains(&"gemini-1.5-pro"));
        assert!(!names.contains(&"llama3")); // 8k
        assert!(!names.contains(&"mistral")); // 32k

        // 1M+ context: only gemini-1.5-pro (2M)
        let huge_ctx = registry.models_within_context(1_000_000);
        assert_eq!(huge_ctx.len(), 1);
        assert_eq!(huge_ctx[0].name, "gemini-1.5-pro");
    }
}
