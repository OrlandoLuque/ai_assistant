//! Model information types

use serde::{Deserialize, Serialize};
use crate::config::AiProvider;

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
}

impl ModelInfo {
    /// Create a new ModelInfo
    pub fn new(name: impl Into<String>, provider: AiProvider) -> Self {
        Self {
            name: name.into(),
            provider,
            size: None,
            modified_at: None,
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
        let model = ModelInfo::new("mistral:7b", AiProvider::Ollama)
            .with_size("4.1 GB");
        assert_eq!(model.size, Some("4.1 GB".to_string()));
    }

    #[test]
    fn test_model_info_display_name() {
        let model = ModelInfo::new("llama3:7b", AiProvider::Ollama);
        assert_eq!(model.display_name(), "llama3:7b");

        let model_with_size = ModelInfo::new("llama3:7b", AiProvider::Ollama)
            .with_size("3.8 GB");
        assert_eq!(model_with_size.display_name(), "llama3:7b (3.8 GB)");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.0 GB");
        assert_eq!(format_size(512 * 1024 * 1024), "512 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100 MB");
    }
}
