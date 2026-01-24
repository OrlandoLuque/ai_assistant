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
