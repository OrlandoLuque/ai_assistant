//! # Virtual Model Registry (v31)
//!
//! A virtual model bundles an `EnrichmentConfig` (RAG, guardrails, compaction,
//! thinking, cost, model selection) + model profile + system prompt as a named
//! "model" that any API consumer (web, SDK, other cluster nodes) can request.
//!
//! ## Key concepts
//!
//! - **VirtualModel**: A configuration bundle exposed as a model name.
//! - **PublishedModel**: Controls visibility of a physical (local) model.
//! - **ModelRegistry**: Central registry for both virtual and physical models.
//! - **ModelResolution**: Result of resolving a model name — virtual, physical, or pass-through.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ai_assistant::virtual_model::{ModelRegistry, VirtualModel};
//! use ai_assistant::server::EnrichmentConfig;
//!
//! let registry = ModelRegistry::new();
//! let vmodel = VirtualModel {
//!     name: "my-rag-assistant".to_string(),
//!     description: "RAG-enabled assistant with guardrails".to_string(),
//!     base_model: "llama3:8b".to_string(),
//!     base_provider: None,
//!     enrichment: EnrichmentConfig::default(),
//!     profile: None,
//!     system_prompt: Some("You are a helpful assistant.".to_string()),
//!     published: true,
//!     created_at: 0,
//!     tags: vec!["rag".to_string()],
//! };
//! registry.register_virtual(vmodel).unwrap();
//! ```

use std::path::Path;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::config::AiProvider;
use crate::models::ModelInfo;
use crate::server::EnrichmentConfig;

/// A virtual model bundles configuration as a named "model" for API consumers.
///
/// Clients request this model by name (e.g. `model: "my-rag-assistant"` in
/// an OpenAI-compatible request), and the server transparently applies the
/// full enrichment pipeline behind the scenes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModel {
    /// Unique name (used as model ID in API requests).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The actual LLM model to use (e.g. "llama3:8b", "gpt-4o").
    pub base_model: String,
    /// Provider override (None = use server default).
    pub base_provider: Option<AiProvider>,
    /// Full enrichment pipeline config (RAG, guardrails, compaction, etc.).
    pub enrichment: EnrichmentConfig,
    /// Generation parameters (temperature, top_p, etc.).
    pub profile: Option<crate::profiles::ModelProfile>,
    /// Default system prompt prepended to user's system prompt.
    pub system_prompt: Option<String>,
    /// Whether this model is visible to API clients and cluster peers.
    pub published: bool,
    /// Creation timestamp (Unix seconds).
    pub created_at: u64,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

/// Controls visibility of a physical (local) model to external API clients.
///
/// By default, physical models are NOT published — the admin must explicitly
/// make them visible. This prevents accidental exposure of local models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedModel {
    /// Model name as reported by the provider (e.g. "llama3:8b").
    pub name: String,
    /// Provider.
    pub provider: AiProvider,
    /// Whether visible to external clients.
    pub published: bool,
    /// Optional display name override (shown to clients instead of raw name).
    pub display_name: Option<String>,
}

/// Result of resolving a model name from the registry.
#[derive(Debug, Clone)]
pub enum ModelResolution {
    /// Physical model — use as-is.
    Physical {
        name: String,
        provider: Option<AiProvider>,
    },
    /// Virtual model — apply enrichment pipeline.
    Virtual(VirtualModel),
    /// Not found in registry — fall back to default behavior.
    PassThrough {
        name: String,
    },
}

/// A model as seen by API clients (OpenAI /v1/models response format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientModel {
    /// Model identifier (used in API requests).
    pub id: String,
    /// Always "model".
    pub object: String,
    /// Creation timestamp.
    pub created: u64,
    /// Owner (provider name or "virtual").
    pub owned_by: String,
    /// "physical" or "virtual".
    pub model_type: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Tags.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// Serializable wrapper for persisting the registry to disk.
#[derive(Debug, Serialize, Deserialize)]
struct RegistrySnapshot {
    virtual_models: Vec<VirtualModel>,
    published_models: Vec<PublishedModel>,
}

/// Central registry for model visibility and virtual models.
///
/// Uses `DashMap` for lock-free concurrent access, consistent with
/// the `AppState` pattern in `server_axum.rs`.
#[derive(Debug)]
pub struct ModelRegistry {
    /// Virtual model definitions (name → VirtualModel).
    virtual_models: DashMap<String, VirtualModel>,
    /// Physical model publish status (name → PublishedModel).
    published_models: DashMap<String, PublishedModel>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            virtual_models: DashMap::new(),
            published_models: DashMap::new(),
        }
    }

    // ── Virtual Model CRUD ──────────────────────────────────────────────

    /// Register a virtual model. Returns error if name already exists.
    pub fn register_virtual(&self, model: VirtualModel) -> Result<(), String> {
        if model.name.is_empty() {
            return Err("Virtual model name cannot be empty".to_string());
        }
        if self.virtual_models.contains_key(&model.name) {
            return Err(format!("Virtual model '{}' already exists", model.name));
        }
        self.virtual_models.insert(model.name.clone(), model);
        Ok(())
    }

    /// Update an existing virtual model. Returns false if not found.
    pub fn update_virtual(&self, model: VirtualModel) -> bool {
        if self.virtual_models.contains_key(&model.name) {
            self.virtual_models.insert(model.name.clone(), model);
            true
        } else {
            false
        }
    }

    /// Unregister a virtual model. Returns true if it existed.
    pub fn unregister_virtual(&self, name: &str) -> bool {
        self.virtual_models.remove(name).is_some()
    }

    /// Get a virtual model by name.
    pub fn get_virtual(&self, name: &str) -> Option<VirtualModel> {
        self.virtual_models.get(name).map(|v| v.clone())
    }

    /// List all virtual models (including unpublished).
    pub fn list_virtual(&self) -> Vec<VirtualModel> {
        self.virtual_models.iter().map(|v| v.value().clone()).collect()
    }

    /// List only published virtual models.
    pub fn list_published_virtual(&self) -> Vec<VirtualModel> {
        self.virtual_models
            .iter()
            .filter(|v| v.published)
            .map(|v| v.value().clone())
            .collect()
    }

    // ── Physical Model Publish Control ──────────────────────────────────

    /// Set the published status of a physical model.
    pub fn set_published(&self, name: &str, provider: AiProvider, published: bool) {
        self.published_models.insert(
            name.to_string(),
            PublishedModel {
                name: name.to_string(),
                provider,
                published,
                display_name: None,
            },
        );
    }

    /// Set the published status with a display name override.
    pub fn set_published_with_display_name(
        &self,
        name: &str,
        provider: AiProvider,
        published: bool,
        display_name: Option<String>,
    ) {
        self.published_models.insert(
            name.to_string(),
            PublishedModel {
                name: name.to_string(),
                provider,
                published,
                display_name,
            },
        );
    }

    /// Check if a physical model is published.
    pub fn is_published(&self, name: &str) -> bool {
        self.published_models
            .get(name)
            .map(|p| p.published)
            .unwrap_or(false) // Not registered = not published
    }

    /// Get publish info for a physical model.
    pub fn get_published(&self, name: &str) -> Option<PublishedModel> {
        self.published_models.get(name).map(|p| p.clone())
    }

    /// List all physical model publish records.
    pub fn list_published_physical(&self) -> Vec<PublishedModel> {
        self.published_models
            .iter()
            .filter(|p| p.published)
            .map(|p| p.value().clone())
            .collect()
    }

    // ── Model Resolution ────────────────────────────────────────────────

    /// Resolve a model name to determine how to handle the request.
    ///
    /// Resolution order:
    /// 1. Virtual model (by name)
    /// 2. Published physical model (by name)
    /// 3. PassThrough (use as raw model name, default behavior)
    pub fn resolve(&self, model_name: &str) -> ModelResolution {
        // 1. Check virtual models first
        if let Some(vmodel) = self.virtual_models.get(model_name) {
            return ModelResolution::Virtual(vmodel.clone());
        }

        // 2. Check published physical models
        if let Some(pmodel) = self.published_models.get(model_name) {
            return ModelResolution::Physical {
                name: pmodel.name.clone(),
                provider: Some(pmodel.provider.clone()),
            };
        }

        // 3. Fall through — use as raw model name
        ModelResolution::PassThrough {
            name: model_name.to_string(),
        }
    }

    // ── Client-visible listing ──────────────────────────────────────────

    /// List all models visible to API clients.
    ///
    /// Includes:
    /// - Published physical models (filtered from `available_models`)
    /// - Published virtual models
    ///
    /// Physical models not in the registry are NOT shown by default.
    pub fn list_client_visible(&self, available_models: &[ModelInfo]) -> Vec<ClientModel> {
        let mut result = Vec::new();

        // Published physical models
        for model in available_models {
            if self.is_published(&model.name) {
                let display = self
                    .published_models
                    .get(&model.name)
                    .and_then(|p| p.display_name.clone());
                result.push(ClientModel {
                    id: display.unwrap_or_else(|| model.name.clone()),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: format!("{:?}", model.provider).to_lowercase(),
                    model_type: "physical".to_string(),
                    description: None,
                    tags: Vec::new(),
                });
            }
        }

        // Published virtual models
        for vmodel in self.list_published_virtual() {
            result.push(ClientModel {
                id: vmodel.name.clone(),
                object: "model".to_string(),
                created: vmodel.created_at,
                owned_by: "virtual".to_string(),
                model_type: "virtual".to_string(),
                description: Some(vmodel.description.clone()),
                tags: vmodel.tags.clone(),
            });
        }

        result
    }

    // ── Persistence ─────────────────────────────────────────────────────

    /// Save the registry to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let snapshot = RegistrySnapshot {
            virtual_models: self.list_virtual(),
            published_models: self
                .published_models
                .iter()
                .map(|p| p.value().clone())
                .collect(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Serialization error: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("Write error: {}", e))
    }

    /// Load the registry from a JSON file. Returns a new registry.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let snapshot: RegistrySnapshot =
            serde_json::from_str(&json).map_err(|e| format!("Parse error: {}", e))?;

        let registry = Self::new();
        for vmodel in snapshot.virtual_models {
            registry.virtual_models.insert(vmodel.name.clone(), vmodel);
        }
        for pmodel in snapshot.published_models {
            registry.published_models.insert(pmodel.name.clone(), pmodel);
        }
        Ok(registry)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_virtual_model(name: &str, published: bool) -> VirtualModel {
        VirtualModel {
            name: name.to_string(),
            description: format!("Test model {}", name),
            base_model: "llama3:8b".to_string(),
            base_provider: None,
            enrichment: EnrichmentConfig::default(),
            profile: None,
            system_prompt: Some("You are helpful.".to_string()),
            published,
            created_at: 1000,
            tags: vec!["test".to_string()],
        }
    }

    fn make_model_info(name: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            provider: AiProvider::Ollama,
            size: None,
            modified_at: None,
            capabilities: None,
        }
    }

    // ── VirtualModel CRUD ───────────────────────────────────────────────

    #[test]
    fn test_register_virtual_model() {
        let registry = ModelRegistry::new();
        let model = make_virtual_model("my-assistant", true);
        assert!(registry.register_virtual(model).is_ok());
        assert_eq!(registry.list_virtual().len(), 1);
    }

    #[test]
    fn test_register_duplicate_virtual_model() {
        let registry = ModelRegistry::new();
        let model1 = make_virtual_model("my-assistant", true);
        let model2 = make_virtual_model("my-assistant", false);
        assert!(registry.register_virtual(model1).is_ok());
        assert!(registry.register_virtual(model2).is_err());
    }

    #[test]
    fn test_register_empty_name_rejected() {
        let registry = ModelRegistry::new();
        let mut model = make_virtual_model("test", true);
        model.name = String::new();
        assert!(registry.register_virtual(model).is_err());
    }

    #[test]
    fn test_get_virtual_model() {
        let registry = ModelRegistry::new();
        let model = make_virtual_model("my-assistant", true);
        registry.register_virtual(model).unwrap();

        let found = registry.get_virtual("my-assistant");
        assert!(found.is_some());
        assert_eq!(found.unwrap().base_model, "llama3:8b");
    }

    #[test]
    fn test_get_virtual_model_not_found() {
        let registry = ModelRegistry::new();
        assert!(registry.get_virtual("nonexistent").is_none());
    }

    #[test]
    fn test_update_virtual_model() {
        let registry = ModelRegistry::new();
        let model = make_virtual_model("my-assistant", true);
        registry.register_virtual(model).unwrap();

        let mut updated = make_virtual_model("my-assistant", true);
        updated.base_model = "gpt-4o".to_string();
        assert!(registry.update_virtual(updated));

        let found = registry.get_virtual("my-assistant").unwrap();
        assert_eq!(found.base_model, "gpt-4o");
    }

    #[test]
    fn test_update_virtual_model_not_found() {
        let registry = ModelRegistry::new();
        let model = make_virtual_model("nonexistent", true);
        assert!(!registry.update_virtual(model));
    }

    #[test]
    fn test_unregister_virtual_model() {
        let registry = ModelRegistry::new();
        let model = make_virtual_model("my-assistant", true);
        registry.register_virtual(model).unwrap();
        assert!(registry.unregister_virtual("my-assistant"));
        assert!(registry.get_virtual("my-assistant").is_none());
    }

    #[test]
    fn test_unregister_nonexistent() {
        let registry = ModelRegistry::new();
        assert!(!registry.unregister_virtual("nonexistent"));
    }

    #[test]
    fn test_list_virtual_models() {
        let registry = ModelRegistry::new();
        registry.register_virtual(make_virtual_model("a", true)).unwrap();
        registry.register_virtual(make_virtual_model("b", false)).unwrap();
        registry.register_virtual(make_virtual_model("c", true)).unwrap();
        assert_eq!(registry.list_virtual().len(), 3);
    }

    #[test]
    fn test_list_published_virtual_models() {
        let registry = ModelRegistry::new();
        registry.register_virtual(make_virtual_model("a", true)).unwrap();
        registry.register_virtual(make_virtual_model("b", false)).unwrap();
        registry.register_virtual(make_virtual_model("c", true)).unwrap();
        let published = registry.list_published_virtual();
        assert_eq!(published.len(), 2);
    }

    // ── PublishedModel ──────────────────────────────────────────────────

    #[test]
    fn test_publish_physical_model() {
        let registry = ModelRegistry::new();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);
        assert!(registry.is_published("llama3:8b"));
    }

    #[test]
    fn test_unpublish_physical_model() {
        let registry = ModelRegistry::new();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);
        assert!(registry.is_published("llama3:8b"));
        registry.set_published("llama3:8b", AiProvider::Ollama, false);
        assert!(!registry.is_published("llama3:8b"));
    }

    #[test]
    fn test_unpublished_by_default() {
        let registry = ModelRegistry::new();
        assert!(!registry.is_published("unknown-model"));
    }

    #[test]
    fn test_get_published_info() {
        let registry = ModelRegistry::new();
        registry.set_published_with_display_name(
            "llama3:8b",
            AiProvider::Ollama,
            true,
            Some("Llama 3 8B".to_string()),
        );
        let info = registry.get_published("llama3:8b").unwrap();
        assert_eq!(info.display_name, Some("Llama 3 8B".to_string()));
    }

    #[test]
    fn test_list_published_physical() {
        let registry = ModelRegistry::new();
        registry.set_published("a", AiProvider::Ollama, true);
        registry.set_published("b", AiProvider::Ollama, false);
        registry.set_published("c", AiProvider::OpenAI, true);
        assert_eq!(registry.list_published_physical().len(), 2);
    }

    // ── Model Resolution ────────────────────────────────────────────────

    #[test]
    fn test_resolve_virtual_model() {
        let registry = ModelRegistry::new();
        registry.register_virtual(make_virtual_model("my-rag", true)).unwrap();
        match registry.resolve("my-rag") {
            ModelResolution::Virtual(v) => assert_eq!(v.base_model, "llama3:8b"),
            _ => panic!("Expected Virtual resolution"),
        }
    }

    #[test]
    fn test_resolve_physical_model() {
        let registry = ModelRegistry::new();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);
        match registry.resolve("llama3:8b") {
            ModelResolution::Physical { name, provider } => {
                assert_eq!(name, "llama3:8b");
                assert!(provider.is_some());
            }
            _ => panic!("Expected Physical resolution"),
        }
    }

    #[test]
    fn test_resolve_passthrough() {
        let registry = ModelRegistry::new();
        match registry.resolve("unknown-model") {
            ModelResolution::PassThrough { name } => assert_eq!(name, "unknown-model"),
            _ => panic!("Expected PassThrough resolution"),
        }
    }

    #[test]
    fn test_resolve_virtual_takes_priority_over_physical() {
        let registry = ModelRegistry::new();
        // Same name as both virtual and physical
        registry.register_virtual(make_virtual_model("dual", true)).unwrap();
        registry.set_published("dual", AiProvider::Ollama, true);
        // Virtual should win
        assert!(matches!(registry.resolve("dual"), ModelResolution::Virtual(_)));
    }

    // ── Client Visible Listing ──────────────────────────────────────────

    #[test]
    fn test_list_client_visible_empty() {
        let registry = ModelRegistry::new();
        let available = vec![make_model_info("llama3:8b")];
        // Nothing published → nothing visible
        assert!(registry.list_client_visible(&available).is_empty());
    }

    #[test]
    fn test_list_client_visible_published_physical() {
        let registry = ModelRegistry::new();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);
        let available = vec![make_model_info("llama3:8b"), make_model_info("mistral:7b")];
        let visible = registry.list_client_visible(&available);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "llama3:8b");
        assert_eq!(visible[0].model_type, "physical");
    }

    #[test]
    fn test_list_client_visible_published_virtual() {
        let registry = ModelRegistry::new();
        registry.register_virtual(make_virtual_model("my-rag", true)).unwrap();
        let available: Vec<ModelInfo> = vec![];
        let visible = registry.list_client_visible(&available);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id, "my-rag");
        assert_eq!(visible[0].model_type, "virtual");
        assert_eq!(visible[0].owned_by, "virtual");
    }

    #[test]
    fn test_list_client_visible_mixed() {
        let registry = ModelRegistry::new();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);
        registry.register_virtual(make_virtual_model("my-rag", true)).unwrap();
        registry.register_virtual(make_virtual_model("hidden", false)).unwrap();
        let available = vec![make_model_info("llama3:8b"), make_model_info("mistral:7b")];
        let visible = registry.list_client_visible(&available);
        // llama3:8b (published) + my-rag (published virtual) = 2
        // mistral:7b (not published) + hidden (not published virtual) excluded
        assert_eq!(visible.len(), 2);
    }

    #[test]
    fn test_list_client_visible_with_display_name() {
        let registry = ModelRegistry::new();
        registry.set_published_with_display_name(
            "llama3:8b",
            AiProvider::Ollama,
            true,
            Some("Llama 3".to_string()),
        );
        let available = vec![make_model_info("llama3:8b")];
        let visible = registry.list_client_visible(&available);
        assert_eq!(visible[0].id, "Llama 3"); // Display name used
    }

    // ── Serialization ───────────────────────────────────────────────────

    #[test]
    fn test_virtual_model_serialization_roundtrip() {
        let model = make_virtual_model("test", true);
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: VirtualModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.base_model, "llama3:8b");
        assert!(deserialized.published);
    }

    #[test]
    fn test_published_model_serialization() {
        let model = PublishedModel {
            name: "test".to_string(),
            provider: AiProvider::Ollama,
            published: true,
            display_name: Some("Test Model".to_string()),
        };
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: PublishedModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert!(deserialized.published);
    }

    #[test]
    fn test_client_model_serialization() {
        let model = ClientModel {
            id: "test".to_string(),
            object: "model".to_string(),
            created: 1000,
            owned_by: "virtual".to_string(),
            model_type: "virtual".to_string(),
            description: Some("A test model".to_string()),
            tags: vec!["rag".to_string()],
        };
        let json = serde_json::to_value(&model).unwrap();
        assert_eq!(json["id"], "test");
        assert_eq!(json["object"], "model");
        assert_eq!(json["model_type"], "virtual");
    }

    #[test]
    fn test_save_and_load_registry() {
        let registry = ModelRegistry::new();
        registry.register_virtual(make_virtual_model("v1", true)).unwrap();
        registry.register_virtual(make_virtual_model("v2", false)).unwrap();
        registry.set_published("llama3:8b", AiProvider::Ollama, true);

        let path = std::env::temp_dir().join("test_model_registry.json");
        registry.save_to_file(&path).unwrap();

        let loaded = ModelRegistry::load_from_file(&path).unwrap();
        assert_eq!(loaded.list_virtual().len(), 2);
        assert!(loaded.is_published("llama3:8b"));
        assert!(loaded.get_virtual("v1").is_some());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    // ── Concurrent Access ───────────────────────────────────────────────

    #[test]
    fn test_concurrent_virtual_model_access() {
        let registry = std::sync::Arc::new(ModelRegistry::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let r = registry.clone();
                std::thread::spawn(move || {
                    let name = format!("model-{}", i);
                    r.register_virtual(make_virtual_model(&name, true)).unwrap();
                    assert!(r.get_virtual(&name).is_some());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(registry.list_virtual().len(), 10);
    }

    #[test]
    fn test_concurrent_publish_access() {
        let registry = std::sync::Arc::new(ModelRegistry::new());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let r = registry.clone();
                std::thread::spawn(move || {
                    let name = format!("model-{}", i);
                    r.set_published(&name, AiProvider::Ollama, true);
                    assert!(r.is_published(&name));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(registry.list_published_physical().len(), 10);
    }

    #[test]
    fn test_default_registry() {
        let registry = ModelRegistry::default();
        assert!(registry.list_virtual().is_empty());
        assert!(registry.list_published_physical().is_empty());
    }
}
