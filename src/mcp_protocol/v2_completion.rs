//! MCP Completions & Suggestions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The kind of reference a completion request targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum CompletionRefType {
    #[serde(rename = "resource_uri")]
    ResourceUri(String),
    #[serde(rename = "prompt_name")]
    PromptName(String),
}

/// A request for argument completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub ref_type: CompletionRefType,
    pub argument_name: String,
    pub partial_value: String,
}

/// A single completion suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionSuggestion {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The result of a completion request, containing matching suggestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    pub suggestions: Vec<CompletionSuggestion>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
}

/// Trait for components that provide argument completions.
pub trait CompletionProvider: Send + Sync {
    /// Return completions matching the request.
    fn complete(&self, request: &CompletionRequest) -> CompletionResult;
    /// Whether this provider can handle the given reference type.
    fn supports_ref_type(&self, ref_type: &CompletionRefType) -> bool;
}

/// A `CompletionProvider` backed by a static set of values per argument name.
pub struct StaticCompletionProvider {
    values: HashMap<String, Vec<String>>,
}

impl StaticCompletionProvider {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Register a list of possible values for an argument.
    pub fn add_values(&mut self, argument_name: String, values: Vec<String>) {
        self.values.insert(argument_name, values);
    }

    /// Remove all values for an argument.
    pub fn remove_values(&mut self, argument_name: &str) {
        self.values.remove(argument_name);
    }
}

impl Default for StaticCompletionProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionProvider for StaticCompletionProvider {
    fn complete(&self, request: &CompletionRequest) -> CompletionResult {
        let matching = match self.values.get(&request.argument_name) {
            Some(vals) => {
                let prefix = request.partial_value.to_lowercase();
                vals.iter()
                    .filter(|v| v.to_lowercase().starts_with(&prefix))
                    .map(|v| CompletionSuggestion {
                        value: v.clone(),
                        label: None,
                        description: None,
                    })
                    .collect::<Vec<_>>()
            }
            None => Vec::new(),
        };
        let total = matching.len();
        CompletionResult {
            suggestions: matching,
            has_more: false,
            total: Some(total),
        }
    }

    fn supports_ref_type(&self, _ref_type: &CompletionRefType) -> bool {
        true // static provider is reference-type agnostic
    }
}

/// Registry that aggregates multiple `CompletionProvider` implementations.
pub struct CompletionRegistry {
    providers: Vec<Box<dyn CompletionProvider>>,
}

impl CompletionRegistry {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register a new completion provider.
    pub fn register(&mut self, provider: Box<dyn CompletionProvider>) {
        self.providers.push(provider);
    }

    /// Query all registered providers and merge their results.
    pub fn complete(&self, request: &CompletionRequest) -> CompletionResult {
        let mut all_suggestions: Vec<CompletionSuggestion> = Vec::new();
        let mut any_has_more = false;
        let mut total_count: usize = 0;

        for provider in &self.providers {
            if provider.supports_ref_type(&request.ref_type) {
                let result = provider.complete(request);
                all_suggestions.extend(result.suggestions);
                if result.has_more {
                    any_has_more = true;
                }
                if let Some(t) = result.total {
                    total_count += t;
                }
            }
        }

        let final_total = if total_count > 0 || !self.providers.is_empty() {
            Some(all_suggestions.len())
        } else {
            None
        };

        CompletionResult {
            suggestions: all_suggestions,
            has_more: any_has_more,
            total: final_total,
        }
    }

    /// Return how many providers are registered.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

impl Default for CompletionRegistry {
    fn default() -> Self {
        Self::new()
    }
}
