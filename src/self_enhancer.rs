//! An [`LlmEnhancer`] backed by the configured model, via the [`LlmProvider`]
//! port (V205; migrated onto the port in the phase-1 hexagonalization).
//!
//! Lets optional LLM-enhanced features (e.g. arbitrary personal-fact extraction
//! in [`crate::fact_extraction`]) run against the same provider/model the user
//! already configured — or a different, stronger one on another machine —
//! without the caller wiring up a client. It depends on the [`LlmProvider`]
//! abstraction, not on a full [`crate::AiAssistant`], so each call is a single
//! stateless completion with no risk of recursing back into the enhancer that
//! invoked it.

use crate::config::AiConfig;
use crate::llm_enhance::LlmEnhancer;
use crate::llm_provider::provider_from_config;
use crate::messages::ChatMessage;

/// An [`LlmEnhancer`] that answers with a one-shot completion through the
/// [`LlmProvider`] port, using `config`'s provider/model/URLs.
pub struct SelfChatEnhancer {
    config: AiConfig,
}

impl SelfChatEnhancer {
    /// Create an enhancer that uses `config`'s provider/model.
    pub fn new(config: AiConfig) -> Self {
        Self { config }
    }
}

impl LlmEnhancer for SelfChatEnhancer {
    fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String, String> {
        // Depend on the LlmProvider port via the phase-3 factory (Ollama ->
        // OllamaAdapter, others -> ConfigLlmProvider) — not a whole AiAssistant:
        // a single stateless call (no memory/extractor attached), so it can
        // never recurse. The prompt is self-contained, so no system prompt.
        let provider = provider_from_config(self.config.clone());
        provider
            .generate(&[ChatMessage::user(prompt)], "")
            .map_err(|e| e.to_string())
    }

    fn model_name(&self) -> &str {
        &self.config.selected_model
    }

    fn is_available(&self) -> bool {
        // The configured provider is assumed reachable; a failed call surfaces
        // as an Err from `generate` and the caller falls back to heuristics.
        true
    }
}
