//! An [`LlmEnhancer`] backed by the assistant's own configured model (V205).
//!
//! Lets optional LLM-enhanced features (e.g. arbitrary personal-fact extraction
//! in [`crate::fact_extraction`]) run against the same provider/model the user
//! already configured, without the caller wiring up a separate client. Each
//! [`generate`](SelfChatEnhancer::generate) call spins a fresh, memory-less
//! [`AiAssistant`] and drives one completion to the end — memory-less so it can
//! never recurse back into the enhancer that invoked it.

use std::time::{Duration, Instant};

use crate::config::AiConfig;
use crate::llm_enhance::LlmEnhancer;
use crate::messages::AiResponse;
use crate::AiAssistant;

/// An [`LlmEnhancer`] that answers by running a one-shot completion on a fresh
/// assistant configured exactly like the caller's.
pub struct SelfChatEnhancer {
    config: AiConfig,
    timeout: Duration,
}

impl SelfChatEnhancer {
    /// Create an enhancer that uses `config`'s provider/model. Default 60s
    /// per-call timeout.
    pub fn new(config: AiConfig) -> Self {
        Self {
            config,
            timeout: Duration::from_secs(60),
        }
    }

    /// Override the per-call timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl LlmEnhancer for SelfChatEnhancer {
    fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String, String> {
        let mut assistant = AiAssistant::new();
        assistant.config = self.config.clone();
        // Deliberately memory-less: no fact ledger / extractor is attached, so
        // this call cannot recurse into another extraction.
        assistant.send_message(prompt.to_string(), "");

        let start = Instant::now();
        loop {
            match assistant.poll_response() {
                Some(AiResponse::Complete(text)) | Some(AiResponse::Cancelled(text)) => {
                    return Ok(text)
                }
                Some(AiResponse::Error(e)) => return Err(e),
                _ => {}
            }
            if start.elapsed() > self.timeout {
                return Err("SelfChatEnhancer: generation timed out".to_string());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
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
