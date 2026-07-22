//! LLM provider **port** — hexagonal architecture, phase 1.
//!
//! Historically the core LLM path dispatched by `match`ing on the [`AiProvider`]
//! enum with HTTP calls inlined in [`crate::providers`]. That is fast to write
//! but couples the domain (the assistant, agents, the fact extractor, tests)
//! directly to the transport: you cannot substitute or mock the model without a
//! live server, and the "which provider" decision is data, not a swappable
//! component.
//!
//! This module introduces a trait-based **port**, [`LlmProvider`], so callers
//! depend on an abstraction. Implementors are the swappable **adapters**: a real
//! backend, a remote endpoint, or a test mock.
//!
//! Phase 1 is deliberately **additive** (strangler-fig): the default adapter,
//! [`ConfigLlmProvider`], simply delegates to the existing
//! `providers::generate_response*` functions, so behaviour is unchanged. Later
//! phases migrate callers onto the port, split each provider into its own
//! adapter, and route transport through the `HttpClient` port.
//!
//! [`AiProvider`]: crate::config::AiProvider

use std::sync::mpsc::Sender;

use anyhow::Result;

use crate::config::AiConfig;
use crate::conversation_control::CancellationToken;
use crate::messages::{AiResponse, ChatMessage};

/// Port for turning a conversation into an LLM response.
///
/// Implementors are the interchangeable adapters. Must be `Send + Sync`: the
/// assistant drives generation from a spawned worker thread, and adapters are
/// shared across threads behind trait objects / `Arc`.
pub trait LlmProvider: Send + Sync {
    /// Generate a full (non-streaming) completion for `conversation` under
    /// `system_prompt`.
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String>;

    /// Generate a streaming completion, emitting [`AiResponse`] chunks on `tx`.
    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()>;

    /// Streaming generation that stops early once `cancel_token` is tripped.
    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()>;
}

/// Default adapter: holds an [`AiConfig`] and delegates to the existing
/// enum-dispatch `providers::generate_response*` functions.
///
/// Introduced without changing behaviour, so the port can be adopted
/// incrementally — every call routes to exactly the same code the assistant
/// used before.
pub struct ConfigLlmProvider {
    config: AiConfig,
}

impl ConfigLlmProvider {
    /// Build an adapter bound to `config`; its `provider`/URLs/keys select the
    /// backend exactly as before.
    pub fn new(config: AiConfig) -> Self {
        Self { config }
    }

    /// The configuration this adapter dispatches on.
    pub fn config(&self) -> &AiConfig {
        &self.config
    }
}

impl LlmProvider for ConfigLlmProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::providers::generate_response(&self.config, conversation, system_prompt)
    }

    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        crate::providers::generate_response_streaming(&self.config, conversation, system_prompt, tx)
    }

    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        crate::providers::generate_response_streaming_cancellable(
            &self.config,
            conversation,
            system_prompt,
            tx,
            cancel_token,
        )
    }
}

/// A trivial [`LlmProvider`] that answers every request with a fixed reply.
///
/// The whole point of the port: inject this via
/// [`crate::AiAssistant::set_llm_provider`] to exercise the domain
/// (`send_message` / agents) **without a live model server** — deterministic,
/// offline tests. Also handy in examples.
pub struct MockLlmProvider {
    reply: String,
}

impl MockLlmProvider {
    /// A mock that returns `reply` for every generation.
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

impl LlmProvider for MockLlmProvider {
    fn generate(&self, _conversation: &[ChatMessage], _system_prompt: &str) -> Result<String> {
        Ok(self.reply.clone())
    }
    fn generate_streaming(
        &self,
        _conversation: &[ChatMessage],
        _system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let _ = tx.send(AiResponse::Complete(self.reply.clone()));
        Ok(())
    }
    fn generate_streaming_cancellable(
        &self,
        _conversation: &[ChatMessage],
        _system_prompt: &str,
        tx: &Sender<AiResponse>,
        _cancel_token: &CancellationToken,
    ) -> Result<()> {
        let _ = tx.send(AiResponse::Complete(self.reply.clone()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn port_is_object_safe_send_sync_and_mockable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn LlmProvider>>();

        let provider: Box<dyn LlmProvider> = Box::new(MockLlmProvider::new("hola"));
        assert_eq!(provider.generate(&[], "sys").unwrap(), "hola");

        let (tx, rx) = channel();
        provider
            .generate_streaming(&[ChatMessage::user("hi")], "sys", &tx)
            .unwrap();
        assert!(matches!(rx.recv().unwrap(), AiResponse::Complete(t) if t == "hola"));
    }

    #[test]
    fn config_adapter_binds_and_exposes_config() {
        let cfg = AiConfig::default();
        let model = cfg.selected_model.clone();
        let adapter = ConfigLlmProvider::new(cfg);
        assert_eq!(adapter.config().selected_model, model);
    }

    /// The payoff of phase 2: a full `AiAssistant` generates through an injected
    /// provider with NO live server — the send/poll domain flow, deterministic.
    #[test]
    fn assistant_generates_via_injected_provider_without_a_server() {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let mut assistant = crate::AiAssistant::new();
        assert!(!assistant.has_custom_llm_provider());
        assistant.set_llm_provider(Arc::new(MockLlmProvider::new("mocked answer 42")));
        assert!(assistant.has_custom_llm_provider());

        assistant.send_message("hello".to_string(), "");

        let start = Instant::now();
        loop {
            match assistant.poll_response() {
                Some(AiResponse::Complete(text)) => {
                    assert!(
                        text.contains("mocked answer 42"),
                        "unexpected reply: {text:?}"
                    );
                    break;
                }
                Some(AiResponse::Error(e)) => panic!("unexpected error: {e}"),
                _ => {}
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "timed out waiting for the mocked response"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
