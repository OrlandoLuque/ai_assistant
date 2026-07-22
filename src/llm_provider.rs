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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    /// A mock adapter: proves the port is usable without any live backend —
    /// the whole point of introducing it.
    struct MockLlmProvider {
        reply: String,
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

    #[test]
    fn port_is_object_safe_send_sync_and_mockable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn LlmProvider>>();

        let provider: Box<dyn LlmProvider> = Box::new(MockLlmProvider {
            reply: "hola".to_string(),
        });
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
}
