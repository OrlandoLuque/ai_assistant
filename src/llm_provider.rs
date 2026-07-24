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
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::config::{AiConfig, AiProvider};
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

    /// Short identifier of the backend this adapter targets (diagnostics). The
    /// default covers adapters that dispatch by config; dedicated adapters
    /// override it.
    fn backend_name(&self) -> &'static str {
        "config-dispatch"
    }
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

/// Dedicated adapter for the native Ollama API — phase 3, the first provider
/// pulled out of the `match &config.provider` dispatch into its own adapter.
///
/// It delegates to the `providers::generate_ollama_*` functions. Ollama is a
/// **local** provider, so it is unaffected by the cloud PII-masking layer in
/// `generate_response` — which is exactly why it is the first safe extraction.
/// (Cloud providers still route through [`ConfigLlmProvider`] so they keep that
/// masking; extracting them cleanly needs a PII-masking *decorator* around the
/// port — see `ai_assistant_plans/HEXAGONAL_PLAN.md`, F3.)
pub struct OllamaAdapter {
    config: AiConfig,
}

impl OllamaAdapter {
    /// Build an Ollama adapter bound to `config` (uses `config.ollama_url`).
    pub fn new(config: AiConfig) -> Self {
        Self { config }
    }
}

impl LlmProvider for OllamaAdapter {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::providers::generate_ollama_response(&self.config, conversation, system_prompt)
    }

    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        crate::providers::generate_ollama_streaming(&self.config, conversation, system_prompt, tx)
    }

    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        crate::providers::generate_ollama_streaming_cancellable(
            &self.config,
            conversation,
            system_prompt,
            tx,
            cancel_token,
        )
    }

    fn backend_name(&self) -> &'static str {
        "ollama"
    }
}

/// Select the [`LlmProvider`] adapter for `config`'s provider. Ollama gets its
/// dedicated [`OllamaAdapter`]; every other provider currently routes through
/// [`ConfigLlmProvider`] (preserving the cloud PII-masking layer). This is the
/// single dispatch point that will grow one adapter per provider as phase 3
/// proceeds.
pub fn provider_from_config(config: AiConfig) -> Box<dyn LlmProvider> {
    match config.provider {
        AiProvider::Ollama => Box::new(OllamaAdapter::new(config)),
        _ => Box::new(ConfigLlmProvider::new(config)),
    }
}

// ── Internal raw per-provider adapters (phase 3) ────────────────────────────
//
// These do the per-provider dispatch by delegating to the low-level
// `providers::*` / `cloud_providers::*` functions, WITHOUT PII masking — the
// mask/unmask stays in `generate_response*`, which wrap these. They exist so the
// three identical `match &config.provider` blocks in `providers.rs` collapse
// into the single `raw_provider_from_config` dispatch below. Each method mirrors
// exactly what the corresponding match arm did.

/// Kobold.cpp: no real streaming — the streaming methods produce the full
/// response as one `Complete` (matching the old match's fallback).
pub(crate) struct RawKoboldProvider {
    pub config: AiConfig,
}

impl LlmProvider for RawKoboldProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::providers::generate_kobold_response(&self.config, conversation, system_prompt)
    }
    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let response =
            crate::providers::generate_kobold_response(&self.config, conversation, system_prompt)?;
        let _ = tx.send(AiResponse::Complete(response));
        Ok(())
    }
    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        if cancel_token.is_cancelled() {
            let _ = tx.send(AiResponse::Cancelled(String::new()));
            return Ok(());
        }
        let response =
            crate::providers::generate_kobold_response(&self.config, conversation, system_prompt)?;
        let _ = tx.send(AiResponse::Complete(response));
        Ok(())
    }
    fn backend_name(&self) -> &'static str {
        "kobold"
    }
}

/// The whole OpenAI-compatible family (LM Studio, vLLM, llama.cpp, OpenAI,
/// Anthropic, Bedrock, Groq, ... — every arm that dispatched to
/// `generate_openai_*`).
pub(crate) struct RawOpenAiCompatProvider {
    pub config: AiConfig,
}

impl LlmProvider for RawOpenAiCompatProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::providers::generate_openai_response(&self.config, conversation, system_prompt)
    }
    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        crate::providers::generate_openai_streaming(&self.config, conversation, system_prompt, tx)
    }
    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        crate::providers::generate_openai_streaming_cancellable(
            &self.config,
            conversation,
            system_prompt,
            tx,
            cancel_token,
        )
    }
    fn backend_name(&self) -> &'static str {
        "openai-compatible"
    }
}

/// Azure OpenAI cloud.
pub(crate) struct RawAzureProvider {
    pub config: AiConfig,
}

impl LlmProvider for RawAzureProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::cloud_providers::generate_azure_openai_cloud(
            &self.config,
            conversation,
            system_prompt,
        )
    }
    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        crate::cloud_providers::generate_azure_openai_streaming(
            &self.config,
            conversation,
            system_prompt,
            tx,
        )
    }
    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        if cancel_token.is_cancelled() {
            let _ = tx.send(AiResponse::Cancelled(String::new()));
            return Ok(());
        }
        crate::cloud_providers::generate_azure_openai_streaming(
            &self.config,
            conversation,
            system_prompt,
            tx,
        )
    }
    fn backend_name(&self) -> &'static str {
        "azure-openai"
    }
}

/// Google Gemini cloud — its own API, no OpenAI-compatible streaming, so the
/// streaming methods produce the full response as one `Complete`.
pub(crate) struct RawGeminiProvider {
    pub config: AiConfig,
}

impl LlmProvider for RawGeminiProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        crate::cloud_providers::generate_gemini_cloud(&self.config, conversation, system_prompt)
    }
    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let response = crate::cloud_providers::generate_gemini_cloud(
            &self.config,
            conversation,
            system_prompt,
        )?;
        let _ = tx.send(AiResponse::Complete(response));
        Ok(())
    }
    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        if cancel_token.is_cancelled() {
            let _ = tx.send(AiResponse::Cancelled(String::new()));
            return Ok(());
        }
        let response = crate::cloud_providers::generate_gemini_cloud(
            &self.config,
            conversation,
            system_prompt,
        )?;
        let _ = tx.send(AiResponse::Complete(response));
        Ok(())
    }
    fn backend_name(&self) -> &'static str {
        "gemini"
    }
}

/// Internal dispatch factory (no PII) used inside `generate_response*` to
/// replace the three identical `match &config.provider` blocks. PII masking is
/// applied by the wrappers around this, so these adapters stay raw.
pub(crate) fn raw_provider_from_config(config: AiConfig) -> Box<dyn LlmProvider> {
    match config.provider {
        AiProvider::Ollama => Box::new(OllamaAdapter::new(config)),
        AiProvider::KoboldCpp => Box::new(RawKoboldProvider { config }),
        AiProvider::AzureOpenAI { .. } => Box::new(RawAzureProvider { config }),
        AiProvider::Gemini => Box::new(RawGeminiProvider { config }),
        // The whole OpenAI-compatible family (incl. OpenAI/Anthropic/Bedrock).
        _ => Box::new(RawOpenAiCompatProvider { config }),
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

    fn backend_name(&self) -> &'static str {
        "mock"
    }
}

/// Adapter that composes an ordered list of **labelled** providers into a
/// **fallback chain**: each `generate*` call tries them in order and returns the
/// first success; if all fail it returns the *first* (primary) provider's error.
/// This mirrors the historical `try_generate_with_fallback` / `generate_sync`
/// fallback in `assistant/` (config + `fallback_providers`, tried in order) so
/// the assistant's fallback behaviour can be expressed purely through the port.
///
/// Each provider carries a **display label**; the label of the one that answered
/// is written into a shared cell (see [`with_winner_sink`](Self::with_winner_sink)),
/// which lets the assistant keep its `fallback_last_provider` indicator working
/// through the port. An empty chain is a usage error: `generate*` return an error
/// rather than panicking.
pub struct FallbackLlmProvider {
    /// `(display label, provider)`, tried in order; index 0 is the primary.
    providers: Vec<(String, Arc<dyn LlmProvider>)>,
    /// Receives the winning provider's label (or `None` when the whole chain
    /// fails). Shared so callers can observe it (e.g. `fallback_last_provider`).
    winner: Arc<Mutex<Option<String>>>,
}

impl FallbackLlmProvider {
    /// Compose labelled `providers` into a fallback chain (index 0 is the
    /// primary). Each entry is `(display label, provider)`.
    pub fn new(providers: Vec<(String, Arc<dyn LlmProvider>)>) -> Self {
        Self {
            providers,
            winner: Arc::new(Mutex::new(None)),
        }
    }

    /// Route the winning provider's label into an existing shared cell instead of
    /// this adapter's own — e.g. the assistant's `fallback_last_provider`.
    pub fn with_winner_sink(mut self, sink: Arc<Mutex<Option<String>>>) -> Self {
        self.winner = sink;
        self
    }

    /// Display label of the provider that last answered successfully, or `None`
    /// if none has yet or the whole chain last failed.
    pub fn last_provider(&self) -> Option<String> {
        self.winner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn record(&self, label: Option<&str>) {
        *self.winner.lock().unwrap_or_else(|e| e.into_inner()) = label.map(str::to_string);
    }

    /// Error for a chain where every provider failed — carries the primary
    /// error, matching the legacy `try_generate_with_fallback` message.
    fn all_failed(first_err: Option<anyhow::Error>) -> anyhow::Error {
        match first_err {
            Some(e) => anyhow::anyhow!("All providers failed. Primary error: {e}"),
            None => anyhow::anyhow!("FallbackLlmProvider: no providers configured"),
        }
    }
}

impl LlmProvider for FallbackLlmProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        let mut first_err = None;
        for (label, p) in &self.providers {
            match p.generate(conversation, system_prompt) {
                Ok(text) => {
                    self.record(Some(label));
                    return Ok(text);
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        self.record(None);
        Err(Self::all_failed(first_err))
    }

    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let mut first_err = None;
        for (label, p) in &self.providers {
            match p.generate_streaming(conversation, system_prompt, tx) {
                Ok(()) => {
                    self.record(Some(label));
                    return Ok(());
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        self.record(None);
        Err(Self::all_failed(first_err))
    }

    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        let mut first_err = None;
        for (i, (label, p)) in self.providers.iter().enumerate() {
            // Match the legacy dispatcher: the primary always runs; cancellation
            // is only checked before each *fallback* attempt.
            if i > 0 && cancel_token.is_cancelled() {
                return Ok(());
            }
            match p.generate_streaming_cancellable(conversation, system_prompt, tx, cancel_token) {
                Ok(()) => {
                    self.record(Some(label));
                    return Ok(());
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        self.record(None);
        Err(Self::all_failed(first_err))
    }

    fn backend_name(&self) -> &'static str {
        "fallback"
    }
}

/// Decorator that applies the cloud **PII masking** boundary around any inner
/// [`LlmProvider`]: personal data in the conversation is tokenised (masked)
/// before it reaches the wrapped provider, and the masked placeholders are
/// restored (unmasked) in the response — per-chunk for streaming, through the
/// same relay the legacy `providers::generate_response*` used.
///
/// Masking is **unconditional**: wrap a provider in this only for backends that
/// must not see raw PII (cloud). Local providers are left undecorated so their
/// data never leaves the machine. Extracted from `generate_response*` so the
/// mask/unmask concern becomes a composable port decorator (hexagonal, Fase 3).
pub struct PiiMaskingProvider {
    inner: Box<dyn LlmProvider>,
}

impl PiiMaskingProvider {
    /// Wrap `inner` so its input is PII-masked and its output PII-unmasked.
    pub fn new(inner: Box<dyn LlmProvider>) -> Self {
        Self { inner }
    }

    /// Mask every message's content, accumulating the placeholder→value map.
    fn mask(conversation: &[ChatMessage]) -> (Vec<ChatMessage>, crate::pii_tokenizer::PiiTokenMap) {
        use crate::pii_tokenizer::{PiiTokenMap, PiiTokenizer};
        let mut pii_map = PiiTokenMap::new();
        let mut tokenizer = PiiTokenizer::with_default();
        let masked = conversation
            .iter()
            .map(|msg| {
                let (masked_content, map) = tokenizer.mask(&msg.content);
                pii_map.extend(map);
                ChatMessage {
                    role: msg.role.clone(),
                    content: masked_content,
                    ..msg.clone()
                }
            })
            .collect();
        (masked, pii_map)
    }
}

impl LlmProvider for PiiMaskingProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        let (masked, pii_map) = Self::mask(conversation);
        let result = self.inner.generate(&masked, system_prompt);
        if pii_map.is_empty() {
            result
        } else {
            result.map(|r| crate::pii_tokenizer::PiiTokenizer::unmask(&r, &pii_map))
        }
    }

    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let (masked, pii_map) = Self::mask(conversation);
        let (dispatch_tx, relay) = crate::providers::pii_unmask_relay(tx, &pii_map);
        let result = self
            .inner
            .generate_streaming(&masked, system_prompt, &dispatch_tx);
        drop(dispatch_tx);
        if let Some(handle) = relay {
            let _ = handle.join();
        }
        result
    }

    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        let (masked, pii_map) = Self::mask(conversation);
        let (dispatch_tx, relay) = crate::providers::pii_unmask_relay(tx, &pii_map);
        let result = self.inner.generate_streaming_cancellable(
            &masked,
            system_prompt,
            &dispatch_tx,
            cancel_token,
        );
        drop(dispatch_tx);
        if let Some(handle) = relay {
            let _ = handle.join();
        }
        result
    }

    fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    /// Test-only adapter that always fails — drives the fallback paths.
    struct FailingProvider;
    impl LlmProvider for FailingProvider {
        fn generate(&self, _c: &[ChatMessage], _s: &str) -> Result<String> {
            Err(anyhow::anyhow!("primary boom"))
        }
        fn generate_streaming(
            &self,
            _c: &[ChatMessage],
            _s: &str,
            _tx: &Sender<AiResponse>,
        ) -> Result<()> {
            Err(anyhow::anyhow!("primary boom"))
        }
        fn generate_streaming_cancellable(
            &self,
            _c: &[ChatMessage],
            _s: &str,
            _tx: &Sender<AiResponse>,
            _t: &CancellationToken,
        ) -> Result<()> {
            Err(anyhow::anyhow!("primary boom"))
        }
        fn backend_name(&self) -> &'static str {
            "failing"
        }
    }

    #[test]
    fn fallback_returns_first_success_and_records_label() {
        let chain = FallbackLlmProvider::new(vec![
            ("Primary".to_string(), Arc::new(FailingProvider)),
            (
                "Second".to_string(),
                Arc::new(MockLlmProvider::new("from second")),
            ),
            (
                "Third".to_string(),
                Arc::new(MockLlmProvider::new("from third")),
            ),
        ]);
        assert_eq!(chain.generate(&[], "sys").unwrap(), "from second");
        assert_eq!(chain.last_provider().as_deref(), Some("Second"));
    }

    #[test]
    fn fallback_primary_success_skips_the_rest() {
        let chain = FallbackLlmProvider::new(vec![
            (
                "Primary".to_string(),
                Arc::new(MockLlmProvider::new("primary")),
            ),
            ("Second".to_string(), Arc::new(FailingProvider)), // never reached
        ]);
        assert_eq!(chain.generate(&[], "sys").unwrap(), "primary");
        assert_eq!(chain.last_provider().as_deref(), Some("Primary"));
    }

    #[test]
    fn fallback_all_fail_returns_primary_error() {
        let chain = FallbackLlmProvider::new(vec![
            ("Primary".to_string(), Arc::new(FailingProvider)),
            ("Second".to_string(), Arc::new(FailingProvider)),
        ]);
        let err = chain.generate(&[], "sys").unwrap_err().to_string();
        assert!(err.contains("All providers failed"), "got: {err}");
        assert!(err.contains("primary boom"), "got: {err}");
        assert_eq!(chain.last_provider(), None);
    }

    #[test]
    fn fallback_streaming_falls_through_to_a_working_provider() {
        let (tx, rx) = channel();
        let chain = FallbackLlmProvider::new(vec![
            ("Primary".to_string(), Arc::new(FailingProvider)),
            (
                "Second".to_string(),
                Arc::new(MockLlmProvider::new("streamed")),
            ),
        ]);
        chain
            .generate_streaming(&[ChatMessage::user("hi")], "sys", &tx)
            .unwrap();
        assert!(matches!(rx.recv().unwrap(), AiResponse::Complete(t) if t == "streamed"));
        assert_eq!(chain.last_provider().as_deref(), Some("Second"));
    }

    #[test]
    fn fallback_winner_sink_is_shared_with_the_caller() {
        // The winning label lands in the shared cell the caller passed in — this
        // is how the assistant keeps its `fallback_last_provider` up to date.
        let sink = Arc::new(Mutex::new(None));
        let chain = FallbackLlmProvider::new(vec![
            ("Primary".to_string(), Arc::new(FailingProvider)),
            ("Second".to_string(), Arc::new(MockLlmProvider::new("ok"))),
        ])
        .with_winner_sink(sink.clone());
        chain.generate(&[], "sys").unwrap();
        assert_eq!(
            sink.lock().unwrap_or_else(|e| e.into_inner()).as_deref(),
            Some("Second")
        );
    }

    #[test]
    fn fallback_empty_chain_errs_without_panicking() {
        let chain = FallbackLlmProvider::new(vec![]);
        assert!(chain.generate(&[], "sys").is_err());
    }

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

    #[test]
    fn factory_dispatches_ollama_to_its_adapter() {
        // Ollama gets the dedicated adapter; other providers fall back to the
        // config-dispatch adapter (which preserves cloud PII masking).
        let mut cfg = AiConfig::default();
        cfg.provider = AiProvider::Ollama;
        assert_eq!(provider_from_config(cfg).backend_name(), "ollama");

        let mut cfg = AiConfig::default();
        cfg.provider = AiProvider::Anthropic;
        assert_eq!(provider_from_config(cfg).backend_name(), "config-dispatch");
    }

    /// The payoff of phase 2: a full `AiAssistant` generates through an injected
    /// provider with NO live server — the send/poll domain flow, deterministic.
    #[test]
    fn assistant_generates_via_injected_provider_without_a_server() {
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

    #[test]
    fn generate_sync_routes_through_injected_provider() {
        // F5.1: the non-streaming path also honours an injected provider.
        let mut a = crate::AiAssistant::new();
        a.set_llm_provider(Arc::new(MockLlmProvider::new("sync mock 7")));
        let out = a.generate_sync("hi".to_string(), "").unwrap();
        assert!(out.contains("sync mock 7"), "got: {out}");
    }

    #[test]
    fn cancellable_streaming_routes_through_injected_provider() {
        // F5.2: the cancellable streaming path also honours an injected provider.
        use std::time::{Duration, Instant};
        let mut assistant = crate::AiAssistant::new();
        assistant.set_llm_provider(Arc::new(MockLlmProvider::new("cancellable mock 99")));
        let _token = assistant.send_message_cancellable("hi".to_string(), "");

        let start = Instant::now();
        loop {
            match assistant.poll_response() {
                Some(AiResponse::Complete(text)) => {
                    assert!(text.contains("cancellable mock 99"), "unexpected: {text:?}");
                    break;
                }
                Some(AiResponse::Error(e)) => panic!("unexpected error: {e}"),
                _ => {}
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "timed out waiting for the mocked cancellable response"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Echoes the last message's content back (so the mask→unmask round-trip is
    /// observable end to end).
    struct EchoLastProvider;
    impl LlmProvider for EchoLastProvider {
        fn generate(&self, conversation: &[ChatMessage], _s: &str) -> Result<String> {
            Ok(conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default())
        }
        fn generate_streaming(
            &self,
            conversation: &[ChatMessage],
            _s: &str,
            tx: &Sender<AiResponse>,
        ) -> Result<()> {
            let last = conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let _ = tx.send(AiResponse::Complete(last));
            Ok(())
        }
        fn generate_streaming_cancellable(
            &self,
            conversation: &[ChatMessage],
            s: &str,
            tx: &Sender<AiResponse>,
            _c: &CancellationToken,
        ) -> Result<()> {
            self.generate_streaming(conversation, s, tx)
        }
    }

    /// Records the content it was actually handed — to prove the inner (cloud)
    /// provider never sees raw PII.
    struct CapturingProvider(Arc<Mutex<String>>);
    impl LlmProvider for CapturingProvider {
        fn generate(&self, conversation: &[ChatMessage], _s: &str) -> Result<String> {
            *self.0.lock().unwrap_or_else(|e| e.into_inner()) = conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok("ok".to_string())
        }
        fn generate_streaming(
            &self,
            conversation: &[ChatMessage],
            _s: &str,
            tx: &Sender<AiResponse>,
        ) -> Result<()> {
            *self.0.lock().unwrap_or_else(|e| e.into_inner()) = conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let _ = tx.send(AiResponse::Complete("ok".to_string()));
            Ok(())
        }
        fn generate_streaming_cancellable(
            &self,
            conversation: &[ChatMessage],
            s: &str,
            tx: &Sender<AiResponse>,
            _c: &CancellationToken,
        ) -> Result<()> {
            self.generate_streaming(conversation, s, tx)
        }
    }

    #[test]
    fn pii_decorator_masks_input_before_the_inner_provider_sees_it() {
        // Security: the wrapped (cloud) provider must never receive raw PII.
        let captured = Arc::new(Mutex::new(String::new()));
        let dec = PiiMaskingProvider::new(Box::new(CapturingProvider(captured.clone())));
        let conv = vec![ChatMessage::user(
            "please email me at alice@example.com about the invoice",
        )];
        let _ = dec.generate(&conv, "sys").unwrap();
        let seen = captured.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert!(
            !seen.contains("alice@example.com"),
            "inner provider saw raw PII: {seen}"
        );
    }

    #[test]
    fn pii_decorator_round_trips_pii_through_mask_and_unmask() {
        // The echo returns the MASKED content; the decorator unmasks it, so the
        // caller gets the original value back.
        let dec = PiiMaskingProvider::new(Box::new(EchoLastProvider));
        let conv = vec![ChatMessage::user(
            "please email me at alice@example.com about the invoice",
        )];
        let out = dec.generate(&conv, "sys").unwrap();
        assert!(
            out.contains("alice@example.com"),
            "unmask should restore the email: {out}"
        );
    }

    #[test]
    fn pii_decorator_unmasks_streaming_chunks() {
        let (tx, rx) = channel();
        let dec = PiiMaskingProvider::new(Box::new(EchoLastProvider));
        let conv = vec![ChatMessage::user("email alice@example.com now")];
        dec.generate_streaming(&conv, "sys", &tx).unwrap();
        let text = match rx.recv().unwrap() {
            AiResponse::Complete(t) => t,
            AiResponse::Chunk(t) => t,
            other => panic!("unexpected: {other:?}"),
        };
        assert!(
            text.contains("alice@example.com"),
            "streaming unmask should restore the email: {text}"
        );
    }
}
