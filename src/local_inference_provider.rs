//! The bridge that was missing: in-process inference, reachable through the
//! normal generation path.
//!
//! # The gap this closes
//!
//! [`crate::local_inference`] could load a GGUF and generate from it — really,
//! with `llama-cpp-2` bindings since V112 — and **nothing routed an
//! [`crate::AiAssistant`] to it**. The loading worked and nobody could ask for
//! it: a caller who wanted a model running inside their own process had to
//! drive the [`Backend`] trait by hand, outside every decorator the crate
//! applies to generation (fallback, PII masking, guardrails).
//!
//! That is why it mattered. Anything built on this crate and run on a machine
//! it has never seen could only talk to a server the user had already
//! installed — which, for somebody who does not know what Ollama is, means
//! nothing at all.
//!
//! # How it connects
//!
//! [`LlmProvider`] is the port every generation path already goes through, and
//! [`crate::AiAssistant::set_llm_provider`] is the seam that swaps what sits
//! behind it. So this is an adapter and not a new branch in a `match`: inject it
//! and everything downstream — fallback chains, masking, the streaming surfaces
//! — works unchanged.
//!
//! A doctest compiles as a *separate crate*, so this is also the guard that the
//! types stay nameable and the trait callable from outside — the hole V322 found
//! in eleven other public types, and the one that unit tests cannot see because
//! they compile inside.
//!
//! ```
//! use ai_assistant::local_inference::{BackendKind, LocalInferenceConfig};
//! use ai_assistant::{ChatMessage, LlmProvider, LocalInferenceProvider};
//!
//! // The stub backend needs no native dependency, which is what lets this
//! // example run anywhere.
//! let config = LocalInferenceConfig::builder(BackendKind::Stub, "").build();
//! let provider = LocalInferenceProvider::load(&config).expect("stub always loads");
//! assert_eq!(provider.label(), "Local (stub)");
//!
//! // Reached through the port, not through an inherent method: this is the
//! // whole point of the adapter.
//! let reply = provider
//!     .generate(&[ChatMessage::user("hola")], "")
//!     .expect("the stub echoes the prompt back");
//! assert!(reply.contains("hola"));
//!
//! // And the claim above, checked rather than asserted in prose: an
//! // `AiAssistant` accepts it through the ordinary injection seam, so every
//! // decorator downstream applies without a new branch anywhere.
//! let mut assistant = ai_assistant::AiAssistant::new();
//! assistant.set_llm_provider(std::sync::Arc::new(provider));
//! assert!(assistant.has_custom_llm_provider());
//! ```

use std::sync::mpsc::Sender;
use std::sync::Mutex;

use anyhow::{anyhow, Result};

use crate::conversation_control::CancellationToken;
use crate::llm_provider::LlmProvider;
use crate::local_inference::{
    load, Backend, BackendError, BackendKind, GenParams, LocalInferenceConfig,
};
use crate::messages::{AiResponse, ChatMessage};

// =============================================================================
// How a conversation becomes a prompt
// =============================================================================

/// How to lay a conversation out for a model that takes one flat string.
///
/// # Why this is a choice and not a detail
///
/// [`Backend::generate`] takes a prompt and tokenises it **raw**. It does not
/// apply the chat template stored in the GGUF, so whoever builds the prompt
/// decides the format — and a model given the wrong one does not fail, it
/// answers worse. That is the quietest way to lose quality there is, so the
/// decision is named here rather than buried in a `format!`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PromptStyle {
    /// `<|im_start|>role\ncontent<|im_end|>`, the format Qwen and its
    /// derivatives expect — including PrismML's Bonsai, which is the small
    /// local model this crate is most often pointed at. The default for that
    /// reason, and for no deeper one.
    #[default]
    ChatMl,
    /// `Role: content`, for base models with no instruction format at all.
    /// Worse than ChatML on anything instruction-tuned; here so that a caller
    /// who knows their model is not forced into a template it never saw.
    Plain,
}

impl PromptStyle {
    /// Lay out a system prompt and a conversation as one string.
    pub fn render(&self, conversation: &[ChatMessage], system_prompt: &str) -> String {
        let mut out = String::new();
        match self {
            PromptStyle::ChatMl => {
                if !system_prompt.trim().is_empty() {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(system_prompt.trim());
                    out.push_str("<|im_end|>\n");
                }
                for m in conversation {
                    out.push_str("<|im_start|>");
                    out.push_str(&m.role);
                    out.push('\n');
                    out.push_str(&m.content);
                    out.push_str("<|im_end|>\n");
                }
                // The turn the model is being asked to write.
                out.push_str("<|im_start|>assistant\n");
            }
            PromptStyle::Plain => {
                if !system_prompt.trim().is_empty() {
                    out.push_str(system_prompt.trim());
                    out.push_str("\n\n");
                }
                for m in conversation {
                    let who = match m.role.as_str() {
                        "user" => "User",
                        "assistant" => "Assistant",
                        "system" => "System",
                        other => other,
                    };
                    out.push_str(who);
                    out.push_str(": ");
                    out.push_str(&m.content);
                    out.push('\n');
                }
                out.push_str("Assistant: ");
            }
        }
        out
    }
}

// =============================================================================
// The provider
// =============================================================================

/// A model running inside this process, presented as an [`LlmProvider`].
pub struct LocalInferenceProvider {
    /// `Backend` takes `&mut self` and is `Send` but not `Sync`; `LlmProvider`
    /// hands out `&self` and must be both. One mutex reconciles them, and it
    /// also states the real constraint: one model, one generation at a time.
    backend: Mutex<Box<dyn Backend>>,
    params: GenParams,
    style: PromptStyle,
    label: String,
}

impl std::fmt::Debug for LocalInferenceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalInferenceProvider")
            .field("label", &self.label)
            .field("style", &self.style)
            .finish_non_exhaustive()
    }
}

impl LocalInferenceProvider {
    /// Load a model and wrap it as a provider.
    ///
    /// Fails the way [`load`] fails: a missing file or a backend whose feature
    /// was not compiled in. Both are worth distinguishing to a caller, and both
    /// are better than a provider that exists and cannot answer.
    pub fn load(config: &LocalInferenceConfig) -> Result<Self, BackendError> {
        let backend = load(config)?;
        let label = label_for(backend.kind());
        Ok(Self {
            backend: Mutex::new(backend),
            params: GenParams::default(),
            style: PromptStyle::default(),
            label,
        })
    }

    /// Sampling settings for every generation.
    pub fn with_params(mut self, params: GenParams) -> Self {
        self.params = params;
        self
    }

    /// Override the prompt layout. See [`PromptStyle`] for why this matters.
    pub fn with_style(mut self, style: PromptStyle) -> Self {
        self.style = style;
        self
    }

    /// What to call this in a "which provider answered" indicator.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Generate, forwarding each chunk to `on_chunk`, and return the whole
    /// reply. The one place that touches the backend.
    fn run(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let prompt = self.style.render(conversation, system_prompt);
        let mut whole = String::new();

        let mut guard = self
            .backend
            .lock()
            // A poisoned mutex means a previous generation panicked inside the
            // native backend. Returning an error beats a deadlock and beats
            // pretending the model is fine.
            .map_err(|_| anyhow!("the local model is in a bad state after an earlier failure"))?;

        guard
            .generate(&prompt, &self.params, &mut |chunk| {
                whole.push_str(chunk);
                on_chunk(chunk);
            })
            .map_err(|e| anyhow!("{e}"))?;

        Ok(whole)
    }
}

fn label_for(kind: BackendKind) -> String {
    match kind {
        BackendKind::Stub => "Local (stub)".to_string(),
        BackendKind::Candle => "Local (Candle)".to_string(),
        BackendKind::LlamaCpp => "Local (llama.cpp)".to_string(),
    }
}

impl LlmProvider for LocalInferenceProvider {
    fn generate(&self, conversation: &[ChatMessage], system_prompt: &str) -> Result<String> {
        self.run(conversation, system_prompt, &mut |_| {})
    }

    fn generate_streaming(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
    ) -> Result<()> {
        let whole = self.run(conversation, system_prompt, &mut |chunk| {
            // A closed receiver is the caller having walked away, not an
            // error worth aborting the generation for: the backend has no way
            // to stop early anyway.
            let _ = tx.send(AiResponse::Chunk(chunk.to_string()));
        })?;
        let _ = tx.send(AiResponse::Complete(whole));
        Ok(())
    }

    fn generate_streaming_cancellable(
        &self,
        conversation: &[ChatMessage],
        system_prompt: &str,
        tx: &Sender<AiResponse>,
        cancel_token: &CancellationToken,
    ) -> Result<()> {
        // Accumulated separately from `run`'s own total, because once cancelled
        // we stop adding to it: `Cancelled` must carry what the caller actually
        // saw, not what the model went on to produce afterwards.
        let mut forwarded = String::new();
        let mut cancelled = false;
        let whole = self.run(conversation, system_prompt, &mut |chunk| {
            if cancelled || cancel_token.is_cancelled() {
                cancelled = true;
                return;
            }
            forwarded.push_str(chunk);
            let _ = tx.send(AiResponse::Chunk(chunk.to_string()));
        })?;

        // What cancelling can and cannot do here, said plainly: the `Backend`
        // trait has no way to stop a generation that has started, so this
        // stops *forwarding* and reports what was produced up to that point.
        // The model keeps running to its own stopping point. Claiming
        // otherwise would be the more comfortable lie.
        if cancelled {
            let _ = tx.send(AiResponse::Cancelled(forwarded));
        } else {
            let _ = tx.send(AiResponse::Complete(whole));
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn stub() -> LocalInferenceProvider {
        let config = LocalInferenceConfig::builder(BackendKind::Stub, "").build();
        LocalInferenceProvider::load(&config).expect("the stub needs no native dependency")
    }

    fn convo(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(role, content)| ChatMessage {
                role: (*role).to_string(),
                content: (*content).to_string(),
                ..ChatMessage::user("")
            })
            .collect()
    }

    #[test]
    fn a_local_model_answers_through_the_normal_path() {
        // The whole point: this is an `LlmProvider`, so everything the crate
        // wraps around generation applies to it unchanged.
        let provider = stub();
        let reply = LlmProvider::generate(&provider, &convo(&[("user", "hello")]), "be brief")
            .expect("stub generates");
        assert!(reply.starts_with("[stub] "), "{reply}");
    }

    #[test]
    fn the_prompt_layout_is_a_decision_and_the_stub_shows_it() {
        // `Backend::generate` tokenises raw: it does not apply the template in
        // the GGUF. So the layout is ours, and a model given the wrong one
        // does not fail -- it answers worse. The stub echoes the prompt, which
        // is what makes that decision observable instead of invisible.
        let provider = stub();
        let reply = LlmProvider::generate(
            &provider,
            &convo(&[("user", "hello"), ("assistant", "hi"), ("user", "again")]),
            "be brief",
        )
        .expect("generates");

        assert!(
            reply.contains("<|im_start|>system\nbe brief<|im_end|>"),
            "{reply}"
        );
        assert!(
            reply.contains("<|im_start|>user\nhello<|im_end|>"),
            "{reply}"
        );
        assert!(
            reply.contains("<|im_start|>assistant\nhi<|im_end|>"),
            "{reply}"
        );
        // And it ends by opening the turn the model is meant to write.
        assert!(reply.ends_with("<|im_start|>assistant\n"), "{reply}");
    }

    #[test]
    fn an_empty_system_prompt_does_not_become_an_empty_system_turn() {
        // An empty `<|im_start|>system<|im_end|>` is not nothing to a model:
        // it is an instruction to be nobody.
        let provider = stub();
        let reply = LlmProvider::generate(&provider, &convo(&[("user", "hello")]), "   ")
            .expect("generates");
        assert!(!reply.contains("system"), "{reply}");
    }

    #[test]
    fn the_plain_style_is_there_for_models_that_never_saw_chatml() {
        let provider = stub().with_style(PromptStyle::Plain);
        let reply = LlmProvider::generate(&provider, &convo(&[("user", "hello")]), "be brief")
            .expect("generates");
        assert!(reply.contains("User: hello"), "{reply}");
        assert!(!reply.contains("<|im_start|>"), "{reply}");
        assert!(reply.ends_with("Assistant: "), "{reply}");
    }

    #[test]
    fn streaming_sends_the_chunks_and_then_the_whole_thing() {
        let provider = stub();
        let (tx, rx) = std::sync::mpsc::channel();
        provider
            .generate_streaming(&convo(&[("user", "hello")]), "", &tx)
            .expect("streams");

        let got: Vec<AiResponse> = rx.try_iter().collect();
        assert!(matches!(got.first(), Some(AiResponse::Chunk(_))), "{got:?}");
        assert!(
            matches!(got.last(), Some(AiResponse::Complete(_))),
            "{got:?}"
        );
    }

    #[test]
    fn cancelling_reports_what_was_produced_rather_than_claiming_it_stopped() {
        // The `Backend` trait cannot stop a generation that has started. This
        // stops forwarding and says `Cancelled` with what it had -- which is
        // the true statement. Reporting `Complete` would be the comfortable
        // lie, and this codebase has paid for those.
        let provider = stub();
        let (tx, rx) = std::sync::mpsc::channel();
        let token = CancellationToken::new();
        token.cancel();

        provider
            .generate_streaming_cancellable(&convo(&[("user", "hello")]), "", &tx, &token)
            .expect("returns");

        let got: Vec<AiResponse> = rx.try_iter().collect();
        assert!(
            !got.iter().any(|r| matches!(r, AiResponse::Chunk(_))),
            "nothing should have been forwarded after cancelling: {got:?}"
        );

        // The payload is the separating assertion, and the reason this test
        // used to pass while the code was wrong: `Cancelled(_)` matched the
        // WHOLE reply just as happily as the empty string. The model ran to its
        // own stopping point -- the stub produces its one chunk regardless --
        // so the only honest payload is what actually reached the caller, which
        // here is nothing at all.
        let payload = got.iter().find_map(|r| match r {
            AiResponse::Cancelled(s) => Some(s.clone()),
            _ => None,
        });
        assert_eq!(
            payload.as_deref(),
            Some(""),
            "Cancelled must carry what was forwarded, not what was generated: {got:?}"
        );
    }

    #[test]
    fn a_backend_whose_feature_is_absent_says_which_one() {
        // Better than a provider that exists and cannot answer. Only asserted
        // when the feature really is absent, so the test is honest in both
        // build configurations.
        #[cfg(not(feature = "local-inference-llama-cpp"))]
        {
            let config = LocalInferenceConfig::builder(BackendKind::LlamaCpp, "").build();
            let err = LocalInferenceProvider::load(&config).expect_err("not compiled in");
            assert!(format!("{err}").contains("llama-cpp-2"), "{err}");
        }
    }

    #[test]
    fn the_label_names_the_engine_that_answered() {
        assert_eq!(stub().label(), "Local (stub)");
        assert_eq!(label_for(BackendKind::LlamaCpp), "Local (llama.cpp)");
        assert_eq!(label_for(BackendKind::Candle), "Local (Candle)");
    }
}
