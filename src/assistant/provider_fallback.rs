//! Provider failover state for the assistant facade.
//!
//! The fourth cluster pulled out of the god-struct in Phase 5 of
//! `ai_assistant_plans/HEXAGONAL_PLAN.md`, after `KnowledgeStore` (V459),
//! `ProceduralMemory` (V460) and `ContextSizeCache` (V461). All three fields
//! were **private**, so this one costs no API break either:
//! `configure_fallback`, `enable_fallback`, `disable_fallback`,
//! `fallback_active` and `last_provider_used` keep their names and signatures
//! and delegate.
//!
//! # What was loose here was not an invariant, it was a question with three answers
//!
//! The facade held `fallback_providers`, `fallback_enabled` and
//! `fallback_last_provider` side by side, and "is failover on?" was re-derived
//! wherever it was needed:
//!
//! | site | what it asked |
//! |---|---|
//! | `models.rs`, `fallback_active()` | `enabled && !providers.is_empty()` |
//! | `messaging.rs`, default generation chain | `enabled` only |
//! | `messaging.rs`, vision streaming | `enabled` only |
//! | `messaging.rs`, vision sync | `!enabled \|\| providers.is_empty()` |
//!
//! None of the four was wrong today, because iterating an empty list adds no
//! links to the chain. They were four chances to disagree, and only the public
//! method stated the rule completely.
//!
//! [`active_providers`](ProviderFallback::active_providers) is the single
//! answer now: the providers to try, empty whenever failover is off. The
//! incomplete question cannot be asked any more, because the fields it needs
//! are not reachable.
//!
//! # And the indicator's lock was written out nine times
//!
//! Recording which provider served a response was spelled
//! `*self.fallback_last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
//! Some(name)` at **nine** call sites: three on the facade and six inside the
//! vision dispatcher, two of those clearing it with `None` instead. Nine copies
//! of a mutex lock and of the poison recovery that keeps one panicked
//! generation thread from killing the indicator for the rest of the session.
//!
//! Now the lock lives in this type's three accessors and nowhere else. The
//! vision dispatcher runs in a spawned thread, so it takes a **clone** of this
//! component: the provider list is a snapshot, exactly as it was when the
//! thread received a `Vec`, while the `Arc` keeps the indicator shared with the
//! facade.

use crate::config::AiProvider;
use std::sync::{Arc, Mutex};

/// The fallback chain: which providers to try after the primary fails, whether
/// to try them at all, and which one actually answered.
///
/// Fields are private, unlike `KnowledgeStore`'s `pub(crate)` ones: nothing
/// outside needs the raw pair, and leaving them reachable is how four spellings
/// of "is failover on" appeared in the first place.
#[derive(Clone)]
pub(crate) struct ProviderFallback {
    /// `(provider, model)` pairs to try in order when the primary fails.
    providers: Vec<(AiProvider, String)>,
    /// Whether automatic failover is enabled. Enabled with an empty list is
    /// **not** active: see [`is_active`](Self::is_active).
    enabled: bool,
    /// Display name of the provider that served the last response. Shared with
    /// background generation threads, which is why it is an `Arc<Mutex<_>>` and
    /// not a plain field: a clone of this component still reports into the same
    /// cell.
    last_provider: Arc<Mutex<Option<String>>>,
}

impl Default for ProviderFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderFallback {
    /// Disabled, with no providers configured and nothing recorded.
    pub(crate) fn new() -> Self {
        Self {
            providers: Vec::new(),
            enabled: false,
            last_provider: Arc::new(Mutex::new(None)),
        }
    }

    /// Replace the configured chain. Does not enable it.
    pub(crate) fn configure(&mut self, providers: Vec<(AiProvider, String)>) {
        self.providers = providers;
    }

    /// Enable automatic failover.
    pub(crate) fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable automatic failover, keeping the configured chain so that
    /// enabling again does not need it supplied a second time.
    pub(crate) fn disable(&mut self) {
        self.enabled = false;
    }

    /// Whether failover will actually happen: enabled **and** at least one
    /// provider configured.
    ///
    /// The one definition. The facade's `fallback_active()` is this.
    pub(crate) fn is_active(&self) -> bool {
        self.enabled && !self.providers.is_empty()
    }

    /// The providers to try, in order. Empty whenever failover is not active,
    /// so a caller that iterates it needs no enabled check of its own.
    pub(crate) fn active_providers(&self) -> &[(AiProvider, String)] {
        if self.is_active() {
            &self.providers
        } else {
            &[]
        }
    }

    /// The raw shared cell, for `FallbackLlmProvider::with_winner_sink`.
    ///
    /// This method exists only because that signature is **public API** and
    /// takes an `Arc<Mutex<Option<String>>>`; narrowing it would be a breaking
    /// change and therefore not this component's call. Everything inside this
    /// crate should use [`record_winner`](Self::record_winner) and
    /// [`last_winner`](Self::last_winner) instead, which are the only places
    /// that take the lock.
    pub(crate) fn sink(&self) -> Arc<Mutex<Option<String>>> {
        self.last_provider.clone()
    }

    /// Record the provider that served a response.
    ///
    /// Only the **vision** path calls this. Text generation goes through the
    /// port, and `FallbackLlmProvider` writes the winner into the cell it got
    /// from [`sink`](Self::sink) on its own; the vision API does not go through
    /// the port at all, which is a known gap in the hexagonal plan. Hence the
    /// `allow` with `vision` off: there is genuinely no caller, and that is the
    /// gap showing rather than dead code.
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) fn record_winner(&self, provider_display_name: &str) {
        *self.lock() = Some(provider_display_name.to_string());
    }

    /// Forget which provider answered, after an attempt where none did.
    ///
    /// Vision-only for the same reason as
    /// [`record_winner`](Self::record_winner).
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) fn clear_winner(&self) {
        *self.lock() = None;
    }

    /// The provider that served the last response, if one has yet.
    pub(crate) fn last_winner(&self) -> Option<String> {
        self.lock().clone()
    }

    /// The one place in the crate that locks the indicator, and therefore the
    /// one place that recovers from a poisoned mutex: a panic in one generation
    /// thread must not take the indicator down for the rest of the session.
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.last_provider.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two() -> Vec<(AiProvider, String)> {
        vec![
            (AiProvider::LMStudio, "model-a".to_string()),
            (AiProvider::Ollama, "model-b".to_string()),
        ]
    }

    #[test]
    fn a_new_chain_is_inactive_and_empty() {
        let f = ProviderFallback::new();
        assert!(!f.is_active());
        assert!(f.active_providers().is_empty());
        assert_eq!(f.last_winner(), None);
    }

    #[test]
    fn default_matches_new() {
        assert!(!ProviderFallback::default().is_active());
    }

    #[test]
    fn configured_but_not_enabled_is_not_active() {
        let mut f = ProviderFallback::new();
        f.configure(two());
        assert!(!f.is_active());
        assert!(
            f.active_providers().is_empty(),
            "a disabled chain must offer nothing to iterate"
        );
    }

    #[test]
    fn enabled_with_nothing_configured_is_not_active() {
        // The half of the rule that three of the four old call sites skipped.
        let mut f = ProviderFallback::new();
        f.enable();
        assert!(!f.is_active());
        assert!(f.active_providers().is_empty());
    }

    #[test]
    fn enabled_and_configured_offers_the_chain_in_order() {
        let mut f = ProviderFallback::new();
        f.configure(two());
        f.enable();
        assert!(f.is_active());
        let active = f.active_providers();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].1, "model-a");
        assert_eq!(active[1].1, "model-b");
    }

    #[test]
    fn disabling_keeps_the_configured_chain_for_next_time() {
        let mut f = ProviderFallback::new();
        f.configure(two());
        f.enable();
        f.disable();
        assert!(!f.is_active());
        f.enable();
        assert_eq!(
            f.active_providers().len(),
            2,
            "disable must not throw away what the caller configured"
        );
    }

    #[test]
    fn configuring_again_replaces_the_chain() {
        let mut f = ProviderFallback::new();
        f.configure(two());
        f.enable();
        f.configure(vec![(AiProvider::OpenAI, "gpt-4o".to_string())]);
        assert_eq!(f.active_providers().len(), 1);
        assert_eq!(f.active_providers()[0].1, "gpt-4o");
    }

    #[test]
    fn the_sink_is_the_same_cell_the_indicator_reads() {
        // If `sink()` ever handed out a fresh Arc, the indicator would report
        // None for ever and nothing else would look wrong.
        let f = ProviderFallback::new();
        let sink = f.sink();
        *sink.lock().unwrap() = Some("TestProvider".to_string());
        assert_eq!(f.last_winner(), Some("TestProvider".to_string()));
    }

    #[test]
    fn a_clone_reports_into_the_same_indicator() {
        // What the vision dispatcher relies on: it gets a clone in a spawned
        // thread, and the facade must still see which provider answered.
        let mut f = ProviderFallback::new();
        f.configure(two());
        f.enable();
        let snapshot = f.clone();
        snapshot.record_winner("Ollama");
        assert_eq!(f.last_winner(), Some("Ollama".to_string()));
        assert_eq!(
            snapshot.active_providers().len(),
            2,
            "the clone carries the chain it was given"
        );
    }

    #[test]
    fn recording_and_clearing_the_winner() {
        let f = ProviderFallback::new();
        f.record_winner("Ollama");
        assert_eq!(f.last_winner(), Some("Ollama".to_string()));
        f.record_winner("LM Studio");
        assert_eq!(f.last_winner(), Some("LM Studio".to_string()));
        f.clear_winner();
        assert_eq!(f.last_winner(), None);
    }

    #[test]
    fn a_poisoned_indicator_still_answers() {
        // A panic in one generation thread must not take the indicator down
        // for the rest of the session. This is the whole reason the accessors
        // go through `unwrap_or_else(|e| e.into_inner())`.
        let f = ProviderFallback::new();
        f.record_winner("Ollama");
        let sink = f.sink();
        let _ = std::thread::spawn(move || {
            let _guard = sink.lock().unwrap();
            panic!("poison the mutex on purpose");
        })
        .join();

        assert!(f.sink().is_poisoned(), "the test must really poison it");
        assert_eq!(f.last_winner(), Some("Ollama".to_string()));
        f.record_winner("Groq");
        assert_eq!(f.last_winner(), Some("Groq".to_string()));
    }
}
