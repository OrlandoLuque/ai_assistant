//! LLM-assisted second-opinion wrapper over any [`StallHeuristic`].
//!
//! The keyword heuristic in [`crate::stall_detection::KeywordStallDetector`]
//! is fast, free, and language-limited. This wrapper lets the caller
//! **augment** it with a single-round LLM verdict for ambiguous cases —
//! without pulling any LLM provider into the core crate.
//!
//! How it works:
//!
//! 1. The wrapper forwards every `observe_*` call to the inner heuristic.
//! 2. On each user message, if the cooldown has elapsed, the wrapper calls
//!    the caller-provided [`LlmVerdictFn`] and caches its verdict.
//! 3. `check()` returns the inner verdict when the inner fires; otherwise it
//!    returns whatever the cached LLM verdict says (falling back to
//!    `Continue` if the LLM abstained or was never consulted).
//!
//! The cooldown keeps the LLM path cheap: at most one call per configurable
//! `min_interval`. Tests use a zero cooldown to run deterministically.
//!
//! Feature-gated behind `stall-detection-llm`. Implies `stall-detection`.
//! No new dependencies — the callback is a plain `Fn` closure.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::stall_detection::{StallDecision, StallHeuristic, StallSignal};

/// Input handed to the LLM verdict callback.
#[derive(Debug, Clone)]
pub struct LlmVerdictInput {
    /// Recent tool-call names (most recent last). Provided to the LLM as a
    /// compact trace. The wrapper keeps at most [`TOOL_TRAIL_CAP`] entries.
    pub recent_tool_names: Vec<String>,
    /// The last user message. The wrapper holds it transiently for the
    /// callback only — it is cleared immediately after the callback returns.
    pub last_user_message: String,
}

/// Maximum number of recent tool names passed to the LLM callback.
pub const TOOL_TRAIL_CAP: usize = 16;

/// Default cooldown between LLM consultations.
pub const DEFAULT_LLM_COOLDOWN: Duration = Duration::from_secs(30);

/// Verdict returned by the LLM callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmVerdict {
    /// LLM judged the trace as stalled; which signal best describes it.
    Stalled(StallSignal),
    /// LLM judged the trace as productive — keep going.
    Continue,
    /// LLM could not decide (ambiguous input, rate-limited, etc.). The
    /// wrapper defers to the inner heuristic.
    Abstain,
}

/// Callback type — the caller's LLM-backed verdict function.
pub type LlmVerdictFn = Arc<dyn Fn(&LlmVerdictInput) -> LlmVerdict + Send + Sync>;

/// Wraps any [`StallHeuristic`] with an LLM-backed second opinion.
///
/// Call [`Self::new`] with the inner heuristic and a verdict callback.
/// Use [`Self::with_min_interval`] to tune the cooldown.
pub struct LlmAssistedStallDetector<H: StallHeuristic> {
    inner: H,
    verdict_fn: LlmVerdictFn,
    min_interval: Duration,
    last_llm_at: Option<Instant>,
    cached_verdict: Option<LlmVerdict>,
    recent_tool_names: Vec<String>,
}

impl<H: StallHeuristic> LlmAssistedStallDetector<H> {
    /// Build a new wrapper around `inner` with the given LLM verdict
    /// callback. Uses [`DEFAULT_LLM_COOLDOWN`] between consultations.
    pub fn new(inner: H, verdict_fn: LlmVerdictFn) -> Self {
        Self {
            inner,
            verdict_fn,
            min_interval: DEFAULT_LLM_COOLDOWN,
            last_llm_at: None,
            cached_verdict: None,
            recent_tool_names: Vec::new(),
        }
    }

    /// Override the cooldown between LLM consultations. Use
    /// `Duration::ZERO` in tests for deterministic behaviour.
    pub fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = interval;
        self
    }

    /// Expose the cached LLM verdict (mainly for introspection / tests).
    pub fn cached_verdict(&self) -> Option<LlmVerdict> {
        self.cached_verdict
    }

    /// Ref access to the wrapped heuristic.
    pub fn inner(&self) -> &H {
        &self.inner
    }

    /// Mutable access to the wrapped heuristic.
    pub fn inner_mut(&mut self) -> &mut H {
        &mut self.inner
    }

    fn cooldown_elapsed(&self, now: Instant) -> bool {
        match self.last_llm_at {
            None => true,
            Some(prev) => now.saturating_duration_since(prev) >= self.min_interval,
        }
    }
}

impl<H: StallHeuristic> StallHeuristic for LlmAssistedStallDetector<H> {
    fn observe_tool_call(&mut self, tool_name: &str, args_hash: u64) {
        self.inner.observe_tool_call(tool_name, args_hash);

        if self.recent_tool_names.len() >= TOOL_TRAIL_CAP {
            self.recent_tool_names.remove(0);
        }
        self.recent_tool_names.push(tool_name.to_string());
    }

    fn observe_user_message(&mut self, text: &str) {
        self.inner.observe_user_message(text);

        let now = Instant::now();
        if !self.cooldown_elapsed(now) {
            return;
        }

        let input = LlmVerdictInput {
            recent_tool_names: self.recent_tool_names.clone(),
            last_user_message: text.to_string(),
        };
        let verdict = (self.verdict_fn)(&input);
        self.last_llm_at = Some(now);
        self.cached_verdict = Some(verdict);
    }

    fn check(&self) -> StallDecision {
        let inner_verdict = self.inner.check();
        if matches!(inner_verdict, StallDecision::Stalled(_)) {
            return inner_verdict;
        }
        match self.cached_verdict {
            Some(LlmVerdict::Stalled(sig)) => StallDecision::Stalled(sig),
            _ => inner_verdict,
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.last_llm_at = None;
        self.cached_verdict = None;
        self.recent_tool_names.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stall_detection::{hash_tool_call, KeywordStallDetector, RateThresholds};
    use std::sync::Mutex;

    fn fixed_verdict(v: LlmVerdict) -> LlmVerdictFn {
        Arc::new(move |_input| v)
    }

    #[test]
    fn forwards_inner_decision_when_inner_stalls() {
        let inner = KeywordStallDetector::new();
        let mut wrapper = LlmAssistedStallDetector::new(inner, fixed_verdict(LlmVerdict::Continue))
            .with_min_interval(Duration::ZERO);

        // Three identical tool calls → inner fires RepeatedToolCall regardless
        // of what the LLM says.
        let h = hash_tool_call("x", b"y");
        for _ in 0..3 {
            wrapper.observe_tool_call("x", h);
        }
        assert_eq!(
            wrapper.check(),
            StallDecision::Stalled(StallSignal::RepeatedToolCall)
        );
    }

    #[test]
    fn llm_verdict_overrides_continue() {
        let inner = KeywordStallDetector::new();
        let mut wrapper = LlmAssistedStallDetector::new(
            inner,
            fixed_verdict(LlmVerdict::Stalled(StallSignal::Frustrated)),
        )
        .with_min_interval(Duration::ZERO);

        wrapper.observe_user_message("some neutral message");
        assert_eq!(
            wrapper.check(),
            StallDecision::Stalled(StallSignal::Frustrated)
        );
    }

    #[test]
    fn abstain_falls_back_to_inner() {
        let inner = KeywordStallDetector::new();
        let mut wrapper = LlmAssistedStallDetector::new(inner, fixed_verdict(LlmVerdict::Abstain))
            .with_min_interval(Duration::ZERO);

        wrapper.observe_user_message("neutral");
        assert_eq!(wrapper.check(), StallDecision::Continue);
    }

    #[test]
    fn cooldown_suppresses_repeated_llm_calls() {
        let call_count = Arc::new(Mutex::new(0u32));
        let cc = call_count.clone();
        let cb: LlmVerdictFn = Arc::new(move |_input| {
            *cc.lock().unwrap() += 1;
            LlmVerdict::Continue
        });

        let inner = KeywordStallDetector::new();
        let mut wrapper =
            LlmAssistedStallDetector::new(inner, cb).with_min_interval(Duration::from_secs(600));

        wrapper.observe_user_message("first");
        wrapper.observe_user_message("second");
        wrapper.observe_user_message("third");

        // Only the first message should trigger the LLM — 10-minute cooldown.
        assert_eq!(*call_count.lock().unwrap(), 1);
    }

    #[test]
    fn zero_cooldown_calls_llm_every_message() {
        let call_count = Arc::new(Mutex::new(0u32));
        let cc = call_count.clone();
        let cb: LlmVerdictFn = Arc::new(move |_input| {
            *cc.lock().unwrap() += 1;
            LlmVerdict::Continue
        });

        let inner = KeywordStallDetector::new();
        let mut wrapper =
            LlmAssistedStallDetector::new(inner, cb).with_min_interval(Duration::ZERO);

        for _ in 0..3 {
            wrapper.observe_user_message("msg");
        }
        assert_eq!(*call_count.lock().unwrap(), 3);
    }

    #[test]
    fn verdict_input_contains_recent_tool_names() {
        let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let cb: LlmVerdictFn = Arc::new(move |input| {
            *cap.lock().unwrap() = input.recent_tool_names.clone();
            LlmVerdict::Continue
        });

        let inner = KeywordStallDetector::new();
        let mut wrapper =
            LlmAssistedStallDetector::new(inner, cb).with_min_interval(Duration::ZERO);

        wrapper.observe_tool_call("alpha", 1);
        wrapper.observe_tool_call("beta", 2);
        wrapper.observe_tool_call("gamma", 3);
        wrapper.observe_user_message("check");

        assert_eq!(
            *captured.lock().unwrap(),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[test]
    fn tool_name_trail_is_capped() {
        let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let cb: LlmVerdictFn = Arc::new(move |input| {
            *cap.lock().unwrap() = input.recent_tool_names.clone();
            LlmVerdict::Continue
        });

        let inner = KeywordStallDetector::new();
        let mut wrapper =
            LlmAssistedStallDetector::new(inner, cb).with_min_interval(Duration::ZERO);

        // Push 2× the cap.
        for i in 0..(TOOL_TRAIL_CAP * 2) {
            wrapper.observe_tool_call(&format!("t{}", i), i as u64);
        }
        wrapper.observe_user_message("check");

        let trail = captured.lock().unwrap();
        assert_eq!(trail.len(), TOOL_TRAIL_CAP);
        // Oldest entries evicted — first element is t16 when cap=16.
        assert_eq!(trail[0], format!("t{}", TOOL_TRAIL_CAP));
    }

    #[test]
    fn reset_clears_wrapper_and_inner() {
        let inner = KeywordStallDetector::new()
            .with_rate_thresholds(RateThresholds::new(Duration::from_secs(60), 2));
        let mut wrapper = LlmAssistedStallDetector::new(
            inner,
            fixed_verdict(LlmVerdict::Stalled(StallSignal::Overheating)),
        )
        .with_min_interval(Duration::ZERO);

        wrapper.observe_tool_call("a", 1);
        wrapper.observe_user_message("go");
        wrapper.reset();

        assert_eq!(wrapper.cached_verdict(), None);
        assert_eq!(wrapper.check(), StallDecision::Continue);
    }

    #[test]
    fn inner_accessors_work() {
        let inner = KeywordStallDetector::new();
        let wrapper = LlmAssistedStallDetector::new(inner, fixed_verdict(LlmVerdict::Continue));
        assert_eq!(wrapper.inner().recent_hash_count(), 0);
    }

    #[test]
    fn default_cooldown_is_thirty_seconds() {
        assert_eq!(DEFAULT_LLM_COOLDOWN, Duration::from_secs(30));
    }

    #[test]
    fn tool_trail_cap_is_stable() {
        assert_eq!(TOOL_TRAIL_CAP, 16);
    }
}
