//! V119 — Stuck Detector + critique-based refinement.
//!
//! A lightweight, framework-agnostic monitor for long-running agent loops.
//! Given a stream of [`AgentObservation`]s (one per agent step), the detector
//! flags when the loop appears to be spinning without progress and emits one
//! or more [`StuckSignal`]s describing why. An optional [`CritiqueRefiner`]
//! turns those signals into a fresh-angle directive that the caller can fold
//! into the agent's next prompt.
//!
//! ## Heuristics
//!
//! - **OutputRepetition** — the agent's textual output has been the same
//!   (or near-duplicate, by Jaccard similarity) for ≥ N steps in a row.
//! - **ActionLoop** — the same canonical action key (e.g. tool name + args
//!   hash) has been issued ≥ N times in the recent window.
//! - **RetryWithoutChange** — the agent retried the same step and got the
//!   same V117 error code ≥ N times. Pairs naturally with the
//!   `ErrorCode` taxonomy: a `PROVIDER_RATE_LIMITED` repeating means
//!   "still rate limited", while `WORKFLOW_NODE_NOT_FOUND` repeating means
//!   "the node really isn't there — stop retrying".
//! - **NoProgress** — no observation has set `progressed = true` for the
//!   last N steps. The caller is responsible for setting that flag based
//!   on its own notion of progress (file written, test passed, score
//!   improved, …).
//!
//! ## Critique pipeline
//!
//! When stuck signals fire, an agent runner can ask a [`CritiqueRefiner`]
//! to produce a free-text directive that nudges the next step out of the
//! loop. The default [`CallbackCritic`] adapter wraps any
//! `Fn(&str) -> Option<String>` callable — typically a thin LLM call.
//! The crate stays library-only: the caller plugs in the actual LLM
//! invocation; the prompt template, signal summarization, and history
//! formatting live here.
//!
//! ## Wiring
//!
//! Multi-agent and autonomous runners can call:
//!
//! ```ignore
//! detector.observe(obs);
//! let signals = detector.check();
//! if !signals.is_empty() {
//!     if let Some(directive) = refiner.refine(&signals, detector.history(), user_intent) {
//!         // fold `directive` into the next agent prompt
//!     }
//! }
//! ```
//!
//! ## Feature flag
//!
//! Gated under `self-correction` — the same flag that hosts
//! [`crate::self_correction`]. This module is *complementary* to that one:
//! `self_correction` runs a tight execute-validate-correct loop on a single
//! task; `stuck_detector` watches an open-ended agent run for higher-level
//! pathologies that can't be expressed as a single validator.

use std::collections::{HashSet, VecDeque};

// ── Observations ────────────────────────────────────────────────────────────

/// One observation from an agent loop. Callers append these as the agent
/// progresses; the detector retains a sliding window for analysis.
#[derive(Debug, Clone)]
pub struct AgentObservation {
    /// 1-indexed step counter (used in signal payloads + critic prompts).
    pub step: usize,
    /// Canonical action key — usually `format!("{tool}:{hash_of_args}")`.
    /// The detector treats two observations as "the same action" iff this
    /// string matches exactly.
    pub action: String,
    /// The agent's free-text output for this step (assistant message, tool
    /// rationale, plan step, …). Used by the OutputRepetition heuristic.
    pub output_text: String,
    /// Optional V117 error code if this step ended in an error
    /// (e.g. `"PROVIDER_RATE_LIMITED"`, `"WORKFLOW_NODE_NOT_FOUND"`).
    /// `None` if the step succeeded.
    pub error_code: Option<String>,
    /// Did the world advance? Set by the caller based on its own definition
    /// of progress (file written, score improved, sub-goal met, …).
    pub progressed: bool,
}

impl AgentObservation {
    /// Convenience constructor for a successful step that did some work.
    pub fn success(step: usize, action: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            step,
            action: action.into(),
            output_text: output.into(),
            error_code: None,
            progressed: true,
        }
    }

    /// Convenience constructor for an errored step.
    pub fn error(
        step: usize,
        action: impl Into<String>,
        output: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            step,
            action: action.into(),
            output_text: output.into(),
            error_code: Some(code.into()),
            progressed: false,
        }
    }
}

// ── Signals ─────────────────────────────────────────────────────────────────

/// Why the detector thinks the agent is stuck. Multiple signals can fire
/// simultaneously (e.g. ActionLoop + RetryWithoutChange when a step is
/// hammering the same failing tool).
#[derive(Debug, Clone, PartialEq)]
pub enum StuckSignal {
    /// Same (or near-duplicate) output text seen `count` times in window.
    /// `sample` is the most recent representative output.
    OutputRepetition { count: usize, sample: String },
    /// The same canonical action key was issued `count` times in window.
    ActionLoop { count: usize, action: String },
    /// The same V117 error code repeated `count` times — the agent is
    /// retrying without changing its approach.
    RetryWithoutChange { count: usize, code: String },
    /// No observation in the last `steps` had `progressed = true`.
    NoProgress { steps: usize },
}

impl StuckSignal {
    /// Single-line human summary, used by the default critic prompt.
    pub fn summary(&self) -> String {
        match self {
            Self::OutputRepetition { count, sample } => {
                let preview: String = sample.chars().take(80).collect();
                format!(
                    "output_repetition×{}: \"{}{}\"",
                    count,
                    preview,
                    if sample.chars().count() > 80 {
                        "…"
                    } else {
                        ""
                    }
                )
            }
            Self::ActionLoop { count, action } => {
                format!("action_loop×{}: {}", count, action)
            }
            Self::RetryWithoutChange { count, code } => {
                format!("retry_without_change×{}: {}", count, code)
            }
            Self::NoProgress { steps } => {
                format!("no_progress for {} steps", steps)
            }
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Tunable thresholds for the stuck detector.
#[derive(Debug, Clone)]
pub struct StuckDetectorConfig {
    /// Sliding-window size — the detector retains the most recent N
    /// observations.
    pub window: usize,
    /// Output-repetition trigger threshold (≥ this many duplicates fires).
    pub repetition_threshold: usize,
    /// Action-loop trigger threshold.
    pub action_loop_threshold: usize,
    /// Retry-without-change trigger threshold (consecutive matching error
    /// codes).
    pub retry_threshold: usize,
    /// Steps without progress before NoProgress fires.
    pub no_progress_threshold: usize,
    /// Jaccard similarity (`[0, 1]`) above which two outputs count as
    /// "the same" for OutputRepetition. Set to `1.0` for exact-match only.
    pub similarity_threshold: f64,
}

impl Default for StuckDetectorConfig {
    fn default() -> Self {
        Self {
            window: 8,
            repetition_threshold: 3,
            action_loop_threshold: 3,
            retry_threshold: 3,
            no_progress_threshold: 5,
            similarity_threshold: 0.85,
        }
    }
}

impl StuckDetectorConfig {
    /// Aggressive config — fires earlier. Good for short-budget interactive
    /// loops where a wasted step is expensive.
    pub fn aggressive() -> Self {
        Self {
            window: 6,
            repetition_threshold: 2,
            action_loop_threshold: 2,
            retry_threshold: 2,
            no_progress_threshold: 3,
            similarity_threshold: 0.7,
        }
    }

    /// Permissive config — gives the agent more rope before flagging.
    /// Good for long-running autonomous runs where occasional repetition
    /// is normal.
    pub fn permissive() -> Self {
        Self {
            window: 16,
            repetition_threshold: 5,
            action_loop_threshold: 5,
            retry_threshold: 5,
            no_progress_threshold: 10,
            similarity_threshold: 0.95,
        }
    }
}

// ── Detector ────────────────────────────────────────────────────────────────

/// Sliding-window stuck detector. Append observations with [`observe`], then
/// call [`check`] to get the current set of stuck signals (empty if
/// nothing's tripped).
///
/// [`observe`]: StuckDetector::observe
/// [`check`]: StuckDetector::check
#[derive(Debug)]
pub struct StuckDetector {
    config: StuckDetectorConfig,
    history: VecDeque<AgentObservation>,
}

impl StuckDetector {
    /// Build a new detector with the given config.
    pub fn new(config: StuckDetectorConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
        }
    }

    /// Append an observation. The history is trimmed to `config.window`.
    pub fn observe(&mut self, obs: AgentObservation) {
        self.history.push_back(obs);
        while self.history.len() > self.config.window {
            self.history.pop_front();
        }
    }

    /// Borrow the current observation history (oldest first).
    pub fn history(&self) -> Vec<&AgentObservation> {
        self.history.iter().collect()
    }

    /// Number of observations currently retained.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the detector has seen any observations.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Reset the history. Useful when the caller wants to give the agent
    /// a clean slate after applying a critic directive.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Run all heuristics against the current window. Returns one signal
    /// per pathology detected; an empty `Vec` means "nothing wrong".
    pub fn check(&self) -> Vec<StuckSignal> {
        let mut out = Vec::new();
        if let Some(s) = self.check_output_repetition() {
            out.push(s);
        }
        if let Some(s) = self.check_action_loop() {
            out.push(s);
        }
        if let Some(s) = self.check_retry_without_change() {
            out.push(s);
        }
        if let Some(s) = self.check_no_progress() {
            out.push(s);
        }
        out
    }

    fn check_output_repetition(&self) -> Option<StuckSignal> {
        if self.history.len() < self.config.repetition_threshold {
            return None;
        }
        // Check the most recent observation against the prior ones; count
        // how many in the window are similar to it. We use the *most recent*
        // as the anchor because it's the one we'd flag.
        let last = self.history.back()?;
        let mut count = 0;
        for obs in &self.history {
            if jaccard_similarity(&obs.output_text, &last.output_text)
                >= self.config.similarity_threshold
            {
                count += 1;
            }
        }
        if count >= self.config.repetition_threshold {
            Some(StuckSignal::OutputRepetition {
                count,
                sample: last.output_text.clone(),
            })
        } else {
            None
        }
    }

    fn check_action_loop(&self) -> Option<StuckSignal> {
        if self.history.len() < self.config.action_loop_threshold {
            return None;
        }
        let last_action = &self.history.back()?.action;
        let count = self
            .history
            .iter()
            .filter(|o| &o.action == last_action)
            .count();
        if count >= self.config.action_loop_threshold {
            Some(StuckSignal::ActionLoop {
                count,
                action: last_action.clone(),
            })
        } else {
            None
        }
    }

    fn check_retry_without_change(&self) -> Option<StuckSignal> {
        if self.history.len() < self.config.retry_threshold {
            return None;
        }
        // Count the longest tail run of matching error codes.
        let last_code = self.history.back()?.error_code.as_ref()?;
        let mut count = 0;
        for obs in self.history.iter().rev() {
            match &obs.error_code {
                Some(code) if code == last_code => count += 1,
                _ => break,
            }
        }
        if count >= self.config.retry_threshold {
            Some(StuckSignal::RetryWithoutChange {
                count,
                code: last_code.clone(),
            })
        } else {
            None
        }
    }

    fn check_no_progress(&self) -> Option<StuckSignal> {
        if self.history.len() < self.config.no_progress_threshold {
            return None;
        }
        // Look at the last N observations; if none have progressed = true,
        // fire NoProgress with that count.
        let n = self.config.no_progress_threshold;
        let recent = self.history.iter().rev().take(n);
        let any_progress = recent.clone().any(|o| o.progressed);
        if !any_progress {
            Some(StuckSignal::NoProgress { steps: n })
        } else {
            None
        }
    }
}

// ── Jaccard similarity (token-set) ──────────────────────────────────────────

/// Whitespace-tokenized Jaccard similarity in `[0, 1]`. Empty inputs return
/// `1.0` (both empty) or `0.0` (one empty) — the standard convention.
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let aset: HashSet<&str> = a.split_whitespace().collect();
    let bset: HashSet<&str> = b.split_whitespace().collect();
    if aset.is_empty() && bset.is_empty() {
        return 1.0;
    }
    if aset.is_empty() || bset.is_empty() {
        return 0.0;
    }
    let inter = aset.intersection(&bset).count() as f64;
    let union = aset.union(&bset).count() as f64;
    inter / union
}

// ── Critique refinement ─────────────────────────────────────────────────────

/// A refiner takes the current stuck signals + history + user intent and
/// returns a free-text directive the caller can fold into the next agent
/// prompt to break the loop. `None` means "no useful directive available";
/// the caller should fall back to whatever escalation it has (abort,
/// hand off to a human, escalate to a stronger model, …).
pub trait CritiqueRefiner {
    fn refine(
        &self,
        signals: &[StuckSignal],
        history: &[AgentObservation],
        user_intent: &str,
    ) -> Option<String>;
}

/// Default refiner: builds a critique prompt from the signals + a short
/// history slice + the user intent, then invokes a caller-supplied callable
/// for the LLM call. The callable's contract is the same as
/// `chain_of_verification`'s LLM verifier:
/// `Fn(&str) -> Option<String>` — `None` on failure / timeout.
pub struct CallbackCritic<F> {
    f: F,
    /// Maximum history entries to include in the prompt (keeps the prompt
    /// bounded). Most-recent entries are preferred.
    pub max_history: usize,
}

impl<F> CallbackCritic<F>
where
    F: Fn(&str) -> Option<String> + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self { f, max_history: 6 }
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Build the critic prompt. Public so callers can preview / replace it.
    pub fn build_prompt(
        &self,
        signals: &[StuckSignal],
        history: &[AgentObservation],
        user_intent: &str,
    ) -> String {
        let mut prompt = String::new();
        prompt.push_str(
            "You are a debugging coach for an autonomous agent. The agent appears stuck — \
             its recent loop has tripped one or more pathology detectors. Read the user's \
             intent, the detected pathologies, and the recent step history, then propose a \
             single concrete directive (1-3 sentences) telling the agent what to try next. \
             Focus on a *different angle* — do not rephrase the prior plan.\n\n",
        );
        prompt.push_str("User intent:\n");
        prompt.push_str(user_intent);
        prompt.push_str("\n\nDetected pathologies:\n");
        for s in signals {
            prompt.push_str("- ");
            prompt.push_str(&s.summary());
            prompt.push('\n');
        }
        prompt.push_str("\nRecent step history (most recent last):\n");
        let take = self.max_history.min(history.len());
        let start = history.len().saturating_sub(take);
        for obs in &history[start..] {
            let preview: String = obs.output_text.chars().take(120).collect();
            let ellipsis = if obs.output_text.chars().count() > 120 {
                "…"
            } else {
                ""
            };
            match &obs.error_code {
                Some(code) => prompt.push_str(&format!(
                    "  step {}: action={} → ERROR {} | {}{}\n",
                    obs.step, obs.action, code, preview, ellipsis
                )),
                None => prompt.push_str(&format!(
                    "  step {}: action={} → ok | {}{}\n",
                    obs.step, obs.action, preview, ellipsis
                )),
            }
        }
        prompt.push_str(
            "\nRespond with the directive only. Do not restate the pathologies or the history.",
        );
        prompt
    }
}

impl<F> CritiqueRefiner for CallbackCritic<F>
where
    F: Fn(&str) -> Option<String> + Send + Sync,
{
    fn refine(
        &self,
        signals: &[StuckSignal],
        history: &[AgentObservation],
        user_intent: &str,
    ) -> Option<String> {
        if signals.is_empty() {
            return None;
        }
        let prompt = self.build_prompt(signals, history, user_intent);
        (self.f)(&prompt).map(|s| s.trim().to_string())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_ok(step: usize, action: &str, output: &str, progressed: bool) -> AgentObservation {
        AgentObservation {
            step,
            action: action.into(),
            output_text: output.into(),
            error_code: None,
            progressed,
        }
    }

    fn obs_err(step: usize, action: &str, output: &str, code: &str) -> AgentObservation {
        AgentObservation::error(step, action, output, code)
    }

    #[test]
    fn jaccard_basics() {
        assert!((jaccard_similarity("", "") - 1.0).abs() < 1e-9);
        assert!((jaccard_similarity("a", "") - 0.0).abs() < 1e-9);
        assert!((jaccard_similarity("a b c", "a b c") - 1.0).abs() < 1e-9);
        // {a,b,c} vs {a,b,d}: |∩|=2, |∪|=4, jaccard = 0.5
        let s = jaccard_similarity("a b c", "a b d");
        assert!((s - 0.5).abs() < 1e-9, "got {}", s);
    }

    #[test]
    fn empty_detector_has_no_signals() {
        let d = StuckDetector::new(StuckDetectorConfig::default());
        assert!(d.is_empty());
        assert!(d.check().is_empty());
    }

    #[test]
    fn output_repetition_fires_above_threshold() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        for i in 1..=3 {
            d.observe(obs_ok(i, &format!("a{}", i), "the same response", true));
        }
        let signals = d.check();
        assert!(signals
            .iter()
            .any(|s| matches!(s, StuckSignal::OutputRepetition { count, .. } if *count >= 3)));
    }

    #[test]
    fn output_repetition_silent_when_outputs_differ() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        d.observe(obs_ok(1, "a", "alpha beta gamma", true));
        d.observe(obs_ok(2, "a", "delta epsilon zeta", true));
        d.observe(obs_ok(3, "a", "eta theta iota", true));
        let signals = d.check();
        assert!(!signals
            .iter()
            .any(|s| matches!(s, StuckSignal::OutputRepetition { .. })));
    }

    #[test]
    fn action_loop_fires_when_same_action_repeats() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        for i in 1..=3 {
            d.observe(obs_ok(
                i,
                "ls:/tmp",
                &format!("listing attempt {}", i),
                true,
            ));
        }
        let signals = d.check();
        assert!(signals.iter().any(|s| matches!(
            s,
            StuckSignal::ActionLoop { count, action }
                if *count == 3 && action == "ls:/tmp"
        )));
    }

    #[test]
    fn retry_without_change_fires_on_repeated_error_code() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        for i in 1..=3 {
            d.observe(obs_err(
                i,
                "call_provider",
                "rate limited again",
                "PROVIDER_RATE_LIMITED",
            ));
        }
        let signals = d.check();
        assert!(signals.iter().any(|s| matches!(
            s,
            StuckSignal::RetryWithoutChange { count, code }
                if *count == 3 && code == "PROVIDER_RATE_LIMITED"
        )));
    }

    #[test]
    fn retry_without_change_silent_when_codes_differ() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        d.observe(obs_err(1, "a", "msg", "PROVIDER_RATE_LIMITED"));
        d.observe(obs_err(2, "b", "msg", "NETWORK_TIMEOUT"));
        d.observe(obs_err(3, "c", "msg", "PROVIDER_RATE_LIMITED"));
        let signals = d.check();
        assert!(!signals
            .iter()
            .any(|s| matches!(s, StuckSignal::RetryWithoutChange { .. })));
    }

    #[test]
    fn no_progress_fires_after_threshold_steps_without_progress() {
        let mut cfg = StuckDetectorConfig::default();
        cfg.no_progress_threshold = 4;
        cfg.window = 8;
        // Ensure other heuristics don't fire — vary action + output, no errors.
        let mut d = StuckDetector::new(cfg);
        for i in 1..=4 {
            d.observe(obs_ok(
                i,
                &format!("a{}", i),
                &format!("unique output {}", i),
                false,
            ));
        }
        let signals = d.check();
        assert!(signals
            .iter()
            .any(|s| matches!(s, StuckSignal::NoProgress { steps } if *steps == 4)));
    }

    #[test]
    fn no_progress_silent_when_any_step_progressed() {
        let mut cfg = StuckDetectorConfig::default();
        cfg.no_progress_threshold = 3;
        let mut d = StuckDetector::new(cfg);
        d.observe(obs_ok(1, "a", "x", false));
        d.observe(obs_ok(2, "b", "y", true)); // progress!
        d.observe(obs_ok(3, "c", "z", false));
        let signals = d.check();
        assert!(!signals
            .iter()
            .any(|s| matches!(s, StuckSignal::NoProgress { .. })));
    }

    #[test]
    fn window_evicts_oldest() {
        let mut cfg = StuckDetectorConfig::default();
        cfg.window = 3;
        let mut d = StuckDetector::new(cfg);
        for i in 1..=10 {
            d.observe(obs_ok(i, "a", "x", true));
        }
        assert_eq!(d.len(), 3);
        // Oldest step in the window is 8, not 1.
        let h = d.history();
        assert_eq!(h.first().map(|o| o.step), Some(8));
    }

    #[test]
    fn signal_summary_has_useful_keywords() {
        let s = StuckSignal::OutputRepetition {
            count: 4,
            sample: "doing the thing again".into(),
        };
        let sum = s.summary();
        assert!(sum.contains("output_repetition"));
        assert!(sum.contains("4"));

        let s = StuckSignal::ActionLoop {
            count: 5,
            action: "tool:foo".into(),
        };
        assert!(s.summary().contains("action_loop"));

        let s = StuckSignal::RetryWithoutChange {
            count: 3,
            code: "WORKFLOW_NODE_NOT_FOUND".into(),
        };
        let sum = s.summary();
        assert!(sum.contains("retry_without_change"));
        assert!(sum.contains("WORKFLOW_NODE_NOT_FOUND"));

        let s = StuckSignal::NoProgress { steps: 7 };
        let sum = s.summary();
        assert!(sum.contains("no_progress"));
        assert!(sum.contains("7"));
    }

    #[test]
    fn callback_critic_invokes_callback_when_signals_present() {
        let critic = CallbackCritic::new(|prompt: &str| Some(format!("len={}", prompt.len())));
        let signals = vec![StuckSignal::ActionLoop {
            count: 3,
            action: "tool:foo".into(),
        }];
        let history = vec![obs_ok(1, "tool:foo", "trying foo", false)];
        let directive = critic.refine(&signals, &history, "fix the bug");
        assert!(directive.is_some());
        assert!(directive.unwrap().starts_with("len="));
    }

    #[test]
    fn callback_critic_returns_none_on_no_signals() {
        let critic = CallbackCritic::new(|_p: &str| panic!("should not call llm"));
        let history = vec![obs_ok(1, "a", "ok", true)];
        assert!(critic.refine(&[], &history, "intent").is_none());
    }

    #[test]
    fn callback_critic_returns_none_when_callback_returns_none() {
        let critic = CallbackCritic::new(|_p: &str| None);
        let signals = vec![StuckSignal::NoProgress { steps: 5 }];
        let history = vec![obs_ok(1, "a", "x", false)];
        assert!(critic.refine(&signals, &history, "intent").is_none());
    }

    #[test]
    fn callback_critic_prompt_includes_intent_signals_history() {
        let critic = CallbackCritic::new(|_p: &str| None);
        let signals = vec![StuckSignal::ActionLoop {
            count: 3,
            action: "shell:ls /tmp".into(),
        }];
        let history = vec![
            obs_ok(1, "shell:ls /tmp", "empty dir", false),
            obs_ok(2, "shell:ls /tmp", "empty dir", false),
        ];
        let prompt = critic.build_prompt(&signals, &history, "find the log file");
        assert!(prompt.contains("find the log file"));
        assert!(prompt.contains("action_loop"));
        assert!(prompt.contains("shell:ls /tmp"));
        // history rendering
        assert!(prompt.contains("step 1"));
        assert!(prompt.contains("step 2"));
    }

    #[test]
    fn callback_critic_max_history_caps_prompt_growth() {
        let critic = CallbackCritic::new(|_p: &str| None).with_max_history(2);
        let history: Vec<AgentObservation> = (1..=10)
            .map(|i| obs_ok(i, "a", &format!("step {}", i), false))
            .collect();
        let prompt =
            critic.build_prompt(&[StuckSignal::NoProgress { steps: 5 }], &history, "intent");
        // Only the last 2 history entries (steps 9, 10) should appear.
        assert!(!prompt.contains("step 1\n"));
        assert!(prompt.contains("step 9"));
        assert!(prompt.contains("step 10"));
    }

    #[test]
    fn presets_have_increasing_thresholds() {
        let agg = StuckDetectorConfig::aggressive();
        let def = StuckDetectorConfig::default();
        let perm = StuckDetectorConfig::permissive();
        assert!(agg.repetition_threshold <= def.repetition_threshold);
        assert!(def.repetition_threshold <= perm.repetition_threshold);
        assert!(agg.no_progress_threshold <= def.no_progress_threshold);
        assert!(def.no_progress_threshold <= perm.no_progress_threshold);
    }

    #[test]
    fn reset_clears_history() {
        let mut d = StuckDetector::new(StuckDetectorConfig::default());
        d.observe(obs_ok(1, "a", "x", true));
        assert_eq!(d.len(), 1);
        d.reset();
        assert!(d.is_empty());
        assert!(d.check().is_empty());
    }
}
