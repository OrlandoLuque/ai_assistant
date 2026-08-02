//! V98 — Self-correction framework.
//!
//! Generic validator-corrector harness: a `CorrectableTask` executes, its
//! output is validated, and on failure the engine builds a feedback prompt
//! and re-executes with the correction. The loop stops when all issues are
//! clear, when the budget is exhausted, or when quality regresses.
//!
//! The trait is deliberately generic: claims, code compilation, tool calls,
//! research citations — any domain with a notion of "execute → validate →
//! feedback" can implement it. V98 ships the trait, the engine, the ledger,
//! and `ClaimVerificationTask`. V99 adds code tasks; V100 adds tool / research
//! / agent-handoff / safety guardrail tasks.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "self-correction")]
//! # fn demo() {
//! use ai_assistant::self_correction::{
//!     SelfCorrectionConfig, SelfCorrectionEngine, CorrectableTask,
//! };
//! # }
//! ```

pub mod agent_handoff;
pub mod claim;
pub mod code;
pub mod engine;
pub mod ledger;
pub mod machine_fix;
pub mod research;
pub mod safety;
pub mod tool_call;

pub use agent_handoff::{
    AgentHandoffTask, HandoffIssue, HandoffRegenerateFn, HandoffValidateFn, HandoffValidationResult,
};
pub use claim::{ClaimIssue, ClaimVerificationTask};
pub use code::{
    cargo_compile_check, cargo_run_tests, parse_cargo_test_failures, CodeCompileTask,
    CodeCompileTaskCell, CodeRegenerateFn, CodeTestTask, CompileCheckResult, CompileFn,
    CompileIssue, TestFn, TestIssue, TestRunResult,
};
pub use engine::SelfCorrectionEngine;
pub use ledger::{CorrectionLedger, LedgerEntry, LedgerError};
pub use machine_fix::{apply_if_verified, apply_suggestions, Suggestion as MachineSuggestion};
pub use research::{
    CitationIssue, CitationRegenerateFn, CitationValidateFn, CitationValidationResult,
    ResearchCitationTask,
};
pub use safety::{
    SafetyCheckResult, SafetyGuardrailTask, SafetyIssue, SafetyIssueSpec, SafetyRegenerateFn,
    SafetyValidateFn,
};
pub use tool_call::{
    ToolCallIssue, ToolCallTask, ToolRegenerateFn, ToolValidateFn, ToolValidationResult,
};

use std::fmt;
use std::path::PathBuf;

// ── Trait & types ──────────────────────────────────────────────────────────

/// A single issue detected by a validator. Concrete tasks carry their own
/// issue enum; the trait only requires `Display`, `Debug`, and a retryability
/// flag.
pub trait Issue: fmt::Display + fmt::Debug + Send + Sync {
    /// Whether this issue can be addressed by regenerating with feedback.
    /// Fatal issues (e.g. RBAC denial, jailbreak detection, PII leak) return
    /// `false` — the engine stops immediately with
    /// [`StopReason::FatalIssue`].
    fn is_retryable(&self) -> bool {
        true
    }
}

/// Outcome of a single `execute` call: the produced output plus the resource
/// cost of producing it. The engine aggregates these for budget tracking.
#[derive(Debug, Clone)]
pub struct TaskOutcome<O> {
    /// The output that was generated.
    pub output: O,
    /// Tokens consumed by this attempt.
    pub tokens_used: usize,
    /// Estimated USD cost of this attempt.
    pub cost_usd: f64,
}

/// Execution error. Carries cost information so the engine can still track
/// budget consumption even when a call fails.
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Human-readable reason for the failure.
    pub reason: String,
    /// Tokens consumed before failure.
    pub tokens_used: usize,
    /// Estimated USD cost of the failed attempt.
    pub cost_usd: f64,
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for TaskError {}

impl TaskError {
    /// Build a `TaskError` with zero cost.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            tokens_used: 0,
            cost_usd: 0.0,
        }
    }
}

/// A task that can be corrected via the self-correction engine.
///
/// The contract is deliberately small:
///
/// 1. `execute` runs the task (possibly with corrective feedback from a
///    prior attempt) and returns either a `TaskOutcome<Output>` or a
///    `TaskError`.
/// 2. `validate` inspects the output and returns a list of issues; an empty
///    list means success.
/// 3. `build_feedback` produces the feedback string that will be passed to
///    the next `execute`. The engine supplies the full prior-attempt history
///    so the task can decide whether to accumulate or summarize.
/// 4. `quality_score` returns a scalar in `[0.0, 1.0]` used for regression
///    and no-improvement detection.
/// 5. `name` returns a static label used for telemetry and ledger entries.
pub trait CorrectableTask {
    /// Typed output of the task (a text response, an AST, a JSON value…).
    type Output;
    /// Concrete issue type for this task.
    type Issue: Issue;

    /// Static label — e.g. `"claim_verification"`, `"code_compile"`.
    fn name(&self) -> &str;

    /// Execute one attempt with optional corrective feedback.
    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError>;

    /// Validate a produced output. Empty `Vec` means no issues.
    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue>;

    /// Build a feedback string for the next attempt. `user_intent` is the
    /// original user prompt / task description; `prior_attempts` contains
    /// the accumulated records so far.
    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String;

    /// Quality score in `[0.0, 1.0]`. Higher is better. Used for regression
    /// and no-improvement detection.
    fn quality_score(&self, output: &Self::Output, issues: &[Self::Issue]) -> f64;
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the self-correction engine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SelfCorrectionConfig {
    /// Maximum number of attempts (including the initial one).
    pub max_attempts: usize,
    /// Maximum total tokens across all attempts.
    pub max_total_tokens: usize,
    /// Maximum total estimated USD cost across all attempts.
    pub max_total_cost_usd: f64,
    /// Maximum total wall-clock time across all attempts (ms).
    pub max_total_time_ms: u64,
    /// Minimum quality-score delta to consider an attempt "improved".
    /// Used for both no-improvement and regression detection.
    pub min_improvement: f64,
    /// Whether to sanitize `prior_response` segments in feedback prompts
    /// (truncate, escape control chars, add delimiters) to mitigate
    /// prompt-injection amplification across attempts.
    pub sanitize_feedback: bool,
    /// Maximum character length of sanitized prior response in the feedback
    /// prompt.
    pub sanitize_max_chars: usize,
    /// Optional path to a JSONL ledger; each run is appended as one entry.
    pub ledger_path: Option<PathBuf>,
}

impl Default for SelfCorrectionConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            max_total_tokens: 16_000,
            max_total_cost_usd: 1.0,
            max_total_time_ms: 60_000,
            min_improvement: 0.05,
            sanitize_feedback: true,
            sanitize_max_chars: 4_000,
            ledger_path: None,
        }
    }
}

impl SelfCorrectionConfig {
    /// Strict config: 2 attempts, tight budget. Good for production API paths
    /// where retries are expensive.
    pub fn strict() -> Self {
        Self {
            max_attempts: 2,
            max_total_tokens: 8_000,
            max_total_cost_usd: 0.25,
            max_total_time_ms: 20_000,
            min_improvement: 0.1,
            ..Self::default()
        }
    }

    /// Permissive config: 5 attempts, generous budget. For offline batch
    /// quality improvement where the goal is best-effort correctness.
    pub fn permissive() -> Self {
        Self {
            max_attempts: 5,
            max_total_tokens: 64_000,
            max_total_cost_usd: 5.0,
            max_total_time_ms: 300_000,
            min_improvement: 0.02,
            ..Self::default()
        }
    }
}

// ── Result types ───────────────────────────────────────────────────────────

/// Why the engine stopped iterating.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StopReason {
    /// All issues cleared — success.
    AllPassed,
    /// Reached `max_attempts` without clearing all issues.
    MaxAttempts,
    /// Aggregate token budget exhausted.
    TokenBudgetExhausted,
    /// Aggregate cost budget exhausted.
    CostBudgetExhausted,
    /// Aggregate time budget exhausted.
    TimeBudgetExhausted,
    /// Two consecutive attempts with delta below `min_improvement`.
    NoImprovement,
    /// Quality score decreased beyond `min_improvement` — regression.
    QualityRegression,
    /// `execute` returned an error with no recoverable strategy.
    RegenerationFailed,
    /// Task signalled a calibrated abstention (treated as success by design).
    CalibratedAbstention,
    /// A non-retryable issue was detected (e.g. RBAC denial, PII leak).
    FatalIssue(String),
}

impl StopReason {
    /// Whether this stop reason represents successful correction.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::AllPassed | Self::CalibratedAbstention)
    }
}

/// Serializable summary of a single attempt. Stored in both the live result
/// and the on-disk ledger.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptRecord {
    /// 1-indexed attempt number.
    pub attempt_num: usize,
    /// Human-readable issue messages (one per `Issue::fmt`).
    pub issues: Vec<String>,
    /// Quality score in `[0.0, 1.0]`.
    pub quality_score: f64,
    /// Tokens consumed by this attempt.
    pub tokens_used: usize,
    /// Estimated USD cost of this attempt.
    pub cost_usd: f64,
    /// Wall-clock duration of this attempt in ms.
    pub elapsed_ms: u64,
    /// Feedback that was passed INTO this attempt (None for the first one).
    pub feedback_given: Option<String>,
    /// Whether this attempt's output passed validation.
    pub succeeded: bool,
}

/// Result of running the self-correction engine.
#[derive(Debug)]
pub struct SelfCorrectionResult<O> {
    /// The best output produced, if any. For `StopReason::AllPassed` this is
    /// the final validated output; otherwise it's the last executed output
    /// (which may still have issues).
    pub final_output: Option<O>,
    /// Per-attempt history.
    pub attempts: Vec<AttemptRecord>,
    /// Whether the engine stopped with `AllPassed` or `CalibratedAbstention`.
    pub succeeded: bool,
    /// Why the engine stopped.
    pub stop_reason: StopReason,
    /// Aggregate tokens across all attempts.
    pub total_tokens: usize,
    /// Aggregate cost across all attempts.
    pub total_cost_usd: f64,
    /// Aggregate wall-clock time (ms) across all attempts.
    pub total_elapsed_ms: u64,
    /// Task name label (from `CorrectableTask::name`).
    pub task_name: String,
}

impl<O> SelfCorrectionResult<O> {
    /// Number of attempts executed.
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }

    /// Whether any attempt succeeded.
    pub fn has_success(&self) -> bool {
        self.attempts.iter().any(|a| a.succeeded)
    }
}

// ── Feedback sanitization ──────────────────────────────────────────────────

/// Sanitize a response string before embedding it in a feedback prompt.
///
/// Mitigates prompt-injection amplification: if a prior LLM response contains
/// text that looks like a prompt directive ("Ignore previous instructions…"),
/// feeding it back verbatim to the LLM would amplify the injection. This
/// function truncates, strips control characters, and wraps with explicit
/// delimiters so the next LLM call sees it as data, not instructions.
pub fn sanitize_for_feedback(text: &str, max_chars: usize) -> String {
    let stripped: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\t' || c == ' ' {
                c
            } else if c.is_control() {
                ' '
            } else {
                c
            }
        })
        .collect();

    let truncated = if stripped.chars().count() > max_chars {
        let mut out: String = stripped.chars().take(max_chars).collect();
        out.push_str("\n…[truncated]");
        out
    } else {
        stripped
    };

    format!("<<<PRIOR_RESPONSE\n{}\n>>>", truncated)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let c = SelfCorrectionConfig::default();
        assert_eq!(c.max_attempts, 3);
        assert!(c.max_total_tokens > 0);
        assert!(c.sanitize_feedback);
    }

    #[test]
    fn test_config_strict_vs_permissive() {
        let s = SelfCorrectionConfig::strict();
        let p = SelfCorrectionConfig::permissive();
        assert!(s.max_attempts <= p.max_attempts);
        assert!(s.max_total_cost_usd < p.max_total_cost_usd);
    }

    #[test]
    fn test_stop_reason_is_success() {
        assert!(StopReason::AllPassed.is_success());
        assert!(StopReason::CalibratedAbstention.is_success());
        assert!(!StopReason::MaxAttempts.is_success());
        assert!(!StopReason::QualityRegression.is_success());
        assert!(!StopReason::FatalIssue("rbac".into()).is_success());
    }

    #[test]
    fn test_sanitize_truncation() {
        let long = "x".repeat(10_000);
        let out = sanitize_for_feedback(&long, 100);
        assert!(out.contains("truncated"));
        assert!(out.contains("<<<PRIOR_RESPONSE"));
        assert!(out.contains(">>>"));
        assert!(out.chars().count() < 300);
    }

    #[test]
    fn test_sanitize_strips_control_chars() {
        let dirty = "hello\x00world\x07test";
        let out = sanitize_for_feedback(dirty, 1000);
        assert!(!out.contains('\x00'));
        assert!(!out.contains('\x07'));
        assert!(out.contains("hello"));
        assert!(out.contains("world"));
    }

    #[test]
    fn test_sanitize_preserves_newlines_and_tabs() {
        let text = "line1\nline2\tcol2";
        let out = sanitize_for_feedback(text, 1000);
        assert!(out.contains('\n'));
        assert!(out.contains('\t'));
    }

    #[test]
    fn test_task_error_display() {
        let e = TaskError::new("llm timed out");
        assert_eq!(format!("{}", e), "llm timed out");
        assert_eq!(e.tokens_used, 0);
    }

    #[test]
    fn test_attempt_record_serialization() {
        let r = AttemptRecord {
            attempt_num: 1,
            issues: vec!["bad".into()],
            quality_score: 0.5,
            tokens_used: 100,
            cost_usd: 0.01,
            elapsed_ms: 200,
            feedback_given: None,
            succeeded: false,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("attempt_num"));
        let back: AttemptRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attempt_num, 1);
    }
}
