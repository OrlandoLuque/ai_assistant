# V98 — Self-Correction Framework

**Version**: 0.2.29 → 0.2.30
**Feature flag**: `self-correction` (opt-in, not in `full`)
**Pattern**: Reflexion / Self-Refine (execute → validate → feedback → regenerate)

## Motivation

Prior to V98 the anti-hallucination pipeline (CoVe, faithfulness scoring,
quality gates) could *detect* issues but had no harness to *recover* from them.
A claim flagged as contradicted was surfaced to the caller; recovery was left
as an exercise. V98 introduces a generic validator-corrector harness so any
task with an `execute → validate` loop can request automatic retry with
feedback.

The trait is deliberately generic. V98 ships it with one concrete task
(`ClaimVerificationTask`). V99 adds code-compile and code-test tasks; V100 adds
tool-call, research-citation, agent-handoff, and safety-guardrail tasks.

## Architecture

### `CorrectableTask` trait — `src/self_correction/mod.rs`

```rust
pub trait CorrectableTask {
    type Output;
    type Issue: Issue;
    fn name(&self) -> &str;
    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError>;
    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue>;
    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String;
    fn quality_score(&self, output: &Self::Output, issues: &[Self::Issue]) -> f64;
}
```

The `Issue` trait carries a single meaningful bit: `is_retryable()`. Fatal
issues (RBAC denial, PII leak, jailbreak detection) return `false` and cause
the engine to stop immediately with `StopReason::FatalIssue(msg)`. All other
issues are regenerated against.

### Engine — `src/self_correction/engine.rs`

`SelfCorrectionEngine::run()` implements the loop:

1. Execute the task (with optional feedback from the prior iteration).
2. Validate the output → list of issues.
3. If any issue is non-retryable → stop with `FatalIssue`.
4. If issues is empty → stop with `AllPassed`.
5. If this is the 2nd+ attempt:
   - If `quality - prev_quality < -min_improvement` → `QualityRegression`.
   - If `|quality - prev_quality| < min_improvement` → `NoImprovement`.
6. Check 4-dim budget: attempts, tokens, cost, time.
7. Build feedback via `task.build_feedback()`, sanitize, pass to next iteration.

### 4-dimensional budget

| Dimension | Default | Purpose |
|-----------|---------|---------|
| `max_attempts` | 3 | Hard cap on retry count |
| `max_total_tokens` | 16,000 | Aggregate token budget across all attempts |
| `max_total_cost_usd` | $1.00 | Aggregate USD cost |
| `max_total_time_ms` | 60,000 | Wall-clock limit |

Plus `min_improvement = 0.05`: minimum quality-score delta between consecutive
attempts to qualify as progress.

### Stop reasons

```
AllPassed                      → success
CalibratedAbstention           → success (model explicitly said "don't know")
MaxAttempts / TokenBudget…     → budget exhaustion, return best-so-far
NoImprovement / QualityRegression → early-stop on non-progress
RegenerationFailed             → execute() returned Err
FatalIssue(msg)                → non-retryable issue detected
```

`StopReason::is_success()` returns true only for `AllPassed` and
`CalibratedAbstention`.

### Feedback sanitization — prompt-injection mitigation

When the prior LLM response is embedded into the next prompt, it's wrapped
in `<<<PRIOR_RESPONSE\n…\n>>>` with control-character stripping and
character-count truncation (default 4000). Without this, a model that emits
"Ignore previous instructions" would amplify the injection across attempts.

### JSONL audit ledger — `src/self_correction/ledger.rs`

Each run appends one JSON object to an append-only `.jsonl` file. The schema
(`LedgerEntry`) includes timestamp, task name, stop reason, per-attempt
records, and aggregates. `CorrectionLedger::read_all()` skips malformed
lines and reports the skip count.

### Concrete task: `ClaimVerificationTask` — `src/self_correction/claim.rs`

Wraps `ChainOfVerification` + `FaithfulnessScorer` + `QualityGateRunner` into
one retry loop.

- Initial attempt: returns the pre-computed response (no LLM call).
- Retry: calls a user-provided `RegenerateFn` closure with feedback.
- Validation produces `ClaimIssue::{Contradicted, Unverifiable, LowFaithfulness, GateFailed, CalibratedAbstention}`.
- Calibrated abstention ("I don't know", "No lo sé", etc.) is detected and
  marked non-retryable with `quality_score = 1.0`, so the engine stops and
  the caller can interpret it as honest success.

## Public API (re-exports in `lib.rs`)

```rust
#[cfg(feature = "self-correction")]
pub use self_correction::{
    CorrectableTask, Issue as CorrectionIssue,
    SelfCorrectionEngine, SelfCorrectionConfig, SelfCorrectionResult,
    StopReason as CorrectionStopReason,
    TaskOutcome as CorrectionTaskOutcome, TaskError as CorrectionTaskError,
    AttemptRecord as CorrectionAttemptRecord,
    CorrectionLedger, LedgerEntry as CorrectionLedgerEntry,
    LedgerError as CorrectionLedgerError,
    ClaimVerificationTask, ClaimIssue,
    sanitize_for_feedback as sanitize_correction_feedback,
};
```

## Tests

36 unit tests across 4 modules, all passing:

- `mod.rs`: 8 tests (config, sanitization, serialization)
- `engine.rs`: 13 tests (happy path, budget exhaustion per-dimension, fatal
  issue, regression, no-improvement, regeneration failure, feedback wiring,
  zero-attempts edge case)
- `ledger.rs`: 4 tests (append/read, count, malformed-line skipping,
  stop-reason string serialization)
- `claim.rs`: 11 tests (initial response returned without LLM call, retry
  triggers closure, abstention detection / non-retryable / quality=1,
  feedback contains prior issues, quality decreases with issue count, end-
  to-end with knowledge)

## Usage example

```rust
use ai_assistant::{
    ClaimVerificationTask, SelfCorrectionEngine, SelfCorrectionConfig,
};

// Minimal — user supplies a regenerate closure
let regen = Box::new(|prompt: &str, feedback: Option<&str>| {
    // call your LLM here; return (response, tokens_used, cost_usd)
    Some(("corrected response".to_string(), 150, 0.003))
});

let task = ClaimVerificationTask::new(
    "What is the speed of light?",
    "About 300,000 km/s in a vacuum.",  // pre-computed initial response
    regen,
)
.with_knowledge("The speed of light is 299,792,458 m/s in a vacuum.");

let engine = SelfCorrectionEngine::new(SelfCorrectionConfig::strict());
let result = engine.run(task, "What is the speed of light?");

if result.succeeded {
    println!("Final: {:?}", result.final_output);
} else {
    println!("Stopped: {:?}", result.stop_reason);
}
println!("Attempts: {}  Tokens: {}  Cost: ${:.4}",
    result.attempt_count(), result.total_tokens, result.total_cost_usd);
```

## What's NOT in V98 (scheduled for V99 / V100)

- Code tasks (V99): `CodeCompileTask`, `CodeTestTask`
- Tool/research/agent/safety tasks (V100)
- Auditor binaries: `ai_corrections` CLI + `ai_corrections_gui` egui (pattern:
  mirror `ai_breeder` / `ai_breeder_gui`)
- MCP tools: `self_correct_claim`, generic `self_correct_structured`
- HTTP server endpoint: `POST /api/v1/correct`
- CLI flag: `ai_cli verify --auto-correct --max-attempts N --max-cost $`
- `SelfCorrectionFileConfig` in `config_file.rs`
- `record_correction_attempt` in `telemetry.rs`

These extend the harness into the surface area. The V98 commit focuses on the
core library primitives so V99/V100 can build on a stable foundation.
