# V119 — Phase B.4 (part 1): Stuck Detector + critique-based refinement

**Date**: 2026-05-05
**Version**: 0.2.66
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § B.4
**Tasks**: #327 (Stuck Detector + critique-based refinement)

## Why

The crate already had a single-task self-correction loop
(`src/self_correction/`, V98-V100): execute → validate → feedback →
retry. Tight, focused, works great for "I have a JSON output and a
validator that says it's wrong."

What was missing: a **higher-level monitor** for *open-ended* agent
runs where there's no per-step validator. Long autonomous runs decide
their own steps; multi-agent loops manifest pathology across turns
rather than within one. The competitive gap audit (Hermes, Autocode,
OpenHands, Goose) flagged this as a Tier-1 hole in B.4: every
competitor ships a "stuck" heuristic + critique-based refinement.

V119 closes part 1 of that gap: a standalone, framework-agnostic
detector module with a default critic adapter. Wire-in to the
autonomous + multi-agent runners is a follow-up so this iteration
can be reviewed and tuned in isolation.

## What changed

### New module: `src/stuck_detector.rs`

Gated under `--features self-correction` (same flag as
`src/self_correction/`). The two modules are complementary:

| Module | Scope | Trigger |
|---|---|---|
| `self_correction` | One task with a validator | Validator returned issues |
| `stuck_detector` | Open-ended agent loop | Heuristic pathology detected |

### Public surface

```rust
pub use stuck_detector::{
    AgentObservation,
    CallbackCritic,
    CritiqueRefiner,
    StuckDetector,
    StuckDetectorConfig,
    StuckSignal,
};
```

Re-exported from the crate root when `self-correction` is enabled.

### `AgentObservation`

```rust
pub struct AgentObservation {
    pub step: usize,
    pub action: String,        // canonical key, e.g. "shell:ls /tmp"
    pub output_text: String,
    pub error_code: Option<String>,  // V117 taxonomy code
    pub progressed: bool,
}
```

Caller-supplied per step. Convenience constructors `::success(...)`
and `::error(...)`.

### `StuckSignal` — four heuristics

| Signal | Fires when |
|---|---|
| `OutputRepetition { count, sample }` | The agent's textual output has been the same (or near-duplicate, by Jaccard ≥ `similarity_threshold`) for ≥ `repetition_threshold` steps in the window. |
| `ActionLoop { count, action }` | The same canonical action key has been issued ≥ `action_loop_threshold` times in the window. |
| `RetryWithoutChange { count, code }` | The same V117 error code has repeated ≥ `retry_threshold` consecutive steps. |
| `NoProgress { steps }` | No observation in the last `no_progress_threshold` steps had `progressed = true`. |

Pathologies are reported as a `Vec<StuckSignal>`, so multiple can
fire simultaneously (e.g. `ActionLoop` + `RetryWithoutChange` when
the agent hammers the same failing tool).

### `RetryWithoutChange` × V117 synergy

This heuristic is the most novel piece. Without the V117 error
taxonomy, "did the agent get the same error twice?" had to compare
free-text strings — fragile against transient providers that include
a timestamp or a request ID in the message. With V117, the comparison
is against stable subsystem codes:

- `PROVIDER_RATE_LIMITED` repeating ⇒ still rate-limited (back off,
  don't switch tools).
- `WORKFLOW_NODE_NOT_FOUND` repeating ⇒ the node really isn't there
  (escalate / abort, don't keep retrying).
- `MCTS_NO_VALID_ACTIONS` repeating ⇒ search has dead-ended (widen
  state expansion or abort).

### `StuckDetectorConfig`

Three presets:

| Preset | Window | Repetition | Action loop | Retry | No progress | Similarity |
|---|---|---|---|---|---|---|
| `aggressive()` | 6 | 2 | 2 | 2 | 3 | 0.70 |
| `default()` | 8 | 3 | 3 | 3 | 5 | 0.85 |
| `permissive()` | 16 | 5 | 5 | 5 | 10 | 0.95 |

### `CritiqueRefiner` + `CallbackCritic`

```rust
pub trait CritiqueRefiner {
    fn refine(
        &self,
        signals: &[StuckSignal],
        history: &[AgentObservation],
        user_intent: &str,
    ) -> Option<String>;
}
```

Default impl wraps any `Fn(&str) -> Option<String> + Send + Sync`,
matching the existing `chain_of_verification::with_llm_verifier`
pattern. The crate stays library-only: caller plugs in the LLM call;
prompt template + signal summarization + history formatting live
here.

The default prompt frames it as a debugging coach asking for a
single fresh-angle directive (1-3 sentences), rather than a rephrase
of the existing plan. Caller can override via `build_prompt(...)`
(public).

## Tests

18 new tests in `stuck_detector::tests`:

| Test | Asserts |
|---|---|
| `jaccard_basics` | empty/equal/disjoint/partial set similarity |
| `empty_detector_has_no_signals` | no false positives at start |
| `output_repetition_fires_above_threshold` | OutputRepetition fires |
| `output_repetition_silent_when_outputs_differ` | silent on distinct outputs |
| `action_loop_fires_when_same_action_repeats` | ActionLoop fires |
| `retry_without_change_fires_on_repeated_error_code` | RetryWithoutChange fires |
| `retry_without_change_silent_when_codes_differ` | silent on alternating codes |
| `no_progress_fires_after_threshold_steps_without_progress` | NoProgress fires |
| `no_progress_silent_when_any_step_progressed` | reset by any progressed step |
| `window_evicts_oldest` | sliding window keeps last N |
| `signal_summary_has_useful_keywords` | each summary contains identifying tokens |
| `callback_critic_invokes_callback_when_signals_present` | callback called |
| `callback_critic_returns_none_on_no_signals` | early return, no LLM call |
| `callback_critic_returns_none_when_callback_returns_none` | error propagation |
| `callback_critic_prompt_includes_intent_signals_history` | prompt assembly |
| `callback_critic_max_history_caps_prompt_growth` | history size cap honored |
| `presets_have_increasing_thresholds` | aggressive ≤ default ≤ permissive |
| `reset_clears_history` | reset wipes window |

All 18 pass under `cargo test --features self-correction`.

## Wiring (deferred to V120/V121)

This iteration ships the standalone module + public re-exports.
The integration into the autonomous agent and multi-agent runners
is intentionally deferred so the detector can be reviewed and tuned
in isolation first. The wire-in is small at each runner:

```rust
// inside the per-step loop:
let obs = AgentObservation {
    step: n,
    action: canonical_action_key(),
    output_text: assistant_msg.clone(),
    error_code: last_error.as_ref().map(|e| <_ as ErrorCode>::code(e).to_string()),
    progressed: world_advanced(),
};
detector.observe(obs);

let signals = detector.check();
if !signals.is_empty() {
    if let Some(directive) = refiner.refine(&signals, &detector.history_owned(), user_intent) {
        next_prompt.push_str(&format!("\n[CRITIC]: {}\n", directive));
        detector.reset();   // give the agent a clean slate after the redirect
    } else {
        // fallback escalation: abort, hand off, or bump model tier
    }
}
```

No public API change in either runner is anticipated.

## What's next

- **V120**: wire `StuckDetector` + `CallbackCritic` into
  `autonomous_agent` (the open-ended runner where stuck-detection
  matters most).
- **V121**: wire into `multi_agent` orchestration (cross-turn
  pathology in handoffs).
- After both: B.5 (parallel tool execution) and B.6 (adversary +
  egress inspectors + `--no-egress`).
