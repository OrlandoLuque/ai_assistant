# V120 — Phase B.4 (part 2): wire StuckDetector into `autonomous_agent`

**Date**: 2026-05-05
**Version**: 0.2.67
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § B.4
**Tasks**: #327 (B.4 — Stuck Detector + critique-based refinement)

## Why

V119 shipped `src/stuck_detector.rs` as a standalone, framework-agnostic
module — observations in, signals out, plus a `CritiqueRefiner` trait
and a `CallbackCritic` adapter. The wire-in to actual runners was
intentionally deferred so the detector could be reviewed and tuned in
isolation. V120 closes that gap for the open-ended runner — the
autonomous agent — where stuck detection matters most.

The autonomous runner is the *worst* place to be stuck silently: there
is no per-step validator (unlike `self_correction`, V98-V100), no peer
to flag the loop (unlike multi-agent), and the policy / sandbox can't
tell the difference between "still working hard" and "hammering a dead
end." A monitor that watches the loop's own observable state is the
right shape of feedback for this runner.

## What changed

### `AutonomousAgentBuilder` — two opt-in setters

Both gated under `--features self-correction`:

```rust
pub fn stuck_detector(mut self, detector: StuckDetector) -> Self;
pub fn critique_refiner(
    mut self,
    refiner: Arc<dyn CritiqueRefiner + Send + Sync>,
) -> Self;
```

Without these, the agent runs exactly as before (no observation, no
signals, no critic). With just the detector, signals fire and are
visible via the new `last_stuck_signals()` accessor — useful for
metrics or external escalation. With both, the agent self-redirects.

### `AutonomousAgent::run_iteration` — observation hook

At the end of every iteration, after tool calls are processed and the
task board is updated, the agent appends an `AgentObservation`:

| Field | Source |
|---|---|
| `step` | `self.iteration` |
| `action` | `canonical_action_key(&parsed)` |
| `output_text` | the assistant message produced this iteration |
| `error_code` | `Some("TOOL_FAILED")` when *all* tool calls in the iteration errored (no successes); `None` otherwise |
| `progressed` | `true` iff at least one tool call succeeded |

If `detector.check()` returns signals and a refiner is installed, the
refiner is asked for a directive; on `Some(directive)` the agent
pushes a `[CRITIC]: <directive>` system message and calls
`detector.reset()` to give the agent a clean slate after the redirect.

### `canonical_action_key` helper

```rust
fn canonical_action_key(parsed: &[ParsedToolCall]) -> String {
    if parsed.is_empty() {
        return "answer".to_string();
    }
    let first = &parsed[0];
    let mut sorted: Vec<(&String, &String)> = first.arguments.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let args = sorted.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(",");
    format!("tool:{}({})", first.name, args)
}
```

Stable ordering of args means `read_file(path=/a)` and
`read_file(path=/b)` get distinct keys — so `ActionLoop` won't
false-positive on a legitimate three-file read — but two identical
calls collapse to the same key, so a real loop fires.

### `error_code = "TOOL_FAILED"` (interim)

V119's `RetryWithoutChange` heuristic compares stable subsystem codes
from V117. The autonomous runner's tool registry currently surfaces
errors as opaque `String`s, so V120 uses a single coarse code,
`TOOL_FAILED`, when *all* tool calls in an iteration errored. That's
enough to trip the heuristic on classic "hammering the same broken
tool" loops; the next layer of fidelity (per-tool V117 codes) is a
follow-up that requires touching `unified_tools::ToolRegistry`'s error
type, out of scope for this slice.

### `last_stuck_signals()` accessor

```rust
pub fn last_stuck_signals(&self) -> &[StuckSignal];
```

Snapshots the signals from the most recent iteration. Cleared in two
cases:
- No signals fired this iteration → cleared.
- Signals fired AND a critic directive was folded in → cleared (the
  redirect is the response).

Stays populated when signals fire but no refiner is installed, or the
refiner returns `None` — useful for an external observer to decide
whether to abort, hand off to a human, or escalate to a stronger model.

### State reset on `run()`

V120 also resets `user_intent`, `last_stuck_signals`, and the detector
itself at the top of each `run()` call. This makes
`AutonomousAgent::run` re-entrant in a sensible way — running it
twice on different tasks doesn't carry over stale observations.

## Tests

4 new tests in `autonomous_loop::tests` (cfg-gated `self-correction`):

| Test | Asserts |
|---|---|
| `test_stuck_detector_observes_each_iteration` | Detector is fed during a normal multi-iteration run; below threshold no signals fire and `last_stuck_signals` stays empty. |
| `test_stuck_detector_fires_on_action_loop_no_refiner` | Same tool call repeated under aggressive thresholds → `ActionLoop` fires and is visible via `last_stuck_signals()`. |
| `test_critic_directive_injected_when_signals_fire` | Same loop with a `CallbackCritic` returning a fixed directive → agent's conversation gains a `[CRITIC]:` system message; signals cleared after the redirect. |
| `test_canonical_action_key_distinct_args` | `read_file(/a)` vs `read_file(/b)` get distinct keys; identical args collapse; empty parse → `"answer"`. |

All 30 `autonomous_loop` tests pass under
`cargo test --features self-correction,autonomous`.

## Compatibility

- Both builder methods are cfg-gated behind `self-correction` and
  default to `None`. Agents built without them behave exactly as
  before — same constructors, same `run()` signature, same
  `AgentResult`.
- The new struct fields are cfg-gated and initialized to `None` /
  empty in `build()`.
- The observation hook is cfg-gated; in non-`self-correction` builds
  the loop is unchanged.

## What's next

- **V121**: wire `StuckDetector` into multi-agent orchestration —
  cross-turn pathology in handoffs (one agent loops on the same
  hand-off message; coordinator never gets a fresh signal). Same
  setters, applied at the orchestrator level.
- **V122 (optional)**: surface V117 error codes through
  `ToolRegistry`'s error type so `RetryWithoutChange` can match on
  `PROVIDER_RATE_LIMITED` / `WORKFLOW_NODE_NOT_FOUND` / etc. instead
  of the coarse `TOOL_FAILED`.
- After V121: B.5 (parallel tool execution) and B.6 (adversary +
  egress inspectors + `--no-egress`).
