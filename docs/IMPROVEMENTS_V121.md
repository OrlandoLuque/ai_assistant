# V121 — Phase B.4 (part 3): wire StuckDetector into `multi_agent::PatternRunner`

**Date**: 2026-05-05
**Version**: 0.2.68
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § B.4
**Tasks**: #327 (B.4 — Stuck Detector + critique-based refinement)

## Why

V119 shipped the standalone `StuckDetector` + `CritiqueRefiner`
trait. V120 wired them into the autonomous-agent runner — the place
where stuck silently is most expensive because there is no peer or
per-step validator to flag the loop. V121 closes the second gap:
multi-agent orchestration.

The pathology in multi-agent is shaped differently from the single
autonomous loop. There is *no* shared step counter; each agent has
its own perspective. But the orchestrator sees the whole transcript
and is the one who decides whose turn is next. So the right place
to monitor for "the conversation has spiralled" is the orchestrator,
not the agents — exactly mirroring the V120 placement, one rung up.

Concretely: in round-robin or debate, the same agent (or pair) can
re-emit near-identical messages, hit the same dead-end action, or
keep retrying without progress. Today the runner doesn't notice. With
V121, the detector observes every transcript append, signals stuck
behaviour, and (when a refiner is installed) folds a `[CRITIC]:`
directive into the next agent's input.

## What changed

### `PatternRunner` — two opt-in setters

Both gated under `--features self-correction`:

```rust
pub fn with_stuck_detector(mut self, detector: StuckDetector) -> Self;
pub fn with_critique_refiner(
    mut self,
    refiner: Arc<dyn CritiqueRefiner + Send + Sync>,
) -> Self;
```

Plus the accessor:

```rust
pub fn last_stuck_signals(&self) -> &[StuckSignal];
```

Same shape as the autonomous-agent surface, intentionally — callers
that already know how to drive `StuckDetector` against the
autonomous runner should be able to drive it against `PatternRunner`
without learning a second API.

### `PatternRunner::run` — observe each transcript append

`run_round_robin`, `run_debate`, and `run_nested_chat` are the three
multi-round patterns where the same agent (or pair) can spiral.
Sequential is single-pass, swarm dispatches by task queue (no
inherent loop shape), broadcast fans out — none benefit. The
wire-in is surgical, not pervasive.

After each agent's reply is pushed onto the transcript:

| Field | Source |
|---|---|
| `step` | `self.transcript.len()` at observation time |
| `action` | `format!("agent:{}", agent_id)` |
| `output_text` | the message body just produced |
| `error_code` | `None` (multi-agent doesn't carry per-message error codes today) |
| `progressed` | `true` |

If `detector.check()` returns signals and a refiner is installed,
the refiner is asked for a directive; on `Some(directive)` the
runner prepends `[CRITIC]: <directive>\n\n` to the *next* agent's
input and resets the detector.

### Action key shape

In V120 (autonomous), the action key was `tool:<name>(args)` because
each iteration's "move" is a tool call. Here the "move" is *which
agent spoke*, so the action key is `agent:<agent_id>`. This makes
`ActionLoop` fire when the orchestrator keeps handing the floor to
the same agent in a tight cycle — which is the failure mode that
matters for round-robin / debate / nested-chat.

### Manual `Debug` impl

`PatternRunner` previously derived `Debug`. With the new
`Option<Arc<dyn CritiqueRefiner + Send + Sync>>` field, the derive
no longer compiles (the trait object isn't `Debug`). V121 replaces
the derive with a hand-written impl that preserves the active
fields verbatim and shows the cfg-gated detector/refiner as opaque
markers when `self-correction` is enabled.

### `Arc` import gating

The `Arc` import in `multi_agent.rs` was previously gated under
`autonomous` only (it was used by an autonomous-only structure).
V121 needs `Arc` under `self-correction` too. Solution: an extra
gated import that brings `Arc` into scope under `self-correction`
*only when `autonomous` is not also enabled*, so the two cfgs don't
clash:

```rust
#[cfg(all(feature = "self-correction", not(feature = "autonomous")))]
use std::sync::Arc;
```

### State reset on `run()`

`run()` resets `last_stuck_signals` and the detector itself at the
top of every call. This keeps `PatternRunner::run` re-entrant in a
sensible way — running the same orchestrator twice on different
inputs doesn't carry over stale observations.

## Tests

4 new tests in `multi_agent::tests` (cfg-gated `self-correction`):

| Test | Asserts |
|---|---|
| `test_pattern_runner_stuck_detector_permissive_no_signals` | Baseline: with permissive thresholds and a short run, no signals fire and `last_stuck_signals()` stays empty. |
| `test_pattern_runner_action_loop_fires_with_single_agent_aggressive` | Single-agent round-robin under aggressive thresholds → same `agent:<id>` every turn → `ActionLoop` fires and is visible. |
| `test_pattern_runner_critic_directive_injected` | Same loop with a `CallbackCritic` returning a fixed directive → at least one transcript message contains `[CRITIC]:`. |
| `test_pattern_runner_run_resets_detector` | Re-running the runner doesn't carry stale observations across tasks. |

All 91 `multi_agent::tests` pass under
`cargo test --features "multi-agent,self-correction" --lib multi_agent::tests`.

### Why permissive in the baseline

The default `StuckDetectorConfig` has `similarity_threshold = 0.85`.
Multi-agent template content (`"[Alice] (round 0) responds to: …"`,
`"[Bob] (round 0) responds to: [Alice] …"`) shares enough whitespace
tokens to clear that bar across a 4-message transcript, which would
fire `OutputRepetition` on perfectly normal traffic. The permissive
preset (`similarity_threshold = 0.95`, `repetition_threshold = 5`,
`action_loop_threshold = 5`) is the right default for orchestrators
where short formulaic templates are normal. The aggressive preset
remains the right pick for short-budget interactive loops where
every wasted step is expensive.

## Compatibility

- Both setters are cfg-gated behind `self-correction` and default
  to `None`. Runners built without them behave exactly as before —
  same builder, same `run()` signature, same `PatternResult`.
- `PatternRunner`'s `Debug` impl is now hand-written; field-for-
  field equivalent for the active fields, with the cfg-gated
  detector/refiner shown as opaque markers when `self-correction`
  is enabled.
- The `Arc` import gating addition is invisible to callers; the
  three feature combinations (`multi-agent` only,
  `multi-agent,self-correction`, `multi-agent,autonomous,
  self-correction`) all compile cleanly.

## What's next

- **V122 (B.5)**: parallel tool execution — when one LLM response
  carries N independent tool calls, execute them concurrently
  rather than sequentially; detect write-after-read dependencies
  to preserve ordering when needed. The autonomous runner will get
  the wire-in first.
- **V123 (B.6)**: adversary + egress inspectors and the
  `--no-egress` policy flag for closed-network operation.
- **Optional follow-up**: surface V117 error codes through the
  multi-agent message envelope so `RetryWithoutChange` can match
  on stable subsystem codes instead of the current
  `error_code = None`.
