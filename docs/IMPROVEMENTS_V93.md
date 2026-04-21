# IMPROVEMENTS V93 — In-crate StallHeuristic (0.2.25)

## Context

Eje 2/6 corrección del informe vs Claude Code leak. The library previously
suggested delegating stall detection to the caller; per the library framing
(`memory/feedback_library_framing.md`) we should resolve it **in-crate**
behind an opt-in feature flag instead. V93 ships that.

A **stall** is the agent state where budget is being spent without user-visible
progress. Two cheap signals cover most cases without an LLM call:

* **Repeated tool calls** — three or more identical `(tool_name, args)` hashes
  in the last 8 invocations.
* **User frustration** — the latest user message classifies as
  `EmotionCategory::Frustrated` via the existing keyword detector.

Either signal fires `StallDecision::Stalled(StallSignal)`. The agent's loop
can then set `LoopStatus::UserStalled`, bump `user_stall_events_total`, and
emit `agent.user_stall_detected` with a `signal` attribute.

## Scope

* **New module** `src/stall_detection.rs` behind `feature = "stall-detection"`
  (implies `autonomous`, `audio`, `analytics`; all three are zero-dep feature
  gates — no new dependencies).
* **Public API:**
  * `StallSignal` (enum: `Frustrated`, `RepeatedToolCall`).
  * `StallDecision` (enum: `Continue`, `Stalled(StallSignal)`).
  * `trait StallHeuristic` — four methods: `observe_tool_call`,
    `observe_user_message`, `check`, `reset`.
  * `KeywordStallDetector` — default impl using a `VecDeque` ring buffer
    (cap 8) of tool-call hashes + `KeywordEmotionDetector` for the latest
    user message.
  * `hash_tool_call(name, args_bytes)` — FNV-1a hash helper (matches the
    hashing style already used by `telemetry::should_sample`).
  * Constants `RING_BUFFER_SIZE = 8`, `REPEAT_THRESHOLD = 3`,
    `SPAN_NAME = "agent.user_stall_detected"`.
* **`LoopStatus::UserStalled`** variant added to `src/agentic_loop.rs`
  (always present so matches stay exhaustive regardless of feature set;
  only ever produced when the feature is enabled).
* **`TelemetryCollector::record_user_stall(&self, signal: &str)`** in
  `src/telemetry.rs`, bumping the new
  `AggregatedMetrics::user_stall_events_total: u64` counter. Signal is a
  `&str` so telemetry stays a thin collector, callable without the
  `stall-detection` feature compiled in.
* **`OtelTracer::start_user_stall_span(&self, signal: &str)`** in
  `src/opentelemetry_integration.rs`. Returns an `AiSpan` with operation
  `agent.user_stall_detected` and attribute `signal=Frustrated|RepeatedToolCall`.
* **14 unit tests** in `stall_detection::tests` plus 1 each in
  `telemetry::tests` and `opentelemetry_integration::tests`.

## Privacy

The heuristic stores **only derived signals** — a `u64` FNV-1a hash per tool
call and an `Option<EmotionCategory>` for the latest user message. Raw message
text is never persisted, matching the guarantees made by the `pii_tokenizer`
module. This invariant is enforced by the module API: `observe_user_message`
returns `()` and the stored state has no `String` field reachable from any
trait method.

## Design notes

### Why the variant is unconditional

`LoopStatus::UserStalled` is always present in the enum — not gated on
`feature = "stall-detection"`. Gating enum variants on features is possible
but forces every exhaustive match site to mirror the cfg, which ripples into
callers' code. The variant adds one discriminant; it is only ever produced by
the stall detector, which *is* feature-gated. Callers without the feature
enabled will never see the variant in practice.

### Why telemetry uses `&str` instead of `StallSignal`

`TelemetryCollector` compiles in every configuration; `stall_detection.rs`
does not. Accepting `&str` lets telemetry stay independent of the heuristic
while still getting a stable signal name (`"Frustrated"` /
`"RepeatedToolCall"`). `StallSignal::as_str()` and the `Display` impl return
exactly those strings, so callers with the feature enabled can pass
`signal.as_str()` directly.

### Signal precedence

When both signals fire in the same tick, `RepeatedToolCall` wins — it carries
the stronger invariant (budget is actively being burned this iteration)
whereas frustration is a lagging indicator (last message may be stale).

### Tie-breaking with 8-slot ring buffer

With `RING_BUFFER_SIZE = 8` and `REPEAT_THRESHOLD = 3`, the worst case is 2
competing tools each reaching 3 repeats. `most_repeated()` returns the first
hash it encounters at max count; that's deterministic given insertion order
and sufficient for the signal-vs-no-signal decision.

## Feature gating

```
stall-detection = ["autonomous", "audio", "analytics"]
```

All three dependencies are zero-dep feature gates in this crate. Enabling
`stall-detection` adds no new transitive dependencies.

## Version bump

`0.2.24 → 0.2.25` (patch-level; additive, no API breakage).

## Verification

```bash
cargo build --features stall-detection
cargo test --features stall-detection --lib stall_detection
# → 14 passed

cargo test --lib telemetry::tests::test_record_user_stall_increments_counter
# → 1 passed (works without stall-detection enabled)

cargo build   # default features, no stall-detection
# → clean
```

## Roadmap (task #155)

* LLM-assisted fallback heuristic — behind a separate feature flag that
  augments the keyword path with a cheap single-round LLM verdict when the
  keyword check is ambiguous.
* Linguistic coverage: current heuristic is English-dominant because
  `KeywordEmotionDetector` is. Multi-language lexicons are a task #155 item.
* Overheating heuristic: integrate tool-call rate + token-burn rate as a
  third signal alongside repetition and frustration.
