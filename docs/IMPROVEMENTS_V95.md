# IMPROVEMENTS V95 — StallHeuristic robustness + LLM-light backend (0.2.27)

## Context

Task #155 in the Claude Code addendum roadmap: harden the V93 StallHeuristic
along three axes identified in the V93 roadmap section:

1. **Overheating** — a rate-based third signal alongside `RepeatedToolCall`
   and `Frustrated`. Catches "many different tool calls fired fast" cases
   that `RepeatedToolCall` (identical-only) misses.
2. **Multi-language coverage** — the V93 detector is English-dominant via
   `KeywordEmotionDetector`. V95 adds compact Spanish / French / German
   frustration lexicons.
3. **LLM-assisted second opinion** — optional wrapper that augments the
   keyword path with a caller-provided LLM verdict on a cooldown. Keeps the
   cheap heuristic as the first line; reserves LLM cost for ambiguous cases.

All three extensions are **additive** and compose with the existing V93 API.
No breaking changes.

## Scope

### New signal + types in `src/stall_detection.rs`

* `StallSignal::Overheating` — new non-exhaustive variant. `as_str()` returns
  `"Overheating"`.
* `StallLanguage` enum: `English`, `Spanish`, `French`, `German` (with
  `as_str()` → ISO-639-1 codes `"en"`/`"es"`/`"fr"`/`"de"`).
* `StallKeywordLexicon` — static per-language frustration keyword lists plus
  `contains_frustration(text, lang) -> bool` helper (case-insensitive).
* `RateThresholds { window: Duration, max_calls: usize }` — with
  `new`, `default`, and constants `DEFAULT_RATE_WINDOW` (60s) and
  `DEFAULT_RATE_MAX_CALLS` (30).

### `KeywordStallDetector` extensions

* Fields: `language`, `rate_thresholds: Option<RateThresholds>`,
  `recent_timestamps: VecDeque<Instant>`, `last_was_frustrated: bool`.
* Builders: `with_language(StallLanguage)`, `with_rate_thresholds(RateThresholds)`.
* Introspection: `language()`, `rate_thresholds()`, `recent_timestamp_count()`.
* `observe_tool_call` now records an `Instant` when rate thresholds are on,
  and evicts entries outside the sliding window.
* `observe_user_message` uses `StallKeywordLexicon` for Spanish/French/German;
  English still goes through `KeywordEmotionDetector` (richer than a word list).
* `check()` evaluates signals with precedence:
  **RepeatedToolCall > Overheating > Frustrated**. Rationale: `RepeatedToolCall`
  has the strongest invariant (definite loop); `Overheating` is real-time;
  `Frustrated` is a lagging signal (last message may be stale).

### New module `src/stall_detection_llm.rs` (feature `stall-detection-llm`)

* `LlmVerdict` enum: `Stalled(StallSignal)`, `Continue`, `Abstain`.
* `LlmVerdictInput { recent_tool_names, last_user_message }` — what the LLM
  sees. The user message is held only for the duration of the callback; the
  wrapper clears references after the call returns.
* `LlmVerdictFn = Arc<dyn Fn(&LlmVerdictInput) -> LlmVerdict + Send + Sync>`.
* `LlmAssistedStallDetector<H: StallHeuristic>` — wraps any heuristic, adds
  an LLM callback on a configurable cooldown (`with_min_interval`, default
  30s via `DEFAULT_LLM_COOLDOWN`).
* `check()` semantics: inner verdict wins when it fires; otherwise cached
  LLM verdict applies; `Abstain` or missing verdict falls back to inner.
* Tool-name trail capped at `TOOL_TRAIL_CAP = 16` entries.

### Telemetry

No new counters. The existing `record_user_stall(signal)` already accepts any
signal name as `&str`, so `"Overheating"` flows through the existing V93
counter (`user_stall_events_total`). Callers that want per-signal breakouts
can use the `signal` property on the emitted event.

### OpenTelemetry

No new span. `start_user_stall_span("Overheating")` works via the existing
V93 helper for the same reason.

## Design notes

### Why precedence is RepeatedToolCall > Overheating > Frustrated

* `RepeatedToolCall` is a hard invariant: same `(tool, args)` three times →
  the agent is definitively looping. Never a false positive for productive
  work.
* `Overheating` is probabilistic: many distinct tool calls fast might mean
  "runaway loop" OR "genuinely busy batch work." We still want to interrupt
  because budget is burning, but only after ruling out the stronger signal.
* `Frustrated` is lagging: the last user message might be stale (user may
  have resolved their frustration by asking a new question). Real-time rate
  evidence should beat a stale sentiment signal.

### Why English still uses `KeywordEmotionDetector`, not the new lexicon

`KeywordEmotionDetector` is richer — it classifies a full emotion range
(Happy/Sad/Angry/Neutral/…) and callers use `last_emotion()` beyond just the
stall decision. Replacing it with a flat word list for English would
regress that capability. The new lexicon is used only for Spanish/French/
German, where the broader detector is not available.

### Why `observe_user_message` triggers the LLM call (not `check`)

`check(&self)` takes `&self` — it cannot mutate the cached verdict or
timestamps. Moving the LLM call to `observe_user_message(&mut self)` keeps
`check` a pure read, gives us a natural rate-limit anchor (one call per user
message at most), and preserves the `StallHeuristic` trait surface.

### Why the callback is `Arc<dyn Fn>` and not a generic parameter

Arc lets callers share one callback across multiple wrappers and clones; the
`Fn` bound avoids the extra verbosity of `FnMut` or `Box<dyn Fn>`. Keeping
the wrapper monomorphic-free over the callback keeps call sites compact.

### Why no new telemetry counter for overheating or LLM verdicts

The existing `record_user_stall(signal)` already accepts any signal string.
Adding per-signal counters would duplicate data: callers can already derive
per-signal counts by partitioning the emitted events on the `signal`
property. Keeping counters minimal honours the library's "no surprises"
principle.

## Feature gating

```
stall-detection-llm = ["stall-detection"]
```

Zero new dependencies. The callback is a plain `Fn` closure supplied by the
caller — whatever LLM provider they use stays on their side.

## AgenticLoop auto-integration

`AgenticLoop` now owns an optional `Box<dyn StallHeuristic>` gated on
`feature = "stall-detection"`. Callers attach a heuristic via a builder:

```rust
let agent = AgenticLoop::new(LoopConfig::default())
    .with_stall_heuristic(Box::new(KeywordStallDetector::new()));
```

During `process()`:

1. The user message is forwarded to `observe_user_message` before the
   first iteration — this is also the LLM cooldown anchor when the
   wrapper from `stall-detection-llm` is used.
2. After each iteration, new `ToolCall`s are hashed via `hash_tool_call`
   and fed through `observe_tool_call`. `check()` then runs; a `Stalled`
   verdict sets `state.status = LoopStatus::UserStalled` and breaks the
   main loop.

`stall_heuristic()` / `stall_heuristic_mut()` accessors let callers inspect
the heuristic after the loop returns (e.g., to read `last_emotion`). When
the feature is disabled, the field and accessors vanish — zero cost, no
API surface. Integration covered by two new unit tests in
`agentic_loop::tests`.

## Version bump

`0.2.26 → 0.2.27` (patch-level; additive, no API breakage).

`StallSignal` is now `#[non_exhaustive]`. This is the safer posture going
forward: future signals (e.g., token-rate overheating, budget exhaustion)
won't require a major bump. Pattern matches on `StallSignal` now need a `_`
arm — existing in-crate matches were updated.

## Verification

```bash
cargo test --features stall-detection --lib stall_detection::tests
# → 30 passed (14 V93 + 16 V95)

cargo test --features stall-detection-llm --lib stall_detection_llm
# → 11 passed

cargo build --features stall-detection-llm
# → clean

cargo build   # default features
# → clean
```

## Roadmap

The original V93 roadmap is now fully addressed. Further work is open-ended:

* **Token-burn signal** — similar to Overheating but gated on cumulative
  tokens per window rather than call count. Would fit cleanly as a fourth
  signal variant + a companion field on `RateThresholds`.
* **Context-sensitive lexicons** — per-domain (code review, research, etc.)
  frustration vocabulary. Would extend `StallKeywordLexicon` with a `Domain`
  axis orthogonal to `StallLanguage`.
* **LLM provider wiring** — optional feature that pre-builds a default
  `LlmVerdictFn` on top of `AiAssistant` with a prompt template. Currently
  the caller wires it themselves (see `examples/` — or inline).
