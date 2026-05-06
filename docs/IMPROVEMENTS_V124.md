# V124 — Phase C.3: OTel adaptive sampler + prompt redaction

**Date**: 2026-05-06
**Version**: 0.2.71
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.3
**Tasks**: #333 (V124 C.3 — adaptive sampler + privacy)

## Why

V118 wired `StructuredError` into OTel span attributes so dashboards
could segment failures by stable subsystem code. That made the
**signal** clean. V124 deals with the **volume** and **content** of
that telemetry:

- A uniform 100% sampling rate is pathological at production scale —
  errors are rare, success spans are millions, and storage costs are
  dominated by the boring case.
- The default span surface was carrying prompts, RAG queries, tool
  arguments, and CoVe claims into every collector by default.
  Callers ran `OtelConfig::default()` and silently leaked
  high-cardinality, privacy-sensitive content to whatever endpoint
  the operator configured.

V124 closes both gaps with a byte-for-byte additive change to
`OtelConfig`.

## What changed

### `SamplingPolicy` enum

```rust
pub enum SamplingPolicy {
    AlwaysOn,                    // default — preserves prior behaviour
    AlwaysOff,                   // drop every span
    Fixed(f64),                  // legacy single-rate
    Adaptive {
        success_rate: f64,       // recommended: 0.01
        error_rate: f64,         // recommended: 1.0
        p99_threshold_ms: u64,   // recommended: 1000
        p99_breach_rate: f64,    // recommended: 1.0
    },
}
```

`SamplingPolicy::adaptive_default()` returns the recommended preset.
The shape is the contract: errors and p99 outliers always go through,
success spans get a low cap.

### Running p99 estimation

`OtelTracer` now keeps a 256-entry `VecDeque<u64>` of recent finished
span durations. `exceeds_p99` answers `true` when *either* the static
threshold is exceeded *or* the running p99 of recent traffic (computed
on demand from a sorted copy) is exceeded. The static threshold gives
predictable behaviour at low traffic; the running p99 catches drift
when the system warms up. The history is updated *before* the sampling
decision, so the p99 estimate is not biased by what gets kept.

### `PrivacyConfig`

```rust
pub struct PrivacyConfig {
    pub redact_prompts: bool,                       // default true
    pub redacted_attribute_keys: Vec<String>,       // GenAI + internal keys
    pub max_prompt_chars: Option<usize>,            // default Some(8000) ≈ 2000 tokens
    pub allow_full_text: bool,                      // default false
}
```

Default redaction key set covers OTel GenAI conventions
(`gen_ai.prompt`, `gen_ai.completion`, `gen_ai.user.message`,
`gen_ai.system.message`) plus our internal surfaces (`rag.query`,
`rag.document`, `tool.input`, `tool.output`, `cove.claim`,
`cove.evidence`). Callers using the standard span builders get
redaction automatically.

Redaction replaces values with `"<redacted:N>"` where `N` is the
original char length. The marker preserves cardinality information
(dashboards can still see "this prompt was 1200 chars" without seeing
the content) while stripping the content itself.

Oversized spans — any redacted-key attribute or `error_message`
exceeding `max_prompt_chars` — are *dropped before sampling*.
Dropped-count is exposed via `OtelTracer::privacy_dropped_count()`
for dashboards.

### `commit_span` — single end-of-span pipeline

Previously `end_span`, `record_error`, and `record_structured_error`
each duplicated the buffer-eviction loop and only `end_span` consulted
sampling. V124 collapses them into a single private `commit_span`:

```
record duration → apply privacy (drop or redact) → sample → push
```

This is the right shape for adding more cross-cutting policies
(rate-limit, per-operation sampling, tail sampling) without
re-touching three sites.

### Compatibility

- `OtelConfig` is `#[non_exhaustive]`; the two new fields
  (`sampling_policy`, `privacy`) have `Default` impls so existing
  `OtelConfig::default()` callers compile unchanged.
- Default policy is `AlwaysOn`. Default privacy master switch is
  `true` with conservative max-chars. Callers who never used the
  redacted keys see zero behavioural difference.
- Callers who *did* use the redacted keys now see `<redacted:N>` in
  their span buffer. Opt out with `cfg.privacy.redact_prompts = false`
  or `cfg.privacy.allow_full_text = true`.
- The legacy `sampling_rate` field is preserved. When
  `sampling_policy = AlwaysOn` and `sampling_rate < 1.0`, the legacy
  rate gates *success* spans only — errors and p99 breaches always
  pass when the policy decision is positive. Callers using only the
  legacy field see the new wiring as an upgrade (errors are now
  always kept).

## Tests

11 new tests in `opentelemetry_integration::tests`:

| Test | Asserts |
|---|---|
| `test_v124_default_config_preserves_old_behaviour` | Default config: rate=1.0, AlwaysOn, redact=true, full_text=false, max_chars=Some(8000), key list contains `gen_ai.prompt`. |
| `test_v124_always_off_drops_every_span` | 5 spans → 0 in buffer. |
| `test_v124_adaptive_keeps_errors_drops_success_at_zero` | success_rate=0 + error_rate=1: 10 oks dropped, 3 errors kept. |
| `test_v124_adaptive_p99_breach_keeps_slow_success_spans` | Forged 100ms success span passes when p99_threshold_ms=5. |
| `test_v124_privacy_redacts_known_prompt_keys` | `gen_ai.prompt` becomes `<redacted:N>`, `non.sensitive` survives intact. |
| `test_v124_privacy_allow_full_text_disables_redaction` | `allow_full_text=true` bypasses redaction. |
| `test_v124_privacy_drops_oversized_prompt_span` | 1000-char prompt with `max_prompt_chars=64` drops the span and increments the counter. |
| `test_v124_privacy_allows_small_prompt_under_budget` | 40-char prompt with `max_prompt_chars=64` is kept and redacted. |
| `test_v124_fixed_policy_zero_drops_all` | `Fixed(0.0)` drops every span. |
| `test_v124_legacy_sampling_rate_still_works_on_success` | Legacy `rate=0.0` drops 5 oks but keeps 1 error (errors bypass legacy gate). |
| `test_v124_adaptive_default_preset` | Pinning the documented field values of the convenience preset. |

All 6,683 lib tests pass under
`cargo test --features "autonomous,self-correction,multi-agent" --lib`.

## What's next

- C.1 supply chain security (`cargo-audit` + `cargo-deny` + SBOM in
  CI) — V125.
- A future iteration may add per-operation policy overrides
  (e.g. always-keep `agent.user_stall_detected`, always-drop
  `cove.verify` when the result is unverifiable) using the same
  `commit_span` shape.
- Tail sampling (decide *after* the trace completes whether to keep
  it) is a natural next layer: the running p99 buffer already gives
  us the data we'd need.
