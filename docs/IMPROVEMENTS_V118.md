# V118 — Phase C.2: wire StructuredError into OTel spans

**Date**: 2026-05-05
**Version**: 0.2.65
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #326 (OTel wiring)

## Why

V113-V117 built a complete error taxonomy: every `AiError`-rooted
error implements `ErrorCode` and emits a stable subsystem-prefixed
code plus structured `(key, String)` fields. The catalog covers 182
codes with `en` + `es` localizations.

But until V118, the `AiSpan` type in `opentelemetry_integration.rs`
only had `AiSpan::fail(&str)` — which dumped the whole error into a
single free-text `error_message` attribute. Any backend that wanted
to "group spans by error type" had to regex-parse that message.

The V113-V117 work was a no-op for OTel consumers until V118 closed
that loop.

## What changed

### 1. `AiSpan::fail_with_structured(&StructuredError)`

Sets the standard fail state plus three attribute families:

```rust
pub fn fail_with_structured(&mut self, structured: &StructuredError) {
    self.status = "error".to_string();
    self.error_message = Some(structured.message.clone());
    self.attributes.insert("error.code".into(), structured.code.into());
    for (k, v) in &structured.fields {
        self.attributes.insert(format!("error.fields.{}", k), v.clone());
    }
    for (i, src) in structured.source_chain.iter().enumerate() {
        self.attributes.insert(format!("error.source_chain.{}", i), src.clone());
    }
    self.finish();
}
```

**Attribute schema**:

| Attribute | Source | Example |
|---|---|---|
| `error.code` | `StructuredError::code` | `"PROVIDER_RATE_LIMITED"` |
| `error.fields.<key>` | `StructuredError::fields` | `error.fields.provider = "openai"`, `error.fields.retry_after = "30"` |
| `error.source_chain.<i>` | `StructuredError::source_chain` | `error.source_chain.0 = "connection refused"` |

The flat-string layout (rather than a single nested JSON blob)
matches OTel's preferred attribute model. Any collector / backend
can index on `error.code` and filter `error.fields.provider="openai"`
without parsing nested structures.

### 2. `AiSpan::fail_structured<E>(&E)` — convenience

For callers that already have an `AiError` (or any `ErrorCode +
std::error::Error`), there's a one-liner that builds the
`StructuredError` and delegates:

```rust
pub fn fail_structured<E>(&mut self, err: &E)
where
    E: ErrorCode + std::error::Error + ?Sized,
{
    let s = StructuredError::from_err(err);
    self.fail_with_structured(&s);
}
```

### 3. `OtelTracer::record_structured_error<E>(span, &err)`

Parallel to the existing `record_error(span, &str)`, but
taxonomy-aware. This is the preferred path for any failure that
already implements `ErrorCode`:

```rust
let span = tracer.start_span("provider.call");
match call_provider().await {
    Ok(resp) => tracer.end_span(span),
    Err(e) => tracer.record_structured_error(span, &e),
}
```

The `record_error(span, &str)` API stays untouched for callers that
only have a free-text message (third-party errors, sentinels, etc.).

## End-to-end example

```rust
use ai_assistant::error::{AiError, ProviderError};
use ai_assistant::opentelemetry_integration::{OtelTracer, OtelConfig};

let tracer = OtelTracer::new(OtelConfig::default());
let span = tracer.start_span("provider.call");

let err = AiError::Provider(ProviderError::RateLimited {
    provider: "openai".into(),
    retry_after: Some(30),
});
tracer.record_structured_error(span, &err);

let s = &tracer.completed_spans()[0];
assert_eq!(s.status, "error");
assert_eq!(s.attributes["error.code"], "PROVIDER_RATE_LIMITED");
assert_eq!(s.attributes["error.fields.provider"], "openai");
assert_eq!(s.attributes["error.fields.retry_after"], "30");
```

## Tests

4 new tests in `opentelemetry_integration::tests`, all pass:

- `test_aispan_fail_with_structured_emits_taxonomy_attributes` —
  end-to-end on `ProviderError::RateLimited`; asserts
  `error.code` + every `error.fields.<key>` is set, and the span is
  finished.
- `test_aispan_fail_structured_convenience` — exercises the
  one-liner on `WorkflowError::NodeNotFound`.
- `test_tracer_record_structured_error` — `OtelTracer` round-trip
  on `ConfigError::UnknownProvider`; checks the span lands in
  `completed_spans` with the right attributes.
- `test_aispan_fail_with_structured_handles_empty_fields` — no
  stray `error.fields.*` / `error.source_chain.*` attrs when the
  structured error carries none.

All 95 `opentelemetry_integration::tests` pass.

## State after V118

Phase C.2 of `plan_tier1_competitive_gaps.md` is **complete**:

| Iter | Scope |
|---|---|
| V113 | Core: `ErrorCode` trait, `StructuredError`, `errors/{en,es}.json` |
| V114 | `AiError` umbrella + 8 sub-types; flip 22 inherent-code tests stay coarse |
| V115 | RAG triad (`RagPipelineError`, `EmbeddingError`, `KpkgError`) |
| V116 | Provider/network triad (Anthropic, OpenAI, HuggingFace, Resilient) |
| V117 | 15 long-tail subsystems under `AiError` |
| **V118** | **OTel wiring — `AiSpan::fail_with_structured` + `record_structured_error`** |

## What's next

Tier 1 Phase B remains:
- **B.4** Stuck Detector + critique-based refinement.
- **B.5** Parallel tool execution.
- **B.6** Adversary + egress inspectors + `--no-egress` flag.
