# V116 — Phase C.2: ErrorCode rollout to provider adapters + resilient registry

**Date**: 2026-05-04
**Version**: 0.2.63
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #324 (provider adapters + provider registry resilience)

## Why

Provider-level errors are the **hottest single failure surface** in
the codebase — every cloud LLM call traverses one of these adapters.
Until V116 they all emitted free-text `Display` strings; oncall and
dashboards had to regex-parse `"API error 429: rate limited"` to
slice by provider × error-type.

After V116, four high-traffic types now emit fine-grained codes plus
structured fields. Of the four, `ResilientError::AllProvidersFailed`
is the most useful: it carries an entire list of `(provider, message)`
pairs that previously stringified into one giant message. The new
`fields()` impl decomposes that into `attempted_count`, `providers`
(comma-separated), and `detail` (per-provider) so alerting can branch
on count or specific providers without parsing.

## What

### `AnthropicAdapterError` (5 codes)

| Variant | Code | Fields |
|---|---|---|
| `Network(s)` | `ANTHROPIC_NETWORK` | `reason` |
| `Serialization(s)` | `ANTHROPIC_SERIALIZATION` | `reason` |
| `Deserialization(s)` | `ANTHROPIC_DESERIALIZATION` | `reason` |
| `Api { code, error_type, message }` | `ANTHROPIC_API` | `status_code`, `error_type`, `message` |
| `RateLimit { retry_after }` | `ANTHROPIC_RATE_LIMITED` | `retry_after_ms` (optional) |

### `OpenAIAdapterError` (5 codes — mirror shape)

`OPENAI_NETWORK`, `OPENAI_SERIALIZATION`, `OPENAI_DESERIALIZATION`,
`OPENAI_API`, `OPENAI_RATE_LIMITED`. Same field shape.

### `HfError` (6 codes)

| Variant | Code | Fields |
|---|---|---|
| `Network(s)` | `HF_NETWORK` | `reason` |
| `Serialization(s)` | `HF_SERIALIZATION` | `reason` |
| `Deserialization(s)` | `HF_DESERIALIZATION` | `reason` |
| `Api { code, message }` | `HF_API` | `status_code`, `message` |
| `ModelLoading` | `HF_MODEL_LOADING` | — |
| `UnexpectedResponse` | `HF_UNEXPECTED_RESPONSE` | — |

### `ResilientError` (2 codes — the one with structure)

| Variant | Code | Fields |
|---|---|---|
| `AllProvidersFailed { errors }` | `RESILIENT_ALL_PROVIDERS_FAILED` | `attempted_count`, `providers` (joined), `detail` (per-provider joined) |
| `NoAvailableProviders` | `RESILIENT_NO_AVAILABLE_PROVIDERS` | — |

The `attempted_count` field is what unlocks "alert if >3 providers
failed in a row" semantics on the dashboard side.

### `errors/{en,es}.json`

65 → 83 entries. The localized templates are concise — the structured
fields carry the variable detail.

## Tests

```bash
cargo test --lib test_errorcode
# test result: ok. 18 passed; 0 failed; 0 ignored.
```

4 new tests, one per migrated module:

- `anthropic_adapter::tests::test_errorcode_anthropic` — covers `Api`
  (3 fields) and `RateLimit` (with `Some(Duration)`).
- `openai_adapter::tests::test_errorcode_openai` — covers `Network`
  (single-field) and `RateLimit { retry_after: None }` (no fields).
- `huggingface_connector::tests::test_errorcode_hf` — covers
  `ModelLoading` (no fields) and `Api` (status_code field).
- `providers::tests::test_errorcode_resilient` — covers
  `NoAvailableProviders` (no fields) and `AllProvidersFailed` with
  2 inner failures (asserts `attempted_count`, `providers`, `detail`
  shape).

## What's next

| Iteration | Scope |
|---|---|
| V117 | Long-tail umbrella variants in `src/error.rs` — `WorkflowError`, `A2AError`, `VoiceAgentError`, `MediaGenerationError`, `DistillationError`, `ConstrainedDecodingError`, `HitlError`, `McpClientError`, `AgentEvalError`, `RedTeamError`, `MctsError`, `DevToolsError`, `EvalSuiteError`, `AdvancedRoutingError`. Then flip `AiError::ErrorCode::code` long-tail arms from coarse-fallback to delegate. |
| V118 | OTel wiring — `opentelemetry_integration.rs::AiSpan` sets `error.code` + `error.fields.*` attributes from `StructuredError::from_err(...).to_json()`. |
| V119 | External locale resolver — runtime drop-in of `errors/<locale>.json`. |

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.62 → 0.2.63 |
| `src/anthropic_adapter.rs` | `impl ErrorCode for AnthropicAdapterError` + 1 test |
| `src/openai_adapter.rs` | `impl ErrorCode for OpenAIAdapterError` + 1 test |
| `src/huggingface_connector.rs` | `impl ErrorCode for HfError` + 1 test |
| `src/providers.rs` | `impl ErrorCode for ResilientError` + 1 test |
| `errors/en.json` | 65 → 83 codes |
| `errors/es.json` | 65 → 83 codes |
| `CHANGELOG.md` | V116 entry |
| `docs/IMPROVEMENTS_V116.md` | this file |
