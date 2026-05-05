# V114 — Phase C.2: ErrorCode rollout to AiError umbrella

**Date**: 2026-05-04
**Version**: 0.2.61
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #322 (umbrella + 8 sub-types onto `ErrorCode`)

## Why

V113 shipped the structured-error core (`ErrorCode` trait,
`StructuredError`, i18n loader) and migrated **one** subsystem
(`local_inference::BackendError`) as the pattern. V114 rolls the
recipe across the umbrella `AiError` plus the 8 sub-types that
account for the bulk of error sites in the codebase.

The migration is **additive** — no `Display`, `Error`, or `From`
impl is rewritten. The existing inherent `AiError::code()` returning
coarse categories (`"PROVIDER"`, `"CONFIG"`, …) is **preserved** so
the 22 tests asserting against it keep passing and any external
consumer keeps working. The new fine-grained code is reachable via
`<AiError as ErrorCode>::code(&err)`.

This sequencing choice matters: a one-shot thiserror-derive migration
of 22 enums + 3832 lines is too risky for one iteration. Additive
trait impls land V113-level structure on the wire today, then V117
flips the inherent method to fine-grained once the long tail also
implements `ErrorCode`.

## What

### Wired sub-types (8)

| Sub-type | Codes (count) | Sample |
|---|---|---|
| `ConfigError` | 5 | `CONFIG_MISSING_VALUE`, `CONFIG_LOAD_FAILED`, `CONFIG_UNKNOWN_PROVIDER` |
| `ProviderError` | 9 | `PROVIDER_RATE_LIMITED { provider, retry_after }`, `PROVIDER_API_ERROR { provider, status_code, message }`, `PROVIDER_CANCELLED` |
| `RagError` | 7 | `RAG_APPEND_ONLY_VIOLATION { operation, source }`, `RAG_DOCUMENT_NOT_FOUND { source }` |
| `NetworkError` | 5 | `NETWORK_TIMEOUT { url, timeout_ms }`, `NETWORK_DNS_ERROR { host }` |
| `ValidationError` | 5 | `VALIDATION_OUT_OF_RANGE { field, min, max, value }` |
| `ResourceLimitError` | 6 | `RESOURCE_BUDGET_EXCEEDED { budget, used, currency }` |
| `IoError` | 1 | `IO_GENERIC { operation, path?, reason }` |
| `SerializationError` | 1 | `SERIALIZATION_ERROR { format, operation, reason }` |

Total: **39 fine-grained codes** added in `errors/en.json` + `errors/es.json`
(plus 4 from V113 = 42 entries today).

### Umbrella behavior

`<AiError as ErrorCode>::code()`:

- For wired sub-types → delegates to inner. `AiError::Provider(ProviderError::Cancelled)` → `"PROVIDER_CANCELLED"`.
- For `Other(detail)` → `"OTHER"`, fields `{ detail }`.
- For long-tail (`Workflow`, `AdvancedMemory`, `A2A`, `VoiceAgent`, `MediaGeneration`, `Distillation`, `ConstrainedDecoding`, `Hitl`, `McpClient`, `AgentEval`, `RedTeam`, `Mcts`, `DevTools`, `EvalSuite`, `AdvancedRouting`) → falls back to coarse category (`"WORKFLOW"`, `"A2A"`, …). These flip to fine-grained in V117 once their enums also implement `ErrorCode`.

`<AiError as ErrorCode>::fields()` aggregates the inner enum's fields,
or `[(detail, msg)]` for `Other`, or `[]` for long-tail.

The inherent `AiError::code()` (returning coarse categories) is
**unchanged**. Both methods coexist; the trait method is name-shadowed
when called as `err.code()` and reached via explicit trait
disambiguation.

### Wire example

```rust
use ai_assistant::error::{AiError, ProviderError};
use ai_assistant::error_taxonomy::{ErrorCode, StructuredError};

let err: AiError = ProviderError::RateLimited {
    provider: "openai".into(),
    retry_after: Some(30),
}.into();

// Coarse — preserved API
assert_eq!(err.code(), "PROVIDER");

// Fine-grained — new
assert_eq!(<AiError as ErrorCode>::code(&err), "PROVIDER_RATE_LIMITED");

// JSON wire shape
let s = StructuredError::from_err(&err);
println!("{}", s.to_json());
// {"code":"PROVIDER_RATE_LIMITED","message":"Rate limited by openai, retry after 30 seconds",
//  "fields":{"provider":"openai","retry_after":"30"},"source_chain":[...]}

// Localized
println!("{}", s.localize("es"));
// "Limitado por openai"
```

## Tests

- **11 new** `error::tests::test_errorcode_*` tests:
  - Per sub-type fine-grained code + fields shape.
  - `AiError` delegates to inner; inherent coarse method preserved.
  - `Other(detail)` emits `OTHER` with `detail` field.
  - Long-tail variants still emit coarse category code.
  - `IoError` / `SerializationError` shape.
  - Full `StructuredError::from_err` + `localize("en")` + `localize("es")` roundtrip.
- **41 existing** `error::tests` regression-clean. Total: 52.

```bash
cargo test --lib error::tests
# test result: ok. 52 passed; 0 failed; 0 ignored.
```

## What's next

| Iteration | Scope |
|---|---|
| V115 | RAG deep modules — `Self-RAG`, `CRAG`, `Graph RAG`, `RAPTOR` error paths inside the implementations themselves, beyond the umbrella `RagError`. |
| V116 | 18 providers — provider-specific submodule error types where they exist (some providers wrap `ProviderError` with extra context that needs its own codes). |
| V117 | Long-tail umbrella variants — `WorkflowError`, `AdvancedMemoryError`, `A2AError`, `VoiceAgentError`, `MediaGenerationError`, `DistillationError`, `ConstrainedDecodingError`, `HitlError`, `McpClientError`, `AgentEvalError`, `RedTeamError`, `MctsError`, `DevToolsError`, `EvalSuiteError`, `AdvancedRoutingError` — onto `ErrorCode`. Then flip `AiError::ErrorCode::code` long-tail arms to delegate. |
| V118 | OTel wiring — `opentelemetry_integration.rs::AiSpan` sets `error.code` + `error.fields.*` attributes from `StructuredError`. Replaces today's free-text `error.message`. |
| V119 | External locale resolver — drop-in `errors/<locale>.json` at runtime (today's loader is in-tree only). |

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.60 → 0.2.61 |
| `src/error.rs` | new V114 block — `impl ErrorCode for {AiError, ConfigError, ProviderError, RagError, NetworkError, ValidationError, ResourceLimitError, IoError, SerializationError}` (~330 LOC). 11 new tests. |
| `errors/en.json` | 4 → 42 codes |
| `errors/es.json` | 4 → 42 codes |
| `CHANGELOG.md` | V114 entry |
| `docs/IMPROVEMENTS_V114.md` | this file |
