# V115 — Phase C.2: ErrorCode rollout to RAG dependency triad

**Date**: 2026-05-04
**Version**: 0.2.62
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #323 (RAG pipeline + embeddings + encrypted knowledge)

## Why

V114 wired the umbrella `RagError` (in `src/error.rs`) onto
`ErrorCode`. That covers the 7 high-level RAG operations exposed
to library callers. But the RAG path has **three layers below
the umbrella**:

```
caller
  └─ AiError::Rag(RagError)              ← V114 (umbrella, 7 codes)
       └─ RagPipelineError               ← V115 (orchestration, 9 codes)
            └─ EmbeddingError            ← V115 (vector ops, 5 codes)
            └─ KpkgError                 ← V115 (encrypted storage, 9 codes)
```

A failed retrieval today surfaces as `RAG_DATABASE` from the
umbrella, with the underlying root cause (decryption failure,
dimension mismatch, vector store down) flattened into the
free-text `reason` field. After V115, `StructuredError::from_err(&err)`
walks the source chain and emits each layer's code, so a structured-log
consumer can match on `KPKG_DECRYPTION_FAILED` directly without
parsing strings.

## What

### `RagPipelineError` (9 codes)

| Variant | Code | Fields |
|---|---|---|
| `NoSources` | `RAG_PIPELINE_NO_SOURCES` | — |
| `MissingRequirement(req)` | `RAG_PIPELINE_MISSING_REQUIREMENT` | `requirement` (via `display_name()`) |
| `QueryProcessingError(s)` | `RAG_PIPELINE_QUERY_PROCESSING` | `reason` |
| `RetrievalError(s)` | `RAG_PIPELINE_RETRIEVAL` | `reason` |
| `PostProcessingError(s)` | `RAG_PIPELINE_POST_PROCESSING` | `reason` |
| `LlmError(s)` | `RAG_PIPELINE_LLM` | `reason` |
| `Timeout` | `RAG_PIPELINE_TIMEOUT` | — |
| `ConfigError(s)` | `RAG_PIPELINE_CONFIG` | `reason` |
| `Internal(s)` | `RAG_PIPELINE_INTERNAL` | `reason` |

### `EmbeddingError` (5 codes)

| Variant | Code | Fields |
|---|---|---|
| `ApiError(s)` | `EMBEDDING_API` | `reason` |
| `ParseError(s)` | `EMBEDDING_PARSE` | `reason` |
| `ConfigError(s)` | `EMBEDDING_CONFIG` | `reason` |
| `EmptyResult` | `EMBEDDING_EMPTY_RESULT` | — |
| `DimensionMismatch { expected, got }` | `EMBEDDING_DIMENSION_MISMATCH` | `expected`, `got` |

### `KpkgError` (9 codes)

| Variant | Code | Fields |
|---|---|---|
| `DataTooShort` | `KPKG_DATA_TOO_SHORT` | — |
| `DecryptionFailed` | `KPKG_DECRYPTION_FAILED` | — |
| `InvalidZipArchive(s)` | `KPKG_INVALID_ZIP` | `reason` |
| `ZipReadError(s)` | `KPKG_ZIP_READ` | `reason` |
| `ZipWriteError(s)` | `KPKG_ZIP_WRITE` | `reason` |
| `InvalidUtf8(path)` | `KPKG_INVALID_UTF8` | `path` |
| `ManifestError(s)` | `KPKG_MANIFEST` | `reason` |
| `EmptyPackage` | `KPKG_EMPTY_PACKAGE` | — |
| `IoError(s)` | `KPKG_IO` | `reason` |

### `errors/{en,es}.json`

42 → 65 entries. Every new code gets a templated message in both
locales with `{field}` placeholders matching the `fields()` keys above.

## Tests

```bash
cargo test --lib test_errorcode
# test result: ok. 14 passed; 0 failed; 0 ignored.
```

3 new tests, one per migrated module:

- `rag_pipeline::tests::test_errorcode_rag_pipeline` — covers
  `NoSources`, `Timeout` (no fields), and `RetrievalError(reason)`
  (single field).
- `neural_embeddings::tests::test_errorcode_embedding` — covers
  `EmptyResult` (no fields), `DimensionMismatch` (multi-field with
  numeric serialization), `ApiError` (single field).
- `encrypted_knowledge::tests::test_errorcode_kpkg` — covers
  `DecryptionFailed` (no fields), `InvalidUtf8(path)` (path field),
  `ManifestError(reason)`.

## What's next

| Iteration | Scope |
|---|---|
| V116 | 18 providers — provider-specific submodule error types (`AnthropicAdapterError`, `OpenAIAdapterError`, `HfError`, `ResilientError` in `providers.rs`, …). |
| V117 | Long-tail umbrella variants — `WorkflowError`, `A2AError`, `VoiceAgentError`, `MediaGenerationError`, `DistillationError`, `ConstrainedDecodingError`, `HitlError`, `McpClientError`, `AgentEvalError`, `RedTeamError`, `MctsError`, `DevToolsError`, `EvalSuiteError`, `AdvancedRoutingError`, plus the rest of the long tail in submodules. Flip `AiError::ErrorCode::code` long-tail arms to delegate. |
| V118 | OTel wiring — `opentelemetry_integration.rs::AiSpan` sets `error.code` + `error.fields.*` from `StructuredError`. |

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.61 → 0.2.62 |
| `src/rag_pipeline.rs` | `impl ErrorCode for RagPipelineError` + 1 test |
| `src/neural_embeddings.rs` | `impl ErrorCode for EmbeddingError` + 1 test |
| `src/encrypted_knowledge.rs` | `impl ErrorCode for KpkgError` + 1 test |
| `errors/en.json` | 42 → 65 codes |
| `errors/es.json` | 42 → 65 codes |
| `CHANGELOG.md` | V115 entry |
| `docs/IMPROVEMENTS_V115.md` | this file |
