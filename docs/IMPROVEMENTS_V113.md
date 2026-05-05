# V113 — Phase C.2 (core): structured error taxonomy

**Date**: 2026-05-04
**Version**: 0.2.60
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.2
**Tasks**: #321 (error taxonomy core + pilot migration)

## Why

Until V113 every subsystem rolled its own error type — `BackendError`,
`AiError`, `RagError`, `NetworkError`, … — and each one printed via a
hand-written `Display` / `Debug` impl. Three problems:

1. **No stable codes.** Structured-log consumers had no way to match
   `"model not found"` across versions because the wire shape was just
   the `Display` string. Renaming a variant or rewording its message was
   silently a breaking change for anyone grepping logs.
2. **No structured fields.** `format!("model not found: {}", path)` flattens
   the path into the message. OTel attributes can't pull it out without
   regex-parsing the human string.
3. **No i18n.** All messages were English-only, baked at the call site.

V113 fixes the **core** — the trait, the wire shape, the i18n loader, the
migration recipe — and migrates **one subsystem** (`local_inference`) as
the pattern. V114+ rolls the recipe across the rest of the codebase.

## What

### `thiserror 2` as a direct dep

Always-on, derive-only — zero runtime cost. Replaces the hand-written
`Display + Error` impls that every subsystem error type was carrying.

### `src/error_taxonomy.rs`

Three pieces:

| Piece | Purpose |
|---|---|
| `pub trait ErrorCode { fn code(&self) -> &'static str; fn fields(&self) -> Vec<(&'static str, String)> }` | Every subsystem error enum implements this. `code()` is screaming-snake-case, prefixed by subsystem (`LOCAL_INFER_*`, `RAG_*`, …) and **never renamed once shipped**. `fields()` extracts the structured payload (paths, IDs, kinds) from the variant. |
| `pub struct StructuredError { code: &'static str, message: String, fields: BTreeMap<String, String>, source_chain: Vec<String> }` | Owned, JSON-serializable. Built from any `ErrorCode + std::error::Error` via `StructuredError::from_err(&err)`. Walks the source chain up to 8 deep. What OTel spans + structured logs emit. |
| i18n loader | `errors/<locale>.json` baked via `include_str!` for `en` + `es`, parsed once into `OnceLock<BTreeMap<&'static str, String>>`. `{field}` placeholders substitute from `StructuredError::fields`. Unknown locales fall through to the underlying `Display`. |

### `errors/en.json` + `errors/es.json`

Codes for the pilot migration:

```json
{
  "LOCAL_INFER_NOT_IMPLEMENTED": "Backend not compiled in: {backend}",
  "LOCAL_INFER_MODEL_NOT_FOUND": "Model file not found: {path}",
  "LOCAL_INFER_IO":              "I/O error during local inference: {io_kind}",
  "LOCAL_INFER_BACKEND":         "Local inference backend error: {detail}"
}
```

Spanish mirror in `es.json`. New codes are added here as subsystems migrate.

### Migration recipe (documented in module header)

```rust
use crate::error_taxonomy::ErrorCode;

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("backend not compiled in: {0}")]
    NotImplemented(&'static str),
    #[error("model not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("backend error: {0}")]
    Backend(String),
}

impl ErrorCode for BackendError {
    fn code(&self) -> &'static str {
        match self {
            BackendError::NotImplemented(_) => "LOCAL_INFER_NOT_IMPLEMENTED",
            BackendError::ModelNotFound(_)  => "LOCAL_INFER_MODEL_NOT_FOUND",
            BackendError::Io(_)             => "LOCAL_INFER_IO",
            BackendError::Backend(_)        => "LOCAL_INFER_BACKEND",
        }
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            BackendError::NotImplemented(name) => vec![("backend", (*name).into())],
            BackendError::ModelNotFound(p)     => vec![("path", p.display().to_string())],
            BackendError::Io(e)                => vec![("io_kind", format!("{:?}", e.kind()))],
            BackendError::Backend(s)           => vec![("detail", s.clone())],
        }
    }
}
```

Then:

- `StructuredError::from_err(&err).to_json()` → wire-ready JSON.
- `StructuredError::from_err(&err).localize("es")` → human-ready Spanish.

### Pilot: `local_inference::BackendError`

First subsystem onto the new taxonomy. Behaviour is **unchanged** —
same variants, same `Display` strings. The hand-written `impl Display`
+ `impl Error` + `impl From<std::io::Error>` are gone, replaced by
`#[derive(thiserror::Error)]` + `#[from]`. The new `impl ErrorCode`
wires the four `LOCAL_INFER_*` codes and per-variant `fields()`.

## Tests

- **7 new** `error_taxonomy::tests`:
  - `from_err` populates code + message + fields from `ErrorCode + Error`.
  - Source chain is walked up to 8 deep, then capped.
  - Substitution: known field, unknown placeholder (left literal),
    unclosed `{` (left literal), no fields (passthrough).
  - JSON roundtrip via `serde_json` preserves shape.
  - Locale fallback: missing locale → underlying `Display`; missing
    code in present locale → underlying `Display`.
- **14 existing** `local_inference` tests pass post-migration.

```bash
cargo test --features local-inference --lib error_taxonomy::tests local_inference::tests
```

## What's next (V114+)

| Iteration | Scope |
|---|---|
| V114 | `error.rs` umbrella `AiError` (22 sub-enums — `ConfigError`, `ProviderError`, `RagError`, `NetworkError`, `ValidationError`, `ResourceLimitError`, …) — fine-grained per-variant codes. Expands `errors/{en,es}.json` accordingly. |
| V115 | RAG subsystem — `Self-RAG`, `CRAG`, `Graph RAG`, `RAPTOR` error paths. |
| V116 | Providers (18) + network/HTTP layer. |
| V117 | Long-tail subsystems (~70 files). |
| V118 | Wire `StructuredError::to_json()` into `opentelemetry_integration.rs::AiSpan` — set `error.code` + `error.fields.*` attributes from the structured form, replacing the current free-text `error.message`. |
| V119 | External locale resolver — let callers drop in extra `errors/<locale>.json` at runtime (today's loader is in-tree only via `include_str!`). |

## Files touched

| File | Change |
|---|---|
| `Cargo.toml` | bump 0.2.59 → 0.2.60; add `thiserror = "2"` direct dep |
| `src/lib.rs` | declare `pub mod error_taxonomy;` |
| `src/error_taxonomy.rs` | new — trait + struct + i18n loader + 7 tests |
| `errors/en.json` | new — 4 `LOCAL_INFER_*` codes |
| `errors/es.json` | new — Spanish mirror |
| `src/local_inference.rs` | `BackendError` migrated to `thiserror` + `ErrorCode`. Hand-written `Display`/`Error`/`From` impls removed. |
| `CHANGELOG.md` | V113 entry |
| `docs/IMPROVEMENTS_V113.md` | this file |
