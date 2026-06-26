# IMPROVEMENTS_V162 — `AiAssistant` public API speaks `AiResult`

**Version:** 0.2.113 → 0.2.114
**Scope:** `src/assistant.rs` (+ CI advisory hygiene)
**Feature:** none new

## Why

The code-quality audit's #1 ergonomics finding: the crate ships a
well-designed `AiError` (hierarchical, `source()`/`suggestion()`/
`is_recoverable()`/`code()`, `#[non_exhaustive]`) but the flagship object
`AiAssistant` returned `anyhow::Result<T>` from **every** public method —
so callers got type-erased `anyhow` errors and couldn't `match` on
`AiError` or use its machinery. This is the first of the deferred
architectural follow-ups (V161 listed it as "deliberately deferred").

## What changed

### `AiAssistant` → `AiResult`

All 38 public methods (and one free helper) that returned the bare
`anyhow::Result<T>` alias now return `crate::error::AiResult<T>`
(= `Result<T, AiError>`). The import at the top of `assistant.rs` moved
from `use anyhow::Result;` to `use crate::error::{AiError, AiResult}`.

The migration is **behavior-preserving**. It is safe because
`impl From<anyhow::Error> for AiError` already exists
(`error.rs:2465`, → `AiError::Other(e.to_string())`): every `?` and
`.context(...)?` inside the bodies auto-converts an `anyhow::Error` into
the same `AiError::Other` it would have produced. Only the **non-`?`**
error productions needed explicit handling, all kept byte-identical:

- `return Err(primary_err)` / `return Err(last_err)` (anyhow values from
  `generate_response` / `generate_vision_response`) → `.into()`.
- `return Err(anyhow::anyhow!(…))` and `anyhow::bail!(…)` →
  `return Err(AiError::Other(format!(…)))` with the same message/args.
- Tail `… .map_err(|e| anyhow::anyhow!(e))` (no `?`) →
  `… .map_err(|e| AiError::Other(e.to_string()))`.
- Tail `return db.foo(…)` returning `anyhow::Result` → `.map_err(AiError::from)`.
- One `ureq` `?` site that relied on anyhow's blanket `From<ureq::Error>`
  → `.map_err(anyhow::Error::from)?` (routes through anyhow, then `?`
  converts to `AiError` — identical message).

The 10 signatures that already returned an explicit error type
(`Result<_, String>`, `Result<_, crate::error::AiError>`) were left
untouched.

### Caller impact: none

No binary or example needed changes: `AiError` implements
`std::error::Error`, so callers' `?` (into an `anyhow::Result` or
`AiResult` fn) and `.context(...)` keep working. Verified with
`cargo check --features full` (lib + all bins) and the example/harness
builds.

### CI advisory hygiene (folded in to keep `master` green)

The RustSec DB published new advisories the week of 2026-06-26:

- **RUSTSEC-2026-0185** — `quinn-proto` 0.11.14 remote memory exhaustion.
  **Fixed** by an in-range bump to **0.11.15** (`cargo update -p
  quinn-proto --precise 0.11.15`; also deduped stale `windows_*` lock
  entries).
- **RUSTSEC-2026-0187** — `lopdf` 0.34.0 stack overflow on deeply-nested
  PDFs. **Ignored** (documented) in all three sync'd places
  (`ci.yml`, `supply-chain.yml`, `deny.toml`): it is purely transitive
  via `pdf-extract` 0.7.12, which pins `lopdf ^0.34`; no `pdf-extract`
  release uses `lopdf >= 0.42` yet, so it cannot be bumped today. Only
  reachable behind the opt-in `documents`/`pdf-extract` feature; DoS, not
  RCE. Re-check date 2026-09-01.

## Tests

No behavior change ⇒ existing tests cover it. Verified: rustfmt clean;
clippy `-D warnings` clean (lib + bins, `FEATURES_STD`, toolchain 1.93.0
= CI); `ai_test_harness --all` 585/585; `cargo audit` clean with the
updated ignore set.

## Still deferred

- Split `advanced_routing.rs` (~10K lines) → `advanced_routing/` (V163).
- Sweep the 59 `#[allow(dead_code)]` (V164).
