# V133 — Repo hygiene: models.dev wired, RUSTSEC patched, zero-warning build

**Date**: 2026-05-11
**Version**: 0.2.80

V133 is a maintenance cycle, not a feature cycle. Three things had
quietly drifted in the repo between V124 and V132:

1. **An unwired module.** `src/models_dev.rs` (the V104.9
   models.dev catalog parser) sat unreferenced from `lib.rs` — it
   compiled in isolation under `cargo check --lib --tests` but no
   external caller could reach it.
2. **A live RUSTSEC advisory** against `wasmtime 36.0.7`
   (RUSTSEC-2026-0114) was tripping the CI security-audit job.
3. **75 `-D warnings` clippy errors** had accumulated in the
   `--release --all-targets` path. The default profile still
   compiled green, so it had gone unnoticed; the strict CI lane
   was failing.

V133 closes all three plus drops three on-disk scratch files
(`new_code.txt`, `new_tests.txt`, `_server_orig.json`) that had
already been integrated into `src/context_composer.rs` but never
deleted.

## Why each was wrong

### Unwired `models_dev` module

The module ships a parser + on-disk cache for the
[models.dev](https://models.dev) catalog: a community-maintained
JSON list of every known LLM with capability metadata
(context window, vision, tool use, pricing, knowledge cutoff).
Without it, every provider's model list had to be hand-curated
inside `models.rs` and could only be refreshed by editing source.

The module had been built — `ModelRegistry::load_from_cache`,
`ModelMetadata`, `fetch_and_cache` — but `lib.rs` never declared
it. The result was 565 lines of dead code that `cargo test
--lib` happily compiled (tests are still reachable) but no
consumer could `use ai_assistant::models_dev::ModelRegistry`.

The library framing rule says caller only configures, everything
lives in-crate. Leaving this module unreachable made the rule
quietly false.

### RUSTSEC-2026-0114 (wasmtime)

`wasmtime 36.0.7` had a memory-safety advisory affecting the WASI
preview1 path. The fix landed in `36.0.9`. Nothing in our usage
exercises the affected codepath, but `cargo audit` still flagged
it on every CI run.

### 75 strict-clippy errors

`cargo clippy --all-targets --release -- -D warnings` had been
breaking for nine versions (since V124-ish). The reasons were
all small but additive:

- 5× `cloned_ref_to_slice_refs` (`&[x.clone()]` patterns in
  test code that pre-date `slice::from_ref`).
- 2× `duplicated_attribute` (`#![cfg(feature = "vision")]` at the
  top of `mmproj.rs` / `embedded_server.rs` when `lib.rs` already
  declared `#[cfg(feature = "vision")] pub mod ...`).
- `large_enum_variant` on `ModelResolution::Virtual(VirtualModel)`
  — the `VirtualModel` payload is ~10× the size of the other
  variants, so every `ModelResolution` was allocated for the
  worst case.
- 2× `lines_filter_map_ok` (`reader.lines().flatten()` — the docs
  call out that this can loop forever on a persistent `Err`).
- `create_without_truncate` in `gguf_downloader.rs` — `.create(true)`
  without an explicit `.truncate()` is a footgun: the default
  changed once already, and on resumable downloads "keep the
  existing bytes" must be explicit.
- Several local cleanups: unused variables, manual prefix
  stripping, `format!` inside `println!` args, doc list
  indentation, an unused `()` type alias, a `loop_counter`
  pattern, `field_reassign_with_default` in tests, immediate
  `push` after `Vec::new()`.

Each individual fix is mechanical. The reason they piled up is
that `--release --all-targets` is slower than the default profile
and rarely run during day-to-day work — only CI tripped on them.

### Scratch files in `src/`

`new_code.txt`, `new_tests.txt`, `_server_orig.json` lived next
to the actual modules. Audit showed all three were drafts that
had already been integrated into `src/context_composer.rs`
(`ContextCompiler`, `SegmentType`, `ConversationCompactor`,
`ToolSearchIndex` + the matching tests) some sessions ago.
Leaving them in the tree confused the next person who ran
`grep` over `src/` looking for a definition.

## What changed

### `src/lib.rs`

```rust
mod models;
pub mod models_dev;   // ← new line
mod providers;
```

The module is exposed via its namespace rather than re-exporting
flat: both `models::ModelRegistry` and `models_dev::ModelRegistry`
exist with different responsibilities, so flat re-export would
collide. The bridge functions (below) handle conversion when
needed.

### `src/models_dev.rs` — bridge helpers

```rust
pub fn provider_from_key(key: &str) -> crate::config::AiProvider { ... }

impl ModelMetadata {
    pub fn to_model_info(&self) -> crate::models::ModelInfo { ... }
}

impl ModelRegistry {
    pub fn to_model_infos(&self) -> Vec<crate::models::ModelInfo> { ... }
}

impl crate::models::ModelRegistry {
    pub fn extend_from_models_dev(&mut self, src: &ModelRegistry) { ... }
}
```

Five new tests cover the bridge: known-provider mapping, the
fallthrough to `OpenAICompatible` for unknown providers, capability
mapping (vision/tool/json/streaming/cost/window/cutoff), in-crate
registry extension preserves order, and round-trip equivalence.
The cache logic and HTTP fetch path (`fetch_and_cache`) were
already covered by the existing 14 tests in this module.

### `Cargo.lock` — wasmtime 36.0.7 → 36.0.9

Stayed in the `36.x` major to avoid the API churn between 36
and 44. `cargo audit` now reports 0 vulnerabilities (the 4
remaining notices are policy-allowed warnings, not advisories).

### Strict-clippy fixes

All 75 errors closed. Highlights:

- `ModelResolution::Virtual(VirtualModel)` → `Virtual(Box<VirtualModel>)`.
  Field access on the variant still autoderefs (`Box<T>`), so
  every call site (`server_axum.rs`, the resolver itself, tests)
  works without explicit deref. Net: every `ModelResolution`
  shrinks from ~2 KB to ~32 B.
- `#![cfg(feature = "vision")]` at the top of `mmproj.rs` and
  `embedded_server.rs` removed — `lib.rs` already gates the
  `pub mod` declaration, so the inner attribute was redundant
  and triggered `duplicated_attribute`.
- `reader.lines().flatten()` → `reader.lines().map_while(Result::ok)`
  in `ai_cli.rs` (two sites). The new form short-circuits on the
  first I/O error instead of looping.
- `OpenOptions::new().create(true).write(true).read(true)` →
  same chain plus `.truncate(false)` in `gguf_downloader.rs`,
  because resumable downloads explicitly want to *not* truncate.
- `&[x.clone()]` → `std::slice::from_ref(&x)` across
  `caching.rs`, `reranker.rs`, `advanced_routing.rs` test code.
- `for (key, _, _, size) in &candidates { ...; freed_entries += 1; }`
  → `for (idx, (key, _, _, size)) in candidates.iter().enumerate()
  { let freed_entries = idx; ... }` in `distributed.rs` —
  clippy was right that `enumerate` makes the intent obvious.
- Local cleanups: `_warnings`, `_result`, `vec![..]` instead of
  immediate push, doc list indentation in `recipes.rs`,
  `strip_prefix("/use ").map(str::trim)` instead of
  `[5..].trim()`, struct-literal `..Default::default()` in
  `rag_tier_tests.rs`, `[{:?}]` directly in `println!` instead
  of nested `format!`.

The build is now green under
`cargo clippy --all-targets --release -- -D warnings`.

### Removed files

```
src/new_code.txt        (537 lines — already in src/context_composer.rs)
src/new_tests.txt       (285 lines — already in src/context_composer.rs tests)
src/_server_orig.json   (30 bytes — stale metadata)
```

Verified before deletion: every `pub` / `pub(crate)` symbol in
`new_code.txt` (`ContextCompiler`, `SegmentType`,
`ConversationCompactor`, `ToolSearchIndex`) is reachable from
the live `context_composer.rs` and all the matching tests in
`new_tests.txt` are also present and passing.

## Compatibility

- **API surface unchanged.** All bridge helpers in `models_dev`
  are new symbols on a previously-unreferenced module. Nothing
  existing changed shape.
- **`ModelResolution::Virtual` is now `Virtual(Box<VirtualModel>)`.**
  External callers that pattern-match the variant (`Virtual(v)`,
  `Virtual(ref v)`) continue to compile because `Box<T>`
  autoderefs for field access. The only break would be code that
  pattern-binds the inner value by *value* and moves a field out
  without going through `Box`; no in-tree caller does that, and
  it would be a niche external pattern.
- **CLI flag surface unchanged.**

## Verification

```bash
# Clippy clean (was 75 errors)
cargo clippy --all-targets --release -- -D warnings

# Library + integration tests
cargo test --lib --release    # 6230 pass; one timing-flaky test
                              # (api_key_rotation::test_key_expiry)
                              # passes in isolation

# Security audit
cargo audit
# → 0 vulnerabilities (was 1: RUSTSEC-2026-0114)

# Binaries build clean
cargo build --release --bins
```

## What V133 deliberately does *not* do

- **No new features.** This is housekeeping. New surfaces
  (anti-hallucination, distributed RAG, GDPR purge, release
  automation) all landed in V125–V132 and are unchanged here.
- **No `models.dev` auto-refresh policy.** The module is now
  reachable from `ai_assistant::models_dev::*`; whether to call
  `fetch_and_cache()` on startup is left to the caller.
  Auto-refresh belongs in a future cycle with a clear cache
  invalidation story.
- **No version bump for wasmtime past 36.x.** `36.0.9` clears
  the advisory. Jumping to `44.x` would mean API churn in the
  WASM feature flag path for marginal benefit.
