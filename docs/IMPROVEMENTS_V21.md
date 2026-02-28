# v21 Improvements Changelog

> Date: 2026-02-28

v21 is another **test coverage push** targeting all remaining `src/*.rs` modules with fewer than 10 unit tests. Brings 40 more source files up to the 10-test minimum, adding +115 new tests across 5 phases.

---

## Summary Metrics

| Metric | v20 | v21 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,848 | 5,963 | +115 |
| Modules with ≥10 tests | ~195 | ~235 | +40 |
| Compiler warnings | 0 | 0 | 0 |
| Clippy warnings | 0 | 0 | 0 |

---

## Phase 1 — 9-test files → 10 (12 files, +12 tests)

| # | File | Before | After | Tests Added |
|---|------|--------|-------|-------------|
| 1 | `src/answer_extraction.rs` | 9 | 10 | +1 |
| 2 | `src/async_support.rs` | 9 | 10 | +1 |
| 3 | `src/binary_integrity.rs` | 9 | 10 | +1 |
| 4 | `src/context.rs` | 9 | 10 | +1 |
| 5 | `src/conversation_templates.rs` | 9 | 10 | +1 |
| 6 | `src/debug.rs` | 9 | 10 | +1 |
| 7 | `src/formatting.rs` | 9 | 10 | +1 |
| 8 | `src/huggingface_connector.rs` | 9 | 10 | +1 |
| 9 | `src/intent.rs` | 9 | 10 | +1 |
| 10 | `src/response_effectiveness.rs` | 9 | 10 | +1 |
| 11 | `src/vision.rs` | 9 | 10 | +1 |
| 12 | `src/wasm.rs` | 9 | 10 | +1 |

---

## Phase 2 — 8-test files → 10 (4 files, +8 tests)

| # | File | Before | After | Tests Added |
|---|------|--------|-------|-------------|
| 1 | `src/injection_detection.rs` | 8 | 10 | +2 |
| 2 | `src/metrics.rs` | 8 | 10 | +2 |
| 3 | `src/patch_application.rs` | 8 | 10 | +2 |
| 4 | `src/pii_detection.rs` | 8 | 10 | +2 |

---

## Phase 3 — 7-test files → 10 (7 files, +21 tests)

| # | File | Before | After | Tests Added |
|---|------|--------|-------|-------------|
| 1 | `src/latency_metrics.rs` | 7 | 10 | +3 |
| 2 | `src/messages.rs` | 7 | 10 | +3 |
| 3 | `src/model_warmup.rs` | 7 | 10 | +3 |
| 4 | `src/prefetch.rs` | 7 | 10 | +3 |
| 5 | `src/profiles.rs` | 7 | 10 | +3 |
| 6 | `src/token_budget.rs` | 7 | 10 | +3 |
| 7 | `src/tool_use.rs` | 7 | 10 | +3 |

---

## Phase 4 — 6-test files → 10 (11 files, +44 tests)

| # | File | Before | After | Tests Added |
|---|------|--------|-------|-------------|
| 1 | `src/confidence_scoring.rs` | 6 | 10 | +4 |
| 2 | `src/config.rs` | 6 | 10 | +4 |
| 3 | `src/connection_pool.rs` | 6 | 10 | +4 |
| 4 | `src/fallback.rs` | 6 | 10 | +4 |
| 5 | `src/i18n.rs` | 6 | 10 | +4 |
| 6 | `src/prompt_chaining.rs` | 6 | 10 | +4 |
| 7 | `src/query_expansion.rs` | 6 | 10 | +4 |
| 8 | `src/sse_streaming.rs` | 6 | 10 | +4 |
| 9 | `src/streaming_compression.rs` | 6 | 10 | +4 |
| 10 | `src/streaming_metrics.rs` | 6 | 10 | +4 |
| 11 | `src/tool_calling.rs` | 6 | 10 | +4 |

---

## Phase 5 — 5-test files → 10 (6 files, +30 tests)

| # | File | Before | After | Tests Added |
|---|------|--------|-------|-------------|
| 1 | `src/benchmark.rs` | 5 | 10 | +5 |
| 2 | `src/citations.rs` | 5 | 10 | +5 |
| 3 | `src/export.rs` | 5 | 10 | +5 |
| 4 | `src/model_ensemble.rs` | 5 | 10 | +5 |
| 5 | `src/prometheus_metrics.rs` | 5 | 10 | +5 |
| 6 | `src/routing.rs` | 5 | 10 | +5 |

---

## Commits

| Hash | Phase | Description |
|------|-------|-------------|
| `4a01851` | Phase 1 | 12 modules 9→10 tests (+12) |
| `ed04c4e` | Phase 2 | 4 modules 8→10 tests (+8) |
| `6a57f7b` | Phase 3 | 7 modules 7→10 tests (+21) |
| `d486431` | Phase 4 | 11 modules 6→10 tests (+44) |
| `a9d3980` | Phase 5 | 6 modules 5→10 tests (+30) |
