# v23 Improvements Changelog

> Date: 2026-02-28

v23 is a **code hygiene** release: replaces raw `println!` with structured logging, eliminates all `#[allow(unused_assignments)]` annotations, and removes an unused import. No new features or tests.

---

## Summary Metrics

| Metric | v22 | v23 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 5,963 | 5,963 | 0 |
| Clippy warnings | 0 | 0 | 0 |
| `println!` in library code | 15 | 0 | -15 |
| `#[allow(unused_assignments)]` | 4 | 0 | -4 |
| `#[allow(unused_imports)]` | 1 | 0 | -1 |
| Remaining `#[allow()]` annotations | 10 | 5 | -5 |

---

## Phase 1 — Replace `println!` with `log` macros (2 files, 15 calls)

| # | File | Lines | Before | After |
|---|------|-------|--------|-------|
| 1 | `src/agent.rs` | 585, 591, 600 | `println!` (verbose) | `log::debug!` |
| 2 | `src/agent.rs` | 605 | `println!` (completion) | `log::info!` |
| 3 | `src/assistant.rs` | 670 | `println!` (context calc) | `log::debug!` |
| 4 | `src/assistant.rs` | 1069, 1078, 1083, 1090 | `println!` (debug) | `log::debug!` |
| 5 | `src/assistant.rs` | 1074, 1085 | `println!` (warnings) | `log::warn!` |
| 6 | `src/assistant.rs` | 2408, 2593 | `println!` (RAG indexed) | `log::info!` |
| 7 | `src/assistant.rs` | 2410, 2595 | `println!` (RAG skipped) | `log::debug!` |

---

## Phase 2 — Eliminate `#[allow(unused_assignments)]` (4 files)

| # | File | Fix |
|---|------|-----|
| 1 | `src/batch.rs` | Use uninitialized `let mut last_error;` instead of `= String::new()` |
| 2 | `src/cot_parsing.rs` | Use `let answer =` at assignment point instead of pre-initialized `let mut` |
| 3 | `src/pii_detection.rs` | Use `all_detections` directly in return, remove intermediate `detections` variable |
| 4 | `src/task_decomposition.rs` | Return `match` expression directly instead of assigning to pre-initialized variable |

---

## Phase 3 — Remove unused imports + audit remaining annotations

**Removed:**

| File | Annotation | Fix |
|------|-----------|-----|
| `src/p2p.rs` | `#[allow(unused_imports)]` on `Dht, DhtNode, NodeId` | Removed import entirely — types not used in file |

**Kept (justified):**

| File | Annotation | Justification |
|------|-----------|---------------|
| `src/constrained_decoding.rs:810` | `clippy::vec_box` | Box needed for raw pointer stability (documented in comment) |
| `src/constrained_decoding.rs:1491` | `dead_code` | State machine enum — variants defined for completeness |
| `src/cloud_connectors.rs:268` | `clippy::result_large_err` | `ureq::Error` is inherently large; wrapping adds complexity |
| `src/quantization.rs:40` | `non_camel_case_types` | Industry-standard naming (FP32, INT8, Q4_K_M) |
| `src/bin/ai_test_harness.rs:10403` | `unused_mut` | Conditional `push` under `cfg(feature = "p2p")` |

---

## Verification

- `cargo check --features full,...` — clean, 0 warnings
- `cargo test --features full,...` — 5,963 passed, 0 failed
- `cargo clippy --features full,...` — 0 warnings
- `cargo check --features full,p2p` — clean (p2p import removal verified)
