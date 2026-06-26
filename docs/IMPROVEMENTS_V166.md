# IMPROVEMENTS_V166 — split `ai_test_harness` into a module

**Version:** 0.2.117 → 0.2.118
**Scope:** `src/bin/ai_test_harness.rs` → `src/bin/ai_test_harness/` + `Cargo.toml`
**Feature:** none new

## Why

The second-largest god file flagged by the audit was the test-harness
binary `src/bin/ai_test_harness.rs` (16,627 lines, 193 fns) — an
unstructured single-file binary. V163 split the largest lib file
(`advanced_routing`); V166 does the same for the harness. It is test
infrastructure (not shipped library code), so the risk is low and the
585-test battery verifies it directly.

## What changed

Pure reorganization — no logic/behavior change. The single file became a
directory binary `src/bin/ai_test_harness/` of 16 files:

| File | ~lines | Contents |
|------|-------:|----------|
| `main.rs` | 1672 | `fn main`, CLI parsing, report types (`TestResult`/`CategoryResult`/`HarnessReport`), runner (`run_test`/`run_test_scored`/`all_categories`), JUnit/TAP/diff writers, `mod` plumbing |
| `macros.rs` | 25 | `assert_eq_test!` / `assert_test!` |
| `basics.rs` | 1165 | core / session / context / security / formatting / streaming / tools / … |
| `features.rs` | 1970 | decision-trees … agent-memory |
| `features2.rs` | 1204 | api-key-rotation … keepalive |
| `chains.rs` | 1214 | integration + chain tests |
| `pipelines.rs` | 801 | pipeline + guardrail tests |
| `rag_graph.rs` | 2366 | rag tiers, knowledge graph, graph quality, multi-layer graph |
| `resilience.rs` | 771 | fallback resilience, conversation quality |
| `stress.rs` | 2140 | all `stress_*` categories |
| `precision.rs` | 1807 | precision suite |
| `eval.rs` | 159 | anti-hallucination, quality-gates, faithfulness, verification, research |
| `p2p.rs` / `containers.rs` | 278 / 526 | `cfg(p2p)` / `cfg(containers)` categories |
| `replay.rs` / `replay_stub.rs` | 551 / 15 | `cfg(rag)` / `cfg(not rag)` replay module |

### Cargo.toml
One line: `[[bin]] ai_test_harness` `path` →
`src/bin/ai_test_harness/main.rs` (`name`, `required-features`, `bench`
unchanged).

### Mechanics
- ~140 `tests_*` category fns became `pub(crate)` so `main.rs` can glob
  them; shared runner helpers stayed private in `main.rs` and submodules
  reach them via `use super::*` (descendant access).
- Both `mod replay` blocks relocated with their exact cfg gates
  (`#[path]` + `#[cfg(feature="rag")]` / `cfg(not)`); the `containers`/
  `p2p` module decls are cfg-gated so they don't warn when off.

## Tests

Verified: `cargo build --bin ai_test_harness --features "full,browser"`
clean (zero warnings); `cargo run … -- --all` → **ALL 585 TESTS PASSED**
(2 skipped, 1 slow) — identical to baseline; rustfmt clean.

## Still open (optional)

- Split `assistant.rs` (8.5K central facade) into `impl` submodules.
- Redesign the public API surface (prelude / re-export collisions).
