# IMPROVEMENTS_V167 — split `assistant.rs` into `impl` submodules

> **This is the last file of the series.** It was written in March 2026 at
> 0.2.119. Everything after it — more than 120 versions — lives in
> [`../CHANGELOG.md`](../CHANGELOG.md), which is the current state of the project.
>
> The series is closed on purpose rather than left to trail off. Its failure mode
> was specific and it bit twice: someone looking for "where are we?" opens the
> highest-numbered `IMPROVEMENTS_V*.md`, gets a coherent, well-written snapshot of
> March, and treats it as today. A document does not have to be wrong to mislead —
> it only has to be findable and undated.
>
> **Where to look instead**
>
> | Question | File |
> |---|---|
> | What changed, and why, version by version | `CHANGELOG.md` (newest first) |
> | How we work — workflow, quality gates, checklist | `docs/modus-operandi.md` |
> | What a subsystem does today | its own `docs/GUIDE_*.md` |
> | Model measurements, with date and backend | `docs/MODEL_BENCHMARKS.md` |
>
> These files (V1–V167) are kept as a record of how the codebase got here. Read
> them as history. Do not read them as state.

**Version:** 0.2.118 → 0.2.119
**Scope:** `src/assistant.rs` → `src/assistant/`
**Feature:** none new

## Why

The last god file flagged by the audit: `assistant.rs` (8,535 lines), the
central `AiAssistant` facade — one enormous `impl AiAssistant` block
divided by `// === Section ===` comments into ~30 concern groups. V163/V166
split the two largest files; V167 finishes the set with the central
object.

## What changed

Pure reorganization — no logic, behavior, or signature change (the V162
`AiResult` return types are untouched). `assistant.rs` became a directory
module of 10 files, each concern group moved into a submodule that carries
its own `impl AiAssistant { ... }`:

| File | ~lines | Concern |
|------|-------:|---------|
| `mod.rs` | 3546 | `AiAssistant` struct + `Default`, helper types (`DocumentInfo`/`FreshContextWarning`/`ContextBudgetStatus`), `new()`/`with_*` constructors, the 4 fallback/summary free fns, `mod` plumbing, and the full `#[cfg(test)] mod tests` |
| `rag.rs` | 1478 | RAG support, global/session notes, knowledge export/import, KPKG→graph (all `#[cfg(feature="rag")]`) |
| `messaging.rs` | 972 | message handling + cancellable streaming |
| `integrations.rs` | 683 | autonomous, butler, scheduler, browser, distributed agents, A/B testing, cost dashboard, chat hooks, MCP client, distillation |
| `context.rs` | 676 | adaptive thinking, dynamic context size, knowledge context, FreshContext advisor, context mgmt, summarization |
| `memory.rs` | 536 | memory + procedural-memory integration |
| `execution.rs` | 230 | container execution, document creation, speech, voice cloning (`cfg(containers|audio)`) |
| `conversation.rs` | 201 | conversation/session/notes mgmt, compaction |
| `models.rs` | 170 | model discovery, provider fallback, API-key mgmt |
| `metrics.rs` | 82 | metrics |

### Why this is safe
- Submodules are **descendants** of the `assistant` module where the
  struct's private fields live, so each `impl AiAssistant` in a submodule
  can read `self.<private_field>` with **no field-visibility change**.
- Only **5** private helper methods called across section boundaries were
  widened to `pub(crate)` (`apply_adaptive_thinking`,
  `maybe_compact_conversation`, `classify_intent_for_budget`,
  `ensure_rag_initialized`, `build_procedural_context`). No public
  signature changed.
- The `#[cfg(test)] mod tests` block is **byte-identical** to before
  (174 `#[test]`), kept in `mod.rs`.
- Two fully-feature-scoped submodules are gated at the `mod` declaration
  (`#[cfg(feature="rag")] mod rag;`, `#[cfg(any(containers,audio))] mod
  execution;`) to avoid empty-module warnings, without changing which
  methods compile in any combo.
- `lib.rs` is **untouched** — `mod assistant;` and every
  `pub use assistant::{...}` re-export resolve identically.

## Tests

Verified: rustfmt clean; clippy `-D warnings` clean (lib + bins,
`FEATURES_STD`, toolchain 1.93.0 = CI); `cargo test --lib -- assistant`
unchanged; `ai_test_harness --all` 585/585.

## Audit follow-ups — status

All three god files the audit flagged are now split: `advanced_routing`
(V163), `ai_test_harness` (V166), `assistant` (V167). The remaining
optional item is the public-API-surface redesign (prelude / re-export
collisions), which is a breaking change left for explicit sign-off.
