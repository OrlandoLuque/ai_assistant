# IMPROVEMENTS V91 — Composable prompt fragments (0.2.23)

## Context

Claude Code's leaked system prompt revealed a composition pattern: ~110
conditional instruction strings, each gated on a runtime signal (platform,
mode, attached IDE, tool availability, plan mode, etc.). The value of that
pattern is **not the specific 110 strings** — those are product decisions —
but the **architecture**: modular, introspectable, override-friendly prompt
assembly.

V91 ports the pattern as a library-shaped primitive. Instead of hardcoding
strings that only fit one product, we provide a composable builder and a small
curated catalog. Callers compose presets, add their own fragments, and swap
defaults without touching the crate.

## Scope — Phase 1 (Rust module)

* New module `src/prompt_fragments.rs` behind feature flag `prompt-fragments`.
* Public API: `PromptBuilder`, `PromptContext`, `PromptFragment`,
  `PromptPreset`, `FragmentCategory`, `Platform`, `AppliedFragment`.
* Built-in catalog (`prompt_fragments::catalog`): 11 fragments covering
  shell notes (Win/Unix), tool-use guidance, plan/execute mode, RAG citation,
  GDPR-EU notice, TDD, git commit conventions, Rust idioms, academic citations.
* Six curated presets: `Minimal`, `ToolUseChatbot`, `RagAssistant`,
  `AgenticLoop`, `ResearchAgent`, `CodeDeveloper`.
* Example: `examples/prompt_fragments.rs` with 4 scenarios.
* 23 unit tests, all pass. Zero clippy warnings in the new module.

Phase 2 (docs) and Phase 3 (butler integration) were completed in the same
patch — see "Status" below.

## Why not just copy the 110 strings?

* They are **product-specific** to Claude Code (hardcoded tool names, IDE
  attachments, plan-mode semantics of their CLI). Porting them verbatim would
  ship dead instructions to every caller.
* We ship the **structural** benefit: priority ranges, category taxonomy,
  introspection via `build_with_trace`, override-by-key semantics. Callers
  own the content decisions.

## Design notes

### Priority ranges (conventional)

| Range    | Category         |
|----------|------------------|
| 0–9      | Safety           |
| 10–19    | ToolGuidance     |
| 20–29    | Context          |
| 30–39    | Style            |
| 40–49    | ModeSpecific     |
| 50–59    | PlatformSpecific |
| 100+     | Domain / Custom  |

Low priority = appears first. Ties break by insertion order.

### Fragment condition

`applies` is an `Arc<dyn Fn(&PromptContext) -> bool + Send + Sync + 'static>`.
`Arc` keeps fragments cheap to clone and safe to share across threads, and the
trait bounds let callers capture state from configuration structs they own.

### Static text in v1

Fragment `text` is a `String` resolved at construction time. Dynamic
interpolation (e.g. injecting `ctx.locale`) is deferred to a follow-up phase;
for v1 the caller can include locale variants as separate fragments or compose
them upstream.

### Trusted input

Fragment text is concatenated verbatim into the system prompt. It must come
from trusted sources — **never** from end-user input, or you create a
prompt-injection vector. Documented at the module level.

### No RAG confusion

Fragments are **instructions** (how the model should behave), not knowledge
(what data the model should know). Retrieval stays in the `rag` feature.

## Quick start

```rust
use ai_assistant::{PromptBuilder, PromptContext, PromptPreset, Platform};

let ctx = PromptContext::default()
    .with_platform(Platform::detect())
    .with_tools(vec!["git".into(), "retrieve".into()])
    .with_region("EU");

let prompt = PromptBuilder::new()
    .with_preset(PromptPreset::CodeDeveloper)
    .add_fragment(ai_assistant::prompt_fragments::catalog::gdpr_eu_notice())
    .build(&ctx);
```

## Testing

* 23 unit tests in `src/prompt_fragments.rs` (builder, fragments, presets,
  overrides, ordering, trace, catalog conditions).
* Example runs clean: `cargo run --example prompt_fragments --features prompt-fragments`.

## Feature gating

Everything lives behind `feature = "prompt-fragments"`. Not included in
`full`, opt-in only. When disabled the module does not compile; callers
without the feature are unaffected.

## Version bump

`0.2.22 → 0.2.23` (patch-level; additive, no API breakage, no new deps).

## Status

* **Phase 1 — Rust module.** Done. Module, catalog, presets, builder, tests,
  example.
* **Phase 2 — Documentation.** Done. `docs/PROMPT_FRAGMENTS.md`, website page
  `prompt_fragments.html`, `feature_matrix.html` row, cross-links from
  `index.html`, `product_overview.html`, `ai_assistant_overview.html`,
  `guide_anti_hallucination.html`, `guide_research.html`.
* **Phase 3 — Butler integration.** Done.
  * `Butler::recommend_prompt_fragments(intent, &report) -> PromptRecommendation`
    returns a seed preset, a list of overlay fragment keys, and a human-readable
    justification string.
  * Rule-based: keyword dispatch (research / code / rag / autonomous / chat)
    with project-type fallback; overlays `git_commit_conventions` when a VCS is
    detected, `rust_idioms` when the project is Rust, and the platform shell
    notes (self-gated by the fragment).
  * CLI subcommand: `ai_cli butler recommend-prompt --intent "<description>"`.
  * 10 unit tests in `butler::tests::prompt_fragments_tests`.
  * LLM-assisted variant deferred — same shape, plugged in behind a separate
    feature flag.
