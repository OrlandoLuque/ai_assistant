# Prompt Fragments — Composable Conditional System Prompts

> Feature flag: `prompt-fragments` — opt-in, not in `full`.

`prompt_fragments` is a library-shaped primitive for assembling system prompts
out of small, conditional pieces. It is the structural equivalent of the ~110
conditional instruction strings in Claude Code's leaked system prompt, but
extensible by the caller rather than hardcoded.

## Contents

1. [What it is (and what it is not)](#what-it-is-and-what-it-is-not)
2. [Why you need it](#why-you-need-it)
3. [Quick start](#quick-start)
4. [API overview](#api-overview)
5. [Built-in catalog](#built-in-catalog)
6. [Presets](#presets)
7. [Writing your own fragments](#writing-your-own-fragments)
8. [Priority and ordering](#priority-and-ordering)
9. [Introspection with `build_with_trace`](#introspection-with-build_with_trace)
10. [Integration points](#integration-points)
11. [Security note: fragment text is trusted](#security-note-fragment-text-is-trusted)

---

## What it is (and what it is not)

**Prompt fragments are instructions.** They tell the model *how* to behave:
which shell conventions apply, whether tools exist, whether plan mode is on,
which compliance notice to include, etc.

**Prompt fragments are not RAG.** RAG (the `rag` feature) answers the
question "what data should I put in the context?". Fragments answer "what
instructions should I give?". Do not confuse them — they solve different
problems with different primitives.

## Why you need it

A static system prompt cannot adapt. Real deployments vary by:

* **Platform:** shell differences (Windows vs POSIX).
* **Mode:** plan vs. execute vs. review vs. autonomous.
* **Tools available:** whether `git`, `retrieve`, `bash`, etc. are callable.
* **Region / compliance:** EU GDPR reminders, US data residency, etc.
* **Domain:** research vs. coding vs. general chat.

Without a composition layer, you either (a) write one massive prompt that
tries to cover everything (wasting tokens and confusing the model), or (b)
maintain N full prompts and pick by handbranching (duplicating text, drifting
between copies). Fragments give you a third option: small units composed at
runtime from signals you already have.

## Quick start

```rust
use ai_assistant::{PromptBuilder, PromptContext, PromptPreset, Platform};

let ctx = PromptContext::default()
    .with_platform(Platform::detect())
    .with_tools(vec!["git".into(), "retrieve".into()])
    .with_locale("en")
    .with_region("EU");

let prompt: String = PromptBuilder::new()
    .with_preset(PromptPreset::CodeDeveloper)
    .add_fragment(ai_assistant::prompt_fragments::catalog::gdpr_eu_notice())
    .build(&ctx);

// Feed `prompt` to your AiAssistant as the base system prompt.
```

## API overview

| Type                | Purpose                                                |
|---------------------|--------------------------------------------------------|
| `PromptBuilder`     | Register fragments and build the final string.         |
| `PromptContext`     | Runtime signals fragments check (platform, tools, …).  |
| `PromptFragment`    | A keyed, categorized, conditional piece of prompt.     |
| `PromptPreset`      | Curated set of fragments for a common agent shape.     |
| `FragmentCategory`  | Safety / ToolGuidance / Context / Style / Mode / …     |
| `Platform`          | Windows / Linux / MacOS / Other — `Platform::detect()`.|
| `AppliedFragment`   | Introspection record from `build_with_trace`.          |

### `PromptBuilder`

```rust
PromptBuilder::new()
    .with_preset(PromptPreset::AgenticLoop)   // seed from preset
    .add_fragment(my_fragment)                 // append / override by key
    .remove_fragment("some_key")               // drop by key
    .build(&ctx)                               // String
```

`build` is infallible. `build_with_trace` returns `(String, Vec<AppliedFragment>)`.

### `PromptContext`

All fields are defaultable; builders fill in what you know:

```rust
PromptContext::default()
    .with_platform(Platform::Linux)
    .with_locale("es")
    .with_region("EU")
    .with_tools(vec!["git".into()])
    .with_custom("experimental", "on");
```

Helpers: `ctx.has_tool("git")`, `ctx.locale_is("es")`.

Under `feature = "autonomous"`, you can also call `.with_mode(OperationMode::Programming)`.

## Built-in catalog

All built-ins live in `ai_assistant::prompt_fragments::catalog::*` and return
a fresh `PromptFragment` per call:

| Key                          | Category          | Fires when…                                  |
|------------------------------|-------------------|----------------------------------------------|
| `windows_shell_note`         | PlatformSpecific  | `platform == Windows`                        |
| `unix_shell_note`            | PlatformSpecific  | `platform in {Linux, MacOS}`                 |
| `tool_use_guidance_general`  | ToolGuidance      | `!tools_available.is_empty()`                |
| `plan_mode_instructions`     | ModeSpecific      | always (caller gates by adding or not)       |
| `execute_mode_instructions`  | ModeSpecific      | always (caller gates by adding or not)       |
| `rag_citation_reminder`      | Context           | a retrieval tool is listed                   |
| `gdpr_eu_notice`             | Safety            | `region.eq_ignore_ascii_case("EU")`          |
| `tdd_workflow`               | Domain            | always                                       |
| `git_commit_conventions`     | Domain            | `has_tool("git")`                            |
| `rust_idioms`                | Domain            | always                                       |
| `academic_citation_style`    | Domain            | always                                       |

The catalog deliberately stays small. Extend it with your own fragments
rather than waiting for upstream additions.

## Presets

| Preset            | Fragments included                                                                          |
|-------------------|---------------------------------------------------------------------------------------------|
| `Minimal`         | *(empty)*                                                                                   |
| `ToolUseChatbot`  | `tool_use_guidance_general`                                                                 |
| `RagAssistant`    | `tool_use_guidance_general`, `rag_citation_reminder`                                        |
| `AgenticLoop`     | `tool_use_guidance_general`, `execute_mode_instructions`                                    |
| `ResearchAgent`   | `tool_use_guidance_general`, `rag_citation_reminder`, `academic_citation_style`             |
| `CodeDeveloper`   | `tool_use_guidance_general`, `tdd_workflow`, `git_commit_conventions`, `rust_idioms`        |

Presets are curated seeds. Mix freely: call `.with_preset(...)` and then
`.add_fragment(...)` for everything the preset leaves out. Adding a fragment
with the same key as one the preset installed **replaces** it.

## Writing your own fragments

```rust
use ai_assistant::{PromptFragment, FragmentCategory, PromptContext};

let frag = PromptFragment::new(
    "company_style_guide",                          // unique key
    "Use British English. Avoid passive voice.",    // text
    FragmentCategory::Style,                        // category
    30,                                             // priority
    |ctx: &PromptContext| {                         // condition
        ctx.custom.get("style").map(|v| v == "british").unwrap_or(false)
    },
);
```

Or for unconditional fragments:

```rust
PromptFragment::always("k", "text", FragmentCategory::Style, 30);
```

## Priority and ordering

Lower priority ⇒ appears first. Ties break by insertion order. Suggested
ranges, so fragments from different sources interleave predictably:

| Range | Category          |
|-------|-------------------|
| 0–9   | Safety            |
| 10–19 | ToolGuidance      |
| 20–29 | Context           |
| 30–39 | Style             |
| 40–49 | ModeSpecific      |
| 50–59 | PlatformSpecific  |
| 100+  | Domain / Custom   |

These are conventions, not enforced — but if you deviate, expect the output
order to surprise other readers of your code.

## Introspection with `build_with_trace`

```rust
let (prompt, trace) = builder.build_with_trace(&ctx);
for applied in &trace {
    eprintln!("{} [{}] prio={}", applied.key, applied.category, applied.priority);
}
```

Each `AppliedFragment` records the key, category, and priority in the exact
order it appears in the output. Useful for:

* Debugging why a fragment did / did not fire.
* Logging which instructions were active when the model answered.
* Emitting a span attribute `fragments_applied=[…]` in OpenTelemetry.

## Integration points

### With `AiAssistant`

```rust
let prompt = PromptBuilder::new().with_preset(PromptPreset::AgenticLoop).build(&ctx);
let assistant = AiAssistant::with_system_prompt(&prompt);
```

### Under `feature = "autonomous"` (OperationMode)

`PromptContext::with_mode(OperationMode)` is available. Write fragments that
branch on it:

```rust
PromptFragment::new(
    "autonomous_guardrails",
    "You are running in autonomous mode. Never escalate privileges without approval.",
    FragmentCategory::Safety,
    3,
    |ctx| ctx.mode == Some(OperationMode::Autonomous),
);
```

### Butler recommendation (Phase 3 — shipped)

`Butler::recommend_prompt_fragments(intent, &report)` turns a natural-language
intent plus a scanned environment into a seed `PromptPreset`, a list of
overlay fragment keys, and a human-readable justification.

```rust
use ai_assistant::butler::Butler;

let mut butler = Butler::new();
let report = butler.scan();
let rec = butler.recommend_prompt_fragments(
    "help me refactor this Rust codebase and add tests",
    &report,
);
println!("{}", rec.justification);
// rec.preset is the seed; iterate rec.extra_fragment_keys and look them up
// in prompt_fragments::catalog::* to overlay on top of the preset.
```

Command-line equivalent (requires `butler` + `prompt-fragments` features):

```bash
ai_cli butler recommend-prompt \
    --intent "help me refactor this Rust codebase and add tests"
```

Keyword dispatch picks the preset (research / code / RAG / autonomous / chat),
with a project-type fallback. Extras overlay on top: `git_commit_conventions`
when a VCS is detected, `rust_idioms` for Rust projects, and the platform
shell notes (the catalog fragment self-gates by host OS). Dedup is handled —
the recommender will not add an extra that the chosen preset already includes.

An LLM-assisted variant is deferred to a follow-up behind a separate feature
flag; the rule-based path already covers the intended shape.

## Security note: fragment text is trusted

Fragment `text` is concatenated **verbatim** into the system prompt. It must
come from trusted sources — your own code, curated config you control.

**Never** build a fragment directly from end-user input. Doing so creates a
prompt-injection vector: a malicious user could supply text that overrides
your safety fragments or leaks instructions.

If you need to surface *user-facing* content into the prompt (e.g. the user's
name, preferences), either:

* sanitize and clamp length before constructing the fragment, or
* put it in `PromptContext.custom` and let a fragment read that key
  explicitly — that way the fragment's trusted text controls the framing and
  the untrusted value is just a parameter.

## Feature gating & deps

* Feature: `prompt-fragments` — opt-in, zero new dependencies.
* Compiles clean without the feature; disabled callers pay nothing.
* Interacts optionally with `autonomous` (reuses `OperationMode`).

## Version

Introduced in v0.2.23. See `docs/IMPROVEMENTS_V91.md` for the design rationale
and status across phases (module, docs, butler integration).
