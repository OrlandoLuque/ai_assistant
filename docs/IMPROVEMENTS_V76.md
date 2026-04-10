# Improvements V76 — Feature Hygiene Pass

## Motivation

During the planning of V77 (docs, helper binaries, use cases) the user asked
why so many modules with hardware or protocol-specific names were compiled
unconditionally. A quick audit of `src/lib.rs` surfaced 15 modules whose name,
dependencies, and call-sites made it obvious they should live behind a feature
flag. Leaving them ungated inflated build times for minimal consumers, forced
unnecessary `dep:` pulls, and blurred the meaning of the 60 Cargo features.

V76 is a **surgical cleanup**: no new functionality, no behavior change for
users that already build with `full` + opt-in features, but a much smaller
surface for users that want to compile only what they need (edge, WASM,
serverless, tests in CI).

## Scope

Following the user instruction "fix first, build on top" (see
`memory/feedback_task_ordering.md`), V76 runs **before** V77 (docs/binaries)
so later work lives on a correctly gated foundation.

Out of scope (deferred to V80): the 64 "ADDITIONAL MODULES" block in
`src/lib.rs` (lines ~1655-1950). Those are lightweight data structures
(`cost`, `memory`, `intent`, `token_budget`, `web_search`, ...) that the
auditor classified as safe ungated and that already contain their own
sub-feature gates.

## Workstream A: Audio cluster → `audio`

| # | Item | File | Estado |
|---|------|------|--------|
| A1 | Gate `audio_filter` behind `audio` (mod + `pub use` block) | lib.rs | HECHO |
| A2 | Gate `audio_model_registry` behind `audio` (mod + `pub use` block) | lib.rs | HECHO |
| A3 | Gate `audio_priority_protocol` behind `audio` (mod decl) | lib.rs | HECHO |
| A4 | Gate `group_queue_host`, `group_queue_runtime` behind `audio` | lib.rs | HECHO |
| A5 | Tighten `mcp_voice_tools` to `all(feature = "tools", feature = "audio")` | lib.rs | HECHO |
| A6 | Tighten `pub use mcp_voice_tools::register_voice_tools` cfg | lib.rs | HECHO |
| A7 | `voice-agent` now implies `audio` in Cargo.toml | Cargo.toml | HECHO |

**Rationale**: `audio_filter` was consumed by `mcp_voice_tools`, `server_axum`
(inside a pre-existing `#[cfg(feature = "audio")]` block), `voice_agent`, and
internally by `group_queue_*`. Making `voice-agent` imply `audio` keeps the
existing voice-agent consumers compiling without adding manual imports.
`mcp_voice_tools` is fundamentally about voice — it always needed both
`tools` and `audio`; the previous gate on `tools` alone was a latent bug.

## Workstream B: GPU-sharing cluster → `gpu-sharing`

| # | Item | File | Estado |
|---|------|------|--------|
| B1 | Gate `gpu_sharing` (mod + `pub use` block) | lib.rs | HECHO |
| B2 | Gate `collusion_detection` (mod + `pub use`) | lib.rs | HECHO |
| B3 | Gate `credit_system` (mod + `pub use`) | lib.rs | HECHO |
| B4 | Gate `dynamic_pricing` (mod + `pub use`) | lib.rs | HECHO |

**Rationale**: the auditor confirmed that `collusion_detection`,
`credit_system` and `dynamic_pricing` are only consumed by `gpu_sharing`
itself (via `crate::X` imports inside `gpu_sharing.rs`). Gating the whole
cluster behind `gpu-sharing` shrinks minimal builds by four files and
removes a trio of modules whose presence without `gpu-sharing` enabled was
dead weight.

## Workstream C: Browser / distributed / video / wasm

| # | Item | File | Estado |
|---|------|------|--------|
| C1 | Gate `browser_policy` + `crawl_policy` behind `browser` | lib.rs | HECHO |
| C2 | Gate `distributed_rag` behind `distributed` | lib.rs | HECHO |
| C3 | Gate `video_filter` behind `video-io` | lib.rs | HECHO |
| C4 | Gate `wasm` + `wasm_hooks` behind `wasm` | lib.rs | HECHO |

All four clusters had zero in-crate consumers beyond their own `pub use`
re-exports, so the cascade risk was LOW.

## Workstream D: Marker feature cleanup

| # | Item | File | Estado |
|---|------|------|--------|
| D1 | Remove `core = []` marker (unused, empty) | Cargo.toml | HECHO |
| D2 | Drop `"core"` from `full = [...]` | Cargo.toml | HECHO |
| D3 | Document `adapters = []` as intentional (labels `adapters_demo` example) | Cargo.toml | HECHO |

`core = []` had zero `#[cfg(feature = "core")]` references in the codebase —
pure noise. `adapters = []` is referenced by `[[example]] adapters_demo` as a
build-time label; removing it would cascade into the example section, so
V76 keeps it with a clarifying comment instead. Full cleanup of marker
features is deferred.

## Files Modified

| File | LOC delta |
|------|-----------|
| `src/lib.rs` | +22 (15 cfg attributes on mod decls + 8 on pub use blocks; 1 widened) |
| `Cargo.toml` | +3 / -3 (voice-agent implies audio; core removed; adapters documented) |
| `CHANGELOG.md` | +15 (v32 entry) |
| `docs/IMPROVEMENTS_V76.md` | new file |
| **Total** | ~45 LOC |

## Verification

```bash
# Full feature set — baseline regression test
cargo check --features "full,autonomous,scheduler,butler,browser,audio-io,\
  gpu-sharing,gui-pro,eval-suite,hitl,webrtc,devtools,home-automation,\
  workflows,advanced-memory,voice-agent"
# → Finished dev profile (pre-existing warnings only)

# Reduced feature set — this is what V76 unlocks
cargo check --lib --features "full,autonomous,scheduler"
# → Finished (previously would compile 15 extra ungated modules)
```

## Security Considerations

None. V76 is a pure compilation-conditionality change. No runtime code was
added, modified, or deleted. Cost tracking, cost projection, and all V75
security mitigations remain intact.

## Deferred to V80 — Audit of ADDITIONAL MODULES

The 64 modules declared ungated in the "ADDITIONAL MODULES" section of
`lib.rs` (lines ~1655-1950) were classified as safe ungated by the V76
auditor. A second-pass audit is scheduled for V80:

- `agent_graph`, `answer_extraction`, `api_key_rotation`, `code_editing`,
  `compute_proof`, `conflict_resolution`, `context_window`,
  `conversation_compaction`, `cost`, `cost_integration`, `dag_executor`,
  `decision_tree`, `distributed_rate_limit`, `fact_verification`,
  `few_shot`, `forecasting`, `health_check`, `i18n`, `intent`, `keepalive`,
  `multimodal_rag`, `openapi_export`, `patch_application`, `pii_tokenizer`,
  `prefetch`, `priority_queue`, `prompt_optimizer`, `quantization`,
  `regeneration`, `reranker`, `request_coalescing`, `request_signing`,
  `response_ranking`, `routing`, `smart_suggestions`, `summarization`,
  `task_planning`, `text_transform`, `token_budget`, `typing_indicator`,
  `ui_hooks`, `user_rate_limit`, `web_search`, `webhooks`, and ~20 more.

Most of them are small (<2k LOC) and have internal feature gates for
their own sub-features (e.g. `advanced_routing` has 21 internal
`#[cfg(...)]`). Gating them would be cosmetic and at least one requires
an ecosystem-wide impact study before committing to a canonical preset.
