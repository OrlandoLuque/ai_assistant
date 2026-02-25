# Improvements v11 — Test Depth + Documentation Completeness

> Date: 2026-02-24
> Tests: 5,156 (up from 5,091)
> Clippy warnings: 0
> Feature flags: 31 (down from 32 — removed ghost `otel` flag)
> Examples: 31 (up from 22)
> Benchmarks: 16

## Summary

v11 addresses test coverage gaps and documentation debt from v10. Weak test spots
(rag_pipeline.rs at 7 tests for 1895 LOC, entities.rs at 7 tests for 1410 LOC,
document_parsing with 0 integration tests for EPUB/DOCX/ODT) get meaningful coverage.
9 features that lacked examples get full demo files. The ghost `otel` feature flag
(declared but never gated) is removed. Documentation (CONCEPTS.md, AGENT_SYSTEM_DESIGN.md,
GUIDE.md, HTML docs) is updated to reflect v10+v11 state.

---

## Phase 1 — Cleanup: otel ghost flag + GUIDE.md numbering

| # | Item | File |
|---|------|------|
| 1.1 | Remove `otel = []` from Cargo.toml (zero `cfg(feature = "otel")` gates anywhere) | `Cargo.toml` |
| 1.2 | Renumber v10 GUIDE.md sections 56-60 → 61-65 (v9 already used 56-60) | `docs/GUIDE.md` |

## Phase 2 — rag_pipeline.rs tests (7 → 36)

Added 29 new tests covering:

| # | Test Area | Tests |
|---|-----------|-------|
| 2.1 | Pipeline config variants (from_rag_config, for_tier, with_config) | 4 |
| 2.2 | check_requirements (missing, satisfied, partial, empty) | 4 |
| 2.3 | Query processing (analysis, expansion, HyDE, empty query) | 5 |
| 2.4 | Post-processing (assembly, fusion, truncation, max chunks) | 4 |
| 2.5 | Stats tracking (duration, LLM calls, chunk count) | 3 |
| 2.6 | Debug logger integration | 2 |
| 2.7 | Edge cases (long queries, special chars, error display, result debug) | 7 |

File: `src/rag_pipeline.rs` (appended to `mod tests`)

## Phase 3 — entities.rs tests (7 → 32)

Added 25 new tests covering:

| # | Test Area | Tests |
|---|-----------|-------|
| 3.1 | EntityType display_name (all variants, Custom) | 2 |
| 3.2 | Entity metadata and position tracking | 3 |
| 3.3 | Extraction by type (programming languages, file paths, phone, money, percentage) | 5 |
| 3.4 | Edge cases (empty text, whitespace-only, very long input) | 3 |
| 3.5 | FactStore operations (by_subject, by_predicate, goals, top_facts, clear, export) | 6 |
| 3.6 | FactExtractor patterns ("I use X", "I work at X") | 2 |
| 3.7 | Context summary (build_context_summary, empty store) | 2 |
| 3.8 | Fact struct methods (new, display) | 2 |

File: `src/entities.rs` (appended to `mod tests`)

## Phase 4 — Document parsing synthetic ZIP tests

Added 12 new integration tests using in-memory ZIP construction:

| # | Test | What |
|---|------|------|
| 4.1 | test_parse_epub_synthetic | Minimal EPUB ZIP → text + metadata extraction |
| 4.2 | test_parse_epub_metadata | dc: elements → title, author, language |
| 4.3 | test_parse_epub_empty_chapters | Empty chapters → graceful handling |
| 4.4 | test_parse_docx_synthetic | Minimal DOCX ZIP → paragraph text extraction |
| 4.5 | test_parse_docx_metadata | docProps/core.xml → metadata |
| 4.6 | test_parse_docx_no_metadata | Missing docProps → graceful fallback |
| 4.7 | test_parse_odt_synthetic | Minimal ODT ZIP → text extraction |
| 4.8 | test_parse_odt_metadata | meta.xml → metadata |
| 4.9 | test_parse_epub_invalid_zip | Garbage bytes → error |
| 4.10 | test_parse_docx_invalid_zip | Garbage bytes → error |
| 4.11 | test_parse_odt_invalid_zip | Garbage bytes → error |
| 4.12 | (extra) Additional validation tests |

File: `src/document_parsing/tests.rs` (new `zip_tests` submodule, `#[cfg(feature = "documents")]`)

Helper functions: `build_epub()`, `build_docx()`, `build_odt()` — create synthetic ZIP archives
in memory using `zip::write::ZipWriter<Cursor<Vec<u8>>>`.

## Phase 5 — 9 Missing Examples

| # | Example | Feature | Key Types |
|---|---------|---------|-----------|
| 5.1 | `workflow_demo.rs` | `workflows` | WorkflowGraph, WorkflowNode, WorkflowRunner, SimpleEvent |
| 5.2 | `prompt_signature_demo.rs` | `prompt-signatures` | Signature, SignatureField, FieldType, CompiledPrompt, BootstrapFewShot |
| 5.3 | `a2a_demo.rs` | `a2a` | AgentCard, AgentSkill, A2ATask, JsonRpcRequest, AgentDirectory |
| 5.4 | `distillation_demo.rs` | `distillation` | TrajectoryCollector, TrajectoryStep, DatasetBuilder, DatasetConfig |
| 5.5 | `hitl_demo.rs` | `hitl` | ApprovalRequest, CallbackApprovalGate, EscalationEvaluator, PolicyEngine |
| 5.6 | `constrained_demo.rs` | `constrained-decoding` | Grammar, GrammarRule, SchemaToGrammar, GrammarBuilder |
| 5.7 | `voice_agent_demo.rs` | `voice-agent` | VoiceAgent, VoiceAgentConfig, VadConfig |
| 5.8 | `media_gen_demo.rs` | `media-generation` | ImageGenConfig, DallEProvider, VideoGenConfig, RunwayProvider |
| 5.9 | `webrtc_demo.rs` | `webrtc` + `voice-agent` | WebRtcTransport, WebRtcConfig, SdpOffer, WebRtcIceCandidate |

Also added WebRTC type re-exports to `lib.rs` (gated on `#[cfg(all(feature = "webrtc", feature = "voice-agent"))]`).

## Phase 6 — CONCEPTS.md v10 update

Added sections 142-147:

| # | Section | Topic |
|---|---------|-------|
| 142 | Module Splitting | Directory submodules, pub use re-exports |
| 143 | TLS Runtime | server-tls feature, rustls, ReadWrite trait |
| 144 | Plugin System Server Hooks | on_request/on_response/on_event, built-in plugins |
| 145 | OpenAPI Export | Auto-generated spec, /openapi.json route |
| 146 | Server CLI Binary | ai_assistant_server, --dry-run, config file |
| 147 | Criterion Benchmarks | 16 benchmarks, what they measure |

File: `docs/CONCEPTS.md`

## Phase 7 — AGENT_SYSTEM_DESIGN.md v10 update

Added sections 48-51 (in Spanish, matching existing style):

| # | Section | Topic |
|---|---------|-------|
| 48 | Arquitectura de Module Splitting | 4 modules split, directory structure |
| 49 | Arquitectura de Integración TLS | Accept loop, ReadWrite trait, TCP/TLS polymorphism |
| 50 | Arquitectura de WebSocket Chat | Upgrade, frame I/O, chat loop |
| 51 | Arquitectura del Sistema de Plugins | Plugin trait, PluginManager, server hooks |

File: `docs/AGENT_SYSTEM_DESIGN.md`

## Phase 8 — HTML docs + TESTING.md

| # | Item | File |
|---|------|------|
| 8.1 | Update subtitle: v17, 5156 tests, 16 benchmarks, 31 examples, 285 files | `docs/feature_matrix.html` |
| 8.2 | Add v16/v17 backup history entries | `docs/feature_matrix.html` |
| 8.3 | Update test counts, source file counts, example counts | `docs/framework_comparison.html` |
| 8.4 | Add v13/v14 backup history entries | `docs/framework_comparison.html` |
| 8.5 | Update TESTING.md: test count 5156, add v10/v11 rows | `docs/TESTING.md` |
| 8.6 | Create IMPROVEMENTS_V11.md | `docs/IMPROVEMENTS_V11.md` |

Mandatory timestamped HTML backups created before editing:
- `docs/backups/feature_matrix_2026-02-24_2200_pre-v11-update.html`
- `docs/backups/framework_comparison_2026-02-24_2200_pre-v11-update.html`

---

## Metrics

| Metric | v10 | v11 |
|--------|-----|-----|
| Tests | 5,091 | 5,156 (+65) |
| Clippy warnings | 0 | 0 |
| Feature flags | 32 | 31 (-otel ghost) |
| Examples | 22 | 31 (+9) |
| Benchmarks | 16 | 16 |
| Source files | ~250 | 285 |

## Commits

1. `Fix otel ghost feature flag and GUIDE.md duplicate section numbering (56-60 → 61-65)`
2. `Boost rag_pipeline.rs test coverage: 7 → 36 tests`
3. `Boost entities.rs test coverage: 7 → 32 tests`
4. `Add synthetic ZIP tests for EPUB/DOCX/ODT parsing: 12 new tests`
5. `Update CONCEPTS.md (sections 142-147) and AGENT_SYSTEM_DESIGN.md (sections 48-51)`
6. `Add 9 missing examples: workflows, prompt-signatures, a2a, distillation, hitl, constrained-decoding, voice-agent, media-generation, webrtc`
7. `Update HTML docs, TESTING.md for v11; create IMPROVEMENTS_V11.md`

## Verification

```bash
cargo check --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools"
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools" --lib
# Result: 5156 passed; 0 failed; 0 ignored
cargo clippy --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools" -- -W clippy::all
cargo bench --features full --no-run
```
