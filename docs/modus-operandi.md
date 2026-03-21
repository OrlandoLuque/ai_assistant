# Modus Operandi — ai_assistant (project-specific addendum)

This file extends the general modus operandi in Claude Code memory.
It contains ONLY project-specific rules, state, and patterns.

## How to continue after a session saturates

When starting a new session, say:
```
Continúa con el desarrollo del proyecto. Lee docs/modus-operandi.md y el último docs/IMPROVEMENTS_V*.md para saber dónde estamos.
```

## Current state (updated 2026-03-15)

- **Roadmap v1–v10**: ALL COMPLETE
- **Roadmap v11–v37**: ALL COMPLETE (v35 partial — Blocks E+G1+I done, B-D-F-G2-G3-H pending)
- **V38**: COMPLETE — Resilience Engineering (Bulkhead, Adaptive Timeouts, Load Shedding, Chaos Engineering, Enhanced DLQ, WS/SSE Auto-Reconnect)
- **V39**: COMPLETE — API Stability Hardening (#[non_exhaustive] on 454 enums + 246 structs, Default/new() constructors, trait stability docs)
- **V40**: COMPLETE — Testing & Debugging Infrastructure (harness scoring, 15 precision tests, 9 CLI flags, regression detection, REPL /test /bench /precision commands)
- **V41**: COMPLETE — Graph Quality Testing (21 KG structural + 16 multi-layer + 10 agent graph + 8 precision = 55 tests across 3 new categories)
- **V42**: IN PROGRESS — Consolidation & Production Readiness (thinking analysis of 6,730 blocks across 40+ sessions, 17/36 items verified already resolved, SmartChunker bounds + MockHttpServer fixes done, 22 items pending)
- **V43**: COMPLETE — FreshContext Total Integration + Fallback Chains (KG in core, overflow truncation, ReferenceResolver bilingüe EN+ES, auto-list tracking, guardrail catch_unwind, SQLite WAL mode, 18 fallback tests, UnifiedDb persistence + schema versioning + write-through sessions + memory store, DiskSpillBuffer, session recovery, 8 conversation quality scored tests)
- **V44**: COMPLETE — Procedural Memory Integration (ProceduralStore wired into conversation pipeline, WORKFLOW GUIDELINES injection, outcome tracking, CRUD API, 10 tests)
- **V45**: COMPLETE — Procedure Import/Export/Defaults + Multi-User Isolation (6 builtin procedures, versioned ProcedureExport, merge/replace import, UserScope classification, per-user session filtering, concepts 192-193)
- **V46**: ai_assistant_core v0.2.0 — Provider Service Mode (serve as OpenAI-compatible proxy, NAT traversal STUN/UPnP/NAT-PMP, ai_serve binary, published to crates.io)
- **V47**: COMPLETE — Distributed Log Correlation (TraceContext propagation, LogCollector buffer+merge, DistributedLogEntry, export JSON/Text/CSV, concepts 194-195, 13 tests)
- **V48**: COMPLETE — Cache Policies Audit (DHT max_entries/bytes/LRU/pinned/invalidation, EntityStore max 5K, Context cache cap 500, SearchCache/ResponseCache FIFO→LRU, EmbeddingCache memory limit, CompressedCache max 5K)
- **V49**: COMPLETE — Distributed Systems Hardening (DhtValue version auto-increment, replica tracking, NodeCapabilities catalog, FailureClassification temp/permanent, hinted handoff wired, reputation routing, NAT traversal integration)
- **V50**: COMPLETE — Distributed MapReduce (MapWorkerRegistry for closure serialization, execute_distributed_with_results, local+remote chunk splitting, self-as-worker, fallback to local, concept 198, 6 tests)
- **Latest**: V50 — Distributed MapReduce (COMPLETE)
- **Test count**: 6,920 lib tests (0 failures) + 39 harness precision + 18 fallback resilience + 8 conversation quality scored tests
- **Source files**: 320 .rs files, ~390K LOC
- **Feature flags**: 55 (+1: chaos-testing)
- **Status**: Experimental — compiles and passes tests, but not validated in production
- **Website**: Separated to `ai_assistant-website` repo (GitHub Pages ready)
- **License**: PolyForm Noncommercial 1.0.0
- **Domain**: ai-assistant.runawaybrains.com (CNAME configured)

## Planning methodology

When creating an implementation plan, follow this iterative process:

1. **Draft initial plan** — outline phases, components, dependencies
2. **Iterate looking for problems** — in each iteration, review the plan searching for:
   - **Gaps**: missing functionality, unhandled scenarios, things left half-done
   - **Stubs/TODOs**: nothing left for "later" — everything fully implemented
   - **Wiring**: every new type/function properly connected to lib.rs, assistant.rs, or wherever it needs to be used
   - **Edge cases**: empty inputs, huge inputs, concurrent access, Unicode, overflow
   - **Memory saturation**: buffers with limits, eviction strategies, no unbounded growth
   - **Fallbacks**: what happens when X fails? Every failure path has a recovery strategy
   - **Tests**: unit tests for every component, plus realistic integration tests
   - **Crash recovery**: corrupted state, partial writes, unexpected shutdowns
3. **Repeat** until the gain from one more iteration is minimal
4. **Report gains per iteration** — before implementing, show what each iteration found/fixed:
   ```
   Iteration 1: +3 edge cases, +2 missing fallbacks, +1 gap in wiring
   Iteration 2: +1 memory limit missing, +1 test gap
   Iteration 3: minor naming tweak only — diminishing returns, plan is solid
   ```

### Validation checklist (before marking anything DONE)

- [ ] Zero compile errors (`cargo check --features "full"`)
- [ ] Zero clippy warnings on new code
- [ ] All new types/functions wired into lib.rs with proper re-exports
- [ ] No name collisions with existing exports (check with `as` aliases if needed)
- [ ] No stubs, no TODOs, no "implement later" — everything complete
- [ ] Unit tests for all new functionality
- [ ] Realistic/integration tests where applicable
- [ ] Edge cases handled (empty, huge, concurrent, Unicode, overflow)
- [ ] Buffers/collections have size limits and eviction
- [ ] Failure paths have fallbacks or graceful degradation
- [ ] Crash/corruption recovery where state is persisted
- [ ] Full test suite passes (`cargo test --features "full,..." --lib`)
- [ ] Documentation updated (IMPROVEMENTS, modus-operandi, TESTING)
- [ ] Commit after each completed phase (not batched at end)

## Project-specific patterns

### lib.rs module organization
Modules are organized by feature gate. Core modules are always available, optional modules
behind `#[cfg(feature = "...")]`. See lib.rs header comments for the full list.

### Feature flag rules
- `dep:X` prefix: if ANY feature uses `dep:X`, ALL must use `dep:X` (never mix)
- `full` feature includes lightweight features only
- Heavy features (`distributed-network`, `autonomous`, `p2p`, `containers`, `audio`, etc.) are opt-in
- See README.md for the complete feature flags table

### Async pattern
Uses `Pin<Box<dyn Future<Output = T> + Send + '_>>` — NOT `async-trait` crate.

### Name collision pattern
When re-exporting types that conflict across modules, use `as` aliases in lib.rs:
```rust
pub use module_a::Foo as ModuleAFoo;
pub use module_b::Foo as ModuleBFoo;
```
See MEMORY.md for the full list of resolved name collisions.

### Wiring checklist (after implementing a new module or type)
1. Add `pub mod <new_module>;` in lib.rs under the correct feature gate section
2. Add `pub use <new_module>::{Type1, Type2, ...};` re-exports (check for name collisions)
3. **Review ALL existing code that could use the new functionality** — don't just export it, wire it into every place where it's useful:
   - `assistant.rs` — if it affects the main user-facing API
   - `server.rs` / `server_axum.rs` — if it should be exposed via HTTP
   - `mcp_protocol/` — if it should be an MCP tool
   - GUI modules — if it has user-visible state
   - Other modules that handle related concerns (e.g., new persistence → wire into session management, new fallback → wire into error paths)
4. Run compile check
5. Run tests
6. Run clippy

### Documentation files to update per phase
- `docs/GUIDE.md` — add numbered sections at end
- `docs/AGENT_SYSTEM_DESIGN.md` — add numbered sections at end
- `docs/TESTING.md` — update test count
- `docs/CONCEPTS.md` — add concept explanations
- `docs/IMPROVEMENTS_V*.md` — mark items HECHO/PARCIAL
- **HTML docs** are in separate `ai_assistant-website` repo

## Test commands

```bash
# Standard full test (most features)
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite,chaos-testing" --lib

# With distributed network
cargo test --features "full,distributed-network" --lib

# P2P only
cargo test --features "full,p2p" --lib -- p2p::

# Quick check (lightweight features only)
cargo test --features full --lib

# --- Harness (V40 flags) ---

# Run harness with verbose output
cargo run --bin ai_test_harness -- --all --verbose

# Filter harness tests by name
cargo run --bin ai_test_harness -- --all --filter="security"

# Run precision tests only
cargo run --bin ai_test_harness -- --category=precision --verbose

# Save baseline and compare
cargo run --bin ai_test_harness -- --all --save-baseline baseline.json
cargo run --bin ai_test_harness -- --all --diff baseline.json

# Sort by duration, summary only
cargo run --bin ai_test_harness -- --all --sort=duration --summary-only

# V41: Graph quality categories
cargo run --bin ai_test_harness --features full -- --category=graph_quality --verbose
cargo run --bin ai_test_harness --features full -- --category=multi_layer_graph --verbose
cargo run --bin ai_test_harness --features full -- --category=agent_graph_quality --verbose
```

## Build check

```bash
# Lightweight features
cargo check --features full

# All features
cargo check --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network,containers,audio"
```

## What's next

- **v35 remaining blocks**: B (source renames), C (container abstraction), D (tool framework consolidation), F (naming edge cases), G2-G3 (memory extensions), H (MCP agent tools)
- **Harness exports**: JUnit XML / TAP output for CI integration
- **Coverage integration**: `cargo-llvm-cov` or `grcov` with harness coverage
- **Comprehensive review**: API consistency, dead code, documentation polish
- **GitHub publication**: repos ready, domain configured
- **PI registration**: Spain (cultura.gob.es), WIPO PROOF, Safe Creative — PENDING

### Backlog (tareas pendientes no urgentes)

- **Web search en GUI**: Cablear web_search.rs al GUI para que el asistente pueda buscar en internet.
  Ya implementados en web_search.rs: DuckDuckGo (scraping), Brave Search (API), SearXNG (self-hosted).
  Pendiente de implementar como SearchProvider: SerpAPI, Tavily, Google Custom Search API, Bing Web Search API.
  Requiere: UI para configurar el endpoint/API key, integrar resultados como contexto RAG antes de enviar al LLM.
- **Build release + GitHub Release**: Ejecutar `scripts/build_release.ps1`, subir el zip a GitHub Releases, commit + push de toda la documentación Getting Started.

## File map (key files)

| File | Purpose |
|------|---------|
| `src/lib.rs` | Module declarations + re-exports |
| `src/assistant.rs` | AiAssistant — main user-facing struct |
| `src/config.rs` | AiProvider enum + AiConfig |
| `src/providers.rs` | Provider routing (generate_response, streaming) |
| `src/server.rs` | Embedded HTTP server (OpenAI-compatible) |
| `src/server_axum.rs` | Axum-based server (standalone/cluster) |
| `src/agent_definition.rs` | AgentDefinition — declarative agent config (JSON/TOML) |
| `src/agent_wiring.rs` | AgentPool, definition→runtime wiring, supervisor |
| `src/autonomous_loop.rs` | AutonomousAgent — loop with policy, sandbox, cost tracking |
| `src/multi_agent.rs` | MultiAgentSession, orchestration strategies |
| `src/memory_service.rs` | Background memory service (episodic, entity, plans) |
| `src/rag.rs` | RAG database (SQLite + FTS5) |
| `src/guardrail_pipeline.rs` | Constitutional AI, PII, toxicity, injection detection |
| `src/mcp_protocol/` | MCP server with 40+ tools |
| `src/advanced_memory/` | Entity memory, episodic, consolidation |
| `src/prompt_signature/` | DSPy-style optimizable prompts |
| `src/document_parsing/` | PDF, EPUB, DOCX, HTML, etc. |
| `src/advanced_routing.rs` | Bandit algorithms, NFA/DFA routing |
| `src/distributed_network.rs` | QUIC/TLS 1.3, node security, anti-entropy |
| `Cargo.toml` | Feature flags + dependencies |
| `src/unified_persistence.rs` | UnifiedDb, SqliteSessionStore, SqliteMemoryStore |
| `docs/IMPROVEMENTS_V43.md` | V43 — FreshContext + Fallback Chains |
| `docs/IMPROVEMENTS_V44.md` | V44 — Procedural Memory Integration |
| `docs/IMPROVEMENTS_V48.md` | V48 — Cache Policies Audit |
| `docs/IMPROVEMENTS_V49.md` | V49 — Distributed Systems Hardening |
| `docs/IMPROVEMENTS_V50.md` | Latest roadmap — Distributed MapReduce |
| `README.md` | Project overview for GitHub |

## Repository structure

| Repo | Content |
|------|---------|
| `ai_assistant` (this) | Rust crate source code + dev docs (full project, PolyForm Noncommercial) |
| `ai_assistant_core` | **Separate crate** published on crates.io as "anzuelo" — lightweight LLM client (MIT/Apache-2.0). Path: `_varios/_dev/ai_assistant_core` |
| `ai_assistant-website` | Landing page + interactive HTML docs (GitHub Pages) |
