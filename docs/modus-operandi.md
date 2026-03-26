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
- **V51**: COMPLETE — Store Limits Hardening (A2A TaskStore max 10K evict completed-first, consolidation FactStore max 5K evict lowest-confidence, entities FactStore enforce 10K evict lowest-reinforcement, LWWMap optional max_entries evict oldest, push_configs max 5K, 8 tests)
- **V52**: COMPLETE — Consolidation & Production Readiness (HTTP log endpoints, JUnit XML/TAP harness export, Block B+F source renames complete (zero aliases), Block C container backend trait, Block G2 EntityStore generalization (query/embedding/TTL), Block G3 PlanStore persistence, Block H MCP agent management tools, 22 tests)
- **V53**: COMPLETE — Block D audit (deprecated tool_use+function_calling, unified_tools canonical), 4 SearchProviders (Google, Bing, SerpAPI, Tavily), web search in GUI, clippy clean
- **V54**: COMPLETE — Block D final (deleted tool_use.rs+function_calling.rs, -1423 LOC), MemoryManager in both context modes, send_message_with_rag(), diagrama flujo 7 corregido
- **V55**: COMPLETE — Adaptive Context Budget Allocator (ContextItem/ContextSource/ContextBudgetAllocator, score-based allocation, extractive compression RECOMP-style, LlmCompressor trait with domain filtering, StrategyBandit UCB1, RagTierDefinition shareable + TierStore import/export, 27 tests)
- **V56**: COMPLETE — Voice/Audio Enhancement (EmotionDetector trait, EmotionState, KeywordEmotionDetector, ExpressiveOpenAiTtsProvider, ElevenLabsProvider, VoiceCodec trait, empathetic loop wiring, diagram 25)
- **V57**: COMPLETE — Agent Interruption & Safety (ToolSafetyProfile, SnapshotStore cross-platform rollback, RollbackStrategy Snapshot/Git 5 modes, KnowledgeWatcher auto-reindex, ToolCallRecord saga tracking)
- **V58**: COMPLETE — StorageContext unified persistence (save_json/load_json atomic, DirtyFlags, drain_writes(), StrategyBandit + RagTierStore persistence)
- **V59**: COMPLETE — Security Hardening (6 concrete vulnerability fixes, ConfigLock, IntegrityChecker, SecurityAlertManager, LearningFreezeConfig, 256-vector audit)
- **V60**: COMPLETE — Semantic Dedup (3-level with batched fusion), Distributed RAG (DocumentScope, SharedChunkMeta, TTL/refresh), ICE NAT types, P2P Security (TrustLevel, MessageAuthorization, PeerAccessControl), concepts 210-217
- **V61**: COMPLETE — BrowserPolicy (URL validation, JS permission levels, SSRF, 16 dangerous patterns, 14 tool permission categories), concepts 218-219
- **V62**: COMPLETE — Agent Methodology (AgentMethodology, WorkflowProtocol 6 phases, ReviewTriggers 8 conditions, QualityGate, 4 presets, wired into AutonomousAgent), RateLimitStrategy (4 strategies, patient preset), RagFeatures +5 new fields (semantic_dedup_fusion, distributed_search, context_budget_allocation, fresh_context, emotion_aware), cancellation propagation (partial response save, HTTP endpoint cancel, CancelTask/CancelAck P2P, MapReduceJob.cancel(), DistributedRagConfig timeout), concepts 220-221
- **V63**: COMPLETE — MCP Task Tools (8 tools: CRUD + soft-delete rollback + FTS5 search, SQLite migration V5), MCP Home Automation (10 tools: lights/switches/climate/scenes/automations, HomeBackend trait, HomeAssistantBackend, SSRF protection, input validation), concepts 225-226
- **V64**: COMPLETE — Universal Event System (8 source types, EventSourceManager, prompt template rendering), MQTT Backend (Zigbee2MQTT/Tasmota/HA Discovery, DeviceRegistry, rate limiting), OpenHAB Backend (REST API), CoAP Backend (UDP, RFC 7252, retransmission), Custom IoT Devices (ThresholdAlerts, StateSource/CommandTarget), mDNS Discovery, 51 attack vectors identified, concepts 227-233
- **Latest**: V64 — Universal Events + Home Automation Expansion (COMPLETE)
- **Test count**: 7,179 lib tests (0 failures) + 39 harness precision + 18 fallback resilience + 8 conversation quality scored tests
- **Source files**: 340 .rs files, ~389K LOC
- **Feature flags**: 57 (+2: home-automation, coap)
- **Feature flags**: 55 (+1: chaos-testing)
- **Status**: Experimental — compiles and passes tests, but not validated in production
- **Website**: Separated to `ai_assistant-website` repo (GitHub Pages ready)
- **License**: PolyForm Noncommercial 1.0.0
- **Domain**: ai-assistant.runawaybrains.com (CNAME configured)

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

- **Wire ContextBudgetAllocator into send_message()**: replace hardcoded injection with allocator (task #57)
- **LLM-enhancement de pipeline completo**: 17 módulos identificados para mejora opcional (conversation compaction, entity NER, guardrails, auto-model selection, etc.)
- **Actualizar diagramas de flujo**: reflejar el nuevo pipeline con allocator
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
| `src/agent_methodology.rs` | AgentMethodology — workflow, reasoning, review triggers, quality gates |
| `src/autonomous_loop.rs` | AutonomousAgent — loop with policy, sandbox, cost tracking, methodology |
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
| `docs/IMPROVEMENTS_V50.md` | V50 — Distributed MapReduce |
| `docs/IMPROVEMENTS_V52.md` | V52 — Consolidation & Production Readiness |
| `docs/IMPROVEMENTS_V62.md` | V62 — Agent Methodology |
| `README.md` | Project overview for GitHub |

## Repository structure

| Repo | Content |
|------|---------|
| `ai_assistant` (this) | Rust crate source code + dev docs (full project, PolyForm Noncommercial) |
| `ai_assistant_core` | **Separate crate** published on crates.io as "anzuelo" — lightweight LLM client (MIT/Apache-2.0). Path: `_varios/_dev/ai_assistant_core` |
| `ai_assistant-website` | Landing page + interactive HTML docs (GitHub Pages) |
