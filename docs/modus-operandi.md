# Modus Operandi — ai_assistant (project-specific addendum)

This file extends the general modus operandi in Claude Code memory.
It contains ONLY project-specific rules, state, and patterns.

## How to continue after a session saturates

When starting a new session, say:
```
Continúa con el desarrollo del proyecto. Lee docs/modus-operandi.md y el último docs/IMPROVEMENTS_V*.md para saber dónde estamos.
```

## Current state (updated 2026-03-14)

- **Roadmap v1–v10**: ALL COMPLETE
- **Roadmap v11–v37**: ALL COMPLETE (v35 partial — Blocks E+G1+I done, B-D-F-G2-G3-H pending)
- **Latest**: V37 — FreshContext mode, MCP knowledge tools, memory integration, advisor API
- **Test count**: 4,892+ lib tests (0 failures)
- **Source files**: 315 .rs files, ~385K LOC
- **Feature flags**: 22+ (see README.md for full table)
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

### Wiring checklist (after implementing a new module)
1. Add `pub mod <new_module>;` in lib.rs under the correct feature gate section
2. Add `pub use <new_module>::{Type1, Type2, ...};` re-exports
3. Run compile check
4. Run tests
5. Run clippy

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
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite" --lib

# With distributed network
cargo test --features "full,distributed-network" --lib

# P2P only
cargo test --features "full,p2p" --lib -- p2p::

# Quick check (lightweight features only)
cargo test --features full --lib
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
| `docs/IMPROVEMENTS_V35.md` | Current roadmap (source of truth) |
| `README.md` | Project overview for GitHub |

## Repository structure

| Repo | Content |
|------|---------|
| `ai_assistant` (this) | Rust crate source code + dev docs |
| `ai_assistant-website` | Landing page + interactive HTML docs (GitHub Pages) |
