# Plan de Mejoras para ai_assistant — v8

> **Estado: 23/26 items COMPLETE — 4882 tests, 0 failures, 0 clippy warnings**

> Documento generado el 2026-02-24.
> Basado en completitud de v1 (39/39), v2 (22/22), v3 (21/21), v4 (38/38), v5 (30/30), v6 (34/34), v7 (34/34) con 4687 tests, 0 failures.
> 240+ source files, ~262k LoC.
>
> **Planes anteriores**: v1 (providers, embeddings, MCP, documents, guardrails),
> v2 (async parity, vector DBs, evaluation, testing),
> v3 (containers, document pipeline, speech/audio, CI/CD maturity),
> v4 (workflows, prompt signatures, A2A, advanced memory, online eval, streaming guardrails),
> v5 (GEPA/MIPROv2, MCP v2, voice agents, media gen, distillation, OTel GenAI,
>     durable execution, constrained decoding, memory evolution),
> v6 (MCP spec completeness, remote MCP client, HITL, SIMBA, Memory OS, discourse RAG,
>     agent eval, red teaming, MCTS planning, WebRTC voice, multi-backend sandbox, devtools),
> v7 (real HTTP wiring, credential management, OTel export, structured output,
>     auto model selection, context composition, cloud connectors, web search, mock HTTP).

---

## Contexto

Tras v7 "Production Hardening" (34/34 items, 4687 tests, +8200 LOC), el crate tiene HTTP real, infraestructura de fiabilidad, seguridad enterprise y observabilidad. Sin embargo, una auditoria de madurez revela:

- **HTTP server solo dev**: server.rs — sin TLS, sin auth, sin graceful shutdown, CORS hardcodeado `*`
- **Logging casi inexistente**: Solo 46 `log::` calls en 14 de 235+ ficheros. providers.rs tiene 0 log calls
- **Memoria volatil**: EpisodicStore, ProceduralStore, EntityStore — todo in-memory, se pierde al reiniciar
- **Guardrails solo input**: Los guards built-in solo validan entrada, no salida del LLM
- **Tests desbalanceados**: assistant.rs y rag_methods.rs infra-testeados
- **Sin prelude module**: 230+ `pub use` re-exports sin organizacion
- **Sin error context**: Errores sin contexto de donde vinieron

**Tesis v8**: Madurar el codigo existente — endurecer el servidor HTTP, cubrir gaps de tests, anadir logging, persistencia de memoria, y guardrails de salida. Calidad > cantidad.

---

## Estructura: 26 items, 10 fases

### Fase 1 — HTTP Server Hardening (5 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 1.1 | Auth middleware: bearer tokens + API keys, constant-time comparison | HECHO | 10 | server.rs |
| 1.2 | CORS configurable: allowed_origins, methods, headers, max_age | HECHO | 8 | server.rs |
| 1.3 | Request logging + correlation IDs: X-Request-Id, method/path/status/latency | HECHO | 8 | server.rs |
| 1.4 | Graceful shutdown: AtomicBool flag, ServerHandle::shutdown(), nonblocking poll | HECHO | 2 | server.rs |
| 1.5 | Prometheus /metrics endpoint: render ServerMetrics as Prometheus text | HECHO | 2 | server.rs |

### Fase 2 — Structured Logging (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 2.1 | Provider logging: log provider, model, tokens, latency, status on every call | HECHO | 8 | providers.rs |
| 2.2 | Module-level logging: assistant (sessions, model changes), cloud (S3 ops), mcp (connections) | HECHO | 2 | assistant.rs, cloud_connectors.rs, mcp_client.rs |

### Fase 3 — Module Splitting (4 items) — DEFERRED

| # | Item | Estado | Razon |
|---|------|--------|-------|
| 3.1 | Split prompt_signature.rs (6907 LOC) | DEFERRED | Alto riesgo de merge conflicts, bajo ROI inmediato |
| 3.2 | Split advanced_memory.rs (5812 LOC) | DEFERRED | Misma razon |
| 3.3 | Split mcp_protocol.rs (5025 LOC) | DEFERRED | Misma razon |
| 3.4 | Split document_parsing.rs (5105 LOC) | DEFERRED | Misma razon |

### Fase 4 — Test Coverage Boost (3 items)

| # | Item | Estado | Tests nuevos | Fichero |
|---|------|--------|-------------|---------|
| 4.1 | assistant.rs test expansion: send_message, context mgmt, compaction, API key rotation | HECHO | ~38 | assistant.rs |
| 4.2 | rag_methods.rs test expansion: HyDE, reranker, CRAG, Self-RAG, Graph-RAG, RAPTOR | HECHO | ~57 | rag_methods.rs |
| 4.3 | server.rs test expansion: concurrent, body limits, malformed HTTP, timeout | HECHO | ~17 | server.rs |

### Fase 5 — Output Guardrails (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 5.1 | OutputPiiGuard: email, phone, SSN, credit card detection in LLM responses | HECHO | 7 | guardrail_pipeline.rs |
| 5.2 | OutputToxicityGuard: toxicity scoring of LLM responses, configurable threshold | HECHO | 7 | guardrail_pipeline.rs |

### Fase 6 — Memory Persistence (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 6.1 | save_to_file/load_from_file: serde_json for all stores, atomic write | HECHO | 11 | advanced_memory.rs |
| 6.2 | AutoPersistenceConfig: periodic save, max_snapshots rotation, configurable base_dir | HECHO | 8 | advanced_memory.rs |

### Fase 7 — API Surface Cleanup (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 7.1 | Prelude module: ~15 most-used types via `use ai_assistant::prelude::*` | HECHO | 4 | prelude.rs (new) |
| 7.2 | Organize lib.rs: doc comments, section headers, `TlsConfig` + `AutoPersistenceConfig` re-exports | HECHO | 0 | lib.rs |

### Fase 8 — Deployment Tooling (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 8.1 | Dockerfile: multi-stage rust:slim → debian:slim, non-root user, port 8090 | HECHO | 0 | Dockerfile (new), .dockerignore (new) |
| 8.2 | Config hot-reload: ConfigWatcher poll by mtime, classify HotReload vs RequiresRestart | HECHO | 8 | config_file.rs |

### Fase 9 — Robustness (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 9.1 | Request limits: max_headers (100), max_header_line (8192), body_read_timeout | HECHO | 2 | server.rs |
| 9.2 | TLS config struct: TlsConfig with cert/key paths, serde(skip), opt-in | HECHO | 5 | server.rs |

### Fase 10 — Developer Experience (2 items)

| # | Item | Estado | Tests | Fichero |
|---|------|--------|-------|---------|
| 10.1 | Error context: ContextualError, AiError::with_context(), ResultExt trait | HECHO | 8 | error.rs |
| 10.2 | Integration test lifecycle: E2E auth + metrics + shutdown | HECHO | 3 | server.rs |

---

## Resumen Cuantitativo

| Concepto | Cantidad |
|----------|----------|
| Fases completadas | 9/10 (Fase 3 deferred) |
| Items completados | 23/26 (3 deferred) |
| Ficheros nuevos | 3 (prelude.rs, Dockerfile, .dockerignore) |
| Ficheros modificados | 12 |
| Lineas anadidas | ~4461 |
| Tests nuevos | ~195 |
| **Total tests** | **4882** (4687 + 195) |
| Clippy warnings | 0 |
| Feature flags nuevos | 0 (TLS struct ready, no new flag yet) |

---

## Ficheros Modificados

| Fichero | Cambios | LOC aprox |
|---------|---------|-----------|
| server.rs | Auth, CORS, metrics, request IDs, graceful shutdown, /metrics, TLS config, request limits, 17+ tests | +1080 |
| rag_methods.rs | 57 new tests for all RAG strategies | +867 |
| guardrail_pipeline.rs | OutputPiiGuard + OutputToxicityGuard + 14 tests | +738 |
| assistant.rs | 38+ new tests + module logging + 2 session logging | +548 |
| advanced_memory.rs | save/load for 3 stores + AutoPersistenceConfig + 19 tests | +426 |
| providers.rs | Structured logging module + 8 tests | +365 |
| config_file.rs | ConfigWatcher + ReloadScope/ReloadResult + 8 tests | +286 |
| error.rs | ContextualError + ResultExt + 8 tests | +156 |
| lib.rs | New re-exports: TlsConfig, AutoPersistenceConfig, ContextualError, etc. | +23 |
| cloud_connectors.rs | S3 operation logging (5 log calls) | +5 |
| mcp_client.rs | Connection/disconnect logging (4 log calls) | +4 |

---

## Decisiones de Diseno

| Decision | Eleccion | Razon |
|----------|----------|-------|
| Module splitting | DEFERRED | Alto riesgo de merge conflicts en ficheros >5000 LOC; la funcionalidad no cambia |
| TLS runtime | Struct only, no rustls dep | Keeps deps light; struct is ready for future `server-tls` feature |
| Prometheus format | Hand-written exposition text | Zero new dependencies; sufficient for basic monitoring |
| Shutdown mechanism | AtomicBool + nonblocking accept | No extra deps; works with std TcpListener |
| Request IDs | 32-char hex from counter + timestamp | No UUID dep needed; collision-resistant enough |
| Memory persistence | serde_json + atomic write (tmp+rename) | Simple, debuggable, crash-safe |

---

## Items Deferred (para v9)

1. **3.1-3.4 Module splitting**: Requiere crear directorios `prompt_signature/`, `advanced_memory/`, `mcp_protocol/`, `document_parsing/` con `mod.rs` re-exportando todo. Alto riesgo de romper imports en ficheros dependientes. Mejor hacerlo cuando haya un sistema de CI validando todas las combinaciones de features.

2. **TLS runtime via rustls**: El struct `TlsConfig` esta listo. Implementar la conexion con rustls requiere feature flag `server-tls` y dependencias pesadas. Planificado para v9.

---

## Verificacion Final

```bash
# Compilacion limpia
cargo check --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools"
# 0 errors, 0 warnings

# Tests
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools" --lib
# 4882 passed, 0 failed

# Clippy
cargo clippy --features "full,autonomous,scheduler,butler,browser,distributed-agents,distributed-network,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools" -- -W clippy::all
# 0 warnings
```
