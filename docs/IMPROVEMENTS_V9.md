# Plan de Mejoras para ai_assistant — v9

> **Estado: EN PROGRESO**

> Documento generado el 2026-02-24.
> Basado en v8 (23/26, 4882 tests, 0 failures, 0 clippy warnings).
> ~271k LoC, 240+ source files.

---

## Contexto

Tras v8 "Code Maturity + Deployment Readiness", el servidor HTTP tiene auth, CORS,
graceful shutdown, Prometheus metrics y request logging. Sin embargo, una auditoria
de produccion revela:

1. **Sin streaming en server** — Solo POST /chat sincrono. Sin SSE ni WebSocket.
   Imposible para UX real donde el usuario ve tokens en tiempo real.

2. **Health check basico** — Solo dice "ok". No reporta estado de dependencias,
   latencia de providers, uso de memoria, sesiones activas, cache hit rate.

3. **Sin rate limiting en server** — RateLimiter existe como struct aislado pero
   no esta integrado en el middleware del servidor.

4. **Sin validacion de request** — El body JSON se parsea sin validar limites
   (max message length, max tokens). Facil de abusar.

5. **Sin endpoints de sesion** — No hay GET /sessions, DELETE /sessions/{id}.
   Los usuarios no pueden gestionar sus conversaciones via API.

6. **Sin compresion de respuesta** — Respuestas grandes van sin gzip. Desperdicio
   de ancho de banda.

7. **Sin API versioning** — Los endpoints estan en raiz (/chat). No hay /api/v1/.

8. **Tests desbalanceados** — conversation_flow.rs tiene solo 4 tests para 439 LOC.
   knowledge_graph.rs tiene 29 tests para 2983 LOC.

9. **Modulos standalone no integrados** — decision_tree, mcts_planner, hitl,
   constrained_decoding, distillation existen pero no estan en AiAssistant API.

10. **Sin auditoria de operaciones** — Cambios de config, rotacion de API keys,
    accesos a datos sensibles no se registran.

**Tesis v9**: Completar la API del servidor, integrar modulos standalone, y
llenar los gaps de tests mas criticos. API completeness + integration wiring.

---

## Estructura: 26 items, 8 fases

### Fase 1 — Server API Completeness (6 items)

| # | Descripcion | Tests | Fichero |
|---|------------|-------|---------|
| 1.1 | **SSE streaming endpoint**: POST /chat/stream con Server-Sent Events. `data: {token}\n\n` format. Connection keepalive. | ~10 | server.rs |
| 1.2 | **Enhanced health check**: GET /health con provider status, memory usage, active sessions, uptime, cache stats. | ~8 | server.rs |
| 1.3 | **Session endpoints**: GET /sessions (list), GET /sessions/{id}, DELETE /sessions/{id}, POST /sessions (new). | ~10 | server.rs |
| 1.4 | **Request validation**: max_message_length, max_system_prompt_length, allowed models list. 422 on invalid. | ~8 | server.rs |
| 1.5 | **Response compression**: gzip Content-Encoding cuando Accept-Encoding: gzip. flate2 ya en deps. | ~6 | server.rs |
| 1.6 | **Rate limiting middleware**: Integrate existing RateLimiter into server. Per-endpoint limits. 429 Too Many Requests. | ~8 | server.rs |

### Fase 2 — Test Coverage Boost (3 items)

| # | Fichero | LOC | Tests actuales | Tests nuevos |
|---|---------|-----|---------------|-------------|
| 2.1 | conversation_flow.rs | 439 | 4 | ~12 (state transitions, flow merge, edge cases) |
| 2.2 | knowledge_graph.rs | 2983 | 29 | ~15 (entity dedup, cycles, traversal, query) |
| 2.3 | vector_db.rs | 3331 | 39 | ~12 (migration, cache eviction, error paths) |

### Fase 3 — Feature Integration into AiAssistant (4 items)

| # | Modulo | Descripcion | Tests | Fichero |
|---|--------|------------|-------|---------|
| 3.1 | constrained_decoding | `assistant.generate_with_grammar(grammar, prompt)` method | ~6 | assistant.rs |
| 3.2 | hitl | `assistant.send_message_with_approval(msg, handler)` method | ~6 | assistant.rs |
| 3.3 | mcp_client | `assistant.connect_mcp_server(url)` + `call_mcp_tool(name, args)` | ~6 | assistant.rs |
| 3.4 | distillation | `assistant.collect_trajectory()` + `export_training_data()` | ~6 | assistant.rs |

### Fase 4 — Audit Logging (2 items)

| # | Descripcion | Tests | Fichero |
|---|------------|-------|---------|
| 4.1 | **AuditLog struct**: timestamp, event_type, actor, details. Thread-safe append. | ~8 | server.rs |
| 4.2 | **Audit events**: config_change, auth_failure, session_delete, api_key_rotation. | ~6 | server.rs |

### Fase 5 — API Versioning (2 items)

| # | Descripcion | Tests | Fichero |
|---|------------|-------|---------|
| 5.1 | **Versioned routes**: /api/v1/chat, /api/v1/health, etc. Backward compat: old routes still work. | ~6 | server.rs |
| 5.2 | **API version header**: X-API-Version response header. Accept-Version request header. | ~4 | server.rs |

### Fase 6 — Structured Error Responses (2 items)

| # | Descripcion | Tests | Fichero |
|---|------------|-------|---------|
| 6.1 | **Error response format**: `{error_code, message, details, retry_after}` JSON. | ~6 | server.rs |
| 6.2 | **Error codes catalog**: INVALID_JSON, AUTH_FAILED, RATE_LIMITED, MODEL_ERROR, etc. | ~4 | server.rs |

### Fase 7 — Persistence Enhancements (3 items)

| # | Descripcion | Tests | Fichero |
|---|------------|-------|---------|
| 7.1 | **Compressed snapshots**: gzip JSON snapshots. `.json.gz` extension. | ~6 | advanced_memory.rs |
| 7.2 | **Checksum verification**: SHA-256 hash stored alongside snapshot. Verify on load. | ~6 | advanced_memory.rs |
| 7.3 | **Session persistence**: Save/load AiAssistant sessions to disk. | ~8 | assistant.rs |

### Fase 8 — Documentation (2 items)

| # | Descripcion | Fichero |
|---|------------|---------|
| 8.1 | **Deployment guide**: Docker, config, TLS, logging, monitoring. | docs/DEPLOYMENT.md |
| 8.2 | **API reference**: All server endpoints with request/response examples. | docs/API_REFERENCE.md |

---

## Resumen Cuantitativo

| Concepto | Cantidad |
|----------|----------|
| Fases | 8 |
| Items totales | 26 |
| Tests nuevos estimados | ~175 |
| Target tests | ~5057 (4882 + 175) |
