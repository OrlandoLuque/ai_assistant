# V38 — Resilience Engineering Completa

**Tesis**: Implementacion completa de patrones de resiliencia avanzada:
Dead Letter Queue mejorada, Bulkhead pattern, Chaos Engineering/Fault Injection,
Auto-reconexion WebSocket y SSE, Adaptive Timeouts, y Load Shedding.

**Estado**: HECHO
**Fecha**: 2026-03-15
**LOC nuevas**: ~2,150 (codigo) + ~1,260 (docs) ≈ 3,410 total

---

## Resumen de cambios

### 1. Enhanced Dead Letter Queue (`src/message_queue.rs`)

Extiende el DLQ basico (add/pop/len) con metadatos ricos y capacidad de replay:
- `DeadLetterEntry`: message + reason + failure_category + attempt_count + timestamps + error_history
- `FailureCategory` enum: Timeout, RateLimited, ProviderUnavailable, InvalidRequest, Unknown
- Metodos nuevos: `add_detailed()`, `replay_one()` (FIFO), `replay_by_category()`,
  `drain_older_than_ms()`, `peek_all()`, `stats()`, `clear()`
- `DlqStats`: total, by_category (HashMap), oldest_age_ms
- Backward compatible: `add()` sigue funcionando (crea Unknown entry)

### 2. Bulkhead Pattern (`src/bulkhead.rs` — NUEVO)

Aislamiento de recursos por semaforo con RAII permits:
- `Bulkhead`: max_concurrent + Condvar para blocking acquire con timeout
- `BulkheadPermit`: RAII guard (Drop decrementa active, poison-safe)
- `BulkheadRegistry`: multiples bulkheads nombrados
- `BulkheadStats`: active, utilization_percent, accepted/rejected/timed_out counters
- Presets: `for_chat()` (10), `for_streaming()` (5), `for_embeddings()` (20), `for_background()` (3)

### 3. Adaptive Timeouts (`src/adaptive_timeout.rs` — NUEVO)

Calculo dinamico de timeouts basado en latencia observada:
- Ring buffer lock-free (AtomicU64) para muestras de latencia
- Calculo de percentiles P50/P95/P99
- Formula: `timeout = clamp(percentile * multiplier, min, max)`
- Presets: `conservative()` (3x P99), `responsive()` (2x P95), `aggressive()` (1.5x P95)
- Integracion con `ResilientExecutor` via `with_adaptive_timeout()`

### 4. Load Shedding (`src/load_shedding.rs` — NUEVO)

Rechazo inteligente de requests bajo presion:
- `LoadContext`: CPU, memoria, queue depth, priority, request age, P95 latency
- `SheddingDecision`: Accept, Shed { reason }, Throttle { delay }
- 4 estrategias: PriorityBased, Probabilistic, OldestFirst, Adaptive
- Priority protection: High-priority nunca se descartan
- Cooldown anti-oscilacion
- Presets: `conservative()`, `aggressive()`, `disabled()`

### 5. Chaos Engineering (`src/fault_injection.rs` — NUEVO, feature `chaos-testing`)

Framework de inyeccion de fallos para testing de resiliencia:
- `FaultType`: Latency, Error, Timeout, ConnectionReset, RateLimited, PartialResponse, CorruptResponse
- `FaultRule`: target (All/Named/Matching) + probabilidad + max_injections
- `FaultInjector`: evaluacion de reglas con PRNG determinista (xorshift64)
- `with_seed(u64)` para tests reproducibles
- Gated: `#[cfg(any(test, feature = "chaos-testing"))]`

### 6. WebSocket Auto-Reconnect (`src/websocket_streaming.rs`)

State machine de reconexion con backoff exponencial:
- `WsConnectionState`: Connected → Disconnected → Reconnecting → GaveUp
- `ResilientWsStream`: caller-driven (no threads internos)
- Callbacks: `on_reconnect`, `on_disconnect`, `on_give_up`
- Backoff exponencial con jitter, configurable
- Presets: `default()` (10 attempts, 1s-60s), `aggressive()`, `quick()`

### 7. SSE Auto-Reconnect (`src/resumable_streaming.rs`)

Reconexion basada en checkpoints con Last-Event-ID:
- `ResilientSseStream`: wrapper sobre `ResumableStream`
- `handle_disconnect()` guarda last_event_id
- `get_resume_chunks()` llama `resume_from(last_event_id)`
- Honra campo `retry:` del servidor SSE
- Callbacks: `on_reconnect`, `on_chunks_lost`
- Deteccion de chunks perdidos por eviccion del replay buffer

### 8. Integration Wiring

- `ResilientExecutor.adaptive_timeout: Option<Arc<AdaptiveTimeout>>` — timeout dinamico
- `ResilientExecutor.dead_letter_queue: Option<Arc<DeadLetterQueue>>` — captura automatica
- Builder methods: `with_adaptive_timeout()`, `with_dead_letter_queue()`
- Feature `chaos-testing` en Cargo.toml (opt-in, no en `full`)

---

## Tests nuevos: ~103

### En `src/bulkhead.rs` (~18 tests):
| Test | Que verifica |
|------|-------------|
| `test_try_acquire_success` | Adquisicion basica |
| `test_try_acquire_full_rejection` | Rechazo cuando lleno |
| `test_acquire_timeout_success` | Blocking acquire con release en otro thread |
| `test_acquire_timeout_expired` | Timeout expira |
| `test_permit_auto_release` | RAII drop libera slot |
| `test_concurrent_acquire_release` | Multi-thread safety |
| `test_stats_tracking` | Contadores accepted/rejected/timed_out |
| `test_registry_*` | Registro, lookup, acquire, stats |
| `test_preset_*` | Configuraciones predefinidas |

### En `src/adaptive_timeout.rs` (~15 tests):
| Test | Que verifica |
|------|-------------|
| `test_initial_timeout` | Timeout antes de muestras |
| `test_adapts_to_*_latency` | Adaptacion a baja/alta latencia |
| `test_respects_*_bound` | Limites min/max |
| `test_concurrent_recording` | Thread safety del ring buffer |
| `test_preset_*` | Presets conservative/responsive/aggressive |

### En `src/load_shedding.rs` (~15 tests):
| Test | Que verifica |
|------|-------------|
| `test_accept_under_normal_load` | Sin shed bajo carga normal |
| `test_high_priority_never_shed` | Proteccion de prioridad |
| `test_priority_based_*` | Estrategia PriorityBased |
| `test_oldest_first_*` | Estrategia OldestFirst |
| `test_adaptive_*` | Estrategia Adaptive |
| `test_disabled_config` | Config disabled acepta todo |
| `test_cooldown` | Anti-oscilacion |

### En `src/fault_injection.rs` (~20 tests):
| Test | Que verifica |
|------|-------------|
| `test_deterministic_with_seed` | Reproducibilidad |
| `test_probability_*` | Probabilidad 0/1 |
| `test_target_*` | Targeting All/Named/Matching |
| `test_max_injections_cap` | Limite de inyecciones |
| `test_enable_disable_rule` | Activacion/desactivacion |
| `test_*_builder` | Convenience constructors |

### En `src/message_queue.rs` (~12 tests nuevos DLQ):
| Test | Que verifica |
|------|-------------|
| `test_dlq_add_detailed` | Entrada rica |
| `test_dlq_replay_one_fifo` | Orden FIFO |
| `test_dlq_replay_by_category` | Replay selectivo |
| `test_dlq_peek_all` | Inspeccion sin consumir |
| `test_dlq_stats` | Estadisticas agregadas |
| `test_dlq_backward_compat_add` | Compatibilidad con add() |

### En `src/websocket_streaming.rs` (~12 tests nuevos):
State machine transitions, backoff, callbacks, presets.

### En `src/resumable_streaming.rs` (~10 tests nuevos):
Disconnect/resume, backoff, server retry override, chunk replay.

---

## Archivos modificados

| Archivo | Cambios |
|---------|---------|
| `src/bulkhead.rs` | +~700 LOC: nuevo modulo completo |
| `src/adaptive_timeout.rs` | +~700 LOC: nuevo modulo completo |
| `src/load_shedding.rs` | +~400 LOC: nuevo modulo completo |
| `src/fault_injection.rs` | +~650 LOC: nuevo modulo completo |
| `src/message_queue.rs` | +~280 LOC: DLQ enhanced + 12 tests |
| `src/websocket_streaming.rs` | +~350 LOC: WS reconnect + 12 tests |
| `src/resumable_streaming.rs` | +~300 LOC: SSE reconnect + 10 tests |
| `src/retry.rs` | +~20 LOC: campos opcionales AdaptiveTimeout + DLQ |
| `src/lib.rs` | +~10 LOC: pub mod + re-exports |
| `Cargo.toml` | +3 LOC: feature chaos-testing |
| `docs/CONCEPTS.md` | +7 secciones (175-181) |
| `docs/GUIDE.md` | +7 secciones (148-154) |
| `docs/TESTING.md` | +actualizado conteo + tabla V38 |
| `docs/IMPROVEMENTS_V38.md` | nuevo |

---

## Estadisticas del proyecto actualizadas

| Metrica | Valor |
|---------|-------|
| **LOC totales** | ~388K |
| **Archivos fuente** | 319 .rs |
| **Tests** | 4,995 (lib, features full+chaos-testing) |
| **Feature flags** | 55 (+1: chaos-testing) |
| **Binarios** | 9 |

---

## Patrones de resiliencia — cobertura completa

| Patron | Modulo | Estado |
|--------|--------|--------|
| Retry + Exponential Backoff | `retry.rs` | Existente |
| Circuit Breaker | `retry.rs`, `cluster/health.rs` | Existente |
| Provider Failover | `fallback.rs`, `assistant.rs` | Existente |
| Health Monitoring | `health_check.rs` | Existente |
| Phi Accrual Failure Detection | `failure_detector.rs` | Existente |
| Connection Pooling | `connection_pool.rs` | Existente |
| Rate Limiting | `security/rate_limiting.rs` | Existente |
| Priority Queue + Backpressure | `request_queue.rs` | Existente |
| Resumable Streaming | `resumable_streaming.rs` | Existente |
| Keep-Alive Management | `keepalive.rs` | Existente |
| **Dead Letter Queue** | `message_queue.rs` | **V38 — Enhanced** |
| **Bulkhead Isolation** | `bulkhead.rs` | **V38 — Nuevo** |
| **Adaptive Timeouts** | `adaptive_timeout.rs` | **V38 — Nuevo** |
| **Load Shedding** | `load_shedding.rs` | **V38 — Nuevo** |
| **Chaos Engineering** | `fault_injection.rs` | **V38 — Nuevo** |
| **WebSocket Auto-Reconnect** | `websocket_streaming.rs` | **V38 — Nuevo** |
| **SSE Auto-Reconnect** | `resumable_streaming.rs` | **V38 — Nuevo** |

---

## Proximos pasos (planificado, no implementado)

- Conectar `LoadShedder` a `RequestQueue.enqueue()` (optional check)
- Conectar `FaultInjector` a `ResilientProviderRegistry` (test-only wiring)
- Conectar `BulkheadRegistry` a `ResilientProviderRegistry`
- Persistencia opcional del DLQ (file/SQLite)
- Dashboard de metricas de resiliencia (via OTel spans)
