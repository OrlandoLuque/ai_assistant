# v30 Improvements Changelog

> Date: 2026-03-06

v30 is the **Microservice Architecture** release: horizontally scalable axum-based server, cluster infrastructure, Redis backend, Swagger UI, Butler feature analyzer, Docker orchestration, and comprehensive documentation.

---

## Summary Metrics

| Metric | v29 | v30 | Delta |
|--------|-----|-----|-------|
| Unit tests (all features) | 6,565 | ~6,900 | ~+335 |
| Clippy warnings | 0 | 0 | 0 |
| New feature flags | 0 | 5 | +5 |
| New source files | 0 | 14 | +14 |
| Lines added | — | ~12,000 | — |

---

## New Feature Flags

| Flag | Dependencies | In `full`? | Purpose |
|------|-------------|-----------|---------|
| `server-axum` | axum, tower, tower-http, dashmap, tokio, reqwest | No | Production async HTTP server |
| `server-axum-tls` | server-axum + axum-server, rustls | No | TLS termination for axum server |
| `server-cluster` | server-axum + distributed-network | No | Cluster mode with CRDT sync |
| `server-openapi` | server-axum + utoipa, utoipa-swagger-ui | No | Swagger UI / OpenAPI 3.0 |
| `redis-backend` | redis (tokio-comp, connection-manager) | No | Redis-backed rate limiting, sessions, cache |

---

## Phase 1: Cargo.toml + Feature Flags + Skeleton

- Added 8 optional dependencies: axum 0.8, axum-extra 0.9, tower 0.5, tower-http 0.6, dashmap 6, axum-server 0.7, utoipa 5, utoipa-swagger-ui 9, redis 1.0, reqwest 0.12
- 5 new feature flag groups
- `lib.rs` conditional module exports for `server_axum`, `cluster`, `redis_backend`

## Phase 2: AppState Decomposition

- `AppState`: assistant (RwLock), config (RwLock), sessions (DashMap), metrics (AtomicU64), rate limiter (DashMap)
- `AxumServerMetrics`: lock-free counters for requests, errors, latency
- `AxumRateLimiter`: per-IP sliding window with DashMap
- `SessionData`: id, messages, metadata, last_active, affinity_node

## Phase 3: Middleware Layers (tower)

- `AuthLayer`: Bearer + API key authentication with constant-time comparison
- `CorsLayer`: tower-http CORS with configurable origins
- `CompressionLayer`: gzip via tower-http
- `RateLimitLayer`: wraps `AxumRateLimiter` for per-IP limiting
- `RequestBodyLimitLayer`: max body size enforcement
- `TimeoutLayer`: request timeout
- `SetRequestIdLayer`: X-Request-Id generation

## Phase 4: Handler Migration — 24 Endpoints

All 24 endpoints ported from `server.rs` to async axum handlers:
- `/health`, `/models`, `/chat`, `/chat/stream` (SSE), `/ws` (WebSocket)
- `/config`, `/config/update`, `/sessions`, `/sessions/{id}`
- `/metrics`, `/openapi.json`, `/api/v1/*` dual-prefix routing
- OpenAI-compatible: `/v1/chat/completions`, `/v1/models`
- Full enrichment pipeline: system prompt, RAG, guardrails, model routing, budget, output guardrails, telemetry

## Phase 5: Server Entrypoint + Graceful Shutdown

- `AxumServer::new(config)` + `run()` with axum::serve + graceful shutdown
- TLS via `axum-server` + rustls behind `server-axum-tls`
- Ctrl+C / SIGTERM signal handling

## Phase 6: Session Affinity

- `SessionId` extractor: from `X-Session-Id` header, cookie, or auto-generated UUID
- DashMap session-to-node mapping for sticky routing
- Session TTL and cleanup background task

## Phase 7: CLI Binaries (3 new)

### `ai_assistant_standalone` — All-in-One Binary
- Server + REPL + Butler auto-config in a single binary
- `--port`, `--repl`, `--auto-config`, `--dry-run`, `--config`
- 9 tests

### `ai_cluster_node` — Cluster Node
- Requires `server-cluster` feature
- `--node-id`, `--bootstrap-peers`, `--join-token`, `--data-dir`, `--quic-port`
- 6 tests

### `ai_proxy` — Lightweight API Gateway
- Routes OpenAI-compatible requests to backend nodes
- Round-robin load balancing, session affinity, health checks
- `--backends`, `--port`, `--health-interval`, `--api-key`
- 7 tests

## Phase 8: ClusterManager

- `ClusterManager`: orchestrates NetworkNode, ConsistentHashRing, HeartbeatManager, AntiEntropySync
- `ClusterState`: CRDT-based shared state (PNCounter rate limits, LWWMap sessions, ORSet active nodes, GCounter request counts)
- Background tasks: heartbeat loop, CRDT sync loop, persistence loop
- Graceful shutdown: drain, persist, remove from ring, shutdown network
- 7 tests

## Phase 9: Distributed Rate Limiter (2-Layer)

- **Layer 1 (local, instant)**: DashMap<IpAddr, SlidingWindowCounter>
- **Layer 2 (global, eventual)**: PNCounter CRDT synced via ClusterManager
- Background sync pushes local increments to CRDT
- 11 tests

## Phase 10: CRDT Persistence

- Atomic snapshots (write tmp + rename)
- WAL (append-only log) for crash recovery
- `TtlValue<T>`: time-based expiry wrapper
- Snapshot compaction and WAL pruning
- 13 tests

## Phase 11: Cluster Health

- Readiness/liveness probes (`/health/ready`, `/health/live`)
- Graceful drain mode with backpressure (503 when overloaded)
- Circuit breaker per peer: Closed -> Open -> Half-Open
- Session migration on node failure
- 13 tests

## Phase 12: Cluster Prometheus Metrics

- AtomicU64 counters: nodes_active, sync_lag, crdt_merges, rebalances, circuit_breaker_opens, sessions_migrated
- `to_prometheus()` — exposition format with node labels
- `MetricsSnapshot` — serializable point-in-time state
- 9 tests

## Phase 13: Redis Backend

- `RedisConfig`: URL, pool size, timeout, key prefix
- `RedisBackend`: async rate limiting (INCR+EXPIRE), session CRUD (SET/GET/DEL with TTL), cache CRUD, health check (PING)
- Uses `redis::aio::ConnectionManager` for auto-reconnection
- 7 tests

## Phase 14: utoipa / Swagger UI

- `#[utoipa::path(...)]` annotations on 6 handlers
- `ToSchema` derive on request/response types
- Swagger UI at `/swagger-ui`, spec at `/swagger-api.json`
- Behind `server-openapi` feature flag

## Phase 15: Butler Feature Flag Analyzer

- `DeploymentScenario`: MinimalChat, ChatWithRag, StandaloneServer, ClusterNode, DeveloperWorkstation, EnterpriseAll
- `CompilationProfile`: features, estimated binary size, estimated RAM, cargo command
- `analyze_features(report, active_flags)`: detects unnecessary flags (containers without Docker, browser without Chrome, audio without Whisper) and missing flags
- `recommend_compilation(scenario)`: returns optimal feature set for deployment target
- `recommend_compilation_from_flags(flags)`: infers scenario from active flags
- 15 tests

## Phase 16: Docker + Infrastructure

- **Dockerfile**: Parameterized `ARG FEATURES` and `ARG BINARY` for flexible builds
- **docker-compose.yml**: 3-node cluster + optional Redis (profile "redis") + optional pgvector (profile "pgvector"), shared network, health checks

## Phase 17: pgvector Setup

- `docs/PGVECTOR_SETUP.md`: Installation guide (Docker, Linux, macOS, Windows), database setup, configuration, troubleshooting

## Phase 18: concepts.html Completion

- 6 new concepts (#165-170) added to both `concepts.html` and `concepts.md`
- #165 Unified BPE Tokenizer
- #166 Emoticon & Emoji Detection
- #167 Benchmark Suite Ecosystem
- #168 MCP Configuration & Evaluation Tools
- #169 OpenAI-Compatible API & Enrichment Pipeline
- #170 Routing DAG & Bandit State Merging

## Phase 19: HTML Documentation

- `docs/tools_catalog.html`: Complete catalog of 7 binaries, CLI examples, benchmark tools, feature flags
- `docs/product_overview.html`: Marketing-oriented product page with use cases, competitive advantages, deployment options

## Phase 20: Markdown Documentation

- `docs/BENCHMARKS.md`: Competitive analysis (binary size, RAM, CPU, startup vs Go/Python/Java/Node)
- `docs/IMPROVEMENTS_V30.md`: This file
- `docs/PGVECTOR_SETUP.md`: pgvector installation and setup guide

---

## New Files Created

| File | LOC | Tests | Description |
|------|-----|-------|-------------|
| `src/server_axum.rs` | ~2500 | 71 | axum-based HTTP server |
| `src/cluster/mod.rs` | ~520 | 7 | ClusterManager |
| `src/cluster/distributed_rate_limit.rs` | ~300 | 11 | 2-layer rate limiter |
| `src/cluster/crdt_persistence.rs` | ~350 | 13 | WAL + snapshots |
| `src/cluster/health.rs` | ~330 | 13 | Readiness, drain, circuit breaker |
| `src/cluster/metrics.rs` | ~250 | 9 | Prometheus metrics |
| `src/redis_backend.rs` | ~300 | 7 | Redis backend |
| `src/bin/ai_assistant_standalone.rs` | ~430 | 9 | Standalone binary |
| `src/bin/ai_cluster_node.rs` | ~350 | 6 | Cluster node binary |
| `src/bin/ai_proxy.rs` | ~460 | 7 | Proxy gateway binary |
| `Dockerfile` | ~60 | — | Parameterized Docker build |
| `docker-compose.yml` | ~100 | — | 3-node cluster + Redis + pgvector |
| `docs/tools_catalog.html` | ~600 | — | Tools catalog |
| `docs/product_overview.html` | ~900 | — | Product overview |

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | 8 deps, 5 features, 3 [[bin]] targets |
| `src/lib.rs` | Conditional exports for server_axum, cluster, redis_backend |
| `src/butler.rs` | Feature flag analyzer (DeploymentScenario, CompilationProfile, analyze_features) |
| `src/bin/ai_assistant_server.rs` | (existing, minor updates) |
| `docs/concepts.html` | 6 new concepts (#165-170) |
| `docs/concepts.md` | 6 new concepts (#165-170) |

## NOT Modified

- `src/server.rs` — kept as-is for zero-dep deployments (192 tests untouched)

---

## Verification

```bash
# Feature flag compiles
cargo check --features server-axum
cargo check --features "server-axum,server-cluster"
cargo check --features "server-axum,server-openapi"
cargo check --features "full,redis-backend"
cargo check --features full  # still works

# New tests
cargo test --features "full,server-axum,butler" --lib -- server_axum::
cargo test --features "full,server-cluster" --lib -- cluster::
cargo test --features "full,redis-backend" --lib -- redis_backend::
cargo test --features "full,autonomous,butler" --lib -- butler::tests::test_

# Existing tests unaffected
cargo test --features "full,autonomous,scheduler,butler,browser,distributed-agents,containers,audio,workflows,prompt-signatures,a2a,voice-agent,media-generation,distillation,constrained-decoding,hitl,webrtc,devtools,eval-suite" --lib
```
