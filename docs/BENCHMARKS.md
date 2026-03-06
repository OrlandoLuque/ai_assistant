# Competitive Benchmarks

Estimated performance characteristics of `ai_assistant` compared to equivalent solutions in other ecosystems. These are projected estimates based on Rust language characteristics and similar Rust HTTP servers (axum, actix-web).

> **Disclaimer**: These are estimates, not measured benchmarks. Actual numbers depend on hardware, configuration, and workload. We encourage users to run their own benchmarks.

---

## Binary Size (Release, stripped)

| Solution | Binary Size | Notes |
|----------|------------|-------|
| **ai_assistant** (minimal) | ~5 MB | `--features core` only |
| **ai_assistant** (full+server) | ~15 MB | `--features full,server-axum` |
| **ai_assistant** (all features) | ~25 MB | All features enabled |
| LangChain (Python) | 300+ MB | Python runtime + dependencies |
| LlamaIndex (Python) | 250+ MB | Python runtime + dependencies |
| Spring AI (Java) | 150+ MB | JVM + dependencies |
| LangChain.js (Node) | 200+ MB | Node runtime + node_modules |

---

## Memory Usage (Idle Server)

| Solution | RAM (idle) | RAM (under load) | Notes |
|----------|-----------|-------------------|-------|
| **ai_assistant** | ~10 MB | ~30-50 MB | No GC, no runtime overhead |
| Python (FastAPI + LangChain) | 200+ MB | 500+ MB | Python interpreter + GC |
| Java (Spring AI) | 150+ MB | 400+ MB | JVM heap + GC |
| Node.js (LangChain.js) | 80+ MB | 200+ MB | V8 heap + GC |
| Go (custom) | ~15 MB | ~50 MB | Closest competitor (compiled, GC) |

---

## Startup Time

| Solution | Cold Start | Notes |
|----------|-----------|-------|
| **ai_assistant** | <100 ms | Native binary, no interpreter |
| Python (FastAPI) | 3-8 s | Import chain, dependency init |
| Java (Spring Boot) | 5-15 s | JVM warmup, classpath scanning |
| Node.js (Express) | 1-3 s | Module resolution, V8 init |
| Go (net/http) | <100 ms | Compiled, similar to Rust |

---

## HTTP Throughput (requests/sec, single core)

Estimated for simple JSON endpoints (health check, model listing):

| Solution | req/s (est.) | P50 Latency | P99 Latency |
|----------|-------------|-------------|-------------|
| **ai_assistant** (axum) | 100K-200K | <0.1 ms | <1 ms |
| Go (net/http) | 80K-150K | <0.1 ms | <1 ms |
| Node.js (Express) | 15K-30K | ~1 ms | ~5 ms |
| Python (FastAPI) | 5K-15K | ~2 ms | ~10 ms |
| Java (Spring Boot) | 30K-60K | ~0.5 ms | ~3 ms |

> Note: For LLM-backed endpoints, throughput is dominated by LLM inference latency, not server overhead.

---

## Feature Completeness (Single Binary)

| Feature | ai_assistant | LangChain | LlamaIndex | Spring AI |
|---------|-------------|-----------|------------|-----------|
| Multi-provider LLM | 15+ providers | 10+ | 5+ | 5+ |
| RAG (5 levels) | Self-RAG, CRAG, Graph, RAPTOR | Basic RAG | Advanced RAG | Basic RAG |
| Vector DB backends | 7 | 5+ | 5+ | 3 |
| Multi-agent orchestration | 5 roles | Yes | No | No |
| Autonomous agent | 5 autonomy levels | Agents | No | No |
| Browser automation | CDP | Playwright | No | No |
| Distributed (CRDTs, DHT) | Built-in | No | No | No |
| QUIC/TLS mesh networking | Built-in | No | No | No |
| HTTP server (embedded) | Built-in | Separate | Separate | Built-in |
| Streaming (SSE + WS) | Built-in | Partial | Partial | Partial |
| MCP protocol | 12 tools | Community | No | No |
| Guardrails pipeline | 7-stage | LangSmith | No | No |
| OpenTelemetry | Built-in | LangSmith | No | Micrometer |
| Voice agent | STT + TTS | No | No | No |
| WASM support | Built-in | No | No | No |
| Total modules | 220+ | ~50 | ~30 | ~20 |
| Total tests | 6,500+ | Varies | Varies | Varies |

---

## Deployment Footprint

| Scenario | ai_assistant | Python Equivalent |
|----------|-------------|-------------------|
| Minimal chat | 5 MB binary, 10 MB RAM | 300 MB venv, 200 MB RAM |
| RAG server | 15 MB binary, 30 MB RAM | 500 MB venv, 400 MB RAM |
| 3-node cluster | 3x 15 MB, Docker Compose | 3x 500 MB, K8s recommended |
| Docker image | ~30 MB (slim) | ~1 GB (python:3.12-slim + deps) |

---

## Zero-GC Advantage

Rust's ownership model means:
- **No garbage collector** — deterministic memory management
- **No GC pauses** — latency is predictable under load
- **No memory bloat** — no retained garbage between collections
- **Thread safety** at compile time — no runtime race condition checks

This is especially relevant for:
- Real-time streaming (SSE/WebSocket) where GC pauses cause stutter
- High-throughput API servers where tail latency matters
- Long-running cluster nodes that must be stable for days/weeks

---

## Running Your Own Benchmarks

```bash
# Criterion benchmarks (16 micro-benchmarks)
cargo bench --features full

# HTTP server load test (requires wrk or hey)
cargo run --features "full,server-axum" --bin ai_assistant_standalone -- --port 8090 &
wrk -t4 -c100 -d30s http://localhost:8090/health

# Memory profiling
cargo build --release --features "full,server-axum" --bin ai_assistant_standalone
valgrind --tool=massif ./target/release/ai_assistant_standalone --port 8090
```
