# Deployment Guide — ai_assistant

> Version: v10 (2026-02-24)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building from Source](#building-from-source)
3. [Docker Deployment](#docker-deployment)
4. [Configuration](#configuration)
5. [Server Configuration](#server-configuration)
6. [Authentication](#authentication)
7. [TLS / HTTPS](#tls--https)
8. [Logging](#logging)
9. [Monitoring](#monitoring)
10. [Memory Persistence](#memory-persistence)
11. [Health Checks](#health-checks)
12. [Rate Limiting](#rate-limiting)
13. [Production Checklist](#production-checklist)

---

## Prerequisites

- **Rust 1.82+** (for building from source)
- **Docker** (optional, for containerized deployment)
- An LLM provider: Ollama, LM Studio, OpenAI, Anthropic, Gemini, etc.

---

## Building from Source

```bash
# Clone the repository
git clone <repo-url>
cd ai_assistant_standalone

# Build with all features
cargo build --release --features full

# Build with specific features only
cargo build --release --features "core,rag,security,streaming"

# The binary is at target/release/ai_assistant
```

### Feature Flags

| Flag | Description | In `full`? |
|------|-----------|-----------|
| `core` | Base LLM providers, config, error handling | Yes |
| `rag` | RAG pipeline, vector DBs, embeddings | Yes |
| `security` | RBAC, PII detection, guardrails | Yes |
| `streaming` | SSE, WebSocket streaming | Yes |
| `multi-agent` | Multi-agent orchestration | Yes |
| `autonomous` | Autonomous agent loop | No |
| `scheduler` | Cron scheduler (requires `autonomous`) | No |
| `browser` | CDP browser automation (requires `autonomous`) | No |
| `distributed-network` | QUIC/TLS P2P networking | No |
| `containers` | Docker-based execution (bollard) | No |
| `audio` | Speech STT/TTS providers | No |

---

## Docker Deployment

### Build the Image

```bash
docker build -t ai-assistant:latest .
```

The Dockerfile uses multi-stage builds:
- **Builder**: `rust:1.82-slim-bookworm` — compiles with `--features full`
- **Runtime**: `debian:bookworm-slim` — minimal runtime (~80MB)

### Run the Container

```bash
docker run -d \
  --name ai-assistant \
  -p 8090:8090 \
  -v ./config:/app/config \
  -v ./data:/app/data \
  -e RUST_LOG=info \
  -e AI_CONFIG_PATH=/app/config/config.toml \
  ai-assistant:latest
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  ai-assistant:
    image: ai-assistant:latest
    ports:
      - "8090:8090"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - RUST_LOG=info
      - AI_CONFIG_PATH=/app/config/config.toml
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8090/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## Configuration

### TOML Format (recommended)

```toml
[provider]
type = "ollama"
model = "llama2"

[urls]
ollama = "http://localhost:11434"
lm_studio = "http://localhost:1234"

[generation]
temperature = 0.7
max_history = 20

[rag]
enabled = true
knowledge_tokens = 2000
conversation_tokens = 1500
```

### JSON Format

```json
{
  "provider": {
    "type": "ollama",
    "model": "llama2"
  },
  "urls": {
    "ollama": "http://localhost:11434"
  },
  "generation": {
    "temperature": 0.7,
    "max_history": 20
  }
}
```

### Config Hot-Reload

The server supports hot-reloading certain configuration fields without restart:

| Field | Hot-reloadable? |
|-------|----------------|
| `model` | Yes |
| `temperature` | Yes |
| `log level` | Yes |
| `host` / `port` | No (requires restart) |
| `TLS cert/key` | No (requires restart) |

Use `ConfigWatcher` to poll for file changes by modification time.

---

## Server Configuration

```rust
use ai_assistant::server::{ServerConfig, AiServer};

let config = ServerConfig {
    host: "0.0.0.0".to_string(),    // Bind to all interfaces
    port: 8090,
    max_body_size: 1_048_576,        // 1 MB
    read_timeout_secs: 30,
    max_headers: 100,                // Anti-abuse: max header count
    max_header_line: 8192,           // Anti-abuse: max header line length
    body_read_timeout_ms: 30_000,    // Anti-slowloris
    max_message_length: 100_000,     // Max chat message chars
    auth: Default::default(),
    cors: Default::default(),
    tls: None,
};

let server = AiServer::new(config);
server.run_blocking(); // Blocks the current thread
```

### Background Mode

```rust
let handle = server.start_background();
// ... do other work ...
handle.shutdown(); // Graceful shutdown
```

---

## Authentication

The server supports bearer tokens and API keys:

```rust
use ai_assistant::server::AuthConfig;

let auth = AuthConfig {
    enabled: true,
    bearer_tokens: vec!["my-secret-token".to_string()],
    api_keys: vec!["api-key-123".to_string()],
    exempt_paths: vec!["/health".to_string()],
};
```

### Request Examples

```bash
# Bearer token
curl -H "Authorization: Bearer my-secret-token" http://localhost:8090/chat

# API key
curl -H "X-API-Key: api-key-123" http://localhost:8090/chat

# Health check (exempt by default)
curl http://localhost:8090/health
```

When auth fails, the server returns `401 Unauthorized` with a JSON error body.
Authentication uses constant-time comparison to prevent timing attacks.

---

## TLS / HTTPS

TLS is configured via the `TlsConfig` struct:

```rust
use ai_assistant::server::TlsConfig;

let tls = TlsConfig {
    cert_path: "/path/to/cert.pem".to_string(),
    key_path: "/path/to/key.pem".to_string(),
};

let config = ServerConfig {
    tls: Some(tls),
    ..Default::default()
};
```

> **TLS runtime** is available via the `server-tls` feature flag:
> `cargo build --features "full,server-tls"`.
> When `tls` is set in `ServerConfig`, the server terminates TLS natively using rustls.
> For production, a reverse proxy (nginx, Caddy, Traefik) is still recommended for
> certificate rotation and additional security layers.

### Reverse Proxy (nginx example)

```nginx
server {
    listen 443 ssl;
    server_name ai.example.com;

    ssl_certificate /etc/letsencrypt/live/ai.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ai.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Request-Id $request_id;
    }
}
```

---

## Logging

The server uses the `log` crate. Set the log level via the `RUST_LOG` environment variable:

```bash
# Basic levels
RUST_LOG=info ./ai_assistant
RUST_LOG=debug ./ai_assistant
RUST_LOG=warn ./ai_assistant

# Per-module filtering
RUST_LOG=ai_assistant=debug,ai_assistant::server=trace ./ai_assistant
```

### What Gets Logged

| Module | Events |
|--------|--------|
| `server` | Request method, path, status, latency, correlation ID |
| `providers` | Provider name, model, token count, latency, retries |
| `assistant` | Session creation, model changes, config loads |
| `cloud_connectors` | S3 operations (upload, download, list, delete) |
| `mcp_client` | MCP server connections, disconnections |

### Correlation IDs

Every request gets a unique 32-character hex correlation ID via the `X-Request-Id` header.
If the client provides `X-Request-Id`, the server reuses it. Otherwise, it generates one.

```bash
# Client-provided correlation ID
curl -H "X-Request-Id: my-trace-id-123" http://localhost:8090/health

# Server-generated (returned in response header)
curl -v http://localhost:8090/health
# < X-Request-Id: a1b2c3d4e5f6...
```

---

## Monitoring

### Prometheus Metrics

The server exposes metrics at `GET /metrics` in Prometheus exposition format:

```
# HELP ai_server_requests_total Total number of HTTP requests.
# TYPE ai_server_requests_total counter
ai_server_requests_total 1234
# HELP ai_server_requests_by_status HTTP requests by status class.
# TYPE ai_server_requests_by_status counter
ai_server_requests_by_status{status="2xx"} 1100
ai_server_requests_by_status{status="4xx"} 120
ai_server_requests_by_status{status="5xx"} 14
# HELP ai_server_request_duration_seconds_total Total time spent processing requests.
# TYPE ai_server_request_duration_seconds_total counter
ai_server_request_duration_seconds_total 45.123456
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'ai-assistant'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8090']
    metrics_path: '/metrics'
```

### Audit Log

The server maintains an in-memory audit log of security-relevant events:
- `AuthSuccess` / `AuthFailure` — authentication outcomes
- `ConfigChange` — configuration modifications via POST /config
- `SessionCreated` / `SessionDeleted` — session lifecycle events
- `RequestProcessed` — general request processing

---

## Memory Persistence

By default, all memory stores (episodic, procedural, entity) are in-memory. Enable persistence to survive restarts:

### Manual Save/Load

```rust
use ai_assistant::advanced_memory::{EpisodicStore, ProceduralStore, EntityStore};

// Save
let store = EpisodicStore::new();
store.save_to_file("/app/data/episodic.json")?;

// Load
let store = EpisodicStore::load_from_file("/app/data/episodic.json")?;
```

### Compressed Snapshots

```rust
// Save compressed (gzip)
store.save_compressed("/app/data/episodic.json.gz")?;

// Load compressed
let store = EpisodicStore::load_compressed("/app/data/episodic.json.gz")?;
```

### Checksum Verification

```rust
// Save with checksum (creates .checksum sidecar file)
store.save_with_checksum("/app/data/episodic.json")?;

// Load with checksum verification
let store = EpisodicStore::load_with_checksum("/app/data/episodic.json")?;
```

### Auto-Persistence

```rust
use ai_assistant::AutoPersistenceConfig;

let config = AutoPersistenceConfig {
    base_dir: "/app/data/memory".to_string(),
    save_interval_secs: 300,    // Save every 5 minutes
    max_snapshots: 10,          // Keep last 10 snapshots
    save_on_drop: true,         // Save when the config is dropped
};
```

---

## Health Checks

The `GET /health` endpoint returns detailed system status:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": "llama2",
  "provider": "Ollama",
  "uptime_secs": 3600,
  "active_sessions": 5,
  "conversation_messages": 42
}
```

Use this for:
- **Docker HEALTHCHECK** (built into the Dockerfile)
- **Kubernetes liveness/readiness probes**
- **Load balancer health checks**

---

## Rate Limiting

The server supports per-instance rate limiting:

```rust
use ai_assistant::server::ServerRateLimiter;

let limiter = ServerRateLimiter::new(60); // 60 requests per minute
```

When the limit is exceeded, the server returns `429 Too Many Requests` with a `Retry-After` header.

---

## Production Checklist

- [ ] **Authentication enabled** — Set `auth.enabled = true` with strong tokens
- [ ] **CORS restricted** — Replace `*` with specific allowed origins
- [ ] **TLS termination** — Use reverse proxy (nginx/Caddy) or wait for `server-tls` feature
- [ ] **Logging configured** — Set `RUST_LOG=info` at minimum
- [ ] **Metrics scraped** — Configure Prometheus to scrape `/metrics`
- [ ] **Memory persisted** — Enable `AutoPersistenceConfig` with appropriate `base_dir`
- [ ] **Rate limiting** — Configure `ServerRateLimiter` to prevent abuse
- [ ] **Body limits** — Review `max_body_size` and `max_message_length`
- [ ] **Health checks** — Verify `/health` returns expected data
- [ ] **Non-root user** — Docker image runs as `aiassistant` user by default
- [ ] **Secrets management** — API keys not hardcoded, loaded from env/config
- [ ] **Backups** — Periodic backup of `/app/data/` directory
