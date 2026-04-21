# `ai_assistant_server` — HTTP + MCP API server

| Field | Value |
|---|---|
| Group | Server |
| Binary path | `src/bin/ai_assistant_server.rs` |
| `required-features` | `full` |

## Purpose

Reference HTTP server bundling the library behind a REST + SSE API and a Model Context Protocol (MCP) endpoint exposing 40+ tools. OpenAI-compatible on `/v1/chat/completions` so it drops into any OpenAI client (Continue.dev, Cursor, LangChain, etc.).

## Build

```bash
cargo build --release --bin ai_assistant_server --features full
```

## Usage

```bash
ai_assistant_server
ai_assistant_server --port 8090 --api-key mysecret
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/models` | List available models |
| POST | `/chat` | Send a message (JSON body) |
| POST | `/chat/stream` | SSE streaming responses |
| POST | `/v1/chat/completions` | OpenAI-compatible endpoint |
| GET | `/config` | View current server config |
| GET | `/metrics` | Prometheus metrics |
| GET | `/openapi.json` | OpenAPI 3.0 spec |
| POST | `/mcp` | MCP JSON-RPC endpoint (40+ tools) |

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/API_REFERENCE.md`](../API_REFERENCE.md)
- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)
