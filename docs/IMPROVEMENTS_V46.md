# V46 — ai_assistant_core v0.2.0: Provider Service Mode

**Estado**: COMPLETADO
**Fecha**: 2026-03-20

---

## Resumen

V46 adds serve mode to ai_assistant_core (separate crate published to crates.io as "anzuelo").
The crate can now act as an OpenAI-compatible proxy server for local LLMs, with NAT traversal
for remote access.

---

## Implementation — HECHO

| # | Tarea | Estado |
|---|-------|--------|
| 1 | **Serve mode** — minimal axum HTTP server proxying to local LLM | HECHO |
| 2 | **Endpoints** — /health, /v1/models, /v1/chat/completions (streaming SSE) | HECHO |
| 3 | **Bearer token auth** — optional authentication | HECHO |
| 4 | **CORS** — permissive CORS for web clients | HECHO |
| 5 | **NAT traversal** — STUN discovery, UPnP IGD, NAT-PMP port mapping | HECHO |
| 6 | **ai_serve binary** — CLI with --port, --backend, --provider, --token, --nat flags | HECHO |
| 7 | **ProviderServiceBuilder** — unified server + NAT setup | HECHO |
| 8 | **11 tests** with mock backend | HECHO |
| 9 | **Published to crates.io** | HECHO |
