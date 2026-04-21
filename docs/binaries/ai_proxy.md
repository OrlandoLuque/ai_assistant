# `ai_proxy` — Hardened LLM gateway

| Field | Value |
|---|---|
| Group | Server |
| Binary path | `src/bin/ai_proxy.rs` |
| `required-features` | `server-axum` (router only); `server-axum,security` (full gateway) |

## Purpose

Production API gateway that sits in front of upstream providers (Ollama, OpenAI, Anthropic, …). Two feature profiles:

- `--features server-axum` — router + round-robin LB + session affinity + health checks + optional Bearer auth (V77 parity).
- `--features "server-axum,security"` — full gateway: rate limit, PII/toxicity/attack guards, budget enforcement, LRU cache, JSONL audit log (V78).

## Build

```bash
# Router only
cargo build --release --bin ai_proxy --features server-axum

# Hardened gateway (recommended)
cargo build --release --bin ai_proxy --features "server-axum,security"
```

## Usage

```bash
# TOML config
ai_proxy --config examples/ai_proxy.toml

# Dry-run: validate and print merged config
ai_proxy --config examples/ai_proxy.toml --dry-run

# CLI overrides win over the file
ai_proxy --config ai_proxy.toml --port 9000 --disable-cache
```

## CLI flags

`--config <PATH>` · `--port <PORT>` · `--backends <a:p,b:p,...>` · `--health-interval <SECS>` · `--audit-log <PATH>` · `--audit-max-files <N>` · `--enable-pii-redaction` · `--disable-cache` · `--cost-snapshot <PATH>` · `--dry-run` · `--api-key <KEY>` **(deprecated — prefer `AI_PROXY_API_KEY` env var)**.

## Middleware pipeline (security feature on)

Rate limit → content-length guard → PII input → toxicity input → attack guard → budget pre-check → cache lookup → backend → PII output → toxicity output → budget post-update → cache store → audit log.

## Response headers

`X-Request-Id` (every response), `X-Cache: HIT|MISS` (chat/completions), `X-Reason` on 429/503 (`budget-exceeded`, `rate_limit`, `output-blocked`, `middleware-error`).

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/IMPROVEMENTS_V78.md`](../IMPROVEMENTS_V78.md) — full design + 13 security mitigations
- [`examples/ai_proxy.toml`](../../examples/ai_proxy.toml)
