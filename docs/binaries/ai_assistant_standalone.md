# `ai_assistant_standalone` — Single-binary HTTP service

| Field | Value |
|---|---|
| Group | Server |
| Binary path | `src/bin/ai_assistant_standalone.rs` |
| `required-features` | `full`, `server-axum` |

## Purpose

Trimmed-down service focused on HTTP only (no MCP, no cluster). Good default for `systemd`-managed deployments or containers where you want a small surface.

## Build

```bash
cargo build --release --bin ai_assistant_standalone --features "full,server-axum"
```

## Usage

```bash
ai_assistant_standalone
ai_assistant_standalone --port 9090
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)
