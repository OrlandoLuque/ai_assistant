# `ai_logs` — Distributed log aggregator

| Field | Value |
|---|---|
| Group | Setup & Ops |
| Binary path | `src/bin/ai_logs.rs` |
| `required-features` | `distributed-network` |

## Purpose

Collector / aggregator for logs streamed by other binaries in a cluster. Reads from `ai_cluster_node` peers over the shared distributed-network transport and exposes a local tail or JSON stream.

## Build

```bash
cargo build --release --bin ai_logs --features distributed-network
```

## Usage

```bash
ai_logs tail
ai_logs tail --node node2 --level warn
ai_logs export --since 10m --format json > logs.ndjson
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_logs_gui`](ai_logs_gui.md) — desktop viewer
- [`ai_cluster_node`](ai_cluster_node.md) — log source
