# `ai_optimize` — Configuration optimiser

| Field | Value |
|---|---|
| Group | Setup & Ops |
| Binary path | `src/bin/ai_optimize.rs` |
| `required-features` | `full` |

## Purpose

Analyses a running deployment (provider mix, typical workloads, hardware) and proposes configuration changes to squeeze more throughput or lower cost. ML-assisted (V71) — learns from snapshot data across runs.

## Build

```bash
cargo build --release --bin ai_optimize --features full
```

## Usage

```bash
ai_optimize --snapshot cost.json
ai_optimize --recommend --format json
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/IMPROVEMENTS_V71.md`](../IMPROVEMENTS_V71.md)
