# `ai_logs_gui` — Log viewer (GUI)

| Field | Value |
|---|---|
| Group | GUI |
| Binary path | `src/bin/ai_logs_gui.rs` |
| `required-features` | `gui-logs` |

## Purpose

Desktop viewer for distributed log streams emitted by [`ai_logs`](ai_logs.md). Provides live tail, per-node filtering, severity colour-coding, and JSON pretty-printing.

## Build

```bash
cargo build --release --bin ai_logs_gui --features gui-logs
```

## Usage

```bash
ai_logs_gui
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_logs`](ai_logs.md) — CLI companion
