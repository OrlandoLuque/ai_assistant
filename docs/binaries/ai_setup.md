# `ai_setup` — Setup wizard (terminal)

| Field | Value |
|---|---|
| Group | Setup & Ops |
| Binary path | `src/bin/ai_setup.rs` |
| `required-features` | `full` |

## Purpose

Interactive terminal wizard that walks a first-time user through: local-provider detection, API-key storage, optional feature enablement, and a final smoke test. The terminal sibling of [`ai_setup_gui`](ai_setup_gui.md).

## Build

```bash
cargo build --release --bin ai_setup --features full
```

## Usage

```bash
ai_setup
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_setup_gui`](ai_setup_gui.md) — graphical variant
- [`docs/GETTING_STARTED.md`](../GETTING_STARTED.md)
