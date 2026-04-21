# `ai_setup_gui` — Setup wizard (GUI)

| Field | Value |
|---|---|
| Group | GUI |
| Binary path | `src/bin/ai_setup_gui.rs` |
| `required-features` | `gui` |

## Purpose

Graphical setup wizard — guides first-time users through provider detection, API-key entry, and smoke tests. Graphical sibling of [`ai_setup`](ai_setup.md).

## Build

```bash
cargo build --release --bin ai_setup_gui --features gui
```

## Usage

```bash
ai_setup_gui
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_setup`](ai_setup.md) — terminal variant
