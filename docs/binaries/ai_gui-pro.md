# `ai_gui-pro` — Advanced desktop GUI

| Field | Value |
|---|---|
| Group | GUI |
| Binary path | `src/bin/ai_gui-pro.rs` |
| `required-features` | `gui-pro` |

## Purpose

Power-user variant of `ai_gui`. Adds prompt templates, multi-session tabs, RAG inspection panels, token/cost meters, and the extended egui widgets surface.

## Build

```bash
cargo build --release --bin ai_gui-pro --features gui-pro
```

## Usage

```bash
ai_gui-pro
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_gui`](ai_gui.md) — simpler variant
