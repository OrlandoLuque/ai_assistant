# `ai_gui` — Desktop chat GUI

| Field | Value |
|---|---|
| Group | GUI |
| Binary path | `src/bin/ai_gui.rs` |
| `required-features` | `gui` |

## Purpose

egui-based cross-platform desktop chat. Ships with provider auto-detection, model browsing, `.kpkg` file opening, and a minimal history panel. Good for "just give me a window to talk to a local LLM".

## Build

```bash
cargo build --release --bin ai_gui --features gui
```

## Usage

```bash
ai_gui
```

On first launch: the app scans for Ollama / LM Studio on the default ports and offers to pick a model.

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_gui-pro`](ai_gui-pro.md) — power-user variant
