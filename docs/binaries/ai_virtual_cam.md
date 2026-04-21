# `ai_virtual_cam` — Virtual camera

| Field | Value |
|---|---|
| Group | Media |
| Binary path | `src/bin/ai_virtual_cam.rs` |
| `required-features` | `video-io` |

## Purpose

Virtual camera output with 19 built-in video effects (blur, edge detect, sepia, cartoon, chromatic aberration, …). Registers a virtual camera device the OS can expose to any capture client.

## Build

```bash
cargo build --release --bin ai_virtual_cam --features video-io
```

## Usage

```bash
ai_virtual_cam --effect blur
ai_virtual_cam --effect edge-detect --source 0
ai_virtual_cam --list-effects
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
