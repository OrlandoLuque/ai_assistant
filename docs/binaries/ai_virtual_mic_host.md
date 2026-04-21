# `ai_virtual_mic_host` — Virtual-mic host driver

| Field | Value |
|---|---|
| Group | Media |
| Binary path | `src/bin/ai_virtual_mic_host.rs` |
| `required-features` | `audio` |

## Purpose

Host-side driver for the virtual microphone pipeline. Runs in the background, accepts audio from one or more `ai_virtual_mic` clients, and exposes a single virtual input device to the OS that other apps (Teams, Zoom, OBS) can capture.

## Build

```bash
cargo build --release --bin ai_virtual_mic_host --features audio
```

## Usage

```bash
ai_virtual_mic_host
ai_virtual_mic_host --device "My Virtual Mic"
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_virtual_mic`](ai_virtual_mic.md) — effect client
