# `ai_virtual_mic` — Virtual microphone

| Field | Value |
|---|---|
| Group | Media |
| Binary path | `src/bin/ai_virtual_mic.rs` |
| `required-features` | `audio-io` |

## Purpose

Virtual microphone client with configurable voice effects: anonymiser, distorter, snore detector, pitch shifter, bandpass filter, and more. Sends its output into the virtual-mic host so it can be selected as an input device in meetings or recordings.

## Build

```bash
cargo build --release --bin ai_virtual_mic --features audio-io
```

## Usage

```bash
ai_virtual_mic --effect anonymize
ai_virtual_mic --effect pitch-shift --semitones -3
```

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`ai_virtual_mic_host`](ai_virtual_mic_host.md) — host driver this client talks to
