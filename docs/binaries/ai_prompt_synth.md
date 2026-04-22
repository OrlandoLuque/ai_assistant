# `ai_prompt_synth` — Fragment Synthesis auditor CLI

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_prompt_synth.rs` |
| `required-features` | `prompt-synthesis` |

## Purpose

Read-only auditor for the `FragmentLedger` produced by `FragmentBandit`. Inspect event history, verify the chain, or aggregate per-arm statistics (selection counts + mean reward).

## Build

```bash
cargo build --release --bin ai_prompt_synth --features prompt-synthesis
```

## Usage

```bash
ai_prompt_synth ledger-show <LEDGER_JSONL> [--last N]
ai_prompt_synth ledger-verify <LEDGER_JSONL>
ai_prompt_synth arms-summary <LEDGER_JSONL>
```

## See also

- [`ai_prompt_synth_gui`](ai_prompt_synth_gui.md) — desktop auditor
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
