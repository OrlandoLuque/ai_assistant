# `ai_prompt_synth_gui` — Fragment Synthesis dashboard

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_prompt_synth_gui.rs` |
| `required-features` | `prompt-synthesis`, `gui-pro` |

## Purpose

Desktop dashboard that replays a `FragmentLedger` JSONL into a per-cluster + per-arm summary: selection counts, samples, mean reward, retirement status, and per-arm reward-history sparklines. Side panel shows ledger chain integrity.

## Build

```bash
cargo build --release --bin ai_prompt_synth_gui --features "prompt-synthesis gui-pro"
```

## Usage

Launch, point at a ledger JSONL file, click **Reload**. Select a cluster to drill into its arms.

## See also

- [`ai_prompt_synth`](ai_prompt_synth.md) — CLI companion
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
