# `ai_skills_gui` — Skill Forge auditor GUI

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_skills_gui.rs` |
| `required-features` | `skill-forge`, `gui-pro` |

## Purpose

Desktop companion to `ai_skills`. Browses a directory of `.skill.json` files, shows metadata, live-verifies content and WASM Blake3 hashes, and renders the `SkillLedger` chain with colour-coded integrity status.

## Build

```bash
cargo build --release --bin ai_skills_gui --features "skill-forge gui-pro"
```

## Usage

Launch the GUI, point the top bar at a skills directory and (optionally) a ledger JSONL file, click **Refresh**. Enable **Auto (5s)** to reload on a timer.

## See also

- [`ai_skills`](ai_skills.md) — CLI companion
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
