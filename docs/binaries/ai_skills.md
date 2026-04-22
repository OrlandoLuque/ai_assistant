# `ai_skills` — Skill Forge auditor CLI

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_skills.rs` |
| `required-features` | `skill-forge` |

## Purpose

Read-only auditor for the Skill Forge registry and hash-chained `SkillLedger`. List skills in a directory, inspect a single skill, verify content + WASM Blake3 hashes, verify the ledger chain, or export a skill bundle for offline review.

## Build

```bash
cargo build --release --bin ai_skills --features skill-forge
```

## Usage

```bash
ai_skills list <SKILLS_DIR>
ai_skills inspect <SKILL.skill.json>
ai_skills verify <SKILL.skill.json>
ai_skills ledger-verify <LEDGER_JSONL>
ai_skills ledger-show <LEDGER_JSONL> [--last N]
ai_skills export <SKILL.skill.json> <OUT_DIR>
```

## See also

- [`ai_skills_gui`](ai_skills_gui.md) — desktop auditor
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
