# `ai_feedback` — Feedback Loop auditor CLI

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_feedback.rs` |
| `required-features` | `feedback-loop` |

## Purpose

Read-only auditor for the `FeedbackDispatcher` `DispatchLedger` and `RetractionLedger`. Replay event history, verify chain integrity, aggregate per-sink dispatch / failure counts and drop-reason stats, or list the full retraction trail.

## Build

```bash
cargo build --release --bin ai_feedback --features feedback-loop
```

## Usage

```bash
ai_feedback ledger-show <DISPATCH_JSONL> [--last N]
ai_feedback ledger-verify <DISPATCH_JSONL>
ai_feedback retractions <RETRACTION_JSONL>
ai_feedback stats <DISPATCH_JSONL>
```

## See also

- [`ai_feedback_gui`](ai_feedback_gui.md) — desktop auditor
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
