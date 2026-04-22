# `ai_feedback_gui` — Feedback Loop dashboard

| Field | Value |
|---|---|
| Group | Self-Learning (V96) |
| Binary path | `src/bin/ai_feedback_gui.rs` |
| `required-features` | `feedback-loop`, `gui-pro` |

## Purpose

Desktop dashboard for the Feedback Loop. Five tabs: **Overview** (totals + chain status), **Sinks** (per-sink dispatch + failure counts), **Drops** (drop-reason breakdown), **Retractions** (GDPR-style retraction trail), **Events** (most recent 500 raw events).

## Build

```bash
cargo build --release --bin ai_feedback_gui --features "feedback-loop gui-pro"
```

## Usage

Launch, point the top bar at a dispatch ledger JSONL and (optionally) a retraction ledger JSONL, click **Reload**. Enable **Auto (5s)** for timer reload.

## See also

- [`ai_feedback`](ai_feedback.md) — CLI companion
- [`docs/IMPROVEMENTS_V96.md`](../IMPROVEMENTS_V96.md) — design rationale
