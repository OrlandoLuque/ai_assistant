# `ai_jobs` — Cron-like job daemon

| Field | Value |
|---|---|
| Group | Setup & Ops |
| Binary path | `src/bin/ai_jobs.rs` |
| `required-features` | `scheduler` |
| Added in | V77 |

## Purpose

Cron-like daemon that runs declared jobs on a schedule. Two runtime modes per job:

- **delegated** *(default)* — shells out to `ai_cli` or any shell command. Always available.
- **embedded** — runs an in-process `AiAssistant` with access to RAG, tools, memory, and session state. Gated behind `--features full`.

Manifest is **JSON** (see [`examples/jobs.json`](../../examples/jobs.json)).

## Build

```bash
# Delegated mode only (minimal)
cargo build --release --bin ai_jobs --features scheduler

# Embedded mode (adds RAG/tools/memory inside the daemon)
cargo build --release --bin ai_jobs --features "scheduler,full"
```

## Subcommands

| Command | Effect |
|---|---|
| `validate <manifest>` | Lint the manifest |
| `list <manifest>` | Human-readable job summary |
| `dry-run <manifest> --minutes <N>` | Simulate next N minutes |
| `run <manifest>` | Start the daemon |

## Usage

```bash
ai_jobs validate examples/jobs.json
ai_jobs list      examples/jobs.json
ai_jobs dry-run   examples/jobs.json --minutes 120
ai_jobs run       examples/jobs.json
```

## Security

- Per-job `timeout_secs` (default 60s)
- `MAX_JOBS = 1000` hard cap
- `std::panic::catch_unwind` protects the daemon from job panics
- Unknown providers downgrade to Ollama with a warning
- API-key env vars are referenced **by name** — never logged

## See also

- [`docs/BINARIES.md`](../BINARIES.md)
- [`docs/IMPROVEMENTS_V77.md`](../IMPROVEMENTS_V77.md)
- [`examples/jobs.json`](../../examples/jobs.json)
