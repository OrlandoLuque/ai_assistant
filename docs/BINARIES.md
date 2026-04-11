# ai_assistant — Binary Catalogue

This document is the authoritative inventory of every executable binary
shipped by the `ai_assistant` crate. It is kept in lockstep with the
companion page
[`ai_assistant-website/binaries.html`](../../ai_assistant-website/binaries.html).

> **Total binaries: 20** (as of V77 — added `ai_jobs`)
>
> Each binary is a first-class consumer of the library and is gated behind
> the feature flags listed below. Binaries with `required-features` that
> are not satisfied by your current build are simply skipped by Cargo.

## Summary table

| # | Binary | Group | `required-features` | Purpose |
|---|--------|-------|---------------------|---------|
| 1 | `ai_cli` | CLI | — | Non-interactive CLI (scan, query, bench, test, cost, ...) |
| 2 | `ai_assistant_cli` | CLI | — | Interactive REPL (chat-style) |
| 3 | `ai_assistant_server` | Server | `full` | HTTP + MCP server |
| 4 | `ai_assistant_standalone` | Server | `full`, `server-axum` | Single-binary service (HTTP only) |
| 5 | `ai_cluster_node` | Server | `full`, `server-cluster` | Distributed cluster node (QUIC mesh) |
| 6 | `ai_proxy` | Server | `server-axum` | Reverse-proxy in front of providers |
| 7 | `ai_gui` | GUI | `gui` | egui desktop chat |
| 8 | `ai_gui-pro` | GUI | `gui-pro` | Extended desktop GUI (power users) |
| 9 | `ai_setup_gui` | GUI | `gui` | Graphical setup wizard |
| 10 | `ai_logs_gui` | GUI | `gui-logs` | Log viewer desktop app |
| 11 | `ai_setup` | Setup & Ops | `full` | Terminal setup wizard |
| 12 | `ai_optimize` | Setup & Ops | `full` | Configuration optimizer |
| 13 | `ai_jobs` **(new in V77)** | Setup & Ops | `scheduler` | Cron-like job daemon (delegated + embedded) |
| 14 | `ai_logs` | Setup & Ops | `distributed-network` | Distributed log aggregator |
| 15 | `ai_virtual_mic` | Media | `audio-io` | Virtual microphone (voice effects) |
| 16 | `ai_virtual_mic_host` | Media | `audio` | Host driver for the virtual mic |
| 17 | `ai_virtual_cam` | Media | `video-io` | Virtual camera (video effects) |
| 18 | `kpkg_tool` | Knowledge | `rag` | Encrypted knowledge package tool (.kpkg) |
| 19 | `ai_gpu_share` | GPU Sharing | `full`, `gpu-sharing` | GPU sharing network CLI |
| 20 | `ai_test_harness` | Testing | `full`, `browser` | Multi-category test harness |

## Detailed descriptions

### CLI & REPL

#### `ai_cli`
Non-interactive CLI for scripting, CI/CD, and one-shot queries. Ships with
10 subcommands including the new V77 `cost` subcommand.

```bash
cargo run --bin ai_cli -- scan
cargo run --bin ai_cli -- query "What is Rust?"
cargo run --bin ai_cli --features full -- cost report --snapshot cost.json
```

Subcommands: `scan`, `providers`, `models`, `config`, `butler`, `query`,
`bench`, `test`, `cost`, `help`.

#### `ai_assistant_cli`
Interactive terminal REPL. Good for exploring models without leaving the
terminal.

### Servers

#### `ai_assistant_server`
Reference HTTP + MCP server. Exposes chat, RAG, tools, and 40+ MCP tools.
Requires `full`.

#### `ai_assistant_standalone`
A trimmed-down, single-binary service focused on HTTP. Good default for
`systemd`-managed deployments.

#### `ai_cluster_node`
Spawns a distributed node that joins a QUIC mesh for cluster-wide RAG and
agent federation. Requires `full`, `server-cluster`.

#### `ai_proxy`
Lightweight reverse proxy that sits in front of upstream providers (Ollama,
OpenAI, Anthropic, …) for auditing, rate limiting, and budget enforcement.

### GUIs

#### `ai_gui`
egui-based desktop chat for everyday interactive use.

#### `ai_gui-pro`
Power-user variant with prompt templates, multi-session tabs, RAG inspection.

#### `ai_setup_gui`
Graphical setup wizard — guides through provider detection, API keys, and
smoke tests.

#### `ai_logs_gui`
Desktop log viewer for distributed `ai_logs` streams.

### Setup & Ops

#### `ai_setup`
Terminal-based setup wizard. Covers provider detection, API key storage,
optional feature enablement.

#### `ai_optimize`
Configuration optimizer — analyzes a running deployment and suggests
changes to squeeze more performance out of the current hardware/provider mix.

#### `ai_jobs` **(new in V77)**
Cron-like job daemon.

```bash
cargo run --bin ai_jobs --features scheduler -- validate examples/jobs.json
cargo run --bin ai_jobs --features scheduler -- list      examples/jobs.json
cargo run --bin ai_jobs --features scheduler -- dry-run   examples/jobs.json --minutes 120
cargo run --bin ai_jobs --features scheduler -- run       examples/jobs.json
```

Two runtime modes per job:

- `delegated` *(default)* — shells out to `ai_cli` or any shell command.
  Always available.
- `embedded` — runs an in-process `AiAssistant` with access to RAG, tools,
  memory, and session state. Gated behind `--features full`.

Manifest is **JSON** (see [`examples/jobs.json`](../examples/jobs.json)).

Security:

- Per-job `timeout_secs` (default 60s)
- `MAX_JOBS = 1000` hard cap
- `std::panic::catch_unwind` protects the daemon from job panics
- Unknown providers downgrade to Ollama with a warning
- API key env vars are referenced by **name** — never logged

#### `ai_logs`
Collector/aggregator for distributed logs from other binaries.

### Media

#### `ai_virtual_mic`
Virtual microphone with configurable voice effects (anonymizer, distorter,
snore detector, pitch shifter, …).

#### `ai_virtual_mic_host`
Host driver that other processes can use to send audio to the virtual mic.

#### `ai_virtual_cam`
Virtual camera with 19 video effects (blur, edge detect, sepia, …).

### Knowledge

#### `kpkg_tool`
Create, read, and verify encrypted knowledge packages (`.kpkg`) used by the
RAG pipeline. AES-256-GCM + optional signatures.

### GPU Sharing

#### `ai_gpu_share`
CLI for participating in an `ai_assistant` GPU sharing network (credit
system, dynamic pricing, collusion detection). Requires `full` + `gpu-sharing`.

### Testing

#### `ai_test_harness`
Multi-category test harness used by CI and by local developers to run
feature-gated integration tests without having to remember every
`cargo test` incantation. Requires `full` + `browser` (for crawl policy
tests).

## Cross-references

- [`docs/USE_CASES.md`](USE_CASES.md) — end-to-end scenarios that wire
  several binaries together.
- [`docs/IMPROVEMENTS_V77.md`](IMPROVEMENTS_V77.md) — rationale behind the
  V77 additions (`ai_jobs`, `ai_cli cost`, etc.).
- [`CHANGELOG.md`](../CHANGELOG.md) — release-by-release history.
