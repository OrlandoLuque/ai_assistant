# IMPROVEMENTS V77 — Docs, Binaries & UX Pass

## Context

V75 (Cost Intelligence) added `CostDashboard` and auto-wiring in
`poll_response()`, but exposed it only via the library API and MCP tools.
V76 (Feature Hygiene) moved 15 modules behind their rightful feature gates.
V77 is the natural next step: **make the existing surface discoverable and
scriptable**.

An audit done at the start of V77 confirmed that:

- **19 binaries already exist** (~41K LOC). There are no missing binaries
  for "butler" or "scheduler" — butler is already a subcommand of `ai_cli`
  and scheduler exists as `examples/scheduler_agent.rs`.
- The real gaps are:
  1. **No CLI access to the V75 cost intelligence** — users can't inspect
     costs from the terminal, only via the API.
  2. **No standalone scheduler daemon** — the pattern exists only as an
     example, not as a binary that loads jobs from a file.
  3. **No centralized docs** for binaries or use cases. The 19 binaries are
     discoverable only by reading `Cargo.toml`; the website has 16 pages
     but lacks `binaries.html` and `use_cases.html`.

V77 closes all three gaps **without creating duplicate binaries**.

---

## Workstreams

### A — `ai_jobs` binary (new)

- **File**: `src/bin/ai_jobs.rs` (~970 LOC), `examples/jobs.json`
- **Feature gate**: `required-features = ["scheduler"]`
- **Two runtime modes**:
  - `delegated` *(default)* — shells out to `ai_cli` or any shell command.
    Always available.
  - `embedded` — runs an in-process `AiAssistant` with access to RAG,
    tools, memory, and session state. Gated behind `#[cfg(feature = "full")]`.
- **Subcommands**: `validate`, `list`, `dry-run`, `run`, `help`
- **Manifest format**: JSON. A parallel schema lives in `ai_jobs.rs`
  (`JobsFile`, `JobConfig`, `ActionConfig`, `JobRuntime`) and is converted
  into the core `scheduler::ScheduledJob` via `into_scheduled_job()`.
  This is intentional — the core `scheduler::*` types are **not**
  `Serialize/Deserialize`, and V77 avoids changing them to keep the
  public surface stable.
- **Security hardening**:
  - `MAX_JOBS = 1000` hard cap at load time
  - Per-job `timeout_secs` (default 60s) enforced via `Child::kill()`
  - `std::panic::catch_unwind` wraps every job execution — a panicking
    embedded agent cannot take the daemon down
  - Unknown providers downgrade to Ollama with a warning
  - API key env vars are referenced by **name** only, never logged
  - Manifest path is canonicalized and logged before use
- **Tests**: 14 unit tests in-file + 6 integration tests in
  `tests/ai_jobs_integration.rs` that spawn the compiled binary via
  `env!("CARGO_BIN_EXE_ai_jobs")`.

### B — `ai_cli cost` subcommand

- **File**: `src/bin/ai_cli.rs` (+~250 LOC)
- **Sub-subcommands**:
  | Command | Purpose |
  |---|---|
  | `cost report [--snapshot <path>]` | Formatted dashboard report |
  | `cost budget --snapshot <path>` | JSON budget status (remaining / used / projected_monthly) |
  | `cost savings --snapshot <path>` | Informational stub — AllocationResult persistence deferred to V78 |
  | `cost projection --snapshot <path>` | Daily / monthly / per-1k cost projections |
  | `cost export --snapshot <path> --output <file.csv> [--force]` | CSV export (refuses to overwrite existing file without `--force`) |
  | `cost help` | Usage |
- **Uses existing V75 API**: `CostDashboard::new()`, `restore()`,
  `format_report()`, `export_csv()`, `projected_*_cost()`,
  `CostDashboardSnapshot`.
- **Tests**: 6 new unit tests for `find_flag_value`, `load_cost_snapshot`,
  and `provider_from_name`.

### C — Use cases docs

- **Files**: `docs/USE_CASES.md`, `ai_assistant-website/use_cases.html`
- **8 end-to-end scenarios** (local RAG chat, CI cost gate, scheduled
  briefings, TLS team server, distributed RAG cluster, voice assistant,
  butler bootstrap, MCP backend).
- Each scenario lists problem, binaries, commands, and required features.

### D — Binary catalogue docs

- **Files**: `docs/BINARIES.md`, `ai_assistant-website/binaries.html`
- Authoritative table of **20 binaries** (19 pre-existing + `ai_jobs`),
  grouped by role: CLI, Servers, GUIs, Setup & Ops, Media, Knowledge,
  GPU Sharing, Testing.
- `binaries.html` and `use_cases.html` are cross-linked, and both are
  linked from `index.html` as new link cards.

### E — Doc-comments audit

- Planned as "add doc-comments to 16 binaries" but turned into a no-op:
  on inspection every binary already has a `//!` doc-comment. The initial
  audit was fooled by the `// Required Notice: Copyright …` line that
  precedes the doc-comment in each file.
- Only `ai_jobs.rs` was missing the copyright notice line; added.

---

## V76 regressions discovered and fixed

While running the new `tests/ai_jobs_integration.rs` test, three binaries
failed to compile with feature flags that worked before V76:

1. `ai_test_harness` — uses `CrawlPolicy` but V76 gated `crawl_policy`
   behind `browser`. Fix: `required-features = ["full", "browser"]`.
2. `ai_virtual_mic_host` — uses `group_queue_host` but V76 gated it behind
   `audio`. Fix: `required-features = ["audio"]`.
3. `ai_gpu_share` — uses `gpu_sharing` but V76 gated it behind `gpu-sharing`;
   the binary only declared `required-features = ["full"]`. Fix: tightened
   to `["full", "gpu-sharing"]`.

These were latent bugs V76 should have caught. V77 lockstep adds them to
`Cargo.toml`.

---

## Files changed

| File | Delta | Type |
|---|---|---|
| `src/bin/ai_jobs.rs` | +970 | NEW |
| `src/bin/ai_cli.rs` | +250 | EDIT (cost subcommand + 6 tests) |
| `Cargo.toml` | +8 | EDIT (bin entry, version bump, 3 V76 fixes) |
| `examples/jobs.json` | +45 | NEW |
| `tests/ai_jobs_integration.rs` | +100 | NEW (6 tests) |
| `docs/BINARIES.md` | +180 | NEW |
| `docs/USE_CASES.md` | +250 | NEW |
| `docs/IMPROVEMENTS_V77.md` | +180 | NEW (this file) |
| `CHANGELOG.md` | +45 | EDIT (v33 entry) |
| `ai_assistant-website/binaries.html` | +470 | NEW |
| `ai_assistant-website/use_cases.html` | +470 | NEW |
| `ai_assistant-website/index.html` | +18 | EDIT (2 link cards) |
| **Total** | **~3000 LOC** | |

---

## Verification

```bash
# Compile checks
cargo check --features "scheduler"
cargo check --features "full,scheduler"
cargo build --bin ai_jobs --features scheduler
cargo build --bin ai_jobs --features "full,scheduler"
cargo build --bin ai_cli --features full

# Tests
cargo test --bin ai_jobs --features scheduler
cargo test --bin ai_cli --features full cost_
cargo test --test ai_jobs_integration --features scheduler

# Manual smoke
./target/debug/ai_jobs validate examples/jobs.json
./target/debug/ai_jobs list examples/jobs.json
./target/debug/ai_jobs dry-run examples/jobs.json --minutes 120
./target/debug/ai_cli cost help
./target/debug/ai_cli cost report --snapshot /tmp/snapshot.json
```

---

## Security summary

| # | Vector | Mitigation |
|---|---|---|
| S-A1 | Shell injection via `jobs.json` | Documented as trusted input (same model as crontab/systemd) |
| S-A2 | Path traversal on manifest / snapshot args | `Path::canonicalize()` + log |
| S-A3 | Unbounded job load | `MAX_JOBS = 1000` |
| S-A4 | Unbounded shell runtime | Default 60s timeout, overridable per-job, `Child::kill()` on timeout |
| S-A5 | API key leakage in logs | Env var referenced by **name** only |
| S-A6 | Panic in embedded agent tumbando daemon | `std::panic::catch_unwind` per job |
| S-A7 | Embedded jobs sin `full` | Detected at parse time, skipped with warning, daemon continues |
| S-B1 | Snapshot path traversal | Canonicalized + read-only |
| S-B2 | Malformed snapshot JSON (NaN/Infinity) | Rejected by V75 `validate_cost()` |
| S-B3 | `cost export --output` overwriting files | Requires `--force` if file exists |
| S-B4 | CSV injection in export | Mitigated in V75 (`sanitize_csv_field`) |

---

## Deferred to V78+

1. **`CostDashboardSnapshot` persistence in `StorageContext`** — V75
   added the snapshot API but not the wiring to disk. V77 `cost export`
   accepts external snapshots; V78 should wire them to storage.
2. **`AllocationResult` savings metrics in the snapshot** — `cost savings`
   currently returns an informational message. To make it work end-to-end,
   `CostDashboardSnapshot` needs `total_tokens_saved` / `compression_ratio`,
   and `AiAssistant::poll_response()` needs to accumulate them.
3. **`ai_cli tool <name> <json>`, `ai_cli workflow run <id>`** — subcommands
   that `ai_jobs` delegated mode invokes. If missing, audit and stub in V78.
4. **Audit of the 64 ungated modules** (deferred from V76 → V80).
5. **`Serialize/Deserialize` on `scheduler::*` core types** — only if a
   future workstream needs it; V77's parallel schema is sufficient for
   `ai_jobs`.
6. **`ai_jobs` run_count persistence across restarts** — currently lost on
   shutdown of the daemon.

---

## Stats

- Version bump: **0.2.8 → 0.2.9**
- New binary: **`ai_jobs`** (total: **20**)
- ~26 new tests (14 ai_jobs unit + 6 ai_cli unit + 6 ai_jobs integration)
- 3 latent V76 compile regressions fixed
- 2 new docs pages (`BINARIES.md`, `USE_CASES.md`) + 2 website pages
- 0 new Cargo dependencies
