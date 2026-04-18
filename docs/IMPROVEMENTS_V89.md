# IMPROVEMENTS V89 — Wire All Binary Stubs (0.2.21)

## Context

Audit of the 20 binaries in `src/bin/` found 5 real stubs (placeholder strings
or unreachable subcommands) across 4 binaries. V89 replaces each with a real
implementation using APIs that were already available in the library, with
**zero new dependencies**.

## Changes

### 1) `src/bin/ai_cli.rs` — Cost savings + tool/workflow subcommands

**`cost_subcommand_savings`** (~L1585–1700):
- Previously printed a stub "Savings analysis not yet implemented" line.
- Now loads a `CostDashboardSnapshot` via `load_cost_snapshot()`, restores it
  with `CostDashboard::restore()`, and reports:
  - Total cost, cost grouped by model (`cost_by_model()`)
  - Top 5 most expensive requests (`most_expensive(5)`)
  - Hypothetical single-model projection vs. actual mixed-model cost, as a
    "you would have paid X — you saved Y (Z%)" comparison.
- Emits JSON on `--json`, human-readable text otherwise.

**`cmd_tool` + `cmd_workflow` + `run_delegated_llm`** (~L1054–1170):
- Added two subcommands `tool <name> --args <json>` and
  `workflow <spec>` that build a prompt and delegate to the configured
  provider through the same `run_delegated_llm` helper (provider/model/url
  flags respected). Usable as a thin subprocess wrapper for other binaries.
- Updated `print_usage()` to document both subcommands.
- Added `tool` and `workflow` arms to the main `match` dispatcher.

### 2) `src/bin/ai_gpu_share.rs` — GPU detection + identity backup/restore

- Replaced the `"GPU detection not yet wired"` placeholder with a local
  `detect_gpu()` (nvidia-smi → CUDA env vars → Apple Silicon `sysctl`). Mirrors
  the detection flow in `butler::GpuDetector` without requiring the `butler`
  feature on the binary.
- Replaced the "Network join placeholder" with `CertificateManager::load_or_create`
  against `./node_identity`, so the first join creates a persistent node
  identity and subsequent joins reuse it.
- `cmd_backup_keys` / `cmd_restore` now call `load_identity` / `save_identity`
  from `CertificateManager`, roundtripping the node cert + key + CA cert.
- Smoke-tested locally: RTX 4080 SUPER (16GB) detected, backup + restore
  produces the same identity on disk.

### 3) `src/bin/ai_jobs.rs` — Delegated tool/workflow runtime

- `run_delegated_tool` and `run_delegated_workflow` were inert stubs. They now
  call a new `run_delegated_ai_cli` helper that spawns `ai_cli` with the right
  subcommand (`tool …` / `workflow …`), mirroring the existing
  `run_delegated_agent` subprocess pattern.
- Both signatures gained a `timeout_secs: u64` parameter. Call sites (L921–926)
  pass `meta.timeout_secs` from the job record so job-level timeouts are
  honored end-to-end.

### 4) `src/bin/ai_gui-pro.rs` — Google Drive OAuth manual flow

- Replaced the non-functional "Authenticate" button in the `gdrive` panel
  with a manual token flow — no new HTTP listener or crate needed:
  - Button opens a modal (`gdrive_oauth_modal`) that instructs the user to
    paste an OAuth Playground token.
  - Token is stored via `save_gdrive_token` in the binary's data dir (0o600
    perms on Unix) and loaded on next launch via `load_gdrive_token`.
  - "Disconnect" button deletes the stored token and clears the UI state.
- `AiGuiApp` gained three fields: `gdrive_oauth_modal: bool`,
  `gdrive_token_input: String`, `gdrive_token: Option<String>`.
- `render_gdrive_oauth_modal` is called each frame before `render_toasts`.

## Files modified

- `Cargo.toml` — version 0.2.20 → 0.2.21
- `src/bin/ai_cli.rs`
- `src/bin/ai_gpu_share.rs`
- `src/bin/ai_jobs.rs`
- `src/bin/ai_gui-pro.rs`
- `docs/IMPROVEMENTS_V89.md` (this file)

## Validation

- `cargo build --release --bin ai_cli` — clean, only pre-existing warnings
- `cargo build --release --bin ai_gpu_share` — clean (2m 43s)
- `cargo build --release --bin ai_jobs` — clean (2m 28s)
- `cargo build --release --bin ai_gui-pro` — clean (3m 29s)
- `cargo test --lib --features full` — **5701 passed, 0 failed**, 29.83s
- `cargo clippy --release --bin ai_cli` — no new warnings on modified ranges
  (1054–1170, 1585–1700); pre-existing warnings only.

## Breaking changes

None. All binary changes are additive (new subcommands, new struct fields on
the GUI app) or replace placeholders that were previously unreachable in any
functioning path.
