# V90.27 — Embedded `llama-server` launcher + CI fixes

**Version**: 0.2.51 → 0.2.52
**Feature flag**: `vision` (additive — no new flag)
**Date**: 2026-04-29

## Why

V90.26 closed the *configuration* gap for vision: callers could declare
a projector path and the library would validate it and detect whether
the running server had it loaded. But the *operational* gap remained —
operators still had to spawn `llama-server` themselves, manage its
PID, watch its `/health`, and remember to kill it on shutdown.

V90.27 lets the library do that work itself. A small, additive
[`embedded_server`] module spawns and supervises a local `llama-server`
(or any compatible binary), feeds it a validated `--mmproj` path
straight from [`crate::mmproj`], waits for `/health`, and guarantees
the child is killed on `Drop` so a panicking caller does not leak the
process.

## Architectural notes

Embedded launching is a **runtime** concern, separate from
[`AiConfig`]. Reasons:

* `AiConfig` is serializable and persisted to disk. A spawned
  `llama-server` is a live OS resource that does not survive
  serialization.
* Many callers want to point at an already-running `llama-server` they
  manage themselves. The launcher is opt-in: existing flows continue
  to work unchanged.

The launcher **does not** parse upstream output for tokens — every
chat / completion request still goes through [`crate::providers`] over
HTTP. The launcher is purely a process-lifetime helper.

## New surface

| Symbol | File | Purpose |
|--------|------|---------|
| `LlamaServerConfig`, `LlamaServerConfigBuilder` | `src/embedded_server.rs` | Fluent config (binary, model, mmproj, host, port, ctx, GPU layers, extras) |
| `EmbeddedLlamaServer` | `src/embedded_server.rs` | Live handle: `start`, `wait_until_ready`, `is_running`, `pid`, `base_url`, `port`, `binary_filename`, `stop`, `Drop` kills child |
| `LaunchError` | `src/embedded_server.rs` | Typed errors (BinaryNotFound, ModelNotFound, MmprojValidation, PathTraversal, ArgContainsNul, InvalidHost, PortTooLow, SpawnFailed, ChildExitedEarly, Timeout) |
| `build_command_args(&config, port)` | `src/embedded_server.rs` | Pure function; argv inspection without spawning |
| `mock_llama_server` | `src/bin/mock_llama_server.rs` | Test-only fake `llama-server`; declared as `[[bin]]` so Cargo builds it for integration tests |

## Validation pipeline (in order)

1. **Path traversal** rejected for `binary_path`, `model_path`,
   `mmproj_path` — `..` components fail before any I/O. Defense-in-depth
   against symlink-race substitution between check and spawn.
2. **NUL bytes** rejected in `extra_args` and `host`. `Command::args`
   would refuse anyway, but the typed error is more useful.
3. **Empty host** rejected. Non-empty validates only that bind succeeds.
4. **File existence** for `binary_path` and `model_path` (must be
   regular files).
5. **mmproj** validated through `MultimodalProjector::from_path` —
   reuses the V90.26 pipeline (size + GGUF magic + traversal).
6. **Port** — `0` triggers OS auto-pick via `TcpListener::bind`; any
   explicit value below 1024 is rejected (operator should run as
   non-root).
7. **Spawn** — `Command::spawn` with `stdout/stderr` piped (configurable
   via `capture_output(false)`).

## Health probe

`wait_until_ready` polls `GET /health` over a raw `TcpStream` (no
`reqwest` dependency added) at 250 ms intervals until either a `2xx`
status arrives or the deadline elapses. The deadline defaults to
`config.ready_timeout` (60 s) but can be overridden per call.

If the child exits before becoming ready, the wait returns
`LaunchError::ChildExitedEarly` immediately rather than running out
the full timeout.

## Drop safety

`EmbeddedLlamaServer::Drop` calls `child.kill()` then `child.wait()`.
`stop()` is idempotent so panicking callers, double-stop, and explicit
shutdown all converge to "child gone, no zombie".

## Test additions

* **`embedded_server::tests`** — 10 unit tests:
  * `build_command_args_minimal`, `build_command_args_with_mmproj_and_extras`
  * `rejects_binary_path_traversal`, `rejects_model_path_traversal`,
    `rejects_mmproj_path_traversal`
  * `rejects_extra_arg_with_nul`, `rejects_empty_host`,
    `rejects_explicit_low_port`
  * `missing_binary_yields_binary_not_found`
  * `launch_error_display_contains_actionable_text`
* **`tests/embedded_server_integration.rs`** — 6 real-process tests
  using `env!("CARGO_BIN_EXE_mock_llama_server")`:
  * `spawns_mock_server_and_health_returns_ok`
  * `drop_kills_child_process`
  * `wait_until_ready_returns_timeout_when_health_never_replies`
  * `auto_picked_port_is_unique_per_call`
  * `binary_filename_is_safe_for_logs`
  * `explicit_port_is_honoured_when_above_threshold`

Total: **16 new vision-gated tests**.

## CI fixes (rolled into the same release)

* `[profile.bench]` added to `Cargo.toml` inheriting `release-fast` so
  criterion benches compile (the default release profile uses
  `panic = "abort"` which is incompatible with criterion's harness).
* `Security Audit` job replaced the v2 action with manual
  `cargo audit --ignore` for four advisories that all live in
  transitive dependencies we cannot bump:
  * `RUSTSEC-2025-0141` — `bincode` unmaintained
  * `RUSTSEC-2024-0436` — `paste` unmaintained
  * `RUSTSEC-2025-0134` — `rustls-pemfile` unmaintained
  * `RUSTSEC-2026-0002` — `lru` unsound `IterMut` (transitive via
    `tantivy`; no patched version reachable yet)

## Out of scope

* Restart-on-crash policy. The handle reports `is_running()` honestly,
  but supervision is the caller's call.
* Auto-download of `llama-server` binaries from upstream releases.
* Stdout/stderr line capture API. The child's pipes are open but not
  exposed yet — add a reader if a caller needs it.
* Embedded launcher for `koboldcpp` or `text-generation-webui`. The
  argv schema differs; a dedicated module per server is cleaner than
  a generic shim.
