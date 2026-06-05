# V145 — rust 1.90 → 1.93 + wasmtime 41 → 45 (0.2.95)

**Status**: SHIPPED 2026-06-06 (0.2.95) — toolchain bump unblocked
the wasmtime 45 jump that V141 explicitly deferred. Closes 13
RUSTSEC advisories in one swing.

**Scope**: bump `rust-toolchain.toml` 1.90.0 → 1.93.0 across the
project (workflows + pin), bump `wasmtime`/`wasmtime-wasi` 41 → 45,
adapt the two `ResourceLimiter` method signatures and the `map_trap`
helper to wasmtime 45's typed `wasmtime::Error`.

**No-goals**: enable WASI in the linker (still gated, V143-009 still
holds), restructure skill_forge, broaden the WASM execution model.

## Por qué

V143/V143.1 push triggered a fresh RUSTSEC sweep that landed 13
wasmtime/cranelift/wiggle advisories on the same Cargo.lock:

| Advisory          | Affected crate         | Class                |
|-------------------|------------------------|----------------------|
| RUSTSEC-2026-0085 | wasmtime               | Sandbox / data leak  |
| RUSTSEC-2026-0086 | wasmtime               | Sandbox escape       |
| RUSTSEC-2026-0087 | wasmtime               | Host panic           |
| RUSTSEC-2026-0088 | wasmtime               | Heap OOB read        |
| RUSTSEC-2026-0089 | wasmtime               | Host panic           |
| RUSTSEC-2026-0091 | wasmtime               | Sandbox escape       |
| RUSTSEC-2026-0092 | wasmtime               | Pooling data leak    |
| RUSTSEC-2026-0093 | wasmtime               | Winch table.fill     |
| RUSTSEC-2026-0094 | wasmtime               | Winch sandbox        |
| RUSTSEC-2026-0095 | wasmtime               | Winch return mask    |
| RUSTSEC-2026-0096 | wasmtime               | aarch64 Cranelift    |
| RUSTSEC-2026-0114 | wasmtime               | Table size panic     |
| RUSTSEC-2026-0149 | wasmtime-wasi          | FilePerms::WRITE     |

None hit a fix line compatible with the 1.90 pin: the lowest fix
for the cluster is wasmtime ≥44.0.2 (requires rustc 1.92) or
≥45.0.0 (requires rustc 1.93). The LTS 36.0.10 path was a
retroceso V141 had already moved past. Option 1 (bump and ship) was
the cleanest answer per the V141 "bump toward margin" guidance.

## Cambios

### Toolchain pin

- `rust-toolchain.toml`: `1.90.0` → `1.93.0`.
- `.github/workflows/ci.yml`: 10 occurrences of
  `dtolnay/rust-toolchain@1.90.0` → `@1.93.0`.
- `.github/workflows/release.yml`: 2 (action + comment).
- `.github/workflows/rustsec-review-monthly.yml`: 1.
- `.github/workflows/supply-chain.yml`: 2.

### Dependency bump

- `Cargo.toml`:
  - `wasmtime = { version = "41", ... }` → `"45"`.
  - `wasmtime-wasi = { version = "41", ... }` → `"45"`.
- `Cargo.lock` regenerated via
  `cargo update -p wasmtime -p wasmtime-wasi` (wasmtime 41.0.4 →
  45.0.1, cranelift 0.128.4 → 0.132.1, wiggle 41.0.4 → 45.0.1).

### Source adaptation (`src/skill_forge/wasm.rs`)

wasmtime 45 split its own error type out of `anyhow`:

- `impl wasmtime::ResourceLimiter for MemoryLimits`:
  - `memory_growing(...) -> anyhow::Result<bool>`
    → `-> wasmtime::Result<bool>`.
  - `table_growing(...) -> anyhow::Result<bool>`
    → `-> wasmtime::Result<bool>`.
- `fn map_trap(skill_id: &SkillId, err: anyhow::Error)`
  → `fn map_trap(skill_id: &SkillId, err: wasmtime::Error)`. Body
  unchanged — `wasmtime::Error: Display`, so the existing
  `format!("{err}")` pattern still works for the fuel/epoch/trap
  string sniffing.

Three call sites (`alloc.call().map_err(...)`,
`run.call().map_err(...)`, instance instantiation) now hand
`wasmtime::Error` straight through. No new conversions, no
behaviour change.

### Audit ignore list

No additions. V144.1 already dropped the stale `RUSTSEC-2026-0002`
entry; the new wasmtime cluster simply disappears from `cargo audit`
and `cargo deny check advisories` once the lock points at 45.0.1.

## Verificación

- `cargo check --features skill-forge` — clean after the wasm.rs
  adapt.
- `cargo check --features full` — clean.
- `cargo test --features skill-forge --lib skill_forge` — 60/60
  pass (includes the `runtime_construction_succeeds` and
  `input_too_large_rejected` ResourceLimiter tests).
- `cargo test --features "full,…,skill-forge" --lib` — 8504 pass,
  0 fail, 1 ignored.
- `cargo fmt --check` — clean.

## Decisiones

- **Toolchain 1.93, not the latest stable**. wasmtime 45's published
  MSRV is 1.93.0. Taking the minimum compatible version keeps the
  next bump's surface small and matches the V141 "bump toward
  margin" pattern (5 mayors of headroom is enough; chasing 1.94+
  is a separate decision).
- **No WASI wiring change**. V143-009 confirmed `wasmtime-wasi` is
  imported but never added to the `Linker`. RUSTSEC-2026-0149
  (path_open bypass) therefore could not have been triggered even
  on the affected versions. The bump still fixes it on principle —
  no exploit story matters here.
- **Skipped Option 3** (temporary ignore list with rationale). The
  runbook art. 3 path was on the table, but the user picked Option
  1 directly: bump rather than defer, ship as V145, no transitional
  V144.x ignore commit.

## Follow-ups

- Sweep for the Node.js 20 deprecation annotations the same CI run
  surfaced (`actions/checkout@v4` running on Node 20). Not in scope
  for V145 — separate workflow tweak.
- Eventual jump to 1.94+ stable when there's a reason (new lang
  feature we want, or another wasmtime bump that needs it). No
  pressure today.
