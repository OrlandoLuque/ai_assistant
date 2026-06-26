# IMPROVEMENTS_V164 — sweep unnecessary `#[allow(dead_code)]`

**Version:** 0.2.115 → 0.2.116
**Scope:** 8 source files (3 lib, 5 bins)
**Feature:** none new

## Why

The code-quality audit's last open item: the crate had 59
`#[allow(dead_code)]` attributes. Most are legitimate (reserved struct
fields, feature-gated-out items, future-use constants), but blanket
`allow(dead_code)` can also hide genuinely-orphaned code. V161 flagged a
"targeted sweep to confirm they're feature-gate artifacts, not orphaned
code"; V164 does it.

## What changed

Each `#[allow(dead_code)]` was removed and the result checked under the
combos that matter, using clippy `-D warnings` as the oracle:

- lib items → `FEATURES_STD` **and** default features;
- bin items → the bin's own `required-features` (e.g. `ai_proxy` with
  `server-axum,security,server-axum-tls`, `ai_gui-pro` with `gui-pro`,
  `ai_breeder` with `prompt-breeder`).

If removing an allow produced a dead-code warning, it was **restored**
(the item really is dead under that build → the allow is doing its job).
Otherwise it stayed removed (the item is used → the allow suppressed
nothing).

**Result: 23 removed, 36 kept** (only attribute lines changed — no item
was deleted, no new allow added).

### Removed (23) — these suppressed nothing
- lib (3): `server_axum.rs` (a `pub` struct — public items never warn),
  `home_automation/mqtt_backend.rs` (the field is read in a `Debug` impl),
  `skill_forge/declarative.rs` (an `_`-prefixed fn — never warns).
- `ai_proxy` (11): items that are now genuinely used (error kinds, routing
  section, cache/audit fields, …) plus inner allows already covered by the
  module-level allows that were kept.
- `ai_gui-pro` (5), `ai_gui` (2): tab enums / `file_path` / `is_kpkg`
  that are now used.
- `ai_breeder` (1), `ai_recipes` (1): `_`-prefixed link-marker fns.

### Kept (36) — genuinely necessary
Reserved struct fields (audio priority protocol, cluster, redis backend,
fault injection), never-constructed enum variants (constrained-decoding
`JsonContext`, `CircuitState::HalfOpen`), reserved policy consts
(`event_source`), deserialize-only request fields (`server.rs`), a couple
of genuinely-dead `pub(crate)` helpers and test markers, and the
module-level allows in `ai_proxy`/`ai_gui*` that cover their inner items.

## Note: a pre-existing, unrelated feature-combo break

While building feature combos for verification, the sweep surfaced a
**pre-existing** compile break (not touched/caused by V164): enabling
`server-axum` + `eval-suite` together fails because
`src/server_axum.rs:2636` references `crate::eval_suite::EvalGenerator`,
which doesn't exist under that combo. It does **not** affect CI
(`FEATURES_STD` does not enable `server-axum`). Tracked for a separate
fix.

## Tests

Verified: rustfmt clean; clippy `-D warnings` clean for the lib + bins
(FEATURES_STD, plus per-bin builds of `ai_proxy`, `ai_gui`, `ai_gui-pro`,
`ai_breeder`); `ai_test_harness --all` 585/585.
