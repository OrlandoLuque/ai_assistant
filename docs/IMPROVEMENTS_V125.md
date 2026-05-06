# V125 — Phase C.1: supply-chain hardening (cargo-deny + SBOM + Renovate)

**Date**: 2026-05-06
**Version**: 0.2.72
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.1
**Tasks**: #334 (V125 C.1 — supply-chain security)

## Why

C.1 calls for `cargo-audit` (already shipping in `ci.yml`),
`cargo-deny` with a license allowlist, an SBOM in CycloneDX format,
a pinned Rust toolchain, and a managed dependency-update policy. The
existing CI ran `cargo audit` only — that catches advisories but does
nothing about license drift (a single GPL transitive contaminates a
future commercial dual-license off PolyForm Noncommercial), nor does
it produce an artifact that downstream consumers can use to verify the
graph they're consuming.

## What changed

### `rust-toolchain.toml`

```toml
[toolchain]
channel = "1.90.0"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

`rustup` and `cargo` honour this file automatically — every developer
and every CI runner now resolves to the same toolchain version
without per-job `dtolnay/rust-toolchain@1.90.0` repetition. Bumping
the channel is now a single-line PR.

### `deny.toml`

`cargo-deny` configuration covering four concerns the existing
`cargo audit` job didn't address:

- **Advisories**: same RustSec database as `cargo-audit`, but applied
  in one pass over the full lock graph. Ignore list mirrors the CI
  audit job's ignore list (synchronisation is verified by the new
  `audit-deny-sync` job below).
- **Licenses**: explicit allowlist of permissive licenses
  (MIT/Apache-2.0/BSD/ISC/Zlib/CC0/MPL-2.0/Unicode/BSL/WTFPL).
  Anything outside the list is denied. PolyForm Noncommercial 1.0.0
  on the wrapper crate is incompatible with copyleft, so even one
  GPL/AGPL/SSPL transitive would block a future commercial
  dual-license — the deny list catches that drift at PR time.
- **Bans**: `wildcards = "deny"` (no `serde = "*"`-style entries),
  `multiple-versions = "warn"`, no specific banned crates yet.
- **Sources**: `unknown-registry = "deny"`, `unknown-git = "deny"` —
  every dependency must come from crates.io (or a future explicit
  allow-listed git remote).

### `.github/workflows/supply-chain.yml`

New workflow with four jobs, separated from `ci.yml` so a
supply-chain failure produces an obvious red signal without blocking
the full feature-matrix run:

| Job | What it does |
|---|---|
| `cargo-deny` | Runs `cargo deny check advisories licenses bans sources`. |
| `cargo-audit` | Mirror of the existing `ci.yml` audit job. Kept here so the supply-chain workflow is self-contained. |
| `audit-deny-sync` | Bash script that extracts `RUSTSEC-*` IDs from `ci.yml`, this workflow, and `deny.toml`'s `[advisories].ignore`, asserts all three lists agree. Catches drift when one place silences an advisory and the other doesn't. |
| `sbom` | Generates CycloneDX 1.4 JSON + XML via `cargo-cyclonedx`. Uploaded as a 90-day artifact on every run; on tag pushes (release flow), attached to the GitHub release alongside the binary zip. |

Triggered on push, PR, and a Monday 06:00 UTC schedule so the
advisory database picks up weekend updates without waiting for the
next push.

### `renovate.json`

Renovate configuration for managed dependency updates:

- Weekly schedule (Monday 06:00 UTC) so PRs land on a predictable
  cadence and bunch nicely with the supply-chain cron.
- Vulnerability alerts run "at any time" with the `security` label
  and direct assignee (`@OrlandoLuque`) so they don't queue behind
  the weekly batch.
- Grouping: rustls + quinn + webpki (cross-dependent); serde stack;
  tokio + futures; dev/build deps separately. One PR per group keeps
  the review surface bounded.
- `lockFileMaintenance` enabled — `Cargo.lock` refresh on the same
  weekly cadence catches transitive bumps Renovate itself doesn't
  surface.
- `dtolnay/rust-toolchain` action explicitly *disabled* for
  Renovate — the channel bump is a human review concern (toolchain
  upgrade can change clippy lint defaults and codegen).

## What's still TODO under C.1 (deferred)

- **Sigstore / cosign signing**: requires key-pair issuance and a
  trust-store decision (Fulcio root vs static key). Will be picked
  up alongside C.4 release automation (V131) where the signing fits
  naturally into the release pipeline.
- **`--locked` enforcement in release builds**: today the `release`
  job in `ci.yml` runs `cargo build --release` without `--locked`.
  V125 doesn't change that to keep the diff scoped; will be added
  in V131 (release automation).

## Compatibility

- All four artifacts (`rust-toolchain.toml`, `deny.toml`,
  `supply-chain.yml`, `renovate.json`) are pure additions. No code
  paths change, no test counts change.
- `rust-toolchain.toml` is honoured by `rustup` automatically.
  Existing CI jobs that pin `dtolnay/rust-toolchain@1.90.0` continue
  to work; they simply land on the same version twice.
- `cargo-deny` and the SBOM generator install on demand inside the
  job; no addition to `Cargo.toml`.

## What's next

- V126 / C.5 — Performance budgets active (`bench_budget.toml` +
  CI regression gate).
