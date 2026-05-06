# V131 — Phase C.4: release automation (final tier-1 cycle)

**Date**: 2026-05-06
**Version**: 0.2.78
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.4
**Tasks**: #340 (V131 C.4 — release automation)

V131 closes the eight-cycle Tier-1 readiness sweep
(C.1 → C.9 — see V125 through V130). After this release the crate
ships the same operational floor expected of competitive
production-ready Rust LLM tooling: pinned toolchain, supply-chain
audit, performance budgets in CI, deprecation policy, encrypted/
signed backup CLI, GDPR right-to-erasure, on-call runbooks, and
a fully-automated release pipeline.

## Why

Before V131 the repo had no release pipeline. A consumer who
wanted to install `ai_cli` had to clone the repo and `cargo build
--release --features full` themselves — long compile, no
binary verification, no SBOM, no signature. The gap also meant
the user's **standing rule** could not be enforced:

> *Every GitHub release must attach the pre-built binary zip +
> SHA-256 sidecars — never release without them.*

Without an automated pipeline that rule lives on the maintainer's
to-do list and gets skipped under pressure. V131 makes it the
default: a tag push *is* the release, and a release that didn't
produce binaries is by construction a CI failure.

## What changed

### `.github/workflows/release.yml` (new)

The release pipeline. Triggered by `v*` tag pushes, with a manual
`workflow_dispatch` `dry_run` flag for testing.

#### Build matrix

| Target | Runner | Archive |
|---|---|---|
| `x86_64-unknown-linux-gnu` | `ubuntu-latest` | `tar.gz` |
| `x86_64-apple-darwin` | `macos-13` (intel) | `tar.gz` |
| `aarch64-apple-darwin` | `macos-latest` (arm) | `tar.gz` |
| `x86_64-pc-windows-msvc` | `windows-latest` | `zip` |

Linux-aarch64 is intentionally **not** in V131 — it requires
either QEMU emulation or a self-hosted arm runner, both of which
add cost / flake without solving a current consumer need. Easy
addition later: one extra matrix row, one extra `cargo install
cross` step.

#### Binaries shipped

Headless only:

```
ai_cli         ai_setup        ai_logs       ai_jobs       ai_backup
ai_local_infer ai_recipes      ai_acp        ai_proxy      ai_breeder
ai_feedback    ai_acp_audit    ai_local_infer_audit
```

GUI binaries (`ai_gui`, `ai_logs_gui`, `ai_recipes_gui`,
`ai_acp_audit_gui`, `ai_local_infer_audit_gui`, `ai_setup_gui`,
`ai_breeder_gui`, `ai_feedback_gui`, `ai_prompt_synth_gui`,
`ai_gui-pro`) are deliberately **not** in the release matrix —
they pull `eframe` and platform-specific windowing deps that
balloon runner time and risk flake. Users who want them build
locally with `cargo build --release --features gui`.

The build loop tolerates per-bin failures with a logged warning,
not a job failure: a fresh feature combination that can't compile
on macOS-arm, say, doesn't kill the whole release. Failed bins
simply don't appear in the archive.

#### Per-archive sidecars

Every archive comes with **four** files:

| Sidecar | Purpose |
|---|---|
| `.sha256` | Integrity. `shasum -a 256 -c <name>.sha256` re-verifies. |
| `.sig` | Cosign keyless signature (sigstore Rekor entry). |
| `.cert` | Cosign-issued ephemeral X.509 certificate, OIDC-bound to this repo's `release.yml` at the tag ref. |

`shasum` and `cosign verify-blob` together let a consumer prove
the artifact came from a `v*` tag push on this repo, signed by
the GitHub Actions OIDC issuer, with bytes intact. No keys need
to be distributed; verification is offline once the cert chain is
trusted.

#### Cosign keyless signing

Uses `sigstore/cosign-installer@v3` and the GH Actions OIDC token
— no key on disk, no key in repo secrets. The signing identity
binds to:

```
sub = repo:<owner>/<repo>:ref=refs/tags/<tag>
iss = https://token.actions.githubusercontent.com
```

A verifier pins this identity in their `cosign verify-blob` call
(see `docs/RELEASE_PROCESS.md` §"Verifying a release"); a
malicious fork or branch cannot satisfy the identity check even
if it produces an otherwise-valid signature.

Cosign signing is conditionally skipped on `workflow_dispatch`
runs (manual). Manual runs may not have OIDC available depending
on runner config; the dry-run path doesn't need a valid
signature anyway.

#### Release publish step

Downloads every per-target artifact, flattens to a single
directory, extracts the V cycle's IMPROVEMENTS doc as the release
body, and calls `softprops/action-gh-release@v2` with
`fail_on_unmatched_files: true` so a missing artifact is a hard
failure — enforces the maintainer's standing "never release
without the binary zip + SHA-256" rule via the build itself.

### Coordination with `supply-chain.yml`

V125 already tag-triggers `supply-chain.yml` which uploads the
CycloneDX SBOM (JSON + XML) to the same release. The two
workflows compose: by the time both finish, the release page
carries `archive + sha256 + sig + cert` per platform plus the
SBOM. No duplication.

### `scripts/check_release_ready.py` (new)

Stdlib-only Python 3.11+ pre-flight check. Verifies four
invariants before a tag is pushed:

1. `Cargo.toml` `version` matches the tag the user is about to
   push.
2. `CHANGELOG.md` carries an `[Unreleased]` entry whose header
   mentions the version. (Catches the V cycle where someone
   bumped the version but forgot the changelog — the release
   body would otherwise be empty.)
3. Working tree is clean (with a short ignore-list for the
   `.claude/settings.local.json` churn that pollutes every
   working tree on this repo).
4. The `--allow-dirty` flag is honoured for CI runs that want to
   skip the working-tree check explicitly.

Smoke-tested on the current repo state — exits 0.

### `docs/RELEASE_PROCESS.md` (new)

The maintainer's release runbook. Covers cadence (patch by
default), pre-flight script, exact commands for cutting a
release, the verification flow consumers should run, and the
rollback policy (delete the GitHub release, ship `+1`, document
in CHANGELOG — never re-tag).

### `docs/IMPROVEMENTS_V131.md`

This file.

### `Cargo.toml`

Version 0.2.77 → 0.2.78. No source change.

## What V131 deliberately does *not* do

* **No `release-plz` config.** `release-plz` is built around
  crates.io publishing flows; the project's intentional posture
  (per `CLAUDE.md`) is that this crate is **not** published —
  not on crates.io, not on a public GitHub. Bringing in
  `release-plz` would add a dep and a config file that exist only
  to be ignored. `cargo-release` is in the same boat. `git tag &&
  git push` is the release primitive; the workflow does the rest.
* **No automated CHANGELOG generation.** Every V cycle writes a
  hand-curated CHANGELOG entry whose authoring quality is the
  whole point — auto-generation from commit messages produces
  release notes that are nearly always worse than the
  IMPROVEMENTS docs we already write.
* **No automatic version bumping in PRs.** Patch-level bumps
  happen at the maintainer's discretion at end of cycle, not
  per-PR.
* **No Linux-arm64 in the matrix.** Add when there's a real
  consumer; today it would only add flake.
* **No GUI binaries in the matrix.** Cross-compile reliability
  for `eframe` is not worth the runner time at this stage.

## Cycle summary (C.1 → C.9 + C.4)

| V | C.x | Title | Version |
|---|---|---|---|
| V125 | C.1 | Supply-chain hardening (deny.toml, SBOM, audit-deny sync, Renovate) | 0.2.72 |
| V126 | C.5 | Active perf budgets (`bench_budget.toml` + CI gate) | 0.2.73 |
| V127 | C.6 | Feature/API lifecycle policy (`docs/FEATURE_LIFECYCLE.md`, deprecation lint) | 0.2.74 |
| V128 | C.7 | `secure_backup` + `ai_backup` CLI (encrypted/signed snapshots) | 0.2.75 |
| V129 | C.8 | GDPR right-to-erasure (`gdpr::purge_user`) + DPIA template | 0.2.76 |
| V130 | C.9 | Operational runbooks (`docs/runbooks/`) | 0.2.77 |
| V131 | C.4 | Release automation (this cycle) | 0.2.78 |
| (V124) | C.3 | OTel adaptive sampler + prompt redaction (pre-Tier-1, included for completeness) | 0.2.71 |

The Tier-1 sweep is complete after this cycle. Anything labelled
"deferred" in V125 (e.g. sigstore signing for V125 artifacts) is
now folded into V131's `release.yml` and applies on the next tag
push.

## Compatibility

Pure additions. No source change beyond version bump. Existing
CI workflows untouched. The only externally-visible change is on
the GitHub release page, which begins to carry signed binaries
for users who want them.
