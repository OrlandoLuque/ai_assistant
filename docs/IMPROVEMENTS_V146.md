# V146 — Node 20 → 24 Action Sweep

**Version:** 0.2.96
**Date:** 2026-06-03
**Trigger:** CI annotation on the V145 push warning that
`actions/checkout@v4` runs on Node 20, force-migrated to Node 24 on
**2026-06-16** (13 days away).

## Why

GitHub announced the Node 20 deprecation in
[Sep 2025](https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/).
The two dates that matter:

- **2026-06-16** — runners switch the default to Node 24. Any
  action still pinned to Node 20 starts failing unless its workflow
  sets `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` or
  `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true`.
- **2026-09-16** — Node 20 is removed from the runner image. No
  opt-out left.

V145 deferred this as out of scope — it was a runtime bump (rustc
+ wasmtime), not a workflow bump. This patch ships the workflow
side before the 06-16 deadline.

## Scope

Workflow YAML only. Zero changes to lib, tests, docs (apart from
this file + CHANGELOG + the website concepts card).

## Action-by-action analysis

Versions confirmed by reading each action's `action.yml` at the
requested tag via `gh api repos/<owner>/<repo>/contents/action.yml?ref=<tag>`:

| Action | Old | New | Notes |
|---|---|---|---|
| `actions/checkout` | `v4` (node20) | `v5` (node24) | 16 occurrences |
| `actions/upload-artifact` | `v4` (node20) | `v6` (node24) | v5 is **still** node20 — first node24 line is v6 |
| `actions/download-artifact` | `v4` (node20) | `v7` (node24) | v5 and v6 are **still** node20 — first node24 line is v7 |
| `actions/github-script` | `v7` (node20) | `v8` (node24) | 1 occurrence |
| `softprops/action-gh-release` | `v2` (node20) | `v3` (node24) | v3.0.0 release notes: pure runtime bump, no API changes |

The `using: node24` line was verified for **every** target tag
before the bump landed — no guesswork.

### Untouched (intentional)

| Action | Status | Why |
|---|---|---|
| `Swatinem/rust-cache@v2` | already node24 | floating major tag tracks the latest |
| `sigstore/cosign-installer@v3` | composite | no Node 20 issue |
| `EmbarkStudios/cargo-deny-action@v2` | docker | no Node 20 issue |
| `codecov/codecov-action@v4` | composite | no Node 20 issue (composite actions run shell, not a bundled JS runtime) |
| `contributor-assistant/github-action@v2.6.1` | node20, **no successor** | latest upstream tag is the same `v2.6.1` from 2024-09. Vendor must ship a Node 24 line. **Tracked as a follow-up.** |

The CLA workflow (`cla.yml`) is the only place
`contributor-assistant/github-action@v2.6.1` is referenced. It
runs on pull-request events from external contributors — we have
none today (project is single-author, unpublished). So the worst
case is: after 2026-09-16, CLA gating breaks for the (zero)
inbound PRs. Acceptable carry.

## Files touched

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- `.github/workflows/supply-chain.yml`
- `.github/workflows/rustsec-review-monthly.yml`
- `Cargo.toml` (0.2.95 → 0.2.96)
- `CHANGELOG.md`
- `docs/IMPROVEMENTS_V146.md` (this file)
- `../ai_assistant-website/concepts.html` (card #314)

## Verification

Local:
- `grep` sweep confirmed no remaining `actions/(checkout|upload-artifact|download-artifact|github-script)@v4` or `softprops/action-gh-release@v2`.

CI:
- Push to master and watch for the Node 20 deprecation `##[warning]`
  annotation disappearing from job logs.

## Follow-ups

- `contributor-assistant/github-action` — re-check upstream
  quarterly; bump as soon as a Node 24 tag exists. Add to the
  rustsec-review-monthly checklist or a separate vendor-watchlist
  doc.
- `codecov/codecov-action` is composite but the floating `@v4`
  pin should track to `@v5` for currency. Low priority — separate
  patch.

## Design decisions

- **Why latest major and not the latest patch?** Workflow files
  use floating major-version tags (`@v5`, not `@v5.1.2`) to ride
  bug fixes automatically. This is consistent with how the repo
  already pins everything else.
- **Why bump in one patch and not split per-action?** The
  deprecation deadline is shared. Splitting buys nothing and
  multiplies the CI traffic.
- **Why no `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` env var as
  a transitional fix?** The bump is cheaper and removes the
  warning today instead of papering over it. The transitional
  flag would be the right call only if some action had no Node
  24 successor — `contributor-assistant` is exactly that case,
  and we explicitly chose to carry the warning there rather than
  add a global override flag.
