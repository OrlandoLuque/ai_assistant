# V148 — codecov-action v4 → v6 (Node 20 follow-up)

**Version:** 0.2.98
**Date:** 2026-06-08
**Trigger:** V146 follow-up. Re-reading the action manifest while
closing out the V146 docs surfaced a misclassification: V146 listed
`codecov/codecov-action@v4` as "composite", but the action's own
`action.yml` at `ref=v4` declares `using: 'node20'`. It was a
Node 20 action all along — and survived the V146 sweep.

## Why

Two reasons to bump now:

1. **Node 20 deprecation.** V146 missed this one. The 2026-06-16
   runner cutover would have surfaced the deprecation warning here
   too. Closing the hole before the deadline.
2. **Currency.** v4 dates from early 2024; v6 is the current
   stable line.

## Action manifest evidence

```
$ gh api repos/codecov/codecov-action/contents/action.yml?ref=v4 | base64 -d | grep -A2 '^runs:'
runs:
  using: 'node20'
  main: 'dist/index.js'

$ gh api repos/codecov/codecov-action/contents/action.yml?ref=v6 | base64 -d | grep -A2 '^runs:'
runs:
  using: "composite"
  steps:
```

v6 is genuinely composite (shell-driven, no bundled JS runtime),
so the Node 20/24 question disappears entirely after this bump.

## Why v6 and not v7

| Tag | Published | Notes |
|---|---|---|
| v7.0.0 | 2026-06-07 01:47Z | GPG key migration (codecovsecurity → codecovsecops). <24h old at time of writing. |
| v6.0.2 | 2026-06-07 02:47Z | Hotfix released **1 hour after** v7.0.0. Signal that the v6 line is being maintained in parallel. |
| v6.0.0 | 2026-03-26 | Introduces Node 24 support in the bundled CLI. ~10 weeks soak time. |

Sticking to v6 for soak. v7 can roll in a separate patch once it
has a week or two of real-world usage.

## Files touched

- `.github/workflows/ci.yml` (1 occurrence, line 191)
- `Cargo.toml` (0.2.97 → 0.2.98)
- `CHANGELOG.md`
- `docs/IMPROVEMENTS_V148.md` (this file)

## Verification

Local:
- `grep` sweep confirmed no remaining `codecov/codecov-action@v4`.
- v6 `action.yml` confirmed `using: "composite"` (no Node runtime
  in the action itself).

CI:
- Push to master and confirm the coverage upload step still
  succeeds with v6 inputs (`files:` and `fail_ci_if_error:` are
  unchanged across the v4→v6 API).

## Design decisions

- **Why a separate patch and not a V146.1?** V146 already shipped
  and the misclassification is a real correction, not a typo.
  Numbering it independently makes the audit trail clearer:
  V146 = the sweep, V148 = the one we missed.
- **Why not include the `contributor-assistant` upstream bump?**
  No Node 24 successor exists yet. Still tracked as a V146
  follow-up.

## Follow-ups

- `codecov/codecov-action@v6` → `@v7` once v7 has ~2 weeks of soak
  time. Pure GPG/account migration per the v7.0.0 release notes,
  so should be a clean patch.
