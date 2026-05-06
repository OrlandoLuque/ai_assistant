# V127 — Phase C.6: feature & API lifecycle policy

**Date**: 2026-05-06
**Version**: 0.2.74
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.6
**Tasks**: #336 (V127 C.6 — feature deprecation policy)

## Why

Pre-1.0 SemVer technically permits any breaking change at minor
bumps, but a single-author library targeting future commercial
licensees can't lean on that escape hatch — silent breakage costs
the user time even when it's "allowed". V127 codifies the
maintenance contract:

1. New feature flags whose API surface isn't yet committed must
   carry an `experimental_*` prefix so the warning is visible at
   the dependency graph level.
2. Any item slated for removal must be `#[deprecated(since = ...)]`
   for at least two patch cycles before being removed, with a
   `note = ` that names the migration path.
3. Each lifecycle transition (graduation, deprecation, removal)
   gets a dedicated `### Feature lifecycle` section in the
   CHANGELOG so a user pinned to an old patch can grep and plan.

The existing codebase has exactly one `#[deprecated]` item
(`AutoApproveAll`) and no `experimental_*` flags, so the policy
applies to all *future* changes. V127 makes the policy explicit and
adds an enforcement gate.

## What changed

### `docs/FEATURE_LIFECYCLE.md`

The full policy document. Defines:

- The three lifecycle states (`experimental_X` → stable `X` →
  `#[deprecated(since)]` → removed) and the transitions between
  them.
- Cargo feature-flag conventions (`kebab-case` for stable,
  `experimental_snake_case` for canaries; canaries cannot stay
  more than two minor cycles).
- `#[deprecated]` requirements: `since = ` and `note = ` are both
  mandatory; removal can only happen ≥ 2 patch versions after
  `since`.
- CHANGELOG conventions: a `### Feature lifecycle` subsection per
  lifecycle-touching release with Graduated / Deprecated / Removed /
  New canary entries.
- What the policy is *not*: not a SemVer override, not retroactive
  on existing flag names, not a substitute for testing.

### `src/agent_policy.rs`

The existing `AutoApproveAll` deprecation grew a `since = "0.2.74"`
field plus a pointer to the lifecycle doc. This is now the
reference example for the convention; future `#[deprecated]`
attributes copy this shape.

```rust
#[deprecated(
    since = "0.2.74",
    note = "Use an explicit ApprovalHandler in production — \
            AutoApproveAll bypasses all safety checks. \
            See docs/FEATURE_LIFECYCLE.md."
)]
pub struct AutoApproveAll;
```

### `scripts/check_deprecation_policy.py`

Stdlib-only Python 3.11+ scanner. Walks `src/**/*.rs`, finds every
`#[deprecated(...)]` attribute (multi-line attribute syntax handled
by tracking bracket/paren depth), and fails the build if any of
them is missing `since = "..."` or `note = "..."`.

Output is plain ASCII — pass and fail messages render cleanly on
every CI runner regardless of locale.

Reports both successes and failures with file path + line number;
the failure message names exactly which fields are missing per
attribute.

### `.github/workflows/ci.yml`

New `deprecation-policy` job sits next to `fmt` in the lint band.
Runs on every push and PR; uses the runner's default Python 3
(no extra setup-python action). One step:

```yaml
deprecation-policy:
  name: Deprecation policy
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run deprecation policy checker
      run: python3 scripts/check_deprecation_policy.py --root src
```

The job is required (default `continue-on-error: false`) so a PR
that introduces a malformed `#[deprecated]` attribute fails CI
loudly.

## Compatibility

- Pure additions plus one annotation update on `AutoApproveAll`.
  No behaviour change.
- The `since = "0.2.74"` value on `AutoApproveAll` reflects
  *when the policy was applied*, not when the deprecation was
  originally announced — for a one-off retrofit on a new policy
  this is the simplest convention. Future deprecations use the
  version number of the introducing PR.

## What's next

- V128 / C.7 — Backup/restore CLI with AES-256-GCM + SHA256 +
  signature, tar.zst archive format.
