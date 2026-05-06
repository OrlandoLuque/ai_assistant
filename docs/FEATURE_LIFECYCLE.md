# Feature & API Lifecycle Policy

**Owner**: Orlando Luque (sole maintainer)
**Last reviewed**: 2026-05-06 (V127, 0.2.74)
**Authority**: this document is the policy reference for any change
that adds, renames, or removes a Cargo feature flag or a public
item.

---

## Why this exists

`ai_assistant` is a single-author library at 0.2.x. Pre-1.0 SemVer
allows breaking changes on every minor bump (0.2.x → 0.3.0), but
silent breakage still costs the future commercial-licensee user
their time. This policy makes change visible: a user who pins
`0.2.x` should be able to read the CHANGELOG and know exactly which
flags or items are on the way out, when they were announced, and
what to migrate to.

The same discipline also makes the eventual 1.0 cut smaller —
items already labelled `experimental_*` or marked `#[deprecated]`
can graduate or disappear cleanly without a separate audit pass.

## The three states

```
                ┌──────────────────┐
                │  experimental_X  │  (canary — may break)
                └─────────┬────────┘
                          │ graduates (drop prefix)
                          ▼
                    ┌─────────┐
                    │    X    │  (stable)
                    └────┬────┘
                         │ supersession or removal
                         ▼
              ┌─────────────────────┐
              │ #[deprecated(since)] │  (announced removal)
              └──────────┬───────────┘
                         │ at least 2 patch cycles
                         ▼
                       removed
```

### State 1 — `experimental_*` (canary)

Use the prefix for any **new Cargo feature flag** whose API surface
is not yet committed:

```toml
[features]
experimental_voice_clone = ["audio-io"]
```

- Compiles and ships like any other flag.
- Covered by tests.
- May be **renamed, redesigned, or removed without deprecation
  cycle** in any patch release. The prefix is the warning.
- A flag must not stay in this state for more than **two minor
  cycles** (~6 months on current release cadence). At that point
  either drop the prefix (graduate) or remove it.

When to use the prefix:
- New providers that haven't been used outside of synthetic tests.
- Subsystems with surface areas still in flux (sub-agent role
  grammar, scheduler trigger DSL, etc.).
- Anything that depends on an upstream crate that is itself
  pre-1.0.

### State 2 — stable (`X`)

Default state for everything currently shipped. Breaking the public
surface of a stable feature requires either:

1. A `0.x → 0.(x+1)` minor bump *and* a deprecation cycle for
   the broken item (preferred), **or**
2. A direct removal in a minor bump *only* if the item was already
   `#[deprecated(since = ...)]` for at least two patch cycles.

Adding fields to a public struct is allowed at patch level if the
struct is `#[non_exhaustive]`; otherwise it must wait for a minor.

### State 3 — `#[deprecated]`

Any public item slated for removal **must** carry:

```rust
#[deprecated(
    since = "0.2.73",
    note = "Use FooBar instead — see docs/FEATURE_LIFECYCLE.md."
)]
pub struct OldThing;
```

- `since = ` is required and points to the version that announced
  the deprecation (not the version of removal).
- `note = ` must name a replacement or a migration path. "Will be
  removed" is not enough — a user reading rustc's warning needs
  to know what to do.
- Removal cannot happen earlier than `since + 2 patch versions` and
  must coincide with at least a patch bump (preferably a minor).

The reference example in this codebase is
`AutoApproveAll` in `src/agent_policy.rs`.

### Removal

Each removal requires a CHANGELOG entry under `### Removed` that
quotes the original `since` line so a user grepping the file can
trace announcement → removal.

## Cargo feature flag rules

Beyond the `experimental_` prefix:

- **Name format**: `kebab-case` for stable flags
  (`audio-io`, `multi-agent`); `snake_case` with `experimental_`
  prefix for canaries.
- **No silent removal**: a flag must appear in CHANGELOG under
  `### Removed` in the patch that drops it.
- **Implies-graph**: a flag's `dependencies = ["a", "b"]` list is
  part of its public surface. Adding to it is non-breaking;
  removing from it (or changing it to a minimal subset) requires
  a deprecation cycle.
- **Default features**: changes to the `default = [...]` list
  follow the same deprecation cycle as removing a flag — a user
  with `default-features = false` may rely on a flag *not* being
  in the default set, or vice versa.

## CHANGELOG conventions (V127 onward)

Every release entry that touches lifecycle gets a dedicated section,
in addition to the usual Added / Changed / Fixed / Removed groups:

```markdown
### Feature lifecycle
- **Graduated**: `experimental_voice_clone` → `voice-clone`. API frozen.
- **Deprecated**: `AutoApproveAll` (since 0.2.73) — use `ApprovalHandler`.
- **Removed**: `OldRagBackend` (announced 0.2.70, see CHANGELOG v83).
- **New canary**: `experimental_lattice_routing` — sub-agent topology
  rewrite, may change shape over the next two minor releases.
```

Releases that don't change any lifecycle state may omit this
section.

## Enforcement

- `scripts/check_deprecation_policy.py` runs in CI under the
  `lint` job (V127). It scans `src/**/*.rs` for any
  `#[deprecated(...)]` attribute and fails the build if it lacks
  a `since = "..."` field. The `note = ` field is also required.
- `cargo metadata` plus a manual review at each minor bump
  catches `experimental_*` flags that have outlived the two-minor
  window.

## What this policy is NOT

- Not a SemVer override. The library is 0.2.x and breaking
  changes at minor are still permitted by SemVer; this policy
  layers a *self-imposed* announcement requirement on top.
- Not retroactive on flag names. Existing stable flags keep their
  current names. The `experimental_` convention applies only to
  flags introduced from V127 onward.
- Not a substitute for testing. A deprecated item still ships
  working code; the deprecation marker is a contract about its
  future, not its present.
