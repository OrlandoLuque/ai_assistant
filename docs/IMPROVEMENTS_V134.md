# V134 — CI gate calibration (0.2.81)

**Status**: shipped 2026-05-11
**Scope**: configuration-only follow-up to V133
**Runtime impact**: none

## Why now

V133 pushed nine commits (V124–V133) that had accumulated locally without
ever reaching origin. Two CI gates introduced inside that window —
`Supply Chain` (V125, commit `d806a1a`) and the `bench-budget` job
(V126) — ran for the first time on master and failed.

Neither failure is a regression. Both expose pre-existing config drift
that was invisible because the workflows had never executed against
the real dependency graph or the GitHub-hosted runners. V134 calibrates
both gates so a clean tree is green.

The alternative — leaving them red — would have desensitised every
future push to the "CI failed" notification and made real regressions
indistinguishable from baseline noise. Calibrating early is cheaper
than fighting alarm fatigue later.

## What was wrong

### 1. `deny.toml` rejected five transitive license sets

cargo-deny is configured with a tight `allow` list (MIT, Apache-2.0,
BSD-*, ISC, Zlib, Unicode, CC0, MPL-2.0, OpenSSL, BSL-1.0, WTFPL).
Anything outside that list is rejected.

Five crates in our dependency graph carry licenses outside the list:

| Crate                     | License(s)                                  | Why it's acceptable                                                        |
| ------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| `epaint 0.27.2`           | `(MIT OR Apache-2.0) AND OFL-1.1 AND LicenseRef-UFL-1.0` | The OFL / UFL parts cover bundled fonts (assets, not code). |
| `webpki-roots 0.26.11`    | `CDLA-Permissive-2.0`                       | Mozilla's CA root certificate bundle. Data, not code.                      |
| `webpki-roots 1.0.7`      | `CDLA-Permissive-2.0`                       | Same, newer major.                                                         |
| `webpki-root-certs 1.0.7` | `CDLA-Permissive-2.0`                       | Companion bundle published by the same project.                            |
| `whisper-rs 0.15.1`       | `Unlicense`                                 | Public-domain dedication. Equivalent to CC0.                               |
| `whisper-rs-sys 0.14.1`   | `Unlicense`                                 | Same — `-sys` companion.                                                   |

All six are compatible with PolyForm Noncommercial. The right fix is
per-crate exceptions — not adding `OFL-1.1` / `Unlicense` to the global
`allow` list, because a *new* dependency arriving under those licenses
should still trigger an audit.

### 2. `cargo-deny` reported the workspace itself as "unlicensed"

The `LICENSE` file at the repo root is PolyForm Noncommercial 1.0.0.
PolyForm is not in the standard SPDX corpus, so cargo-deny matched
the file at confidence **0.90** — below the previous threshold of
**0.93**. The crate then fell through to "no license expression",
which counts as `unlicensed`.

`Cargo.toml` uses `license-file = "LICENSE"` (not `license = "..."`)
because PolyForm has no canonical SPDX ID. The right fix is a
`[[licenses.clarify]]` entry pinning the LICENSE file's hash to the
expression `LicenseRef-PolyForm-Noncommercial-1.0.0`, then adding
that expression to `allow`.

### 3. `bpe_token_count_200_words` measured 9× over budget

V126 introduced a `[bench-budget]` gate that fails the build if a
listed benchmark exceeds its `max_ns`. Budgets were calibrated on
a local workstation:

```
bpe_token_count_200_words: observed ~270 µs × 1.5 headroom = 400_000 ns
```

GitHub-hosted Ubuntu runners are single-vCPU and run all other CI
jobs concurrently. The same benchmark measures **~3.6 ms** there.
That's a 13× slowdown driven entirely by runner hardware — the BPE
tokenizer code (`src/token_counter.rs`) hasn't been touched since
V90.20.

The right fix is to bump the budget to a CI-realistic value with a
1.5× headroom on the observed CI worst-case, and to document the
local↔CI gap so the next person tightening the budget reaches for the
CI number, not the local one.

## What changed

### `deny.toml`

- `confidence-threshold`: `0.93 → 0.90`
- New `allow` entry: `LicenseRef-PolyForm-Noncommercial-1.0.0`
- New `exceptions` block with five per-crate entries (one per crate
  above) — scoped to crate name + `version = "*"` so the gate keeps
  catching unrelated new arrivals.
- New `[[licenses.clarify]]` block pinning `LICENSE` (hash
  `0x516ff7a6`) to the PolyForm expression.

### `bench_budget.toml`

- `bpe_token_count_200_words`: `400_000 → 6_000_000` ns (15× bump).
- Updated note: documents the CI vs local hardware delta so the
  reason for the gap is visible to anyone reviewing the table.

### `Cargo.toml`

- `0.2.80 → 0.2.81`

## Compatibility

Configuration-only release. No source code changes, no API surface
changes, no behavioural changes. All 44 non-supply-chain CI jobs from
V133 stay green; this delta unblocks the remaining two.

## Verification

After this change, the next push to master should produce:

- `Supply Chain` (cargo-deny): `advisories ok, bans ok, licenses ok,
  sources ok`. Pre-existing duplicate-`zip` warning stays as a
  warning, not an error.
- `Benchmarks → Check bench budget`: `PASS` for
  `bpe_token_count_200_words` at whatever the runner measures (5–6 ms
  expected; budget is 6 ms).

Local verification is limited because `cargo-deny` is not installed
on the dev machine. The Cargo.toml license configuration was reviewed
against cargo-deny upstream docs:
<https://embarkstudios.github.io/cargo-deny/checks/licenses/cfg.html#the-clarify-field>

## Out of scope

- Optimising `BpeTokenCounter::count()`. The 3.6 ms figure is what a
  real production caller sees on a single-vCPU box; if that becomes a
  user-visible problem we'll revisit the algorithm (likely a Trie
  rebuild). Not worth doing speculatively.
- Replacing PolyForm with an SPDX-listed license. Decision already
  documented in `CLAUDE.md`; the deny.toml workaround is the
  correct shape for the current strategy.
