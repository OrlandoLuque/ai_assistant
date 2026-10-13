# Runbook: handling a RUSTSEC advisory

**Severity**: depends on the advisory. The audit gate fails → start here.
**Owner**: maintainer (this is a one-person project; on-call is the author).
**Last reviewed**: 2026-10-13 (V401 / V402).

This runbook covers two situations: (a) a new RUSTSEC advisory has
just appeared in CI, and (b) the monthly review issue opened by
`.github/workflows/rustsec-review-monthly.yml` is sitting in the
queue and needs to be processed.

## 1. Symptoms

* CI fails on the `cargo-deny` or `cargo-audit` step.
* A new issue with label `supply-chain` + `monthly-review` was just
  opened by the bot.
* `cargo audit` locally surfaces an ID not in `deny.toml`.

## 2. Decision tree

```
new RUSTSEC ID in the audit output
        │
        ▼
 Is there a fixed version of the affected crate?
        ├── yes ──► bump the dep (direct or transitive). PR + done.
        │
        ▼ no
 Does our code path actually trigger the bug?
        ├── yes ──► patch our usage to avoid it (workaround),
        │          or pin to a non-affected version, or drop the dep.
        │          Document in CHANGELOG + commit.
        │
        ▼ no
 Add to `deny.toml#advisories.ignore` AND to the two `cargo audit`
 invocations (ci.yml + supply-chain.yml). The `audit-deny-sync`
 job will reject the PR if the three lists drift.

 The entry MUST include in a comment:
    * Why it doesn't apply (one or two sentences), MEASURED --
      see step 3 on how, because five of eight were wrong.
    * A `Re-check by:` DATE. Not optional, and not replaceable by
      an upstream event: `scripts/check_recheck_dates.py` fails CI
      on any date in the past.
    * Optionally, the upstream event that would let the entry be
      DELETED rather than renewed ("when crate X publishes Y").
```

## 3. Adding an ignore — the policy

A `deny.toml` entry without comment + `Re-check by:` date fails
review, and now fails CI. Example of an **acceptable** entry:

```toml
# tantivy → lru IterMut unsoundness. The affected codepath
# (mutating an entry obtained from IterMut while the iterator is
# alive) is not used by tantivy's query/indexing paths we exercise.
# Re-check by: 2027-02-15 OR when lancedb/tantivy bump lru >= 0.16.
"RUSTSEC-2026-0002",
```

Example of an **unacceptable** entry:

```toml
# upstream issue, ignore
"RUSTSEC-XXXX-NNNN",
```

### Measure the reason. Do not reason about it.

This is the part the policy used to leave implicit, and it cost
five of the eight entries (V401, 2026-10-13):

* **Ask the resolved graph: `cargo tree --all-features -e features
  -i <crate>`.** `cargo metadata` and a crate's `[dependencies]`
  table list what is *declared*, optional-and-switched-off
  included, so they will happily name a crate that has nothing to
  do with your build. The `quick-xml` entry blamed a Linux GUI
  file-dialog chain parsing trusted build-time XML; the resolved
  graph puts the crate there through our own `tabular-polars`
  feature → `object_store` with `cloud` and `http` enabled, i.e.
  XML off the network. The whole trust argument rested on the
  wrong crate.
* **Count the dependants before keying a re-check on one of
  them.** `paste`'s trigger was "when egui drops the paste dep".
  It has 22 dependants and egui is not among them, so the trigger
  could never fire. `ttf-parser`'s said the same about egui while
  `lopdf ← pdf-extract` holds a second, independent root.
* **Check whether the crate is really transitive.**
  `rustls-pemfile` was documented as "only present transitively
  via reqwest's TLS backend". We declare it ourselves, for
  `server-tls`, and reqwest is not in the chain at all.
* **A suppression whose written reason is false looks reviewed, so
  nobody re-reads it.** That is the whole failure mode. It is
  cheaper to measure the reason once than to carry a plausible one
  for a year.

### Renewing moves the date

A re-check does **not** add a note under an expired deadline — it
sets a new one. V385 re-checked `lru` properly, recorded what it
measured, and left `Re-check by: 2026-09-01` in place; the entry
then read as freshly reviewed while being permanently overdue.
Keep the history of what was measured, and put the new date last:
the gate reads the LAST date in the block.

## 4. Monthly review — processing the issue

The bot opens an issue listing every ignored entry with one of two
markers:

* **STILL ACTIVE — verify ignore is still justified**: `cargo audit`
  still reports this advisory. For each, open the advisory, read it
  again, re-read the justification in `deny.toml`, and decide:
  * **keep**: comment in the issue "kept — still applies, re-check
    by <date>". Update the comment in `deny.toml` if the trigger
    has slipped.
  * **fix**: bump the dep. Remove the ignore from `deny.toml` +
    both `cargo audit` invocations + commit.
* **NO LONGER REPORTED — consider removing**: `cargo audit` does
  not report this advisory anymore (typically because the
  vulnerable version has been pruned from the lockfile by a
  transitive bump). Remove the ignore from all three places. The
  `audit-deny-sync` job will reject the PR if any one of the three
  still references the ID.

The bot also lists **new advisories** not yet in `deny.toml`. Treat
each as a fresh entry to step 2's decision tree.

Once every entry has been processed: close the issue. The next
monthly run will open a new one.

> **This step was not performed for five months.** The workflow ran
> on the 1st of June, July, August, September and October 2026 and
> opened its issue every time; all five were still open on
> 2026-10-13, and six of the eight re-check dates had expired with
> one re-check recorded between them. The bot is not broken — an
> issue in the tracker of a one-developer repository is not a
> forcing function.
>
> So the mechanical half moved into CI
> (`scripts/check_recheck_dates.py`, which fails on an expired
> date) and this issue is now for the half a script cannot do:
> re-reading the advisory and re-measuring the justification. If
> five identical issues pile up again, the answer is not more
> discipline — it is to make the workflow update ONE issue instead
> of opening a new one each month.

## 5. Mitigate (CI red, fix not ready)

If the CI is blocking a release and the proper fix needs time:

* Add the ID to all three places (`deny.toml`,
  `ci.yml`, `supply-chain.yml`) with a comment marking it
  `TEMPORARY — expected fix in <PR/version>` and a re-check date no
  more than 4 weeks out.
* File a follow-up commit/PR to land the proper fix before the
  re-check date.

This unblocks merging without losing track, and what keeps it
from being lost is now the CI gate rather than the monthly nag:
`scripts/check_recheck_dates.py` fails the `Supply Chain` workflow
the day the date passes. (This paragraph used to say "the monthly
review will nag if the temporary entry survives past its
re-check". It did nag. Nagging is not the same as stopping the
work, which is the lesson of 2026-10-13.)

## 6. Postmortem

If a RUSTSEC ended up affecting us in production:

* Record: which ignore was wrong, when the advisory was first
  available, what the operational impact was.
* **"When it was first available" is not the date `cargo audit`
  showed it.** The five `quinn-proto` advisories of 2026-10-12 are
  dated 2026-08-17 and 2026-09-14 and entered the RustSec database
  on 2026-10-12 at 20:05, six of them in one commit (`ffa3dd5a`).
  For nearly two months `cargo audit` printed `ok` with five real
  denial-of-service advisories in the lockfile. A green audit means
  "nothing in the lockfile matches RustSec", not "no known
  vulnerabilities" — check the advisory's own `date` field against
  when the file appeared in the database (`git log --diff-filter=A`
  inside `~/.cargo/advisory-db`).
* Tighten the policy in this runbook if the failure was structural
  (e.g. "we missed it because the dep was transitive 5 levels deep
  — add cargo-deny `--graph-depth 10` to the audit job").

## Related

* `deny.toml#advisories.ignore` — the canonical ignore list.
* `.github/workflows/ci.yml` (audit job) — gating per PR.
* `.github/workflows/supply-chain.yml` — gating + weekly schedule.
* `.github/workflows/rustsec-review-monthly.yml` — monthly nag.
* `scripts/check_recheck_dates.py` — the gate that fails CI on an
  expired `Re-check by:` date. Note it can turn the build red with
  no code change, the day a deadline passes; inside 21 days of a
  deadline it warns instead.
* `scripts/check_deny_graph.py` — `deny.toml` must walk the whole
  feature graph (N146); without it `cargo deny` checked 299 of 955
  crates and printed the same `ok`.
