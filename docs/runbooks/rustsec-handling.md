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
* `cargo audit` locally surfaces an ID not in `deny.toml` -- but only if
  the local database is current, which by default it is **not** on this
  machine. See "The local database, and how to make it current" below.
* **The same commit has two different verdicts.** CI clones the advisory
  database fresh on every run, so the `push` run and a later `schedule`
  run of the *same SHA* can disagree. Measured on V493: the push run
  (00:25Z) reported one vulnerability, the scheduled run sixteen hours
  later reported two. **A red you have already diagnosed is not a red you
  have finished reading** -- and because a `schedule` failure interrupts
  nobody, nothing will tell you it grew.

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
# webbrowser 0.8.15: command injection via a crafted URL. Exactly one
# dependant, egui-winit 0.27.2, requiring `^0.8.3`; the fix is 1.2.2,
# which that `^` cannot reach. Unix-only, and it needs an attacker who
# already controls the process environment; the egui features are not
# in `full`.
# Re-check by: 2027-03-15 OR when the egui stack is upgraded past 0.27
# (tracked as N37) — then DELETE this entry, do not renew it.
"RUSTSEC-2026-0257",
```

That example is live, and it is deliberately one whose reason was
**re-measured and survived unchanged** (V401) — of eight entries audited
that day, it was the only one. The previous example here was
`RUSTSEC-2026-0002`, held up as the model entry until V493 deleted it:
see below.

Example of an **unacceptable** entry:

```toml
# upstream issue, ignore
"RUSTSEC-XXXX-NNNN",
```

### When `cargo-deny` and `cargo-audit` disagree, neither is broken

V494 (2026-11-24) hit this and it looks alarming: `cargo-deny` reported
**one** vulnerability and `cargo-audit` reported **two**, on the same
tree, in the same run, with the ignore-list sync job passing. The lists
had not drifted. The two tools answer different questions:

| tool | what it reads | scope |
|---|---|---|
| `cargo-audit` | `Cargo.lock`, whole | every crate the lock names (1078) |
| `cargo-deny` | a resolved dependency graph | filtered by **features** and by **target** |

The crate only `cargo-audit` saw was `cookie`, pulled by `cookie_store`,
which is an **optional** dependency of `ureq` behind its `cookies`
feature -- and this crate declares `ureq = { features = ["json"] }`. So
`cookie` is in the lock without being in any compilation. Cargo says so
itself if you read the build log: ``Downloaded cookie v0.18.3`` followed
by ``Finished `dev` profile ... in 0.93s``. It downloads what the lock
names and compiles what the graph needs.

**Two absences, two different messages, and do not confuse them.**
`cargo tree -i` answers this directly, and its wording is the
diagnostic:

| query | cargo says | means |
|---|---|---|
| `-i lru` (deleted in V493) | `error: package ID specification 'lru' did not match any packages` | not in the lock at all |
| `-i cookie`, even with `--target all --all-features` | `warning: nothing to print.` | **in** the lock, and no path from this crate reaches it under any feature or target |

The first is the strongest absence claim available without compiling.
The second is one notch weaker in form and just as conclusive for
whether the code is built: nothing we can enable gets there. Both are
stronger than "we do not call it", which only speaks about today's
source. Note that `cargo tree` appends a hint about narrowing `--target`
to the second message even when `--target all` was passed, so the hint
is not a sign you asked wrong.

**Which one decides?** For whether CI goes red: `cargo-audit`, because
it reads the lock. For whether the vulnerable code is reachable:
`cargo-deny`'s narrower graph, and "the feature is not enabled" is
stronger evidence than "we do not call it" -- the code is not compiled
at all.

**The trap is reading deny's silence as absolution.** It means "not in
my graph", not "not in what you ship". Ask both, and when they differ,
say which question you are answering.

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

### Deleting rather than renewing — and why the trigger is usually wrong

Worked example, V493 (2026-11-23): the two `lru` entries were deleted,
not renewed, because the crate left the graph. What makes it worth
writing down is **how the condition came true**.

Both written triggers said the same thing: *when `tantivy` bumps `lru`
>= 0.16*. Neither of them happened. What happened is that **`lancedb`
stopped using `tantivy`** — the chain was
`lru ← tantivy 0.24.2 ← lance/lance-index ← lancedb 0.26.2`, and
`lancedb` is now 0.41.0 with `tantivy` absent from the lock entirely.
Two advisories lost their subject at once, for a reason stated in
neither entry.

So, two rules that cost nothing and would have caught it:

* **Name the link whose disappearance ends the entry, not only the
  version bump you expect.** A trigger phrased as "when X bumps Y" is
  unfalsifiable by the far more common event: somebody upstream drops
  X. "when `lru` leaves the lock" is checkable in one command and does
  not predict the mechanism.
* **The instrument that decides deletion is `Cargo.lock`, not
  `cargo audit` and not `cargo deny`.** The lock is not filtered by
  features or by target — it carries all of them — so zero matches for
  `name = "X"` is the strongest absence claim available without
  compiling: no feature combination and no target can pull it in. If the
  crate is absent, no advisory about that crate can fire.

And the trap on this machine, which nearly deleted the wrong entry:
`cargo deny check advisories` prints
`warning[advisory-not-detected]: no crate matched advisory criteria`
for **six** entries, and for those same six it also prints
`warning[unknown-advisory]: advisory not found in any advisory
database`. Read the output in full and cross the two warnings: an
`advisory-not-detected` **without** its `unknown-advisory` is a real
signal; with it, it means "I do not have the data".

### The local database, and how to make it current

**This section replaces a claim that was half wrong.** It used to say the
local copy is from 2026-07-16, that `cargo deny fetch` fails silently
over git's "dubious ownership", and -- by implication -- that nothing
could be done short of changing the machine's global git configuration.
The first two halves were right. The third was not, and it cost a night
of pushing security fixes unverified.

Measured 2026-11-24: the clone was **515 commits behind**, its real HEAD
was 2026-07-13 (the 2026-07-16 was the *file* mtime, three days later --
so a date read off the filesystem is not the database's age), and the
cause is not a preference. The files were created by an account whose
SID belongs to **another machine**, so git refuses the repository and
prints `(inconvertible)` where the owner's name would go.
`safe.directory` is the mechanism git's own error message proposes, and
it can be passed **per command**, changing nothing persistent:

```bash
for SD in "$HOME/.cargo/advisory-db"           "$HOME/.cargo/advisory-db/advisory-db-3157b0e258782691"; do
  git -c safe.directory="$SD" -C "$SD" fetch origin main
  git -c safe.directory="$SD" -C "$SD" merge --ff-only origin/main
done
```

Three things that cost time if you do not know them:

* **There are two clones and one is inside the other.** The outer is
  `cargo-audit`'s, the inner is `cargo-deny`'s. Updating one leaves the
  other answering with four-month-old data and a green exit code, which
  is worse than both being stale.
* **`cargo audit` needs `--no-fetch`** to use the local copy. Its own
  fetch fails with a *different* message, `failed to find 'origin'
  remote`, which reads like a broken repository and is the same
  ownership problem seen by its embedded git.
* **The check that it is now faithful is two numbers, not a date.**
  After the update, local printed `Loaded 1563 security advisories`
  against `1078 crate dependencies` -- identical to CI's log for the
  same day. Compare those with CI's; that is what turns the local audit
  into a predictor of CI's verdict, and until then it was not one.

`cargo deny fetch` still fails on its own, because its embedded git does
not honour a command-line `-c`. Keeping the database current without
hand-running the loop above needs a decision about mechanism, and that
decision is the maintainer's.


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

**An `informational` advisory turns CI red exactly like a vulnerability.**
The audit step runs with `-D unsound -D yanked -D unmaintained`, so
there is no "it is only a warning" tier here. V494 escaped
`RUSTSEC-2026-0595` (`parquet`, `informational = "unsound"`, published
the same day as the two it did fix) only because `parquet` has zero
`name = "parquet"` entries in the lock. **That near miss was a crate,
not a policy.**

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
