# Runbook: handling a RUSTSEC advisory

**Severity**: depends on the advisory. The audit gate fails → start here.
**Owner**: maintainer (this is a one-person project; on-call is the author).
**Last reviewed**: 2026-05-26 (V142).

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
    * Why it doesn't apply (one or two sentences).
    * A re-check trigger: a date OR an upstream event
      ("when crate X publishes version Y").
```

## 3. Adding an ignore — the policy

A `deny.toml` entry without comment + re-check trigger fails review.
Example of an **acceptable** entry:

```toml
# tantivy → lru IterMut unsoundness. The affected codepath
# (mutating an entry obtained from IterMut while the iterator is
# alive) is not used by tantivy's query/indexing paths we exercise.
# Re-check by: when tantivy 0.25+ ships OR 2026-08-01.
"RUSTSEC-2026-0002",
```

Example of an **unacceptable** entry:

```toml
# upstream issue, ignore
"RUSTSEC-XXXX-NNNN",
```

The monthly review will flag both the same way, but the second
gives the reviewer no way to judge whether the ignore is still
valid.

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

## 5. Mitigate (CI red, fix not ready)

If the CI is blocking a release and the proper fix needs time:

* Add the ID to all three places (`deny.toml`,
  `ci.yml`, `supply-chain.yml`) with a comment marking it
  `TEMPORARY — expected fix in <PR/version>` and a re-check date no
  more than 4 weeks out.
* File a follow-up commit/PR to land the proper fix before the
  re-check date.

This unblocks merging without losing track. The monthly review will
nag if the temporary entry survives past its re-check.

## 6. Postmortem

If a RUSTSEC ended up affecting us in production:

* Record: which ignore was wrong, when the advisory was first
  available, what the operational impact was.
* Tighten the policy in this runbook if the failure was structural
  (e.g. "we missed it because the dep was transitive 5 levels deep
  — add cargo-deny `--graph-depth 10` to the audit job").

## Related

* `deny.toml#advisories.ignore` — the canonical ignore list.
* `.github/workflows/ci.yml` (audit job) — gating per PR.
* `.github/workflows/supply-chain.yml` — gating + weekly schedule.
* `.github/workflows/rustsec-review-monthly.yml` — monthly nag.
