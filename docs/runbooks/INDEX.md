# Operational Runbooks

Each runbook is a short, opinionated playbook for a specific
production failure. They are written for an on-call operator at
2 a.m., not for a leisurely architectural read. Format is fixed:

1. **Symptoms** — what the operator actually sees first.
2. **Likely causes** — ordered by frequency in our deployments.
3. **Diagnose** — concrete commands. No "look around" steps.
4. **Mitigate** — get traffic moving again. Acceptable to be ugly.
5. **Resolve** — proper fix. Schedule, do not rush.
6. **Postmortem** — what to record and what to change.

If you change a runbook, update the *Last reviewed* date so the
next operator can tell whether the playbook is fresh.

## Runbooks

| Slug | Symptom in one line |
|---|---|
| [llama-server-down](llama-server-down.md) | `ai_local_infer` 5xx / health probe fails / generation hangs |
| [vector-db-corruption](vector-db-corruption.md) | RAG returns garbage, panics on query, or fails to open |
| [scheduler-missed-job](scheduler-missed-job.md) | Cron-style job didn't fire; downstream alert |
| [rbac-token-expired](rbac-token-expired.md) | API requests get 401/403 immediately after deploy or rotation |
| [backup-verify-failed](backup-verify-failed.md) | `ai_backup verify` returns non-zero |
| [rag-empty-results](rag-empty-results.md) | RAG retrieves zero chunks for queries that should hit |
| [rustsec-handling](rustsec-handling.md) | `cargo audit`/`cargo deny` red, or monthly review issue open |

## When in doubt

* **Don't** delete state without a backup. Every storage subsystem
  in this crate has either a `secure_backup`-able state directory
  or a `dump` API. Use them before destructive actions.
* **Don't** silence alerts to "make it green." File the postmortem.
* **Do** capture `ai_logs --since=15m --json > snapshot.json`
  before you start rolling fixes — most regressions get harder to
  diagnose once you've changed three things.

## Related docs

* `docs/DEPLOYMENT.md` — provisioning, capacity planning, env vars.
* `docs/BINARIES.md` — what each `ai_*` CLI does.
* `docs/FEATURE_LIFECYCLE.md` — feature flag policy.
* `docs/DPIA_TEMPLATE.md` — GDPR / data-subject erasure flow.

*Last reviewed: 2026-05-26 (V142).*
