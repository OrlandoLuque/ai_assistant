# V130 — Phase C.9: operational runbooks

**Date**: 2026-05-06
**Version**: 0.2.77
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.9
**Tasks**: #339 (V130 C.9 — runbooks)

## Why

Tier-1 readiness checklist § C.9 calls for documented operational
runbooks: "what does an on-call operator do at 2 a.m. when X
happens?" — for `llama-server` crashes, vector DB corruption,
scheduler skips, RBAC token expiration, and similar production
failures. Before V130 the codebase had:

* Architecture docs (`docs/AGENT_SYSTEM_DESIGN.md`,
  `docs/CONCEPTS.md`) — useful for design review, useless at 2 a.m.
* Per-feature guides (`docs/GUIDE.md`,
  `docs/GUIDE_ANTI_HALLUCINATION.md`) — feature-mode, not
  failure-mode.
* No directory of incident playbooks. No fixed format. No
  reviewed-on dates so an operator could tell whether a doc was
  fresh.

V130's deliverable is the missing layer: pure docs, no code, but
rigorously formatted so on-call quality stays high as the docs
accumulate.

## What changed

### `docs/runbooks/INDEX.md` (new)

The directory contract. Documents the six-section template every
runbook follows (Symptoms → Likely causes → Diagnose → Mitigate
→ Resolve → Postmortem), the table of available runbooks, and
short cross-cutting "when in doubt" guidance. Carries a
*Last reviewed* date so freshness is observable.

### Six runbooks

| File | Covers |
|---|---|
| `docs/runbooks/llama-server-down.md` | `llama-server` crashes / OOM-kill / model file corrupted / GPU driver hang / port collision / version drift |
| `docs/runbooks/vector-db-corruption.md` | HNSW / SQLite / LanceDB / pgvector backends — corruption diagnosis and recovery from `secure_backup` snapshots or source documents |
| `docs/runbooks/scheduler-missed-job.md` | Scheduler not running / clock skew / queue starvation / stale lock file / TZ mismatch |
| `docs/runbooks/rbac-token-expired.md` | TTL elapsed / signing-key rotation / clock skew / scope tightening / IdP outage |
| `docs/runbooks/backup-verify-failed.md` | V128 `ai_backup verify` non-zero — sidecar mismatch, crypto failure, format error, signature failure |
| `docs/runbooks/rag-empty-results.md` | RAG opens but returns 0 hits — embedding-model mismatch, threshold too high, filter excludes everything, empty index, reranker stuck, tenant isolation bug |

### Why these six

The plan called out four (llama-server, vector DB, scheduler,
RBAC). Two more are included because (a) they map directly to
recently-shipped surfaces — `backup-verify-failed` covers V128's
`ai_backup` CLI; `rag-empty-results` is a non-corruption RAG
failure mode that gets confused with corruption otherwise — and
(b) they are real observed failure modes in this codebase's domain,
not hypotheticals.

### Format invariants every runbook obeys

1. **Six sections, in this order.** Symptoms first because that's
   what the operator sees first. Postmortem last because that's
   what they file when traffic is back.
2. **Severity, owner, last-reviewed at the top.** No need to scroll
   to know if you're the right person and whether the doc is stale.
3. **A "Likely causes" table sorted by frequency.** This is the
   part operators actually use to triage; alphabetic order would
   waste time.
4. **Concrete commands, not "look around".** Where a command is
   platform-specific, both Linux and Windows variants are given.
5. **References to other runbooks where the symptom can be
   confused.** E.g. `vector-db-corruption.md` explicitly redirects
   to `rag-empty-results.md` when the symptom is "0 hits but index
   opens cleanly", and vice versa.

### Things V130 deliberately does not do

* It does **not** add CI lint for runbook freshness. A six-month
  *Last reviewed* date is a soft signal, not a CI gate; mechanising
  it tends to produce date-bumps without re-reads.
* It does **not** replace `docs/DEPLOYMENT.md` (provisioning) or
  `docs/BINARIES.md` (per-CLI reference). Runbooks are for failure;
  those docs are for green-path operations.
* It does **not** ship runbooks for every crate feature — only the
  load-bearing ones with realistic failure modes. New runbooks
  should be added as new failure modes are observed in production
  (which is exactly the Postmortem section's job).

### `Cargo.toml`

Version 0.2.76 → 0.2.77. No code change.

## Compatibility

Pure docs. Zero code change, zero feature change, zero API change,
zero test change. The crate's behaviour is identical to V129.

## What's next

- V131 / C.4 — release automation (`release-plz` or `cargo-release`,
  cross-platform binary matrix, signed releases with SBOM and
  SHA-256 sidecars attached to the GitHub release; sigstore
  signing carried over from V125).
