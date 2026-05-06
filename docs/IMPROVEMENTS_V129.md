# V129 — Phase C.8: GDPR right-to-erasure (`gdpr::purge_user`)

**Date**: 2026-05-06
**Version**: 0.2.76
**Plan**: `ai_assistant_plans/plan_tier1_competitive_gaps.md` § C.8
**Tasks**: #338 (V129 C.8 — GDPR right-to-erasure)

## Why

Article 17 of the EU GDPR (the "Right to be Forgotten") requires a
controller to erase a data subject's personal data on request,
without undue delay, across **every** system that holds it. Before
V129 the codebase had several scattered, partial APIs:

| API | Coverage |
|---|---|
| `ChatSessionStore::delete_session` | session metadata only |
| `RagDb::clear_session_history` | per-user, per-session messages |
| `RagDb::delete_knowledge_notes` | per-user, per-source notes |
| `RagDb::delete_session_notes` | per-user, per-session notes |
| `mcp_task_tools::purge_expired` | tombstone cleanup, *not* user-keyed |

A pre-V129 audit (see exploration log) catalogued **17+ subsystems**
that retain user-attributable data — in addition to the four above,
modules like `procedural` memory, `conversation_analytics`,
`user_engagement`, `multi_layer_graph::UserGraph`, `ab_testing`,
`feedback_loop::ledger`, and the central `security::audit` log all
key by `user_id` (or `principal`, `signer`, `owner`). None of them
exposed a coordinated erasure path. None emitted an audit record
proving a subject's request had been honoured. None handled the
*regulatory dual obligation*: erase the subject's data, **but**
retain proof that processing occurred.

V129's deliverable is the missing orchestration layer plus the
compliance documentation that lets a controller actually use it.

## What changed

### `src/gdpr.rs` (new module, behind feature `gdpr`)

Five small public surfaces:

```rust
pub trait PurgeAdapter: Send {
    fn name(&self) -> &'static str;
    fn purge_user(&mut self, user_id: &str) -> Result<usize, String>;
}

pub fn purge_user(
    user_id: &str,
    adapters: &mut [Box<dyn PurgeAdapter>],
    audit:    Option<&mut AuditLogger>,
) -> Result<PurgeReport, PurgeError>;

pub fn hash_user_id(user_id: &str) -> String;     // SHA-256 hex

pub struct MapPurgeAdapter<'a, V: Send> { ... }   // reference impl

pub struct PurgeReport {
    pub user_id_sha256: String,            // never the raw id
    pub started_at:     DateTime<Utc>,
    pub finished_at:    DateTime<Utc>,
    pub per_subsystem:  Vec<SubsystemPurge>,
    pub total_records:  usize,
    pub partial_failures: Vec<PartialFailure>,
    pub audit_redacted:        bool,
    pub audit_records_redacted: usize,
}
```

### Three design decisions worth flagging

1. **Adapter pattern, not a hard-coded module list.** With ~17
   user-attributable subsystems and feature gating that varies
   per-deployment, hard-wiring would produce either (a) tight
   coupling that breaks when a feature is off, or (b) a per-feature
   `#[cfg]` matrix in one module. Instead: each subsystem (or each
   *deployment's* concrete store types) implements the trait. The
   crate ships `MapPurgeAdapter` as a reference plus the
   orchestrator. Crate consumers write thin adapters for their own
   stores.

2. **Audit is *redacted*, not deleted.** Regulators (CNIL guidance
   2020-12-15, GDPR Art. 5(1)(f) accountability) consistently treat
   the audit trail as a *protection* for data subjects, not a
   secondary copy of their data. Deleting audit events on erasure
   destroys evidence that processing happened. The orchestrator
   therefore:
   - calls `AuditLogger::redact_user(user_id)` which walks every
     event in place: `user_id` field → `"[ERASED]"`, any `details`
     value matching the user_id → `"[ERASED]"`, any `details` key
     in the PII keylist (`email`, `username`, `name`, `principal`,
     `ip`, `phone`) → `"[ERASED]"`;
   - appends a single `AuditEventType::DataErased` event carrying
     only a SHA-256 hash of the erased id, the per-subsystem record
     count, and the partial-failure count.
   The audit *trail* survives. The *linkage to the subject* does
   not.

3. **Best-effort with structured failure.** A single adapter
   failure does not abort the call. Failures land in
   `PurgeReport::partial_failures` and the orchestration continues.
   This matches reality — distributed sub-stores fail
   independently — and gives a compliance officer something
   actionable. The call only returns `Err` when (a) the user_id is
   empty after trimming, or (b) zero adapters were supplied (almost
   certainly a misconfiguration).

### Hashing

`hash_user_id` is SHA-256 hex (lowercase, 64 chars). Stable across
processes and platforms. A compliance officer can prove a specific
request was honoured by re-hashing the original id and comparing
against the audit detail; the system itself never retains the raw
id post-purge.

### `src/security/audit.rs` — two additions

* `AuditEventType::DataErased` variant (the enum is
  `#[non_exhaustive]`, so this is non-breaking for existing
  pattern-match consumers).
* `AuditLogger::redact_user(&mut self, user_id: &str) -> usize`.
  Returns the number of events whose `user_id` field was rewritten.
  Standalone — usable without invoking `gdpr::purge_user`.

### Tests (9 new, all passing)

| Test | Covers |
|---|---|
| `hash_is_deterministic_and_64_hex_chars` | SHA-256 stability and shape |
| `purges_user_across_adapters_leaves_others_untouched` | Per-user isolation across multiple adapters |
| `rejects_empty_user_id` | Trimmed-empty-id rejection |
| `rejects_no_adapters` | Misconfiguration guard |
| `collects_partial_failures_without_aborting` | Best-effort semantics |
| `audit_log_is_redacted_in_place_and_data_erased_event_emitted` | The whole audit story end-to-end |
| `idempotent_when_user_has_no_data` | Second call returns 0, no failure |
| `report_has_consistent_timing` | `finished_at >= started_at`, hash length |
| `map_purge_adapter_matches_trait` | Reference adapter behaviour |

`cargo test --lib --features full` reports **6212 passing** (V128
baseline 6203 + 9 new from V129).

### `docs/DPIA_TEMPLATE.md` (new)

A template Data Protection Impact Assessment a controller fills in
when deploying `ai_assistant`. Eleven sections covering:
controller/processor identification, processing description,
necessity & proportionality, subsystems handling personal data
(pre-filled table mapping cargo features to data categories),
Articles 15–22 rights, security measures, risks, consultation,
records of processing, review cadence, sign-off. Two appendices:
feature-flag mapping and an erasure runbook including the exact
`gdpr::purge_user` call shape.

The template is opinionated where the library can be authoritative
(subsystem inventory, applicable controls) and `<TODO>`-marked
where only the controller can speak (legal entity, retention
periods, lawful basis).

### `Cargo.toml`

* New feature `gdpr = ["dep:sha2"]`.
* Added to the `full` feature set.
* Version 0.2.75 → 0.2.76.

### `src/lib.rs`

```rust
#[cfg(feature = "gdpr")] pub mod gdpr;
```

between `formatting` and `gguf_downloader`.

## What this V cycle deliberately does *not* do

* It does **not** ship concrete adapters for every internal subsystem.
  The `RagDb` already has SQL-level deletion APIs that call sites can
  wrap; the same is true of any HashMap-backed store via
  `MapPurgeAdapter`. Wiring all 17+ subsystems would mean editing
  17+ modules in one V cycle and require feature-conditional adapter
  glue throughout. The orchestrator + reference adapter is the
  integration point; deployment-specific adapter sets are the
  controller's job (and are documented as such in the DPIA template).
* It does **not** handle Articles 15 (subject access) or 20 (data
  portability). Those are separate compliance flows; conflating them
  into one module would entangle "export" and "delete" semantics in
  ways the GDPR does not.
* It does **not** modify any append-only ledger. Ledgers preserve
  integrity by being immutable; the GDPR-compliant pattern there is
  to emit a `RetractionRequested`-shaped event (already present in
  `feedback_loop::ledger`) and document the limitation in the DPIA.
  Adapters for ledger-backed subsystems should emit such an event
  and return the count of trajectories marked retracted.

## Compatibility

* Pure addition. The new `gdpr` feature is in `full`, so any caller
  on `default-features = ["full"]` picks it up automatically.
  Callers on a narrower set keep their dep graph.
* `AuditEventType::DataErased` is a new variant on a
  `#[non_exhaustive]` enum — non-breaking.
* `AuditLogger::redact_user` is a new method — non-breaking.
* No existing source files changed except `src/lib.rs` (one new
  `pub mod` line), `src/security/audit.rs` (one new variant + one
  new method), and `Cargo.toml`.

## What's next

- V130 / C.9 — operational runbooks (`docs/runbooks/`) for
  llama-server crashes, vector DB corruption, scheduler skips, RBAC
  token expiration. Pure docs, no code.
- V131 / C.4 — release automation (`release-plz` / `cargo-release`,
  cross-platform binary matrix, sign + SBOM + SHA256 attached to
  GH release; sigstore signing carried over from V125).
