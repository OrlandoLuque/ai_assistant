# Data Protection Impact Assessment (DPIA) — Template

**Crate**: `ai_assistant`
**Document version**: V129 (2026-05-06)
**Applies to**: any deployment of this library that processes personal
data of natural persons subject to the EU GDPR, the UK GDPR, the
Swiss LPD, or equivalent regimes (LGPD-BR, PIPEDA-CA, CCPA-US).

---

## How to use this template

This file is **a template, not a completed assessment.** Each section
below contains:

* **What to fill in** — instructions for the controller doing the DPIA.
* **Pre-filled facts about `ai_assistant`** — items the library
  authors can answer once and re-use across deployments. These appear
  in `> blockquotes`.

Replace every `<TODO>` with deployment-specific information. Sign and
date Section 11 once complete. Retain the signed copy for the period
required by your data-protection regulator (typically the lifetime of
the processing operation plus three years).

A DPIA is mandatory under GDPR Article 35 when processing is "likely
to result in a high risk to the rights and freedoms of natural
persons" — including systematic large-scale processing, automated
decision-making, profiling, and processing of special-category data.
LLM-driven assistants frequently meet at least one of these criteria,
so completing this template is strongly recommended even when not
strictly mandatory.

---

## 1. Controller and processor identification

| Role | Identity | Contact |
|---|---|---|
| Data controller | `<TODO: legal entity>` | `<TODO: email>` |
| DPO (if appointed) | `<TODO>` | `<TODO>` |
| Joint controllers (if any) | `<TODO>` | `<TODO>` |
| Processors | `<TODO: e.g. cloud LLM provider, hosting>` | `<TODO>` |

> **`ai_assistant` upstream**: not a controller and not a processor in
> the GDPR sense. It is *software* the controller embeds. The
> controller remains responsible for every processing activity
> performed by the deployed binary.

## 2. Description of processing

* **Purpose**: `<TODO: e.g. answering customer-support questions>`
* **Lawful basis** (Art. 6): `<TODO: contract / consent / legitimate
  interest / legal obligation>`
* **Lawful basis for special-category data** (Art. 9), if any:
  `<TODO>`
* **Categories of data subjects**: `<TODO: e.g. customers, employees>`
* **Categories of personal data**: `<TODO: e.g. names, email
  addresses, conversation transcripts, IP addresses>`
* **Special categories** (Art. 9), if any: `<TODO>`
* **Recipients**: `<TODO: e.g. cloud LLM provider X, internal
  analytics team, regulator on request>`
* **Transfers outside the EEA** and safeguards: `<TODO: SCCs,
  adequacy decision, binding corporate rules>`
* **Retention period** per category: `<TODO>`

## 3. Necessity and proportionality assessment

* **Why is this processing necessary** for the stated purpose?
  `<TODO>`
* **Could the purpose be achieved with less data, less intrusive
  means, or aggregation?** `<TODO>`
* **Have data minimisation, accuracy, storage limitation, and
  purpose-limitation principles been implemented?** `<TODO>`

## 4. Subsystems handling personal data

This section catalogues every component of `ai_assistant` enabled in
the deployment that may store user-attributable data. Mark each row
*Enabled* or *Not present* and add controls.

| Subsystem (cargo feature) | Stores | User key | Controls |
|---|---|---|---|
| `security::audit` (always on) | Audit events | `user_id`, `session_id` | TTL, redaction via `gdpr::purge_user` |
| `session` (always on) | Chat history (RAM) | `session_id` | `delete_session` |
| `rag` (`rag`) | SQLite: users, conversation_history, knowledge_notes | `user_id` | `clear_session_history`, custom `PurgeAdapter` |
| `memory_management` (always on) | Long-term memory | `session_id` | Caller-managed |
| `advanced_memory::procedural` (`advanced-memory`) | Per-user procedures | `user_id` | Caller-managed; needs `PurgeAdapter` |
| `conversation_analytics` (`analytics`) | Event log | `user_id`, `session_id` | Caller-managed; needs `PurgeAdapter` |
| `user_engagement` (`analytics`) | Engagement metrics | `user_id` | Caller-managed; needs `PurgeAdapter` |
| `ab_testing` (`eval`) | Variant assignments | `user_id`, `experiment_id` | Caller-managed |
| `multi_layer_graph` (`autonomous`) | UserGraph beliefs | `user_id` | Caller-managed |
| `feedback_loop::ledger` (`feedback-loop`) | Append-only trajectories | `principal` | RetractionRequested event; immutable storage |
| `secure_backup` (`backup`) | Encrypted snapshots | (file-level) | Re-create after purge with caller-supplied source set |

## 5. Data subject rights

* **Article 15 — access**: `<TODO: process to export a subject's data>`
* **Article 16 — rectification**: `<TODO>`
* **Article 17 — erasure**: implemented via `gdpr::purge_user`. The
  library emits an `AuditEventType::DataErased` record carrying a
  SHA-256 of the erased `user_id`. The audit log is *redacted in
  place* (linkages to the subject are blanked) but events themselves
  are retained for accountability under Article 5(1)(f).
  Document the operational runbook (who triggers the call, how the
  PurgeReport is archived) at: `<TODO: link to runbook>`.
* **Article 18 — restriction**: `<TODO>`
* **Article 20 — portability**: `<TODO>`
* **Article 21 — objection**: `<TODO>`
* **Article 22 — automated decisions**: `<TODO: confirm whether the
  LLM's output qualifies as an automated decision with legal or
  similar significant effect>`

## 6. Security measures

| Measure | `ai_assistant` feature / configuration |
|---|---|
| Encryption at rest of backups | `backup` (AES-256-GCM, HKDF-SHA256) |
| Tamper-evident integrity | Ed25519 sign/verify in `secure_backup` |
| Audit trail | `security::audit` |
| RBAC | `access_control` module |
| PII redaction in logs | `log_redaction` module |
| Rate limiting | `security::rate_limiting` |
| Input sanitisation | `security::sanitization` |
| Append-only ledgers | `feedback_loop::ledger`, `prompt_breeder::ledger` |
| Right to erasure | `gdpr::purge_user` |
| Supply-chain hardening | `cargo-deny`, `cargo-audit`, SBOM (V125) |

`<TODO>`: organisational measures (access policies, training, breach
playbook, retention enforcement).

## 7. Risks to data subjects

For each identified risk, score Likelihood × Severity (1–5 each) and
list mitigations.

| # | Risk | Likelihood | Severity | Mitigations |
|---|---|---|---|---|
| 1 | Re-identification via prompt logs | `<TODO>` | `<TODO>` | `log_redaction`, audit retention TTL, PurgeAdapter |
| 2 | Model-side training on captured prompts | `<TODO>` | `<TODO>` | `<TODO: provider DPA, opt-out flags>` |
| 3 | Cross-tenant leakage in shared RAG store | `<TODO>` | `<TODO>` | Per-user isolation (`user_id` keys), `clear_session_history` |
| 4 | Inadvertent retention of erased data in cached embeddings | `<TODO>` | `<TODO>` | Custom `PurgeAdapter` for embedding cache |
| 5 | Audit log itself becoming a PII repository | `<TODO>` | `<TODO>` | `redact_user` runs as part of every purge |
| 6 | Append-only ledger preventing erasure | `<TODO>` | `<TODO>` | RetractionRequested events; document the limit explicitly to subjects |
| ... | `<TODO: domain-specific risks>` | | | |

Residual risk after mitigation per row: `<TODO>`.

## 8. Consultation

* Internal: `<TODO: who reviewed this DPIA — DPO, security, legal>`
* Subjects (if relevant): `<TODO>`
* Supervisory authority: `<TODO: only required when residual risk
  remains high after mitigations>`

## 9. Records of processing (Article 30)

Cross-reference: `<TODO: link to your Records of Processing>`.

## 10. Review cadence

This DPIA must be reviewed:

* On any material change to processing purposes or means.
* When `ai_assistant` is upgraded across a feature-set change (new
  subsystem in the `full` feature, new module storing personal data).
* At least every 24 months, even absent change.

Next review due: `<TODO: YYYY-MM-DD>`.

## 11. Sign-off

| Name | Role | Date | Signature |
|---|---|---|---|
| `<TODO>` | Data controller representative | `<TODO>` | `<TODO>` |
| `<TODO>` | DPO | `<TODO>` | `<TODO>` |
| `<TODO>` | CISO / equivalent | `<TODO>` | `<TODO>` |

---

## Appendix A — Mapping to `ai_assistant` features

The library is delivered as a Cargo crate with feature flags. Only
features actually compiled into the deployment matter for this DPIA.
List the feature set used at deployment time:

```toml
[dependencies]
ai_assistant = { version = "0.2.x", default-features = false, features = [<TODO>] }
```

If `default-features = ["full"]`, every subsystem in Section 4 is
enabled.

## Appendix B — Erasure runbook (template)

1. **Receive** the data-subject request through `<TODO: intake
   channel>`.
2. **Verify identity** of the requester per `<TODO: verification
   policy>`.
3. **Construct the adapter set** for every storage layer the
   deployment uses. Adapters live in the controller's code, not the
   library; the library only ships `PurgeAdapter` and one reference
   `MapPurgeAdapter` impl.
4. **Invoke**:
   ```rust
   let report = ai_assistant::gdpr::purge_user(
       &user_id,
       &mut adapters,
       Some(&mut audit_logger),
   )?;
   ```
5. **Persist** the `PurgeReport` (it is `Serialize`) to the compliance
   evidence store. Retain for `<TODO: e.g. 3 years>`.
6. **Notify** the subject within 30 days (Art. 12(3)). Communicate any
   `partial_failures` and the remediation timeline.
7. **Notify recipients** (Art. 19) — propagate the erasure to every
   processor or sub-controller listed in Section 2.
