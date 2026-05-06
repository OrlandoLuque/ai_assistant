//! V129 (C.8) — GDPR Article 17: Right to Erasure ("Right to be Forgotten").
//!
//! This module gives a single, audited entry point for honouring a
//! data-subject deletion request across every subsystem of the crate
//! that may store user-attributable data:
//!
//! ```text
//! gdpr::purge_user(user_id, &mut adapters, Some(&mut audit_logger))
//!     -> Result<PurgeReport, PurgeError>
//! ```
//!
//! ### Design
//!
//! * **Adapter pattern** — each storage subsystem (RAG SQLite tables,
//!   in-memory engagement tracker, multi-layer user graph, ...)
//!   implements [`PurgeAdapter`]. Adapters are owned by the caller so
//!   the crate doesn't have to enumerate every concrete type at
//!   compile time, and so callers can plug in their own stores.
//!
//! * **Audit is redacted, not deleted.** Regulators (CNIL guidance
//!   2020-12-15, Article 5(1)(f) accountability) want proof that
//!   processing occurred *and* that the erasure request was honoured.
//!   The orchestrator therefore (a) emits a single
//!   [`AuditEventType::DataErased`] record carrying only a
//!   *SHA-256 hash* of the erased user_id, and (b) walks the existing
//!   audit log in place to overwrite every reference to the raw
//!   user_id with `"[ERASED]"`. The audit *trail* survives the erasure;
//!   the *linkage to the data subject* does not.
//!
//! * **Best-effort with structured failure** — if one adapter fails,
//!   the orchestrator continues with the rest and records the failure
//!   in [`PurgeReport::partial_failures`]. The whole call only returns
//!   `Err` if the audit emission itself fails or no adapters are
//!   supplied.
//!
//! * **No raw user_id leaves this module after the call.** The audit
//!   detail map carries only the SHA-256 hex digest. This lets a
//!   compliance officer prove "request for user X was honoured" by
//!   re-hashing the original ID, without the system retaining PII.
//!
//! ### What this module does *not* do
//!
//! * It does not enumerate all 17+ user-attributable subsystems found
//!   in the codebase. Adapters for the obvious subsystems
//!   ([`MapPurgeAdapter`] for any in-memory `HashMap<String, _>`-shaped
//!   store, plus the [`AuditLogger`] redaction handled inline) are
//!   provided. Crate consumers add adapters for their own concrete
//!   store types — a `RagDbPurgeAdapter`, a `MemoryManagerPurgeAdapter`,
//!   etc. The adapter trait is the integration point, intentionally
//!   small (two methods).
//!
//! * It does not implement Article 20 (data portability — separate
//!   module concern) or Article 15 (subject access — separate again).
//!   This is purely the erasure path.
//!
//! ### Compliance evidence
//!
//! [`PurgeReport`] is `Serialize`. Persist it (file, S3, immutable
//! ledger) for the retention window your jurisdiction requires
//! (typically 1–3 years in the EU under accountability principles).

use std::collections::HashMap;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::security::{AuditEvent, AuditEventType, AuditLogger};

// =====================================================================
// Errors
// =====================================================================

/// Top-level error from [`purge_user`].
///
/// Subsystem-level failures are *not* surfaced through this enum; they
/// are collected in [`PurgeReport::partial_failures`] so a partial
/// erasure is still observable. Returning a hard error would prevent
/// the caller from seeing how many subsystems *did* succeed.
#[derive(Debug, thiserror::Error)]
pub enum PurgeError {
    /// `user_id` was empty after trimming.
    #[error("user_id must not be empty")]
    EmptyUserId,
    /// No adapters were supplied. A purge call with no adapters is
    /// almost certainly a mis-configuration; we refuse it loudly so a
    /// compliance officer doesn't sign off on a no-op.
    #[error("at least one PurgeAdapter must be supplied")]
    NoAdapters,
}

/// Failure observed while running a single adapter. Recorded in the
/// report; does not abort the orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFailure {
    pub subsystem: String,
    pub error: String,
}

// =====================================================================
// Adapter trait
// =====================================================================

/// One subsystem's contribution to a GDPR erasure request.
///
/// The trait is deliberately tiny — two methods — so adapters are
/// trivial to write for in-house storage layers. Implementors should
/// be **idempotent**: calling `purge_user` twice for the same id on
/// a system that no longer has any matching records must succeed and
/// return `0`.
pub trait PurgeAdapter: Send {
    /// Stable name of the subsystem. Appears in [`PurgeReport`] and the
    /// audit detail map. Use a short identifier (`"rag.user_table"`,
    /// `"memory.procedural"`).
    fn name(&self) -> &'static str;

    /// Erase every record attributable to `user_id` from this
    /// subsystem. Return the number of records removed (best-effort —
    /// a coarse estimate is acceptable when an exact count is
    /// expensive).
    fn purge_user(&mut self, user_id: &str) -> Result<usize, String>;
}

// =====================================================================
// Reference adapter — in-memory map keyed by user_id
// =====================================================================

/// Reference [`PurgeAdapter`] for any in-memory `HashMap<String, _>`-
/// shaped store keyed by `user_id`. Useful as a building block for
/// quick adapters over caches and analytics counters.
///
/// ```rust,ignore
/// let mut counters: HashMap<String, u64> = HashMap::new();
/// counters.insert("alice".into(), 7);
/// counters.insert("bob".into(),   3);
///
/// let mut adapter = ai_assistant::gdpr::MapPurgeAdapter::new(
///     "test.counters",
///     &mut counters,
/// );
/// let report = ai_assistant::gdpr::purge_user(
///     "alice",
///     &mut [Box::new(adapter)],
///     None,
/// )?;
/// assert_eq!(report.total_records, 1);
/// ```
pub struct MapPurgeAdapter<'a, V: Send> {
    name: &'static str,
    map: &'a mut HashMap<String, V>,
}

impl<'a, V: Send> MapPurgeAdapter<'a, V> {
    pub fn new(name: &'static str, map: &'a mut HashMap<String, V>) -> Self {
        Self { name, map }
    }
}

impl<'a, V: Send> PurgeAdapter for MapPurgeAdapter<'a, V> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn purge_user(&mut self, user_id: &str) -> Result<usize, String> {
        Ok(self.map.remove(user_id).map(|_| 1).unwrap_or(0))
    }
}

// =====================================================================
// Report
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemPurge {
    pub name: String,
    pub records_removed: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeReport {
    /// Hex SHA-256 of the original user_id. The raw user_id is *never*
    /// stored in the report.
    pub user_id_sha256: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub per_subsystem: Vec<SubsystemPurge>,
    pub total_records: usize,
    pub partial_failures: Vec<PartialFailure>,
    /// Whether the central audit log was redacted in place. False if
    /// no `AuditLogger` was supplied.
    pub audit_redacted: bool,
    pub audit_records_redacted: usize,
}

// =====================================================================
// Hashing
// =====================================================================

/// SHA-256 of `user_id` as lowercase hex. Stable across processes and
/// platforms — safe to persist and re-derive for compliance audits.
pub fn hash_user_id(user_id: &str) -> String {
    let mut h = Sha256::new();
    h.update(user_id.as_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

// =====================================================================
// Orchestrator
// =====================================================================

/// Walk every adapter, redact the audit log if supplied, and return a
/// structured [`PurgeReport`].
///
/// Adapters run sequentially in the order given. A subsystem failure
/// is recorded in `partial_failures` and the next adapter is invoked.
pub fn purge_user(
    user_id: &str,
    adapters: &mut [Box<dyn PurgeAdapter>],
    audit: Option<&mut AuditLogger>,
) -> Result<PurgeReport, PurgeError> {
    let trimmed = user_id.trim();
    if trimmed.is_empty() {
        return Err(PurgeError::EmptyUserId);
    }
    if adapters.is_empty() {
        return Err(PurgeError::NoAdapters);
    }

    let started_at = Utc::now();
    let user_id_sha256 = hash_user_id(trimmed);

    let mut per_subsystem = Vec::with_capacity(adapters.len());
    let mut partial_failures = Vec::new();
    let mut total_records = 0usize;

    for adapter in adapters.iter_mut() {
        let name = adapter.name().to_string();
        let t0 = Instant::now();
        match adapter.purge_user(trimmed) {
            Ok(n) => {
                total_records += n;
                per_subsystem.push(SubsystemPurge {
                    name,
                    records_removed: n,
                    duration_ms: t0.elapsed().as_millis() as u64,
                });
            }
            Err(e) => {
                partial_failures.push(PartialFailure {
                    subsystem: name.clone(),
                    error: e,
                });
                per_subsystem.push(SubsystemPurge {
                    name,
                    records_removed: 0,
                    duration_ms: t0.elapsed().as_millis() as u64,
                });
            }
        }
    }

    // Redact + emit on the audit log, if one was supplied.
    let mut audit_redacted = false;
    let mut audit_records_redacted = 0usize;
    if let Some(logger) = audit {
        audit_records_redacted = logger.redact_user(trimmed);
        audit_redacted = true;
        let event = AuditEvent::new(AuditEventType::DataErased)
            .with_detail("user_id_sha256", &user_id_sha256)
            .with_detail("subsystem_count", &per_subsystem.len().to_string())
            .with_detail("total_records_removed", &total_records.to_string())
            .with_detail("partial_failures", &partial_failures.len().to_string())
            .with_detail(
                "audit_records_redacted",
                &audit_records_redacted.to_string(),
            );
        logger.log(event);
    }

    Ok(PurgeReport {
        user_id_sha256,
        started_at,
        finished_at: Utc::now(),
        per_subsystem,
        total_records,
        partial_failures,
        audit_redacted,
        audit_records_redacted,
    })
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AuditConfig, AuditEvent, AuditEventType, AuditLogger};
    use std::collections::HashMap;

    /// A test adapter backed by a HashMap of `user_id -> Vec<record>`.
    /// More realistic than [`MapPurgeAdapter`] (which has a single
    /// value per user), and lets us assert "removes all records for X".
    struct VecMapAdapter {
        name: &'static str,
        store: HashMap<String, Vec<String>>,
    }

    impl PurgeAdapter for VecMapAdapter {
        fn name(&self) -> &'static str {
            self.name
        }
        fn purge_user(&mut self, user_id: &str) -> Result<usize, String> {
            Ok(self.store.remove(user_id).map(|v| v.len()).unwrap_or(0))
        }
    }

    /// Adapter that always fails — exercises the partial-failure path.
    struct AlwaysFailsAdapter;
    impl PurgeAdapter for AlwaysFailsAdapter {
        fn name(&self) -> &'static str {
            "test.fails"
        }
        fn purge_user(&mut self, _user_id: &str) -> Result<usize, String> {
            Err("simulated failure".into())
        }
    }

    fn vec_adapter(name: &'static str, seed: &[(&str, &[&str])]) -> VecMapAdapter {
        let mut store = HashMap::new();
        for (uid, recs) in seed {
            store.insert(
                (*uid).to_string(),
                recs.iter().map(|s| (*s).to_string()).collect(),
            );
        }
        VecMapAdapter { name, store }
    }

    #[test]
    fn hash_is_deterministic_and_64_hex_chars() {
        let a = hash_user_id("alice@example.com");
        let b = hash_user_id("alice@example.com");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        assert_ne!(a, hash_user_id("bob@example.com"));
    }

    #[test]
    fn purges_user_across_adapters_leaves_others_untouched() {
        let mut a1 = vec_adapter("test.a", &[("alice", &["x", "y"]), ("bob", &["z"])]);
        let mut a2 = vec_adapter("test.b", &[("alice", &["q"]), ("carol", &["r"])]);

        let report = {
            let mut adapters: Vec<Box<dyn PurgeAdapter>> = vec![
                Box::new(std::mem::replace(
                    &mut a1,
                    VecMapAdapter {
                        name: "test.a",
                        store: HashMap::new(),
                    },
                )),
                Box::new(std::mem::replace(
                    &mut a2,
                    VecMapAdapter {
                        name: "test.b",
                        store: HashMap::new(),
                    },
                )),
            ];
            purge_user("alice", &mut adapters, None).unwrap()
        };

        assert_eq!(report.total_records, 3); // 2 + 1
        assert_eq!(report.per_subsystem.len(), 2);
        assert_eq!(report.per_subsystem[0].records_removed, 2);
        assert_eq!(report.per_subsystem[1].records_removed, 1);
        assert!(report.partial_failures.is_empty());
        assert!(!report.audit_redacted);
        assert_eq!(report.user_id_sha256, hash_user_id("alice"));
    }

    #[test]
    fn rejects_empty_user_id() {
        let mut adapters: Vec<Box<dyn PurgeAdapter>> = vec![Box::new(AlwaysFailsAdapter)];
        assert!(matches!(
            purge_user("", &mut adapters, None),
            Err(PurgeError::EmptyUserId)
        ));
        assert!(matches!(
            purge_user("   ", &mut adapters, None),
            Err(PurgeError::EmptyUserId)
        ));
    }

    #[test]
    fn rejects_no_adapters() {
        let mut empty: Vec<Box<dyn PurgeAdapter>> = vec![];
        assert!(matches!(
            purge_user("alice", &mut empty, None),
            Err(PurgeError::NoAdapters)
        ));
    }

    #[test]
    fn collects_partial_failures_without_aborting() {
        let a = vec_adapter("test.ok", &[("alice", &["x"])]);
        let mut adapters: Vec<Box<dyn PurgeAdapter>> =
            vec![Box::new(AlwaysFailsAdapter), Box::new(a)];

        let report = purge_user("alice", &mut adapters, None).unwrap();

        assert_eq!(report.partial_failures.len(), 1);
        assert_eq!(report.partial_failures[0].subsystem, "test.fails");
        assert_eq!(report.total_records, 1); // ok adapter still ran
    }

    #[test]
    fn audit_log_is_redacted_in_place_and_data_erased_event_emitted() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        // Seed audit log with three events: two for alice, one for bob.
        logger.log(
            AuditEvent::new(AuditEventType::MessageSent)
                .with_user("alice")
                .with_detail("email", "alice@example.com")
                .with_detail("content_length", "42"),
        );
        logger.log(AuditEvent::new(AuditEventType::ResponseReceived).with_user("alice"));
        logger.log(AuditEvent::new(AuditEventType::MessageSent).with_user("bob"));

        let mut adapters: Vec<Box<dyn PurgeAdapter>> =
            vec![Box::new(vec_adapter("test.x", &[("alice", &["one"])]))];

        let report = purge_user("alice", &mut adapters, Some(&mut logger)).unwrap();

        assert!(report.audit_redacted);
        assert_eq!(report.audit_records_redacted, 2);

        // Alice's events: user_id is now "[ERASED]", email blanked, content_length kept.
        let events: Vec<&AuditEvent> = logger.get_events().iter().collect();
        let n_erased = events
            .iter()
            .filter(|e| e.user_id.as_deref() == Some("[ERASED]"))
            .count();
        assert_eq!(n_erased, 2);

        // Bob's event is intact.
        let bob_evt = events
            .iter()
            .find(|e| e.user_id.as_deref() == Some("bob"))
            .expect("bob's event must survive");
        assert_eq!(bob_evt.user_id.as_deref(), Some("bob"));

        // Email key is redacted on alice's events; content_length is kept.
        let first_alice = events
            .iter()
            .find(|e| {
                e.user_id.as_deref() == Some("[ERASED]")
                    && e.event_type == AuditEventType::MessageSent
            })
            .expect("alice's MessageSent must exist");
        assert_eq!(
            first_alice.details.get("email").map(String::as_str),
            Some("[ERASED]")
        );
        assert_eq!(
            first_alice
                .details
                .get("content_length")
                .map(String::as_str),
            Some("42")
        );

        // A DataErased event has been appended carrying the SHA-256 hash.
        let data_erased = events
            .iter()
            .find(|e| e.event_type == AuditEventType::DataErased)
            .expect("DataErased event must be appended");
        assert_eq!(
            data_erased
                .details
                .get("user_id_sha256")
                .map(String::as_str),
            Some(hash_user_id("alice").as_str())
        );
        assert_eq!(
            data_erased
                .details
                .get("total_records_removed")
                .map(String::as_str),
            Some("1")
        );
        // user_id field is intentionally None to avoid PII leakage.
        assert!(data_erased.user_id.is_none());
    }

    #[test]
    fn idempotent_when_user_has_no_data() {
        let mut adapters: Vec<Box<dyn PurgeAdapter>> =
            vec![Box::new(vec_adapter("test.empty", &[("alice", &["one"])]))];
        // First call removes 1.
        let r1 = purge_user("alice", &mut adapters, None).unwrap();
        assert_eq!(r1.total_records, 1);
        // Second call removes 0 (idempotent), no failures.
        let r2 = purge_user("alice", &mut adapters, None).unwrap();
        assert_eq!(r2.total_records, 0);
        assert!(r2.partial_failures.is_empty());
    }

    #[test]
    fn report_has_consistent_timing() {
        let mut adapters: Vec<Box<dyn PurgeAdapter>> =
            vec![Box::new(vec_adapter("test.timing", &[("alice", &["one"])]))];
        let report = purge_user("alice", &mut adapters, None).unwrap();
        assert!(report.finished_at >= report.started_at);
        // SHA-256 hex always 64 chars.
        assert_eq!(report.user_id_sha256.len(), 64);
    }

    #[test]
    fn map_purge_adapter_matches_trait() {
        let mut store: HashMap<String, u64> = HashMap::new();
        store.insert("alice".into(), 7);
        store.insert("bob".into(), 3);
        {
            let mut adapter = MapPurgeAdapter::new("test.counters", &mut store);
            assert_eq!(adapter.name(), "test.counters");
            assert_eq!(adapter.purge_user("alice").unwrap(), 1);
            assert_eq!(adapter.purge_user("alice").unwrap(), 0); // idempotent
        }
        assert!(!store.contains_key("alice"));
        assert!(store.contains_key("bob"));
    }
}
