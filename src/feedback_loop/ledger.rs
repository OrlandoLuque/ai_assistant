//! `DispatchLedger` + `RetractionLedger` — hash-chained event logs for the
//! feedback loop.
//!
//! Mirrors `prompt_synthesis::FragmentLedger` in shape so auditors see a
//! consistent record across subsystems. Each event carries Blake3 self-hash,
//! chain-link to the previous event, and an optional Ed25519 signature via
//! the `DispatchSigner` trait (re-uses the same pattern as the rest of
//! V96).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use super::trajectory::{Outcome, PrivacyTier, TrajectoryId};

// =============================================================================
// Events
// =============================================================================

/// One entry in the dispatch ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DispatchEvent {
    pub seq: u64,
    pub prev_hash_hex: String,
    pub self_hash_hex: String,
    pub signature_hex: String,
    pub signer: String,
    pub timestamp: DateTime<Utc>,
    pub kind: DispatchEventKind,
}

impl DispatchEvent {
    pub fn verify_self_hash(&self) -> bool {
        self.self_hash_hex == compute_self_hash(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DispatchEventKind {
    /// A trajectory reached the dispatcher.
    TrajectoryReceived {
        trajectory: TrajectoryId,
        principal: String,
        privacy: PrivacyTier,
        outcome: Outcome,
    },
    /// Dispatched to a named sink successfully.
    SinkDispatched {
        trajectory: TrajectoryId,
        sink: String,
    },
    /// A sink rejected or errored on a trajectory.
    SinkFailed {
        trajectory: TrajectoryId,
        sink: String,
        reason: String,
    },
    /// Trajectory dropped before dispatch (freeze / privacy / insufficient reward).
    TrajectoryDropped {
        trajectory: TrajectoryId,
        reason: String,
    },
    /// A retraction was requested for a previously-dispatched trajectory.
    RetractionRequested {
        trajectory: TrajectoryId,
        reason: String,
    },
    /// Retraction executed against a sink (sink was told to tombstone).
    RetractionPropagated {
        trajectory: TrajectoryId,
        sink: String,
    },
    /// Freeze toggled.
    FreezeChanged { frozen: bool },
}

// =============================================================================
// Hash + signer
// =============================================================================

fn canonical_bytes_for_hashing(ev: &DispatchEvent) -> Vec<u8> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        seq: u64,
        prev_hash_hex: &'a str,
        signer: &'a str,
        timestamp: DateTime<Utc>,
        kind: &'a DispatchEventKind,
    }
    let c = Canonical {
        seq: ev.seq,
        prev_hash_hex: &ev.prev_hash_hex,
        signer: &ev.signer,
        timestamp: ev.timestamp,
        kind: &ev.kind,
    };
    serde_json::to_vec(&c).unwrap_or_else(|_| format!("serialize-error:{}", ev.seq).into_bytes())
}

fn compute_self_hash(ev: &DispatchEvent) -> String {
    hash_hex(&canonical_bytes_for_hashing(ev))
}

fn hash_hex(bytes: &[u8]) -> String {
    #[cfg(feature = "feedback-loop")]
    {
        blake3::hash(bytes).to_hex().to_string()
    }
    #[cfg(not(feature = "feedback-loop"))]
    {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        format!("defhash:{:016x}", h.finish())
    }
}

/// Signs each event's self-hash. `NoopDispatchSigner` is the default for tests.
pub trait DispatchSigner: Send + Sync {
    fn signer_id(&self) -> String;
    fn sign(&self, self_hash_hex: &str) -> String;
}

pub struct NoopDispatchSigner {
    id: String,
}

impl NoopDispatchSigner {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl DispatchSigner for NoopDispatchSigner {
    fn signer_id(&self) -> String {
        self.id.clone()
    }
    fn sign(&self, _self_hash_hex: &str) -> String {
        String::new()
    }
}

// =============================================================================
// Ledger
// =============================================================================

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DispatchLedgerError {
    ChainBroken { seq: u64, reason: String },
    SelfHashMismatch { seq: u64 },
    Poisoned,
    SeqGap { expected: u64, got: u64 },
}

impl std::fmt::Display for DispatchLedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainBroken { seq, reason } => {
                write!(f, "chain broken at seq {seq}: {reason}")
            }
            Self::SelfHashMismatch { seq } => write!(f, "self-hash mismatch at seq {seq}"),
            Self::Poisoned => f.write_str("dispatch ledger poisoned"),
            Self::SeqGap { expected, got } => {
                write!(f, "sequence gap: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for DispatchLedgerError {}

#[derive(Debug, Default)]
struct LedgerInner {
    events: Vec<DispatchEvent>,
}

#[derive(Clone)]
pub struct DispatchLedger {
    signer: Arc<dyn DispatchSigner>,
    inner: Arc<RwLock<LedgerInner>>,
}

impl DispatchLedger {
    pub fn new(signer: Arc<dyn DispatchSigner>) -> Self {
        Self {
            signer,
            inner: Arc::new(RwLock::new(LedgerInner::default())),
        }
    }

    pub fn append(&self, kind: DispatchEventKind) -> Result<DispatchEvent, DispatchLedgerError> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| DispatchLedgerError::Poisoned)?;
        let seq = inner.events.len() as u64;
        let prev_hash_hex = inner
            .events
            .last()
            .map(|e| e.self_hash_hex.clone())
            .unwrap_or_default();
        let mut ev = DispatchEvent {
            seq,
            prev_hash_hex,
            self_hash_hex: String::new(),
            signature_hex: String::new(),
            signer: self.signer.signer_id(),
            timestamp: Utc::now(),
            kind,
        };
        ev.self_hash_hex = compute_self_hash(&ev);
        ev.signature_hex = self.signer.sign(&ev.self_hash_hex);
        inner.events.push(ev.clone());
        Ok(ev)
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|i| i.events.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn events(&self) -> Vec<DispatchEvent> {
        self.inner
            .read()
            .map(|i| i.events.clone())
            .unwrap_or_default()
    }

    pub fn verify_chain(&self) -> Result<(), DispatchLedgerError> {
        let inner = self
            .inner
            .read()
            .map_err(|_| DispatchLedgerError::Poisoned)?;
        let mut prev_self_hash: Option<String> = None;
        for (idx, ev) in inner.events.iter().enumerate() {
            if ev.seq != idx as u64 {
                return Err(DispatchLedgerError::SeqGap {
                    expected: idx as u64,
                    got: ev.seq,
                });
            }
            if !ev.verify_self_hash() {
                return Err(DispatchLedgerError::SelfHashMismatch { seq: ev.seq });
            }
            if let Some(prev) = &prev_self_hash {
                if &ev.prev_hash_hex != prev {
                    return Err(DispatchLedgerError::ChainBroken {
                        seq: ev.seq,
                        reason: format!("prev_hash stored={} expected={}", ev.prev_hash_hex, prev),
                    });
                }
            } else if !ev.prev_hash_hex.is_empty() {
                return Err(DispatchLedgerError::ChainBroken {
                    seq: ev.seq,
                    reason: "first event must have empty prev_hash".into(),
                });
            }
            prev_self_hash = Some(ev.self_hash_hex.clone());
        }
        Ok(())
    }
}

impl std::fmt::Debug for DispatchLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchLedger")
            .field("len", &self.len())
            .finish()
    }
}

// =============================================================================
// RetractionLedger (thin wrapper: same chain semantics, different focus)
// =============================================================================

/// Dedicated ledger for retraction tombstones. Shares the chain semantics
/// with `DispatchLedger` but carries only retraction-related events. Keeps
/// GDPR/retraction audits separable from normal dispatch events.
#[derive(Clone, Debug)]
pub struct RetractionLedger {
    inner: DispatchLedger,
}

impl RetractionLedger {
    pub fn new(signer: Arc<dyn DispatchSigner>) -> Self {
        Self {
            inner: DispatchLedger::new(signer),
        }
    }

    pub fn record_request(
        &self,
        trajectory: &TrajectoryId,
        reason: impl Into<String>,
    ) -> Result<DispatchEvent, DispatchLedgerError> {
        self.inner.append(DispatchEventKind::RetractionRequested {
            trajectory: trajectory.clone(),
            reason: reason.into(),
        })
    }

    pub fn record_propagated(
        &self,
        trajectory: &TrajectoryId,
        sink: impl Into<String>,
    ) -> Result<DispatchEvent, DispatchLedgerError> {
        self.inner.append(DispatchEventKind::RetractionPropagated {
            trajectory: trajectory.clone(),
            sink: sink.into(),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn events(&self) -> Vec<DispatchEvent> {
        self.inner.events()
    }

    pub fn verify_chain(&self) -> Result<(), DispatchLedgerError> {
        self.inner.verify_chain()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn signer() -> Arc<dyn DispatchSigner> {
        Arc::new(NoopDispatchSigner::new("node-a"))
    }

    fn sample_kind() -> DispatchEventKind {
        DispatchEventKind::TrajectoryReceived {
            trajectory: TrajectoryId::new(),
            principal: "alice".into(),
            privacy: PrivacyTier::Internal,
            outcome: Outcome::Success,
        }
    }

    #[test]
    fn new_dispatch_ledger_is_empty() {
        let l = DispatchLedger::new(signer());
        assert!(l.is_empty());
        assert!(l.verify_chain().is_ok());
    }

    #[test]
    fn append_links_prev_hash() {
        let l = DispatchLedger::new(signer());
        let e0 = l.append(sample_kind()).unwrap();
        let e1 = l.append(sample_kind()).unwrap();
        assert!(e0.prev_hash_hex.is_empty());
        assert_eq!(e1.prev_hash_hex, e0.self_hash_hex);
    }

    #[test]
    fn dispatch_verify_chain_ok_after_many_appends() {
        let l = DispatchLedger::new(signer());
        for _ in 0..5 {
            l.append(sample_kind()).unwrap();
        }
        l.verify_chain().unwrap();
    }

    #[test]
    fn dispatch_self_hash_roundtrips() {
        let l = DispatchLedger::new(signer());
        let e = l.append(sample_kind()).unwrap();
        assert!(e.verify_self_hash());
    }

    #[test]
    fn retraction_ledger_records_requests_and_propagations() {
        let l = RetractionLedger::new(signer());
        let t = TrajectoryId::new();
        let req = l.record_request(&t, "gdpr").unwrap();
        let prop = l.record_propagated(&t, "memory_sink").unwrap();
        assert_eq!(req.seq, 0);
        assert_eq!(prop.seq, 1);
        assert_eq!(l.len(), 2);
        l.verify_chain().unwrap();
    }

    #[test]
    fn noop_signer_signature_is_empty() {
        let l = DispatchLedger::new(signer());
        let e = l.append(sample_kind()).unwrap();
        assert!(e.signature_hex.is_empty());
    }
}
