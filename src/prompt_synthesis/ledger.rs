//! `FragmentLedger` — hash-chained event log for prompt-synthesis activity.
//!
//! Mirrors the pattern established by `skill_forge::ledger`. Each event
//! carries its own Blake3 self-hash + chain link to the previous event. Ed25519
//! signatures ride on top via the same `LedgerSigner` trait re-used here.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use super::arm::{ArmOrigin, PromptArmId, ProviderFingerprint};
use super::bandit::SelectionReason;
use super::intent::IntentClusterId;

// =============================================================================
// Event shape
// =============================================================================

/// One entry in the fragment ledger. Immutable once appended.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FragmentEvent {
    pub seq: u64,
    pub prev_hash_hex: String,
    pub self_hash_hex: String,
    pub signature_hex: String,
    pub signer: String,
    pub timestamp: DateTime<Utc>,
    pub kind: FragmentEventKind,
}

impl FragmentEvent {
    /// Recompute the canonical hash and compare to the stored one.
    pub fn verify_self_hash(&self) -> bool {
        self.self_hash_hex == compute_self_hash(self)
    }
}

/// What was recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FragmentEventKind {
    ArmCreated {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
        arm: PromptArmId,
        origin: ArmOrigin,
    },
    ArmSelected {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
        arm: PromptArmId,
        reason: SelectionReason,
        score: f32,
    },
    RewardRecorded {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
        arm: PromptArmId,
        reward: f32,
    },
    ArmRetired {
        cluster: IntentClusterId,
        provider: ProviderFingerprint,
        arm: PromptArmId,
        reason: String,
    },
    ClusterResized {
        before: usize,
        after: usize,
        removed: usize,
    },
    FreezeChanged {
        frozen: bool,
    },
}

// =============================================================================
// Hash + signer
// =============================================================================

fn canonical_bytes_for_hashing(ev: &FragmentEvent) -> Vec<u8> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        seq: u64,
        prev_hash_hex: &'a str,
        signer: &'a str,
        timestamp: DateTime<Utc>,
        kind: &'a FragmentEventKind,
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

fn compute_self_hash(ev: &FragmentEvent) -> String {
    hash_hex(&canonical_bytes_for_hashing(ev))
}

fn hash_hex(bytes: &[u8]) -> String {
    #[cfg(feature = "prompt-synthesis")]
    {
        blake3::hash(bytes).to_hex().to_string()
    }
    #[cfg(not(feature = "prompt-synthesis"))]
    {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        format!("defhash:{:016x}", h.finish())
    }
}

/// Signer used to produce the `signature_hex` field. Implementations may
/// leave it empty (unsigned) or produce an Ed25519 signature.
pub trait FragmentSigner: Send + Sync {
    fn signer_id(&self) -> String;
    fn sign(&self, self_hash_hex: &str) -> String;
}

/// Signer that never signs — emits empty strings. Convenient default for
/// tests and single-tenant deployments where Ed25519 is overkill.
pub struct NoopFragmentSigner {
    id: String,
}

impl NoopFragmentSigner {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl FragmentSigner for NoopFragmentSigner {
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
pub enum FragmentLedgerError {
    /// Chain integrity failure at the given seq.
    ChainBroken { seq: u64, reason: String },
    /// Self-hash does not match canonical.
    SelfHashMismatch { seq: u64 },
    /// Lock poisoned.
    Poisoned,
    /// Sequence gap (events not contiguous from 0).
    SeqGap { expected: u64, got: u64 },
}

impl std::fmt::Display for FragmentLedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainBroken { seq, reason } => {
                write!(f, "chain broken at seq {seq}: {reason}")
            }
            Self::SelfHashMismatch { seq } => write!(f, "self-hash mismatch at seq {seq}"),
            Self::Poisoned => f.write_str("ledger lock poisoned"),
            Self::SeqGap { expected, got } => {
                write!(f, "sequence gap: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for FragmentLedgerError {}

#[derive(Debug, Default)]
struct LedgerInner {
    events: Vec<FragmentEvent>,
}

/// Append-only hash-chained ledger. Thread-safe via internal `RwLock`.
#[derive(Clone)]
pub struct FragmentLedger {
    signer: Arc<dyn FragmentSigner>,
    inner: Arc<RwLock<LedgerInner>>,
}

impl FragmentLedger {
    pub fn new(signer: Arc<dyn FragmentSigner>) -> Self {
        Self {
            signer,
            inner: Arc::new(RwLock::new(LedgerInner::default())),
        }
    }

    /// Append a new event. Returns a clone of the event as appended (with
    /// seq/prev/self/signature filled in).
    pub fn append(&self, kind: FragmentEventKind) -> Result<FragmentEvent, FragmentLedgerError> {
        let mut inner = self
            .inner
            .write()
            .map_err(|_| FragmentLedgerError::Poisoned)?;
        let seq = inner.events.len() as u64;
        let prev_hash_hex = inner
            .events
            .last()
            .map(|e| e.self_hash_hex.clone())
            .unwrap_or_default();
        let mut ev = FragmentEvent {
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

    /// Cloned snapshot of all events — safe to iterate without locking.
    pub fn events(&self) -> Vec<FragmentEvent> {
        self.inner
            .read()
            .map(|i| i.events.clone())
            .unwrap_or_default()
    }

    /// Verify the full hash chain + self-hashes. Does NOT verify Ed25519
    /// signatures — callers who need that must provide their own verifier.
    pub fn verify_chain(&self) -> Result<(), FragmentLedgerError> {
        let inner = self
            .inner
            .read()
            .map_err(|_| FragmentLedgerError::Poisoned)?;
        let mut prev_self_hash: Option<String> = None;
        for (idx, ev) in inner.events.iter().enumerate() {
            if ev.seq != idx as u64 {
                return Err(FragmentLedgerError::SeqGap {
                    expected: idx as u64,
                    got: ev.seq,
                });
            }
            if !ev.verify_self_hash() {
                return Err(FragmentLedgerError::SelfHashMismatch { seq: ev.seq });
            }
            if let Some(prev) = &prev_self_hash {
                if &ev.prev_hash_hex != prev {
                    return Err(FragmentLedgerError::ChainBroken {
                        seq: ev.seq,
                        reason: format!("prev_hash stored={} expected={}", ev.prev_hash_hex, prev),
                    });
                }
            } else if !ev.prev_hash_hex.is_empty() {
                return Err(FragmentLedgerError::ChainBroken {
                    seq: ev.seq,
                    reason: "first event must have empty prev_hash".into(),
                });
            }
            prev_self_hash = Some(ev.self_hash_hex.clone());
        }
        Ok(())
    }
}

impl std::fmt::Debug for FragmentLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FragmentLedger")
            .field("len", &self.len())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn signer() -> Arc<dyn FragmentSigner> {
        Arc::new(NoopFragmentSigner::new("tenant-a/ep0"))
    }

    fn sample_event_kind() -> FragmentEventKind {
        FragmentEventKind::ArmCreated {
            cluster: IntentClusterId(0),
            provider: ProviderFingerprint::new("ollama", "m"),
            arm: PromptArmId::new("a"),
            origin: ArmOrigin::Manual,
        }
    }

    #[test]
    fn new_ledger_is_empty() {
        let l = FragmentLedger::new(signer());
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        assert!(l.verify_chain().is_ok());
    }

    #[test]
    fn append_increments_seq_and_links_prev_hash() {
        let l = FragmentLedger::new(signer());
        let e0 = l.append(sample_event_kind()).unwrap();
        let e1 = l.append(sample_event_kind()).unwrap();
        assert_eq!(e0.seq, 0);
        assert_eq!(e1.seq, 1);
        assert!(e0.prev_hash_hex.is_empty());
        assert_eq!(e1.prev_hash_hex, e0.self_hash_hex);
    }

    #[test]
    fn verify_chain_ok_on_intact_ledger() {
        let l = FragmentLedger::new(signer());
        for _ in 0..5 {
            l.append(sample_event_kind()).unwrap();
        }
        l.verify_chain().unwrap();
    }

    #[test]
    fn self_hash_roundtrips() {
        let l = FragmentLedger::new(signer());
        let e = l.append(sample_event_kind()).unwrap();
        assert!(e.verify_self_hash());
    }

    #[test]
    fn noop_signer_produces_empty_signature() {
        let l = FragmentLedger::new(signer());
        let e = l.append(sample_event_kind()).unwrap();
        assert!(e.signature_hex.is_empty());
    }

    #[test]
    fn signer_id_carried_in_events() {
        let l = FragmentLedger::new(signer());
        let e = l.append(sample_event_kind()).unwrap();
        assert_eq!(e.signer, "tenant-a/ep0");
    }
}
