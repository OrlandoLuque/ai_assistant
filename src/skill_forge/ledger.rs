//! Hash-chained, optionally-signed ledger of skill registry events.
//!
//! The ledger is the **source of truth for history**: every insert / promote /
//! retract / integrity-check emits a `LedgerEvent` with a Blake3 `prev_hash`
//! linking it to the previous entry (Bitcoin / git-commit style). When the
//! `skill-forge` feature is on, events are additionally signed with Ed25519;
//! verifiers check both the chain and the signatures.
//!
//! Storage: in-memory `Vec<LedgerEvent>` with an optional `append_to_file`
//! hook. Retention / compaction lives above this module.
//!
//! RBAC is enforced at the auditor API layer (see `ai_skills` binary), not
//! here — the ledger itself accepts all appends from trusted callers.

use super::registry::{SkillId, SkillStatus, SkillVersion};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, RwLock};

// =============================================================================
// Event model
// =============================================================================

/// One entry in the skill ledger. Immutable once appended.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LedgerEvent {
    /// Monotonic sequence number, starts at 0.
    pub seq: u64,
    /// Blake3 hash of the previous event's canonical bytes, hex-encoded.
    /// Empty string for `seq == 0`.
    pub prev_hash_hex: String,
    /// Blake3 hash of THIS event's canonical bytes (excluding `self_hash_hex`
    /// and `signature_hex`), hex-encoded. Computed by `compute_self_hash`.
    pub self_hash_hex: String,
    /// Optional Ed25519 signature over `self_hash_hex`, hex-encoded.
    /// Empty when the `skill-forge` feature is off or no signer is configured.
    pub signature_hex: String,
    /// Signer identifier: tenant + key rotation epoch (e.g. `tenant-a/ep3`).
    pub signer: String,
    /// UTC timestamp.
    pub timestamp: DateTime<Utc>,
    /// The event payload.
    pub kind: LedgerEventKind,
}

/// What happened.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LedgerEventKind {
    /// A new skill (id, version) was registered.
    Registered {
        skill: SkillId,
        version: SkillVersion,
        content_hash_hex: String,
    },
    /// A skill transitioned to a new status.
    StatusChanged {
        skill: SkillId,
        version: SkillVersion,
        from: SkillStatus,
        to: SkillStatus,
        reason: String,
    },
    /// A skill was retracted (soft delete).
    Retracted {
        skill: SkillId,
        version: SkillVersion,
        reason: String,
    },
    /// An integrity check was performed. Records expected vs actual hash.
    IntegrityCheck {
        skill: SkillId,
        version: SkillVersion,
        expected_hex: String,
        actual_hex: String,
        passed: bool,
    },
    /// An auditor viewed the ledger. "Audit the auditor" trail.
    AuditAccess { viewer: String, scope: String },
}

impl LedgerEvent {
    /// Compute the Blake3 hash of this event's canonical bytes, excluding
    /// `self_hash_hex` and `signature_hex`.
    pub fn compute_self_hash(&self) -> String {
        let bytes = canonical_bytes_for_hashing(self);
        hash_hex(&bytes)
    }

    /// Check: recompute this event's self-hash and compare to stored.
    pub fn verify_self_hash(&self) -> bool {
        self.self_hash_hex == self.compute_self_hash()
    }
}

/// Serialize everything except `self_hash_hex` and `signature_hex` — those
/// are derived from the rest.
fn canonical_bytes_for_hashing(ev: &LedgerEvent) -> Vec<u8> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        seq: u64,
        prev_hash_hex: &'a str,
        signer: &'a str,
        timestamp: DateTime<Utc>,
        kind: &'a LedgerEventKind,
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

// =============================================================================
// Hash + signature backends (feature-gated)
// =============================================================================

#[cfg(feature = "skill-forge")]
fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(not(feature = "skill-forge"))]
fn hash_hex(bytes: &[u8]) -> String {
    // Stable FNV-1a fallback. NOT cryptographically secure; used only when
    // the `skill-forge` feature is off so the type is still usable in tests
    // and downstream crates.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("fnv64:{h:016x}")
}

// =============================================================================
// Signer abstraction
// =============================================================================

/// Trait for signing ledger events. Production uses an Ed25519-backed impl
/// (requires `skill-forge`); tests can use `NoopSigner`.
pub trait LedgerSigner: Send + Sync {
    /// Sign the given bytes. Returns hex-encoded signature.
    /// Empty string means "no signature" (valid for unsigned ledgers).
    fn sign(&self, message: &[u8]) -> String;
    /// Identifier of this signer, recorded in each event.
    fn signer_id(&self) -> String;
}

/// No-op signer: returns empty signature. Use in tests or when signing is
/// disabled.
#[derive(Debug, Clone, Default)]
pub struct NoopSigner {
    pub id: String,
}

impl LedgerSigner for NoopSigner {
    fn sign(&self, _message: &[u8]) -> String {
        String::new()
    }
    fn signer_id(&self) -> String {
        self.id.clone()
    }
}

/// Ed25519 signer backed by `ed25519-dalek`. Only available with the
/// `skill-forge` feature.
#[cfg(feature = "skill-forge")]
pub struct Ed25519Signer {
    key: ed25519_dalek::SigningKey,
    id: String,
}

#[cfg(feature = "skill-forge")]
impl Ed25519Signer {
    /// Construct from an existing signing key.
    pub fn new(key: ed25519_dalek::SigningKey, id: impl Into<String>) -> Self {
        Self { key, id: id.into() }
    }

    /// Derive a tenant-specific signing key from master bytes via HKDF-SHA256.
    ///
    /// The master key is never used directly; each tenant (identified by
    /// `tenant_id` and `epoch`) gets its own derived 32-byte seed.
    pub fn derive_from_master(
        master: &[u8],
        tenant_id: &str,
        epoch: u32,
    ) -> Result<Self, LedgerVerifyError> {
        let hk = hkdf::Hkdf::<sha2::Sha256>::new(None, master);
        let info = format!("ai_assistant/skill_forge/v1/tenant={tenant_id}/epoch={epoch}");
        let mut seed = [0u8; 32];
        hk.expand(info.as_bytes(), &mut seed)
            .map_err(|e| LedgerVerifyError::KeyDerivation(e.to_string()))?;
        let key = ed25519_dalek::SigningKey::from_bytes(&seed);
        let id = format!("{tenant_id}/ep{epoch}");
        Ok(Self::new(key, id))
    }

    /// Verifying key (public) for this signer.
    pub fn verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        self.key.verifying_key()
    }
}

#[cfg(feature = "skill-forge")]
impl LedgerSigner for Ed25519Signer {
    fn sign(&self, message: &[u8]) -> String {
        use ed25519_dalek::Signer;
        let sig = self.key.sign(message);
        hex_encode(&sig.to_bytes())
    }
    fn signer_id(&self) -> String {
        self.id.clone()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = from_hex(bytes[i])?;
        let lo = from_hex(bytes[i + 1])?;
        out.push(hi * 16 + lo);
        i += 2;
    }
    Some(out)
}

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// =============================================================================
// Verification errors
// =============================================================================

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LedgerVerifyError {
    SeqGap { expected: u64, actual: u64 },
    ChainBroken { at_seq: u64 },
    SelfHashMismatch { at_seq: u64 },
    SignatureInvalid { at_seq: u64, signer: String },
    SignatureMissing { at_seq: u64 },
    KeyDerivation(String),
}

impl fmt::Display for LedgerVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeqGap { expected, actual } => {
                write!(f, "sequence gap: expected {expected}, got {actual}")
            }
            Self::ChainBroken { at_seq } => write!(f, "chain broken at seq {at_seq}"),
            Self::SelfHashMismatch { at_seq } => write!(f, "self-hash mismatch at seq {at_seq}"),
            Self::SignatureInvalid { at_seq, signer } => {
                write!(f, "invalid signature at seq {at_seq} signer={signer}")
            }
            Self::SignatureMissing { at_seq } => write!(f, "signature missing at seq {at_seq}"),
            Self::KeyDerivation(e) => write!(f, "key derivation error: {e}"),
        }
    }
}

impl std::error::Error for LedgerVerifyError {}

// =============================================================================
// Ledger
// =============================================================================

/// Append-only hash-chained ledger. Thread-safe.
#[derive(Clone)]
pub struct SkillLedger {
    inner: Arc<RwLock<LedgerInner>>,
    signer: Arc<dyn LedgerSigner>,
}

struct LedgerInner {
    events: Vec<LedgerEvent>,
}

impl SkillLedger {
    /// New empty ledger with the given signer.
    pub fn new(signer: Arc<dyn LedgerSigner>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LedgerInner { events: Vec::new() })),
            signer,
        }
    }

    /// Append an event. Fills in `seq`, `prev_hash_hex`, `self_hash_hex`,
    /// `signature_hex`, `signer`, `timestamp`. Caller provides `kind`.
    pub fn append(&self, kind: LedgerEventKind) -> LedgerEvent {
        let mut inner = match self.inner.write() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let seq = inner.events.len() as u64;
        let prev_hash_hex = inner
            .events
            .last()
            .map(|e| e.self_hash_hex.clone())
            .unwrap_or_default();
        let mut ev = LedgerEvent {
            seq,
            prev_hash_hex,
            self_hash_hex: String::new(),
            signature_hex: String::new(),
            signer: self.signer.signer_id(),
            timestamp: Utc::now(),
            kind,
        };
        ev.self_hash_hex = ev.compute_self_hash();
        ev.signature_hex = self.signer.sign(ev.self_hash_hex.as_bytes());
        inner.events.push(ev.clone());
        ev
    }

    /// Return a snapshot of all events.
    pub fn events(&self) -> Vec<LedgerEvent> {
        self.inner
            .read()
            .map(|g| g.events.clone())
            .unwrap_or_default()
    }

    /// Number of events in the ledger.
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.events.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Verify the full chain: sequence, prev_hash linkage, self-hash integrity.
    /// Does NOT verify signatures (use `verify_signatures` for that).
    pub fn verify_chain(&self) -> Result<(), LedgerVerifyError> {
        let events = self.events();
        for (i, ev) in events.iter().enumerate() {
            let expected_seq = i as u64;
            if ev.seq != expected_seq {
                return Err(LedgerVerifyError::SeqGap {
                    expected: expected_seq,
                    actual: ev.seq,
                });
            }
            if !ev.verify_self_hash() {
                return Err(LedgerVerifyError::SelfHashMismatch { at_seq: ev.seq });
            }
            let expected_prev = if i == 0 {
                String::new()
            } else {
                events[i - 1].self_hash_hex.clone()
            };
            if ev.prev_hash_hex != expected_prev {
                return Err(LedgerVerifyError::ChainBroken { at_seq: ev.seq });
            }
        }
        Ok(())
    }

    /// Verify signatures using the supplied verifier. `verifier` receives
    /// `(signer_id, message_bytes, signature_hex)` and returns whether the
    /// signature validates.
    pub fn verify_signatures<F>(&self, verifier: F) -> Result<(), LedgerVerifyError>
    where
        F: Fn(&str, &[u8], &str) -> bool,
    {
        let events = self.events();
        for ev in &events {
            if ev.signature_hex.is_empty() {
                // Unsigned event — skip (not all ledgers sign).
                continue;
            }
            if !verifier(&ev.signer, ev.self_hash_hex.as_bytes(), &ev.signature_hex) {
                return Err(LedgerVerifyError::SignatureInvalid {
                    at_seq: ev.seq,
                    signer: ev.signer.clone(),
                });
            }
        }
        Ok(())
    }
}

impl fmt::Debug for SkillLedger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkillLedger")
            .field("len", &self.len())
            .field("signer", &self.signer.signer_id())
            .finish()
    }
}

// =============================================================================
// Utility: verify an Ed25519 signature against a public key.
// =============================================================================

#[cfg(feature = "skill-forge")]
pub fn verify_ed25519(
    verifying_key: &ed25519_dalek::VerifyingKey,
    message: &[u8],
    signature_hex: &str,
) -> bool {
    let Some(sig_bytes) = hex_decode(signature_hex) else {
        return false;
    };
    let Ok(sig_arr) = <[u8; 64]>::try_from(sig_bytes.as_slice()) else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(&sig_arr);
    use ed25519_dalek::Verifier;
    verifying_key.verify(message, &sig).is_ok()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger_noop() -> SkillLedger {
        SkillLedger::new(Arc::new(NoopSigner {
            id: "test/ep0".into(),
        }))
    }

    #[test]
    fn new_ledger_is_empty() {
        let l = ledger_noop();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn append_increments_seq() {
        let l = ledger_noop();
        let e1 = l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        let e2 = l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s2"),
            version: SkillVersion(1),
            content_hash_hex: "def".into(),
        });
        assert_eq!(e1.seq, 0);
        assert_eq!(e2.seq, 1);
    }

    #[test]
    fn append_links_prev_hash() {
        let l = ledger_noop();
        let e1 = l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        let e2 = l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s2"),
            version: SkillVersion(1),
            content_hash_hex: "def".into(),
        });
        assert_eq!(e2.prev_hash_hex, e1.self_hash_hex);
        assert_eq!(e1.prev_hash_hex, "");
    }

    #[test]
    fn verify_chain_on_intact_ledger_ok() {
        let l = ledger_noop();
        for i in 0..5 {
            l.append(LedgerEventKind::Registered {
                skill: SkillId::new(format!("s{i}")),
                version: SkillVersion(1),
                content_hash_hex: format!("h{i}"),
            });
        }
        l.verify_chain().expect("chain ok");
    }

    #[test]
    fn self_hash_mismatch_detected() {
        let l = ledger_noop();
        l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        // Tamper with the events by rewriting the inner vec.
        {
            let mut inner = l.inner.write().unwrap();
            inner.events[0].self_hash_hex = "tampered".into();
        }
        let err = l.verify_chain().unwrap_err();
        match err {
            LedgerVerifyError::SelfHashMismatch { at_seq: 0 } => {}
            other => panic!("expected SelfHashMismatch, got {other}"),
        }
    }

    #[test]
    fn chain_broken_detected() {
        let l = ledger_noop();
        l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s2"),
            version: SkillVersion(1),
            content_hash_hex: "def".into(),
        });
        // Break the chain by rewriting prev_hash on event 1.
        {
            let mut inner = l.inner.write().unwrap();
            inner.events[1].prev_hash_hex = "bogus-prev".into();
            // Recompute self_hash so self-hash check passes — only the chain
            // linkage should fail.
            inner.events[1].self_hash_hex = inner.events[1].compute_self_hash();
        }
        let err = l.verify_chain().unwrap_err();
        match err {
            LedgerVerifyError::ChainBroken { at_seq: 1 } => {}
            other => panic!("expected ChainBroken, got {other}"),
        }
    }

    #[test]
    fn noop_signer_leaves_signature_empty() {
        let l = ledger_noop();
        let e = l.append(LedgerEventKind::AuditAccess {
            viewer: "orlando".into(),
            scope: "list".into(),
        });
        assert!(e.signature_hex.is_empty());
    }

    #[test]
    fn verify_signatures_skips_unsigned() {
        let l = ledger_noop();
        l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        l.verify_signatures(|_, _, _| false)
            .expect("no signed events => nothing to verify");
    }

    #[test]
    fn hex_roundtrip() {
        let v = vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0xff];
        let s = hex_encode(&v);
        assert_eq!(s, "deadbeef00ff");
        let back = hex_decode(&s).unwrap();
        assert_eq!(back, v);
    }

    #[cfg(feature = "skill-forge")]
    #[test]
    fn ed25519_signer_produces_verifiable_signature() {
        use ed25519_dalek::SigningKey;
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let signer = Ed25519Signer::new(key, "test/ep0");
        let vkey = signer.verifying_key();
        let l = SkillLedger::new(Arc::new(signer));
        let ev = l.append(LedgerEventKind::Registered {
            skill: SkillId::new("s1"),
            version: SkillVersion(1),
            content_hash_hex: "abc".into(),
        });
        assert!(!ev.signature_hex.is_empty());
        assert!(verify_ed25519(
            &vkey,
            ev.self_hash_hex.as_bytes(),
            &ev.signature_hex
        ));
    }

    #[cfg(feature = "skill-forge")]
    #[test]
    fn derive_from_master_is_deterministic() {
        let master = [42u8; 64];
        let s1 = Ed25519Signer::derive_from_master(&master, "tenant-a", 1).unwrap();
        let s2 = Ed25519Signer::derive_from_master(&master, "tenant-a", 1).unwrap();
        assert_eq!(s1.verifying_key().to_bytes(), s2.verifying_key().to_bytes());
        let s3 = Ed25519Signer::derive_from_master(&master, "tenant-b", 1).unwrap();
        assert_ne!(s1.verifying_key().to_bytes(), s3.verifying_key().to_bytes());
    }
}
