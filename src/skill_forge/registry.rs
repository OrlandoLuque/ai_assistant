//! Skill registry — in-memory store of `SkillDefinition`s keyed by `SkillId`.
//!
//! Mutations (insert / promote / retract) emit events to a `SkillLedger`
//! (see [`ledger`](super::ledger)). The registry itself holds current-state
//! views; the ledger is the source-of-truth for history.

use super::capability::CapabilitySet;
use super::declarative::SkillStep;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// ID and version
// =============================================================================

/// Stable identifier for a skill. Opaque string; conventionally `kebab-case`
/// plus short suffix (e.g. `extract-invoice-totals`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SkillId(pub String);

impl SkillId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SkillId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Monotonic version number per skill id. Starts at 1 for a new skill;
/// increments on every `insert` of a new definition with the same id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SkillVersion(pub u32);

impl SkillVersion {
    pub const INITIAL: SkillVersion = SkillVersion(1);
    pub fn next(self) -> Self {
        SkillVersion(self.0 + 1)
    }
}

impl fmt::Display for SkillVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// =============================================================================
// Execution mode
// =============================================================================

/// How a skill is executed. Two modes only — no additional runtime by design
/// (see feedback memory `feedback_minimize_execution_modes`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SkillMode {
    /// LLM-authored sequence of `SkillStep`s, interpreted by
    /// `DeclarativeExecutor`. Safe-by-construction — no code execution.
    Declarative(Vec<SkillStep>),

    /// LLM-authored Rust source compiled to `wasm32-wasip1` and executed in
    /// a wasmtime sandbox with fuel + memory limits + capability gating.
    Wasm(WasmArtifact),
}

/// Persistent WASM artifact: bytes, integrity hash, signature, and the fingerprint
/// of the compile toolchain (so we can detect environment drift).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WasmArtifact {
    /// WASM module bytes. Usually a few KiB to a few MiB.
    pub bytes: Vec<u8>,
    /// Blake3 hash of `bytes`, hex-encoded. Used by ledger + integrity check.
    pub blake3_hex: String,
    /// Ed25519 signature of `blake3_hex` (hex-encoded). Verified on load.
    pub ed25519_sig_hex: String,
    /// Identifier of the signing key (tenant id + key rotation epoch).
    pub signed_by: String,
    /// Fingerprint of the rustc/cargo toolchain that produced this artifact
    /// (e.g. `rustc 1.84.0 (..) target wasm32-wasip1`). Purely informational —
    /// used by auditors to correlate with build logs.
    pub compile_fingerprint: String,
    /// Source Rust path (relative to skill store root). `None` if source was
    /// discarded — not recommended; most tenants require source retention.
    pub source_path: Option<String>,
}

// =============================================================================
// Status lifecycle
// =============================================================================

/// Lifecycle stage of a skill, shared with the promotion pipeline.
///
/// Transitions:
/// - `Exploring` → `Exploited` (promotion gates passed)
/// - `Exploited` → `Frozen` (admin / LearningFreezeConfig)
/// - any → `Retired` (deprecated / failed quality)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SkillStatus {
    /// Newly registered, under evaluation. Reward samples accumulate.
    Exploring,
    /// Passed promotion gates. Preferred in dispatch.
    Exploited,
    /// Read-only — accepts invocations but not reward updates.
    Frozen,
    /// Deprecated — not dispatched. Artifacts kept for audit.
    Retired,
}

impl Default for SkillStatus {
    fn default() -> Self {
        Self::Exploring
    }
}

impl fmt::Display for SkillStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exploring => f.write_str("exploring"),
            Self::Exploited => f.write_str("exploited"),
            Self::Frozen => f.write_str("frozen"),
            Self::Retired => f.write_str("retired"),
        }
    }
}

// =============================================================================
// Skill definition
// =============================================================================

/// Full definition of a skill at a given version.
///
/// The content hash covers all fields except `status` (which is a runtime
/// label). `status` changes do not invalidate the hash — they are recorded
/// as ledger events instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SkillDefinition {
    pub id: SkillId,
    pub version: SkillVersion,
    /// Human-readable name. Not unique — `id` is the stable key.
    pub name: String,
    /// One-paragraph description of what the skill does.
    pub description: String,
    /// Execution mode.
    pub mode: SkillMode,
    /// Capabilities required to execute. Enforced at dispatch + WASM runtime.
    pub capabilities: CapabilitySet,
    /// Blake3 hash of the canonical serialization of this definition (hex).
    /// Used by ledger + integrity check. Computed by `content_hash()`.
    pub content_hash_hex: String,
    /// Current lifecycle status. Transient — use ledger for history.
    pub status: SkillStatus,
    /// Tenant that owns this skill. Cross-tenant dispatch is denied unless
    /// the skill is marked `shared_cross_tenant: true`.
    pub tenant: String,
    /// Opt-in sharing flag for multi-tenant deployments. Default false.
    pub shared_cross_tenant: bool,
}

impl SkillDefinition {
    /// Compute the canonical content hash from the in-memory fields.
    ///
    /// Status is excluded — it is a runtime label, not part of the
    /// immutable skill identity.
    pub fn compute_content_hash(&self) -> String {
        // Serialize a stable canonical form. We clone + null out `content_hash_hex`
        // and `status` so they don't contribute to the hash of themselves.
        #[derive(Serialize)]
        struct Canonical<'a> {
            id: &'a SkillId,
            version: SkillVersion,
            name: &'a str,
            description: &'a str,
            mode: &'a SkillMode,
            capabilities: &'a CapabilitySet,
            tenant: &'a str,
            shared_cross_tenant: bool,
        }
        let canonical = Canonical {
            id: &self.id,
            version: self.version,
            name: &self.name,
            description: &self.description,
            mode: &self.mode,
            capabilities: &self.capabilities,
            tenant: &self.tenant,
            shared_cross_tenant: self.shared_cross_tenant,
        };
        // serde_json is stable enough for canonicalization given we control
        // the Serialize impls. If we ever need true canonical JSON we'll
        // add a dedicated canonicalizer.
        let bytes = match serde_json::to_vec(&canonical) {
            Ok(b) => b,
            // If serialization fails (should not — our types are trivially
            // serializable), fall back to a sentinel hash. This avoids unwrap()
            // per project policy while still producing a unique-ish value.
            Err(_) => format!("serialize-error:{}:{}", self.id.0, self.version.0).into_bytes(),
        };
        #[cfg(feature = "skill-forge")]
        {
            let hash = blake3::hash(&bytes);
            return hash.to_hex().to_string();
        }
        #[cfg(not(feature = "skill-forge"))]
        {
            // Without skill-forge feature, fall back to a simple sha2-free
            // DefaultHasher. This path only exists so downstream crates can
            // reference the type without pulling blake3.
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            format!("defhash:{:016x}", h.finish())
        }
    }
}

/// Inputs passed to a skill invocation.
///
/// JSON-shaped — simple and language-agnostic. Extraction into strongly-typed
/// fields happens inside the skill (Declarative: via variable bindings;
/// WASM: via the `get_input` host import).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillInputs(pub serde_json::Value);

impl SkillInputs {
    pub fn new(v: serde_json::Value) -> Self {
        Self(v)
    }
    pub fn empty() -> Self {
        Self(serde_json::Value::Null)
    }
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.as_object().and_then(|o| o.get(key))
    }
}

/// Output of a skill invocation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillOutput {
    /// Structured result (JSON).
    pub value: serde_json::Value,
    /// Optional human-readable trace lines emitted during execution.
    pub trace: Vec<String>,
    /// Fuel / instructions consumed (WASM only). Informational.
    pub fuel_consumed: u64,
    /// Wall-clock time in milliseconds.
    pub wall_ms: u64,
}

// =============================================================================
// Errors
// =============================================================================

/// Errors surfaced by the skill subsystem.
///
/// No `thiserror` — aligns with the rest of the crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SkillError {
    /// Skill not found in the registry.
    NotFound(SkillId),
    /// Capability required was not granted.
    CapabilityDenied { skill: SkillId, capability: String },
    /// Content hash mismatch — persisted hash does not match recomputed one.
    IntegrityMismatch {
        skill: SkillId,
        expected: String,
        actual: String,
    },
    /// Ed25519 signature verification failed.
    SignatureInvalid { skill: SkillId, signer: String },
    /// Execution exceeded resource limits.
    ResourceExhausted { skill: SkillId, what: &'static str },
    /// Execution failed at runtime.
    ExecutionFailed { skill: SkillId, message: String },
    /// Cross-tenant access denied.
    CrossTenantDenied {
        skill: SkillId,
        caller_tenant: String,
        skill_tenant: String,
    },
    /// Skill is retired or frozen and cannot be invoked for writes.
    InvalidStatus {
        skill: SkillId,
        status: SkillStatus,
        operation: &'static str,
    },
    /// Input validation failed.
    BadInput { skill: SkillId, message: String },
    /// I/O error (artifact read / write).
    Io(String),
    /// Serialization error (JSON / bincode / etc).
    Serialization(String),
}

impl fmt::Display for SkillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "skill not found: {id}"),
            Self::CapabilityDenied { skill, capability } => {
                write!(f, "capability '{capability}' denied for skill {skill}")
            }
            Self::IntegrityMismatch { skill, expected, actual } => write!(
                f,
                "integrity mismatch for {skill}: expected={expected} actual={actual}"
            ),
            Self::SignatureInvalid { skill, signer } => {
                write!(f, "invalid signature for {skill} (signer={signer})")
            }
            Self::ResourceExhausted { skill, what } => {
                write!(f, "resource exhausted for {skill}: {what}")
            }
            Self::ExecutionFailed { skill, message } => {
                write!(f, "execution failed for {skill}: {message}")
            }
            Self::CrossTenantDenied { skill, caller_tenant, skill_tenant } => write!(
                f,
                "cross-tenant access denied: caller={caller_tenant} skill={skill} owner={skill_tenant}"
            ),
            Self::InvalidStatus { skill, status, operation } => write!(
                f,
                "skill {skill} is {status}; operation {operation} not allowed"
            ),
            Self::BadInput { skill, message } => write!(f, "bad input for {skill}: {message}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl std::error::Error for SkillError {}

/// Errors specific to registry operations (insert / promote / retract).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SkillRegistryError {
    /// A skill with this id+version already exists.
    DuplicateVersion {
        skill: SkillId,
        version: SkillVersion,
    },
    /// Attempted to promote a non-existent skill.
    NotFound(SkillId),
    /// Illegal state transition (e.g. Retired → Exploiting).
    IllegalTransition {
        skill: SkillId,
        from: SkillStatus,
        to: SkillStatus,
    },
    /// Underlying skill error (integrity / signature).
    Skill(SkillError),
    /// Registry is frozen (`LearningFreezeConfig::freeze_skill_forge == true`).
    /// Mutations (insert / status change) are rejected until unfrozen.
    Frozen,
}

impl fmt::Display for SkillRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateVersion { skill, version } => {
                write!(f, "duplicate skill version: {skill} {version}")
            }
            Self::NotFound(id) => write!(f, "skill not found: {id}"),
            Self::IllegalTransition { skill, from, to } => {
                write!(f, "illegal status transition for {skill}: {from} -> {to}")
            }
            Self::Skill(e) => write!(f, "{e}"),
            Self::Frozen => f.write_str("skill forge is frozen — mutation rejected"),
        }
    }
}

impl std::error::Error for SkillRegistryError {}

impl From<SkillError> for SkillRegistryError {
    fn from(e: SkillError) -> Self {
        Self::Skill(e)
    }
}

// =============================================================================
// Registry
// =============================================================================

/// Thread-safe in-memory registry of skill definitions.
///
/// Persistence is a separate concern — the registry is the authoritative
/// runtime view. Audit binaries read the `SkillLedger` for history.
#[derive(Debug, Default, Clone)]
pub struct SkillRegistry {
    inner: Arc<RwLock<RegistryInner>>,
    frozen: Arc<AtomicBool>,
}

#[derive(Debug, Default)]
struct RegistryInner {
    /// All versions per skill id. Latest version is `by_id[id].last()`.
    by_id: HashMap<SkillId, Vec<SkillDefinition>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runtime freeze switch. Mirrors `LearningFreezeConfig::freeze_skill_forge`.
    /// When frozen, `insert` and `set_status` reject with `SkillRegistryError::Frozen`.
    pub fn set_frozen(&self, frozen: bool) {
        self.frozen.store(frozen, Ordering::SeqCst);
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::SeqCst)
    }

    /// Insert a new skill definition. Fails if `(id, version)` already exists.
    pub fn insert(&self, def: SkillDefinition) -> Result<(), SkillRegistryError> {
        if self.is_frozen() {
            return Err(SkillRegistryError::Frozen);
        }
        // Verify integrity at insert time: recompute content hash and compare.
        let expected = def.compute_content_hash();
        if expected != def.content_hash_hex {
            return Err(SkillRegistryError::Skill(SkillError::IntegrityMismatch {
                skill: def.id.clone(),
                expected,
                actual: def.content_hash_hex.clone(),
            }));
        }

        let mut inner = self.inner.write().map_err(|_| {
            SkillRegistryError::Skill(SkillError::Io("registry lock poisoned".into()))
        })?;
        let versions = inner.by_id.entry(def.id.clone()).or_default();
        if versions.iter().any(|d| d.version == def.version) {
            return Err(SkillRegistryError::DuplicateVersion {
                skill: def.id,
                version: def.version,
            });
        }
        versions.push(def);
        // Keep versions sorted so `.last()` is the newest.
        versions.sort_by_key(|d| d.version);
        Ok(())
    }

    /// Get the latest version of a skill.
    pub fn latest(&self, id: &SkillId) -> Option<SkillDefinition> {
        let inner = self.inner.read().ok()?;
        inner.by_id.get(id)?.last().cloned()
    }

    /// Get a specific version.
    pub fn get(&self, id: &SkillId, version: SkillVersion) -> Option<SkillDefinition> {
        let inner = self.inner.read().ok()?;
        inner
            .by_id
            .get(id)?
            .iter()
            .find(|d| d.version == version)
            .cloned()
    }

    /// List all skill ids currently registered.
    pub fn list_ids(&self) -> Vec<SkillId> {
        self.inner
            .read()
            .map(|inner| inner.by_id.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// List all versions of a given skill.
    pub fn versions(&self, id: &SkillId) -> Vec<SkillDefinition> {
        self.inner
            .read()
            .map(|inner| inner.by_id.get(id).cloned().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Transition the latest version of a skill to a new status.
    ///
    /// Enforces the lifecycle DAG:
    /// - Exploring → {Exploited, Retired}
    /// - Exploited → {Frozen, Retired}
    /// - Frozen → {Exploited, Retired}
    /// - Retired → (terminal)
    pub fn set_status(
        &self,
        id: &SkillId,
        new_status: SkillStatus,
    ) -> Result<SkillStatus, SkillRegistryError> {
        if self.is_frozen() {
            return Err(SkillRegistryError::Frozen);
        }
        let mut inner = self.inner.write().map_err(|_| {
            SkillRegistryError::Skill(SkillError::Io("registry lock poisoned".into()))
        })?;
        let versions = inner
            .by_id
            .get_mut(id)
            .ok_or_else(|| SkillRegistryError::NotFound(id.clone()))?;
        let latest = versions
            .last_mut()
            .ok_or_else(|| SkillRegistryError::NotFound(id.clone()))?;
        let from = latest.status;
        let allowed = match (from, new_status) {
            (SkillStatus::Exploring, SkillStatus::Exploited) => true,
            (SkillStatus::Exploring, SkillStatus::Retired) => true,
            (SkillStatus::Exploited, SkillStatus::Frozen) => true,
            (SkillStatus::Exploited, SkillStatus::Retired) => true,
            (SkillStatus::Frozen, SkillStatus::Exploited) => true,
            (SkillStatus::Frozen, SkillStatus::Retired) => true,
            (a, b) if a == b => true,
            _ => false,
        };
        if !allowed {
            return Err(SkillRegistryError::IllegalTransition {
                skill: id.clone(),
                from,
                to: new_status,
            });
        }
        latest.status = new_status;
        Ok(from)
    }

    /// Count of all (id, version) pairs.
    pub fn total_versions(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.by_id.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_forge::capability::CapabilitySet;
    use crate::skill_forge::declarative::{SkillStep, StepKind};

    fn mk_def(id: &str, version: u32, status: SkillStatus) -> SkillDefinition {
        let mut def = SkillDefinition {
            id: SkillId::new(id),
            version: SkillVersion(version),
            name: id.to_string(),
            description: "test skill".into(),
            mode: SkillMode::Declarative(vec![SkillStep {
                kind: StepKind::Plan {
                    prompt: "do the thing".into(),
                },
                bind: None,
            }]),
            capabilities: CapabilitySet::empty(),
            content_hash_hex: String::new(),
            status,
            tenant: "default".into(),
            shared_cross_tenant: false,
        };
        def.content_hash_hex = def.compute_content_hash();
        def
    }

    #[test]
    fn skill_id_roundtrip() {
        let id = SkillId::new("my-skill");
        assert_eq!(id.as_str(), "my-skill");
        assert_eq!(id.to_string(), "my-skill");
    }

    #[test]
    fn version_next_monotonic() {
        let v = SkillVersion::INITIAL;
        assert_eq!(v.0, 1);
        assert_eq!(v.next().0, 2);
    }

    #[test]
    fn status_default_exploring() {
        assert_eq!(SkillStatus::default(), SkillStatus::Exploring);
    }

    #[test]
    fn insert_and_latest_roundtrip() {
        let reg = SkillRegistry::new();
        let d = mk_def("s1", 1, SkillStatus::Exploring);
        reg.insert(d.clone()).expect("insert");
        let got = reg.latest(&d.id).expect("latest");
        assert_eq!(got.version, SkillVersion(1));
    }

    #[test]
    fn insert_duplicate_version_fails() {
        let reg = SkillRegistry::new();
        let d = mk_def("s1", 1, SkillStatus::Exploring);
        reg.insert(d.clone()).expect("insert 1");
        let err = reg.insert(d.clone()).unwrap_err();
        match err {
            SkillRegistryError::DuplicateVersion { .. } => {}
            other => panic!("expected DuplicateVersion, got {other:?}"),
        }
    }

    #[test]
    fn insert_multiple_versions_latest_is_newest() {
        let reg = SkillRegistry::new();
        reg.insert(mk_def("s1", 1, SkillStatus::Exploring)).unwrap();
        reg.insert(mk_def("s1", 2, SkillStatus::Exploring)).unwrap();
        reg.insert(mk_def("s1", 3, SkillStatus::Exploring)).unwrap();
        assert_eq!(
            reg.latest(&SkillId::new("s1")).unwrap().version,
            SkillVersion(3)
        );
    }

    #[test]
    fn insert_content_hash_mismatch_fails() {
        let reg = SkillRegistry::new();
        let mut d = mk_def("s1", 1, SkillStatus::Exploring);
        d.content_hash_hex = "tampered".into();
        match reg.insert(d) {
            Err(SkillRegistryError::Skill(SkillError::IntegrityMismatch { .. })) => {}
            other => panic!("expected IntegrityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn status_transition_exploring_to_exploited() {
        let reg = SkillRegistry::new();
        reg.insert(mk_def("s1", 1, SkillStatus::Exploring)).unwrap();
        let prev = reg
            .set_status(&SkillId::new("s1"), SkillStatus::Exploited)
            .unwrap();
        assert_eq!(prev, SkillStatus::Exploring);
        assert_eq!(
            reg.latest(&SkillId::new("s1")).unwrap().status,
            SkillStatus::Exploited
        );
    }

    #[test]
    fn status_transition_retired_is_terminal() {
        let reg = SkillRegistry::new();
        reg.insert(mk_def("s1", 1, SkillStatus::Exploring)).unwrap();
        reg.set_status(&SkillId::new("s1"), SkillStatus::Retired)
            .unwrap();
        let err = reg
            .set_status(&SkillId::new("s1"), SkillStatus::Exploited)
            .unwrap_err();
        match err {
            SkillRegistryError::IllegalTransition { .. } => {}
            other => panic!("expected IllegalTransition, got {other:?}"),
        }
    }

    #[test]
    fn list_ids_returns_all() {
        let reg = SkillRegistry::new();
        reg.insert(mk_def("a", 1, SkillStatus::Exploring)).unwrap();
        reg.insert(mk_def("b", 1, SkillStatus::Exploring)).unwrap();
        let ids = reg.list_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn content_hash_changes_with_fields() {
        let d1 = mk_def("s1", 1, SkillStatus::Exploring);
        let mut d2 = d1.clone();
        d2.description = "changed".into();
        d2.content_hash_hex = d2.compute_content_hash();
        assert_ne!(d1.content_hash_hex, d2.content_hash_hex);
    }

    #[test]
    fn content_hash_stable_for_status_changes() {
        // status is NOT part of the hash
        let mut d1 = mk_def("s1", 1, SkillStatus::Exploring);
        let h1 = d1.compute_content_hash();
        d1.status = SkillStatus::Exploited;
        let h2 = d1.compute_content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn skill_error_display_messages() {
        let e = SkillError::NotFound(SkillId::new("x"));
        assert!(format!("{e}").contains("not found"));
        let e = SkillError::CapabilityDenied {
            skill: SkillId::new("x"),
            capability: "net".into(),
        };
        assert!(format!("{e}").contains("capability"));
    }

    #[test]
    fn frozen_registry_rejects_insert_and_status_change() {
        let reg = SkillRegistry::new();
        reg.insert(mk_def("s1", 1, SkillStatus::Exploring)).unwrap();
        reg.set_frozen(true);
        assert!(reg.is_frozen());

        // insert is rejected
        match reg.insert(mk_def("s2", 1, SkillStatus::Exploring)) {
            Err(SkillRegistryError::Frozen) => {}
            other => panic!("expected Frozen on insert, got {other:?}"),
        }

        // status change is rejected
        match reg.set_status(&SkillId::new("s1"), SkillStatus::Exploited) {
            Err(SkillRegistryError::Frozen) => {}
            other => panic!("expected Frozen on set_status, got {other:?}"),
        }

        // unfreeze restores mutation
        reg.set_frozen(false);
        assert!(!reg.is_frozen());
        reg.insert(mk_def("s2", 1, SkillStatus::Exploring)).unwrap();
        reg.set_status(&SkillId::new("s1"), SkillStatus::Exploited)
            .unwrap();
    }
}
