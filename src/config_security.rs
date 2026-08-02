//! Configuration Security — locking, integrity verification, and encryption support.
//!
//! Provides:
//! - `ConfigLock`: prevents runtime modification of config sections
//! - `IntegrityChecker`: content-digest tamper detection for persisted files
//!   (SHA-256 with the `security` feature; see the type's docs for what that
//!   does and does not defend against — it is not an HMAC)
//! - `SecurityAlertManager`: proactive alerts for key rotation, permissions, etc.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

// ============================================================================
// Config Locking
// ============================================================================

/// Sections of configuration that can be independently locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConfigSection {
    /// LLM provider settings and API keys.
    Providers,
    /// Security settings (guardrails, injection detection, PII).
    Security,
    /// RAG tier configuration.
    RagTiers,
    /// Agent autonomy levels and policies.
    AgentPolicy,
    /// Network egress rules.
    NetworkPolicy,
    /// Cost/budget limits.
    BudgetLimits,
    /// Rollback strategy settings.
    RollbackStrategy,
    /// Learning subsystem settings (bandit, procedures, etc.).
    Learning,
    /// Home automation configuration (backends, credentials, device access).
    HomeAutomation,
}

impl std::fmt::Display for ConfigSection {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Providers => write!(f, "providers"),
            Self::Security => write!(f, "security"),
            Self::RagTiers => write!(f, "rag_tiers"),
            Self::AgentPolicy => write!(f, "agent_policy"),
            Self::NetworkPolicy => write!(f, "network_policy"),
            Self::BudgetLimits => write!(f, "budget_limits"),
            Self::RollbackStrategy => write!(f, "rollback_strategy"),
            Self::Learning => write!(f, "learning"),
            _ => write!(f, "unknown"),
        }
    }
}

/// What is required to unlock a locked section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UnlockRequirement {
    /// Requires the master passphrase/key.
    MasterKey,
    /// Requires user confirmation in GUI.
    UserConfirmation,
    /// Requires both master key and user confirmation.
    Both,
    /// Cannot be unlocked at runtime — edit config on disk with the key.
    Immutable,
}

impl Default for UnlockRequirement {
    fn default() -> Self {
        Self::MasterKey
    }
}

/// Configuration lock state — prevents runtime modification of config sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigLock {
    /// Which sections are locked.
    locked_sections: HashSet<ConfigSection>,
    /// Whether ALL sections are locked (overrides individual).
    fully_locked: bool,
    /// Hash of config at lock time for tamper detection.
    locked_hash: Option<String>,
    /// What's required to unlock.
    unlock_requires: UnlockRequirement,
    /// Log of lock/unlock attempts.
    audit_log: Vec<LockAuditEntry>,
}

/// Audit entry for lock/unlock attempts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockAuditEntry {
    pub timestamp: u64,
    pub action: String,
    pub section: Option<ConfigSection>,
    pub success: bool,
    pub source: String,
}

impl ConfigLock {
    /// Create a new unlocked config.
    pub fn new() -> Self {
        Self {
            locked_sections: HashSet::new(),
            fully_locked: false,
            locked_hash: None,
            unlock_requires: UnlockRequirement::default(),
            audit_log: Vec::new(),
        }
    }

    /// Lock a specific section.
    pub fn lock_section(&mut self, section: ConfigSection) {
        self.locked_sections.insert(section);
        self.log("lock_section", Some(section), true, "api");
    }

    /// Lock ALL sections.
    pub fn lock_all(&mut self) {
        self.fully_locked = true;
        self.log("lock_all", None, true, "api");
    }

    /// Check if a section is locked.
    pub fn is_locked(&self, section: ConfigSection) -> bool {
        self.fully_locked || self.locked_sections.contains(&section)
    }

    /// Check if ANY section is locked.
    pub fn any_locked(&self) -> bool {
        self.fully_locked || !self.locked_sections.is_empty()
    }

    /// Attempt to unlock a section. Returns true if successful.
    pub fn unlock_section(&mut self, section: ConfigSection, source: &str) -> bool {
        if self.fully_locked {
            self.log(
                "unlock_section_denied_fully_locked",
                Some(section),
                false,
                source,
            );
            return false;
        }
        let removed = self.locked_sections.remove(&section);
        self.log("unlock_section", Some(section), removed, source);
        removed
    }

    /// Attempt to unlock all. Returns true if successful.
    pub fn unlock_all(&mut self, source: &str) -> bool {
        self.fully_locked = false;
        self.locked_sections.clear();
        self.log("unlock_all", None, true, source);
        true
    }

    /// Get the unlock requirement.
    pub fn unlock_requirement(&self) -> &UnlockRequirement {
        &self.unlock_requires
    }

    /// Set the unlock requirement.
    pub fn set_unlock_requirement(&mut self, req: UnlockRequirement) {
        self.unlock_requires = req;
    }

    /// Get all locked sections.
    pub fn locked_sections(&self) -> Vec<ConfigSection> {
        if self.fully_locked {
            vec![
                ConfigSection::Providers,
                ConfigSection::Security,
                ConfigSection::RagTiers,
                ConfigSection::AgentPolicy,
                ConfigSection::NetworkPolicy,
                ConfigSection::BudgetLimits,
                ConfigSection::RollbackStrategy,
                ConfigSection::Learning,
            ]
        } else {
            self.locked_sections.iter().copied().collect()
        }
    }

    /// Get audit log.
    pub fn audit_log(&self) -> &[LockAuditEntry] {
        &self.audit_log
    }

    fn log(&mut self, action: &str, section: Option<ConfigSection>, success: bool, source: &str) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.audit_log.push(LockAuditEntry {
            timestamp: ts,
            action: action.to_string(),
            section,
            success,
            source: source.to_string(),
        });
        // Cap audit log at 1000 entries
        if self.audit_log.len() > 1000 {
            self.audit_log.drain(..100);
        }
    }
}

impl Default for ConfigLock {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Integrity Checker
// ============================================================================

/// File integrity verification by content digest.
///
/// Stores a digest of persisted files and verifies it on load, catching
/// modified config files, poisoned learning state and the like.
///
/// # What this does and does not defend against
///
/// This doc used to claim **HMAC-SHA256**, and the digest function was named
/// `sha256_hex`. It was neither: the body was FNV-1a, a non-cryptographic hash
/// (V265 corrected both).
///
/// * **With the `security` feature** (on in `full`, hence in the default build)
///   the digest is real **SHA-256**.
/// * **Without it**, the digest falls back to FNV-1a, which detects *accidental*
///   corruption only. It is trivially forgeable, so an attacker who can write
///   the file can also make the digest match.
///
/// In neither case is this an HMAC: there is no secret key, so a digest stored
/// beside the file it protects can be recomputed by anyone who can rewrite both.
/// It raises the bar against tampering; it does not close the door.
pub struct IntegrityChecker {
    /// Known checksums: filename → hex SHA256.
    checksums: std::collections::HashMap<String, String>,
    /// Path to the checksums file.
    checksums_path: PathBuf,
}

impl IntegrityChecker {
    /// Create a new integrity checker with the given data directory.
    pub fn new(data_dir: &Path) -> Self {
        let checksums_path = data_dir.join("integrity.json");
        let mut checker = Self {
            checksums: std::collections::HashMap::new(),
            checksums_path,
        };
        checker.load_checksums();
        checker
    }

    /// Compute SHA-256 checksum of a file.
    pub fn compute_checksum(path: &Path) -> Result<String, String> {
        let data = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        Ok(Self::content_digest_hex(&data))
    }

    /// Record the current checksum of a file.
    pub fn record(&mut self, name: &str, path: &Path) -> Result<(), String> {
        let checksum = Self::compute_checksum(path)?;
        self.checksums.insert(name.to_string(), checksum);
        self.save_checksums()
    }

    /// Verify a file against its recorded checksum.
    pub fn verify(&self, name: &str, path: &Path) -> IntegrityResult {
        let recorded = match self.checksums.get(name) {
            Some(c) => c,
            None => return IntegrityResult::NoChecksum,
        };

        let current = match Self::compute_checksum(path) {
            Ok(c) => c,
            Err(_) => return IntegrityResult::FileNotFound,
        };

        if current == *recorded {
            IntegrityResult::Ok
        } else {
            IntegrityResult::Tampered {
                expected: recorded.clone(),
                actual: current,
            }
        }
    }

    /// Verify ALL recorded files. Returns list of issues.
    pub fn verify_all(&self, data_dir: &Path) -> Vec<(String, IntegrityResult)> {
        let mut results = Vec::new();
        for (name, _) in &self.checksums {
            let path = data_dir.join(format!("{}.json", name));
            let result = self.verify(name, &path);
            if !matches!(result, IntegrityResult::Ok) {
                results.push((name.clone(), result));
            }
        }
        results
    }

    /// Content digest, hex-encoded. **SHA-256** when the `security` feature is
    /// on (it is, in `full`); a non-cryptographic fallback otherwise.
    ///
    /// Renamed from `sha256_hex` in V265: the old name and its doc comment both
    /// claimed SHA-256 while the body was FNV-1a. A function whose name asserts
    /// a cryptographic guarantee it does not provide will be trusted by whoever
    /// reads the call site — which is the whole failure mode. The TODO it
    /// carried ("replace when ring/sha2 is added") had also gone stale: `sha2`
    /// is a dependency.
    fn content_digest_hex(data: &[u8]) -> String {
        #[cfg(feature = "security")]
        {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(data);
            return format!("{:x}", hasher.finalize());
        }
        #[cfg(not(feature = "security"))]
        {
            // FNV-1a 128-bit. Detects accidental corruption; forgeable on
            // purpose, which the type's docs state plainly.
            let mut h: u128 = 0xcbf29ce484222325;
            for &byte in data {
                h ^= byte as u128;
                h = h.wrapping_mul(0x100000001b3);
            }
            let h2 = h.wrapping_mul(0x517cc1b727220a95);
            format!("{:032x}{:032x}", h, h2)
        }
    }

    fn save_checksums(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.checksums)
            .map_err(|e| format!("Serialize: {}", e))?;
        let tmp = self.checksums_path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write: {}", e))?;
        std::fs::rename(&tmp, &self.checksums_path).map_err(|e| format!("Rename: {}", e))?;
        Ok(())
    }

    fn load_checksums(&mut self) {
        if let Ok(data) = std::fs::read_to_string(&self.checksums_path) {
            if let Ok(map) = serde_json::from_str(&data) {
                self.checksums = map;
            }
        }
    }
}

/// Result of an integrity check.
#[derive(Debug, Clone)]
pub enum IntegrityResult {
    /// File matches its recorded checksum.
    Ok,
    /// No checksum was recorded for this file.
    NoChecksum,
    /// File not found on disk.
    FileNotFound,
    /// File has been modified since last recorded checksum.
    Tampered { expected: String, actual: String },
}

// ============================================================================
// Security Alerts
// ============================================================================

/// Types of proactive security alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SecurityAlert {
    /// Key should be rotated (age exceeds recommended maximum).
    KeyRotationDue { days_since_rotation: u64 },
    /// No backup key exists.
    NoBackupKey,
    /// File has insecure permissions.
    InsecureFilePermissions { path: String, current: String },
    /// Configuration has been tampered with.
    ConfigTamperDetected { file: String },
    /// Sensitive data is unencrypted.
    UnencryptedSensitiveData { description: String },
    /// No encryption configured at all.
    NoEncryptionConfigured,
    /// Lock bypass attempt detected.
    LockBypassAttempt { source: String, section: String },
    /// Learning subsystem anomaly.
    LearningAnomaly {
        subsystem: String,
        description: String,
    },
}

impl std::fmt::Display for SecurityAlert {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyRotationDue {
                days_since_rotation,
            } => {
                write!(
                    f,
                    "Key rotation due ({} days since last rotation)",
                    days_since_rotation
                )
            }
            Self::NoBackupKey => write!(f, "No backup key exists — risk of data loss"),
            Self::InsecureFilePermissions { path, current } => {
                write!(f, "Insecure permissions on {}: {}", path, current)
            }
            Self::ConfigTamperDetected { file } => {
                write!(f, "Configuration tamper detected: {}", file)
            }
            Self::UnencryptedSensitiveData { description } => {
                write!(f, "Unencrypted sensitive data: {}", description)
            }
            Self::NoEncryptionConfigured => write!(f, "No encryption configured"),
            Self::LockBypassAttempt { source, section } => {
                write!(
                    f,
                    "Lock bypass attempt from {} on section {}",
                    source, section
                )
            }
            Self::LearningAnomaly {
                subsystem,
                description,
            } => {
                write!(f, "Learning anomaly in {}: {}", subsystem, description)
            }
            _ => write!(f, "Security alert"),
        }
    }
}

/// Manages security alerts with deduplication and cooldown.
pub struct SecurityAlertManager {
    alerts: Vec<(SecurityAlert, u64)>, // (alert, timestamp)
    cooldown_secs: u64,
    max_alerts: usize,
}

impl SecurityAlertManager {
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            cooldown_secs: 300, // 5 min cooldown between same alert type
            max_alerts: 100,
        }
    }

    /// Emit an alert (respects cooldown to avoid spam).
    pub fn emit(&mut self, alert: SecurityAlert) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Check cooldown: don't emit same alert type within cooldown window
        let alert_str = format!("{}", alert);
        let recent_duplicate = self.alerts.iter().rev().any(|(a, ts)| {
            format!("{}", a) == alert_str && now.saturating_sub(*ts) < self.cooldown_secs
        });

        if !recent_duplicate {
            log::warn!("[security-alert] {}", alert);
            self.alerts.push((alert, now));
            if self.alerts.len() > self.max_alerts {
                self.alerts.drain(..10);
            }
        }
    }

    /// Get all active alerts.
    pub fn active_alerts(&self) -> &[(SecurityAlert, u64)] {
        &self.alerts
    }

    /// Clear all alerts.
    pub fn clear(&mut self) {
        self.alerts.clear();
    }

    /// Number of alerts.
    pub fn count(&self) -> usize {
        self.alerts.len()
    }
}

impl Default for SecurityAlertManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_lock_section() {
        let mut lock = ConfigLock::new();
        assert!(!lock.is_locked(ConfigSection::Security));

        lock.lock_section(ConfigSection::Security);
        assert!(lock.is_locked(ConfigSection::Security));
        assert!(!lock.is_locked(ConfigSection::Providers));
        assert!(lock.any_locked());
    }

    #[test]
    fn test_config_lock_all() {
        let mut lock = ConfigLock::new();
        lock.lock_all();
        assert!(lock.is_locked(ConfigSection::Security));
        assert!(lock.is_locked(ConfigSection::Providers));
        assert!(lock.is_locked(ConfigSection::Learning));
    }

    #[test]
    fn test_config_unlock_section() {
        let mut lock = ConfigLock::new();
        lock.lock_section(ConfigSection::Security);
        assert!(lock.is_locked(ConfigSection::Security));

        lock.unlock_section(ConfigSection::Security, "test");
        assert!(!lock.is_locked(ConfigSection::Security));
    }

    #[test]
    fn test_config_unlock_denied_when_fully_locked() {
        let mut lock = ConfigLock::new();
        lock.lock_all();

        let result = lock.unlock_section(ConfigSection::Security, "test");
        assert!(!result); // Can't unlock individual when fully locked
        assert!(lock.is_locked(ConfigSection::Security));
    }

    #[test]
    fn test_config_lock_audit_log() {
        let mut lock = ConfigLock::new();
        lock.lock_section(ConfigSection::Security);
        lock.unlock_section(ConfigSection::Security, "admin");

        assert_eq!(lock.audit_log().len(), 2);
        assert_eq!(lock.audit_log()[0].action, "lock_section");
        assert_eq!(lock.audit_log()[1].action, "unlock_section");
    }

    #[test]
    fn test_integrity_checker_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ai_test_integrity_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&dir);

        let test_file = dir.join("test_config.json");
        std::fs::write(&test_file, r#"{"key": "value"}"#).unwrap();

        let mut checker = IntegrityChecker::new(&dir);
        checker.record("test_config", &test_file).unwrap();

        // Verify OK
        assert!(matches!(
            checker.verify("test_config", &test_file),
            IntegrityResult::Ok
        ));

        // Tamper the file
        std::fs::write(&test_file, r#"{"key": "TAMPERED"}"#).unwrap();

        // Verify TAMPERED
        assert!(matches!(
            checker.verify("test_config", &test_file),
            IntegrityResult::Tampered { .. }
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_integrity_no_checksum() {
        let dir = std::env::temp_dir().join(format!("ai_test_integrity2_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&dir);

        let checker = IntegrityChecker::new(&dir);
        let result = checker.verify("nonexistent", Path::new("/tmp/nope"));
        assert!(matches!(result, IntegrityResult::NoChecksum));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_security_alert_manager() {
        let mut mgr = SecurityAlertManager::new();
        mgr.emit(SecurityAlert::NoEncryptionConfigured);
        assert_eq!(mgr.count(), 1);

        // Same alert within cooldown — should be deduplicated
        mgr.emit(SecurityAlert::NoEncryptionConfigured);
        assert_eq!(mgr.count(), 1);

        // Different alert — should be added
        mgr.emit(SecurityAlert::NoBackupKey);
        assert_eq!(mgr.count(), 2);
    }

    #[test]
    fn test_security_alert_display() {
        let alert = SecurityAlert::KeyRotationDue {
            days_since_rotation: 95,
        };
        let s = format!("{}", alert);
        assert!(s.contains("95 days"));
    }

    #[test]
    fn test_config_section_display() {
        assert_eq!(ConfigSection::Security.to_string(), "security");
        assert_eq!(ConfigSection::Learning.to_string(), "learning");
    }
}
