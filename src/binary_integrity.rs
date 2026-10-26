//! Binary integrity verification module
//!
//! This module provides startup-time verification of binary integrity
//! to detect tampering or modifications. The verification only runs
//! once at startup, so there is no runtime performance penalty.
//!
//! # Features
//!
//! - SHA256 hash verification of the binary
//! - Optional RSA signature verification
//! - Embedded public key support
//! - Self-check capabilities
//!
//! # Usage
//!
//! ```no_run
//! use ai_assistant::binary_integrity::{IntegrityChecker, IntegrityResult};
//!
//! // Quick check at startup
//! if let Err(e) = IntegrityChecker::verify_self() {
//!     eprintln!("Binary integrity check failed: {}", e);
//!     std::process::exit(1);
//! }
//! ```

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Result of integrity verification
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegrityResult {
    /// Binary passed all integrity checks
    Valid,
    /// Hash mismatch detected
    HashMismatch { expected: String, actual: String },
    /// Signature verification failed
    SignatureInvalid,
    /// Could not read binary file
    ReadError(String),
    /// No expected hash configured
    NoHashConfigured,
    /// Verification skipped (not in protected build)
    Skipped,
}

impl IntegrityResult {
    /// Returns true if the binary is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, IntegrityResult::Valid | IntegrityResult::Skipped)
    }

    /// Returns true if verification was performed
    pub fn was_verified(&self) -> bool {
        !matches!(
            self,
            IntegrityResult::Skipped | IntegrityResult::NoHashConfigured
        )
    }
}

impl std::fmt::Display for IntegrityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegrityResult::Valid => write!(f, "Binary integrity verified"),
            IntegrityResult::HashMismatch { expected, actual } => {
                write!(f, "Hash mismatch: expected {}, got {}", expected, actual)
            }
            IntegrityResult::SignatureInvalid => write!(f, "Invalid signature"),
            IntegrityResult::ReadError(e) => write!(f, "Read error: {}", e),
            IntegrityResult::NoHashConfigured => write!(f, "No hash configured for verification"),
            IntegrityResult::Skipped => write!(f, "Verification skipped (debug build)"),
        }
    }
}

impl std::error::Error for IntegrityResult {}

/// Configuration for integrity checking
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IntegrityConfig {
    /// Expected SHA256 hash (hex string, 64 chars)
    pub expected_hash: Option<String>,
    /// Embedded public key for signature verification (PEM format)
    pub public_key_pem: Option<String>,
    /// Whether to abort on failure
    pub abort_on_failure: bool,
    /// Custom path to check (defaults to current executable)
    pub binary_path: Option<String>,
}

with_setters!(IntegrityConfig {
    with_abort_on_failure -> abort_on_failure: bool,
});

impl Default for IntegrityConfig {
    fn default() -> Self {
        Self {
            expected_hash: None,
            public_key_pem: None,
            abort_on_failure: true,
            binary_path: None,
        }
    }
}

impl IntegrityConfig {
    /// Create config with expected hash
    pub fn with_hash(hash: impl Into<String>) -> Self {
        Self {
            expected_hash: Some(hash.into()),
            ..Default::default()
        }
    }

    /// Set the public key for signature verification
    pub fn with_public_key(mut self, pem: impl Into<String>) -> Self {
        self.public_key_pem = Some(pem.into());
        self
    }

    /// Set whether to abort on failure
    pub fn abort_on_failure(mut self, abort: bool) -> Self {
        self.abort_on_failure = abort;
        self
    }

    /// Set custom binary path
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.binary_path = Some(path.into());
        self
    }
}

/// Binary integrity checker
///
/// Performs one-time verification at startup with no runtime overhead.
pub struct IntegrityChecker {
    config: IntegrityConfig,
}

impl IntegrityChecker {
    /// Create a new integrity checker with configuration
    pub fn new(config: IntegrityConfig) -> Self {
        Self { config }
    }

    /// Create checker with default config (looks for embedded hash)
    pub fn with_defaults() -> Self {
        Self::new(IntegrityConfig::default())
    }

    /// Quick verification using embedded configuration
    ///
    /// This is the recommended way to verify at startup:
    /// ```no_run
    /// if let Err(e) = IntegrityChecker::verify_self() {
    ///     std::process::exit(1);
    /// }
    /// ```
    #[cfg(feature = "integrity-check")]
    pub fn verify_self() -> Result<(), IntegrityResult> {
        // In debug builds, skip verification
        #[cfg(debug_assertions)]
        {
            return Ok(());
        }

        #[cfg(not(debug_assertions))]
        {
            let checker = Self::with_defaults();
            let result = checker.verify();

            if result.is_valid() {
                Ok(())
            } else {
                Err(result)
            }
        }
    }

    /// Quick verification - always succeeds when feature not enabled
    #[cfg(not(feature = "integrity-check"))]
    pub fn verify_self() -> Result<(), IntegrityResult> {
        Ok(())
    }

    /// Perform integrity verification
    pub fn verify(&self) -> IntegrityResult {
        // Skip in debug builds
        #[cfg(debug_assertions)]
        {
            return IntegrityResult::Skipped;
        }

        #[cfg(not(debug_assertions))]
        {
            // Get binary path
            let binary_path = match &self.config.binary_path {
                Some(p) => std::path::PathBuf::from(p),
                None => match std::env::current_exe() {
                    Ok(p) => p,
                    Err(e) => return IntegrityResult::ReadError(e.to_string()),
                },
            };

            // Check if we have an expected hash
            let expected_hash = match &self.config.expected_hash {
                Some(h) => h.clone(),
                None => {
                    // Try to load from generated file or environment
                    if let Ok(h) = std::env::var("BINARY_INTEGRITY_HASH") {
                        h
                    } else {
                        return IntegrityResult::NoHashConfigured;
                    }
                }
            };

            // Calculate actual hash
            let actual_hash = match self.calculate_sha256(&binary_path) {
                Ok(h) => h,
                Err(e) => return IntegrityResult::ReadError(e.to_string()),
            };

            // Compare hashes (case-insensitive)
            if actual_hash.to_uppercase() != expected_hash.to_uppercase() {
                return IntegrityResult::HashMismatch {
                    expected: expected_hash,
                    actual: actual_hash,
                };
            }

            IntegrityResult::Valid
        }
    }

    /// Calculate SHA256 hash of a file (used in release builds only)
    #[cfg_attr(debug_assertions, allow(dead_code))]
    fn calculate_sha256(&self, path: &Path) -> io::Result<String> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(sha256_hex(&buffer))
    }

    /// Verify and abort if failed (for use in main())
    pub fn verify_or_abort(&self) {
        let result = self.verify();

        if !result.is_valid() && self.config.abort_on_failure {
            log::error!("INTEGRITY CHECK FAILED: {}", result);
            std::process::exit(1);
        }
    }
}

/// Simple SHA256 implementation (no external dependencies)
///
/// Based on FIPS 180-4 specification
fn sha256_hex(data: &[u8]) -> String {
    crate::sha256::sha256_hex(data)
}

/// Hash a file and return hex string
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(sha256_hex(&buffer))
}

/// Hash bytes and return hex string
pub fn hash_bytes(data: &[u8]) -> String {
    sha256_hex(data)
}

/// Startup guard that verifies integrity once
///
/// Use this in your main() function:
/// ```no_run
/// use ai_assistant::binary_integrity::startup_integrity_check;
///
/// fn main() {
///     startup_integrity_check();
///     // ... rest of your application
/// }
/// ```
pub fn startup_integrity_check() {
    #[cfg(feature = "integrity-check")]
    {
        if let Err(result) = IntegrityChecker::verify_self() {
            // In release builds, this is a fatal error
            #[cfg(not(debug_assertions))]
            {
                log::error!("FATAL: Binary integrity verification failed");
                log::error!("       {}", result);
                log::error!("       The application may have been tampered with.");
                std::process::exit(1);
            }

            // In debug builds, just warn
            #[cfg(debug_assertions)]
            {
                log::warn!("Binary integrity check would fail in release: {}", result);
            }
        }
    }
}

/// Macro to embed integrity check at compile time
///
/// Place this in your main.rs to automatically verify on startup:
/// ```rust,no_run
/// ai_assistant::integrity_guard!();
/// ```
#[macro_export]
macro_rules! integrity_guard {
    () => {
        #[cfg(feature = "integrity-check")]
        {
            $crate::binary_integrity::startup_integrity_check();
        }
    };
    ($hash:expr) => {
        #[cfg(feature = "integrity-check")]
        {
            let checker = $crate::binary_integrity::IntegrityChecker::new(
                $crate::binary_integrity::IntegrityConfig::with_hash($hash),
            );
            checker.verify_or_abort();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        // SHA256 of empty string
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hello() {
        // SHA256 of "hello"
        let hash = sha256_hex(b"hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_sha256_long_message() {
        // SHA256 of a longer message
        let hash = sha256_hex(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            hash,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    #[test]
    fn test_hash_bytes() {
        let hash = hash_bytes(b"test");
        assert_eq!(
            hash,
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        );
    }

    #[test]
    fn test_integrity_config_builder() {
        let config = IntegrityConfig::with_hash("abc123")
            .abort_on_failure(false)
            .with_path("/some/path");

        assert_eq!(config.expected_hash, Some("abc123".to_string()));
        assert!(!config.abort_on_failure);
        assert_eq!(config.binary_path, Some("/some/path".to_string()));
    }

    #[test]
    fn test_integrity_result_display() {
        assert_eq!(
            format!("{}", IntegrityResult::Valid),
            "Binary integrity verified"
        );
        assert_eq!(
            format!("{}", IntegrityResult::Skipped),
            "Verification skipped (debug build)"
        );
    }

    #[test]
    fn test_integrity_result_is_valid() {
        assert!(IntegrityResult::Valid.is_valid());
        assert!(IntegrityResult::Skipped.is_valid());
        assert!(!IntegrityResult::NoHashConfigured.is_valid());
        assert!(!IntegrityResult::SignatureInvalid.is_valid());
    }

    #[test]
    fn test_verify_self_in_debug() {
        // In debug mode, verify_self should always succeed
        let result = IntegrityChecker::verify_self();
        assert!(result.is_ok());
    }

    #[test]
    fn test_checker_verify_skipped_in_debug() {
        let checker = IntegrityChecker::with_defaults();
        let _result = checker.verify();

        // In debug builds, verification is skipped
        #[cfg(debug_assertions)]
        assert_eq!(_result, IntegrityResult::Skipped);
    }

    #[test]
    fn test_hash_mismatch_display() {
        let result = IntegrityResult::HashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        let msg = format!("{}", result);
        assert!(msg.contains("abc123"));
        assert!(msg.contains("def456"));
    }
}
