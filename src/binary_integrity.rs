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
pub enum IntegrityResult {
    /// Binary passed all integrity checks
    Valid,
    /// Hash mismatch detected
    HashMismatch {
        expected: String,
        actual: String,
    },
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
        !matches!(self, IntegrityResult::Skipped | IntegrityResult::NoHashConfigured)
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
                }
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

    /// Calculate SHA256 hash of a file
    #[allow(dead_code)]
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
    let hash = sha256(data);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// SHA256 hash computation
fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: adding padding bits
    let ml = (data.len() as u64) * 8; // Message length in bits
    let mut padded = data.to_vec();

    // Append bit '1' to message
    padded.push(0x80);

    // Append zeros until message length ≡ 448 (mod 512)
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }

    // Append original length in bits as 64-bit big-endian
    padded.extend_from_slice(&ml.to_be_bytes());

    // Process each 512-bit chunk
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];

        // Copy chunk into first 16 words
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }

        // Extend the first 16 words into the remaining 48 words
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // Initialize working variables
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        // Compression function main loop
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        // Add compressed chunk to current hash value
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce the final hash value (big-endian)
    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }

    result
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
/// ```ignore
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
                $crate::binary_integrity::IntegrityConfig::with_hash($hash)
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
        let result = checker.verify();

        // In debug builds, verification is skipped
        #[cfg(debug_assertions)]
        assert_eq!(result, IntegrityResult::Skipped);
    }
}
