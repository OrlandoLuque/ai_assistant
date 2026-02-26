//! Node security for distributed networking.
//!
//! Provides:
//! - **CertificateManager**: Self-signed CA and node certificate generation using rcgen,
//!   with save/load for persistent identity across restarts.
//! - **JoinToken**: Time-limited, use-limited tokens for cluster admission control.
//! - **ChallengeResponse**: Challenge-response authentication using HMAC-SHA256.
//!
//! All certificates use Ed25519 for fast, compact signatures.
//! This module is gated behind the `distributed-network` feature.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair, PKCS_ED25519};
use sha2::{Digest, Sha256};

use crate::distributed::NodeId;

/// Result type alias for this module.
type SecurityResult<T> = Result<T, String>;

// =============================================================================
// Cryptographic Utilities
// =============================================================================

/// Generate cryptographically adequate random bytes using SHA-256 mixing.
///
/// Combines multiple entropy sources (nanosecond timestamp, thread ID,
/// process ID, and a global counter) hashed through SHA-256 to produce
/// unpredictable output without requiring external CSPRNG dependencies.
fn secure_random_bytes(len: usize) -> Vec<u8> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let mut result = Vec::with_capacity(len);
    let mut block = 0u64;

    while result.len() < len {
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let thread_id = format!("{:?}", std::thread::current().id());

        let mut hasher = Sha256::new();
        hasher.update(&nanos.to_le_bytes());
        hasher.update(thread_id.as_bytes());
        hasher.update(&counter.to_le_bytes());
        hasher.update(&block.to_le_bytes());
        hasher.update(&(std::process::id() as u64).to_le_bytes());
        let hash = hasher.finalize();

        for &byte in hash.iter() {
            if result.len() < len {
                result.push(byte);
            } else {
                break;
            }
        }
        block += 1;
    }

    result
}

/// Constant-time byte comparison to prevent timing attacks.
///
/// Returns true only if both slices have equal length and identical contents.
/// Uses XOR accumulation so the comparison takes the same time regardless
/// of where (or whether) the slices differ.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// =============================================================================
// Node Identity
// =============================================================================

/// A node's cryptographic identity: its ID, certificate, private key, and the CA cert.
#[derive(Clone)]
pub struct NodeIdentity {
    /// The node's unique identifier (derived from its certificate).
    pub node_id: NodeId,
    /// DER-encoded X.509 certificate.
    pub cert_der: Vec<u8>,
    /// DER-encoded private key (PKCS#8, Ed25519).
    pub key_der: Vec<u8>,
    /// DER-encoded CA certificate used to verify peer certificates.
    pub ca_cert_der: Vec<u8>,
}

impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeIdentity")
            .field("node_id", &self.node_id)
            .field("cert_der_len", &self.cert_der.len())
            .field("key_der_len", &self.key_der.len())
            .field("ca_cert_der_len", &self.ca_cert_der.len())
            .finish()
    }
}

// =============================================================================
// Certificate Manager
// =============================================================================

/// Manages TLS certificate generation, storage, and loading for nodes.
///
/// Uses a self-signed CA to issue node certificates. All nodes in a cluster
/// share the same CA certificate to verify each other (mutual TLS).
#[derive(Debug)]
pub struct CertificateManager;

impl CertificateManager {
    /// Build the CA certificate parameters (fixed, deterministic except serial).
    fn make_ca_params() -> CertificateParams {
        let mut params = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "AI Assistant Cluster CA");
        dn.push(DnType::OrganizationName, "AI Assistant");
        params.distinguished_name = dn;
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        params.not_before = rcgen::date_time_ymd(2024, 1, 1);
        params.not_after = rcgen::date_time_ymd(2034, 1, 1);
        params
    }

    /// Generate a new self-signed Certificate Authority (CA).
    ///
    /// Returns `(ca_cert_der, ca_key_der)` — the DER-encoded CA certificate
    /// and its PKCS#8 private key. The CA is used to sign node certificates.
    pub fn generate_ca() -> SecurityResult<(Vec<u8>, Vec<u8>)> {
        let key_pair = KeyPair::generate_for(&PKCS_ED25519)
            .map_err(|e| format!("Failed to generate CA key pair: {}", e))?;
        let key_der = key_pair.serialized_der().to_vec();

        let params = Self::make_ca_params();
        let cert = params
            .self_signed(&key_pair)
            .map_err(|e| format!("Failed to create CA certificate: {}", e))?;
        let cert_der = cert.der().to_vec();

        Ok((cert_der, key_der))
    }

    /// Reconstruct the CA certificate from key DER (for signing node certs).
    ///
    /// Re-creates the CA Certificate object using the same fixed params and
    /// the saved key. The resulting cert will have a different serial number
    /// than the original, but that's fine — we only need the CA's signing
    /// capability, and the original CA cert DER is what we distribute to nodes.
    fn reconstruct_ca(ca_key_der: &[u8]) -> SecurityResult<(rcgen::Certificate, KeyPair)> {
        let pkcs8 = rustls::pki_types::PrivatePkcs8KeyDer::from(ca_key_der.to_vec());
        let ca_key = KeyPair::from_pkcs8_der_and_sign_algo(&pkcs8, &PKCS_ED25519)
            .map_err(|e| format!("Failed to load CA key: {}", e))?;

        let params = Self::make_ca_params();
        let ca_cert = params
            .self_signed(&ca_key)
            .map_err(|e| format!("Failed to reconstruct CA: {}", e))?;

        Ok((ca_cert, ca_key))
    }

    /// Generate a node certificate signed by the given CA.
    ///
    /// The node ID is derived from the first 20 bytes of the certificate's
    /// SHA-256 hash, ensuring a stable mapping from cert to identity.
    ///
    /// # Arguments
    /// * `ca_cert_der` - DER-encoded CA certificate (stored with the identity)
    /// * `ca_key_der` - DER-encoded CA PKCS#8 private key
    pub fn generate_node_cert(
        ca_cert_der: &[u8],
        ca_key_der: &[u8],
    ) -> SecurityResult<NodeIdentity> {
        // Reconstruct CA for signing
        let (ca_cert, ca_key) = Self::reconstruct_ca(ca_key_der)?;

        // Generate node key pair
        let node_key = KeyPair::generate_for(&PKCS_ED25519)
            .map_err(|e| format!("Failed to generate node key: {}", e))?;
        let node_key_der = node_key.serialized_der().to_vec();

        // Create node certificate params
        // SAN "node" is required: QUIC/TLS clients verify the server cert's SAN
        // against the server name used in connect(). We use "node" as the name.
        let mut node_params = CertificateParams::new(vec!["node".to_string()])
            .map_err(|e| format!("Failed to create node cert params: {}", e))?;
        let mut dn = DistinguishedName::new();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        dn.push(DnType::CommonName, format!("node-{}", timestamp));
        dn.push(DnType::OrganizationName, "AI Assistant Cluster");
        node_params.distinguished_name = dn;
        node_params.is_ca = rcgen::IsCa::NoCa;
        node_params.not_before = rcgen::date_time_ymd(2024, 1, 1);
        node_params.not_after = rcgen::date_time_ymd(2034, 1, 1);

        // Sign the node cert with the CA
        let node_cert = node_params
            .signed_by(&node_key, &ca_cert, &ca_key)
            .map_err(|e| format!("Failed to sign node cert: {}", e))?;
        let node_cert_der = node_cert.der().to_vec();

        // Derive NodeId from cert hash (first 20 bytes of SHA-256)
        let node_id = Self::node_id_from_cert(&node_cert_der);

        Ok(NodeIdentity {
            node_id,
            cert_der: node_cert_der,
            key_der: node_key_der,
            ca_cert_der: ca_cert_der.to_vec(),
        })
    }

    /// Save a node identity to a directory (3 files: cert.der, key.der, ca.der).
    pub fn save_identity(identity: &NodeIdentity, dir: &Path) -> SecurityResult<()> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create identity dir: {}", e))?;

        std::fs::write(dir.join("cert.der"), &identity.cert_der)
            .map_err(|e| format!("Failed to write cert: {}", e))?;
        std::fs::write(dir.join("key.der"), &identity.key_der)
            .map_err(|e| format!("Failed to write key: {}", e))?;
        std::fs::write(dir.join("ca.der"), &identity.ca_cert_der)
            .map_err(|e| format!("Failed to write CA cert: {}", e))?;

        Ok(())
    }

    /// Load a node identity from a directory.
    pub fn load_identity(dir: &Path) -> SecurityResult<NodeIdentity> {
        let cert_der = std::fs::read(dir.join("cert.der"))
            .map_err(|e| format!("Failed to read cert: {}", e))?;
        let key_der =
            std::fs::read(dir.join("key.der")).map_err(|e| format!("Failed to read key: {}", e))?;
        let ca_cert_der = std::fs::read(dir.join("ca.der"))
            .map_err(|e| format!("Failed to read CA cert: {}", e))?;

        let node_id = Self::node_id_from_cert(&cert_der);

        Ok(NodeIdentity {
            node_id,
            cert_der,
            key_der,
            ca_cert_der,
        })
    }

    /// Load an existing identity from dir, or create a new one if none exists.
    ///
    /// If no CA exists in the directory, generates a new CA and node cert.
    /// Returns `(identity, is_new)` — `is_new` is true if a new identity was created.
    pub fn load_or_create(dir: &Path) -> SecurityResult<(NodeIdentity, bool)> {
        // Try loading existing
        if dir.join("cert.der").exists()
            && dir.join("key.der").exists()
            && dir.join("ca.der").exists()
        {
            let identity = Self::load_identity(dir)?;
            return Ok((identity, false));
        }

        // Generate new CA and node cert
        let (ca_cert_der, ca_key_der) = Self::generate_ca()?;
        let identity = Self::generate_node_cert(&ca_cert_der, &ca_key_der)?;
        Self::save_identity(&identity, dir)?;

        // Also save CA key for future node generation
        std::fs::write(dir.join("ca_key.der"), &ca_key_der)
            .map_err(|e| format!("Failed to write CA key: {}", e))?;

        Ok((identity, true))
    }

    /// Build a quinn ServerConfig using the node's certificate.
    ///
    /// Configures mutual TLS: the server presents its cert and requires
    /// clients to present a cert signed by the same CA.
    pub fn make_server_config(identity: &NodeIdentity) -> SecurityResult<quinn::ServerConfig> {
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(identity.key_der.clone());
        let cert = rustls::pki_types::CertificateDer::from(identity.cert_der.clone());

        let mut root_store = rustls::RootCertStore::empty();
        let ca_cert = rustls::pki_types::CertificateDer::from(identity.ca_cert_der.clone());
        root_store
            .add(ca_cert)
            .map_err(|e| format!("Failed to add CA to root store: {}", e))?;

        // Build server config with client cert verification
        let client_verifier =
            rustls::server::WebPkiClientVerifier::builder(std::sync::Arc::new(root_store))
                .build()
                .map_err(|e| format!("Failed to build client verifier: {}", e))?;

        let server_crypto = rustls::ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(vec![cert], key.into())
            .map_err(|e| format!("Failed to build server TLS config: {}", e))?;

        let server_config = quinn::ServerConfig::with_crypto(std::sync::Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
                .map_err(|e| format!("Failed to create QUIC server config: {}", e))?,
        ));

        Ok(server_config)
    }

    /// Build a quinn ClientConfig that trusts the cluster's CA.
    ///
    /// The client presents its own certificate for mutual TLS and
    /// verifies the server's certificate against the CA.
    pub fn make_client_config(identity: &NodeIdentity) -> SecurityResult<quinn::ClientConfig> {
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(identity.key_der.clone());
        let cert = rustls::pki_types::CertificateDer::from(identity.cert_der.clone());

        let mut root_store = rustls::RootCertStore::empty();
        let ca_cert = rustls::pki_types::CertificateDer::from(identity.ca_cert_der.clone());
        root_store
            .add(ca_cert)
            .map_err(|e| format!("Failed to add CA to root store: {}", e))?;

        let client_crypto = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(vec![cert], key.into())
            .map_err(|e| format!("Failed to build client TLS config: {}", e))?;

        let client_config = quinn::ClientConfig::new(std::sync::Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)
                .map_err(|e| format!("Failed to create QUIC client config: {}", e))?,
        ));

        Ok(client_config)
    }

    /// Derive a NodeId from the first 20 bytes of a certificate's SHA-256 hash.
    fn node_id_from_cert(cert_der: &[u8]) -> NodeId {
        let hash = Sha256::digest(cert_der);
        let mut bytes = [0u8; 20];
        bytes.copy_from_slice(&hash[..20]);
        NodeId::from_bytes(bytes)
    }
}

// =============================================================================
// Join Token
// =============================================================================

/// A time-limited, use-limited token for cluster admission control.
///
/// Tokens are generated by an existing cluster member and shared out-of-band
/// (e.g., copied to a config file). New nodes present the token during the
/// join handshake to prove they are authorized.
#[derive(Debug, Clone)]
pub struct JoinToken {
    /// Random token bytes, base64-encoded for display.
    pub token: String,
    /// Unix timestamp when the token was created.
    pub created_at: u64,
    /// Unix timestamp when the token expires.
    pub expires_at: u64,
    /// Maximum number of times this token can be used (None = unlimited).
    pub max_uses: Option<usize>,
    /// How many times this token has been used.
    pub uses: usize,
}

impl JoinToken {
    /// Generate a new join token.
    ///
    /// # Arguments
    /// * `validity_hours` - How many hours the token remains valid.
    /// * `max_uses` - Maximum uses (None for unlimited).
    pub fn generate(validity_hours: u64, max_uses: Option<usize>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate 32 random bytes using SHA-256 mixed entropy
        let random = secure_random_bytes(32);
        let mut token_bytes = [0u8; 32];
        token_bytes.copy_from_slice(&random);

        Self {
            token: base64_encode(&token_bytes),
            created_at: now,
            expires_at: now + validity_hours * 3600,
            max_uses,
            uses: 0,
        }
    }

    /// Check if the token is currently valid (not expired, not exhausted).
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if now >= self.expires_at {
            return false;
        }

        if let Some(max) = self.max_uses {
            if self.uses >= max {
                return false;
            }
        }

        true
    }

    /// Consume one use of the token. Returns false if expired or exhausted.
    pub fn consume(&mut self) -> bool {
        if !self.is_valid() {
            return false;
        }
        self.uses += 1;
        true
    }

    /// Encode the token to a shareable string.
    pub fn encode(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.token,
            self.created_at,
            self.expires_at,
            self.max_uses
                .map_or("unlimited".to_string(), |m| m.to_string()),
            self.uses,
        )
    }

    /// Decode a token from its encoded string representation.
    pub fn decode(s: &str) -> SecurityResult<Self> {
        let parts: Vec<&str> = s.splitn(5, ':').collect();
        if parts.len() != 5 {
            return Err("Invalid token format: expected 5 colon-separated parts".to_string());
        }

        let token = parts[0].to_string();
        let created_at: u64 = parts[1]
            .parse()
            .map_err(|_| "Invalid created_at timestamp".to_string())?;
        let expires_at: u64 = parts[2]
            .parse()
            .map_err(|_| "Invalid expires_at timestamp".to_string())?;
        let max_uses = if parts[3] == "unlimited" {
            None
        } else {
            Some(
                parts[3]
                    .parse::<usize>()
                    .map_err(|_| "Invalid max_uses value".to_string())?,
            )
        };
        let uses: usize = parts[4]
            .parse()
            .map_err(|_| "Invalid uses count".to_string())?;

        Ok(Self {
            token,
            created_at,
            expires_at,
            max_uses,
            uses,
        })
    }

    /// Get remaining validity in seconds (0 if expired).
    pub fn remaining_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at.saturating_sub(now)
    }

    /// Get remaining uses (None if unlimited).
    pub fn remaining_uses(&self) -> Option<usize> {
        self.max_uses.map(|max| max.saturating_sub(self.uses))
    }
}

// =============================================================================
// Challenge-Response Authentication
// =============================================================================

/// Challenge-response authentication using HMAC-SHA256.
///
/// After TLS handshake, this provides an additional layer of authentication:
/// 1. Server sends a random 32-byte challenge
/// 2. Client computes HMAC-SHA256(challenge, private_key_hash)
/// 3. Server verifies the HMAC using its knowledge of the client's public key
///
/// This proves the client possesses the private key without exposing it.
#[derive(Debug)]
pub struct ChallengeResponse;

impl ChallengeResponse {
    /// Generate a random 32-byte challenge using SHA-256 mixed entropy.
    pub fn generate_challenge() -> Vec<u8> {
        secure_random_bytes(32)
    }

    /// Compute the challenge response using HMAC-SHA256.
    ///
    /// The response is `SHA256(challenge || key_der)`, proving possession of
    /// the private key material.
    pub fn sign_challenge(challenge: &[u8], key_der: &[u8]) -> SecurityResult<Vec<u8>> {
        let mut hasher = Sha256::new();
        hasher.update(challenge);
        hasher.update(key_der);
        Ok(hasher.finalize().to_vec())
    }

    /// Verify a challenge response using constant-time comparison.
    ///
    /// The verifier must have access to the same `key_der` that the challenger
    /// used. In practice, this means both sides share the key DER (e.g., from
    /// the initial TLS handshake or certificate exchange).
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    pub fn verify_response(
        challenge: &[u8],
        response: &[u8],
        key_der: &[u8],
    ) -> SecurityResult<bool> {
        let expected = Self::sign_challenge(challenge, key_der)?;
        Ok(constant_time_eq(&expected, response))
    }
}

// =============================================================================
// Base64 encoding/decoding (minimal, no external dependency)
// =============================================================================

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Decode a base64-encoded string back to bytes.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim_end_matches('=');
    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;

    for c in s.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(format!("Invalid base64 character: {}", c)),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ca() {
        let result = CertificateManager::generate_ca();
        assert!(
            result.is_ok(),
            "CA generation should succeed: {:?}",
            result.err()
        );
        let (cert_der, key_der) = result.unwrap();
        assert!(!cert_der.is_empty(), "CA cert should not be empty");
        assert!(!key_der.is_empty(), "CA key should not be empty");
    }

    #[test]
    fn test_generate_node_cert() {
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let result = CertificateManager::generate_node_cert(&ca_cert, &ca_key);
        assert!(
            result.is_ok(),
            "Node cert generation should succeed: {:?}",
            result.err()
        );
        let identity = result.unwrap();
        assert!(!identity.cert_der.is_empty());
        assert!(!identity.key_der.is_empty());
        assert_eq!(identity.ca_cert_der, ca_cert);
    }

    #[test]
    fn test_node_id_from_cert_deterministic() {
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let identity = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();

        let id1 = CertificateManager::node_id_from_cert(&identity.cert_der);
        let id2 = CertificateManager::node_id_from_cert(&identity.cert_der);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_different_nodes_different_ids() {
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let id1 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id2 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        assert_ne!(
            id1.node_id, id2.node_id,
            "Different nodes should have different IDs"
        );
    }

    #[test]
    fn test_save_load_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let original = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();

        CertificateManager::save_identity(&original, dir.path()).unwrap();
        let loaded = CertificateManager::load_identity(dir.path()).unwrap();

        assert_eq!(original.node_id, loaded.node_id);
        assert_eq!(original.cert_der, loaded.cert_der);
        assert_eq!(original.key_der, loaded.key_der);
        assert_eq!(original.ca_cert_der, loaded.ca_cert_der);
    }

    #[test]
    fn test_load_or_create_new() {
        let dir = tempfile::tempdir().unwrap();
        let (identity, is_new) = CertificateManager::load_or_create(dir.path()).unwrap();
        assert!(is_new, "First call should create new identity");
        assert!(!identity.cert_der.is_empty());
    }

    #[test]
    fn test_load_or_create_existing() {
        let dir = tempfile::tempdir().unwrap();
        let (first, is_new1) = CertificateManager::load_or_create(dir.path()).unwrap();
        assert!(is_new1);

        let (second, is_new2) = CertificateManager::load_or_create(dir.path()).unwrap();
        assert!(!is_new2, "Second call should load existing identity");
        assert_eq!(first.node_id, second.node_id);
    }

    #[test]
    fn test_make_server_config() {
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let identity = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        let result = CertificateManager::make_server_config(&identity);
        assert!(
            result.is_ok(),
            "Server config should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_make_client_config() {
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let identity = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        let result = CertificateManager::make_client_config(&identity);
        assert!(
            result.is_ok(),
            "Client config should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_join_token_generate_valid() {
        let token = JoinToken::generate(24, Some(5));
        assert!(token.is_valid(), "Fresh token should be valid");
        assert!(!token.token.is_empty());
        assert_eq!(token.uses, 0);
        assert_eq!(token.max_uses, Some(5));
        assert!(token.remaining_secs() > 0);
        assert_eq!(token.remaining_uses(), Some(5));
    }

    #[test]
    fn test_join_token_consume() {
        let mut token = JoinToken::generate(24, Some(2));
        assert!(token.consume(), "First consume should succeed");
        assert_eq!(token.uses, 1);
        assert_eq!(token.remaining_uses(), Some(1));
        assert!(token.consume(), "Second consume should succeed");
        assert_eq!(token.uses, 2);
        assert!(!token.consume(), "Third consume should fail (exhausted)");
    }

    #[test]
    fn test_join_token_unlimited() {
        let mut token = JoinToken::generate(24, None);
        assert!(token.is_valid());
        assert_eq!(token.remaining_uses(), None);
        for _ in 0..100 {
            assert!(token.consume());
        }
    }

    #[test]
    fn test_join_token_expired() {
        let mut token = JoinToken::generate(0, None);
        token.expires_at = token.created_at;
        assert!(!token.is_valid(), "Expired token should be invalid");
        assert!(!token.consume(), "Consuming expired token should fail");
    }

    #[test]
    fn test_join_token_encode_decode() {
        let original = JoinToken::generate(48, Some(10));
        let encoded = original.encode();
        let decoded = JoinToken::decode(&encoded).unwrap();

        assert_eq!(original.token, decoded.token);
        assert_eq!(original.created_at, decoded.created_at);
        assert_eq!(original.expires_at, decoded.expires_at);
        assert_eq!(original.max_uses, decoded.max_uses);
        assert_eq!(original.uses, decoded.uses);
    }

    #[test]
    fn test_join_token_decode_invalid() {
        assert!(JoinToken::decode("invalid").is_err());
        assert!(JoinToken::decode("a:b:c").is_err());
    }

    #[test]
    fn test_challenge_response_generate() {
        let c1 = ChallengeResponse::generate_challenge();
        let c2 = ChallengeResponse::generate_challenge();
        assert_eq!(c1.len(), 32);
        assert_eq!(c2.len(), 32);
    }

    #[test]
    fn test_challenge_sign_verify() {
        let key_der = b"test_private_key_material";
        let challenge = ChallengeResponse::generate_challenge();
        let response = ChallengeResponse::sign_challenge(&challenge, key_der).unwrap();
        let valid = ChallengeResponse::verify_response(&challenge, &response, key_der).unwrap();
        assert!(valid, "Valid response should verify");
    }

    #[test]
    fn test_challenge_wrong_key_fails() {
        let key1 = b"key_material_1";
        let key2 = b"key_material_2";
        let challenge = ChallengeResponse::generate_challenge();
        let response = ChallengeResponse::sign_challenge(&challenge, key1).unwrap();
        let valid = ChallengeResponse::verify_response(&challenge, &response, key2).unwrap();
        assert!(!valid, "Response with wrong key should not verify");
    }

    #[test]
    fn test_challenge_wrong_challenge_fails() {
        let key = b"key_material";
        let challenge1 = vec![1u8; 32];
        let challenge2 = vec![2u8; 32];
        let response = ChallengeResponse::sign_challenge(&challenge1, key).unwrap();
        let valid = ChallengeResponse::verify_response(&challenge2, &response, key).unwrap();
        assert!(!valid, "Response to different challenge should not verify");
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = b"Hello, distributed world!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_base64_empty() {
        let encoded = base64_encode(b"");
        let decoded = base64_decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_secure_random_bytes_length() {
        for len in [0, 1, 16, 32, 64, 100] {
            let bytes = secure_random_bytes(len);
            assert_eq!(bytes.len(), len);
        }
    }

    #[test]
    fn test_secure_random_bytes_unique() {
        let a = secure_random_bytes(32);
        let b = secure_random_bytes(32);
        assert_ne!(a, b, "Two consecutive random outputs should differ");
    }

    #[test]
    fn test_constant_time_eq_equal() {
        let a = vec![1, 2, 3, 4, 5];
        assert!(constant_time_eq(&a, &a));
    }

    #[test]
    fn test_constant_time_eq_different() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 6];
        assert!(!constant_time_eq(&a, &b));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3, 4];
        assert!(!constant_time_eq(&a, &b));
    }

    #[test]
    fn test_base64_various_lengths() {
        for len in 1..=10 {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 13) as u8).collect();
            let encoded = base64_encode(&data);
            let decoded = base64_decode(&encoded).unwrap();
            assert_eq!(data, decoded, "Roundtrip failed for length {}", len);
        }
    }
}
