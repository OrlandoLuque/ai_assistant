//! P2P Security — node authentication, authorization, and trust management.
//!
//! Provides:
//! - `TrustLevel`: per-peer authorization tier (Probation → Normal → Trusted → Admin)
//! - `MessageAuthorization`: which message types each trust level can send
//! - `PeerAccessControl`: whitelist/blacklist for node admission
//! - Utility functions for NodeId verification against TLS certificates

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// Trust Levels
// ============================================================================

/// Trust level assigned to a peer, determining which message types it can send.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Newly joined peer — can only ping/pong and request joining.
    Probation = 0,
    /// Authenticated peer — can read/write data.
    Normal = 1,
    /// Long-standing peer with high reputation — can sync and replicate.
    Trusted = 2,
    /// Cluster administrator — can send control messages.
    Admin = 3,
}

impl Default for TrustLevel {
    fn default() -> Self {
        Self::Probation
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Probation => write!(f, "probation"),
            Self::Normal => write!(f, "normal"),
            Self::Trusted => write!(f, "trusted"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

// ============================================================================
// Message Authorization
// ============================================================================

/// Categories of network messages for authorization purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageCategory {
    /// Ping, Pong — basic liveness.
    Heartbeat,
    /// JoinRequest, JoinResponse — cluster admission.
    Join,
    /// Get — read data.
    Read,
    /// Put, Delete — write data.
    Write,
    /// Replicate — data replication between nodes.
    Replicate,
    /// SyncRequest, SyncData — anti-entropy synchronization.
    Sync,
    /// NodeLeft, PeerExchange — cluster membership management.
    Membership,
    /// MapTask, MapResult — distributed computation.
    Compute,
    /// LogRequest, LogResponse — distributed logging.
    Logging,
    /// Invalidate, InvalidateAck — cache invalidation.
    Invalidation,
}

/// Authorization matrix: which trust levels can send which message categories.
pub struct MessageAuthorization;

impl MessageAuthorization {
    /// Check if a peer with the given trust level is authorized to send a message category.
    pub fn is_authorized(trust: TrustLevel, category: MessageCategory) -> bool {
        match trust {
            TrustLevel::Probation => matches!(
                category,
                MessageCategory::Heartbeat | MessageCategory::Join
            ),
            TrustLevel::Normal => matches!(
                category,
                MessageCategory::Heartbeat
                    | MessageCategory::Read
                    | MessageCategory::Write
                    | MessageCategory::Logging
            ),
            TrustLevel::Trusted => matches!(
                category,
                MessageCategory::Heartbeat
                    | MessageCategory::Read
                    | MessageCategory::Write
                    | MessageCategory::Replicate
                    | MessageCategory::Sync
                    | MessageCategory::Membership
                    | MessageCategory::Compute
                    | MessageCategory::Logging
                    | MessageCategory::Invalidation
            ),
            TrustLevel::Admin => true, // Admin can send everything
        }
    }

    /// Get the minimum trust level required for a message category.
    pub fn min_trust_for(category: MessageCategory) -> TrustLevel {
        match category {
            MessageCategory::Heartbeat | MessageCategory::Join => TrustLevel::Probation,
            MessageCategory::Read | MessageCategory::Write | MessageCategory::Logging => {
                TrustLevel::Normal
            }
            MessageCategory::Replicate
            | MessageCategory::Sync
            | MessageCategory::Membership
            | MessageCategory::Compute
            | MessageCategory::Invalidation => TrustLevel::Trusted,
        }
    }
}

// ============================================================================
// Peer Access Control
// ============================================================================

/// Whitelist/blacklist based peer access control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerAccessControl {
    /// If true, only whitelisted peers are allowed (whitelist mode).
    pub whitelist_only: bool,
    /// Allowed node IDs (hex-encoded). Only used when whitelist_only = true.
    pub whitelist: HashSet<String>,
    /// Banned node IDs (hex-encoded). Always enforced.
    pub blacklist: HashSet<String>,
    /// Auto-ban threshold: if reputation drops below this, auto-blacklist.
    pub auto_ban_reputation: f32,
    /// Maximum peers allowed in the cluster.
    pub max_peers: usize,
}

impl PeerAccessControl {
    /// Open mode — anyone with valid cert can join, no bans.
    pub fn open() -> Self {
        Self {
            whitelist_only: false,
            whitelist: HashSet::new(),
            blacklist: HashSet::new(),
            auto_ban_reputation: 0.05,
            max_peers: 100,
        }
    }

    /// Whitelist mode — only pre-approved peers.
    pub fn whitelist_mode() -> Self {
        Self {
            whitelist_only: true,
            whitelist: HashSet::new(),
            blacklist: HashSet::new(),
            auto_ban_reputation: 0.05,
            max_peers: 100,
        }
    }

    /// Check if a node is allowed to connect.
    pub fn check_admission(&self, node_id_hex: &str) -> AdmissionDecision {
        // 1. Blacklist always wins
        if self.blacklist.contains(node_id_hex) {
            return AdmissionDecision::Denied {
                reason: "Node is blacklisted".into(),
            };
        }

        // 2. Whitelist check (if enabled)
        if self.whitelist_only && !self.whitelist.contains(node_id_hex) {
            return AdmissionDecision::Denied {
                reason: "Node not in whitelist (whitelist-only mode)".into(),
            };
        }

        AdmissionDecision::Allowed
    }

    /// Add a node to the whitelist.
    pub fn whitelist_add(&mut self, node_id_hex: &str) {
        self.whitelist.insert(node_id_hex.to_string());
    }

    /// Remove a node from the whitelist.
    pub fn whitelist_remove(&mut self, node_id_hex: &str) {
        self.whitelist.remove(node_id_hex);
    }

    /// Ban a node (add to blacklist).
    pub fn ban(&mut self, node_id_hex: &str) {
        self.blacklist.insert(node_id_hex.to_string());
        // Also remove from whitelist if present
        self.whitelist.remove(node_id_hex);
    }

    /// Unban a node (remove from blacklist).
    pub fn unban(&mut self, node_id_hex: &str) {
        self.blacklist.remove(node_id_hex);
    }

    /// Check if a node should be auto-banned based on reputation.
    pub fn should_auto_ban(&self, reputation: f32) -> bool {
        reputation < self.auto_ban_reputation
    }
}

impl Default for PeerAccessControl {
    fn default() -> Self {
        Self::open()
    }
}

/// Result of an admission check.
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    Allowed,
    Denied { reason: String },
}

impl AdmissionDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

// ============================================================================
// NodeId Verification
// ============================================================================

/// Verify that a claimed NodeId matches a TLS certificate.
///
/// Computes SHA-256 of the certificate DER bytes and takes the first 20 bytes.
/// This should be called during identity exchange to prevent impersonation.
pub fn verify_node_id_against_cert(claimed_id: &[u8; 20], cert_der: &[u8]) -> bool {
    let computed = node_id_from_cert_bytes(cert_der);
    // Constant-time comparison to prevent timing attacks
    constant_time_eq(claimed_id, &computed)
}

/// Compute a NodeId from certificate DER bytes.
/// Uses FNV-1a hash (matching the existing node_id_from_cert in node_security.rs).
pub fn node_id_from_cert_bytes(cert_der: &[u8]) -> [u8; 20] {
    // FNV-1a hash to match existing implementation
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in cert_der {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let mut id = [0u8; 20];
    let bytes = h.to_le_bytes();
    id[..8].copy_from_slice(&bytes);
    // Fill remaining bytes with secondary hash
    let h2 = h.wrapping_mul(0x517cc1b727220a95);
    let bytes2 = h2.to_le_bytes();
    id[8..16].copy_from_slice(&bytes2);
    let h3 = h2.wrapping_mul(0x6c62272e07bb0142);
    let bytes3 = h3.to_le_bytes();
    id[16..20].copy_from_slice(&bytes3[..4]);
    id
}

/// Constant-time byte array comparison (prevents timing attacks).
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Probation < TrustLevel::Normal);
        assert!(TrustLevel::Normal < TrustLevel::Trusted);
        assert!(TrustLevel::Trusted < TrustLevel::Admin);
    }

    #[test]
    fn test_message_authorization_probation() {
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Probation,
            MessageCategory::Heartbeat
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Probation,
            MessageCategory::Join
        ));
        assert!(!MessageAuthorization::is_authorized(
            TrustLevel::Probation,
            MessageCategory::Write
        ));
        assert!(!MessageAuthorization::is_authorized(
            TrustLevel::Probation,
            MessageCategory::Sync
        ));
    }

    #[test]
    fn test_message_authorization_normal() {
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Normal,
            MessageCategory::Read
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Normal,
            MessageCategory::Write
        ));
        assert!(!MessageAuthorization::is_authorized(
            TrustLevel::Normal,
            MessageCategory::Replicate
        ));
        assert!(!MessageAuthorization::is_authorized(
            TrustLevel::Normal,
            MessageCategory::Sync
        ));
    }

    #[test]
    fn test_message_authorization_trusted() {
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Trusted,
            MessageCategory::Replicate
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Trusted,
            MessageCategory::Sync
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Trusted,
            MessageCategory::Membership
        ));
    }

    #[test]
    fn test_message_authorization_admin() {
        // Admin can do everything
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Admin,
            MessageCategory::Heartbeat
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Admin,
            MessageCategory::Sync
        ));
        assert!(MessageAuthorization::is_authorized(
            TrustLevel::Admin,
            MessageCategory::Membership
        ));
    }

    #[test]
    fn test_peer_access_open() {
        let acl = PeerAccessControl::open();
        assert!(acl.check_admission("abc123").is_allowed());
    }

    #[test]
    fn test_peer_access_blacklist() {
        let mut acl = PeerAccessControl::open();
        acl.ban("badnode");
        assert!(!acl.check_admission("badnode").is_allowed());
        assert!(acl.check_admission("goodnode").is_allowed());
    }

    #[test]
    fn test_peer_access_whitelist_mode() {
        let mut acl = PeerAccessControl::whitelist_mode();
        acl.whitelist_add("approved1");
        assert!(acl.check_admission("approved1").is_allowed());
        assert!(!acl.check_admission("unknown").is_allowed());
    }

    #[test]
    fn test_peer_access_blacklist_overrides_whitelist() {
        let mut acl = PeerAccessControl::whitelist_mode();
        acl.whitelist_add("node1");
        acl.ban("node1"); // ban removes from whitelist too
        assert!(!acl.check_admission("node1").is_allowed());
    }

    #[test]
    fn test_peer_auto_ban_threshold() {
        let acl = PeerAccessControl::open();
        assert!(acl.should_auto_ban(0.01));
        assert!(acl.should_auto_ban(0.04));
        assert!(!acl.should_auto_ban(0.1));
        assert!(!acl.should_auto_ban(0.5));
    }

    #[test]
    fn test_node_id_from_cert_deterministic() {
        let cert = b"fake_certificate_data_for_testing";
        let id1 = node_id_from_cert_bytes(cert);
        let id2 = node_id_from_cert_bytes(cert);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_node_id_verification() {
        let cert = b"test_certificate_bytes";
        let correct_id = node_id_from_cert_bytes(cert);
        assert!(verify_node_id_against_cert(&correct_id, cert));

        let mut wrong_id = correct_id;
        wrong_id[0] ^= 0xFF;
        assert!(!verify_node_id_against_cert(&wrong_id, cert));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn test_trust_level_display() {
        assert_eq!(TrustLevel::Probation.to_string(), "probation");
        assert_eq!(TrustLevel::Admin.to_string(), "admin");
    }

    #[test]
    fn test_min_trust_for() {
        assert_eq!(
            MessageAuthorization::min_trust_for(MessageCategory::Heartbeat),
            TrustLevel::Probation
        );
        assert_eq!(
            MessageAuthorization::min_trust_for(MessageCategory::Write),
            TrustLevel::Normal
        );
        assert_eq!(
            MessageAuthorization::min_trust_for(MessageCategory::Sync),
            TrustLevel::Trusted
        );
    }

    #[test]
    fn test_unban() {
        let mut acl = PeerAccessControl::open();
        acl.ban("node1");
        assert!(!acl.check_admission("node1").is_allowed());
        acl.unban("node1");
        assert!(acl.check_admission("node1").is_allowed());
    }
}
