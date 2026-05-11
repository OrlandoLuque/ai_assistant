//! Distributed RAG — shared knowledge across cluster nodes via DHT.
//!
//! Documents can be scoped as Private (local only) or Shared (distributed
//! via DHT to all cluster nodes). Queries merge local + distributed results.

use serde::{Deserialize, Serialize};

/// Scope of a document in the RAG knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DocumentScope {
    /// Document stays on this node only. Never shared. Default.
    Private,
    /// Document chunks distributed via DHT to cluster nodes.
    Shared,
}

impl Default for DocumentScope {
    fn default() -> Self {
        Self::Private
    }
}

impl std::fmt::Display for DocumentScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            Self::Private => write!(f, "private"),
            Self::Shared => write!(f, "shared"),
            _ => write!(f, "unknown"),
        }
    }
}

/// Metadata for a shared chunk stored in the DHT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedChunkMeta {
    /// Hash of the chunk content (DHT key).
    pub chunk_hash: String,
    /// Original document source name.
    pub source: String,
    /// Document ID that owns this chunk.
    pub document_id: String,
    /// Node ID that indexed this chunk.
    pub owner_node: String,
    /// Chunk text content.
    pub content: String,
    /// Section/heading within the document.
    pub section: String,
    /// Relevance score from original indexing.
    pub base_score: f32,
    /// Token count.
    pub tokens: usize,
    /// TTL in seconds (0 = no expiry).
    pub ttl_secs: u64,
    /// Creation timestamp.
    pub created_at: u64,
}

impl SharedChunkMeta {
    /// Check if this chunk has expired.
    pub fn is_expired(&self) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now > self.created_at + self.ttl_secs
    }
}

/// Configuration for distributed RAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DistributedRagConfig {
    /// Whether distributed RAG is enabled.
    pub enabled: bool,
    /// Default scope for new documents.
    pub default_scope: DocumentScope,
    /// TTL for shared chunks in seconds (default: 24 hours).
    pub shared_chunk_ttl_secs: u64,
    /// How often to refresh TTL of owned shared chunks (seconds).
    pub refresh_interval_secs: u64,
    /// Maximum number of distributed chunks to include per query.
    pub max_distributed_chunks: usize,
    /// Whether to encrypt shared chunks in the DHT.
    pub encrypt_shared: bool,
    /// Timeout per distributed query in seconds. If a peer doesn't respond
    /// within this time, its results are skipped. Default: 5 seconds.
    pub query_timeout_secs: u64,
    /// Whether distributed queries are cancellable. When true, the query
    /// checks a cancellation flag between peer responses and stops early.
    pub cancellable: bool,
}

impl Default for DistributedRagConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_scope: DocumentScope::Private,
            shared_chunk_ttl_secs: 86400, // 24 hours
            refresh_interval_secs: 3600,  // refresh every hour
            max_distributed_chunks: 20,
            encrypt_shared: false,
            query_timeout_secs: 5,
            cancellable: true,
        }
    }
}

/// Result of a distributed RAG query.
#[derive(Debug, Clone)]
pub struct DistributedRagResult {
    /// Chunks from local RAG.
    pub local_chunks: usize,
    /// Chunks from distributed DHT.
    pub distributed_chunks: usize,
    /// Total chunks after merge + dedup.
    pub total_chunks: usize,
    /// Nodes that contributed chunks.
    pub contributing_nodes: Vec<String>,
}

// ============================================================================
// ICE (Interactive Connectivity Establishment) Types
// ============================================================================

/// ICE candidate for NAT traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCandidate {
    /// Candidate type.
    pub candidate_type: IceCandidateType,
    /// IP address.
    pub address: String,
    /// Port.
    pub port: u16,
    /// Priority (higher = preferred).
    pub priority: u32,
    /// Protocol (UDP/TCP).
    pub protocol: String,
}

/// Type of ICE candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IceCandidateType {
    /// Direct local IP.
    Host,
    /// Discovered via STUN (public IP).
    ServerReflexive,
    /// Discovered during connectivity checks.
    PeerReflexive,
    /// Via TURN relay server.
    Relay,
}

impl std::fmt::Display for IceCandidateType {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => write!(f, "host"),
            Self::ServerReflexive => write!(f, "srflx"),
            Self::PeerReflexive => write!(f, "prflx"),
            Self::Relay => write!(f, "relay"),
            _ => write!(f, "unknown"),
        }
    }
}

/// ICE agent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IceState {
    New,
    Gathering,
    Checking,
    Connected,
    Completed,
    Failed,
    Closed,
}

impl Default for IceState {
    fn default() -> Self {
        Self::New
    }
}

/// ICE agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IceConfig {
    /// STUN servers for server-reflexive candidates.
    pub stun_servers: Vec<String>,
    /// TURN servers for relay candidates.
    pub turn_servers: Vec<TurnServerConfig>,
    /// Timeout for connectivity checks (ms).
    pub check_timeout_ms: u64,
    /// Whether to gather relay candidates (requires TURN server).
    pub gather_relay: bool,
}

impl Default for IceConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            turn_servers: Vec::new(),
            check_timeout_ms: 5000,
            gather_relay: false,
        }
    }
}

/// TURN server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnServerConfig {
    /// TURN server URL (turn:host:port).
    pub url: String,
    /// Username for TURN authentication.
    pub username: String,
    /// Credential (should use SecureString in production).
    pub credential: String,
}

/// Result of ICE candidate gathering.
#[derive(Debug, Clone)]
pub struct IceGatherResult {
    pub candidates: Vec<IceCandidate>,
    pub state: IceState,
    pub selected: Option<IceCandidate>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_scope_default() {
        assert_eq!(DocumentScope::default(), DocumentScope::Private);
    }

    #[test]
    fn test_shared_chunk_not_expired() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let chunk = SharedChunkMeta {
            chunk_hash: "abc".into(),
            source: "doc.md".into(),
            document_id: "doc1".into(),
            owner_node: "node1".into(),
            content: "test".into(),
            section: "intro".into(),
            base_score: 0.8,
            tokens: 10,
            ttl_secs: 3600,
            created_at: now,
        };
        assert!(!chunk.is_expired());
    }

    #[test]
    fn test_shared_chunk_expired() {
        let chunk = SharedChunkMeta {
            chunk_hash: "abc".into(),
            source: "doc.md".into(),
            document_id: "doc1".into(),
            owner_node: "node1".into(),
            content: "test".into(),
            section: "intro".into(),
            base_score: 0.8,
            tokens: 10,
            ttl_secs: 1,
            created_at: 1000, // far in the past
        };
        assert!(chunk.is_expired());
    }

    #[test]
    fn test_shared_chunk_no_ttl() {
        let chunk = SharedChunkMeta {
            chunk_hash: "abc".into(),
            source: "doc.md".into(),
            document_id: "doc1".into(),
            owner_node: "node1".into(),
            content: "test".into(),
            section: "intro".into(),
            base_score: 0.8,
            tokens: 10,
            ttl_secs: 0, // no expiry
            created_at: 1000,
        };
        assert!(!chunk.is_expired());
    }

    #[test]
    fn test_ice_candidate_type_display() {
        assert_eq!(IceCandidateType::Host.to_string(), "host");
        assert_eq!(IceCandidateType::ServerReflexive.to_string(), "srflx");
        assert_eq!(IceCandidateType::Relay.to_string(), "relay");
    }

    #[test]
    fn test_ice_state_default() {
        assert_eq!(IceState::default(), IceState::New);
    }

    #[test]
    fn test_distributed_rag_config_default() {
        let config = DistributedRagConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.default_scope, DocumentScope::Private);
        assert_eq!(config.shared_chunk_ttl_secs, 86400);
    }
}
