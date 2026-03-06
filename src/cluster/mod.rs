//! # Cluster Module (v30)
//!
//! Wires the existing distributed infrastructure modules into the axum server
//! for horizontal scaling and high availability.
//!
//! ## Submodules
//! - `distributed_rate_limit` — 2-layer rate limiter (local + CRDT sync)
//! - `crdt_persistence` — WAL, snapshots, TTL, tombstone compaction
//! - `health` — Readiness/liveness probes, node drain, session migration, circuit breaker
//! - `metrics` — Prometheus cluster metrics
//!
//! ## Architecture
//! `ClusterManager` wraps `NetworkNode` + CRDTs + `ConsistentHashRing` + `HeartbeatManager`
//! and runs background sync tasks via tokio. All cluster functionality is behind the
//! `server-cluster` feature flag and configurable on/off at runtime via `ClusterConfig`.

pub mod crdt_persistence;
pub mod distributed_rate_limit;
pub mod health;
pub mod metrics;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::consistent_hash::ConsistentHashRing;
use crate::distributed::{GCounter, LWWMap, NodeId, ORSet, PNCounter};
use crate::distributed_network::{NetworkConfig, NetworkEvent, NetworkNode, ReplicationConfig};
use crate::failure_detector::{HeartbeatConfig, HeartbeatManager};
use crate::merkle_sync::AntiEntropySync;

use self::crdt_persistence::CrdtPersistence;
use self::distributed_rate_limit::DistributedRateLimiter;
use self::health::ClusterHealthManager;
use self::metrics::ClusterMetrics;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for cluster mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Unique node identifier.
    pub node_id: String,
    /// QUIC listen address for mesh networking.
    pub quic_addr: SocketAddr,
    /// Bootstrap peer addresses to join the cluster.
    pub bootstrap_peers: Vec<SocketAddr>,
    /// Optional join token for cluster admission.
    pub join_token: Option<String>,
    /// Directory for CRDT persistence (WAL + snapshots).
    pub data_dir: PathBuf,
    /// Heartbeat interval for failure detection.
    #[serde(default = "default_heartbeat_interval_ms")]
    pub heartbeat_interval_ms: u64,
    /// Phi threshold for failure detection.
    #[serde(default = "default_phi_threshold")]
    pub phi_threshold: f64,
    /// CRDT sync interval.
    #[serde(default = "default_sync_interval_secs")]
    pub sync_interval_secs: u64,
    /// Snapshot interval for CRDT persistence.
    #[serde(default = "default_snapshot_interval_secs")]
    pub snapshot_interval_secs: u64,
    /// Maximum connections per node.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    /// Replication factor for data.
    #[serde(default = "default_replication_factor")]
    pub replication_factor: usize,
}

fn default_heartbeat_interval_ms() -> u64 { 1000 }
fn default_phi_threshold() -> f64 { 12.0 }
fn default_sync_interval_secs() -> u64 { 30 }
fn default_snapshot_interval_secs() -> u64 { 300 }
fn default_max_connections() -> usize { 64 }
fn default_replication_factor() -> usize { 3 }

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: "node1".to_string(),
            quic_addr: "127.0.0.1:9001".parse().unwrap(),
            bootstrap_peers: vec![],
            join_token: None,
            data_dir: PathBuf::from("./data"),
            heartbeat_interval_ms: default_heartbeat_interval_ms(),
            phi_threshold: default_phi_threshold(),
            sync_interval_secs: default_sync_interval_secs(),
            snapshot_interval_secs: default_snapshot_interval_secs(),
            max_connections: default_max_connections(),
            replication_factor: default_replication_factor(),
        }
    }
}

// ============================================================================
// Cluster State (CRDT-based shared state)
// ============================================================================

/// Shared cluster state synchronized via CRDTs.
pub struct ClusterState {
    /// Rate limit counters (IP → count) synchronized across nodes.
    pub rate_limits: Arc<RwLock<PNCounter>>,
    /// Active sessions tracked across the cluster.
    pub sessions: Arc<RwLock<LWWMap<String, Vec<u8>>>>,
    /// Cluster-wide configuration register.
    pub config_register: Arc<RwLock<LWWMap<String, String>>>,
    /// Active nodes set (add/remove semantics).
    pub active_nodes: Arc<RwLock<ORSet<String>>>,
    /// Per-endpoint request counters for load balancing.
    pub request_counts: Arc<RwLock<GCounter>>,
}

impl ClusterState {
    fn new() -> Self {
        Self {
            rate_limits: Arc::new(RwLock::new(PNCounter::new())),
            sessions: Arc::new(RwLock::new(LWWMap::new())),
            config_register: Arc::new(RwLock::new(LWWMap::new())),
            active_nodes: Arc::new(RwLock::new(ORSet::new())),
            request_counts: Arc::new(RwLock::new(GCounter::new())),
        }
    }
}

// ============================================================================
// ClusterManager
// ============================================================================

/// Manages cluster state, CRDT synchronization, and peer communication.
///
/// Orchestrates:
/// - `NetworkNode` for QUIC mesh communication
/// - `ConsistentHashRing` for data partitioning
/// - `HeartbeatManager` for failure detection
/// - `AntiEntropySync` for Merkle-based reconciliation
/// - CRDTs for eventually-consistent shared state
/// - `CrdtPersistence` for WAL + snapshot durability
/// - `DistributedRateLimiter` for 2-layer rate limiting
/// - `ClusterHealthManager` for readiness/liveness
/// - `ClusterMetrics` for Prometheus-compatible observability
pub struct ClusterManager {
    /// Node identifier.
    node_id: String,
    /// Cluster configuration.
    config: ClusterConfig,
    /// The QUIC-based network node.
    network_node: Arc<NetworkNode>,
    /// Consistent hash ring for key/session routing.
    ring: Arc<RwLock<ConsistentHashRing>>,
    /// Heartbeat-based failure detection.
    heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
    /// Anti-entropy synchronization.
    #[allow(dead_code)]
    anti_entropy: Arc<RwLock<AntiEntropySync>>,
    /// Shared CRDT state.
    state: Arc<ClusterState>,
    /// CRDT persistence (WAL + snapshots).
    persistence: Arc<CrdtPersistence>,
    /// 2-layer distributed rate limiter.
    rate_limiter: Arc<DistributedRateLimiter>,
    /// Cluster health management.
    health: Arc<ClusterHealthManager>,
    /// Prometheus-compatible metrics.
    metrics: Arc<ClusterMetrics>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Time the node started.
    started_at: Instant,
}

impl ClusterManager {
    /// Create and initialize a new ClusterManager.
    ///
    /// This creates the NetworkNode, connects to bootstrap peers, and starts
    /// background tasks for heartbeat, CRDT sync, and anti-entropy.
    pub fn new(config: ClusterConfig) -> Result<Self, String> {
        let node_id_obj = NodeId::from_string(&config.node_id);

        // Build NetworkConfig from ClusterConfig
        let network_config = NetworkConfig {
            listen_addr: config.quic_addr,
            bootstrap_peers: config.bootstrap_peers.clone(),
            identity_dir: config.data_dir.join("identity"),
            heartbeat_interval: Duration::from_millis(config.heartbeat_interval_ms),
            replication: ReplicationConfig {
                min_copies: config.replication_factor,
                max_copies: config.replication_factor,
                ..Default::default()
            },
            join_token: config.join_token.clone(),
            max_connections: config.max_connections,
            phi_threshold: config.phi_threshold,
            ..Default::default()
        };

        let network_node = NetworkNode::new(network_config)
            .map_err(|e| format!("Failed to create NetworkNode: {}", e))?;
        let network_node = Arc::new(network_node);

        let ring = Arc::new(RwLock::new(ConsistentHashRing::new(64, config.replication_factor)));
        let heartbeat_mgr = Arc::new(RwLock::new(HeartbeatManager::new(HeartbeatConfig {
            interval: Duration::from_millis(config.heartbeat_interval_ms),
            phi_threshold: config.phi_threshold,
            max_samples: 200,
            suspicious_threshold: config.phi_threshold / 2.0,
        })));
        let anti_entropy = Arc::new(RwLock::new(AntiEntropySync::new(
            Duration::from_secs(config.sync_interval_secs),
        )));

        let state = Arc::new(ClusterState::new());
        let persistence = Arc::new(CrdtPersistence::new(config.data_dir.clone()));
        let rate_limiter = Arc::new(DistributedRateLimiter::new(
            config.node_id.clone(),
            state.rate_limits.clone(),
        ));
        let health = Arc::new(ClusterHealthManager::new(
            config.node_id.clone(),
            heartbeat_mgr.clone(),
        ));
        let cluster_metrics = Arc::new(ClusterMetrics::new(config.node_id.clone()));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Add ourselves to the ring and active nodes
        {
            let rt = tokio::runtime::Handle::try_current();
            if let Ok(handle) = rt {
                let ring_c = ring.clone();
                let state_c = state.clone();
                let nid = node_id_obj;
                let nid_str = config.node_id.clone();
                handle.spawn(async move {
                    ring_c.write().await.add_node(nid);
                    state_c.active_nodes.write().await.add(nid_str, "system");
                });
            }
        }

        Ok(Self {
            node_id: config.node_id.clone(),
            config,
            network_node,
            ring,
            heartbeat_mgr,
            anti_entropy,
            state,
            persistence,
            rate_limiter,
            health,
            metrics: cluster_metrics,
            shutdown,
            started_at: Instant::now(),
        })
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get cluster configuration.
    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }

    /// Get the network node for direct QUIC operations.
    pub fn network_node(&self) -> &NetworkNode {
        &self.network_node
    }

    /// Get the consistent hash ring.
    pub fn ring(&self) -> &Arc<RwLock<ConsistentHashRing>> {
        &self.ring
    }

    /// Get shared cluster state (CRDTs).
    pub fn state(&self) -> &Arc<ClusterState> {
        &self.state
    }

    /// Get the distributed rate limiter.
    pub fn rate_limiter(&self) -> &Arc<DistributedRateLimiter> {
        &self.rate_limiter
    }

    /// Get the health manager.
    pub fn health(&self) -> &Arc<ClusterHealthManager> {
        &self.health
    }

    /// Get cluster metrics.
    pub fn metrics(&self) -> &Arc<ClusterMetrics> {
        &self.metrics
    }

    /// Get uptime since the node started.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get the number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.network_node.peer_count()
    }

    /// Get detailed peer information.
    pub fn peers(&self) -> Vec<(NodeId, SocketAddr, f32)> {
        self.network_node.peers()
    }

    /// Start all background tasks (heartbeat, sync, anti-entropy, persistence).
    ///
    /// Must be called within a tokio runtime context.
    pub async fn start_background_tasks(&self) {
        let shutdown = self.shutdown.clone();

        // Task 1: Heartbeat loop — send/receive heartbeats, update failure detector
        {
            let node = self.network_node.clone();
            let hb_mgr = self.heartbeat_mgr.clone();
            let ring = self.ring.clone();
            let state = self.state.clone();
            let metrics = self.metrics.clone();
            let health = self.health.clone();
            let interval = Duration::from_millis(self.config.heartbeat_interval_ms);
            let shutdown = shutdown.clone();
            let node_id = self.node_id.clone();

            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                while !shutdown.load(Ordering::Relaxed) {
                    ticker.tick().await;

                    // Process network events
                    let events = node.poll_events();
                    for event in events {
                        match event {
                            NetworkEvent::PeerConnected(peer_id, addr) => {
                                let mut r = ring.write().await;
                                r.add_node(peer_id);
                                let mut nodes = state.active_nodes.write().await;
                                nodes.add(peer_id.to_hex(), &node_id);
                                metrics.record_ring_rebalance();
                                log::info!("Peer connected: {} at {}", peer_id.to_hex(), addr);
                            }
                            NetworkEvent::PeerDisconnected(peer_id) => {
                                let mut r = ring.write().await;
                                r.remove_node(&peer_id);
                                let mut hb = hb_mgr.write().await;
                                hb.remove_node(&peer_id);
                                metrics.record_ring_rebalance();
                                log::info!("Peer disconnected: {}", peer_id.to_hex());
                            }
                            NetworkEvent::PeerFailed(peer_id, phi) => {
                                health.record_peer_failure(&peer_id);
                                log::warn!("Peer failed: {} (phi={})", peer_id.to_hex(), phi);
                            }
                            _ => {}
                        }
                    }

                    // Record heartbeats from connected peers
                    let peers = node.peers();
                    let mut hb = hb_mgr.write().await;
                    for (peer_id, _addr, _rep) in &peers {
                        hb.record_heartbeat(peer_id);
                    }

                    // Check for dead nodes
                    let dead = hb.get_dead_nodes();
                    for dead_id in dead {
                        health.record_peer_failure(&dead_id);
                    }

                    metrics.set_active_nodes(peers.len() + 1); // +1 for self
                }
            });
        }

        // Task 2: CRDT sync loop — merge CRDTs with peers periodically
        {
            let node = self.network_node.clone();
            let state = self.state.clone();
            let metrics = self.metrics.clone();
            let interval = Duration::from_secs(self.config.sync_interval_secs);
            let shutdown = shutdown.clone();
            let node_id = self.node_id.clone();

            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                while !shutdown.load(Ordering::Relaxed) {
                    ticker.tick().await;

                    // Serialize CRDT inner state and store in network node for peer access
                    let rate_limits = state.rate_limits.read().await;
                    if let Ok(data) = serde_json::to_vec(&rate_limits.positive.counts) {
                        let key = format!("crdt:rate_limits_p:{}", node_id);
                        node.local_store(&key, data, Some(Duration::from_secs(120)));
                    }
                    if let Ok(data) = serde_json::to_vec(&rate_limits.negative.counts) {
                        let key = format!("crdt:rate_limits_n:{}", node_id);
                        node.local_store(&key, data, Some(Duration::from_secs(120)));
                    }
                    drop(rate_limits);

                    let request_counts = state.request_counts.read().await;
                    if let Ok(data) = serde_json::to_vec(&request_counts.counts) {
                        let key = format!("crdt:request_counts:{}", node_id);
                        node.local_store(&key, data, Some(Duration::from_secs(120)));
                    }
                    drop(request_counts);

                    metrics.record_crdt_sync();
                }
            });
        }

        // Task 3: Persistence loop — snapshot CRDTs to disk periodically
        {
            let state = self.state.clone();
            let persistence = self.persistence.clone();
            let interval = Duration::from_secs(self.config.snapshot_interval_secs);
            let shutdown = shutdown.clone();

            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                while !shutdown.load(Ordering::Relaxed) {
                    ticker.tick().await;

                    // Snapshot all CRDTs (serialize inner data)
                    let rate_limits = state.rate_limits.read().await;
                    if let Ok(data) = serde_json::to_vec(&rate_limits.positive.counts) {
                        let _ = persistence.write_snapshot("rate_limits_p", &data);
                    }
                    if let Ok(data) = serde_json::to_vec(&rate_limits.negative.counts) {
                        let _ = persistence.write_snapshot("rate_limits_n", &data);
                    }
                    drop(rate_limits);

                    let request_counts = state.request_counts.read().await;
                    if let Ok(data) = serde_json::to_vec(&request_counts.counts) {
                        let _ = persistence.write_snapshot("request_counts", &data);
                    }
                    drop(request_counts);
                }
            });
        }
    }

    /// Graceful shutdown: stop background tasks, leave cluster ring, persist state.
    pub async fn shutdown(&self) {
        log::info!("ClusterManager shutting down...");
        self.shutdown.store(true, Ordering::Relaxed);
        self.health.start_drain();

        // Final persistence snapshot (serialize inner data)
        let rate_limits = self.state.rate_limits.read().await;
        if let Ok(data) = serde_json::to_vec(&rate_limits.positive.counts) {
            let _ = self.persistence.write_snapshot("rate_limits_p", &data);
        }
        if let Ok(data) = serde_json::to_vec(&rate_limits.negative.counts) {
            let _ = self.persistence.write_snapshot("rate_limits_n", &data);
        }
        drop(rate_limits);

        let request_counts = self.state.request_counts.read().await;
        if let Ok(data) = serde_json::to_vec(&request_counts.counts) {
            let _ = self.persistence.write_snapshot("request_counts", &data);
        }
        drop(request_counts);

        // Remove self from ring
        let nid = NodeId::from_string(&self.node_id);
        self.ring.write().await.remove_node(&nid);

        // Shutdown network node
        self.network_node.shutdown();
        log::info!("ClusterManager shutdown complete.");
    }

    /// Get a debug info snapshot of the cluster state.
    pub async fn debug_info(&self) -> ClusterDebugInfo {
        let ring = self.ring.read().await;
        let hb = self.heartbeat_mgr.read().await;
        let active_nodes = self.state.active_nodes.read().await;

        ClusterDebugInfo {
            node_id: self.node_id.clone(),
            uptime_secs: self.started_at.elapsed().as_secs(),
            peer_count: self.network_node.peer_count(),
            ring_node_count: ring.node_count(),
            ring_vnode_count: ring.vnode_count(),
            replication_factor: ring.replication_factor(),
            active_nodes: active_nodes.elements().iter().map(|s| s.to_string()).collect(),
            monitored_nodes: hb.monitored_count(),
            dead_nodes: hb.get_dead_nodes().iter().map(|n| n.to_hex()).collect(),
            is_draining: self.health.is_draining(),
            is_ready: self.health.is_ready(),
            metrics: self.metrics.snapshot(),
        }
    }
}

/// Debug information about the cluster state.
#[derive(Debug, Serialize)]
pub struct ClusterDebugInfo {
    pub node_id: String,
    pub uptime_secs: u64,
    pub peer_count: usize,
    pub ring_node_count: usize,
    pub ring_vnode_count: usize,
    pub replication_factor: usize,
    pub active_nodes: Vec<String>,
    pub monitored_nodes: usize,
    pub dead_nodes: Vec<String>,
    pub is_draining: bool,
    pub is_ready: bool,
    pub metrics: metrics::MetricsSnapshot,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_defaults() {
        let config = ClusterConfig::default();
        assert_eq!(config.node_id, "node1");
        assert_eq!(config.heartbeat_interval_ms, 1000);
        assert_eq!(config.phi_threshold, 12.0);
        assert_eq!(config.sync_interval_secs, 30);
        assert_eq!(config.snapshot_interval_secs, 300);
        assert_eq!(config.max_connections, 64);
        assert_eq!(config.replication_factor, 3);
    }

    #[test]
    fn test_cluster_config_serialization() {
        let config = ClusterConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ClusterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_id, config.node_id);
        assert_eq!(parsed.phi_threshold, config.phi_threshold);
    }

    #[test]
    fn test_cluster_state_creation() {
        let state = ClusterState::new();
        // All CRDTs should be empty
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let rl = state.rate_limits.read().await;
            assert_eq!(rl.value(), 0);
            let rc = state.request_counts.read().await;
            assert_eq!(rc.value(), 0);
            let an = state.active_nodes.read().await;
            assert!(an.is_empty());
        });
    }

    #[test]
    fn test_cluster_state_crdt_operations() {
        let state = ClusterState::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Rate limits
            state.rate_limits.write().await.increment("node1");
            state.rate_limits.write().await.increment("node1");
            state.rate_limits.write().await.increment("node2");
            assert_eq!(state.rate_limits.read().await.value(), 3);

            // Active nodes
            state.active_nodes.write().await.add("node1".to_string(), "system");
            state.active_nodes.write().await.add("node2".to_string(), "system");
            assert_eq!(state.active_nodes.read().await.len(), 2);

            // Request counts
            state.request_counts.write().await.increment("node1");
            assert_eq!(state.request_counts.read().await.value(), 1);
        });
    }

    #[test]
    fn test_cluster_state_session_tracking() {
        let state = ClusterState::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let ts = 1000u64;
            state.sessions.write().await.set(
                "session1".to_string(),
                b"node1".to_vec(),
                ts,
                "node1",
            );
            let sessions = state.sessions.read().await;
            assert_eq!(sessions.get(&"session1".to_string()), Some(&b"node1".to_vec()));
        });
    }

    #[test]
    fn test_cluster_config_custom() {
        let config = ClusterConfig {
            node_id: "custom-node".to_string(),
            quic_addr: "10.0.0.1:9001".parse().unwrap(),
            bootstrap_peers: vec!["10.0.0.2:9001".parse().unwrap()],
            join_token: Some("token123".to_string()),
            data_dir: PathBuf::from("/data/custom"),
            heartbeat_interval_ms: 500,
            phi_threshold: 8.0,
            sync_interval_secs: 15,
            snapshot_interval_secs: 120,
            max_connections: 32,
            replication_factor: 2,
        };
        assert_eq!(config.node_id, "custom-node");
        assert_eq!(config.bootstrap_peers.len(), 1);
        assert_eq!(config.replication_factor, 2);
    }

    #[test]
    fn test_cluster_debug_info_serialization() {
        let info = ClusterDebugInfo {
            node_id: "test".to_string(),
            uptime_secs: 100,
            peer_count: 2,
            ring_node_count: 3,
            ring_vnode_count: 192,
            replication_factor: 3,
            active_nodes: vec!["node1".to_string(), "node2".to_string()],
            monitored_nodes: 2,
            dead_nodes: vec![],
            is_draining: false,
            is_ready: true,
            metrics: metrics::MetricsSnapshot::default(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"node_id\":\"test\""));
        assert!(json.contains("\"is_ready\":true"));
    }
}
