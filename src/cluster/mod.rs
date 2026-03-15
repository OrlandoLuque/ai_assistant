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
#[non_exhaustive]
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
    /// Published model catalog: "{node_id}:{model_name}" → serialized ModelAdvertisement.
    /// Synchronized across nodes via CRDT merge.
    pub model_catalog: Arc<RwLock<LWWMap<String, String>>>,
}

impl ClusterState {
    fn new() -> Self {
        Self {
            rate_limits: Arc::new(RwLock::new(PNCounter::new())),
            sessions: Arc::new(RwLock::new(LWWMap::new())),
            config_register: Arc::new(RwLock::new(LWWMap::new())),
            active_nodes: Arc::new(RwLock::new(ORSet::new())),
            request_counts: Arc::new(RwLock::new(GCounter::new())),
            model_catalog: Arc::new(RwLock::new(LWWMap::new())),
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

                    // Sync model catalog: serialize all entries for peer access
                    let model_catalog = state.model_catalog.read().await;
                    for cat_key in model_catalog.keys() {
                        if let Some(value) = model_catalog.get(cat_key) {
                            let key = format!("crdt:model_catalog:{}:{}", node_id, cat_key);
                            node.local_store(
                                &key,
                                value.as_bytes().to_vec(),
                                Some(Duration::from_secs(120)),
                            );
                        }
                    }
                    drop(model_catalog);

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

                    // Snapshot model catalog
                    let model_catalog = state.model_catalog.read().await;
                    let catalog_entries: Vec<(&String, &String)> = model_catalog
                        .keys()
                        .iter()
                        .filter_map(|k| model_catalog.get(k).map(|v| (*k, v)))
                        .collect();
                    if let Ok(data) = serde_json::to_vec(&catalog_entries) {
                        let _ = persistence.write_snapshot("model_catalog", &data);
                    }
                    drop(model_catalog);
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

        let published_model_count = {
            let catalog = self.state.model_catalog.read().await;
            let mut count = 0;
            for key in catalog.keys() {
                if let Some(value) = catalog.get(key) {
                    if let Ok(ad) = serde_json::from_str::<ModelAdvertisement>(value) {
                        if ad.published {
                            count += 1;
                        }
                    }
                }
            }
            count
        };

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
            published_model_count,
            metrics: self.metrics.snapshot(),
        }
    }
}

// ============================================================================
// Model Catalog — advertisements for inter-node model discovery
// ============================================================================

/// A model advertisement broadcast via CRDT to cluster peers.
///
/// Each published model (physical or virtual) is serialized to JSON and stored
/// in `ClusterState::model_catalog` keyed by `"{node_id}:{model_name}"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelAdvertisement {
    /// The node that owns this model.
    pub node_id: String,
    /// Model name as used by API clients.
    pub model_name: String,
    /// "physical" or "virtual".
    pub model_type: String,
    /// Provider name (for physical models).
    pub provider: Option<String>,
    /// Human-readable description.
    pub description: Option<String>,
    /// Whether this model is currently published (false = tombstone).
    pub published: bool,
}

impl ModelAdvertisement {
    /// Create a catalog key for this advertisement.
    pub fn catalog_key(&self) -> String {
        format!("{}:{}", self.node_id, self.model_name)
    }
}

impl ClusterManager {
    /// Advertise a model in the cluster catalog.
    ///
    /// Inserts (or updates) a `ModelAdvertisement` into the CRDT-backed catalog,
    /// which will be replicated to peers during the next sync cycle.
    pub async fn advertise_model(&self, ad: ModelAdvertisement) {
        let key = ad.catalog_key();
        let value = serde_json::to_string(&ad).unwrap_or_default();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut catalog = self.state.model_catalog.write().await;
        catalog.set(key, value, ts, &self.node_id);
    }

    /// Withdraw (unpublish) a model from the cluster catalog.
    ///
    /// Sets `published = false` in the advertisement so peers stop routing to it.
    pub async fn withdraw_model(&self, model_name: &str) {
        let key = format!("{}:{}", self.node_id, model_name);
        let ad = ModelAdvertisement {
            node_id: self.node_id.clone(),
            model_name: model_name.to_string(),
            model_type: "physical".to_string(),
            provider: None,
            description: None,
            published: false,
        };
        let value = serde_json::to_string(&ad).unwrap_or_default();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut catalog = self.state.model_catalog.write().await;
        catalog.set(key, value, ts, &self.node_id);
    }

    /// List all published models from all peers (including self).
    pub async fn list_published_models(&self) -> Vec<ModelAdvertisement> {
        let catalog = self.state.model_catalog.read().await;
        let mut result = Vec::new();
        for key in catalog.keys() {
            if let Some(value) = catalog.get(key) {
                if let Ok(ad) = serde_json::from_str::<ModelAdvertisement>(value) {
                    if ad.published {
                        result.push(ad);
                    }
                }
            }
        }
        result
    }

    /// List published models from a specific peer node.
    pub async fn list_peer_models(&self, peer_node_id: &str) -> Vec<ModelAdvertisement> {
        let catalog = self.state.model_catalog.read().await;
        let mut result = Vec::new();
        let prefix = format!("{}:", peer_node_id);
        for key in catalog.keys() {
            if key.starts_with(&prefix) {
                if let Some(value) = catalog.get(key) {
                    if let Ok(ad) = serde_json::from_str::<ModelAdvertisement>(value) {
                        if ad.published {
                            result.push(ad);
                        }
                    }
                }
            }
        }
        result
    }

    /// Find which node hosts a given model name.
    ///
    /// Returns all advertisements for the model (could be on multiple nodes).
    pub async fn find_model(&self, model_name: &str) -> Vec<ModelAdvertisement> {
        let catalog = self.state.model_catalog.read().await;
        let mut result = Vec::new();
        let suffix = format!(":{}", model_name);
        for key in catalog.keys() {
            if key.ends_with(&suffix) {
                if let Some(value) = catalog.get(key) {
                    if let Ok(ad) = serde_json::from_str::<ModelAdvertisement>(value) {
                        if ad.published {
                            result.push(ad);
                        }
                    }
                }
            }
        }
        result
    }

    /// Get the model catalog for CRDT sync serialization.
    pub fn model_catalog(&self) -> &Arc<RwLock<LWWMap<String, String>>> {
        &self.state.model_catalog
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
    pub published_model_count: usize,
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
            published_model_count: 0,
            metrics: metrics::MetricsSnapshot::default(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"node_id\":\"test\""));
        assert!(json.contains("\"is_ready\":true"));
    }

    // ========================================================================
    // Model Catalog tests
    // ========================================================================

    #[test]
    fn test_model_advertisement_serialization() {
        let ad = ModelAdvertisement {
            node_id: "node1".to_string(),
            model_name: "llama3:8b".to_string(),
            model_type: "physical".to_string(),
            provider: Some("ollama".to_string()),
            description: Some("Llama 3 8B model".to_string()),
            published: true,
        };
        let json = serde_json::to_string(&ad).unwrap();
        let parsed: ModelAdvertisement = serde_json::from_str(&json).unwrap();
        assert_eq!(ad, parsed);
        assert!(json.contains("\"model_name\":\"llama3:8b\""));
    }

    #[test]
    fn test_model_advertisement_catalog_key() {
        let ad = ModelAdvertisement {
            node_id: "node1".to_string(),
            model_name: "gpt-4o".to_string(),
            model_type: "physical".to_string(),
            provider: Some("openai".to_string()),
            description: None,
            published: true,
        };
        assert_eq!(ad.catalog_key(), "node1:gpt-4o");
    }

    #[test]
    fn test_model_catalog_crdt_operations() {
        let state = ClusterState::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let ad = ModelAdvertisement {
                node_id: "node1".to_string(),
                model_name: "llama3:8b".to_string(),
                model_type: "physical".to_string(),
                provider: Some("ollama".to_string()),
                description: None,
                published: true,
            };
            let key = ad.catalog_key();
            let value = serde_json::to_string(&ad).unwrap();

            // Insert into catalog
            state.model_catalog.write().await.set(key.clone(), value, 1000, "node1");

            // Read back
            let catalog = state.model_catalog.read().await;
            let stored = catalog.get(&key).unwrap();
            let parsed: ModelAdvertisement = serde_json::from_str(stored).unwrap();
            assert_eq!(parsed.model_name, "llama3:8b");
            assert!(parsed.published);
        });
    }

    #[test]
    fn test_model_catalog_crdt_merge() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state1 = ClusterState::new();
            let state2 = ClusterState::new();

            // Node1 publishes a model
            let ad1 = ModelAdvertisement {
                node_id: "node1".to_string(),
                model_name: "llama3:8b".to_string(),
                model_type: "physical".to_string(),
                provider: Some("ollama".to_string()),
                description: None,
                published: true,
            };
            state1.model_catalog.write().await.set(
                ad1.catalog_key(),
                serde_json::to_string(&ad1).unwrap(),
                1000,
                "node1",
            );

            // Node2 publishes a different model
            let ad2 = ModelAdvertisement {
                node_id: "node2".to_string(),
                model_name: "gpt-4o".to_string(),
                model_type: "physical".to_string(),
                provider: Some("openai".to_string()),
                description: None,
                published: true,
            };
            state2.model_catalog.write().await.set(
                ad2.catalog_key(),
                serde_json::to_string(&ad2).unwrap(),
                1000,
                "node2",
            );

            // Merge state2 into state1
            let catalog2 = state2.model_catalog.read().await;
            state1.model_catalog.write().await.merge(&catalog2);
            drop(catalog2);

            // State1 should now have both
            let catalog = state1.model_catalog.read().await;
            assert!(catalog.get(&"node1:llama3:8b".to_string()).is_some());
            assert!(catalog.get(&"node2:gpt-4o".to_string()).is_some());
        });
    }

    #[test]
    fn test_model_catalog_unpublish_via_crdt() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = ClusterState::new();

            // Publish a model
            let ad = ModelAdvertisement {
                node_id: "node1".to_string(),
                model_name: "llama3:8b".to_string(),
                model_type: "physical".to_string(),
                provider: Some("ollama".to_string()),
                description: None,
                published: true,
            };
            state.model_catalog.write().await.set(
                ad.catalog_key(),
                serde_json::to_string(&ad).unwrap(),
                1000,
                "node1",
            );

            // Unpublish it (higher timestamp)
            let ad_unpub = ModelAdvertisement {
                published: false,
                ..ad.clone()
            };
            state.model_catalog.write().await.set(
                ad_unpub.catalog_key(),
                serde_json::to_string(&ad_unpub).unwrap(),
                2000,
                "node1",
            );

            // Should reflect unpublished
            let catalog = state.model_catalog.read().await;
            let stored = catalog.get(&"node1:llama3:8b".to_string()).unwrap();
            let parsed: ModelAdvertisement = serde_json::from_str(stored).unwrap();
            assert!(!parsed.published);
        });
    }

    #[test]
    fn test_model_catalog_lww_conflict_resolution() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state1 = ClusterState::new();
            let state2 = ClusterState::new();
            let key = "node1:llama3:8b".to_string();

            // Node1 publishes at t=1000
            let ad_pub = ModelAdvertisement {
                node_id: "node1".to_string(),
                model_name: "llama3:8b".to_string(),
                model_type: "physical".to_string(),
                provider: Some("ollama".to_string()),
                description: None,
                published: true,
            };
            state1.model_catalog.write().await.set(
                key.clone(),
                serde_json::to_string(&ad_pub).unwrap(),
                1000,
                "node1",
            );

            // Node2 has a stale unpublish at t=500
            let ad_unpub = ModelAdvertisement {
                published: false,
                ..ad_pub.clone()
            };
            state2.model_catalog.write().await.set(
                key.clone(),
                serde_json::to_string(&ad_unpub).unwrap(),
                500,
                "node2",
            );

            // Merge state2 into state1 — LWW should keep the t=1000 (published) version
            let catalog2 = state2.model_catalog.read().await;
            state1.model_catalog.write().await.merge(&catalog2);
            drop(catalog2);

            let catalog = state1.model_catalog.read().await;
            let stored = catalog.get(&key).unwrap();
            let parsed: ModelAdvertisement = serde_json::from_str(stored).unwrap();
            assert!(parsed.published, "LWW should keep the later (published) version");
        });
    }

    #[test]
    fn test_model_catalog_multiple_models_per_node() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = ClusterState::new();

            let models = vec![
                ("llama3:8b", "physical", Some("ollama"), true),
                ("gpt-4o", "physical", Some("openai"), true),
                ("rag-assistant", "virtual", None, true),
                ("internal-only", "virtual", None, false), // unpublished
            ];

            for (name, mtype, provider, published) in &models {
                let ad = ModelAdvertisement {
                    node_id: "node1".to_string(),
                    model_name: name.to_string(),
                    model_type: mtype.to_string(),
                    provider: provider.map(|s| s.to_string()),
                    description: None,
                    published: *published,
                };
                state.model_catalog.write().await.set(
                    ad.catalog_key(),
                    serde_json::to_string(&ad).unwrap(),
                    1000,
                    "node1",
                );
            }

            // Count published
            let catalog = state.model_catalog.read().await;
            let mut published_count = 0;
            for key in catalog.keys() {
                if let Some(value) = catalog.get(key) {
                    if let Ok(ad) = serde_json::from_str::<ModelAdvertisement>(value) {
                        if ad.published {
                            published_count += 1;
                        }
                    }
                }
            }
            assert_eq!(published_count, 3);
        });
    }

    #[test]
    fn test_model_catalog_multi_node_merge() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state1 = ClusterState::new();
            let state2 = ClusterState::new();
            let state3 = ClusterState::new();

            // Each node publishes different models
            for (node_id, model, state) in [
                ("node1", "llama3:8b", &state1),
                ("node2", "gpt-4o", &state2),
                ("node3", "mixtral:8x7b", &state3),
            ] {
                let ad = ModelAdvertisement {
                    node_id: node_id.to_string(),
                    model_name: model.to_string(),
                    model_type: "physical".to_string(),
                    provider: Some("test".to_string()),
                    description: None,
                    published: true,
                };
                state.model_catalog.write().await.set(
                    ad.catalog_key(),
                    serde_json::to_string(&ad).unwrap(),
                    1000,
                    node_id,
                );
            }

            // Merge all into state1
            let cat2 = state2.model_catalog.read().await;
            state1.model_catalog.write().await.merge(&cat2);
            drop(cat2);
            let cat3 = state3.model_catalog.read().await;
            state1.model_catalog.write().await.merge(&cat3);
            drop(cat3);

            // State1 should have all 3
            let catalog = state1.model_catalog.read().await;
            assert!(catalog.get(&"node1:llama3:8b".to_string()).is_some());
            assert!(catalog.get(&"node2:gpt-4o".to_string()).is_some());
            assert!(catalog.get(&"node3:mixtral:8x7b".to_string()).is_some());
        });
    }

    #[test]
    fn test_model_catalog_virtual_model_advertisement() {
        let ad = ModelAdvertisement {
            node_id: "node1".to_string(),
            model_name: "rag-assistant".to_string(),
            model_type: "virtual".to_string(),
            provider: None,
            description: Some("RAG-enhanced assistant with guardrails".to_string()),
            published: true,
        };
        let json = serde_json::to_string(&ad).unwrap();
        let parsed: ModelAdvertisement = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_type, "virtual");
        assert!(parsed.provider.is_none());
        assert!(parsed.description.is_some());
    }

    #[test]
    fn test_model_catalog_empty_state() {
        let state = ClusterState::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let catalog = state.model_catalog.read().await;
            assert!(catalog.keys().is_empty());
        });
    }
}
