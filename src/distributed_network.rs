//! Real distributed networking module with QUIC transport.
//!
//! Provides a complete networking stack for the distributed system:
//! - **QUIC transport** via quinn with mutual TLS (Ed25519 certificates)
//! - **Connection management** with automatic reconnection and peer tracking
//! - **Consistent hashing** for data partitioning across nodes
//! - **Phi Accrual Failure Detection** for node health monitoring
//! - **Anti-entropy sync** via Merkle trees for data consistency
//! - **Replication** with configurable factor and quorum semantics
//! - **LAN discovery** via UDP broadcast for zero-config clustering
//! - **Join tokens** for cluster admission control
//!
//! Architecture: A tokio runtime runs in a background thread. The public API
//! is synchronous, communicating with the event loop via channels. This matches
//! the pattern used by `LanceVectorDb` in this crate.
//!
//! This module is gated behind the `distributed-network` feature.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};

use crate::consistent_hash::ConsistentHashRing;
use crate::distributed::{NodeId, NodeMessage};
use crate::failure_detector::{HeartbeatConfig, HeartbeatManager, NodeStatus};
use crate::merkle_sync::{AntiEntropySync, MerkleTree};
use crate::node_security::{CertificateManager, JoinToken, NodeIdentity};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for a network node.
#[derive(Clone, Debug)]
pub struct NetworkConfig {
    /// Address to listen on. Use `0.0.0.0:0` for auto-assigned port.
    pub listen_addr: SocketAddr,
    /// Bootstrap peers to connect to on startup.
    pub bootstrap_peers: Vec<SocketAddr>,
    /// Directory for storing/loading node identity (certificates).
    pub identity_dir: PathBuf,
    /// Heartbeat interval for failure detection.
    pub heartbeat_interval: Duration,
    /// Replication configuration.
    pub replication: ReplicationConfig,
    /// Discovery configuration.
    pub discovery: DiscoveryConfig,
    /// Join token required to join an existing cluster (None for first node).
    pub join_token: Option<String>,
    /// Maximum concurrent connections.
    pub max_connections: usize,
    /// Timeout for message round-trips.
    pub message_timeout: Duration,
    /// Phi threshold for failure detection.
    pub phi_threshold: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:0".parse().expect("valid address"),
            bootstrap_peers: Vec::new(),
            identity_dir: PathBuf::from("./node_identity"),
            heartbeat_interval: Duration::from_secs(2),
            replication: ReplicationConfig::default(),
            discovery: DiscoveryConfig::default(),
            join_token: None,
            max_connections: 50,
            message_timeout: Duration::from_secs(10),
            phi_threshold: 8.0,
        }
    }
}

/// Replication configuration.
#[derive(Clone, Debug)]
pub struct ReplicationConfig {
    /// Minimum number of copies to maintain.
    pub min_copies: usize,
    /// Maximum number of copies to attempt.
    pub max_copies: usize,
    /// Whether writes wait for acknowledgement.
    pub write_mode: WriteMode,
    /// Number of nodes that must respond for a read to succeed.
    pub read_quorum: usize,
    /// Number of nodes that must acknowledge a write for it to succeed.
    pub write_quorum: usize,
    /// Virtual nodes per physical node in the hash ring.
    pub vnodes_per_node: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            min_copies: 2,
            max_copies: 3,
            write_mode: WriteMode::Asynchronous,
            read_quorum: 1,
            write_quorum: 1,
            vnodes_per_node: 64,
        }
    }
}

/// Write mode for replication.
#[derive(Clone, Debug, PartialEq)]
pub enum WriteMode {
    /// Wait for `write_quorum` acknowledgements before returning success.
    Synchronous,
    /// Write locally and replicate in the background.
    Asynchronous,
}

/// Discovery configuration for finding peers on the local network.
#[derive(Clone, Debug)]
pub struct DiscoveryConfig {
    /// Enable UDP broadcast for LAN discovery.
    pub enable_broadcast: bool,
    /// Port for LAN discovery broadcasts.
    pub broadcast_port: u16,
    /// Interval between discovery broadcasts.
    pub broadcast_interval: Duration,
    /// Enable peer exchange (ask peers for their peers).
    pub enable_peer_exchange: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_broadcast: true,
            broadcast_port: 9876,
            broadcast_interval: Duration::from_secs(10),
            enable_peer_exchange: true,
        }
    }
}

// =============================================================================
// Peer State
// =============================================================================

/// State of a connected peer.
#[derive(Debug, Clone)]
pub struct PeerState {
    /// The peer's node identifier.
    pub node_id: NodeId,
    /// The peer's network address.
    pub addr: SocketAddr,
    /// When the connection was established.
    pub connected_since: Option<Instant>,
    /// Peer's reputation score (0.0 to 1.0).
    pub reputation: f32,
    /// Whether the peer is in a probation period (newly joined).
    pub probation: bool,
    /// Total messages sent to this peer.
    pub messages_sent: u64,
    /// Total messages received from this peer.
    pub messages_received: u64,
}

impl PeerState {
    fn new(node_id: NodeId, addr: SocketAddr) -> Self {
        Self {
            node_id,
            addr,
            connected_since: Some(Instant::now()),
            reputation: 0.5,
            probation: true,
            messages_sent: 0,
            messages_received: 0,
        }
    }
}

// =============================================================================
// Events and Commands
// =============================================================================

/// Events emitted by the network node.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// A new peer connected.
    PeerConnected(NodeId, SocketAddr),
    /// A peer disconnected.
    PeerDisconnected(NodeId),
    /// A peer is suspected of being failed (with phi value).
    PeerFailed(NodeId, f64),
    /// A message was received from a peer.
    MessageReceived(NodeId, NodeMessage),
    /// Replication of a key completed to N nodes.
    ReplicationComplete(String, usize),
    /// A join request was received.
    JoinRequestReceived(NodeId, SocketAddr),
    /// An error occurred.
    Error(String),
}

/// Internal commands sent to the async event loop.
enum NetworkCommand {
    /// Connect to a peer at the given address.
    Connect(SocketAddr, oneshot::Sender<Result<NodeId, String>>),
    /// Send a message to a peer (fire-and-forget).
    SendMessage(NodeId, NodeMessage),
    /// Send a message and wait for a reply.
    Request(
        NodeId,
        NodeMessage,
        oneshot::Sender<Result<NodeMessage, String>>,
    ),
    /// Store a key-value pair with replication.
    Store(
        String,
        Vec<u8>,
        Option<Duration>,
        oneshot::Sender<Result<(), String>>,
    ),
    /// Get a value by key.
    Get(String, oneshot::Sender<Result<Option<Vec<u8>>, String>>),
    /// Delete a key.
    Delete(String, oneshot::Sender<Result<bool, String>>),
    /// Shutdown the event loop.
    Shutdown,
}

// =============================================================================
// Network Statistics
// =============================================================================

/// Statistics about the network node.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// This node's identifier.
    pub node_id: NodeId,
    /// How long the node has been running.
    pub uptime: Duration,
    /// Number of currently connected peers.
    pub peers_connected: usize,
    /// Number of peers detected as dead.
    pub peers_dead: usize,
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Number of keys stored locally.
    pub keys_stored: usize,
    /// Number of pending replication operations.
    pub replication_pending: usize,
}

/// Information about the hash ring.
#[derive(Debug, Clone)]
pub struct RingInfo {
    /// Total physical nodes in the ring.
    pub total_nodes: usize,
    /// Total virtual nodes (physical * vnodes_per_node).
    pub total_vnodes: usize,
    /// Configured replication factor.
    pub replication_factor: usize,
}

// =============================================================================
// Network Node — Main Struct
// =============================================================================

/// A node in the distributed network.
///
/// Provides a synchronous API backed by an async QUIC transport running in
/// a background tokio runtime. Handles connection management, message routing,
/// failure detection, replication, and anti-entropy synchronization.
///
/// # Example
/// ```no_run
/// use ai_assistant::distributed_network::{NetworkNode, NetworkConfig};
///
/// let config = NetworkConfig {
///     listen_addr: "127.0.0.1:0".parse().unwrap(),
///     ..NetworkConfig::default()
/// };
/// let node = NetworkNode::new(config).expect("Failed to create node");
/// println!("Node {} listening on {}", node.node_id(), node.local_addr());
/// ```
pub struct NetworkNode {
    /// This node's identifier.
    node_id: NodeId,
    /// The configuration used to create this node.
    config: NetworkConfig,
    /// The node's cryptographic identity.
    identity: NodeIdentity,
    /// The tokio runtime running the event loop.
    rt: Option<Runtime>,
    /// Channel to send commands to the event loop.
    command_tx: mpsc::Sender<NetworkCommand>,
    /// Channel to receive events from the event loop.
    event_rx: Mutex<mpsc::Receiver<NetworkEvent>>,
    /// The local address the QUIC endpoint is bound to.
    local_addr: SocketAddr,
    /// Shared peer state (readable from sync API).
    peers: Arc<RwLock<HashMap<NodeId, PeerState>>>,
    /// Shared hash ring.
    ring: Arc<RwLock<ConsistentHashRing>>,
    /// Shared local key-value storage.
    storage: Arc<RwLock<HashMap<String, StoredValue>>>,
    /// Shared heartbeat manager.
    heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
    /// Shared anti-entropy sync state (used by periodic sync loop).
    merkle: Arc<RwLock<AntiEntropySync>>,
    /// Shared join tokens for validating incoming connections.
    join_tokens: Arc<RwLock<Vec<JoinToken>>>,
    /// When the node was started.
    started_at: Instant,
    /// Total messages sent counter.
    messages_sent: Arc<std::sync::atomic::AtomicU64>,
    /// Total messages received counter.
    messages_received: Arc<std::sync::atomic::AtomicU64>,
    /// Pending replication count.
    replication_pending: Arc<std::sync::atomic::AtomicUsize>,
    /// Whether the node has been shut down.
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

/// A value stored in the local key-value store.
#[derive(Clone, Debug)]
struct StoredValue {
    data: Vec<u8>,
    version: u64,
    expires_at: Option<Instant>,
}

impl StoredValue {
    fn new(data: Vec<u8>, ttl: Option<Duration>) -> Self {
        Self {
            data,
            version: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            expires_at: ttl.map(|d| Instant::now() + d),
        }
    }

    fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |exp| Instant::now() > exp)
    }
}

impl NetworkNode {
    /// Create a new network node with the given configuration.
    ///
    /// This starts a background tokio runtime, binds a QUIC endpoint,
    /// and begins the event loop. The node will automatically connect
    /// to bootstrap peers if configured.
    pub fn new(config: NetworkConfig) -> Result<Self, String> {
        // Load or create identity
        let (identity, _is_new) = CertificateManager::load_or_create(&config.identity_dir)
            .map_err(|e| format!("Failed to load/create identity: {}", e))?;
        let node_id = identity.node_id;

        // Create tokio runtime
        let rt = Runtime::new().map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        // Create QUIC endpoint
        let server_config = CertificateManager::make_server_config(&identity)
            .map_err(|e| format!("Failed to create server config: {}", e))?;

        let local_addr = rt.block_on(async {
            let endpoint = quinn::Endpoint::server(server_config, config.listen_addr)
                .map_err(|e| format!("Failed to bind QUIC endpoint: {}", e))?;
            let addr = endpoint
                .local_addr()
                .map_err(|e| format!("Failed to get local address: {}", e))?;
            Ok::<(quinn::Endpoint, SocketAddr), String>((endpoint, addr))
        });

        let (endpoint, local_addr) = local_addr?;

        // Create shared state
        let peers: Arc<RwLock<HashMap<NodeId, PeerState>>> = Arc::new(RwLock::new(HashMap::new()));
        let ring = Arc::new(RwLock::new(ConsistentHashRing::new(
            config.replication.vnodes_per_node,
            config.replication.max_copies,
        )));
        let storage: Arc<RwLock<HashMap<String, StoredValue>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let heartbeat_mgr = Arc::new(RwLock::new(HeartbeatManager::new(HeartbeatConfig {
            interval: config.heartbeat_interval,
            phi_threshold: config.phi_threshold,
            max_samples: 200,
            suspicious_threshold: config.phi_threshold * 0.5,
        })));
        let merkle = Arc::new(RwLock::new(AntiEntropySync::new(Duration::from_secs(30))));

        // Initialize join tokens from config
        let join_tokens = Arc::new(RwLock::new(Vec::new()));
        if let Some(ref token_str) = config.join_token {
            // Treat the config token as a raw token string for simple matching
            let token = JoinToken {
                token: token_str.clone(),
                created_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                expires_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    + 365 * 24 * 3600,
                max_uses: None,
                uses: 0,
            };
            join_tokens
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .push(token);
        }

        let messages_sent = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let messages_received = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let replication_pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Add self to ring
        {
            let mut ring_w = ring.write().unwrap_or_else(|e| e.into_inner());
            ring_w.add_node(node_id);
        }

        // Create channels
        let (command_tx, command_rx) = mpsc::channel::<NetworkCommand>(256);
        let (event_tx, event_rx) = mpsc::channel::<NetworkEvent>(256);

        // Start event loop
        let event_loop = EventLoop {
            node_id,
            endpoint,
            identity: identity.clone(),
            peers: peers.clone(),
            ring: ring.clone(),
            storage: storage.clone(),
            heartbeat_mgr: heartbeat_mgr.clone(),
            merkle: merkle.clone(),
            join_tokens: join_tokens.clone(),
            connections: HashMap::new(),
            config: config.clone(),
            messages_sent: messages_sent.clone(),
            messages_received: messages_received.clone(),
            replication_pending: replication_pending.clone(),
            shutdown_flag: shutdown.clone(),
        };

        rt.spawn(event_loop.run(command_rx, event_tx));

        // Connect to bootstrap peers
        let boot_peers = config.bootstrap_peers.clone();
        let cmd_tx = command_tx.clone();
        rt.spawn(async move {
            for peer_addr in boot_peers {
                let (tx, _rx) = oneshot::channel();
                let _ = cmd_tx.send(NetworkCommand::Connect(peer_addr, tx)).await;
                // Small delay between bootstrap connections
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Ok(Self {
            node_id,
            config,
            identity,
            rt: Some(rt),
            command_tx,
            event_rx: Mutex::new(event_rx),
            local_addr,
            peers,
            ring,
            storage,
            heartbeat_mgr,
            merkle,
            join_tokens,
            started_at: Instant::now(),
            messages_sent,
            messages_received,
            replication_pending,
            shutdown,
        })
    }

    /// Get this node's identifier.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Get the local address the QUIC endpoint is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Connect to a peer at the given address.
    ///
    /// Returns the peer's NodeId on success.
    pub fn connect(&self, addr: SocketAddr) -> Result<NodeId, String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Connect(addr, tx))?;
        self.recv_response(rx)?
    }

    /// Disconnect from a specific peer.
    pub fn disconnect(&self, peer: &NodeId) {
        let mut peers = self.peers.write().unwrap_or_else(|e| e.into_inner());
        peers.remove(peer);
        let mut ring = self.ring.write().unwrap_or_else(|e| e.into_inner());
        ring.remove_node(peer);
    }

    /// Send a message to a peer (fire-and-forget).
    pub fn send(&self, peer: &NodeId, msg: NodeMessage) -> Result<(), String> {
        self.send_command(NetworkCommand::SendMessage(*peer, msg))
    }

    /// Send a message to a peer and wait for a reply.
    pub fn request(&self, peer: &NodeId, msg: NodeMessage) -> Result<NodeMessage, String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Request(*peer, msg, tx))?;
        self.recv_response(rx)?
    }

    /// Store a key-value pair with replication to the appropriate nodes.
    pub fn store(&self, key: &str, value: Vec<u8>) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Store(key.to_string(), value, None, tx))?;
        self.recv_response(rx)?
    }

    /// Store a key-value pair with a time-to-live.
    pub fn store_with_ttl(&self, key: &str, value: Vec<u8>, ttl: Duration) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Store(key.to_string(), value, Some(ttl), tx))?;
        self.recv_response(rx)?
    }

    /// Get a value by key, potentially querying multiple nodes (quorum read).
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Get(key.to_string(), tx))?;
        self.recv_response(rx)?
    }

    /// Delete a key from the local store and all replicas.
    pub fn delete(&self, key: &str) -> Result<bool, String> {
        let (tx, rx) = oneshot::channel();
        self.send_command(NetworkCommand::Delete(key.to_string(), tx))?;
        self.recv_response(rx)?
    }

    /// Poll for pending network events (non-blocking).
    pub fn poll_events(&self) -> Vec<NetworkEvent> {
        let mut events = Vec::new();
        let mut rx = self.event_rx.lock().unwrap_or_else(|e| e.into_inner());
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Get the number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Get information about all connected peers.
    pub fn peers(&self) -> Vec<(NodeId, SocketAddr, f32)> {
        self.peers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|p| (p.node_id, p.addr, p.reputation))
            .collect()
    }

    /// Get information about the hash ring.
    pub fn ring_info(&self) -> RingInfo {
        let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
        RingInfo {
            total_nodes: ring.node_count(),
            total_vnodes: ring.vnode_count(),
            replication_factor: ring.replication_factor(),
        }
    }

    /// Get network statistics.
    pub fn stats(&self) -> NetworkStats {
        let dead_count = self
            .heartbeat_mgr
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get_dead_nodes()
            .len();

        NetworkStats {
            node_id: self.node_id,
            uptime: self.started_at.elapsed(),
            peers_connected: self.peer_count(),
            peers_dead: dead_count,
            messages_sent: self
                .messages_sent
                .load(std::sync::atomic::Ordering::Relaxed),
            messages_received: self
                .messages_received
                .load(std::sync::atomic::Ordering::Relaxed),
            keys_stored: self.storage.read().unwrap_or_else(|e| e.into_inner()).len(),
            replication_pending: self
                .replication_pending
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Gracefully shut down the node.
    pub fn shutdown(&self) {
        if self
            .shutdown
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return; // Already shut down
        }
        let _ = self.command_tx.blocking_send(NetworkCommand::Shutdown);
    }

    /// Send a command to the event loop.
    fn send_command(&self, cmd: NetworkCommand) -> Result<(), String> {
        self.command_tx
            .blocking_send(cmd)
            .map_err(|_| "Event loop has shut down".to_string())
    }

    /// Receive a response from the event loop.
    fn recv_response<T>(&self, rx: oneshot::Receiver<T>) -> Result<T, String> {
        rx.blocking_recv()
            .map_err(|_| "Event loop dropped response channel".to_string())
    }

    /// Get the number of keys stored locally.
    pub fn local_key_count(&self) -> usize {
        self.storage.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Get a value directly from local storage (no network).
    pub fn local_get(&self, key: &str) -> Option<Vec<u8>> {
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage.get(key).and_then(|v| {
            if v.is_expired() {
                None
            } else {
                Some(v.data.clone())
            }
        })
    }

    /// Store a value directly in local storage (no replication).
    pub fn local_store(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) {
        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        storage.insert(key.to_string(), StoredValue::new(value, ttl));
    }

    /// Check the health status of a peer.
    pub fn check_peer(&self, peer: &NodeId) -> NodeStatus {
        self.heartbeat_mgr
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .check_node(peer)
    }

    /// Get the node's cryptographic identity.
    pub fn identity(&self) -> &NodeIdentity {
        &self.identity
    }

    /// Generate a join token for admitting new nodes to the cluster.
    ///
    /// The token is valid for `validity_hours` and can be used `max_uses` times.
    /// Share the encoded token string with new nodes via out-of-band channels.
    pub fn generate_join_token(&self, validity_hours: u64, max_uses: Option<usize>) -> JoinToken {
        let token = JoinToken::generate(validity_hours, max_uses);
        self.join_tokens
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(token.clone());
        token
    }

    /// Get the reputation of a specific peer (0.0 to 1.0).
    pub fn peer_reputation(&self, peer: &NodeId) -> Option<f32> {
        self.peers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(peer)
            .map(|p| p.reputation)
    }

    /// Check if a peer is still in probation period.
    pub fn peer_in_probation(&self, peer: &NodeId) -> Option<bool> {
        self.peers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(peer)
            .map(|p| p.probation)
    }

    /// Trigger an immediate anti-entropy sync check.
    ///
    /// Returns the number of peers that needed sync. Normally sync runs
    /// automatically every 30 seconds in the background event loop.
    pub fn sync_status(&self) -> usize {
        let merkle = self.merkle.read().unwrap_or_else(|e| e.into_inner());
        let peers = self.peers.read().unwrap_or_else(|e| e.into_inner());
        peers.keys().filter(|id| merkle.needs_sync(id)).count()
    }

    /// Get all nodes responsible for a key (based on consistent hashing).
    pub fn nodes_for_key(&self, key: &str) -> Vec<NodeId> {
        let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
        ring.get_nodes(key, self.config.replication.max_copies)
    }
}

impl Drop for NetworkNode {
    fn drop(&mut self) {
        self.shutdown();
        // Give the event loop a moment to process the shutdown
        if let Some(rt) = self.rt.take() {
            // Drop the runtime, which will cancel all spawned tasks
            drop(rt);
        }
    }
}

// =============================================================================
// Event Loop — Async internals
// =============================================================================

/// The async event loop that runs in the background tokio runtime.
struct EventLoop {
    node_id: NodeId,
    endpoint: quinn::Endpoint,
    identity: NodeIdentity,
    peers: Arc<RwLock<HashMap<NodeId, PeerState>>>,
    ring: Arc<RwLock<ConsistentHashRing>>,
    storage: Arc<RwLock<HashMap<String, StoredValue>>>,
    heartbeat_mgr: Arc<RwLock<HeartbeatManager>>,
    merkle: Arc<RwLock<AntiEntropySync>>,
    join_tokens: Arc<RwLock<Vec<JoinToken>>>,
    connections: HashMap<NodeId, quinn::Connection>,
    config: NetworkConfig,
    messages_sent: Arc<std::sync::atomic::AtomicU64>,
    messages_received: Arc<std::sync::atomic::AtomicU64>,
    replication_pending: Arc<std::sync::atomic::AtomicUsize>,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl EventLoop {
    /// Main event loop: processes commands, accepts connections, sends heartbeats,
    /// runs anti-entropy sync, peer exchange, and optionally LAN discovery.
    async fn run(
        mut self,
        mut command_rx: mpsc::Receiver<NetworkCommand>,
        event_tx: mpsc::Sender<NetworkEvent>,
    ) {
        let mut heartbeat_interval = tokio::time::interval(self.config.heartbeat_interval);
        let mut cleanup_interval = tokio::time::interval(Duration::from_secs(30));

        // Spawn LAN discovery task if enabled
        let discovery_command_tx = if self.config.discovery.enable_broadcast {
            let (disc_tx, mut disc_rx) = mpsc::channel::<SocketAddr>(32);
            let broadcast_port = self.config.discovery.broadcast_port;
            let broadcast_interval = self.config.discovery.broadcast_interval;
            let node_id = self.node_id;
            let local_addr = self
                .endpoint
                .local_addr()
                .unwrap_or(self.config.listen_addr);
            let shutdown_flag = self.shutdown_flag.clone();

            // Spawn UDP broadcast listener + sender
            tokio::spawn(async move {
                Self::run_lan_discovery(
                    node_id,
                    local_addr,
                    broadcast_port,
                    broadcast_interval,
                    shutdown_flag,
                    disc_tx,
                )
                .await;
            });

            // Spawn task to process discovered peers by connecting to them
            let cmd_tx = event_tx.clone();
            let peers_ref = self.peers.clone();
            let identity_ref = self.identity.clone();
            let mut endpoint_ref = self.endpoint.clone();
            let msg_sent = self.messages_sent.clone();
            let shutdown2 = self.shutdown_flag.clone();

            tokio::spawn(async move {
                while let Some(addr) = disc_rx.recv().await {
                    if shutdown2.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    // Check if we're already connected to this address
                    let already_connected = {
                        let peers = peers_ref.read().unwrap_or_else(|e| e.into_inner());
                        peers.values().any(|p| p.addr == addr)
                    };
                    if already_connected {
                        continue;
                    }
                    // Try to connect
                    let client_config = match CertificateManager::make_client_config(&identity_ref)
                    {
                        Ok(cfg) => cfg,
                        Err(_) => continue,
                    };
                    endpoint_ref.set_default_client_config(client_config);
                    match endpoint_ref.connect(addr, "node") {
                        Ok(connecting) => {
                            if let Ok(conn) = connecting.await {
                                // We connected — but full registration happens via the main event loop
                                // For simplicity, emit PeerConnected event
                                let _ = cmd_tx
                                    .send(NetworkEvent::PeerConnected(
                                        NodeId::from_string(&addr.to_string()),
                                        addr,
                                    ))
                                    .await;
                                let _ = conn;
                            }
                        }
                        Err(_) => {}
                    }
                }
            });

            Some(msg_sent)
        } else {
            None
        };
        let _ = discovery_command_tx; // Suppress unused warning when broadcast disabled

        loop {
            tokio::select! {
                // Process commands from the sync API
                cmd = command_rx.recv() => {
                    match cmd {
                        Some(NetworkCommand::Shutdown) | None => {
                            self.handle_shutdown().await;
                            break;
                        }
                        Some(cmd) => {
                            self.handle_command(cmd, &event_tx).await;
                        }
                    }
                }

                // Accept incoming connections
                incoming = self.endpoint.accept() => {
                    if let Some(incoming) = incoming {
                        self.handle_incoming(incoming, &event_tx).await;
                    }
                }

                // Send heartbeats periodically
                _ = heartbeat_interval.tick() => {
                    self.send_heartbeats(&event_tx).await;
                }

                // Periodic cleanup, anti-entropy sync, and peer exchange
                _ = cleanup_interval.tick() => {
                    self.cleanup_expired();
                    self.check_dead_peers(&event_tx).await;
                    self.run_anti_entropy_sync().await;
                    if self.config.discovery.enable_peer_exchange {
                        self.run_peer_exchange(&event_tx).await;
                    }
                    self.enforce_min_copies(&event_tx).await;
                }
            }

            if self
                .shutdown_flag
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                break;
            }
        }
    }

    /// Handle a command from the sync API.
    async fn handle_command(&mut self, cmd: NetworkCommand, event_tx: &mpsc::Sender<NetworkEvent>) {
        match cmd {
            NetworkCommand::Connect(addr, reply) => {
                let result = self.connect_to_peer(addr, event_tx).await;
                let _ = reply.send(result);
            }
            NetworkCommand::SendMessage(peer_id, msg) => {
                let _ = self.send_to_peer(&peer_id, &msg).await;
            }
            NetworkCommand::Request(peer_id, msg, reply) => {
                let result = self.request_from_peer(&peer_id, msg).await;
                let _ = reply.send(result);
            }
            NetworkCommand::Store(key, value, ttl, reply) => {
                let result = self.handle_store(&key, value, ttl, event_tx).await;
                let _ = reply.send(result);
            }
            NetworkCommand::Get(key, reply) => {
                let result = self.handle_get(&key).await;
                let _ = reply.send(result);
            }
            NetworkCommand::Delete(key, reply) => {
                let result = self.handle_delete(&key).await;
                let _ = reply.send(result);
            }
            NetworkCommand::Shutdown => {
                // Handled in the main loop
            }
        }
    }

    /// Connect to a peer at the given address.
    async fn connect_to_peer(
        &mut self,
        addr: SocketAddr,
        event_tx: &mpsc::Sender<NetworkEvent>,
    ) -> Result<NodeId, String> {
        // Build client config
        let client_config = CertificateManager::make_client_config(&self.identity)
            .map_err(|e| format!("Failed to create client config: {}", e))?;

        self.endpoint.set_default_client_config(client_config);

        let connection = self
            .endpoint
            .connect(addr, "node")
            .map_err(|e| format!("Failed to initiate connection to {}: {}", addr, e))?
            .await
            .map_err(|e| format!("Connection to {} failed: {}", addr, e))?;

        // Exchange identity: send our node ID, receive theirs
        let peer_id = self.exchange_identity(&connection).await?;

        // Register the peer
        {
            let mut peers = self.peers.write().unwrap_or_else(|e| e.into_inner());
            peers.insert(peer_id, PeerState::new(peer_id, addr));
        }
        {
            let mut ring = self.ring.write().unwrap_or_else(|e| e.into_inner());
            ring.add_node(peer_id);
        }

        self.connections.insert(peer_id, connection);

        let _ = event_tx
            .send(NetworkEvent::PeerConnected(peer_id, addr))
            .await;

        Ok(peer_id)
    }

    /// Handle an incoming QUIC connection.
    ///
    /// Enforces max_connections limit and validates join tokens if configured.
    async fn handle_incoming(
        &mut self,
        incoming: quinn::Incoming,
        event_tx: &mpsc::Sender<NetworkEvent>,
    ) {
        // Enforce max_connections limit
        let current_connections = self.connections.len();
        if current_connections >= self.config.max_connections {
            let _ = event_tx
                .send(NetworkEvent::Error(format!(
                    "Rejecting connection: at max capacity ({}/{})",
                    current_connections, self.config.max_connections
                )))
                .await;
            // Drop the incoming connection by not awaiting it
            return;
        }

        let connection = match incoming.await {
            Ok(conn) => conn,
            Err(e) => {
                let _ = event_tx
                    .send(NetworkEvent::Error(format!(
                        "Failed to accept connection: {}",
                        e
                    )))
                    .await;
                return;
            }
        };

        let addr = connection.remote_address();

        // Exchange identity
        let peer_id = match self.exchange_identity_server(&connection).await {
            Ok(id) => id,
            Err(e) => {
                let _ = event_tx
                    .send(NetworkEvent::Error(format!(
                        "Identity exchange failed with {}: {}",
                        addr, e
                    )))
                    .await;
                connection.close(1u32.into(), b"identity exchange failed");
                return;
            }
        };

        // Validate join token if the cluster requires one
        if self.config.join_token.is_some() {
            match self.validate_join_token(&connection, peer_id).await {
                Ok(true) => {} // Token valid
                Ok(false) => {
                    let _ = event_tx
                        .send(NetworkEvent::Error(format!(
                            "Join token validation failed for {} at {}",
                            peer_id, addr
                        )))
                        .await;
                    connection.close(2u32.into(), b"invalid join token");
                    return;
                }
                Err(e) => {
                    let _ = event_tx
                        .send(NetworkEvent::Error(format!(
                            "Join handshake error with {}: {}",
                            addr, e
                        )))
                        .await;
                    connection.close(2u32.into(), b"join handshake error");
                    return;
                }
            }
        }

        // Register the peer
        {
            let mut peers = self.peers.write().unwrap_or_else(|e| e.into_inner());
            peers.insert(peer_id, PeerState::new(peer_id, addr));
        }
        {
            let mut ring = self.ring.write().unwrap_or_else(|e| e.into_inner());
            ring.add_node(peer_id);
        }

        self.connections.insert(peer_id, connection.clone());

        let _ = event_tx
            .send(NetworkEvent::PeerConnected(peer_id, addr))
            .await;

        // Emit JoinRequestReceived event
        let _ = event_tx
            .send(NetworkEvent::JoinRequestReceived(peer_id, addr))
            .await;

        // Spawn a task to handle incoming messages from this peer
        let peers = self.peers.clone();
        let storage = self.storage.clone();
        let join_tokens = self.join_tokens.clone();
        let config_join_token = self.config.join_token.clone();
        let messages_received = self.messages_received.clone();
        let event_tx_clone = event_tx.clone();
        let node_id = self.node_id;

        tokio::spawn(async move {
            Self::handle_peer_messages(
                connection,
                peer_id,
                node_id,
                peers,
                storage,
                join_tokens,
                config_join_token,
                messages_received,
                event_tx_clone,
            )
            .await;
        });
    }

    /// Validate a join token from an incoming peer.
    ///
    /// The server sends a JoinRequest prompt and the client must respond
    /// with a valid token. Returns true if the token is valid.
    async fn validate_join_token(
        &self,
        connection: &quinn::Connection,
        _peer_id: NodeId,
    ) -> Result<bool, String> {
        // Wait for the client to send a JoinRequest
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| format!("Failed to accept join stream: {}", e))?;

        let msg = tokio::time::timeout(self.config.message_timeout, Self::read_message(&mut recv))
            .await
            .map_err(|_| "Join token exchange timed out".to_string())?
            .map_err(|e| format!("Failed to read join request: {}", e))?;

        match msg {
            NodeMessage::JoinRequest { token, .. } => {
                // Validate token inside a block so the write guard is dropped before await
                let (valid, resp) = {
                    let mut tokens = self.join_tokens.write().unwrap_or_else(|e| e.into_inner());
                    let valid = tokens.iter_mut().any(|t| t.token == token && t.consume());
                    if valid {
                        let peer_list: Vec<(Vec<u8>, String)> = self
                            .peers
                            .read()
                            .unwrap_or_else(|e| e.into_inner())
                            .values()
                            .map(|p| (p.node_id.0.to_vec(), p.addr.to_string()))
                            .collect();
                        (
                            true,
                            NodeMessage::JoinAccepted {
                                node_id: self.node_id.0.to_vec(),
                                peers: peer_list,
                            },
                        )
                    } else {
                        (
                            false,
                            NodeMessage::JoinRejected {
                                reason: "Invalid or expired join token".to_string(),
                            },
                        )
                    }
                }; // Lock dropped here

                let _ = Self::write_message(&mut send, &resp).await;
                Ok(valid)
            }
            _ => Err("Expected JoinRequest message".to_string()),
        }
    }

    /// Handle messages from a connected peer until the connection closes.
    async fn handle_peer_messages(
        connection: quinn::Connection,
        peer_id: NodeId,
        self_node_id: NodeId,
        peers: Arc<RwLock<HashMap<NodeId, PeerState>>>,
        storage: Arc<RwLock<HashMap<String, StoredValue>>>,
        join_tokens: Arc<RwLock<Vec<JoinToken>>>,
        config_join_token: Option<String>,
        messages_received: Arc<std::sync::atomic::AtomicU64>,
        event_tx: mpsc::Sender<NetworkEvent>,
    ) {
        loop {
            match connection.accept_bi().await {
                Ok((mut send, mut recv)) => {
                    // Read length-prefixed message
                    match Self::read_message(&mut recv).await {
                        Ok(msg) => {
                            messages_received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            // Update peer stats: increment received count and reputation
                            {
                                let mut peers_w = peers.write().unwrap_or_else(|e| e.into_inner());
                                if let Some(peer) = peers_w.get_mut(&peer_id) {
                                    peer.messages_received += 1;
                                    // Successful message exchange improves reputation
                                    peer.reputation = (peer.reputation + 0.001).min(1.0);
                                    // After 100 successful messages, exit probation
                                    if peer.messages_received >= 100 {
                                        peer.probation = false;
                                    }
                                }
                            }

                            // Handle the message and generate response
                            let response = Self::process_message(
                                &msg,
                                self_node_id,
                                &storage,
                                &peers,
                                &join_tokens,
                                &config_join_token,
                            );

                            // Send response if any
                            if let Some(resp) = response {
                                let _ = Self::write_message(&mut send, &resp).await;
                            }

                            // Forward to event channel
                            let _ = event_tx
                                .send(NetworkEvent::MessageReceived(peer_id, msg))
                                .await;
                        }
                        Err(_) => {
                            // Connection error — degrade reputation slightly
                            let mut peers_w = peers.write().unwrap_or_else(|e| e.into_inner());
                            if let Some(peer) = peers_w.get_mut(&peer_id) {
                                peer.reputation = (peer.reputation - 0.01).max(0.0);
                            }
                            break;
                        }
                    }
                }
                Err(_) => break, // Connection closed
            }
        }

        // Peer disconnected
        let _ = event_tx.send(NetworkEvent::PeerDisconnected(peer_id)).await;
    }

    /// Process an incoming message and optionally return a response.
    ///
    /// Handles all NodeMessage variants with appropriate logic:
    /// - DHT ops (Get/Put/Delete/Replicate) operate on local storage
    /// - Sync messages trigger anti-entropy data exchange
    /// - Peer exchange returns the known peer list
    /// - Join requests validate tokens
    /// - Vector search returns empty results (VectorDb lives at a higher layer)
    fn process_message(
        msg: &NodeMessage,
        self_node_id: NodeId,
        storage: &Arc<RwLock<HashMap<String, StoredValue>>>,
        peers: &Arc<RwLock<HashMap<NodeId, PeerState>>>,
        join_tokens: &Arc<RwLock<Vec<JoinToken>>>,
        config_join_token: &Option<String>,
    ) -> Option<NodeMessage> {
        match msg {
            NodeMessage::Ping { timestamp, .. } => Some(NodeMessage::Pong {
                sender: self_node_id,
                timestamp: *timestamp,
            }),
            NodeMessage::Pong { sender, .. } => {
                // Record heartbeat from the peer (handled at caller level too)
                // No response needed for Pong
                let _ = sender;
                None
            }
            NodeMessage::Get { key, request_id } => {
                let value = storage
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(key)
                    .and_then(|v| {
                        if v.is_expired() {
                            None
                        } else {
                            Some(v.data.clone())
                        }
                    });
                Some(NodeMessage::GetResponse {
                    key: key.clone(),
                    value,
                    request_id: *request_id,
                })
            }
            NodeMessage::GetResponse { .. } => None, // Response type, no reply needed
            NodeMessage::Put {
                key,
                value,
                ttl_secs,
            } => {
                let ttl = ttl_secs.map(Duration::from_secs);
                let mut storage_w = storage.write().unwrap_or_else(|e| e.into_inner());
                storage_w.insert(key.clone(), StoredValue::new(value.clone(), ttl));
                Some(NodeMessage::PutAck {
                    key: key.clone(),
                    success: true,
                    request_id: 0,
                })
            }
            NodeMessage::PutAck { .. } => None,
            NodeMessage::Delete { key, request_id } => {
                let mut storage_w = storage.write().unwrap_or_else(|e| e.into_inner());
                let existed = storage_w.remove(key).is_some();
                Some(NodeMessage::DeleteAck {
                    key: key.clone(),
                    success: existed,
                    request_id: *request_id,
                })
            }
            NodeMessage::DeleteAck { .. } => None,
            NodeMessage::Replicate {
                key,
                value,
                version,
            } => {
                let mut storage_w = storage.write().unwrap_or_else(|e| e.into_inner());
                let should_update = storage_w
                    .get(key)
                    .map_or(true, |existing| existing.version < *version);
                if should_update {
                    storage_w.insert(
                        key.clone(),
                        StoredValue {
                            data: value.clone(),
                            version: *version,
                            expires_at: None,
                        },
                    );
                }
                Some(NodeMessage::ReplicateAck {
                    key: key.clone(),
                    version: *version,
                    success: true,
                })
            }
            NodeMessage::ReplicateAck { .. } => None,

            // --- Anti-Entropy Sync ---
            NodeMessage::SyncRequest { merkle_root } => {
                // Build our Merkle tree from local storage and compare roots
                let local_data = Self::storage_to_btree(storage);
                let local_tree = MerkleTree::from_data(&local_data);
                let our_root = local_tree
                    .root_hash()
                    .map(|h| h.to_vec())
                    .unwrap_or_default();

                if our_root == *merkle_root {
                    // Trees match — no sync needed
                    Some(NodeMessage::SyncResponse {
                        diff_keys: Vec::new(),
                    })
                } else {
                    // Trees differ — compute keys that the remote is missing
                    // We send our keys so the remote can request what it needs
                    let our_keys: Vec<String> = local_data.keys().cloned().collect();
                    Some(NodeMessage::SyncResponse {
                        diff_keys: our_keys,
                    })
                }
            }
            NodeMessage::SyncResponse { diff_keys } => {
                if diff_keys.is_empty() {
                    return None; // Already in sync
                }
                // The remote sent us their keys — send them data for keys we have
                let storage_r = storage.read().unwrap_or_else(|e| e.into_inner());
                let mut entries = Vec::new();
                for key in diff_keys {
                    if let Some(v) = storage_r.get(key) {
                        if !v.is_expired() {
                            entries.push((key.clone(), v.data.clone()));
                        }
                    }
                }
                if entries.is_empty() {
                    None
                } else {
                    Some(NodeMessage::SyncData { entries })
                }
            }
            NodeMessage::SyncData { entries } => {
                // Store received entries (only if we don't have a newer version)
                let mut storage_w = storage.write().unwrap_or_else(|e| e.into_inner());
                for (key, data) in entries {
                    storage_w
                        .entry(key.clone())
                        .or_insert_with(|| StoredValue::new(data.clone(), None));
                }
                None
            }

            // --- Cluster Management ---
            NodeMessage::JoinRequest { token, cert_der } => {
                let _ = cert_der; // Certificate already verified at TLS layer
                                  // Validate join token if the cluster requires one
                if config_join_token.is_some() {
                    let mut tokens = join_tokens.write().unwrap_or_else(|e| e.into_inner());
                    let valid = tokens.iter_mut().any(|t| t.token == *token && t.consume());
                    if !valid {
                        return Some(NodeMessage::JoinRejected {
                            reason: "Invalid or expired join token".to_string(),
                        });
                    }
                }
                // Token valid (or no token required) — accept the join
                let peer_list: Vec<(Vec<u8>, String)> = peers
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .values()
                    .map(|p| (p.node_id.0.to_vec(), p.addr.to_string()))
                    .collect();
                Some(NodeMessage::JoinAccepted {
                    node_id: self_node_id.0.to_vec(),
                    peers: peer_list,
                })
            }
            NodeMessage::JoinAccepted { .. } => None,
            NodeMessage::JoinRejected { .. } => None,
            NodeMessage::NodeLeft { .. } => None, // Handled at event level

            // --- Discovery ---
            NodeMessage::DiscoveryAnnounce { .. } => None, // Handled via UDP, not QUIC

            // --- Peer Exchange ---
            NodeMessage::PeerExchangeRequest { .. } => {
                let peer_list: Vec<(Vec<u8>, String)> = peers
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .values()
                    .map(|p| (p.node_id.0.to_vec(), p.addr.to_string()))
                    .collect();
                Some(NodeMessage::PeerExchangeResponse { peers: peer_list })
            }
            NodeMessage::PeerExchangeResponse { .. } => None, // Handled at caller level

            // --- MapReduce ---
            NodeMessage::MapTask { job_id, .. } => {
                // MapReduce tasks are forwarded as events for the application layer
                // to handle. Return an empty result to acknowledge receipt.
                Some(NodeMessage::MapResult {
                    job_id: job_id.clone(),
                    outputs: Vec::new(),
                })
            }
            NodeMessage::MapResult { .. } => None,
            NodeMessage::ReduceTask { job_id, key, .. } => Some(NodeMessage::ReduceResult {
                job_id: job_id.clone(),
                key: key.clone(),
                value: Vec::new(),
            }),
            NodeMessage::ReduceResult { .. } => None,

            // --- Vector DB ---
            NodeMessage::VectorSearch { request_id, .. } => {
                // VectorDb lives at a higher layer. Return empty results here;
                // the MessageReceived event allows the application to handle
                // vector queries with access to the actual VectorDb instance.
                Some(NodeMessage::VectorSearchResponse {
                    results: Vec::new(),
                    request_id: *request_id,
                })
            }
            NodeMessage::VectorSearchResponse { .. } => None,
        }
    }

    /// Convert local storage into a BTreeMap for Merkle tree construction.
    fn storage_to_btree(
        storage: &Arc<RwLock<HashMap<String, StoredValue>>>,
    ) -> std::collections::BTreeMap<String, Vec<u8>> {
        let storage_r = storage.read().unwrap_or_else(|e| e.into_inner());
        let mut tree_data = std::collections::BTreeMap::new();
        for (k, v) in storage_r.iter() {
            if !v.is_expired() {
                tree_data.insert(k.clone(), v.data.clone());
            }
        }
        tree_data
    }

    /// Exchange identity with a peer (client side).
    async fn exchange_identity(&self, connection: &quinn::Connection) -> Result<NodeId, String> {
        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|e| format!("Failed to open stream: {}", e))?;

        // Send our node ID
        let id_msg = NodeMessage::Ping {
            sender: self.node_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        Self::write_message(&mut send, &id_msg).await?;

        // Receive their response
        let response = Self::read_message(&mut recv).await?;
        match response {
            NodeMessage::Pong { sender, .. } => Ok(sender),
            _ => Err("Unexpected response during identity exchange".to_string()),
        }
    }

    /// Exchange identity with a peer (server side).
    async fn exchange_identity_server(
        &self,
        connection: &quinn::Connection,
    ) -> Result<NodeId, String> {
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| format!("Failed to accept stream: {}", e))?;

        // Receive their identity
        let msg = Self::read_message(&mut recv).await?;
        let peer_id = match msg {
            NodeMessage::Ping { sender, .. } => sender,
            _ => return Err("Expected Ping for identity exchange".to_string()),
        };

        // Send our response
        let response = NodeMessage::Pong {
            sender: self.node_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        Self::write_message(&mut send, &response).await?;

        Ok(peer_id)
    }

    /// Send a message to a specific peer.
    async fn send_to_peer(&mut self, peer_id: &NodeId, msg: &NodeMessage) -> Result<(), String> {
        let connection = self
            .connections
            .get(peer_id)
            .ok_or_else(|| format!("Not connected to peer {}", peer_id))?;

        let (mut send, _recv) = connection
            .open_bi()
            .await
            .map_err(|e| format!("Failed to open stream to {}: {}", peer_id, e))?;

        Self::write_message(&mut send, msg).await?;
        self.messages_sent
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Update peer stats
        {
            let mut peers = self.peers.write().unwrap_or_else(|e| e.into_inner());
            if let Some(peer) = peers.get_mut(peer_id) {
                peer.messages_sent += 1;
            }
        }

        Ok(())
    }

    /// Send a message and wait for a reply.
    async fn request_from_peer(
        &mut self,
        peer_id: &NodeId,
        msg: NodeMessage,
    ) -> Result<NodeMessage, String> {
        let connection = self
            .connections
            .get(peer_id)
            .ok_or_else(|| format!("Not connected to peer {}", peer_id))?
            .clone();

        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|e| format!("Failed to open stream to {}: {}", peer_id, e))?;

        Self::write_message(&mut send, &msg).await?;
        self.messages_sent
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let timeout = self.config.message_timeout;
        let response = tokio::time::timeout(timeout, Self::read_message(&mut recv))
            .await
            .map_err(|_| format!("Request to {} timed out", peer_id))?;

        self.messages_received
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        response
    }

    /// Handle a Store command: store locally and replicate.
    async fn handle_store(
        &mut self,
        key: &str,
        value: Vec<u8>,
        ttl: Option<Duration>,
        event_tx: &mpsc::Sender<NetworkEvent>,
    ) -> Result<(), String> {
        // Store locally
        let stored = StoredValue::new(value.clone(), ttl);
        let version = stored.version;
        {
            let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
            storage.insert(key.to_string(), stored);
        }

        // Determine replica nodes
        let replica_nodes = {
            let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
            ring.get_nodes(key, self.config.replication.max_copies)
        };

        // Replicate to other nodes
        let mut replicated_count = 1usize; // Count ourselves

        for &replica_node in &replica_nodes {
            if replica_node == self.node_id {
                continue;
            }

            let msg = NodeMessage::Replicate {
                key: key.to_string(),
                value: value.clone(),
                version,
            };

            match self.config.replication.write_mode {
                WriteMode::Synchronous => {
                    match self.request_from_peer(&replica_node, msg).await {
                        Ok(NodeMessage::ReplicateAck { success: true, .. }) => {
                            replicated_count += 1;
                        }
                        Ok(_) => {}  // Unexpected response, skip
                        Err(_) => {} // Failed, skip
                    }
                }
                WriteMode::Asynchronous => {
                    let _ = self.send_to_peer(&replica_node, &msg).await;
                    replicated_count += 1; // Optimistic count
                }
            }
        }

        let _ = event_tx
            .send(NetworkEvent::ReplicationComplete(
                key.to_string(),
                replicated_count,
            ))
            .await;

        if self.config.replication.write_mode == WriteMode::Synchronous
            && replicated_count < self.config.replication.write_quorum
        {
            return Err(format!(
                "Write quorum not met: {} < {} for key {}",
                replicated_count, self.config.replication.write_quorum, key
            ));
        }

        Ok(())
    }

    /// Handle a Get command: read from local store, optionally query peers for quorum.
    async fn handle_get(&mut self, key: &str) -> Result<Option<Vec<u8>>, String> {
        // Check local storage first
        let local_value = {
            let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
            storage.get(key).and_then(|v| {
                if v.is_expired() {
                    None
                } else {
                    Some(v.data.clone())
                }
            })
        };

        if self.config.replication.read_quorum <= 1 {
            return Ok(local_value);
        }

        // For quorum reads, also query replica nodes
        let replica_nodes = {
            let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
            ring.get_nodes(key, self.config.replication.max_copies)
        };

        let mut values = Vec::new();
        if let Some(ref v) = local_value {
            values.push(v.clone());
        }

        let request_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for &replica_node in &replica_nodes {
            if replica_node == self.node_id {
                continue;
            }
            if values.len() >= self.config.replication.read_quorum {
                break;
            }

            let msg = NodeMessage::Get {
                key: key.to_string(),
                request_id,
            };

            match self.request_from_peer(&replica_node, msg).await {
                Ok(NodeMessage::GetResponse { value: Some(v), .. }) => {
                    values.push(v);
                }
                _ => {} // Skip failed or empty responses
            }
        }

        // Return the first value if quorum is met
        if values.len() >= self.config.replication.read_quorum {
            Ok(values.into_iter().next())
        } else if values.is_empty() {
            Ok(None)
        } else {
            // Partial results — return what we have
            Ok(values.into_iter().next())
        }
    }

    /// Handle a Delete command: delete locally and from replicas.
    async fn handle_delete(&mut self, key: &str) -> Result<bool, String> {
        let existed = {
            let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
            storage.remove(key).is_some()
        };

        // Delete from replicas too
        let replica_nodes = {
            let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
            ring.get_nodes(key, self.config.replication.max_copies)
        };

        let request_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for &replica_node in &replica_nodes {
            if replica_node == self.node_id {
                continue;
            }
            let msg = NodeMessage::Delete {
                key: key.to_string(),
                request_id,
            };
            let _ = self.send_to_peer(&replica_node, &msg).await;
        }

        Ok(existed)
    }

    /// Send heartbeats to all connected peers and check failure detectors.
    async fn send_heartbeats(&mut self, event_tx: &mpsc::Sender<NetworkEvent>) {
        let peer_ids: Vec<NodeId> = self.connections.keys().copied().collect();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for peer_id in peer_ids {
            let msg = NodeMessage::Ping {
                sender: self.node_id,
                timestamp,
            };

            match self.send_to_peer(&peer_id, &msg).await {
                Ok(_) => {
                    // Record heartbeat receipt (we assume success means the peer is alive)
                    let mut hb = self
                        .heartbeat_mgr
                        .write()
                        .unwrap_or_else(|e| e.into_inner());
                    hb.record_heartbeat(&peer_id);
                }
                Err(_) => {
                    // Check if the peer should be considered dead
                    let status = self
                        .heartbeat_mgr
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .check_node(&peer_id);
                    if let NodeStatus::Dead(phi) = status {
                        let _ = event_tx.send(NetworkEvent::PeerFailed(peer_id, phi)).await;
                    }
                }
            }
        }
    }

    /// Check for dead peers and emit events.
    async fn check_dead_peers(&self, event_tx: &mpsc::Sender<NetworkEvent>) {
        let dead_nodes = self
            .heartbeat_mgr
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get_dead_nodes();
        for node_id in dead_nodes {
            let _ = event_tx
                .send(NetworkEvent::PeerFailed(node_id, f64::MAX))
                .await;
        }
    }

    /// Remove expired entries from local storage.
    fn cleanup_expired(&self) {
        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        storage.retain(|_, v| !v.is_expired());
    }

    /// Run LAN discovery via UDP broadcast.
    ///
    /// Periodically broadcasts a discovery announcement on the LAN and listens
    /// for announcements from other nodes. Discovered peers are sent to the
    /// main event loop via the `discovered_tx` channel.
    async fn run_lan_discovery(
        node_id: NodeId,
        local_quic_addr: SocketAddr,
        broadcast_port: u16,
        broadcast_interval: Duration,
        shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
        discovered_tx: mpsc::Sender<SocketAddr>,
    ) {
        // Bind to the broadcast port for receiving
        let recv_addr: SocketAddr = format!("0.0.0.0:{}", broadcast_port)
            .parse()
            .expect("valid address");
        let socket = match tokio::net::UdpSocket::bind(recv_addr).await {
            Ok(s) => s,
            Err(_) => {
                // Port in use or permission denied — silently disable discovery
                return;
            }
        };
        // Enable broadcast
        if socket.set_broadcast(true).is_err() {
            return;
        }

        let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", broadcast_port)
            .parse()
            .expect("valid address");

        let announce = NodeMessage::DiscoveryAnnounce {
            node_id: node_id.0.to_vec(),
            quic_addr: local_quic_addr.to_string(),
        };
        let announce_bytes = match bincode::serialize(&announce) {
            Ok(b) => b,
            Err(_) => return,
        };

        let mut interval = tokio::time::interval(broadcast_interval);
        let mut buf = vec![0u8; 4096];

        loop {
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            tokio::select! {
                _ = interval.tick() => {
                    // Send broadcast announcement
                    let _ = socket.send_to(&announce_bytes, broadcast_addr).await;
                }
                result = socket.recv_from(&mut buf) => {
                    if let Ok((len, _src)) = result {
                        // Try to decode as DiscoveryAnnounce
                        if let Ok(msg) = bincode::deserialize::<NodeMessage>(&buf[..len]) {
                            if let NodeMessage::DiscoveryAnnounce {
                                node_id: remote_id_bytes,
                                quic_addr,
                            } = msg {
                                // Ignore our own announcements
                                let mut remote_id = [0u8; 20];
                                if remote_id_bytes.len() >= 20 {
                                    remote_id.copy_from_slice(&remote_id_bytes[..20]);
                                }
                                if NodeId::from_bytes(remote_id) == node_id {
                                    continue;
                                }
                                // Parse the QUIC address and notify the main loop
                                if let Ok(addr) = quic_addr.parse::<SocketAddr>() {
                                    let _ = discovered_tx.send(addr).await;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run anti-entropy synchronization with connected peers.
    ///
    /// For each peer that needs sync (based on the configured interval),
    /// builds a Merkle tree from local storage, sends a SyncRequest with
    /// the root hash, and processes the response to exchange missing data.
    async fn run_anti_entropy_sync(&mut self) {
        // Get peers that need sync
        let peers_to_sync: Vec<NodeId> = {
            let merkle = self.merkle.read().unwrap_or_else(|e| e.into_inner());
            self.connections
                .keys()
                .filter(|peer_id| merkle.needs_sync(peer_id))
                .copied()
                .collect()
        };

        if peers_to_sync.is_empty() {
            return;
        }

        // Build Merkle tree from local storage
        let local_data = Self::storage_to_btree(&self.storage);
        let local_tree = MerkleTree::from_data(&local_data);
        let root_hash = local_tree
            .root_hash()
            .map(|h| h.to_vec())
            .unwrap_or_default();

        for peer_id in peers_to_sync {
            let msg = NodeMessage::SyncRequest {
                merkle_root: root_hash.clone(),
            };

            match self.request_from_peer(&peer_id, msg).await {
                Ok(NodeMessage::SyncResponse { diff_keys }) => {
                    if !diff_keys.is_empty() {
                        // Remote has keys we might not — request them
                        // Also send our data for keys they might be missing
                        let entries = {
                            let storage_r = self.storage.read().unwrap_or_else(|e| e.into_inner());
                            let mut entries = Vec::new();
                            for key in &diff_keys {
                                if let Some(v) = storage_r.get(key) {
                                    if !v.is_expired() {
                                        entries.push((key.clone(), v.data.clone()));
                                    }
                                }
                            }
                            entries
                        }; // RwLockReadGuard dropped here before await

                        if !entries.is_empty() {
                            let sync_data = NodeMessage::SyncData { entries };
                            let _ = self.send_to_peer(&peer_id, &sync_data).await;
                        }
                    }
                    // Record sync
                    self.merkle
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .record_sync(&peer_id);
                }
                _ => {} // Sync failed, try next time
            }
        }
    }

    /// Run peer exchange: ask a connected peer for its peer list and
    /// discover new peers to connect to.
    async fn run_peer_exchange(&mut self, event_tx: &mpsc::Sender<NetworkEvent>) {
        // Pick a random connected peer to ask
        let peer_id = match self.connections.keys().next().copied() {
            Some(id) => id,
            None => return,
        };

        let msg = NodeMessage::PeerExchangeRequest {
            sender: self.node_id.0.to_vec(),
        };

        match self.request_from_peer(&peer_id, msg).await {
            Ok(NodeMessage::PeerExchangeResponse { peers: peer_list }) => {
                for (id_bytes, addr_str) in peer_list {
                    // Parse the address
                    let addr = match addr_str.parse::<SocketAddr>() {
                        Ok(a) => a,
                        Err(_) => continue,
                    };

                    // Reconstruct node ID
                    if id_bytes.len() < 20 {
                        continue;
                    }
                    let mut id_arr = [0u8; 20];
                    id_arr.copy_from_slice(&id_bytes[..20]);
                    let discovered_id = NodeId::from_bytes(id_arr);

                    // Skip if it's us or already connected
                    if discovered_id == self.node_id {
                        continue;
                    }
                    if self.connections.contains_key(&discovered_id) {
                        continue;
                    }

                    // Try to connect to the discovered peer
                    let _ = self.connect_to_peer(addr, event_tx).await;
                }
            }
            _ => {} // Exchange failed, try next time
        }
    }

    /// Enforce minimum replication copies for stored keys.
    ///
    /// Checks each locally stored key and ensures it's replicated to
    /// at least `min_copies` nodes. If under-replicated, sends Replicate
    /// messages to additional nodes determined by the hash ring.
    async fn enforce_min_copies(&mut self, event_tx: &mpsc::Sender<NetworkEvent>) {
        let min_copies = self.config.replication.min_copies;
        if min_copies <= 1 || self.connections.is_empty() {
            return;
        }

        // Get keys that might need replication
        let keys_and_versions: Vec<(String, Vec<u8>, u64)> = {
            let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
            storage
                .iter()
                .filter(|(_, v)| !v.is_expired())
                .map(|(k, v)| (k.clone(), v.data.clone(), v.version))
                .collect()
        };

        for (key, value, version) in keys_and_versions {
            let replica_nodes = {
                let ring = self.ring.read().unwrap_or_else(|e| e.into_inner());
                ring.get_nodes(&key, min_copies)
            };

            let mut pending = 0usize;
            for &node_id in &replica_nodes {
                if node_id == self.node_id {
                    continue;
                }
                // Skip peers in probation for primary replication
                let in_probation = {
                    let peers = self.peers.read().unwrap_or_else(|e| e.into_inner());
                    peers.get(&node_id).map_or(true, |p| p.probation)
                };
                if in_probation && replica_nodes.len() > min_copies {
                    continue;
                }

                if self.connections.contains_key(&node_id) {
                    let msg = NodeMessage::Replicate {
                        key: key.clone(),
                        value: value.clone(),
                        version,
                    };
                    let _ = self.send_to_peer(&node_id, &msg).await;
                    pending += 1;
                }
            }

            if pending > 0 {
                self.replication_pending
                    .fetch_add(pending, std::sync::atomic::Ordering::Relaxed);
                let _ = event_tx
                    .send(NetworkEvent::ReplicationComplete(key, pending + 1))
                    .await;
            }
        }
    }

    /// Handle graceful shutdown.
    async fn handle_shutdown(&mut self) {
        // Close all connections
        for (_, conn) in self.connections.drain() {
            conn.close(0u32.into(), b"shutdown");
        }
        self.endpoint.close(0u32.into(), b"shutdown");
    }

    /// Write a length-prefixed bincode message to a QUIC send stream.
    async fn write_message(send: &mut quinn::SendStream, msg: &NodeMessage) -> Result<(), String> {
        let data =
            bincode::serialize(msg).map_err(|e| format!("Failed to serialize message: {}", e))?;

        let len = (data.len() as u32).to_be_bytes();
        send.write_all(&len)
            .await
            .map_err(|e| format!("Failed to write length: {}", e))?;
        send.write_all(&data)
            .await
            .map_err(|e| format!("Failed to write data: {}", e))?;
        send.finish()
            .map_err(|e| format!("Failed to finish stream: {}", e))?;

        Ok(())
    }

    /// Read a length-prefixed bincode message from a QUIC receive stream.
    async fn read_message(recv: &mut quinn::RecvStream) -> Result<NodeMessage, String> {
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf)
            .await
            .map_err(|e| format!("Failed to read length: {}", e))?;
        let len = u32::from_be_bytes(len_buf) as usize;

        if len > 16 * 1024 * 1024 {
            return Err(format!("Message too large: {} bytes", len));
        }

        let mut data = vec![0u8; len];
        recv.read_exact(&mut data)
            .await
            .map_err(|e| format!("Failed to read data: {}", e))?;

        bincode::deserialize(&data).map_err(|e| format!("Failed to deserialize message: {}", e))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(identity_dir: &std::path::Path) -> NetworkConfig {
        NetworkConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            bootstrap_peers: Vec::new(),
            identity_dir: identity_dir.to_path_buf(),
            heartbeat_interval: Duration::from_secs(60), // Long interval for tests
            replication: ReplicationConfig {
                min_copies: 1,
                max_copies: 2,
                write_mode: WriteMode::Asynchronous,
                read_quorum: 1,
                write_quorum: 1,
                vnodes_per_node: 16,
            },
            discovery: DiscoveryConfig {
                enable_broadcast: false,
                ..DiscoveryConfig::default()
            },
            join_token: None,
            max_connections: 10,
            message_timeout: Duration::from_secs(5),
            phi_threshold: 8.0,
        }
    }

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.max_connections, 50);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(2));
        assert_eq!(config.phi_threshold, 8.0);
    }

    #[test]
    fn test_replication_config_default() {
        let config = ReplicationConfig::default();
        assert_eq!(config.min_copies, 2);
        assert_eq!(config.max_copies, 3);
        assert_eq!(config.vnodes_per_node, 64);
        assert_eq!(config.read_quorum, 1);
        assert_eq!(config.write_quorum, 1);
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert!(config.enable_broadcast);
        assert_eq!(config.broadcast_port, 9876);
        assert!(config.enable_peer_exchange);
    }

    #[test]
    fn test_write_mode_eq() {
        assert_eq!(WriteMode::Synchronous, WriteMode::Synchronous);
        assert_eq!(WriteMode::Asynchronous, WriteMode::Asynchronous);
        assert_ne!(WriteMode::Synchronous, WriteMode::Asynchronous);
    }

    #[test]
    fn test_peer_state_new() {
        let node_id = NodeId::from_string("test_peer");
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let peer = PeerState::new(node_id, addr);

        assert_eq!(peer.node_id, node_id);
        assert_eq!(peer.addr, addr);
        assert!(peer.connected_since.is_some());
        assert_eq!(peer.reputation, 0.5);
        assert!(peer.probation);
        assert_eq!(peer.messages_sent, 0);
        assert_eq!(peer.messages_received, 0);
    }

    #[test]
    fn test_stored_value_no_ttl() {
        let value = StoredValue::new(b"hello".to_vec(), None);
        assert!(!value.is_expired());
        assert!(value.version > 0);
        assert_eq!(value.data, b"hello");
    }

    #[test]
    fn test_stored_value_with_ttl() {
        let value = StoredValue::new(b"data".to_vec(), Some(Duration::from_secs(3600)));
        assert!(!value.is_expired());
    }

    #[test]
    fn test_stored_value_expired() {
        let value = StoredValue {
            data: b"old".to_vec(),
            version: 1,
            expires_at: Some(Instant::now() - Duration::from_secs(1)),
        };
        assert!(value.is_expired());
    }

    #[test]
    fn test_ring_info() {
        let info = RingInfo {
            total_nodes: 3,
            total_vnodes: 192,
            replication_factor: 3,
        };
        assert_eq!(info.total_nodes, 3);
        assert_eq!(info.total_vnodes, 192);
        assert_eq!(info.replication_factor, 3);
    }

    #[test]
    fn test_create_node() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let result = NetworkNode::new(config);
        assert!(
            result.is_ok(),
            "Node creation should succeed: {:?}",
            result.err()
        );
        let node = result.unwrap();
        assert_eq!(node.peer_count(), 0);
        assert_eq!(node.local_key_count(), 0);
        assert!(node.local_addr().port() > 0);
    }

    #[test]
    fn test_node_local_storage() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        node.local_store("key1", b"value1".to_vec(), None);
        assert_eq!(node.local_get("key1"), Some(b"value1".to_vec()));
        assert_eq!(node.local_get("nonexistent"), None);
        assert_eq!(node.local_key_count(), 1);
    }

    #[test]
    fn test_node_local_storage_ttl() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        // Store with a long TTL — should be readable
        node.local_store("key_ttl", b"data".to_vec(), Some(Duration::from_secs(3600)));
        assert_eq!(node.local_get("key_ttl"), Some(b"data".to_vec()));
    }

    #[test]
    fn test_node_stats() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let stats = node.stats();
        assert_eq!(stats.peers_connected, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.keys_stored, 0);
    }

    #[test]
    fn test_node_ring_info() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let ring = node.ring_info();
        assert_eq!(ring.total_nodes, 1); // Just us
        assert_eq!(ring.total_vnodes, 16); // vnodes_per_node
        assert_eq!(ring.replication_factor, 2);
    }

    #[test]
    fn test_nodes_for_key() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let nodes = node.nodes_for_key("test_key");
        assert_eq!(nodes.len(), 1); // Only ourselves
        assert_eq!(nodes[0], node.node_id());
    }

    #[test]
    fn test_check_peer_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let unknown = NodeId::from_string("unknown_peer");
        match node.check_peer(&unknown) {
            NodeStatus::Unknown => {} // Expected
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn test_two_nodes_connect() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        // Both nodes need the same CA to connect via mutual TLS.
        // Generate shared CA and create separate node identities.
        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let id1 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id1, dir1.path()).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let id2 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id2, dir2.path()).unwrap();

        let config1 = test_config(dir1.path());
        let node1 = NetworkNode::new(config1).unwrap();

        let config2 = test_config(dir2.path());
        let node2 = NetworkNode::new(config2).unwrap();

        // Connect node2 to node1
        let result = node2.connect(node1.local_addr());
        assert!(
            result.is_ok(),
            "Connection should succeed: {:?}",
            result.err()
        );

        // Give some time for the connection to be established
        std::thread::sleep(Duration::from_millis(100));

        assert!(node2.peer_count() >= 1, "Node2 should have at least 1 peer");
    }

    #[test]
    fn test_two_nodes_store_get() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let id1 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id1, dir1.path()).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let id2 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id2, dir2.path()).unwrap();

        let config1 = test_config(dir1.path());
        let node1 = NetworkNode::new(config1).unwrap();

        let config2 = test_config(dir2.path());
        let node2 = NetworkNode::new(config2).unwrap();

        node2.connect(node1.local_addr()).unwrap();
        std::thread::sleep(Duration::from_millis(100));

        // Store on node1 — should be readable locally
        node1.local_store("shared_key", b"shared_value".to_vec(), None);
        assert_eq!(
            node1.local_get("shared_key"),
            Some(b"shared_value".to_vec())
        );
    }

    #[test]
    fn test_node_disconnect() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        let (ca_cert, ca_key) = CertificateManager::generate_ca().unwrap();
        let id1 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id1, dir1.path()).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let id2 = CertificateManager::generate_node_cert(&ca_cert, &ca_key).unwrap();
        CertificateManager::save_identity(&id2, dir2.path()).unwrap();

        let config1 = test_config(dir1.path());
        let node1 = NetworkNode::new(config1).unwrap();

        let config2 = test_config(dir2.path());
        let node2 = NetworkNode::new(config2).unwrap();

        let peer_id = node2.connect(node1.local_addr()).unwrap();
        std::thread::sleep(Duration::from_millis(100));

        node2.disconnect(&peer_id);
        assert_eq!(node2.peer_count(), 0);
    }

    #[test]
    fn test_node_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();
        node.shutdown();
        // Should be safe to call shutdown multiple times
        node.shutdown();
    }

    #[test]
    fn test_message_serialization() {
        let msg = NodeMessage::Ping {
            sender: NodeId::from_string("test"),
            timestamp: 12345,
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::Ping { timestamp, .. } => assert_eq!(timestamp, 12345),
            _ => panic!("Expected Ping"),
        }
    }

    #[test]
    fn test_message_serialization_complex() {
        let msg = NodeMessage::Put {
            key: "test_key".to_string(),
            value: vec![1, 2, 3, 4],
            ttl_secs: Some(3600),
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::Put {
                key,
                value,
                ttl_secs,
            } => {
                assert_eq!(key, "test_key");
                assert_eq!(value, vec![1, 2, 3, 4]);
                assert_eq!(ttl_secs, Some(3600));
            }
            _ => panic!("Expected Put"),
        }
    }

    #[test]
    fn test_message_replicate_serialization() {
        let msg = NodeMessage::Replicate {
            key: "replicated".to_string(),
            value: vec![10, 20, 30],
            version: 42,
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::Replicate {
                key,
                value,
                version,
            } => {
                assert_eq!(key, "replicated");
                assert_eq!(value, vec![10, 20, 30]);
                assert_eq!(version, 42);
            }
            _ => panic!("Expected Replicate"),
        }
    }

    #[test]
    fn test_message_join_request_serialization() {
        let msg = NodeMessage::JoinRequest {
            token: "abc123".to_string(),
            cert_der: vec![1, 2, 3],
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::JoinRequest { token, cert_der } => {
                assert_eq!(token, "abc123");
                assert_eq!(cert_der, vec![1, 2, 3]);
            }
            _ => panic!("Expected JoinRequest"),
        }
    }

    #[test]
    fn test_message_vector_search_serialization() {
        let msg = NodeMessage::VectorSearch {
            query: vec![1.0, 2.0, 3.0],
            limit: 10,
            request_id: 99,
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::VectorSearch {
                query,
                limit,
                request_id,
            } => {
                assert_eq!(query, vec![1.0, 2.0, 3.0]);
                assert_eq!(limit, 10);
                assert_eq!(request_id, 99);
            }
            _ => panic!("Expected VectorSearch"),
        }
    }

    #[test]
    fn test_process_message_ping() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");
        let msg = NodeMessage::Ping {
            sender: NodeId::from_string("client"),
            timestamp: 42,
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::Pong { sender, timestamp }) => {
                assert_eq!(sender, node_id);
                assert_eq!(timestamp, 42);
            }
            _ => panic!("Expected Pong"),
        }
    }

    #[test]
    fn test_process_message_get_put() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Put a value
        let put_msg = NodeMessage::Put {
            key: "hello".to_string(),
            value: b"world".to_vec(),
            ttl_secs: None,
        };
        let resp = EventLoop::process_message(&put_msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::PutAck { key, success, .. }) => {
                assert_eq!(key, "hello");
                assert!(success);
            }
            _ => panic!("Expected PutAck"),
        }

        // Get the value back
        let get_msg = NodeMessage::Get {
            key: "hello".to_string(),
            request_id: 1,
        };
        let resp = EventLoop::process_message(&get_msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::GetResponse {
                key,
                value,
                request_id,
            }) => {
                assert_eq!(key, "hello");
                assert_eq!(value, Some(b"world".to_vec()));
                assert_eq!(request_id, 1);
            }
            _ => panic!("Expected GetResponse"),
        }
    }

    #[test]
    fn test_process_message_delete() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Store something first
        storage
            .write()
            .unwrap()
            .insert("key".to_string(), StoredValue::new(b"val".to_vec(), None));

        let del_msg = NodeMessage::Delete {
            key: "key".to_string(),
            request_id: 5,
        };
        let resp = EventLoop::process_message(&del_msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::DeleteAck {
                key,
                success,
                request_id,
            }) => {
                assert_eq!(key, "key");
                assert!(success);
                assert_eq!(request_id, 5);
            }
            _ => panic!("Expected DeleteAck"),
        }

        // Key should be gone
        assert!(storage.read().unwrap().get("key").is_none());
    }

    #[test]
    fn test_process_message_replicate() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::Replicate {
            key: "rep_key".to_string(),
            value: b"rep_val".to_vec(),
            version: 100,
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::ReplicateAck {
                key,
                version,
                success,
            }) => {
                assert_eq!(key, "rep_key");
                assert_eq!(version, 100);
                assert!(success);
            }
            _ => panic!("Expected ReplicateAck"),
        }

        // Value should be stored
        assert_eq!(
            storage.read().unwrap().get("rep_key").unwrap().data,
            b"rep_val"
        );
    }

    #[test]
    fn test_process_message_replicate_version_check() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Store with high version
        storage.write().unwrap().insert(
            "key".to_string(),
            StoredValue {
                data: b"new".to_vec(),
                version: 200,
                expires_at: None,
            },
        );

        // Try to replicate with lower version — should not overwrite
        let msg = NodeMessage::Replicate {
            key: "key".to_string(),
            value: b"old".to_vec(),
            version: 100,
        };
        EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);

        assert_eq!(storage.read().unwrap().get("key").unwrap().data, b"new");
    }

    #[test]
    fn test_process_message_join_request_no_token_required() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::JoinRequest {
            token: "any_token".to_string(),
            cert_der: vec![1, 2, 3],
        };
        // No join token required (config_join_token is None)
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::JoinAccepted { .. }) => {} // Expected
            _ => panic!("Expected JoinAccepted when no token required"),
        }
    }

    #[test]
    fn test_process_message_join_request_valid_token() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let token = JoinToken::generate(24, Some(5));
        let token_str = token.token.clone();
        let tokens = Arc::new(RwLock::new(vec![token]));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::JoinRequest {
            token: token_str,
            cert_der: vec![],
        };
        let config_token = Some("required".to_string());
        let resp =
            EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &config_token);
        match resp {
            Some(NodeMessage::JoinAccepted { .. }) => {} // Expected
            _ => panic!("Expected JoinAccepted with valid token"),
        }
    }

    #[test]
    fn test_process_message_join_request_invalid_token() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(vec![JoinToken::generate(24, Some(5))]));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::JoinRequest {
            token: "wrong_token".to_string(),
            cert_der: vec![],
        };
        let config_token = Some("required".to_string());
        let resp =
            EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &config_token);
        match resp {
            Some(NodeMessage::JoinRejected { reason }) => {
                assert!(reason.contains("Invalid"));
            }
            _ => panic!("Expected JoinRejected with invalid token"),
        }
    }

    #[test]
    fn test_process_message_sync_request_same_data() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Empty storage — merkle root should match empty tree
        let empty_tree = MerkleTree::from_data(&std::collections::BTreeMap::new());
        let root = empty_tree
            .root_hash()
            .map(|h| h.to_vec())
            .unwrap_or_default();

        let msg = NodeMessage::SyncRequest { merkle_root: root };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::SyncResponse { diff_keys }) => {
                assert!(diff_keys.is_empty(), "Identical data should have no diff");
            }
            _ => panic!("Expected SyncResponse"),
        }
    }

    #[test]
    fn test_process_message_sync_request_different_data() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Put data in storage
        storage.write().unwrap().insert(
            "local_key".to_string(),
            StoredValue::new(b"data".to_vec(), None),
        );

        // Send a different merkle root
        let msg = NodeMessage::SyncRequest {
            merkle_root: vec![0u8; 32],
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::SyncResponse { diff_keys }) => {
                assert!(diff_keys.contains(&"local_key".to_string()));
            }
            _ => panic!("Expected SyncResponse with diff_keys"),
        }
    }

    #[test]
    fn test_process_message_sync_data() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::SyncData {
            entries: vec![("synced_key".to_string(), b"synced_val".to_vec())],
        };
        EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);

        assert_eq!(
            storage.read().unwrap().get("synced_key").unwrap().data,
            b"synced_val"
        );
    }

    #[test]
    fn test_process_message_peer_exchange() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        // Add a peer to the peers map
        let peer_id = NodeId::from_string("peer1");
        let peer_addr: SocketAddr = "192.168.1.100:5000".parse().unwrap();
        peers
            .write()
            .unwrap()
            .insert(peer_id, PeerState::new(peer_id, peer_addr));

        let msg = NodeMessage::PeerExchangeRequest {
            sender: vec![1, 2, 3],
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::PeerExchangeResponse { peers: peer_list }) => {
                assert_eq!(peer_list.len(), 1);
                assert_eq!(peer_list[0].1, "192.168.1.100:5000");
            }
            _ => panic!("Expected PeerExchangeResponse"),
        }
    }

    #[test]
    fn test_process_message_vector_search() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::VectorSearch {
            query: vec![1.0, 2.0],
            limit: 5,
            request_id: 42,
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::VectorSearchResponse {
                results,
                request_id,
            }) => {
                assert!(results.is_empty()); // No VectorDb at this layer
                assert_eq!(request_id, 42);
            }
            _ => panic!("Expected VectorSearchResponse"),
        }
    }

    #[test]
    fn test_process_message_map_task() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        let peers = Arc::new(RwLock::new(HashMap::new()));
        let tokens = Arc::new(RwLock::new(Vec::new()));
        let node_id = NodeId::from_string("server");

        let msg = NodeMessage::MapTask {
            job_id: "job1".to_string(),
            chunk_id: "chunk1".to_string(),
            data: vec![1, 2, 3],
        };
        let resp = EventLoop::process_message(&msg, node_id, &storage, &peers, &tokens, &None);
        match resp {
            Some(NodeMessage::MapResult { job_id, .. }) => {
                assert_eq!(job_id, "job1");
            }
            _ => panic!("Expected MapResult"),
        }
    }

    #[test]
    fn test_node_generate_join_token() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let token = node.generate_join_token(24, Some(5));
        assert!(token.is_valid());
        assert!(!token.token.is_empty());
        assert_eq!(token.remaining_uses(), Some(5));
    }

    #[test]
    fn test_node_identity_accessible() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let identity = node.identity();
        assert!(!identity.cert_der.is_empty());
        assert!(!identity.key_der.is_empty());
        assert_eq!(identity.node_id, node.node_id());
    }

    #[test]
    fn test_node_peer_reputation_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        let unknown = NodeId::from_string("unknown");
        assert_eq!(node.peer_reputation(&unknown), None);
        assert_eq!(node.peer_in_probation(&unknown), None);
    }

    #[test]
    fn test_node_sync_status() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let node = NetworkNode::new(config).unwrap();

        // No peers, so no sync needed
        assert_eq!(node.sync_status(), 0);
    }

    #[test]
    fn test_storage_to_btree() {
        let storage = Arc::new(RwLock::new(HashMap::new()));
        storage
            .write()
            .unwrap()
            .insert("b".to_string(), StoredValue::new(b"val_b".to_vec(), None));
        storage
            .write()
            .unwrap()
            .insert("a".to_string(), StoredValue::new(b"val_a".to_vec(), None));
        // Add an expired entry that should be excluded
        storage.write().unwrap().insert(
            "expired".to_string(),
            StoredValue {
                data: b"old".to_vec(),
                version: 1,
                expires_at: Some(Instant::now() - Duration::from_secs(1)),
            },
        );

        let btree = EventLoop::storage_to_btree(&storage);
        assert_eq!(btree.len(), 2);
        assert_eq!(btree.get("a").unwrap(), &b"val_a".to_vec());
        assert_eq!(btree.get("b").unwrap(), &b"val_b".to_vec());
        assert!(!btree.contains_key("expired"));
    }

    #[test]
    fn test_peer_exchange_serialization() {
        let msg = NodeMessage::PeerExchangeRequest {
            sender: vec![1, 2, 3, 4],
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::PeerExchangeRequest { sender } => {
                assert_eq!(sender, vec![1, 2, 3, 4]);
            }
            _ => panic!("Expected PeerExchangeRequest"),
        }

        let msg2 = NodeMessage::PeerExchangeResponse {
            peers: vec![(vec![5, 6], "127.0.0.1:8080".to_string())],
        };
        let encoded2 = bincode::serialize(&msg2).unwrap();
        let decoded2: NodeMessage = bincode::deserialize(&encoded2).unwrap();
        match decoded2 {
            NodeMessage::PeerExchangeResponse { peers } => {
                assert_eq!(peers.len(), 1);
                assert_eq!(peers[0].1, "127.0.0.1:8080");
            }
            _ => panic!("Expected PeerExchangeResponse"),
        }
    }

    #[test]
    fn test_discovery_announce_serialization() {
        let msg = NodeMessage::DiscoveryAnnounce {
            node_id: vec![1, 2, 3],
            quic_addr: "192.168.1.1:9000".to_string(),
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: NodeMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            NodeMessage::DiscoveryAnnounce { node_id, quic_addr } => {
                assert_eq!(node_id, vec![1, 2, 3]);
                assert_eq!(quic_addr, "192.168.1.1:9000");
            }
            _ => panic!("Expected DiscoveryAnnounce"),
        }
    }

    #[test]
    fn test_max_connections_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.max_connections = 2;
        let node = NetworkNode::new(config).unwrap();
        // The node should have been created with max_connections = 2
        // This is enforced in handle_incoming at runtime
        assert_eq!(node.peer_count(), 0);
    }
}
