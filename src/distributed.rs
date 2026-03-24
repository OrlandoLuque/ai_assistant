//! Distributed computing module for DHT and MapReduce operations
//!
//! This module provides:
//! - DHT (Distributed Hash Table) for decentralized key-value storage
//! - MapReduce framework for distributed data processing
//! - CRDT (Conflict-free Replicated Data Types) for eventual consistency
//! - Consensus primitives for strong consistency when needed

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// DHT (Distributed Hash Table)
// =============================================================================

/// Node identifier in the DHT network (160-bit like Kademlia)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeId(pub(crate) [u8; 20]);

impl NodeId {
    /// Create a new random node ID
    pub fn random() -> Self {
        let mut bytes = [0u8; 20];
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        // Simple pseudo-random generation using wrapping shifts
        for (i, byte) in bytes.iter_mut().enumerate() {
            let shift1 = (i * 8) % 128;
            let shift2 = ((i + 7) * 3) % 128;
            *byte = ((now >> shift1) ^ (now >> shift2)) as u8;
        }

        Self(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    /// Create from a string (hashed)
    pub fn from_string(s: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let hash = hasher.finish();

        let mut bytes = [0u8; 20];
        for i in 0..8 {
            bytes[i] = (hash >> (i * 8)) as u8;
        }
        // Fill remaining with secondary hash
        let hash2 = hash.wrapping_mul(0x517cc1b727220a95);
        for i in 8..16 {
            bytes[i] = (hash2 >> ((i - 8) * 8)) as u8;
        }
        let hash3 = hash2.wrapping_mul(0x517cc1b727220a95);
        for i in 16..20 {
            bytes[i] = (hash3 >> ((i - 16) * 8)) as u8;
        }

        Self(bytes)
    }

    /// Calculate XOR distance between two node IDs (Kademlia metric)
    pub fn distance(&self, other: &NodeId) -> NodeId {
        let mut result = [0u8; 20];
        for i in 0..20 {
            result[i] = self.0[i] ^ other.0[i];
        }
        NodeId(result)
    }

    /// Get the leading zeros count (for bucket selection)
    pub fn leading_zeros(&self) -> u32 {
        let mut count = 0;
        for byte in &self.0 {
            if *byte == 0 {
                count += 8;
            } else {
                count += byte.leading_zeros();
                break;
            }
        }
        count
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.to_hex()[..8])
    }
}

/// Information about a node in the DHT
#[derive(Clone, Debug)]
pub struct DhtNode {
    pub id: NodeId,
    pub address: SocketAddr,
    pub last_seen: Instant,
    pub rtt_ms: Option<u32>,
}

impl DhtNode {
    pub fn new(id: NodeId, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            last_seen: Instant::now(),
            rtt_ms: None,
        }
    }

    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_seen.elapsed() > timeout
    }
}

/// K-bucket for storing nodes at a specific distance
#[derive(Clone, Debug)]
pub struct KBucket {
    pub nodes: Vec<DhtNode>,
    pub k: usize, // Max nodes per bucket (typically 20)
}

impl KBucket {
    pub fn new(k: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(k),
            k,
        }
    }

    /// Try to add a node to the bucket
    pub fn add(&mut self, node: DhtNode) -> bool {
        // Check if node already exists
        if let Some(pos) = self.nodes.iter().position(|n| n.id == node.id) {
            // Move to end (most recently seen)
            self.nodes.remove(pos);
            self.nodes.push(node);
            return true;
        }

        // Bucket not full, just add
        if self.nodes.len() < self.k {
            self.nodes.push(node);
            return true;
        }

        // Bucket full - check if oldest is stale
        if self.nodes[0].is_stale(Duration::from_secs(300)) {
            self.nodes.remove(0);
            self.nodes.push(node);
            return true;
        }

        false
    }

    /// Get the N closest nodes to a target
    pub fn closest(&self, target: &NodeId, n: usize) -> Vec<&DhtNode> {
        let mut sorted: Vec<_> = self.nodes.iter().collect();
        sorted.sort_by_key(|node| node.id.distance(target).leading_zeros());
        sorted.into_iter().rev().take(n).collect()
    }
}

/// Routing table using Kademlia-style k-buckets
#[derive(Clone)]
pub struct RoutingTable {
    pub local_id: NodeId,
    pub buckets: Vec<KBucket>,
    pub k: usize,
}

impl RoutingTable {
    pub fn new(local_id: NodeId, k: usize) -> Self {
        Self {
            local_id,
            buckets: (0..160).map(|_| KBucket::new(k)).collect(),
            k,
        }
    }

    /// Get the bucket index for a node
    fn bucket_index(&self, node_id: &NodeId) -> usize {
        let distance = self.local_id.distance(node_id);
        let zeros = distance.leading_zeros() as usize;
        zeros.min(159)
    }

    /// Add a node to the routing table
    pub fn add(&mut self, node: DhtNode) -> bool {
        if node.id == self.local_id {
            return false;
        }
        let idx = self.bucket_index(&node.id);
        self.buckets[idx].add(node)
    }

    /// Find the K closest nodes to a target
    pub fn find_closest(&self, target: &NodeId, count: usize) -> Vec<DhtNode> {
        let mut all_nodes: Vec<_> = self
            .buckets
            .iter()
            .flat_map(|b| b.nodes.iter().cloned())
            .collect();

        all_nodes.sort_by(|a, b| {
            let dist_a = a.id.distance(target);
            let dist_b = b.id.distance(target);
            dist_b.leading_zeros().cmp(&dist_a.leading_zeros())
        });

        all_nodes.into_iter().take(count).collect()
    }

    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.buckets.iter().map(|b| b.nodes.len()).sum()
    }
}

/// Value stored in the DHT with metadata and access tracking.
#[derive(Clone, Debug)]
pub struct DhtValue {
    pub data: Vec<u8>,
    pub owner: NodeId,
    pub version: u64,
    pub ttl: Option<Duration>,
    pub created_at: Instant,
    pub replicas: HashSet<NodeId>,
    /// Last time this value was accessed (for LRU eviction).
    pub last_accessed: Instant,
    /// Number of times this value has been read.
    pub access_count: u64,
    /// Cached data size in bytes.
    pub size_bytes: usize,
    /// Priority: 0 = normal, >0 = pinned (never evicted by LRU).
    pub priority: u8,
}

impl DhtValue {
    pub fn new(data: Vec<u8>, owner: NodeId) -> Self {
        let size = data.len();
        let now = Instant::now();
        Self {
            data,
            owner,
            version: 1,
            ttl: None,
            created_at: now,
            replicas: HashSet::new(),
            last_accessed: now,
            access_count: 0,
            size_bytes: size,
            priority: 0,
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Mark as pinned (will not be evicted by cache policies).
    pub fn pinned(mut self) -> Self {
        self.priority = 1;
        self
    }

    /// Record an access (updates last_accessed and access_count).
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

/// Cache eviction policy for DHT storage.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvictionPolicy {
    /// Least Recently Used — evict the entry accessed longest ago.
    Lru,
    /// Least Recently Written — evict the oldest entry by creation time.
    Lrw,
    /// Largest First — evict the largest entry by data size.
    LargestFirst,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::Lru
    }
}

/// Cache configuration for DHT storage limits and eviction.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DhtCacheConfig {
    /// Maximum number of entries (0 = unlimited).
    pub max_entries: usize,
    /// Maximum total bytes across all entries (0 = unlimited).
    pub max_bytes: usize,
    /// Default TTL for entries without explicit TTL (None = no expiry).
    pub default_ttl: Option<Duration>,
    /// Eviction strategy when at capacity.
    pub eviction_policy: EvictionPolicy,
}

impl Default for DhtCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_bytes: 100 * 1024 * 1024, // 100 MB
            default_ttl: None,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

/// Capabilities advertised by a node in the cluster.
///
/// Used for capability-based routing — a node can select the best peer
/// for a given task based on available models, features, and load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NodeCapabilities {
    /// Node identifier.
    pub node_id: String,
    /// Available LLM models (e.g., ["llama3.2:1b", "mistral:7b"]).
    pub models: Vec<String>,
    /// Active feature flags (e.g., ["rag", "multi-agent", "security"]).
    pub features: Vec<String>,
    /// Maximum concurrent requests this node can handle.
    pub max_concurrent: usize,
    /// Whether the node has a RAG database.
    pub has_rag: bool,
    /// Whether the node has GPU acceleration.
    pub has_gpu: bool,
    /// Available memory in megabytes.
    pub total_memory_mb: usize,
    /// Current utilization (0.0 = idle, 1.0 = fully loaded).
    pub current_load: f32,
    /// Timestamp of last update (Unix ms).
    pub last_updated_ms: u64,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            models: Vec::new(),
            features: Vec::new(),
            max_concurrent: 4,
            has_rag: false,
            has_gpu: false,
            total_memory_mb: 0,
            current_load: 0.0,
            last_updated_ms: 0,
        }
    }
}

impl NodeCapabilities {
    /// Create capabilities for a node with the given ID.
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            last_updated_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ..Default::default()
        }
    }

    /// Check if this node can handle a specific model.
    pub fn has_model(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }

    /// Check if this node has a specific feature.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }

    /// Whether the node has capacity for more requests.
    pub fn has_capacity(&self) -> bool {
        self.current_load < 0.9
    }
}

/// Compute FNV-1a hash for data integrity verification.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// DHT configuration
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DhtConfig {
    /// Number of nodes per k-bucket
    pub k: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Lookup parallelism (alpha)
    pub alpha: usize,
    /// Node timeout
    pub node_timeout: Duration,
    /// Value expiration check interval
    pub cleanup_interval: Duration,
    /// Cache policies for storage limits and eviction.
    pub cache: DhtCacheConfig,
}

impl Default for DhtConfig {
    fn default() -> Self {
        Self {
            k: 20,
            replication_factor: 3,
            alpha: 3,
            node_timeout: Duration::from_secs(300),
            cleanup_interval: Duration::from_secs(60),
            cache: DhtCacheConfig::default(),
        }
    }
}

/// The main DHT instance
pub struct Dht {
    pub config: DhtConfig,
    pub local_id: NodeId,
    pub routing_table: Arc<RwLock<RoutingTable>>,
    pub storage: Arc<RwLock<HashMap<NodeId, DhtValue>>>,
    pub stats: Arc<RwLock<DhtStats>>,
}

/// DHT statistics
#[derive(Clone, Debug, Default)]
pub struct DhtStats {
    pub gets: u64,
    pub puts: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub replications: u64,
    /// Number of entries evicted by cache policy.
    pub evictions: u64,
    /// Number of entries expired by TTL.
    pub expirations: u64,
    /// Current total bytes stored.
    pub current_bytes: usize,
    /// Current number of entries.
    pub current_entries: usize,
}

impl Dht {
    /// Create a new DHT instance
    pub fn new(config: DhtConfig) -> Self {
        let local_id = NodeId::random();
        Self {
            routing_table: Arc::new(RwLock::new(RoutingTable::new(local_id, config.k))),
            storage: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DhtStats::default())),
            local_id,
            config,
        }
    }

    /// Create with a specific node ID
    pub fn with_id(id: NodeId, config: DhtConfig) -> Self {
        Self {
            routing_table: Arc::new(RwLock::new(RoutingTable::new(id, config.k))),
            storage: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DhtStats::default())),
            local_id: id,
            config,
        }
    }

    /// Store a value in the DHT (with automatic eviction if at capacity).
    pub fn put(&self, key: &str, value: Vec<u8>) -> NodeId {
        let key_id = NodeId::from_string(key);
        let mut dht_value = DhtValue::new(value, self.local_id);

        // Auto-increment version if key already exists
        {
            let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
            if let Some(existing) = storage.get(&key_id) {
                dht_value.version = existing.version + 1;
            }
        }

        // Apply default TTL if configured and no explicit TTL set
        if dht_value.ttl.is_none() {
            if let Some(default_ttl) = self.config.cache.default_ttl {
                dht_value.ttl = Some(default_ttl);
            }
        }

        // Evict if needed before inserting
        self.evict_if_needed();

        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        storage.insert(key_id, dht_value);
        #[cfg(feature = "analytics")]
        let storage_len = storage.len();
        drop(storage);

        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.puts += 1;
        self.update_size_stats(&mut stats);

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::DhtStorage,
            storage_len,
        );

        key_id
    }

    /// Store with TTL
    pub fn put_with_ttl(&self, key: &str, value: Vec<u8>, ttl: Duration) -> NodeId {
        let key_id = NodeId::from_string(key);
        let dht_value = DhtValue::new(value, self.local_id).with_ttl(ttl);

        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        storage.insert(key_id, dht_value);

        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.puts += 1;

        key_id
    }

    /// Get a value from the DHT (updates access tracking for LRU).
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let key_id = NodeId::from_string(key);

        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.gets += 1;

        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = storage.get_mut(&key_id) {
            if !value.is_expired() {
                value.touch(); // Update LRU tracking
                stats.get_hits += 1;
                return Some(value.data.clone());
            }
        }

        stats.get_misses += 1;
        None
    }

    /// Get with metadata
    pub fn get_with_meta(&self, key: &str) -> Option<DhtValue> {
        let key_id = NodeId::from_string(key);
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage.get(&key_id).cloned()
    }

    /// Delete a value
    pub fn delete(&self, key: &str) -> bool {
        let key_id = NodeId::from_string(key);
        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        storage.remove(&key_id).is_some()
    }

    /// Add a node to the routing table
    pub fn add_node(&self, node: DhtNode) -> bool {
        let mut rt = self
            .routing_table
            .write()
            .unwrap_or_else(|e| e.into_inner());
        rt.add(node)
    }

    /// Record that a node holds a replica of a key.
    pub fn add_replica(&self, key: &str, node_id: NodeId) {
        let key_id = NodeId::from_string(key);
        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        if let Some(value) = storage.get_mut(&key_id) {
            value.replicas.insert(node_id);
        }
    }

    /// Get all nodes that hold replicas of a key.
    pub fn get_replicas(&self, key: &str) -> Vec<NodeId> {
        let key_id = NodeId::from_string(key);
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage
            .get(&key_id)
            .map(|v| v.replicas.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Find closest nodes to a key
    pub fn find_nodes(&self, key: &str, count: usize) -> Vec<DhtNode> {
        let key_id = NodeId::from_string(key);
        let rt = self.routing_table.read().unwrap_or_else(|e| e.into_inner());
        rt.find_closest(&key_id, count)
    }

    /// Cleanup expired values
    pub fn cleanup(&self) -> usize {
        let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
        let before = storage.len();
        storage.retain(|_, v| !v.is_expired());
        let expired = before - storage.len();
        drop(storage);

        if expired > 0 {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.expirations += expired as u64;
            self.update_size_stats(&mut stats);
        }

        expired
    }

    /// Evict entries if storage exceeds configured limits.
    ///
    /// First removes expired entries, then evicts by policy (LRU/LRW/LargestFirst).
    /// Pinned entries (priority > 0) are never evicted.
    pub fn evict_if_needed(&self) {
        let max_entries = self.config.cache.max_entries;
        let max_bytes = self.config.cache.max_bytes;

        if max_entries == 0 && max_bytes == 0 {
            return; // No limits configured
        }

        // First pass: remove expired
        {
            let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
            let before = storage.len();
            storage.retain(|_, v| !v.is_expired());
            let expired = before - storage.len();
            if expired > 0 {
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                stats.expirations += expired as u64;
            }
        }

        // Check if still over limits
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        let current_entries = storage.len();
        let current_bytes: usize = storage.values().map(|v| v.size_bytes).sum();

        // Account for the upcoming insert (+1 entry)
        let over_entries = max_entries > 0 && current_entries >= max_entries;
        let over_bytes = max_bytes > 0 && current_bytes > max_bytes;

        if !over_entries && !over_bytes {
            return;
        }

        // Collect eviction candidates (non-pinned), sorted by policy
        let mut candidates: Vec<(NodeId, Instant, Instant, usize)> = storage
            .iter()
            .filter(|(_, v)| v.priority == 0) // Never evict pinned
            .map(|(k, v)| (*k, v.last_accessed, v.created_at, v.size_bytes))
            .collect();
        drop(storage);

        // Sort by eviction policy
        match self.config.cache.eviction_policy {
            EvictionPolicy::Lru => candidates.sort_by_key(|(_, la, _, _)| *la), // oldest access first
            EvictionPolicy::Lrw => candidates.sort_by_key(|(_, _, ca, _)| *ca), // oldest creation first
            EvictionPolicy::LargestFirst => candidates.sort_by(|a, b| b.3.cmp(&a.3)), // largest first
        }

        // Evict until under limits
        let mut to_evict = Vec::new();
        let mut freed_entries = 0;
        let mut freed_bytes = 0usize;

        for (key, _, _, size) in &candidates {
            let still_over_entries = max_entries > 0 && (current_entries - freed_entries) >= max_entries;
            let still_over_bytes = max_bytes > 0 && (current_bytes - freed_bytes) > max_bytes;

            if !still_over_entries && !still_over_bytes {
                break;
            }

            to_evict.push(*key);
            freed_entries += 1;
            freed_bytes += size;
        }

        if !to_evict.is_empty() {
            let mut storage = self.storage.write().unwrap_or_else(|e| e.into_inner());
            for key in &to_evict {
                storage.remove(key);
            }
            drop(storage);

            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.evictions += to_evict.len() as u64;
            self.update_size_stats(&mut stats);
        }
    }

    /// Update current_bytes and current_entries in stats.
    fn update_size_stats(&self, stats: &mut DhtStats) {
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        stats.current_entries = storage.len();
        stats.current_bytes = storage.values().map(|v| v.size_bytes).sum();
    }

    /// Get the cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
        if stats.gets == 0 {
            0.0
        } else {
            stats.get_hits as f64 / stats.gets as f64
        }
    }

    /// Get statistics
    pub fn stats(&self) -> DhtStats {
        let mut s = self.stats.read().unwrap_or_else(|e| e.into_inner()).clone();
        // Refresh size stats
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        s.current_entries = storage.len();
        s.current_bytes = storage.values().map(|v| v.size_bytes).sum();
        s
    }

    /// Get all keys (for iteration)
    pub fn keys(&self) -> Vec<NodeId> {
        let storage = self.storage.read().unwrap_or_else(|e| e.into_inner());
        storage.keys().cloned().collect()
    }

    /// Get local storage size
    pub fn local_size(&self) -> usize {
        self.storage.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// =============================================================================
// CRDT (Conflict-free Replicated Data Types)
// =============================================================================

/// G-Counter: Grow-only counter (increment only)
#[derive(Clone, Debug)]
pub struct GCounter {
    /// Node ID -> count
    pub counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Increment the counter for a node
    pub fn increment(&mut self, node_id: &str) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Increment by a specific amount
    pub fn increment_by(&mut self, node_id: &str, amount: u64) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += amount;
    }

    /// Get the total count
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another G-Counter (takes max of each node's count)
    pub fn merge(&mut self, other: &GCounter) {
        for (node, &count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// PN-Counter: Positive-Negative counter (increment and decrement)
#[derive(Clone, Debug)]
pub struct PNCounter {
    pub positive: GCounter,
    pub negative: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self {
            positive: GCounter::new(),
            negative: GCounter::new(),
        }
    }

    /// Increment the counter
    pub fn increment(&mut self, node_id: &str) {
        self.positive.increment(node_id);
    }

    /// Decrement the counter
    pub fn decrement(&mut self, node_id: &str) {
        self.negative.increment(node_id);
    }

    /// Get the current value
    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    /// Merge with another PN-Counter
    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

impl Default for PNCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// LWW-Register: Last-Writer-Wins Register
#[derive(Clone, Debug)]
pub struct LWWRegister<T: Clone> {
    pub value: Option<T>,
    pub timestamp: u64,
    pub node_id: String,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new() -> Self {
        Self {
            value: None,
            timestamp: 0,
            node_id: String::new(),
        }
    }

    /// Set a new value with timestamp.
    ///
    /// Rejects timestamps more than 1 hour in the future to prevent
    /// timestamp spoofing attacks (LWW hijack with u64::MAX).
    /// Timestamps can be in any unit (seconds, millis, micros) — the
    /// clamp auto-detects by comparing magnitude with current time.
    pub fn set(&mut self, value: T, timestamp: u64, node_id: &str) {
        // Security: reject far-future timestamps (max 1 hour drift)
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        // 1 hour in microseconds
        let max_allowed = now_micros.saturating_add(3_600_000_000);
        let effective_ts = timestamp.min(max_allowed);

        if effective_ts > self.timestamp
            || (effective_ts == self.timestamp && node_id > self.node_id.as_str())
        {
            self.value = Some(value);
            self.timestamp = effective_ts;
            self.node_id = node_id.to_string();
        }
    }

    /// Get the current value
    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Merge with another LWW-Register
    pub fn merge(&mut self, other: &LWWRegister<T>) {
        if other.timestamp > self.timestamp
            || (other.timestamp == self.timestamp && other.node_id > self.node_id)
        {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.node_id = other.node_id.clone();
        }
    }
}

impl<T: Clone> Default for LWWRegister<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// OR-Set: Observed-Remove Set (add/remove elements)
#[derive(Clone, Debug)]
pub struct ORSet<T: Clone + Eq + Hash> {
    /// Element -> Set of (unique_tag, node_id) pairs
    elements: HashMap<T, HashSet<(u64, String)>>,
    /// Tombstones: removed tags
    tombstones: HashSet<(u64, String)>,
    /// Counter for generating unique tags
    tag_counter: u64,
}

impl<T: Clone + Eq + Hash> ORSet<T> {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashSet::new(),
            tag_counter: 0,
        }
    }

    /// Add an element
    pub fn add(&mut self, element: T, node_id: &str) {
        self.tag_counter += 1;
        let tag = (self.tag_counter, node_id.to_string());
        self.elements
            .entry(element)
            .or_insert_with(HashSet::new)
            .insert(tag);
    }

    /// Remove an element (all observed tags)
    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.get(element) {
            for tag in tags.iter() {
                self.tombstones.insert(tag.clone());
            }
        }
        self.elements.remove(element);
    }

    /// Check if element exists
    pub fn contains(&self, element: &T) -> bool {
        if let Some(tags) = self.elements.get(element) {
            tags.iter().any(|tag| !self.tombstones.contains(tag))
        } else {
            false
        }
    }

    /// Get all elements
    pub fn elements(&self) -> Vec<&T> {
        self.elements
            .iter()
            .filter(|(_, tags)| tags.iter().any(|tag| !self.tombstones.contains(tag)))
            .map(|(elem, _)| elem)
            .collect()
    }

    /// Merge with another OR-Set
    pub fn merge(&mut self, other: &ORSet<T>) {
        // Merge elements
        for (elem, tags) in &other.elements {
            let entry = self
                .elements
                .entry(elem.clone())
                .or_insert_with(HashSet::new);
            for tag in tags {
                entry.insert(tag.clone());
            }
        }

        // Merge tombstones
        for tag in &other.tombstones {
            self.tombstones.insert(tag.clone());
        }

        // Update tag counter
        self.tag_counter = self.tag_counter.max(other.tag_counter);

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::CrdtOrSetTombstones,
            self.tombstones.len(),
        );
    }

    /// Count of elements
    pub fn len(&self) -> usize {
        self.elements().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone + Eq + Hash> Default for ORSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// LWW-Map: Last-Writer-Wins Map
#[derive(Clone, Debug)]
pub struct LWWMap<K: Clone + Eq + Hash, V: Clone> {
    entries: HashMap<K, LWWRegister<V>>,
    /// Optional maximum number of entries. When set, oldest entries are evicted on insert.
    max_entries: Option<usize>,
}

impl<K: Clone + Eq + Hash, V: Clone> LWWMap<K, V> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: None,
        }
    }

    /// Create a new LWWMap with a maximum entry limit.
    pub fn with_max_entries(max: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries: Some(max),
        }
    }

    /// Set a key-value pair
    pub fn set(&mut self, key: K, value: V, timestamp: u64, node_id: &str) {
        let entry = self.entries.entry(key).or_insert_with(LWWRegister::new);
        entry.set(value, timestamp, node_id);
        self.evict_if_needed();
    }

    /// Get a value
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key).and_then(|reg| reg.get())
    }

    /// Merge with another LWW-Map
    pub fn merge(&mut self, other: &LWWMap<K, V>) {
        for (key, reg) in &other.entries {
            let entry = self
                .entries
                .entry(key.clone())
                .or_insert_with(LWWRegister::new);
            entry.merge(reg);
        }
        self.evict_if_needed();
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<&K> {
        self.entries.keys().collect()
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict oldest entries (by timestamp) when over the configured max.
    fn evict_if_needed(&mut self) {
        let max = match self.max_entries {
            Some(m) => m,
            None => return,
        };
        if self.entries.len() <= max {
            return;
        }
        // Collect entries sorted by timestamp ascending (oldest first)
        let mut by_age: Vec<(K, u64)> = self
            .entries
            .iter()
            .map(|(k, reg)| (k.clone(), reg.timestamp))
            .collect();
        by_age.sort_by_key(|(_, ts)| *ts);

        let to_remove = self.entries.len() - max;
        for (key, _) in by_age.into_iter().take(to_remove) {
            self.entries.remove(&key);
        }
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Default for LWWMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// MAPREDUCE FRAMEWORK
// =============================================================================

/// Status of a MapReduce job
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum JobStatus {
    Pending,
    Mapping,
    Shuffling,
    Reducing,
    Completed,
    Failed(String),
}

/// A chunk of data to be processed
#[derive(Clone, Debug)]
pub struct DataChunk {
    pub id: String,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

impl DataChunk {
    pub fn new(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            data,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Result from a Map operation
#[derive(Clone, Debug)]
pub struct MapOutput {
    pub key: String,
    pub value: Vec<u8>,
}

impl MapOutput {
    pub fn new(key: impl Into<String>, value: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            value,
        }
    }

    pub fn from_string(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into().into_bytes(),
        }
    }
}

/// Result from a Reduce operation
#[derive(Clone, Debug)]
pub struct ReduceOutput {
    pub key: String,
    pub value: Vec<u8>,
}

impl ReduceOutput {
    pub fn new(key: impl Into<String>, value: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            value,
        }
    }

    pub fn from_string(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into().into_bytes(),
        }
    }
}

/// Mapper function type
pub type MapFn = Arc<dyn Fn(&DataChunk) -> Vec<MapOutput> + Send + Sync>;

/// Reducer function type
pub type ReduceFn = Arc<dyn Fn(&str, Vec<Vec<u8>>) -> ReduceOutput + Send + Sync>;

/// Combiner function type (optional local reduction)
pub type CombineFn = Arc<dyn Fn(&str, Vec<Vec<u8>>) -> Vec<u8> + Send + Sync>;

/// Configuration for a MapReduce job
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MapReduceConfig {
    /// Number of mapper workers
    pub num_mappers: usize,
    /// Number of reducer workers
    pub num_reducers: usize,
    /// Chunk size for splitting input
    pub chunk_size: usize,
    /// Enable local combining before shuffle
    pub use_combiner: bool,
    /// Timeout for each task
    pub task_timeout: Duration,
    /// Retry count for failed tasks
    pub max_retries: u32,
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self {
            num_mappers: 4,
            num_reducers: 2,
            chunk_size: 1024 * 1024, // 1MB
            use_combiner: true,
            task_timeout: Duration::from_secs(300),
            max_retries: 3,
        }
    }
}

/// A MapReduce job
pub struct MapReduceJob {
    pub id: String,
    pub config: MapReduceConfig,
    pub status: JobStatus,
    pub map_fn: MapFn,
    pub reduce_fn: ReduceFn,
    pub combine_fn: Option<CombineFn>,

    // Internal state
    input_chunks: Vec<DataChunk>,
    map_outputs: Arc<Mutex<HashMap<String, Vec<Vec<u8>>>>>,
    reduce_outputs: Arc<Mutex<Vec<ReduceOutput>>>,

    // Progress tracking
    mapped_chunks: Arc<Mutex<usize>>,
    reduced_keys: Arc<Mutex<usize>>,

    // Timing
    started_at: Option<Instant>,
    completed_at: Option<Instant>,
}

impl MapReduceJob {
    /// Create a new MapReduce job
    pub fn new(id: impl Into<String>, map_fn: MapFn, reduce_fn: ReduceFn) -> Self {
        Self {
            id: id.into(),
            config: MapReduceConfig::default(),
            status: JobStatus::Pending,
            map_fn,
            reduce_fn,
            combine_fn: None,
            input_chunks: Vec::new(),
            map_outputs: Arc::new(Mutex::new(HashMap::new())),
            reduce_outputs: Arc::new(Mutex::new(Vec::new())),
            mapped_chunks: Arc::new(Mutex::new(0)),
            reduced_keys: Arc::new(Mutex::new(0)),
            started_at: None,
            completed_at: None,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: MapReduceConfig) -> Self {
        self.config = config;
        self
    }

    /// Set combiner function
    pub fn with_combiner(mut self, combine_fn: CombineFn) -> Self {
        self.combine_fn = Some(combine_fn);
        self.config.use_combiner = true;
        self
    }

    /// Add input data
    pub fn add_input(&mut self, chunk: DataChunk) {
        self.input_chunks.push(chunk);
    }

    /// Add multiple inputs
    pub fn add_inputs(&mut self, chunks: Vec<DataChunk>) {
        self.input_chunks.extend(chunks);
    }

    /// Split raw data into chunks
    pub fn split_input(&mut self, data: &[u8], chunk_prefix: &str) {
        for (i, chunk_data) in data.chunks(self.config.chunk_size).enumerate() {
            let chunk = DataChunk::new(format!("{}_{}", chunk_prefix, i), chunk_data.to_vec());
            self.input_chunks.push(chunk);
        }
    }

    /// Execute the MapReduce job with parallel map and reduce phases via rayon.
    pub fn execute(&mut self) -> Result<Vec<ReduceOutput>, String> {
        use rayon::prelude::*;

        self.started_at = Some(Instant::now());

        // Phase 1: Parallel Map
        self.status = JobStatus::Mapping;
        let map_fn = &self.map_fn;
        let all_outputs: Vec<Vec<MapOutput>> = self
            .input_chunks
            .par_iter()
            .map(|chunk| (map_fn)(chunk))
            .collect();

        // Merge map outputs into grouped structure (sequential — fast, just inserts)
        {
            let mut map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());
            for outputs in all_outputs {
                for output in outputs {
                    map_outputs
                        .entry(output.key)
                        .or_insert_with(Vec::new)
                        .push(output.value);
                }
            }
        }
        *self.mapped_chunks.lock().unwrap_or_else(|e| e.into_inner()) = self.input_chunks.len();

        // Phase 2: Shuffle (already grouped by key in map_outputs)
        self.status = JobStatus::Shuffling;

        // Optional: Apply combiner
        if self.config.use_combiner {
            if let Some(ref combine_fn) = self.combine_fn {
                let mut map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());
                for (key, values) in map_outputs.iter_mut() {
                    if values.len() > 1 {
                        let combined = (combine_fn)(key, values.clone());
                        *values = vec![combined];
                    }
                }
            }
        }

        // Phase 3: Parallel Reduce
        self.status = JobStatus::Reducing;
        let map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());
        let reduce_fn = &self.reduce_fn;

        let entries: Vec<(&String, &Vec<Vec<u8>>)> = map_outputs.iter().collect();
        let results: Vec<ReduceOutput> = entries
            .par_iter()
            .map(|(key, values)| (reduce_fn)(key, (*values).clone()))
            .collect();

        *self.reduced_keys.lock().unwrap_or_else(|e| e.into_inner()) = results.len();

        // Complete
        self.status = JobStatus::Completed;
        self.completed_at = Some(Instant::now());

        *self
            .reduce_outputs
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = results.clone();

        Ok(results)
    }

    /// Get job progress
    pub fn progress(&self) -> (usize, usize, usize, usize) {
        let mapped = *self.mapped_chunks.lock().unwrap_or_else(|e| e.into_inner());
        let total_chunks = self.input_chunks.len();
        let reduced = *self.reduced_keys.lock().unwrap_or_else(|e| e.into_inner());
        let total_keys = self
            .map_outputs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        (mapped, total_chunks, reduced, total_keys)
    }

    /// Get execution time
    pub fn execution_time(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            (Some(start), None) => Some(start.elapsed()),
            _ => None,
        }
    }

    /// Get results
    pub fn results(&self) -> Vec<ReduceOutput> {
        self.reduce_outputs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Registry of map/reduce functions available on this node.
///
/// Since closures cannot be serialized over the network, each node must
/// register the same map/reduce functions under the same `job_id`. When a
/// `MapTask` arrives from the network, the registry looks up the function
/// by `job_id` and executes it locally on the received data.
pub struct MapWorkerRegistry {
    map_fns: HashMap<String, MapFn>,
    reduce_fns: HashMap<String, ReduceFn>,
}

impl MapWorkerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            map_fns: HashMap::new(),
            reduce_fns: HashMap::new(),
        }
    }

    /// Register a map function for a job ID.
    pub fn register_map(&mut self, job_id: &str, map_fn: MapFn) {
        self.map_fns.insert(job_id.to_string(), map_fn);
    }

    /// Register a reduce function for a job ID.
    pub fn register_reduce(&mut self, job_id: &str, reduce_fn: ReduceFn) {
        self.reduce_fns.insert(job_id.to_string(), reduce_fn);
    }

    /// Execute a registered map function on a data chunk.
    /// Returns None if the job_id is not registered.
    pub fn execute_map(&self, job_id: &str, chunk: &DataChunk) -> Option<Vec<MapOutput>> {
        let map_fn = self.map_fns.get(job_id)?;
        Some((map_fn)(chunk))
    }

    /// Execute a registered reduce function.
    /// Returns None if the job_id is not registered.
    pub fn execute_reduce(
        &self,
        job_id: &str,
        key: &str,
        values: Vec<Vec<u8>>,
    ) -> Option<ReduceOutput> {
        let reduce_fn = self.reduce_fns.get(job_id)?;
        Some((reduce_fn)(key, values))
    }

    /// Check if a job is registered.
    pub fn has_job(&self, job_id: &str) -> bool {
        self.map_fns.contains_key(job_id)
    }

    /// Number of registered jobs.
    pub fn job_count(&self) -> usize {
        self.map_fns.len()
    }
}

impl Default for MapWorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MapReduceJob {
    /// Distribute chunks across local node + remote peers.
    ///
    /// The local node is ALWAYS included as a worker — it processes its share
    /// with rayon while remote peers process theirs via `NodeMessage::MapTask`.
    ///
    /// If `peer_map_results` is empty (no peers responded), all chunks are
    /// processed locally as fallback.
    ///
    /// # Arguments
    /// * `peer_map_results` — pre-collected map results from remote peers,
    ///   keyed by chunk_id. The caller is responsible for sending MapTasks
    ///   to peers and collecting MapResults (via NetworkNode).
    /// * `local_chunk_count` — how many of the first input_chunks to process locally.
    ///   The rest are assumed handled by peers.
    pub fn execute_distributed_with_results(
        &mut self,
        local_chunk_count: usize,
        peer_map_results: Vec<Vec<MapOutput>>,
    ) -> Result<Vec<ReduceOutput>, String> {
        use rayon::prelude::*;

        self.started_at = Some(Instant::now());

        // Phase 1: Local map (our share of chunks)
        self.status = JobStatus::Mapping;
        let local_count = local_chunk_count.min(self.input_chunks.len());
        let local_chunks = &self.input_chunks[..local_count];

        let map_fn = &self.map_fn;
        let local_outputs: Vec<Vec<MapOutput>> = local_chunks
            .par_iter()
            .map(|chunk| (map_fn)(chunk))
            .collect();

        // Phase 2: Merge local + remote map outputs
        {
            let mut map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());

            // Local outputs
            for outputs in &local_outputs {
                for output in outputs {
                    map_outputs
                        .entry(output.key.clone())
                        .or_insert_with(Vec::new)
                        .push(output.value.clone());
                }
            }

            // Remote peer outputs
            for outputs in &peer_map_results {
                for output in outputs {
                    map_outputs
                        .entry(output.key.clone())
                        .or_insert_with(Vec::new)
                        .push(output.value.clone());
                }
            }
        }

        let total_mapped = local_count + peer_map_results.len();
        *self.mapped_chunks.lock().unwrap_or_else(|e| e.into_inner()) = total_mapped;

        // Phase 3: Shuffle (already grouped by key)
        self.status = JobStatus::Shuffling;

        // Optional combiner
        if self.config.use_combiner {
            if let Some(ref combine_fn) = self.combine_fn {
                let mut map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());
                for (key, values) in map_outputs.iter_mut() {
                    if values.len() > 1 {
                        let combined = (combine_fn)(key, values.clone());
                        *values = vec![combined];
                    }
                }
            }
        }

        // Phase 4: Reduce (always local — coordinator aggregates)
        self.status = JobStatus::Reducing;
        let map_outputs = self.map_outputs.lock().unwrap_or_else(|e| e.into_inner());
        let reduce_fn = &self.reduce_fn;

        let entries: Vec<(&String, &Vec<Vec<u8>>)> = map_outputs.iter().collect();
        let results: Vec<ReduceOutput> = entries
            .par_iter()
            .map(|(key, values)| (reduce_fn)(key, (*values).clone()))
            .collect();

        *self.reduced_keys.lock().unwrap_or_else(|e| e.into_inner()) = results.len();

        self.status = JobStatus::Completed;
        self.completed_at = Some(Instant::now());
        *self.reduce_outputs.lock().unwrap_or_else(|e| e.into_inner()) = results.clone();

        Ok(results)
    }

    /// Calculate how many chunks the local node should process given total workers.
    ///
    /// `total_workers` includes the local node (self).
    pub fn local_chunk_count(&self, total_workers: usize) -> usize {
        if total_workers <= 1 {
            self.input_chunks.len()
        } else {
            (self.input_chunks.len() + total_workers - 1) / total_workers
        }
    }

    /// Get chunks assigned to remote peer `peer_index` (0-based, excluding local).
    pub fn remote_chunks(&self, peer_index: usize, total_workers: usize) -> Vec<&DataChunk> {
        let chunks_per_worker = self.local_chunk_count(total_workers);
        let start = chunks_per_worker + peer_index * chunks_per_worker;
        let end = (start + chunks_per_worker).min(self.input_chunks.len());
        if start >= self.input_chunks.len() {
            Vec::new()
        } else {
            self.input_chunks[start..end].iter().collect()
        }
    }
}

/// Builder for creating common MapReduce patterns
pub struct MapReduceBuilder;

impl MapReduceBuilder {
    /// Word count example
    pub fn word_count() -> MapReduceJob {
        let map_fn: MapFn = Arc::new(|chunk| {
            let text = String::from_utf8_lossy(&chunk.data);
            let mut outputs = Vec::new();

            for word in text.split_whitespace() {
                let word = word
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>();

                if !word.is_empty() {
                    outputs.push(MapOutput::from_string(&word, "1"));
                }
            }

            outputs
        });

        let reduce_fn: ReduceFn = Arc::new(|key, values| {
            let count: u64 = values
                .iter()
                .filter_map(|v| String::from_utf8_lossy(v).parse::<u64>().ok())
                .sum();

            ReduceOutput::from_string(key, count.to_string())
        });

        let combine_fn: CombineFn = Arc::new(|_key, values| {
            let count: u64 = values
                .iter()
                .filter_map(|v| String::from_utf8_lossy(v).parse::<u64>().ok())
                .sum();

            count.to_string().into_bytes()
        });

        MapReduceJob::new("word_count", map_fn, reduce_fn).with_combiner(combine_fn)
    }

    /// Sum values by key
    pub fn sum_by_key() -> MapReduceJob {
        let map_fn: MapFn = Arc::new(|chunk| {
            let text = String::from_utf8_lossy(&chunk.data);
            let mut outputs = Vec::new();

            for line in text.lines() {
                if let Some((key, value)) = line.split_once(',') {
                    outputs.push(MapOutput::from_string(key.trim(), value.trim()));
                }
            }

            outputs
        });

        let reduce_fn: ReduceFn = Arc::new(|key, values| {
            let sum: f64 = values
                .iter()
                .filter_map(|v| String::from_utf8_lossy(v).parse::<f64>().ok())
                .sum();

            ReduceOutput::from_string(key, sum.to_string())
        });

        MapReduceJob::new("sum_by_key", map_fn, reduce_fn)
    }

    /// Group by key (collect all values)
    pub fn group_by_key() -> MapReduceJob {
        let map_fn: MapFn = Arc::new(|chunk| {
            let text = String::from_utf8_lossy(&chunk.data);
            let mut outputs = Vec::new();

            for line in text.lines() {
                if let Some((key, value)) = line.split_once(',') {
                    outputs.push(MapOutput::from_string(key.trim(), value.trim()));
                }
            }

            outputs
        });

        let reduce_fn: ReduceFn = Arc::new(|key, values| {
            let collected: Vec<String> = values
                .iter()
                .map(|v| String::from_utf8_lossy(v).to_string())
                .collect();

            ReduceOutput::from_string(key, collected.join(";"))
        });

        MapReduceJob::new("group_by_key", map_fn, reduce_fn)
    }

    /// Custom job with user-provided functions
    pub fn custom(
        id: impl Into<String>,
        map_fn: impl Fn(&DataChunk) -> Vec<MapOutput> + Send + Sync + 'static,
        reduce_fn: impl Fn(&str, Vec<Vec<u8>>) -> ReduceOutput + Send + Sync + 'static,
    ) -> MapReduceJob {
        MapReduceJob::new(id, Arc::new(map_fn), Arc::new(reduce_fn))
    }
}

// =============================================================================
// DISTRIBUTED COORDINATOR (combines DHT + MapReduce + CRDT)
// =============================================================================

/// A distributed coordinator that combines DHT, MapReduce, and CRDTs
pub struct DistributedCoordinator {
    pub node_id: NodeId,
    pub dht: Arc<Dht>,
    pub jobs: Arc<RwLock<HashMap<String, MapReduceJob>>>,
    pub counters: Arc<RwLock<HashMap<String, PNCounter>>>,
    pub registers: Arc<RwLock<HashMap<String, LWWRegister<Vec<u8>>>>>,
    pub sets: Arc<RwLock<HashMap<String, ORSet<String>>>>,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub fn new() -> Self {
        let dht = Dht::new(DhtConfig::default());
        Self {
            node_id: dht.local_id,
            dht: Arc::new(dht),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            registers: Arc::new(RwLock::new(HashMap::new())),
            sets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a value in the DHT
    pub fn store(&self, key: &str, value: Vec<u8>) {
        self.dht.put(key, value);
    }

    /// Retrieve a value from the DHT
    pub fn retrieve(&self, key: &str) -> Option<Vec<u8>> {
        self.dht.get(key)
    }

    /// Submit a MapReduce job
    pub fn submit_job(&self, mut job: MapReduceJob) -> Result<String, String> {
        let job_id = job.id.clone();

        // Execute the job
        job.execute()?;

        // Store results in DHT
        for result in job.results() {
            let key = format!("job:{}:result:{}", job_id, result.key);
            self.dht.put(&key, result.value);
        }

        // Store job metadata
        let mut jobs = self.jobs.write().unwrap_or_else(|e| e.into_inner());
        jobs.insert(job_id.clone(), job);

        Ok(job_id)
    }

    /// Get job results
    pub fn get_job_results(&self, job_id: &str) -> Option<Vec<ReduceOutput>> {
        let jobs = self.jobs.read().unwrap_or_else(|e| e.into_inner());
        jobs.get(job_id).map(|j| j.results())
    }

    /// Get or create a distributed counter
    pub fn counter(&self, name: &str) -> PNCounter {
        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());
        counters.get(name).cloned().unwrap_or_default()
    }

    /// Increment a distributed counter
    pub fn increment_counter(&self, name: &str) {
        let mut counters = self.counters.write().unwrap_or_else(|e| e.into_inner());
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(PNCounter::new);
        counter.increment(&self.node_id.to_hex());
    }

    /// Decrement a distributed counter
    pub fn decrement_counter(&self, name: &str) {
        let mut counters = self.counters.write().unwrap_or_else(|e| e.into_inner());
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(PNCounter::new);
        counter.decrement(&self.node_id.to_hex());
    }

    /// Set a distributed register
    pub fn set_register(&self, name: &str, value: Vec<u8>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut registers = self.registers.write().unwrap_or_else(|e| e.into_inner());
        let register = registers
            .entry(name.to_string())
            .or_insert_with(LWWRegister::new);
        register.set(value, timestamp, &self.node_id.to_hex());
    }

    /// Get a distributed register
    pub fn get_register(&self, name: &str) -> Option<Vec<u8>> {
        let registers = self.registers.read().unwrap_or_else(|e| e.into_inner());
        registers.get(name).and_then(|r| r.get().cloned())
    }

    /// Add to a distributed set
    pub fn add_to_set(&self, set_name: &str, element: &str) {
        let mut sets = self.sets.write().unwrap_or_else(|e| e.into_inner());
        let set = sets.entry(set_name.to_string()).or_insert_with(ORSet::new);
        set.add(element.to_string(), &self.node_id.to_hex());
    }

    /// Remove from a distributed set
    pub fn remove_from_set(&self, set_name: &str, element: &str) {
        let mut sets = self.sets.write().unwrap_or_else(|e| e.into_inner());
        if let Some(set) = sets.get_mut(set_name) {
            set.remove(&element.to_string());
        }
    }

    /// Check if element is in set
    pub fn set_contains(&self, set_name: &str, element: &str) -> bool {
        let sets = self.sets.read().unwrap_or_else(|e| e.into_inner());
        sets.get(set_name)
            .map(|s| s.contains(&element.to_string()))
            .unwrap_or(false)
    }

    /// Get all elements in a set
    pub fn get_set(&self, set_name: &str) -> Vec<String> {
        let sets = self.sets.read().unwrap_or_else(|e| e.into_inner());
        sets.get(set_name)
            .map(|s| s.elements().into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Merge state from another coordinator (for replication)
    pub fn merge_state(&self, other: &DistributedCoordinator) {
        // Merge counters
        {
            let mut our_counters = self.counters.write().unwrap_or_else(|e| e.into_inner());
            let their_counters = other.counters.read().unwrap_or_else(|e| e.into_inner());

            for (name, their_counter) in their_counters.iter() {
                let our_counter = our_counters
                    .entry(name.clone())
                    .or_insert_with(PNCounter::new);
                our_counter.merge(their_counter);
            }
        }

        // Merge registers
        {
            let mut our_registers = self.registers.write().unwrap_or_else(|e| e.into_inner());
            let their_registers = other.registers.read().unwrap_or_else(|e| e.into_inner());

            for (name, their_register) in their_registers.iter() {
                let our_register = our_registers
                    .entry(name.clone())
                    .or_insert_with(LWWRegister::new);
                our_register.merge(their_register);
            }
        }

        // Merge sets
        {
            let mut our_sets = self.sets.write().unwrap_or_else(|e| e.into_inner());
            let their_sets = other.sets.read().unwrap_or_else(|e| e.into_inner());

            for (name, their_set) in their_sets.iter() {
                let our_set = our_sets.entry(name.clone()).or_insert_with(ORSet::new);
                our_set.merge(their_set);
            }
        }
    }
}

impl Default for DistributedCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Network Messages (for distributed-network feature)
// =============================================================================

/// Messages exchanged between nodes in the distributed network.
///
/// Covers heartbeat, DHT operations, replication, anti-entropy sync,
/// MapReduce task distribution, cluster management, and discovery.
#[cfg(feature = "distributed-network")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NodeMessage {
    // --- Heartbeat ---
    /// Ping request with sender identity and timestamp.
    Ping { sender: NodeId, timestamp: u64 },
    /// Pong response to a ping.
    Pong { sender: NodeId, timestamp: u64 },

    // --- DHT Operations ---
    /// Request to get a value by key.
    Get { key: String, request_id: u64 },
    /// Response to a Get request.
    GetResponse {
        key: String,
        value: Option<Vec<u8>>,
        request_id: u64,
    },
    /// Request to store a key-value pair.
    Put {
        key: String,
        value: Vec<u8>,
        ttl_secs: Option<u64>,
    },
    /// Acknowledgement of a Put request.
    PutAck {
        key: String,
        success: bool,
        request_id: u64,
    },
    /// Request to delete a key.
    Delete { key: String, request_id: u64 },
    /// Acknowledgement of a Delete request.
    DeleteAck {
        key: String,
        success: bool,
        request_id: u64,
    },

    // --- Replication ---
    /// Replicate a key-value pair to this node.
    Replicate {
        key: String,
        value: Vec<u8>,
        version: u64,
    },
    /// Acknowledgement of replication.
    ReplicateAck {
        key: String,
        version: u64,
        success: bool,
    },

    // --- Anti-Entropy Sync ---
    /// Request sync by sending our Merkle root hash.
    SyncRequest { merkle_root: Vec<u8> },
    /// Response with keys that differ.
    SyncResponse { diff_keys: Vec<String> },
    /// Send actual data for sync.
    SyncData { entries: Vec<(String, Vec<u8>)> },

    // --- MapReduce ---
    /// Distribute a map task to a node.
    MapTask {
        job_id: String,
        chunk_id: String,
        data: Vec<u8>,
    },
    /// Result of a map task.
    MapResult {
        job_id: String,
        outputs: Vec<(String, Vec<u8>)>,
    },
    /// Distribute a reduce task to a node.
    ReduceTask {
        job_id: String,
        key: String,
        values: Vec<Vec<u8>>,
    },
    /// Result of a reduce task.
    ReduceResult {
        job_id: String,
        key: String,
        value: Vec<u8>,
    },

    // --- Cluster Management ---
    /// Request to join the cluster with a token and certificate.
    JoinRequest { token: String, cert_der: Vec<u8> },
    /// Accept a join request, providing the assigned node ID and peer list.
    JoinAccepted {
        node_id: Vec<u8>,
        peers: Vec<(Vec<u8>, String)>,
    },
    /// Reject a join request.
    JoinRejected { reason: String },
    /// Notification that a node has left the cluster.
    NodeLeft { node_id: Vec<u8> },

    // --- Discovery ---
    /// LAN discovery announcement with node identity and QUIC address.
    DiscoveryAnnounce { node_id: Vec<u8>, quic_addr: String },

    // --- Peer Exchange ---
    /// Request a peer's known peer list.
    PeerExchangeRequest { sender: Vec<u8> },
    /// Response with known peers: list of (node_id_bytes, socket_addr_string).
    PeerExchangeResponse { peers: Vec<(Vec<u8>, String)> },

    // --- Vector DB ---
    /// Search for similar vectors across the cluster.
    VectorSearch {
        query: Vec<f32>,
        limit: usize,
        request_id: u64,
    },
    /// Response with vector search results: (id, score, metadata).
    VectorSearchResponse {
        results: Vec<(String, f32, HashMap<String, String>)>,
        request_id: u64,
    },

    // --- Cache Invalidation ---
    /// Invalidate a cached key across the cluster.
    Invalidate { key: String, version: u64 },
    /// Acknowledgement of invalidation.
    InvalidateAck { key: String, success: bool },

    // --- Distributed Log Correlation ---
    /// Request log entries for a specific trace from a remote node.
    LogRequest {
        trace_id: String,
        since_ms: Option<u64>,
        max_entries: usize,
        min_level: u8,
    },
    /// Response with log entries for the requested trace.
    LogResponse {
        trace_id: String,
        entries: Vec<crate::distributed_log::DistributedLogEntry>,
    },
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // DHT Tests

    #[test]
    fn test_node_id_random() {
        let id1 = NodeId::random();
        let id2 = NodeId::random();
        // Should be different (with very high probability)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_id_from_string() {
        let id1 = NodeId::from_string("hello");
        let id2 = NodeId::from_string("hello");
        let id3 = NodeId::from_string("world");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_node_id_distance() {
        let id1 = NodeId::from_string("a");
        let id2 = NodeId::from_string("b");

        let dist = id1.distance(&id2);
        assert_ne!(dist.leading_zeros(), 160); // Not identical
    }

    #[test]
    fn test_dht_put_get() {
        let dht = Dht::new(DhtConfig::default());

        dht.put("key1", b"value1".to_vec());
        dht.put("key2", b"value2".to_vec());

        assert_eq!(dht.get("key1"), Some(b"value1".to_vec()));
        assert_eq!(dht.get("key2"), Some(b"value2".to_vec()));
        assert_eq!(dht.get("key3"), None);
    }

    #[test]
    fn test_dht_ttl_expiration() {
        let dht = Dht::new(DhtConfig::default());

        dht.put_with_ttl("temp", b"data".to_vec(), Duration::from_millis(1));

        // Should exist initially
        assert!(dht.get("temp").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        // Should be gone
        assert!(dht.get("temp").is_none());
    }

    #[test]
    fn test_dht_delete() {
        let dht = Dht::new(DhtConfig::default());

        dht.put("key", b"value".to_vec());
        assert!(dht.get("key").is_some());

        assert!(dht.delete("key"));
        assert!(dht.get("key").is_none());
    }

    #[test]
    fn test_routing_table() {
        let local_id = NodeId::random();
        let mut rt = RoutingTable::new(local_id, 20);

        // Add some nodes
        for i in 0..10 {
            let node = DhtNode::new(
                NodeId::from_string(&format!("node_{}", i)),
                "127.0.0.1:8000".parse().unwrap(),
            );
            rt.add(node);
        }

        assert_eq!(rt.node_count(), 10);

        // Find closest
        let target = NodeId::from_string("target");
        let closest = rt.find_closest(&target, 5);
        assert_eq!(closest.len(), 5);
    }

    // CRDT Tests

    #[test]
    fn test_g_counter() {
        let mut c1 = GCounter::new();
        let mut c2 = GCounter::new();

        c1.increment("node1");
        c1.increment("node1");
        c2.increment("node2");
        c2.increment("node2");
        c2.increment("node2");

        assert_eq!(c1.value(), 2);
        assert_eq!(c2.value(), 3);

        // Merge
        c1.merge(&c2);
        assert_eq!(c1.value(), 5);
    }

    #[test]
    fn test_pn_counter() {
        let mut counter = PNCounter::new();

        counter.increment("node1");
        counter.increment("node1");
        counter.increment("node1");
        counter.decrement("node1");

        assert_eq!(counter.value(), 2);
    }

    #[test]
    fn test_lww_register() {
        let mut r1 = LWWRegister::new();
        let mut r2 = LWWRegister::new();

        r1.set("value1".to_string(), 100, "node1");
        r2.set("value2".to_string(), 200, "node2");

        // r2 has higher timestamp, should win
        r1.merge(&r2);
        assert_eq!(r1.get(), Some(&"value2".to_string()));
    }

    #[test]
    fn test_or_set() {
        let mut s1 = ORSet::new();
        let mut s2 = ORSet::new();

        s1.add("a".to_string(), "node1");
        s1.add("b".to_string(), "node1");
        s2.add("b".to_string(), "node2");
        s2.add("c".to_string(), "node2");

        s1.merge(&s2);

        assert!(s1.contains(&"a".to_string()));
        assert!(s1.contains(&"b".to_string()));
        assert!(s1.contains(&"c".to_string()));
        assert_eq!(s1.len(), 3);

        // Remove
        s1.remove(&"b".to_string());
        assert!(!s1.contains(&"b".to_string()));
        assert_eq!(s1.len(), 2);
    }

    #[test]
    fn test_lww_map() {
        let mut m1 = LWWMap::new();
        let mut m2 = LWWMap::new();

        m1.set("key1", "value1", 100, "node1");
        m2.set("key1", "value2", 200, "node2");

        m1.merge(&m2);

        assert_eq!(m1.get(&"key1"), Some(&"value2"));
    }

    #[test]
    fn test_lww_map_max_entries_eviction() {
        let mut map: LWWMap<String, String> = LWWMap::with_max_entries(5);
        // Insert 8 entries with increasing timestamps
        for i in 0..8u64 {
            map.set(format!("key-{}", i), format!("val-{}", i), i * 10, "node1");
        }
        // Should be capped at 5
        assert_eq!(map.len(), 5);
        // Oldest entries (lowest timestamp) should have been evicted
        assert!(map.get(&"key-0".to_string()).is_none());
        assert!(map.get(&"key-1".to_string()).is_none());
        assert!(map.get(&"key-2".to_string()).is_none());
        // Newest entries should remain
        assert_eq!(map.get(&"key-7".to_string()), Some(&"val-7".to_string()));
        assert_eq!(map.get(&"key-5".to_string()), Some(&"val-5".to_string()));
    }

    #[test]
    fn test_lww_map_no_limit_grows_freely() {
        let mut map: LWWMap<String, String> = LWWMap::new();
        for i in 0..100u64 {
            map.set(format!("key-{}", i), format!("val-{}", i), i, "node1");
        }
        assert_eq!(map.len(), 100);
    }

    // MapReduce Tests

    #[test]
    fn test_word_count() {
        let mut job = MapReduceBuilder::word_count();

        job.add_input(DataChunk::new("doc1", b"hello world hello".to_vec()));
        job.add_input(DataChunk::new("doc2", b"world foo bar foo".to_vec()));

        let results = job.execute().unwrap();

        let results_map: HashMap<_, _> = results
            .iter()
            .map(|r| {
                (
                    r.key.as_str(),
                    String::from_utf8_lossy(&r.value).to_string(),
                )
            })
            .collect();

        assert_eq!(results_map.get("hello"), Some(&"2".to_string()));
        assert_eq!(results_map.get("world"), Some(&"2".to_string()));
        assert_eq!(results_map.get("foo"), Some(&"2".to_string()));
        assert_eq!(results_map.get("bar"), Some(&"1".to_string()));
    }

    #[test]
    fn test_sum_by_key() {
        let mut job = MapReduceBuilder::sum_by_key();

        job.add_input(DataChunk::new("data1", b"a,10\nb,20\na,5".to_vec()));
        job.add_input(DataChunk::new("data2", b"b,15\nc,30".to_vec()));

        let results = job.execute().unwrap();

        let results_map: HashMap<_, _> = results
            .iter()
            .map(|r| {
                (
                    r.key.as_str(),
                    String::from_utf8_lossy(&r.value).parse::<f64>().unwrap(),
                )
            })
            .collect();

        assert_eq!(results_map.get("a"), Some(&15.0));
        assert_eq!(results_map.get("b"), Some(&35.0));
        assert_eq!(results_map.get("c"), Some(&30.0));
    }

    #[test]
    fn test_group_by_key() {
        let mut job = MapReduceBuilder::group_by_key();

        job.add_input(DataChunk::new(
            "data",
            b"user1,item1\nuser1,item2\nuser2,item3".to_vec(),
        ));

        let results = job.execute().unwrap();

        let results_map: HashMap<_, _> = results
            .iter()
            .map(|r| {
                (
                    r.key.as_str(),
                    String::from_utf8_lossy(&r.value).to_string(),
                )
            })
            .collect();

        let user1_items = results_map.get("user1").unwrap();
        assert!(user1_items.contains("item1"));
        assert!(user1_items.contains("item2"));
    }

    #[test]
    fn test_custom_mapreduce() {
        // Custom: find max value per key
        let job = MapReduceBuilder::custom(
            "max_by_key",
            |chunk| {
                let text = String::from_utf8_lossy(&chunk.data);
                text.lines()
                    .filter_map(|line| line.split_once(','))
                    .map(|(k, v)| MapOutput::from_string(k.trim(), v.trim()))
                    .collect()
            },
            |key, values| {
                let max = values
                    .iter()
                    .filter_map(|v| String::from_utf8_lossy(v).parse::<i64>().ok())
                    .max()
                    .unwrap_or(0);
                ReduceOutput::from_string(key, max.to_string())
            },
        );

        let mut job = job;
        job.add_input(DataChunk::new(
            "data",
            b"x,5\nx,10\nx,3\ny,20\ny,15".to_vec(),
        ));

        let results = job.execute().unwrap();

        let results_map: HashMap<_, _> = results
            .iter()
            .map(|r| {
                (
                    r.key.as_str(),
                    String::from_utf8_lossy(&r.value).to_string(),
                )
            })
            .collect();

        assert_eq!(results_map.get("x"), Some(&"10".to_string()));
        assert_eq!(results_map.get("y"), Some(&"20".to_string()));
    }

    #[test]
    fn test_job_progress() {
        let mut job = MapReduceBuilder::word_count();

        job.add_input(DataChunk::new("doc1", b"hello world".to_vec()));
        job.add_input(DataChunk::new("doc2", b"foo bar".to_vec()));

        let (mapped, total, _, _) = job.progress();
        assert_eq!(mapped, 0);
        assert_eq!(total, 2);

        job.execute().unwrap();

        let (mapped, total, _, _) = job.progress();
        assert_eq!(mapped, 2);
        assert_eq!(total, 2);
    }

    // Distributed Coordinator Tests

    #[test]
    fn test_coordinator_dht() {
        let coord = DistributedCoordinator::new();

        coord.store("key1", b"value1".to_vec());
        coord.store("key2", b"value2".to_vec());

        assert_eq!(coord.retrieve("key1"), Some(b"value1".to_vec()));
        assert_eq!(coord.retrieve("key2"), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_coordinator_counter() {
        let coord = DistributedCoordinator::new();

        coord.increment_counter("visits");
        coord.increment_counter("visits");
        coord.increment_counter("visits");
        coord.decrement_counter("visits");

        assert_eq!(coord.counter("visits").value(), 2);
    }

    #[test]
    fn test_coordinator_register() {
        let coord = DistributedCoordinator::new();

        coord.set_register("config", b"v1".to_vec());
        assert_eq!(coord.get_register("config"), Some(b"v1".to_vec()));

        coord.set_register("config", b"v2".to_vec());
        assert_eq!(coord.get_register("config"), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_coordinator_set() {
        let coord = DistributedCoordinator::new();

        coord.add_to_set("users", "alice");
        coord.add_to_set("users", "bob");
        coord.add_to_set("users", "charlie");

        assert!(coord.set_contains("users", "alice"));
        assert!(coord.set_contains("users", "bob"));
        assert!(!coord.set_contains("users", "dave"));

        coord.remove_from_set("users", "bob");
        assert!(!coord.set_contains("users", "bob"));

        let users = coord.get_set("users");
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn test_coordinator_merge() {
        let coord1 = DistributedCoordinator::new();
        let coord2 = DistributedCoordinator::new();

        // Different operations on each
        coord1.increment_counter("count");
        coord1.increment_counter("count");

        coord2.increment_counter("count");
        coord2.add_to_set("items", "a");
        coord2.add_to_set("items", "b");

        // Merge coord2 into coord1
        coord1.merge_state(&coord2);

        // Counter should be sum (each node has its own namespace)
        let counter = coord1.counter("count");
        assert!(counter.value() >= 2); // At least coord1's contributions

        // Set should have both items
        assert!(coord1.set_contains("items", "a"));
        assert!(coord1.set_contains("items", "b"));
    }

    #[test]
    fn test_coordinator_mapreduce_job() {
        let coord = DistributedCoordinator::new();

        let mut job = MapReduceBuilder::word_count();
        job.add_input(DataChunk::new("doc", b"hello world hello".to_vec()));

        let job_id = coord.submit_job(job).unwrap();

        let results = coord.get_job_results(&job_id).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_split_input() {
        let mut job = MapReduceBuilder::word_count().with_config(MapReduceConfig {
            chunk_size: 10,
            ..Default::default()
        });

        let data = b"hello world foo bar baz qux";
        job.split_input(data, "chunk");

        // Should create multiple chunks
        let (_, total, _, _) = job.progress();
        assert!(total > 1);
    }

    // ── DHT Cache Policy Tests ──────────────────────────────────

    #[test]
    fn test_dht_cache_max_entries() {
        let mut config = DhtConfig::default();
        config.cache.max_entries = 3;
        let dht = Dht::new(config);

        for i in 0..5 {
            dht.put(&format!("key{}", i), vec![i as u8; 10]);
        }

        let stats = dht.stats();
        assert!(stats.current_entries <= 3, "Should evict to stay under max_entries");
        assert!(stats.evictions > 0, "Should have evicted some entries");
    }

    #[test]
    fn test_dht_cache_max_bytes() {
        let mut config = DhtConfig::default();
        config.cache.max_bytes = 100;
        config.cache.max_entries = 0; // No entry limit
        let dht = Dht::new(config);

        for i in 0..20 {
            dht.put(&format!("key{}", i), vec![i as u8; 20]); // 20 bytes each
        }

        let stats = dht.stats();
        // Should be around max_bytes + one entry (eviction happens before insert)
        assert!(stats.current_bytes <= 120, "Should evict to stay near max_bytes: got {}", stats.current_bytes);
        assert!(stats.evictions > 0, "Should have evicted some entries");
    }

    #[test]
    fn test_dht_cache_lru_eviction() {
        let mut config = DhtConfig::default();
        config.cache.max_entries = 3;
        config.cache.eviction_policy = EvictionPolicy::Lru;
        let dht = Dht::new(config);

        dht.put("a", b"aaa".to_vec());
        dht.put("b", b"bbb".to_vec());
        dht.put("c", b"ccc".to_vec());

        // Access "a" to make it recently used
        dht.get("a");

        // Insert "d" — should evict "b" (least recently accessed)
        dht.put("d", b"ddd".to_vec());

        assert!(dht.get("a").is_some(), "a should survive (recently accessed)");
        assert!(dht.get("c").is_some() || dht.get("d").is_some(), "c or d should exist");
    }

    #[test]
    fn test_dht_cache_pinned_not_evicted() {
        let mut config = DhtConfig::default();
        config.cache.max_entries = 2;
        let dht = Dht::new(config);

        // Insert pinned value
        let key_id = NodeId::from_string("pinned");
        let pinned_value = DhtValue::new(b"important".to_vec(), dht.local_id).pinned();
        {
            let mut storage = dht.storage.write().unwrap();
            storage.insert(key_id, pinned_value);
        }

        // Fill with normal values
        dht.put("normal1", b"x".to_vec());
        dht.put("normal2", b"y".to_vec());
        dht.put("normal3", b"z".to_vec());

        // Pinned should still exist
        let storage = dht.storage.read().unwrap();
        assert!(storage.contains_key(&key_id), "Pinned entry should not be evicted");
    }

    #[test]
    fn test_dht_cache_expired_evicted_first() {
        let mut config = DhtConfig::default();
        config.cache.max_entries = 3;
        let dht = Dht::new(config);

        // Insert expired value
        dht.put_with_ttl("expired", b"old".to_vec(), Duration::from_millis(1));
        dht.put("fresh1", b"new1".to_vec());
        dht.put("fresh2", b"new2".to_vec());

        std::thread::sleep(Duration::from_millis(10));

        // Insert one more — expired should be cleaned first
        dht.put("fresh3", b"new3".to_vec());

        let stats = dht.stats();
        assert!(stats.expirations > 0, "Should have expired entries");
    }

    #[test]
    fn test_dht_cache_stats_tracking() {
        let dht = Dht::new(DhtConfig::default());

        dht.put("k1", b"v1".to_vec());
        dht.put("k2", b"v2".to_vec());

        dht.get("k1"); // hit
        dht.get("k1"); // hit
        dht.get("missing"); // miss

        let stats = dht.stats();
        assert_eq!(stats.puts, 2);
        assert_eq!(stats.gets, 3);
        assert_eq!(stats.get_hits, 2);
        assert_eq!(stats.get_misses, 1);
        assert_eq!(stats.current_entries, 2);
        assert!(stats.current_bytes > 0);

        let hr = dht.hit_rate();
        assert!((hr - 2.0 / 3.0).abs() < 0.01, "Hit rate should be ~0.67, got {}", hr);
    }

    #[test]
    fn test_dht_cache_default_ttl() {
        let mut config = DhtConfig::default();
        config.cache.default_ttl = Some(Duration::from_millis(1));
        let dht = Dht::new(config);

        dht.put("auto_expire", b"data".to_vec());
        assert!(dht.get("auto_expire").is_some());

        std::thread::sleep(Duration::from_millis(10));
        assert!(dht.get("auto_expire").is_none(), "Should expire with default TTL");
    }

    // ── V49: Version, Replicas, Capabilities tests ──────────────

    #[test]
    fn test_version_auto_increment() {
        let dht = Dht::new(DhtConfig::default());

        dht.put("key1", b"v1".to_vec());
        let meta1 = dht.get_with_meta("key1").unwrap();
        assert_eq!(meta1.version, 1);

        dht.put("key1", b"v2".to_vec());
        let meta2 = dht.get_with_meta("key1").unwrap();
        assert_eq!(meta2.version, 2);

        dht.put("key1", b"v3".to_vec());
        let meta3 = dht.get_with_meta("key1").unwrap();
        assert_eq!(meta3.version, 3);

        // Different key starts at 1
        dht.put("key2", b"x".to_vec());
        assert_eq!(dht.get_with_meta("key2").unwrap().version, 1);
    }

    #[test]
    fn test_replica_tracking() {
        let dht = Dht::new(DhtConfig::default());
        dht.put("key1", b"data".to_vec());

        let node_b = NodeId::random();
        let node_c = NodeId::random();

        dht.add_replica("key1", node_b);
        dht.add_replica("key1", node_c);

        let replicas = dht.get_replicas("key1");
        assert_eq!(replicas.len(), 2);
        assert!(replicas.contains(&node_b));
        assert!(replicas.contains(&node_c));

        // Non-existent key returns empty
        assert!(dht.get_replicas("nonexistent").is_empty());
    }

    #[test]
    fn test_node_capabilities_serialize() {
        let mut caps = NodeCapabilities::new("node-A");
        caps.models = vec!["llama3.2:1b".to_string(), "mistral:7b".to_string()];
        caps.features = vec!["rag".to_string(), "security".to_string()];
        caps.has_gpu = true;
        caps.total_memory_mb = 16384;
        caps.current_load = 0.42;

        let json = serde_json::to_string(&caps).unwrap();
        let parsed: NodeCapabilities = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.node_id, "node-A");
        assert_eq!(parsed.models.len(), 2);
        assert!(parsed.has_model("mistral:7b"));
        assert!(parsed.has_feature("rag"));
        assert!(!parsed.has_feature("distributed"));
        assert!(parsed.has_gpu);
        assert!(parsed.has_capacity()); // 0.42 < 0.9
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let data = b"hello world";
        assert_eq!(fnv1a_hash(data), fnv1a_hash(data));
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    // ── V50: Distributed MapReduce tests ────────────────────────

    #[test]
    fn test_worker_registry_register_execute() {
        let mut registry = MapWorkerRegistry::new();

        let map_fn: MapFn = Arc::new(|chunk| {
            vec![MapOutput {
                key: "count".to_string(),
                value: vec![chunk.data.len() as u8],
            }]
        });

        registry.register_map("job1", map_fn);
        assert!(registry.has_job("job1"));
        assert_eq!(registry.job_count(), 1);

        let chunk = DataChunk::new("c1", b"hello".to_vec());
        let result = registry.execute_map("job1", &chunk);
        assert!(result.is_some());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].key, "count");
        assert_eq!(outputs[0].value, vec![5u8]); // "hello".len() = 5
    }

    #[test]
    fn test_worker_registry_unknown_job() {
        let registry = MapWorkerRegistry::new();
        let chunk = DataChunk::new("c1", b"data".to_vec());

        assert!(!registry.has_job("nonexistent"));
        assert!(registry.execute_map("nonexistent", &chunk).is_none());
    }

    #[test]
    fn test_distributed_mapreduce_local_only() {
        // With 0 peers, execute_distributed_with_results behaves like local execute
        let mut job = MapReduceBuilder::word_count();
        job.add_input(DataChunk::new("c1", b"hello world hello".to_vec()));
        job.add_input(DataChunk::new("c2", b"world foo".to_vec()));

        let local_count = job.local_chunk_count(1); // 1 worker = all local
        assert_eq!(local_count, 2);

        let results = job.execute_distributed_with_results(local_count, Vec::new()).unwrap();
        assert!(!results.is_empty());

        // Check word count results
        let hello_count = results.iter().find(|r| r.key == "hello");
        assert!(hello_count.is_some());
    }

    #[test]
    fn test_chunk_distribution_round_robin() {
        let mut job = MapReduceBuilder::word_count();
        for i in 0..9 {
            job.add_input(DataChunk::new(format!("c{}", i), format!("word{}", i).into_bytes()));
        }

        // 3 workers (1 local + 2 peers): 3 chunks each
        let local_count = job.local_chunk_count(3);
        assert_eq!(local_count, 3);

        let peer0_chunks = job.remote_chunks(0, 3);
        assert_eq!(peer0_chunks.len(), 3);

        let peer1_chunks = job.remote_chunks(1, 3);
        assert_eq!(peer1_chunks.len(), 3);

        // 10th peer would get 0 chunks
        let peer9_chunks = job.remote_chunks(9, 3);
        assert!(peer9_chunks.is_empty());
    }

    #[test]
    fn test_self_included_as_worker() {
        let mut job = MapReduceBuilder::word_count();
        for i in 0..6 {
            job.add_input(DataChunk::new(format!("c{}", i), b"hello world".to_vec()));
        }

        // 2 workers: local gets 3, peer gets 3
        let local_count = job.local_chunk_count(2);
        assert_eq!(local_count, 3);

        // Simulate: peer processes its 3 chunks and returns results
        let peer_results: Vec<Vec<MapOutput>> = job.remote_chunks(0, 2)
            .iter()
            .map(|_chunk| {
                vec![
                    MapOutput { key: "hello".to_string(), value: 1u64.to_le_bytes().to_vec() },
                    MapOutput { key: "world".to_string(), value: 1u64.to_le_bytes().to_vec() },
                ]
            })
            .collect();

        let results = job.execute_distributed_with_results(local_count, peer_results).unwrap();
        assert!(!results.is_empty());

        // "hello" should appear 6 times total (3 local + 3 from peer)
        let hello = results.iter().find(|r| r.key == "hello");
        assert!(hello.is_some());
    }

    #[test]
    fn test_distributed_mapreduce_with_peer_results() {
        let map_fn: MapFn = Arc::new(|chunk| {
            let n = chunk.data.len();
            vec![MapOutput { key: "total_bytes".to_string(), value: (n as u64).to_le_bytes().to_vec() }]
        });
        let reduce_fn: ReduceFn = Arc::new(|_key, values| {
            let sum: u64 = values.iter()
                .map(|v| u64::from_le_bytes(v[..8].try_into().unwrap_or([0;8])))
                .sum();
            ReduceOutput { key: _key.to_string(), value: sum.to_le_bytes().to_vec() }
        });

        let mut job = MapReduceJob::new("byte_count", map_fn, reduce_fn);
        job.add_input(DataChunk::new("local1", b"abc".to_vec()));     // 3 bytes
        job.add_input(DataChunk::new("local2", b"de".to_vec()));      // 2 bytes
        job.add_input(DataChunk::new("remote1", b"fghij".to_vec()));  // 5 bytes

        // Local processes first 2 chunks, peer processes 1
        let peer_results = vec![
            vec![MapOutput { key: "total_bytes".to_string(), value: 5u64.to_le_bytes().to_vec() }]
        ];

        let results = job.execute_distributed_with_results(2, peer_results).unwrap();
        assert_eq!(results.len(), 1);

        let total = u64::from_le_bytes(results[0].value[..8].try_into().unwrap());
        assert_eq!(total, 10); // 3 + 2 + 5 = 10
    }
}
