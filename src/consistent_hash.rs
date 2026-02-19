//! Consistent hash ring with virtual nodes for data partitioning.
//!
//! Implements a consistent hashing ring where each physical node is mapped to multiple
//! virtual nodes (vnodes) distributed around a 64-bit hash space. This ensures:
//! - Uniform data distribution across nodes
//! - Minimal data movement when nodes join/leave (~1/N keys affected)
//! - Configurable replication factor (N distinct physical nodes per key)
//!
//! This module is gated behind the `distributed-network` feature.

use std::collections::{BTreeMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

#[cfg(feature = "distributed")]
use crate::distributed::NodeId;

/// A range of hash space affected by a node change.
#[derive(Debug, Clone)]
pub struct KeyRange {
    /// Start of the affected hash range (inclusive).
    pub start: u64,
    /// End of the affected hash range (exclusive, wraps around).
    pub end: u64,
    /// The node that now owns this range.
    pub node_id: NodeId,
}

/// Consistent hash ring with virtual nodes for data partitioning.
///
/// Each physical node is represented by `vnodes_per_node` positions on a 64-bit
/// hash ring. Keys are mapped to the ring and assigned to the first node found
/// by walking clockwise from the key's hash position.
///
/// For replication, `get_nodes(key, n)` returns `n` distinct physical nodes
/// by continuing clockwise past the primary node.
pub struct ConsistentHashRing {
    /// Hash position -> NodeId mapping (sorted by position via BTreeMap).
    ring: BTreeMap<u64, NodeId>,
    /// Number of virtual nodes per physical node.
    vnodes_per_node: usize,
    /// Set of physical nodes in the ring.
    nodes: HashSet<NodeId>,
    /// Number of distinct physical nodes to return for each key.
    replication_factor: usize,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring.
    ///
    /// # Arguments
    /// * `vnodes_per_node` - Number of virtual nodes per physical node (64 is typical)
    /// * `replication_factor` - How many distinct physical nodes should store each key
    pub fn new(vnodes_per_node: usize, replication_factor: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            vnodes_per_node: vnodes_per_node.max(1),
            nodes: HashSet::new(),
            replication_factor: replication_factor.max(1),
        }
    }

    /// Add a physical node to the ring, creating `vnodes_per_node` virtual nodes.
    ///
    /// Returns the hash ranges that are now owned by this node (affected ranges
    /// where data may need to be migrated).
    pub fn add_node(&mut self, node_id: NodeId) -> Vec<KeyRange> {
        if self.nodes.contains(&node_id) {
            return Vec::new();
        }
        self.nodes.insert(node_id);

        let mut affected = Vec::new();
        for i in 0..self.vnodes_per_node {
            let hash = Self::hash_vnode(&node_id, i);
            self.ring.insert(hash, node_id);

            // The affected range is from the previous vnode to this one
            if let Some((&prev_hash, _)) = self.ring.range(..hash).next_back() {
                affected.push(KeyRange {
                    start: prev_hash.wrapping_add(1),
                    end: hash,
                    node_id,
                });
            } else if let Some((&last_hash, _)) = self.ring.iter().next_back() {
                // Wraps around: from last position to this one
                if last_hash != hash {
                    affected.push(KeyRange {
                        start: last_hash.wrapping_add(1),
                        end: hash,
                        node_id,
                    });
                }
            }
        }

        affected
    }

    /// Remove a physical node from the ring.
    ///
    /// Returns the hash ranges that need to be reassigned to other nodes.
    pub fn remove_node(&mut self, node_id: &NodeId) -> Vec<KeyRange> {
        if !self.nodes.remove(node_id) {
            return Vec::new();
        }

        let mut affected = Vec::new();
        let positions: Vec<u64> = self.ring.iter()
            .filter(|(_, nid)| *nid == node_id)
            .map(|(&pos, _)| pos)
            .collect();

        for pos in &positions {
            self.ring.remove(pos);
        }

        // Calculate affected ranges: each removed vnode's range is now handled
        // by the next node clockwise
        for &pos in &positions {
            if let Some(next_node) = self.find_next_node(pos) {
                affected.push(KeyRange {
                    start: pos,
                    end: pos,
                    node_id: next_node,
                });
            }
        }

        affected
    }

    /// Get the primary node responsible for a key.
    pub fn get_node(&self, key: &str) -> Option<NodeId> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = Self::hash_key(key);
        self.find_next_node(hash)
    }

    /// Get `n` distinct physical nodes responsible for a key (primary + replicas).
    ///
    /// Walks clockwise around the ring from the key's hash position,
    /// collecting distinct physical nodes until `n` are found or the ring
    /// is exhausted.
    ///
    /// # Arguments
    /// * `key` - The key to look up
    /// * `n` - Number of distinct physical nodes to return
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = Self::hash_key(key);
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let ring_len = self.ring.len();

        // Collect entries starting from the hash position going clockwise
        let after = self.ring.range(hash..);
        let before = self.ring.range(..hash);
        let iter = after.chain(before);

        for (_, node_id) in iter {
            if seen.insert(*node_id) {
                result.push(*node_id);
                if result.len() >= n {
                    break;
                }
            }
            // Safety: if we've checked all vnodes, stop
            if seen.len() >= self.nodes.len() || result.len() >= ring_len {
                break;
            }
        }

        result
    }

    /// Check if a node is in the ring.
    pub fn contains_node(&self, node_id: &NodeId) -> bool {
        self.nodes.contains(node_id)
    }

    /// Number of physical nodes in the ring.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of virtual nodes in the ring.
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Get all physical nodes in the ring.
    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.iter().copied().collect()
    }

    /// Get the replication factor.
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }

    /// Deterministic hash of a string key to a position on the ring.
    fn hash_key(key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Deterministic hash of a (node_id, vnode_index) pair to a ring position.
    fn hash_vnode(node_id: &NodeId, vnode_index: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        node_id.0.hash(&mut hasher);
        vnode_index.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the next node clockwise from a hash position.
    fn find_next_node(&self, hash: u64) -> Option<NodeId> {
        // Look for the first vnode at or after the hash
        if let Some((_, node_id)) = self.ring.range(hash..).next() {
            return Some(*node_id);
        }
        // Wrap around to the beginning
        self.ring.values().next().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(s: &str) -> NodeId {
        NodeId::from_string(s)
    }

    #[test]
    fn test_empty_ring() {
        let ring = ConsistentHashRing::new(16, 3);
        assert_eq!(ring.node_count(), 0);
        assert_eq!(ring.vnode_count(), 0);
        assert!(ring.get_node("test_key").is_none());
        assert!(ring.get_nodes("test_key", 3).is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut ring = ConsistentHashRing::new(16, 3);
        let node = make_node("node_a");
        ring.add_node(node);

        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 16);
        assert!(ring.contains_node(&node));

        // All keys should map to the single node
        for i in 0..100 {
            let key = format!("key_{}", i);
            assert_eq!(ring.get_node(&key), Some(node));
        }
    }

    #[test]
    fn test_add_remove_nodes() {
        let mut ring = ConsistentHashRing::new(16, 3);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");
        let node_c = make_node("node_c");

        ring.add_node(node_a);
        ring.add_node(node_b);
        ring.add_node(node_c);

        assert_eq!(ring.node_count(), 3);
        assert_eq!(ring.vnode_count(), 48); // 3 * 16

        ring.remove_node(&node_b);
        assert_eq!(ring.node_count(), 2);
        assert_eq!(ring.vnode_count(), 32); // 2 * 16
        assert!(!ring.contains_node(&node_b));
    }

    #[test]
    fn test_get_nodes_distinct() {
        let mut ring = ConsistentHashRing::new(64, 3);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");
        let node_c = make_node("node_c");

        ring.add_node(node_a);
        ring.add_node(node_b);
        ring.add_node(node_c);

        // get_nodes should return distinct physical nodes
        for i in 0..50 {
            let key = format!("test_key_{}", i);
            let nodes = ring.get_nodes(&key, 3);
            assert_eq!(nodes.len(), 3, "Should return 3 distinct nodes for key {}", key);

            // All nodes should be unique
            let unique: HashSet<_> = nodes.iter().collect();
            assert_eq!(unique.len(), 3, "All returned nodes should be unique");
        }
    }

    #[test]
    fn test_get_nodes_limited_by_ring_size() {
        let mut ring = ConsistentHashRing::new(16, 3);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");

        ring.add_node(node_a);
        ring.add_node(node_b);

        // Requesting 5 nodes but only 2 exist
        let nodes = ring.get_nodes("key", 5);
        assert_eq!(nodes.len(), 2, "Should return only available nodes");
    }

    #[test]
    fn test_key_distribution() {
        let mut ring = ConsistentHashRing::new(64, 1);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");
        let node_c = make_node("node_c");

        ring.add_node(node_a);
        ring.add_node(node_b);
        ring.add_node(node_c);

        // Count key assignments
        let mut counts = std::collections::HashMap::new();
        let total_keys = 3000;
        for i in 0..total_keys {
            let key = format!("data_key_{}", i);
            let node = ring.get_node(&key).unwrap();
            *counts.entry(node).or_insert(0u32) += 1;
        }

        // Each node should get roughly 1/3 of keys (with some variance)
        // With 64 vnodes and 3000 keys, expect ~1000 +/- 300 per node
        for (node, count) in &counts {
            assert!(
                *count > 500 && *count < 1500,
                "Node {:?} got {} keys, expected ~1000 (distribution should be reasonably uniform)",
                node.to_hex(),
                count
            );
        }
    }

    #[test]
    fn test_minimal_redistribution() {
        let mut ring = ConsistentHashRing::new(64, 1);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");

        ring.add_node(node_a);
        ring.add_node(node_b);

        // Record initial assignments
        let total_keys = 1000;
        let initial: Vec<_> = (0..total_keys)
            .map(|i| {
                let key = format!("key_{}", i);
                (key.clone(), ring.get_node(&key).unwrap())
            })
            .collect();

        // Add a third node
        let node_c = make_node("node_c");
        ring.add_node(node_c);

        // Count how many keys moved
        let mut moved = 0;
        for (key, old_node) in &initial {
            let new_node = ring.get_node(key).unwrap();
            if new_node != *old_node {
                moved += 1;
            }
        }

        // Adding 1 node to 3 should move roughly 1/3 of keys
        // Allow generous range for statistical variation
        let expected_max = (total_keys as f64 * 0.55) as usize;
        assert!(
            moved < expected_max,
            "Adding a node moved {} keys out of {}, expected < {} (minimal redistribution)",
            moved,
            total_keys,
            expected_max
        );
    }

    #[test]
    fn test_duplicate_add() {
        let mut ring = ConsistentHashRing::new(16, 3);
        let node = make_node("node_a");

        ring.add_node(node);
        let affected = ring.add_node(node); // Duplicate

        assert!(affected.is_empty(), "Duplicate add should return empty");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 16);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut ring = ConsistentHashRing::new(16, 3);
        let node = make_node("node_a");
        let affected = ring.remove_node(&node);
        assert!(affected.is_empty());
    }

    #[test]
    fn test_affected_ranges_on_add() {
        let mut ring = ConsistentHashRing::new(8, 1);
        let node_a = make_node("node_a");
        let affected = ring.add_node(node_a);

        // First node: all ranges map to it
        assert!(!affected.is_empty(), "Adding first node should produce affected ranges");
        for range in &affected {
            assert_eq!(range.node_id, node_a);
        }
    }

    #[test]
    fn test_deterministic_hashing() {
        let mut ring1 = ConsistentHashRing::new(32, 2);
        let mut ring2 = ConsistentHashRing::new(32, 2);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");

        ring1.add_node(node_a);
        ring1.add_node(node_b);
        ring2.add_node(node_a);
        ring2.add_node(node_b);

        // Same ring configuration should produce same results
        for i in 0..100 {
            let key = format!("key_{}", i);
            assert_eq!(
                ring1.get_node(&key),
                ring2.get_node(&key),
                "Same ring should produce same assignment for key {}",
                key
            );
        }
    }

    #[test]
    fn test_nodes_list() {
        let mut ring = ConsistentHashRing::new(8, 1);
        let node_a = make_node("node_a");
        let node_b = make_node("node_b");
        let node_c = make_node("node_c");

        ring.add_node(node_a);
        ring.add_node(node_b);
        ring.add_node(node_c);

        let nodes = ring.nodes();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&node_a));
        assert!(nodes.contains(&node_b));
        assert!(nodes.contains(&node_c));
    }
}
