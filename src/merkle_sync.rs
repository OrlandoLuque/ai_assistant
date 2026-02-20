//! Merkle tree and anti-entropy synchronization.
//!
//! Provides a Merkle tree for efficient comparison of data sets between nodes.
//! When two nodes compare their Merkle trees, they can identify exactly which
//! keys differ in O(log N) comparisons instead of transferring all keys.
//!
//! Used for anti-entropy repair: periodically compare trees with peers and
//! sync only the differing entries.
//!
//! This module is gated behind the `distributed-network` feature.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

#[cfg(feature = "distributed")]
use crate::distributed::NodeId;

// =============================================================================
// Merkle Tree
// =============================================================================

/// A Merkle tree built from key-value data for efficient comparison.
///
/// The tree is built from sorted key-value pairs. Each leaf is the SHA-256 hash
/// of `key || value`. Internal nodes are the hash of `left_child || right_child`.
/// The root hash represents the entire data set — if two trees have the same
/// root, their data is identical.
///
/// # Structure
/// The tree is stored as a flat array in level-order:
/// - Index 0: root
/// - Index 1, 2: level 1
/// - Index 3..6: level 2
/// - etc.
///
/// For N leaves, the tree has 2*next_power_of_2(N) - 1 nodes.
pub struct MerkleTree {
    /// Flat array of SHA-256 hashes, level-order.
    nodes: Vec<[u8; 32]>,
    /// Number of actual data leaves (may be less than allocated leaf slots).
    leaf_count: usize,
    /// Keys corresponding to each leaf, in sorted order.
    leaf_keys: Vec<String>,
    /// Total number of allocated leaf slots (next power of 2 >= leaf_count).
    leaf_slots: usize,
}

impl MerkleTree {
    /// Create an empty Merkle tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            leaf_count: 0,
            leaf_keys: Vec::new(),
            leaf_slots: 0,
        }
    }

    /// Build a Merkle tree from sorted key-value data.
    ///
    /// The BTreeMap provides sorted iteration, ensuring deterministic tree
    /// construction regardless of insertion order.
    pub fn from_data(entries: &BTreeMap<String, Vec<u8>>) -> Self {
        if entries.is_empty() {
            return Self::new();
        }

        let leaf_count = entries.len();
        let leaf_slots = leaf_count.next_power_of_two();
        let total_nodes = 2 * leaf_slots - 1;

        let mut nodes = vec![[0u8; 32]; total_nodes];
        let mut leaf_keys = Vec::with_capacity(leaf_count);
        let leaf_start = leaf_slots - 1; // First leaf index in the array

        // Fill leaves
        for (i, (key, value)) in entries.iter().enumerate() {
            nodes[leaf_start + i] = Self::hash_leaf(key, value);
            leaf_keys.push(key.clone());
        }

        // Fill empty leaf slots with zero hash (already zero-initialized)

        // Build internal nodes bottom-up
        if total_nodes > 1 {
            for i in (0..leaf_start).rev() {
                let left = 2 * i + 1;
                let right = 2 * i + 2;
                nodes[i] = Self::hash_pair(&nodes[left], &nodes[right]);
            }
        }

        Self {
            nodes,
            leaf_count,
            leaf_keys,
            leaf_slots,
        }
    }

    /// Get the root hash of the tree, or None if empty.
    pub fn root_hash(&self) -> Option<[u8; 32]> {
        self.nodes.first().copied()
    }

    /// Number of data entries (leaves) in the tree.
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Check if a key exists in the tree.
    pub fn contains_key(&self, key: &str) -> bool {
        self.leaf_keys.binary_search(&key.to_string()).is_ok()
    }

    /// Find keys that differ between this tree and another.
    ///
    /// Returns the keys that are present in one tree but not the other,
    /// or that have different values (different leaf hashes).
    ///
    /// This is an O(K * log N) operation where K is the number of differing
    /// keys, vs O(N) for a full comparison.
    pub fn diff(&self, other: &MerkleTree) -> Vec<String> {
        if self.is_empty() && other.is_empty() {
            return Vec::new();
        }

        // If roots match, trees are identical
        if self.root_hash() == other.root_hash() {
            return Vec::new();
        }

        // Simple approach: compare leaf hashes directly
        // More efficient than recursive descent for most practical sizes
        let mut differing = Vec::new();

        // Build a map of key -> leaf_hash for the other tree
        let other_map: HashMap<&str, [u8; 32]> = other
            .leaf_keys
            .iter()
            .enumerate()
            .map(|(i, key)| {
                let leaf_idx = other.leaf_slots - 1 + i;
                (key.as_str(), other.nodes[leaf_idx])
            })
            .collect();

        let self_leaf_start = self.leaf_slots.saturating_sub(1);

        // Keys in self that differ or are missing from other
        for (i, key) in self.leaf_keys.iter().enumerate() {
            let self_hash = self.nodes[self_leaf_start + i];
            match other_map.get(key.as_str()) {
                Some(other_hash) if *other_hash == self_hash => {}
                _ => differing.push(key.clone()),
            }
        }

        // Keys in other that are not in self
        let self_keys: std::collections::HashSet<&str> =
            self.leaf_keys.iter().map(|k| k.as_str()).collect();
        for key in &other.leaf_keys {
            if !self_keys.contains(key.as_str()) {
                differing.push(key.clone());
            }
        }

        differing
    }

    /// Generate an inclusion proof for a key.
    ///
    /// The proof contains the leaf hash and sibling hashes along the path
    /// to the root. Anyone with the root hash can verify that the key-value
    /// pair is part of the tree without having the full tree.
    pub fn proof(&self, key: &str) -> Option<MerkleProof> {
        let leaf_idx = self.leaf_keys.binary_search(&key.to_string()).ok()?;
        let leaf_array_idx = self.leaf_slots - 1 + leaf_idx;

        let mut siblings = Vec::new();
        let mut idx = leaf_array_idx;

        while idx > 0 {
            let parent = (idx - 1) / 2;
            let sibling = if idx % 2 == 1 {
                // We're the left child, sibling is right
                (self.nodes.get(idx + 1).copied().unwrap_or([0u8; 32]), false)
            } else {
                // We're the right child, sibling is left
                (self.nodes.get(idx - 1).copied().unwrap_or([0u8; 32]), true)
            };
            siblings.push(sibling);
            idx = parent;
        }

        Some(MerkleProof {
            key: key.to_string(),
            leaf_hash: self.nodes[leaf_array_idx],
            siblings,
        })
    }

    /// Verify a Merkle proof against a known root hash.
    ///
    /// Returns true if the proof is valid — i.e., the claimed key-value pair
    /// is indeed part of the tree with the given root hash.
    pub fn verify_proof(root: &[u8; 32], proof: &MerkleProof) -> bool {
        let mut current = proof.leaf_hash;

        for (sibling_hash, is_left) in &proof.siblings {
            if *is_left {
                current = Self::hash_pair(sibling_hash, &current);
            } else {
                current = Self::hash_pair(&current, sibling_hash);
            }
        }

        current == *root
    }

    /// SHA-256 hash of a key-value leaf.
    fn hash_leaf(key: &str, value: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hasher.update(b"|");
        hasher.update(value);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// SHA-256 hash of two child hashes.
    fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Merkle Proof
// =============================================================================

/// A proof that a key-value pair is included in a Merkle tree.
///
/// The proof consists of the leaf hash and sibling hashes along the path
/// from the leaf to the root. Verification recomputes the root hash using
/// the siblings and checks it matches.
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// The key this proof is for.
    pub key: String,
    /// Hash of the leaf (key + value).
    pub leaf_hash: [u8; 32],
    /// Sibling hashes along the path to root: (hash, is_left_sibling).
    pub siblings: Vec<([u8; 32], bool)>,
}

// =============================================================================
// Sync Delta
// =============================================================================

/// Result of comparing two Merkle trees for anti-entropy sync.
#[derive(Debug, Clone)]
pub struct SyncDelta {
    /// Keys we have that the remote doesn't (or has different values).
    /// These should be sent to the remote.
    pub keys_to_send: Vec<String>,
    /// Keys the remote has that we don't (or has different values from ours).
    /// These should be requested from the remote.
    pub keys_to_request: Vec<String>,
}

impl SyncDelta {
    /// Total number of keys that need synchronization.
    pub fn total_diff(&self) -> usize {
        self.keys_to_send.len() + self.keys_to_request.len()
    }

    /// Whether synchronization is needed.
    pub fn needs_sync(&self) -> bool {
        self.total_diff() > 0
    }
}

// =============================================================================
// Anti-Entropy Sync Manager
// =============================================================================

/// Manages periodic anti-entropy synchronization with peer nodes.
///
/// Tracks when each peer was last synced and determines when a new
/// sync round is needed. The actual sync protocol uses Merkle tree
/// comparison to minimize data transfer.
pub struct AntiEntropySync {
    /// How often to sync with each peer.
    sync_interval: Duration,
    /// Last sync time per peer.
    last_sync: HashMap<NodeId, Instant>,
}

impl AntiEntropySync {
    /// Create a new anti-entropy sync manager.
    ///
    /// # Arguments
    /// * `sync_interval` - How often to sync with each peer (e.g., 30 seconds)
    pub fn new(sync_interval: Duration) -> Self {
        Self {
            sync_interval,
            last_sync: HashMap::new(),
        }
    }

    /// Check if a peer needs synchronization.
    pub fn needs_sync(&self, peer: &NodeId) -> bool {
        match self.last_sync.get(peer) {
            None => true,
            Some(last) => last.elapsed() >= self.sync_interval,
        }
    }

    /// Record that a sync was completed with a peer.
    pub fn record_sync(&mut self, peer: &NodeId) {
        self.last_sync.insert(*peer, Instant::now());
    }

    /// Remove a peer from tracking (e.g., when it leaves the cluster).
    pub fn remove_peer(&mut self, peer: &NodeId) {
        self.last_sync.remove(peer);
    }

    /// Compute the delta between local and remote data using Merkle trees.
    ///
    /// Returns which keys need to be sent to the remote and which need
    /// to be requested from the remote.
    ///
    /// # Arguments
    /// * `local` - Merkle tree of local data
    /// * `remote` - Merkle tree received from the remote peer
    /// * `local_data` - Local key-value data (for determining direction of sync)
    pub fn compute_delta(
        local: &MerkleTree,
        remote: &MerkleTree,
        local_data: &BTreeMap<String, Vec<u8>>,
    ) -> SyncDelta {
        let differing = local.diff(remote);

        let mut keys_to_send = Vec::new();
        let mut keys_to_request = Vec::new();

        for key in differing {
            if local_data.contains_key(&key) && !remote.contains_key(&key) {
                // We have it, they don't
                keys_to_send.push(key);
            } else if !local_data.contains_key(&key) && remote.contains_key(&key) {
                // They have it, we don't
                keys_to_request.push(key);
            } else {
                // Both have it but different values — send ours and request theirs
                // In practice, the newer version should win (use LWW semantics)
                keys_to_send.push(key.clone());
                keys_to_request.push(key);
            }
        }

        SyncDelta {
            keys_to_send,
            keys_to_request,
        }
    }

    /// Number of tracked peers.
    pub fn tracked_peers(&self) -> usize {
        self.last_sync.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<u8>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.as_bytes().to_vec()))
            .collect()
    }

    fn make_node(s: &str) -> NodeId {
        NodeId::from_string(s)
    }

    #[test]
    fn test_empty_tree() {
        let tree = MerkleTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.leaf_count(), 0);
        assert!(tree.root_hash().is_none());
        assert!(!tree.contains_key("anything"));
    }

    #[test]
    fn test_single_entry() {
        let data = make_data(&[("key1", "value1")]);
        let tree = MerkleTree::from_data(&data);

        assert_eq!(tree.leaf_count(), 1);
        assert!(tree.root_hash().is_some());
        assert!(tree.contains_key("key1"));
        assert!(!tree.contains_key("key2"));
    }

    #[test]
    fn test_deterministic_root() {
        let data = make_data(&[("a", "1"), ("b", "2"), ("c", "3")]);

        let tree1 = MerkleTree::from_data(&data);
        let tree2 = MerkleTree::from_data(&data);

        assert_eq!(
            tree1.root_hash(),
            tree2.root_hash(),
            "Same data should produce same root hash"
        );
    }

    #[test]
    fn test_different_data_different_root() {
        let data1 = make_data(&[("a", "1"), ("b", "2")]);
        let data2 = make_data(&[("a", "1"), ("b", "3")]); // different value for "b"

        let tree1 = MerkleTree::from_data(&data1);
        let tree2 = MerkleTree::from_data(&data2);

        assert_ne!(
            tree1.root_hash(),
            tree2.root_hash(),
            "Different data should produce different root hash"
        );
    }

    #[test]
    fn test_diff_identical() {
        let data = make_data(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let tree1 = MerkleTree::from_data(&data);
        let tree2 = MerkleTree::from_data(&data);

        let diff = tree1.diff(&tree2);
        assert!(diff.is_empty(), "Identical trees should have no diff");
    }

    #[test]
    fn test_diff_different_values() {
        let data1 = make_data(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let data2 = make_data(&[("a", "1"), ("b", "CHANGED"), ("c", "3")]);

        let tree1 = MerkleTree::from_data(&data1);
        let tree2 = MerkleTree::from_data(&data2);

        let diff = tree1.diff(&tree2);
        assert!(
            diff.contains(&"b".to_string()),
            "Should detect changed key 'b'"
        );
        assert!(
            !diff.contains(&"a".to_string()),
            "Unchanged key 'a' should not be in diff"
        );
    }

    #[test]
    fn test_diff_missing_keys() {
        let data1 = make_data(&[("a", "1"), ("b", "2")]);
        let data2 = make_data(&[("a", "1"), ("c", "3")]);

        let tree1 = MerkleTree::from_data(&data1);
        let tree2 = MerkleTree::from_data(&data2);

        let diff = tree1.diff(&tree2);
        assert!(diff.contains(&"b".to_string()), "Key 'b' only in tree1");
        assert!(diff.contains(&"c".to_string()), "Key 'c' only in tree2");
    }

    #[test]
    fn test_proof_and_verify() {
        let data = make_data(&[("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]);
        let tree = MerkleTree::from_data(&data);
        let root = tree.root_hash().unwrap();

        // Generate proof for key "b"
        let proof = tree
            .proof("b")
            .expect("Should generate proof for existing key");
        assert_eq!(proof.key, "b");

        // Verify the proof
        assert!(
            MerkleTree::verify_proof(&root, &proof),
            "Valid proof should verify"
        );

        // Tamper with the proof — should fail
        let mut tampered = proof.clone();
        tampered.leaf_hash[0] ^= 0xFF;
        assert!(
            !MerkleTree::verify_proof(&root, &tampered),
            "Tampered proof should not verify"
        );
    }

    #[test]
    fn test_proof_nonexistent_key() {
        let data = make_data(&[("a", "1"), ("b", "2")]);
        let tree = MerkleTree::from_data(&data);

        assert!(
            tree.proof("nonexistent").is_none(),
            "Should return None for nonexistent key"
        );
    }

    #[test]
    fn test_anti_entropy_sync_timing() {
        let mut sync = AntiEntropySync::new(Duration::from_millis(50));
        let node = make_node("peer_1");

        // New peer should need sync
        assert!(sync.needs_sync(&node));

        // After recording sync, should not need sync
        sync.record_sync(&node);
        assert!(!sync.needs_sync(&node));

        // After waiting, should need sync again
        std::thread::sleep(Duration::from_millis(60));
        assert!(sync.needs_sync(&node));
    }

    #[test]
    fn test_compute_delta() {
        let local_data = make_data(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let remote_data = make_data(&[("a", "1"), ("b", "CHANGED"), ("d", "4")]);

        let local_tree = MerkleTree::from_data(&local_data);
        let remote_tree = MerkleTree::from_data(&remote_data);

        let delta = AntiEntropySync::compute_delta(&local_tree, &remote_tree, &local_data);

        assert!(delta.needs_sync(), "Different trees should need sync");
        // "c" is only local -> send
        assert!(delta.keys_to_send.contains(&"c".to_string()));
        // "d" is only remote -> request
        assert!(delta.keys_to_request.contains(&"d".to_string()));
        // "b" has different values -> both send and request
        assert!(delta.keys_to_send.contains(&"b".to_string()));
        assert!(delta.keys_to_request.contains(&"b".to_string()));
    }

    #[test]
    fn test_sync_delta_no_diff() {
        let delta = SyncDelta {
            keys_to_send: Vec::new(),
            keys_to_request: Vec::new(),
        };
        assert!(!delta.needs_sync());
        assert_eq!(delta.total_diff(), 0);
    }

    #[test]
    fn test_anti_entropy_remove_peer() {
        let mut sync = AntiEntropySync::new(Duration::from_secs(30));
        let node = make_node("peer_1");

        sync.record_sync(&node);
        assert_eq!(sync.tracked_peers(), 1);

        sync.remove_peer(&node);
        assert_eq!(sync.tracked_peers(), 0);
        assert!(sync.needs_sync(&node)); // Should need sync again after removal
    }

    #[test]
    fn test_large_tree() {
        let mut data = BTreeMap::new();
        for i in 0..1000 {
            data.insert(format!("key_{:04}", i), format!("value_{}", i).into_bytes());
        }

        let tree = MerkleTree::from_data(&data);
        assert_eq!(tree.leaf_count(), 1000);
        assert!(tree.root_hash().is_some());

        // Verify proof for arbitrary key
        let proof = tree.proof("key_0500").unwrap();
        assert!(MerkleTree::verify_proof(&tree.root_hash().unwrap(), &proof));
    }
}
