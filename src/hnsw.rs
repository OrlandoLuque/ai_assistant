//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! Pure-Rust implementation of the HNSW algorithm for efficient vector similarity search.
//! Provides sub-linear query time compared to brute-force search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::{AiError, AiResult};
use crate::vector_db::{
    BackendInfo, DistanceMetric, MetadataFilter, StoredVector, VectorDb, VectorDbConfig,
    VectorSearchResult,
};

/// HNSW index configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node at layer 0 (default: 16).
    pub m: usize,
    /// Maximum number of connections per node at higher layers (default: 32).
    pub m_max: usize,
    /// Size of the dynamic candidate list during construction (default: 200).
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search (default: 50).
    pub ef_search: usize,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// A node in the HNSW graph, holding the vector, metadata, and graph connections.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Unique identifier for this node.
    pub id: String,
    /// The vector data.
    pub vector: Vec<f32>,
    /// Associated metadata.
    pub metadata: serde_json::Value,
    /// Timestamp when inserted (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Connections at each layer. connections[layer] = set of neighbor indices.
    connections: Vec<HashSet<usize>>,
    /// The maximum layer this node exists on.
    #[allow(dead_code)]
    max_layer: usize,
    /// Tombstone flag for lazy deletion.
    deleted: bool,
}

/// Candidate for priority queue (min-heap by distance).
#[derive(Debug, Clone)]
struct Candidate {
    index: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (BinaryHeap is a max-heap)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-distance candidate (for maintaining the farthest neighbor).
#[derive(Debug, Clone)]
struct FarCandidate {
    index: usize,
    distance: f32,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for FarCandidate {}

impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    id_map: HashMap<String, usize>,
    entry_point: Option<usize>,
    current_max_level: usize,
}

impl HnswIndex {
    /// Create a new HNSW index with the given configuration.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            id_map: HashMap::new(),
            entry_point: None,
            current_max_level: 0,
        }
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) {
        if let Some(&existing_idx) = self.id_map.get(id) {
            // Update existing node
            self.nodes[existing_idx].vector = vector;
            self.nodes[existing_idx].metadata = metadata;
            self.nodes[existing_idx].deleted = false;
            return;
        }

        let level = self.random_level(id);
        let idx = self.nodes.len();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let node = HnswNode {
            id: id.to_string(),
            vector,
            metadata,
            timestamp: now,
            connections: (0..=level).map(|_| HashSet::new()).collect(),
            max_layer: level,
            deleted: false,
        };

        self.nodes.push(node);
        self.id_map.insert(id.to_string(), idx);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.current_max_level = level;
            return;
        }

        let entry = self.entry_point.expect("entry_point guaranteed by early return above");

        // Navigate from top layer to the node's insertion layer
        let mut current = entry;
        let top_layer = self.current_max_level;

        // Greedy search from top to level+1
        for layer in (level + 1..=top_layer).rev() {
            current = self.greedy_closest(current, idx, layer);
        }

        // Insert at each layer from min(level, top_layer) down to 0
        let start_layer = level.min(top_layer);
        for layer in (0..=start_layer).rev() {
            let neighbors = self.search_layer(current, idx, self.config.ef_construction, layer);
            let selected = self.select_neighbors(&neighbors, layer);

            // Connect new node to selected neighbors
            for &neighbor_idx in &selected {
                self.nodes[idx].connections[layer].insert(neighbor_idx);
                // Also add reverse connection
                if layer < self.nodes[neighbor_idx].connections.len() {
                    self.nodes[neighbor_idx].connections[layer].insert(idx);
                    // Prune if exceeds max connections
                    let max_conn = if layer == 0 {
                        self.config.m * 2
                    } else {
                        self.config.m_max
                    };
                    if self.nodes[neighbor_idx].connections[layer].len() > max_conn {
                        self.prune_connections(neighbor_idx, layer, max_conn);
                    }
                }
            }

            if !neighbors.is_empty() {
                current = neighbors[0];
            }
        }

        // Update entry point if new node has higher level
        if level > self.current_max_level {
            self.entry_point = Some(idx);
            self.current_max_level = level;
        }
    }

    /// Search for the k nearest neighbors to the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<VectorSearchResult> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let entry = self.entry_point.expect("entry_point guaranteed by is_none check above");

        // Navigate from top to layer 1
        let mut current = entry;
        for layer in (1..=self.current_max_level).rev() {
            current = self.greedy_closest_query(current, query, layer);
        }

        // Search at layer 0
        let candidates = self.search_layer_query(current, query, self.config.ef_search.max(k), 0);

        // Collect results, filtering tombstones
        let mut results: Vec<VectorSearchResult> = candidates
            .iter()
            .filter(|&&idx| !self.nodes[idx].deleted)
            .map(|&idx| {
                let dist = self.distance(query, &self.nodes[idx].vector);
                let score = self.config.metric.to_similarity(dist);
                VectorSearchResult {
                    id: self.nodes[idx].id.clone(),
                    score,
                    metadata: self.nodes[idx].metadata.clone(),
                    vector: Some(self.nodes[idx].vector.clone()),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search with metadata filtering.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filters: &[MetadataFilter],
    ) -> Vec<VectorSearchResult> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let entry = self.entry_point.expect("entry_point guaranteed by is_none check above");

        // Navigate from top to layer 1
        let mut current = entry;
        for layer in (1..=self.current_max_level).rev() {
            current = self.greedy_closest_query(current, query, layer);
        }

        // Search at layer 0 with extra ef to compensate for filtering
        let ef = (self.config.ef_search.max(k) * 4).min(self.nodes.len());
        let candidates = self.search_layer_query(current, query, ef, 0);

        // Collect results, filtering tombstones and applying metadata filters
        let mut results: Vec<VectorSearchResult> = candidates
            .iter()
            .filter(|&&idx| {
                let node = &self.nodes[idx];
                !node.deleted && filters.iter().all(|f| f.matches(&node.metadata))
            })
            .map(|&idx| {
                let dist = self.distance(query, &self.nodes[idx].vector);
                let score = self.config.metric.to_similarity(dist);
                VectorSearchResult {
                    id: self.nodes[idx].id.clone(),
                    score,
                    metadata: self.nodes[idx].metadata.clone(),
                    vector: Some(self.nodes[idx].vector.clone()),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Mark a vector as deleted (lazy tombstone).
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(&idx) = self.id_map.get(id) {
            self.nodes[idx].deleted = true;
            true
        } else {
            false
        }
    }

    /// Get the number of active (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a stored vector by ID.
    pub fn get(&self, id: &str) -> Option<&HnswNode> {
        self.id_map.get(id).and_then(|&idx| {
            let node = &self.nodes[idx];
            if node.deleted {
                None
            } else {
                Some(node)
            }
        })
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.id_map.clear();
        self.entry_point = None;
        self.current_max_level = 0;
    }

    /// Export all active (non-deleted) vectors as StoredVector.
    pub fn export_all(&self) -> Vec<StoredVector> {
        self.nodes
            .iter()
            .filter(|n| !n.deleted)
            .map(|n| StoredVector {
                id: n.id.clone(),
                vector: n.vector.clone(),
                metadata: n.metadata.clone(),
                timestamp: n.timestamp,
            })
            .collect()
    }

    // === Private methods ===

    /// Generate a random level for a new node using a hash-based approach.
    fn random_level(&self, id: &str) -> usize {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset
        for byte in id.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
        // Also mix in the current node count for variety
        hash ^= self.nodes.len() as u64;
        hash = hash.wrapping_mul(0x100000001b3);

        let ml = 1.0 / (self.config.m as f64).ln();
        let r = (hash as f64) / (u64::MAX as f64);
        let level = (-r.ln() * ml).floor() as usize;
        level.min(16) // Cap at 16 layers
    }

    /// Greedy search for closest node to target at a given layer.
    fn greedy_closest(&self, start: usize, target: usize, layer: usize) -> usize {
        let target_vec = &self.nodes[target].vector;
        self.greedy_closest_query(start, target_vec, layer)
    }

    /// Greedy search for closest node to query vector at a given layer.
    fn greedy_closest_query(&self, start: usize, query: &[f32], layer: usize) -> usize {
        let mut current = start;
        let mut current_dist = self.distance(query, &self.nodes[current].vector);

        loop {
            let mut changed = false;
            if layer < self.nodes[current].connections.len() {
                for &neighbor in &self.nodes[current].connections[layer] {
                    if neighbor < self.nodes.len() {
                        let dist = self.distance(query, &self.nodes[neighbor].vector);
                        if dist < current_dist {
                            current_dist = dist;
                            current = neighbor;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        current
    }

    /// Search a layer for ef nearest neighbors to target node.
    fn search_layer(&self, entry: usize, target: usize, ef: usize, layer: usize) -> Vec<usize> {
        let target_vec = &self.nodes[target].vector;
        self.search_layer_query(entry, target_vec, ef, layer)
    }

    /// Search a layer for ef nearest neighbors to query vector.
    fn search_layer_query(
        &self,
        entry: usize,
        query: &[f32],
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // min-heap
        let mut results = BinaryHeap::new(); // max-heap (farthest first)

        let entry_dist = self.distance(query, &self.nodes[entry].vector);
        visited.insert(entry);
        candidates.push(Candidate {
            index: entry,
            distance: entry_dist,
        });
        results.push(FarCandidate {
            index: entry,
            distance: entry_dist,
        });

        while let Some(Candidate {
            index: current,
            distance: current_dist,
        }) = candidates.pop()
        {
            // If current is farther than the farthest result, stop
            if let Some(farthest) = results.peek() {
                if current_dist > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            if layer < self.nodes[current].connections.len() {
                for &neighbor in &self.nodes[current].connections[layer] {
                    if neighbor < self.nodes.len() && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        let dist = self.distance(query, &self.nodes[neighbor].vector);

                        let should_add = results.len() < ef || {
                            if let Some(farthest) = results.peek() {
                                dist < farthest.distance
                            } else {
                                true
                            }
                        };

                        if should_add {
                            candidates.push(Candidate {
                                index: neighbor,
                                distance: dist,
                            });
                            results.push(FarCandidate {
                                index: neighbor,
                                distance: dist,
                            });
                            if results.len() > ef {
                                results.pop(); // Remove farthest
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| c.index)
            .collect()
    }

    /// Select neighbors from candidates (simple selection: take closest m).
    fn select_neighbors(&self, candidates: &[usize], layer: usize) -> Vec<usize> {
        let max_conn = if layer == 0 {
            self.config.m * 2
        } else {
            self.config.m
        };
        if candidates.len() <= max_conn {
            return candidates.to_vec();
        }
        candidates[..max_conn].to_vec()
    }

    /// Prune connections for a node to keep only the closest neighbors.
    fn prune_connections(&mut self, node_idx: usize, layer: usize, max_conn: usize) {
        let node_vec = self.nodes[node_idx].vector.clone();
        let mut neighbors: Vec<(usize, f32)> = self.nodes[node_idx].connections[layer]
            .iter()
            .filter(|&&idx| idx < self.nodes.len())
            .map(|&idx| (idx, self.distance(&node_vec, &self.nodes[idx].vector)))
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbors.truncate(max_conn);

        self.nodes[node_idx].connections[layer] =
            neighbors.into_iter().map(|(idx, _)| idx).collect();
    }

    /// Compute distance between two vectors using the configured metric.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.config.metric.calculate(a, b)
    }
}

/// HNSW-backed vector database implementing the VectorDb trait.
pub struct HnswVectorDb {
    index: HnswIndex,
    config: VectorDbConfig,
}

impl HnswVectorDb {
    /// Create a new HNSW vector database with the given configuration.
    pub fn new(config: VectorDbConfig) -> Self {
        let hnsw_config = HnswConfig {
            metric: config.distance_metric,
            ..Default::default()
        };
        Self {
            index: HnswIndex::new(hnsw_config),
            config,
        }
    }

    /// Create a new HNSW vector database with custom HNSW parameters.
    pub fn with_hnsw_config(config: VectorDbConfig, hnsw_config: HnswConfig) -> Self {
        Self {
            index: HnswIndex::new(hnsw_config),
            config,
        }
    }

    /// Get a reference to the underlying HNSW index.
    pub fn index(&self) -> &HnswIndex {
        &self.index
    }
}

impl VectorDb for HnswVectorDb {
    fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: serde_json::Value) -> AiResult<()> {
        if vector.len() != self.config.dimensions {
            return Err(AiError::Other(format!(
                "Expected {} dimensions, got {}",
                self.config.dimensions,
                vector.len()
            )));
        }
        self.index.insert(id, vector, metadata);
        Ok(())
    }

    fn insert_batch(
        &mut self,
        vectors: Vec<(String, Vec<f32>, serde_json::Value)>,
    ) -> AiResult<usize> {
        let mut inserted = 0;
        for (id, vector, metadata) in vectors {
            if self.insert(&id, vector, metadata).is_ok() {
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&[MetadataFilter]>,
    ) -> AiResult<Vec<VectorSearchResult>> {
        if let Some(filters) = filter {
            Ok(self.index.search_filtered(query, limit, filters))
        } else {
            Ok(self.index.search(query, limit))
        }
    }

    fn get(&self, id: &str) -> AiResult<Option<StoredVector>> {
        Ok(self.index.get(id).map(|node| StoredVector {
            id: node.id.clone(),
            vector: node.vector.clone(),
            metadata: node.metadata.clone(),
            timestamp: node.timestamp,
        }))
    }

    fn delete(&mut self, id: &str) -> AiResult<bool> {
        Ok(self.index.remove(id))
    }

    fn count(&self) -> usize {
        self.index.len()
    }

    fn clear(&mut self) -> AiResult<()> {
        self.index.clear();
        Ok(())
    }

    fn health_check(&self) -> AiResult<bool> {
        Ok(true)
    }

    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "HNSW",
            tier: 0,
            supports_persistence: false,
            supports_filtering: true,
            supports_export: true,
            max_recommended_vectors: Some(1_000_000),
        }
    }

    fn export_all(&self) -> AiResult<Vec<StoredVector>> {
        Ok(self.index.export_all())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize) -> VectorDbConfig {
        VectorDbConfig {
            dimensions: dim,
            collection_name: "test".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({}));
        index.insert("b", vec![0.9, 0.1, 0.0], serde_json::json!({}));
        index.insert("c", vec![0.0, 1.0, 0.0], serde_json::json!({}));

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // Exact match
    }

    #[test]
    fn test_hnsw_remove() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        index.insert("b", vec![0.0, 1.0], serde_json::json!({}));
        assert_eq!(index.len(), 2);

        assert!(index.remove("a"));
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_hnsw_empty_search() {
        let index = HnswIndex::new(HnswConfig::default());
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_get() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert(
            "test",
            vec![1.0, 2.0, 3.0],
            serde_json::json!({"key": "value"}),
        );

        let node = index.get("test").unwrap();
        assert_eq!(node.id, "test");
        assert_eq!(node.vector, vec![1.0, 2.0, 3.0]);

        assert!(index.get("nonexistent").is_none());
    }

    #[test]
    fn test_hnsw_len_and_is_empty() {
        let mut index = HnswIndex::new(HnswConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        index.insert("a", vec![1.0], serde_json::json!({}));
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hnsw_update_existing() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({"v": 1}));
        index.insert("a", vec![0.0, 1.0], serde_json::json!({"v": 2}));

        assert_eq!(index.len(), 1);
        let node = index.get("a").unwrap();
        assert_eq!(node.vector, vec![0.0, 1.0]);
    }

    #[test]
    fn test_hnsw_euclidean_metric() {
        let config = HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let mut index = HnswIndex::new(config);
        index.insert("origin", vec![0.0, 0.0], serde_json::json!({}));
        index.insert("close", vec![1.0, 0.0], serde_json::json!({}));
        index.insert("far", vec![10.0, 10.0], serde_json::json!({}));

        let results = index.search(&[0.0, 0.0], 1);
        assert_eq!(results[0].id, "origin");
    }

    #[test]
    fn test_hnsw_many_vectors() {
        let mut index = HnswIndex::new(HnswConfig {
            ef_construction: 50,
            ef_search: 30,
            ..Default::default()
        });

        // Insert 100 vectors
        for i in 0..100 {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / 100.0;
            let vec = vec![angle.cos(), angle.sin()];
            index.insert(&format!("v{}", i), vec, serde_json::json!({"i": i}));
        }

        assert_eq!(index.len(), 100);

        // Search should return results
        let results = index.search(&[1.0, 0.0], 5);
        assert_eq!(results.len(), 5);
        // v0 should be closest to [1.0, 0.0] (angle=0)
        assert_eq!(results[0].id, "v0");
    }

    #[test]
    fn test_hnsw_db_trait() {
        let config = make_config(3);
        let mut db = HnswVectorDb::new(config);

        db.insert("a", vec![1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("b", vec![0.0, 1.0, 0.0], serde_json::json!({}))
            .unwrap();

        assert_eq!(db.count(), 2);

        let results = db.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, "a");

        let stored = db.get("a").unwrap().unwrap();
        assert_eq!(stored.vector, vec![1.0, 0.0, 0.0]);

        assert!(db.delete("a").unwrap());
        assert_eq!(db.count(), 1);
    }

    #[test]
    fn test_hnsw_db_dimension_check() {
        let config = make_config(3);
        let mut db = HnswVectorDb::new(config);
        let result = db.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_hnsw_db_health_check() {
        let config = make_config(3);
        let db = HnswVectorDb::new(config);
        assert!(db.health_check().unwrap());
    }

    #[test]
    fn test_hnsw_db_backend_info() {
        let config = make_config(3);
        let db = HnswVectorDb::new(config);
        let info = db.backend_info();
        assert_eq!(info.name, "HNSW");
        assert_eq!(info.tier, 0);
        assert!(info.supports_filtering);
        assert!(info.supports_export);
    }

    #[test]
    fn test_hnsw_remove_nonexistent() {
        let mut index = HnswIndex::new(HnswConfig::default());
        assert!(!index.remove("doesnt_exist"));
    }

    #[test]
    fn test_hnsw_search_after_remove_and_reinsert() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        index.insert("b", vec![0.0, 1.0], serde_json::json!({}));

        index.remove("a");
        index.insert("c", vec![0.9, 0.1], serde_json::json!({}));

        let results = index.search(&[1.0, 0.0], 2);
        // 'a' should not be in results (deleted)
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_hnsw_dot_product_metric() {
        let config = HnswConfig {
            metric: DistanceMetric::DotProduct,
            ..Default::default()
        };
        let mut index = HnswIndex::new(config);
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        index.insert("b", vec![0.5, 0.5], serde_json::json!({}));

        let results = index.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hnsw_single_element() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("only", vec![42.0], serde_json::json!({}));

        let results = index.search(&[42.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "only");
    }

    #[test]
    fn test_hnsw_clear() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        index.insert("b", vec![0.0, 1.0], serde_json::json!({}));
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn test_hnsw_db_clear() {
        let config = make_config(2);
        let mut db = HnswVectorDb::new(config);
        db.insert("a", vec![1.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("b", vec![0.0, 1.0], serde_json::json!({}))
            .unwrap();
        assert_eq!(db.count(), 2);

        db.clear().unwrap();
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn test_hnsw_db_insert_batch() {
        let config = make_config(2);
        let mut db = HnswVectorDb::new(config);
        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0], serde_json::json!({})),
            ("b".to_string(), vec![0.0, 1.0], serde_json::json!({})),
            ("bad".to_string(), vec![1.0], serde_json::json!({})), // Wrong dim
        ];
        let inserted = db.insert_batch(vectors).unwrap();
        assert_eq!(inserted, 2);
        assert_eq!(db.count(), 2);
    }

    #[test]
    fn test_hnsw_db_export_all() {
        let config = make_config(2);
        let mut db = HnswVectorDb::new(config);
        db.insert("a", vec![1.0, 0.0], serde_json::json!({"k": "v"}))
            .unwrap();
        db.insert("b", vec![0.0, 1.0], serde_json::json!({}))
            .unwrap();

        let exported = db.export_all().unwrap();
        assert_eq!(exported.len(), 2);

        let ids: HashSet<String> = exported.iter().map(|v| v.id.clone()).collect();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
    }

    #[test]
    fn test_hnsw_search_zero_k() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        let results = index.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_db_with_custom_config() {
        let vdb_config = make_config(2);
        let hnsw_config = HnswConfig {
            m: 8,
            m_max: 16,
            ef_construction: 100,
            ef_search: 25,
            metric: DistanceMetric::Euclidean,
        };
        let mut db = HnswVectorDb::with_hnsw_config(vdb_config, hnsw_config);

        db.insert("a", vec![0.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("b", vec![1.0, 0.0], serde_json::json!({}))
            .unwrap();
        db.insert("c", vec![10.0, 10.0], serde_json::json!({}))
            .unwrap();

        let results = db.search(&[0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_hnsw_get_deleted_returns_none() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a", vec![1.0, 0.0], serde_json::json!({}));
        index.remove("a");
        assert!(index.get("a").is_none());
    }
}
