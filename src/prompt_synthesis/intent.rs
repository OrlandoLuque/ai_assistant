//! Intent clustering: adaptive 1..64 clusters over query embeddings.
//!
//! The manager keeps a small K-means-like set of centroids. On each new
//! query we compute cosine similarity to the nearest centroid; if below a
//! threshold we grow a new cluster (up to `MAX_CLUSTERS`). If clusters
//! stay empty for a configurable window we shrink. Deterministic by choice
//! of tiebreak (first index wins).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::defaults::{MAX_CLUSTERS, MIN_CLUSTERS};

/// Opaque id for a cluster. Monotonic within a manager lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentClusterId(pub(crate) u32);

impl fmt::Display for IntentClusterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cluster_{}", self.0)
    }
}

/// A normalized embedding vector. Owners are responsible for normalization —
/// `IntentClusterManager` treats vectors as already unit-length for the
/// purposes of cosine similarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentEmbedding {
    pub vector: Vec<f32>,
}

impl IntentEmbedding {
    /// Construct from any slice; caller should normalize via `normalized()`.
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector }
    }

    /// Return a unit-length copy (L2). Zero vectors stay zero.
    pub fn normalized(&self) -> Self {
        let norm = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return self.clone();
        }
        Self {
            vector: self.vector.iter().map(|x| x / norm).collect(),
        }
    }

    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Cosine similarity, assuming both vectors are L2-normalized.
    pub fn cosine(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// One cluster: a centroid plus membership statistics. Centroids are
/// updated via running average after each `assign`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntentCluster {
    pub id: IntentClusterId,
    pub centroid: IntentEmbedding,
    /// Number of embeddings assigned so far.
    pub count: u64,
    /// Optional label for UI/ledger. Derived by the caller (e.g. top keyword).
    pub label: Option<String>,
}

/// Configuration for `IntentClusterManager`.
#[derive(Debug, Clone)]
pub struct IntentClusterManagerConfig {
    /// Minimum cosine similarity to an existing centroid to reuse it.
    /// Below this we grow a new cluster (subject to max).
    pub grow_threshold: f32,
    /// Maximum clusters kept alive. Defaults to 64 per the plan.
    pub max_clusters: usize,
    /// Minimum clusters — never shrink below this. Defaults to 1.
    pub min_clusters: usize,
    /// After this many consecutive `assign` calls without any hit, a cluster
    /// is eligible for shrink. 0 disables shrinking.
    pub shrink_idle_count: u64,
}

impl Default for IntentClusterManagerConfig {
    fn default() -> Self {
        Self {
            grow_threshold: 0.80,
            max_clusters: MAX_CLUSTERS,
            min_clusters: MIN_CLUSTERS,
            shrink_idle_count: 0,
        }
    }
}

/// Adaptive cluster manager. Not thread-safe — wrap in `Mutex`/`RwLock` if
/// shared across tasks.
#[derive(Debug)]
pub struct IntentClusterManager {
    cfg: IntentClusterManagerConfig,
    clusters: Vec<IntentCluster>,
    next_id: u32,
    /// Per-cluster "steps since last hit" counter, keyed by cluster id.
    idle_counter: HashMap<u32, u64>,
    /// Monotonic clock used to drive the idle counter.
    step: u64,
}

impl IntentClusterManager {
    pub fn new(cfg: IntentClusterManagerConfig) -> Self {
        Self {
            cfg,
            clusters: Vec::new(),
            next_id: 0,
            idle_counter: HashMap::new(),
            step: 0,
        }
    }

    /// Number of live clusters.
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Snapshot of all live clusters (cloned).
    pub fn clusters(&self) -> Vec<IntentCluster> {
        self.clusters.clone()
    }

    /// Find the cluster for an embedding, creating one if necessary. Returns
    /// the assigned cluster id. Updates centroid as a running mean.
    pub fn assign(&mut self, embedding: &IntentEmbedding) -> IntentClusterId {
        self.step = self.step.saturating_add(1);
        let hit = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.centroid.cosine(&embedding.normalized())))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((idx, sim)) = hit {
            if sim >= self.cfg.grow_threshold {
                self.update_cluster(idx, embedding);
                let id = self.clusters[idx].id;
                self.idle_counter.insert(id.0, 0);
                return id;
            }
        }
        // Grow (bounded).
        if self.clusters.len() < self.cfg.max_clusters {
            let id = IntentClusterId(self.next_id);
            self.next_id = self.next_id.wrapping_add(1);
            self.clusters.push(IntentCluster {
                id,
                centroid: embedding.normalized(),
                count: 1,
                label: None,
            });
            self.idle_counter.insert(id.0, 0);
            return id;
        }
        // At cap — fall back to the best hit (even if below threshold).
        if let Some((idx, _sim)) = hit {
            self.update_cluster(idx, embedding);
            let id = self.clusters[idx].id;
            self.idle_counter.insert(id.0, 0);
            return id;
        }
        // Should not happen — no clusters + at cap means cap==0. Create one
        // anyway so the caller always gets a valid id.
        let id = IntentClusterId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.clusters.push(IntentCluster {
            id,
            centroid: embedding.normalized(),
            count: 1,
            label: None,
        });
        self.idle_counter.insert(id.0, 0);
        id
    }

    /// Tick idle counters; returns the number of clusters that were pruned.
    /// Call from background cadence (e.g. once per query or per minute).
    pub fn tick_and_prune(&mut self) -> usize {
        if self.cfg.shrink_idle_count == 0 {
            return 0;
        }
        // Increment everyone; any hit resets during assign.
        for c in &self.clusters {
            let entry = self.idle_counter.entry(c.id.0).or_insert(0);
            *entry = entry.saturating_add(1);
        }
        let threshold = self.cfg.shrink_idle_count;
        let min = self.cfg.min_clusters;
        let before = self.clusters.len();
        // Sort descending by idle count so we prune the stalest first.
        let mut ordered: Vec<(usize, u64)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, *self.idle_counter.get(&c.id.0).unwrap_or(&0)))
            .collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1));
        let mut to_remove: Vec<usize> = ordered
            .into_iter()
            .filter(|(_, idle)| *idle >= threshold)
            .map(|(i, _)| i)
            .collect();
        // Remove largest-index-first to keep other indices valid.
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            if self.clusters.len() <= min {
                break;
            }
            let removed = self.clusters.remove(idx);
            self.idle_counter.remove(&removed.id.0);
        }
        before - self.clusters.len()
    }

    /// Best-effort label setter. Caller derives labels out-of-band (e.g.
    /// from common keywords) and stamps them on the cluster for UI.
    pub fn set_label(&mut self, id: IntentClusterId, label: impl Into<String>) -> bool {
        if let Some(c) = self.clusters.iter_mut().find(|c| c.id == id) {
            c.label = Some(label.into());
            true
        } else {
            false
        }
    }

    fn update_cluster(&mut self, idx: usize, embedding: &IntentEmbedding) {
        let c = &mut self.clusters[idx];
        c.count = c.count.saturating_add(1);
        if c.centroid.vector.len() != embedding.vector.len() {
            // Shape changed — reset centroid to this sample to stay usable.
            c.centroid = embedding.normalized();
            return;
        }
        let n = c.count as f32;
        let inv_n = 1.0 / n;
        for (cv, ev) in c.centroid.vector.iter_mut().zip(embedding.vector.iter()) {
            *cv = *cv * (1.0 - inv_n) + *ev * inv_n;
        }
        // Re-normalize lazily to avoid drift.
        let norm = c.centroid.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut c.centroid.vector {
                *v /= norm;
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3(a: f32, b: f32, c: f32) -> IntentEmbedding {
        IntentEmbedding::new(vec![a, b, c]).normalized()
    }

    #[test]
    fn normalized_has_unit_length() {
        let n = IntentEmbedding::new(vec![3.0, 4.0, 0.0]).normalized();
        let len: f32 = n.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalized_handles_zero_vector() {
        let n = IntentEmbedding::new(vec![0.0, 0.0]).normalized();
        assert_eq!(n.vector, vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec3(1.0, 2.0, 3.0);
        assert!((v.cosine(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_of_orthogonal_is_zero() {
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        assert!(a.cosine(&b).abs() < 1e-5);
    }

    #[test]
    fn manager_creates_first_cluster() {
        let mut m = IntentClusterManager::new(IntentClusterManagerConfig::default());
        let id = m.assign(&vec3(1.0, 0.0, 0.0));
        assert_eq!(m.len(), 1);
        assert_eq!(id.0, 0);
    }

    #[test]
    fn manager_reuses_similar_cluster() {
        let mut m = IntentClusterManager::new(IntentClusterManagerConfig::default());
        let a = m.assign(&vec3(1.0, 0.0, 0.0));
        let b = m.assign(&vec3(0.99, 0.01, 0.0));
        assert_eq!(a, b);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn manager_grows_for_dissimilar_embedding() {
        let mut m = IntentClusterManager::new(IntentClusterManagerConfig::default());
        m.assign(&vec3(1.0, 0.0, 0.0));
        m.assign(&vec3(0.0, 1.0, 0.0));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn manager_respects_max_clusters() {
        let mut cfg = IntentClusterManagerConfig::default();
        cfg.max_clusters = 2;
        cfg.grow_threshold = 0.99; // almost always grow until cap
        let mut m = IntentClusterManager::new(cfg);
        m.assign(&vec3(1.0, 0.0, 0.0));
        m.assign(&vec3(0.0, 1.0, 0.0));
        m.assign(&vec3(0.0, 0.0, 1.0));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn manager_prunes_idle_clusters() {
        let mut cfg = IntentClusterManagerConfig::default();
        cfg.shrink_idle_count = 1;
        cfg.min_clusters = 1;
        let mut m = IntentClusterManager::new(cfg);
        m.assign(&vec3(1.0, 0.0, 0.0));
        m.assign(&vec3(0.0, 1.0, 0.0));
        assert_eq!(m.len(), 2);
        // No assign between ticks → both ages go up, prune to min_clusters.
        m.tick_and_prune();
        m.tick_and_prune();
        assert!(m.len() >= 1);
        assert!(m.len() <= 2);
    }

    #[test]
    fn set_label_updates_cluster() {
        let mut m = IntentClusterManager::new(IntentClusterManagerConfig::default());
        let id = m.assign(&vec3(1.0, 0.0, 0.0));
        assert!(m.set_label(id, "code_help"));
        assert_eq!(m.clusters()[0].label.as_deref(), Some("code_help"));
    }

    #[test]
    fn set_label_returns_false_for_missing() {
        let mut m = IntentClusterManager::new(IntentClusterManagerConfig::default());
        assert!(!m.set_label(IntentClusterId(42), "x"));
    }
}
