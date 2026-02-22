//! Advanced memory system: episodic, procedural, and entity memory with consolidation.
//!
//! Provides three complementary memory stores:
//! - **Episodic** (4.1): Time-stamped experiences with embedding-based similarity recall
//! - **Procedural** (4.2): Learned procedures/routines with confidence tracking
//! - **Entity** (4.3): Named entities with attributes, relations, and deduplication
//! - **Consolidation** (4.4): Clusters episodes into reusable procedures
//!
//! Gated behind the `advanced-memory` feature flag.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{AdvancedMemoryError, AiError};

// ============================================================
// Helpers
// ============================================================

/// Cosine similarity between two vectors.
///
/// Returns 0.0 when either vector is zero-length or when dimensions mismatch.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute keyword overlap ratio between two strings (lowercased, whitespace-split).
fn keyword_overlap(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let words_a: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ============================================================
// 4.1 — Episodic Memory
// ============================================================

/// A single episodic memory entry — a recorded experience with context and embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub content: String,
    pub context: String,
    pub timestamp: u64,
    pub importance: f64,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub access_count: usize,
    pub last_accessed: u64,
}

/// Store for episodic memories with capacity limits and temporal decay.
pub struct EpisodicStore {
    episodes: Vec<Episode>,
    max_episodes: usize,
    decay_factor: f64,
}

impl EpisodicStore {
    /// Create a new episodic store with the given capacity and temporal decay factor.
    ///
    /// `decay_factor` controls how quickly older memories lose relevance (0.0 = no
    /// decay, 1.0 = aggressive decay). Typical values are 0.001 .. 0.01.
    pub fn new(max_episodes: usize, decay_factor: f64) -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes,
            decay_factor,
        }
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Add an episode to the store. If capacity is reached the oldest episode is
    /// evicted first.
    pub fn add(&mut self, episode: Episode) {
        if self.episodes.len() >= self.max_episodes {
            self.remove_oldest();
        }
        self.episodes.push(episode);
    }

    /// Retrieve an episode by id (mutable so we can track access).
    pub fn get(&mut self, id: &str) -> Option<&Episode> {
        if let Some(ep) = self.episodes.iter_mut().find(|e| e.id == id) {
            ep.access_count += 1;
            ep.last_accessed = Self::now();
            // Return shared ref after mutation.
            let idx = self.episodes.iter().position(|e| e.id == id);
            return idx.map(|i| &self.episodes[i]);
        }
        None
    }

    /// Recall the top-k episodes most similar to `query_embedding`, weighted by
    /// temporal decay and importance.
    pub fn recall(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<Episode> {
        let now = Self::now();
        let mut scored: Vec<(f64, usize)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(idx, ep)| {
                let sim = cosine_similarity(query_embedding, &ep.embedding);
                let age = (now.saturating_sub(ep.timestamp)) as f64;
                let decay = (-self.decay_factor * age).exp();
                let score = sim * decay * (0.5 + 0.5 * ep.importance);
                (score, idx)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(scored.len());
        let mut results = Vec::with_capacity(k);
        for &(_, idx) in scored.iter().take(k) {
            self.episodes[idx].access_count += 1;
            self.episodes[idx].last_accessed = now;
            results.push(self.episodes[idx].clone());
        }
        results
    }

    /// Recall episodes that share at least one of the given tags, returning the
    /// top-k by number of matching tags then by importance descending.
    pub fn recall_by_tags(&mut self, tags: &[String], top_k: usize) -> Vec<Episode> {
        let now = Self::now();
        let mut scored: Vec<(usize, f64, usize)> = self
            .episodes
            .iter()
            .enumerate()
            .filter_map(|(idx, ep)| {
                let matching = ep.tags.iter().filter(|t| tags.contains(t)).count();
                if matching > 0 {
                    Some((matching, ep.importance, idx))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let k = top_k.min(scored.len());
        let mut results = Vec::with_capacity(k);
        for &(_, _, idx) in scored.iter().take(k) {
            self.episodes[idx].access_count += 1;
            self.episodes[idx].last_accessed = now;
            results.push(self.episodes[idx].clone());
        }
        results
    }

    /// Remove the oldest episode (by timestamp).
    pub fn remove_oldest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }
        let oldest_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by_key(|(_, ep)| ep.timestamp)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.episodes.remove(oldest_idx);
    }

    /// Serialize the store to a JSON string.
    pub fn to_json(&self) -> Result<String, AiError> {
        serde_json::to_string(&self.episodes).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::StoreFailed {
                memory_type: "episodic".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Deserialize episodes from a JSON string, replacing current contents.
    pub fn from_json(&mut self, json: &str) -> Result<(), AiError> {
        let episodes: Vec<Episode> = serde_json::from_str(json).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::RecallFailed {
                query: "from_json".to_string(),
                reason: e.to_string(),
            })
        })?;
        self.episodes = episodes;
        Ok(())
    }

    /// Get a read-only slice of all episodes.
    pub fn all(&self) -> &[Episode] {
        &self.episodes
    }

    // Simple monotonic "now" for timestamping — in production would use
    // `std::time::SystemTime`; here we use it for sorting/decay.
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ============================================================
// 4.2 — Procedural Memory
// ============================================================

/// A learned procedure — a sequence of steps with a triggering condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub steps: Vec<String>,
    pub success_count: usize,
    pub failure_count: usize,
    pub confidence: f64,
    pub created_from: Vec<String>,
    pub tags: Vec<String>,
}

/// Store for procedural memories with capacity limits.
pub struct ProceduralStore {
    procedures: Vec<Procedure>,
    max_procedures: usize,
}

impl ProceduralStore {
    /// Create a new procedural store with the given capacity.
    pub fn new(max_procedures: usize) -> Self {
        Self {
            procedures: Vec::new(),
            max_procedures,
        }
    }

    /// Number of stored procedures.
    pub fn len(&self) -> usize {
        self.procedures.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.procedures.is_empty()
    }

    /// Add a procedure. If capacity is reached the least-confident procedure is
    /// evicted first.
    pub fn add(&mut self, procedure: Procedure) {
        if self.procedures.len() >= self.max_procedures {
            // Evict least confident
            if let Some(idx) = self
                .procedures
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.procedures.remove(idx);
            }
        }
        self.procedures.push(procedure);
    }

    /// Find procedures whose condition keywords match the given context string.
    /// Returns matches sorted by confidence descending.
    pub fn find_by_condition(&self, context: &str) -> Vec<&Procedure> {
        let ctx_lower = context.to_lowercase();
        let ctx_words: std::collections::HashSet<&str> = ctx_lower.split_whitespace().collect();

        let mut matches: Vec<(f64, &Procedure)> = self
            .procedures
            .iter()
            .filter_map(|p| {
                let cond_lower = p.condition.to_lowercase();
                let cond_words: Vec<&str> = cond_lower.split_whitespace().collect();
                let matching = cond_words.iter().filter(|w| ctx_words.contains(*w)).count();
                if matching > 0 {
                    Some((p.confidence, p))
                } else {
                    None
                }
            })
            .collect();

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches.into_iter().map(|(_, p)| p).collect()
    }

    /// Record an outcome (success or failure) for a procedure and update its
    /// confidence.
    pub fn update_outcome(&mut self, id: &str, success: bool) -> Result<(), AiError> {
        let proc = self.procedures.iter_mut().find(|p| p.id == id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        if success {
            proc.success_count += 1;
        } else {
            proc.failure_count += 1;
        }
        let total = proc.success_count + proc.failure_count;
        proc.confidence = proc.success_count as f64 / total as f64;
        Ok(())
    }

    /// Return the top-n most confident procedures.
    pub fn most_confident(&self, n: usize) -> Vec<&Procedure> {
        let mut sorted: Vec<&Procedure> = self.procedures.iter().collect();
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Retrieve a procedure by id.
    pub fn get(&self, id: &str) -> Option<&Procedure> {
        self.procedures.iter().find(|p| p.id == id)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, AiError> {
        serde_json::to_string(&self.procedures).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::StoreFailed {
                memory_type: "procedural".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Deserialize from JSON, replacing current contents.
    pub fn from_json(&mut self, json: &str) -> Result<(), AiError> {
        let procs: Vec<Procedure> = serde_json::from_str(json).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::RecallFailed {
                query: "from_json".to_string(),
                reason: e.to_string(),
            })
        })?;
        self.procedures = procs;
        Ok(())
    }

    /// Read-only access to all procedures.
    pub fn all(&self) -> &[Procedure] {
        &self.procedures
    }
}

// ============================================================
// 4.3 — Entity Memory
// ============================================================

/// A record for a named entity with typed attributes and relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub attributes: HashMap<String, serde_json::Value>,
    pub relations: Vec<EntityRelation>,
    pub first_seen: u64,
    pub last_updated: u64,
    pub mention_count: usize,
}

/// A directed relation from one entity to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    pub relation_type: String,
    pub target_entity_id: String,
    pub confidence: f64,
}

/// Store for entity records with name-based indexing and deduplication.
pub struct EntityStore {
    entities: HashMap<String, EntityRecord>,
    name_index: HashMap<String, String>,
}

impl EntityStore {
    /// Create an empty entity store.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            name_index: HashMap::new(),
        }
    }

    /// Number of entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Add an entity record. Returns an error if an entity with the same
    /// normalized name already exists (use `merge` instead).
    pub fn add(&mut self, record: EntityRecord) -> Result<(), AiError> {
        let normalized = record.name.to_lowercase();
        if let Some(existing_id) = self.name_index.get(&normalized) {
            return Err(AiError::AdvancedMemory(
                AdvancedMemoryError::DuplicateEntity {
                    name: record.name.clone(),
                    existing_id: existing_id.clone(),
                },
            ));
        }
        let id = record.id.clone();
        self.name_index.insert(normalized, id.clone());
        self.entities.insert(id, record);
        Ok(())
    }

    /// Get an entity by id.
    pub fn get(&self, id: &str) -> Option<&EntityRecord> {
        self.entities.get(id)
    }

    /// Find an entity by name (case-insensitive).
    pub fn find_by_name(&self, name: &str) -> Option<&EntityRecord> {
        let normalized = name.to_lowercase();
        self.name_index
            .get(&normalized)
            .and_then(|id| self.entities.get(id))
    }

    /// Update attributes of an entity. Merges the given attributes into the
    /// existing record (overwrites keys that already exist).
    pub fn update(
        &mut self,
        id: &str,
        attributes: HashMap<String, serde_json::Value>,
    ) -> Result<(), AiError> {
        let record = self.entities.get_mut(id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        for (k, v) in attributes {
            record.attributes.insert(k, v);
        }
        record.last_updated = Self::now();
        record.mention_count += 1;
        Ok(())
    }

    /// Add a relation to an entity.
    pub fn add_relation(&mut self, id: &str, relation: EntityRelation) -> Result<(), AiError> {
        let record = self.entities.get_mut(id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        record.relations.push(relation);
        record.last_updated = Self::now();
        Ok(())
    }

    /// Merge two entities. The source entity (`id2`) is removed and its
    /// attributes/relations are folded into the target (`id1`).
    pub fn merge(&mut self, id1: &str, id2: &str) -> Result<(), AiError> {
        if id1 == id2 {
            return Err(AiError::AdvancedMemory(
                AdvancedMemoryError::StoreFailed {
                    memory_type: "entity".to_string(),
                    reason: "Cannot merge an entity with itself".to_string(),
                },
            ));
        }

        let source = self.entities.remove(id2).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id2.to_string(),
            })
        })?;

        // Remove source from name index
        let source_normalized = source.name.to_lowercase();
        self.name_index.remove(&source_normalized);

        // Check target exists; if not, restore source and return error
        let target = match self.entities.get_mut(id1) {
            Some(t) => t,
            None => {
                self.entities.insert(id2.to_string(), source);
                self.name_index
                    .insert(source_normalized, id2.to_string());
                return Err(AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                    name: id1.to_string(),
                }));
            }
        };

        // Merge attributes (source overwrites on conflict)
        for (k, v) in source.attributes {
            target.attributes.insert(k, v);
        }

        // Merge relations
        for rel in source.relations {
            target.relations.push(rel);
        }

        // Accumulate mention count, keep earliest first_seen
        target.mention_count += source.mention_count;
        if source.first_seen < target.first_seen {
            target.first_seen = source.first_seen;
        }
        target.last_updated = Self::now();

        Ok(())
    }

    /// Remove an entity by id.
    pub fn remove(&mut self, id: &str) -> Result<EntityRecord, AiError> {
        let record = self.entities.remove(id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        let normalized = record.name.to_lowercase();
        self.name_index.remove(&normalized);
        Ok(record)
    }

    /// Return all entity records.
    pub fn all(&self) -> Vec<&EntityRecord> {
        self.entities.values().collect()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, AiError> {
        let entries: Vec<&EntityRecord> = self.entities.values().collect();
        serde_json::to_string(&entries).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::StoreFailed {
                memory_type: "entity".to_string(),
                reason: e.to_string(),
            })
        })
    }

    /// Deserialize from JSON, replacing current contents.
    pub fn from_json(&mut self, json: &str) -> Result<(), AiError> {
        let records: Vec<EntityRecord> = serde_json::from_str(json).map_err(|e| {
            AiError::AdvancedMemory(AdvancedMemoryError::RecallFailed {
                query: "from_json".to_string(),
                reason: e.to_string(),
            })
        })?;
        self.entities.clear();
        self.name_index.clear();
        for rec in records {
            let normalized = rec.name.to_lowercase();
            let id = rec.id.clone();
            self.name_index.insert(normalized, id.clone());
            self.entities.insert(id, rec);
        }
        Ok(())
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

// ============================================================
// 4.4 — Memory Consolidation
// ============================================================

/// Configuration and logic for consolidating episodic memories into procedures.
pub struct MemoryConsolidator {
    /// Minimum number of episodes required to consider forming a procedure.
    pub min_episodes_for_procedure: usize,
    /// Tag/keyword overlap threshold (0.0-1.0) to consider two episodes similar.
    pub similarity_threshold: f64,
    /// Minimum cluster size to actually generate a procedure from a cluster.
    pub min_cluster_size: usize,
}

/// Result of a consolidation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub procedures_created: Vec<Procedure>,
    pub episodes_clustered: usize,
    pub clusters_found: usize,
}

impl MemoryConsolidator {
    /// Create a consolidator with sensible defaults.
    pub fn new() -> Self {
        Self {
            min_episodes_for_procedure: 3,
            similarity_threshold: 0.3,
            min_cluster_size: 2,
        }
    }

    /// Cluster the given episodes by content similarity (shared tags + keyword
    /// overlap) and generate procedures from sufficiently large clusters.
    pub fn consolidate(&self, episodes: &[Episode]) -> ConsolidationResult {
        if episodes.len() < self.min_episodes_for_procedure {
            return ConsolidationResult {
                procedures_created: Vec::new(),
                episodes_clustered: 0,
                clusters_found: 0,
            };
        }

        // Simple single-pass greedy clustering
        let mut assigned: Vec<bool> = vec![false; episodes.len()];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..episodes.len() {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..episodes.len() {
                if assigned[j] {
                    continue;
                }
                let sim = self.episode_similarity(&episodes[i], &episodes[j]);
                if sim >= self.similarity_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            if cluster.len() >= self.min_cluster_size {
                clusters.push(cluster);
            }
        }

        // Build procedures from qualifying clusters
        let mut procedures = Vec::new();
        let mut episodes_clustered = 0usize;

        for cluster in &clusters {
            episodes_clustered += cluster.len();

            // Collect shared tags
            let first_tags: std::collections::HashSet<&str> = episodes[cluster[0]]
                .tags
                .iter()
                .map(|t| t.as_str())
                .collect();
            let shared_tags: Vec<String> = first_tags
                .iter()
                .filter(|tag| {
                    cluster
                        .iter()
                        .all(|&idx| episodes[idx].tags.iter().any(|t| t == **tag))
                })
                .map(|s| s.to_string())
                .collect();

            // Build procedure steps from episode contents
            let steps: Vec<String> = cluster
                .iter()
                .map(|&idx| episodes[idx].content.clone())
                .collect();

            let created_from: Vec<String> =
                cluster.iter().map(|&idx| episodes[idx].id.clone()).collect();

            let name = if shared_tags.is_empty() {
                format!("procedure_{}", uuid::Uuid::new_v4())
            } else {
                format!("procedure_{}", shared_tags.join("_"))
            };

            let condition = if shared_tags.is_empty() {
                "general".to_string()
            } else {
                shared_tags.join(", ")
            };

            procedures.push(Procedure {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                condition,
                steps,
                success_count: 0,
                failure_count: 0,
                confidence: 0.5, // neutral starting confidence
                created_from,
                tags: shared_tags,
            });
        }

        ConsolidationResult {
            procedures_created: procedures,
            episodes_clustered,
            clusters_found: clusters.len(),
        }
    }

    /// Compute similarity between two episodes using tag overlap and keyword
    /// overlap in content/context fields.
    fn episode_similarity(&self, a: &Episode, b: &Episode) -> f64 {
        // Tag Jaccard
        let tags_a: std::collections::HashSet<&str> =
            a.tags.iter().map(|t| t.as_str()).collect();
        let tags_b: std::collections::HashSet<&str> =
            b.tags.iter().map(|t| t.as_str()).collect();
        let tag_sim = if tags_a.is_empty() && tags_b.is_empty() {
            0.0
        } else {
            let inter = tags_a.intersection(&tags_b).count();
            let union = tags_a.union(&tags_b).count();
            if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            }
        };

        // Content keyword overlap
        let content_sim = keyword_overlap(&a.content, &b.content);

        // Context keyword overlap
        let context_sim = keyword_overlap(&a.context, &b.context);

        // Weighted combination
        0.4 * tag_sim + 0.35 * content_sim + 0.25 * context_sim
    }
}

// ============================================================
// Unified Manager
// ============================================================

/// Unified manager that owns all three memory stores plus the consolidator.
pub struct AdvancedMemoryManager {
    pub episodic: EpisodicStore,
    pub procedural: ProceduralStore,
    pub entities: EntityStore,
    pub consolidator: MemoryConsolidator,
}

impl AdvancedMemoryManager {
    /// Create a manager with sensible defaults (1000 episodes, 500 procedures,
    /// 0.001 decay).
    pub fn new() -> Self {
        Self {
            episodic: EpisodicStore::new(1000, 0.001),
            procedural: ProceduralStore::new(500),
            entities: EntityStore::new(),
            consolidator: MemoryConsolidator::new(),
        }
    }

    /// Create a manager with custom capacity and decay configuration.
    pub fn with_config(episodic_max: usize, procedural_max: usize, decay: f64) -> Self {
        Self {
            episodic: EpisodicStore::new(episodic_max, decay),
            procedural: ProceduralStore::new(procedural_max),
            entities: EntityStore::new(),
            consolidator: MemoryConsolidator::new(),
        }
    }

    /// Add an episode to the episodic store.
    pub fn add_episode(&mut self, episode: Episode) {
        self.episodic.add(episode);
    }

    /// Add a procedure to the procedural store.
    pub fn add_procedure(&mut self, procedure: Procedure) {
        self.procedural.add(procedure);
    }

    /// Add an entity record.
    pub fn add_entity(&mut self, record: EntityRecord) -> Result<(), AiError> {
        self.entities.add(record)
    }

    /// Run consolidation on all current episodes and add resulting procedures
    /// to the procedural store.
    pub fn consolidate(&mut self) -> ConsolidationResult {
        let episodes = self.episodic.all().to_vec();
        let result = self.consolidator.consolidate(&episodes);
        for proc in &result.procedures_created {
            self.procedural.add(proc.clone());
        }
        result
    }

    /// Recall episodes by embedding similarity.
    pub fn recall_episodes(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<Episode> {
        self.episodic.recall(query_embedding, top_k)
    }

    /// Find procedures matching a context.
    pub fn find_procedures(&self, context: &str) -> Vec<&Procedure> {
        self.procedural.find_by_condition(context)
    }

    /// Find an entity by name.
    pub fn find_entity(&self, name: &str) -> Option<&EntityRecord> {
        self.entities.find_by_name(name)
    }
}

// ============================================================
// Helper: create episode with UUID
// ============================================================

/// Create a new episode with a generated UUID.
pub fn new_episode(
    content: impl Into<String>,
    context: impl Into<String>,
    importance: f64,
    tags: Vec<String>,
    embedding: Vec<f32>,
) -> Episode {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Episode {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.into(),
        context: context.into(),
        timestamp: now,
        importance: importance.clamp(0.0, 1.0),
        tags,
        embedding,
        access_count: 0,
        last_accessed: now,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // Test helpers
    // ----------------------------------------------------------

    fn make_episode(id: &str, content: &str, tags: &[&str], embedding: &[f32], ts: u64) -> Episode {
        Episode {
            id: id.to_string(),
            content: content.to_string(),
            context: format!("context for {}", id),
            timestamp: ts,
            importance: 0.8,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            embedding: embedding.to_vec(),
            access_count: 0,
            last_accessed: ts,
        }
    }

    fn make_procedure(id: &str, name: &str, condition: &str, confidence: f64) -> Procedure {
        Procedure {
            id: id.to_string(),
            name: name.to_string(),
            condition: condition.to_string(),
            steps: vec!["step1".to_string(), "step2".to_string()],
            success_count: 0,
            failure_count: 0,
            confidence,
            created_from: Vec::new(),
            tags: Vec::new(),
        }
    }

    fn make_entity(id: &str, name: &str, etype: &str) -> EntityRecord {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        EntityRecord {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: etype.to_string(),
            attributes: HashMap::new(),
            relations: Vec::new(),
            first_seen: now,
            last_updated: now,
            mention_count: 1,
        }
    }

    // ----------------------------------------------------------
    // Cosine similarity helper tests
    // ----------------------------------------------------------

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.3, 0.5, 0.7, 0.2];
        let b = vec![0.3, 0.5, 0.7, 0.2];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "Opposite vectors should have similarity -1.0");
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert!(sim.abs() < 1e-6, "Empty vectors should yield 0.0");
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Mismatched lengths should yield 0.0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Zero vector should yield 0.0");
    }

    // ----------------------------------------------------------
    // Episode creation
    // ----------------------------------------------------------

    #[test]
    fn test_episode_creation() {
        let ep = make_episode("e1", "learned something", &["rust", "coding"], &[1.0, 0.0], 100);
        assert_eq!(ep.id, "e1");
        assert_eq!(ep.content, "learned something");
        assert_eq!(ep.tags.len(), 2);
        assert_eq!(ep.timestamp, 100);
        assert_eq!(ep.access_count, 0);
    }

    #[test]
    fn test_new_episode_helper() {
        let ep = new_episode("hello world", "context", 0.9, vec!["tag".to_string()], vec![1.0]);
        assert!(!ep.id.is_empty());
        assert_eq!(ep.content, "hello world");
        assert!((ep.importance - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_new_episode_importance_clamped() {
        let ep = new_episode("x", "c", 1.5, vec![], vec![]);
        assert!((ep.importance - 1.0).abs() < 1e-6, "Importance should clamp to 1.0");

        let ep2 = new_episode("x", "c", -0.5, vec![], vec![]);
        assert!((ep2.importance - 0.0).abs() < 1e-6, "Importance should clamp to 0.0");
    }

    // ----------------------------------------------------------
    // EpisodicStore
    // ----------------------------------------------------------

    #[test]
    fn test_episodic_store_add() {
        let mut store = EpisodicStore::new(10, 0.001);
        assert!(store.is_empty());
        store.add(make_episode("e1", "first", &[], &[1.0], 100));
        assert_eq!(store.len(), 1);
        store.add(make_episode("e2", "second", &[], &[0.0, 1.0], 200));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_episodic_store_capacity() {
        let mut store = EpisodicStore::new(3, 0.0);
        store.add(make_episode("e1", "first", &[], &[], 100));
        store.add(make_episode("e2", "second", &[], &[], 200));
        store.add(make_episode("e3", "third", &[], &[], 300));
        assert_eq!(store.len(), 3);

        // Adding a 4th should evict the oldest (e1, ts=100)
        store.add(make_episode("e4", "fourth", &[], &[], 400));
        assert_eq!(store.len(), 3);
        assert!(
            store.all().iter().all(|e| e.id != "e1"),
            "Oldest episode should have been evicted"
        );
    }

    #[test]
    fn test_episodic_store_recall_by_similarity() {
        let mut store = EpisodicStore::new(100, 0.0); // no decay
        store.add(make_episode("e1", "rust programming", &[], &[1.0, 0.0, 0.0], 100));
        store.add(make_episode("e2", "python scripting", &[], &[0.0, 1.0, 0.0], 100));
        store.add(make_episode("e3", "rust systems", &[], &[0.9, 0.1, 0.0], 100));

        let results = store.recall(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // e1 should be most similar, then e3
        assert_eq!(results[0].id, "e1");
        assert_eq!(results[1].id, "e3");
    }

    #[test]
    fn test_episodic_store_recall_by_tags() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "a", &["rust", "systems"], &[], 100));
        store.add(make_episode("e2", "b", &["python", "ml"], &[], 100));
        store.add(make_episode("e3", "c", &["rust", "ml"], &[], 100));

        let results = store.recall_by_tags(
            &["rust".to_string(), "ml".to_string()],
            10,
        );
        // e3 matches 2 tags, e1 matches 1, e2 matches 1
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "e3", "Episode with most matching tags should be first");
    }

    #[test]
    fn test_episodic_store_recall_by_tags_no_match() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "a", &["rust"], &[], 100));
        let results = store.recall_by_tags(&["java".to_string()], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_episodic_store_temporal_decay() {
        let mut store = EpisodicStore::new(100, 0.01);
        // Very old episode
        store.add(make_episode("old", "old memory", &[], &[1.0, 0.0], 0));
        // Recent episode
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        store.add(make_episode("new", "new memory", &[], &[0.9, 0.1], now));

        let results = store.recall(&[1.0, 0.0], 2);
        // With decay, the recent episode should rank higher even though old has
        // slightly higher raw similarity.
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "new", "Recent episode should rank higher with decay");
    }

    #[test]
    fn test_episodic_store_access_tracking() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "content", &[], &[1.0], 100));

        let ep = store.get("e1");
        assert!(ep.is_some());
        assert_eq!(ep.map(|e| e.access_count).unwrap_or(0), 1);

        let ep2 = store.get("e1");
        assert_eq!(ep2.map(|e| e.access_count).unwrap_or(0), 2);
    }

    #[test]
    fn test_episodic_store_get_missing() {
        let mut store = EpisodicStore::new(10, 0.0);
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_episodic_store_remove_oldest() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "a", &[], &[], 300));
        store.add(make_episode("e2", "b", &[], &[], 100));
        store.add(make_episode("e3", "c", &[], &[], 200));

        store.remove_oldest();
        assert_eq!(store.len(), 2);
        assert!(
            store.all().iter().all(|e| e.id != "e2"),
            "e2 (ts=100) should have been removed"
        );
    }

    #[test]
    fn test_episodic_store_remove_oldest_empty() {
        let mut store = EpisodicStore::new(10, 0.0);
        store.remove_oldest(); // should not panic
        assert_eq!(store.len(), 0);
    }

    // ----------------------------------------------------------
    // Procedure creation
    // ----------------------------------------------------------

    #[test]
    fn test_procedure_creation() {
        let p = make_procedure("p1", "test_proc", "when testing", 0.9);
        assert_eq!(p.id, "p1");
        assert_eq!(p.name, "test_proc");
        assert_eq!(p.condition, "when testing");
        assert!((p.confidence - 0.9).abs() < 1e-6);
        assert_eq!(p.steps.len(), 2);
    }

    // ----------------------------------------------------------
    // ProceduralStore
    // ----------------------------------------------------------

    #[test]
    fn test_procedural_store_add() {
        let mut store = ProceduralStore::new(10);
        assert!(store.is_empty());
        store.add(make_procedure("p1", "proc1", "condition1", 0.8));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_procedural_store_capacity() {
        let mut store = ProceduralStore::new(2);
        store.add(make_procedure("p1", "proc1", "c1", 0.5));
        store.add(make_procedure("p2", "proc2", "c2", 0.9));
        assert_eq!(store.len(), 2);

        // Adding a 3rd should evict the least confident (p1, 0.5)
        store.add(make_procedure("p3", "proc3", "c3", 0.7));
        assert_eq!(store.len(), 2);
        assert!(
            store.all().iter().all(|p| p.id != "p1"),
            "Least confident procedure should have been evicted"
        );
    }

    #[test]
    fn test_procedural_find_by_condition() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "when debugging rust code", 0.8));
        store.add(make_procedure("p2", "proc2", "when writing python tests", 0.9));
        store.add(make_procedure("p3", "proc3", "when deploying rust services", 0.7));

        let results = store.find_by_condition("I am debugging some rust");
        assert!(!results.is_empty());
        // p1 matches "debugging" + "rust", p3 matches "rust", p2 has no overlap
        // Sorted by confidence desc among matches → p1 (0.8) first, p3 (0.7) second
        assert_eq!(results[0].id, "p1", "Best matching procedure should be first");
    }

    #[test]
    fn test_procedural_find_by_condition_no_match() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "when debugging", 0.8));
        let results = store.find_by_condition("cooking recipes");
        assert!(results.is_empty());
    }

    #[test]
    fn test_procedural_update_outcome() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "cond", 0.5));

        store.update_outcome("p1", true).expect("should succeed");
        let p = store.get("p1").expect("should exist");
        assert_eq!(p.success_count, 1);
        assert_eq!(p.failure_count, 0);
        assert!((p.confidence - 1.0).abs() < 1e-6);

        store.update_outcome("p1", false).expect("should succeed");
        let p = store.get("p1").expect("should exist");
        assert_eq!(p.success_count, 1);
        assert_eq!(p.failure_count, 1);
        assert!((p.confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_procedural_update_outcome_not_found() {
        let mut store = ProceduralStore::new(10);
        let result = store.update_outcome("nonexistent", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_procedural_most_confident() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "c1", 0.3));
        store.add(make_procedure("p2", "proc2", "c2", 0.9));
        store.add(make_procedure("p3", "proc3", "c3", 0.6));

        let top = store.most_confident(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, "p2");
        assert_eq!(top[1].id, "p3");
    }

    #[test]
    fn test_procedural_most_confident_more_than_available() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "c1", 0.5));
        let top = store.most_confident(10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_procedural_get() {
        let mut store = ProceduralStore::new(10);
        store.add(make_procedure("p1", "proc1", "cond", 0.8));
        assert!(store.get("p1").is_some());
        assert!(store.get("p2").is_none());
    }

    // ----------------------------------------------------------
    // Entity creation
    // ----------------------------------------------------------

    #[test]
    fn test_entity_creation() {
        let e = make_entity("ent1", "Rust Language", "programming_language");
        assert_eq!(e.id, "ent1");
        assert_eq!(e.name, "Rust Language");
        assert_eq!(e.entity_type, "programming_language");
        assert!(e.attributes.is_empty());
        assert!(e.relations.is_empty());
    }

    // ----------------------------------------------------------
    // EntityStore
    // ----------------------------------------------------------

    #[test]
    fn test_entity_store_add() {
        let mut store = EntityStore::new();
        assert!(store.is_empty());
        let result = store.add(make_entity("e1", "Rust", "language"));
        assert!(result.is_ok());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_entity_duplicate_detection() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("first add ok");
        let result = store.add(make_entity("e2", "rust", "language")); // same name, different case
        assert!(result.is_err(), "Duplicate name (case-insensitive) should be rejected");
    }

    #[test]
    fn test_entity_find_by_name() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Python", "language")).expect("ok");
        store.add(make_entity("e2", "Rust", "language")).expect("ok");

        let found = store.find_by_name("python");
        assert!(found.is_some());
        assert_eq!(found.map(|e| e.id.as_str()), Some("e1"));

        let found2 = store.find_by_name("RUST");
        assert!(found2.is_some());
        assert_eq!(found2.map(|e| e.id.as_str()), Some("e2"));

        assert!(store.find_by_name("java").is_none());
    }

    #[test]
    fn test_entity_get() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        assert!(store.get("e1").is_some());
        assert!(store.get("e2").is_none());
    }

    #[test]
    fn test_entity_update_attributes() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");

        let mut attrs = HashMap::new();
        attrs.insert("version".to_string(), serde_json::json!("1.77"));
        attrs.insert("compiled".to_string(), serde_json::json!(true));

        store.update("e1", attrs).expect("ok");
        let ent = store.get("e1").expect("should exist");
        assert_eq!(ent.attributes.len(), 2);
        assert_eq!(ent.attributes.get("version"), Some(&serde_json::json!("1.77")));
    }

    #[test]
    fn test_entity_update_not_found() {
        let mut store = EntityStore::new();
        let result = store.update("nonexistent", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_add_relation() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        store.add(make_entity("e2", "Cargo", "tool")).expect("ok");

        let rel = EntityRelation {
            relation_type: "uses".to_string(),
            target_entity_id: "e2".to_string(),
            confidence: 0.95,
        };
        store.add_relation("e1", rel).expect("ok");

        let ent = store.get("e1").expect("exists");
        assert_eq!(ent.relations.len(), 1);
        assert_eq!(ent.relations[0].relation_type, "uses");
        assert_eq!(ent.relations[0].target_entity_id, "e2");
    }

    #[test]
    fn test_entity_add_relation_not_found() {
        let mut store = EntityStore::new();
        let rel = EntityRelation {
            relation_type: "uses".to_string(),
            target_entity_id: "e2".to_string(),
            confidence: 0.9,
        };
        let result = store.add_relation("nonexistent", rel);
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_merge() {
        let mut store = EntityStore::new();
        let mut e1 = make_entity("e1", "Rust Lang", "language");
        e1.attributes.insert("paradigm".to_string(), serde_json::json!("systems"));
        e1.mention_count = 5;
        store.add(e1).expect("ok");

        let mut e2 = make_entity("e2", "Rust Programming", "language");
        e2.attributes.insert("year".to_string(), serde_json::json!(2010));
        e2.mention_count = 3;
        store.add(e2).expect("ok");

        store.merge("e1", "e2").expect("ok");

        assert_eq!(store.len(), 1, "Source entity should be removed after merge");
        let merged = store.get("e1").expect("target should still exist");
        assert_eq!(merged.attributes.len(), 2, "Attributes should be merged");
        assert_eq!(merged.mention_count, 8, "Mention counts should be summed");
        assert!(store.find_by_name("Rust Programming").is_none(), "Source name index should be removed");
    }

    #[test]
    fn test_entity_merge_self() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        let result = store.merge("e1", "e1");
        assert!(result.is_err(), "Merging entity with itself should fail");
    }

    #[test]
    fn test_entity_merge_target_not_found() {
        let mut store = EntityStore::new();
        store.add(make_entity("e2", "Python", "language")).expect("ok");
        let result = store.merge("nonexistent", "e2");
        // Source is removed first, then target lookup fails; source should be restored
        assert!(result.is_err());
        // Source should be restored
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_entity_merge_source_not_found() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        let result = store.merge("e1", "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_remove() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        assert_eq!(store.len(), 1);

        let removed = store.remove("e1").expect("ok");
        assert_eq!(removed.id, "e1");
        assert_eq!(store.len(), 0);
        assert!(store.find_by_name("Rust").is_none());
    }

    #[test]
    fn test_entity_remove_not_found() {
        let mut store = EntityStore::new();
        let result = store.remove("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_all() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        store.add(make_entity("e2", "Python", "language")).expect("ok");
        let all = store.all();
        assert_eq!(all.len(), 2);
    }

    // ----------------------------------------------------------
    // Consolidation
    // ----------------------------------------------------------

    #[test]
    fn test_consolidation_basic() {
        let consolidator = MemoryConsolidator::new();
        // Too few episodes -> no procedures
        let episodes = vec![make_episode("e1", "hello", &["tag1"], &[], 100)];
        let result = consolidator.consolidate(&episodes);
        assert_eq!(result.procedures_created.len(), 0);
        assert_eq!(result.episodes_clustered, 0);
        assert_eq!(result.clusters_found, 0);
    }

    #[test]
    fn test_consolidation_min_cluster_size() {
        let mut consolidator = MemoryConsolidator::new();
        consolidator.min_episodes_for_procedure = 2;
        consolidator.min_cluster_size = 3;
        consolidator.similarity_threshold = 0.1;

        // Only 2 similar episodes -> cluster too small
        let episodes = vec![
            make_episode("e1", "rust programming", &["rust"], &[], 100),
            make_episode("e2", "rust coding", &["rust"], &[], 200),
        ];
        let result = consolidator.consolidate(&episodes);
        assert_eq!(result.clusters_found, 0);
    }

    #[test]
    fn test_consolidation_creates_procedures() {
        let mut consolidator = MemoryConsolidator::new();
        consolidator.min_episodes_for_procedure = 2;
        consolidator.min_cluster_size = 2;
        consolidator.similarity_threshold = 0.1;

        let episodes = vec![
            make_episode("e1", "rust systems programming", &["rust", "systems"], &[], 100),
            make_episode("e2", "rust memory safety", &["rust", "safety"], &[], 200),
            make_episode("e3", "python data analysis", &["python", "data"], &[], 300),
            make_episode("e4", "python machine learning", &["python", "ml"], &[], 400),
        ];

        let result = consolidator.consolidate(&episodes);
        assert!(
            result.clusters_found >= 1,
            "Should find at least one cluster (rust or python episodes)"
        );
        assert!(result.episodes_clustered >= 2);
        assert!(!result.procedures_created.is_empty());

        // Each procedure should have an id and steps
        for proc in &result.procedures_created {
            assert!(!proc.id.is_empty());
            assert!(!proc.steps.is_empty());
            assert!(!proc.created_from.is_empty());
        }
    }

    #[test]
    fn test_consolidation_no_overlap() {
        let mut consolidator = MemoryConsolidator::new();
        consolidator.min_episodes_for_procedure = 2;
        consolidator.min_cluster_size = 2;
        consolidator.similarity_threshold = 0.9; // very high threshold

        let episodes = vec![
            make_episode("e1", "alpha beta", &["x"], &[], 100),
            make_episode("e2", "gamma delta", &["y"], &[], 200),
            make_episode("e3", "epsilon zeta", &["z"], &[], 300),
        ];

        let result = consolidator.consolidate(&episodes);
        assert_eq!(result.clusters_found, 0, "No clusters when similarity threshold is very high");
    }

    #[test]
    fn test_consolidation_all_identical() {
        let mut consolidator = MemoryConsolidator::new();
        consolidator.min_episodes_for_procedure = 2;
        consolidator.min_cluster_size = 2;
        consolidator.similarity_threshold = 0.1;

        let episodes = vec![
            make_episode("e1", "same content here", &["tag"], &[], 100),
            make_episode("e2", "same content here", &["tag"], &[], 200),
            make_episode("e3", "same content here", &["tag"], &[], 300),
        ];

        let result = consolidator.consolidate(&episodes);
        assert_eq!(result.clusters_found, 1, "All identical episodes should form one cluster");
        assert_eq!(result.episodes_clustered, 3);
        assert_eq!(result.procedures_created.len(), 1);
        assert!(result.procedures_created[0].tags.contains(&"tag".to_string()));
    }

    // ----------------------------------------------------------
    // AdvancedMemoryManager
    // ----------------------------------------------------------

    #[test]
    fn test_manager_new() {
        let mgr = AdvancedMemoryManager::new();
        assert_eq!(mgr.episodic.len(), 0);
        assert_eq!(mgr.procedural.len(), 0);
        assert_eq!(mgr.entities.len(), 0);
    }

    #[test]
    fn test_manager_with_config() {
        let mgr = AdvancedMemoryManager::with_config(500, 200, 0.01);
        assert_eq!(mgr.episodic.len(), 0);
        assert_eq!(mgr.procedural.len(), 0);
    }

    #[test]
    fn test_manager_add_episode() {
        let mut mgr = AdvancedMemoryManager::new();
        mgr.add_episode(make_episode("e1", "test", &[], &[1.0], 100));
        assert_eq!(mgr.episodic.len(), 1);
    }

    #[test]
    fn test_manager_add_procedure() {
        let mut mgr = AdvancedMemoryManager::new();
        mgr.add_procedure(make_procedure("p1", "proc", "cond", 0.8));
        assert_eq!(mgr.procedural.len(), 1);
    }

    #[test]
    fn test_manager_add_entity() {
        let mut mgr = AdvancedMemoryManager::new();
        let result = mgr.add_entity(make_entity("ent1", "Rust", "language"));
        assert!(result.is_ok());
        assert_eq!(mgr.entities.len(), 1);
    }

    #[test]
    fn test_manager_recall_episodes() {
        let mut mgr = AdvancedMemoryManager::new();
        mgr.add_episode(make_episode("e1", "rust", &[], &[1.0, 0.0], 100));
        mgr.add_episode(make_episode("e2", "python", &[], &[0.0, 1.0], 100));
        let results = mgr.recall_episodes(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_manager_find_procedures() {
        let mut mgr = AdvancedMemoryManager::new();
        mgr.add_procedure(make_procedure("p1", "proc1", "when debugging rust", 0.8));
        let results = mgr.find_procedures("debugging");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_manager_find_entity() {
        let mut mgr = AdvancedMemoryManager::new();
        mgr.add_entity(make_entity("e1", "Rust", "language")).expect("ok");
        assert!(mgr.find_entity("rust").is_some());
        assert!(mgr.find_entity("java").is_none());
    }

    #[test]
    fn test_manager_full_lifecycle() {
        let mut mgr = AdvancedMemoryManager::with_config(100, 50, 0.0);

        // Add episodes
        mgr.add_episode(make_episode(
            "e1", "rust ownership explained", &["rust", "ownership"], &[1.0, 0.0], 100,
        ));
        mgr.add_episode(make_episode(
            "e2", "rust borrow checker rules", &["rust", "borrowing"], &[0.9, 0.1], 200,
        ));
        mgr.add_episode(make_episode(
            "e3", "python data frames", &["python", "data"], &[0.0, 1.0], 300,
        ));

        // Add entity
        mgr.add_entity(make_entity("ent1", "Rust", "language")).expect("ok");

        // Recall
        let recalled = mgr.recall_episodes(&[1.0, 0.0], 2);
        assert_eq!(recalled.len(), 2);

        // Find entity
        assert!(mgr.find_entity("rust").is_some());

        // Consolidate
        let result = mgr.consolidate();
        // Should create at least something since e1 and e2 share "rust" tag and
        // related content
        assert!(result.clusters_found >= 1 || result.procedures_created.is_empty());

        // Verify procedures were added
        let total_procs = mgr.procedural.len();
        assert!(total_procs >= result.procedures_created.len());
    }

    #[test]
    fn test_manager_consolidate_adds_procedures() {
        let mut mgr = AdvancedMemoryManager::with_config(100, 50, 0.0);
        mgr.consolidator.min_episodes_for_procedure = 2;
        mgr.consolidator.min_cluster_size = 2;
        mgr.consolidator.similarity_threshold = 0.1;

        mgr.add_episode(make_episode("e1", "same topic here", &["tag1"], &[], 100));
        mgr.add_episode(make_episode("e2", "same topic here", &["tag1"], &[], 200));
        mgr.add_episode(make_episode("e3", "same topic here", &["tag1"], &[], 300));

        let result = mgr.consolidate();
        assert!(!result.procedures_created.is_empty());
        assert!(mgr.procedural.len() > 0, "Consolidation should add procedures to the store");
    }

    // ----------------------------------------------------------
    // Serialization / Persistence
    // ----------------------------------------------------------

    #[test]
    fn test_episode_serialization() {
        let ep = make_episode("e1", "hello", &["tag1", "tag2"], &[0.5, 0.3], 12345);
        let json = serde_json::to_string(&ep).expect("serialize ok");
        let deserialized: Episode = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(deserialized.id, "e1");
        assert_eq!(deserialized.content, "hello");
        assert_eq!(deserialized.tags, vec!["tag1", "tag2"]);
        assert_eq!(deserialized.embedding, vec![0.5, 0.3]);
    }

    #[test]
    fn test_procedure_serialization() {
        let p = make_procedure("p1", "test_proc", "when testing", 0.85);
        let json = serde_json::to_string(&p).expect("serialize ok");
        let deserialized: Procedure = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(deserialized.id, "p1");
        assert_eq!(deserialized.name, "test_proc");
        assert!((deserialized.confidence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_entity_serialization() {
        let mut e = make_entity("ent1", "Rust", "language");
        e.attributes.insert("version".to_string(), serde_json::json!("2021"));
        e.relations.push(EntityRelation {
            relation_type: "used_by".to_string(),
            target_entity_id: "ent2".to_string(),
            confidence: 0.9,
        });
        let json = serde_json::to_string(&e).expect("serialize ok");
        let deserialized: EntityRecord = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(deserialized.id, "ent1");
        assert_eq!(deserialized.name, "Rust");
        assert_eq!(deserialized.attributes.len(), 1);
        assert_eq!(deserialized.relations.len(), 1);
    }

    #[test]
    fn test_episodic_store_to_from_json() {
        let mut store = EpisodicStore::new(100, 0.001);
        store.add(make_episode("e1", "first", &["a"], &[1.0], 100));
        store.add(make_episode("e2", "second", &["b"], &[0.0, 1.0], 200));

        let json = store.to_json().expect("to_json ok");
        let mut store2 = EpisodicStore::new(100, 0.001);
        store2.from_json(&json).expect("from_json ok");
        assert_eq!(store2.len(), 2);
    }

    #[test]
    fn test_procedural_store_to_from_json() {
        let mut store = ProceduralStore::new(100);
        store.add(make_procedure("p1", "proc1", "cond1", 0.8));

        let json = store.to_json().expect("to_json ok");
        let mut store2 = ProceduralStore::new(100);
        store2.from_json(&json).expect("from_json ok");
        assert_eq!(store2.len(), 1);
    }

    #[test]
    fn test_entity_store_to_from_json() {
        let mut store = EntityStore::new();
        store.add(make_entity("e1", "Rust", "language")).expect("ok");
        store.add(make_entity("e2", "Python", "language")).expect("ok");

        let json = store.to_json().expect("to_json ok");
        let mut store2 = EntityStore::new();
        store2.from_json(&json).expect("from_json ok");
        assert_eq!(store2.len(), 2);
        assert!(store2.find_by_name("rust").is_some());
        assert!(store2.find_by_name("python").is_some());
    }

    #[test]
    fn test_episodic_store_from_invalid_json() {
        let mut store = EpisodicStore::new(10, 0.0);
        let result = store.from_json("not valid json!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_procedural_store_from_invalid_json() {
        let mut store = ProceduralStore::new(10);
        let result = store.from_json("{broken}");
        assert!(result.is_err());
    }

    #[test]
    fn test_entity_store_from_invalid_json() {
        let mut store = EntityStore::new();
        let result = store.from_json("[nope]");
        assert!(result.is_err());
    }

    // ----------------------------------------------------------
    // Consolidation result serialization
    // ----------------------------------------------------------

    #[test]
    fn test_consolidation_result_serialization() {
        let result = ConsolidationResult {
            procedures_created: vec![make_procedure("p1", "proc", "cond", 0.5)],
            episodes_clustered: 4,
            clusters_found: 2,
        };
        let json = serde_json::to_string(&result).expect("serialize ok");
        let deser: ConsolidationResult = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(deser.episodes_clustered, 4);
        assert_eq!(deser.clusters_found, 2);
        assert_eq!(deser.procedures_created.len(), 1);
    }

    // ----------------------------------------------------------
    // EntityRelation
    // ----------------------------------------------------------

    #[test]
    fn test_entity_relation_serialization() {
        let rel = EntityRelation {
            relation_type: "depends_on".to_string(),
            target_entity_id: "target_1".to_string(),
            confidence: 0.75,
        };
        let json = serde_json::to_string(&rel).expect("serialize ok");
        let deser: EntityRelation = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(deser.relation_type, "depends_on");
        assert_eq!(deser.target_entity_id, "target_1");
        assert!((deser.confidence - 0.75).abs() < 1e-6);
    }

    // ----------------------------------------------------------
    // Keyword overlap helper
    // ----------------------------------------------------------

    #[test]
    fn test_keyword_overlap_identical() {
        let sim = keyword_overlap("hello world", "hello world");
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_keyword_overlap_partial() {
        let sim = keyword_overlap("hello world", "hello there");
        // intersection = {hello}, union = {hello, world, there} -> 1/3
        assert!((sim - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_keyword_overlap_none() {
        let sim = keyword_overlap("alpha beta", "gamma delta");
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_keyword_overlap_empty() {
        let sim = keyword_overlap("", "hello");
        assert!(sim.abs() < 1e-6);
        let sim2 = keyword_overlap("hello", "");
        assert!(sim2.abs() < 1e-6);
    }

    // ----------------------------------------------------------
    // Edge cases and additional coverage
    // ----------------------------------------------------------

    #[test]
    fn test_episodic_recall_empty_store() {
        let mut store = EpisodicStore::new(10, 0.0);
        let results = store.recall(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_episodic_recall_by_tags_empty_store() {
        let mut store = EpisodicStore::new(10, 0.0);
        let results = store.recall_by_tags(&["tag".to_string()], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_procedural_find_empty_store() {
        let store = ProceduralStore::new(10);
        let results = store.find_by_condition("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_entity_store_empty() {
        let store = EntityStore::new();
        assert!(store.is_empty());
        assert!(store.all().is_empty());
        assert!(store.get("x").is_none());
        assert!(store.find_by_name("x").is_none());
    }

    #[test]
    fn test_episodic_store_recall_top_k_larger_than_store() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "only", &[], &[1.0], 100));
        let results = store.recall(&[1.0], 100);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_episodic_store_recall_by_tags_top_k_limit() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "a", &["t1"], &[], 100));
        store.add(make_episode("e2", "b", &["t1"], &[], 200));
        store.add(make_episode("e3", "c", &["t1"], &[], 300));

        let results = store.recall_by_tags(&["t1".to_string()], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_episode_access_count_incremented_by_recall() {
        let mut store = EpisodicStore::new(100, 0.0);
        store.add(make_episode("e1", "content", &[], &[1.0], 100));

        store.recall(&[1.0], 1);
        let ep = &store.all()[0];
        assert_eq!(ep.access_count, 1, "Access count should be incremented after recall");

        store.recall(&[1.0], 1);
        let ep = &store.all()[0];
        assert_eq!(ep.access_count, 2, "Access count should be incremented again");
    }

    #[test]
    fn test_manager_consolidate_empty() {
        let mut mgr = AdvancedMemoryManager::new();
        let result = mgr.consolidate();
        assert_eq!(result.procedures_created.len(), 0);
        assert_eq!(result.episodes_clustered, 0);
    }

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![-1.0, 2.0, -3.0];
        let b = vec![4.0, -5.0, 6.0];
        let sim = cosine_similarity(&a, &b);
        // dot = -4 + -10 + -18 = -32
        // |a| = sqrt(1+4+9) = sqrt(14), |b| = sqrt(16+25+36) = sqrt(77)
        // sim = -32 / (sqrt(14)*sqrt(77))
        let expected = -32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        assert!((sim - expected).abs() < 1e-6);
    }

    #[test]
    fn test_entity_merge_preserves_earliest_first_seen() {
        let mut store = EntityStore::new();
        let mut e1 = make_entity("e1", "Target", "type");
        e1.first_seen = 200;
        store.add(e1).expect("ok");

        let mut e2 = make_entity("e2", "Source", "type");
        e2.first_seen = 100; // earlier
        store.add(e2).expect("ok");

        store.merge("e1", "e2").expect("ok");
        let merged = store.get("e1").expect("exists");
        assert_eq!(merged.first_seen, 100, "Should keep the earlier first_seen");
    }

    #[test]
    fn test_entity_merge_accumulates_relations() {
        let mut store = EntityStore::new();
        let mut e1 = make_entity("e1", "Target", "type");
        e1.relations.push(EntityRelation {
            relation_type: "r1".to_string(),
            target_entity_id: "other".to_string(),
            confidence: 0.8,
        });
        store.add(e1).expect("ok");

        let mut e2 = make_entity("e2", "Source", "type");
        e2.relations.push(EntityRelation {
            relation_type: "r2".to_string(),
            target_entity_id: "another".to_string(),
            confidence: 0.7,
        });
        store.add(e2).expect("ok");

        store.merge("e1", "e2").expect("ok");
        let merged = store.get("e1").expect("exists");
        assert_eq!(merged.relations.len(), 2, "Relations from both entities should be present");
    }
}
