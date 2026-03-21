//! Entity memory: named entities with attributes, relations, and deduplication.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{AdvancedMemoryError, AiError};

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

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
    /// Optional embedding vector for semantic search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Optional TTL: entity expires at this Unix timestamp (0 = no expiry).
    #[serde(default)]
    pub expires_at: u64,
}

/// Query builder for filtering entities by type, attributes, or text.
#[derive(Debug, Clone, Default)]
pub struct EntityQuery {
    /// Filter by entity type (exact match).
    pub entity_type: Option<String>,
    /// Attribute contains this substring (key or value).
    pub attribute_contains: Option<String>,
    /// Name contains this substring (case-insensitive).
    pub name_contains: Option<String>,
    /// Minimum mention count.
    pub min_mentions: Option<usize>,
    /// Maximum results to return.
    pub limit: Option<usize>,
}

/// A directed relation from one entity to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelation {
    pub relation_type: String,
    pub target_entity_id: String,
    pub confidence: f64,
}

/// Store for entity records with name-based indexing and deduplication.
#[derive(Debug)]
pub struct EntityStore {
    entities: HashMap<String, EntityRecord>,
    name_index: HashMap<String, String>,
    /// Maximum number of entities (0 = unlimited).
    max_entities: usize,
}

impl EntityStore {
    /// Create an empty entity store with default limits.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            name_index: HashMap::new(),
            max_entities: 5_000,
        }
    }

    /// Create with a custom max entities limit.
    pub fn with_max(max_entities: usize) -> Self {
        Self {
            entities: HashMap::new(),
            name_index: HashMap::new(),
            max_entities,
        }
    }

    /// Evict the least-mentioned entity if at capacity.
    fn evict_if_needed(&mut self) {
        if self.max_entities == 0 || self.entities.len() < self.max_entities {
            return;
        }
        // Find entity with lowest mention_count (and no relations, if possible)
        if let Some((evict_id, _)) = self
            .entities
            .iter()
            .filter(|(_, r)| r.relations.is_empty()) // prefer evicting entities without relations
            .min_by_key(|(_, r)| r.mention_count)
            .map(|(id, r)| (id.clone(), r.mention_count))
            .or_else(|| {
                self.entities
                    .iter()
                    .min_by_key(|(_, r)| r.mention_count)
                    .map(|(id, r)| (id.clone(), r.mention_count))
            })
        {
            if let Some(record) = self.entities.remove(&evict_id) {
                let normalized = record.name.to_lowercase();
                self.name_index.remove(&normalized);
            }
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

    /// Add an entity record. Evicts least-mentioned entity if at capacity.
    /// Returns an error if an entity with the same normalized name already exists.
    pub fn add(&mut self, record: EntityRecord) -> Result<(), AiError> {
        self.evict_if_needed();
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

    /// List all distinct entity types.
    pub fn list_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .entities
            .values()
            .map(|e| e.entity_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        types.sort();
        types
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

    /// Query entities with filters.
    pub fn query(&self, q: &EntityQuery) -> Vec<&EntityRecord> {
        let mut results: Vec<&EntityRecord> = self
            .entities
            .values()
            .filter(|e| {
                if let Some(ref t) = q.entity_type {
                    if e.entity_type.to_lowercase() != t.to_lowercase() {
                        return false;
                    }
                }
                if let Some(ref name_q) = q.name_contains {
                    let lower = name_q.to_lowercase();
                    if !e.name.to_lowercase().contains(&lower) {
                        return false;
                    }
                }
                if let Some(min) = q.min_mentions {
                    if e.mention_count < min {
                        return false;
                    }
                }
                if let Some(ref attr_q) = q.attribute_contains {
                    let lower = attr_q.to_lowercase();
                    let has_match = e.attributes.iter().any(|(k, v)| {
                        k.to_lowercase().contains(&lower)
                            || v.to_string().to_lowercase().contains(&lower)
                    });
                    if !has_match {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort by mention_count descending (most relevant first)
        results.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));

        if let Some(limit) = q.limit {
            results.truncate(limit);
        }
        results
    }

    /// Find entities by type.
    pub fn find_by_type(&self, entity_type: &str) -> Vec<&EntityRecord> {
        let lower = entity_type.to_lowercase();
        self.entities
            .values()
            .filter(|e| e.entity_type.to_lowercase() == lower)
            .collect()
    }

    /// Search entities by semantic similarity (cosine distance to query embedding).
    /// Returns (entity, similarity_score) sorted by score descending.
    pub fn search_similar(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Vec<(&EntityRecord, f32)> {
        let mut scored: Vec<(&EntityRecord, f32)> = self
            .entities
            .values()
            .filter_map(|e| {
                let emb = e.embedding.as_ref()?;
                let sim = cosine_similarity(query_embedding, emb);
                Some((e, sim))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Remove expired entities (those with expires_at > 0 and < now).
    /// Returns the number of entities evicted.
    pub fn evict_expired(&mut self) -> usize {
        let now = Self::now();
        let expired_ids: Vec<String> = self
            .entities
            .iter()
            .filter(|(_, e)| e.expires_at > 0 && e.expires_at < now)
            .map(|(id, _)| id.clone())
            .collect();
        let count = expired_ids.len();
        for id in &expired_ids {
            if let Some(rec) = self.entities.remove(id) {
                self.name_index.remove(&rec.name.to_lowercase());
            }
        }
        count
    }

    /// Set a TTL on an entity (expiry in seconds from now).
    pub fn set_ttl(&mut self, id: &str, ttl_secs: u64) -> Result<(), AiError> {
        let entity = self.entities.get_mut(id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        entity.expires_at = Self::now() + ttl_secs;
        Ok(())
    }

    /// Set an embedding vector on an entity.
    pub fn set_embedding(&mut self, id: &str, embedding: Vec<f32>) -> Result<(), AiError> {
        let entity = self.entities.get_mut(id).ok_or_else(|| {
            AiError::AdvancedMemory(AdvancedMemoryError::EntityNotFound {
                name: id.to_string(),
            })
        })?;
        entity.embedding = Some(embedding);
        entity.last_updated = Self::now();
        Ok(())
    }

    /// Count of entities by type.
    pub fn count_by_type(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for e in self.entities.values() {
            *counts.entry(e.entity_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Save the entity store to a JSON file. Uses atomic write (temp file + rename).
    #[allow(clippy::inherent_to_string)]
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<String, String> {
        let entries: Vec<&EntityRecord> = self.entities.values().collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| format!("Serialize error: {}", e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("Rename error: {}", e))?;
        Ok(json)
    }

    /// Load an entity store from a JSON file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let records: Vec<EntityRecord> =
            serde_json::from_str(&data).map_err(|e| format!("Deserialize error: {}", e))?;
        let mut store = Self::new();
        for rec in records {
            let normalized = rec.name.to_lowercase();
            let id = rec.id.clone();
            store.name_index.insert(normalized, id.clone());
            store.entities.insert(id, rec);
        }
        Ok(store)
    }
}
