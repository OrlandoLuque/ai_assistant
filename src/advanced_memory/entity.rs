//! Entity memory: named entities with attributes, relations, and deduplication.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{AdvancedMemoryError, AiError};

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
#[derive(Debug)]
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

    /// Save the entity store to a JSON file. Uses atomic write (temp file + rename).
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
