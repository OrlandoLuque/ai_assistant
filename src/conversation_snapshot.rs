//! Conversation snapshots
//!
//! Save and restore conversation state.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Snapshot metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMetadata {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: u64,
    pub message_count: usize,
    pub size_bytes: usize,
    pub tags: Vec<String>,
    pub version: String,
}

/// Conversation message for snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

/// Full conversation snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationSnapshot {
    pub metadata: SnapshotMetadata,
    pub messages: Vec<SnapshotMessage>,
    pub context: HashMap<String, String>,
    pub memory: Vec<MemoryItem>,
}

/// Memory item in snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryItem {
    pub key: String,
    pub value: String,
    pub importance: f64,
}

impl ConversationSnapshot {
    pub fn new(name: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            metadata: SnapshotMetadata {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.to_string(),
                description: None,
                created_at: now,
                message_count: 0,
                size_bytes: 0,
                tags: Vec::new(),
                version: "1.0".to_string(),
            },
            messages: Vec::new(),
            context: HashMap::new(),
            memory: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: &str) -> Self {
        self.metadata.description = Some(description.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.metadata.tags.push(tag.to_string());
        self
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.messages.push(SnapshotMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: now,
            metadata: HashMap::new(),
        });

        self.update_metadata();
    }

    pub fn set_context(&mut self, key: &str, value: &str) {
        self.context.insert(key.to_string(), value.to_string());
    }

    pub fn add_memory(&mut self, key: &str, value: &str, importance: f64) {
        self.memory.push(MemoryItem {
            key: key.to_string(),
            value: value.to_string(),
            importance,
        });
    }

    fn update_metadata(&mut self) {
        self.metadata.message_count = self.messages.len();
        // Estimate size
        let json = serde_json::to_string(self).unwrap_or_default();
        self.metadata.size_bytes = json.len();
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Serialize to internal binary format (bincode+gzip when feature enabled).
    #[cfg(feature = "binary-storage")]
    pub fn to_binary(&self) -> Result<Vec<u8>, anyhow::Error> {
        crate::internal_storage::serialize_internal(self)
    }

    /// Deserialize from internal format (auto-detects binary or JSON).
    #[cfg(feature = "binary-storage")]
    pub fn from_binary(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        crate::internal_storage::deserialize_internal(bytes)
    }
}

/// Snapshot manager
pub struct SnapshotManager {
    snapshots: HashMap<String, ConversationSnapshot>,
    storage_path: Option<String>,
    max_snapshots: usize,
}

impl SnapshotManager {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            storage_path: None,
            max_snapshots: 100,
        }
    }

    pub fn with_storage(mut self, path: &str) -> Self {
        self.storage_path = Some(path.to_string());
        self
    }

    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    pub fn create(&mut self, name: &str) -> &mut ConversationSnapshot {
        let snapshot = ConversationSnapshot::new(name);
        let id = snapshot.metadata.id.clone();
        self.snapshots.insert(id.clone(), snapshot);
        self.snapshots.get_mut(&id).expect("key just inserted")
    }

    pub fn save(&mut self, snapshot: ConversationSnapshot) -> String {
        let id = snapshot.metadata.id.clone();

        // Enforce max snapshots
        if self.snapshots.len() >= self.max_snapshots {
            // Remove oldest
            if let Some(oldest) = self.get_oldest_id() {
                self.snapshots.remove(&oldest);
            }
        }

        self.snapshots.insert(id.clone(), snapshot);

        // Save to disk if path configured
        if let Some(ref path) = self.storage_path {
            self.save_to_disk(&id, path);
        }

        id
    }

    pub fn load(&self, id: &str) -> Option<&ConversationSnapshot> {
        self.snapshots.get(id)
    }

    pub fn load_mut(&mut self, id: &str) -> Option<&mut ConversationSnapshot> {
        self.snapshots.get_mut(id)
    }

    pub fn delete(&mut self, id: &str) -> Option<ConversationSnapshot> {
        let snapshot = self.snapshots.remove(id);

        // Delete from disk
        if let Some(ref path) = self.storage_path {
            let file_path = format!("{}/{}.json", path, id);
            let _ = std::fs::remove_file(file_path);
        }

        snapshot
    }

    pub fn list(&self) -> Vec<&SnapshotMetadata> {
        self.snapshots.values().map(|s| &s.metadata).collect()
    }

    pub fn search_by_tag(&self, tag: &str) -> Vec<&ConversationSnapshot> {
        self.snapshots
            .values()
            .filter(|s| s.metadata.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn search_by_name(&self, query: &str) -> Vec<&ConversationSnapshot> {
        let query_lower = query.to_lowercase();
        self.snapshots
            .values()
            .filter(|s| s.metadata.name.to_lowercase().contains(&query_lower))
            .collect()
    }

    fn get_oldest_id(&self) -> Option<String> {
        self.snapshots
            .values()
            .min_by_key(|s| s.metadata.created_at)
            .map(|s| s.metadata.id.clone())
    }

    fn save_to_disk(&self, id: &str, base_path: &str) {
        if let Some(snapshot) = self.snapshots.get(id) {
            let path = std::path::Path::new(base_path).join(format!("{}.bin", id));
            let _ = crate::internal_storage::save_internal(snapshot, &path);
        }
    }

    /// Load snapshots from disk. Supports both legacy JSON (.json) and
    /// binary (.bin) snapshot files via auto-detection.
    pub fn load_from_disk(&mut self, base_path: &str) -> std::io::Result<usize> {
        let mut count = 0;

        for entry in std::fs::read_dir(base_path)? {
            let entry = entry?;
            let path = entry.path();

            let is_snapshot_file = path
                .extension()
                .map(|e| e == "json" || e == "bin")
                .unwrap_or(false);

            if is_snapshot_file {
                // Use internal_storage to auto-detect format
                if let Ok(snapshot) =
                    crate::internal_storage::load_internal::<ConversationSnapshot>(&path)
                {
                    self.snapshots
                        .insert(snapshot.metadata.id.clone(), snapshot);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    pub fn export(&self, id: &str) -> Option<String> {
        self.snapshots.get(id).and_then(|s| s.to_json().ok())
    }

    pub fn import(&mut self, json: &str) -> Result<String, serde_json::Error> {
        let snapshot = ConversationSnapshot::from_json(json)?;
        let id = self.save(snapshot);
        Ok(id)
    }
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot diff
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub added_messages: Vec<SnapshotMessage>,
    pub removed_messages: Vec<String>,
    pub context_changes: HashMap<String, (Option<String>, Option<String>)>,
    pub memory_changes: Vec<MemoryChange>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MemoryChange {
    Added(MemoryItem),
    Removed(String),
    Modified {
        key: String,
        old: String,
        new: String,
    },
}

impl SnapshotDiff {
    pub fn compare(old: &ConversationSnapshot, new: &ConversationSnapshot) -> Self {
        let old_msg_ids: std::collections::HashSet<_> =
            old.messages.iter().map(|m| m.id.clone()).collect();
        let new_msg_ids: std::collections::HashSet<_> =
            new.messages.iter().map(|m| m.id.clone()).collect();

        let added_messages: Vec<_> = new
            .messages
            .iter()
            .filter(|m| !old_msg_ids.contains(&m.id))
            .cloned()
            .collect();

        let removed_messages: Vec<_> = old_msg_ids.difference(&new_msg_ids).cloned().collect();

        let mut context_changes = HashMap::new();
        for (key, new_val) in &new.context {
            let old_val = old.context.get(key);
            if old_val != Some(new_val) {
                context_changes.insert(key.clone(), (old_val.cloned(), Some(new_val.clone())));
            }
        }
        for (key, old_val) in &old.context {
            if !new.context.contains_key(key) {
                context_changes.insert(key.clone(), (Some(old_val.clone()), None));
            }
        }

        let mut memory_changes = Vec::new();
        let old_memory: HashMap<_, _> = old.memory.iter().map(|m| (m.key.clone(), m)).collect();
        let new_memory: HashMap<_, _> = new.memory.iter().map(|m| (m.key.clone(), m)).collect();

        for (key, new_item) in &new_memory {
            match old_memory.get(key) {
                Some(old_item) if old_item.value != new_item.value => {
                    memory_changes.push(MemoryChange::Modified {
                        key: key.clone(),
                        old: old_item.value.clone(),
                        new: new_item.value.clone(),
                    });
                }
                None => {
                    memory_changes.push(MemoryChange::Added((*new_item).clone()));
                }
                _ => {}
            }
        }

        for key in old_memory.keys() {
            if !new_memory.contains_key(key) {
                memory_changes.push(MemoryChange::Removed(key.clone()));
            }
        }

        SnapshotDiff {
            added_messages,
            removed_messages,
            context_changes,
            memory_changes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.added_messages.is_empty()
            && self.removed_messages.is_empty()
            && self.context_changes.is_empty()
            && self.memory_changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let mut snapshot = ConversationSnapshot::new("Test Conversation")
            .with_description("A test conversation")
            .with_tag("test");

        snapshot.add_message("user", "Hello!");
        snapshot.add_message("assistant", "Hi there!");

        assert_eq!(snapshot.messages.len(), 2);
        assert_eq!(snapshot.metadata.name, "Test Conversation");
    }

    #[test]
    fn test_snapshot_serialization() {
        let mut snapshot = ConversationSnapshot::new("Test");
        snapshot.add_message("user", "Test message");

        let json = snapshot.to_json().unwrap();
        let loaded = ConversationSnapshot::from_json(&json).unwrap();

        assert_eq!(loaded.metadata.name, "Test");
        assert_eq!(loaded.messages.len(), 1);
    }

    #[test]
    fn test_snapshot_manager() {
        let mut manager = SnapshotManager::new();

        let snapshot = manager.create("Test");
        let id = snapshot.metadata.id.clone();

        assert!(manager.load(&id).is_some());
    }

    #[test]
    fn test_snapshot_context_and_memory() {
        let mut snapshot = ConversationSnapshot::new("Test");
        snapshot.set_context("model", "gpt-4");
        snapshot.set_context("temp", "0.7");
        snapshot.add_memory("user_name", "Alice", 0.9);
        assert_eq!(snapshot.context.get("model").unwrap(), "gpt-4");
        assert_eq!(snapshot.memory.len(), 1);
        assert!((snapshot.memory[0].importance - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snapshot_bytes_roundtrip() {
        let mut snapshot = ConversationSnapshot::new("ByteTest");
        snapshot.add_message("user", "Hello bytes!");
        let bytes = snapshot.to_bytes().unwrap();
        let restored = ConversationSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored.metadata.name, "ByteTest");
        assert_eq!(restored.messages.len(), 1);
    }

    #[test]
    fn test_manager_save_delete() {
        let mut manager = SnapshotManager::new();
        let snapshot = ConversationSnapshot::new("ToDelete");
        let id = manager.save(snapshot);
        assert!(manager.load(&id).is_some());
        let deleted = manager.delete(&id);
        assert!(deleted.is_some());
        assert!(manager.load(&id).is_none());
    }

    #[test]
    fn test_manager_search_by_tag() {
        let mut manager = SnapshotManager::new();
        let s1 = ConversationSnapshot::new("Tagged").with_tag("important");
        let s2 = ConversationSnapshot::new("Untagged");
        manager.save(s1);
        manager.save(s2);
        let results = manager.search_by_tag("important");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.name, "Tagged");
    }

    #[test]
    fn test_manager_search_by_name() {
        let mut manager = SnapshotManager::new();
        manager.save(ConversationSnapshot::new("Alpha Session"));
        manager.save(ConversationSnapshot::new("Beta Session"));
        manager.save(ConversationSnapshot::new("Gamma Talk"));
        let results = manager.search_by_name("session");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_diff_context_and_memory() {
        let mut old = ConversationSnapshot::new("Old");
        old.set_context("key1", "val1");
        old.add_memory("fact", "old_value", 0.5);

        let mut new = ConversationSnapshot::new("New");
        new.set_context("key1", "val2");
        new.set_context("key2", "new");
        new.add_memory("fact", "new_value", 0.8);

        let diff = SnapshotDiff::compare(&old, &new);
        assert!(!diff.context_changes.is_empty());
        assert!(!diff.memory_changes.is_empty());
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_snapshot_diff() {
        let mut old = ConversationSnapshot::new("Old");
        old.add_message("user", "Hello");

        let mut new = ConversationSnapshot::new("New");
        new.add_message("user", "Hello");
        new.add_message("assistant", "Hi!");

        let diff = SnapshotDiff::compare(&old, &new);
        assert!(!diff.added_messages.is_empty());
    }
}
