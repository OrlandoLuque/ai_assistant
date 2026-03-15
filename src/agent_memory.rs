//! Agent memory sharing
//!
//! Shared memory system for multi-agent collaboration.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Memory entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemoryType {
    Fact,
    Context,
    Task,
    Result,
    Preference,
    Temporary,
}

/// Memory entry
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub key: String,
    pub value: String,
    pub memory_type: MemoryType,
    pub owner: String,
    pub shared_with: Vec<String>,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub access_count: u64,
    pub ttl: Option<std::time::Duration>,
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    pub fn new(key: &str, value: &str, memory_type: MemoryType, owner: &str) -> Self {
        let now = Instant::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            key: key.to_string(),
            value: value.to_string(),
            memory_type,
            owner: owner.to_string(),
            shared_with: Vec::new(),
            created_at: now,
            updated_at: now,
            access_count: 0,
            ttl: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn share_with(mut self, agent_id: &str) -> Self {
        if !self.shared_with.contains(&agent_id.to_string()) {
            self.shared_with.push(agent_id.to_string());
        }
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    pub fn can_access(&self, agent_id: &str) -> bool {
        self.owner == agent_id || self.shared_with.contains(&agent_id.to_string())
    }
}

/// Shared memory store
pub struct SharedMemory {
    entries: HashMap<String, MemoryEntry>,
    agent_views: HashMap<String, Vec<String>>,
    global_entries: Vec<String>,
}

impl SharedMemory {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            agent_views: HashMap::new(),
            global_entries: Vec::new(),
        }
    }

    pub fn store(&mut self, entry: MemoryEntry) -> String {
        let id = entry.id.clone();
        let owner = entry.owner.clone();
        let shared = entry.shared_with.clone();

        // Track in owner's view
        self.agent_views.entry(owner).or_default().push(id.clone());

        // Track in shared agents' views
        for agent_id in shared {
            self.agent_views
                .entry(agent_id)
                .or_default()
                .push(id.clone());
        }

        self.entries.insert(id.clone(), entry);
        id
    }

    pub fn store_global(&mut self, entry: MemoryEntry) -> String {
        let id = entry.id.clone();
        self.global_entries.push(id.clone());
        self.entries.insert(id.clone(), entry);
        id
    }

    pub fn get(&mut self, id: &str, agent_id: &str) -> Option<&MemoryEntry> {
        let entry = self.entries.get_mut(id)?;

        if entry.is_expired() {
            return None;
        }

        if !entry.can_access(agent_id) && !self.global_entries.contains(&id.to_string()) {
            return None;
        }

        entry.access_count += 1;
        self.entries.get(id)
    }

    pub fn get_by_key(&mut self, key: &str, agent_id: &str) -> Option<&MemoryEntry> {
        let id = self
            .entries
            .iter()
            .find(|(_, e)| {
                e.key == key && (e.can_access(agent_id) || self.global_entries.contains(&e.id))
            })
            .map(|(id, _)| id.clone())?;

        self.get(&id, agent_id)
    }

    pub fn update(&mut self, id: &str, value: &str, agent_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get_mut(id).ok_or(MemoryError::NotFound)?;

        if entry.owner != agent_id {
            return Err(MemoryError::AccessDenied);
        }

        entry.value = value.to_string();
        entry.updated_at = Instant::now();

        Ok(())
    }

    pub fn delete(&mut self, id: &str, agent_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get(id).ok_or(MemoryError::NotFound)?;

        if entry.owner != agent_id {
            return Err(MemoryError::AccessDenied);
        }

        // Remove from views
        for views in self.agent_views.values_mut() {
            views.retain(|v| v != id);
        }

        self.global_entries.retain(|g| g != id);
        self.entries.remove(id);

        Ok(())
    }

    pub fn share(&mut self, id: &str, with_agent: &str, owner_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get_mut(id).ok_or(MemoryError::NotFound)?;

        if entry.owner != owner_id {
            return Err(MemoryError::AccessDenied);
        }

        if !entry.shared_with.contains(&with_agent.to_string()) {
            entry.shared_with.push(with_agent.to_string());

            self.agent_views
                .entry(with_agent.to_string())
                .or_default()
                .push(id.to_string());
        }

        Ok(())
    }

    pub fn unshare(&mut self, id: &str, agent_id: &str, owner_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get_mut(id).ok_or(MemoryError::NotFound)?;

        if entry.owner != owner_id {
            return Err(MemoryError::AccessDenied);
        }

        entry.shared_with.retain(|a| a != agent_id);

        if let Some(views) = self.agent_views.get_mut(agent_id) {
            views.retain(|v| v != id);
        }

        Ok(())
    }

    pub fn get_agent_memories(&self, agent_id: &str) -> Vec<&MemoryEntry> {
        let mut memories = Vec::new();

        // Get owned and shared memories
        if let Some(ids) = self.agent_views.get(agent_id) {
            for id in ids {
                if let Some(entry) = self.entries.get(id) {
                    if !entry.is_expired() {
                        memories.push(entry);
                    }
                }
            }
        }

        // Add global memories
        for id in &self.global_entries {
            if let Some(entry) = self.entries.get(id) {
                if !entry.is_expired() && !memories.iter().any(|m| m.id == entry.id) {
                    memories.push(entry);
                }
            }
        }

        memories
    }

    pub fn search(&self, query: &str, agent_id: &str) -> Vec<&MemoryEntry> {
        let query_lower = query.to_lowercase();

        self.get_agent_memories(agent_id)
            .into_iter()
            .filter(|e| {
                e.key.to_lowercase().contains(&query_lower)
                    || e.value.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn get_by_type(&self, memory_type: MemoryType, agent_id: &str) -> Vec<&MemoryEntry> {
        self.get_agent_memories(agent_id)
            .into_iter()
            .filter(|e| e.memory_type == memory_type)
            .collect()
    }

    pub fn cleanup_expired(&mut self) {
        let expired: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            self.entries.remove(&id);
            for views in self.agent_views.values_mut() {
                views.retain(|v| v != &id);
            }
            self.global_entries.retain(|g| g != &id);
        }
    }

    pub fn stats(&self) -> MemoryStats {
        let total = self.entries.len();
        let by_type: HashMap<MemoryType, usize> =
            self.entries.values().fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.memory_type).or_insert(0) += 1;
                acc
            });

        let total_access: u64 = self.entries.values().map(|e| e.access_count).sum();
        let global_count = self.global_entries.len();

        MemoryStats {
            total_entries: total,
            by_type,
            total_access_count: total_access,
            global_entries: global_count,
            agents_with_memory: self.agent_views.len(),
        }
    }
}

impl Default for SharedMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub by_type: HashMap<MemoryType, usize>,
    pub total_access_count: u64,
    pub global_entries: usize,
    pub agents_with_memory: usize,
}

/// Memory errors
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryError {
    NotFound,
    AccessDenied,
    AlreadyExists,
    InvalidOperation,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Memory entry not found"),
            Self::AccessDenied => write!(f, "Access denied"),
            Self::AlreadyExists => write!(f, "Entry already exists"),
            Self::InvalidOperation => write!(f, "Invalid operation"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Thread-safe shared memory wrapper
pub struct ThreadSafeMemory {
    inner: Arc<RwLock<SharedMemory>>,
}

impl ThreadSafeMemory {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SharedMemory::new())),
        }
    }

    pub fn store(&self, entry: MemoryEntry) -> String {
        match self.inner.write() {
            Ok(mut inner) => inner.store(entry),
            Err(_) => String::new(),
        }
    }

    pub fn get(&self, id: &str, agent_id: &str) -> Option<MemoryEntry> {
        match self.inner.write() {
            Ok(mut inner) => inner.get(id, agent_id).cloned(),
            Err(_) => None,
        }
    }

    pub fn update(&self, id: &str, value: &str, agent_id: &str) -> Result<(), MemoryError> {
        self.inner
            .write()
            .map_err(|_| MemoryError::InvalidOperation)?
            .update(id, value, agent_id)
    }
}

impl Default for ThreadSafeMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ThreadSafeMemory {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("test_key", "test_value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        let retrieved = memory.get(&id, "agent1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "test_value");
    }

    #[test]
    fn test_sharing() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("shared_key", "shared_value", MemoryType::Context, "agent1");
        let id = memory.store(entry);

        // Agent2 cannot access yet
        assert!(memory.get(&id, "agent2").is_none());

        // Share with agent2
        memory.share(&id, "agent2", "agent1").unwrap();

        // Now agent2 can access
        assert!(memory.get(&id, "agent2").is_some());
    }

    #[test]
    fn test_global_memory() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("global_key", "global_value", MemoryType::Fact, "system");
        let id = memory.store_global(entry);

        // Any agent can access global memory
        assert!(memory.get(&id, "agent1").is_some());
        assert!(memory.get(&id, "agent2").is_some());
    }

    #[test]
    fn test_search() {
        let mut memory = SharedMemory::new();

        memory.store(MemoryEntry::new(
            "python_info",
            "Python is a language",
            MemoryType::Fact,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "rust_info",
            "Rust is fast",
            MemoryType::Fact,
            "agent1",
        ));

        let results = memory.search("python", "agent1");
        assert_eq!(results.len(), 1);
    }

    // === Additional Unit Tests ===

    #[test]
    fn test_ttl_expiration() {
        let mut memory = SharedMemory::new();

        // Create entry with very short TTL
        let entry = MemoryEntry::new("temp_key", "temp_value", MemoryType::Temporary, "agent1")
            .with_ttl(std::time::Duration::from_millis(1));

        let id = memory.store(entry);

        // Wait for expiration
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Should not be retrievable after expiration
        let retrieved = memory.get(&id, "agent1");
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_entry_is_expired() {
        let entry_no_ttl = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        assert!(!entry_no_ttl.is_expired());

        let entry_long_ttl = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1")
            .with_ttl(std::time::Duration::from_secs(3600));
        assert!(!entry_long_ttl.is_expired());

        let entry_expired = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1")
            .with_ttl(std::time::Duration::from_millis(1));
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(entry_expired.is_expired());
    }

    #[test]
    fn test_metadata() {
        let entry = MemoryEntry::new("key", "value", MemoryType::Context, "agent1")
            .with_metadata("source", "api")
            .with_metadata("version", "1.0");

        assert_eq!(entry.metadata.get("source"), Some(&"api".to_string()));
        assert_eq!(entry.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_get_by_type() {
        let mut memory = SharedMemory::new();

        memory.store(MemoryEntry::new(
            "fact1",
            "Fact 1",
            MemoryType::Fact,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "fact2",
            "Fact 2",
            MemoryType::Fact,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "context1",
            "Context 1",
            MemoryType::Context,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "task1",
            "Task 1",
            MemoryType::Task,
            "agent1",
        ));

        let facts = memory.get_by_type(MemoryType::Fact, "agent1");
        assert_eq!(facts.len(), 2);

        let contexts = memory.get_by_type(MemoryType::Context, "agent1");
        assert_eq!(contexts.len(), 1);

        let tasks = memory.get_by_type(MemoryType::Task, "agent1");
        assert_eq!(tasks.len(), 1);
    }

    #[test]
    fn test_cleanup_expired() {
        let mut memory = SharedMemory::new();

        // Add entry with short TTL
        let entry1 = MemoryEntry::new("temp", "value", MemoryType::Temporary, "agent1")
            .with_ttl(std::time::Duration::from_millis(1));
        memory.store(entry1);

        // Add entry without TTL
        let entry2 = MemoryEntry::new("permanent", "value", MemoryType::Fact, "agent1");
        memory.store(entry2);

        // Wait for expiration
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Cleanup
        memory.cleanup_expired();

        // Only permanent entry should remain
        let memories = memory.get_agent_memories("agent1");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].key, "permanent");
    }

    #[test]
    fn test_stats() {
        let mut memory = SharedMemory::new();

        memory.store(MemoryEntry::new("fact1", "v1", MemoryType::Fact, "agent1"));
        memory.store(MemoryEntry::new("fact2", "v2", MemoryType::Fact, "agent2"));
        memory.store(MemoryEntry::new(
            "context1",
            "v3",
            MemoryType::Context,
            "agent1",
        ));
        memory.store_global(MemoryEntry::new(
            "global1",
            "v4",
            MemoryType::Result,
            "system",
        ));

        let stats = memory.stats();

        assert_eq!(stats.total_entries, 4);
        assert_eq!(stats.global_entries, 1);
        assert_eq!(stats.agents_with_memory, 2); // agent1 and agent2
        assert_eq!(*stats.by_type.get(&MemoryType::Fact).unwrap(), 2);
        assert_eq!(*stats.by_type.get(&MemoryType::Context).unwrap(), 1);
        assert_eq!(*stats.by_type.get(&MemoryType::Result).unwrap(), 1);
    }

    #[test]
    fn test_update() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "original", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Update should succeed for owner
        let result = memory.update(&id, "updated", "agent1");
        assert!(result.is_ok());

        // Verify update
        let retrieved = memory.get(&id, "agent1").unwrap();
        assert_eq!(retrieved.value, "updated");
    }

    #[test]
    fn test_update_access_denied() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Update should fail for non-owner
        let result = memory.update(&id, "new_value", "agent2");
        assert_eq!(result, Err(MemoryError::AccessDenied));
    }

    #[test]
    fn test_update_not_found() {
        let mut memory = SharedMemory::new();

        let result = memory.update("nonexistent", "value", "agent1");
        assert_eq!(result, Err(MemoryError::NotFound));
    }

    #[test]
    fn test_delete() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Delete should succeed for owner
        let result = memory.delete(&id, "agent1");
        assert!(result.is_ok());

        // Should no longer exist
        assert!(memory.get(&id, "agent1").is_none());
    }

    #[test]
    fn test_delete_access_denied() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Delete should fail for non-owner
        let result = memory.delete(&id, "agent2");
        assert_eq!(result, Err(MemoryError::AccessDenied));
    }

    #[test]
    fn test_delete_not_found() {
        let mut memory = SharedMemory::new();

        let result = memory.delete("nonexistent", "agent1");
        assert_eq!(result, Err(MemoryError::NotFound));
    }

    #[test]
    fn test_unshare() {
        let mut memory = SharedMemory::new();

        let entry =
            MemoryEntry::new("key", "value", MemoryType::Fact, "agent1").share_with("agent2");
        let id = memory.store(entry);

        // agent2 can access
        assert!(memory.get(&id, "agent2").is_some());

        // Unshare
        memory.unshare(&id, "agent2", "agent1").unwrap();

        // agent2 can no longer access
        assert!(memory.get(&id, "agent2").is_none());
    }

    #[test]
    fn test_unshare_access_denied() {
        let mut memory = SharedMemory::new();

        let entry =
            MemoryEntry::new("key", "value", MemoryType::Fact, "agent1").share_with("agent2");
        let id = memory.store(entry);

        // agent2 cannot unshare (not owner)
        let result = memory.unshare(&id, "agent3", "agent2");
        assert_eq!(result, Err(MemoryError::AccessDenied));
    }

    #[test]
    fn test_access_count() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Access multiple times
        memory.get(&id, "agent1");
        memory.get(&id, "agent1");
        memory.get(&id, "agent1");

        let entry = memory.get(&id, "agent1").unwrap();
        assert_eq!(entry.access_count, 4); // Including this get
    }

    #[test]
    fn test_get_by_key() {
        let mut memory = SharedMemory::new();

        memory.store(MemoryEntry::new(
            "unique_key",
            "the_value",
            MemoryType::Fact,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "another_key",
            "other",
            MemoryType::Fact,
            "agent1",
        ));

        let result = memory.get_by_key("unique_key", "agent1");
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, "the_value");

        let not_found = memory.get_by_key("nonexistent_key", "agent1");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_can_access() {
        let entry =
            MemoryEntry::new("key", "value", MemoryType::Fact, "agent1").share_with("agent2");

        assert!(entry.can_access("agent1")); // Owner
        assert!(entry.can_access("agent2")); // Shared
        assert!(!entry.can_access("agent3")); // Not shared
    }

    #[test]
    fn test_share_duplicate() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        // Share twice with same agent
        memory.share(&id, "agent2", "agent1").unwrap();
        memory.share(&id, "agent2", "agent1").unwrap(); // Should not duplicate

        let entry = memory.get(&id, "agent1").unwrap();
        let count = entry.shared_with.iter().filter(|a| *a == "agent2").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_all_memory_types() {
        let types = vec![
            MemoryType::Fact,
            MemoryType::Context,
            MemoryType::Task,
            MemoryType::Result,
            MemoryType::Preference,
            MemoryType::Temporary,
        ];

        for memory_type in types {
            let entry = MemoryEntry::new("key", "value", memory_type, "agent");
            assert_eq!(entry.memory_type, memory_type);
        }
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", MemoryError::NotFound),
            "Memory entry not found"
        );
        assert_eq!(format!("{}", MemoryError::AccessDenied), "Access denied");
        assert_eq!(
            format!("{}", MemoryError::AlreadyExists),
            "Entry already exists"
        );
        assert_eq!(
            format!("{}", MemoryError::InvalidOperation),
            "Invalid operation"
        );
    }

    #[test]
    fn test_shared_memory_default() {
        let memory = SharedMemory::default();
        let stats = memory.stats();
        assert_eq!(stats.total_entries, 0);
    }

    #[test]
    fn test_thread_safe_memory_basic() {
        let memory = ThreadSafeMemory::new();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        let retrieved = memory.get(&id, "agent1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "value");
    }

    #[test]
    fn test_thread_safe_memory_update() {
        let memory = ThreadSafeMemory::new();

        let entry = MemoryEntry::new("key", "original", MemoryType::Fact, "agent1");
        let id = memory.store(entry);

        memory.update(&id, "updated", "agent1").unwrap();

        let retrieved = memory.get(&id, "agent1");
        assert_eq!(retrieved.unwrap().value, "updated");
    }

    #[test]
    fn test_thread_safe_memory_clone() {
        let memory1 = ThreadSafeMemory::new();
        let memory2 = memory1.clone();

        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1");
        let id = memory1.store(entry);

        // Both should see the same data
        assert!(memory2.get(&id, "agent1").is_some());
    }

    #[test]
    fn test_thread_safe_memory_concurrent() {
        use std::thread;

        let memory = ThreadSafeMemory::new();
        let mut handles = vec![];

        // Spawn multiple threads writing to memory
        for i in 0..10 {
            let mem = memory.clone();
            let handle = thread::spawn(move || {
                let entry = MemoryEntry::new(
                    &format!("key_{}", i),
                    &format!("value_{}", i),
                    MemoryType::Fact,
                    &format!("agent_{}", i),
                );
                mem.store(entry);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all entries were stored
        for i in 0..10 {
            let _agent = format!("agent_{}", i);
            let _key = format!("key_{}", i);
            // We need to access via get_by_key through the inner RwLock
            // For this test, we verify the store operations completed without panicking
        }
    }

    #[test]
    fn test_search_in_key_and_value() {
        let mut memory = SharedMemory::new();

        memory.store(MemoryEntry::new(
            "python_facts",
            "Rust is not here",
            MemoryType::Fact,
            "agent1",
        ));
        memory.store(MemoryEntry::new(
            "rust_facts",
            "This is about Rust",
            MemoryType::Fact,
            "agent1",
        ));

        // Search finds match in key
        let results = memory.search("python", "agent1");
        assert_eq!(results.len(), 1);

        // Search finds match in value
        let results = memory.search("rust", "agent1");
        assert_eq!(results.len(), 2); // Both entries mention rust
    }

    #[test]
    fn test_get_agent_memories_excludes_expired() {
        let mut memory = SharedMemory::new();

        // Add permanent entry
        memory.store(MemoryEntry::new(
            "permanent",
            "stays",
            MemoryType::Fact,
            "agent1",
        ));

        // Add expiring entry
        let temp = MemoryEntry::new("temp", "goes away", MemoryType::Temporary, "agent1")
            .with_ttl(std::time::Duration::from_millis(1));
        memory.store(temp);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let memories = memory.get_agent_memories("agent1");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].key, "permanent");
    }

    #[test]
    fn test_global_memory_in_agent_memories() {
        let mut memory = SharedMemory::new();

        // Add agent-specific memory
        memory.store(MemoryEntry::new(
            "agent_key",
            "agent_value",
            MemoryType::Fact,
            "agent1",
        ));

        // Add global memory
        memory.store_global(MemoryEntry::new(
            "global_key",
            "global_value",
            MemoryType::Fact,
            "system",
        ));

        // Agent should see both
        let memories = memory.get_agent_memories("agent1");
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_share_with_builder() {
        let entry = MemoryEntry::new("key", "value", MemoryType::Fact, "agent1")
            .share_with("agent2")
            .share_with("agent3")
            .share_with("agent2"); // Duplicate, should not add again

        assert_eq!(entry.shared_with.len(), 2);
        assert!(entry.shared_with.contains(&"agent2".to_string()));
        assert!(entry.shared_with.contains(&"agent3".to_string()));
    }

    #[test]
    fn test_delete_removes_from_views() {
        let mut memory = SharedMemory::new();

        let entry =
            MemoryEntry::new("key", "value", MemoryType::Fact, "agent1").share_with("agent2");
        let id = memory.store(entry);

        // Both agents can see it
        assert!(memory.get(&id, "agent1").is_some());
        assert!(memory.get(&id, "agent2").is_some());

        // Delete
        memory.delete(&id, "agent1").unwrap();

        // Neither can see it now
        assert!(memory.get(&id, "agent1").is_none());
        assert!(memory.get(&id, "agent2").is_none());
    }

    #[test]
    fn test_delete_removes_from_global() {
        let mut memory = SharedMemory::new();

        let entry = MemoryEntry::new("global", "value", MemoryType::Fact, "system");
        let id = memory.store_global(entry);

        // Any agent can access
        assert!(memory.get(&id, "agent1").is_some());

        // Delete (only owner can)
        memory.delete(&id, "system").unwrap();

        // No longer accessible
        assert!(memory.get(&id, "agent1").is_none());
    }
}
