//! Agent memory sharing
//!
//! Shared memory system for multi-agent collaboration.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Memory entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        self.agent_views
            .entry(owner)
            .or_default()
            .push(id.clone());

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
        let id = self.entries.iter()
            .find(|(_, e)| e.key == key && (e.can_access(agent_id) || self.global_entries.contains(&e.id)))
            .map(|(id, _)| id.clone())?;

        self.get(&id, agent_id)
    }

    pub fn update(&mut self, id: &str, value: &str, agent_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get_mut(id)
            .ok_or(MemoryError::NotFound)?;

        if entry.owner != agent_id {
            return Err(MemoryError::AccessDenied);
        }

        entry.value = value.to_string();
        entry.updated_at = Instant::now();

        Ok(())
    }

    pub fn delete(&mut self, id: &str, agent_id: &str) -> Result<(), MemoryError> {
        let entry = self.entries.get(id)
            .ok_or(MemoryError::NotFound)?;

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
        let entry = self.entries.get_mut(id)
            .ok_or(MemoryError::NotFound)?;

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
        let entry = self.entries.get_mut(id)
            .ok_or(MemoryError::NotFound)?;

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
                e.key.to_lowercase().contains(&query_lower) ||
                e.value.to_lowercase().contains(&query_lower)
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
        let expired: Vec<_> = self.entries.iter()
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
        let by_type: HashMap<MemoryType, usize> = self.entries.values()
            .fold(HashMap::new(), |mut acc, e| {
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
        self.inner.write().unwrap().store(entry)
    }

    pub fn get(&self, id: &str, agent_id: &str) -> Option<MemoryEntry> {
        self.inner.write().unwrap().get(id, agent_id).cloned()
    }

    pub fn update(&self, id: &str, value: &str, agent_id: &str) -> Result<(), MemoryError> {
        self.inner.write().unwrap().update(id, value, agent_id)
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

        memory.store(MemoryEntry::new("python_info", "Python is a language", MemoryType::Fact, "agent1"));
        memory.store(MemoryEntry::new("rust_info", "Rust is fast", MemoryType::Fact, "agent1"));

        let results = memory.search("python", "agent1");
        assert_eq!(results.len(), 1);
    }
}
