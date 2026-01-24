//! Incremental sync
//!
//! Synchronize conversation state incrementally.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Sync operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SyncOperation {
    Add,
    Update,
    Delete,
}

/// Sync entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncEntry {
    pub id: String,
    pub operation: SyncOperation,
    pub entity_type: String,
    pub entity_id: String,
    pub data: Option<String>,
    pub timestamp: u64,
    pub version: u64,
}

impl SyncEntry {
    pub fn add(entity_type: &str, entity_id: &str, data: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            operation: SyncOperation::Add,
            entity_type: entity_type.to_string(),
            entity_id: entity_id.to_string(),
            data: Some(data.to_string()),
            timestamp: now,
            version: 1,
        }
    }

    pub fn update(entity_type: &str, entity_id: &str, data: &str, version: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            operation: SyncOperation::Update,
            entity_type: entity_type.to_string(),
            entity_id: entity_id.to_string(),
            data: Some(data.to_string()),
            timestamp: now,
            version,
        }
    }

    pub fn delete(entity_type: &str, entity_id: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            operation: SyncOperation::Delete,
            entity_type: entity_type.to_string(),
            entity_id: entity_id.to_string(),
            data: None,
            timestamp: now,
            version: 0,
        }
    }
}

/// Sync state for tracking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncState {
    pub client_id: String,
    pub last_sync_timestamp: u64,
    pub last_sync_version: u64,
    pub entity_versions: HashMap<String, u64>,
}

impl SyncState {
    pub fn new(client_id: &str) -> Self {
        Self {
            client_id: client_id.to_string(),
            last_sync_timestamp: 0,
            last_sync_version: 0,
            entity_versions: HashMap::new(),
        }
    }
}

/// Sync log for storing changes
pub struct SyncLog {
    entries: Vec<SyncEntry>,
    max_entries: usize,
    current_version: u64,
}

impl SyncLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            current_version: 0,
        }
    }

    pub fn append(&mut self, mut entry: SyncEntry) -> u64 {
        self.current_version += 1;
        entry.version = self.current_version;

        self.entries.push(entry);

        // Trim old entries
        if self.entries.len() > self.max_entries {
            self.entries.drain(0..(self.entries.len() - self.max_entries));
        }

        self.current_version
    }

    pub fn get_since(&self, version: u64) -> Vec<&SyncEntry> {
        self.entries.iter()
            .filter(|e| e.version > version)
            .collect()
    }

    pub fn get_since_timestamp(&self, timestamp: u64) -> Vec<&SyncEntry> {
        self.entries.iter()
            .filter(|e| e.timestamp > timestamp)
            .collect()
    }

    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    pub fn compact(&mut self) {
        // Keep only latest operation per entity
        let mut latest: HashMap<String, &SyncEntry> = HashMap::new();

        for entry in &self.entries {
            let key = format!("{}:{}", entry.entity_type, entry.entity_id);
            latest.insert(key, entry);
        }

        let compacted: Vec<SyncEntry> = latest.values()
            .map(|e| (*e).clone())
            .collect();

        self.entries = compacted;
    }
}

impl Default for SyncLog {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Sync delta for transmission
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncDelta {
    pub from_version: u64,
    pub to_version: u64,
    pub entries: Vec<SyncEntry>,
    pub is_full_sync: bool,
}

impl SyncDelta {
    pub fn new(from_version: u64, to_version: u64, entries: Vec<SyncEntry>) -> Self {
        Self {
            from_version,
            to_version,
            entries,
            is_full_sync: false,
        }
    }

    pub fn full_sync(entries: Vec<SyncEntry>) -> Self {
        let to_version = entries.iter().map(|e| e.version).max().unwrap_or(0);
        Self {
            from_version: 0,
            to_version,
            entries,
            is_full_sync: true,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Incremental sync manager
pub struct IncrementalSyncManager {
    log: SyncLog,
    client_states: HashMap<String, SyncState>,
}

impl IncrementalSyncManager {
    pub fn new() -> Self {
        Self {
            log: SyncLog::default(),
            client_states: HashMap::new(),
        }
    }

    pub fn record_add(&mut self, entity_type: &str, entity_id: &str, data: &str) -> u64 {
        let entry = SyncEntry::add(entity_type, entity_id, data);
        self.log.append(entry)
    }

    pub fn record_update(&mut self, entity_type: &str, entity_id: &str, data: &str) -> u64 {
        let version = self.log.current_version() + 1;
        let entry = SyncEntry::update(entity_type, entity_id, data, version);
        self.log.append(entry)
    }

    pub fn record_delete(&mut self, entity_type: &str, entity_id: &str) -> u64 {
        let entry = SyncEntry::delete(entity_type, entity_id);
        self.log.append(entry)
    }

    pub fn get_delta(&self, client_id: &str) -> SyncDelta {
        let from_version = self.client_states.get(client_id)
            .map(|s| s.last_sync_version)
            .unwrap_or(0);

        let entries: Vec<SyncEntry> = self.log.get_since(from_version)
            .into_iter()
            .cloned()
            .collect();

        SyncDelta::new(from_version, self.log.current_version(), entries)
    }

    pub fn get_full_sync(&self) -> SyncDelta {
        let entries: Vec<SyncEntry> = self.log.entries.iter().cloned().collect();
        SyncDelta::full_sync(entries)
    }

    pub fn acknowledge(&mut self, client_id: &str, version: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let state = self.client_states
            .entry(client_id.to_string())
            .or_insert_with(|| SyncState::new(client_id));

        state.last_sync_version = version;
        state.last_sync_timestamp = now;
    }

    pub fn apply_delta(&mut self, delta: &SyncDelta) -> Result<usize, SyncError> {
        let mut applied = 0;

        for entry in &delta.entries {
            if entry.version > self.log.current_version() || delta.is_full_sync {
                self.log.append(entry.clone());
                applied += 1;
            }
        }

        Ok(applied)
    }

    pub fn get_client_state(&self, client_id: &str) -> Option<&SyncState> {
        self.client_states.get(client_id)
    }

    pub fn current_version(&self) -> u64 {
        self.log.current_version()
    }

    pub fn compact(&mut self) {
        self.log.compact();
    }
}

impl Default for IncrementalSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Sync errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncError {
    VersionMismatch,
    InvalidDelta,
    ConflictDetected,
    NetworkError,
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionMismatch => write!(f, "Version mismatch"),
            Self::InvalidDelta => write!(f, "Invalid delta"),
            Self::ConflictDetected => write!(f, "Conflict detected"),
            Self::NetworkError => write!(f, "Network error"),
        }
    }
}

impl std::error::Error for SyncError {}

/// Two-way sync coordinator
pub struct TwoWaySyncCoordinator {
    local: IncrementalSyncManager,
    pending_outbound: Vec<SyncEntry>,
    pending_inbound: Vec<SyncEntry>,
}

impl TwoWaySyncCoordinator {
    pub fn new() -> Self {
        Self {
            local: IncrementalSyncManager::new(),
            pending_outbound: Vec::new(),
            pending_inbound: Vec::new(),
        }
    }

    pub fn local_change(&mut self, entry: SyncEntry) {
        self.local.log.append(entry.clone());
        self.pending_outbound.push(entry);
    }

    pub fn receive_remote(&mut self, delta: SyncDelta) {
        for entry in delta.entries {
            self.pending_inbound.push(entry);
        }
    }

    pub fn get_outbound_delta(&mut self) -> SyncDelta {
        let entries = std::mem::take(&mut self.pending_outbound);
        SyncDelta::new(
            self.local.current_version().saturating_sub(entries.len() as u64),
            self.local.current_version(),
            entries,
        )
    }

    pub fn apply_inbound(&mut self) -> Vec<SyncEntry> {
        let entries = std::mem::take(&mut self.pending_inbound);
        for entry in &entries {
            self.local.log.append(entry.clone());
        }
        entries
    }

    pub fn has_pending(&self) -> bool {
        !self.pending_outbound.is_empty() || !self.pending_inbound.is_empty()
    }
}

impl Default for TwoWaySyncCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_log() {
        let mut log = SyncLog::new(100);

        log.append(SyncEntry::add("message", "1", "Hello"));
        log.append(SyncEntry::add("message", "2", "World"));

        assert_eq!(log.current_version(), 2);
    }

    #[test]
    fn test_get_delta() {
        let mut manager = IncrementalSyncManager::new();

        manager.record_add("message", "1", "First");
        manager.record_add("message", "2", "Second");

        let delta = manager.get_delta("client1");
        assert_eq!(delta.entries.len(), 2);
    }

    #[test]
    fn test_acknowledge() {
        let mut manager = IncrementalSyncManager::new();

        manager.record_add("message", "1", "Test");
        manager.acknowledge("client1", 1);

        let state = manager.get_client_state("client1").unwrap();
        assert_eq!(state.last_sync_version, 1);
    }

    #[test]
    fn test_two_way_sync() {
        let mut coordinator = TwoWaySyncCoordinator::new();

        coordinator.local_change(SyncEntry::add("message", "1", "Local change"));

        let outbound = coordinator.get_outbound_delta();
        assert_eq!(outbound.entries.len(), 1);
    }
}
