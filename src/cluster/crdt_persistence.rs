//! # CRDT Persistence (Phase 10)
//!
//! Durability layer for CRDT state:
//! - **Snapshots**: Periodic full serialization of CRDT state to disk
//! - **WAL (Write-Ahead Log)**: Append-only log of CRDT operations for crash recovery
//! - **TTL wrapper**: Time-based expiry for rate limit windows and sessions
//! - **Compaction**: Garbage collect old WAL entries and ORSet tombstones
//!
//! All operations are designed to be safe for concurrent access and crash recovery.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ============================================================================
// CrdtPersistence
// ============================================================================

/// Manages CRDT persistence via snapshots and WAL.
pub struct CrdtPersistence {
    /// Base directory for all persistence data.
    data_dir: PathBuf,
    /// Snapshot subdirectory.
    snapshot_dir: PathBuf,
    /// WAL subdirectory.
    wal_dir: PathBuf,
}

impl CrdtPersistence {
    /// Create a new persistence manager.
    ///
    /// Creates the data directories if they don't exist.
    pub fn new(data_dir: PathBuf) -> Self {
        let snapshot_dir = data_dir.join("snapshots");
        let wal_dir = data_dir.join("wal");
        let _ = fs::create_dir_all(&snapshot_dir);
        let _ = fs::create_dir_all(&wal_dir);
        Self {
            data_dir,
            snapshot_dir,
            wal_dir,
        }
    }

    /// Write a snapshot of a named CRDT to disk.
    ///
    /// Uses atomic write (write to temp, then rename) for crash safety.
    pub fn write_snapshot(&self, name: &str, data: &[u8]) -> Result<PathBuf, String> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let filename = format!("{}_{}.snap", name, ts);
        let path = self.snapshot_dir.join(&filename);
        let tmp_path = self.snapshot_dir.join(format!("{}.tmp", filename));

        fs::write(&tmp_path, data)
            .map_err(|e| format!("Failed to write snapshot tmp: {}", e))?;
        fs::rename(&tmp_path, &path)
            .map_err(|e| format!("Failed to rename snapshot: {}", e))?;

        Ok(path)
    }

    /// Load the latest snapshot for a named CRDT.
    ///
    /// Returns the data bytes, or None if no snapshot exists.
    pub fn load_latest_snapshot(&self, name: &str) -> Result<Option<Vec<u8>>, String> {
        let prefix = format!("{}_", name);
        let mut snapshots: Vec<_> = fs::read_dir(&self.snapshot_dir)
            .map_err(|e| format!("Failed to read snapshot dir: {}", e))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_string_lossy().starts_with(&prefix)
                    && entry.file_name().to_string_lossy().ends_with(".snap")
            })
            .collect();

        if snapshots.is_empty() {
            return Ok(None);
        }

        // Sort by name (timestamp is embedded) to get latest
        snapshots.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

        let latest = &snapshots[0];
        let data = fs::read(latest.path())
            .map_err(|e| format!("Failed to read snapshot: {}", e))?;
        Ok(Some(data))
    }

    /// Append a WAL entry for crash recovery.
    pub fn append_wal(&self, name: &str, operation: &WalEntry) -> Result<(), String> {
        let wal_file = self.wal_dir.join(format!("{}.wal", name));
        let data = serde_json::to_vec(operation)
            .map_err(|e| format!("Failed to serialize WAL entry: {}", e))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_file)
            .map_err(|e| format!("Failed to open WAL file: {}", e))?;

        file.write_all(&data)
            .map_err(|e| format!("Failed to write WAL entry: {}", e))?;
        file.write_all(b"\n")
            .map_err(|e| format!("Failed to write WAL newline: {}", e))?;

        Ok(())
    }

    /// Read all WAL entries for a named CRDT (for replay after crash).
    pub fn read_wal(&self, name: &str) -> Result<Vec<WalEntry>, String> {
        let wal_file = self.wal_dir.join(format!("{}.wal", name));
        if !wal_file.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&wal_file)
            .map_err(|e| format!("Failed to read WAL file: {}", e))?;

        let mut entries = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    log::warn!("Skipping corrupt WAL entry: {}", e);
                }
            }
        }
        Ok(entries)
    }

    /// Compact WAL: truncate after successful snapshot.
    pub fn compact_wal(&self, name: &str) -> Result<(), String> {
        let wal_file = self.wal_dir.join(format!("{}.wal", name));
        if wal_file.exists() {
            fs::write(&wal_file, b"")
                .map_err(|e| format!("Failed to compact WAL: {}", e))?;
        }
        Ok(())
    }

    /// Remove old snapshots, keeping only the N most recent.
    pub fn prune_snapshots(&self, name: &str, keep: usize) -> Result<usize, String> {
        let prefix = format!("{}_", name);
        let mut snapshots: Vec<_> = fs::read_dir(&self.snapshot_dir)
            .map_err(|e| format!("Failed to read snapshot dir: {}", e))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_string_lossy().starts_with(&prefix)
                    && entry.file_name().to_string_lossy().ends_with(".snap")
            })
            .collect();

        if snapshots.len() <= keep {
            return Ok(0);
        }

        snapshots.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

        let mut removed = 0;
        for old in &snapshots[keep..] {
            if fs::remove_file(old.path()).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Get the data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

// ============================================================================
// WAL Entry
// ============================================================================

/// A single WAL operation for crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Timestamp (millis since epoch).
    pub timestamp_ms: u64,
    /// Node that performed the operation.
    pub node_id: String,
    /// CRDT name.
    pub crdt_name: String,
    /// Operation type.
    pub operation: WalOperation,
}

/// CRDT operation types for WAL replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WalOperation {
    /// Increment a counter.
    Increment { key: String, amount: u64 },
    /// Decrement a counter.
    Decrement { key: String, amount: u64 },
    /// Set a register value.
    SetRegister { key: String, value: Vec<u8>, timestamp: u64 },
    /// Add to a set.
    AddToSet { element: String },
    /// Remove from a set.
    RemoveFromSet { element: String },
    /// Set a map entry.
    SetMap { key: String, value: Vec<u8>, timestamp: u64 },
    /// Merge with remote state (full serialized CRDT).
    Merge { remote_state: Vec<u8> },
}

// ============================================================================
// TTL Wrapper
// ============================================================================

/// Wraps a value with a time-to-live for automatic expiration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtlValue<T: Clone> {
    /// The wrapped value.
    pub value: T,
    /// Creation timestamp (millis since epoch).
    pub created_at_ms: u64,
    /// TTL in milliseconds.
    pub ttl_ms: u64,
}

impl<T: Clone> TtlValue<T> {
    /// Create a new TTL-wrapped value.
    pub fn new(value: T, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            value,
            created_at_ms: now,
            ttl_ms: ttl.as_millis() as u64,
        }
    }

    /// Check if this value has expired.
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.created_at_ms) > self.ttl_ms
    }

    /// Get the remaining time before expiration.
    pub fn remaining(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let elapsed = now.saturating_sub(self.created_at_ms);
        if elapsed >= self.ttl_ms {
            Duration::ZERO
        } else {
            Duration::from_millis(self.ttl_ms - elapsed)
        }
    }

    /// Get the inner value (regardless of expiration).
    pub fn get(&self) -> &T {
        &self.value
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("crdt_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_persistence_new_creates_dirs() {
        let dir = temp_dir().join("persist_new");
        let _p = CrdtPersistence::new(dir.clone());
        assert!(dir.join("snapshots").exists());
        assert!(dir.join("wal").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_and_load_snapshot() {
        let dir = temp_dir().join("snap_test");
        let p = CrdtPersistence::new(dir.clone());

        let data = b"test snapshot data";
        p.write_snapshot("counter", data).unwrap();

        let loaded = p.load_latest_snapshot("counter").unwrap();
        assert_eq!(loaded, Some(data.to_vec()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_latest_snapshot_returns_newest() {
        let dir = temp_dir().join("snap_latest");
        let p = CrdtPersistence::new(dir.clone());

        p.write_snapshot("counter", b"old").unwrap();
        thread::sleep(Duration::from_millis(5));
        p.write_snapshot("counter", b"new").unwrap();

        let loaded = p.load_latest_snapshot("counter").unwrap();
        assert_eq!(loaded, Some(b"new".to_vec()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_snapshot_none_when_empty() {
        let dir = temp_dir().join("snap_empty");
        let p = CrdtPersistence::new(dir.clone());

        let loaded = p.load_latest_snapshot("nonexistent").unwrap();
        assert!(loaded.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wal_append_and_read() {
        let dir = temp_dir().join("wal_test");
        let p = CrdtPersistence::new(dir.clone());

        let entry = WalEntry {
            timestamp_ms: 1000,
            node_id: "node1".to_string(),
            crdt_name: "counter".to_string(),
            operation: WalOperation::Increment {
                key: "node1".to_string(),
                amount: 5,
            },
        };

        p.append_wal("counter", &entry).unwrap();
        p.append_wal("counter", &entry).unwrap();

        let entries = p.read_wal("counter").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].node_id, "node1");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wal_read_empty() {
        let dir = temp_dir().join("wal_empty");
        let p = CrdtPersistence::new(dir.clone());

        let entries = p.read_wal("nonexistent").unwrap();
        assert!(entries.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compact_wal() {
        let dir = temp_dir().join("wal_compact");
        let p = CrdtPersistence::new(dir.clone());

        let entry = WalEntry {
            timestamp_ms: 1000,
            node_id: "node1".to_string(),
            crdt_name: "counter".to_string(),
            operation: WalOperation::Increment {
                key: "node1".to_string(),
                amount: 1,
            },
        };

        p.append_wal("counter", &entry).unwrap();
        assert!(!p.read_wal("counter").unwrap().is_empty());

        p.compact_wal("counter").unwrap();
        assert!(p.read_wal("counter").unwrap().is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prune_snapshots() {
        let dir = temp_dir().join("snap_prune");
        let p = CrdtPersistence::new(dir.clone());

        for i in 0..5 {
            p.write_snapshot("counter", format!("data{}", i).as_bytes()).unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        let removed = p.prune_snapshots("counter", 2).unwrap();
        assert_eq!(removed, 3);

        // Verify latest is still accessible
        let loaded = p.load_latest_snapshot("counter").unwrap();
        assert!(loaded.is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ttl_value_not_expired() {
        let val = TtlValue::new(42u64, Duration::from_secs(60));
        assert!(!val.is_expired());
        assert_eq!(*val.get(), 42);
        assert!(val.remaining() > Duration::ZERO);
    }

    #[test]
    fn test_ttl_value_expired() {
        let val = TtlValue {
            value: 42u64,
            created_at_ms: 0, // Way in the past
            ttl_ms: 1,
        };
        assert!(val.is_expired());
        assert_eq!(val.remaining(), Duration::ZERO);
    }

    #[test]
    fn test_ttl_value_serialization() {
        let val = TtlValue::new("hello".to_string(), Duration::from_secs(30));
        let json = serde_json::to_string(&val).unwrap();
        let parsed: TtlValue<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.value, "hello");
        assert_eq!(parsed.ttl_ms, 30_000);
    }

    #[test]
    fn test_wal_operation_variants() {
        let ops = vec![
            WalOperation::Increment { key: "k".to_string(), amount: 1 },
            WalOperation::Decrement { key: "k".to_string(), amount: 1 },
            WalOperation::SetRegister { key: "k".to_string(), value: vec![1, 2], timestamp: 100 },
            WalOperation::AddToSet { element: "e".to_string() },
            WalOperation::RemoveFromSet { element: "e".to_string() },
            WalOperation::SetMap { key: "k".to_string(), value: vec![3], timestamp: 200 },
            WalOperation::Merge { remote_state: vec![4, 5] },
        ];

        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let _: WalOperation = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_wal_entry_serialization() {
        let entry = WalEntry {
            timestamp_ms: 12345,
            node_id: "node1".to_string(),
            crdt_name: "sessions".to_string(),
            operation: WalOperation::SetMap {
                key: "sess1".to_string(),
                value: vec![1, 2, 3],
                timestamp: 12345,
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: WalEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.timestamp_ms, 12345);
        assert_eq!(parsed.crdt_name, "sessions");
    }
}
