//! Unified Storage Context — persist and restore all subsystem state.
//!
//! Inspired by LlamaIndex's StorageContext pattern: a single object that
//! coordinates persistence of all subsystems to a configurable data directory.
//!
//! # Design
//!
//! - JSON files for config-like data (tiers, bandit, watcher, snapshots)
//! - SQLite (via UnifiedDb) for write-heavy data (audit, graph layers)
//! - Auto-save with configurable intervals and dirty-flag batching
//! - drain_writes() guarantee for error paths and shutdown

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// Configuration for the storage context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StorageConfig {
    /// Root directory for all persisted data.
    pub data_dir: PathBuf,
    /// Auto-save interval in seconds (0 = manual only).
    pub auto_save_interval_secs: u64,
    /// Whether to restore state on startup.
    pub restore_on_startup: bool,
    /// Whether to persist on shutdown (Drop).
    pub persist_on_shutdown: bool,
    /// Schema version for migration support.
    pub schema_version: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(".ai_assistant_data"),
            auto_save_interval_secs: 30,
            restore_on_startup: true,
            persist_on_shutdown: true,
            schema_version: 1,
        }
    }
}

/// Tracks which subsystems have unsaved changes.
#[derive(Debug, Default)]
pub struct DirtyFlags {
    pub bandit: AtomicBool,
    pub tiers: AtomicBool,
    pub procedures: AtomicBool,
    pub snapshots: AtomicBool,
    pub watcher: AtomicBool,
    pub audit: AtomicBool,
    pub graph_layers: AtomicBool,
}

impl DirtyFlags {
    /// Mark a subsystem as dirty (has unsaved changes).
    pub fn mark(&self, subsystem: Subsystem) {
        match subsystem {
            Subsystem::Bandit => self.bandit.store(true, Ordering::Relaxed),
            Subsystem::Tiers => self.tiers.store(true, Ordering::Relaxed),
            Subsystem::Procedures => self.procedures.store(true, Ordering::Relaxed),
            Subsystem::Snapshots => self.snapshots.store(true, Ordering::Relaxed),
            Subsystem::Watcher => self.watcher.store(true, Ordering::Relaxed),
            Subsystem::Audit => self.audit.store(true, Ordering::Relaxed),
            Subsystem::GraphLayers => self.graph_layers.store(true, Ordering::Relaxed),
        }
    }

    /// Check and clear a dirty flag. Returns true if it was dirty.
    pub fn take(&self, subsystem: Subsystem) -> bool {
        match subsystem {
            Subsystem::Bandit => self.bandit.swap(false, Ordering::Relaxed),
            Subsystem::Tiers => self.tiers.swap(false, Ordering::Relaxed),
            Subsystem::Procedures => self.procedures.swap(false, Ordering::Relaxed),
            Subsystem::Snapshots => self.snapshots.swap(false, Ordering::Relaxed),
            Subsystem::Watcher => self.watcher.swap(false, Ordering::Relaxed),
            Subsystem::Audit => self.audit.swap(false, Ordering::Relaxed),
            Subsystem::GraphLayers => self.graph_layers.swap(false, Ordering::Relaxed),
        }
    }

    /// Check if any subsystem is dirty.
    pub fn any_dirty(&self) -> bool {
        self.bandit.load(Ordering::Relaxed)
            || self.tiers.load(Ordering::Relaxed)
            || self.procedures.load(Ordering::Relaxed)
            || self.snapshots.load(Ordering::Relaxed)
            || self.watcher.load(Ordering::Relaxed)
            || self.audit.load(Ordering::Relaxed)
            || self.graph_layers.load(Ordering::Relaxed)
    }
}

/// Identifiers for persistable subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsystem {
    Bandit,
    Tiers,
    Procedures,
    Snapshots,
    Watcher,
    Audit,
    GraphLayers,
}

/// Unified storage context that coordinates persistence of all subsystems.
pub struct StorageContext {
    config: StorageConfig,
    dirty: DirtyFlags,
}

impl StorageContext {
    /// Create a new storage context with the given configuration.
    /// Creates the data directory if it doesn't exist.
    pub fn new(config: StorageConfig) -> Self {
        let _ = std::fs::create_dir_all(&config.data_dir);
        Self {
            config,
            dirty: DirtyFlags::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StorageConfig::default())
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
    }

    /// Get the path for a specific subsystem's data file.
    pub fn subsystem_path(&self, name: &str) -> PathBuf {
        self.config.data_dir.join(format!("{}.json", name))
    }

    /// Get the dirty flags (for subsystems to mark themselves).
    pub fn dirty(&self) -> &DirtyFlags {
        &self.dirty
    }

    /// Save a serializable value to a JSON file in the data directory.
    /// Uses atomic write (temp file + rename) for crash safety.
    pub fn save_json<T: Serialize>(&self, name: &str, data: &T) -> Result<(), String> {
        let path = self.subsystem_path(name);
        let json =
            serde_json::to_string_pretty(data).map_err(|e| format!("Serialize {}: {}", name, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json).map_err(|e| format!("Write {}: {}", name, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("Rename {}: {}", name, e))?;
        Ok(())
    }

    /// Load a deserializable value from a JSON file in the data directory.
    pub fn load_json<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T, String> {
        let path = self.subsystem_path(name);
        let data =
            std::fs::read_to_string(&path).map_err(|e| format!("Read {}: {}", name, e))?;
        serde_json::from_str(&data).map_err(|e| format!("Deserialize {}: {}", name, e))
    }

    /// Check if a subsystem's data file exists.
    pub fn exists(&self, name: &str) -> bool {
        self.subsystem_path(name).exists()
    }

    /// Drain all dirty writes — flush every subsystem that has pending changes.
    /// Call this in error paths, shutdown handlers, and Drop.
    ///
    /// Takes closures for each subsystem's save logic since StorageContext
    /// doesn't own the subsystem data.
    pub fn drain_writes<F>(&self, mut save_fn: F)
    where
        F: FnMut(Subsystem) -> Result<(), String>,
    {
        let subsystems = [
            Subsystem::Bandit,
            Subsystem::Tiers,
            Subsystem::Procedures,
            Subsystem::Snapshots,
            Subsystem::Watcher,
            Subsystem::Audit,
            Subsystem::GraphLayers,
        ];

        for sub in &subsystems {
            if self.dirty.take(*sub) {
                if let Err(e) = save_fn(*sub) {
                    eprintln!("[StorageContext] drain_writes error for {:?}: {}", sub, e);
                }
            }
        }
    }

    /// List all persisted subsystem files in the data directory.
    pub fn list_persisted(&self) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.config.data_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.path().file_stem() {
                    if entry.path().extension().map(|e| e == "json").unwrap_or(false) {
                        names.push(name.to_string_lossy().to_string());
                    }
                }
            }
        }
        names.sort();
        names
    }

    /// Total size of all persisted data in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.config.data_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                }
            }
        }
        total
    }
}

impl Default for StorageContext {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_test_storage_{}",
            uuid::Uuid::new_v4()
        ));
        dir
    }

    #[test]
    fn test_storage_context_creates_dir() {
        let dir = test_dir();
        let config = StorageConfig {
            data_dir: dir.clone(),
            ..Default::default()
        };
        let _ctx = StorageContext::new(config);
        assert!(dir.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_json_roundtrip() {
        let dir = test_dir();
        let ctx = StorageContext::new(StorageConfig {
            data_dir: dir.clone(),
            ..Default::default()
        });

        // Save
        let data = serde_json::json!({"key": "value", "count": 42});
        ctx.save_json("test_data", &data).unwrap();

        // Load
        let loaded: serde_json::Value = ctx.load_json("test_data").unwrap();
        assert_eq!(loaded["key"], "value");
        assert_eq!(loaded["count"], 42);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dirty_flags() {
        let flags = DirtyFlags::default();
        assert!(!flags.any_dirty());

        flags.mark(Subsystem::Bandit);
        assert!(flags.any_dirty());

        let was_dirty = flags.take(Subsystem::Bandit);
        assert!(was_dirty);
        assert!(!flags.any_dirty());
    }

    #[test]
    fn test_drain_writes() {
        let dir = test_dir();
        let ctx = StorageContext::new(StorageConfig {
            data_dir: dir.clone(),
            ..Default::default()
        });

        ctx.dirty().mark(Subsystem::Bandit);
        ctx.dirty().mark(Subsystem::Tiers);

        let mut saved = Vec::new();
        ctx.drain_writes(|sub| {
            saved.push(sub);
            Ok(())
        });

        assert!(saved.contains(&Subsystem::Bandit));
        assert!(saved.contains(&Subsystem::Tiers));
        assert!(!ctx.dirty().any_dirty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_exists_and_list() {
        let dir = test_dir();
        let ctx = StorageContext::new(StorageConfig {
            data_dir: dir.clone(),
            ..Default::default()
        });

        assert!(!ctx.exists("bandit"));
        ctx.save_json("bandit", &serde_json::json!({})).unwrap();
        assert!(ctx.exists("bandit"));

        let listed = ctx.list_persisted();
        assert!(listed.contains(&"bandit".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_total_size() {
        let dir = test_dir();
        let ctx = StorageContext::new(StorageConfig {
            data_dir: dir.clone(),
            ..Default::default()
        });

        assert_eq!(ctx.total_size_bytes(), 0);
        ctx.save_json("test", &serde_json::json!({"data": "hello"}))
            .unwrap();
        assert!(ctx.total_size_bytes() > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_subsystem_path() {
        let ctx = StorageContext::new(StorageConfig {
            data_dir: PathBuf::from("/tmp/test"),
            ..Default::default()
        });
        let path = ctx.subsystem_path("bandit");
        assert_eq!(path, PathBuf::from("/tmp/test/bandit.json"));
    }
}
