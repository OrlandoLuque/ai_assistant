//! Tool Safety Profiles and Filesystem Snapshot/Rollback System
//!
//! Provides safety classification for tools (interruptible, reversible, destructive)
//! and a cross-platform SnapshotStore for file operation rollback.
//!
//! # Design
//!
//! Every tool can declare a `ToolSafetyProfile` describing its behavior:
//! - Can it be interrupted mid-execution?
//! - Does it have side effects?
//! - Are those side effects reversible?
//! - Should a snapshot be taken before execution?
//!
//! The `SnapshotStore` provides a cross-platform trash/backup system
//! (NOT dependent on OS trash — works on Linux server, macOS, Windows).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Tool Safety Profile
// ============================================================================

/// Safety classification for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSafetyProfile {
    /// Can the tool be stopped mid-execution?
    pub interruptible: bool,
    /// Does the tool modify external state (files, APIs, databases)?
    pub has_side_effects: bool,
    /// Can the side effects be undone?
    pub reversible: bool,
    /// Should a filesystem snapshot be taken before execution?
    pub snapshot_before: bool,
    /// Maximum execution time before forced timeout (ms). 0 = no timeout.
    pub timeout_ms: u64,
    /// Does this tool support dry-run mode?
    pub dry_run_supported: bool,
    /// Heartbeat interval for long-running tools (ms). 0 = no heartbeat.
    pub heartbeat_interval_ms: u64,
}

impl ToolSafetyProfile {
    /// Read-only tool (search, query, read_file). Safe to interrupt, no side effects.
    pub fn read_only() -> Self {
        Self {
            interruptible: true,
            has_side_effects: false,
            reversible: true,
            snapshot_before: false,
            timeout_ms: 30_000,
            dry_run_supported: false,
            heartbeat_interval_ms: 0,
        }
    }

    /// Destructive but reversible (write_file, delete_file, edit_file).
    pub fn destructive_reversible() -> Self {
        Self {
            interruptible: false,
            has_side_effects: true,
            reversible: true,
            snapshot_before: true,
            timeout_ms: 60_000,
            dry_run_supported: true,
            heartbeat_interval_ms: 0,
        }
    }

    /// Destructive and irreversible (send_email, post_to_api).
    pub fn destructive_irreversible() -> Self {
        Self {
            interruptible: false,
            has_side_effects: true,
            reversible: false,
            snapshot_before: false,
            timeout_ms: 30_000,
            dry_run_supported: true,
            heartbeat_interval_ms: 0,
        }
    }

    /// Long-running interruptible (run_code, deploy, web_scrape).
    pub fn long_running() -> Self {
        Self {
            interruptible: true,
            has_side_effects: true,
            reversible: false,
            snapshot_before: false,
            timeout_ms: 300_000, // 5 minutes
            dry_run_supported: false,
            heartbeat_interval_ms: 5_000,
        }
    }
}

impl Default for ToolSafetyProfile {
    fn default() -> Self {
        Self::read_only()
    }
}

// ============================================================================
// File Operations
// ============================================================================

/// Type of filesystem operation being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FileOperation {
    Write,
    Edit,
    Delete,
    Create,
    CreateDir,
    Rename,
    Move,
    Copy,
    Chmod,
    Append,
}

impl std::fmt::Display for FileOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write => write!(f, "write"),
            Self::Edit => write!(f, "edit"),
            Self::Delete => write!(f, "delete"),
            Self::Create => write!(f, "create"),
            Self::CreateDir => write!(f, "create_dir"),
            Self::Rename => write!(f, "rename"),
            Self::Move => write!(f, "move"),
            Self::Copy => write!(f, "copy"),
            Self::Chmod => write!(f, "chmod"),
            Self::Append => write!(f, "append"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// File Snapshot
// ============================================================================

/// Snapshot of a file's state before a destructive operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    /// What operation was performed.
    pub operation: FileOperation,
    /// Original path of the file.
    pub original_path: PathBuf,
    /// Path where the backup is stored (in trash dir).
    pub backup_path: Option<PathBuf>,
    /// Original file content (for small files, stored inline).
    #[serde(skip)]
    pub content: Option<Vec<u8>>,
    /// Original file size in bytes.
    pub original_size: Option<u64>,
    /// Destination path (for rename/move/copy operations).
    pub destination_path: Option<PathBuf>,
    /// Unix timestamp when the snapshot was created.
    pub timestamp: u64,
    /// Tool call ID that triggered this operation.
    pub tool_call_id: String,
    /// Task/iteration ID for grouping.
    pub iteration_id: String,
    /// Whether this snapshot has been compensated (rolled back).
    pub compensated: bool,
}

// ============================================================================
// Rollback Strategy
// ============================================================================

/// How to handle rollback of file operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RollbackStrategy {
    /// Internal snapshot store (default, cross-platform, no git dependency).
    Snapshot,
    /// Git-based rollback (opt-in).
    Git(GitRollbackConfig),
    /// No rollback tracking.
    None,
}

impl Default for RollbackStrategy {
    fn default() -> Self {
        Self::Snapshot
    }
}

/// Git-based rollback configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRollbackConfig {
    /// Git rollback mode.
    pub mode: GitRollbackMode,
    /// Auto-squash commits on successful task completion.
    pub auto_squash_on_success: bool,
    /// Auto-cleanup temporary branches on completion.
    pub auto_cleanup_branch: bool,
    /// Prefix for automatic commits.
    pub commit_prefix: String,
    /// Prefix for temporary branches.
    pub branch_prefix: String,
    /// Maximum auto-commits before forced squash.
    pub max_auto_commits: usize,
}

impl Default for GitRollbackConfig {
    fn default() -> Self {
        Self {
            mode: GitRollbackMode::Branch,
            auto_squash_on_success: true,
            auto_cleanup_branch: true,
            commit_prefix: "[ai-checkpoint]".to_string(),
            branch_prefix: "ai-agent/".to_string(),
            max_auto_commits: 50,
        }
    }
}

/// Git rollback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GitRollbackMode {
    /// Commit on current branch. Rollback = reset.
    Commit,
    /// Git stash. Single level.
    Stash,
    /// Temporary branch. Clean main history.
    Branch,
    /// Isolated worktree. Maximum isolation.
    Worktree,
    /// Lightweight tags. Multi-level, cheap.
    Tag,
}

// ============================================================================
// Snapshot Store
// ============================================================================

/// Cross-platform snapshot store for file operation rollback.
///
/// Uses an internal trash directory (NOT the OS trash). Works identically
/// on Linux (with or without desktop), macOS, and Windows.
pub struct SnapshotStore {
    /// Directory for storing backups.
    trash_dir: PathBuf,
    /// Active snapshots indexed by tool_call_id.
    snapshots: HashMap<String, Vec<FileSnapshot>>,
    /// Maximum number of snapshots to retain (LRU eviction).
    max_snapshots: usize,
    /// Time-to-live for snapshots before auto-cleanup.
    ttl: Duration,
    /// Total snapshot count.
    total_count: usize,
}

impl SnapshotStore {
    /// Create a new snapshot store with the given trash directory.
    pub fn new(trash_dir: impl Into<PathBuf>) -> Self {
        let dir = trash_dir.into();
        let _ = std::fs::create_dir_all(&dir);
        Self {
            trash_dir: dir,
            snapshots: HashMap::new(),
            max_snapshots: 100,
            ttl: Duration::from_secs(3600), // 1 hour
            total_count: 0,
        }
    }

    /// Create with default trash directory.
    pub fn default_dir() -> Self {
        Self::new(".ai_assistant_trash")
    }

    /// Set maximum snapshots.
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Set TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Take a snapshot before a write/edit operation.
    pub fn snapshot_write(
        &mut self,
        path: &Path,
        tool_call_id: &str,
        iteration_id: &str,
    ) -> Result<FileSnapshot, String> {
        let content = std::fs::read(path).ok();
        let original_size = std::fs::metadata(path).ok().map(|m| m.len());

        let snapshot = FileSnapshot {
            operation: if content.is_some() {
                FileOperation::Write
            } else {
                FileOperation::Create
            },
            original_path: path.to_path_buf(),
            backup_path: None,
            content,
            original_size,
            destination_path: None,
            timestamp: now_secs(),
            tool_call_id: tool_call_id.to_string(),
            iteration_id: iteration_id.to_string(),
            compensated: false,
        };

        self.store_snapshot(snapshot.clone());
        Ok(snapshot)
    }

    /// Take a snapshot before a delete operation (moves file to trash).
    pub fn snapshot_delete(
        &mut self,
        path: &Path,
        tool_call_id: &str,
        iteration_id: &str,
    ) -> Result<FileSnapshot, String> {
        if !path.exists() {
            return Err(format!("File does not exist: {}", path.display()));
        }

        // Move to trash instead of deleting
        let trash_subdir = self.trash_dir.join(tool_call_id);
        let _ = std::fs::create_dir_all(&trash_subdir);

        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let backup_path = trash_subdir.join(&filename);

        std::fs::rename(path, &backup_path)
            .map_err(|e| format!("Failed to move to trash: {}", e))?;

        let snapshot = FileSnapshot {
            operation: FileOperation::Delete,
            original_path: path.to_path_buf(),
            backup_path: Some(backup_path),
            content: None,
            original_size: None,
            destination_path: None,
            timestamp: now_secs(),
            tool_call_id: tool_call_id.to_string(),
            iteration_id: iteration_id.to_string(),
            compensated: false,
        };

        self.store_snapshot(snapshot.clone());
        Ok(snapshot)
    }

    /// Take a snapshot before a rename/move operation.
    pub fn snapshot_rename(
        &mut self,
        from: &Path,
        to: &Path,
        tool_call_id: &str,
        iteration_id: &str,
    ) -> Result<FileSnapshot, String> {
        let snapshot = FileSnapshot {
            operation: FileOperation::Rename,
            original_path: from.to_path_buf(),
            backup_path: None,
            content: None,
            original_size: None,
            destination_path: Some(to.to_path_buf()),
            timestamp: now_secs(),
            tool_call_id: tool_call_id.to_string(),
            iteration_id: iteration_id.to_string(),
            compensated: false,
        };

        self.store_snapshot(snapshot.clone());
        Ok(snapshot)
    }

    /// Rollback a single snapshot.
    pub fn rollback(&mut self, snapshot: &FileSnapshot) -> Result<(), String> {
        match snapshot.operation {
            FileOperation::Write | FileOperation::Edit => {
                // Restore original content
                if let Some(ref content) = snapshot.content {
                    std::fs::write(&snapshot.original_path, content)
                        .map_err(|e| format!("Rollback write failed: {}", e))?;
                }
            }
            FileOperation::Create => {
                // Delete the created file
                if snapshot.original_path.exists() {
                    std::fs::remove_file(&snapshot.original_path)
                        .map_err(|e| format!("Rollback create failed: {}", e))?;
                }
            }
            FileOperation::Delete => {
                // Move back from trash
                if let Some(ref backup) = snapshot.backup_path {
                    if backup.exists() {
                        std::fs::rename(backup, &snapshot.original_path)
                            .map_err(|e| format!("Rollback delete failed: {}", e))?;
                    }
                }
            }
            FileOperation::Rename | FileOperation::Move => {
                // Move back to original path
                if let Some(ref dest) = snapshot.destination_path {
                    if dest.exists() {
                        std::fs::rename(dest, &snapshot.original_path)
                            .map_err(|e| format!("Rollback rename failed: {}", e))?;
                    }
                }
            }
            FileOperation::Copy => {
                // Delete the copy
                if let Some(ref dest) = snapshot.destination_path {
                    if dest.exists() {
                        std::fs::remove_file(dest)
                            .map_err(|e| format!("Rollback copy failed: {}", e))?;
                    }
                }
            }
            FileOperation::Append => {
                // Truncate to original size
                if let Some(original_size) = snapshot.original_size {
                    let file = std::fs::OpenOptions::new()
                        .write(true)
                        .open(&snapshot.original_path)
                        .map_err(|e| format!("Rollback append open failed: {}", e))?;
                    file.set_len(original_size)
                        .map_err(|e| format!("Rollback append truncate failed: {}", e))?;
                }
            }
            FileOperation::CreateDir => {
                // Remove directory if empty
                if snapshot.original_path.exists() {
                    let _ = std::fs::remove_dir(&snapshot.original_path);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Rollback all snapshots from a specific iteration (in reverse order — LIFO).
    pub fn rollback_iteration(&mut self, iteration_id: &str) -> Result<usize, String> {
        let mut to_rollback: Vec<FileSnapshot> = self
            .snapshots
            .values()
            .flatten()
            .filter(|s| s.iteration_id == iteration_id && !s.compensated)
            .cloned()
            .collect();

        // LIFO order — reverse chronological
        to_rollback.sort_by_key(|e| std::cmp::Reverse(e.timestamp));

        let mut count = 0;
        for snapshot in &to_rollback {
            self.rollback(snapshot)?;
            count += 1;
        }

        // Mark as compensated
        for snapshots in self.snapshots.values_mut() {
            for s in snapshots.iter_mut() {
                if s.iteration_id == iteration_id {
                    s.compensated = true;
                }
            }
        }

        Ok(count)
    }

    /// List all active (non-compensated) snapshots.
    pub fn list_active(&self) -> Vec<&FileSnapshot> {
        self.snapshots
            .values()
            .flatten()
            .filter(|s| !s.compensated)
            .collect()
    }

    /// Total number of active snapshots.
    pub fn active_count(&self) -> usize {
        self.list_active().len()
    }

    /// Clean up expired snapshots based on TTL.
    pub fn cleanup_expired(&mut self) -> usize {
        let now = now_secs();
        let ttl_secs = self.ttl.as_secs();
        let mut removed = 0;

        self.snapshots.retain(|_, snapshots| {
            let before = snapshots.len();
            snapshots.retain(|s| {
                let expired = now.saturating_sub(s.timestamp) > ttl_secs;
                if expired {
                    // Clean up backup files
                    if let Some(ref backup) = s.backup_path {
                        let _ = std::fs::remove_file(backup);
                    }
                }
                !expired
            });
            removed += before - snapshots.len();
            !snapshots.is_empty()
        });

        // Clean up empty trash subdirectories
        if let Ok(entries) = std::fs::read_dir(&self.trash_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let _ = std::fs::remove_dir(entry.path()); // only removes if empty
                }
            }
        }

        removed
    }

    /// Store a snapshot internally.
    fn store_snapshot(&mut self, snapshot: FileSnapshot) {
        self.snapshots
            .entry(snapshot.tool_call_id.clone())
            .or_default()
            .push(snapshot);
        self.total_count += 1;

        // Evict if over limit
        if self.total_count > self.max_snapshots {
            self.cleanup_expired();
        }
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::default_dir()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// Tool Call Record (for saga compensation tracking)
// ============================================================================

/// Record of a tool call with optional snapshot for compensation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Tool name.
    pub tool_name: String,
    /// Unique call ID.
    pub call_id: String,
    /// Arguments passed to the tool.
    pub arguments: serde_json::Value,
    /// Whether the call succeeded.
    pub success: bool,
    /// Output from the tool.
    pub output: String,
    /// Safety profile of the tool.
    pub safety: ToolSafetyProfile,
    /// File snapshot taken before execution (if any).
    #[serde(skip)]
    pub snapshot: Option<FileSnapshot>,
    /// Whether compensation has been executed.
    pub compensated: bool,
    /// Iteration this call belongs to.
    pub iteration_id: String,
    /// Timestamp.
    pub timestamp: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ai_assistant_test_safety_{}", uuid::Uuid::new_v4()));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_safety_profiles() {
        let ro = ToolSafetyProfile::read_only();
        assert!(ro.interruptible);
        assert!(!ro.has_side_effects);

        let dr = ToolSafetyProfile::destructive_reversible();
        assert!(!dr.interruptible);
        assert!(dr.has_side_effects);
        assert!(dr.reversible);
        assert!(dr.snapshot_before);

        let di = ToolSafetyProfile::destructive_irreversible();
        assert!(!di.reversible);

        let lr = ToolSafetyProfile::long_running();
        assert!(lr.interruptible);
        assert!(lr.heartbeat_interval_ms > 0);
    }

    #[test]
    fn test_snapshot_write_and_rollback() {
        let dir = test_dir();
        let file_path = dir.join("test.txt");
        fs::write(&file_path, "original content").unwrap();

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        // Snapshot before write
        let snapshot = store
            .snapshot_write(&file_path, "call-1", "iter-1")
            .unwrap();
        assert!(snapshot.content.is_some());
        assert_eq!(
            String::from_utf8_lossy(snapshot.content.as_ref().unwrap()),
            "original content"
        );

        // Simulate write
        fs::write(&file_path, "modified content").unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "modified content");

        // Rollback
        store.rollback(&snapshot).unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "original content");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_delete_and_rollback() {
        let dir = test_dir();
        let file_path = dir.join("to_delete.txt");
        fs::write(&file_path, "important data").unwrap();

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        // Snapshot delete (moves to trash)
        let snapshot = store
            .snapshot_delete(&file_path, "call-2", "iter-1")
            .unwrap();
        assert!(!file_path.exists()); // File moved to trash
        assert!(snapshot.backup_path.as_ref().unwrap().exists());

        // Rollback (restore from trash)
        store.rollback(&snapshot).unwrap();
        assert!(file_path.exists());
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "important data");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_rename_and_rollback() {
        let dir = test_dir();
        let from = dir.join("old_name.txt");
        let to = dir.join("new_name.txt");
        fs::write(&from, "content").unwrap();

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        let snapshot = store
            .snapshot_rename(&from, &to, "call-3", "iter-1")
            .unwrap();

        // Simulate rename
        fs::rename(&from, &to).unwrap();
        assert!(!from.exists());
        assert!(to.exists());

        // Rollback
        store.rollback(&snapshot).unwrap();
        assert!(from.exists());
        assert!(!to.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rollback_iteration_lifo() {
        let dir = test_dir();
        let file1 = dir.join("file1.txt");
        let file2 = dir.join("file2.txt");
        fs::write(&file1, "original1").unwrap();
        fs::write(&file2, "original2").unwrap();

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        // Two operations in the same iteration
        store.snapshot_write(&file1, "call-a", "iter-1").unwrap();
        fs::write(&file1, "modified1").unwrap();

        store.snapshot_write(&file2, "call-b", "iter-1").unwrap();
        fs::write(&file2, "modified2").unwrap();

        // Rollback entire iteration
        let count = store.rollback_iteration("iter-1").unwrap();
        assert_eq!(count, 2);
        assert_eq!(fs::read_to_string(&file1).unwrap(), "original1");
        assert_eq!(fs::read_to_string(&file2).unwrap(), "original2");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_create_rollback() {
        let dir = test_dir();
        let file_path = dir.join("new_file.txt");

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        // Snapshot for a file that doesn't exist yet (create operation)
        let snapshot = store
            .snapshot_write(&file_path, "call-c", "iter-2")
            .unwrap();
        assert_eq!(snapshot.operation, FileOperation::Create);

        // Simulate create
        fs::write(&file_path, "new content").unwrap();
        assert!(file_path.exists());

        // Rollback = delete
        store.rollback(&snapshot).unwrap();
        assert!(!file_path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_active_count() {
        let dir = test_dir();
        let file = dir.join("counted.txt");
        fs::write(&file, "data").unwrap();

        let trash = dir.join("trash");
        let mut store = SnapshotStore::new(&trash);

        assert_eq!(store.active_count(), 0);
        store.snapshot_write(&file, "c1", "i1").unwrap();
        assert_eq!(store.active_count(), 1);
        store.snapshot_write(&file, "c2", "i1").unwrap();
        assert_eq!(store.active_count(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_file_operation_display() {
        assert_eq!(FileOperation::Write.to_string(), "write");
        assert_eq!(FileOperation::Delete.to_string(), "delete");
        assert_eq!(FileOperation::Rename.to_string(), "rename");
        assert_eq!(FileOperation::Append.to_string(), "append");
    }

    #[test]
    fn test_rollback_strategy_default() {
        let strategy = RollbackStrategy::default();
        assert!(matches!(strategy, RollbackStrategy::Snapshot));
    }

    #[test]
    fn test_git_rollback_config_default() {
        let config = GitRollbackConfig::default();
        assert_eq!(config.mode, GitRollbackMode::Branch);
        assert!(config.auto_squash_on_success);
        assert!(config.auto_cleanup_branch);
    }
}
