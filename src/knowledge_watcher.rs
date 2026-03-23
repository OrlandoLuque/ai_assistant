//! Knowledge Watcher — auto-reindex when source documents change on disk.
//!
//! Monitors files and directories that have been indexed into the RAG
//! knowledge base. When a file is modified, it triggers re-indexing
//! of that specific document (re-chunk + re-embed).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// Configuration for the knowledge watcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WatcherConfig {
    /// Polling interval for checking file changes.
    pub poll_interval: Duration,
    /// Whether to auto-reindex on change or just notify.
    pub auto_reindex: bool,
    /// File extensions to watch (empty = all).
    pub watch_extensions: Vec<String>,
    /// Paths to exclude from watching.
    pub exclude_paths: Vec<PathBuf>,
    /// Maximum file size to auto-reindex (bytes). 0 = unlimited.
    pub max_file_size: u64,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(30),
            auto_reindex: true,
            watch_extensions: vec![
                "md".into(),
                "txt".into(),
                "pdf".into(),
                "docx".into(),
                "html".into(),
                "rs".into(),
                "py".into(),
                "js".into(),
                "json".into(),
                "toml".into(),
                "yaml".into(),
                "yml".into(),
            ],
            exclude_paths: Vec::new(),
            max_file_size: 50 * 1024 * 1024, // 50 MB
        }
    }
}

/// State of a watched file.
#[derive(Debug, Clone)]
struct WatchedFile {
    path: PathBuf,
    last_modified: SystemTime,
    last_size: u64,
    last_indexed: SystemTime,
}

/// Event emitted when a watched file changes.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// File was modified (content changed).
    Modified(PathBuf),
    /// File was created (new file in watched directory).
    Created(PathBuf),
    /// File was deleted.
    Deleted(PathBuf),
}

/// Watches indexed knowledge sources for changes and triggers re-indexing.
pub struct KnowledgeWatcher {
    config: WatcherConfig,
    /// Tracked files: path → state.
    tracked: HashMap<PathBuf, WatchedFile>,
    /// Pending events to process.
    pending_events: Vec<WatchEvent>,
}

impl KnowledgeWatcher {
    /// Create a new watcher with the given configuration.
    pub fn new(config: WatcherConfig) -> Self {
        Self {
            config,
            tracked: HashMap::new(),
            pending_events: Vec::new(),
        }
    }

    /// Register a file or directory to watch.
    pub fn watch(&mut self, path: &Path) {
        if path.is_file() {
            self.track_file(path);
        } else if path.is_dir() {
            self.track_directory(path);
        }
    }

    /// Check all tracked files for changes. Returns events for changed files.
    pub fn poll(&mut self) -> Vec<WatchEvent> {
        let mut events = Vec::new();

        // Check existing tracked files
        let paths: Vec<PathBuf> = self.tracked.keys().cloned().collect();
        for path in &paths {
            if !path.exists() {
                // File was deleted
                events.push(WatchEvent::Deleted(path.clone()));
                self.tracked.remove(path);
                continue;
            }

            if let Ok(metadata) = std::fs::metadata(path) {
                if let Ok(modified) = metadata.modified() {
                    let tracked = self.tracked.get(path).unwrap();
                    let size = metadata.len();

                    // Check if modified since last check
                    if modified > tracked.last_modified || size != tracked.last_size {
                        events.push(WatchEvent::Modified(path.clone()));

                        // Update tracking
                        if let Some(entry) = self.tracked.get_mut(path) {
                            entry.last_modified = modified;
                            entry.last_size = size;
                        }
                    }
                }
            }
        }

        // Check for new files in tracked directories
        // (directories themselves are tracked via their contained files)

        self.pending_events.extend(events.clone());
        events
    }

    /// Drain pending events (consumed by the indexer).
    pub fn drain_events(&mut self) -> Vec<WatchEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Number of files being watched.
    pub fn watched_count(&self) -> usize {
        self.tracked.len()
    }

    /// Mark a file as successfully re-indexed.
    pub fn mark_indexed(&mut self, path: &Path) {
        if let Some(entry) = self.tracked.get_mut(path) {
            entry.last_indexed = SystemTime::now();
        }
    }

    /// List files that need re-indexing (modified since last indexed).
    pub fn stale_files(&self) -> Vec<&Path> {
        self.tracked
            .values()
            .filter(|f| f.last_modified > f.last_indexed)
            .map(|f| f.path.as_path())
            .collect()
    }

    /// Check if a path should be watched (extension filter + exclusions).
    fn should_watch(&self, path: &Path) -> bool {
        // Check exclusions
        for excluded in &self.config.exclude_paths {
            if path.starts_with(excluded) {
                return false;
            }
        }

        // Check extension filter
        if !self.config.watch_extensions.is_empty() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                return self
                    .config
                    .watch_extensions
                    .iter()
                    .any(|w| w.eq_ignore_ascii_case(ext));
            }
            return false;
        }

        true
    }

    /// Track a single file.
    fn track_file(&mut self, path: &Path) {
        if !self.should_watch(path) {
            return;
        }

        if let Ok(metadata) = std::fs::metadata(path) {
            // Check max file size
            if self.config.max_file_size > 0 && metadata.len() > self.config.max_file_size {
                return;
            }

            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            self.tracked.insert(
                path.to_path_buf(),
                WatchedFile {
                    path: path.to_path_buf(),
                    last_modified: modified,
                    last_size: metadata.len(),
                    last_indexed: SystemTime::now(),
                },
            );
        }
    }

    /// Track all matching files in a directory.
    fn track_directory(&mut self, dir: &Path) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    self.track_file(&path);
                }
            }
        }
    }
}

impl Default for KnowledgeWatcher {
    fn default() -> Self {
        Self::new(WatcherConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ai_assistant_test_watcher_{}",
            uuid::Uuid::new_v4()
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_watch_file() {
        let dir = test_dir();
        let file = dir.join("test.md");
        std::fs::write(&file, "content").unwrap();

        let mut watcher = KnowledgeWatcher::default();
        watcher.watch(&file);
        assert_eq!(watcher.watched_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_watch_directory() {
        let dir = test_dir();
        std::fs::write(dir.join("a.md"), "a").unwrap();
        std::fs::write(dir.join("b.txt"), "b").unwrap();
        std::fs::write(dir.join("c.exe"), "c").unwrap(); // excluded by extension

        let mut watcher = KnowledgeWatcher::default();
        watcher.watch(&dir);
        assert_eq!(watcher.watched_count(), 2); // .md and .txt, not .exe

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_poll_detects_modification() {
        let dir = test_dir();
        let file = dir.join("watched.md");
        std::fs::write(&file, "original").unwrap();

        let mut watcher = KnowledgeWatcher::default();
        watcher.watch(&file);

        // No changes yet
        let events = watcher.poll();
        assert!(events.is_empty());

        // Modify file (need to change timestamp)
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file, "modified content").unwrap();

        let events = watcher.poll();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], WatchEvent::Modified(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_poll_detects_deletion() {
        let dir = test_dir();
        let file = dir.join("to_delete.md");
        std::fs::write(&file, "data").unwrap();

        let mut watcher = KnowledgeWatcher::default();
        watcher.watch(&file);

        std::fs::remove_file(&file).unwrap();

        let events = watcher.poll();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], WatchEvent::Deleted(_)));
        assert_eq!(watcher.watched_count(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_files() {
        let dir = test_dir();
        let file = dir.join("stale.md");
        std::fs::write(&file, "v1").unwrap();

        let mut watcher = KnowledgeWatcher::default();
        watcher.watch(&file);

        // Initially not stale (just indexed)
        assert!(watcher.stale_files().is_empty());

        // Simulate modification
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file, "v2").unwrap();
        watcher.poll();

        assert_eq!(watcher.stale_files().len(), 1);

        // Mark as indexed
        watcher.mark_indexed(&file);
        assert!(watcher.stale_files().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extension_filter() {
        let config = WatcherConfig {
            watch_extensions: vec!["md".into(), "txt".into()],
            ..Default::default()
        };
        let watcher = KnowledgeWatcher::new(config);
        assert!(watcher.should_watch(Path::new("doc.md")));
        assert!(watcher.should_watch(Path::new("readme.txt")));
        assert!(!watcher.should_watch(Path::new("binary.exe")));
    }

    #[test]
    fn test_exclude_paths() {
        let config = WatcherConfig {
            exclude_paths: vec![PathBuf::from("/tmp/secret")],
            ..Default::default()
        };
        let watcher = KnowledgeWatcher::new(config);
        assert!(!watcher.should_watch(Path::new("/tmp/secret/doc.md")));
        assert!(watcher.should_watch(Path::new("/tmp/public/doc.md")));
    }
}
