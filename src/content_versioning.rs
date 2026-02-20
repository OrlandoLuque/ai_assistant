//! # Content Versioning
//!
//! This module tracks content changes over time by storing snapshots,
//! computing line-based diffs using a longest common subsequence (LCS) algorithm,
//! and providing change summaries. It supports both an in-memory store and an
//! optional SQLite-backed store behind the `rag` feature flag.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Change Type
// ─────────────────────────────────────────────────────────────────────────────

/// The type of change detected between two content versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// A line was added in the new version.
    Added,
    /// A line was removed from the old version.
    Removed,
    /// A line was modified between versions.
    Modified,
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSnapshot {
    /// Unique identifier for this version.
    pub version_id: String,
    /// Identifier of the content being tracked.
    pub content_id: String,
    /// The full content text at this point in time.
    pub content: String,
    /// When this snapshot was created.
    pub timestamp: DateTime<Utc>,
    /// Hash of the content for quick comparison.
    pub content_hash: u64,
    /// Size of the content in bytes.
    pub size_bytes: usize,
    /// Arbitrary metadata associated with this snapshot.
    pub metadata: HashMap<String, String>,
    /// Optional human-readable label for this version.
    pub label: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Change
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a single change between two content versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChange {
    /// The type of change.
    pub change_type: ChangeType,
    /// The line number in the old version (if applicable).
    pub old_line: Option<usize>,
    /// The line number in the new version (if applicable).
    pub new_line: Option<usize>,
    /// The content of the changed line.
    pub content: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Version Diff
// ─────────────────────────────────────────────────────────────────────────────

/// The computed difference between two content versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// The version ID of the older snapshot.
    pub old_version: String,
    /// The version ID of the newer snapshot.
    pub new_version: String,
    /// List of individual changes.
    pub changes: Vec<ContentChange>,
    /// Number of lines added.
    pub lines_added: usize,
    /// Number of lines removed.
    pub lines_removed: usize,
    /// Number of lines modified.
    pub lines_modified: usize,
    /// Whether the two versions are identical.
    pub identical: bool,
    /// Similarity ratio between 0.0 and 1.0.
    pub similarity: f64,
}

impl VersionDiff {
    /// Returns a human-readable summary of the diff.
    pub fn summary(&self) -> String {
        if self.identical {
            return format!(
                "Versions {} and {} are identical.",
                self.old_version, self.new_version
            );
        }
        format!(
            "Diff {}->{}: +{} added, -{} removed, ~{} modified (similarity: {:.1}%)",
            self.old_version,
            self.new_version,
            self.lines_added,
            self.lines_removed,
            self.lines_modified,
            self.similarity * 100.0
        )
    }

    /// Returns only the additions from the diff.
    pub fn additions(&self) -> Vec<&ContentChange> {
        self.changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Added)
            .collect()
    }

    /// Returns only the removals from the diff.
    pub fn removals(&self) -> Vec<&ContentChange> {
        self.changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Removed)
            .collect()
    }

    /// Produces a unified-diff-style string representation.
    pub fn to_unified_diff(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("--- version {}\n", self.old_version));
        output.push_str(&format!("+++ version {}\n", self.new_version));

        for change in &self.changes {
            match change.change_type {
                ChangeType::Added => {
                    let line = change.new_line.unwrap_or(0);
                    output.push_str(&format!("+{:>4} | {}\n", line, change.content));
                }
                ChangeType::Removed => {
                    let line = change.old_line.unwrap_or(0);
                    output.push_str(&format!("-{:>4} | {}\n", line, change.content));
                }
                ChangeType::Modified => {
                    let old_line = change.old_line.unwrap_or(0);
                    output.push_str(&format!("~{:>4} | {}\n", old_line, change.content));
                }
            }
        }

        output
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Versioning Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the content versioning store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Maximum number of versions to keep per content ID.
    pub max_versions: usize,
    /// Whether to automatically compute diffs when adding versions.
    pub auto_diff: bool,
    /// Minimum change threshold (0.0 - 1.0) to store a new version.
    /// A value of 0.0 means all changes are stored.
    pub change_threshold: f64,
    /// Whether to store the full content in snapshots.
    pub store_content: bool,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            max_versions: 50,
            auto_diff: true,
            change_threshold: 0.0,
            store_content: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Version History
// ─────────────────────────────────────────────────────────────────────────────

/// The full version history for a single content ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// The content ID this history belongs to.
    pub content_id: String,
    /// Ordered list of snapshots (oldest first).
    pub snapshots: Vec<ContentSnapshot>,
}

impl VersionHistory {
    /// Returns the most recent snapshot, if any.
    pub fn latest(&self) -> Option<&ContentSnapshot> {
        self.snapshots.last()
    }

    /// Returns the snapshot with the given version ID.
    pub fn get_version(&self, version_id: &str) -> Option<&ContentSnapshot> {
        self.snapshots.iter().find(|s| s.version_id == version_id)
    }

    /// Returns the number of snapshots stored.
    pub fn version_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Version Store (in-memory)
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory content version store that tracks changes over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentVersionStore {
    /// Configuration for versioning behavior.
    pub config: VersioningConfig,
    /// Map of content ID to its version history.
    histories: HashMap<String, VersionHistory>,
}

impl ContentVersionStore {
    /// Creates a new store with the given configuration.
    pub fn new(config: VersioningConfig) -> Self {
        Self {
            config,
            histories: HashMap::new(),
        }
    }

    /// Adds a new version of the given content.
    ///
    /// Returns the version ID if the content was stored, or `None` if the change
    /// was below the configured threshold.
    pub fn add_version(&mut self, content_id: &str, content: &str) -> Option<String> {
        self.add_version_with_metadata(content_id, content, HashMap::new(), None)
    }

    /// Adds a new version with metadata and an optional label.
    ///
    /// Returns the version ID if stored, or `None` if below threshold.
    pub fn add_version_with_metadata(
        &mut self,
        content_id: &str,
        content: &str,
        metadata: HashMap<String, String>,
        label: Option<String>,
    ) -> Option<String> {
        let new_hash = compute_hash(content);

        // Check threshold: if there is a previous version, compute similarity
        if self.config.change_threshold > 0.0 {
            if let Some(history) = self.histories.get(content_id) {
                if let Some(latest) = history.latest() {
                    let similarity = calculate_similarity(&latest.content, content);
                    // If the content hasn't changed enough, don't store
                    if (1.0 - similarity) < self.config.change_threshold {
                        return None;
                    }
                }
            }
        }

        // Check if content is identical to latest (always skip if hash matches)
        if let Some(history) = self.histories.get(content_id) {
            if let Some(latest) = history.latest() {
                if latest.content_hash == new_hash && latest.content == content {
                    return None;
                }
            }
        }

        let version_id = Uuid::new_v4().to_string();
        let stored_content = if self.config.store_content {
            content.to_string()
        } else {
            String::new()
        };

        let snapshot = ContentSnapshot {
            version_id: version_id.clone(),
            content_id: content_id.to_string(),
            content: stored_content,
            timestamp: Utc::now(),
            content_hash: new_hash,
            size_bytes: content.len(),
            metadata,
            label,
        };

        let history = self
            .histories
            .entry(content_id.to_string())
            .or_insert_with(|| VersionHistory {
                content_id: content_id.to_string(),
                snapshots: Vec::new(),
            });

        history.snapshots.push(snapshot);

        // Trim history if needed
        self.trim_history(content_id);

        Some(version_id)
    }

    /// Computes the diff between two specific versions of a content ID.
    pub fn diff(
        &self,
        content_id: &str,
        old_version: &str,
        new_version: &str,
    ) -> Result<VersionDiff> {
        let history = self
            .histories
            .get(content_id)
            .ok_or_else(|| anyhow::anyhow!("Content ID '{}' not found", content_id))?;

        let old_snapshot = history
            .get_version(old_version)
            .ok_or_else(|| anyhow::anyhow!("Old version '{}' not found", old_version))?;

        let new_snapshot = history
            .get_version(new_version)
            .ok_or_else(|| anyhow::anyhow!("New version '{}' not found", new_version))?;

        let changes = compute_diff(&old_snapshot.content, &new_snapshot.content);
        let similarity = calculate_similarity(&old_snapshot.content, &new_snapshot.content);

        let lines_added = changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Added)
            .count();
        let lines_removed = changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Removed)
            .count();
        let lines_modified = changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Modified)
            .count();
        let identical = changes.is_empty();

        Ok(VersionDiff {
            old_version: old_version.to_string(),
            new_version: new_version.to_string(),
            changes,
            lines_added,
            lines_removed,
            lines_modified,
            identical,
            similarity,
        })
    }

    /// Computes the diff between the last two versions of a content ID.
    pub fn diff_latest(&self, content_id: &str) -> Result<VersionDiff> {
        let history = self
            .histories
            .get(content_id)
            .ok_or_else(|| anyhow::anyhow!("Content ID '{}' not found", content_id))?;

        if history.snapshots.len() < 2 {
            return Err(anyhow::anyhow!(
                "Need at least 2 versions to compute a diff, found {}",
                history.snapshots.len()
            ));
        }

        let n = history.snapshots.len();
        let old_version = &history.snapshots[n - 2].version_id;
        let new_version = &history.snapshots[n - 1].version_id;

        self.diff(content_id, old_version, new_version)
    }

    /// Returns the version history for a content ID, if it exists.
    pub fn history(&self, content_id: &str) -> Option<&VersionHistory> {
        self.histories.get(content_id)
    }

    /// Returns a list of all tracked content IDs.
    pub fn content_ids(&self) -> Vec<&str> {
        self.histories.keys().map(|s| s.as_str()).collect()
    }

    /// Checks if the new content differs from the latest stored version (by hash).
    pub fn has_changed(&self, content_id: &str, new_content: &str) -> bool {
        let new_hash = compute_hash(new_content);
        match self.histories.get(content_id) {
            Some(history) => match history.latest() {
                Some(latest) => latest.content_hash != new_hash,
                None => true,
            },
            None => true,
        }
    }

    /// Removes all version history for a content ID.
    pub fn clear_history(&mut self, content_id: &str) {
        self.histories.remove(content_id);
    }

    /// Exports the entire store as a JSON string.
    pub fn export(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Imports store data from a JSON string, merging with existing data.
    pub fn import(&mut self, json: &str) -> Result<()> {
        let imported: ContentVersionStore = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

        for (content_id, history) in imported.histories {
            self.histories.insert(content_id, history);
        }

        Ok(())
    }

    /// Export to internal binary format (bincode+gzip when feature enabled).
    #[cfg(feature = "binary-storage")]
    pub fn export_bytes(&self) -> Result<Vec<u8>> {
        crate::internal_storage::serialize_internal(self)
    }

    /// Import from internal binary format (auto-detects binary or JSON).
    #[cfg(feature = "binary-storage")]
    pub fn import_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let imported: ContentVersionStore = crate::internal_storage::deserialize_internal(bytes)?;
        for (content_id, history) in imported.histories {
            self.histories.insert(content_id, history);
        }
        Ok(())
    }

    /// Trims the history for a content ID to the configured max_versions.
    fn trim_history(&mut self, content_id: &str) {
        if let Some(history) = self.histories.get_mut(content_id) {
            let max = self.config.max_versions;
            if history.snapshots.len() > max {
                let excess = history.snapshots.len() - max;
                history.snapshots.drain(0..excess);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Computes a hash of the given content using FNV-1a.
fn compute_hash(content: &str) -> u64 {
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in content.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Computes the LCS (Longest Common Subsequence) table for two slices of lines.
fn compute_lcs_table(old_lines: &[&str], new_lines: &[&str]) -> Vec<Vec<usize>> {
    let m = old_lines.len();
    let n = new_lines.len();
    let mut table = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if old_lines[i - 1] == new_lines[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }

    table
}

/// Computes a line-based diff between old and new content using LCS.
fn compute_diff(old: &str, new: &str) -> Vec<ContentChange> {
    if old == new {
        return Vec::new();
    }

    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let table = compute_lcs_table(&old_lines, &new_lines);

    let mut changes = Vec::new();

    // Backtrack through the LCS table to produce the diff
    let mut i = old_lines.len();
    let mut j = new_lines.len();
    let mut result: Vec<ContentChange> = Vec::new();

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
            // Lines match, part of LCS - no change
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || table[i][j - 1] >= table[i - 1][j]) {
            // Line was added in new
            result.push(ContentChange {
                change_type: ChangeType::Added,
                old_line: None,
                new_line: Some(j),
                content: new_lines[j - 1].to_string(),
            });
            j -= 1;
        } else if i > 0 && (j == 0 || table[i - 1][j] > table[i][j - 1]) {
            // Line was removed from old
            result.push(ContentChange {
                change_type: ChangeType::Removed,
                old_line: Some(i),
                new_line: None,
                content: old_lines[i - 1].to_string(),
            });
            i -= 1;
        } else {
            // Should not happen with correct LCS, but handle gracefully
            break;
        }
    }

    // Reverse since we walked backwards
    result.reverse();
    changes.extend(result);

    changes
}

/// Calculates the similarity between two content strings as a ratio.
/// Returns 2 * lcs_length / (old_lines + new_lines).
/// Returns 1.0 if both are empty, 0.0 if one is empty and the other is not.
fn calculate_similarity(old: &str, new: &str) -> f64 {
    if old == new {
        return 1.0;
    }

    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let total = old_lines.len() + new_lines.len();
    if total == 0 {
        return 1.0;
    }

    let table = compute_lcs_table(&old_lines, &new_lines);
    let lcs_length = table[old_lines.len()][new_lines.len()];

    (2.0 * lcs_length as f64) / total as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// SQLite Version Store (behind `rag` feature)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "rag")]
pub mod sqlite_store {
    use super::*;
    use rusqlite::{params, Connection};

    /// SQLite-backed content version store for persistent storage.
    #[derive(Debug)]
    pub struct SqliteVersionStore {
        conn: Connection,
        config: VersioningConfig,
    }

    impl SqliteVersionStore {
        /// Opens (or creates) the SQLite database at the given path and initializes tables.
        pub fn new(db_path: &str, config: VersioningConfig) -> Result<Self> {
            let conn = Connection::open(db_path)
                .map_err(|e| anyhow::anyhow!("Failed to open database: {}", e))?;

            let store = Self { conn, config };
            store.init_tables()?;
            Ok(store)
        }

        /// Creates the schema if it does not already exist.
        fn init_tables(&self) -> Result<()> {
            self.conn
                .execute_batch(
                    "CREATE TABLE IF NOT EXISTS content_versions (
                        id TEXT PRIMARY KEY,
                        content_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        hash INTEGER NOT NULL,
                        size INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        label TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_content_versions_content_id
                        ON content_versions(content_id);
                    CREATE INDEX IF NOT EXISTS idx_content_versions_timestamp
                        ON content_versions(content_id, timestamp);",
                )
                .map_err(|e| anyhow::anyhow!("Failed to initialize tables: {}", e))?;
            Ok(())
        }

        /// Adds a new version. Returns the version ID if stored.
        pub fn add_version(&self, content_id: &str, content: &str) -> Result<Option<String>> {
            self.add_version_with_metadata(content_id, content, HashMap::new(), None)
        }

        /// Adds a new version with metadata and label.
        pub fn add_version_with_metadata(
            &self,
            content_id: &str,
            content: &str,
            metadata: HashMap<String, String>,
            label: Option<String>,
        ) -> Result<Option<String>> {
            let new_hash = compute_hash(content) as i64;

            // Check threshold
            if self.config.change_threshold > 0.0 {
                if let Some(latest) = self.get_latest_content(content_id)? {
                    let similarity = calculate_similarity(&latest, content);
                    if (1.0 - similarity) < self.config.change_threshold {
                        return Ok(None);
                    }
                }
            }

            // Check for identical content
            if let Some(latest_hash) = self.get_latest_hash(content_id)? {
                if latest_hash == new_hash as u64 {
                    if let Some(latest_content) = self.get_latest_content(content_id)? {
                        if latest_content == content {
                            return Ok(None);
                        }
                    }
                }
            }

            let version_id = Uuid::new_v4().to_string();
            let timestamp = Utc::now().to_rfc3339();
            let stored_content = if self.config.store_content {
                content
            } else {
                ""
            };
            let metadata_json = serde_json::to_string(&metadata)?;

            self.conn
                .execute(
                    "INSERT INTO content_versions (id, content_id, content, timestamp, hash, size, metadata_json, label)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        version_id,
                        content_id,
                        stored_content,
                        timestamp,
                        new_hash,
                        content.len() as i64,
                        metadata_json,
                        label,
                    ],
                )
                .map_err(|e| anyhow::anyhow!("Failed to insert version: {}", e))?;

            // Trim old versions
            self.trim_history(content_id)?;

            Ok(Some(version_id))
        }

        /// Computes the diff between two specific versions.
        pub fn diff(
            &self,
            content_id: &str,
            old_version: &str,
            new_version: &str,
        ) -> Result<VersionDiff> {
            let old_content = self.get_version_content(content_id, old_version)?;
            let new_content = self.get_version_content(content_id, new_version)?;

            let changes = compute_diff(&old_content, &new_content);
            let similarity = calculate_similarity(&old_content, &new_content);

            let lines_added = changes
                .iter()
                .filter(|c| c.change_type == ChangeType::Added)
                .count();
            let lines_removed = changes
                .iter()
                .filter(|c| c.change_type == ChangeType::Removed)
                .count();
            let lines_modified = changes
                .iter()
                .filter(|c| c.change_type == ChangeType::Modified)
                .count();
            let identical = changes.is_empty();

            Ok(VersionDiff {
                old_version: old_version.to_string(),
                new_version: new_version.to_string(),
                changes,
                lines_added,
                lines_removed,
                lines_modified,
                identical,
                similarity,
            })
        }

        /// Computes the diff between the last two versions.
        pub fn diff_latest(&self, content_id: &str) -> Result<VersionDiff> {
            let versions = self.get_version_ids(content_id)?;
            if versions.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Need at least 2 versions to compute a diff, found {}",
                    versions.len()
                ));
            }
            let n = versions.len();
            self.diff(content_id, &versions[n - 2], &versions[n - 1])
        }

        /// Returns the version history for a content ID.
        pub fn history(&self, content_id: &str) -> Result<Option<VersionHistory>> {
            let mut stmt = self.conn.prepare(
                "SELECT id, content_id, content, timestamp, hash, size, metadata_json, label
                 FROM content_versions
                 WHERE content_id = ?1
                 ORDER BY timestamp ASC",
            )?;

            let snapshots: Vec<ContentSnapshot> = stmt
                .query_map(params![content_id], |row| {
                    let id: String = row.get(0)?;
                    let cid: String = row.get(1)?;
                    let content: String = row.get(2)?;
                    let ts_str: String = row.get(3)?;
                    let hash: i64 = row.get(4)?;
                    let size: i64 = row.get(5)?;
                    let meta_json: String = row.get(6)?;
                    let label: Option<String> = row.get(7)?;
                    Ok((id, cid, content, ts_str, hash, size, meta_json, label))
                })?
                .filter_map(|r| r.ok())
                .map(|(id, cid, content, ts_str, hash, size, meta_json, label)| {
                    let timestamp = chrono::DateTime::parse_from_rfc3339(&ts_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());
                    let metadata: HashMap<String, String> =
                        serde_json::from_str(&meta_json).unwrap_or_default();
                    ContentSnapshot {
                        version_id: id,
                        content_id: cid,
                        content,
                        timestamp,
                        content_hash: hash as u64,
                        size_bytes: size as usize,
                        metadata,
                        label,
                    }
                })
                .collect();

            if snapshots.is_empty() {
                Ok(None)
            } else {
                Ok(Some(VersionHistory {
                    content_id: content_id.to_string(),
                    snapshots,
                }))
            }
        }

        /// Returns all tracked content IDs.
        pub fn content_ids(&self) -> Result<Vec<String>> {
            let mut stmt = self
                .conn
                .prepare("SELECT DISTINCT content_id FROM content_versions")?;
            let ids: Vec<String> = stmt
                .query_map([], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(ids)
        }

        /// Checks if new content differs from the latest stored version.
        pub fn has_changed(&self, content_id: &str, new_content: &str) -> Result<bool> {
            let new_hash = compute_hash(new_content);
            match self.get_latest_hash(content_id)? {
                Some(latest_hash) => Ok(latest_hash != new_hash),
                None => Ok(true),
            }
        }

        /// Clears all version history for a content ID.
        pub fn clear_history(&self, content_id: &str) -> Result<()> {
            self.conn.execute(
                "DELETE FROM content_versions WHERE content_id = ?1",
                params![content_id],
            )?;
            Ok(())
        }

        /// Exports the store as a JSON string.
        pub fn export(&self) -> Result<String> {
            let ids = self.content_ids()?;
            let mut histories = HashMap::new();
            for id in &ids {
                if let Some(history) = self.history(id)? {
                    histories.insert(id.clone(), history);
                }
            }
            let data = serde_json::to_string_pretty(&histories)?;
            Ok(data)
        }

        /// Imports version data from a JSON string.
        pub fn import(&self, json: &str) -> Result<()> {
            let histories: HashMap<String, VersionHistory> = serde_json::from_str(json)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

            for (_content_id, history) in histories {
                for snapshot in &history.snapshots {
                    let timestamp = snapshot.timestamp.to_rfc3339();
                    let metadata_json = serde_json::to_string(&snapshot.metadata)?;
                    self.conn.execute(
                        "INSERT OR IGNORE INTO content_versions (id, content_id, content, timestamp, hash, size, metadata_json, label)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                        params![
                            snapshot.version_id,
                            snapshot.content_id,
                            snapshot.content,
                            timestamp,
                            snapshot.content_hash as i64,
                            snapshot.size_bytes as i64,
                            metadata_json,
                            snapshot.label,
                        ],
                    )?;
                }
            }

            Ok(())
        }

        // ─── Private helpers ─────────────────────────────────────────────

        fn get_latest_hash(&self, content_id: &str) -> Result<Option<u64>> {
            let result: Option<i64> = self
                .conn
                .query_row(
                    "SELECT hash FROM content_versions WHERE content_id = ?1 ORDER BY timestamp DESC LIMIT 1",
                    params![content_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(result.map(|h| h as u64))
        }

        fn get_latest_content(&self, content_id: &str) -> Result<Option<String>> {
            let result: Option<String> = self
                .conn
                .query_row(
                    "SELECT content FROM content_versions WHERE content_id = ?1 ORDER BY timestamp DESC LIMIT 1",
                    params![content_id],
                    |row| row.get(0),
                )
                .ok();
            Ok(result)
        }

        fn get_version_content(&self, content_id: &str, version_id: &str) -> Result<String> {
            self.conn
                .query_row(
                    "SELECT content FROM content_versions WHERE content_id = ?1 AND id = ?2",
                    params![content_id, version_id],
                    |row| row.get(0),
                )
                .map_err(|e| anyhow::anyhow!("Version '{}' not found: {}", version_id, e))
        }

        fn get_version_ids(&self, content_id: &str) -> Result<Vec<String>> {
            let mut stmt = self.conn.prepare(
                "SELECT id FROM content_versions WHERE content_id = ?1 ORDER BY timestamp ASC",
            )?;
            let ids: Vec<String> = stmt
                .query_map(params![content_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(ids)
        }

        fn trim_history(&self, content_id: &str) -> Result<()> {
            let max = self.config.max_versions as i64;
            // Count existing versions
            let count: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM content_versions WHERE content_id = ?1",
                params![content_id],
                |row| row.get(0),
            )?;

            if count > max {
                let excess = count - max;
                self.conn.execute(
                    "DELETE FROM content_versions WHERE id IN (
                        SELECT id FROM content_versions
                        WHERE content_id = ?1
                        ORDER BY timestamp ASC
                        LIMIT ?2
                    )",
                    params![content_id, excess],
                )?;
            }
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_version_and_history() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        let v1 = store.add_version("doc1", "Hello, world!");
        assert!(v1.is_some());

        let v2 = store.add_version("doc1", "Hello, Rust!");
        assert!(v2.is_some());

        let history = store.history("doc1").unwrap();
        assert_eq!(history.version_count(), 2);
        assert_eq!(history.latest().unwrap().content, "Hello, Rust!");
    }

    #[test]
    fn test_identical_content_not_stored() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        let v1 = store.add_version("doc1", "Same content");
        assert!(v1.is_some());

        // Adding the exact same content should return None
        let v2 = store.add_version("doc1", "Same content");
        assert!(v2.is_none());

        let history = store.history("doc1").unwrap();
        assert_eq!(history.version_count(), 1);
    }

    #[test]
    fn test_diff_computation() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        let old_content = "line1\nline2\nline3\n";
        let new_content = "line1\nmodified\nline3\nline4\n";

        let v1 = store.add_version("doc1", old_content).unwrap();
        let v2 = store.add_version("doc1", new_content).unwrap();

        let diff = store.diff("doc1", &v1, &v2).unwrap();
        assert!(!diff.identical);
        assert!(diff.lines_added > 0 || diff.lines_removed > 0);
        assert!(diff.similarity > 0.0 && diff.similarity < 1.0);

        // Verify the unified diff output is non-empty
        let unified = diff.to_unified_diff();
        assert!(!unified.is_empty());
        assert!(unified.contains("---"));
        assert!(unified.contains("+++"));
    }

    #[test]
    fn test_lcs_diff_correctness() {
        // Test the LCS diff algorithm directly
        let old = "A\nB\nC\nD\nE";
        let new = "A\nX\nC\nD\nF";

        let changes = compute_diff(old, new);

        // B should be removed, X should be added
        let removed: Vec<&ContentChange> = changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Removed)
            .collect();
        let added: Vec<&ContentChange> = changes
            .iter()
            .filter(|c| c.change_type == ChangeType::Added)
            .collect();

        assert!(removed.iter().any(|c| c.content == "B"));
        assert!(removed.iter().any(|c| c.content == "E"));
        assert!(added.iter().any(|c| c.content == "X"));
        assert!(added.iter().any(|c| c.content == "F"));

        // A, C, D should not appear in changes (they are common)
        let all_change_contents: Vec<&str> = changes.iter().map(|c| c.content.as_str()).collect();
        assert!(!all_change_contents.contains(&"A"));
        assert!(!all_change_contents.contains(&"C"));
        assert!(!all_change_contents.contains(&"D"));
    }

    #[test]
    fn test_similarity_calculation() {
        // Identical content
        assert_eq!(calculate_similarity("hello\nworld", "hello\nworld"), 1.0);

        // Completely different
        let sim = calculate_similarity("A\nB\nC", "X\nY\nZ");
        assert_eq!(sim, 0.0);

        // Partially similar
        let sim = calculate_similarity("A\nB\nC\nD", "A\nB\nX\nD");
        // LCS is A, B, D (length 3), total lines = 8, similarity = 6/8 = 0.75
        assert!((sim - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_has_changed() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        // Non-existent content is considered changed
        assert!(store.has_changed("doc1", "anything"));

        store.add_version("doc1", "original");

        assert!(!store.has_changed("doc1", "original"));
        assert!(store.has_changed("doc1", "modified"));
    }

    #[test]
    fn test_trim_history() {
        let config = VersioningConfig {
            max_versions: 3,
            ..Default::default()
        };
        let mut store = ContentVersionStore::new(config);

        store.add_version("doc1", "v1");
        store.add_version("doc1", "v2");
        store.add_version("doc1", "v3");
        store.add_version("doc1", "v4");
        store.add_version("doc1", "v5");

        let history = store.history("doc1").unwrap();
        assert_eq!(history.version_count(), 3);
        // The latest should be v5
        assert_eq!(history.latest().unwrap().content, "v5");
    }

    #[test]
    fn test_export_import() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config.clone());

        store.add_version("doc1", "content A");
        store.add_version("doc1", "content B");
        store.add_version("doc2", "other content");

        let json = store.export();
        assert!(!json.is_empty());

        // Import into a new store
        let mut store2 = ContentVersionStore::new(config);
        store2.import(&json).unwrap();

        assert_eq!(store2.history("doc1").unwrap().version_count(), 2);
        assert_eq!(store2.history("doc2").unwrap().version_count(), 1);
    }

    #[test]
    fn test_diff_latest() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        store.add_version("doc1", "first version\nline2");
        store.add_version("doc1", "first version\nline2\nline3");

        let diff = store.diff_latest("doc1").unwrap();
        assert!(!diff.identical);
        assert_eq!(diff.lines_added, 1);
        assert_eq!(diff.lines_removed, 0);
    }

    #[test]
    fn test_change_threshold() {
        let config = VersioningConfig {
            change_threshold: 0.5, // Require at least 50% change
            ..Default::default()
        };
        let mut store = ContentVersionStore::new(config);

        store.add_version("doc1", "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ");

        // Small change (only 1 line out of 10 differs) - should be below threshold
        let v2 = store.add_version("doc1", "A\nB\nC\nD\nE\nF\nG\nH\nI\nX");
        assert!(v2.is_none());

        // Large change (most lines differ) - should exceed threshold
        let v3 = store.add_version("doc1", "X\nY\nZ\nW\nV\nU\nT\nS\nR\nQ");
        assert!(v3.is_some());
    }

    #[test]
    fn test_version_diff_summary() {
        let diff = VersionDiff {
            old_version: "v1".to_string(),
            new_version: "v2".to_string(),
            changes: vec![
                ContentChange {
                    change_type: ChangeType::Added,
                    old_line: None,
                    new_line: Some(3),
                    content: "new line".to_string(),
                },
                ContentChange {
                    change_type: ChangeType::Removed,
                    old_line: Some(2),
                    new_line: None,
                    content: "old line".to_string(),
                },
            ],
            lines_added: 1,
            lines_removed: 1,
            lines_modified: 0,
            identical: false,
            similarity: 0.75,
        };

        let summary = diff.summary();
        assert!(summary.contains("+1 added"));
        assert!(summary.contains("-1 removed"));
        assert!(summary.contains("75.0%"));

        assert_eq!(diff.additions().len(), 1);
        assert_eq!(diff.removals().len(), 1);
    }

    #[test]
    fn test_clear_history() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        store.add_version("doc1", "content");
        assert!(store.history("doc1").is_some());

        store.clear_history("doc1");
        assert!(store.history("doc1").is_none());
    }

    #[test]
    fn test_content_ids() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        store.add_version("alpha", "a");
        store.add_version("beta", "b");
        store.add_version("gamma", "c");

        let mut ids = store.content_ids();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_add_version_with_metadata() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "test_user".to_string());
        meta.insert("source".to_string(), "unit_test".to_string());

        let vid = store
            .add_version_with_metadata("doc1", "content here", meta, Some("initial".to_string()))
            .unwrap();

        let history = store.history("doc1").unwrap();
        let snapshot = history.get_version(&vid).unwrap();
        assert_eq!(snapshot.metadata.get("author").unwrap(), "test_user");
        assert_eq!(snapshot.label, Some("initial".to_string()));
    }

    #[test]
    fn test_compute_hash_consistency() {
        let h1 = compute_hash("hello world");
        let h2 = compute_hash("hello world");
        let h3 = compute_hash("hello world!");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_empty_diff() {
        let changes = compute_diff("same\ncontent", "same\ncontent");
        assert!(changes.is_empty());
    }

    #[test]
    fn test_diff_error_cases() {
        let config = VersioningConfig::default();
        let store = ContentVersionStore::new(config);

        // Non-existent content ID
        let result = store.diff("nonexistent", "v1", "v2");
        assert!(result.is_err());

        // Non-existent version ID
        let mut store2 = ContentVersionStore::new(VersioningConfig::default());
        store2.add_version("doc1", "content");
        let result = store2.diff("doc1", "nonexistent_v1", "nonexistent_v2");
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_latest_insufficient_versions() {
        let config = VersioningConfig::default();
        let mut store = ContentVersionStore::new(config);

        store.add_version("doc1", "only one version");

        let result = store.diff_latest("doc1");
        assert!(result.is_err());
    }
}
