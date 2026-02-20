//! Persistence utilities: backup, compaction, session migration

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// Database Backup
// ============================================================================

/// Configuration for automatic backups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup directory path
    pub backup_dir: PathBuf,
    /// Maximum number of backups to keep
    pub max_backups: usize,
    /// Backup interval in hours
    pub interval_hours: u32,
    /// Compress backups
    pub compress: bool,
    /// Include in filename timestamp format
    pub timestamp_format: String,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backup_dir: PathBuf::from("./backups"),
            max_backups: 10,
            interval_hours: 24,
            compress: true,
            timestamp_format: "%Y%m%d_%H%M%S".to_string(),
        }
    }
}

/// Information about a backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// Backup filename
    pub filename: String,
    /// Full path to backup
    pub path: PathBuf,
    /// When the backup was created
    pub created_at: DateTime<Utc>,
    /// Size in bytes
    pub size_bytes: u64,
    /// Is compressed
    pub compressed: bool,
    /// Original database path
    pub source_db: String,
}

/// Database backup manager
pub struct BackupManager {
    config: BackupConfig,
    last_backup: Option<DateTime<Utc>>,
}

impl BackupManager {
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config,
            last_backup: None,
        }
    }

    /// Check if backup is needed based on interval
    pub fn should_backup(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        match self.last_backup {
            None => true,
            Some(last) => {
                let elapsed = Utc::now() - last;
                elapsed.num_hours() >= self.config.interval_hours as i64
            }
        }
    }

    /// Create a backup of the database
    pub fn create_backup(&mut self, db_path: &Path) -> Result<BackupInfo> {
        // Ensure backup directory exists
        fs::create_dir_all(&self.config.backup_dir).context("Failed to create backup directory")?;

        let timestamp = Utc::now();
        let timestamp_str = timestamp.format(&self.config.timestamp_format).to_string();

        let db_name = db_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("database");

        let extension = if self.config.compress { "db.gz" } else { "db" };
        let backup_filename = format!("{}_{}.{}", db_name, timestamp_str, extension);
        let backup_path = self.config.backup_dir.join(&backup_filename);

        // Read source database
        let mut source = fs::File::open(db_path).context("Failed to open source database")?;
        let mut data = Vec::new();
        source
            .read_to_end(&mut data)
            .context("Failed to read source database")?;

        // Write backup (with optional compression)
        let size_bytes = if self.config.compress {
            let compressed = Self::compress_data(&data)?;
            let mut backup_file =
                fs::File::create(&backup_path).context("Failed to create backup file")?;
            backup_file
                .write_all(&compressed)
                .context("Failed to write compressed backup")?;
            compressed.len() as u64
        } else {
            fs::copy(db_path, &backup_path).context("Failed to copy database")?
        };

        self.last_backup = Some(timestamp);

        // Clean old backups
        self.cleanup_old_backups()?;

        Ok(BackupInfo {
            filename: backup_filename,
            path: backup_path,
            created_at: timestamp,
            size_bytes,
            compressed: self.config.compress,
            source_db: db_path.to_string_lossy().to_string(),
        })
    }

    /// Restore database from backup
    pub fn restore_backup(&self, backup_path: &Path, target_path: &Path) -> Result<()> {
        let mut backup_file = fs::File::open(backup_path).context("Failed to open backup file")?;
        let mut data = Vec::new();
        backup_file
            .read_to_end(&mut data)
            .context("Failed to read backup file")?;

        // Decompress if needed
        let decompressed = if backup_path.extension().map(|e| e == "gz").unwrap_or(false) {
            Self::decompress_data(&data)?
        } else {
            data
        };

        // Write to target
        let mut target_file =
            fs::File::create(target_path).context("Failed to create target database")?;
        target_file
            .write_all(&decompressed)
            .context("Failed to write restored database")?;

        Ok(())
    }

    /// List available backups
    pub fn list_backups(&self) -> Result<Vec<BackupInfo>> {
        if !self.config.backup_dir.exists() {
            return Ok(Vec::new());
        }

        let mut backups = Vec::new();

        for entry in fs::read_dir(&self.config.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let is_backup = ext == "db" || ext == "gz";
                    if is_backup {
                        let metadata = fs::metadata(&path)?;
                        let filename = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();

                        // Try to parse timestamp from filename
                        let created_at =
                            Self::parse_backup_timestamp(&filename).unwrap_or_else(|| {
                                metadata
                                    .modified()
                                    .map(|t| DateTime::from(t))
                                    .unwrap_or_else(|_| Utc::now())
                            });

                        backups.push(BackupInfo {
                            filename,
                            path: path.clone(),
                            created_at,
                            size_bytes: metadata.len(),
                            compressed: ext == "gz",
                            source_db: String::new(),
                        });
                    }
                }
            }
        }

        // Sort by creation time (newest first)
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(backups)
    }

    /// Delete a backup
    pub fn delete_backup(&self, backup_path: &Path) -> Result<()> {
        fs::remove_file(backup_path).context("Failed to delete backup")?;
        Ok(())
    }

    /// Clean up old backups beyond max_backups limit
    fn cleanup_old_backups(&self) -> Result<()> {
        let mut backups = self.list_backups()?;

        if backups.len() > self.config.max_backups {
            // Remove oldest backups
            while backups.len() > self.config.max_backups {
                if let Some(oldest) = backups.pop() {
                    let _ = fs::remove_file(&oldest.path);
                }
            }
        }

        Ok(())
    }

    fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    fn parse_backup_timestamp(filename: &str) -> Option<DateTime<Utc>> {
        // Try to extract timestamp from filename like "database_20240115_120000.db.gz"
        let parts: Vec<&str> = filename.split('_').collect();
        if parts.len() >= 3 {
            let date_part = parts[parts.len() - 2];
            let time_part = parts[parts.len() - 1].split('.').next()?;

            let datetime_str = format!("{}_{}", date_part, time_part);
            chrono::NaiveDateTime::parse_from_str(&datetime_str, "%Y%m%d_%H%M%S")
                .ok()
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
        } else {
            None
        }
    }

    /// Get total backup size
    pub fn total_backup_size(&self) -> Result<u64> {
        let backups = self.list_backups()?;
        Ok(backups.iter().map(|b| b.size_bytes).sum())
    }
}

// ============================================================================
// Database Compaction
// ============================================================================

/// Configuration for database compaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Enable automatic compaction
    pub enabled: bool,
    /// Trigger compaction when fragmentation exceeds this percentage
    pub fragmentation_threshold: f32,
    /// Minimum time between compactions (hours)
    pub min_interval_hours: u32,
    /// Vacuum after compaction
    pub vacuum_after: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fragmentation_threshold: 25.0,
            min_interval_hours: 24,
            vacuum_after: true,
        }
    }
}

/// Result of compaction
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub size_before: u64,
    pub size_after: u64,
    pub space_saved: u64,
    pub fragmentation_before: f32,
    pub duration_ms: u64,
}

/// Database compactor for SQLite
pub struct DatabaseCompactor {
    config: CompactionConfig,
    last_compaction: Option<DateTime<Utc>>,
}

impl DatabaseCompactor {
    pub fn new(config: CompactionConfig) -> Self {
        Self {
            config,
            last_compaction: None,
        }
    }

    /// Check if compaction is needed
    pub fn should_compact(&self, fragmentation: f32) -> bool {
        if !self.config.enabled {
            return false;
        }

        if fragmentation < self.config.fragmentation_threshold {
            return false;
        }

        match self.last_compaction {
            None => true,
            Some(last) => {
                let elapsed = Utc::now() - last;
                elapsed.num_hours() >= self.config.min_interval_hours as i64
            }
        }
    }

    /// Compact a SQLite database (requires rag feature)
    #[cfg(feature = "rag")]
    pub fn compact(&mut self, db_path: &Path) -> Result<CompactionResult> {
        let start = std::time::Instant::now();

        // Get size before
        let size_before = fs::metadata(db_path)?.len();

        // Estimate fragmentation (simplified - actual would query sqlite)
        let fragmentation_before = self.estimate_fragmentation(db_path)?;

        // Open connection and run VACUUM
        let conn = rusqlite::Connection::open(db_path)?;

        // Run VACUUM
        conn.execute("VACUUM", [])?;

        // Run ANALYZE for query optimization
        conn.execute("ANALYZE", [])?;

        // Optionally optimize FTS tables
        let _ = conn.execute(
            "INSERT INTO knowledge_fts(knowledge_fts) VALUES('optimize')",
            [],
        );

        conn.close().map_err(|(_, e)| e)?;

        // Get size after
        let size_after = fs::metadata(db_path)?.len();

        self.last_compaction = Some(Utc::now());

        Ok(CompactionResult {
            size_before,
            size_after,
            space_saved: size_before.saturating_sub(size_after),
            fragmentation_before,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Estimate database fragmentation (requires rag feature)
    #[cfg(feature = "rag")]
    fn estimate_fragmentation(&self, db_path: &Path) -> Result<f32> {
        let conn = rusqlite::Connection::open(db_path)?;

        // Get page count and freelist count
        let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let freelist_count: i64 = conn.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;

        conn.close().map_err(|(_, e)| e)?;

        if page_count == 0 {
            return Ok(0.0);
        }

        Ok((freelist_count as f32 / page_count as f32) * 100.0)
    }

    /// Get database statistics (requires rag feature)
    #[cfg(feature = "rag")]
    pub fn get_stats(&self, db_path: &Path) -> Result<DatabaseStats> {
        let conn = rusqlite::Connection::open(db_path)?;

        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let freelist_count: i64 = conn.query_row("PRAGMA freelist_count", [], |r| r.get(0))?;

        conn.close().map_err(|(_, e)| e)?;

        let total_size = page_size as u64 * page_count as u64;
        let used_size = page_size as u64 * (page_count - freelist_count) as u64;
        let fragmentation = if page_count > 0 {
            (freelist_count as f32 / page_count as f32) * 100.0
        } else {
            0.0
        };

        Ok(DatabaseStats {
            total_size,
            used_size,
            free_size: total_size - used_size,
            page_size: page_size as u64,
            page_count: page_count as u64,
            freelist_count: freelist_count as u64,
            fragmentation,
        })
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub total_size: u64,
    pub used_size: u64,
    pub free_size: u64,
    pub page_size: u64,
    pub page_count: u64,
    pub freelist_count: u64,
    pub fragmentation: f32,
}

// ============================================================================
// Session Migration to RAG
// ============================================================================

/// Configuration for session migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Minimum messages to migrate a session
    pub min_messages: usize,
    /// Maximum age for sessions to migrate (days)
    pub max_age_days: u32,
    /// Delete original sessions after migration
    pub delete_after_migration: bool,
    /// Index conversations for RAG search
    pub index_for_search: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            min_messages: 4,
            max_age_days: 365,
            delete_after_migration: false,
            index_for_search: true,
        }
    }
}

/// Result of migration
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub sessions_migrated: usize,
    pub messages_migrated: usize,
    pub sessions_skipped: usize,
    pub errors: Vec<String>,
}

/// Session migrator
pub struct SessionMigrator {
    config: MigrationConfig,
}

impl SessionMigrator {
    pub fn new(config: MigrationConfig) -> Self {
        Self { config }
    }

    /// Migrate sessions from JSON store to RAG database
    #[cfg(feature = "rag")]
    pub fn migrate_sessions(
        &self,
        sessions: &[crate::session::ChatSession],
        rag_db: &crate::rag::RagDb,
        user_id: &str,
    ) -> MigrationResult {
        let mut result = MigrationResult {
            sessions_migrated: 0,
            messages_migrated: 0,
            sessions_skipped: 0,
            errors: Vec::new(),
        };

        let now = Utc::now();

        for session in sessions {
            // Check minimum messages
            if session.messages.len() < self.config.min_messages {
                result.sessions_skipped += 1;
                continue;
            }

            // Check age
            let age_days = (now - session.updated_at).num_days();
            if age_days > self.config.max_age_days as i64 {
                result.sessions_skipped += 1;
                continue;
            }

            // Migrate messages
            for msg in &session.messages {
                match rag_db.store_message(user_id, &session.id, msg, false) {
                    Ok(_) => {
                        result.messages_migrated += 1;
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "Failed to migrate message in session {}: {}",
                            session.id, e
                        ));
                    }
                }
            }

            result.sessions_migrated += 1;
        }

        result
    }

    /// Check which sessions are eligible for migration
    pub fn get_eligible_sessions<'a>(
        &self,
        sessions: &'a [crate::session::ChatSession],
    ) -> Vec<&'a crate::session::ChatSession> {
        let now = Utc::now();

        sessions
            .iter()
            .filter(|s| {
                s.messages.len() >= self.config.min_messages
                    && (now - s.updated_at).num_days() <= self.config.max_age_days as i64
            })
            .collect()
    }
}

// ============================================================================
// Data Export/Import
// ============================================================================

/// Complete export of all assistant data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullExport {
    pub version: String,
    pub exported_at: DateTime<Utc>,
    pub sessions: Vec<crate::session::ChatSession>,
    pub preferences: crate::session::UserPreferences,
    pub config: crate::config::AiConfig,
    #[cfg(feature = "rag")]
    pub knowledge: Option<crate::rag::KnowledgeExport>,
}

impl FullExport {
    pub fn new(
        sessions: Vec<crate::session::ChatSession>,
        preferences: crate::session::UserPreferences,
        config: crate::config::AiConfig,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            exported_at: Utc::now(),
            sessions,
            preferences,
            config,
            #[cfg(feature = "rag")]
            knowledge: None,
        }
    }

    #[cfg(feature = "rag")]
    pub fn with_knowledge(mut self, knowledge: crate::rag::KnowledgeExport) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    /// Export to JSON file
    pub fn to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Import from JSON file
    pub fn from_file(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let export: Self = serde_json::from_str(&json)?;
        Ok(export)
    }

    /// Export to compressed file
    pub fn to_compressed_file(&self, path: &Path) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let json = serde_json::to_string(self)?;
        let file = fs::File::create(path)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(json.as_bytes())?;
        encoder.finish()?;
        Ok(())
    }

    /// Import from compressed file
    pub fn from_compressed_file(path: &Path) -> Result<Self> {
        use flate2::read::GzDecoder;

        let file = fs::File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut json = String::new();
        decoder.read_to_string(&mut json)?;
        let export: Self = serde_json::from_str(&json)?;
        Ok(export)
    }
}

// ============================================================================
// Persistent Cache
// ============================================================================

/// Configuration for persistent cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCacheConfig {
    /// Maximum entries to keep in cache
    pub max_entries: usize,
    /// Default TTL in seconds
    pub default_ttl_seconds: u64,
    /// Enable compression for large values
    pub compress_large_values: bool,
    /// Threshold for compression (bytes)
    pub compression_threshold: usize,
    /// Auto-cleanup expired entries
    pub auto_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

impl Default for PersistentCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            default_ttl_seconds: 3600, // 1 hour
            compress_large_values: true,
            compression_threshold: 1024, // 1KB
            auto_cleanup: true,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

/// A cached entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key
    pub key: String,
    /// Cached value (JSON serialized)
    pub value: String,
    /// Is the value compressed
    pub compressed: bool,
    /// When the entry was created
    pub created_at: DateTime<Utc>,
    /// When the entry expires
    pub expires_at: DateTime<Utc>,
    /// Number of times this entry was accessed
    pub access_count: u64,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
}

/// Persistent cache stored in SQLite
#[cfg(feature = "rag")]
pub struct PersistentCache {
    conn: rusqlite::Connection,
    config: PersistentCacheConfig,
    last_cleanup: Option<DateTime<Utc>>,
}

#[cfg(feature = "rag")]
impl PersistentCache {
    /// Open or create a persistent cache
    pub fn open(db_path: &Path, config: PersistentCacheConfig) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = rusqlite::Connection::open(db_path)?;

        let cache = Self {
            conn,
            config,
            last_cleanup: None,
        };

        cache.init_tables()?;
        Ok(cache)
    }

    fn init_tables(&self) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                compressed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL
            )",
            [],
        )?;

        // Index for expiration queries
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
            [],
        )?;

        Ok(())
    }

    /// Get a value from cache
    pub fn get<T: serde::de::DeserializeOwned>(&mut self, key: &str) -> Result<Option<T>> {
        // Auto cleanup if needed
        if self.config.auto_cleanup {
            self.maybe_cleanup()?;
        }

        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Get entry and check expiration
        let result: Option<(String, bool, String)> = self
            .conn
            .query_row(
                "SELECT value, compressed, expires_at FROM cache_entries WHERE key = ?1",
                [key],
                |row| Ok((row.get(0)?, row.get::<_, i32>(1)? != 0, row.get(2)?)),
            )
            .ok();

        let (value, compressed, expires_at) = match result {
            Some(v) => v,
            None => return Ok(None),
        };

        // Check expiration
        if let Ok(exp) = chrono::DateTime::parse_from_rfc3339(&expires_at) {
            if exp.with_timezone(&Utc) < now {
                // Entry expired, delete it
                let _ = self
                    .conn
                    .execute("DELETE FROM cache_entries WHERE key = ?1", [key]);
                return Ok(None);
            }
        }

        // Update access count and last_accessed
        let _ = self.conn.execute(
            "UPDATE cache_entries SET access_count = access_count + 1, last_accessed = ?1 WHERE key = ?2",
            rusqlite::params![now_str, key],
        );

        // Decompress if needed
        let json = if compressed {
            let decoded = base64_decode(&value)?;
            let decompressed = BackupManager::decompress_data(&decoded)?;
            String::from_utf8(decompressed)?
        } else {
            value
        };

        let parsed: T = serde_json::from_str(&json)?;
        Ok(Some(parsed))
    }

    /// Set a value in cache with default TTL
    pub fn set<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        self.set_with_ttl(key, value, self.config.default_ttl_seconds)
    }

    /// Set a value in cache with custom TTL
    pub fn set_with_ttl<T: Serialize>(
        &mut self,
        key: &str,
        value: &T,
        ttl_seconds: u64,
    ) -> Result<()> {
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(ttl_seconds as i64);

        let json = serde_json::to_string(value)?;

        // Compress if large
        let (stored_value, compressed) = if self.config.compress_large_values
            && json.len() > self.config.compression_threshold
        {
            let compressed_data = BackupManager::compress_data(json.as_bytes())?;
            (base64_encode(&compressed_data), true)
        } else {
            (json, false)
        };

        self.conn.execute(
            "INSERT INTO cache_entries (key, value, compressed, created_at, expires_at, access_count, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?4)
             ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                compressed = excluded.compressed,
                expires_at = excluded.expires_at,
                access_count = cache_entries.access_count + 1,
                last_accessed = excluded.created_at",
            rusqlite::params![key, stored_value, compressed as i32, now.to_rfc3339(), expires_at.to_rfc3339()],
        )?;

        // Enforce max entries
        self.enforce_max_entries()?;

        Ok(())
    }

    /// Delete a value from cache
    pub fn delete(&mut self, key: &str) -> Result<bool> {
        let deleted = self
            .conn
            .execute("DELETE FROM cache_entries WHERE key = ?1", [key])?;
        Ok(deleted > 0)
    }

    /// Check if a key exists in cache
    pub fn exists(&self, key: &str) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM cache_entries WHERE key = ?1 AND expires_at > ?2",
            rusqlite::params![key, now],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Clear all cache entries
    pub fn clear(&mut self) -> Result<usize> {
        let deleted = self.conn.execute("DELETE FROM cache_entries", [])?;
        Ok(deleted)
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) -> Result<usize> {
        let now = Utc::now().to_rfc3339();
        let deleted = self
            .conn
            .execute("DELETE FROM cache_entries WHERE expires_at < ?1", [&now])?;
        self.last_cleanup = Some(Utc::now());
        Ok(deleted)
    }

    fn maybe_cleanup(&mut self) -> Result<()> {
        let should_cleanup = match self.last_cleanup {
            None => true,
            Some(last) => {
                let elapsed = (Utc::now() - last).num_seconds();
                elapsed as u64 >= self.config.cleanup_interval_seconds
            }
        };

        if should_cleanup {
            let _ = self.cleanup_expired();
        }

        Ok(())
    }

    fn enforce_max_entries(&mut self) -> Result<()> {
        let count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM cache_entries", [], |row| row.get(0))?;

        if count > self.config.max_entries {
            // Delete oldest entries based on last_accessed
            let to_delete = count - self.config.max_entries;
            self.conn.execute(
                "DELETE FROM cache_entries WHERE key IN (
                    SELECT key FROM cache_entries ORDER BY last_accessed ASC LIMIT ?1
                )",
                [to_delete],
            )?;
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Result<CacheStats> {
        let total_entries: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM cache_entries", [], |row| row.get(0))?;

        let now = Utc::now().to_rfc3339();
        let expired_entries: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?1",
            [&now],
            |row| row.get(0),
        )?;

        let total_access_count: u64 = self.conn.query_row(
            "SELECT COALESCE(SUM(access_count), 0) FROM cache_entries",
            [],
            |row| row.get(0),
        )?;

        let compressed_entries: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM cache_entries WHERE compressed = 1",
            [],
            |row| row.get(0),
        )?;

        let total_size: usize = self.conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(value)), 0) FROM cache_entries",
            [],
            |row| row.get(0),
        )?;

        Ok(CacheStats {
            total_entries,
            valid_entries: total_entries - expired_entries,
            expired_entries,
            total_access_count,
            compressed_entries,
            total_size_bytes: total_size,
        })
    }

    /// List all cache keys
    pub fn list_keys(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT key FROM cache_entries ORDER BY key")?;
        let rows = stmt.query_map([], |row| row.get(0))?;

        let mut keys = Vec::new();
        for row in rows {
            keys.push(row?);
        }
        Ok(keys)
    }

    /// Get entry metadata without the value
    pub fn get_entry_info(&self, key: &str) -> Result<Option<CacheEntryInfo>> {
        let result = self.conn.query_row(
            "SELECT compressed, created_at, expires_at, access_count, last_accessed, LENGTH(value)
             FROM cache_entries WHERE key = ?1",
            [key],
            |row| {
                Ok(CacheEntryInfo {
                    key: key.to_string(),
                    compressed: row.get::<_, i32>(0)? != 0,
                    created_at: row.get(1)?,
                    expires_at: row.get(2)?,
                    access_count: row.get(3)?,
                    last_accessed: row.get(4)?,
                    size_bytes: row.get(5)?,
                })
            },
        );

        match result {
            Ok(info) => Ok(Some(info)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub expired_entries: usize,
    pub total_access_count: u64,
    pub compressed_entries: usize,
    pub total_size_bytes: usize,
}

/// Cache entry info (without value)
#[derive(Debug, Clone)]
pub struct CacheEntryInfo {
    pub key: String,
    pub compressed: bool,
    pub created_at: String,
    pub expires_at: String,
    pub access_count: u64,
    pub last_accessed: String,
    pub size_bytes: usize,
}

// Base64 helper functions
#[allow(dead_code)]
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let mut chunks = data.chunks_exact(3);

    for chunk in chunks.by_ref() {
        let n = (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8 | chunk[2] as u32;
        result.push(ALPHABET[(n >> 18) as usize & 63] as char);
        result.push(ALPHABET[(n >> 12) as usize & 63] as char);
        result.push(ALPHABET[(n >> 6) as usize & 63] as char);
        result.push(ALPHABET[n as usize & 63] as char);
    }

    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        let mut n = (remainder[0] as u32) << 16;
        result.push(ALPHABET[(n >> 18) as usize & 63] as char);

        if remainder.len() > 1 {
            n |= (remainder[1] as u32) << 8;
            result.push(ALPHABET[(n >> 12) as usize & 63] as char);
            result.push(ALPHABET[(n >> 6) as usize & 63] as char);
        } else {
            result.push(ALPHABET[(n >> 12) as usize & 63] as char);
            result.push('=');
        }
        result.push('=');
    }

    result
}

#[allow(dead_code)]
fn base64_decode(data: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let data = data.trim_end_matches('=');

    let decode_char = |c: char| -> Option<u8> {
        match c {
            'A'..='Z' => Some(c as u8 - b'A'),
            'a'..='z' => Some(c as u8 - b'a' + 26),
            '0'..='9' => Some(c as u8 - b'0' + 52),
            '+' => Some(62),
            '/' => Some(63),
            _ => None,
        }
    };

    let chars: Vec<u8> = data.chars().filter_map(decode_char).collect();

    let mut chunks = chars.chunks_exact(4);

    for chunk in chunks.by_ref() {
        let n = (chunk[0] as u32) << 18
            | (chunk[1] as u32) << 12
            | (chunk[2] as u32) << 6
            | chunk[3] as u32;
        result.push((n >> 16) as u8);
        result.push((n >> 8) as u8);
        result.push(n as u8);
    }

    let remainder = chunks.remainder();
    if remainder.len() >= 2 {
        let n = (remainder[0] as u32) << 18
            | (remainder[1] as u32) << 12
            | remainder.get(2).map(|&c| (c as u32) << 6).unwrap_or(0);

        result.push((n >> 16) as u8);
        if remainder.len() >= 3 {
            result.push((n >> 8) as u8);
        }
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_backup_manager() {
        let dir = tempdir().unwrap();
        let config = BackupConfig {
            backup_dir: dir.path().to_path_buf(),
            compress: false,
            ..Default::default()
        };
        let mut manager = BackupManager::new(config);

        // Create a test database file
        let db_path = dir.path().join("test.db");
        fs::write(&db_path, b"test database content").unwrap();

        // Create backup
        let backup = manager.create_backup(&db_path).unwrap();
        assert!(backup.path.exists());
        assert!(backup.size_bytes > 0);
    }

    #[test]
    fn test_compression() {
        let original = b"Hello, this is a test message for compression!";
        let compressed = BackupManager::compress_data(original).unwrap();
        let decompressed = BackupManager::decompress_data(&compressed).unwrap();
        assert_eq!(original.to_vec(), decompressed);
    }

    #[test]
    fn test_should_backup() {
        let config = BackupConfig {
            enabled: true,
            interval_hours: 24,
            ..Default::default()
        };
        let manager = BackupManager::new(config);

        // Should backup if never backed up
        assert!(manager.should_backup());
    }

    #[test]
    fn test_backup_cleanup() {
        let dir = tempdir().unwrap();
        let backup_dir = dir.path().join("backups");
        let config = BackupConfig {
            backup_dir: backup_dir.clone(),
            max_backups: 2,
            compress: false,
            ..Default::default()
        };
        let mut manager = BackupManager::new(config);

        let db_path = dir.path().join("test.db");
        fs::write(&db_path, b"database content").unwrap();

        // Create 3 backups with slight delay so timestamps differ
        let _b1 = manager.create_backup(&db_path).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let _b2 = manager.create_backup(&db_path).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let _b3 = manager.create_backup(&db_path).unwrap();

        // After creating 3 with max_backups=2, only 2 should remain
        let backups = manager.list_backups().unwrap();
        assert_eq!(
            backups.len(),
            2,
            "Expected 2 backups after cleanup, got {}",
            backups.len()
        );
    }

    #[test]
    fn test_restore_backup() {
        let dir = tempdir().unwrap();
        let config = BackupConfig {
            backup_dir: dir.path().join("backups"),
            compress: false,
            ..Default::default()
        };
        let mut manager = BackupManager::new(config);

        let original_content = b"important database content 12345";
        let db_path = dir.path().join("source.db");
        fs::write(&db_path, original_content).unwrap();

        let backup_info = manager.create_backup(&db_path).unwrap();

        // Restore to a new location
        let restored_path = dir.path().join("restored.db");
        manager
            .restore_backup(&backup_info.path, &restored_path)
            .unwrap();

        let restored_content = fs::read(&restored_path).unwrap();
        assert_eq!(
            restored_content, original_content,
            "Restored content must match original"
        );
    }

    #[test]
    fn test_compressed_backup_roundtrip() {
        let dir = tempdir().unwrap();
        let config = BackupConfig {
            backup_dir: dir.path().join("backups"),
            compress: true,
            ..Default::default()
        };
        let mut manager = BackupManager::new(config);

        let original_content =
            b"compressed database content with repeated data repeated data repeated data";
        let db_path = dir.path().join("source.db");
        fs::write(&db_path, original_content).unwrap();

        let backup_info = manager.create_backup(&db_path).unwrap();
        assert!(backup_info.compressed, "Backup should be compressed");
        assert!(
            backup_info.filename.ends_with(".db.gz"),
            "Filename should end with .db.gz"
        );

        // Restore the compressed backup
        let restored_path = dir.path().join("restored.db");
        manager
            .restore_backup(&backup_info.path, &restored_path)
            .unwrap();

        let restored_content = fs::read(&restored_path).unwrap();
        assert_eq!(
            restored_content, original_content,
            "Decompressed content must match original"
        );
    }

    #[test]
    fn test_list_backups_sorted() {
        let dir = tempdir().unwrap();
        let backup_dir = dir.path().join("backups");
        let config = BackupConfig {
            backup_dir: backup_dir.clone(),
            max_backups: 10,
            compress: false,
            ..Default::default()
        };
        let mut manager = BackupManager::new(config);

        let db_path = dir.path().join("test.db");
        fs::write(&db_path, b"db content").unwrap();

        // Create backups with delays so timestamps differ
        let b1 = manager.create_backup(&db_path).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let b2 = manager.create_backup(&db_path).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let b3 = manager.create_backup(&db_path).unwrap();

        let backups = manager.list_backups().unwrap();
        assert_eq!(backups.len(), 3);

        // Newest first: b3 > b2 > b1
        assert_eq!(
            backups[0].filename, b3.filename,
            "First backup should be newest"
        );
        assert_eq!(
            backups[1].filename, b2.filename,
            "Second backup should be middle"
        );
        assert_eq!(
            backups[2].filename, b1.filename,
            "Third backup should be oldest"
        );
    }

    #[test]
    fn test_should_backup_disabled() {
        let config = BackupConfig {
            enabled: false,
            ..Default::default()
        };
        let manager = BackupManager::new(config);

        assert!(
            !manager.should_backup(),
            "should_backup must return false when enabled=false"
        );
    }
}
