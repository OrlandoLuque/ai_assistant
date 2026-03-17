//! Unified SQLite persistence layer.
//!
//! Consolidates session storage and memory snapshots into a single SQLite
//! database with proper schema versioning and write-through semantics.
//!
//! # Components
//!
//! - **`UnifiedDb`** — central coordinator: opens `unified.db`, manages
//!   schema versions, runs numbered migrations.
//! - **`SqliteSessionStore`** — write-through session storage backed by SQLite.
//!   Every `save_session` / `delete_session` is immediately persisted.
//! - **`SqliteMemoryStore`** — replaces compressed JSON snapshots with SQLite
//!   rows; supports atomic writes via transactions.
//!
//! # Schema versioning
//!
//! A `schema_versions` table tracks every migration that has been applied.
//! Migrations are numbered (V1, V2, …) and run exactly once, in order.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::messages::ChatMessage;
use crate::session::{ChatSession, ChatSessionStore, UserPreferences};

// ============================================================================
// Schema version tracking
// ============================================================================

/// A record of a single applied migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SchemaVersion {
    /// Migration number (1-based).
    pub version: u32,
    /// Human-readable description.
    pub description: String,
    /// RFC 3339 timestamp when this migration was applied.
    pub applied_at: String,
}

impl SchemaVersion {
    /// Create a new schema version record.
    pub fn new(version: u32, description: impl Into<String>) -> Self {
        Self {
            version,
            description: description.into(),
            applied_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Summary returned after running migrations.
#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    /// How many migrations were applied in this run.
    pub applied: u32,
    /// The database version after all migrations.
    pub current_version: u32,
    /// Any non-fatal warnings encountered during migration.
    pub warnings: Vec<String>,
}

// ============================================================================
// UnifiedDb — central persistence coordinator
// ============================================================================

/// Unified SQLite persistence coordinator.
///
/// Opens (or creates) a single `unified.db` that stores:
/// - Schema version history
/// - Sessions and their messages
/// - Memory snapshots
///
/// All writes use WAL mode and a 5-second busy timeout for safe concurrency.
pub struct UnifiedDb {
    conn: rusqlite::Connection,
    db_path: PathBuf,
}

impl std::fmt::Debug for UnifiedDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedDb")
            .field("db_path", &self.db_path)
            .finish()
    }
}

impl UnifiedDb {
    /// Open or create the unified database at the given path.
    ///
    /// Creates parent directories if needed, enables WAL mode, creates the
    /// schema-version table, and runs any pending migrations.
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory for {}", db_path.display()))?;
        }

        let conn = rusqlite::Connection::open(db_path)
            .with_context(|| format!("Failed to open unified db: {}", db_path.display()))?;

        // WAL mode for concurrent reads + busy timeout for lock contention
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let db = Self {
            conn,
            db_path: db_path.to_path_buf(),
        };

        db.init_schema_versions()?;
        db.run_migrations()?;

        Ok(db)
    }

    /// Return the filesystem path of this database.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Return the current schema version (0 if no migrations applied).
    pub fn current_version(&self) -> Result<u32> {
        let version: u32 = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_versions",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(version)
    }

    /// List all applied schema versions.
    pub fn applied_versions(&self) -> Result<Vec<SchemaVersion>> {
        let mut stmt = self
            .conn
            .prepare("SELECT version, description, applied_at FROM schema_versions ORDER BY version")?;

        let versions = stmt
            .query_map([], |row| {
                Ok(SchemaVersion {
                    version: row.get(0)?,
                    description: row.get(1)?,
                    applied_at: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(versions)
    }

    /// Borrow the raw connection (for advanced use / testing).
    pub fn connection(&self) -> &rusqlite::Connection {
        &self.conn
    }

    // --- Internal --------------------------------------------------------

    /// Create the schema_versions table if it doesn't exist.
    fn init_schema_versions(&self) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )",
            [],
        )?;
        Ok(())
    }

    /// Run all pending migrations in order.
    fn run_migrations(&self) -> Result<MigrationReport> {
        let current = self.current_version()?;
        let mut report = MigrationReport {
            current_version: current,
            ..Default::default()
        };

        // List of all migrations. Each entry: (version, description, SQL).
        let migrations: &[(u32, &str, &str)] = &[
            (
                1,
                "Sessions and messages tables",
                "CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    preferences_json TEXT NOT NULL DEFAULT '{}',
                    context_notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_session_messages_session
                    ON session_messages(session_id, sort_order);",
            ),
            (
                2,
                "Memory snapshots table",
                "CREATE TABLE IF NOT EXISTS memory_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    store_name TEXT NOT NULL,
                    data BLOB NOT NULL,
                    compressed INTEGER NOT NULL DEFAULT 0,
                    checksum INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memory_snapshots_store
                    ON memory_snapshots(store_name, created_at);",
            ),
            (
                3,
                "Session FTS5 for message search",
                "CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                    content,
                    content=session_messages,
                    content_rowid=id
                );
                CREATE TRIGGER IF NOT EXISTS session_msg_ai AFTER INSERT ON session_messages BEGIN
                    INSERT INTO session_messages_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS session_msg_ad AFTER DELETE ON session_messages BEGIN
                    INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS session_msg_au AFTER UPDATE ON session_messages BEGIN
                    INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO session_messages_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;",
            ),
            (
                4,
                "Memory entries table for key-value persistence",
                "CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    store_name TEXT NOT NULL,
                    entry_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(store_name, entry_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_entries_store
                    ON memory_entries(store_name);",
            ),
        ];

        for &(version, description, sql) in migrations {
            if version <= current {
                continue;
            }

            // Run migration inside a transaction
            let tx = self.conn.unchecked_transaction()?;
            match tx.execute_batch(sql) {
                Ok(()) => {
                    tx.execute(
                        "INSERT INTO schema_versions (version, description, applied_at) VALUES (?1, ?2, ?3)",
                        rusqlite::params![version, description, chrono::Utc::now().to_rfc3339()],
                    )?;
                    tx.commit()?;
                    report.applied += 1;
                    report.current_version = version;
                }
                Err(e) => {
                    // Transaction auto-rolls back on drop
                    report.warnings.push(format!(
                        "Migration V{} ({}) failed: {}",
                        version, description, e
                    ));
                    // Stop applying further migrations on failure
                    break;
                }
            }
        }

        Ok(report)
    }
}

// ============================================================================
// SqliteSessionStore — write-through session storage
// ============================================================================

/// Write-through session store backed by SQLite.
///
/// Every mutation (`save_session`, `delete_session`, `add_message`) is
/// immediately persisted. Reads come from the database, not from an
/// in-memory cache, so data is always consistent.
pub struct SqliteSessionStore<'a> {
    db: &'a UnifiedDb,
}

impl<'a> SqliteSessionStore<'a> {
    /// Create a new session store using the given `UnifiedDb`.
    pub fn new(db: &'a UnifiedDb) -> Self {
        Self { db }
    }

    /// Save a full session (upsert). Replaces all messages atomically.
    pub fn save_session(&self, session: &ChatSession) -> Result<()> {
        let tx = self.db.conn.unchecked_transaction()?;

        let prefs_json = serde_json::to_string(&session.preferences)
            .unwrap_or_else(|_| "{}".to_string());

        // Upsert session metadata
        tx.execute(
            "INSERT INTO sessions (id, name, preferences_json, context_notes, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                preferences_json = excluded.preferences_json,
                context_notes = excluded.context_notes,
                updated_at = excluded.updated_at",
            rusqlite::params![
                session.id,
                session.name,
                prefs_json,
                session.context_notes,
                session.created_at.to_rfc3339(),
                session.updated_at.to_rfc3339(),
            ],
        )?;

        // Replace all messages: delete old, insert new in order
        tx.execute(
            "DELETE FROM session_messages WHERE session_id = ?1",
            rusqlite::params![session.id],
        )?;

        let mut insert = tx.prepare(
            "INSERT INTO session_messages (session_id, role, content, timestamp, sort_order)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for (i, msg) in session.messages.iter().enumerate() {
            insert.execute(rusqlite::params![
                session.id,
                msg.role,
                msg.content,
                msg.timestamp.to_rfc3339(),
                i as i64,
            ])?;
        }

        drop(insert);
        tx.commit()?;
        Ok(())
    }

    /// Append a single message to an existing session (write-through).
    ///
    /// More efficient than `save_session` when only adding a new message.
    pub fn append_message(&self, session_id: &str, msg: &ChatMessage) -> Result<()> {
        // Get current max sort_order
        let max_order: i64 = self
            .db
            .conn
            .query_row(
                "SELECT COALESCE(MAX(sort_order), -1) FROM session_messages WHERE session_id = ?1",
                rusqlite::params![session_id],
                |row| row.get(0),
            )
            .unwrap_or(-1);

        self.db.conn.execute(
            "INSERT INTO session_messages (session_id, role, content, timestamp, sort_order)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                session_id,
                msg.role,
                msg.content,
                msg.timestamp.to_rfc3339(),
                max_order + 1,
            ],
        )?;

        // Touch session updated_at
        self.db.conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            rusqlite::params![chrono::Utc::now().to_rfc3339(), session_id],
        )?;

        Ok(())
    }

    /// Load a session by ID, including all its messages.
    pub fn load_session(&self, session_id: &str) -> Result<Option<ChatSession>> {
        let row = self.db.conn.query_row(
            "SELECT id, name, preferences_json, context_notes, created_at, updated_at
             FROM sessions WHERE id = ?1",
            rusqlite::params![session_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        );

        let (id, name, prefs_json, context_notes, created_str, updated_str) = match row {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let preferences: UserPreferences =
            serde_json::from_str(&prefs_json).unwrap_or_default();

        let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        // Load messages in order
        let mut stmt = self.db.conn.prepare(
            "SELECT role, content, timestamp FROM session_messages
             WHERE session_id = ?1 ORDER BY sort_order ASC",
        )?;

        let messages: Vec<ChatMessage> = stmt
            .query_map(rusqlite::params![session_id], |row| {
                let role: String = row.get(0)?;
                let content: String = row.get(1)?;
                let ts_str: String = row.get(2)?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(&ts_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_else(|_| chrono::Utc::now());
                Ok(ChatMessage {
                    role,
                    content,
                    timestamp,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(Some(ChatSession {
            id,
            name,
            messages,
            preferences,
            created_at,
            updated_at,
            context_notes,
        }))
    }

    /// List all sessions (metadata only, no messages) sorted by last update.
    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut stmt = self.db.conn.prepare(
            "SELECT s.id, s.name, s.created_at, s.updated_at,
                    (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.id)
             FROM sessions s ORDER BY s.updated_at DESC",
        )?;

        let summaries = stmt
            .query_map([], |row| {
                Ok(SessionSummary {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                    message_count: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(summaries)
    }

    /// Delete a session and all its messages.
    pub fn delete_session(&self, session_id: &str) -> Result<bool> {
        // Messages are cascade-deleted via FK, but SQLite foreign_keys must be enabled
        self.db
            .conn
            .execute("PRAGMA foreign_keys = ON", [])?;

        let deleted = self.db.conn.execute(
            "DELETE FROM sessions WHERE id = ?1",
            rusqlite::params![session_id],
        )?;

        Ok(deleted > 0)
    }

    /// Search session messages using FTS5 full-text search.
    pub fn search_messages(&self, query: &str, limit: usize) -> Result<Vec<MessageSearchResult>> {
        let mut stmt = self.db.conn.prepare(
            "SELECT m.session_id, m.role, m.content, m.timestamp,
                    rank
             FROM session_messages_fts fts
             JOIN session_messages m ON m.id = fts.rowid
             WHERE session_messages_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let results = stmt
            .query_map(rusqlite::params![query, limit as i64], |row| {
                Ok(MessageSearchResult {
                    session_id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    timestamp: row.get(3)?,
                    rank: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }

    /// Import all sessions from an existing `ChatSessionStore`.
    ///
    /// Skips sessions that already exist in the database (by ID).
    pub fn import_from_store(&self, store: &ChatSessionStore) -> Result<ImportReport> {
        let mut report = ImportReport::default();

        for session in &store.sessions {
            let exists: bool = self
                .db
                .conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM sessions WHERE id = ?1",
                    rusqlite::params![session.id],
                    |row| row.get(0),
                )
                .unwrap_or(false);

            if exists {
                report.skipped += 1;
                continue;
            }

            match self.save_session(session) {
                Ok(()) => {
                    report.imported += 1;
                    report.messages_imported += session.messages.len();
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Session {}: {}", session.id, e));
                }
            }
        }

        Ok(report)
    }

    /// Return the total number of sessions.
    pub fn session_count(&self) -> Result<usize> {
        let count: i64 = self
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Return the total number of messages across all sessions.
    pub fn message_count(&self) -> Result<usize> {
        let count: i64 = self.db.conn.query_row(
            "SELECT COUNT(*) FROM session_messages",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// Summary of a session (metadata only).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionSummary {
    /// Session ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Creation timestamp (RFC 3339).
    pub created_at: String,
    /// Last update timestamp (RFC 3339).
    pub updated_at: String,
    /// Number of messages in this session.
    pub message_count: i64,
}

impl SessionSummary {
    /// Create a new session summary.
    pub fn new(id: String, name: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id,
            name,
            created_at: now.clone(),
            updated_at: now,
            message_count: 0,
        }
    }
}

/// Result of a full-text search across session messages.
#[derive(Debug, Clone)]
pub struct MessageSearchResult {
    /// Which session this message belongs to.
    pub session_id: String,
    /// Message role (user / assistant / system).
    pub role: String,
    /// Full message content.
    pub content: String,
    /// RFC 3339 timestamp.
    pub timestamp: String,
    /// FTS5 rank score (lower = more relevant).
    pub rank: f64,
}

/// Report from `import_from_store`.
#[derive(Debug, Clone, Default)]
pub struct ImportReport {
    /// Sessions successfully imported.
    pub imported: usize,
    /// Sessions skipped (already exist).
    pub skipped: usize,
    /// Total messages imported.
    pub messages_imported: usize,
    /// Errors encountered (session ID + message).
    pub errors: Vec<String>,
}

// ============================================================================
// SqliteMemoryStore — memory snapshot persistence
// ============================================================================

/// SQLite-backed memory snapshot store.
///
/// Replaces file-based compressed JSON snapshots with SQLite rows.
/// Supports atomic writes and rotation (max snapshots per store).
pub struct SqliteMemoryStore<'a> {
    db: &'a UnifiedDb,
    /// Maximum number of snapshots to keep per store name.
    pub max_snapshots: usize,
}

impl<'a> SqliteMemoryStore<'a> {
    /// Create a new memory store using the given `UnifiedDb`.
    pub fn new(db: &'a UnifiedDb) -> Self {
        Self {
            db,
            max_snapshots: 5,
        }
    }

    /// Create with a custom max_snapshots limit.
    pub fn with_max_snapshots(db: &'a UnifiedDb, max_snapshots: usize) -> Self {
        Self { db, max_snapshots }
    }

    /// Save a memory snapshot (optionally compressed).
    ///
    /// If `compressed` is true, `data` should be gzip-compressed.
    /// Automatically rotates old snapshots beyond `max_snapshots`.
    pub fn save_snapshot(
        &self,
        store_name: &str,
        data: &[u8],
        compressed: bool,
        metadata_json: &str,
    ) -> Result<i64> {
        let checksum = compute_checksum(data);
        let now = chrono::Utc::now().to_rfc3339();

        let id = self.db.conn.query_row(
            "INSERT INTO memory_snapshots (store_name, data, compressed, checksum, created_at, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             RETURNING id",
            rusqlite::params![
                store_name,
                data,
                compressed as i32,
                checksum as i64,
                now,
                metadata_json,
            ],
            |row| row.get(0),
        )?;

        // Rotate: keep only the latest max_snapshots
        self.rotate_snapshots(store_name)?;

        Ok(id)
    }

    /// Load the most recent snapshot for a store.
    pub fn load_latest(&self, store_name: &str) -> Result<Option<MemorySnapshot>> {
        let row = self.db.conn.query_row(
            "SELECT id, data, compressed, checksum, created_at, metadata_json
             FROM memory_snapshots
             WHERE store_name = ?1
             ORDER BY created_at DESC
             LIMIT 1",
            rusqlite::params![store_name],
            |row| {
                Ok(MemorySnapshot {
                    id: row.get(0)?,
                    store_name: store_name.to_string(),
                    data: row.get(1)?,
                    compressed: row.get::<_, i32>(2)? != 0,
                    checksum: row.get::<_, i64>(3)? as u64,
                    created_at: row.get(4)?,
                    metadata_json: row.get(5)?,
                })
            },
        );

        match row {
            Ok(snap) => {
                // Verify checksum
                let computed = compute_checksum(&snap.data);
                if computed != snap.checksum {
                    anyhow::bail!(
                        "Checksum mismatch for store '{}': stored={}, computed={}",
                        store_name,
                        snap.checksum,
                        computed
                    );
                }
                Ok(Some(snap))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Load a specific snapshot by ID.
    pub fn load_by_id(&self, snapshot_id: i64) -> Result<Option<MemorySnapshot>> {
        let row = self.db.conn.query_row(
            "SELECT id, store_name, data, compressed, checksum, created_at, metadata_json
             FROM memory_snapshots WHERE id = ?1",
            rusqlite::params![snapshot_id],
            |row| {
                Ok(MemorySnapshot {
                    id: row.get(0)?,
                    store_name: row.get(1)?,
                    data: row.get(2)?,
                    compressed: row.get::<_, i32>(3)? != 0,
                    checksum: row.get::<_, i64>(4)? as u64,
                    created_at: row.get(5)?,
                    metadata_json: row.get(6)?,
                })
            },
        );

        match row {
            Ok(snap) => Ok(Some(snap)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all snapshots for a store (newest first).
    pub fn list_snapshots(&self, store_name: &str) -> Result<Vec<SnapshotSummary>> {
        let mut stmt = self.db.conn.prepare(
            "SELECT id, LENGTH(data), compressed, created_at, metadata_json
             FROM memory_snapshots
             WHERE store_name = ?1
             ORDER BY created_at DESC",
        )?;

        let summaries = stmt
            .query_map(rusqlite::params![store_name], |row| {
                Ok(SnapshotSummary {
                    id: row.get(0)?,
                    store_name: store_name.to_string(),
                    size_bytes: row.get(1)?,
                    compressed: row.get::<_, i32>(2)? != 0,
                    created_at: row.get(3)?,
                    metadata_json: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(summaries)
    }

    /// Delete a specific snapshot.
    pub fn delete_snapshot(&self, snapshot_id: i64) -> Result<bool> {
        let deleted = self.db.conn.execute(
            "DELETE FROM memory_snapshots WHERE id = ?1",
            rusqlite::params![snapshot_id],
        )?;
        Ok(deleted > 0)
    }

    /// Delete all snapshots for a store.
    pub fn clear_store(&self, store_name: &str) -> Result<usize> {
        let deleted = self.db.conn.execute(
            "DELETE FROM memory_snapshots WHERE store_name = ?1",
            rusqlite::params![store_name],
        )?;
        Ok(deleted)
    }

    /// Save a key-value memory entry (upsert).
    pub fn put_entry(
        &self,
        store_name: &str,
        key: &str,
        value_json: &str,
        importance: f64,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.db.conn.execute(
            "INSERT INTO memory_entries (store_name, entry_key, value_json, importance, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)
             ON CONFLICT(store_name, entry_key) DO UPDATE SET
                value_json = excluded.value_json,
                importance = excluded.importance,
                updated_at = excluded.updated_at",
            rusqlite::params![store_name, key, value_json, importance, now],
        )?;
        Ok(())
    }

    /// Get a memory entry by store name and key.
    pub fn get_entry(&self, store_name: &str, key: &str) -> Result<Option<MemoryEntry>> {
        let row = self.db.conn.query_row(
            "SELECT id, value_json, importance, created_at, updated_at
             FROM memory_entries WHERE store_name = ?1 AND entry_key = ?2",
            rusqlite::params![store_name, key],
            |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    store_name: store_name.to_string(),
                    key: key.to_string(),
                    value_json: row.get(1)?,
                    importance: row.get(2)?,
                    created_at: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            },
        );

        match row {
            Ok(entry) => Ok(Some(entry)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List all entries for a store, sorted by importance descending.
    pub fn list_entries(&self, store_name: &str) -> Result<Vec<MemoryEntry>> {
        let mut stmt = self.db.conn.prepare(
            "SELECT id, entry_key, value_json, importance, created_at, updated_at
             FROM memory_entries WHERE store_name = ?1
             ORDER BY importance DESC",
        )?;

        let entries = stmt
            .query_map(rusqlite::params![store_name], |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    store_name: store_name.to_string(),
                    key: row.get(1)?,
                    value_json: row.get(2)?,
                    importance: row.get(3)?,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    /// Delete a memory entry.
    pub fn delete_entry(&self, store_name: &str, key: &str) -> Result<bool> {
        let deleted = self.db.conn.execute(
            "DELETE FROM memory_entries WHERE store_name = ?1 AND entry_key = ?2",
            rusqlite::params![store_name, key],
        )?;
        Ok(deleted > 0)
    }

    /// Return the snapshot count for a store.
    pub fn snapshot_count(&self, store_name: &str) -> Result<usize> {
        let count: i64 = self.db.conn.query_row(
            "SELECT COUNT(*) FROM memory_snapshots WHERE store_name = ?1",
            rusqlite::params![store_name],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    // --- Internal --------------------------------------------------------

    /// Remove oldest snapshots beyond max_snapshots for a given store.
    fn rotate_snapshots(&self, store_name: &str) -> Result<()> {
        let count = self.snapshot_count(store_name)?;
        if count <= self.max_snapshots {
            return Ok(());
        }

        let to_delete = count - self.max_snapshots;
        self.db.conn.execute(
            "DELETE FROM memory_snapshots WHERE id IN (
                SELECT id FROM memory_snapshots
                WHERE store_name = ?1
                ORDER BY created_at ASC
                LIMIT ?2
            )",
            rusqlite::params![store_name, to_delete as i64],
        )?;

        Ok(())
    }
}

/// A loaded memory snapshot.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Database row ID.
    pub id: i64,
    /// Which store this belongs to.
    pub store_name: String,
    /// Raw data (may be compressed).
    pub data: Vec<u8>,
    /// Whether `data` is gzip-compressed.
    pub compressed: bool,
    /// FNV-1a checksum of `data`.
    pub checksum: u64,
    /// RFC 3339 timestamp.
    pub created_at: String,
    /// JSON metadata.
    pub metadata_json: String,
}

/// Summary of a snapshot (without the data blob).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SnapshotSummary {
    /// Database row ID.
    pub id: i64,
    /// Which store this belongs to.
    pub store_name: String,
    /// Size of the data in bytes.
    pub size_bytes: i64,
    /// Whether the data is compressed.
    pub compressed: bool,
    /// RFC 3339 timestamp.
    pub created_at: String,
    /// JSON metadata.
    pub metadata_json: String,
}

impl SnapshotSummary {
    /// Create a new snapshot summary.
    pub fn new(id: i64, store_name: String) -> Self {
        Self {
            id,
            store_name,
            size_bytes: 0,
            compressed: false,
            created_at: chrono::Utc::now().to_rfc3339(),
            metadata_json: "{}".to_string(),
        }
    }
}

/// A key-value memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryEntry {
    /// Database row ID.
    pub id: i64,
    /// Store name this entry belongs to.
    pub store_name: String,
    /// Entry key.
    pub key: String,
    /// JSON-serialized value.
    pub value_json: String,
    /// Importance score (higher = more important).
    pub importance: f64,
    /// RFC 3339 creation timestamp.
    pub created_at: String,
    /// RFC 3339 last-update timestamp.
    pub updated_at: String,
}

impl MemoryEntry {
    /// Create a new memory entry.
    pub fn new(store_name: String, key: String, value_json: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: 0,
            store_name,
            key,
            value_json,
            importance: 1.0,
            created_at: now.clone(),
            updated_at: now,
        }
    }
}

// ============================================================================
// Utility
// ============================================================================

/// Compute FNV-1a checksum (same algorithm as `AutoPersistenceConfig::compute_checksum`).
pub fn compute_checksum(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn temp_db() -> (UnifiedDb, NamedTempFile) {
        let tmp = NamedTempFile::new().expect("temp file");
        let db = UnifiedDb::open(tmp.path()).expect("open unified db");
        (db, tmp)
    }

    #[test]
    fn test_schema_version_tracking() {
        let (db, _tmp) = temp_db();

        let version = db.current_version().expect("get version");
        assert_eq!(version, 4, "all 4 migrations should have run");

        let versions = db.applied_versions().expect("list versions");
        assert_eq!(versions.len(), 4);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[3].version, 4);
    }

    #[test]
    fn test_migration_idempotency() {
        let (db, tmp) = temp_db();
        let v1 = db.current_version().expect("v1");

        // Re-open same database — migrations should not re-run
        drop(db);
        let db2 = UnifiedDb::open(tmp.path()).expect("reopen");
        let v2 = db2.current_version().expect("v2");
        assert_eq!(v1, v2);

        let versions = db2.applied_versions().expect("versions");
        assert_eq!(versions.len(), 4, "still 4 — no duplicates");
    }

    #[test]
    fn test_session_write_through_crud() {
        let (db, _tmp) = temp_db();
        let store = SqliteSessionStore::new(&db);

        // Create a session
        let mut session = ChatSession::new("Test session");
        session.messages.push(ChatMessage::user("Hello"));
        session
            .messages
            .push(ChatMessage::assistant("Hi there!"));

        // Save
        store.save_session(&session).expect("save");
        assert_eq!(store.session_count().expect("count"), 1);
        assert_eq!(store.message_count().expect("msg count"), 2);

        // Load
        let loaded = store
            .load_session(&session.id)
            .expect("load")
            .expect("should exist");
        assert_eq!(loaded.name, "Test session");
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].role, "user");
        assert_eq!(loaded.messages[1].content, "Hi there!");

        // Append message
        store
            .append_message(&session.id, &ChatMessage::user("How are you?"))
            .expect("append");
        assert_eq!(store.message_count().expect("msg count"), 3);

        // Delete
        let deleted = store.delete_session(&session.id).expect("delete");
        assert!(deleted);
        assert_eq!(store.session_count().expect("count"), 0);
    }

    #[test]
    fn test_session_list_and_search() {
        let (db, _tmp) = temp_db();
        let store = SqliteSessionStore::new(&db);

        // Create two sessions
        let mut s1 = ChatSession::new("Alpha session");
        s1.messages
            .push(ChatMessage::user("Tell me about quantum computing"));
        store.save_session(&s1).expect("save s1");

        let mut s2 = ChatSession::new("Beta session");
        s2.messages
            .push(ChatMessage::user("What is machine learning?"));
        store.save_session(&s2).expect("save s2");

        // List
        let list = store.list_sessions().expect("list");
        assert_eq!(list.len(), 2);

        // FTS search
        let results = store.search_messages("quantum", 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, s1.id);
    }

    #[test]
    fn test_session_upsert() {
        let (db, _tmp) = temp_db();
        let store = SqliteSessionStore::new(&db);

        let mut session = ChatSession::new("Evolving session");
        session.messages.push(ChatMessage::user("First"));
        store.save_session(&session).expect("save v1");

        // Update: change name and add message
        session.name = "Updated session".to_string();
        session.messages.push(ChatMessage::assistant("Second"));
        session.touch();
        store.save_session(&session).expect("save v2");

        assert_eq!(store.session_count().expect("count"), 1);
        let loaded = store
            .load_session(&session.id)
            .expect("load")
            .expect("exists");
        assert_eq!(loaded.name, "Updated session");
        assert_eq!(loaded.messages.len(), 2);
    }

    #[test]
    fn test_import_from_chat_session_store() {
        let (db, _tmp) = temp_db();
        let store = SqliteSessionStore::new(&db);

        // Build a ChatSessionStore with 2 sessions (explicit IDs to avoid
        // timestamp collision when both are created in the same millisecond)
        let mut css = ChatSessionStore::new();
        let mut s1 = ChatSession::new("Imported 1");
        s1.id = "import_test_1".to_string();
        s1.messages.push(ChatMessage::user("Hello from JSON"));
        css.save_session(s1.clone());

        let mut s2 = ChatSession::new("Imported 2");
        s2.id = "import_test_2".to_string();
        s2.messages.push(ChatMessage::user("Another one"));
        s2.messages
            .push(ChatMessage::assistant("Response"));
        css.save_session(s2);

        // Import
        let report = store.import_from_store(&css).expect("import");
        assert_eq!(report.imported, 2);
        assert_eq!(report.messages_imported, 3);
        assert_eq!(report.skipped, 0);
        assert!(report.errors.is_empty());

        // Re-import should skip all
        let report2 = store.import_from_store(&css).expect("re-import");
        assert_eq!(report2.imported, 0);
        assert_eq!(report2.skipped, 2);
    }

    #[test]
    fn test_memory_snapshot_save_load() {
        let (db, _tmp) = temp_db();
        let mem = SqliteMemoryStore::new(&db);

        let data = b"test memory data for episodic store";
        let id = mem
            .save_snapshot("episodic", data, false, "{\"version\":1}")
            .expect("save snapshot");
        assert!(id > 0);

        // Load latest
        let snap = mem
            .load_latest("episodic")
            .expect("load")
            .expect("should exist");
        assert_eq!(snap.data, data);
        assert!(!snap.compressed);
        assert_eq!(snap.store_name, "episodic");

        // Load by ID
        let snap2 = mem
            .load_by_id(id)
            .expect("load by id")
            .expect("exists");
        assert_eq!(snap2.id, id);
    }

    #[test]
    fn test_memory_snapshot_rotation() {
        let (db, _tmp) = temp_db();
        let mem = SqliteMemoryStore::with_max_snapshots(&db, 3);

        // Save 5 snapshots
        for i in 0..5u8 {
            mem.save_snapshot("rotating", &[i], false, "{}")
                .expect("save");
        }

        // Only 3 should remain
        let count = mem.snapshot_count("rotating").expect("count");
        assert_eq!(count, 3);

        // Latest should be the last one saved
        let latest = mem
            .load_latest("rotating")
            .expect("load")
            .expect("exists");
        assert_eq!(latest.data, vec![4u8]);
    }

    #[test]
    fn test_memory_snapshot_checksum_verification() {
        let (db, _tmp) = temp_db();
        let mem = SqliteMemoryStore::new(&db);

        let data = b"integrity check data";
        mem.save_snapshot("integrity", data, false, "{}")
            .expect("save");

        // Corrupt the checksum in the database
        db.conn
            .execute(
                "UPDATE memory_snapshots SET checksum = 0 WHERE store_name = 'integrity'",
                [],
            )
            .expect("corrupt");

        // Load should fail with checksum mismatch
        let result = mem.load_latest("integrity");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Checksum mismatch")
        );
    }

    #[test]
    fn test_memory_entry_crud() {
        let (db, _tmp) = temp_db();
        let mem = SqliteMemoryStore::new(&db);

        // Put entry
        mem.put_entry("facts", "user_name", "\"Orlando\"", 0.9)
            .expect("put");

        // Get entry
        let entry = mem
            .get_entry("facts", "user_name")
            .expect("get")
            .expect("exists");
        assert_eq!(entry.value_json, "\"Orlando\"");
        assert!((entry.importance - 0.9).abs() < f64::EPSILON);

        // Update (upsert)
        mem.put_entry("facts", "user_name", "\"Lander\"", 1.0)
            .expect("upsert");
        let updated = mem
            .get_entry("facts", "user_name")
            .expect("get")
            .expect("exists");
        assert_eq!(updated.value_json, "\"Lander\"");

        // List entries
        mem.put_entry("facts", "user_role", "\"developer\"", 0.5)
            .expect("put 2");
        let entries = mem.list_entries("facts").expect("list");
        assert_eq!(entries.len(), 2);
        // Sorted by importance DESC → "user_name" (1.0) first
        assert_eq!(entries[0].key, "user_name");

        // Delete
        let deleted = mem.delete_entry("facts", "user_role").expect("delete");
        assert!(deleted);
        let entries2 = mem.list_entries("facts").expect("list2");
        assert_eq!(entries2.len(), 1);
    }

    #[test]
    fn test_memory_clear_store() {
        let (db, _tmp) = temp_db();
        let mem = SqliteMemoryStore::new(&db);

        mem.save_snapshot("cleanup", b"data1", false, "{}")
            .expect("s1");
        mem.save_snapshot("cleanup", b"data2", false, "{}")
            .expect("s2");
        mem.save_snapshot("other", b"keep", false, "{}")
            .expect("s3");

        let cleared = mem.clear_store("cleanup").expect("clear");
        assert_eq!(cleared, 2);

        // "other" store untouched
        assert_eq!(mem.snapshot_count("other").expect("c"), 1);
    }

    #[test]
    fn test_compute_checksum_deterministic() {
        let data = b"hello world";
        let c1 = compute_checksum(data);
        let c2 = compute_checksum(data);
        assert_eq!(c1, c2);

        // Different data → different checksum
        let c3 = compute_checksum(b"hello worle");
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_empty_database_operations() {
        let (db, _tmp) = temp_db();
        let store = SqliteSessionStore::new(&db);
        let mem = SqliteMemoryStore::new(&db);

        // Sessions
        assert_eq!(store.session_count().expect("c"), 0);
        assert!(store.load_session("nonexistent").expect("load").is_none());
        assert!(!store.delete_session("nonexistent").expect("del"));
        assert!(store.list_sessions().expect("list").is_empty());
        assert!(store.search_messages("anything", 10).expect("search").is_empty());

        // Memory
        assert_eq!(mem.snapshot_count("any").expect("c"), 0);
        assert!(mem.load_latest("any").expect("load").is_none());
        assert!(mem.list_snapshots("any").expect("list").is_empty());
        assert!(mem.get_entry("any", "key").expect("get").is_none());
    }
}
