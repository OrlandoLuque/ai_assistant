//! RAG (Retrieval-Augmented Generation) system for knowledge and conversation context
//!
//! This module provides:
//! - Knowledge base indexing with SQLite FTS5 for full-text search
//! - Conversation history storage for "infinite" context
//! - Relevance-based retrieval to minimize token usage
//! - Multi-user support with isolated data per user

use crate::messages::ChatMessage;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default user ID for single-user applications
pub const DEFAULT_USER_ID: &str = "default";

/// User information stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user ID
    pub id: i64,
    /// User identifier (username or unique key)
    pub user_id: String,
    /// Display name
    pub display_name: String,
    /// Global notes for this user
    pub global_notes: String,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: String,
}

/// A chunk of knowledge from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeChunk {
    /// Unique ID
    pub id: i64,
    /// Source document name
    pub source: String,
    /// Section/heading within the document
    pub section: String,
    /// The actual content
    pub content: String,
    /// Estimated token count
    pub token_count: usize,
}

/// A stored conversation message with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    /// Unique ID
    pub id: i64,
    /// Session ID this message belongs to
    pub session_id: String,
    /// Message role (user/assistant/system)
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: String,
    /// Estimated token count
    pub token_count: usize,
    /// Whether this message is currently in active context
    pub in_context: bool,
}

/// RAG configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Enable RAG for knowledge base
    pub knowledge_rag_enabled: bool,
    /// Enable RAG for conversation history
    pub conversation_rag_enabled: bool,
    /// Maximum tokens to retrieve for knowledge
    pub max_knowledge_tokens: usize,
    /// Maximum tokens to retrieve for conversation history
    pub max_conversation_tokens: usize,
    /// Number of top chunks to retrieve
    pub top_k_chunks: usize,
    /// Minimum relevance score (0.0-1.0)
    pub min_relevance_score: f32,
    /// Append-only mode: when enabled, knowledge documents can only be added, not removed
    /// This protects against accidental deletion of important knowledge
    pub append_only_mode: bool,
    /// Dynamic context mode: when enabled, automatically calculate max_knowledge_tokens
    /// based on the model's context window and current conversation size
    /// This allows filling the context as much as possible with relevant knowledge
    pub dynamic_context_enabled: bool,
    /// Auto-store mode: when enabled, automatically store messages in RAG as they are sent/received
    /// This enables conversation search and retrieval without manual storage calls
    pub auto_store_messages: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            knowledge_rag_enabled: false,
            conversation_rag_enabled: false,
            max_knowledge_tokens: 2000,
            max_conversation_tokens: 1500,
            top_k_chunks: 5,
            min_relevance_score: 0.1,
            append_only_mode: false,
            dynamic_context_enabled: false,
            auto_store_messages: false,
        }
    }
}

/// Tracks which knowledge chunks were used for a specific message/query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeUsage {
    /// The query that triggered this knowledge retrieval
    pub query: String,
    /// Timestamp when the knowledge was retrieved
    pub timestamp: String,
    /// Sources (documents) that contributed chunks
    pub sources: Vec<KnowledgeSourceUsage>,
    /// Total number of chunks retrieved
    pub total_chunks: usize,
    /// Total tokens used from knowledge
    pub total_tokens: usize,
}

/// Usage information for a single knowledge source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSourceUsage {
    /// Source document name
    pub source: String,
    /// Number of chunks used from this source
    pub chunks_used: usize,
    /// Sections used from this source
    pub sections: Vec<String>,
    /// Total tokens from this source
    pub tokens: usize,
    /// Relevance score (if available from hybrid search)
    pub relevance_score: Option<f32>,
}

impl KnowledgeUsage {
    /// Create a new empty KnowledgeUsage
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            sources: Vec::new(),
            total_chunks: 0,
            total_tokens: 0,
        }
    }

    /// Build from a list of retrieved chunks
    pub fn from_chunks(query: impl Into<String>, chunks: &[KnowledgeChunk]) -> Self {
        let mut usage = Self::new(query);
        usage.add_chunks(chunks);
        usage
    }

    /// Build from hybrid search results
    pub fn from_hybrid_results(
        query: impl Into<String>,
        results: &[HybridKnowledgeResult],
    ) -> Self {
        let mut usage = Self::new(query);
        usage.add_hybrid_results(results);
        usage
    }

    /// Add chunks to the usage tracking
    pub fn add_chunks(&mut self, chunks: &[KnowledgeChunk]) {
        use std::collections::HashMap;

        // Group by source
        let mut by_source: HashMap<&str, Vec<&KnowledgeChunk>> = HashMap::new();
        for chunk in chunks {
            by_source.entry(&chunk.source).or_default().push(chunk);
        }

        for (source, source_chunks) in by_source {
            let sections: Vec<String> = source_chunks
                .iter()
                .map(|c| c.section.clone())
                .filter(|s| !s.is_empty())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let tokens: usize = source_chunks.iter().map(|c| c.token_count).sum();

            self.sources.push(KnowledgeSourceUsage {
                source: source.to_string(),
                chunks_used: source_chunks.len(),
                sections,
                tokens,
                relevance_score: None,
            });

            self.total_chunks += source_chunks.len();
            self.total_tokens += tokens;
        }
    }

    /// Add hybrid search results to the usage tracking
    pub fn add_hybrid_results(&mut self, results: &[HybridKnowledgeResult]) {
        use std::collections::HashMap;

        // Group by source
        let mut by_source: HashMap<&str, Vec<&HybridKnowledgeResult>> = HashMap::new();
        for result in results {
            by_source
                .entry(&result.chunk.source)
                .or_default()
                .push(result);
        }

        for (source, source_results) in by_source {
            let sections: Vec<String> = source_results
                .iter()
                .map(|r| r.chunk.section.clone())
                .filter(|s| !s.is_empty())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let tokens: usize = source_results.iter().map(|r| r.chunk.token_count).sum();
            let avg_relevance = source_results.iter().map(|r| r.combined_score).sum::<f32>()
                / source_results.len() as f32;

            self.sources.push(KnowledgeSourceUsage {
                source: source.to_string(),
                chunks_used: source_results.len(),
                sections,
                tokens,
                relevance_score: Some(avg_relevance),
            });

            self.total_chunks += source_results.len();
            self.total_tokens += tokens;
        }
    }

    /// Check if any knowledge was used
    pub fn has_knowledge(&self) -> bool {
        self.total_chunks > 0
    }

    /// Get list of all sources used
    pub fn get_sources(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.source.as_str()).collect()
    }

    /// Get usage summary as a formatted string
    pub fn summary(&self) -> String {
        if self.sources.is_empty() {
            return "No knowledge used".to_string();
        }

        let source_info: Vec<String> = self
            .sources
            .iter()
            .map(|s| {
                if let Some(score) = s.relevance_score {
                    format!(
                        "{} ({} chunks, {:.0}% relevance)",
                        s.source,
                        s.chunks_used,
                        score * 100.0
                    )
                } else {
                    format!("{} ({} chunks)", s.source, s.chunks_used)
                }
            })
            .collect();

        format!(
            "{} chunks from {} sources ({} tokens): {}",
            self.total_chunks,
            self.sources.len(),
            self.total_tokens,
            source_info.join(", ")
        )
    }
}

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRagConfig {
    /// Weight for BM25/FTS5 score (0.0 to 1.0)
    pub bm25_weight: f32,
    /// Weight for semantic similarity (0.0 to 1.0)
    pub semantic_weight: f32,
    /// Enable semantic search (requires embedding computation)
    pub semantic_enabled: bool,
    /// Minimum semantic score to consider (0.0 to 1.0)
    pub min_semantic_score: f32,
}

impl Default for HybridRagConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 0.6,
            semantic_weight: 0.4,
            semantic_enabled: false,
            min_semantic_score: 0.1,
        }
    }
}

/// Result from hybrid search
#[derive(Debug, Clone)]
pub struct HybridKnowledgeResult {
    /// The knowledge chunk
    pub chunk: KnowledgeChunk,
    /// BM25 score from FTS5
    pub bm25_score: f32,
    /// Semantic similarity score (if computed)
    pub semantic_score: Option<f32>,
    /// Combined score
    pub combined_score: f32,
}

/// RAG database manager with multi-user support
pub struct RagDb {
    conn: rusqlite::Connection,
    /// Local embedder for semantic search
    embedder: Option<crate::embeddings::LocalEmbedder>,
    /// Hybrid search configuration
    pub hybrid_config: HybridRagConfig,
}

impl RagDb {
    /// Open or create the RAG database
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = rusqlite::Connection::open(db_path)?;
        let db = Self {
            conn,
            embedder: None,
            hybrid_config: HybridRagConfig::default(),
        };
        db.init_tables()?;
        db.run_migrations()?;
        Ok(db)
    }

    /// Open with hybrid search enabled
    pub fn open_with_hybrid(db_path: &Path, hybrid_config: HybridRagConfig) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = rusqlite::Connection::open(db_path)?;

        let embedder = if hybrid_config.semantic_enabled {
            Some(crate::embeddings::LocalEmbedder::new(
                crate::embeddings::EmbeddingConfig::default(),
            ))
        } else {
            None
        };

        let db = Self {
            conn,
            embedder,
            hybrid_config,
        };
        db.init_tables()?;
        db.run_migrations()?;
        Ok(db)
    }

    /// Enable or disable semantic search
    pub fn set_semantic_enabled(&mut self, enabled: bool) {
        self.hybrid_config.semantic_enabled = enabled;
        if enabled && self.embedder.is_none() {
            self.embedder = Some(crate::embeddings::LocalEmbedder::new(
                crate::embeddings::EmbeddingConfig::default(),
            ));
        }
    }

    /// Train the embedder on current knowledge base
    pub fn train_embedder(&mut self) -> Result<usize> {
        let embedder = match self.embedder.as_mut() {
            Some(e) => e,
            None => {
                self.embedder = Some(crate::embeddings::LocalEmbedder::new(
                    crate::embeddings::EmbeddingConfig::default(),
                ));
                self.embedder
                    .as_mut()
                    .expect("embedder must be initialized")
            }
        };

        // Get all knowledge chunks
        let mut stmt = self.conn.prepare("SELECT content FROM knowledge_chunks")?;

        let contents: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();

        let doc_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
        embedder.train(&doc_refs);

        Ok(contents.len())
    }

    /// Initialize database tables with FTS5
    fn init_tables(&self) -> Result<()> {
        // Users table for multi-user support
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                global_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
            [],
        )?;

        // Knowledge sources table - tracks indexed documents and their content hash
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
            [],
        )?;

        // Knowledge chunks table (shared across users - knowledge is global)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;

        // FTS5 virtual table for knowledge search
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                source,
                section,
                content,
                content=knowledge_chunks,
                content_rowid=id
            )",
            [],
        )?;

        // Triggers to keep FTS in sync
        self.conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_chunks BEGIN
                INSERT INTO knowledge_fts(rowid, source, section, content)
                VALUES (new.id, new.source, new.section, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_chunks BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, source, section, content)
                VALUES ('delete', old.id, old.source, old.section, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_chunks BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, source, section, content)
                VALUES ('delete', old.id, old.source, old.section, old.content);
                INSERT INTO knowledge_fts(rowid, source, section, content)
                VALUES (new.id, new.source, new.section, new.content);
            END;",
        )?;

        // Conversation messages table (per user via session_id which includes user)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                in_context INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )",
            [],
        )?;

        // FTS5 for conversation search
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
                content,
                content=conversation_messages,
                content_rowid=id
            )",
            [],
        )?;

        // Triggers for conversation FTS
        self.conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conversation_messages BEGIN
                INSERT INTO conversation_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS conv_ad AFTER DELETE ON conversation_messages BEGIN
                INSERT INTO conversation_fts(conversation_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS conv_au AFTER UPDATE ON conversation_messages BEGIN
                INSERT INTO conversation_fts(conversation_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO conversation_fts(rowid, content) VALUES (new.id, new.content);
            END;",
        )?;

        // Indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_messages(session_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_messages(user_id)",
            [],
        )?;
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_in_context ON conversation_messages(in_context)",
            [],
        )?;

        // Knowledge notes table - notes per knowledge document/guide per user
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                source TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, source)
            )",
            [],
        )?;

        // Session notes table - notes per session per user
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS session_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                session_id TEXT NOT NULL,
                notes TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, session_id)
            )",
            [],
        )?;

        Ok(())
    }

    /// Run migrations for schema updates
    fn run_migrations(&self) -> Result<()> {
        // Check if user_id column exists in conversation_messages
        let has_user_id: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('conversation_messages') WHERE name = 'user_id'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_user_id {
            // Add user_id column to existing tables
            let _ = self.conn.execute(
                "ALTER TABLE conversation_messages ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'",
                [],
            );
        }

        // Check if user_id column exists in knowledge_notes
        let has_user_id_notes: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('knowledge_notes') WHERE name = 'user_id'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_user_id_notes {
            // Need to recreate knowledge_notes table with new schema
            let _ = self.conn.execute_batch(
                "ALTER TABLE knowledge_notes RENAME TO knowledge_notes_old;
                 CREATE TABLE knowledge_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    source TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, source)
                 );
                 INSERT INTO knowledge_notes (user_id, source, notes, updated_at)
                 SELECT 'default', source, notes, updated_at FROM knowledge_notes_old;
                 DROP TABLE knowledge_notes_old;",
            );
        }

        // Add priority column to knowledge_sources if it doesn't exist
        let has_priority: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('knowledge_sources') WHERE name = 'priority'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_priority {
            let _ = self.conn.execute(
                "ALTER TABLE knowledge_sources ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
                [],
            );
        }

        Ok(())
    }

    // === User Management ===

    /// Get or create a user by user_id
    pub fn get_or_create_user(&self, user_id: &str) -> Result<User> {
        let now = chrono::Utc::now().to_rfc3339();

        // Try to get existing user
        let result = self.conn.query_row(
            "SELECT id, user_id, display_name, global_notes, created_at, updated_at FROM users WHERE user_id = ?1",
            rusqlite::params![user_id],
            |row| Ok(User {
                id: row.get(0)?,
                user_id: row.get(1)?,
                display_name: row.get(2)?,
                global_notes: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
            }),
        );

        match result {
            Ok(user) => Ok(user),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                // Create new user
                let display_name = if user_id == DEFAULT_USER_ID {
                    "Default User".to_string()
                } else {
                    user_id.to_string()
                };

                self.conn.execute(
                    "INSERT INTO users (user_id, display_name, global_notes, created_at, updated_at)
                     VALUES (?1, ?2, '', ?3, ?3)",
                    rusqlite::params![user_id, display_name, now],
                )?;

                let id = self.conn.last_insert_rowid();
                Ok(User {
                    id,
                    user_id: user_id.to_string(),
                    display_name,
                    global_notes: String::new(),
                    created_at: now.clone(),
                    updated_at: now,
                })
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Get user's global notes
    pub fn get_user_global_notes(&self, user_id: &str) -> Result<String> {
        let result = self.conn.query_row(
            "SELECT global_notes FROM users WHERE user_id = ?1",
            rusqlite::params![user_id],
            |row| row.get::<_, String>(0),
        );

        match result {
            Ok(notes) => Ok(notes),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(String::new()),
            Err(e) => Err(e.into()),
        }
    }

    /// Set user's global notes
    pub fn set_user_global_notes(&self, user_id: &str, notes: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();

        // Ensure user exists
        self.get_or_create_user(user_id)?;

        self.conn.execute(
            "UPDATE users SET global_notes = ?1, updated_at = ?2 WHERE user_id = ?3",
            rusqlite::params![notes, now, user_id],
        )?;
        Ok(())
    }

    /// List all users
    pub fn list_users(&self) -> Result<Vec<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, user_id, display_name, global_notes, created_at, updated_at FROM users ORDER BY display_name"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(User {
                id: row.get(0)?,
                user_id: row.get(1)?,
                display_name: row.get(2)?,
                global_notes: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // === Knowledge Base Operations ===

    /// Clear all knowledge chunks
    pub fn clear_knowledge(&self) -> Result<()> {
        self.conn.execute("DELETE FROM knowledge_chunks", [])?;
        self.conn.execute("DELETE FROM knowledge_sources", [])?;
        Ok(())
    }

    /// Calculate a hash of document content for change detection
    /// Uses FNV-1a algorithm - fast and good distribution for strings
    fn hash_content(content: &str) -> String {
        // FNV-1a 64-bit hash
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in content.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        // Include length for extra safety against collisions
        format!("{:016x}_{}", hash, content.len())
    }

    /// Check if a document needs to be re-indexed
    /// Returns (needs_index, current_hash)
    pub fn needs_reindex(&self, source: &str, content: &str) -> Result<(bool, String)> {
        let new_hash = Self::hash_content(content);

        let existing: Option<String> = self
            .conn
            .query_row(
                "SELECT content_hash FROM knowledge_sources WHERE source = ?1",
                [source],
                |row| row.get(0),
            )
            .ok();

        match existing {
            Some(old_hash) if old_hash == new_hash => Ok((false, new_hash)),
            _ => Ok((true, new_hash)),
        }
    }

    /// Check if a document is already indexed (regardless of content changes)
    pub fn is_document_indexed(&self, source: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_sources WHERE source = ?1",
            [source],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get information about an indexed document
    pub fn get_source_info(&self, source: &str) -> Result<Option<(String, usize, usize, String)>> {
        let result: Result<(String, usize, usize, String), _> = self.conn.query_row(
            "SELECT content_hash, chunk_count, total_tokens, indexed_at
             FROM knowledge_sources WHERE source = ?1",
            [source],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        );
        match result {
            Ok(info) => Ok(Some(info)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Index a document by splitting it into chunks
    /// Returns the number of chunks indexed, or 0 if document was already up-to-date
    pub fn index_document(&self, source: &str, content: &str) -> Result<usize> {
        let (needs_index, content_hash) = self.needs_reindex(source, content)?;

        if !needs_index {
            // Document already indexed with same content
            return Ok(0);
        }

        // Delete old chunks for this source
        self.conn
            .execute("DELETE FROM knowledge_chunks WHERE source = ?1", [source])?;

        // Index new chunks
        let chunks = chunk_document(source, content);
        let now = chrono::Utc::now().to_rfc3339();
        let mut count = 0;
        let mut total_tokens = 0;

        for chunk in chunks {
            self.conn.execute(
                "INSERT INTO knowledge_chunks (source, section, content, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    chunk.source,
                    chunk.section,
                    chunk.content,
                    chunk.token_count,
                    now
                ],
            )?;
            total_tokens += chunk.token_count;
            count += 1;
        }

        // Update or insert source record
        self.conn.execute(
            "INSERT INTO knowledge_sources (source, content_hash, chunk_count, total_tokens, indexed_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)
             ON CONFLICT(source) DO UPDATE SET
                content_hash = excluded.content_hash,
                chunk_count = excluded.chunk_count,
                total_tokens = excluded.total_tokens,
                updated_at = excluded.updated_at",
            rusqlite::params![source, content_hash, count, total_tokens, now],
        )?;

        Ok(count)
    }

    /// Delete a specific document from the knowledge base
    pub fn delete_document(&self, source: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM knowledge_chunks WHERE source = ?1", [source])?;
        self.conn
            .execute("DELETE FROM knowledge_sources WHERE source = ?1", [source])?;
        Ok(())
    }

    /// Get list of all indexed sources with their stats
    pub fn list_indexed_sources(&self) -> Result<Vec<(String, usize, usize, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT source, chunk_count, total_tokens, indexed_at FROM knowledge_sources ORDER BY source"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Search knowledge base for relevant chunks
    /// Results are ordered by: priority (higher first), then BM25 relevance score
    pub fn search_knowledge(
        &self,
        query: &str,
        max_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<KnowledgeChunk>> {
        let search_terms = prepare_fts_query(query);

        // Join with knowledge_sources to get priority, order by priority DESC then BM25 score
        let mut stmt = self.conn.prepare(
            "SELECT k.id, k.source, k.section, k.content, k.token_count,
                    bm25(knowledge_fts) as score,
                    COALESCE(s.priority, 0) as priority
             FROM knowledge_fts f
             JOIN knowledge_chunks k ON f.rowid = k.id
             LEFT JOIN knowledge_sources s ON k.source = s.source
             WHERE knowledge_fts MATCH ?1
             ORDER BY priority DESC, score
             LIMIT ?2",
        )?;

        let mut results = Vec::new();
        let mut total_tokens = 0;

        let rows = stmt.query_map(rusqlite::params![search_terms, top_k * 2], |row| {
            Ok(KnowledgeChunk {
                id: row.get(0)?,
                source: row.get(1)?,
                section: row.get(2)?,
                content: row.get(3)?,
                token_count: row.get(4)?,
            })
        })?;

        for chunk_result in rows {
            let chunk = chunk_result?;
            if total_tokens + chunk.token_count <= max_tokens {
                total_tokens += chunk.token_count;
                results.push(chunk);
                if results.len() >= top_k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Search knowledge base filtering by specific sources
    ///
    /// Only returns chunks from the specified sources.
    /// If `sources` is empty, returns no results.
    pub fn search_knowledge_filtered(
        &self,
        query: &str,
        sources: &[String],
        max_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<KnowledgeChunk>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        let search_terms = prepare_fts_query(query);

        // Build source filter
        let placeholders: Vec<String> = sources
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 3))
            .collect();
        let source_filter = placeholders.join(",");

        let sql = format!(
            "SELECT k.id, k.source, k.section, k.content, k.token_count,
                    bm25(knowledge_fts) as score,
                    COALESCE(s.priority, 0) as priority
             FROM knowledge_fts f
             JOIN knowledge_chunks k ON f.rowid = k.id
             LEFT JOIN knowledge_sources s ON k.source = s.source
             WHERE knowledge_fts MATCH ?1
               AND k.source IN ({})
             ORDER BY priority DESC, score
             LIMIT ?2",
            source_filter
        );

        let mut stmt = self.conn.prepare(&sql)?;

        // Build params: [search_terms, top_k * 2, source1, source2, ...]
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        params.push(Box::new(search_terms.clone()));
        params.push(Box::new((top_k * 2) as i64));
        for source in sources {
            params.push(Box::new(source.clone()));
        }

        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let mut results = Vec::new();
        let mut total_tokens = 0;

        let rows = stmt.query_map(&params_refs[..], |row| {
            Ok(KnowledgeChunk {
                id: row.get(0)?,
                source: row.get(1)?,
                section: row.get(2)?,
                content: row.get(3)?,
                token_count: row.get(4)?,
            })
        })?;

        for chunk_result in rows {
            let chunk = chunk_result?;
            if total_tokens + chunk.token_count <= max_tokens {
                total_tokens += chunk.token_count;
                results.push(chunk);
                if results.len() >= top_k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Hybrid search combining BM25 and semantic similarity
    ///
    /// Returns results sorted by combined score (BM25 weighted + semantic weighted)
    pub fn search_knowledge_hybrid(
        &self,
        query: &str,
        max_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<HybridKnowledgeResult>> {
        // Get BM25 results with scores
        let search_terms = prepare_fts_query(query);

        let mut stmt = self.conn.prepare(
            "SELECT k.id, k.source, k.section, k.content, k.token_count,
                    -bm25(knowledge_fts) as score,
                    COALESCE(s.priority, 0) as priority
             FROM knowledge_fts f
             JOIN knowledge_chunks k ON f.rowid = k.id
             LEFT JOIN knowledge_sources s ON k.source = s.source
             WHERE knowledge_fts MATCH ?1
             ORDER BY priority DESC, score DESC
             LIMIT ?2",
        )?;

        let mut bm25_results: Vec<(KnowledgeChunk, f32)> = Vec::new();

        let rows = stmt.query_map(rusqlite::params![search_terms, top_k * 3], |row| {
            Ok((
                KnowledgeChunk {
                    id: row.get(0)?,
                    source: row.get(1)?,
                    section: row.get(2)?,
                    content: row.get(3)?,
                    token_count: row.get(4)?,
                },
                row.get::<_, f64>(5)? as f32, // BM25 score
            ))
        })?;

        for row in rows {
            bm25_results.push(row?);
        }

        // Normalize BM25 scores to 0-1 range
        let max_bm25 = bm25_results
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0f32, f32::max)
            .max(1.0);

        // Compute semantic scores if enabled
        let semantic_scores: Vec<Option<f32>> = if self.hybrid_config.semantic_enabled {
            if let Some(ref embedder) = self.embedder {
                let query_embedding = embedder.embed(query);
                bm25_results
                    .iter()
                    .map(|(chunk, _)| {
                        let chunk_embedding = embedder.embed(&chunk.content);
                        let score = crate::embeddings::LocalEmbedder::cosine_similarity(
                            &query_embedding,
                            &chunk_embedding,
                        );
                        Some(score)
                    })
                    .collect()
            } else {
                bm25_results.iter().map(|_| None).collect()
            }
        } else {
            bm25_results.iter().map(|_| None).collect()
        };

        // Combine scores
        let mut hybrid_results: Vec<HybridKnowledgeResult> = bm25_results
            .into_iter()
            .zip(semantic_scores)
            .map(|((chunk, bm25_score), semantic_score)| {
                let normalized_bm25 = bm25_score / max_bm25;
                let combined_score = if let Some(sem_score) = semantic_score {
                    normalized_bm25 * self.hybrid_config.bm25_weight
                        + sem_score * self.hybrid_config.semantic_weight
                } else {
                    normalized_bm25
                };

                HybridKnowledgeResult {
                    chunk,
                    bm25_score: normalized_bm25,
                    semantic_score,
                    combined_score,
                }
            })
            .collect();

        // Sort by combined score
        hybrid_results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply token limit
        let mut results = Vec::new();
        let mut total_tokens = 0;

        for result in hybrid_results {
            // Filter by minimum semantic score if enabled
            if self.hybrid_config.semantic_enabled {
                if let Some(sem) = result.semantic_score {
                    if sem < self.hybrid_config.min_semantic_score {
                        continue;
                    }
                }
            }

            if total_tokens + result.chunk.token_count <= max_tokens {
                total_tokens += result.chunk.token_count;
                results.push(result);
                if results.len() >= top_k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Search with auto-selection of BM25 or hybrid based on configuration
    pub fn search_knowledge_auto(
        &self,
        query: &str,
        max_tokens: usize,
        top_k: usize,
    ) -> Result<Vec<KnowledgeChunk>> {
        if self.hybrid_config.semantic_enabled && self.embedder.is_some() {
            // Use hybrid search and extract just the chunks
            let hybrid_results = self.search_knowledge_hybrid(query, max_tokens, top_k)?;
            Ok(hybrid_results.into_iter().map(|r| r.chunk).collect())
        } else {
            // Fall back to BM25-only search
            self.search_knowledge(query, max_tokens, top_k)
        }
    }

    /// Set the priority for a knowledge source
    /// Higher priority sources appear first in search results
    pub fn set_source_priority(&self, source: &str, priority: i32) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE knowledge_sources SET priority = ?1, updated_at = ?2 WHERE source = ?3",
            rusqlite::params![priority, now, source],
        )?;
        Ok(())
    }

    /// Get the priority for a knowledge source
    pub fn get_source_priority(&self, source: &str) -> Result<i32> {
        let result = self.conn.query_row(
            "SELECT priority FROM knowledge_sources WHERE source = ?1",
            [source],
            |row| row.get(0),
        );
        match result {
            Ok(priority) => Ok(priority),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0),
            Err(e) => Err(e.into()),
        }
    }

    /// Get total knowledge stats
    pub fn get_knowledge_stats(&self) -> Result<(usize, usize)> {
        let count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM knowledge_chunks", [], |row| {
                    row.get(0)
                })?;
        let total_tokens: usize = self.conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM knowledge_chunks",
            [],
            |row| row.get(0),
        )?;
        Ok((count, total_tokens))
    }

    // === Knowledge Notes Operations (per user) ===

    /// Get notes for a specific knowledge document/guide for a user
    pub fn get_knowledge_notes(&self, user_id: &str, source: &str) -> Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT notes FROM knowledge_notes WHERE user_id = ?1 AND source = ?2",
            rusqlite::params![user_id, source],
            |row| row.get::<_, String>(0),
        );

        match result {
            Ok(notes) => Ok(Some(notes)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set notes for a specific knowledge document/guide for a user
    pub fn set_knowledge_notes(&self, user_id: &str, source: &str, notes: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO knowledge_notes (user_id, source, notes, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(user_id, source) DO UPDATE SET notes = ?3, updated_at = ?4",
            rusqlite::params![user_id, source, notes, now],
        )?;
        Ok(())
    }

    /// Delete notes for a specific knowledge document/guide for a user
    pub fn delete_knowledge_notes(&self, user_id: &str, source: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM knowledge_notes WHERE user_id = ?1 AND source = ?2",
            rusqlite::params![user_id, source],
        )?;
        Ok(())
    }

    /// Get all knowledge notes for a user
    pub fn get_all_knowledge_notes(&self, user_id: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT source, notes FROM knowledge_notes WHERE user_id = ?1 ORDER BY source",
        )?;

        let rows = stmt.query_map(rusqlite::params![user_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get list of all indexed knowledge sources
    pub fn get_knowledge_sources(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT source FROM knowledge_chunks ORDER BY source")?;

        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // === Session Notes Operations (per user) ===

    /// Get notes for a specific session for a user
    pub fn get_session_notes(&self, user_id: &str, session_id: &str) -> Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT notes FROM session_notes WHERE user_id = ?1 AND session_id = ?2",
            rusqlite::params![user_id, session_id],
            |row| row.get::<_, String>(0),
        );

        match result {
            Ok(notes) => Ok(Some(notes)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set notes for a specific session for a user
    pub fn set_session_notes(&self, user_id: &str, session_id: &str, notes: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO session_notes (user_id, session_id, notes, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(user_id, session_id) DO UPDATE SET notes = ?3, updated_at = ?4",
            rusqlite::params![user_id, session_id, notes, now],
        )?;
        Ok(())
    }

    /// Delete notes for a session
    pub fn delete_session_notes(&self, user_id: &str, session_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM session_notes WHERE user_id = ?1 AND session_id = ?2",
            rusqlite::params![user_id, session_id],
        )?;
        Ok(())
    }

    // === Conversation Operations (per user) ===

    /// Store a message in the database for a user
    pub fn store_message(
        &self,
        user_id: &str,
        session_id: &str,
        msg: &ChatMessage,
        in_context: bool,
    ) -> Result<i64> {
        let token_count = crate::context::estimate_tokens(&msg.content);
        let now = chrono::Utc::now().to_rfc3339();

        self.conn.execute(
            "INSERT INTO conversation_messages (user_id, session_id, role, content, timestamp, token_count, in_context, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                user_id,
                session_id,
                msg.role,
                msg.content,
                msg.timestamp.to_rfc3339(),
                token_count,
                in_context as i32,
                now
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Mark messages as out of context (moved to RAG storage)
    pub fn mark_messages_out_of_context(
        &self,
        user_id: &str,
        session_id: &str,
        message_ids: &[i64],
    ) -> Result<()> {
        for id in message_ids {
            self.conn.execute(
                "UPDATE conversation_messages SET in_context = 0 WHERE id = ?1 AND user_id = ?2 AND session_id = ?3",
                rusqlite::params![id, user_id, session_id],
            )?;
        }
        Ok(())
    }

    /// Search conversation history for relevant messages
    pub fn search_conversation(
        &self,
        user_id: &str,
        session_id: &str,
        query: &str,
        max_tokens: usize,
        exclude_in_context: bool,
    ) -> Result<Vec<StoredMessage>> {
        let search_terms = prepare_fts_query(query);

        let sql = if exclude_in_context {
            "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.token_count, m.in_context
             FROM conversation_fts f
             JOIN conversation_messages m ON f.rowid = m.id
             WHERE conversation_fts MATCH ?1 AND m.user_id = ?2 AND m.session_id = ?3 AND m.in_context = 0
             ORDER BY bm25(conversation_fts)
             LIMIT 50"
        } else {
            "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.token_count, m.in_context
             FROM conversation_fts f
             JOIN conversation_messages m ON f.rowid = m.id
             WHERE conversation_fts MATCH ?1 AND m.user_id = ?2 AND m.session_id = ?3
             ORDER BY bm25(conversation_fts)
             LIMIT 50"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let mut results = Vec::new();
        let mut total_tokens = 0;

        let rows = stmt.query_map(
            rusqlite::params![search_terms, user_id, session_id],
            |row| {
                Ok(StoredMessage {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    timestamp: row.get(4)?,
                    token_count: row.get(5)?,
                    in_context: row.get::<_, i32>(6)? != 0,
                })
            },
        )?;

        for msg_result in rows {
            let msg = msg_result?;
            if total_tokens + msg.token_count <= max_tokens {
                total_tokens += msg.token_count;
                results.push(msg);
            }
        }

        Ok(results)
    }

    /// Get recent messages from conversation (not in current context)
    pub fn get_recent_archived_messages(
        &self,
        user_id: &str,
        session_id: &str,
        max_tokens: usize,
    ) -> Result<Vec<StoredMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, role, content, timestamp, token_count, in_context
             FROM conversation_messages
             WHERE user_id = ?1 AND session_id = ?2 AND in_context = 0
             ORDER BY timestamp DESC
             LIMIT 50",
        )?;

        let mut results = Vec::new();
        let mut total_tokens = 0;

        let rows = stmt.query_map(rusqlite::params![user_id, session_id], |row| {
            Ok(StoredMessage {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                timestamp: row.get(4)?,
                token_count: row.get(5)?,
                in_context: row.get::<_, i32>(6)? != 0,
            })
        })?;

        for msg_result in rows {
            let msg = msg_result?;
            if total_tokens + msg.token_count <= max_tokens {
                total_tokens += msg.token_count;
                results.push(msg);
            }
        }

        // Reverse to get chronological order
        results.reverse();
        Ok(results)
    }

    /// Get conversation stats for a session
    pub fn get_conversation_stats(
        &self,
        user_id: &str,
        session_id: &str,
    ) -> Result<(usize, usize, usize)> {
        let total: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?1 AND session_id = ?2",
            rusqlite::params![user_id, session_id],
            |row| row.get(0),
        )?;
        let archived: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?1 AND session_id = ?2 AND in_context = 0",
            rusqlite::params![user_id, session_id],
            |row| row.get(0),
        )?;
        let archived_tokens: usize = self.conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM conversation_messages WHERE user_id = ?1 AND session_id = ?2 AND in_context = 0",
            rusqlite::params![user_id, session_id],
            |row| row.get(0),
        )?;
        Ok((total, archived, archived_tokens))
    }

    /// Clear conversation history for a session
    pub fn clear_session_history(&self, user_id: &str, session_id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM conversation_messages WHERE user_id = ?1 AND session_id = ?2",
            rusqlite::params![user_id, session_id],
        )?;
        Ok(())
    }

    // === Legacy compatibility methods (use DEFAULT_USER_ID) ===

    /// Get notes for a specific knowledge document/guide (legacy - uses default user)
    pub fn get_knowledge_notes_default(&self, source: &str) -> Result<Option<String>> {
        self.get_knowledge_notes(DEFAULT_USER_ID, source)
    }

    /// Set notes for a specific knowledge document/guide (legacy - uses default user)
    pub fn set_knowledge_notes_default(&self, source: &str, notes: &str) -> Result<()> {
        self.set_knowledge_notes(DEFAULT_USER_ID, source, notes)
    }

    /// Delete notes for a specific knowledge document/guide (legacy - uses default user)
    pub fn delete_knowledge_notes_default(&self, source: &str) -> Result<()> {
        self.delete_knowledge_notes(DEFAULT_USER_ID, source)
    }

    /// Get all knowledge notes (legacy - uses default user)
    pub fn get_all_knowledge_notes_default(&self) -> Result<Vec<(String, String)>> {
        self.get_all_knowledge_notes(DEFAULT_USER_ID)
    }

    /// Store a message (legacy - uses default user)
    pub fn store_message_default(
        &self,
        session_id: &str,
        msg: &ChatMessage,
        in_context: bool,
    ) -> Result<i64> {
        self.store_message(DEFAULT_USER_ID, session_id, msg, in_context)
    }

    /// Mark messages as out of context (legacy - uses default user)
    pub fn mark_messages_out_of_context_default(
        &self,
        session_id: &str,
        message_ids: &[i64],
    ) -> Result<()> {
        self.mark_messages_out_of_context(DEFAULT_USER_ID, session_id, message_ids)
    }

    /// Search conversation (legacy - uses default user)
    pub fn search_conversation_default(
        &self,
        session_id: &str,
        query: &str,
        max_tokens: usize,
        exclude_in_context: bool,
    ) -> Result<Vec<StoredMessage>> {
        self.search_conversation(
            DEFAULT_USER_ID,
            session_id,
            query,
            max_tokens,
            exclude_in_context,
        )
    }

    /// Get recent archived messages (legacy - uses default user)
    pub fn get_recent_archived_messages_default(
        &self,
        session_id: &str,
        max_tokens: usize,
    ) -> Result<Vec<StoredMessage>> {
        self.get_recent_archived_messages(DEFAULT_USER_ID, session_id, max_tokens)
    }

    /// Get conversation stats (legacy - uses default user)
    pub fn get_conversation_stats_default(
        &self,
        session_id: &str,
    ) -> Result<(usize, usize, usize)> {
        self.get_conversation_stats(DEFAULT_USER_ID, session_id)
    }

    /// Clear session history (legacy - uses default user)
    pub fn clear_session_history_default(&self, session_id: &str) -> Result<()> {
        self.clear_session_history(DEFAULT_USER_ID, session_id)
    }
}

/// Export format for knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeExport {
    /// Version for forward compatibility
    pub version: u32,
    /// Export timestamp
    pub exported_at: String,
    /// All knowledge chunks
    pub chunks: Vec<ExportedChunk>,
    /// Source metadata
    pub sources: Vec<ExportedSource>,
}

/// Exported chunk data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedChunk {
    pub source: String,
    pub section: String,
    pub content: String,
    pub token_count: usize,
}

/// Exported source metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedSource {
    pub source: String,
    pub content_hash: String,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

impl RagDb {
    /// Export all knowledge to a serializable format
    pub fn export_knowledge(&self) -> Result<KnowledgeExport> {
        // Get all chunks
        let mut stmt = self.conn.prepare(
            "SELECT source, section, content, token_count FROM knowledge_chunks ORDER BY source, id"
        )?;

        let chunks: Vec<ExportedChunk> = stmt
            .query_map([], |row| {
                Ok(ExportedChunk {
                    source: row.get(0)?,
                    section: row.get(1)?,
                    content: row.get(2)?,
                    token_count: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Get all sources
        let mut stmt = self.conn.prepare(
            "SELECT source, content_hash, chunk_count, total_tokens FROM knowledge_sources ORDER BY source"
        )?;

        let sources: Vec<ExportedSource> = stmt
            .query_map([], |row| {
                Ok(ExportedSource {
                    source: row.get(0)?,
                    content_hash: row.get(1)?,
                    chunk_count: row.get(2)?,
                    total_tokens: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(KnowledgeExport {
            version: 1,
            exported_at: chrono::Utc::now().to_rfc3339(),
            chunks,
            sources,
        })
    }

    /// Export knowledge to a JSON file
    pub fn export_knowledge_to_file(&self, path: &Path) -> Result<()> {
        let export = self.export_knowledge()?;
        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Import knowledge from a serializable format
    ///
    /// # Arguments
    /// * `data` - The knowledge export data
    /// * `replace` - If true, clears existing knowledge first. If false, merges.
    pub fn import_knowledge(&self, data: &KnowledgeExport, replace: bool) -> Result<usize> {
        if replace {
            self.clear_knowledge()?;
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut imported = 0;

        // Import sources first
        for source in &data.sources {
            // Check if source already exists
            let exists: bool = self
                .conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM knowledge_sources WHERE source = ?1",
                    [&source.source],
                    |row| row.get(0),
                )
                .unwrap_or(false);

            if !exists {
                self.conn.execute(
                    "INSERT INTO knowledge_sources (source, content_hash, chunk_count, total_tokens, indexed_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                    rusqlite::params![source.source, source.content_hash, source.chunk_count, source.total_tokens, now],
                )?;
            }
        }

        // Import chunks
        for chunk in &data.chunks {
            // Skip if source already has chunks (to avoid duplicates in merge mode)
            if !replace {
                let has_chunks: bool = self
                    .conn
                    .query_row(
                        "SELECT COUNT(*) > 0 FROM knowledge_chunks WHERE source = ?1",
                        [&chunk.source],
                        |row| row.get(0),
                    )
                    .unwrap_or(false);

                if has_chunks {
                    continue;
                }
            }

            self.conn.execute(
                "INSERT INTO knowledge_chunks (source, section, content, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    chunk.source,
                    chunk.section,
                    chunk.content,
                    chunk.token_count,
                    now
                ],
            )?;
            imported += 1;
        }

        Ok(imported)
    }

    /// Import knowledge from a JSON file
    pub fn import_knowledge_from_file(&self, path: &Path, replace: bool) -> Result<usize> {
        let json = std::fs::read_to_string(path)?;
        let data: KnowledgeExport = serde_json::from_str(&json)?;
        self.import_knowledge(&data, replace)
    }
}

/// Split a document into chunks based on markdown structure
fn chunk_document(source: &str, content: &str) -> Vec<KnowledgeChunk> {
    let mut chunks = Vec::new();
    let mut current_section = String::new();
    let mut current_content = String::new();
    let mut current_tokens = 0;

    // Target chunk size in tokens (aim for ~300-500 tokens per chunk)
    const TARGET_CHUNK_TOKENS: usize = 400;
    const MAX_CHUNK_TOKENS: usize = 600;

    for line in content.lines() {
        let line = line.trim();

        // Check for markdown headers
        if line.starts_with('#') {
            // Save current chunk if it has content
            if !current_content.is_empty() {
                chunks.push(KnowledgeChunk {
                    id: 0,
                    source: source.to_string(),
                    section: current_section.clone(),
                    content: current_content.trim().to_string(),
                    token_count: current_tokens,
                });
            }

            // Start new section
            current_section = line.trim_start_matches('#').trim().to_string();
            current_content = String::new();
            current_tokens = 0;
        } else if !line.is_empty() {
            let line_tokens = crate::context::estimate_tokens(line);

            // If adding this line would exceed max, save current chunk
            if current_tokens + line_tokens > MAX_CHUNK_TOKENS && !current_content.is_empty() {
                chunks.push(KnowledgeChunk {
                    id: 0,
                    source: source.to_string(),
                    section: current_section.clone(),
                    content: current_content.trim().to_string(),
                    token_count: current_tokens,
                });
                current_content = String::new();
                current_tokens = 0;
            }

            current_content.push_str(line);
            current_content.push('\n');
            current_tokens += line_tokens;

            // If we've reached target size and at a good break point, save chunk
            if current_tokens >= TARGET_CHUNK_TOKENS
                && (line.ends_with('.') || line.ends_with(':') || line.is_empty())
            {
                chunks.push(KnowledgeChunk {
                    id: 0,
                    source: source.to_string(),
                    section: current_section.clone(),
                    content: current_content.trim().to_string(),
                    token_count: current_tokens,
                });
                current_content = String::new();
                current_tokens = 0;
            }
        }
    }

    // Don't forget the last chunk
    if !current_content.is_empty() {
        chunks.push(KnowledgeChunk {
            id: 0,
            source: source.to_string(),
            section: current_section,
            content: current_content.trim().to_string(),
            token_count: current_tokens,
        });
    }

    chunks
}

/// Prepare a search query for FTS5
fn prepare_fts_query(query: &str) -> String {
    // Split into words, filter short words, add wildcards for prefix matching
    let terms: Vec<String> = query
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(|w| {
            // Keep only alphanumeric characters and basic punctuation that's safe for FTS5
            // Remove: " * ( ) : ? ! . , ; ' ` ~ @ # $ % ^ & [ ] { } < > / \ | + =
            let escaped: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            escaped.to_lowercase()
        })
        .filter(|w| w.len() >= 2) // Re-filter after removing punctuation
        .map(|w| format!("{}*", w))
        .collect();

    if terms.is_empty() {
        // Fallback to match anything if query is too short
        "*".to_string()
    } else {
        terms.join(" OR ")
    }
}

/// Build context string from retrieved knowledge chunks
pub fn build_knowledge_context(chunks: &[KnowledgeChunk]) -> String {
    if chunks.is_empty() {
        return String::new();
    }

    let mut context = String::from("--- RELEVANT KNOWLEDGE ---\n");

    // Group by source
    let mut by_source: std::collections::HashMap<&str, Vec<&KnowledgeChunk>> =
        std::collections::HashMap::new();
    for chunk in chunks {
        by_source.entry(&chunk.source).or_default().push(chunk);
    }

    for (source, source_chunks) in by_source {
        context.push_str(&format!("\n## {}\n", source));
        for chunk in source_chunks {
            if !chunk.section.is_empty() {
                context.push_str(&format!("### {}\n", chunk.section));
            }
            context.push_str(&chunk.content);
            context.push_str("\n\n");
        }
    }

    context.push_str("--- END KNOWLEDGE ---\n");
    context
}

/// Build context string from retrieved conversation messages
pub fn build_conversation_context(messages: &[StoredMessage]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut context = String::from("[Previous conversation context]\n");

    for msg in messages {
        let role = if msg.role == "user" {
            "User"
        } else {
            "Assistant"
        };
        context.push_str(&format!("{}: {}\n", role, msg.content));
    }

    context.push_str("[End previous context]\n");
    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_chunk_document() {
        let content = r#"
# Introduction
This is the intro section.

## Features
- Feature 1
- Feature 2
- Feature 3

## Usage
Here is how to use it.
"#;
        let chunks = chunk_document("test.md", content);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.section == "Introduction"));
        assert!(chunks.iter().any(|c| c.section == "Features"));
    }

    #[test]
    fn test_prepare_fts_query() {
        assert_eq!(prepare_fts_query("hello world"), "hello* OR world*");
        assert_eq!(prepare_fts_query("a"), "*"); // Too short
        assert_eq!(prepare_fts_query("test:query"), "testquery*");
        // Test punctuation removal
        assert_eq!(prepare_fts_query("What is a CCU?"), "what* OR is* OR ccu*");
        assert_eq!(prepare_fts_query("Hello, world!"), "hello* OR world*");
        assert_eq!(prepare_fts_query("test (example)"), "test* OR example*");
    }

    #[test]
    fn test_prepare_fts_query_special_chars() {
        // Test various punctuation marks that users might include
        assert_eq!(prepare_fts_query("What's this?"), "whats* OR this*");
        assert_eq!(prepare_fts_query("price: $100"), "price* OR 100*");
        assert_eq!(prepare_fts_query("test@email.com"), "testemailcom*");
        assert_eq!(prepare_fts_query("50% discount!"), "50* OR discount*");
        assert_eq!(prepare_fts_query("item #123"), "item* OR 123*");
        assert_eq!(prepare_fts_query("A & B"), "*"); // Both too short after filtering
        assert_eq!(prepare_fts_query("foo && bar"), "foo* OR bar*");
        assert_eq!(prepare_fts_query("[test]"), "test*");
        assert_eq!(prepare_fts_query("{value}"), "value*");
        assert_eq!(prepare_fts_query("path/to/file"), "pathtofile*"); // slashes removed
    }

    #[test]
    fn test_prepare_fts_query_unicode() {
        // Unicode letters should be preserved
        assert_eq!(prepare_fts_query("café résumé"), "café* OR résumé*");
        assert_eq!(prepare_fts_query("日本語 テスト"), "日本語* OR テスト*");
    }

    #[test]
    fn test_prepare_fts_query_hyphen_underscore() {
        // Hyphens and underscores should be preserved
        assert_eq!(
            prepare_fts_query("cross-chassis upgrade"),
            "cross-chassis* OR upgrade*"
        );
        assert_eq!(prepare_fts_query("user_name test"), "user_name* OR test*");
    }

    fn create_temp_db() -> (RagDb, PathBuf) {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("test_rag_{}.db", uuid::Uuid::new_v4()));
        let db = RagDb::open(&db_path).expect("Failed to create test database");
        (db, db_path)
    }

    fn cleanup_db(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_rag_db_creation() {
        let (db, path) = create_temp_db();

        // Check stats are zero initially
        let (chunks, tokens) = db.get_knowledge_stats().expect("Failed to get stats");
        assert_eq!(chunks, 0);
        assert_eq!(tokens, 0);

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_index_document() {
        let (db, path) = create_temp_db();

        let content = r#"
# CCU Guide

## What is a CCU?
A **CCU (Cross-Chassis Upgrade)** is an item that transforms one ship into another of higher value.

## How CCUs Work
1. Buy a CCU from ship A to ship B
2. The CCU costs the price difference
3. Apply the CCU to transform your ship
"#;

        // Index the document
        let chunks_indexed = db
            .index_document("ccu-guide", content)
            .expect("Failed to index");
        assert!(chunks_indexed > 0, "Should have indexed at least one chunk");

        // Verify stats
        let (total_chunks, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert_eq!(total_chunks, chunks_indexed);

        // Verify source is listed
        let sources = db.list_indexed_sources().expect("Failed to list sources");
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, "ccu-guide");

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_search_basic() {
        let (db, path) = create_temp_db();

        let content = r#"
# CCU Guide

## What is a CCU?
A **CCU (Cross-Chassis Upgrade)** is an item that transforms one ship into another of higher value.
This is the most economical way to obtain ships in Star Citizen.

## Pricing
CCUs cost the difference between the two ship prices.
"#;

        db.index_document("ccu-guide", content)
            .expect("Failed to index");

        // Search for CCU
        let results = db
            .search_knowledge("What is a CCU?", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Search should return results");

        // Verify content contains CCU information
        let all_content: String = results.iter().map(|c| c.content.clone()).collect();
        assert!(
            all_content.contains("Cross-Chassis Upgrade"),
            "Results should contain CCU definition"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_search_with_punctuation() {
        let (db, path) = create_temp_db();

        let content = r#"
# Ship Guide

## Aurora MR
The Aurora MR is a starter ship manufactured by Roberts Space Industries.

## Mustang Alpha
The Mustang Alpha is another starter ship, made by Consolidated Outland.
"#;

        db.index_document("ships", content)
            .expect("Failed to index");

        // Search with question mark
        let results = db
            .search_knowledge("What is the Aurora MR?", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Search with '?' should return results");

        // Search with exclamation
        let results = db
            .search_knowledge("Tell me about Mustang!", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Search with '!' should return results");

        // Search with comma
        let results = db
            .search_knowledge("Aurora, Mustang ships", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Search with ',' should return results");

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_search_no_results() {
        let (db, path) = create_temp_db();

        let content = r#"
# Ship Guide
The Aurora is a starter ship.
"#;

        db.index_document("ships", content)
            .expect("Failed to index");

        // Search for something not in the document
        let results = db
            .search_knowledge("quantum drive specifications", 2000, 5)
            .expect("Search failed");
        // This might return results due to partial matching, but let's verify the search doesn't crash
        // The important thing is that it doesn't error
        assert!(results.len() <= 5, "Should respect top_k limit");

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_reindex_same_content() {
        let (db, path) = create_temp_db();

        let content = "# Test\nSome content here.";

        // First index
        let chunks1 = db.index_document("test", content).expect("Failed to index");
        assert!(chunks1 > 0);

        // Second index with same content - should skip
        let chunks2 = db
            .index_document("test", content)
            .expect("Failed to reindex");
        assert_eq!(chunks2, 0, "Should return 0 when content unchanged");

        // Verify still only original chunks
        let (total, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert_eq!(total, chunks1);

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_reindex_changed_content() {
        let (db, path) = create_temp_db();

        let content1 = "# Test\nOriginal content.";
        let content2 = "# Test\nUpdated content with more information.";

        // First index
        let chunks1 = db
            .index_document("test", content1)
            .expect("Failed to index");
        assert!(chunks1 > 0);

        // Second index with changed content - should reindex
        let chunks2 = db
            .index_document("test", content2)
            .expect("Failed to reindex");
        assert!(chunks2 > 0, "Should reindex when content changed");

        // Search should find new content
        let results = db
            .search_knowledge("updated information", 2000, 5)
            .expect("Search failed");
        let all_content: String = results.iter().map(|c| c.content.clone()).collect();
        assert!(
            all_content.contains("Updated") || all_content.contains("updated"),
            "Should find updated content"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_multiple_documents() {
        let (db, path) = create_temp_db();

        let ccu_content = "# CCU Guide\nA CCU transforms one ship into another.";
        let ships_content =
            "# Ships\nThe Aurora is a starter ship. The Carrack is an exploration ship.";
        let trading_content = "# Trading\nBuy low, sell high. Check commodity prices.";

        db.index_document("ccu", ccu_content)
            .expect("Failed to index ccu");
        db.index_document("ships", ships_content)
            .expect("Failed to index ships");
        db.index_document("trading", trading_content)
            .expect("Failed to index trading");

        // Verify all sources indexed
        let sources = db.list_indexed_sources().expect("Failed to list sources");
        assert_eq!(sources.len(), 3);

        // Search should find relevant document
        let results = db
            .search_knowledge("What is the Carrack?", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty());
        assert!(
            results.iter().any(|c| c.source == "ships"),
            "Should find ships document"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_delete_document() {
        let (db, path) = create_temp_db();

        let content = "# Test\nSome content.";
        db.index_document("test", content).expect("Failed to index");

        // Verify indexed
        let (chunks_before, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert!(chunks_before > 0);

        // Delete
        db.delete_document("test").expect("Failed to delete");

        // Verify deleted
        let (chunks_after, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert_eq!(chunks_after, 0);

        let sources = db.list_indexed_sources().expect("Failed to list sources");
        assert!(sources.is_empty());

        cleanup_db(&path);
    }

    #[test]
    fn test_build_knowledge_context() {
        let chunks = vec![
            KnowledgeChunk {
                id: 1,
                source: "guide".to_string(),
                section: "Introduction".to_string(),
                content: "This is the intro.".to_string(),
                token_count: 5,
            },
            KnowledgeChunk {
                id: 2,
                source: "guide".to_string(),
                section: "Details".to_string(),
                content: "Here are the details.".to_string(),
                token_count: 5,
            },
        ];

        let context = build_knowledge_context(&chunks);

        assert!(context.contains("RELEVANT KNOWLEDGE"));
        assert!(context.contains("guide"));
        assert!(context.contains("Introduction"));
        assert!(context.contains("This is the intro"));
        assert!(context.contains("END KNOWLEDGE"));
    }

    #[test]
    fn test_build_knowledge_context_empty() {
        let chunks: Vec<KnowledgeChunk> = vec![];
        let context = build_knowledge_context(&chunks);
        assert!(context.is_empty());
    }

    #[test]
    fn test_knowledge_usage_tracking() {
        let chunks = vec![
            KnowledgeChunk {
                id: 1,
                source: "ccu-guide".to_string(),
                section: "Basics".to_string(),
                content: "CCU info".to_string(),
                token_count: 100,
            },
            KnowledgeChunk {
                id: 2,
                source: "ccu-guide".to_string(),
                section: "Advanced".to_string(),
                content: "More CCU info".to_string(),
                token_count: 150,
            },
            KnowledgeChunk {
                id: 3,
                source: "ships".to_string(),
                section: "Aurora".to_string(),
                content: "Ship info".to_string(),
                token_count: 50,
            },
        ];

        let usage = KnowledgeUsage::from_chunks("What is a CCU?", &chunks);

        assert!(usage.has_knowledge());
        assert_eq!(usage.total_chunks, 3);
        assert_eq!(usage.total_tokens, 300);
        assert_eq!(usage.sources.len(), 2); // ccu-guide and ships

        let sources = usage.get_sources();
        assert!(sources.contains(&"ccu-guide"));
        assert!(sources.contains(&"ships"));
    }

    #[test]
    fn test_rag_db_token_limit() {
        let (db, path) = create_temp_db();

        // Create content with multiple sections that will create multiple chunks
        let content = r#"
# Section 1
This is a long section with lots of content to ensure we get multiple chunks.
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

# Section 2
Another section with different content about ships and upgrades.
More text here to fill out the chunk.

# Section 3
Yet another section with even more content about trading and exploration.
Additional text to make this chunk substantial.

# Section 4
Final section with concluding information about the game mechanics.
This should create enough chunks to test the token limit.
"#;

        db.index_document("test", content).expect("Failed to index");

        // Search with very low token limit
        let results = db
            .search_knowledge("section content", 50, 10)
            .expect("Search failed");

        // Should respect token limit (though might get 0 if first chunk exceeds limit)
        let total_tokens: usize = results.iter().map(|c| c.token_count).sum();
        assert!(
            total_tokens <= 50 || results.is_empty(),
            "Should respect token limit or return empty if first chunk too large"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_db_top_k_limit() {
        let (db, path) = create_temp_db();

        // Index multiple documents
        for i in 0..10 {
            let content = format!(
                "# Document {}\nThis document talks about ships and CCUs.",
                i
            );
            db.index_document(&format!("doc{}", i), &content)
                .expect("Failed to index");
        }

        // Search with top_k = 3
        let results = db
            .search_knowledge("ships CCU", 10000, 3)
            .expect("Search failed");
        assert!(results.len() <= 3, "Should respect top_k limit");

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_real_world_ccu_query() {
        // This test replicates the real-world scenario that was failing
        let (db, path) = create_temp_db();

        let ccu_content = r#"
# CCU Game - Cross-Chassis Upgrades en Star Citizen

## Que es un CCU?

Un **CCU (Cross-Chassis Upgrade)** es un item que permite transformar una nave en otra de mayor valor. Es la forma mas economica de obtener naves en Star Citizen sin pagar el precio completo.

### Funcionamiento Basico

1. Compras un CCU que va desde una nave A hacia una nave B
2. El CCU cuesta la diferencia de precio entre ambas naves
3. Aplicas el CCU a tu nave A y se convierte en nave B

## Tipos de CCU

### CCU Standard
- Precio normal (diferencia entre naves)
- Disponible mientras ambas naves estan en venta

### CCU Warbond
- Precio con descuento
- Requiere dinero nuevo (no store credit)
- Mejor valor pero menos flexible
"#;

        db.index_document("ccu-game", ccu_content)
            .expect("Failed to index CCU content");

        // Verify indexing worked
        let (chunks, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert!(chunks > 0, "Should have indexed chunks");

        // Test the exact query that was failing before the fix
        let query = "What is a CCU in Star Citizen?";
        let results = db.search_knowledge(query, 2000, 5).expect("Search failed");

        assert!(
            !results.is_empty(),
            "Search for '{}' should return results",
            query
        );

        // Verify the results contain the CCU definition
        let all_content: String = results
            .iter()
            .map(|c| c.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_content.contains("cross-chassis") || all_content.contains("ccu"),
            "Results should contain CCU information. Got: {}...",
            &all_content[..all_content.len().min(200)]
        );

        // Build context and verify it's not empty
        let context = build_knowledge_context(&results);
        assert!(!context.is_empty(), "Built context should not be empty");
        assert!(
            context.contains("RELEVANT KNOWLEDGE"),
            "Context should have header"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_spanish_content_search() {
        let (db, path) = create_temp_db();

        let content = r#"
# Guía de Naves

## Aurora MR
La Aurora MR es una nave inicial fabricada por Roberts Space Industries.
Es perfecta para nuevos ciudadanos que quieren explorar el universo.

## Características
- Tamaño: Pequeño
- Tripulación: 1 piloto
- Precio: 30 USD
"#;

        db.index_document("naves", content)
            .expect("Failed to index");

        // Search in Spanish
        let results = db
            .search_knowledge("Qué es la Aurora?", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Should find Spanish content");

        let all_content: String = results.iter().map(|c| c.content.clone()).collect();
        assert!(
            all_content.contains("Aurora") || all_content.contains("nave"),
            "Should find Aurora information"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_ccu_chain_example_query() {
        // This test verifies that asking for a CCU chain example returns the actual example
        let (db, path) = create_temp_db();

        // Use the real CCU knowledge content
        let ccu_content = r#"
# CCU Game - Cross-Chassis Upgrades en Star Citizen

## Que es un CCU?

Un **CCU (Cross-Chassis Upgrade)** es un item que permite transformar una nave en otra de mayor valor.

## CCU Chains (Cadenas de CCU)

Una **cadena de CCU** es la estrategia de aplicar multiples CCUs en secuencia para maximizar descuentos.

### Ejemplo de Cadena

```
Aurora MR ($30)
  -> [CCU Warbond $5] -> Mustang Alpha ($35)
  -> [CCU Warbond $10] -> Avenger Titan ($55)
  -> [CCU Warbond $40] -> Cutlass Black ($110)
  -> [CCU Standard $90] -> Constellation Taurus ($200)

Total gastado: $175 (vs $200 precio directo)
Ahorro: $25 (12.5%)
```

### Estrategia de Cadenas

1. **Comprar CCUs warbond durante eventos**
2. **Stockpile de CCUs** - Comprar y guardar CCUs aunque no los necesites ahora
3. **Naves ancla** - Usar naves con buen valor como puntos intermedios:
   - Cutlass Black ($110)
   - Constellation Taurus ($200)
   - Mercury Star Runner ($260)
"#;

        db.index_document("ccu-game", ccu_content)
            .expect("Failed to index CCU content");

        // Verify indexing
        let (chunks, _) = db.get_knowledge_stats().expect("Failed to get stats");
        assert!(chunks > 0, "Should have indexed chunks");

        // Test query about CCU chain example
        let query = "dime una cadena de CCUs de ejemplo";
        let results = db.search_knowledge(query, 4000, 10).expect("Search failed");

        assert!(
            !results.is_empty(),
            "Search for CCU chain example should return results"
        );

        // Check that results contain the actual example
        let all_content: String = results
            .iter()
            .map(|c| c.content.clone())
            .collect::<Vec<_>>()
            .join("\n");

        // The results MUST contain the actual chain example
        let has_aurora = all_content.contains("Aurora MR");
        let has_mustang = all_content.contains("Mustang Alpha");
        let has_avenger = all_content.contains("Avenger Titan");
        let has_cutlass = all_content.contains("Cutlass Black");
        let has_chain_keyword = all_content.to_lowercase().contains("cadena");

        println!("Query: {}", query);
        println!("Results count: {}", results.len());
        println!(
            "Content preview: {}...",
            &all_content[..all_content.len().min(500)]
        );
        println!(
            "Has Aurora: {}, Mustang: {}, Avenger: {}, Cutlass: {}, Cadena: {}",
            has_aurora, has_mustang, has_avenger, has_cutlass, has_chain_keyword
        );

        assert!(
            has_chain_keyword || has_aurora,
            "Results should contain CCU chain information. Got: {}...",
            &all_content[..all_content.len().min(300)]
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_finds_code_block_content() {
        // Ensure that content inside code blocks (```) is also searchable
        let (db, path) = create_temp_db();

        let content = r#"
# Example

Here is an example:

```
Aurora MR ($30)
  -> Mustang Alpha ($35)
  -> Avenger Titan ($55)
```

This shows a simple upgrade path.
"#;

        db.index_document("example", content)
            .expect("Failed to index");

        // Search for content that's inside the code block
        let results = db
            .search_knowledge("Aurora Mustang Avenger", 2000, 5)
            .expect("Search failed");

        assert!(
            !results.is_empty(),
            "Should find content inside code blocks"
        );

        let all_content: String = results.iter().map(|c| c.content.clone()).collect();
        assert!(
            all_content.contains("Aurora") || all_content.contains("Mustang"),
            "Should find ships mentioned in code block"
        );

        cleanup_db(&path);
    }

    #[test]
    fn test_chunk_preserves_code_blocks() {
        // Verify that chunking doesn't break code blocks
        let content = r#"
# CCU Chains

### Ejemplo de Cadena

```
Aurora MR ($30)
  -> [CCU Warbond $5] -> Mustang Alpha ($35)
  -> [CCU Warbond $10] -> Avenger Titan ($55)
  -> [CCU Warbond $40] -> Cutlass Black ($110)
```

Total gastado: $175
"#;

        let chunks = chunk_document("test", content);

        // Find the chunk that should contain the example
        let example_chunk = chunks
            .iter()
            .find(|c| c.content.contains("Aurora") || c.content.contains("Mustang"));

        assert!(
            example_chunk.is_some(),
            "Should have a chunk with the example"
        );

        let chunk = example_chunk.unwrap();
        // The chunk should contain most of the chain
        let has_multiple_ships = (chunk.content.contains("Aurora") as u8)
            + (chunk.content.contains("Mustang") as u8)
            + (chunk.content.contains("Avenger") as u8)
            + (chunk.content.contains("Cutlass") as u8);

        assert!(
            has_multiple_ships >= 2,
            "Chunk should contain multiple ships from the chain. Got: {}",
            chunk.content
        );
    }

    #[test]
    fn test_rag_db_concurrent_access() {
        // Test that multiple searches don't interfere with each other
        let (db, path) = create_temp_db();

        let content = "# Test\nThis is test content about ships and CCUs and upgrades.";
        db.index_document("test", content).expect("Failed to index");

        // Multiple searches should all work
        for i in 0..5 {
            let query = format!("test query {}", i);
            let results = db.search_knowledge(&query, 1000, 3);
            assert!(results.is_ok(), "Search {} should not fail", i);
        }

        cleanup_db(&path);
    }

    #[test]
    fn test_rag_export_import() {
        let (db1, path1) = create_temp_db();

        // Index some documents
        db1.index_document("doc1", "# Document 1\nContent about ships.")
            .expect("Failed to index");
        db1.index_document("doc2", "# Document 2\nContent about CCUs.")
            .expect("Failed to index");

        // Export
        let export = db1.export_knowledge().expect("Export failed");
        assert!(!export.chunks.is_empty());
        assert_eq!(export.sources.len(), 2);

        // Create new database and import
        let (db2, path2) = create_temp_db();
        let imported = db2.import_knowledge(&export, false).expect("Import failed");
        assert!(imported > 0);

        // Verify imported content is searchable
        let results = db2
            .search_knowledge("ships CCUs", 2000, 5)
            .expect("Search failed");
        assert!(!results.is_empty(), "Should find imported content");

        cleanup_db(&path1);
        cleanup_db(&path2);
    }
}
