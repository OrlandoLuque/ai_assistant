//! RAG (Retrieval-Augmented Generation) system for knowledge and conversation context
//!
//! This module provides:
//! - Knowledge base indexing with SQLite FTS5 for full-text search
//! - Conversation history storage for "infinite" context
//! - Relevance-based retrieval to minimize token usage
//! - Multi-user support with isolated data per user

use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::messages::ChatMessage;

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
    pub fn from_hybrid_results(query: impl Into<String>, results: &[HybridKnowledgeResult]) -> Self {
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
            let sections: Vec<String> = source_chunks.iter()
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
            by_source.entry(&result.chunk.source).or_default().push(result);
        }

        for (source, source_results) in by_source {
            let sections: Vec<String> = source_results.iter()
                .map(|r| r.chunk.section.clone())
                .filter(|s| !s.is_empty())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let tokens: usize = source_results.iter().map(|r| r.chunk.token_count).sum();
            let avg_relevance = source_results.iter()
                .map(|r| r.combined_score)
                .sum::<f32>() / source_results.len() as f32;

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

        let source_info: Vec<String> = self.sources.iter()
            .map(|s| {
                if let Some(score) = s.relevance_score {
                    format!("{} ({} chunks, {:.0}% relevance)", s.source, s.chunks_used, score * 100.0)
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
            Some(crate::embeddings::LocalEmbedder::new(crate::embeddings::EmbeddingConfig::default()))
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
                crate::embeddings::EmbeddingConfig::default()
            ));
        }
    }

    /// Train the embedder on current knowledge base
    pub fn train_embedder(&mut self) -> Result<usize> {
        let embedder = match self.embedder.as_mut() {
            Some(e) => e,
            None => {
                self.embedder = Some(crate::embeddings::LocalEmbedder::new(
                    crate::embeddings::EmbeddingConfig::default()
                ));
                self.embedder.as_mut().unwrap()
            }
        };

        // Get all knowledge chunks
        let mut stmt = self.conn.prepare(
            "SELECT content FROM knowledge_chunks"
        )?;

        let contents: Vec<String> = stmt.query_map([], |row| {
            row.get::<_, String>(0)
        })?.filter_map(|r| r.ok()).collect();

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
            END;"
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
            END;"
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
                 DROP TABLE knowledge_notes_old;"
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

        let rows = stmt.query_map([], |row| Ok(User {
            id: row.get(0)?,
            user_id: row.get(1)?,
            display_name: row.get(2)?,
            global_notes: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
        }))?;

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

        let existing: Option<String> = self.conn.query_row(
            "SELECT content_hash FROM knowledge_sources WHERE source = ?1",
            [source],
            |row| row.get(0),
        ).ok();

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
        self.conn.execute(
            "DELETE FROM knowledge_chunks WHERE source = ?1",
            [source],
        )?;

        // Index new chunks
        let chunks = chunk_document(source, content);
        let now = chrono::Utc::now().to_rfc3339();
        let mut count = 0;
        let mut total_tokens = 0;

        for chunk in chunks {
            self.conn.execute(
                "INSERT INTO knowledge_chunks (source, section, content, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![chunk.source, chunk.section, chunk.content, chunk.token_count, now],
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
        self.conn.execute("DELETE FROM knowledge_chunks WHERE source = ?1", [source])?;
        self.conn.execute("DELETE FROM knowledge_sources WHERE source = ?1", [source])?;
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
    pub fn search_knowledge(&self, query: &str, max_tokens: usize, top_k: usize) -> Result<Vec<KnowledgeChunk>> {
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
             LIMIT ?2"
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
        let placeholders: Vec<String> = sources.iter().enumerate()
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
    pub fn search_knowledge_hybrid(&self, query: &str, max_tokens: usize, top_k: usize) -> Result<Vec<HybridKnowledgeResult>> {
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
             LIMIT ?2"
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
        let max_bm25 = bm25_results.iter()
            .map(|(_, s)| *s)
            .fold(0.0f32, f32::max)
            .max(1.0);

        // Compute semantic scores if enabled
        let semantic_scores: Vec<Option<f32>> = if self.hybrid_config.semantic_enabled {
            if let Some(ref embedder) = self.embedder {
                let query_embedding = embedder.embed(query);
                bm25_results.iter()
                    .map(|(chunk, _)| {
                        let chunk_embedding = embedder.embed(&chunk.content);
                        let score = crate::embeddings::LocalEmbedder::cosine_similarity(
                            &query_embedding,
                            &chunk_embedding
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
        let mut hybrid_results: Vec<HybridKnowledgeResult> = bm25_results.into_iter()
            .zip(semantic_scores.into_iter())
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
            b.combined_score.partial_cmp(&a.combined_score)
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
    pub fn search_knowledge_auto(&self, query: &str, max_tokens: usize, top_k: usize) -> Result<Vec<KnowledgeChunk>> {
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
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_chunks",
            [],
            |row| row.get(0),
        )?;
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
            "SELECT source, notes FROM knowledge_notes WHERE user_id = ?1 ORDER BY source"
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
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT source FROM knowledge_chunks ORDER BY source"
        )?;

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
    pub fn store_message(&self, user_id: &str, session_id: &str, msg: &ChatMessage, in_context: bool) -> Result<i64> {
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
    pub fn mark_messages_out_of_context(&self, user_id: &str, session_id: &str, message_ids: &[i64]) -> Result<()> {
        for id in message_ids {
            self.conn.execute(
                "UPDATE conversation_messages SET in_context = 0 WHERE id = ?1 AND user_id = ?2 AND session_id = ?3",
                rusqlite::params![id, user_id, session_id],
            )?;
        }
        Ok(())
    }

    /// Search conversation history for relevant messages
    pub fn search_conversation(&self, user_id: &str, session_id: &str, query: &str, max_tokens: usize, exclude_in_context: bool) -> Result<Vec<StoredMessage>> {
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

        let rows = stmt.query_map(rusqlite::params![search_terms, user_id, session_id], |row| {
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

        Ok(results)
    }

    /// Get recent messages from conversation (not in current context)
    pub fn get_recent_archived_messages(&self, user_id: &str, session_id: &str, max_tokens: usize) -> Result<Vec<StoredMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, role, content, timestamp, token_count, in_context
             FROM conversation_messages
             WHERE user_id = ?1 AND session_id = ?2 AND in_context = 0
             ORDER BY timestamp DESC
             LIMIT 50"
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
    pub fn get_conversation_stats(&self, user_id: &str, session_id: &str) -> Result<(usize, usize, usize)> {
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
    pub fn store_message_default(&self, session_id: &str, msg: &ChatMessage, in_context: bool) -> Result<i64> {
        self.store_message(DEFAULT_USER_ID, session_id, msg, in_context)
    }

    /// Mark messages as out of context (legacy - uses default user)
    pub fn mark_messages_out_of_context_default(&self, session_id: &str, message_ids: &[i64]) -> Result<()> {
        self.mark_messages_out_of_context(DEFAULT_USER_ID, session_id, message_ids)
    }

    /// Search conversation (legacy - uses default user)
    pub fn search_conversation_default(&self, session_id: &str, query: &str, max_tokens: usize, exclude_in_context: bool) -> Result<Vec<StoredMessage>> {
        self.search_conversation(DEFAULT_USER_ID, session_id, query, max_tokens, exclude_in_context)
    }

    /// Get recent archived messages (legacy - uses default user)
    pub fn get_recent_archived_messages_default(&self, session_id: &str, max_tokens: usize) -> Result<Vec<StoredMessage>> {
        self.get_recent_archived_messages(DEFAULT_USER_ID, session_id, max_tokens)
    }

    /// Get conversation stats (legacy - uses default user)
    pub fn get_conversation_stats_default(&self, session_id: &str) -> Result<(usize, usize, usize)> {
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

        let chunks: Vec<ExportedChunk> = stmt.query_map([], |row| {
            Ok(ExportedChunk {
                source: row.get(0)?,
                section: row.get(1)?,
                content: row.get(2)?,
                token_count: row.get(3)?,
            })
        })?.filter_map(|r| r.ok()).collect();

        // Get all sources
        let mut stmt = self.conn.prepare(
            "SELECT source, content_hash, chunk_count, total_tokens FROM knowledge_sources ORDER BY source"
        )?;

        let sources: Vec<ExportedSource> = stmt.query_map([], |row| {
            Ok(ExportedSource {
                source: row.get(0)?,
                content_hash: row.get(1)?,
                chunk_count: row.get(2)?,
                total_tokens: row.get(3)?,
            })
        })?.filter_map(|r| r.ok()).collect();

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
            let exists: bool = self.conn.query_row(
                "SELECT COUNT(*) > 0 FROM knowledge_sources WHERE source = ?1",
                [&source.source],
                |row| row.get(0),
            ).unwrap_or(false);

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
                let has_chunks: bool = self.conn.query_row(
                    "SELECT COUNT(*) > 0 FROM knowledge_chunks WHERE source = ?1",
                    [&chunk.source],
                    |row| row.get(0),
                ).unwrap_or(false);

                if has_chunks {
                    continue;
                }
            }

            self.conn.execute(
                "INSERT INTO knowledge_chunks (source, section, content, token_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![chunk.source, chunk.section, chunk.content, chunk.token_count, now],
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
            // Escape special FTS5 characters
            let escaped = w
                .replace('"', "")
                .replace('*', "")
                .replace('(', "")
                .replace(')', "")
                .replace(':', "")
                .to_lowercase();
            format!("{}*", escaped)
        })
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
    let mut by_source: std::collections::HashMap<&str, Vec<&KnowledgeChunk>> = std::collections::HashMap::new();
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
        let role = if msg.role == "user" { "User" } else { "Assistant" };
        context.push_str(&format!("{}: {}\n", role, msg.content));
    }

    context.push_str("[End previous context]\n");
    context
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
