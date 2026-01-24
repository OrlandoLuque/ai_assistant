//! Automatic RAG indexing of imported content.
//!
//! This module provides automatic chunking, indexing, and storage of documents
//! into a SQLite database for RAG retrieval. It supports multiple chunking
//! strategies, incremental re-indexing, and maintains document metadata for
//! efficient updates.
//!
//! This module is gated behind the `rag` feature in `lib.rs`.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(feature = "document-formats")]
use super::document_parsing::ParsedDocument;

// ============================================================================
// Enums
// ============================================================================

/// Strategy used to chunk document content for indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexChunkingStrategy {
    /// Split by paragraph boundaries (double newlines).
    Paragraph,
    /// Split by sentence boundaries.
    Sentence,
    /// Use a sliding window with configurable overlap.
    SlidingWindow,
    /// Adaptive: paragraph-based, merging small chunks and splitting large ones.
    Adaptive,
}

impl Default for IndexChunkingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Position of a chunk within its source document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkPosition {
    /// The first chunk in the document.
    First,
    /// A chunk in the middle of the document.
    Middle,
    /// The last chunk in the document.
    Last,
    /// The only chunk in the document.
    Only,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Metadata for a single indexed document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDocumentMeta {
    /// Source identifier (path or URI).
    pub source: String,
    /// When the document was first indexed.
    pub first_indexed: DateTime<Utc>,
    /// When the document was last re-indexed.
    pub last_indexed: DateTime<Utc>,
    /// Hash of the content for change detection.
    pub content_hash: u64,
    /// MIME-like content type (e.g. "text/plain", "text/markdown").
    pub content_type: String,
    /// Number of chunks produced.
    pub chunk_count: usize,
    /// Total character count.
    pub char_count: usize,
    /// Estimated token count.
    pub token_count: usize,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// A single indexable chunk of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexableChunk {
    /// The text content of this chunk.
    pub content: String,
    /// Zero-based index within the source document.
    pub chunk_index: usize,
    /// Source identifier.
    pub source: String,
    /// Optional section title.
    pub section: Option<String>,
    /// Rich metadata about this chunk.
    pub metadata: ChunkMetadata,
}

/// Metadata associated with a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source identifier.
    pub source: String,
    /// When the chunk was imported/indexed.
    pub import_date: DateTime<Utc>,
    /// Content type of the source.
    pub content_type: String,
    /// Character offset within the source document.
    pub char_offset: usize,
    /// Position of this chunk in the document.
    pub position: ChunkPosition,
    /// Optional section title.
    pub section: Option<String>,
    /// Tags for classification and filtering.
    pub tags: Vec<String>,
}

/// Result of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingResult {
    /// Number of documents successfully indexed.
    pub documents_indexed: usize,
    /// Total chunks created across all documents.
    pub chunks_created: usize,
    /// Documents skipped (already up-to-date).
    pub documents_skipped: usize,
    /// Documents that failed to index.
    pub documents_failed: usize,
    /// Error messages from failed documents.
    pub errors: Vec<String>,
    /// Total time taken in milliseconds.
    pub duration_ms: u64,
}

impl IndexingResult {
    fn empty() -> Self {
        Self {
            documents_indexed: 0,
            chunks_created: 0,
            documents_skipped: 0,
            documents_failed: 0,
            errors: Vec::new(),
            duration_ms: 0,
        }
    }

    fn merge(&mut self, other: &IndexingResult) {
        self.documents_indexed += other.documents_indexed;
        self.chunks_created += other.chunks_created;
        self.documents_skipped += other.documents_skipped;
        self.documents_failed += other.documents_failed;
        self.errors.extend(other.errors.iter().cloned());
    }
}

/// Configuration for the auto-indexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIndexConfig {
    /// The chunking strategy to use.
    pub chunking_strategy: IndexChunkingStrategy,
    /// Target number of tokens per chunk.
    pub target_chunk_tokens: usize,
    /// Number of overlapping tokens between adjacent chunks.
    pub overlap_tokens: usize,
    /// Maximum tokens allowed in a single chunk.
    pub max_chunk_tokens: usize,
    /// Minimum tokens allowed in a single chunk.
    pub min_chunk_tokens: usize,
    /// If true, force re-indexing even if content hash matches.
    pub force_reindex: bool,
    /// File extensions that are supported for indexing.
    pub supported_extensions: Vec<String>,
    /// Default tags applied to all indexed chunks.
    pub default_tags: Vec<String>,
    /// Maximum document size in bytes (documents larger are skipped).
    pub max_document_size: usize,
    /// Whether to preserve source metadata in chunks.
    pub preserve_metadata: bool,
}

impl Default for AutoIndexConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: IndexChunkingStrategy::Adaptive,
            target_chunk_tokens: 300,
            overlap_tokens: 30,
            max_chunk_tokens: 500,
            min_chunk_tokens: 50,
            force_reindex: false,
            supported_extensions: vec![
                "txt".to_string(),
                "md".to_string(),
                "html".to_string(),
                "epub".to_string(),
                "docx".to_string(),
                "odt".to_string(),
            ],
            default_tags: Vec::new(),
            max_document_size: 50 * 1024 * 1024, // 50 MB
            preserve_metadata: true,
        }
    }
}

/// The indexing state: tracks all indexed documents and global counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexState {
    /// Map from source identifier to its metadata.
    pub documents: HashMap<String, IndexedDocumentMeta>,
    /// Total number of chunks across all documents.
    pub total_chunks: usize,
    /// When the last full re-index was performed.
    pub last_full_reindex: Option<DateTime<Utc>>,
}

impl Default for IndexState {
    fn default() -> Self {
        Self {
            documents: HashMap::new(),
            total_chunks: 0,
            last_full_reindex: None,
        }
    }
}

/// Statistics about the current index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of indexed documents.
    pub total_documents: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Estimated total tokens across all chunks.
    pub total_tokens_estimated: usize,
    /// Average chunks per document.
    pub avg_chunks_per_doc: f64,
    /// Count of documents per content type.
    pub content_types: HashMap<String, usize>,
}

// ============================================================================
// AutoIndexer
// ============================================================================

/// The main auto-indexer: manages chunking, storage, and retrieval of indexed content.
pub struct AutoIndexer {
    /// Configuration for the indexer.
    config: AutoIndexConfig,
    /// Current indexing state.
    state: IndexState,
    /// SQLite connection for persistent storage.
    db: Connection,
}

impl AutoIndexer {
    /// Create a new `AutoIndexer` with the given database path and configuration.
    ///
    /// Opens (or creates) the SQLite database and initializes the required tables.
    pub fn new(db_path: &str, config: AutoIndexConfig) -> Result<Self> {
        let db = Connection::open(db_path)?;
        let mut indexer = Self {
            config,
            state: IndexState::default(),
            db,
        };
        indexer.init_tables()?;
        indexer.load_state_from_db()?;
        Ok(indexer)
    }

    /// Index a raw text string with the given source identifier and content type.
    pub fn index_text(
        &mut self,
        content: &str,
        source: &str,
        content_type: &str,
    ) -> Result<IndexingResult> {
        let start = Instant::now();

        if !self.config.force_reindex && !self.needs_reindex(source, content) {
            return Ok(IndexingResult {
                documents_skipped: 1,
                duration_ms: start.elapsed().as_millis() as u64,
                ..IndexingResult::empty()
            });
        }

        // Remove existing chunks for this source
        self.remove_chunks(source)?;

        // Chunk the content
        let chunks = self.chunk_content(content, source, content_type);
        let chunk_count = chunks.len();

        // Store the chunks
        self.store_chunks(&chunks)?;

        // Update state
        let now = Utc::now();
        let content_hash = Self::compute_hash(content);
        let token_count = Self::estimate_tokens(content);

        let meta = IndexedDocumentMeta {
            source: source.to_string(),
            first_indexed: self
                .state
                .documents
                .get(source)
                .map(|m| m.first_indexed)
                .unwrap_or(now),
            last_indexed: now,
            content_hash,
            content_type: content_type.to_string(),
            chunk_count,
            char_count: content.len(),
            token_count,
            metadata: HashMap::new(),
        };

        self.state.total_chunks = self
            .state
            .total_chunks
            .saturating_sub(
                self.state
                    .documents
                    .get(source)
                    .map(|m| m.chunk_count)
                    .unwrap_or(0),
            )
            + chunk_count;

        self.save_document_meta(&meta)?;
        self.state.documents.insert(source.to_string(), meta);

        Ok(IndexingResult {
            documents_indexed: 1,
            chunks_created: chunk_count,
            duration_ms: start.elapsed().as_millis() as u64,
            ..IndexingResult::empty()
        })
    }

    /// Index a `ParsedDocument` (available when the `document-formats` feature is enabled).
    #[cfg(feature = "document-formats")]
    pub fn index_parsed_document(
        &mut self,
        doc: &super::document_parsing::ParsedDocument,
    ) -> Result<IndexingResult> {
        let source = doc
            .source_path
            .as_deref()
            .unwrap_or("unknown")
            .to_string();
        let content_type = match doc.format {
            super::document_parsing::DocumentFormat::Epub => "application/epub+zip",
            super::document_parsing::DocumentFormat::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            super::document_parsing::DocumentFormat::Odt => {
                "application/vnd.oasis.opendocument.text"
            }
            super::document_parsing::DocumentFormat::Html => "text/html",
            super::document_parsing::DocumentFormat::PlainText => "text/plain",
        };
        self.index_text(&doc.text, &source, content_type)
    }

    /// Index a single file from the filesystem.
    #[cfg(feature = "document-formats")]
    pub fn index_file(&mut self, path: &Path) -> Result<IndexingResult> {
        let path_str = path.to_string_lossy().to_string();

        // Check file size
        let file_meta = std::fs::metadata(path)?;
        if file_meta.len() as usize > self.config.max_document_size {
            return Ok(IndexingResult {
                documents_skipped: 1,
                errors: vec![format!(
                    "File too large: {} ({} bytes, max {})",
                    path_str,
                    file_meta.len(),
                    self.config.max_document_size
                )],
                ..IndexingResult::empty()
            });
        }

        // Check extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !self.config.supported_extensions.contains(&ext) {
            return Ok(IndexingResult {
                documents_skipped: 1,
                errors: vec![format!("Unsupported extension: .{}", ext)],
                ..IndexingResult::empty()
            });
        }

        // Parse the file using the document parser
        let parser_config = super::document_parsing::DocumentParserConfig::default();
        let parser = super::document_parsing::DocumentParser::new(parser_config);
        let doc = parser.parse_file(path)?;
        self.index_parsed_document(&doc)
    }

    /// Index all supported files in a directory, optionally recursively.
    #[cfg(feature = "document-formats")]
    pub fn index_directory(&mut self, dir: &Path, recursive: bool) -> Result<IndexingResult> {
        let start = Instant::now();
        let mut result = IndexingResult::empty();

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    result.documents_failed += 1;
                    result.errors.push(format!("Failed to read entry: {}", err));
                    continue;
                }
            };

            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    match self.index_directory(&path, true) {
                        Ok(sub_result) => result.merge(&sub_result),
                        Err(err) => {
                            result.documents_failed += 1;
                            result
                                .errors
                                .push(format!("Failed to index dir {:?}: {}", path, err));
                        }
                    }
                }
            } else if path.is_file() {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if self.config.supported_extensions.contains(&ext) {
                    match self.index_file(&path) {
                        Ok(sub_result) => result.merge(&sub_result),
                        Err(err) => {
                            result.documents_failed += 1;
                            result
                                .errors
                                .push(format!("Failed to index {:?}: {}", path, err));
                        }
                    }
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Re-index only documents whose content has changed since the last indexing.
    pub fn incremental_reindex(&mut self) -> Result<IndexingResult> {
        let start = Instant::now();
        let mut result = IndexingResult::empty();

        // Collect sources that exist as files
        let sources: Vec<String> = self.state.documents.keys().cloned().collect();

        for source in sources {
            let path = Path::new(&source);
            if path.exists() && path.is_file() {
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        if self.needs_reindex(&source, &content) {
                            let content_type = self.guess_content_type(&source);
                            match self.index_text(&content, &source, &content_type) {
                                Ok(sub) => result.merge(&sub),
                                Err(err) => {
                                    result.documents_failed += 1;
                                    result.errors.push(format!(
                                        "Failed to reindex {}: {}",
                                        source, err
                                    ));
                                }
                            }
                        } else {
                            result.documents_skipped += 1;
                        }
                    }
                    Err(err) => {
                        result.documents_failed += 1;
                        result
                            .errors
                            .push(format!("Failed to read {}: {}", source, err));
                    }
                }
            } else {
                result.documents_skipped += 1;
            }
        }

        self.state.last_full_reindex = Some(Utc::now());
        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Check if a document needs to be re-indexed (content hash differs).
    pub fn needs_reindex(&self, source: &str, content: &str) -> bool {
        if self.config.force_reindex {
            return true;
        }
        match self.state.documents.get(source) {
            Some(meta) => meta.content_hash != Self::compute_hash(content),
            None => true,
        }
    }

    /// Remove a document and all its chunks from the index.
    pub fn remove_document(&mut self, source: &str) -> Result<()> {
        self.remove_chunks(source)?;
        if let Some(meta) = self.state.documents.remove(source) {
            self.state.total_chunks = self.state.total_chunks.saturating_sub(meta.chunk_count);
        }
        self.db
            .execute("DELETE FROM indexed_documents WHERE source = ?1", params![source])?;
        Ok(())
    }

    /// Get a reference to the current index state.
    pub fn state(&self) -> &IndexState {
        &self.state
    }

    /// Get metadata for a specific document by source identifier.
    pub fn document_meta(&self, source: &str) -> Option<&IndexedDocumentMeta> {
        self.state.documents.get(source)
    }

    /// Get all indexed document metadata entries.
    pub fn indexed_documents(&self) -> Vec<&IndexedDocumentMeta> {
        self.state.documents.values().collect()
    }

    /// Compute statistics about the current index.
    pub fn stats(&self) -> IndexStats {
        let total_documents = self.state.documents.len();
        let total_chunks = self.state.total_chunks;
        let total_tokens_estimated: usize = self
            .state
            .documents
            .values()
            .map(|m| m.token_count)
            .sum();

        let avg_chunks_per_doc = if total_documents > 0 {
            total_chunks as f64 / total_documents as f64
        } else {
            0.0
        };

        let mut content_types: HashMap<String, usize> = HashMap::new();
        for meta in self.state.documents.values() {
            *content_types.entry(meta.content_type.clone()).or_insert(0) += 1;
        }

        IndexStats {
            total_documents,
            total_chunks,
            total_tokens_estimated,
            avg_chunks_per_doc,
            content_types,
        }
    }

    /// Export the current state as a JSON string.
    pub fn export_state(&self) -> String {
        serde_json::to_string_pretty(&self.state).unwrap_or_default()
    }

    /// Import state from a JSON string, replacing the current state.
    pub fn import_state(&mut self, json: &str) -> Result<()> {
        let state: IndexState = serde_json::from_str(json)?;
        self.state = state;
        Ok(())
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    /// Chunk content using the configured strategy.
    fn chunk_content(
        &self,
        content: &str,
        source: &str,
        content_type: &str,
    ) -> Vec<IndexableChunk> {
        let raw_chunks = match self.config.chunking_strategy {
            IndexChunkingStrategy::Paragraph => self.chunk_by_paragraph(content),
            IndexChunkingStrategy::Sentence => self.chunk_by_sentence(content),
            IndexChunkingStrategy::SlidingWindow => self.chunk_sliding_window(content),
            IndexChunkingStrategy::Adaptive => self.chunk_adaptive(content),
        };

        let total = raw_chunks.len();
        let now = Utc::now();

        raw_chunks
            .into_iter()
            .enumerate()
            .map(|(i, (text, char_offset))| {
                let position = Self::determine_position(i, total);
                IndexableChunk {
                    content: text.clone(),
                    chunk_index: i,
                    source: source.to_string(),
                    section: None,
                    metadata: ChunkMetadata {
                        source: source.to_string(),
                        import_date: now,
                        content_type: content_type.to_string(),
                        char_offset,
                        position,
                        section: None,
                        tags: self.config.default_tags.clone(),
                    },
                }
            })
            .collect()
    }

    /// Store chunks into the SQLite database.
    fn store_chunks(&mut self, chunks: &[IndexableChunk]) -> Result<()> {
        let tx = self.db.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO indexed_chunks (source, chunk_index, content, section, metadata_json, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )?;

            for chunk in chunks {
                let metadata_json = serde_json::to_string(&chunk.metadata)?;
                let created_at = chunk.metadata.import_date.to_rfc3339();
                stmt.execute(params![
                    chunk.source,
                    chunk.chunk_index as i64,
                    chunk.content,
                    chunk.section,
                    metadata_json,
                    created_at,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Remove all chunks for a given source from the database.
    fn remove_chunks(&mut self, source: &str) -> Result<()> {
        self.db
            .execute("DELETE FROM indexed_chunks WHERE source = ?1", params![source])?;
        Ok(())
    }

    /// Compute a simple hash of the content string for change detection.
    fn compute_hash(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Initialize the SQLite tables if they do not already exist.
    fn init_tables(&self) -> Result<()> {
        self.db.execute_batch(
            "CREATE TABLE IF NOT EXISTS indexed_chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                section TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_documents (
                source TEXT PRIMARY KEY,
                first_indexed TEXT NOT NULL,
                last_indexed TEXT NOT NULL,
                content_hash INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON indexed_chunks(source);",
        )?;
        Ok(())
    }

    /// Load index state from the database on startup.
    fn load_state_from_db(&mut self) -> Result<()> {
        let mut stmt = self.db.prepare(
            "SELECT source, first_indexed, last_indexed, content_hash, content_type, \
             chunk_count, char_count, token_count, metadata_json FROM indexed_documents",
        )?;

        let rows = stmt.query_map([], |row| {
            let source: String = row.get(0)?;
            let first_indexed_str: String = row.get(1)?;
            let last_indexed_str: String = row.get(2)?;
            let content_hash: i64 = row.get(3)?;
            let content_type: String = row.get(4)?;
            let chunk_count: i64 = row.get(5)?;
            let char_count: i64 = row.get(6)?;
            let token_count: i64 = row.get(7)?;
            let metadata_json: Option<String> = row.get(8)?;

            Ok((
                source,
                first_indexed_str,
                last_indexed_str,
                content_hash as u64,
                content_type,
                chunk_count as usize,
                char_count as usize,
                token_count as usize,
                metadata_json,
            ))
        })?;

        let mut total_chunks = 0usize;
        for row in rows {
            let (
                source,
                first_indexed_str,
                last_indexed_str,
                content_hash,
                content_type,
                chunk_count,
                char_count,
                token_count,
                metadata_json,
            ) = row?;

            let first_indexed = DateTime::parse_from_rfc3339(&first_indexed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let last_indexed = DateTime::parse_from_rfc3339(&last_indexed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let metadata: HashMap<String, String> = metadata_json
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default();

            total_chunks += chunk_count;

            self.state.documents.insert(
                source.clone(),
                IndexedDocumentMeta {
                    source,
                    first_indexed,
                    last_indexed,
                    content_hash,
                    content_type,
                    chunk_count,
                    char_count,
                    token_count,
                    metadata,
                },
            );
        }

        self.state.total_chunks = total_chunks;
        Ok(())
    }

    /// Persist document metadata to the database.
    fn save_document_meta(&self, meta: &IndexedDocumentMeta) -> Result<()> {
        let metadata_json = serde_json::to_string(&meta.metadata)?;
        self.db.execute(
            "INSERT OR REPLACE INTO indexed_documents \
             (source, first_indexed, last_indexed, content_hash, content_type, \
              chunk_count, char_count, token_count, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                meta.source,
                meta.first_indexed.to_rfc3339(),
                meta.last_indexed.to_rfc3339(),
                meta.content_hash as i64,
                meta.content_type,
                meta.chunk_count as i64,
                meta.char_count as i64,
                meta.token_count as i64,
                metadata_json,
            ],
        )?;
        Ok(())
    }

    /// Split content using a sliding window approach.
    fn chunk_sliding_window(&self, content: &str) -> Vec<(String, usize)> {
        let target_chars = self.config.target_chunk_tokens * 4;
        let overlap_chars = self.config.overlap_tokens * 4;
        let max_chars = self.config.max_chunk_tokens * 4;

        if content.len() <= max_chars {
            return vec![(content.to_string(), 0)];
        }

        let mut chunks = Vec::new();
        let mut offset = 0usize;

        while offset < content.len() {
            let end = (offset + target_chars).min(content.len());

            // Try to find a natural boundary near the end
            let chunk_end = if end < content.len() {
                Self::find_break_point(content, end, target_chars / 4)
            } else {
                end
            };

            let chunk_text = &content[offset..chunk_end];
            chunks.push((chunk_text.to_string(), offset));

            if chunk_end >= content.len() {
                break;
            }

            // Advance by (target - overlap)
            let advance = if target_chars > overlap_chars {
                target_chars - overlap_chars
            } else {
                target_chars
            };
            offset += advance;
            if offset >= chunk_end {
                offset = chunk_end;
            }
        }

        chunks
    }

    /// Split content by paragraph boundaries (double newlines).
    fn chunk_by_paragraph(&self, content: &str) -> Vec<(String, usize)> {
        let mut chunks = Vec::new();
        let mut offset = 0usize;

        // Split on double newlines
        let paragraphs: Vec<&str> = content.split("\n\n").collect();

        for para in &paragraphs {
            let trimmed = para.trim();
            if !trimmed.is_empty() {
                // Find the actual offset of this paragraph in the original content
                let para_offset = content[offset..]
                    .find(trimmed)
                    .map(|pos| offset + pos)
                    .unwrap_or(offset);
                chunks.push((trimmed.to_string(), para_offset));
                offset = para_offset + trimmed.len();
            }
        }

        chunks
    }

    /// Split content by sentence boundaries.
    fn chunk_by_sentence(&self, content: &str) -> Vec<(String, usize)> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_offset = 0usize;
        let mut chunk_start = 0usize;
        let target_chars = self.config.target_chunk_tokens * 4;

        let sentences = Self::split_sentences(content);

        for (sentence, sent_offset) in &sentences {
            if current_chunk.is_empty() {
                chunk_start = *sent_offset;
            }

            let would_be_len = current_chunk.len() + sentence.len() + 1;

            if would_be_len > target_chars && !current_chunk.is_empty() {
                chunks.push((current_chunk.trim().to_string(), chunk_start));
                current_chunk = String::new();
                chunk_start = *sent_offset;
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        if !current_chunk.trim().is_empty() {
            chunks.push((current_chunk.trim().to_string(), chunk_start));
        }

        chunks
    }

    /// Adaptive chunking: paragraph-based, merging small chunks and splitting large ones.
    fn chunk_adaptive(&self, content: &str) -> Vec<(String, usize)> {
        let paragraphs = self.chunk_by_paragraph(content);
        let min_chars = self.config.min_chunk_tokens * 4;
        let max_chars = self.config.max_chunk_tokens * 4;
        let target_chars = self.config.target_chunk_tokens * 4;

        let mut result: Vec<(String, usize)> = Vec::new();
        let mut buffer = String::new();
        let mut buffer_offset = 0usize;

        for (para, offset) in paragraphs {
            if para.len() > max_chars {
                // Flush buffer first
                if !buffer.trim().is_empty() {
                    result.push((buffer.trim().to_string(), buffer_offset));
                    buffer = String::new();
                }
                // Split the large paragraph using sliding window
                let sub_chunks = self.chunk_sliding_window(&para);
                for (sub_text, sub_off) in sub_chunks {
                    result.push((sub_text, offset + sub_off));
                }
            } else if buffer.len() + para.len() + 2 > target_chars && !buffer.is_empty() {
                // Current buffer + new paragraph exceeds target, flush buffer
                result.push((buffer.trim().to_string(), buffer_offset));
                buffer = para;
                buffer_offset = offset;
            } else {
                // Merge into buffer
                if buffer.is_empty() {
                    buffer_offset = offset;
                } else {
                    buffer.push_str("\n\n");
                }
                buffer.push_str(&para);
            }
        }

        // Flush remaining buffer
        if !buffer.trim().is_empty() {
            // If the remaining buffer is too small, merge with previous
            if buffer.len() < min_chars && !result.is_empty() {
                let last_idx = result.len() - 1;
                result[last_idx].0.push_str("\n\n");
                result[last_idx].0.push_str(buffer.trim());
            } else {
                result.push((buffer.trim().to_string(), buffer_offset));
            }
        }

        // Handle edge case: empty input
        if result.is_empty() && !content.trim().is_empty() {
            result.push((content.trim().to_string(), 0));
        }

        result
    }

    /// Estimate the number of tokens in a text (approximation: chars / 4).
    fn estimate_tokens(text: &str) -> usize {
        (text.len() + 3) / 4
    }

    /// Determine the position of a chunk given its index and total count.
    fn determine_position(index: usize, total: usize) -> ChunkPosition {
        if total <= 1 {
            ChunkPosition::Only
        } else if index == 0 {
            ChunkPosition::First
        } else if index == total - 1 {
            ChunkPosition::Last
        } else {
            ChunkPosition::Middle
        }
    }

    /// Find a suitable break point near `pos` within `search_range` characters.
    fn find_break_point(content: &str, pos: usize, search_range: usize) -> usize {
        let start = if pos > search_range {
            pos - search_range
        } else {
            pos
        };
        let end = (pos + search_range).min(content.len());

        // Look backward from pos for a newline, period, or space
        let search_slice = &content[start..pos];
        if let Some(nl_pos) = search_slice.rfind('\n') {
            return start + nl_pos + 1;
        }
        if let Some(dot_pos) = search_slice.rfind(". ") {
            return start + dot_pos + 2;
        }
        if let Some(sp_pos) = search_slice.rfind(' ') {
            return start + sp_pos + 1;
        }

        // Look forward for a space
        let forward_slice = &content[pos..end];
        if let Some(sp_pos) = forward_slice.find(' ') {
            return pos + sp_pos + 1;
        }

        pos
    }

    /// Split text into sentences with their character offsets.
    fn split_sentences(content: &str) -> Vec<(String, usize)> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut current_start = 0usize;
        let mut offset = 0usize;

        for ch in content.chars() {
            current.push(ch);
            offset += ch.len_utf8();

            if (ch == '.' || ch == '!' || ch == '?') && current.len() > 1 {
                // Check if followed by space or end
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push((trimmed, current_start));
                }
                current = String::new();
                current_start = offset;
            }
        }

        // Remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push((trimmed, current_start));
        }

        sentences
    }

    /// Guess content type from a file path extension.
    fn guess_content_type(&self, source: &str) -> String {
        let path = Path::new(source);
        match path.extension().and_then(|e| e.to_str()) {
            Some("md") => "text/markdown".to_string(),
            Some("html") | Some("htm") => "text/html".to_string(),
            Some("epub") => "application/epub+zip".to_string(),
            Some("docx") => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    .to_string()
            }
            Some("odt") => "application/vnd.oasis.opendocument.text".to_string(),
            _ => "text/plain".to_string(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AutoIndexConfig {
        AutoIndexConfig {
            target_chunk_tokens: 50,
            overlap_tokens: 10,
            max_chunk_tokens: 100,
            min_chunk_tokens: 10,
            ..AutoIndexConfig::default()
        }
    }

    fn create_test_indexer() -> AutoIndexer {
        AutoIndexer::new(":memory:", test_config()).expect("Failed to create indexer")
    }

    #[test]
    fn test_index_text_basic() {
        let mut indexer = create_test_indexer();
        let content = "Hello world. This is a test document with some content.";
        let result = indexer
            .index_text(content, "test.txt", "text/plain")
            .unwrap();

        assert_eq!(result.documents_indexed, 1);
        assert!(result.chunks_created > 0);
        assert_eq!(result.documents_skipped, 0);
        assert_eq!(result.documents_failed, 0);

        // State should be updated
        assert_eq!(indexer.state().documents.len(), 1);
        assert!(indexer.document_meta("test.txt").is_some());
    }

    #[test]
    fn test_skip_unchanged_document() {
        let mut indexer = create_test_indexer();
        let content = "Hello world. This is a test document.";

        // First index
        let r1 = indexer
            .index_text(content, "test.txt", "text/plain")
            .unwrap();
        assert_eq!(r1.documents_indexed, 1);

        // Second index with same content should skip
        let r2 = indexer
            .index_text(content, "test.txt", "text/plain")
            .unwrap();
        assert_eq!(r2.documents_indexed, 0);
        assert_eq!(r2.documents_skipped, 1);
    }

    #[test]
    fn test_reindex_changed_document() {
        let mut indexer = create_test_indexer();
        let content1 = "Hello world. First version.";
        let content2 = "Hello world. Second version with more content added.";

        let r1 = indexer
            .index_text(content1, "test.txt", "text/plain")
            .unwrap();
        assert_eq!(r1.documents_indexed, 1);
        let hash1 = indexer.document_meta("test.txt").unwrap().content_hash;

        let r2 = indexer
            .index_text(content2, "test.txt", "text/plain")
            .unwrap();
        assert_eq!(r2.documents_indexed, 1);
        let hash2 = indexer.document_meta("test.txt").unwrap().content_hash;

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_remove_document() {
        let mut indexer = create_test_indexer();
        let content = "Some document content for removal testing.";
        indexer
            .index_text(content, "to_remove.txt", "text/plain")
            .unwrap();

        assert!(indexer.document_meta("to_remove.txt").is_some());

        indexer.remove_document("to_remove.txt").unwrap();
        assert!(indexer.document_meta("to_remove.txt").is_none());
        assert_eq!(indexer.state().documents.len(), 0);
    }

    #[test]
    fn test_stats_computation() {
        let mut indexer = create_test_indexer();
        indexer
            .index_text("First document content.", "doc1.txt", "text/plain")
            .unwrap();
        indexer
            .index_text("Second document content.", "doc2.md", "text/markdown")
            .unwrap();

        let stats = indexer.stats();
        assert_eq!(stats.total_documents, 2);
        assert!(stats.total_chunks >= 2);
        assert!(stats.total_tokens_estimated > 0);
        assert!(stats.content_types.contains_key("text/plain"));
        assert!(stats.content_types.contains_key("text/markdown"));
    }

    #[test]
    fn test_export_import_state() {
        let mut indexer = create_test_indexer();
        indexer
            .index_text("Test content for export.", "export.txt", "text/plain")
            .unwrap();

        let exported = indexer.export_state();
        assert!(!exported.is_empty());

        // Create a new indexer and import
        let mut indexer2 = create_test_indexer();
        indexer2.import_state(&exported).unwrap();
        assert_eq!(indexer2.state().documents.len(), 1);
        assert!(indexer2.document_meta("export.txt").is_some());
    }

    #[test]
    fn test_chunking_strategies() {
        let config_para = AutoIndexConfig {
            chunking_strategy: IndexChunkingStrategy::Paragraph,
            ..test_config()
        };
        let indexer = AutoIndexer::new(":memory:", config_para).unwrap();
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = indexer.chunk_by_paragraph(content);
        assert_eq!(chunks.len(), 3);

        let config_sent = AutoIndexConfig {
            chunking_strategy: IndexChunkingStrategy::Sentence,
            ..test_config()
        };
        let indexer2 = AutoIndexer::new(":memory:", config_sent).unwrap();
        let content2 = "First sentence. Second sentence. Third sentence.";
        let chunks2 = indexer2.chunk_by_sentence(content2);
        assert!(!chunks2.is_empty());
    }

    #[test]
    fn test_determine_position() {
        assert_eq!(AutoIndexer::determine_position(0, 1), ChunkPosition::Only);
        assert_eq!(AutoIndexer::determine_position(0, 3), ChunkPosition::First);
        assert_eq!(AutoIndexer::determine_position(1, 3), ChunkPosition::Middle);
        assert_eq!(AutoIndexer::determine_position(2, 3), ChunkPosition::Last);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(AutoIndexer::estimate_tokens(""), 0);
        assert_eq!(AutoIndexer::estimate_tokens("abcd"), 1);
        assert_eq!(AutoIndexer::estimate_tokens("abcdefgh"), 2);
        // chars / 4, rounded up
        assert_eq!(AutoIndexer::estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let h1 = AutoIndexer::compute_hash("hello world");
        let h2 = AutoIndexer::compute_hash("hello world");
        let h3 = AutoIndexer::compute_hash("different content");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_needs_reindex() {
        let mut indexer = create_test_indexer();
        let content = "Some content.";
        indexer
            .index_text(content, "check.txt", "text/plain")
            .unwrap();

        assert!(!indexer.needs_reindex("check.txt", content));
        assert!(indexer.needs_reindex("check.txt", "Different content."));
        assert!(indexer.needs_reindex("unknown.txt", content));
    }

    #[test]
    fn test_adaptive_merges_small_paragraphs() {
        let config = AutoIndexConfig {
            chunking_strategy: IndexChunkingStrategy::Adaptive,
            target_chunk_tokens: 100,
            min_chunk_tokens: 20,
            max_chunk_tokens: 200,
            ..AutoIndexConfig::default()
        };
        let indexer = AutoIndexer::new(":memory:", config).unwrap();

        // Very short paragraphs should get merged
        let content = "A.\n\nB.\n\nC.\n\nD.";
        let chunks = indexer.chunk_adaptive(content);
        // All 4 tiny paragraphs should be merged into fewer chunks
        assert!(chunks.len() < 4);
    }

    #[test]
    fn test_sliding_window_overlap() {
        let config = AutoIndexConfig {
            chunking_strategy: IndexChunkingStrategy::SlidingWindow,
            target_chunk_tokens: 10, // 40 chars
            overlap_tokens: 5,       // 20 chars overlap
            max_chunk_tokens: 15,    // 60 chars max
            ..AutoIndexConfig::default()
        };
        let indexer = AutoIndexer::new(":memory:", config).unwrap();

        let content = "a".repeat(200); // 200 chars, well above max
        let chunks = indexer.chunk_sliding_window(&content);
        assert!(chunks.len() > 1);

        // Each chunk should start before the previous one ends (overlap)
        if chunks.len() >= 2 {
            let first_end = chunks[0].1 + chunks[0].0.len();
            let second_start = chunks[1].1;
            // The second chunk should start before the first one ends (overlap region)
            assert!(second_start < first_end);
        }
    }
}
