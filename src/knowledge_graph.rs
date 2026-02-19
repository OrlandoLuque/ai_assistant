//! Knowledge Graph for Graph RAG
//!
//! This module provides a complete knowledge graph implementation for enhanced
//! retrieval-augmented generation. It extracts entities and relationships from
//! documents using LLM-based analysis and stores them in a SQLite database.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Knowledge Graph System                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
//! │  │  Documents  │───►│  GraphBuilder    │───►│  KnowledgeGraphStore  │  │
//! │  │             │    │                  │    │                       │  │
//! │  │ "Aegis makes│    │ 1. Chunk text    │    │  SQLite Database:     │  │
//! │  │  the Sabre" │    │ 2. Extract       │    │  • entities           │  │
//! │  │             │    │    entities      │    │  • relations          │  │
//! │  └─────────────┘    │ 3. Extract       │    │  • entity_mentions    │  │
//! │                     │    relations     │    │  • chunks             │  │
//! │                     │ 4. Store graph   │    │                       │  │
//! │                     └──────────────────┘    └───────────────────────┘  │
//! │                                                       │                │
//! │                                                       ▼                │
//! │  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
//! │  │   Query     │───►│   GraphQuery     │───►│   Retrieved Chunks    │  │
//! │  │             │    │                  │    │                       │  │
//! │  │ "What ships │    │ 1. Extract query │    │  Chunks mentioning:   │  │
//! │  │  does Aegis │    │    entities      │    │  • Aegis              │  │
//! │  │  make?"     │    │ 2. Traverse      │    │  • Sabre              │  │
//! │  │             │    │    relations     │    │  • Gladius            │  │
//! │  └─────────────┘    │ 3. Collect       │    │  • Vanguard           │  │
//! │                     │    chunks        │    │  • ...                │  │
//! │                     └──────────────────┘    └───────────────────────┘  │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use ai_assistant::knowledge_graph::{
//!     KnowledgeGraph, KnowledgeGraphConfig, LlmEntityExtractor
//! };
//!
//! // Create the graph
//! let config = KnowledgeGraphConfig::default();
//! let mut graph = KnowledgeGraph::open("knowledge.db", config)?;
//!
//! // Index documents (requires LLM for entity extraction)
//! let extractor = LlmEntityExtractor::new(llm_callback);
//! graph.index_document("doc1", "Aegis Dynamics manufactures the Sabre fighter.", &extractor)?;
//!
//! // Query the graph
//! let callback = graph.as_graph_callback();
//! // Use with RagPipeline...
//! ```
//!
//! # Entity Types
//!
//! The system recognizes various entity types:
//! - **Organization**: Companies, factions, groups
//! - **Product**: Ships, vehicles, weapons, items
//! - **Person**: Characters, NPCs
//! - **Location**: Places, systems, planets
//! - **Concept**: Game mechanics, features
//! - **Event**: Missions, battles, historical events
//!
//! # Relation Types
//!
//! Common relation types extracted:
//! - `manufactures`: Organization → Product
//! - `located_in`: Entity → Location
//! - `part_of`: Entity → Entity
//! - `variant_of`: Product → Product
//! - `uses`: Entity → Product
//! - `related_to`: General relationship

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::rag_pipeline::{GraphCallback, GraphRelation, RetrievedChunk};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Knowledge Graph system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Maximum depth for graph traversal (default: 2)
    pub max_traversal_depth: usize,

    /// Maximum entities to return per query (default: 50)
    pub max_entities_per_query: usize,

    /// Maximum chunks to return per entity (default: 5)
    pub max_chunks_per_entity: usize,

    /// Minimum confidence score for relations (0.0-1.0, default: 0.5)
    pub min_relation_confidence: f32,

    /// Chunk size for document processing (default: 1000 chars)
    pub chunk_size: usize,

    /// Chunk overlap (default: 200 chars)
    pub chunk_overlap: usize,

    /// Enable entity alias resolution (default: true)
    pub resolve_aliases: bool,

    /// Enable relation inference (default: false)
    pub infer_relations: bool,

    /// Cache size for entity lookups (default: 1000)
    pub cache_size: usize,

    /// Entity types to extract (empty = all)
    pub entity_types: Vec<String>,

    /// Relation types to extract (empty = all)
    pub relation_types: Vec<String>,
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            max_traversal_depth: 2,
            max_entities_per_query: 50,
            max_chunks_per_entity: 5,
            min_relation_confidence: 0.5,
            chunk_size: 1000,
            chunk_overlap: 200,
            resolve_aliases: true,
            infer_relations: false,
            cache_size: 1000,
            entity_types: vec![],
            relation_types: vec![],
        }
    }
}

// ============================================================================
// Core Types
// ============================================================================

/// An entity in the knowledge graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: i64,
    /// Canonical name
    pub name: String,
    /// Entity type (Organization, Product, Person, Location, etc.)
    pub entity_type: EntityType,
    /// Alternative names/aliases
    pub aliases: Vec<String>,
    /// Additional metadata as JSON
    pub metadata: HashMap<String, String>,
    /// When this entity was first seen
    pub created_at: String,
    /// When this entity was last updated
    pub updated_at: String,
}

/// Types of entities recognized by the system
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Organization,
    Product,
    Person,
    Location,
    Concept,
    Event,
    Other,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Organization => "organization",
            EntityType::Product => "product",
            EntityType::Person => "person",
            EntityType::Location => "location",
            EntityType::Concept => "concept",
            EntityType::Event => "event",
            EntityType::Other => "other",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "organization" | "org" | "company" | "faction" => EntityType::Organization,
            "product" | "ship" | "vehicle" | "weapon" | "item" => EntityType::Product,
            "person" | "character" | "npc" => EntityType::Person,
            "location" | "place" | "system" | "planet" | "station" => EntityType::Location,
            "concept" | "mechanic" | "feature" => EntityType::Concept,
            "event" | "mission" | "battle" => EntityType::Event,
            _ => EntityType::Other,
        }
    }

    pub fn all() -> &'static [EntityType] {
        &[
            EntityType::Organization,
            EntityType::Product,
            EntityType::Person,
            EntityType::Location,
            EntityType::Concept,
            EntityType::Event,
            EntityType::Other,
        ]
    }
}

/// A relation between two entities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relation {
    /// Unique identifier
    pub id: i64,
    /// Source entity ID
    pub from_entity_id: i64,
    /// Target entity ID
    pub to_entity_id: i64,
    /// Type of relationship
    pub relation_type: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Evidence/source text that established this relation
    pub evidence: Option<String>,
    /// Source chunk ID
    pub source_chunk_id: Option<i64>,
    /// When this relation was created
    pub created_at: String,
}

/// A mention of an entity in a chunk
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityMention {
    /// Entity ID
    pub entity_id: i64,
    /// Chunk ID where mentioned
    pub chunk_id: i64,
    /// Character position in chunk
    pub position: Option<usize>,
    /// Surrounding context
    pub context: Option<String>,
    /// How many times mentioned in this chunk
    pub mention_count: u32,
}

/// A text chunk stored in the graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphChunk {
    /// Unique identifier
    pub id: i64,
    /// Source document identifier
    pub source_doc: String,
    /// Chunk content
    pub content: String,
    /// Position in source document
    pub position: usize,
    /// Content hash for deduplication
    pub content_hash: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// When indexed
    pub created_at: String,
}

/// Statistics about the knowledge graph
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_entities: usize,
    pub total_relations: usize,
    pub total_chunks: usize,
    pub total_mentions: usize,
    pub entities_by_type: HashMap<String, usize>,
    pub relations_by_type: HashMap<String, usize>,
}

/// Result of entity extraction from text
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relations
    pub relations: Vec<ExtractedRelation>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// An entity extracted from text (before storage)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name as found in text
    pub name: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Aliases found
    pub aliases: Vec<String>,
    /// Position in text
    pub position: Option<usize>,
    /// Surrounding context
    pub context: Option<String>,
}

/// A relation extracted from text (before storage)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedRelation {
    /// Source entity name
    pub from_entity: String,
    /// Target entity name
    pub to_entity: String,
    /// Relation type
    pub relation_type: String,
    /// Confidence score
    pub confidence: f32,
    /// Evidence text
    pub evidence: Option<String>,
}

// ============================================================================
// Entity Extraction Trait
// ============================================================================

/// Trait for extracting entities and relations from text
pub trait EntityExtractor: Send + Sync {
    /// Extract entities and relations from a text chunk
    fn extract(&self, text: &str) -> Result<ExtractionResult>;

    /// Extract entities from a query (simpler, for retrieval)
    fn extract_query_entities(&self, query: &str) -> Result<Vec<String>>;
}

// ============================================================================
// LLM-based Entity Extractor
// ============================================================================

/// LLM-based entity and relation extractor
pub struct LlmEntityExtractor<F>
where
    F: Fn(&str, &str) -> Result<String> + Send + Sync,
{
    /// LLM generation function: (system_prompt, user_prompt) -> response
    llm_fn: F,
    /// Entity types to focus on (empty = all)
    entity_types: Vec<EntityType>,
    /// Custom extraction prompt template
    custom_prompt: Option<String>,
}

impl<F> LlmEntityExtractor<F>
where
    F: Fn(&str, &str) -> Result<String> + Send + Sync,
{
    /// Create a new LLM-based extractor
    ///
    /// The `llm_fn` takes (system_prompt, user_prompt) and returns the response.
    pub fn new(llm_fn: F) -> Self {
        Self {
            llm_fn,
            entity_types: vec![],
            custom_prompt: None,
        }
    }

    /// Focus on specific entity types
    pub fn with_entity_types(mut self, types: Vec<EntityType>) -> Self {
        self.entity_types = types;
        self
    }

    /// Use a custom extraction prompt
    pub fn with_custom_prompt(mut self, prompt: String) -> Self {
        self.custom_prompt = Some(prompt);
        self
    }

    fn build_extraction_prompt(&self, text: &str) -> (String, String) {
        let system_prompt = r#"You are an entity and relation extraction system. Extract structured information from text.

Output Format (JSON):
{
  "entities": [
    {"name": "Entity Name", "type": "organization|product|person|location|concept|event|other", "aliases": ["alt name"]}
  ],
  "relations": [
    {"from": "Entity1", "to": "Entity2", "type": "relation_type", "confidence": 0.9, "evidence": "source text"}
  ]
}

Common relation types:
- manufactures: Organization → Product
- located_in: Entity → Location
- part_of: Component → Whole
- variant_of: Variant → Base
- uses: User → Tool/Weapon
- operates: Organization → Product
- founded_by: Organization → Person
- affiliated_with: Entity → Organization
- related_to: General association

Guidelines:
- Only extract entities explicitly mentioned
- Use canonical names when possible
- Include aliases (abbreviations, nicknames)
- Assign confidence based on how explicit the relation is
- Include evidence text for relations
- Be precise with entity types"#;

        let entity_filter = if self.entity_types.is_empty() {
            String::new()
        } else {
            let types: Vec<&str> = self.entity_types.iter().map(|t| t.as_str()).collect();
            format!("\n\nFocus on these entity types: {}", types.join(", "))
        };

        let user_prompt = if let Some(ref custom) = self.custom_prompt {
            format!("{}\n\nText to analyze:\n{}", custom, text)
        } else {
            format!(
                "Extract entities and relations from this text:{}\n\nText:\n{}",
                entity_filter, text
            )
        };

        (system_prompt.to_string(), user_prompt)
    }

    fn build_query_prompt(&self, query: &str) -> (String, String) {
        let system_prompt = r#"Extract entity names from this query. Return only a JSON array of entity names.
Example: ["Aegis", "Sabre", "Origin"]
Only include proper nouns and specific named entities, not generic terms."#;

        let user_prompt = format!("Query: {}", query);
        (system_prompt.to_string(), user_prompt)
    }

    fn parse_extraction_response(&self, response: &str) -> Result<ExtractionResult> {
        // Try to find JSON in the response
        let json_str = self.extract_json(response)?;

        #[derive(Deserialize)]
        struct RawExtraction {
            entities: Option<Vec<RawEntity>>,
            relations: Option<Vec<RawRelation>>,
        }

        #[derive(Deserialize)]
        struct RawEntity {
            name: String,
            #[serde(rename = "type")]
            entity_type: Option<String>,
            aliases: Option<Vec<String>>,
        }

        #[derive(Deserialize)]
        struct RawRelation {
            from: String,
            to: String,
            #[serde(rename = "type")]
            relation_type: Option<String>,
            confidence: Option<f32>,
            evidence: Option<String>,
        }

        let raw: RawExtraction = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse extraction JSON: {}", e))?;

        let entities = raw
            .entities
            .unwrap_or_default()
            .into_iter()
            .map(|e| ExtractedEntity {
                name: e.name,
                entity_type: e
                    .entity_type
                    .map(|t| EntityType::from_str(&t))
                    .unwrap_or(EntityType::Other),
                aliases: e.aliases.unwrap_or_default(),
                position: None,
                context: None,
            })
            .collect();

        let relations = raw
            .relations
            .unwrap_or_default()
            .into_iter()
            .map(|r| ExtractedRelation {
                from_entity: r.from,
                to_entity: r.to,
                relation_type: r.relation_type.unwrap_or_else(|| "related_to".to_string()),
                confidence: r.confidence.unwrap_or(0.7),
                evidence: r.evidence,
            })
            .collect();

        Ok(ExtractionResult {
            entities,
            relations,
            processing_time_ms: 0,
        })
    }

    fn extract_json(&self, text: &str) -> Result<String> {
        // Try to find JSON object or array
        if let Some(start) = text.find('{') {
            let mut depth = 0;
            let mut end = start;
            for (i, c) in text[start..].chars().enumerate() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth == 0 {
                return Ok(text[start..end].to_string());
            }
        }

        // Try array
        if let Some(start) = text.find('[') {
            let mut depth = 0;
            let mut end = start;
            for (i, c) in text[start..].chars().enumerate() {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth == 0 {
                return Ok(text[start..end].to_string());
            }
        }

        Err(anyhow!("No valid JSON found in response"))
    }
}

impl<F> EntityExtractor for LlmEntityExtractor<F>
where
    F: Fn(&str, &str) -> Result<String> + Send + Sync,
{
    fn extract(&self, text: &str) -> Result<ExtractionResult> {
        let start = Instant::now();
        let (system, user) = self.build_extraction_prompt(text);

        let response = (self.llm_fn)(&system, &user)?;

        let mut result = self.parse_extraction_response(&response)?;
        result.processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    fn extract_query_entities(&self, query: &str) -> Result<Vec<String>> {
        let (system, user) = self.build_query_prompt(query);
        let response = (self.llm_fn)(&system, &user)?;

        let json_str = self.extract_json(&response)?;
        let entities: Vec<String> = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse entity array: {}", e))?;

        Ok(entities)
    }
}

// ============================================================================
// Simple Pattern-based Extractor (no LLM required)
// ============================================================================

/// Simple pattern-based entity extractor (no LLM required)
///
/// Uses regex patterns and a dictionary of known entities.
/// Less accurate but much faster and doesn't require API calls.
pub struct PatternEntityExtractor {
    /// Known entities by name
    known_entities: HashMap<String, EntityType>,
    /// Aliases mapping to canonical names
    aliases: HashMap<String, String>,
    /// Patterns for entity detection
    patterns: Vec<(regex::Regex, EntityType)>,
}

impl PatternEntityExtractor {
    /// Create a new pattern-based extractor
    pub fn new() -> Self {
        Self {
            known_entities: HashMap::new(),
            aliases: HashMap::new(),
            patterns: vec![],
        }
    }

    /// Add a known entity
    pub fn add_entity(mut self, name: &str, entity_type: EntityType) -> Self {
        self.known_entities
            .insert(name.to_lowercase(), entity_type);
        self
    }

    /// Add multiple entities
    pub fn add_entities(mut self, entities: &[(&str, EntityType)]) -> Self {
        for (name, entity_type) in entities {
            self.known_entities
                .insert(name.to_lowercase(), *entity_type);
        }
        self
    }

    /// Add an alias for an entity
    pub fn add_alias(mut self, alias: &str, canonical: &str) -> Self {
        self.aliases
            .insert(alias.to_lowercase(), canonical.to_string());
        self
    }

    /// Add a pattern for entity detection
    pub fn add_pattern(mut self, pattern: &str, entity_type: EntityType) -> Result<Self> {
        let re = regex::Regex::new(pattern)?;
        self.patterns.push((re, entity_type));
        Ok(self)
    }

    fn find_entities_in_text(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut found = HashMap::new();
        let text_lower = text.to_lowercase();

        // Check known entities
        for (name, entity_type) in &self.known_entities {
            if text_lower.contains(name) {
                found.insert(
                    name.clone(),
                    ExtractedEntity {
                        name: name.clone(),
                        entity_type: *entity_type,
                        aliases: vec![],
                        position: text_lower.find(name),
                        context: None,
                    },
                );
            }
        }

        // Check aliases
        for (alias, canonical) in &self.aliases {
            if text_lower.contains(alias) {
                if let Some(entity_type) = self.known_entities.get(&canonical.to_lowercase()) {
                    let entry = found.entry(canonical.to_lowercase()).or_insert_with(|| {
                        ExtractedEntity {
                            name: canonical.clone(),
                            entity_type: *entity_type,
                            aliases: vec![],
                            position: text_lower.find(alias),
                            context: None,
                        }
                    });
                    if !entry.aliases.contains(alias) {
                        entry.aliases.push(alias.clone());
                    }
                }
            }
        }

        // Apply patterns
        for (pattern, entity_type) in &self.patterns {
            for caps in pattern.captures_iter(text) {
                if let Some(m) = caps.get(0) {
                    let name = m.as_str().to_string();
                    found.entry(name.to_lowercase()).or_insert_with(|| {
                        ExtractedEntity {
                            name,
                            entity_type: *entity_type,
                            aliases: vec![],
                            position: Some(m.start()),
                            context: None,
                        }
                    });
                }
            }
        }

        found.into_values().collect()
    }
}

impl Default for PatternEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityExtractor for PatternEntityExtractor {
    fn extract(&self, text: &str) -> Result<ExtractionResult> {
        let start = Instant::now();
        let entities = self.find_entities_in_text(text);

        // Simple relation extraction: if two entities appear in same sentence, they're related
        let mut relations = Vec::new();
        let sentences: Vec<&str> = text.split(['.', '!', '?']).collect();

        for sentence in sentences {
            let sentence_entities = self.find_entities_in_text(sentence);
            for i in 0..sentence_entities.len() {
                for j in (i + 1)..sentence_entities.len() {
                    relations.push(ExtractedRelation {
                        from_entity: sentence_entities[i].name.clone(),
                        to_entity: sentence_entities[j].name.clone(),
                        relation_type: "related_to".to_string(),
                        confidence: 0.5,
                        evidence: Some(sentence.trim().to_string()),
                    });
                }
            }
        }

        Ok(ExtractionResult {
            entities,
            relations,
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn extract_query_entities(&self, query: &str) -> Result<Vec<String>> {
        let entities = self.find_entities_in_text(query);
        Ok(entities.into_iter().map(|e| e.name).collect())
    }
}

// ============================================================================
// Knowledge Graph Store (SQLite)
// ============================================================================

/// SQLite-based knowledge graph storage
///
/// Uses `Mutex<Connection>` to allow `Send + Sync` for use with `GraphCallback`.
pub struct KnowledgeGraphStore {
    conn: Mutex<Connection>,
    config: KnowledgeGraphConfig,
}

// Safety: We wrap Connection in Mutex, making it safe for concurrent access
unsafe impl Send for KnowledgeGraphStore {}
unsafe impl Sync for KnowledgeGraphStore {}

impl KnowledgeGraphStore {
    /// Open or create a knowledge graph database
    pub fn open(path: impl AsRef<Path>, config: KnowledgeGraphConfig) -> Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn: Mutex::new(conn), config };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory knowledge graph (for testing)
    pub fn in_memory(config: KnowledgeGraphConfig) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn: Mutex::new(conn), config };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.execute_batch(
            r#"
            -- Entities table
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                entity_type TEXT NOT NULL,
                aliases TEXT DEFAULT '[]',  -- JSON array
                metadata TEXT DEFAULT '{}', -- JSON object
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

            -- Entity aliases table (for fast lookup)
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT NOT NULL COLLATE NOCASE,
                entity_id INTEGER NOT NULL,
                PRIMARY KEY (alias),
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases(entity_id);

            -- Relations table
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entity_id INTEGER NOT NULL,
                to_entity_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence TEXT,
                source_chunk_id INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (from_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (to_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (source_chunk_id) REFERENCES chunks(id) ON DELETE SET NULL,
                UNIQUE(from_entity_id, to_entity_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

            -- Chunks table
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc TEXT NOT NULL,
                content TEXT NOT NULL,
                position INTEGER NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_doc);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

            -- Entity mentions in chunks
            CREATE TABLE IF NOT EXISTS entity_mentions (
                entity_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                position INTEGER,
                context TEXT,
                mention_count INTEGER DEFAULT 1,
                PRIMARY KEY (entity_id, chunk_id),
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
            CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON entity_mentions(chunk_id);

            -- Full-text search on chunks
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
            "#,
        )?;

        Ok(())
    }

    // === Entity Operations ===

    /// Get or create an entity by name
    pub fn get_or_create_entity(
        &self,
        name: &str,
        entity_type: EntityType,
        aliases: &[String],
    ) -> Result<i64> {
        // First, check if entity exists (uses internal locking)
        if let Some(id) = self.find_entity_id(name)? {
            // Update aliases if needed
            if !aliases.is_empty() {
                self.add_aliases(id, aliases)?;
            }
            return Ok(id);
        }

        // Check aliases
        for alias in aliases {
            if let Some(id) = self.find_entity_id(alias)? {
                self.add_aliases(id, aliases)?;
                return Ok(id);
            }
        }

        // Create new entity
        let aliases_json = serde_json::to_string(aliases)?;
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.execute(
            "INSERT INTO entities (name, entity_type, aliases) VALUES (?1, ?2, ?3)",
            params![name, entity_type.as_str(), aliases_json],
        )?;

        let id = conn.last_insert_rowid();
        drop(conn); // Release lock before calling add_aliases

        // Add aliases to lookup table
        self.add_aliases(id, aliases)?;

        Ok(id)
    }

    /// Find entity ID by name or alias
    pub fn find_entity_id(&self, name: &str) -> Result<Option<i64>> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;

        // Check main name
        let id: Option<i64> = conn
            .query_row(
                "SELECT id FROM entities WHERE name = ?1 COLLATE NOCASE",
                params![name],
                |row| row.get(0),
            )
            .optional()?;

        if id.is_some() {
            return Ok(id);
        }

        // Check aliases
        if self.config.resolve_aliases {
            let id: Option<i64> = conn
                .query_row(
                    "SELECT entity_id FROM entity_aliases WHERE alias = ?1 COLLATE NOCASE",
                    params![name],
                    |row| row.get(0),
                )
                .optional()?;
            return Ok(id);
        }

        Ok(None)
    }

    /// Add aliases to an entity
    fn add_aliases(&self, entity_id: i64, aliases: &[String]) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        for alias in aliases {
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?1, ?2)",
                params![alias, entity_id],
            )?;
        }
        Ok(())
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: i64) -> Result<Option<Entity>> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.query_row(
            r#"SELECT id, name, entity_type, aliases, metadata, created_at, updated_at
               FROM entities WHERE id = ?1"#,
            params![id],
            |row| {
                let aliases_json: String = row.get(3)?;
                let metadata_json: String = row.get(4)?;
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: EntityType::from_str(&row.get::<_, String>(2)?),
                    aliases: serde_json::from_str(&aliases_json).unwrap_or_default(),
                    metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
                    created_at: row.get(5)?,
                    updated_at: row.get(6)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    /// Get entity by name
    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        if let Some(id) = self.find_entity_id(name)? {
            self.get_entity(id)
        } else {
            Ok(None)
        }
    }

    // === Relation Operations ===

    /// Add a relation between entities
    pub fn add_relation(
        &self,
        from_entity_id: i64,
        to_entity_id: i64,
        relation_type: &str,
        confidence: f32,
        evidence: Option<&str>,
        source_chunk_id: Option<i64>,
    ) -> Result<i64> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.execute(
            r#"INSERT INTO relations (from_entity_id, to_entity_id, relation_type, confidence, evidence, source_chunk_id)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6)
               ON CONFLICT(from_entity_id, to_entity_id, relation_type) DO UPDATE SET
                   confidence = MAX(confidence, excluded.confidence),
                   evidence = COALESCE(excluded.evidence, evidence)"#,
            params![
                from_entity_id,
                to_entity_id,
                relation_type,
                confidence,
                evidence,
                source_chunk_id
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get relations from an entity
    pub fn get_relations_from(&self, entity_id: i64, depth: usize) -> Result<Vec<GraphRelation>> {
        let mut relations = Vec::new();
        let mut visited = HashSet::new();
        let mut current_entities = vec![entity_id];

        for _ in 0..depth {
            if current_entities.is_empty() {
                break;
            }

            let mut next_entities = Vec::new();

            for eid in &current_entities {
                if visited.contains(eid) {
                    continue;
                }
                visited.insert(*eid);

                // Collect relation data within the lock scope
                let relation_data: Vec<(String, String, String, f32)> = {
                    let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
                    let mut stmt = conn.prepare(
                        r#"SELECT e1.name, e2.name, r.relation_type, r.confidence
                           FROM relations r
                           JOIN entities e1 ON r.from_entity_id = e1.id
                           JOIN entities e2 ON r.to_entity_id = e2.id
                           WHERE r.from_entity_id = ?1 AND r.confidence >= ?2"#,
                    )?;

                    let rows = stmt.query_map(params![eid, self.config.min_relation_confidence], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, f32>(3)?,
                        ))
                    })?;

                    rows.collect::<std::result::Result<Vec<_>, _>>()?
                };

                for (from, to, rel_type, confidence) in relation_data {
                    relations.push(GraphRelation {
                        from,
                        to: to.clone(),
                        relation_type: rel_type,
                        weight: confidence,
                    });

                    // Add target entity for next depth (uses find_entity_id which has its own lock)
                    if let Some(target_id) = self.find_entity_id(&to)? {
                        if !visited.contains(&target_id) {
                            next_entities.push(target_id);
                        }
                    }
                }
            }

            current_entities = next_entities;
        }

        Ok(relations)
    }

    // === Chunk Operations ===

    /// Add a chunk to the graph
    pub fn add_chunk(&self, source_doc: &str, content: &str, position: usize) -> Result<i64> {
        let content_hash = self.hash_content(content);
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;

        // Check for existing chunk
        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM chunks WHERE content_hash = ?1",
                params![content_hash],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(id) = existing {
            return Ok(id);
        }

        conn.execute(
            "INSERT INTO chunks (source_doc, content, position, content_hash) VALUES (?1, ?2, ?3, ?4)",
            params![source_doc, content, position, content_hash],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Link an entity to a chunk
    pub fn link_entity_to_chunk(
        &self,
        entity_id: i64,
        chunk_id: i64,
        position: Option<usize>,
        context: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.execute(
            r#"INSERT INTO entity_mentions (entity_id, chunk_id, position, context, mention_count)
               VALUES (?1, ?2, ?3, ?4, 1)
               ON CONFLICT(entity_id, chunk_id) DO UPDATE SET
                   mention_count = mention_count + 1"#,
            params![entity_id, chunk_id, position, context],
        )?;
        Ok(())
    }

    /// Get chunks mentioning entities
    pub fn get_chunks_for_entities(&self, entity_ids: &[i64]) -> Result<Vec<RetrievedChunk>> {
        if entity_ids.is_empty() {
            return Ok(vec![]);
        }

        let placeholders: Vec<String> = (0..entity_ids.len()).map(|i| format!("?{}", i + 1)).collect();
        let query = format!(
            r#"SELECT DISTINCT c.id, c.source_doc, c.content, c.position,
                      GROUP_CONCAT(DISTINCT e.name) as entities,
                      SUM(em.mention_count) as total_mentions
               FROM chunks c
               JOIN entity_mentions em ON c.id = em.chunk_id
               JOIN entities e ON em.entity_id = e.id
               WHERE em.entity_id IN ({})
               GROUP BY c.id
               ORDER BY total_mentions DESC
               LIMIT ?{}"#,
            placeholders.join(", "),
            entity_ids.len() + 1
        );

        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        let mut stmt = conn.prepare(&query)?;

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = entity_ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn rusqlite::ToSql>)
            .collect();
        params_vec.push(Box::new(
            self.config.max_chunks_per_entity * entity_ids.len(),
        ));

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            let entities_str: String = row.get(4)?;
            let mention_count: i64 = row.get(5)?;
            Ok(RetrievedChunk {
                chunk_id: format!("graph_{}", row.get::<_, i64>(0)?),
                content: row.get(2)?,
                source: row.get(1)?,
                section: None,
                score: (mention_count as f32 / 10.0).min(1.0), // Normalize mention count
                keyword_score: None,
                semantic_score: None,
                token_count: 0, // Will be calculated if needed
                position: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("entities".to_string(), entities_str);
                    m.insert("position".to_string(), row.get::<_, i64>(3)?.to_string());
                    m.insert("retrieval_method".to_string(), "graph".to_string());
                    m
                },
            })
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn hash_content(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    // === Statistics ===

    /// Get graph statistics
    pub fn get_stats(&self) -> Result<GraphStats> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;

        let total_entities: usize = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;

        let total_relations: usize = conn
            .query_row("SELECT COUNT(*) FROM relations", [], |row| row.get(0))?;

        let total_chunks: usize = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let total_mentions: usize = conn
            .query_row("SELECT COUNT(*) FROM entity_mentions", [], |row| row.get(0))?;

        let mut entities_by_type = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
            })?;
            for row in rows {
                let (t, c) = row?;
                entities_by_type.insert(t, c);
            }
        }

        let mut relations_by_type = HashMap::new();
        {
            let mut stmt = conn
                .prepare("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
            })?;
            for row in rows {
                let (t, c) = row?;
                relations_by_type.insert(t, c);
            }
        }

        Ok(GraphStats {
            total_entities,
            total_relations,
            total_chunks,
            total_mentions,
            entities_by_type,
            relations_by_type,
        })
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock error: {}", e))?;
        conn.execute_batch(
            r#"
            DELETE FROM entity_mentions;
            DELETE FROM relations;
            DELETE FROM entity_aliases;
            DELETE FROM chunks;
            DELETE FROM entities;
            DELETE FROM chunks_fts;
            "#,
        )?;
        Ok(())
    }
}

// ============================================================================
// Knowledge Graph (High-level API)
// ============================================================================

/// High-level knowledge graph interface
///
/// Combines storage, extraction, and querying into a single API.
pub struct KnowledgeGraph {
    store: KnowledgeGraphStore,
    config: KnowledgeGraphConfig,
    /// Cache for entity lookups
    entity_cache: Arc<RwLock<HashMap<String, i64>>>,
}

impl KnowledgeGraph {
    /// Open or create a knowledge graph
    pub fn open(path: impl AsRef<Path>, config: KnowledgeGraphConfig) -> Result<Self> {
        let store = KnowledgeGraphStore::open(path, config.clone())?;
        Ok(Self {
            store,
            config,
            entity_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create an in-memory knowledge graph
    pub fn in_memory(config: KnowledgeGraphConfig) -> Result<Self> {
        let store = KnowledgeGraphStore::in_memory(config.clone())?;
        Ok(Self {
            store,
            config,
            entity_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Index a document into the knowledge graph
    ///
    /// This will:
    /// 1. Split the document into chunks
    /// 2. Extract entities and relations from each chunk
    /// 3. Store everything in the graph
    pub fn index_document(
        &mut self,
        doc_id: &str,
        content: &str,
        extractor: &dyn EntityExtractor,
    ) -> Result<IndexingResult> {
        let start = Instant::now();
        let mut result = IndexingResult::default();

        // Split into chunks
        let chunks = self.chunk_text(content);
        result.chunks_processed = chunks.len();

        for (position, chunk) in chunks.iter().enumerate() {
            // Add chunk to store
            let chunk_id = self.store.add_chunk(doc_id, chunk, position)?;

            // Extract entities and relations
            match extractor.extract(chunk) {
                Ok(extraction) => {
                    // Store entities
                    for entity in &extraction.entities {
                        let entity_id = self.store.get_or_create_entity(
                            &entity.name,
                            entity.entity_type,
                            &entity.aliases,
                        )?;

                        // Link to chunk
                        self.store.link_entity_to_chunk(
                            entity_id,
                            chunk_id,
                            entity.position,
                            entity.context.as_deref(),
                        )?;

                        // Update cache
                        self.entity_cache
                            .write()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(entity.name.to_lowercase(), entity_id);

                        result.entities_extracted += 1;
                    }

                    // Store relations
                    for relation in &extraction.relations {
                        if relation.confidence >= self.config.min_relation_confidence {
                            // Find or create entities
                            let from_id = self.store.get_or_create_entity(
                                &relation.from_entity,
                                EntityType::Other,
                                &[],
                            )?;
                            let to_id = self.store.get_or_create_entity(
                                &relation.to_entity,
                                EntityType::Other,
                                &[],
                            )?;

                            self.store.add_relation(
                                from_id,
                                to_id,
                                &relation.relation_type,
                                relation.confidence,
                                relation.evidence.as_deref(),
                                Some(chunk_id),
                            )?;

                            result.relations_extracted += 1;
                        }
                    }
                }
                Err(e) => {
                    result.errors.push(format!("Chunk {}: {}", position, e));
                }
            }
        }

        result.processing_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Query the graph for entities mentioned in a query
    pub fn query(&self, query: &str, extractor: &dyn EntityExtractor) -> Result<GraphQueryResult> {
        let start = Instant::now();

        // Extract entities from query
        let query_entities = extractor.extract_query_entities(query)?;

        if query_entities.is_empty() {
            return Ok(GraphQueryResult {
                entities_found: vec![],
                relations: vec![],
                chunks: vec![],
                processing_time_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Find entity IDs
        let mut entity_ids = Vec::new();
        let mut entities_found = Vec::new();

        for name in &query_entities {
            if let Some(entity) = self.store.get_entity_by_name(name)? {
                entity_ids.push(entity.id);
                entities_found.push(entity);
            }
        }

        // Get relations
        let mut all_relations = Vec::new();
        let mut related_entity_ids = HashSet::new();

        for id in &entity_ids {
            let relations = self
                .store
                .get_relations_from(*id, self.config.max_traversal_depth)?;
            for rel in &relations {
                if let Some(target_id) = self.store.find_entity_id(&rel.to)? {
                    related_entity_ids.insert(target_id);
                }
            }
            all_relations.extend(relations);
        }

        // Add related entity IDs
        entity_ids.extend(related_entity_ids);
        entity_ids.truncate(self.config.max_entities_per_query);

        // Get chunks
        let chunks = self.store.get_chunks_for_entities(&entity_ids)?;

        Ok(GraphQueryResult {
            entities_found,
            relations: all_relations,
            chunks,
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Create a GraphCallback implementation for use with RagPipeline
    pub fn as_graph_callback<'a>(
        &'a self,
        extractor: &'a dyn EntityExtractor,
    ) -> KnowledgeGraphCallback<'a> {
        KnowledgeGraphCallback {
            graph: self,
            extractor,
        }
    }

    /// Get graph statistics
    pub fn stats(&self) -> Result<GraphStats> {
        self.store.get_stats()
    }

    /// Clear the entire graph
    pub fn clear(&self) -> Result<()> {
        self.store.clear()?;
        self.entity_cache.write().unwrap_or_else(|e| e.into_inner()).clear();
        Ok(())
    }

    fn chunk_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.config.chunk_size).min(chars.len());

            // Try to break at sentence boundary
            let chunk_end = if end < chars.len() {
                let search_start = end.saturating_sub(100);
                let search_range = &chars[search_start..end];
                if let Some(pos) = search_range
                    .iter()
                    .rposition(|&c| c == '.' || c == '!' || c == '?')
                {
                    search_start + pos + 1
                } else {
                    end
                }
            } else {
                end
            };

            let chunk: String = chars[start..chunk_end].iter().collect();
            if !chunk.trim().is_empty() {
                chunks.push(chunk);
            }

            start = if chunk_end > start + self.config.chunk_overlap {
                chunk_end - self.config.chunk_overlap
            } else {
                chunk_end
            };
        }

        chunks
    }
}

/// Result of document indexing
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexingResult {
    pub chunks_processed: usize,
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub processing_time_ms: u64,
    pub errors: Vec<String>,
}

/// Result of a graph query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphQueryResult {
    pub entities_found: Vec<Entity>,
    pub relations: Vec<GraphRelation>,
    pub chunks: Vec<RetrievedChunk>,
    pub processing_time_ms: u64,
}

// ============================================================================
// GraphCallback Implementation
// ============================================================================

/// Adapter that implements GraphCallback for use with RagPipeline
pub struct KnowledgeGraphCallback<'a> {
    graph: &'a KnowledgeGraph,
    extractor: &'a dyn EntityExtractor,
}

impl<'a> GraphCallback for KnowledgeGraphCallback<'a> {
    fn extract_entities(&self, text: &str) -> std::result::Result<Vec<String>, String> {
        self.extractor
            .extract_query_entities(text)
            .map_err(|e| e.to_string())
    }

    fn get_related(
        &self,
        entity: &str,
        depth: usize,
    ) -> std::result::Result<Vec<GraphRelation>, String> {
        let entity_id = self
            .graph
            .store
            .find_entity_id(entity)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Entity not found: {}", entity))?;

        self.graph
            .store
            .get_relations_from(entity_id, depth)
            .map_err(|e| e.to_string())
    }

    fn get_entity_chunks(
        &self,
        entities: &[String],
    ) -> std::result::Result<Vec<RetrievedChunk>, String> {
        let mut entity_ids = Vec::new();

        for name in entities {
            if let Some(id) = self
                .graph
                .store
                .find_entity_id(name)
                .map_err(|e| e.to_string())?
            {
                entity_ids.push(id);
            }
        }

        self.graph
            .store
            .get_chunks_for_entities(&entity_ids)
            .map_err(|e| e.to_string())
    }
}

// ============================================================================
// Builder for Easy Construction
// ============================================================================

/// Builder for creating a KnowledgeGraph with common entity dictionaries
pub struct KnowledgeGraphBuilder {
    config: KnowledgeGraphConfig,
    known_entities: Vec<(String, EntityType)>,
    aliases: Vec<(String, String)>,
}

impl KnowledgeGraphBuilder {
    pub fn new() -> Self {
        Self {
            config: KnowledgeGraphConfig::default(),
            known_entities: vec![],
            aliases: vec![],
        }
    }

    pub fn with_config(mut self, config: KnowledgeGraphConfig) -> Self {
        self.config = config;
        self
    }

    /// Add Star Citizen specific entities
    pub fn with_star_citizen_entities(mut self) -> Self {
        // Manufacturers
        let manufacturers = [
            "Aegis Dynamics",
            "Anvil Aerospace",
            "Aopoa",
            "Argo Astronautics",
            "Banu",
            "CNOU",
            "Consolidated Outland",
            "Crusader Industries",
            "Drake Interplanetary",
            "Esperia",
            "Gatac",
            "Greycat Industrial",
            "Kruger Intergalactic",
            "MISC",
            "Musashi Industrial",
            "Origin Jumpworks",
            "Roberts Space Industries",
            "RSI",
            "Tumbril",
            "Vanduul",
        ];
        for m in manufacturers {
            self.known_entities
                .push((m.to_string(), EntityType::Organization));
        }

        // Add aliases
        self.aliases
            .push(("RSI".to_string(), "Roberts Space Industries".to_string()));
        self.aliases
            .push(("MISC".to_string(), "Musashi Industrial".to_string()));
        self.aliases.push((
            "Consolidated Outland".to_string(),
            "CNOU".to_string(),
        ));

        self
    }

    /// Add custom entities
    pub fn add_entity(mut self, name: &str, entity_type: EntityType) -> Self {
        self.known_entities.push((name.to_string(), entity_type));
        self
    }

    /// Add an alias
    pub fn add_alias(mut self, alias: &str, canonical: &str) -> Self {
        self.aliases.push((alias.to_string(), canonical.to_string()));
        self
    }

    /// Build the knowledge graph at the given path
    pub fn build(self, path: impl AsRef<Path>) -> Result<KnowledgeGraph> {
        let graph = KnowledgeGraph::open(path, self.config)?;

        // Pre-populate with known entities
        for (name, entity_type) in &self.known_entities {
            graph.store.get_or_create_entity(name, *entity_type, &[])?;
        }

        // Add aliases
        for (alias, canonical) in &self.aliases {
            if let Some(entity_id) = graph.store.find_entity_id(canonical)? {
                graph.store.add_aliases(entity_id, &[alias.clone()])?;
            }
        }

        Ok(graph)
    }

    /// Build an in-memory knowledge graph
    pub fn build_in_memory(self) -> Result<KnowledgeGraph> {
        let graph = KnowledgeGraph::in_memory(self.config)?;

        for (name, entity_type) in &self.known_entities {
            graph.store.get_or_create_entity(name, *entity_type, &[])?;
        }

        for (alias, canonical) in &self.aliases {
            if let Some(entity_id) = graph.store.find_entity_id(canonical)? {
                graph.store.add_aliases(entity_id, &[alias.clone()])?;
            }
        }

        Ok(graph)
    }
}

impl Default for KnowledgeGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_conversion() {
        assert_eq!(EntityType::from_str("organization"), EntityType::Organization);
        assert_eq!(EntityType::from_str("ship"), EntityType::Product);
        assert_eq!(EntityType::from_str("unknown"), EntityType::Other);
        assert_eq!(EntityType::Organization.as_str(), "organization");
    }

    #[test]
    fn test_pattern_extractor() {
        let extractor = PatternEntityExtractor::new()
            .add_entity("Aegis", EntityType::Organization)
            .add_entity("Sabre", EntityType::Product)
            .add_alias("Aegis Dynamics", "Aegis");

        let text = "Aegis Dynamics manufactures the Sabre fighter.";
        let result = extractor.extract(text).unwrap();

        assert!(!result.entities.is_empty());
        assert!(result.entities.iter().any(|e| e.name.to_lowercase() == "aegis"));
        assert!(result.entities.iter().any(|e| e.name.to_lowercase() == "sabre"));
    }

    #[test]
    fn test_knowledge_graph_store() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        // Create entity
        let id = store
            .get_or_create_entity("Aegis", EntityType::Organization, &["Aegis Dynamics".to_string()])
            .unwrap();
        assert!(id > 0);

        // Find by name
        assert_eq!(store.find_entity_id("Aegis").unwrap(), Some(id));

        // Find by alias
        assert_eq!(store.find_entity_id("Aegis Dynamics").unwrap(), Some(id));

        // Get entity
        let entity = store.get_entity(id).unwrap().unwrap();
        assert_eq!(entity.name, "Aegis");
        assert_eq!(entity.entity_type, EntityType::Organization);
    }

    #[test]
    fn test_knowledge_graph_relations() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let aegis_id = store
            .get_or_create_entity("Aegis", EntityType::Organization, &[])
            .unwrap();
        let sabre_id = store
            .get_or_create_entity("Sabre", EntityType::Product, &[])
            .unwrap();

        store
            .add_relation(aegis_id, sabre_id, "manufactures", 0.9, Some("Aegis makes the Sabre"), None)
            .unwrap();

        let relations = store.get_relations_from(aegis_id, 1).unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].from, "Aegis");
        assert_eq!(relations[0].to, "Sabre");
        assert_eq!(relations[0].relation_type, "manufactures");
    }

    #[test]
    fn test_knowledge_graph_chunks() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        // Add chunk
        let chunk_id = store
            .add_chunk("doc1", "Aegis manufactures the Sabre fighter.", 0)
            .unwrap();

        // Add entity and link
        let aegis_id = store
            .get_or_create_entity("Aegis", EntityType::Organization, &[])
            .unwrap();
        store
            .link_entity_to_chunk(aegis_id, chunk_id, Some(0), Some("Aegis manufactures"))
            .unwrap();

        // Query chunks
        let chunks = store.get_chunks_for_entities(&[aegis_id]).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Aegis"));
    }

    #[test]
    fn test_knowledge_graph_builder() {
        let graph = KnowledgeGraphBuilder::new()
            .with_star_citizen_entities()
            .add_entity("Custom Entity", EntityType::Concept)
            .build_in_memory()
            .unwrap();

        let stats = graph.stats().unwrap();
        assert!(stats.total_entities > 0);

        // Check RSI alias works
        let rsi = graph.store.get_entity_by_name("RSI").unwrap();
        assert!(rsi.is_some());
    }

    #[test]
    fn test_chunking() {
        let config = KnowledgeGraphConfig {
            chunk_size: 100,
            chunk_overlap: 20,
            ..Default::default()
        };
        let graph = KnowledgeGraph::in_memory(config).unwrap();

        let text = "This is a test sentence. ".repeat(20);
        let chunks = graph.chunk_text(&text);

        assert!(chunks.len() > 1);
        // Check overlap exists
        if chunks.len() >= 2 {
            let end_of_first = &chunks[0][chunks[0].len().saturating_sub(20)..];
            assert!(chunks[1].contains(end_of_first.trim()));
        }
    }

    #[test]
    fn test_graph_stats() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        store
            .get_or_create_entity("Aegis", EntityType::Organization, &[])
            .unwrap();
        store
            .get_or_create_entity("Origin", EntityType::Organization, &[])
            .unwrap();
        store
            .get_or_create_entity("Sabre", EntityType::Product, &[])
            .unwrap();

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_entities, 3);
        assert_eq!(stats.entities_by_type.get("organization"), Some(&2));
        assert_eq!(stats.entities_by_type.get("product"), Some(&1));
    }
}
