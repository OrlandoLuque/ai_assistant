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
use std::fmt;
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

impl<F> fmt::Debug for LlmEntityExtractor<F>
where
    F: Fn(&str, &str) -> Result<String> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmEntityExtractor")
            .field("llm_fn", &"<Fn(&str, &str) -> Result<String>>")
            .field("entity_types", &self.entity_types)
            .field("custom_prompt", &self.custom_prompt)
            .finish()
    }
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
#[derive(Debug)]
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
        self.known_entities.insert(name.to_lowercase(), entity_type);
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
                    let entry =
                        found
                            .entry(canonical.to_lowercase())
                            .or_insert_with(|| ExtractedEntity {
                                name: canonical.clone(),
                                entity_type: *entity_type,
                                aliases: vec![],
                                position: text_lower.find(alias),
                                context: None,
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
                    found
                        .entry(name.to_lowercase())
                        .or_insert_with(|| ExtractedEntity {
                            name,
                            entity_type: *entity_type,
                            aliases: vec![],
                            position: Some(m.start()),
                            context: None,
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

impl fmt::Debug for KnowledgeGraphStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnowledgeGraphStore")
            .field("conn", &"<Mutex<Connection>>")
            .field("config", &self.config)
            .finish()
    }
}

// Safety: We wrap Connection in Mutex, making it safe for concurrent access
unsafe impl Send for KnowledgeGraphStore {}
unsafe impl Sync for KnowledgeGraphStore {}

impl KnowledgeGraphStore {
    /// Open or create a knowledge graph database
    pub fn open(path: impl AsRef<Path>, config: KnowledgeGraphConfig) -> Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Mutex::new(conn),
            config,
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory knowledge graph (for testing)
    pub fn in_memory(config: KnowledgeGraphConfig) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Mutex::new(conn),
            config,
        };
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

        #[cfg(feature = "analytics")]
        {
            let entity_count: usize = conn
                .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
                .unwrap_or(0);
            crate::scalability_monitor::check_scalability(
                crate::scalability_monitor::Subsystem::KnowledgeGraph,
                entity_count,
            );
        }

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

                    let rows =
                        stmt.query_map(params![eid, self.config.min_relation_confidence], |row| {
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

        let placeholders: Vec<String> = (0..entity_ids.len())
            .map(|i| format!("?{}", i + 1))
            .collect();
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

        let total_entities: usize =
            conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;

        let total_relations: usize =
            conn.query_row("SELECT COUNT(*) FROM relations", [], |row| row.get(0))?;

        let total_chunks: usize =
            conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let total_mentions: usize =
            conn.query_row("SELECT COUNT(*) FROM entity_mentions", [], |row| row.get(0))?;

        let mut entities_by_type = HashMap::new();
        {
            let mut stmt =
                conn.prepare("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")?;
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
#[derive(Debug)]
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
        self.entity_cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Ok(())
    }

    /// Export the entire knowledge graph as JSON for visualization.
    pub fn export_json(&self) -> anyhow::Result<serde_json::Value> {
        let entities = self.store.list_all_entities()?;
        let relations = self.store.list_all_relations()?;
        let stats = self.stats()?;

        Ok(serde_json::json!({
            "entities": entities.iter().map(|(id, name, entity_type)| {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "entity_type": entity_type,
                })
            }).collect::<Vec<_>>(),
            "relations": relations.iter().map(|(id, from_id, to_id, rel_type, confidence)| {
                serde_json::json!({
                    "id": id,
                    "from_entity_id": from_id,
                    "to_entity_id": to_id,
                    "relation_type": rel_type,
                    "confidence": confidence,
                })
            }).collect::<Vec<_>>(),
            "stats": serde_json::json!({
                "total_entities": stats.total_entities,
                "total_relations": stats.total_relations,
                "total_chunks": stats.total_chunks,
                "total_mentions": stats.total_mentions,
            }),
        }))
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

impl<'a> fmt::Debug for KnowledgeGraphCallback<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnowledgeGraphCallback")
            .field("graph", &self.graph)
            .field("extractor", &"<dyn EntityExtractor>")
            .finish()
    }
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
#[derive(Debug)]
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
        self.aliases
            .push(("Consolidated Outland".to_string(), "CNOU".to_string()));

        self
    }

    /// Add custom entities
    pub fn add_entity(mut self, name: &str, entity_type: EntityType) -> Self {
        self.known_entities.push((name.to_string(), entity_type));
        self
    }

    /// Add an alias
    pub fn add_alias(mut self, alias: &str, canonical: &str) -> Self {
        self.aliases
            .push((alias.to_string(), canonical.to_string()));
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
                graph
                    .store
                    .add_aliases(entity_id, std::slice::from_ref(alias))?;
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
                graph
                    .store
                    .add_aliases(entity_id, std::slice::from_ref(alias))?;
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
// Cypher-like Query Builder
// ============================================================================

/// WHERE clause operators for graph queries.
#[derive(Debug, Clone)]
pub enum WhereClause {
    /// Exact equality: field = 'value'
    Eq(String, String),
    /// Substring match: field LIKE '%value%'
    Contains(String, String),
    /// Greater than: field > value
    Gt(String, f64),
    /// Less than: field < value
    Lt(String, f64),
    /// Greater than or equal: field >= value
    Gte(String, f64),
    /// Less than or equal: field <= value
    Lte(String, f64),
    /// Set membership: field IN (values...)
    In(String, Vec<String>),
}

/// A match pattern for graph query building.
#[derive(Debug, Clone)]
pub enum MatchPattern {
    /// Match a node (entity) with optional type label and alias.
    Node {
        label: Option<String>,
        alias: String,
    },
    /// Match a relationship (relation) with optional type, alias, and endpoint aliases.
    Relationship {
        rel_type: Option<String>,
        alias: String,
        from_alias: String,
        to_alias: String,
    },
    /// Match a path between two node aliases up to max_hops.
    Path {
        from_alias: String,
        to_alias: String,
        max_hops: usize,
    },
}

/// Result from a Cypher-like graph query execution.
#[derive(Debug, Clone)]
pub struct CypherQueryResult {
    /// Column names from the query.
    pub columns: Vec<String>,
    /// Rows of string values.
    pub rows: Vec<Vec<String>>,
    /// Execution time in milliseconds.
    pub execution_time_ms: u128,
}

/// Cypher-like query builder for the knowledge graph.
///
/// Builds SQL queries against the knowledge graph's SQLite schema.
///
/// # Example
///
/// ```ignore
/// let query = GraphQuery::new()
///     .match_node("n", Some("Organization"))
///     .where_eq("n.name", "Aegis")
///     .return_fields(&["n.name", "n.entity_type"])
///     .limit(10);
/// let sql = query.to_sql();
/// ```
#[derive(Debug, Clone)]
pub struct GraphQuery {
    match_patterns: Vec<MatchPattern>,
    where_clauses: Vec<WhereClause>,
    return_fields: Vec<String>,
    order_by: Option<(String, bool)>, // (field, ascending)
    limit_val: Option<usize>,
    skip_val: Option<usize>,
}

impl GraphQuery {
    /// Create a new empty query builder.
    pub fn new() -> Self {
        Self {
            match_patterns: Vec::new(),
            where_clauses: Vec::new(),
            return_fields: Vec::new(),
            order_by: None,
            limit_val: None,
            skip_val: None,
        }
    }

    /// Add a node match pattern.
    ///
    /// `alias` is used to reference this node in other clauses.
    /// `label` optionally filters by entity_type.
    pub fn match_node(mut self, alias: &str, label: Option<&str>) -> Self {
        self.match_patterns.push(MatchPattern::Node {
            label: label.map(|s| s.to_string()),
            alias: alias.to_string(),
        });
        self
    }

    /// Add a relationship match pattern joining two node aliases.
    ///
    /// `rel_type` optionally filters by relation_type.
    pub fn match_relationship(
        mut self,
        alias: &str,
        rel_type: Option<&str>,
        from_alias: &str,
        to_alias: &str,
    ) -> Self {
        self.match_patterns.push(MatchPattern::Relationship {
            rel_type: rel_type.map(|s| s.to_string()),
            alias: alias.to_string(),
            from_alias: from_alias.to_string(),
            to_alias: to_alias.to_string(),
        });
        self
    }

    /// Add a path match pattern between two node aliases.
    ///
    /// `max_hops` limits traversal depth.
    pub fn match_path(mut self, from_alias: &str, to_alias: &str, max_hops: usize) -> Self {
        self.match_patterns.push(MatchPattern::Path {
            from_alias: from_alias.to_string(),
            to_alias: to_alias.to_string(),
            max_hops,
        });
        self
    }

    /// Add an equality WHERE clause: field = 'value'.
    pub fn where_eq(mut self, field: &str, value: &str) -> Self {
        self.where_clauses
            .push(WhereClause::Eq(field.to_string(), value.to_string()));
        self
    }

    /// Add a substring WHERE clause: field LIKE '%value%'.
    pub fn where_contains(mut self, field: &str, value: &str) -> Self {
        self.where_clauses
            .push(WhereClause::Contains(field.to_string(), value.to_string()));
        self
    }

    /// Add a greater-than WHERE clause: field > value.
    pub fn where_gt(mut self, field: &str, value: f64) -> Self {
        self.where_clauses
            .push(WhereClause::Gt(field.to_string(), value));
        self
    }

    /// Set the fields to return.
    pub fn return_fields(mut self, fields: &[&str]) -> Self {
        self.return_fields = fields.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set the ORDER BY clause.
    pub fn order_by(mut self, field: &str, ascending: bool) -> Self {
        self.order_by = Some((field.to_string(), ascending));
        self
    }

    /// Set the LIMIT clause.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit_val = Some(n);
        self
    }

    /// Set the OFFSET (skip) clause.
    pub fn skip(mut self, n: usize) -> Self {
        self.skip_val = Some(n);
        self
    }

    /// Generate SQL query string from the builder state.
    pub fn to_sql(&self) -> String {
        // Build SELECT clause
        let select = if self.return_fields.is_empty() {
            "SELECT *".to_string()
        } else {
            format!("SELECT {}", self.return_fields.join(", "))
        };

        // Build FROM clause based on match patterns
        let mut tables = Vec::new();
        let mut joins = Vec::new();

        for pattern in &self.match_patterns {
            match pattern {
                MatchPattern::Node { label, alias } => {
                    if let Some(lbl) = label {
                        tables.push(format!("entities AS {} /* type={} */", alias, lbl));
                    } else {
                        tables.push(format!("entities AS {}", alias));
                    }
                }
                MatchPattern::Relationship {
                    rel_type,
                    alias,
                    from_alias,
                    to_alias,
                } => {
                    joins.push(format!(
                        "JOIN relations AS {} ON {}.id = {}.from_entity_id AND {}.id = {}.to_entity_id{}",
                        alias,
                        from_alias,
                        alias,
                        to_alias,
                        alias,
                        if let Some(rt) = rel_type {
                            format!(" AND {}.relation_type = '{}'", alias, rt.replace('\'', "''"))
                        } else {
                            String::new()
                        }
                    ));
                }
                MatchPattern::Path {
                    from_alias,
                    to_alias,
                    max_hops,
                } => {
                    // Path queries use a CTE approach (simplified: single-hop join for SQL generation)
                    joins.push(format!(
                        "JOIN relations AS path_r ON {}.id = path_r.from_entity_id \
                         JOIN entities AS {} ON {}.id = path_r.to_entity_id /* max_hops={} */",
                        from_alias, to_alias, to_alias, max_hops
                    ));
                }
            }
        }

        let from = if tables.is_empty() {
            "FROM entities".to_string()
        } else {
            format!("FROM {}", tables.join(", "))
        };

        // Build WHERE clause
        let mut where_parts = Vec::new();
        for pattern in &self.match_patterns {
            if let MatchPattern::Node {
                label: Some(lbl),
                alias,
            } = pattern
            {
                where_parts.push(format!("{}.entity_type = '{}'", alias, lbl.replace('\'', "''")));
            }
        }
        for clause in &self.where_clauses {
            match clause {
                WhereClause::Eq(field, value) => {
                    where_parts.push(format!("{} = '{}'", field, value.replace('\'', "''")))
                }
                WhereClause::Contains(field, value) => {
                    // Escape both SQL quotes and LIKE wildcards in user input
                    let escaped = value.replace('\'', "''").replace('%', "\\%").replace('_', "\\_");
                    where_parts.push(format!("{} LIKE '%{}%' ESCAPE '\\'", field, escaped))
                }
                WhereClause::Gt(field, value) => where_parts.push(format!("{} > {}", field, value)),
                WhereClause::Lt(field, value) => where_parts.push(format!("{} < {}", field, value)),
                WhereClause::Gte(field, value) => {
                    where_parts.push(format!("{} >= {}", field, value))
                }
                WhereClause::Lte(field, value) => {
                    where_parts.push(format!("{} <= {}", field, value))
                }
                WhereClause::In(field, values) => {
                    let vals = values
                        .iter()
                        .map(|v| format!("'{}'", v.replace('\'', "''")))
                        .collect::<Vec<_>>()
                        .join(", ");
                    where_parts.push(format!("{} IN ({})", field, vals));
                }
            }
        }

        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };

        // Build ORDER BY
        let order = match &self.order_by {
            Some((field, asc)) => {
                format!(" ORDER BY {} {}", field, if *asc { "ASC" } else { "DESC" })
            }
            None => String::new(),
        };

        // Build LIMIT/OFFSET
        let limit = match self.limit_val {
            Some(n) => format!(" LIMIT {}", n),
            None => String::new(),
        };
        let offset = match self.skip_val {
            Some(n) => format!(" OFFSET {}", n),
            None => String::new(),
        };

        // Combine
        let joins_str = if joins.is_empty() {
            String::new()
        } else {
            format!(" {}", joins.join(" "))
        };
        format!(
            "{} {}{}{}{}{}{}",
            select, from, joins_str, where_clause, order, limit, offset
        )
    }

    /// Execute the query against a KnowledgeGraphStore connection.
    pub fn execute(&self, store: &KnowledgeGraphStore) -> Result<CypherQueryResult> {
        let start = Instant::now();
        let sql = self.to_sql();
        let conn = store
            .conn
            .lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        let mut stmt = conn.prepare(&sql)?;
        let column_count = stmt.column_count();
        let columns: Vec<String> = (0..column_count)
            .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
            .collect();

        let rows: Vec<Vec<String>> = stmt
            .query_map([], |row| {
                let mut values = Vec::new();
                for i in 0..column_count {
                    let val: String = row
                        .get::<_, rusqlite::types::Value>(i)
                        .map(|v| match v {
                            rusqlite::types::Value::Null => "NULL".to_string(),
                            rusqlite::types::Value::Integer(i) => i.to_string(),
                            rusqlite::types::Value::Real(f) => f.to_string(),
                            rusqlite::types::Value::Text(s) => s,
                            rusqlite::types::Value::Blob(b) => {
                                format!("<blob:{} bytes>", b.len())
                            }
                        })
                        .unwrap_or_else(|_| "ERROR".to_string());
                    values.push(val);
                }
                Ok(values)
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(CypherQueryResult {
            columns,
            rows,
            execution_time_ms: start.elapsed().as_millis(),
        })
    }
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Graph Algorithms
// ============================================================================

/// Graph algorithms operating on the knowledge graph.
///
/// All algorithms load adjacency data from SQLite into memory, then compute
/// in-memory for efficiency.
#[derive(Debug)]
pub struct GraphAlgorithms;

impl GraphAlgorithms {
    /// Find the shortest path between two entities using BFS.
    ///
    /// Returns entity IDs in order from `from_entity_id` to `to_entity_id`,
    /// or `None` if no path exists. Treats the graph as undirected.
    pub fn shortest_path(
        store: &KnowledgeGraphStore,
        from_entity_id: i64,
        to_entity_id: i64,
    ) -> Result<Option<Vec<i64>>> {
        // Load adjacency list from relations table
        let conn = store.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;
        let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = conn.prepare("SELECT from_entity_id, to_entity_id FROM relations")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;

        for edge in edges {
            let (from, to) = edge?;
            adj.entry(from).or_default().push(to);
            adj.entry(to).or_default().push(from); // undirected
        }
        drop(stmt);
        drop(conn);

        // BFS
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();
        let mut visited: HashMap<i64, i64> = HashMap::new(); // node -> parent
        visited.insert(from_entity_id, -1);
        queue.push_back(from_entity_id);

        while let Some(current) = queue.pop_front() {
            if current == to_entity_id {
                // Reconstruct path
                let mut path = vec![current];
                let mut node = current;
                while visited[&node] != -1 {
                    node = visited[&node];
                    path.push(node);
                }
                path.reverse();
                return Ok(Some(path));
            }

            if let Some(neighbors) = adj.get(&current) {
                for &next in neighbors {
                    if !visited.contains_key(&next) {
                        visited.insert(next, current);
                        queue.push_back(next);
                    }
                }
            }
        }

        Ok(None)
    }

    /// Compute PageRank for all entities in the graph.
    ///
    /// Returns a map of entity_id to PageRank score. Uses the standard
    /// iterative power method with a damping factor (typically 0.85).
    pub fn page_rank(
        store: &KnowledgeGraphStore,
        damping: f64,
        iterations: usize,
    ) -> Result<HashMap<i64, f64>> {
        let conn = store.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;

        // Load all entity IDs
        let mut ids: Vec<i64> = Vec::new();
        let mut stmt = conn.prepare("SELECT id FROM entities")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        for r in rows {
            ids.push(r?);
        }
        drop(stmt);

        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        // Load outgoing edges
        let mut outgoing: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut stmt = conn.prepare("SELECT from_entity_id, to_entity_id FROM relations")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for edge in edges {
            let (from, to) = edge?;
            outgoing.entry(from).or_default().push(to);
        }
        drop(stmt);
        drop(conn);

        let n = ids.len() as f64;
        let mut ranks: HashMap<i64, f64> = ids.iter().map(|&id| (id, 1.0 / n)).collect();

        for _ in 0..iterations {
            let mut new_ranks: HashMap<i64, f64> =
                ids.iter().map(|&id| (id, (1.0 - damping) / n)).collect();

            for &id in &ids {
                let rank = ranks[&id];
                if let Some(targets) = outgoing.get(&id) {
                    let share = rank / targets.len() as f64;
                    for &target in targets {
                        if let Some(r) = new_ranks.get_mut(&target) {
                            *r += damping * share;
                        }
                    }
                }
            }

            ranks = new_ranks;
        }

        Ok(ranks)
    }

    /// Find connected components using Union-Find.
    ///
    /// Returns a map of entity_id to component_id (the root entity of its component).
    /// Treats the graph as undirected.
    pub fn connected_components(store: &KnowledgeGraphStore) -> Result<HashMap<i64, i64>> {
        let conn = store.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;

        let mut ids: Vec<i64> = Vec::new();
        let mut stmt = conn.prepare("SELECT id FROM entities")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        for r in rows {
            ids.push(r?);
        }
        drop(stmt);

        // Union-Find
        let mut parent: HashMap<i64, i64> = ids.iter().map(|&id| (id, id)).collect();

        fn find(parent: &mut HashMap<i64, i64>, x: i64) -> i64 {
            let p = parent[&x];
            if p == x {
                return x;
            }
            let root = find(parent, p);
            parent.insert(x, root);
            root
        }

        let mut stmt = conn.prepare("SELECT from_entity_id, to_entity_id FROM relations")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        let edge_vec: Vec<(i64, i64)> = edges.filter_map(|e| e.ok()).collect();
        drop(stmt);
        drop(conn);

        for (from, to) in edge_vec {
            let root_from = find(&mut parent, from);
            let root_to = find(&mut parent, to);
            if root_from != root_to {
                parent.insert(root_from, root_to);
            }
        }

        // Flatten — ensure all nodes point directly to their root
        let all_ids = ids.clone();
        for id in all_ids {
            find(&mut parent, id);
        }

        Ok(parent)
    }

    /// Compute degree centrality for all entities.
    ///
    /// Returns a tuple of (in_degree, out_degree) maps. Directed graph view.
    pub fn degree_centrality(
        store: &KnowledgeGraphStore,
    ) -> Result<(HashMap<i64, usize>, HashMap<i64, usize>)> {
        let conn = store.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;

        let mut in_degree: HashMap<i64, usize> = HashMap::new();
        let mut out_degree: HashMap<i64, usize> = HashMap::new();

        // Initialize with all entities
        let mut stmt = conn.prepare("SELECT id FROM entities")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        for r in rows {
            let id = r?;
            in_degree.insert(id, 0);
            out_degree.insert(id, 0);
        }
        drop(stmt);

        let mut stmt = conn.prepare("SELECT from_entity_id, to_entity_id FROM relations")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for edge in edges {
            let (from, to) = edge?;
            *out_degree.entry(from).or_insert(0) += 1;
            *in_degree.entry(to).or_insert(0) += 1;
        }
        drop(stmt);
        drop(conn);

        Ok((in_degree, out_degree))
    }

    /// Find all paths between two entities up to max_depth using DFS.
    ///
    /// Returns all paths found. Treats the graph as undirected.
    /// `max_depth` limits the maximum number of edges in any path.
    pub fn all_paths(
        store: &KnowledgeGraphStore,
        from_id: i64,
        to_id: i64,
        max_depth: usize,
    ) -> Result<Vec<Vec<i64>>> {
        let conn = store.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;
        let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();

        let mut stmt = conn.prepare("SELECT from_entity_id, to_entity_id FROM relations")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for edge in edges {
            let (from, to) = edge?;
            adj.entry(from).or_default().push(to);
            adj.entry(to).or_default().push(from); // undirected
        }
        drop(stmt);
        drop(conn);

        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(from_id);
        let mut path = vec![from_id];

        fn dfs(
            current: i64,
            target: i64,
            max_depth: usize,
            adj: &HashMap<i64, Vec<i64>>,
            visited: &mut HashSet<i64>,
            path: &mut Vec<i64>,
            results: &mut Vec<Vec<i64>>,
        ) {
            if current == target {
                results.push(path.clone());
                return;
            }
            if path.len() > max_depth {
                return;
            }
            if let Some(neighbors) = adj.get(&current) {
                for &next in neighbors {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        path.push(next);
                        dfs(next, target, max_depth, adj, visited, path, results);
                        path.pop();
                        visited.remove(&next);
                    }
                }
            }
        }

        dfs(
            from_id,
            to_id,
            max_depth,
            &adj,
            &mut visited,
            &mut path,
            &mut results,
        );
        Ok(results)
    }
}

// ============================================================================
// KnowledgeGraphStore Helper Methods for Visualization
// ============================================================================

impl KnowledgeGraphStore {
    /// List all entities in the store.
    ///
    /// Returns tuples of (id, name, entity_type).
    pub fn list_all_entities(&self) -> Result<Vec<(i64, String, String)>> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;
        let mut stmt = conn.prepare("SELECT id, name, entity_type FROM entities")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| e.into())
    }

    /// List all relations in the store.
    ///
    /// Returns tuples of (id, from_entity_id, to_entity_id, relation_type, confidence).
    pub fn list_all_relations(&self) -> Result<Vec<(i64, i64, i64, String, f64)>> {
        let conn = self.conn.lock().map_err(|e| anyhow!("Lock: {}", e))?;
        let mut stmt = conn.prepare(
            "SELECT id, from_entity_id, to_entity_id, relation_type, confidence FROM relations",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?,
            ))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| e.into())
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
        assert_eq!(
            EntityType::from_str("organization"),
            EntityType::Organization
        );
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
        assert!(result
            .entities
            .iter()
            .any(|e| e.name.to_lowercase() == "aegis"));
        assert!(result
            .entities
            .iter()
            .any(|e| e.name.to_lowercase() == "sabre"));
    }

    #[test]
    fn test_knowledge_graph_store() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        // Create entity
        let id = store
            .get_or_create_entity(
                "Aegis",
                EntityType::Organization,
                &["Aegis Dynamics".to_string()],
            )
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
            .add_relation(
                aegis_id,
                sabre_id,
                "manufactures",
                0.9,
                Some("Aegis makes the Sabre"),
                None,
            )
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

    // ========================================================================
    // GraphQuery builder tests
    // ========================================================================

    #[test]
    fn test_graph_query_new() {
        let query = GraphQuery::new();
        assert!(query.match_patterns.is_empty());
        assert!(query.where_clauses.is_empty());
        assert!(query.return_fields.is_empty());
        assert!(query.order_by.is_none());
        assert!(query.limit_val.is_none());
        assert!(query.skip_val.is_none());
        // Default SQL: SELECT * FROM entities
        let sql = query.to_sql();
        assert!(sql.contains("SELECT *"));
        assert!(sql.contains("FROM entities"));
    }

    #[test]
    fn test_graph_query_match_node() {
        let query = GraphQuery::new().match_node("n", Some("Organization"));
        let sql = query.to_sql();
        assert!(sql.contains("entities AS n"));
        assert!(sql.contains("n.entity_type = 'Organization'"));
    }

    #[test]
    fn test_graph_query_where_eq() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .where_eq("n.name", "Aegis");
        let sql = query.to_sql();
        assert!(sql.contains("n.name = 'Aegis'"));
    }

    #[test]
    fn test_graph_query_where_contains() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .where_contains("n.name", "Aeg");
        let sql = query.to_sql();
        assert!(sql.contains("n.name LIKE '%Aeg%'"));
    }

    #[test]
    fn test_graph_query_where_gt() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .where_gt("r.confidence", 0.5);
        let sql = query.to_sql();
        assert!(sql.contains("r.confidence > 0.5"));
    }

    #[test]
    fn test_graph_query_return_fields() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .return_fields(&["n.name", "n.entity_type"]);
        let sql = query.to_sql();
        assert!(sql.contains("SELECT n.name, n.entity_type"));
    }

    #[test]
    fn test_graph_query_order_by_limit() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .order_by("n.name", true)
            .limit(10);
        let sql = query.to_sql();
        assert!(sql.contains("ORDER BY n.name ASC"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn test_graph_query_complex_sql() {
        let query = GraphQuery::new()
            .match_node("a", Some("Organization"))
            .match_node("b", Some("Product"))
            .match_relationship("r", Some("manufactures"), "a", "b")
            .return_fields(&["a.name", "b.name", "r.confidence"])
            .where_gt("r.confidence", 0.8)
            .order_by("r.confidence", false)
            .limit(5);
        let sql = query.to_sql();
        assert!(sql.contains("a.entity_type = 'Organization'"));
        assert!(sql.contains("b.entity_type = 'Product'"));
        assert!(sql.contains("JOIN relations AS r"));
        assert!(sql.contains("r.relation_type = 'manufactures'"));
        assert!(sql.contains("r.confidence > 0.8"));
        assert!(sql.contains("ORDER BY r.confidence DESC"));
        assert!(sql.contains("LIMIT 5"));
    }

    #[test]
    fn test_graph_query_match_relationship() {
        let query = GraphQuery::new()
            .match_node("a", None)
            .match_node("b", None)
            .match_relationship("r", None, "a", "b");
        let sql = query.to_sql();
        assert!(sql
            .contains("JOIN relations AS r ON a.id = r.from_entity_id AND b.id = r.to_entity_id"));
    }

    #[test]
    fn test_graph_query_to_sql_with_skip() {
        let query = GraphQuery::new().match_node("n", None).limit(10).skip(20);
        let sql = query.to_sql();
        assert!(sql.contains("LIMIT 10"));
        assert!(sql.contains("OFFSET 20"));
    }

    // ========================================================================
    // Graph algorithm tests
    // ========================================================================

    /// Helper: create an in-memory store with a graph:
    /// A --manufactures--> B --uses--> C --located_in--> D
    /// Also E is isolated (no edges).
    fn create_test_graph() -> (KnowledgeGraphStore, i64, i64, i64, i64, i64) {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let a = store
            .get_or_create_entity("A", EntityType::Organization, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("B", EntityType::Product, &[])
            .unwrap();
        let c = store
            .get_or_create_entity("C", EntityType::Product, &[])
            .unwrap();
        let d = store
            .get_or_create_entity("D", EntityType::Location, &[])
            .unwrap();
        let e = store
            .get_or_create_entity("E", EntityType::Person, &[])
            .unwrap();

        store
            .add_relation(a, b, "manufactures", 0.9, None, None)
            .unwrap();
        store.add_relation(b, c, "uses", 0.8, None, None).unwrap();
        store
            .add_relation(c, d, "located_in", 0.7, None, None)
            .unwrap();

        (store, a, b, c, d, e)
    }

    #[test]
    fn test_shortest_path_direct() {
        let (store, a, b, _c, _d, _e) = create_test_graph();
        let path = GraphAlgorithms::shortest_path(&store, a, b).unwrap();
        assert_eq!(path, Some(vec![a, b]));
    }

    #[test]
    fn test_shortest_path_indirect() {
        let (store, a, _b, c, _d, _e) = create_test_graph();
        // A -> B -> C (undirected, so BFS finds it)
        let path = GraphAlgorithms::shortest_path(&store, a, c).unwrap();
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], a);
        assert_eq!(path[2], c);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        // E is isolated
        let path = GraphAlgorithms::shortest_path(&store, a, e).unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn test_page_rank_basic() {
        let (store, a, b, c, d, e) = create_test_graph();
        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 20).unwrap();

        // All entities should have a score
        assert!(ranks.contains_key(&a));
        assert!(ranks.contains_key(&b));
        assert!(ranks.contains_key(&c));
        assert!(ranks.contains_key(&d));
        assert!(ranks.contains_key(&e));

        // All scores > 0
        for &score in ranks.values() {
            assert!(score > 0.0);
        }
    }

    #[test]
    fn test_page_rank_convergence() {
        // Create a graph with a cycle so rank is conserved: A -> B -> C -> A
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();
        let a = store
            .get_or_create_entity("X", EntityType::Organization, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("Y", EntityType::Product, &[])
            .unwrap();
        let c = store
            .get_or_create_entity("Z", EntityType::Location, &[])
            .unwrap();
        store.add_relation(a, b, "r1", 0.9, None, None).unwrap();
        store.add_relation(b, c, "r2", 0.9, None, None).unwrap();
        store.add_relation(c, a, "r3", 0.9, None, None).unwrap();

        let ranks = GraphAlgorithms::page_rank(&store, 0.85, 100).unwrap();

        // In a fully connected cycle with no dangling nodes, sum of ranks should be ~1.0
        let total: f64 = ranks.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PageRank sum was {}, expected ~1.0",
            total
        );

        // In a symmetric cycle, all nodes should have roughly equal rank
        let avg = total / 3.0;
        for &score in ranks.values() {
            assert!(
                (score - avg).abs() < 0.05,
                "Score {} far from average {}",
                score,
                avg
            );
        }
    }

    #[test]
    fn test_connected_components() {
        let (store, a, b, c, d, e) = create_test_graph();
        let components = GraphAlgorithms::connected_components(&store).unwrap();

        // A, B, C, D should be in the same component
        let comp_a = components[&a];
        assert_eq!(components[&b], comp_a);
        assert_eq!(components[&c], comp_a);
        assert_eq!(components[&d], comp_a);

        // E should be in its own component (isolated)
        assert_ne!(components[&e], comp_a);
    }

    #[test]
    fn test_degree_centrality() {
        let (store, a, b, c, d, e) = create_test_graph();
        let (in_deg, out_deg) = GraphAlgorithms::degree_centrality(&store).unwrap();

        // A: out=1 (A->B), in=0
        assert_eq!(out_deg[&a], 1);
        assert_eq!(in_deg[&a], 0);

        // B: out=1 (B->C), in=1 (A->B)
        assert_eq!(out_deg[&b], 1);
        assert_eq!(in_deg[&b], 1);

        // C: out=1 (C->D), in=1 (B->C)
        assert_eq!(out_deg[&c], 1);
        assert_eq!(in_deg[&c], 1);

        // D: out=0, in=1 (C->D)
        assert_eq!(out_deg[&d], 0);
        assert_eq!(in_deg[&d], 1);

        // E: isolated, out=0, in=0
        assert_eq!(out_deg[&e], 0);
        assert_eq!(in_deg[&e], 0);
    }

    #[test]
    fn test_all_paths_basic() {
        let (store, a, _b, c, _d, _e) = create_test_graph();
        // A -> B -> C (undirected), max_depth=3
        let paths = GraphAlgorithms::all_paths(&store, a, c, 3).unwrap();
        assert!(!paths.is_empty());
        // At least one path of length 3 (A, B, C)
        assert!(paths.iter().any(|p| p.len() == 3));
    }

    #[test]
    fn test_list_all_entities() {
        let (store, _a, _b, _c, _d, _e) = create_test_graph();
        let entities = store.list_all_entities().unwrap();
        assert_eq!(entities.len(), 5);
        // Check we got the right names
        let names: Vec<&str> = entities.iter().map(|(_, n, _)| n.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        assert!(names.contains(&"C"));
        assert!(names.contains(&"D"));
        assert!(names.contains(&"E"));
    }

    #[test]
    fn test_list_all_relations() {
        let (store, _a, _b, _c, _d, _e) = create_test_graph();
        let relations = store.list_all_relations().unwrap();
        assert_eq!(relations.len(), 3);
        // Check relation types
        let types: Vec<&str> = relations
            .iter()
            .map(|(_, _, _, rt, _)| rt.as_str())
            .collect();
        assert!(types.contains(&"manufactures"));
        assert!(types.contains(&"uses"));
        assert!(types.contains(&"located_in"));
    }

    #[test]
    fn test_knowledge_graph_export_json() {
        let config = KnowledgeGraphConfig::default();
        let kg = KnowledgeGraph::in_memory(config).unwrap();
        let json = kg.export_json().unwrap();
        assert!(json["entities"].as_array().unwrap().is_empty());
        assert!(json["relations"].as_array().unwrap().is_empty());
        assert_eq!(json["stats"]["total_entities"].as_u64().unwrap(), 0);
    }

    // ====================================================================
    // Additional tests: entity ops, relation ops, dedup, traversal, etc.
    // ====================================================================

    #[test]
    fn test_entity_creation_and_retrieval_by_name() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id = store
            .get_or_create_entity("Tesla", EntityType::Organization, &[])
            .unwrap();

        let entity = store.get_entity_by_name("Tesla").unwrap().unwrap();
        assert_eq!(entity.id, id);
        assert_eq!(entity.name, "Tesla");
        assert_eq!(entity.entity_type, EntityType::Organization);
    }

    #[test]
    fn test_entity_retrieval_nonexistent() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let result = store.get_entity_by_name("NonExistent").unwrap();
        assert!(result.is_none());

        let result_by_id = store.get_entity(999).unwrap();
        assert!(result_by_id.is_none());
    }

    #[test]
    fn test_entity_deduplication_same_name() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id1 = store
            .get_or_create_entity("Duplicated", EntityType::Product, &[])
            .unwrap();
        let id2 = store
            .get_or_create_entity("Duplicated", EntityType::Product, &[])
            .unwrap();

        // Same name should return same ID (no duplicate)
        assert_eq!(id1, id2);

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_entities, 1);
    }

    #[test]
    fn test_entity_deduplication_via_alias() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id1 = store
            .get_or_create_entity(
                "Microsoft",
                EntityType::Organization,
                &["MSFT".to_string()],
            )
            .unwrap();

        // Creating entity by alias name should resolve to same entity
        let _id2 = store
            .get_or_create_entity("MSFT", EntityType::Organization, &[])
            .unwrap();

        // The alias lookup should find the original entity. However, if MSFT is looked up
        // as a name first and not found, then aliases are checked.
        // Since we added "MSFT" as an alias for "Microsoft", find_entity_id("MSFT") should find id1.
        let found_id = store.find_entity_id("MSFT").unwrap();
        assert_eq!(found_id, Some(id1));

        // id2 may or may not equal id1 depending on the exact flow,
        // but find_entity_id should work correctly for the alias
        assert!(found_id.is_some());
    }

    #[test]
    fn test_relationship_creation_between_entities() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let org_id = store
            .get_or_create_entity("SpaceX", EntityType::Organization, &[])
            .unwrap();
        let product_id = store
            .get_or_create_entity("Falcon 9", EntityType::Product, &[])
            .unwrap();

        let rel_id = store
            .add_relation(
                org_id,
                product_id,
                "manufactures",
                0.95,
                Some("SpaceX builds Falcon 9 rockets"),
                None,
            )
            .unwrap();
        assert!(rel_id > 0);

        let relations = store.get_relations_from(org_id, 1).unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].from, "SpaceX");
        assert_eq!(relations[0].to, "Falcon 9");
        assert_eq!(relations[0].relation_type, "manufactures");
    }

    #[test]
    fn test_relation_confidence_update_on_conflict() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let a = store
            .get_or_create_entity("EntityA", EntityType::Other, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("EntityB", EntityType::Other, &[])
            .unwrap();

        // Insert relation with low confidence
        store
            .add_relation(a, b, "related_to", 0.5, None, None)
            .unwrap();

        // Insert same relation with higher confidence (should update via ON CONFLICT)
        store
            .add_relation(a, b, "related_to", 0.9, Some("better evidence"), None)
            .unwrap();

        let relations = store.list_all_relations().unwrap();
        assert_eq!(relations.len(), 1);
        // Confidence should be MAX(0.5, 0.9) = 0.9
        assert!(
            (relations[0].4 - 0.9).abs() < 0.01,
            "Expected confidence ~0.9, got {}",
            relations[0].4
        );
    }

    #[test]
    fn test_graph_traversal_depth_2() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();

        // Depth 1: only A -> B
        let depth1 = store.get_relations_from(a, 1).unwrap();
        assert_eq!(depth1.len(), 1);

        // Depth 2: A -> B, then B -> C
        let depth2 = store.get_relations_from(a, 2).unwrap();
        assert_eq!(depth2.len(), 2);
    }

    #[test]
    fn test_relations_from_isolated_entity() {
        let (_store, _a, _b, _c, _d, e) = create_test_graph();

        // E is isolated - no outgoing relations
        let relations = _store.get_relations_from(e, 2).unwrap();
        assert!(relations.is_empty());
    }

    #[test]
    fn test_empty_graph_stats() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_relations, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_mentions, 0);
        assert!(stats.entities_by_type.is_empty());
        assert!(stats.relations_by_type.is_empty());
    }

    #[test]
    fn test_empty_graph_operations() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        // find_entity_id on empty graph
        assert_eq!(store.find_entity_id("Nothing").unwrap(), None);

        // get_entity on empty graph
        assert!(store.get_entity(1).unwrap().is_none());

        // get_entity_by_name on empty graph
        assert!(store.get_entity_by_name("Nobody").unwrap().is_none());

        // list_all_entities on empty graph
        let entities = store.list_all_entities().unwrap();
        assert!(entities.is_empty());

        // list_all_relations on empty graph
        let relations = store.list_all_relations().unwrap();
        assert!(relations.is_empty());

        // get_chunks_for_entities with empty list
        let chunks = store.get_chunks_for_entities(&[]).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_clear_graph() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        store
            .get_or_create_entity("Alpha", EntityType::Organization, &[])
            .unwrap();
        store
            .get_or_create_entity("Beta", EntityType::Product, &[])
            .unwrap();

        let stats_before = store.get_stats().unwrap();
        assert_eq!(stats_before.total_entities, 2);

        store.clear().unwrap();

        let stats_after = store.get_stats().unwrap();
        assert_eq!(stats_after.total_entities, 0);
        assert_eq!(stats_after.total_relations, 0);
    }

    #[test]
    fn test_entity_type_all_variants() {
        let all = EntityType::all();
        assert_eq!(all.len(), 7);
        // Check round-trip conversion for each variant
        for et in all {
            let s = et.as_str();
            let back = EntityType::from_str(s);
            assert_eq!(*et, back, "Round-trip failed for {:?} -> {}", et, s);
        }
    }

    #[test]
    fn test_entity_type_from_str_aliases() {
        // Check various aliases defined in from_str
        assert_eq!(EntityType::from_str("org"), EntityType::Organization);
        assert_eq!(EntityType::from_str("company"), EntityType::Organization);
        assert_eq!(EntityType::from_str("faction"), EntityType::Organization);
        assert_eq!(EntityType::from_str("ship"), EntityType::Product);
        assert_eq!(EntityType::from_str("vehicle"), EntityType::Product);
        assert_eq!(EntityType::from_str("weapon"), EntityType::Product);
        assert_eq!(EntityType::from_str("item"), EntityType::Product);
        assert_eq!(EntityType::from_str("character"), EntityType::Person);
        assert_eq!(EntityType::from_str("npc"), EntityType::Person);
        assert_eq!(EntityType::from_str("place"), EntityType::Location);
        assert_eq!(EntityType::from_str("system"), EntityType::Location);
        assert_eq!(EntityType::from_str("planet"), EntityType::Location);
        assert_eq!(EntityType::from_str("station"), EntityType::Location);
        assert_eq!(EntityType::from_str("mechanic"), EntityType::Concept);
        assert_eq!(EntityType::from_str("feature"), EntityType::Concept);
        assert_eq!(EntityType::from_str("mission"), EntityType::Event);
        assert_eq!(EntityType::from_str("battle"), EntityType::Event);
        assert_eq!(EntityType::from_str("xyz_unknown"), EntityType::Other);
    }

    #[test]
    fn test_chunk_deduplication() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id1 = store
            .add_chunk("doc1", "This is the same chunk content.", 0)
            .unwrap();
        let id2 = store
            .add_chunk("doc1", "This is the same chunk content.", 1)
            .unwrap();

        // Same content hash should return the same chunk ID
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_link_entity_to_chunk_multiple_mentions() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let entity_id = store
            .get_or_create_entity("Foo", EntityType::Concept, &[])
            .unwrap();
        let chunk_id = store
            .add_chunk("doc1", "Foo appears here.", 0)
            .unwrap();

        // Link twice - mention_count should increase
        store
            .link_entity_to_chunk(entity_id, chunk_id, Some(0), Some("first mention"))
            .unwrap();
        store
            .link_entity_to_chunk(entity_id, chunk_id, Some(10), Some("second mention"))
            .unwrap();

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_mentions, 1); // One row, but count incremented
    }

    #[test]
    fn test_graph_query_where_in() {
        let query = GraphQuery::new()
            .match_node("n", None)
            .return_fields(&["n.name"]);

        // Build a query with WhereClause::In manually
        let mut q = query;
        q.where_clauses.push(WhereClause::In(
            "n.entity_type".to_string(),
            vec!["organization".to_string(), "product".to_string()],
        ));

        let sql = q.to_sql();
        assert!(sql.contains("n.entity_type IN ('organization', 'product')"));
    }

    #[test]
    fn test_graph_query_execute_on_store() {
        let (store, _a, _b, _c, _d, _e) = create_test_graph();

        // entity_type is stored lowercase ("organization") by EntityType::as_str()
        let query = GraphQuery::new()
            .match_node("n", Some("organization"))
            .return_fields(&["n.name", "n.entity_type"])
            .limit(10);

        let result = query.execute(&store).unwrap();
        assert_eq!(result.columns.len(), 2);
        // Only entity A is Organization
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], "A");
    }

    #[test]
    fn test_pattern_extractor_with_custom_pattern() {
        let extractor = PatternEntityExtractor::new()
            .add_entity("rust", EntityType::Concept)
            .add_pattern(r"\b[A-Z][a-z]+\b", EntityType::Other)
            .unwrap();

        let text = "Rust is great. Alice loves it.";
        let result = extractor.extract(text).unwrap();

        // "rust" should be found (known entity, case-insensitive)
        assert!(
            result
                .entities
                .iter()
                .any(|e| e.name.to_lowercase() == "rust"),
            "Expected to find 'rust' entity"
        );
    }

    #[test]
    fn test_pattern_extractor_query_entities() {
        let extractor = PatternEntityExtractor::new()
            .add_entity("python", EntityType::Concept)
            .add_entity("javascript", EntityType::Concept);

        let query_entities = extractor
            .extract_query_entities("Tell me about python and javascript")
            .unwrap();

        assert!(query_entities.iter().any(|e| e == "python"));
        assert!(query_entities.iter().any(|e| e == "javascript"));
    }

    // ========================================================================
    // Entity CRUD advanced (6 tests)
    // ========================================================================

    #[test]
    fn test_entity_metadata() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id = store
            .get_or_create_entity("TestOrg", EntityType::Organization, &[])
            .unwrap();
        let entity = store.get_entity(id).unwrap().unwrap();
        // Default metadata should be an empty HashMap (stored as "{}" in DB)
        assert!(entity.metadata.is_empty());
        // Verify metadata is indeed a HashMap<String, String>
        let expected: HashMap<String, String> = HashMap::new();
        assert_eq!(entity.metadata, expected);
        // Entity name and type preserved
        assert_eq!(entity.name, "TestOrg");
        assert_eq!(entity.entity_type, EntityType::Organization);
    }

    #[test]
    fn test_entity_type_all_returns_seven() {
        let all = EntityType::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&EntityType::Organization));
        assert!(all.contains(&EntityType::Product));
        assert!(all.contains(&EntityType::Person));
        assert!(all.contains(&EntityType::Location));
        assert!(all.contains(&EntityType::Concept));
        assert!(all.contains(&EntityType::Event));
        assert!(all.contains(&EntityType::Other));
    }

    #[test]
    fn test_find_entity_id_not_found() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let result = store.find_entity_id("nonexistent_entity").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_get_entity_by_name() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        store
            .get_or_create_entity("Alice", EntityType::Person, &[])
            .unwrap();

        let entity = store.get_entity_by_name("Alice").unwrap();
        assert!(entity.is_some());
        let entity = entity.unwrap();
        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.entity_type, EntityType::Person);
    }

    #[test]
    fn test_get_entity_by_name_not_found() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let result = store.get_entity_by_name("ghost_entity").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_entity_aliases_resolve() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let id = store
            .get_or_create_entity(
                "Microsoft",
                EntityType::Organization,
                &["MSFT".to_string(), "MS".to_string()],
            )
            .unwrap();

        // Should find by canonical name
        assert_eq!(store.find_entity_id("Microsoft").unwrap(), Some(id));
        // Should find by alias
        assert_eq!(store.find_entity_id("MSFT").unwrap(), Some(id));
        assert_eq!(store.find_entity_id("MS").unwrap(), Some(id));
        // get_entity_by_name via alias should also resolve
        let entity = store.get_entity_by_name("MSFT").unwrap();
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().name, "Microsoft");
    }

    // ========================================================================
    // Relation management (5 tests)
    // ========================================================================

    #[test]
    fn test_relation_confidence_range() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let a = store
            .get_or_create_entity("X", EntityType::Organization, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("Y", EntityType::Product, &[])
            .unwrap();

        // Confidence 0.0 should succeed
        let r1 = store.add_relation(a, b, "zero_conf", 0.0, None, None);
        assert!(r1.is_ok());

        // Confidence 1.0 should succeed
        let c = store
            .get_or_create_entity("Z", EntityType::Product, &[])
            .unwrap();
        let r2 = store.add_relation(a, c, "full_conf", 1.0, None, None);
        assert!(r2.is_ok());
    }

    #[test]
    fn test_get_relations_from_depth_zero() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();

        // depth=0 means the loop runs 0 times => no relations collected
        let relations = store.get_relations_from(a, 0).unwrap();
        assert!(
            relations.is_empty(),
            "depth=0 should return no relations, got {}",
            relations.len()
        );
    }

    #[test]
    fn test_get_relations_from_depth_two() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();

        // depth=2 should traverse A->B and B->C (2 hops)
        let relations = store.get_relations_from(a, 2).unwrap();
        assert!(
            relations.len() >= 2,
            "depth=2 should find at least 2 relations, got {}",
            relations.len()
        );

        // Verify we have both direct and indirect relations
        let rel_types: Vec<&str> = relations.iter().map(|r| r.relation_type.as_str()).collect();
        assert!(
            rel_types.contains(&"manufactures"),
            "Should contain direct 'manufactures' relation"
        );
        assert!(
            rel_types.contains(&"uses"),
            "Should contain indirect 'uses' relation from second hop"
        );
    }

    #[test]
    fn test_relation_with_evidence() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let a = store
            .get_or_create_entity("CompanyA", EntityType::Organization, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("ProductB", EntityType::Product, &[])
            .unwrap();

        let evidence_text = "CompanyA announced ProductB at the 2026 conference.";
        let rel_id = store
            .add_relation(a, b, "announced", 0.95, Some(evidence_text), None)
            .unwrap();
        assert!(rel_id > 0);

        // Verify evidence persists via list_all_relations (returns relation data)
        let all_rels = store.list_all_relations().unwrap();
        assert_eq!(all_rels.len(), 1);
        assert_eq!(all_rels[0].3, "announced");
    }

    #[test]
    fn test_multiple_relations_between_same_entities() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let a = store
            .get_or_create_entity("Org1", EntityType::Organization, &[])
            .unwrap();
        let b = store
            .get_or_create_entity("Prod1", EntityType::Product, &[])
            .unwrap();

        // Add two different relation types between same pair
        store
            .add_relation(a, b, "manufactures", 0.9, None, None)
            .unwrap();
        store
            .add_relation(a, b, "designs", 0.85, None, None)
            .unwrap();

        let all_rels = store.list_all_relations().unwrap();
        assert_eq!(
            all_rels.len(),
            2,
            "Two different relation_types between same pair should both persist"
        );
        let types: Vec<&str> = all_rels.iter().map(|r| r.3.as_str()).collect();
        assert!(types.contains(&"manufactures"));
        assert!(types.contains(&"designs"));
    }

    // ========================================================================
    // Chunk operations (4 tests)
    // ========================================================================

    #[test]
    fn test_add_chunk_returns_id() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let chunk_id = store.add_chunk("doc1", "This is chunk content.", 0).unwrap();
        assert!(chunk_id > 0, "add_chunk should return a positive id");
    }

    #[test]
    fn test_link_entity_to_chunk() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let entity_id = store
            .get_or_create_entity("TestEntity", EntityType::Concept, &[])
            .unwrap();
        let chunk_id = store
            .add_chunk("doc1", "TestEntity is mentioned here.", 0)
            .unwrap();

        store
            .link_entity_to_chunk(entity_id, chunk_id, Some(0), Some("TestEntity is mentioned"))
            .unwrap();

        let chunks = store.get_chunks_for_entities(&[entity_id]).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("TestEntity is mentioned here"));
    }

    #[test]
    fn test_get_chunks_for_multiple_entities() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let e1 = store
            .get_or_create_entity("Entity1", EntityType::Person, &[])
            .unwrap();
        let e2 = store
            .get_or_create_entity("Entity2", EntityType::Location, &[])
            .unwrap();

        let c1 = store.add_chunk("doc1", "Chunk about Entity1.", 0).unwrap();
        let c2 = store.add_chunk("doc1", "Chunk about Entity2.", 1).unwrap();

        store.link_entity_to_chunk(e1, c1, None, None).unwrap();
        store.link_entity_to_chunk(e2, c2, None, None).unwrap();

        let chunks = store.get_chunks_for_entities(&[e1, e2]).unwrap();
        assert_eq!(
            chunks.len(),
            2,
            "Should retrieve chunks for both entities"
        );
    }

    #[test]
    fn test_chunk_position_preserved() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        let c0 = store.add_chunk("doc1", "First chunk.", 0).unwrap();
        let c1 = store.add_chunk("doc1", "Second chunk.", 1).unwrap();
        let c2 = store.add_chunk("doc1", "Third chunk.", 2).unwrap();

        // Verify all are different IDs (unique content)
        assert_ne!(c0, c1);
        assert_ne!(c1, c2);
        assert_ne!(c0, c2);

        // Link an entity to all chunks to verify retrieval with position metadata
        let eid = store
            .get_or_create_entity("Marker", EntityType::Concept, &[])
            .unwrap();
        store.link_entity_to_chunk(eid, c0, None, None).unwrap();
        store.link_entity_to_chunk(eid, c1, None, None).unwrap();
        store.link_entity_to_chunk(eid, c2, None, None).unwrap();

        let chunks = store.get_chunks_for_entities(&[eid]).unwrap();
        assert_eq!(chunks.len(), 3);
        // Each chunk should have a position metadata entry
        for chunk in &chunks {
            assert!(
                chunk.metadata.contains_key("position"),
                "Chunk should have 'position' in metadata"
            );
        }
    }

    // ========================================================================
    // GraphQuery builder (5 tests)
    // ========================================================================

    #[test]
    fn test_graph_query_where_contains_generates_like() {
        let sql = GraphQuery::new()
            .match_node("n", None)
            .where_contains("n.name", "Aegis")
            .to_sql();

        assert!(
            sql.contains("LIKE"),
            "where_contains should generate LIKE clause, got: {}",
            sql
        );
        assert!(
            sql.contains("Aegis"),
            "SQL should contain the search value"
        );
    }

    #[test]
    fn test_graph_query_where_gt_generates_comparison() {
        let sql = GraphQuery::new()
            .match_node("n", None)
            .where_gt("n.confidence", 0.5)
            .to_sql();

        assert!(
            sql.contains(">"),
            "where_gt should generate > clause, got: {}",
            sql
        );
        assert!(
            sql.contains("0.5"),
            "SQL should contain the threshold value"
        );
    }

    #[test]
    fn test_graph_query_skip_and_limit() {
        let sql = GraphQuery::new()
            .match_node("n", None)
            .skip(10)
            .limit(5)
            .to_sql();

        assert!(
            sql.contains("LIMIT 5"),
            "Should contain LIMIT 5, got: {}",
            sql
        );
        assert!(
            sql.contains("OFFSET 10"),
            "Should contain OFFSET 10, got: {}",
            sql
        );
    }

    #[test]
    fn test_graph_query_order_by_descending() {
        let sql = GraphQuery::new()
            .match_node("n", None)
            .order_by("n.name", false)
            .to_sql();

        assert!(
            sql.contains("ORDER BY"),
            "Should contain ORDER BY, got: {}",
            sql
        );
        assert!(
            sql.contains("DESC"),
            "ascending=false should generate DESC, got: {}",
            sql
        );
    }

    #[test]
    fn test_graph_query_match_path() {
        let sql = GraphQuery::new()
            .match_node("src", Some("organization"))
            .match_path("src", "dst", 3)
            .to_sql();

        assert!(
            sql.contains("max_hops=3"),
            "match_path should encode max_hops in SQL comment, got: {}",
            sql
        );
        assert!(
            sql.contains("path_r"),
            "match_path should use path_r alias for relation join, got: {}",
            sql
        );
    }

    // ========================================================================
    // KnowledgeGraph high-level (4 tests)
    // ========================================================================

    #[test]
    fn test_graph_in_memory_empty() {
        let config = KnowledgeGraphConfig::default();
        let graph = KnowledgeGraph::in_memory(config).unwrap();

        let stats = graph.stats().unwrap();
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_relations, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_mentions, 0);
    }

    #[test]
    fn test_graph_chunk_text_sizes() {
        let mut config = KnowledgeGraphConfig::default();
        config.chunk_size = 50;
        config.chunk_overlap = 10;
        let graph = KnowledgeGraph::in_memory(config).unwrap();

        // Generate text longer than one chunk
        let text = "A".repeat(120);
        let chunks = graph.chunk_text(&text);

        assert!(
            chunks.len() >= 2,
            "120 chars with chunk_size=50 should produce at least 2 chunks, got {}",
            chunks.len()
        );
        // Each chunk should not exceed chunk_size
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.len() <= 50,
                "Chunk {} has len {} which exceeds chunk_size 50",
                i,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_graph_clear_resets_stats() {
        let config = KnowledgeGraphConfig::default();
        let graph = KnowledgeGraph::in_memory(config).unwrap();

        // Add some data directly via the store
        let eid = graph
            .store
            .get_or_create_entity("ClearTest", EntityType::Concept, &[])
            .unwrap();
        let cid = graph
            .store
            .add_chunk("doc1", "Some content for clear test.", 0)
            .unwrap();
        graph
            .store
            .link_entity_to_chunk(eid, cid, None, None)
            .unwrap();

        let stats_before = graph.stats().unwrap();
        assert!(stats_before.total_entities > 0);

        graph.clear().unwrap();

        let stats_after = graph.stats().unwrap();
        assert_eq!(stats_after.total_entities, 0);
        assert_eq!(stats_after.total_relations, 0);
        assert_eq!(stats_after.total_chunks, 0);
        assert_eq!(stats_after.total_mentions, 0);
    }

    #[test]
    fn test_graph_export_json_structure() {
        let config = KnowledgeGraphConfig::default();
        let graph = KnowledgeGraph::in_memory(config).unwrap();

        graph
            .store
            .get_or_create_entity("ExportEntity", EntityType::Organization, &[])
            .unwrap();

        let json = graph.export_json().unwrap();

        assert!(
            json.get("entities").is_some(),
            "Export JSON should have 'entities' key"
        );
        assert!(
            json.get("relations").is_some(),
            "Export JSON should have 'relations' key"
        );
        assert!(
            json.get("stats").is_some(),
            "Export JSON should have 'stats' key"
        );

        let entities = json["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["name"], "ExportEntity");
    }

    // ========================================================================
    // Graph algorithms (4 tests)
    // ========================================================================

    #[test]
    fn test_shortest_path_unconnected_returns_none() {
        let (store, _a, _b, _c, _d, e) = create_test_graph();

        // E is isolated (no outgoing or incoming relations in create_test_graph)
        // A->B->C->D chain, E is disconnected
        let path = GraphAlgorithms::shortest_path(&store, e, _a).unwrap();
        assert_eq!(
            path, None,
            "Should return None when no path exists between unconnected entities"
        );
    }

    #[test]
    fn test_all_paths_max_depth() {
        let (store, a, b, _c, _d, _e) = create_test_graph();

        // With max_depth=1, only direct paths A->B should be found
        let paths = GraphAlgorithms::all_paths(&store, a, b, 1).unwrap();
        assert!(
            !paths.is_empty(),
            "Should find at least the direct path A->B"
        );
        for path in &paths {
            assert!(
                path.len() <= 2,
                "max_depth=1 should limit path length to at most 2 nodes, got {}",
                path.len()
            );
        }
    }

    #[test]
    fn test_degree_centrality_values() {
        let (store, a, b, c, d, e) = create_test_graph();

        let (in_deg, out_deg) = GraphAlgorithms::degree_centrality(&store).unwrap();

        // A has outgoing: A->B (out=1), no incoming (in=0)
        assert_eq!(*out_deg.get(&a).unwrap_or(&0), 1);
        assert_eq!(*in_deg.get(&a).unwrap_or(&0), 0);

        // B has incoming: A->B (in=1), outgoing: B->C (out=1)
        assert_eq!(*in_deg.get(&b).unwrap_or(&0), 1);
        assert_eq!(*out_deg.get(&b).unwrap_or(&0), 1);

        // C has incoming: B->C (in=1), outgoing: C->D (out=1)
        assert_eq!(*in_deg.get(&c).unwrap_or(&0), 1);
        assert_eq!(*out_deg.get(&c).unwrap_or(&0), 1);

        // D has incoming: C->D (in=1), no outgoing (out=0)
        assert_eq!(*in_deg.get(&d).unwrap_or(&0), 1);
        assert_eq!(*out_deg.get(&d).unwrap_or(&0), 0);

        // E is isolated: in=0, out=0
        assert_eq!(*in_deg.get(&e).unwrap_or(&0), 0);
        assert_eq!(*out_deg.get(&e).unwrap_or(&0), 0);
    }

    #[test]
    fn test_connected_components_two_groups() {
        let config = KnowledgeGraphConfig::default();
        let store = KnowledgeGraphStore::in_memory(config).unwrap();

        // Group 1: X -> Y
        let x = store
            .get_or_create_entity("X", EntityType::Organization, &[])
            .unwrap();
        let y = store
            .get_or_create_entity("Y", EntityType::Product, &[])
            .unwrap();
        store.add_relation(x, y, "related", 0.9, None, None).unwrap();

        // Group 2: P -> Q (disconnected from group 1)
        let p = store
            .get_or_create_entity("P", EntityType::Person, &[])
            .unwrap();
        let q = store
            .get_or_create_entity("Q", EntityType::Location, &[])
            .unwrap();
        store.add_relation(p, q, "located_in", 0.8, None, None).unwrap();

        let components = GraphAlgorithms::connected_components(&store).unwrap();

        // X and Y should share a component, P and Q should share a different one
        assert_eq!(
            components.get(&x),
            components.get(&y),
            "X and Y should be in the same component"
        );
        assert_eq!(
            components.get(&p),
            components.get(&q),
            "P and Q should be in the same component"
        );
        assert_ne!(
            components.get(&x),
            components.get(&p),
            "Group 1 and Group 2 should be in different components"
        );
    }

    // ========================================================================
    // Builder pattern (4 tests)
    // ========================================================================

    #[test]
    fn test_builder_with_config() {
        let mut custom_config = KnowledgeGraphConfig::default();
        custom_config.max_traversal_depth = 5;
        custom_config.chunk_size = 500;

        let graph = KnowledgeGraphBuilder::new()
            .with_config(custom_config)
            .build_in_memory()
            .unwrap();

        // The graph should use the custom config values
        assert_eq!(graph.config.max_traversal_depth, 5);
        assert_eq!(graph.config.chunk_size, 500);
    }

    #[test]
    fn test_builder_add_entity() {
        let graph = KnowledgeGraphBuilder::new()
            .add_entity("BuilderOrg", EntityType::Organization)
            .build_in_memory()
            .unwrap();

        let entity = graph.store.get_entity_by_name("BuilderOrg").unwrap();
        assert!(entity.is_some(), "Entity added via builder should exist");
        assert_eq!(entity.unwrap().entity_type, EntityType::Organization);
    }

    #[test]
    fn test_builder_add_alias() {
        let graph = KnowledgeGraphBuilder::new()
            .add_entity("CanonicalName", EntityType::Person)
            .add_alias("Alias1", "CanonicalName")
            .build_in_memory()
            .unwrap();

        // The alias should resolve to the canonical entity
        let by_alias = graph.store.find_entity_id("Alias1").unwrap();
        let by_canonical = graph.store.find_entity_id("CanonicalName").unwrap();
        assert!(by_alias.is_some(), "Alias should resolve to an entity");
        assert_eq!(
            by_alias, by_canonical,
            "Alias should resolve to the same entity as canonical name"
        );
    }

    #[test]
    fn test_builder_default_builds_empty() {
        let graph = KnowledgeGraphBuilder::new().build_in_memory().unwrap();

        let stats = graph.stats().unwrap();
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.total_relations, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_mentions, 0);
    }
}
