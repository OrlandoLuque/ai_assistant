//! Multi-Layer Knowledge Graph System
//!
//! This module provides a layered knowledge graph system with:
//! - Layer 1: Knowledge Graph (primary, verified data from knowledge packs)
//! - Layer 2: Internet Graph (complementary, web-sourced data with TTL)
//! - Layer 3: User Graph (persistent user preferences and beliefs)
//! - Layer 4: Session Graph (temporary, per-conversation context)
//!
//! Features:
//! - Contradiction detection between layers
//! - Priority-based query merging
//! - User beliefs with confidence levels
//! - Automatic entity extraction from conversations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// Note: KnowledgeGraph integration is optional and depends on the rag feature
// When needed, use crate::KnowledgeGraph, crate::GraphEntity, etc.

// =============================================================================
// GRAPH LAYERS
// =============================================================================

/// Identifies which layer a piece of data comes from
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphLayer {
    /// Primary knowledge from verified sources (knowledge packs)
    Knowledge,
    /// Complementary data from internet sources
    Internet,
    /// User-stated preferences and beliefs
    User,
    /// Temporary data from current conversation
    Session,
}

impl GraphLayer {
    /// Get the priority of this layer (higher = more trusted)
    pub fn priority(&self) -> u8 {
        match self {
            GraphLayer::Knowledge => 100,
            GraphLayer::User => 80, // User beliefs are important but not verified
            GraphLayer::Internet => 50, // Internet data needs verification
            GraphLayer::Session => 30, // Session data is contextual
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            GraphLayer::Knowledge => "Knowledge Pack",
            GraphLayer::User => "User Belief",
            GraphLayer::Internet => "Internet",
            GraphLayer::Session => "Session",
        }
    }
}

/// Confidence level of data
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Verified data from official sources
    Verified,
    /// Data from web sources (may need verification)
    WebSource,
    /// User explicitly stated this
    UserStated,
    /// AI inferred from context
    Inferred,
}

impl ConfidenceLevel {
    pub fn score(&self) -> f32 {
        match self {
            ConfidenceLevel::Verified => 1.0,
            ConfidenceLevel::UserStated => 0.9,
            ConfidenceLevel::WebSource => 0.6,
            ConfidenceLevel::Inferred => 0.4,
        }
    }
}

// =============================================================================
// USER BELIEFS
// =============================================================================

/// Type of user belief
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefType {
    /// User preference ("I prefer X")
    Preference,
    /// User ownership ("I have X")
    Ownership,
    /// User goal ("I want X")
    Goal,
    /// User opinion ("X is better than Y")
    Opinion,
    /// User-stated fact ("X has Y shields") - may contradict knowledge
    Fact,
}

impl BeliefType {
    pub fn display_name(&self) -> &'static str {
        match self {
            BeliefType::Preference => "Preference",
            BeliefType::Ownership => "Ownership",
            BeliefType::Goal => "Goal",
            BeliefType::Opinion => "Opinion",
            BeliefType::Fact => "Stated Fact",
        }
    }
}

/// A belief or preference stated by the user
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserBelief {
    /// Unique identifier
    pub id: String,
    /// The statement as expressed by the user
    pub statement: String,
    /// Main entity this belief is about
    pub subject_entity: Option<String>,
    /// Type of belief
    pub belief_type: BeliefType,
    /// When this belief was expressed
    pub expressed_at: u64, // Unix timestamp
    /// Session where this was expressed
    pub session_id: String,
    /// Confidence in the extraction (0.0-1.0)
    pub confidence: f32,
    /// Whether this belief is still considered active
    pub active: bool,
}

impl UserBelief {
    pub fn new(
        statement: impl Into<String>,
        subject_entity: Option<String>,
        belief_type: BeliefType,
        session_id: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: format!("belief_{}", now),
            statement: statement.into(),
            subject_entity,
            belief_type,
            expressed_at: now,
            session_id: session_id.into(),
            confidence,
            active: true,
        }
    }
}

// =============================================================================
// CONTRADICTIONS
// =============================================================================

/// Source of a piece of contradictory information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContradictionSource {
    /// Which layer this came from
    pub layer: GraphLayer,
    /// Name of the source (e.g., "ships.kpkg", "reddit.com/r/starcitizen")
    pub source_name: String,
    /// URL if from internet
    pub source_url: Option<String>,
    /// When this data was recorded
    pub timestamp: u64,
}

/// How a contradiction was resolved
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContradictionResolution {
    /// Primary (knowledge) source is trustworthy
    PrimaryTrustworthy,
    /// Internet source has more recent info
    InternetMoreRecent,
    /// User explicitly chose a resolution
    UserOverride,
    /// Not yet resolved - show both to user
    Unresolved,
}

/// A detected contradiction between data sources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contradiction {
    /// Unique identifier
    pub id: String,
    /// When this was detected
    pub detected_at: u64,
    /// The primary (trusted) source
    pub primary_source: ContradictionSource,
    /// The conflicting source
    pub conflicting_source: ContradictionSource,
    /// Entity involved
    pub entity: String,
    /// Attribute that contradicts
    pub attribute: String,
    /// Value from primary source
    pub primary_value: String,
    /// Value from conflicting source
    pub conflicting_value: String,
    /// How this was resolved (if at all)
    pub resolution: ContradictionResolution,
    /// Whether user was notified
    pub user_notified: bool,
}

impl Contradiction {
    pub fn new(
        entity: impl Into<String>,
        attribute: impl Into<String>,
        primary_value: impl Into<String>,
        conflicting_value: impl Into<String>,
        primary_source: ContradictionSource,
        conflicting_source: ContradictionSource,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: format!("contradiction_{}", now),
            detected_at: now,
            primary_source,
            conflicting_source,
            entity: entity.into(),
            attribute: attribute.into(),
            primary_value: primary_value.into(),
            conflicting_value: conflicting_value.into(),
            resolution: ContradictionResolution::Unresolved,
            user_notified: false,
        }
    }
}

/// Log of contradictions for export/reporting
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContradictionLog {
    /// Version of the log format
    pub version: String,
    /// Anonymized user identifier
    pub user_id: String,
    /// Application version
    pub app_version: String,
    /// When this log was generated
    pub generated_at: u64,
    /// All recorded contradictions
    pub contradictions: Vec<Contradiction>,
    /// Versions of knowledge packs used
    pub knowledge_pack_versions: HashMap<String, String>,
}

impl ContradictionLog {
    pub fn new(user_id: impl Into<String>, app_version: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version: "1.0".to_string(),
            user_id: user_id.into(),
            app_version: app_version.into(),
            generated_at: now,
            contradictions: Vec::new(),
            knowledge_pack_versions: HashMap::new(),
        }
    }

    /// Export log as JSON for reporting
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// =============================================================================
// LAYERED ENTITY (entity with layer metadata)
// =============================================================================

/// An entity with layer metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayeredEntity {
    /// Entity name
    pub name: String,
    /// Entity type (e.g., "Product", "Organization")
    pub entity_type: String,
    /// Which layer this came from
    pub layer: GraphLayer,
    /// Confidence level
    pub confidence: ConfidenceLevel,
    /// Source identifier
    pub source: String,
    /// When this was added
    pub timestamp: u64,
    /// Time-to-live for cache (internet layer)
    pub ttl_seconds: Option<u64>,
}

impl LayeredEntity {
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now > self.timestamp + ttl
        } else {
            false
        }
    }
}

// =============================================================================
// BELIEF EXTRACTOR (extracts beliefs from text)
// =============================================================================

/// Patterns for extracting user beliefs from text
pub struct BeliefExtractor {
    /// Patterns that indicate preference
    preference_patterns: Vec<&'static str>,
    /// Patterns that indicate ownership
    ownership_patterns: Vec<&'static str>,
    /// Patterns that indicate goals
    goal_patterns: Vec<&'static str>,
    /// Patterns that indicate opinions
    opinion_patterns: Vec<&'static str>,
}

impl Default for BeliefExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl BeliefExtractor {
    pub fn new() -> Self {
        Self {
            preference_patterns: vec![
                "prefiero",
                "me gusta más",
                "i prefer",
                "i like",
                "me inclino por",
                "mi favorito",
                "my favorite",
            ],
            ownership_patterns: vec![
                "tengo un",
                "tengo una",
                "i have a",
                "i own",
                "mi ",
                "my ",
                "compré",
                "i bought",
            ],
            goal_patterns: vec![
                "quiero",
                "i want",
                "me gustaría",
                "i would like",
                "planeo",
                "i plan to",
                "voy a comprar",
                "i'm going to buy",
            ],
            opinion_patterns: vec![
                "creo que",
                "pienso que",
                "i think",
                "i believe",
                "en mi opinión",
                "in my opinion",
                "me parece que",
                "es mejor",
                "is better",
                "es el mejor",
                "is the best",
            ],
        }
    }

    /// Extract beliefs from a user message
    pub fn extract(&self, message: &str, session_id: &str, entities: &[String]) -> Vec<UserBelief> {
        let mut beliefs = Vec::new();
        let lower = message.to_lowercase();

        // Check for preferences
        for pattern in &self.preference_patterns {
            if lower.contains(pattern) {
                let entity = entities.first().map(|s| s.to_string());
                beliefs.push(UserBelief::new(
                    message,
                    entity,
                    BeliefType::Preference,
                    session_id,
                    0.8,
                ));
                break;
            }
        }

        // Check for ownership
        for pattern in &self.ownership_patterns {
            if lower.contains(pattern) {
                let entity = entities.first().map(|s| s.to_string());
                beliefs.push(UserBelief::new(
                    message,
                    entity,
                    BeliefType::Ownership,
                    session_id,
                    0.9,
                ));
                break;
            }
        }

        // Check for goals
        for pattern in &self.goal_patterns {
            if lower.contains(pattern) {
                let entity = entities.first().map(|s| s.to_string());
                beliefs.push(UserBelief::new(
                    message,
                    entity,
                    BeliefType::Goal,
                    session_id,
                    0.85,
                ));
                break;
            }
        }

        // Check for opinions
        for pattern in &self.opinion_patterns {
            if lower.contains(pattern) {
                let entity = entities.first().map(|s| s.to_string());
                beliefs.push(UserBelief::new(
                    message,
                    entity,
                    BeliefType::Opinion,
                    session_id,
                    0.7,
                ));
                break;
            }
        }

        beliefs
    }
}

// =============================================================================
// SESSION GRAPH (per-conversation graph)
// =============================================================================

/// A lightweight graph for a single session
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionGraph {
    /// Session identifier
    pub session_id: String,
    /// Entities mentioned in this session
    pub entities: Vec<LayeredEntity>,
    /// Relations discovered in this session
    pub relations: Vec<(String, String, String)>, // (from, relation_type, to)
    /// When this session started
    pub created_at: u64,
    /// Last update time
    pub updated_at: u64,
}

impl SessionGraph {
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            session_id: session_id.into(),
            entities: Vec::new(),
            relations: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add an entity to the session graph
    pub fn add_entity(
        &mut self,
        name: impl Into<String>,
        entity_type: impl Into<String>,
        source: impl Into<String>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let name = name.into();

        // Don't add duplicates
        if self.entities.iter().any(|e| e.name == name) {
            return;
        }

        self.entities.push(LayeredEntity {
            name,
            entity_type: entity_type.into(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Inferred,
            source: source.into(),
            timestamp: now,
            ttl_seconds: None,
        });
        self.updated_at = now;
    }

    /// Add a relation to the session graph
    pub fn add_relation(
        &mut self,
        from: impl Into<String>,
        relation_type: impl Into<String>,
        to: impl Into<String>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.relations
            .push((from.into(), relation_type.into(), to.into()));
        self.updated_at = now;
    }

    /// Get all entity names
    pub fn entity_names(&self) -> Vec<&str> {
        self.entities.iter().map(|e| e.name.as_str()).collect()
    }

    /// Check if an entity is in this session
    pub fn has_entity(&self, name: &str) -> bool {
        self.entities
            .iter()
            .any(|e| e.name.eq_ignore_ascii_case(name))
    }
}

// =============================================================================
// USER GRAPH (persistent user data)
// =============================================================================

/// Persistent user preferences and beliefs
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UserGraph {
    /// User identifier
    pub user_id: String,
    /// User beliefs
    pub beliefs: Vec<UserBelief>,
    /// User-confirmed facts (may contradict knowledge)
    pub confirmed_facts: HashMap<String, String>,
    /// When this was last updated
    pub updated_at: u64,
}

impl UserGraph {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            beliefs: Vec::new(),
            confirmed_facts: HashMap::new(),
            updated_at: 0,
        }
    }

    /// Add a belief
    pub fn add_belief(&mut self, belief: UserBelief) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.beliefs.push(belief);
        self.updated_at = now;
    }

    /// Get active beliefs about an entity
    pub fn beliefs_about(&self, entity: &str) -> Vec<&UserBelief> {
        self.beliefs
            .iter()
            .filter(|b| {
                b.active
                    && b.subject_entity
                        .as_ref()
                        .map_or(false, |e| e.eq_ignore_ascii_case(entity))
            })
            .collect()
    }

    /// Get all active beliefs of a type
    pub fn beliefs_of_type(&self, belief_type: BeliefType) -> Vec<&UserBelief> {
        self.beliefs
            .iter()
            .filter(|b| b.active && b.belief_type == belief_type)
            .collect()
    }

    /// Get all active beliefs
    pub fn all_active_beliefs(&self) -> Vec<&UserBelief> {
        self.beliefs.iter().filter(|b| b.active).collect()
    }

    /// Deactivate a belief by ID (soft delete)
    pub fn deactivate_belief(&mut self, belief_id: &str) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(belief) = self.beliefs.iter_mut().find(|b| b.id == belief_id) {
            belief.active = false;
            self.updated_at = now;
            true
        } else {
            false
        }
    }

    /// Remove a belief by ID (hard delete)
    pub fn remove_belief(&mut self, belief_id: &str) -> bool {
        let len_before = self.beliefs.len();
        self.beliefs.retain(|b| b.id != belief_id);
        let removed = self.beliefs.len() < len_before;
        if removed {
            self.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
        removed
    }

    /// Load from file (auto-detects binary or legacy JSON format).
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        crate::internal_storage::load_internal(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save to file using internal storage format.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        crate::internal_storage::save_internal(self, path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// INTERNET GRAPH ENTRY (cached web data)
// =============================================================================

/// Cached data from internet sources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InternetGraphEntry {
    /// Entity this is about
    pub entity: String,
    /// Attribute
    pub attribute: String,
    /// Value
    pub value: String,
    /// Source URL
    pub source_url: String,
    /// When this was fetched
    pub fetched_at: u64,
    /// TTL in seconds
    pub ttl_seconds: u64,
    /// Whether this contradicts knowledge graph
    pub contradicts_knowledge: bool,
}

impl InternetGraphEntry {
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now > self.fetched_at + self.ttl_seconds
    }
}

/// Internet knowledge cache
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InternetGraph {
    /// Cached entries
    pub entries: Vec<InternetGraphEntry>,
    /// Default TTL for new entries
    pub default_ttl_seconds: u64,
}

impl InternetGraph {
    pub fn new(default_ttl_seconds: u64) -> Self {
        Self {
            entries: Vec::new(),
            default_ttl_seconds,
        }
    }

    /// Add an entry
    pub fn add_entry(&mut self, entry: InternetGraphEntry) {
        // Remove any existing entry for same entity+attribute
        self.entries
            .retain(|e| !(e.entity == entry.entity && e.attribute == entry.attribute));
        self.entries.push(entry);
    }

    /// Get non-expired entries for an entity
    pub fn get_entries(&self, entity: &str) -> Vec<&InternetGraphEntry> {
        self.entries
            .iter()
            .filter(|e| e.entity.eq_ignore_ascii_case(entity) && !e.is_expired())
            .collect()
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) {
        self.entries.retain(|e| !e.is_expired());
    }

    /// Load from file (auto-detects binary or legacy JSON format).
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        crate::internal_storage::load_internal(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save to file using internal storage format.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        crate::internal_storage::save_internal(self, path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// MULTI-LAYER QUERY RESULT
// =============================================================================

/// Result from querying across multiple layers
#[derive(Clone, Debug, Default)]
pub struct MultiLayerQueryResult {
    /// Entities found with their layer information
    pub entities: Vec<(LayeredEntity, GraphLayer)>,
    /// Relations found
    pub relations: Vec<(String, String, String, GraphLayer)>,
    /// User beliefs relevant to the query
    pub relevant_beliefs: Vec<UserBelief>,
    /// Any contradictions found
    pub contradictions: Vec<Contradiction>,
    /// Context string for LLM
    pub context: String,
}

// =============================================================================
// MULTI-LAYER GRAPH COORDINATOR
// =============================================================================

/// Coordinator for all graph layers
pub struct MultiLayerGraph {
    /// Session graphs (keyed by session_id)
    pub session_graphs: HashMap<String, SessionGraph>,
    /// User graph (persistent)
    pub user_graph: UserGraph,
    /// Internet graph (cached web data)
    pub internet_graph: InternetGraph,
    /// Detected contradictions
    pub contradictions: Vec<Contradiction>,
    /// Belief extractor
    pub belief_extractor: BeliefExtractor,
    /// Path for user graph persistence
    pub user_graph_path: Option<PathBuf>,
    /// Path for internet graph persistence
    pub internet_graph_path: Option<PathBuf>,
    /// Path for contradiction log
    pub contradiction_log_path: Option<PathBuf>,
}

impl Default for MultiLayerGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiLayerGraph {
    pub fn new() -> Self {
        Self {
            session_graphs: HashMap::new(),
            user_graph: UserGraph::new("default"),
            internet_graph: InternetGraph::new(86400), // 24h default TTL
            contradictions: Vec::new(),
            belief_extractor: BeliefExtractor::new(),
            user_graph_path: None,
            internet_graph_path: None,
            contradiction_log_path: None,
        }
    }

    /// Create with persistence paths
    pub fn with_persistence(
        user_graph_path: PathBuf,
        internet_graph_path: PathBuf,
        contradiction_log_path: PathBuf,
    ) -> Self {
        let mut graph = Self::new();
        graph.user_graph_path = Some(user_graph_path.clone());
        graph.internet_graph_path = Some(internet_graph_path.clone());
        graph.contradiction_log_path = Some(contradiction_log_path);

        // Load existing data
        if let Ok(ug) = UserGraph::load(&user_graph_path) {
            graph.user_graph = ug;
        }
        if let Ok(ig) = InternetGraph::load(&internet_graph_path) {
            graph.internet_graph = ig;
        }

        graph
    }

    /// Get or create a session graph
    pub fn get_or_create_session(&mut self, session_id: &str) -> &mut SessionGraph {
        if !self.session_graphs.contains_key(session_id) {
            self.session_graphs
                .insert(session_id.to_string(), SessionGraph::new(session_id));
        }
        self.session_graphs
            .get_mut(session_id)
            .expect("key just inserted")
    }

    /// Process a user message and extract entities/beliefs
    pub fn process_user_message(
        &mut self,
        session_id: &str,
        message: &str,
        extracted_entities: &[String],
    ) {
        // Add entities to session graph
        let session = self.get_or_create_session(session_id);
        for entity in extracted_entities {
            session.add_entity(entity, "Unknown", "user_message");
        }

        // Extract beliefs
        let beliefs = self
            .belief_extractor
            .extract(message, session_id, extracted_entities);
        for belief in beliefs {
            self.user_graph.add_belief(belief);
        }

        // Save user graph
        self.save_user_graph();
    }

    /// Process an assistant response and extract entities
    pub fn process_assistant_response(
        &mut self,
        session_id: &str,
        _response: &str,
        extracted_entities: &[String],
    ) {
        // Add entities to session graph
        let session = self.get_or_create_session(session_id);
        for entity in extracted_entities {
            session.add_entity(entity, "Unknown", "assistant_response");
        }
    }

    /// Add internet data and check for contradictions
    pub fn add_internet_data(
        &mut self,
        entity: &str,
        attribute: &str,
        value: &str,
        source_url: &str,
        knowledge_value: Option<&str>,
    ) -> Option<Contradiction> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let contradicts = knowledge_value.map_or(false, |kv| kv != value);

        let entry = InternetGraphEntry {
            entity: entity.to_string(),
            attribute: attribute.to_string(),
            value: value.to_string(),
            source_url: source_url.to_string(),
            fetched_at: now,
            ttl_seconds: self.internet_graph.default_ttl_seconds,
            contradicts_knowledge: contradicts,
        };

        self.internet_graph.add_entry(entry);

        // Create contradiction if detected
        if contradicts {
            if let Some(kv) = knowledge_value {
                let contradiction = Contradiction::new(
                    entity,
                    attribute,
                    kv,
                    value,
                    ContradictionSource {
                        layer: GraphLayer::Knowledge,
                        source_name: "knowledge_pack".to_string(),
                        source_url: None,
                        timestamp: 0, // Unknown
                    },
                    ContradictionSource {
                        layer: GraphLayer::Internet,
                        source_name: source_url.to_string(),
                        source_url: Some(source_url.to_string()),
                        timestamp: now,
                    },
                );
                self.contradictions.push(contradiction.clone());
                self.save_contradiction_log();
                return Some(contradiction);
            }
        }

        self.save_internet_graph();
        None
    }

    /// Build context for LLM from all layers
    pub fn build_context(&self, session_id: &str, query_entities: &[String]) -> String {
        let mut context = String::new();

        // User beliefs (high priority)
        let relevant_beliefs: Vec<_> = query_entities
            .iter()
            .flat_map(|e| self.user_graph.beliefs_about(e))
            .collect();

        if !relevant_beliefs.is_empty() {
            context.push_str("## User Beliefs\n");
            for belief in relevant_beliefs {
                context.push_str(&format!(
                    "- [{}] {}\n",
                    belief.belief_type.display_name(),
                    belief.statement
                ));
            }
            context.push('\n');
        }

        // Session context
        if let Some(session) = self.session_graphs.get(session_id) {
            if !session.entities.is_empty() {
                context.push_str("## Previously Mentioned in This Session\n");
                for entity in &session.entities {
                    context.push_str(&format!("- {} ({})\n", entity.name, entity.entity_type));
                }
                context.push('\n');
            }
        }

        // Internet data (with contradiction warnings)
        let internet_data: Vec<_> = query_entities
            .iter()
            .flat_map(|e| self.internet_graph.get_entries(e))
            .collect();

        if !internet_data.is_empty() {
            context.push_str("## Internet Data (unverified)\n");
            for entry in internet_data {
                let warning = if entry.contradicts_knowledge {
                    " [CONTRADICTS KNOWLEDGE]"
                } else {
                    ""
                };
                context.push_str(&format!(
                    "- {}: {} = {}{}\n",
                    entry.entity, entry.attribute, entry.value, warning
                ));
            }
            context.push('\n');
        }

        context
    }

    /// Get unresolved contradictions
    pub fn get_unresolved_contradictions(&self) -> Vec<&Contradiction> {
        self.contradictions
            .iter()
            .filter(|c| c.resolution == ContradictionResolution::Unresolved)
            .collect()
    }

    /// Resolve a contradiction
    pub fn resolve_contradiction(
        &mut self,
        contradiction_id: &str,
        resolution: ContradictionResolution,
    ) -> bool {
        if let Some(c) = self
            .contradictions
            .iter_mut()
            .find(|c| c.id == contradiction_id)
        {
            c.resolution = resolution;
            c.user_notified = true;
            self.save_contradiction_log();
            true
        } else {
            false
        }
    }

    /// Remove a belief from the user graph
    pub fn remove_belief(&mut self, belief_id: &str) -> bool {
        let removed = self.user_graph.remove_belief(belief_id);
        if removed {
            self.save_user_graph();
        }
        removed
    }

    /// Deactivate a belief from the user graph
    pub fn deactivate_belief(&mut self, belief_id: &str) -> bool {
        let deactivated = self.user_graph.deactivate_belief(belief_id);
        if deactivated {
            self.save_user_graph();
        }
        deactivated
    }

    /// Get all active beliefs
    pub fn get_all_beliefs(&self) -> Vec<&UserBelief> {
        self.user_graph.all_active_beliefs()
    }

    /// Clean up a session
    pub fn cleanup_session(&mut self, session_id: &str) {
        self.session_graphs.remove(session_id);
    }

    /// Save user graph to disk
    fn save_user_graph(&self) {
        if let Some(ref path) = self.user_graph_path {
            let _ = self.user_graph.save(path);
        }
    }

    /// Save internet graph to disk
    fn save_internet_graph(&self) {
        if let Some(ref path) = self.internet_graph_path {
            let _ = self.internet_graph.save(path);
        }
    }

    /// Save contradiction log using internal storage format
    fn save_contradiction_log(&self) {
        if let Some(ref path) = self.contradiction_log_path {
            let log = ContradictionLog {
                version: "1.0".to_string(),
                user_id: self.user_graph.user_id.clone(),
                app_version: env!("CARGO_PKG_VERSION").to_string(),
                generated_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                contradictions: self.contradictions.clone(),
                knowledge_pack_versions: HashMap::new(),
            };
            let _ = crate::internal_storage::save_internal(&log, path);
        }
    }

    /// Export contradiction log for reporting
    pub fn export_contradiction_log(&self) -> Result<String, serde_json::Error> {
        let log = ContradictionLog {
            version: "1.0".to_string(),
            user_id: self.user_graph.user_id.clone(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            generated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            contradictions: self.contradictions.clone(),
            knowledge_pack_versions: HashMap::new(),
        };
        log.export_json()
    }

    /// Get statistics
    pub fn stats(&self) -> MultiLayerGraphStats {
        MultiLayerGraphStats {
            session_count: self.session_graphs.len(),
            total_session_entities: self.session_graphs.values().map(|s| s.entities.len()).sum(),
            user_beliefs_count: self.user_graph.beliefs.len(),
            internet_entries_count: self.internet_graph.entries.len(),
            contradiction_count: self.contradictions.len(),
            unresolved_contradictions: self
                .contradictions
                .iter()
                .filter(|c| c.resolution == ContradictionResolution::Unresolved)
                .count(),
        }
    }
}

/// Statistics for the multi-layer graph
#[derive(Clone, Debug, Default)]
pub struct MultiLayerGraphStats {
    pub session_count: usize,
    pub total_session_entities: usize,
    pub user_beliefs_count: usize,
    pub internet_entries_count: usize,
    pub contradiction_count: usize,
    pub unresolved_contradictions: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_extractor() {
        let extractor = BeliefExtractor::new();
        let entities = vec!["Gladius".to_string()];

        let beliefs = extractor.extract(
            "Creo que el Gladius es el mejor fighter",
            "session1",
            &entities,
        );
        assert!(!beliefs.is_empty());
        assert_eq!(beliefs[0].belief_type, BeliefType::Opinion);

        let beliefs = extractor.extract("Tengo un Gladius", "session1", &entities);
        assert!(!beliefs.is_empty());
        assert_eq!(beliefs[0].belief_type, BeliefType::Ownership);

        let beliefs = extractor.extract("Quiero comprar un Sabre", "session1", &entities);
        assert!(!beliefs.is_empty());
        assert_eq!(beliefs[0].belief_type, BeliefType::Goal);
    }

    #[test]
    fn test_session_graph() {
        let mut session = SessionGraph::new("test_session");
        session.add_entity("Gladius", "Ship", "user_message");
        session.add_entity("Sabre", "Ship", "user_message");

        assert!(session.has_entity("Gladius"));
        assert!(session.has_entity("Sabre"));
        assert!(!session.has_entity("Arrow"));
        assert_eq!(session.entity_names().len(), 2);
    }

    #[test]
    fn test_user_graph() {
        let mut graph = UserGraph::new("user1");

        let belief = UserBelief::new(
            "I think the Gladius is the best",
            Some("Gladius".to_string()),
            BeliefType::Opinion,
            "session1",
            0.8,
        );
        graph.add_belief(belief);

        let beliefs = graph.beliefs_about("Gladius");
        assert_eq!(beliefs.len(), 1);

        let opinions = graph.beliefs_of_type(BeliefType::Opinion);
        assert_eq!(opinions.len(), 1);
    }

    #[test]
    fn test_contradiction_detection() {
        let mut mlg = MultiLayerGraph::new();

        let contradiction = mlg.add_internet_data(
            "Sabre",
            "shields",
            "3",
            "https://reddit.com/example",
            Some("2"),
        );

        assert!(contradiction.is_some());
        assert_eq!(mlg.contradictions.len(), 1);
        assert_eq!(mlg.get_unresolved_contradictions().len(), 1);
    }

    #[test]
    fn test_multi_layer_graph_context() {
        let mut mlg = MultiLayerGraph::new();

        // Add user belief
        mlg.user_graph.add_belief(UserBelief::new(
            "The Gladius is great for AC",
            Some("Gladius".to_string()),
            BeliefType::Opinion,
            "session1",
            0.8,
        ));

        // Add session entity
        let session = mlg.get_or_create_session("session1");
        session.add_entity("Gladius", "Ship", "test");

        let context = mlg.build_context("session1", &["Gladius".to_string()]);
        assert!(context.contains("User Beliefs"));
        assert!(context.contains("Previously Mentioned"));
    }
}
