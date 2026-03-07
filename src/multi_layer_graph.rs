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
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphLayer {
    /// Primary knowledge from verified sources (knowledge packs)
    Knowledge,
    /// Complementary data from internet sources
    Internet,
    /// User-stated preferences and beliefs
    User,
    /// Temporary data from current conversation
    Session,
    /// User-defined custom layer
    Custom(String),
}

impl GraphLayer {
    /// Get the priority of this layer (higher = more trusted)
    pub fn priority(&self) -> u8 {
        match self {
            GraphLayer::Knowledge => 100,
            GraphLayer::User => 80, // User beliefs are important but not verified
            GraphLayer::Internet => 50, // Internet data needs verification
            GraphLayer::Session => 30, // Session data is contextual
            GraphLayer::Custom(_) => 0, // Custom layers have lowest default priority
        }
    }

    /// Get display name
    pub fn display_name(&self) -> String {
        match self {
            GraphLayer::Knowledge => "Knowledge Pack".to_string(),
            GraphLayer::User => "User Belief".to_string(),
            GraphLayer::Internet => "Internet".to_string(),
            GraphLayer::Session => "Session".to_string(),
            GraphLayer::Custom(name) => name.clone(),
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

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::MultiLayerGraph,
            self.entities.len(),
        );
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
    /// Configuration for layer priorities and behaviors (7.1)
    pub layer_configs: HashMap<GraphLayer, LayerConfig>,
    /// Custom layers beyond the 4 built-in ones (7.2)
    pub custom_layers: HashMap<String, Vec<LayeredEntity>>,
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
            layer_configs: HashMap::new(),
            custom_layers: HashMap::new(),
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
            .expect("session graph must exist after insert in get_or_create_session")
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
// V4 ADDITIONS: Multi-layer graph improvements (items 7.1-7.7, 8.5)
// =============================================================================

/// Configuration for a specific graph layer (7.1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Priority override for the layer (higher = more trusted)
    pub priority: u8,
    /// Sync policy for multi-agent contexts (8.5)
    pub sync_policy: SyncPolicy,
    /// How to resolve intra-layer entity conflicts (7.6)
    pub conflict_policy: ConflictPolicy,
    /// Maximum number of entities allowed in this layer
    pub max_entities: Option<usize>,
}

impl LayerConfig {
    /// Create a new layer configuration with defaults
    pub fn new(priority: u8) -> Self {
        Self {
            priority,
            sync_policy: SyncPolicy::Shared,
            conflict_policy: ConflictPolicy::LastWriteWins,
            max_entities: None,
        }
    }
}

/// Sync policy for a layer in multi-agent contexts (8.5)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncPolicy {
    /// All agents can read and write
    Shared,
    /// Only the owning agent can write
    Private,
    /// No modifications allowed
    ReadOnly,
}

/// How to resolve intra-layer entity conflicts (7.6)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// Most recently written entity wins
    LastWriteWins,
    /// Entity with highest confidence wins
    HighestConfidence,
    /// Merge attributes from both entities
    Merge,
    /// Require manual resolution
    Manual,
}

/// Unified view across all layers (7.3)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedView {
    /// All entities merged from all layers
    pub entities: Vec<UnifiedEntity>,
    /// All relations from all layers
    pub relations: Vec<UnifiedRelation>,
}

/// An entity merged from multiple layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedEntity {
    /// Entity name
    pub name: String,
    /// Entity type
    pub entity_type: String,
    /// Which layers this entity appears in
    pub layers: Vec<GraphLayer>,
    /// Merged attributes from all layers
    pub merged_attributes: HashMap<String, serde_json::Value>,
    /// Combined confidence (weighted by layer priority)
    pub confidence: f32,
}

/// A relation in the unified view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRelation {
    /// Source entity name
    pub source: String,
    /// Target entity name
    pub target: String,
    /// Relation type (e.g., "belongs_to", "related_to")
    pub relation_type: String,
    /// Which layer this relation originated from
    pub layer: GraphLayer,
    /// Confidence score
    pub confidence: f32,
}

/// Cluster of related entities (7.4)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCluster {
    /// Unique cluster identifier
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Names of entities in this cluster
    pub entity_names: Vec<String>,
    /// Layer this cluster belongs to
    pub layer: GraphLayer,
    /// Cohesion score (0.0-1.0): how tightly related the entities are
    pub cohesion: f64,
}

/// Inferred cross-layer relation (7.5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredRelation {
    /// Source entity name
    pub source_entity: String,
    /// Layer of the source entity
    pub source_layer: GraphLayer,
    /// Target entity name
    pub target_entity: String,
    /// Layer of the target entity
    pub target_layer: GraphLayer,
    /// Type of inferred relation
    pub relation_type: String,
    /// Confidence in the inference (0.0-1.0)
    pub confidence: f32,
    /// Decay factor over time (0.0-1.0, 1.0 = no decay)
    pub decay: f64,
}

/// Diff between two graph states (7.7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Entities added (layer, entity name)
    pub added_entities: Vec<(GraphLayer, String)>,
    /// Entities removed (layer, entity name)
    pub removed_entities: Vec<(GraphLayer, String)>,
    /// Entities modified (layer, name, description of change)
    pub modified_entities: Vec<(GraphLayer, String, String)>,
    /// Relations added (source, relation_type, target)
    pub added_relations: Vec<(String, String, String)>,
    /// Relations removed (source, relation_type, target)
    pub removed_relations: Vec<(String, String, String)>,
}

impl GraphDiff {
    /// Create an empty diff
    pub fn empty() -> Self {
        Self {
            added_entities: Vec::new(),
            removed_entities: Vec::new(),
            modified_entities: Vec::new(),
            added_relations: Vec::new(),
            removed_relations: Vec::new(),
        }
    }

    /// Whether this diff has any changes
    pub fn is_empty(&self) -> bool {
        self.added_entities.is_empty()
            && self.removed_entities.is_empty()
            && self.modified_entities.is_empty()
            && self.added_relations.is_empty()
            && self.removed_relations.is_empty()
    }
}

/// Merge strategy for combining diffs (7.7)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep our version on conflict
    Ours,
    /// Keep their version on conflict
    Theirs,
    /// Include both sides (union)
    Union,
}

// =============================================================================
// V4 METHODS on MultiLayerGraph
// =============================================================================

impl MultiLayerGraph {
    // =========================================================================
    // 7.1: Layer configuration
    // =========================================================================

    /// Configure a layer with custom priority and policies
    pub fn configure_layer(&mut self, layer: GraphLayer, config: LayerConfig) {
        self.layer_configs.insert(layer, config);
    }

    /// Get the configuration for a layer, if any
    pub fn get_layer_config(&self, layer: &GraphLayer) -> Option<&LayerConfig> {
        self.layer_configs.get(layer)
    }

    /// Get effective priority for a layer (config override or built-in default)
    pub fn effective_priority(&self, layer: &GraphLayer) -> u8 {
        self.layer_configs
            .get(layer)
            .map(|c| c.priority)
            .unwrap_or_else(|| layer.priority())
    }

    // =========================================================================
    // 7.2: Custom layers
    // =========================================================================

    /// Register a custom layer and return the corresponding GraphLayer variant
    pub fn add_custom_layer(&mut self, name: &str) -> GraphLayer {
        self.custom_layers
            .entry(name.to_string())
            .or_insert_with(Vec::new);
        GraphLayer::Custom(name.to_string())
    }

    /// Get entities in a custom layer
    pub fn get_custom_layer(&self, name: &str) -> Option<&Vec<LayeredEntity>> {
        self.custom_layers.get(name)
    }

    /// Add an entity to a custom layer
    pub fn add_to_custom_layer(
        &mut self,
        layer_name: &str,
        entity: LayeredEntity,
    ) -> Result<(), String> {
        let entities = self
            .custom_layers
            .get_mut(layer_name)
            .ok_or_else(|| format!("Custom layer '{}' not found", layer_name))?;

        // Check max_entities limit from config
        let layer_key = GraphLayer::Custom(layer_name.to_string());
        if let Some(config) = self.layer_configs.get(&layer_key) {
            if let Some(max) = config.max_entities {
                if entities.len() >= max {
                    return Err(format!(
                        "Custom layer '{}' reached max entities limit ({})",
                        layer_name, max
                    ));
                }
            }
        }

        // Don't add duplicates
        if !entities.iter().any(|e| e.name == entity.name) {
            entities.push(entity);
        }
        Ok(())
    }

    // =========================================================================
    // 7.3: Unified view
    // =========================================================================

    /// Build a unified view across all layers, optionally filtered by session
    pub fn query_unified(&self, session_id: Option<&str>) -> UnifiedView {
        let mut entity_map: HashMap<String, UnifiedEntity> = HashMap::new();
        let mut relations = Vec::new();

        // Collect session entities
        if let Some(sid) = session_id {
            if let Some(session) = self.session_graphs.get(sid) {
                for entity in &session.entities {
                    let entry = entity_map
                        .entry(entity.name.clone())
                        .or_insert_with(|| UnifiedEntity {
                            name: entity.name.clone(),
                            entity_type: entity.entity_type.clone(),
                            layers: Vec::new(),
                            merged_attributes: HashMap::new(),
                            confidence: 0.0,
                        });
                    if !entry.layers.contains(&GraphLayer::Session) {
                        entry.layers.push(GraphLayer::Session);
                    }
                    let priority = self.effective_priority(&GraphLayer::Session) as f32;
                    if priority > entry.confidence {
                        entry.confidence = entity.confidence.score() * (priority / 100.0);
                    }
                }
                // Session relations
                for (from, rel_type, to) in &session.relations {
                    relations.push(UnifiedRelation {
                        source: from.clone(),
                        target: to.clone(),
                        relation_type: rel_type.clone(),
                        layer: GraphLayer::Session,
                        confidence: 0.3,
                    });
                }
            }
        } else {
            // If no session specified, include all sessions
            for session in self.session_graphs.values() {
                for entity in &session.entities {
                    let entry = entity_map
                        .entry(entity.name.clone())
                        .or_insert_with(|| UnifiedEntity {
                            name: entity.name.clone(),
                            entity_type: entity.entity_type.clone(),
                            layers: Vec::new(),
                            merged_attributes: HashMap::new(),
                            confidence: 0.0,
                        });
                    if !entry.layers.contains(&GraphLayer::Session) {
                        entry.layers.push(GraphLayer::Session);
                    }
                }
                for (from, rel_type, to) in &session.relations {
                    relations.push(UnifiedRelation {
                        source: from.clone(),
                        target: to.clone(),
                        relation_type: rel_type.clone(),
                        layer: GraphLayer::Session,
                        confidence: 0.3,
                    });
                }
            }
        }

        // Collect internet entities (non-expired)
        for entry in &self.internet_graph.entries {
            if !entry.is_expired() {
                let unified = entity_map
                    .entry(entry.entity.clone())
                    .or_insert_with(|| UnifiedEntity {
                        name: entry.entity.clone(),
                        entity_type: "InternetEntity".to_string(),
                        layers: Vec::new(),
                        merged_attributes: HashMap::new(),
                        confidence: 0.0,
                    });
                if !unified.layers.contains(&GraphLayer::Internet) {
                    unified.layers.push(GraphLayer::Internet);
                }
                // Store attribute as merged attribute
                unified.merged_attributes.insert(
                    entry.attribute.clone(),
                    serde_json::Value::String(entry.value.clone()),
                );
                let priority = self.effective_priority(&GraphLayer::Internet) as f32;
                let score = 0.6 * (priority / 100.0);
                if score > unified.confidence {
                    unified.confidence = score;
                }
            }
        }

        // Collect user belief entities
        for belief in self.user_graph.all_active_beliefs() {
            if let Some(ref subject) = belief.subject_entity {
                let unified = entity_map
                    .entry(subject.clone())
                    .or_insert_with(|| UnifiedEntity {
                        name: subject.clone(),
                        entity_type: "UserEntity".to_string(),
                        layers: Vec::new(),
                        merged_attributes: HashMap::new(),
                        confidence: 0.0,
                    });
                if !unified.layers.contains(&GraphLayer::User) {
                    unified.layers.push(GraphLayer::User);
                }
                unified.merged_attributes.insert(
                    format!("belief_{}", belief.belief_type.display_name()),
                    serde_json::Value::String(belief.statement.clone()),
                );
                let priority = self.effective_priority(&GraphLayer::User) as f32;
                let score = belief.confidence * (priority / 100.0);
                if score > unified.confidence {
                    unified.confidence = score;
                }
            }
        }

        // Collect custom layer entities
        for (layer_name, entities) in &self.custom_layers {
            let layer = GraphLayer::Custom(layer_name.clone());
            for entity in entities {
                let unified = entity_map
                    .entry(entity.name.clone())
                    .or_insert_with(|| UnifiedEntity {
                        name: entity.name.clone(),
                        entity_type: entity.entity_type.clone(),
                        layers: Vec::new(),
                        merged_attributes: HashMap::new(),
                        confidence: 0.0,
                    });
                if !unified.layers.contains(&layer) {
                    unified.layers.push(layer.clone());
                }
                let priority = self.effective_priority(&layer) as f32;
                let score = entity.confidence.score() * (priority / 100.0);
                if score > unified.confidence {
                    unified.confidence = score;
                }
            }
        }

        // Sort entities by confidence descending
        let mut entities: Vec<UnifiedEntity> = entity_map.into_values().collect();
        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        UnifiedView {
            entities,
            relations,
        }
    }

    // =========================================================================
    // 7.4: Entity clustering
    // =========================================================================

    /// Cluster entities in a layer based on shared relations.
    /// Returns clusters where entities are connected by at least one relation.
    pub fn cluster_entities(
        &self,
        layer: &GraphLayer,
        session_id: Option<&str>,
        min_cluster_size: usize,
    ) -> Vec<GraphCluster> {
        // Gather entities and relations for the specified layer
        let (entities, relations) = self.collect_layer_data(layer, session_id);

        if entities.is_empty() {
            return Vec::new();
        }

        // Build adjacency from relations
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for (from, _rel, to) in &relations {
            adjacency
                .entry(from.clone())
                .or_default()
                .push(to.clone());
            adjacency
                .entry(to.clone())
                .or_default()
                .push(from.clone());
        }

        // Connected components via BFS
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut clusters = Vec::new();
        let mut cluster_idx = 0u64;

        for entity_name in &entities {
            if visited.contains(entity_name) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(entity_name.clone());
            visited.insert(entity_name.clone());

            while let Some(current) = queue.pop_front() {
                component.push(current.clone());
                if let Some(neighbors) = adjacency.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) && entities.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }

            if component.len() >= min_cluster_size {
                // Cohesion: ratio of actual edges to possible edges
                let n = component.len() as f64;
                let max_edges = if n > 1.0 { n * (n - 1.0) / 2.0 } else { 1.0 };
                let mut actual_edges = 0u64;
                for (from, _rel, to) in &relations {
                    if component.contains(from) && component.contains(to) {
                        actual_edges += 1;
                    }
                }
                let cohesion = (actual_edges as f64) / max_edges;

                clusters.push(GraphCluster {
                    id: format!("cluster_{}", cluster_idx),
                    label: format!(
                        "{} cluster {}",
                        layer.display_name(),
                        cluster_idx
                    ),
                    entity_names: component,
                    layer: layer.clone(),
                    cohesion,
                });
                cluster_idx += 1;
            }
        }

        clusters
    }

    /// Collect entity names and relations for a specific layer
    fn collect_layer_data(
        &self,
        layer: &GraphLayer,
        session_id: Option<&str>,
    ) -> (Vec<String>, Vec<(String, String, String)>) {
        let mut entities = Vec::new();
        let mut relations = Vec::new();

        match layer {
            GraphLayer::Session => {
                if let Some(sid) = session_id {
                    if let Some(session) = self.session_graphs.get(sid) {
                        entities = session.entities.iter().map(|e| e.name.clone()).collect();
                        relations = session.relations.clone();
                    }
                } else {
                    for session in self.session_graphs.values() {
                        for e in &session.entities {
                            if !entities.contains(&e.name) {
                                entities.push(e.name.clone());
                            }
                        }
                        relations.extend(session.relations.clone());
                    }
                }
            }
            GraphLayer::Internet => {
                for entry in &self.internet_graph.entries {
                    if !entry.is_expired() && !entities.contains(&entry.entity) {
                        entities.push(entry.entity.clone());
                    }
                }
                // Internet entries don't have explicit relations in the current model
            }
            GraphLayer::User => {
                for belief in self.user_graph.all_active_beliefs() {
                    if let Some(ref subject) = belief.subject_entity {
                        if !entities.contains(subject) {
                            entities.push(subject.clone());
                        }
                    }
                }
            }
            GraphLayer::Knowledge => {
                // Knowledge layer entities come from external knowledge packs;
                // we don't store them directly in MultiLayerGraph.
            }
            GraphLayer::Custom(name) => {
                if let Some(layer_entities) = self.custom_layers.get(name) {
                    entities = layer_entities.iter().map(|e| e.name.clone()).collect();
                }
            }
        }

        (entities, relations)
    }

    // =========================================================================
    // 7.5: Cross-layer inference
    // =========================================================================

    /// Infer relations between entities that appear in multiple layers.
    /// If the same entity name appears in two different layers, we create an
    /// inferred "same_as" relation. Confidence is based on name match quality.
    pub fn infer_cross_layer(&self, session_id: Option<&str>) -> Vec<InferredRelation> {
        let mut inferred = Vec::new();

        // Collect all entity names per layer
        let layers = [
            GraphLayer::Session,
            GraphLayer::User,
            GraphLayer::Internet,
            GraphLayer::Knowledge,
        ];
        let mut layer_entities: Vec<(GraphLayer, Vec<String>)> = Vec::new();

        for layer in &layers {
            let (entities, _) = self.collect_layer_data(layer, session_id);
            if !entities.is_empty() {
                layer_entities.push((layer.clone(), entities));
            }
        }

        // Also include custom layers
        for layer_name in self.custom_layers.keys() {
            let layer = GraphLayer::Custom(layer_name.clone());
            let (entities, _) = self.collect_layer_data(&layer, session_id);
            if !entities.is_empty() {
                layer_entities.push((layer, entities));
            }
        }

        // Compare every pair of layers
        for i in 0..layer_entities.len() {
            for j in (i + 1)..layer_entities.len() {
                let (ref layer_a, ref entities_a) = layer_entities[i];
                let (ref layer_b, ref entities_b) = layer_entities[j];

                for name_a in entities_a {
                    for name_b in entities_b {
                        if name_a.eq_ignore_ascii_case(name_b) {
                            inferred.push(InferredRelation {
                                source_entity: name_a.clone(),
                                source_layer: layer_a.clone(),
                                target_entity: name_b.clone(),
                                target_layer: layer_b.clone(),
                                relation_type: "same_as".to_string(),
                                confidence: if name_a == name_b { 1.0 } else { 0.9 },
                                decay: 1.0, // No decay initially
                            });
                        }
                    }
                }
            }
        }

        inferred
    }

    // =========================================================================
    // 7.6: Conflict resolution
    // =========================================================================

    /// Resolve a conflict for an entity within a specific layer.
    /// When multiple entries exist for the same entity, use the given policy
    /// to pick or merge the result.
    pub fn resolve_conflict(
        &self,
        entity_name: &str,
        layer: &GraphLayer,
        policy: &ConflictPolicy,
    ) -> Option<LayeredEntity> {
        // Collect all matching entities in this layer
        let candidates = self.find_entities_in_layer(entity_name, layer);

        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(candidates[0].clone());
        }

        match policy {
            ConflictPolicy::LastWriteWins => {
                // Pick the one with the highest timestamp
                candidates
                    .iter()
                    .max_by_key(|e| e.timestamp)
                    .cloned()
            }
            ConflictPolicy::HighestConfidence => {
                // Pick the one with the highest confidence score
                candidates
                    .iter()
                    .max_by(|a, b| {
                        a.confidence
                            .score()
                            .partial_cmp(&b.confidence.score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
            }
            ConflictPolicy::Merge => {
                // Merge: take the most recent timestamp, highest confidence,
                // and combine sources
                let mut merged = candidates[0].clone();
                for candidate in &candidates[1..] {
                    if candidate.timestamp > merged.timestamp {
                        merged.timestamp = candidate.timestamp;
                    }
                    if candidate.confidence.score() > merged.confidence.score() {
                        merged.confidence = candidate.confidence;
                    }
                    if !merged.source.contains(&candidate.source) {
                        merged.source = format!("{}, {}", merged.source, candidate.source);
                    }
                }
                Some(merged)
            }
            ConflictPolicy::Manual => {
                // For manual resolution, return the earliest candidate (by timestamp)
                // as a placeholder; the caller should present all options to the user
                candidates
                    .iter()
                    .min_by_key(|e| e.timestamp)
                    .cloned()
            }
        }
    }

    /// Find all entities matching a name in a specific layer
    fn find_entities_in_layer(&self, entity_name: &str, layer: &GraphLayer) -> Vec<LayeredEntity> {
        let mut results = Vec::new();

        match layer {
            GraphLayer::Session => {
                for session in self.session_graphs.values() {
                    for entity in &session.entities {
                        if entity.name.eq_ignore_ascii_case(entity_name) {
                            results.push(entity.clone());
                        }
                    }
                }
            }
            GraphLayer::Internet => {
                for entry in &self.internet_graph.entries {
                    if entry.entity.eq_ignore_ascii_case(entity_name) && !entry.is_expired() {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        results.push(LayeredEntity {
                            name: entry.entity.clone(),
                            entity_type: "InternetEntity".to_string(),
                            layer: GraphLayer::Internet,
                            confidence: ConfidenceLevel::WebSource,
                            source: entry.source_url.clone(),
                            timestamp: now,
                            ttl_seconds: Some(entry.ttl_seconds),
                        });
                    }
                }
            }
            GraphLayer::User => {
                for belief in &self.user_graph.beliefs {
                    if belief.active {
                        if let Some(ref subject) = belief.subject_entity {
                            if subject.eq_ignore_ascii_case(entity_name) {
                                results.push(LayeredEntity {
                                    name: subject.clone(),
                                    entity_type: belief.belief_type.display_name().to_string(),
                                    layer: GraphLayer::User,
                                    confidence: ConfidenceLevel::UserStated,
                                    source: format!("belief:{}", belief.id),
                                    timestamp: belief.expressed_at,
                                    ttl_seconds: None,
                                });
                            }
                        }
                    }
                }
            }
            GraphLayer::Knowledge => {
                // Knowledge entities are managed externally; nothing to search here
            }
            GraphLayer::Custom(name) => {
                if let Some(entities) = self.custom_layers.get(name) {
                    for entity in entities {
                        if entity.name.eq_ignore_ascii_case(entity_name) {
                            results.push(entity.clone());
                        }
                    }
                }
            }
        }

        results
    }

    // =========================================================================
    // 7.7: Diff and merge
    // =========================================================================

    /// Compute the diff between this graph and another graph.
    /// Compares session entities (for the given session_id), internet entries,
    /// user beliefs, and custom layers.
    pub fn diff(&self, other: &MultiLayerGraph, session_id: Option<&str>) -> GraphDiff {
        let mut diff = GraphDiff::empty();

        // Compare session entities
        let self_session_entities = self.collect_session_entity_names(session_id);
        let other_session_entities = other.collect_session_entity_names(session_id);

        for name in &other_session_entities {
            if !self_session_entities.contains(name) {
                diff.added_entities
                    .push((GraphLayer::Session, name.clone()));
            }
        }
        for name in &self_session_entities {
            if !other_session_entities.contains(name) {
                diff.removed_entities
                    .push((GraphLayer::Session, name.clone()));
            }
        }

        // Compare session relations
        let self_session_relations = self.collect_session_relations(session_id);
        let other_session_relations = other.collect_session_relations(session_id);

        for rel in &other_session_relations {
            if !self_session_relations.contains(rel) {
                diff.added_relations.push(rel.clone());
            }
        }
        for rel in &self_session_relations {
            if !other_session_relations.contains(rel) {
                diff.removed_relations.push(rel.clone());
            }
        }

        // Compare internet entries
        let self_internet: std::collections::HashSet<String> = self
            .internet_graph
            .entries
            .iter()
            .filter(|e| !e.is_expired())
            .map(|e| e.entity.clone())
            .collect();
        let other_internet: std::collections::HashSet<String> = other
            .internet_graph
            .entries
            .iter()
            .filter(|e| !e.is_expired())
            .map(|e| e.entity.clone())
            .collect();

        for name in &other_internet {
            if !self_internet.contains(name) {
                diff.added_entities
                    .push((GraphLayer::Internet, name.clone()));
            }
        }
        for name in &self_internet {
            if !other_internet.contains(name) {
                diff.removed_entities
                    .push((GraphLayer::Internet, name.clone()));
            }
        }

        // Compare custom layers
        for (layer_name, other_entities) in &other.custom_layers {
            let layer = GraphLayer::Custom(layer_name.clone());
            let self_entities = self
                .custom_layers
                .get(layer_name)
                .map(|v| v.iter().map(|e| e.name.clone()).collect::<Vec<_>>())
                .unwrap_or_default();
            let other_names: Vec<String> = other_entities.iter().map(|e| e.name.clone()).collect();

            for name in &other_names {
                if !self_entities.contains(name) {
                    diff.added_entities.push((layer.clone(), name.clone()));
                }
            }
        }
        for (layer_name, self_entities) in &self.custom_layers {
            let layer = GraphLayer::Custom(layer_name.clone());
            let other_names: Vec<String> = other
                .custom_layers
                .get(layer_name)
                .map(|v| v.iter().map(|e| e.name.clone()).collect())
                .unwrap_or_default();

            for entity in self_entities {
                if !other_names.contains(&entity.name) {
                    diff.removed_entities
                        .push((layer.clone(), entity.name.clone()));
                }
            }
        }

        diff
    }

    /// Apply a diff to this graph using the specified merge strategy
    pub fn apply_diff(&mut self, diff: &GraphDiff, session_id: &str, strategy: &MergeStrategy) {
        match strategy {
            MergeStrategy::Union => {
                // Add all added entities and relations, remove all removed
                self.apply_additions(diff, session_id);
                self.apply_removals(diff, session_id);
            }
            MergeStrategy::Theirs => {
                // Accept all changes from the diff
                self.apply_additions(diff, session_id);
                self.apply_removals(diff, session_id);
            }
            MergeStrategy::Ours => {
                // Keep our state: only apply additions (don't remove anything)
                self.apply_additions(diff, session_id);
            }
        }
    }

    /// Apply entity and relation additions from a diff
    fn apply_additions(&mut self, diff: &GraphDiff, session_id: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (layer, name) in &diff.added_entities {
            match layer {
                GraphLayer::Session => {
                    let session = self.get_or_create_session(session_id);
                    session.add_entity(name, "Unknown", "diff_apply");
                }
                GraphLayer::Custom(layer_name) => {
                    // Ensure custom layer exists
                    self.custom_layers
                        .entry(layer_name.clone())
                        .or_insert_with(Vec::new);
                    let entity = LayeredEntity {
                        name: name.clone(),
                        entity_type: "Unknown".to_string(),
                        layer: layer.clone(),
                        confidence: ConfidenceLevel::Inferred,
                        source: "diff_apply".to_string(),
                        timestamp: now,
                        ttl_seconds: None,
                    };
                    let _ = self.add_to_custom_layer(layer_name, entity);
                }
                _ => {
                    // Internet and Knowledge additions require more context;
                    // add to session as a fallback
                    let session = self.get_or_create_session(session_id);
                    session.add_entity(name, "Unknown", "diff_apply");
                }
            }
        }

        for (from, rel_type, to) in &diff.added_relations {
            let session = self.get_or_create_session(session_id);
            session.add_relation(from, rel_type, to);
        }
    }

    /// Apply entity and relation removals from a diff
    fn apply_removals(&mut self, diff: &GraphDiff, session_id: &str) {
        for (layer, name) in &diff.removed_entities {
            match layer {
                GraphLayer::Session => {
                    if let Some(session) = self.session_graphs.get_mut(session_id) {
                        session.entities.retain(|e| e.name != *name);
                    }
                }
                GraphLayer::Custom(layer_name) => {
                    if let Some(entities) = self.custom_layers.get_mut(layer_name) {
                        entities.retain(|e| e.name != *name);
                    }
                }
                _ => {
                    // For Internet/Knowledge layers, skip removal (managed externally)
                }
            }
        }

        for (from, rel_type, to) in &diff.removed_relations {
            if let Some(session) = self.session_graphs.get_mut(session_id) {
                session
                    .relations
                    .retain(|r| !(r.0 == *from && r.1 == *rel_type && r.2 == *to));
            }
        }
    }

    /// Collect session entity names for diff comparison
    fn collect_session_entity_names(&self, session_id: Option<&str>) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(sid) = session_id {
            if let Some(session) = self.session_graphs.get(sid) {
                names = session.entities.iter().map(|e| e.name.clone()).collect();
            }
        } else {
            for session in self.session_graphs.values() {
                for e in &session.entities {
                    if !names.contains(&e.name) {
                        names.push(e.name.clone());
                    }
                }
            }
        }
        names
    }

    /// Collect session relations for diff comparison
    fn collect_session_relations(
        &self,
        session_id: Option<&str>,
    ) -> Vec<(String, String, String)> {
        let mut rels = Vec::new();
        if let Some(sid) = session_id {
            if let Some(session) = self.session_graphs.get(sid) {
                rels = session.relations.clone();
            }
        } else {
            for session in self.session_graphs.values() {
                rels.extend(session.relations.clone());
            }
        }
        rels
    }
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

    // =========================================================================
    // V4 Tests: GraphLayer::Custom variant
    // =========================================================================

    #[test]
    fn test_graph_layer_custom() {
        let layer = GraphLayer::Custom("MyLayer".to_string());
        assert_eq!(layer, GraphLayer::Custom("MyLayer".to_string()));
        assert_ne!(layer, GraphLayer::Session);
        assert_ne!(layer, GraphLayer::Custom("Other".to_string()));
    }

    #[test]
    fn test_graph_layer_custom_clone() {
        let layer = GraphLayer::Custom("Test".to_string());
        let cloned = layer.clone();
        assert_eq!(layer, cloned);
    }

    #[test]
    fn test_graph_layer_display_name_builtin() {
        assert_eq!(GraphLayer::Knowledge.display_name(), "Knowledge Pack");
        assert_eq!(GraphLayer::User.display_name(), "User Belief");
        assert_eq!(GraphLayer::Internet.display_name(), "Internet");
        assert_eq!(GraphLayer::Session.display_name(), "Session");
    }

    #[test]
    fn test_graph_layer_display_name_custom() {
        let layer = GraphLayer::Custom("Research Notes".to_string());
        assert_eq!(layer.display_name(), "Research Notes");
    }

    #[test]
    fn test_graph_layer_priority_builtin() {
        assert_eq!(GraphLayer::Knowledge.priority(), 100);
        assert_eq!(GraphLayer::User.priority(), 80);
        assert_eq!(GraphLayer::Internet.priority(), 50);
        assert_eq!(GraphLayer::Session.priority(), 30);
    }

    #[test]
    fn test_graph_layer_priority_custom() {
        let layer = GraphLayer::Custom("custom".to_string());
        assert_eq!(layer.priority(), 0);
    }

    #[test]
    fn test_graph_layer_hash_custom() {
        let mut set = std::collections::HashSet::new();
        set.insert(GraphLayer::Custom("A".to_string()));
        set.insert(GraphLayer::Custom("B".to_string()));
        set.insert(GraphLayer::Custom("A".to_string())); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_graph_layer_serialize_custom() {
        let layer = GraphLayer::Custom("Serialized".to_string());
        let json = serde_json::to_string(&layer).expect("serialize");
        assert!(json.contains("Custom"));
        assert!(json.contains("Serialized"));
        let deserialized: GraphLayer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, layer);
    }

    // =========================================================================
    // V4 Tests: LayerConfig
    // =========================================================================

    #[test]
    fn test_layer_config_creation() {
        let config = LayerConfig::new(50);
        assert_eq!(config.priority, 50);
        assert_eq!(config.sync_policy, SyncPolicy::Shared);
        assert_eq!(config.conflict_policy, ConflictPolicy::LastWriteWins);
        assert!(config.max_entities.is_none());
    }

    #[test]
    fn test_layer_config_with_max_entities() {
        let config = LayerConfig {
            priority: 75,
            sync_policy: SyncPolicy::Private,
            conflict_policy: ConflictPolicy::HighestConfidence,
            max_entities: Some(100),
        };
        assert_eq!(config.priority, 75);
        assert_eq!(config.max_entities, Some(100));
    }

    #[test]
    fn test_layer_config_serialize() {
        let config = LayerConfig::new(42);
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: LayerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.priority, 42);
    }

    // =========================================================================
    // V4 Tests: SyncPolicy
    // =========================================================================

    #[test]
    fn test_sync_policy_variants() {
        assert_eq!(SyncPolicy::Shared, SyncPolicy::Shared);
        assert_ne!(SyncPolicy::Shared, SyncPolicy::Private);
        assert_ne!(SyncPolicy::Private, SyncPolicy::ReadOnly);
    }

    #[test]
    fn test_sync_policy_serialize() {
        let policy = SyncPolicy::Private;
        let json = serde_json::to_string(&policy).expect("serialize");
        let deserialized: SyncPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, SyncPolicy::Private);
    }

    #[test]
    fn test_sync_policy_clone() {
        let policy = SyncPolicy::ReadOnly;
        let cloned = policy.clone();
        assert_eq!(policy, cloned);
    }

    // =========================================================================
    // V4 Tests: ConflictPolicy
    // =========================================================================

    #[test]
    fn test_conflict_policy_variants() {
        assert_eq!(ConflictPolicy::LastWriteWins, ConflictPolicy::LastWriteWins);
        assert_ne!(ConflictPolicy::LastWriteWins, ConflictPolicy::HighestConfidence);
        assert_ne!(ConflictPolicy::Merge, ConflictPolicy::Manual);
        let all = [
            ConflictPolicy::LastWriteWins,
            ConflictPolicy::HighestConfidence,
            ConflictPolicy::Merge,
            ConflictPolicy::Manual,
        ];
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_conflict_policy_serialize() {
        let policy = ConflictPolicy::Merge;
        let json = serde_json::to_string(&policy).expect("serialize");
        let deserialized: ConflictPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, ConflictPolicy::Merge);
    }

    // =========================================================================
    // V4 Tests: Configure layer (7.1)
    // =========================================================================

    #[test]
    fn test_configure_layer() {
        let mut mlg = MultiLayerGraph::new();
        let config = LayerConfig::new(90);
        mlg.configure_layer(GraphLayer::Session, config);
        assert!(mlg.get_layer_config(&GraphLayer::Session).is_some());
        assert_eq!(mlg.get_layer_config(&GraphLayer::Session).map(|c| c.priority), Some(90));
    }

    #[test]
    fn test_configure_layer_custom() {
        let mut mlg = MultiLayerGraph::new();
        let layer = GraphLayer::Custom("special".to_string());
        let config = LayerConfig {
            priority: 60,
            sync_policy: SyncPolicy::Private,
            conflict_policy: ConflictPolicy::HighestConfidence,
            max_entities: Some(50),
        };
        mlg.configure_layer(layer.clone(), config);
        let retrieved = mlg.get_layer_config(&layer).expect("config should exist");
        assert_eq!(retrieved.priority, 60);
        assert_eq!(retrieved.sync_policy, SyncPolicy::Private);
        assert_eq!(retrieved.max_entities, Some(50));
    }

    #[test]
    fn test_get_layer_config_none() {
        let mlg = MultiLayerGraph::new();
        assert!(mlg.get_layer_config(&GraphLayer::Knowledge).is_none());
    }

    #[test]
    fn test_effective_priority_default() {
        let mlg = MultiLayerGraph::new();
        assert_eq!(mlg.effective_priority(&GraphLayer::Knowledge), 100);
        assert_eq!(mlg.effective_priority(&GraphLayer::Session), 30);
    }

    #[test]
    fn test_effective_priority_override() {
        let mut mlg = MultiLayerGraph::new();
        mlg.configure_layer(GraphLayer::Session, LayerConfig::new(99));
        assert_eq!(mlg.effective_priority(&GraphLayer::Session), 99);
    }

    // =========================================================================
    // V4 Tests: Custom layers (7.2)
    // =========================================================================

    #[test]
    fn test_add_custom_layer() {
        let mut mlg = MultiLayerGraph::new();
        let layer = mlg.add_custom_layer("research");
        assert_eq!(layer, GraphLayer::Custom("research".to_string()));
        assert!(mlg.get_custom_layer("research").is_some());
        assert!(mlg.get_custom_layer("research").expect("exists").is_empty());
    }

    #[test]
    fn test_add_custom_layer_idempotent() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("notes");
        mlg.add_custom_layer("notes"); // second call shouldn't create duplicate
        assert_eq!(mlg.custom_layers.len(), 1);
    }

    #[test]
    fn test_get_custom_layer_not_found() {
        let mlg = MultiLayerGraph::new();
        assert!(mlg.get_custom_layer("nonexistent").is_none());
    }

    #[test]
    fn test_custom_layer_entities() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("research");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let entity = LayeredEntity {
            name: "Quantum Computing".to_string(),
            entity_type: "Topic".to_string(),
            layer: GraphLayer::Custom("research".to_string()),
            confidence: ConfidenceLevel::Verified,
            source: "paper".to_string(),
            timestamp: now,
            ttl_seconds: None,
        };
        let result = mlg.add_to_custom_layer("research", entity);
        assert!(result.is_ok());
        let entities = mlg.get_custom_layer("research").expect("layer exists");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Quantum Computing");
    }

    #[test]
    fn test_add_to_custom_layer_not_found() {
        let mut mlg = MultiLayerGraph::new();
        let entity = LayeredEntity {
            name: "Test".to_string(),
            entity_type: "Unknown".to_string(),
            layer: GraphLayer::Custom("missing".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };
        let result = mlg.add_to_custom_layer("missing", entity);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_custom_layer_max_entities() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("limited");
        let layer = GraphLayer::Custom("limited".to_string());
        mlg.configure_layer(layer, LayerConfig {
            priority: 10,
            sync_policy: SyncPolicy::Shared,
            conflict_policy: ConflictPolicy::LastWriteWins,
            max_entities: Some(2),
        });

        let make_entity = |name: &str| LayeredEntity {
            name: name.to_string(),
            entity_type: "Item".to_string(),
            layer: GraphLayer::Custom("limited".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };

        assert!(mlg.add_to_custom_layer("limited", make_entity("A")).is_ok());
        assert!(mlg.add_to_custom_layer("limited", make_entity("B")).is_ok());
        let result = mlg.add_to_custom_layer("limited", make_entity("C"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max entities limit"));
    }

    #[test]
    fn test_custom_layer_no_duplicate_entities() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("dedup");
        let entity = LayeredEntity {
            name: "Same".to_string(),
            entity_type: "Item".to_string(),
            layer: GraphLayer::Custom("dedup".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };
        let _ = mlg.add_to_custom_layer("dedup", entity.clone());
        let _ = mlg.add_to_custom_layer("dedup", entity);
        assert_eq!(mlg.get_custom_layer("dedup").expect("exists").len(), 1);
    }

    // =========================================================================
    // V4 Tests: Unified view (7.3)
    // =========================================================================

    #[test]
    fn test_query_unified_empty() {
        let mlg = MultiLayerGraph::new();
        let view = mlg.query_unified(None);
        assert!(view.entities.is_empty());
        assert!(view.relations.is_empty());
    }

    #[test]
    fn test_query_unified_with_session_entities() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("Alpha", "Type1", "test");
        session.add_entity("Beta", "Type2", "test");

        let view = mlg.query_unified(Some("s1"));
        assert_eq!(view.entities.len(), 2);
        let names: Vec<&str> = view.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alpha"));
        assert!(names.contains(&"Beta"));
    }

    #[test]
    fn test_query_unified_with_session_relations() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_relation("A", "related_to", "B");

        let view = mlg.query_unified(Some("s1"));
        assert_eq!(view.relations.len(), 1);
        assert_eq!(view.relations[0].source, "A");
        assert_eq!(view.relations[0].target, "B");
        assert_eq!(view.relations[0].relation_type, "related_to");
        assert_eq!(view.relations[0].layer, GraphLayer::Session);
    }

    #[test]
    fn test_query_unified_with_user_beliefs() {
        let mut mlg = MultiLayerGraph::new();
        mlg.user_graph.add_belief(UserBelief::new(
            "I like Rust",
            Some("Rust".to_string()),
            BeliefType::Preference,
            "s1",
            0.9,
        ));

        let view = mlg.query_unified(None);
        assert_eq!(view.entities.len(), 1);
        assert_eq!(view.entities[0].name, "Rust");
        assert!(view.entities[0].layers.contains(&GraphLayer::User));
    }

    #[test]
    fn test_query_unified_with_custom_layer() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("extra");
        let entity = LayeredEntity {
            name: "CustomItem".to_string(),
            entity_type: "Misc".to_string(),
            layer: GraphLayer::Custom("extra".to_string()),
            confidence: ConfidenceLevel::Verified,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };
        let _ = mlg.add_to_custom_layer("extra", entity);

        let view = mlg.query_unified(None);
        assert_eq!(view.entities.len(), 1);
        assert_eq!(view.entities[0].name, "CustomItem");
        assert!(view.entities[0].layers.contains(&GraphLayer::Custom("extra".to_string())));
    }

    #[test]
    fn test_query_unified_merges_same_entity() {
        let mut mlg = MultiLayerGraph::new();
        // Entity in session
        let session = mlg.get_or_create_session("s1");
        session.add_entity("Overlap", "Ship", "test");
        // Same entity as user belief
        mlg.user_graph.add_belief(UserBelief::new(
            "I own Overlap",
            Some("Overlap".to_string()),
            BeliefType::Ownership,
            "s1",
            0.9,
        ));

        let view = mlg.query_unified(Some("s1"));
        // Should be 1 unified entity with 2 layers
        assert_eq!(view.entities.len(), 1);
        assert_eq!(view.entities[0].name, "Overlap");
        assert!(view.entities[0].layers.contains(&GraphLayer::Session));
        assert!(view.entities[0].layers.contains(&GraphLayer::User));
    }

    #[test]
    fn test_unified_entity_merged_attributes() {
        let mut mlg = MultiLayerGraph::new();
        mlg.user_graph.add_belief(UserBelief::new(
            "Gladius is great",
            Some("Gladius".to_string()),
            BeliefType::Opinion,
            "s1",
            0.8,
        ));

        let view = mlg.query_unified(None);
        assert_eq!(view.entities.len(), 1);
        assert!(!view.entities[0].merged_attributes.is_empty());
    }

    #[test]
    fn test_query_unified_all_sessions() {
        let mut mlg = MultiLayerGraph::new();
        let s1 = mlg.get_or_create_session("s1");
        s1.add_entity("EntityA", "Type", "test");
        let s2 = mlg.get_or_create_session("s2");
        s2.add_entity("EntityB", "Type", "test");

        let view = mlg.query_unified(None);
        assert_eq!(view.entities.len(), 2);
    }

    #[test]
    fn test_unified_view_sorted_by_confidence() {
        let mut mlg = MultiLayerGraph::new();
        // Session entity (low confidence)
        let session = mlg.get_or_create_session("s1");
        session.add_entity("LowConf", "T", "test");
        // User belief entity (higher confidence)
        mlg.user_graph.add_belief(UserBelief::new(
            "HighConf is important",
            Some("HighConf".to_string()),
            BeliefType::Fact,
            "s1",
            1.0,
        ));

        let view = mlg.query_unified(Some("s1"));
        assert!(view.entities.len() >= 2);
        // First entity should have higher or equal confidence
        assert!(view.entities[0].confidence >= view.entities[1].confidence);
    }

    // =========================================================================
    // V4 Tests: Clustering (7.4)
    // =========================================================================

    #[test]
    fn test_cluster_empty_layer() {
        let mlg = MultiLayerGraph::new();
        let clusters = mlg.cluster_entities(&GraphLayer::Session, None, 1);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_entities_no_relations() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");

        // Without relations, each entity is its own component of size 1
        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 2);
        // min_cluster_size=2, and no edges connect them => no clusters of size >= 2
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_entities_basic() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_entity("C", "T", "test");
        session.add_relation("A", "relates", "B");
        session.add_relation("B", "relates", "C");

        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 2);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].entity_names.len(), 3);
        assert!(clusters[0].cohesion > 0.0);
        assert_eq!(clusters[0].layer, GraphLayer::Session);
    }

    #[test]
    fn test_cluster_entities_two_components() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_entity("C", "T", "test");
        session.add_entity("D", "T", "test");
        session.add_relation("A", "r", "B");
        session.add_relation("C", "r", "D");

        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 2);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_cluster_cohesion_full_graph() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_entity("C", "T", "test");
        // Fully connected: A-B, B-C, A-C
        session.add_relation("A", "r", "B");
        session.add_relation("B", "r", "C");
        session.add_relation("A", "r", "C");

        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 2);
        assert_eq!(clusters.len(), 1);
        // 3 edges / 3 possible edges = 1.0
        assert!((clusters[0].cohesion - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cluster_min_size_filter() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_relation("A", "r", "B");

        // min_cluster_size = 3 should exclude this 2-entity cluster
        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 3);
        assert!(clusters.is_empty());

        // min_cluster_size = 2 should include it
        let clusters = mlg.cluster_entities(&GraphLayer::Session, Some("s1"), 2);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn test_cluster_custom_layer() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("mydata");
        let make_entity = |name: &str| LayeredEntity {
            name: name.to_string(),
            entity_type: "Item".to_string(),
            layer: GraphLayer::Custom("mydata".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };
        let _ = mlg.add_to_custom_layer("mydata", make_entity("X"));
        let _ = mlg.add_to_custom_layer("mydata", make_entity("Y"));

        // Custom layers have no relations mechanism yet, so no clusters >= 2
        let clusters = mlg.cluster_entities(
            &GraphLayer::Custom("mydata".to_string()),
            None,
            1,
        );
        // Each entity is its own cluster of size 1
        assert_eq!(clusters.len(), 2);
    }

    // =========================================================================
    // V4 Tests: Cross-layer inference (7.5)
    // =========================================================================

    #[test]
    fn test_infer_cross_layer_empty() {
        let mlg = MultiLayerGraph::new();
        let inferred = mlg.infer_cross_layer(None);
        assert!(inferred.is_empty());
    }

    #[test]
    fn test_infer_cross_layer() {
        let mut mlg = MultiLayerGraph::new();
        // Entity in session
        let session = mlg.get_or_create_session("s1");
        session.add_entity("Gladius", "Ship", "test");
        // Same entity as user belief
        mlg.user_graph.add_belief(UserBelief::new(
            "Gladius is great",
            Some("Gladius".to_string()),
            BeliefType::Opinion,
            "s1",
            0.8,
        ));

        let inferred = mlg.infer_cross_layer(Some("s1"));
        assert!(!inferred.is_empty());
        assert_eq!(inferred[0].relation_type, "same_as");
        assert_eq!(inferred[0].source_entity, "Gladius");
        assert_eq!(inferred[0].target_entity, "Gladius");
        assert_eq!(inferred[0].confidence, 1.0); // exact match
    }

    #[test]
    fn test_infer_cross_layer_case_insensitive() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("gladius", "Ship", "test");
        mlg.user_graph.add_belief(UserBelief::new(
            "GLADIUS is great",
            Some("GLADIUS".to_string()),
            BeliefType::Opinion,
            "s1",
            0.8,
        ));

        let inferred = mlg.infer_cross_layer(Some("s1"));
        assert!(!inferred.is_empty());
        // Different case => confidence 0.9
        assert!((inferred[0].confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_inferred_relation_decay() {
        let ir = InferredRelation {
            source_entity: "A".to_string(),
            source_layer: GraphLayer::Session,
            target_entity: "A".to_string(),
            target_layer: GraphLayer::User,
            relation_type: "same_as".to_string(),
            confidence: 1.0,
            decay: 0.95,
        };
        assert!((ir.decay - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_infer_cross_layer_with_custom() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("extra");
        let entity = LayeredEntity {
            name: "SharedEntity".to_string(),
            entity_type: "Topic".to_string(),
            layer: GraphLayer::Custom("extra".to_string()),
            confidence: ConfidenceLevel::Verified,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        };
        let _ = mlg.add_to_custom_layer("extra", entity);

        let session = mlg.get_or_create_session("s1");
        session.add_entity("SharedEntity", "Topic", "test");

        let inferred = mlg.infer_cross_layer(Some("s1"));
        assert!(!inferred.is_empty());
        // Should find same_as between Session and Custom
        let has_cross = inferred.iter().any(|ir| {
            (ir.source_layer == GraphLayer::Session
                && ir.target_layer == GraphLayer::Custom("extra".to_string()))
                || (ir.source_layer == GraphLayer::Custom("extra".to_string())
                    && ir.target_layer == GraphLayer::Session)
        });
        assert!(has_cross);
    }

    // =========================================================================
    // V4 Tests: Conflict resolution (7.6)
    // =========================================================================

    #[test]
    fn test_resolve_conflict_not_found() {
        let mlg = MultiLayerGraph::new();
        let result = mlg.resolve_conflict("nonexistent", &GraphLayer::Session, &ConflictPolicy::LastWriteWins);
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_conflict_single_entity() {
        let mut mlg = MultiLayerGraph::new();
        let session = mlg.get_or_create_session("s1");
        session.add_entity("Solo", "Ship", "test");

        let result = mlg.resolve_conflict("Solo", &GraphLayer::Session, &ConflictPolicy::LastWriteWins);
        assert!(result.is_some());
        assert_eq!(result.expect("should exist").name, "Solo");
    }

    #[test]
    fn test_resolve_conflict_last_write_wins() {
        let mut mlg = MultiLayerGraph::new();
        // Add same entity in two different sessions (different timestamps)
        let s1 = mlg.get_or_create_session("s1");
        s1.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Inferred,
            source: "old".to_string(),
            timestamp: 1000,
            ttl_seconds: None,
        });
        let s2 = mlg.get_or_create_session("s2");
        s2.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Verified,
            source: "new".to_string(),
            timestamp: 2000,
            ttl_seconds: None,
        });

        let result = mlg.resolve_conflict("Ship", &GraphLayer::Session, &ConflictPolicy::LastWriteWins);
        assert!(result.is_some());
        let resolved = result.expect("should resolve");
        assert_eq!(resolved.source, "new");
        assert_eq!(resolved.timestamp, 2000);
    }

    #[test]
    fn test_resolve_conflict_highest_confidence() {
        let mut mlg = MultiLayerGraph::new();
        let s1 = mlg.get_or_create_session("s1");
        s1.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Inferred, // 0.4
            source: "low_conf".to_string(),
            timestamp: 2000,
            ttl_seconds: None,
        });
        let s2 = mlg.get_or_create_session("s2");
        s2.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Verified, // 1.0
            source: "high_conf".to_string(),
            timestamp: 1000,
            ttl_seconds: None,
        });

        let result = mlg.resolve_conflict("Ship", &GraphLayer::Session, &ConflictPolicy::HighestConfidence);
        assert!(result.is_some());
        assert_eq!(result.expect("resolved").source, "high_conf");
    }

    #[test]
    fn test_resolve_conflict_merge() {
        let mut mlg = MultiLayerGraph::new();
        let s1 = mlg.get_or_create_session("s1");
        s1.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Inferred,
            source: "source_a".to_string(),
            timestamp: 1000,
            ttl_seconds: None,
        });
        let s2 = mlg.get_or_create_session("s2");
        s2.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Verified,
            source: "source_b".to_string(),
            timestamp: 2000,
            ttl_seconds: None,
        });

        let result = mlg.resolve_conflict("Ship", &GraphLayer::Session, &ConflictPolicy::Merge);
        assert!(result.is_some());
        let merged = result.expect("merged");
        assert_eq!(merged.timestamp, 2000); // latest timestamp
        assert_eq!(merged.confidence, ConfidenceLevel::Verified); // highest confidence
        assert!(merged.source.contains("source_a"));
        assert!(merged.source.contains("source_b"));
    }

    #[test]
    fn test_resolve_conflict_manual() {
        let mut mlg = MultiLayerGraph::new();
        let s1 = mlg.get_or_create_session("s1");
        s1.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Inferred,
            source: "first".to_string(),
            timestamp: 1000,
            ttl_seconds: None,
        });
        let s2 = mlg.get_or_create_session("s2");
        s2.entities.push(LayeredEntity {
            name: "Ship".to_string(),
            entity_type: "Vehicle".to_string(),
            layer: GraphLayer::Session,
            confidence: ConfidenceLevel::Verified,
            source: "second".to_string(),
            timestamp: 2000,
            ttl_seconds: None,
        });

        let result = mlg.resolve_conflict("Ship", &GraphLayer::Session, &ConflictPolicy::Manual);
        assert!(result.is_some());
        // Manual returns the first candidate
        assert_eq!(result.expect("manual").source, "first");
    }

    #[test]
    fn test_resolve_conflict_custom_layer() {
        let mut mlg = MultiLayerGraph::new();
        mlg.add_custom_layer("data");
        let e1 = LayeredEntity {
            name: "Item".to_string(),
            entity_type: "Thing".to_string(),
            layer: GraphLayer::Custom("data".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "s1".to_string(),
            timestamp: 100,
            ttl_seconds: None,
        };
        let e2 = LayeredEntity {
            name: "Item".to_string(),
            entity_type: "Thing".to_string(),
            layer: GraphLayer::Custom("data".to_string()),
            confidence: ConfidenceLevel::Verified,
            source: "s2".to_string(),
            timestamp: 200,
            ttl_seconds: None,
        };
        // Directly push to test multiple entities with same name
        mlg.custom_layers.get_mut("data").expect("exists").push(e1);
        mlg.custom_layers.get_mut("data").expect("exists").push(e2);

        let result = mlg.resolve_conflict(
            "Item",
            &GraphLayer::Custom("data".to_string()),
            &ConflictPolicy::HighestConfidence,
        );
        assert!(result.is_some());
        assert_eq!(result.expect("resolved").source, "s2");
    }

    // =========================================================================
    // V4 Tests: Diff and merge (7.7)
    // =========================================================================

    #[test]
    fn test_graph_diff_empty() {
        let diff = GraphDiff::empty();
        assert!(diff.is_empty());
        assert!(diff.added_entities.is_empty());
        assert!(diff.removed_entities.is_empty());
        assert!(diff.modified_entities.is_empty());
        assert!(diff.added_relations.is_empty());
        assert!(diff.removed_relations.is_empty());
    }

    #[test]
    fn test_graph_diff_not_empty() {
        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "New".to_string()));
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_graph_diff_identical_graphs() {
        let mut g1 = MultiLayerGraph::new();
        let mut g2 = MultiLayerGraph::new();
        let s1 = g1.get_or_create_session("s1");
        s1.add_entity("A", "T", "test");
        let s2 = g2.get_or_create_session("s1");
        s2.add_entity("A", "T", "test");

        let diff = g1.diff(&g2, Some("s1"));
        assert!(diff.is_empty());
    }

    #[test]
    fn test_graph_diff_added_entities() {
        let g1 = MultiLayerGraph::new();
        let mut g2 = MultiLayerGraph::new();
        let session = g2.get_or_create_session("s1");
        session.add_entity("NewEntity", "Type", "test");

        let diff = g1.diff(&g2, Some("s1"));
        assert_eq!(diff.added_entities.len(), 1);
        assert_eq!(diff.added_entities[0].1, "NewEntity");
        assert_eq!(diff.added_entities[0].0, GraphLayer::Session);
    }

    #[test]
    fn test_graph_diff_removed_entities() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("OldEntity", "Type", "test");
        let g2 = MultiLayerGraph::new();

        let diff = g1.diff(&g2, Some("s1"));
        assert_eq!(diff.removed_entities.len(), 1);
        assert_eq!(diff.removed_entities[0].1, "OldEntity");
    }

    #[test]
    fn test_graph_diff_added_relations() {
        let g1 = MultiLayerGraph::new();
        let mut g2 = MultiLayerGraph::new();
        let session = g2.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_relation("A", "links", "B");

        let diff = g1.diff(&g2, Some("s1"));
        assert_eq!(diff.added_relations.len(), 1);
        assert_eq!(diff.added_relations[0], ("A".to_string(), "links".to_string(), "B".to_string()));
    }

    #[test]
    fn test_graph_diff_removed_relations() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("X", "T", "test");
        session.add_entity("Y", "T", "test");
        session.add_relation("X", "connects", "Y");
        let g2 = MultiLayerGraph::new();

        let diff = g1.diff(&g2, Some("s1"));
        assert_eq!(diff.removed_relations.len(), 1);
    }

    #[test]
    fn test_graph_diff_custom_layer() {
        let mut g1 = MultiLayerGraph::new();
        g1.add_custom_layer("custom");
        let _ = g1.add_to_custom_layer("custom", LayeredEntity {
            name: "Existing".to_string(),
            entity_type: "T".to_string(),
            layer: GraphLayer::Custom("custom".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        });

        let mut g2 = MultiLayerGraph::new();
        g2.add_custom_layer("custom");
        let _ = g2.add_to_custom_layer("custom", LayeredEntity {
            name: "Existing".to_string(),
            entity_type: "T".to_string(),
            layer: GraphLayer::Custom("custom".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        });
        let _ = g2.add_to_custom_layer("custom", LayeredEntity {
            name: "NewCustom".to_string(),
            entity_type: "T".to_string(),
            layer: GraphLayer::Custom("custom".to_string()),
            confidence: ConfidenceLevel::Inferred,
            source: "test".to_string(),
            timestamp: 0,
            ttl_seconds: None,
        });

        let diff = g1.diff(&g2, None);
        assert_eq!(diff.added_entities.len(), 1);
        assert_eq!(diff.added_entities[0].1, "NewCustom");
    }

    #[test]
    fn test_apply_diff_union() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("Keep", "T", "test");
        session.add_entity("Remove", "T", "test");

        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "Added".to_string()));
        diff.removed_entities.push((GraphLayer::Session, "Remove".to_string()));

        g1.apply_diff(&diff, "s1", &MergeStrategy::Union);

        let session = g1.session_graphs.get("s1").expect("session exists");
        let names: Vec<&str> = session.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Keep"));
        assert!(names.contains(&"Added"));
        assert!(!names.contains(&"Remove"));
    }

    #[test]
    fn test_apply_diff_ours() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("Keep", "T", "test");
        session.add_entity("AlsoKeep", "T", "test");

        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "New".to_string()));
        diff.removed_entities.push((GraphLayer::Session, "AlsoKeep".to_string()));

        // Ours strategy: add but don't remove
        g1.apply_diff(&diff, "s1", &MergeStrategy::Ours);

        let session = g1.session_graphs.get("s1").expect("session exists");
        let names: Vec<&str> = session.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Keep"));
        assert!(names.contains(&"AlsoKeep")); // NOT removed
        assert!(names.contains(&"New")); // added
    }

    #[test]
    fn test_apply_diff_theirs() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("Old", "T", "test");

        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "Their".to_string()));
        diff.removed_entities.push((GraphLayer::Session, "Old".to_string()));

        g1.apply_diff(&diff, "s1", &MergeStrategy::Theirs);

        let session = g1.session_graphs.get("s1").expect("session exists");
        let names: Vec<&str> = session.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Their"));
        assert!(!names.contains(&"Old"));
    }

    #[test]
    fn test_apply_diff_with_relations() {
        let mut g1 = MultiLayerGraph::new();
        g1.get_or_create_session("s1");

        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "X".to_string()));
        diff.added_entities.push((GraphLayer::Session, "Y".to_string()));
        diff.added_relations.push(("X".to_string(), "connects".to_string(), "Y".to_string()));

        g1.apply_diff(&diff, "s1", &MergeStrategy::Union);

        let session = g1.session_graphs.get("s1").expect("exists");
        assert_eq!(session.entities.len(), 2);
        assert_eq!(session.relations.len(), 1);
    }

    #[test]
    fn test_apply_diff_custom_layer() {
        let mut g1 = MultiLayerGraph::new();
        g1.add_custom_layer("extra");

        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Custom("extra".to_string()), "CustomNew".to_string()));

        g1.apply_diff(&diff, "s1", &MergeStrategy::Union);

        let entities = g1.get_custom_layer("extra").expect("layer exists");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "CustomNew");
    }

    #[test]
    fn test_apply_diff_remove_relation() {
        let mut g1 = MultiLayerGraph::new();
        let session = g1.get_or_create_session("s1");
        session.add_entity("A", "T", "test");
        session.add_entity("B", "T", "test");
        session.add_relation("A", "rel", "B");

        let mut diff = GraphDiff::empty();
        diff.removed_relations.push(("A".to_string(), "rel".to_string(), "B".to_string()));

        g1.apply_diff(&diff, "s1", &MergeStrategy::Union);

        let session = g1.session_graphs.get("s1").expect("exists");
        assert!(session.relations.is_empty());
    }

    // =========================================================================
    // V4 Tests: MergeStrategy
    // =========================================================================

    #[test]
    fn test_merge_strategy_variants() {
        assert_eq!(MergeStrategy::Ours, MergeStrategy::Ours);
        assert_ne!(MergeStrategy::Ours, MergeStrategy::Theirs);
        assert_ne!(MergeStrategy::Theirs, MergeStrategy::Union);
    }

    #[test]
    fn test_merge_strategy_clone() {
        let strategy = MergeStrategy::Union;
        let cloned = strategy.clone();
        assert_eq!(strategy, cloned);
    }

    // =========================================================================
    // V4 Tests: GraphDiff serialization
    // =========================================================================

    #[test]
    fn test_graph_diff_serialize() {
        let mut diff = GraphDiff::empty();
        diff.added_entities.push((GraphLayer::Session, "Entity".to_string()));
        diff.modified_entities.push((GraphLayer::User, "Modified".to_string(), "changed type".to_string()));

        let json = serde_json::to_string(&diff).expect("serialize");
        let deserialized: GraphDiff = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.added_entities.len(), 1);
        assert_eq!(deserialized.modified_entities.len(), 1);
    }

    // =========================================================================
    // V4 Tests: InferredRelation
    // =========================================================================

    #[test]
    fn test_inferred_relation_serialize() {
        let ir = InferredRelation {
            source_entity: "A".to_string(),
            source_layer: GraphLayer::Session,
            target_entity: "B".to_string(),
            target_layer: GraphLayer::Custom("x".to_string()),
            relation_type: "same_as".to_string(),
            confidence: 0.85,
            decay: 0.99,
        };
        let json = serde_json::to_string(&ir).expect("serialize");
        let deserialized: InferredRelation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.source_entity, "A");
        assert_eq!(deserialized.target_layer, GraphLayer::Custom("x".to_string()));
    }

    // =========================================================================
    // V4 Tests: GraphCluster
    // =========================================================================

    #[test]
    fn test_graph_cluster_serialize() {
        let cluster = GraphCluster {
            id: "c0".to_string(),
            label: "test cluster".to_string(),
            entity_names: vec!["A".to_string(), "B".to_string()],
            layer: GraphLayer::Session,
            cohesion: 0.75,
        };
        let json = serde_json::to_string(&cluster).expect("serialize");
        let deserialized: GraphCluster = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.id, "c0");
        assert_eq!(deserialized.entity_names.len(), 2);
    }

    // =========================================================================
    // V4 Tests: UnifiedView
    // =========================================================================

    #[test]
    fn test_unified_view_serialize() {
        let view = UnifiedView {
            entities: vec![UnifiedEntity {
                name: "Test".to_string(),
                entity_type: "Type".to_string(),
                layers: vec![GraphLayer::Session, GraphLayer::User],
                merged_attributes: HashMap::new(),
                confidence: 0.8,
            }],
            relations: vec![UnifiedRelation {
                source: "A".to_string(),
                target: "B".to_string(),
                relation_type: "related".to_string(),
                layer: GraphLayer::Internet,
                confidence: 0.5,
            }],
        };
        let json = serde_json::to_string(&view).expect("serialize");
        let deserialized: UnifiedView = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.entities.len(), 1);
        assert_eq!(deserialized.relations.len(), 1);
    }

    #[test]
    fn test_unified_entity_multi_layer() {
        let entity = UnifiedEntity {
            name: "Multi".to_string(),
            entity_type: "Thing".to_string(),
            layers: vec![
                GraphLayer::Session,
                GraphLayer::User,
                GraphLayer::Custom("extra".to_string()),
            ],
            merged_attributes: {
                let mut m = HashMap::new();
                m.insert("key".to_string(), serde_json::Value::String("val".to_string()));
                m
            },
            confidence: 0.95,
        };
        assert_eq!(entity.layers.len(), 3);
        assert!(entity.merged_attributes.contains_key("key"));
    }
}
