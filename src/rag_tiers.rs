//! RAG Tier System - Configurable retrieval augmented generation with cost/quality tradeoffs
//!
//! This module provides a comprehensive tier system for RAG (Retrieval-Augmented Generation)
//! that allows applications to choose between different levels of sophistication,
//! from simple keyword search to full agentic retrieval.
//!
//! # Tiers Overview
//!
//! | Tier | Extra LLM Calls | Description |
//! |------|-----------------|-------------|
//! | Disabled | 0 | No RAG, only conversation context |
//! | Fast | 0 | FTS5 keyword search only |
//! | Semantic | 0 | FTS5 + embeddings + hybrid search |
//! | Enhanced | 1-2 | + Query expansion + reranking |
//! | Thorough | 3-5 | + Multi-query + compression + self-reflection |
//! | Agentic | N | Iterative agent that searches until satisfied |
//! | Graph | N+ | + Knowledge graph traversal |
//! | Full | N+ | All features enabled |
//!
//! # Usage
//!
//! ```rust
//! use ai_assistant::rag_tiers::{RagConfig, RagTier, RagFeatures};
//!
//! // Simple: use a predefined tier
//! let config = RagConfig::with_tier(RagTier::Enhanced);
//!
//! // Advanced: customize specific features
//! let mut config = RagConfig::with_tier(RagTier::Enhanced);
//! config.features.contextual_compression = true;  // Add from higher tier
//! config.features.reranking = false;  // Disable from current tier
//! config.use_custom_features = true;  // Use custom instead of tier defaults
//!
//! // Check requirements
//! let reqs = config.check_requirements();
//! for req in reqs {
//!     println!("Required: {:?}", req);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// RAG Features - Fine-grained control over individual capabilities
// ============================================================================

/// Individual RAG features that can be enabled/disabled
///
/// Each feature has associated costs (LLM calls, compute time) and requirements
/// (embedding models, databases, etc.)
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RagFeatures {
    // === Retrieval Methods ===
    /// FTS5 full-text search (keyword matching via BM25)
    /// Cost: None | Requires: SQLite FTS5
    pub fts_search: bool,

    /// Semantic search using embedding vectors
    /// Cost: Embedding computation | Requires: Embedding model
    pub semantic_search: bool,

    /// Hybrid search: combine FTS + semantic with score fusion
    /// Cost: Both FTS and semantic | Requires: Both enabled
    pub hybrid_search: bool,

    // === Query Enhancement ===
    /// Expand query with synonyms and related terms (no LLM)
    /// Cost: None | Requires: Synonym dictionary
    pub synonym_expansion: bool,

    /// LLM-based query expansion (generate query variants)
    /// Cost: 1 LLM call | Requires: None
    pub query_expansion: bool,

    /// Decompose complex queries into multiple sub-queries
    /// Cost: 1 LLM call | Requires: None
    pub multi_query: bool,

    /// Generate hypothetical document to search (HyDE)
    /// Cost: 1 LLM call + embedding | Requires: Embedding model
    pub hyde: bool,

    // === Result Processing ===
    /// Rerank results using LLM scoring
    /// Cost: 1 LLM call | Requires: None
    pub reranking: bool,

    /// Use cross-encoder model for more accurate reranking
    /// Cost: Inference | Requires: Cross-encoder model
    pub cross_encoder_rerank: bool,

    /// Combine multiple result sets with Reciprocal Rank Fusion
    /// Cost: None | Requires: Multiple retrieval methods
    pub fusion_rrf: bool,

    /// Extract only relevant parts from retrieved chunks
    /// Cost: 1 LLM call per chunk | Requires: None
    pub contextual_compression: bool,

    /// Include surrounding context for matched sentences
    /// Cost: None | Requires: Stored sentence boundaries
    pub sentence_window: bool,

    /// Return parent document when child chunk matches
    /// Cost: None | Requires: Parent-child relationships stored
    pub parent_document: bool,

    // === Self-Improvement ===
    /// Self-evaluate if more context is needed
    /// Cost: 1-2 LLM calls | Requires: None
    pub self_reflection: bool,

    /// Evaluate retrieval quality and retry if poor (CRAG)
    /// Cost: 1-2 LLM calls | Requires: None
    pub corrective_rag: bool,

    /// Dynamically choose retrieval strategy based on query
    /// Cost: 1 LLM call | Requires: None
    pub adaptive_strategy: bool,

    // === Advanced (Tier 5+) ===
    /// Iterative agent that searches until satisfied
    /// Cost: N LLM calls (unbounded) | Requires: None
    pub agentic_mode: bool,

    /// Build and query knowledge graph
    /// Cost: Entity extraction + graph ops | Requires: Graph database
    pub graph_rag: bool,

    /// Hierarchical corpus summarization (RAPTOR)
    /// Cost: Pre-processing | Requires: Pre-built summary tree
    pub raptor: bool,

    /// Multi-modal retrieval (images, tables)
    /// Cost: Vision model | Requires: Vision-capable model
    pub multimodal: bool,
}

impl RagFeatures {
    /// Create with all features disabled
    pub fn none() -> Self {
        Self::default()
    }

    /// Create with all features enabled
    pub fn all() -> Self {
        Self {
            fts_search: true,
            semantic_search: true,
            hybrid_search: true,
            synonym_expansion: true,
            query_expansion: true,
            multi_query: true,
            hyde: true,
            reranking: true,
            cross_encoder_rerank: true,
            fusion_rrf: true,
            contextual_compression: true,
            sentence_window: true,
            parent_document: true,
            self_reflection: true,
            corrective_rag: true,
            adaptive_strategy: true,
            agentic_mode: true,
            graph_rag: true,
            raptor: true,
            multimodal: true,
        }
    }

    /// Count enabled features
    pub fn enabled_count(&self) -> usize {
        let mut count = 0;
        if self.fts_search { count += 1; }
        if self.semantic_search { count += 1; }
        if self.hybrid_search { count += 1; }
        if self.synonym_expansion { count += 1; }
        if self.query_expansion { count += 1; }
        if self.multi_query { count += 1; }
        if self.hyde { count += 1; }
        if self.reranking { count += 1; }
        if self.cross_encoder_rerank { count += 1; }
        if self.fusion_rrf { count += 1; }
        if self.contextual_compression { count += 1; }
        if self.sentence_window { count += 1; }
        if self.parent_document { count += 1; }
        if self.self_reflection { count += 1; }
        if self.corrective_rag { count += 1; }
        if self.adaptive_strategy { count += 1; }
        if self.agentic_mode { count += 1; }
        if self.graph_rag { count += 1; }
        if self.raptor { count += 1; }
        if self.multimodal { count += 1; }
        count
    }

    /// Get list of enabled feature names
    pub fn enabled_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();
        if self.fts_search { features.push("fts_search"); }
        if self.semantic_search { features.push("semantic_search"); }
        if self.hybrid_search { features.push("hybrid_search"); }
        if self.synonym_expansion { features.push("synonym_expansion"); }
        if self.query_expansion { features.push("query_expansion"); }
        if self.multi_query { features.push("multi_query"); }
        if self.hyde { features.push("hyde"); }
        if self.reranking { features.push("reranking"); }
        if self.cross_encoder_rerank { features.push("cross_encoder_rerank"); }
        if self.fusion_rrf { features.push("fusion_rrf"); }
        if self.contextual_compression { features.push("contextual_compression"); }
        if self.sentence_window { features.push("sentence_window"); }
        if self.parent_document { features.push("parent_document"); }
        if self.self_reflection { features.push("self_reflection"); }
        if self.corrective_rag { features.push("corrective_rag"); }
        if self.adaptive_strategy { features.push("adaptive_strategy"); }
        if self.agentic_mode { features.push("agentic_mode"); }
        if self.graph_rag { features.push("graph_rag"); }
        if self.raptor { features.push("raptor"); }
        if self.multimodal { features.push("multimodal"); }
        features
    }
}

// ============================================================================
// RAG Tiers - Predefined feature sets for common use cases
// ============================================================================

/// Predefined RAG tiers with increasing sophistication and cost
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RagTier {
    /// No RAG - only uses conversation context
    Disabled,

    /// Fast: FTS5 keyword search only (0 extra LLM calls)
    /// Best for: Low latency, simple queries
    Fast,

    /// Semantic: FTS5 + embeddings + hybrid search (0 extra LLM calls)
    /// Best for: Better recall without latency cost
    /// Requires: Embedding model
    Semantic,

    /// Enhanced: + Query expansion + reranking + sentence window (1-2 LLM calls)
    /// Best for: Balanced quality/cost for most applications
    Enhanced,

    /// Thorough: + Multi-query + compression + self-reflection (3-5 LLM calls)
    /// Best for: High-stakes queries where accuracy matters
    Thorough,

    /// Agentic: Iterative agent that searches until satisfied (N LLM calls)
    /// Best for: Complex queries requiring multiple retrieval rounds
    /// Warning: Can use many LLM calls
    Agentic,

    /// Graph: + Knowledge graph traversal (N+ LLM calls)
    /// Best for: Queries requiring relationship understanding
    /// Requires: Graph database setup
    Graph,

    /// Full: All features enabled
    /// Best for: Maximum capability regardless of cost
    /// Warning: Highest latency and cost
    Full,

    /// Custom: User-defined feature selection
    /// Use RagConfig.features to specify exactly which features to enable
    Custom,
}

impl Default for RagTier {
    fn default() -> Self {
        RagTier::Fast
    }
}

impl RagTier {
    /// Get the display name for this tier
    pub fn display_name(&self) -> &'static str {
        match self {
            RagTier::Disabled => "Disabled",
            RagTier::Fast => "Fast",
            RagTier::Semantic => "Semantic",
            RagTier::Enhanced => "Enhanced",
            RagTier::Thorough => "Thorough",
            RagTier::Agentic => "Agentic",
            RagTier::Graph => "Graph",
            RagTier::Full => "Full",
            RagTier::Custom => "Custom",
        }
    }

    /// Get a description of this tier
    pub fn description(&self) -> &'static str {
        match self {
            RagTier::Disabled => "No retrieval, only conversation context",
            RagTier::Fast => "Keyword search only, minimal latency",
            RagTier::Semantic => "Keyword + semantic search, better recall",
            RagTier::Enhanced => "Adds query expansion and reranking",
            RagTier::Thorough => "Multi-query with self-reflection",
            RagTier::Agentic => "Iterative agent-based retrieval",
            RagTier::Graph => "Knowledge graph traversal",
            RagTier::Full => "All features enabled",
            RagTier::Custom => "User-defined feature selection",
        }
    }

    /// Convert tier to default feature flags
    pub fn to_features(&self) -> RagFeatures {
        match self {
            RagTier::Disabled => RagFeatures::none(),

            RagTier::Fast => RagFeatures {
                fts_search: true,
                ..Default::default()
            },

            RagTier::Semantic => RagFeatures {
                fts_search: true,
                semantic_search: true,
                hybrid_search: true,
                fusion_rrf: true,
                ..Default::default()
            },

            RagTier::Enhanced => RagFeatures {
                fts_search: true,
                semantic_search: true,
                hybrid_search: true,
                synonym_expansion: true,
                query_expansion: true,
                reranking: true,
                sentence_window: true,
                fusion_rrf: true,
                ..Default::default()
            },

            RagTier::Thorough => RagFeatures {
                fts_search: true,
                semantic_search: true,
                hybrid_search: true,
                synonym_expansion: true,
                multi_query: true,
                reranking: true,
                contextual_compression: true,
                sentence_window: true,
                fusion_rrf: true,
                self_reflection: true,
                corrective_rag: true,
                ..Default::default()
            },

            RagTier::Agentic => RagFeatures {
                fts_search: true,
                semantic_search: true,
                hybrid_search: true,
                synonym_expansion: true,
                multi_query: true,
                reranking: true,
                contextual_compression: true,
                sentence_window: true,
                fusion_rrf: true,
                self_reflection: true,
                corrective_rag: true,
                adaptive_strategy: true,
                agentic_mode: true,
                ..Default::default()
            },

            RagTier::Graph => RagFeatures {
                fts_search: true,
                semantic_search: true,
                hybrid_search: true,
                synonym_expansion: true,
                multi_query: true,
                reranking: true,
                contextual_compression: true,
                sentence_window: true,
                parent_document: true,
                fusion_rrf: true,
                self_reflection: true,
                corrective_rag: true,
                adaptive_strategy: true,
                agentic_mode: true,
                graph_rag: true,
                ..Default::default()
            },

            RagTier::Full => RagFeatures::all(),

            RagTier::Custom => RagFeatures::none(), // User fills in
        }
    }

    /// Estimated range of extra LLM calls for this tier
    /// Returns (minimum, maximum) where None means unbounded
    pub fn estimated_extra_calls(&self) -> (usize, Option<usize>) {
        match self {
            RagTier::Disabled => (0, Some(0)),
            RagTier::Fast => (0, Some(0)),
            RagTier::Semantic => (0, Some(0)),
            RagTier::Enhanced => (1, Some(2)),
            RagTier::Thorough => (3, Some(5)),
            RagTier::Agentic => (2, None), // Unbounded
            RagTier::Graph => (3, None),   // Unbounded
            RagTier::Full => (5, None),    // Unbounded
            RagTier::Custom => (0, None),  // Depends on features
        }
    }

    /// Whether this tier requires an embedding model
    pub fn requires_embeddings(&self) -> bool {
        matches!(
            self,
            RagTier::Semantic
                | RagTier::Enhanced
                | RagTier::Thorough
                | RagTier::Agentic
                | RagTier::Graph
                | RagTier::Full
        )
    }

    /// All tiers in order of increasing sophistication
    pub fn all_tiers() -> &'static [RagTier] {
        &[
            RagTier::Disabled,
            RagTier::Fast,
            RagTier::Semantic,
            RagTier::Enhanced,
            RagTier::Thorough,
            RagTier::Agentic,
            RagTier::Graph,
            RagTier::Full,
            RagTier::Custom,
        ]
    }

    /// Standard tiers excluding Custom (useful for UI selectors)
    ///
    /// Returns tiers that have predefined feature sets, ordered by complexity.
    pub fn standard_tiers() -> &'static [RagTier] {
        &[
            RagTier::Disabled,
            RagTier::Fast,
            RagTier::Semantic,
            RagTier::Enhanced,
            RagTier::Thorough,
            RagTier::Agentic,
            RagTier::Graph,
            RagTier::Full,
        ]
    }

    /// Convert to snake_case string representation
    ///
    /// Useful for serialization, config files, API responses, and logging.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert_eq!(RagTier::Enhanced.as_str(), "enhanced");
    /// assert_eq!(RagTier::Graph.as_str(), "graph");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            RagTier::Disabled => "disabled",
            RagTier::Fast => "fast",
            RagTier::Semantic => "semantic",
            RagTier::Enhanced => "enhanced",
            RagTier::Thorough => "thorough",
            RagTier::Agentic => "agentic",
            RagTier::Graph => "graph",
            RagTier::Full => "full",
            RagTier::Custom => "custom",
        }
    }

    /// Parse from string (case-insensitive)
    ///
    /// Returns `None` if the string doesn't match any tier.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert_eq!(RagTier::from_str("enhanced"), Some(RagTier::Enhanced));
    /// assert_eq!(RagTier::from_str("SEMANTIC"), Some(RagTier::Semantic));
    /// assert_eq!(RagTier::from_str("invalid"), None);
    /// ```
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "disabled" | "none" | "off" => Some(RagTier::Disabled),
            "fast" | "basic" | "keyword" => Some(RagTier::Fast),
            "semantic" | "vector" | "embedding" => Some(RagTier::Semantic),
            "enhanced" | "standard" => Some(RagTier::Enhanced),
            "thorough" | "deep" => Some(RagTier::Thorough),
            "agentic" | "agent" | "iterative" => Some(RagTier::Agentic),
            "graph" | "knowledge_graph" | "kg" => Some(RagTier::Graph),
            "full" | "max" | "maximum" | "all" => Some(RagTier::Full),
            "custom" => Some(RagTier::Custom),
            _ => None,
        }
    }

    /// Get the numeric complexity level (0-8)
    ///
    /// Higher values indicate more sophisticated (and costly) retrieval.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert_eq!(RagTier::Disabled.complexity_level(), 0);
    /// assert_eq!(RagTier::Fast.complexity_level(), 1);
    /// assert_eq!(RagTier::Full.complexity_level(), 7);
    /// ```
    pub fn complexity_level(&self) -> u8 {
        match self {
            RagTier::Disabled => 0,
            RagTier::Fast => 1,
            RagTier::Semantic => 2,
            RagTier::Enhanced => 3,
            RagTier::Thorough => 4,
            RagTier::Agentic => 5,
            RagTier::Graph => 6,
            RagTier::Full => 7,
            RagTier::Custom => 8, // Custom is considered highest as it can include anything
        }
    }

    /// Check if RAG is enabled at all for this tier
    ///
    /// ```
    /// use ai_assistant::RagTier;
    /// assert!(!RagTier::Disabled.is_enabled());
    /// assert!(RagTier::Fast.is_enabled());
    /// ```
    pub fn is_enabled(&self) -> bool {
        !matches!(self, RagTier::Disabled)
    }

    /// Check if this tier uses LLM calls for retrieval enhancement
    ///
    /// Tiers from Enhanced and above use LLM calls for query expansion,
    /// reranking, self-reflection, etc.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert!(!RagTier::Fast.uses_llm_for_retrieval());
    /// assert!(!RagTier::Semantic.uses_llm_for_retrieval());
    /// assert!(RagTier::Enhanced.uses_llm_for_retrieval());
    /// ```
    pub fn uses_llm_for_retrieval(&self) -> bool {
        matches!(
            self,
            RagTier::Enhanced
                | RagTier::Thorough
                | RagTier::Agentic
                | RagTier::Graph
                | RagTier::Full
        )
    }

    /// Check if this tier has unbounded LLM call potential
    ///
    /// Agentic, Graph, and Full tiers can make unlimited LLM calls
    /// as they iterate until satisfied.
    pub fn is_unbounded(&self) -> bool {
        matches!(self, RagTier::Agentic | RagTier::Graph | RagTier::Full)
    }

    /// Get the previous (simpler) tier, if any
    ///
    /// Returns None for Disabled and Custom tiers.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert_eq!(RagTier::Enhanced.previous(), Some(RagTier::Semantic));
    /// assert_eq!(RagTier::Fast.previous(), Some(RagTier::Disabled));
    /// assert_eq!(RagTier::Disabled.previous(), None);
    /// ```
    pub fn previous(&self) -> Option<Self> {
        match self {
            RagTier::Disabled => None,
            RagTier::Fast => Some(RagTier::Disabled),
            RagTier::Semantic => Some(RagTier::Fast),
            RagTier::Enhanced => Some(RagTier::Semantic),
            RagTier::Thorough => Some(RagTier::Enhanced),
            RagTier::Agentic => Some(RagTier::Thorough),
            RagTier::Graph => Some(RagTier::Agentic),
            RagTier::Full => Some(RagTier::Graph),
            RagTier::Custom => None, // Custom has no ordering
        }
    }

    /// Get the next (more sophisticated) tier, if any
    ///
    /// Returns None for Full and Custom tiers.
    /// ```
    /// use ai_assistant::RagTier;
    /// assert_eq!(RagTier::Enhanced.next(), Some(RagTier::Thorough));
    /// assert_eq!(RagTier::Full.next(), None);
    /// ```
    pub fn next(&self) -> Option<Self> {
        match self {
            RagTier::Disabled => Some(RagTier::Fast),
            RagTier::Fast => Some(RagTier::Semantic),
            RagTier::Semantic => Some(RagTier::Enhanced),
            RagTier::Enhanced => Some(RagTier::Thorough),
            RagTier::Thorough => Some(RagTier::Agentic),
            RagTier::Agentic => Some(RagTier::Graph),
            RagTier::Graph => Some(RagTier::Full),
            RagTier::Full => None,
            RagTier::Custom => None, // Custom has no ordering
        }
    }

    /// Upgrade to the next tier if possible
    ///
    /// Returns the current tier if already at maximum or Custom.
    pub fn upgrade(&self) -> Self {
        self.next().unwrap_or(*self)
    }

    /// Downgrade to the previous tier if possible
    ///
    /// Returns the current tier if already at minimum or Custom.
    pub fn downgrade(&self) -> Self {
        self.previous().unwrap_or(*self)
    }

    /// Get emoji indicator for the tier (useful for UI)
    pub fn emoji(&self) -> &'static str {
        match self {
            RagTier::Disabled => "⭕",
            RagTier::Fast => "⚡",
            RagTier::Semantic => "🔍",
            RagTier::Enhanced => "✨",
            RagTier::Thorough => "🔬",
            RagTier::Agentic => "🤖",
            RagTier::Graph => "🕸️",
            RagTier::Full => "🌟",
            RagTier::Custom => "⚙️",
        }
    }

    /// Get a short label suitable for compact UI (e.g., buttons)
    pub fn short_label(&self) -> &'static str {
        match self {
            RagTier::Disabled => "Off",
            RagTier::Fast => "Fast",
            RagTier::Semantic => "Sem",
            RagTier::Enhanced => "Enh",
            RagTier::Thorough => "Thor",
            RagTier::Agentic => "Agent",
            RagTier::Graph => "Graph",
            RagTier::Full => "Full",
            RagTier::Custom => "Cust",
        }
    }

    /// Check if this tier is at least as sophisticated as another
    ///
    /// ```
    /// use ai_assistant::RagTier;
    /// assert!(RagTier::Enhanced.is_at_least(RagTier::Semantic));
    /// assert!(RagTier::Enhanced.is_at_least(RagTier::Enhanced));
    /// assert!(!RagTier::Fast.is_at_least(RagTier::Semantic));
    /// ```
    pub fn is_at_least(&self, other: RagTier) -> bool {
        // Custom is special - it's only >= itself
        if *self == RagTier::Custom || other == RagTier::Custom {
            return *self == other;
        }
        self.complexity_level() >= other.complexity_level()
    }

    /// Create a tier from complexity level
    ///
    /// Clamps to valid range (0-7 for standard tiers).
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => RagTier::Disabled,
            1 => RagTier::Fast,
            2 => RagTier::Semantic,
            3 => RagTier::Enhanced,
            4 => RagTier::Thorough,
            5 => RagTier::Agentic,
            6 => RagTier::Graph,
            _ => RagTier::Full, // 7+
        }
    }

    /// Get recommended tier for a given latency budget (in milliseconds)
    ///
    /// Suggests the most sophisticated tier that should fit within the budget.
    /// This is a heuristic based on typical LLM response times.
    pub fn for_latency_budget(budget_ms: u64) -> Self {
        match budget_ms {
            0..=500 => RagTier::Fast,           // Sub-second: keyword only
            501..=1500 => RagTier::Semantic,    // 1-1.5s: add embeddings
            1501..=3000 => RagTier::Enhanced,   // 1.5-3s: add 1-2 LLM calls
            3001..=6000 => RagTier::Thorough,   // 3-6s: add more LLM calls
            _ => RagTier::Agentic,              // 6s+: allow agentic
        }
    }

    /// Get recommended tier for a given cost budget (in API call equivalents)
    ///
    /// Suggests the most sophisticated tier that fits within N extra LLM calls.
    pub fn for_call_budget(max_calls: usize) -> Self {
        match max_calls {
            0 => RagTier::Semantic,     // 0 extra calls: keyword + embeddings
            1..=2 => RagTier::Enhanced, // 1-2 calls: add expansion + rerank
            3..=5 => RagTier::Thorough, // 3-5 calls: full enhancement
            _ => RagTier::Agentic,      // 5+: allow iteration
        }
    }
}

// ============================================================================
// RAG Requirements - What each feature needs to function
// ============================================================================

/// Requirements that must be met for certain RAG features
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RagRequirement {
    /// Needs an embedding model (local or API)
    EmbeddingModel,

    /// Needs a cross-encoder model for accurate reranking
    CrossEncoderModel,

    /// Needs a graph database (e.g., Neo4j, petgraph)
    GraphDatabase,

    /// Needs pre-processed corpus with summary hierarchy
    PreprocessedCorpus,

    /// Needs a vision-capable model
    VisionModel,

    /// Needs synonym dictionary
    SynonymDictionary,

    /// Needs parent-child document relationships stored
    DocumentHierarchy,

    /// Needs sentence boundary storage
    SentenceBoundaries,
}

impl RagRequirement {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            RagRequirement::EmbeddingModel => "Embedding Model",
            RagRequirement::CrossEncoderModel => "Cross-Encoder Model",
            RagRequirement::GraphDatabase => "Graph Database",
            RagRequirement::PreprocessedCorpus => "Pre-processed Corpus",
            RagRequirement::VisionModel => "Vision Model",
            RagRequirement::SynonymDictionary => "Synonym Dictionary",
            RagRequirement::DocumentHierarchy => "Document Hierarchy",
            RagRequirement::SentenceBoundaries => "Sentence Boundaries",
        }
    }

    /// Get description of what's needed
    pub fn description(&self) -> &'static str {
        match self {
            RagRequirement::EmbeddingModel => "A model that converts text to vectors (e.g., text-embedding-ada-002, nomic-embed)",
            RagRequirement::CrossEncoderModel => "A model that scores query-document pairs (e.g., ms-marco-MiniLM)",
            RagRequirement::GraphDatabase => "A graph database like Neo4j or in-memory petgraph",
            RagRequirement::PreprocessedCorpus => "Run RAPTOR preprocessing to build summary hierarchy",
            RagRequirement::VisionModel => "A vision-capable model like LLaVA or GPT-4V",
            RagRequirement::SynonymDictionary => "Built-in or custom synonym mappings",
            RagRequirement::DocumentHierarchy => "Store parent-child relationships during indexing",
            RagRequirement::SentenceBoundaries => "Store sentence offsets during chunking",
        }
    }
}

// ============================================================================
// RAG Configuration - Main configuration struct
// ============================================================================

/// Main RAG configuration combining tier selection with custom features
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagConfig {
    /// Selected tier (use Custom for manual feature selection)
    pub tier: RagTier,

    /// Feature flags (when tier = Custom, or to override tier defaults)
    pub features: RagFeatures,

    /// Whether to use custom features (true) or tier defaults (false)
    pub use_custom_features: bool,

    // === Limits ===
    /// Maximum extra LLM calls allowed per query (hard cap)
    pub max_extra_llm_calls: usize,

    /// Maximum chunks to retrieve
    pub max_chunks: usize,

    /// Maximum tokens for knowledge context
    pub max_knowledge_tokens: usize,

    /// Use dynamic context sizing (fill up to model's limit)
    pub dynamic_context: bool,

    /// Minimum relevance score for chunks (0.0 to 1.0)
    pub min_relevance_score: f32,

    // === Model Config ===
    /// Embedding model name (if different from chat model)
    pub embedding_model: Option<String>,

    /// Cross-encoder model name (for cross_encoder_rerank)
    pub rerank_model: Option<String>,

    // === Behavior ===
    /// Enable debug logging for RAG operations
    pub debug_enabled: bool,

    /// Path for debug log files (if enabled)
    pub debug_log_path: Option<String>,

    /// Log individual retrieval steps
    pub log_retrieval_steps: bool,

    /// Log LLM calls made by RAG
    pub log_llm_calls: bool,

    // === Advanced ===
    /// For agentic mode: max iterations before giving up
    pub agentic_max_iterations: usize,

    /// For self-reflection: threshold to trigger re-retrieval
    pub self_reflection_threshold: f32,

    /// For CRAG: threshold for "retrieval is bad"
    pub crag_quality_threshold: f32,

    /// Weights for hybrid search fusion
    pub hybrid_weights: HybridWeights,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            tier: RagTier::Fast,
            features: RagFeatures::none(),
            use_custom_features: false,
            max_extra_llm_calls: 5,
            max_chunks: 10,
            max_knowledge_tokens: 4000,
            dynamic_context: true,
            min_relevance_score: 0.1,
            embedding_model: None,
            rerank_model: None,
            debug_enabled: false,
            debug_log_path: None,
            log_retrieval_steps: false,
            log_llm_calls: false,
            agentic_max_iterations: 5,
            self_reflection_threshold: 0.5,
            crag_quality_threshold: 0.3,
            hybrid_weights: HybridWeights::default(),
        }
    }
}

impl RagConfig {
    /// Create with a specific tier
    pub fn with_tier(tier: RagTier) -> Self {
        Self {
            tier,
            features: tier.to_features(),
            use_custom_features: false,
            ..Default::default()
        }
    }

    /// Create with custom features
    pub fn with_features(features: RagFeatures) -> Self {
        Self {
            tier: RagTier::Custom,
            features,
            use_custom_features: true,
            ..Default::default()
        }
    }

    /// Get the effective features (resolves tier vs custom)
    pub fn effective_features(&self) -> RagFeatures {
        if self.use_custom_features {
            self.features.clone()
        } else {
            self.tier.to_features()
        }
    }

    /// Estimate extra LLM calls needed for current config
    pub fn estimate_extra_calls(&self) -> (usize, Option<usize>) {
        let f = self.effective_features();
        let mut min = 0;
        let mut max = 0;
        let mut unbounded = false;

        if f.query_expansion {
            min += 1;
            max += 1;
        }
        if f.multi_query {
            min += 1;
            max += 1;
        }
        if f.hyde {
            min += 1;
            max += 1;
        }
        if f.reranking {
            min += 1;
            max += 1;
        }
        if f.contextual_compression {
            min += 1;
            max += 3; // Per-chunk
        }
        if f.self_reflection {
            min += 1;
            max += 2;
        }
        if f.corrective_rag {
            min += 1;
            max += 2;
        }
        if f.adaptive_strategy {
            min += 1;
            max += 1;
        }
        if f.agentic_mode {
            unbounded = true;
            min += 2;
        }
        if f.graph_rag {
            unbounded = true;
            min += 1;
        }

        // Apply hard cap
        max = max.min(self.max_extra_llm_calls);

        if unbounded {
            (min.min(self.max_extra_llm_calls), None)
        } else {
            (min, Some(max))
        }
    }

    /// Check what requirements are needed for current config
    pub fn check_requirements(&self) -> Vec<RagRequirement> {
        let f = self.effective_features();
        let mut reqs = Vec::new();

        if f.semantic_search || f.hybrid_search || f.hyde {
            reqs.push(RagRequirement::EmbeddingModel);
        }
        if f.cross_encoder_rerank {
            reqs.push(RagRequirement::CrossEncoderModel);
        }
        if f.graph_rag {
            reqs.push(RagRequirement::GraphDatabase);
        }
        if f.raptor {
            reqs.push(RagRequirement::PreprocessedCorpus);
        }
        if f.multimodal {
            reqs.push(RagRequirement::VisionModel);
        }
        if f.synonym_expansion {
            reqs.push(RagRequirement::SynonymDictionary);
        }
        if f.parent_document {
            reqs.push(RagRequirement::DocumentHierarchy);
        }
        if f.sentence_window {
            reqs.push(RagRequirement::SentenceBoundaries);
        }

        // Deduplicate
        reqs.sort_by_key(|r| format!("{:?}", r));
        reqs.dedup();
        reqs
    }

    /// Check if a specific feature is enabled
    pub fn is_feature_enabled(&self, feature: &str) -> bool {
        let f = self.effective_features();
        match feature {
            "fts_search" => f.fts_search,
            "semantic_search" => f.semantic_search,
            "hybrid_search" => f.hybrid_search,
            "synonym_expansion" => f.synonym_expansion,
            "query_expansion" => f.query_expansion,
            "multi_query" => f.multi_query,
            "hyde" => f.hyde,
            "reranking" => f.reranking,
            "cross_encoder_rerank" => f.cross_encoder_rerank,
            "fusion_rrf" => f.fusion_rrf,
            "contextual_compression" => f.contextual_compression,
            "sentence_window" => f.sentence_window,
            "parent_document" => f.parent_document,
            "self_reflection" => f.self_reflection,
            "corrective_rag" => f.corrective_rag,
            "adaptive_strategy" => f.adaptive_strategy,
            "agentic_mode" => f.agentic_mode,
            "graph_rag" => f.graph_rag,
            "raptor" => f.raptor,
            "multimodal" => f.multimodal,
            _ => false,
        }
    }

    /// Enable debug logging with a specific path
    pub fn with_debug(mut self, path: impl Into<String>) -> Self {
        self.debug_enabled = true;
        self.debug_log_path = Some(path.into());
        self.log_retrieval_steps = true;
        self.log_llm_calls = true;
        self
    }

    /// Set max LLM calls
    pub fn with_max_calls(mut self, max: usize) -> Self {
        self.max_extra_llm_calls = max;
        self
    }

    /// Set max chunks
    pub fn with_max_chunks(mut self, max: usize) -> Self {
        self.max_chunks = max;
        self
    }

    /// Set embedding model
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    /// Get a summary of the configuration
    pub fn summary(&self) -> String {
        let f = self.effective_features();
        let (min_calls, max_calls) = self.estimate_extra_calls();
        let max_str = max_calls.map(|m| m.to_string()).unwrap_or("∞".to_string());

        format!(
            "RAG Config: {} tier, {} features enabled, {}-{} LLM calls",
            self.tier.display_name(),
            f.enabled_count(),
            min_calls,
            max_str
        )
    }
}

// ============================================================================
// Hybrid Search Weights
// ============================================================================

/// Weights for combining different search methods in hybrid search
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridWeights {
    /// Weight for keyword/BM25/FTS score (0.0 to 1.0)
    pub keyword: f32,
    /// Weight for semantic/embedding score (0.0 to 1.0)
    pub semantic: f32,
    /// Weight for recency (0.0 to 1.0)
    pub recency: f32,
    /// Weight for source priority (0.0 to 1.0)
    pub priority: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            keyword: 0.4,
            semantic: 0.5,
            recency: 0.05,
            priority: 0.05,
        }
    }
}

impl HybridWeights {
    /// Create balanced weights
    pub fn balanced() -> Self {
        Self {
            keyword: 0.5,
            semantic: 0.5,
            recency: 0.0,
            priority: 0.0,
        }
    }

    /// Favor keyword matching
    pub fn keyword_heavy() -> Self {
        Self {
            keyword: 0.7,
            semantic: 0.3,
            recency: 0.0,
            priority: 0.0,
        }
    }

    /// Favor semantic matching
    pub fn semantic_heavy() -> Self {
        Self {
            keyword: 0.3,
            semantic: 0.7,
            recency: 0.0,
            priority: 0.0,
        }
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.keyword + self.semantic + self.recency + self.priority;
        if total > 0.0 {
            self.keyword /= total;
            self.semantic /= total;
            self.recency /= total;
            self.priority /= total;
        }
    }
}

// ============================================================================
// RAG Statistics
// ============================================================================

/// Statistics about RAG operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RagStats {
    /// Total queries processed
    pub queries_processed: usize,
    /// Total LLM calls made by RAG
    pub llm_calls: usize,
    /// Total chunks retrieved
    pub chunks_retrieved: usize,
    /// Total chunks after filtering
    pub chunks_used: usize,
    /// Average relevance score
    pub avg_relevance: f32,
    /// Cache hits (if caching enabled)
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Feature usage counts
    pub feature_usage: HashMap<String, usize>,
    /// Time spent in retrieval (ms)
    pub retrieval_time_ms: u64,
    /// Time spent in LLM calls (ms)
    pub llm_time_ms: u64,
}

impl RagStats {
    /// Record a query
    pub fn record_query(&mut self) {
        self.queries_processed += 1;
    }

    /// Record LLM calls
    pub fn record_llm_calls(&mut self, count: usize, time_ms: u64) {
        self.llm_calls += count;
        self.llm_time_ms += time_ms;
    }

    /// Record chunk retrieval
    pub fn record_retrieval(&mut self, retrieved: usize, used: usize, time_ms: u64) {
        self.chunks_retrieved += retrieved;
        self.chunks_used += used;
        self.retrieval_time_ms += time_ms;
    }

    /// Record feature usage
    pub fn record_feature(&mut self, feature: &str) {
        *self.feature_usage.entry(feature.to_string()).or_insert(0) += 1;
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }

    /// Get average LLM calls per query
    pub fn avg_llm_calls_per_query(&self) -> f32 {
        if self.queries_processed == 0 {
            0.0
        } else {
            self.llm_calls as f32 / self.queries_processed as f32
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "RAG Stats: {} queries, {} LLM calls ({:.1}/query), {} chunks retrieved, {:.0}% cache hit rate",
            self.queries_processed,
            self.llm_calls,
            self.avg_llm_calls_per_query(),
            self.chunks_retrieved,
            self.cache_hit_rate() * 100.0
        )
    }
}

// ============================================================================
// Auto-Selection Helper
// ============================================================================

/// Hints about the current environment for auto-selecting tier
#[derive(Clone, Debug, Default)]
pub struct TierSelectionHints {
    /// Estimated model speed (tokens/second)
    pub model_speed: Option<f32>,
    /// Whether embeddings are available
    pub has_embeddings: bool,
    /// Whether graph database is available
    pub has_graph_db: bool,
    /// User preference for speed vs quality
    pub preference: UserPreference,
    /// Query complexity (simple keyword, question, complex reasoning)
    pub query_complexity: QueryComplexity,
}

/// User preference for speed vs quality tradeoff
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UserPreference {
    /// Prefer fast responses
    Speed,
    #[default]
    /// Balance speed and quality
    Balanced,
    /// Prefer high quality
    Quality,
    /// Maximum quality regardless of cost
    MaxQuality,
}

/// Estimated query complexity
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueryComplexity {
    /// Simple keyword lookup
    Simple,
    #[default]
    /// Standard question
    Standard,
    /// Complex multi-part question
    Complex,
    /// Requires reasoning or multi-step retrieval
    Reasoning,
}

/// Auto-select appropriate tier based on hints
pub fn auto_select_tier(hints: &TierSelectionHints) -> RagTier {
    // Start with preference-based baseline
    let baseline = match hints.preference {
        UserPreference::Speed => RagTier::Fast,
        UserPreference::Balanced => RagTier::Semantic,
        UserPreference::Quality => RagTier::Enhanced,
        UserPreference::MaxQuality => RagTier::Thorough,
    };

    // Adjust based on query complexity
    let adjusted = match hints.query_complexity {
        QueryComplexity::Simple => {
            // Keep as-is or downgrade
            baseline
        }
        QueryComplexity::Standard => baseline,
        QueryComplexity::Complex => {
            // Upgrade one tier if not already high
            match baseline {
                RagTier::Fast => RagTier::Semantic,
                RagTier::Semantic => RagTier::Enhanced,
                RagTier::Enhanced => RagTier::Thorough,
                other => other,
            }
        }
        QueryComplexity::Reasoning => {
            // Use at least Thorough
            match baseline {
                RagTier::Fast | RagTier::Semantic | RagTier::Enhanced => RagTier::Thorough,
                other => other,
            }
        }
    };

    // Check if embeddings are available for semantic tiers
    if !hints.has_embeddings
        && matches!(
            adjusted,
            RagTier::Semantic | RagTier::Enhanced | RagTier::Thorough
        )
    {
        return RagTier::Fast; // Fall back to keyword-only
    }

    // Check if graph is available for graph tier
    if !hints.has_graph_db && matches!(adjusted, RagTier::Graph) {
        return RagTier::Agentic; // Fall back to agentic without graph
    }

    adjusted
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_to_features() {
        let fast = RagTier::Fast.to_features();
        assert!(fast.fts_search);
        assert!(!fast.semantic_search);

        let semantic = RagTier::Semantic.to_features();
        assert!(semantic.fts_search);
        assert!(semantic.semantic_search);
        assert!(semantic.hybrid_search);

        let full = RagTier::Full.to_features();
        assert_eq!(full.enabled_count(), 20); // All features
    }

    #[test]
    fn test_config_with_tier() {
        let config = RagConfig::with_tier(RagTier::Enhanced);
        assert_eq!(config.tier, RagTier::Enhanced);
        assert!(!config.use_custom_features);

        let features = config.effective_features();
        assert!(features.reranking);
        assert!(features.query_expansion);
    }

    #[test]
    fn test_custom_features() {
        let mut features = RagFeatures::none();
        features.fts_search = true;
        features.reranking = true;

        let config = RagConfig::with_features(features);
        assert_eq!(config.tier, RagTier::Custom);
        assert!(config.use_custom_features);

        let effective = config.effective_features();
        assert!(effective.fts_search);
        assert!(effective.reranking);
        assert!(!effective.semantic_search);
    }

    #[test]
    fn test_estimate_calls() {
        let fast = RagConfig::with_tier(RagTier::Fast);
        assert_eq!(fast.estimate_extra_calls(), (0, Some(0)));

        let enhanced = RagConfig::with_tier(RagTier::Enhanced);
        let (min, max) = enhanced.estimate_extra_calls();
        assert!(min >= 1);
        assert!(max.is_some() && max.unwrap() >= min);

        let agentic = RagConfig::with_tier(RagTier::Agentic);
        let (_, max) = agentic.estimate_extra_calls();
        assert!(max.is_none()); // Unbounded
    }

    #[test]
    fn test_requirements() {
        let fast = RagConfig::with_tier(RagTier::Fast);
        let reqs = fast.check_requirements();
        assert!(reqs.is_empty()); // No special requirements

        let semantic = RagConfig::with_tier(RagTier::Semantic);
        let reqs = semantic.check_requirements();
        assert!(reqs.contains(&RagRequirement::EmbeddingModel));

        let graph = RagConfig::with_tier(RagTier::Graph);
        let reqs = graph.check_requirements();
        assert!(reqs.contains(&RagRequirement::GraphDatabase));
    }

    #[test]
    fn test_auto_select() {
        let speed_hints = TierSelectionHints {
            preference: UserPreference::Speed,
            has_embeddings: true,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&speed_hints), RagTier::Fast);

        let quality_hints = TierSelectionHints {
            preference: UserPreference::Quality,
            has_embeddings: true,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&quality_hints), RagTier::Enhanced);

        // No embeddings should fall back
        let no_embed_hints = TierSelectionHints {
            preference: UserPreference::Quality,
            has_embeddings: false,
            ..Default::default()
        };
        assert_eq!(auto_select_tier(&no_embed_hints), RagTier::Fast);
    }

    #[test]
    fn test_hybrid_weights() {
        let mut weights = HybridWeights::default();
        let _total_before = weights.keyword + weights.semantic + weights.recency + weights.priority;
        weights.normalize();
        let total_after = weights.keyword + weights.semantic + weights.recency + weights.priority;
        assert!((total_after - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rag_stats() {
        let mut stats = RagStats::default();
        stats.record_query();
        stats.record_llm_calls(2, 500);
        stats.record_retrieval(10, 5, 100);
        stats.cache_hits = 3;
        stats.cache_misses = 7;

        assert_eq!(stats.queries_processed, 1);
        assert_eq!(stats.llm_calls, 2);
        assert!((stats.cache_hit_rate() - 0.3).abs() < 0.001);
        assert!((stats.avg_llm_calls_per_query() - 2.0).abs() < 0.001);
    }
}
