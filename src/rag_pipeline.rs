//! RAG Processing Pipeline - Orchestrates all RAG methods based on configuration
//!
//! This module provides the main pipeline that coordinates all RAG (Retrieval-Augmented
//! Generation) operations based on the configured tier and features. It handles:
//!
//! - Query processing (analysis, expansion, decomposition)
//! - Multi-stage retrieval (keyword, semantic, hybrid)
//! - Post-processing (reranking, compression, fusion)
//! - Self-improvement (reflection, corrective RAG, adaptive strategy)
//! - Advanced features (agentic, graph, RAPTOR)
//!
//! # Architecture
//!
//! The pipeline is organized into stages:
//!
//! ```text
//! ┌─────────────────┐
//! │  Query Input    │
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │ Query Processing│  ← Analysis, Expansion, HyDE, Multi-Query
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │    Retrieval    │  ← Keyword, Semantic, Hybrid, Graph
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │ Post-Processing │  ← Fusion, Reranking, Compression
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │ Self-Improvement│  ← Reflection, CRAG, Adaptive
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │ Context Assembly│  ← Final context for LLM
//! └────────────────┘
//! ```
//!
//! # Usage
//!
//! `process` is synchronous and takes the four callbacks the pipeline needs —
//! this example used to show `process(query).await?` with a single argument,
//! which never compiled against any version of this API. Doc examples are not
//! built by CI (`--lib` and `--test '*'`, never `--doc`), so it went unnoticed.
//!
//! ```rust
//! use ai_assistant::rag_pipeline::{
//!     EmbeddingCallback, LlmCallback, RagPipeline, RetrievalCallback, RetrievedChunk,
//! };
//! use ai_assistant::rag_tiers::{RagTier, RagTierConfig};
//!
//! struct MyLlm;
//! impl LlmCallback for MyLlm {
//!     fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
//!         Ok("...".to_string())
//!     }
//!     fn model_name(&self) -> &str { "my-model" }
//! }
//!
//! struct MyEmbedder;
//! impl EmbeddingCallback for MyEmbedder {
//!     fn embed(&self, _text: &str) -> Result<Vec<f32>, String> { Ok(vec![0.0; 3]) }
//!     fn model_name(&self) -> &str { "my-embedder" }
//!     fn dimension(&self) -> usize { 3 }
//! }
//!
//! struct MyIndex;
//! impl RetrievalCallback for MyIndex {
//!     fn keyword_search(&self, _q: &str, _n: usize) -> Result<Vec<RetrievedChunk>, String> {
//!         Ok(Vec::new())
//!     }
//!     fn semantic_search(&self, _e: &[f32], _n: usize) -> Result<Vec<RetrievedChunk>, String> {
//!         Ok(Vec::new())
//!     }
//!     fn get_chunk(&self, _id: &str) -> Result<Option<RetrievedChunk>, String> { Ok(None) }
//! }
//!
//! let mut pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Enhanced));
//! let result = pipeline.process(
//!     "What is the Aurora MR's cargo capacity?",
//!     &MyLlm,
//!     Some(&MyEmbedder),
//!     &MyIndex,
//!     None, // optional GraphCallback
//! )?;
//!
//! println!("Retrieved context: {}", result.context);
//! println!("Sources: {:?}", result.sources);
//! # Ok::<(), ai_assistant::rag_pipeline::RagPipelineError>(())
//! ```
//!
//! [`VectorDbRetrieval`] saves writing that `RetrievalCallback` by hand when the
//! chunks live in a vector database.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::rag_debug::{
    RagDebugConfig, RagDebugLogger, RagDebugStep, RagQuerySession, ScoreChange,
};
use crate::rag_tiers::{RagFeatures, RagRequirement, RagStats, RagTier, RagTierConfig};

// ============================================================================
// Pipeline Result Types
// ============================================================================

/// Result of a RAG pipeline execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagPipelineResult {
    /// The assembled context for the LLM
    pub context: String,

    /// Retrieved chunks with metadata
    pub chunks: Vec<RetrievedChunk>,

    /// Sources used (file names, document IDs, etc.)
    pub sources: Vec<String>,

    /// Total tokens in context
    pub token_count: usize,

    /// Whether context was truncated
    pub was_truncated: bool,

    /// Processing statistics
    pub stats: RagPipelineStats,

    /// Queries used (original + expanded/decomposed)
    pub queries_used: Vec<String>,

    /// Debug session ID (if debugging enabled)
    pub debug_session_id: Option<String>,
}

/// A retrieved chunk with scoring information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// Unique chunk identifier
    pub chunk_id: String,

    /// The chunk content
    pub content: String,

    /// Source document/file
    pub source: String,

    /// Section/heading within source
    pub section: Option<String>,

    /// Relevance score (0.0 to 1.0)
    pub score: f32,

    /// Keyword/BM25 score component
    pub keyword_score: Option<f32>,

    /// Semantic similarity score component
    pub semantic_score: Option<f32>,

    /// Token count
    pub token_count: usize,

    /// Position in original document
    pub position: Option<PipelineChunkPosition>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Position information for a chunk
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineChunkPosition {
    pub start_offset: usize,
    pub end_offset: usize,
    pub paragraph_index: Option<usize>,
    pub sentence_indices: Option<Vec<usize>>,
}

/// Statistics from pipeline execution
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RagPipelineStats {
    /// Total pipeline duration (ms)
    pub total_duration_ms: u64,

    /// Query processing duration (ms)
    pub query_processing_ms: u64,

    /// Retrieval duration (ms)
    pub retrieval_ms: u64,

    /// Post-processing duration (ms)
    pub post_processing_ms: u64,

    /// Self-improvement duration (ms)
    pub self_improvement_ms: u64,

    /// Context assembly duration (ms)
    pub assembly_ms: u64,

    /// Number of LLM calls made
    pub llm_calls: usize,

    /// LLM input tokens
    pub llm_input_tokens: usize,

    /// LLM output tokens
    pub llm_output_tokens: usize,

    /// Chunks retrieved before filtering
    pub chunks_retrieved: usize,

    /// Chunks after filtering/reranking
    pub chunks_filtered: usize,

    /// Chunks used in final context
    pub chunks_used: usize,

    /// Features that were actually executed
    pub features_executed: Vec<String>,

    /// Agentic iterations (if applicable)
    pub agentic_iterations: usize,

    /// Self-reflection triggered
    pub self_reflection_triggered: bool,

    /// CRAG action taken
    pub crag_action: Option<String>,
}

// ============================================================================
// Pipeline Error
// ============================================================================

/// Errors that can occur during pipeline execution
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RagPipelineError {
    /// No retrieval sources available
    NoSources,

    /// Required feature not available
    MissingRequirement(RagRequirement),

    /// Query processing failed
    QueryProcessingError(String),

    /// Retrieval failed
    RetrievalError(String),

    /// Post-processing failed
    PostProcessingError(String),

    /// LLM call failed
    LlmError(String),

    /// Timeout exceeded
    Timeout,

    /// Configuration error
    ConfigError(String),

    /// Internal error
    Internal(String),
}

impl std::fmt::Display for RagPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RagPipelineError::NoSources => write!(f, "No retrieval sources available"),
            RagPipelineError::MissingRequirement(req) => {
                write!(f, "Missing requirement: {}", req.display_name())
            }
            RagPipelineError::QueryProcessingError(s) => write!(f, "Query processing error: {}", s),
            RagPipelineError::RetrievalError(s) => write!(f, "Retrieval error: {}", s),
            RagPipelineError::PostProcessingError(s) => write!(f, "Post-processing error: {}", s),
            RagPipelineError::LlmError(s) => write!(f, "LLM error: {}", s),
            RagPipelineError::Timeout => write!(f, "Pipeline timeout exceeded"),
            RagPipelineError::ConfigError(s) => write!(f, "Configuration error: {}", s),
            RagPipelineError::Internal(s) => write!(f, "Internal error: {}", s),
        }
    }
}

impl std::error::Error for RagPipelineError {}

impl crate::error_taxonomy::ErrorCode for RagPipelineError {
    fn code(&self) -> &'static str {
        match self {
            RagPipelineError::NoSources => "RAG_PIPELINE_NO_SOURCES",
            RagPipelineError::MissingRequirement(_) => "RAG_PIPELINE_MISSING_REQUIREMENT",
            RagPipelineError::QueryProcessingError(_) => "RAG_PIPELINE_QUERY_PROCESSING",
            RagPipelineError::RetrievalError(_) => "RAG_PIPELINE_RETRIEVAL",
            RagPipelineError::PostProcessingError(_) => "RAG_PIPELINE_POST_PROCESSING",
            RagPipelineError::LlmError(_) => "RAG_PIPELINE_LLM",
            RagPipelineError::Timeout => "RAG_PIPELINE_TIMEOUT",
            RagPipelineError::ConfigError(_) => "RAG_PIPELINE_CONFIG",
            RagPipelineError::Internal(_) => "RAG_PIPELINE_INTERNAL",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            RagPipelineError::NoSources | RagPipelineError::Timeout => Vec::new(),
            RagPipelineError::MissingRequirement(req) => {
                vec![("requirement", req.display_name().to_string())]
            }
            RagPipelineError::QueryProcessingError(s)
            | RagPipelineError::RetrievalError(s)
            | RagPipelineError::PostProcessingError(s)
            | RagPipelineError::LlmError(s)
            | RagPipelineError::ConfigError(s)
            | RagPipelineError::Internal(s) => vec![("reason", s.clone())],
        }
    }
}

// ============================================================================
// Pipeline Callbacks/Traits
// ============================================================================

/// Callback for LLM calls within the pipeline
pub trait LlmCallback: Send + Sync {
    /// Generate a completion for the given prompt
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String>;

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Estimate token count for text
    fn estimate_tokens(&self, text: &str) -> usize {
        // Default: rough estimate based on words
        text.split_whitespace().count() * 4 / 3
    }
}

/// Callback for embedding generation
pub trait EmbeddingCallback: Send + Sync {
    /// Generate embedding for text
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;

    /// Generate embeddings for multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get model name
    fn model_name(&self) -> &str;
}

/// Callback for retrieval from indexed sources
pub trait RetrievalCallback: Send + Sync {
    /// Keyword/FTS search
    fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<RetrievedChunk>, String>;

    /// Semantic search with embedding
    fn semantic_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RetrievedChunk>, String>;

    /// Get chunk by ID
    fn get_chunk(&self, chunk_id: &str) -> Result<Option<RetrievedChunk>, String>;

    /// Get parent document for a chunk
    fn get_parent(&self, _chunk_id: &str) -> Result<Option<RetrievedChunk>, String> {
        Ok(None) // Default: no parent hierarchy
    }

    /// Get surrounding sentences for a chunk
    fn get_sentence_window(
        &self,
        _chunk_id: &str,
        _window_size: usize,
    ) -> Result<Vec<RetrievedChunk>, String> {
        Ok(vec![]) // Default: no sentence window
    }
}

/// Callback for graph traversal
pub trait GraphCallback: Send + Sync {
    /// Find entities in text
    fn extract_entities(&self, text: &str) -> Result<Vec<String>, String>;

    /// Get related entities
    fn get_related(&self, entity: &str, depth: usize) -> Result<Vec<GraphRelation>, String>;

    /// Get chunks mentioning entities
    fn get_entity_chunks(&self, entities: &[String]) -> Result<Vec<RetrievedChunk>, String>;
}

/// A relationship in the knowledge graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphRelation {
    pub from: String,
    pub to: String,
    pub relation_type: String,
    pub weight: f32,
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Additional pipeline configuration beyond RagTierConfig
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RagPipelineConfig {
    /// Base RAG configuration
    pub rag_config: RagTierConfig,

    /// Timeout for entire pipeline (ms)
    pub timeout_ms: Option<u64>,

    /// Timeout for individual LLM calls (ms)
    pub llm_timeout_ms: u64,

    /// Whether to continue on non-fatal errors
    pub continue_on_error: bool,

    /// Minimum chunks required for valid context
    pub min_chunks: usize,

    /// Enable caching of intermediate results
    pub enable_caching: bool,

    /// Deduplication threshold (similarity)
    pub dedup_threshold: f32,

    /// Query expansion: max variants to generate
    pub max_query_variants: usize,

    /// Multi-query: max sub-queries
    pub max_sub_queries: usize,

    /// Reranking: top-k to keep
    pub rerank_top_k: usize,

    /// Compression: target tokens per chunk
    pub compression_target_tokens: usize,

    /// Sentence window: sentences before/after
    pub sentence_window_size: usize,

    /// Agentic: max search iterations
    pub agentic_max_iterations: usize,

    /// Graph: max traversal depth
    pub graph_max_depth: usize,
}

impl Default for RagPipelineConfig {
    fn default() -> Self {
        Self {
            rag_config: RagTierConfig::default(),
            timeout_ms: Some(30000), // 30 seconds
            llm_timeout_ms: 10000,   // 10 seconds per LLM call
            continue_on_error: true,
            min_chunks: 1,
            enable_caching: true,
            dedup_threshold: 0.9,
            max_query_variants: 5,
            max_sub_queries: 4,
            rerank_top_k: 10,
            compression_target_tokens: 200,
            sentence_window_size: 2,
            agentic_max_iterations: 5,
            graph_max_depth: 2,
        }
    }
}

impl RagPipelineConfig {
    /// Create from a RagTierConfig
    pub fn from_rag_config(config: RagTierConfig) -> Self {
        Self {
            rag_config: config,
            ..Default::default()
        }
    }

    /// Create for a specific tier
    pub fn for_tier(tier: RagTier) -> Self {
        Self::from_rag_config(RagTierConfig::with_tier(tier))
    }
}

// ============================================================================
// The Pipeline
// ============================================================================

/// The main RAG processing pipeline
pub struct RagPipeline {
    config: RagPipelineConfig,
    debug_logger: Arc<RagDebugLogger>,
    stats: RagStats,
}

/// How many chunks are ever put in front of a model for ranking.
///
/// The prompt used to carry every chunk while asking for 100 tokens back, so a
/// long list was answered with a truncated ranking and the tail silently became
/// "not mentioned". Bounding the question is cheaper than parsing a cut answer,
/// and the chunks past the cap are the least relevant ones by definition --
/// they arrive already ordered by the fusion.
const MAX_CHUNKS_TO_RERANK: usize = 20;

/// The result of asking a model to reorder chunks.
///
/// `note` exists because "reranking did not happen" and "reranking happened"
/// used to be indistinguishable: both came back as `Ok(Vec<..>)`.
#[derive(Debug)]
struct Reranked {
    chunks: Vec<RetrievedChunk>,
    /// How many passages the model actually placed.
    ranked: usize,
    /// Set when the ranking was unusable or incomplete.
    note: Option<&'static str>,
}

impl RagPipeline {
    /// Create a new pipeline with configuration
    pub fn new(rag_config: RagTierConfig) -> Self {
        Self::with_config(RagPipelineConfig::from_rag_config(rag_config))
    }

    /// Create with full pipeline configuration
    pub fn with_config(config: RagPipelineConfig) -> Self {
        let debug_config = if config.rag_config.debug_enabled {
            RagDebugConfig {
                enabled: true,
                level: crate::rag_debug::RagDebugLevel::Detailed,
                log_to_file: config.rag_config.debug_log_path.is_some(),
                log_path: config.rag_config.debug_log_path.clone().map(Into::into),
                log_to_stderr: true,
                ..Default::default()
            }
        } else {
            RagDebugConfig::default()
        };

        Self {
            config,
            debug_logger: Arc::new(RagDebugLogger::new(debug_config)),
            stats: RagStats::default(),
        }
    }

    /// Create with custom debug logger
    pub fn with_debug_logger(config: RagPipelineConfig, logger: Arc<RagDebugLogger>) -> Self {
        Self {
            config,
            debug_logger: logger,
            stats: RagStats::default(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &RagPipelineConfig {
        &self.config
    }

    /// Get the debug logger
    pub fn debug_logger(&self) -> Arc<RagDebugLogger> {
        self.debug_logger.clone()
    }

    /// Get accumulated statistics
    pub fn stats(&self) -> &RagStats {
        &self.stats
    }

    /// Check if all requirements are met
    pub fn check_requirements(
        &self,
        has_embeddings: bool,
        has_graph: bool,
        has_cross_encoder: bool,
    ) -> Vec<RagRequirement> {
        let reqs = self.config.rag_config.check_requirements();
        let mut missing = Vec::new();

        for req in reqs {
            let met = match &req {
                RagRequirement::EmbeddingModel => has_embeddings,
                RagRequirement::GraphDatabase => has_graph,
                RagRequirement::CrossEncoderModel => has_cross_encoder,
                // These are data requirements, assume met if not checked
                RagRequirement::PreprocessedCorpus => true,
                RagRequirement::VisionModel => true,
                RagRequirement::SynonymDictionary => true,
                RagRequirement::DocumentHierarchy => true,
                RagRequirement::SentenceBoundaries => true,
                RagRequirement::DiscourseBoundaries => true,
                RagRequirement::WebSearchProvider => true,
                RagRequirement::MultipleRerankerModels => true,
                RagRequirement::AdvancedMemorySystem => true,
                RagRequirement::MultiLayerGraphSetup => true,
            };

            if !met {
                missing.push(req);
            }
        }

        missing
    }

    /// Process a query through the pipeline
    ///
    /// This is the main entry point. It orchestrates all RAG stages based on
    /// the configured tier and features.
    pub fn process(
        &mut self,
        query: &str,
        llm: &dyn LlmCallback,
        embeddings: Option<&dyn EmbeddingCallback>,
        retrieval: &dyn RetrievalCallback,
        graph: Option<&dyn GraphCallback>,
    ) -> Result<RagPipelineResult, RagPipelineError> {
        let start = Instant::now();
        let features = self.config.rag_config.effective_features();

        // Start debug session
        let debug_session = self.debug_logger.start_query(query);
        debug_session.set_tier(self.config.rag_config.tier.display_name());
        debug_session.set_features(
            features
                .enabled_features()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let mut stats = RagPipelineStats::default();
        let mut llm_calls_remaining = self.config.rag_config.max_extra_llm_calls;

        // Stage 1: Query Processing
        let query_start = Instant::now();
        let processed_queries = self.process_queries(
            query,
            &features,
            llm,
            embeddings,
            &mut llm_calls_remaining,
            &debug_session,
        )?;
        stats.query_processing_ms = query_start.elapsed().as_millis() as u64;

        // Stage 2: Retrieval
        let retrieval_start = Instant::now();
        let mut chunks = self.retrieve(
            &processed_queries,
            &features,
            embeddings,
            retrieval,
            graph,
            &debug_session,
        )?;
        stats.retrieval_ms = retrieval_start.elapsed().as_millis() as u64;
        stats.chunks_retrieved = chunks.len();

        // Stage 3: Post-Processing
        let post_start = Instant::now();
        chunks = self.post_process(
            &chunks,
            query,
            &features,
            llm,
            retrieval,
            &mut llm_calls_remaining,
            &debug_session,
        )?;
        stats.post_processing_ms = post_start.elapsed().as_millis() as u64;
        stats.chunks_filtered = chunks.len();

        // Stage 4: Self-Improvement (may trigger re-retrieval)
        let self_improve_start = Instant::now();
        let (final_chunks, self_improve_triggered) = self.self_improve(
            query,
            chunks,
            &features,
            llm,
            embeddings,
            retrieval,
            &mut llm_calls_remaining,
            &debug_session,
        )?;
        stats.self_improvement_ms = self_improve_start.elapsed().as_millis() as u64;
        stats.self_reflection_triggered = self_improve_triggered;

        // Stage 5: Context Assembly
        let assembly_start = Instant::now();
        let (context, used_chunks, was_truncated) =
            self.assemble_context(&final_chunks, &features, &debug_session);
        stats.assembly_ms = assembly_start.elapsed().as_millis() as u64;
        stats.chunks_used = used_chunks.len();

        // Calculate final stats
        stats.total_duration_ms = start.elapsed().as_millis() as u64;
        stats.llm_calls = self.config.rag_config.max_extra_llm_calls - llm_calls_remaining;
        stats.features_executed = features
            .enabled_features()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Collect sources
        let sources: Vec<String> = used_chunks
            .iter()
            .map(|c| c.source.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Complete debug session
        let session_id = debug_session.session_id().to_string();
        debug_session.set_context(&context);
        debug_session.complete(None);

        // Update global stats
        self.stats.record_query();
        self.stats
            .record_llm_calls(stats.llm_calls, stats.total_duration_ms);
        self.stats.record_retrieval(
            stats.chunks_retrieved,
            stats.chunks_used,
            stats.retrieval_ms,
        );

        let token_count = estimate_tokens(&context);
        Ok(RagPipelineResult {
            context,
            chunks: used_chunks,
            sources,
            token_count,
            was_truncated,
            stats,
            queries_used: processed_queries,
            debug_session_id: Some(session_id),
        })
    }

    // ========================================================================
    // Stage 1: Query Processing
    // ========================================================================

    fn process_queries(
        &self,
        query: &str,
        features: &RagFeatures,
        llm: &dyn LlmCallback,
        embeddings: Option<&dyn EmbeddingCallback>,
        llm_calls: &mut usize,
        debug: &RagQuerySession<'_>,
    ) -> Result<Vec<String>, RagPipelineError> {
        let mut queries = vec![query.to_string()];

        // Synonym expansion (no LLM)
        if features.synonym_expansion {
            let start = Instant::now();
            let expanded = self.expand_synonyms(query);
            if !expanded.is_empty() {
                debug.log_expansion(query, expanded.clone(), "synonym", start.elapsed());
                queries.extend(expanded);
            }
        }

        // LLM-based query expansion
        if features.query_expansion && *llm_calls > 0 {
            let start = Instant::now();
            match self.llm_expand_query(query, llm) {
                Ok(expanded) => {
                    *llm_calls -= 1;
                    debug.log_expansion(query, expanded.clone(), "llm", start.elapsed());
                    queries.extend(expanded);
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("query_expansion", &e);
                }
                Err(e) => return Err(RagPipelineError::QueryProcessingError(e)),
            }
        }

        // Multi-query decomposition
        if features.multi_query && *llm_calls > 0 {
            let start = Instant::now();
            match self.decompose_query(query, llm) {
                Ok(sub_queries) => {
                    *llm_calls -= 1;
                    debug.log_step(RagDebugStep::MultiQuery {
                        original: query.to_string(),
                        sub_queries: sub_queries.clone(),
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                    queries.extend(sub_queries);
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("multi_query", &e);
                }
                Err(e) => return Err(RagPipelineError::QueryProcessingError(e)),
            }
        }

        // HyDE (Hypothetical Document Embeddings)
        if features.hyde && embeddings.is_some() && *llm_calls > 0 {
            let start = Instant::now();
            match self.generate_hyde(query, llm) {
                Ok(hypothetical) => {
                    *llm_calls -= 1;
                    debug.log_step(RagDebugStep::HyDE {
                        query: query.to_string(),
                        hypothetical_doc: truncate(&hypothetical, 200),
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                    queries.push(hypothetical);
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("hyde", &e);
                }
                Err(e) => return Err(RagPipelineError::QueryProcessingError(e)),
            }
        }

        // Deduplicate queries
        let unique: Vec<String> = queries
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .take(self.config.max_query_variants + self.config.max_sub_queries + 2)
            .collect();

        Ok(unique)
    }

    fn expand_synonyms(&self, query: &str) -> Vec<String> {
        // Basic synonym expansion - in production, use a proper dictionary
        let synonyms: HashMap<&str, &[&str]> = [
            ("ship", &["vessel", "spacecraft", "craft"][..]),
            ("weapons", &["armament", "guns", "turrets"][..]),
            ("speed", &["velocity", "performance"][..]),
            ("cargo", &["storage", "hold", "capacity"][..]),
            ("price", &["cost", "value", "usd"][..]),
        ]
        .into_iter()
        .collect();

        let mut expanded = Vec::new();
        let lower = query.to_lowercase();

        for (word, syns) in synonyms {
            if lower.contains(word) {
                for syn in syns {
                    expanded.push(lower.replace(word, syn));
                }
            }
        }

        expanded.into_iter().take(3).collect()
    }

    fn llm_expand_query(&self, query: &str, llm: &dyn LlmCallback) -> Result<Vec<String>, String> {
        let prompt = format!(
            "Generate 3 alternative ways to phrase this search query. \
             Return ONLY the alternatives, one per line, no numbering:\n\n\
             Query: {}\n\nAlternatives:",
            query
        );

        let response = llm.generate(&prompt, 200)?;

        let variants: Vec<String> = response
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && l.len() > 3)
            .take(self.config.max_query_variants)
            .map(|s| s.to_string())
            .collect();

        Ok(variants)
    }

    fn decompose_query(&self, query: &str, llm: &dyn LlmCallback) -> Result<Vec<String>, String> {
        let prompt = format!(
            "Break down this complex question into simpler sub-questions. \
             Return ONLY the sub-questions, one per line, no numbering:\n\n\
             Question: {}\n\nSub-questions:",
            query
        );

        let response = llm.generate(&prompt, 300)?;

        let sub_queries: Vec<String> = response
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && l.len() > 5 && l.contains('?'))
            .take(self.config.max_sub_queries)
            .map(|s| s.to_string())
            .collect();

        Ok(sub_queries)
    }

    fn generate_hyde(&self, query: &str, llm: &dyn LlmCallback) -> Result<String, String> {
        let prompt = format!(
            "Write a short paragraph that would answer this question. \
             Write as if you're an expert providing factual information:\n\n\
             Question: {}\n\nAnswer:",
            query
        );

        llm.generate(&prompt, 300)
    }

    // ========================================================================
    // Stage 2: Retrieval
    // ========================================================================

    fn retrieve(
        &self,
        queries: &[String],
        features: &RagFeatures,
        embeddings: Option<&dyn EmbeddingCallback>,
        retrieval: &dyn RetrievalCallback,
        graph: Option<&dyn GraphCallback>,
        debug: &RagQuerySession<'_>,
    ) -> Result<Vec<RetrievedChunk>, RagPipelineError> {
        let mut all_chunks: Vec<RetrievedChunk> = Vec::new();
        let limit = self.config.rag_config.max_chunks;

        for query in queries {
            // Keyword search
            if features.fts_search {
                let start = Instant::now();
                match retrieval.keyword_search(query, limit) {
                    Ok(chunks) => {
                        let top_score = chunks.first().map(|c| c.score);
                        debug.log_keyword_search(query, chunks.len(), top_score, start.elapsed());
                        all_chunks.extend(chunks);
                    }
                    Err(e) if self.config.continue_on_error => {
                        debug.log_warning("keyword_search", &e);
                    }
                    Err(e) => return Err(RagPipelineError::RetrievalError(e)),
                }
            }

            // Semantic search
            if features.semantic_search {
                if let Some(embed_fn) = embeddings {
                    let start = Instant::now();
                    match embed_fn.embed(query) {
                        Ok(embedding) => match retrieval.semantic_search(&embedding, limit) {
                            Ok(chunks) => {
                                let top_sim = chunks.first().map(|c| c.score);
                                debug.log_semantic_search(
                                    query,
                                    embed_fn.model_name(),
                                    chunks.len(),
                                    top_sim,
                                    start.elapsed(),
                                );
                                all_chunks.extend(chunks);
                            }
                            Err(e) if self.config.continue_on_error => {
                                debug.log_warning("semantic_search", &e);
                            }
                            Err(e) => return Err(RagPipelineError::RetrievalError(e)),
                        },
                        Err(e) if self.config.continue_on_error => {
                            debug.log_warning("embedding", &e);
                        }
                        Err(e) => return Err(RagPipelineError::RetrievalError(e)),
                    }
                }
            }

            // Graph RAG
            if features.graph_rag {
                if let Some(graph_fn) = graph {
                    let start = Instant::now();
                    match self.graph_retrieve(query, graph_fn) {
                        Ok((chunks, traversal_stats)) => {
                            debug.log_step(RagDebugStep::GraphTraversal {
                                start_entities: traversal_stats.0,
                                traversal_depth: traversal_stats.1,
                                nodes_visited: traversal_stats.2,
                                relationships_found: traversal_stats.3,
                                duration_ms: start.elapsed().as_millis() as u64,
                            });
                            all_chunks.extend(chunks);
                        }
                        Err(e) if self.config.continue_on_error => {
                            debug.log_warning("graph_rag", &e);
                        }
                        Err(e) => return Err(RagPipelineError::RetrievalError(e)),
                    }
                }
            }
        }

        // Hybrid fusion if both keyword and semantic were used
        if features.hybrid_search && features.fts_search && features.semantic_search {
            let start = Instant::now();
            let (keyword_count, semantic_count) = count_by_score_type(&all_chunks);
            all_chunks = self.hybrid_fusion(all_chunks, features);
            debug.log_step(RagDebugStep::HybridFusion {
                keyword_results: keyword_count,
                semantic_results: semantic_count,
                fused_results: all_chunks.len(),
                method: if features.fusion_rrf {
                    "rrf"
                } else {
                    "weighted"
                }
                .to_string(),
                weights: Some(
                    [
                        (
                            "keyword".to_string(),
                            self.config.rag_config.hybrid_weights.keyword,
                        ),
                        (
                            "semantic".to_string(),
                            self.config.rag_config.hybrid_weights.semantic,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                ),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Deduplicate
        all_chunks = self.deduplicate_chunks(all_chunks);

        Ok(all_chunks)
    }

    fn graph_retrieve(
        &self,
        query: &str,
        graph: &dyn GraphCallback,
    ) -> Result<(Vec<RetrievedChunk>, (Vec<String>, usize, usize, usize)), String> {
        // Extract entities from query
        let entities = graph.extract_entities(query)?;

        if entities.is_empty() {
            return Ok((vec![], (vec![], 0, 0, 0)));
        }

        // Traverse graph for related entities
        let mut all_entities = entities.clone();
        let mut relationships = 0;

        for entity in &entities {
            let related = graph.get_related(entity, self.config.graph_max_depth)?;
            relationships += related.len();
            for rel in related {
                if !all_entities.contains(&rel.to) {
                    all_entities.push(rel.to);
                }
            }
        }

        // Get chunks for all entities
        let chunks = graph.get_entity_chunks(&all_entities)?;

        Ok((
            chunks,
            (
                entities,
                self.config.graph_max_depth,
                all_entities.len(),
                relationships,
            ),
        ))
    }

    fn hybrid_fusion(
        &self,
        chunks: Vec<RetrievedChunk>,
        features: &RagFeatures,
    ) -> Vec<RetrievedChunk> {
        if features.fusion_rrf {
            self.reciprocal_rank_fusion(chunks)
        } else {
            self.weighted_fusion(chunks)
        }
    }

    /// Fuse the keyword and semantic rankings into one.
    ///
    /// The arithmetic lives in [`crate::rank_fusion`], shared with the other
    /// two copies of it that existed in this crate. The normalisation to 0-1
    /// that V316 added is now the shared default rather than a local fix — see
    /// that module for why raw RRF output always fell under
    /// `min_relevance_score`.
    ///
    /// What stays here is the adapter: which chunk belongs in which ranking,
    /// and how the fused scores get written back.
    fn reciprocal_rank_fusion(&self, chunks: Vec<RetrievedChunk>) -> Vec<RetrievedChunk> {
        let ranked_ids = |score_of: fn(&RetrievedChunk) -> Option<f32>| -> Vec<String> {
            let mut ranked: Vec<&RetrievedChunk> =
                chunks.iter().filter(|c| score_of(c).is_some()).collect();
            ranked.sort_by(|a, b| {
                score_of(b)
                    .unwrap_or(0.0)
                    .partial_cmp(&score_of(a).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.into_iter().map(|c| c.chunk_id.clone()).collect()
        };

        let lists = vec![
            ranked_ids(|c| c.keyword_score),
            ranked_ids(|c| c.semantic_score),
        ];
        let fused = crate::rank_fusion::fuse_ranked_ids_to_map(
            &lists,
            &crate::rank_fusion::RrfOptions::default(),
        );

        // A chunk that appears in neither ranking keeps the score it arrived
        // with. It used to be dropped: the old code built its result from the
        // fusion map alone, so anything retrieved by a route that sets neither
        // `keyword_score` nor `semantic_score` — graph traversal, memory —
        // disappeared here without a word. Fusing is merging; deleting is
        // somebody else's job, and there is a relevance floor two steps down
        // that already does it in the open.
        let mut result: Vec<RetrievedChunk> = chunks;
        for chunk in &mut result {
            if let Some(score) = fused.get(&chunk.chunk_id) {
                chunk.score = *score;
            }
        }

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        result
    }

    fn weighted_fusion(&self, chunks: Vec<RetrievedChunk>) -> Vec<RetrievedChunk> {
        let weights = &self.config.rag_config.hybrid_weights;

        // Group by chunk_id
        let mut grouped: HashMap<String, RetrievedChunk> = HashMap::new();

        for chunk in chunks {
            let entry = grouped
                .entry(chunk.chunk_id.clone())
                .or_insert(chunk.clone());

            // Combine scores
            let kw = chunk.keyword_score.unwrap_or(0.0) * weights.keyword;
            let sem = chunk.semantic_score.unwrap_or(0.0) * weights.semantic;

            let new_score = kw + sem;
            if new_score > entry.score {
                entry.score = new_score;
            }
        }

        let mut result: Vec<_> = grouped.into_values().collect();
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    fn deduplicate_chunks(&self, chunks: Vec<RetrievedChunk>) -> Vec<RetrievedChunk> {
        let mut seen: HashMap<String, RetrievedChunk> = HashMap::new();

        for chunk in chunks {
            let entry = seen.entry(chunk.chunk_id.clone()).or_insert(chunk.clone());
            if chunk.score > entry.score {
                *entry = chunk;
            }
        }

        let mut result: Vec<_> = seen.into_values().collect();
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    // ========================================================================
    // Stage 3: Post-Processing
    // ========================================================================

    fn post_process(
        &self,
        chunks: &[RetrievedChunk],
        query: &str,
        features: &RagFeatures,
        llm: &dyn LlmCallback,
        retrieval: &dyn RetrievalCallback,
        llm_calls: &mut usize,
        debug: &RagQuerySession<'_>,
    ) -> Result<Vec<RetrievedChunk>, RagPipelineError> {
        let mut processed = chunks.to_vec();

        // Sentence window expansion
        if features.sentence_window {
            let start = Instant::now();
            let original_count = processed.len();
            processed = self.apply_sentence_window(&processed, retrieval);
            debug.log_step(RagDebugStep::SentenceWindow {
                matched_sentences: original_count,
                window_size: self.config.sentence_window_size,
                expanded_chunks: processed.len(),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Parent document retrieval
        if features.parent_document {
            let start = Instant::now();
            let child_count = processed.len();
            processed = self.apply_parent_document(&processed, retrieval);
            let parent_count = processed.len() - child_count;
            debug.log_step(RagDebugStep::ParentDocument {
                child_matches: child_count,
                parent_docs_retrieved: parent_count as usize,
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        // LLM-based reranking
        if features.reranking && *llm_calls > 0 {
            let start = Instant::now();
            let before_scores: Vec<_> = processed
                .iter()
                .enumerate()
                .map(|(i, c)| (c.chunk_id.clone(), c.score, i))
                .collect();

            match self.llm_rerank(&processed, query, llm) {
                Ok(reranked) => {
                    *llm_calls -= 1;

                    // Said out loud, because it used to be impossible to tell
                    // apart from success: a model whose answer holds no usable
                    // ranking leaves the order exactly as it was, and that is a
                    // different outcome from having reordered.
                    if let Some(note) = reranked.note {
                        debug.log_warning("reranking", note);
                    }

                    let score_changes: Vec<ScoreChange> = reranked
                        .chunks
                        .iter()
                        .enumerate()
                        .filter_map(|(new_rank, chunk)| {
                            before_scores
                                .iter()
                                .find(|(id, _, _)| id == &chunk.chunk_id)
                                .map(|(id, old_score, old_rank)| ScoreChange {
                                    chunk_id: id.clone(),
                                    before: *old_score,
                                    after: chunk.score,
                                    rank_before: *old_rank,
                                    rank_after: new_rank,
                                })
                        })
                        .collect();

                    debug.log_step(RagDebugStep::Reranking {
                        input_count: processed.len(),
                        output_count: reranked.chunks.len(),
                        // The count belongs in the method: "llm" alone read as
                        // "the model ordered these", which was true only
                        // sometimes.
                        method: format!("llm ({} of {} placed)", reranked.ranked, processed.len()),
                        score_changes,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });

                    processed = reranked.chunks;
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("reranking", &e);
                }
                Err(e) => return Err(RagPipelineError::PostProcessingError(e)),
            }
        }

        // Contextual compression
        if features.contextual_compression && *llm_calls > 0 {
            let start = Instant::now();
            let input_tokens: usize = processed.iter().map(|c| c.token_count).sum();

            match self.compress_chunks(&processed, query, llm, llm_calls) {
                Ok(compressed) => {
                    let output_tokens: usize = compressed.iter().map(|c| c.token_count).sum();
                    let ratio = if output_tokens > 0 {
                        input_tokens as f32 / output_tokens as f32
                    } else {
                        1.0
                    };

                    debug.log_step(RagDebugStep::ContextualCompression {
                        input_chunks: processed.len(),
                        input_tokens,
                        output_chunks: compressed.len(),
                        output_tokens,
                        compression_ratio: ratio,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });

                    processed = compressed;
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("compression", &e);
                }
                Err(e) => return Err(RagPipelineError::PostProcessingError(e)),
            }
        }

        // Filter by minimum score
        let min_score = self.config.rag_config.min_relevance_score;
        processed.retain(|c| c.score >= min_score);

        // Limit to top-k
        processed.truncate(self.config.rerank_top_k);

        Ok(processed)
    }

    fn apply_sentence_window(
        &self,
        chunks: &[RetrievedChunk],
        retrieval: &dyn RetrievalCallback,
    ) -> Vec<RetrievedChunk> {
        let mut expanded = Vec::new();

        for chunk in chunks {
            match retrieval.get_sentence_window(&chunk.chunk_id, self.config.sentence_window_size) {
                Ok(window) if !window.is_empty() => {
                    expanded.extend(window);
                }
                _ => {
                    expanded.push(chunk.clone());
                }
            }
        }

        self.deduplicate_chunks(expanded)
    }

    fn apply_parent_document(
        &self,
        chunks: &[RetrievedChunk],
        retrieval: &dyn RetrievalCallback,
    ) -> Vec<RetrievedChunk> {
        let mut result = chunks.to_vec();

        for chunk in chunks {
            if let Ok(Some(parent)) = retrieval.get_parent(&chunk.chunk_id) {
                if !result.iter().any(|c| c.chunk_id == parent.chunk_id) {
                    result.push(parent);
                }
            }
        }

        result
    }

    /// Ask a model to put the chunks in order.
    ///
    /// # It reorders. It does not re-score.
    ///
    /// This used to overwrite every score with `1.0 - position / total`, which
    /// threw away what the keyword search, the semantic search and the fusion
    /// had all worked out, and replaced relevance with position. That single
    /// decision produced three separate silent failures:
    ///
    /// 1. **An answer with no usable numbers destroyed the ranking.** Nothing
    ///    matched, so every chunk fell through to the "not mentioned" branch
    ///    and got the same flat score. The function returned `Ok`, the caller
    ///    believed reranking had happened, and the ordering computed upstream
    ///    was gone.
    /// 2. **The sentinel sat exactly on the filter.** Unmentioned chunks were
    ///    given `0.1`, and `min_relevance_score` defaults to `0.1`; they
    ///    survived by exact equality of two independent `f32` literals. Raise
    ///    the floor to 0.15 and the whole tail vanished — combined with (1),
    ///    that meant a garbled answer emptied the context entirely and nothing
    ///    said so.
    /// 3. **It laundered upstream defects.** Because scores were overwritten,
    ///    any scale bug before this point became invisible to every tier that
    ///    reranks. That is exactly how V316's RRF defect went unnoticed until
    ///    it showed up in `RagTier::Semantic`, the one tier that does not
    ///    rerank.
    ///
    /// Keeping the original scores fixes all three at once: the relevance
    /// floor now filters on relevance, an unusable answer costs the order and
    /// nothing else, and a bad scale upstream stays visible.
    fn llm_rerank(
        &self,
        chunks: &[RetrievedChunk],
        query: &str,
        llm: &dyn LlmCallback,
    ) -> Result<Reranked, String> {
        if chunks.is_empty() {
            return Ok(Reranked {
                chunks: vec![],
                ranked: 0,
                note: None,
            });
        }

        // Only the head of the list is put to the model. The prompt used to
        // carry every chunk while asking for 100 tokens back: with forty or
        // fifty passages the list of numbers does not fit in the answer, so it
        // was cut off and the entire tail was treated as "not mentioned".
        // Anything past the cap keeps its place, after the ranked ones.
        let to_rank = chunks.len().min(MAX_CHUNKS_TO_RERANK);

        let mut prompt = format!(
            "Given this query: \"{}\"\n\n\
             Rank the following text passages by relevance (most relevant first).\n\
             Return ONLY the passage numbers in order, comma-separated (e.g., \"3,1,5,2,4\"):\n\n",
            query
        );
        for (i, chunk) in chunks.iter().take(to_rank).enumerate() {
            let preview = truncate(&chunk.content, 200);
            prompt.push_str(&format!("{}. {}\n\n", i + 1, preview));
        }
        prompt.push_str("Ranking (most relevant first): ");

        // Sized to the answer we asked for rather than a flat 100: a ranking of
        // n passages needs roughly n numbers plus separators.
        let answer_budget = (to_rank * 6 + 16).clamp(64, 512);
        let response = llm.generate(&prompt, answer_budget)?;

        let mut order: Vec<usize> = Vec::with_capacity(to_rank);
        let mut seen = HashSet::new();
        for number in response
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0 && n <= to_rank)
        {
            let index = number - 1;
            if seen.insert(index) {
                order.push(index);
            }
        }

        if order.is_empty() {
            // Nothing usable came back. Say so and hand the chunks straight
            // through: the order from upstream is a real answer, and replacing
            // it with a flattened one was strictly worse than doing nothing.
            return Ok(Reranked {
                chunks: chunks.to_vec(),
                ranked: 0,
                note: Some("the model's answer held no usable ranking; order left as it was"),
            });
        }

        let mut reranked: Vec<RetrievedChunk> = order.iter().map(|&i| chunks[i].clone()).collect();
        // Everything the model did not place, in the order it already had.
        // Their scores are untouched: they are the relevance computed upstream,
        // not a sentinel, so the filter below still filters on relevance.
        reranked.extend(
            chunks
                .iter()
                .enumerate()
                .filter(|(i, _)| !seen.contains(i))
                .map(|(_, c)| c.clone()),
        );

        let ranked = order.len();
        Ok(Reranked {
            chunks: reranked,
            ranked,
            note: (ranked < to_rank)
                .then_some("the model ranked only part of the passages; the rest kept their order"),
        })
    }

    fn compress_chunks(
        &self,
        chunks: &[RetrievedChunk],
        query: &str,
        llm: &dyn LlmCallback,
        llm_calls: &mut usize,
    ) -> Result<Vec<RetrievedChunk>, String> {
        let mut compressed = Vec::new();

        for chunk in chunks {
            if *llm_calls == 0 {
                compressed.push(chunk.clone());
                continue;
            }

            let prompt = format!(
                "Extract only the parts relevant to answering: \"{}\"\n\n\
                 Text: {}\n\n\
                 Relevant extract (be concise):",
                query, chunk.content
            );

            match llm.generate(&prompt, self.config.compression_target_tokens) {
                Ok(extracted) => {
                    *llm_calls -= 1;
                    let mut c = chunk.clone();
                    c.content = extracted.trim().to_string();
                    c.token_count = estimate_tokens(&c.content);
                    compressed.push(c);
                }
                Err(_) => {
                    compressed.push(chunk.clone());
                }
            }
        }

        Ok(compressed)
    }

    // ========================================================================
    // Stage 4: Self-Improvement
    // ========================================================================

    fn self_improve(
        &self,
        query: &str,
        chunks: Vec<RetrievedChunk>,
        features: &RagFeatures,
        llm: &dyn LlmCallback,
        embeddings: Option<&dyn EmbeddingCallback>,
        retrieval: &dyn RetrievalCallback,
        llm_calls: &mut usize,
        debug: &RagQuerySession<'_>,
    ) -> Result<(Vec<RetrievedChunk>, bool), RagPipelineError> {
        let mut current_chunks = chunks;
        let mut triggered = false;

        // Self-reflection
        if features.self_reflection && *llm_calls > 0 {
            let start = Instant::now();
            match self.evaluate_sufficiency(query, &current_chunks, llm) {
                Ok((sufficient, confidence, reason)) => {
                    *llm_calls -= 1;

                    debug.log_step(RagDebugStep::SelfReflection {
                        query: query.to_string(),
                        context_summary: format!("{} chunks", current_chunks.len()),
                        is_sufficient: sufficient,
                        confidence,
                        reason: reason.clone(),
                        duration_ms: start.elapsed().as_millis() as u64,
                    });

                    if !sufficient && confidence < self.config.rag_config.self_reflection_threshold
                    {
                        triggered = true;
                        // Could trigger re-retrieval here
                    }
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("self_reflection", &e);
                }
                Err(e) => return Err(RagPipelineError::LlmError(e)),
            }
        }

        // Corrective RAG (CRAG)
        if features.corrective_rag && *llm_calls > 0 {
            let start = Instant::now();
            match self.evaluate_retrieval_quality(query, &current_chunks, llm) {
                Ok((quality, action)) => {
                    *llm_calls -= 1;

                    debug.log_step(RagDebugStep::CorrectiveRag {
                        retrieval_quality: quality,
                        action_taken: action.clone(),
                        reason: None,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });

                    if quality < self.config.rag_config.crag_quality_threshold {
                        triggered = true;
                        // CRAG would trigger corrective action here
                    }
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("corrective_rag", &e);
                }
                Err(e) => return Err(RagPipelineError::LlmError(e)),
            }
        }

        // Agentic mode
        if features.agentic_mode && *llm_calls > 0 {
            let chunks_for_agentic = current_chunks.clone();
            match self.agentic_retrieve(
                query,
                chunks_for_agentic,
                llm,
                embeddings,
                retrieval,
                llm_calls,
                debug,
            ) {
                Ok(chunks) => {
                    current_chunks = chunks;
                    triggered = true;
                }
                Err(e) if self.config.continue_on_error => {
                    debug.log_warning("agentic", &e);
                }
                Err(e) => return Err(RagPipelineError::LlmError(e)),
            }
        }

        Ok((current_chunks, triggered))
    }

    fn evaluate_sufficiency(
        &self,
        query: &str,
        chunks: &[RetrievedChunk],
        llm: &dyn LlmCallback,
    ) -> Result<(bool, f32, Option<String>), String> {
        let context: String = chunks
            .iter()
            .map(|c| truncate(&c.content, 300))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(
            "Question: {}\n\n\
             Available context:\n{}\n\n\
             Can this context sufficiently answer the question?\n\
             Reply with: YES/NO, confidence (0-100), brief reason\n\
             Format: YES|85|Contains relevant information about...",
            query, context
        );

        let response = llm.generate(&prompt, 100)?;

        // Parse response
        let parts: Vec<&str> = response.split('|').collect();
        let sufficient = parts
            .first()
            .map(|s| s.trim().to_uppercase().contains("YES"))
            .unwrap_or(false);
        let confidence = parts
            .get(1)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|c| c / 100.0)
            .unwrap_or(0.5);
        let reason = parts.get(2).map(|s| s.trim().to_string());

        Ok((sufficient, confidence, reason))
    }

    fn evaluate_retrieval_quality(
        &self,
        query: &str,
        chunks: &[RetrievedChunk],
        llm: &dyn LlmCallback,
    ) -> Result<(f32, String), String> {
        let context: String = chunks
            .iter()
            .take(5)
            .map(|c| truncate(&c.content, 200))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(
            "Question: {}\n\n\
             Retrieved passages:\n{}\n\n\
             Rate the overall relevance of these passages (0-100).\n\
             Reply with: score, action (use_as_is/refine/retry)\n\
             Format: 75|use_as_is",
            query, context
        );

        let response = llm.generate(&prompt, 50)?;

        let parts: Vec<&str> = response.split('|').collect();
        let quality = parts
            .first()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|q| q / 100.0)
            .unwrap_or(0.5);
        let action = parts
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "use_as_is".to_string());

        Ok((quality, action))
    }

    fn agentic_retrieve(
        &self,
        query: &str,
        initial_chunks: Vec<RetrievedChunk>,
        llm: &dyn LlmCallback,
        _embeddings: Option<&dyn EmbeddingCallback>,
        retrieval: &dyn RetrievalCallback,
        llm_calls: &mut usize,
        debug: &RagQuerySession<'_>,
    ) -> Result<Vec<RetrievedChunk>, String> {
        let mut all_chunks = initial_chunks;
        let mut iteration = 0;

        while iteration < self.config.agentic_max_iterations && *llm_calls > 0 {
            iteration += 1;
            let start = Instant::now();

            // Ask LLM what to do next
            let context_summary: String = all_chunks
                .iter()
                .take(3)
                .map(|c| truncate(&c.content, 100))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Goal: Answer \"{}\"\n\n\
                 Current context ({} passages):\n{}\n\n\
                 Should I search for more information?\n\
                 Reply: DONE (if sufficient) or SEARCH: <refined query>",
                query,
                all_chunks.len(),
                context_summary
            );

            let response = llm.generate(&prompt, 100)?;
            *llm_calls -= 1;

            let (action, observation, is_complete) = if response.to_uppercase().contains("DONE") {
                (
                    "done".to_string(),
                    "Sufficient information gathered".to_string(),
                    true,
                )
            } else if let Some(refined) = response
                .split("SEARCH:")
                .nth(1)
                .map(|s| s.trim().to_string())
            {
                // Perform additional search
                let new_chunks = retrieval.keyword_search(&refined, 5).unwrap_or_default();
                let found = new_chunks.len();
                all_chunks.extend(new_chunks);
                all_chunks = self.deduplicate_chunks(all_chunks);
                (
                    format!("search: {}", refined),
                    format!("Found {} new passages", found),
                    false,
                )
            } else {
                ("unknown".to_string(), response.clone(), true)
            };

            debug.log_step(RagDebugStep::AgenticIteration {
                iteration,
                action: action.clone(),
                observation,
                is_complete,
                duration_ms: start.elapsed().as_millis() as u64,
            });

            if is_complete {
                break;
            }
        }

        Ok(all_chunks)
    }

    // ========================================================================
    // Stage 5: Context Assembly
    // ========================================================================

    fn assemble_context(
        &self,
        chunks: &[RetrievedChunk],
        _features: &RagFeatures,
        debug: &RagQuerySession<'_>,
    ) -> (String, Vec<RetrievedChunk>, bool) {
        let start = Instant::now();
        let max_tokens = self.config.rag_config.max_knowledge_tokens;

        let mut context_parts = Vec::new();
        let mut used_chunks = Vec::new();
        let mut total_tokens = 0;
        let mut truncated = false;

        for chunk in chunks {
            if total_tokens + chunk.token_count > max_tokens {
                truncated = true;
                break;
            }

            // Format chunk with source attribution
            let formatted = format!("[Source: {}]\n{}\n", chunk.source, chunk.content.trim());

            context_parts.push(formatted);
            used_chunks.push(chunk.clone());
            total_tokens += chunk.token_count;
        }

        let sources: Vec<String> = used_chunks
            .iter()
            .map(|c| c.source.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let context = context_parts.join("\n---\n");

        debug.log_step(RagDebugStep::ContextAssembly {
            total_chunks: used_chunks.len(),
            total_tokens,
            sources,
            truncated,
            duration_ms: start.elapsed().as_millis() as u64,
        });

        (context, used_chunks, truncated)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}...",
            crate::text_util::truncate_str(s, max_len.saturating_sub(3))
        )
    }
}

fn estimate_tokens(text: &str) -> usize {
    crate::context::estimate_tokens(text)
}

fn count_by_score_type(chunks: &[RetrievedChunk]) -> (usize, usize) {
    let keyword = chunks.iter().filter(|c| c.keyword_score.is_some()).count();
    let semantic = chunks.iter().filter(|c| c.semantic_score.is_some()).count();
    (keyword, semantic)
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// VectorDb-backed retrieval (V316)
// ============================================================================

/// Retrieval backed by any [`crate::vector_db::VectorDb`] implementation.
///
/// # Why this exists
///
/// [`RetrievalCallback::semantic_search`] takes a query embedding and is meant
/// to return the nearest chunks — which is exactly what a vector database does.
/// Until V316 the crate shipped **no implementation of it**, so the ~3.700-line
/// vector subsystem and the RAG pipeline had no way to meet: every caller had to
/// write the glue by hand, and the only in-repo example of that glue (in the
/// evaluation suite) pre-computed a list outside the pipeline and ignored the
/// embedding it was handed.
///
/// It also gives [`crate::rag_methods::HierarchicalRouter`] something real to
/// point at. That router classifies a query and names a retriever — `"dense"`
/// for factual and conversational queries — but naming is all it does; nothing
/// mapped that name to a dense retriever.
///
/// # Keyword search is not silently faked
///
/// A vector database has no lexical index. Rather than return an empty result —
/// which the pipeline cannot tell apart from "nothing matched" —
/// [`Self::keyword_search`] returns an error unless you supply a delegate with
/// [`Self::with_keyword_search`]. Pairing an FTS5-backed callback for keywords
/// with this one for vectors is the hybrid setup the pipeline was built for.
///
/// # Example
///
/// ```no_run
/// # use ai_assistant::rag_pipeline::VectorDbRetrieval;
/// # use ai_assistant::vector_db::{InMemoryVectorDb, VectorDbConfig};
/// let db = InMemoryVectorDb::new(VectorDbConfig::default());
/// let retrieval = VectorDbRetrieval::new(&db);
/// // hand `&retrieval` to RagPipeline::process
/// ```
pub struct VectorDbRetrieval<'a> {
    db: &'a dyn crate::vector_db::VectorDb,
    content_key: String,
    source: String,
    keyword: Option<&'a dyn RetrievalCallback>,
}

impl<'a> VectorDbRetrieval<'a> {
    /// Wrap a vector database, reading chunk text from the `"content"` metadata
    /// key.
    pub fn new(db: &'a dyn crate::vector_db::VectorDb) -> Self {
        Self {
            db,
            content_key: "content".to_string(),
            source: "vectordb".to_string(),
            keyword: None,
        }
    }

    /// Read chunk text from a different metadata key.
    ///
    /// The vector store holds arbitrary JSON metadata, so the text has to be
    /// found by convention; `"content"` is the default because that is what the
    /// rest of the crate writes.
    pub fn with_content_key(mut self, key: impl Into<String>) -> Self {
        self.content_key = key.into();
        self
    }

    /// Set the `source` recorded on every returned chunk (default
    /// `"vectordb"`). Useful when several stores feed one pipeline and the
    /// answer should say which one a passage came from.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Delegate keyword search to another callback, making this a hybrid
    /// retriever: lexical hits from `delegate`, vector hits from here.
    ///
    /// Without this, [`Self::keyword_search`] reports that it cannot do it
    /// instead of quietly returning nothing.
    pub fn with_keyword_search(mut self, delegate: &'a dyn RetrievalCallback) -> Self {
        self.keyword = Some(delegate);
        self
    }

    /// Turn one stored vector's metadata into a chunk.
    fn to_chunk(&self, id: &str, score: f32, metadata: &serde_json::Value) -> RetrievedChunk {
        let content = metadata
            .get(&self.content_key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        // Everything else in the metadata object is carried through as strings
        // rather than dropped: the store is the caller's, and a filter or a
        // citation may depend on a key this adapter knows nothing about.
        let mut extra = HashMap::new();
        if let Some(obj) = metadata.as_object() {
            for (k, v) in obj {
                if k == &self.content_key {
                    continue;
                }
                let as_text = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                extra.insert(k.clone(), as_text);
            }
        }

        RetrievedChunk {
            chunk_id: id.to_string(),
            token_count: crate::context::estimate_tokens(&content),
            content,
            source: self.source.clone(),
            section: None,
            score,
            keyword_score: None,
            semantic_score: Some(score),
            position: None,
            metadata: extra,
        }
    }
}

impl RetrievalCallback for VectorDbRetrieval<'_> {
    fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<RetrievedChunk>, String> {
        match self.keyword {
            Some(delegate) => delegate.keyword_search(query, limit),
            None => Err("VectorDbRetrieval has no lexical index; supply one with \
                 with_keyword_search() or disable keyword search for this tier"
                .to_string()),
        }
    }

    fn semantic_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RetrievedChunk>, String> {
        let hits = self
            .db
            .search(embedding, limit, None)
            .map_err(|e| format!("vector search failed: {}", e))?;
        Ok(hits
            .iter()
            .map(|h| self.to_chunk(&h.id, h.score, &h.metadata))
            .collect())
    }

    fn get_chunk(&self, chunk_id: &str) -> Result<Option<RetrievedChunk>, String> {
        let stored = self
            .db
            .get(chunk_id)
            .map_err(|e| format!("vector get failed: {}", e))?;
        // A direct fetch is not a similarity result, so there is no score to
        // report; 1.0 would claim a perfect match that was never measured.
        Ok(stored.map(|s| self.to_chunk(&s.id, 0.0, &s.metadata)))
    }
}

#[cfg(test)]
mod vector_db_retrieval_tests {
    //! The seam between the RAG pipeline and the vector subsystem. Before V316
    //! nothing in the crate implemented `RetrievalCallback` against a
    //! `VectorDb`; the one in-repo attempt (evaluation suite) took `_emb` and
    //! ignored it, returning a list computed outside the pipeline. So the first
    //! test here is the one that attempt would fail: it asks two different
    //! questions and requires two different answers.
    use super::*;
    use crate::vector_db::{DistanceMetric, InMemoryVectorDb, VectorDb, VectorDbConfig};

    /// Stands in for an FTS5-backed callback: it answers by word, never by
    /// vector, which is the half a vector store cannot do.
    struct LexicalStub;
    impl RetrievalCallback for LexicalStub {
        fn keyword_search(&self, _q: &str, limit: usize) -> Result<Vec<RetrievedChunk>, String> {
            Ok((0..limit.min(2))
                .map(|i| RetrievedChunk {
                    chunk_id: format!("lex_{}", i),
                    content: format!("lexical hit {}", i),
                    source: "fts".to_string(),
                    section: None,
                    score: 0.5,
                    keyword_score: Some(0.5),
                    semantic_score: None,
                    token_count: 3,
                    position: None,
                    metadata: HashMap::new(),
                })
                .collect())
        }
        fn semantic_search(&self, _e: &[f32], _l: usize) -> Result<Vec<RetrievedChunk>, String> {
            Ok(Vec::new())
        }
        fn get_chunk(&self, _id: &str) -> Result<Option<RetrievedChunk>, String> {
            Ok(None)
        }
    }

    struct StubLlm;
    impl LlmCallback for StubLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
            Ok("stub".to_string())
        }
        fn model_name(&self) -> &str {
            "stub-llm"
        }
    }

    /// Three orthogonal unit vectors, so "nearest" is unambiguous and the test
    /// does not depend on how any embedding model happens to behave.
    fn seeded_db() -> InMemoryVectorDb {
        let config = VectorDbConfig {
            dimensions: 3,
            distance_metric: DistanceMetric::Cosine,
            collection_name: "seam".to_string(),
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);
        db.insert(
            "x",
            vec![1.0, 0.0, 0.0],
            serde_json::json!({"content": "about the X axis", "page": 7}),
        )
        .expect("insert x");
        db.insert(
            "y",
            vec![0.0, 1.0, 0.0],
            serde_json::json!({"content": "about the Y axis"}),
        )
        .expect("insert y");
        db.insert(
            "z",
            vec![0.0, 0.0, 1.0],
            serde_json::json!({"content": "about the Z axis"}),
        )
        .expect("insert z");
        db
    }

    #[test]
    fn the_query_embedding_actually_selects_the_chunk() {
        let db = seeded_db();
        let retrieval = VectorDbRetrieval::new(&db);

        let near_x = retrieval.semantic_search(&[1.0, 0.0, 0.0], 1).expect("x");
        let near_z = retrieval.semantic_search(&[0.0, 0.0, 1.0], 1).expect("z");

        assert_eq!(near_x[0].chunk_id, "x");
        assert_eq!(near_z[0].chunk_id, "z");
        assert_ne!(
            near_x[0].chunk_id, near_z[0].chunk_id,
            "a retriever that ignores its embedding returns the same chunk for \
             every query; that is the failure this test exists to catch"
        );
    }

    #[test]
    fn the_similarity_is_reported_as_a_semantic_score() {
        let db = seeded_db();
        let retrieval = VectorDbRetrieval::new(&db);
        let hits = retrieval
            .semantic_search(&[1.0, 0.0, 0.0], 3)
            .expect("hits");

        assert_eq!(hits.len(), 3);
        assert!(
            hits[0].semantic_score.is_some(),
            "the score came from vector similarity and must be labelled as such"
        );
        assert!(
            hits[0].keyword_score.is_none(),
            "no lexical index was consulted; claiming a keyword score would be a lie"
        );
        assert!(
            hits[0].score >= hits[1].score,
            "hits must arrive ordered: {:?}",
            hits.iter().map(|h| h.score).collect::<Vec<_>>()
        );
    }

    #[test]
    fn keyword_search_says_it_cannot_rather_than_returning_nothing() {
        // The whole point: an empty Vec is indistinguishable from "no document
        // matched", which is how a missing capability turns into a silently
        // worse answer.
        let db = seeded_db();
        let retrieval = VectorDbRetrieval::new(&db);

        let err = retrieval
            .keyword_search("axis", 5)
            .expect_err("a vector store has no lexical index");
        assert!(
            err.contains("with_keyword_search"),
            "the error must name the way out: {err}"
        );
    }

    #[test]
    fn a_delegate_makes_it_a_hybrid_retriever() {
        let db = seeded_db();
        let lexical = LexicalStub;
        let retrieval = VectorDbRetrieval::new(&db).with_keyword_search(&lexical);

        let by_word = retrieval.keyword_search("anything", 2).expect("delegated");
        assert_eq!(by_word.len(), 2);
        assert!(by_word[0].chunk_id.starts_with("lex_"));

        // ...and the vector side still answers from the database.
        let by_vector = retrieval.semantic_search(&[0.0, 1.0, 0.0], 1).expect("vec");
        assert_eq!(by_vector[0].chunk_id, "y");
    }

    #[test]
    fn metadata_the_adapter_does_not_understand_is_carried_through() {
        let db = seeded_db();
        let retrieval = VectorDbRetrieval::new(&db);
        let hits = retrieval.semantic_search(&[1.0, 0.0, 0.0], 1).expect("hit");

        assert_eq!(hits[0].content, "about the X axis");
        assert_eq!(
            hits[0].metadata.get("page").map(String::as_str),
            Some("7"),
            "the store is the caller's; a key this adapter never heard of must survive"
        );
        assert!(
            !hits[0].metadata.contains_key("content"),
            "the text is in `content`, not duplicated into the metadata map"
        );
    }

    #[test]
    fn the_content_key_is_configurable() {
        let config = VectorDbConfig {
            dimensions: 2,
            collection_name: "other".to_string(),
            ..Default::default()
        };
        let mut db = InMemoryVectorDb::new(config);
        db.insert("a", vec![1.0, 0.0], serde_json::json!({"passage": "hello"}))
            .expect("insert");

        let default_key = VectorDbRetrieval::new(&db);
        assert_eq!(
            default_key.semantic_search(&[1.0, 0.0], 1).expect("hit")[0].content,
            "",
            "with the wrong key the text is simply absent, not invented"
        );

        let right_key = VectorDbRetrieval::new(&db).with_content_key("passage");
        assert_eq!(
            right_key.semantic_search(&[1.0, 0.0], 1).expect("hit")[0].content,
            "hello"
        );
    }

    #[test]
    fn get_chunk_fetches_by_id_and_claims_no_score() {
        let db = seeded_db();
        let retrieval = VectorDbRetrieval::new(&db).with_source("papers");

        let found = retrieval.get_chunk("y").expect("lookup").expect("y exists");
        assert_eq!(found.content, "about the Y axis");
        assert_eq!(found.source, "papers");
        assert_eq!(
            found.score, 0.0,
            "a fetch by id measured no similarity; a 1.0 here would be a made-up match"
        );

        assert!(
            retrieval.get_chunk("nope").expect("lookup").is_none(),
            "a missing id is None, not an error"
        );
    }

    #[test]
    fn the_pipeline_retrieves_through_it_end_to_end() {
        // The seam is only worth anything if RagPipeline can actually use it.
        struct FixedEmbedding;
        impl EmbeddingCallback for FixedEmbedding {
            fn embed(&self, _text: &str) -> Result<Vec<f32>, String> {
                Ok(vec![0.0, 0.0, 1.0]) // points at "z"
            }
            fn model_name(&self) -> &str {
                "fixed-3d"
            }
            fn dimension(&self) -> usize {
                3
            }
        }

        let db = seeded_db();
        let lexical = LexicalStub;
        let retrieval = VectorDbRetrieval::new(&db).with_keyword_search(&lexical);

        let mut pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Semantic));
        let result = pipeline
            .process(
                "which axis?",
                &StubLlm,
                Some(&FixedEmbedding),
                &retrieval,
                None,
            )
            .expect("the pipeline must run with a vector-backed retriever");

        assert!(
            !result.chunks.is_empty(),
            "the pipeline retrieved nothing through the adapter"
        );
        assert!(
            result.chunks.iter().any(|c| c.chunk_id == "z"),
            "the Semantic tier must have reached the vector store, not only the              lexical delegate: {:?}",
            result.chunks.iter().map(|c| &c.chunk_id).collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_errorcode_rag_pipeline() {
        use crate::error_taxonomy::ErrorCode;

        let e = RagPipelineError::NoSources;
        assert_eq!(e.code(), "RAG_PIPELINE_NO_SOURCES");
        assert!(e.fields().is_empty());

        let e = RagPipelineError::Timeout;
        assert_eq!(e.code(), "RAG_PIPELINE_TIMEOUT");
        assert!(e.fields().is_empty());

        let e = RagPipelineError::RetrievalError("vector store down".into());
        assert_eq!(e.code(), "RAG_PIPELINE_RETRIEVAL");
        let f = e.fields();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].0, "reason");
        assert_eq!(f[0].1, "vector store down");
    }

    // Mock implementations for testing
    struct MockLlm;
    impl LlmCallback for MockLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
            Ok("Mock response".to_string())
        }
        fn model_name(&self) -> &str {
            "mock-llm"
        }
    }

    struct MockRetrieval;
    impl RetrievalCallback for MockRetrieval {
        fn keyword_search(
            &self,
            _query: &str,
            limit: usize,
        ) -> Result<Vec<RetrievedChunk>, String> {
            Ok((0..limit.min(3))
                .map(|i| RetrievedChunk {
                    chunk_id: format!("chunk_{}", i),
                    content: format!("Test content {}", i),
                    source: "test.md".to_string(),
                    section: None,
                    score: 0.9 - (i as f32 * 0.1),
                    keyword_score: Some(0.9 - (i as f32 * 0.1)),
                    semantic_score: None,
                    token_count: 10,
                    position: None,
                    metadata: HashMap::new(),
                })
                .collect())
        }

        fn semantic_search(
            &self,
            _embedding: &[f32],
            limit: usize,
        ) -> Result<Vec<RetrievedChunk>, String> {
            Ok((0..limit.min(3))
                .map(|i| RetrievedChunk {
                    chunk_id: format!("semantic_{}", i),
                    content: format!("Semantic content {}", i),
                    source: "test.md".to_string(),
                    section: None,
                    score: 0.85 - (i as f32 * 0.1),
                    keyword_score: None,
                    semantic_score: Some(0.85 - (i as f32 * 0.1)),
                    token_count: 15,
                    position: None,
                    metadata: HashMap::new(),
                })
                .collect())
        }

        fn get_chunk(&self, _chunk_id: &str) -> Result<Option<RetrievedChunk>, String> {
            Ok(None)
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let pipeline = RagPipeline::new(config);
        assert_eq!(pipeline.config().rag_config.tier, RagTier::Fast);
    }

    #[test]
    fn test_pipeline_config() {
        let config = RagPipelineConfig::for_tier(RagTier::Enhanced);
        assert_eq!(config.rag_config.tier, RagTier::Enhanced);
        assert!(config.rag_config.effective_features().reranking);
    }

    #[test]
    fn test_process_basic() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);

        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline.process("test query", &llm, None, &retrieval, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.context.is_empty());
        assert!(!result.chunks.is_empty());
    }

    #[test]
    fn test_deduplicate_chunks() {
        let pipeline = RagPipeline::new(RagTierConfig::default());

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "1".to_string(),
                content: "content".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.9,
                keyword_score: Some(0.9),
                semantic_score: None,
                token_count: 10,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "1".to_string(), // Duplicate
                content: "content".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.8, // Lower score
                keyword_score: Some(0.8),
                semantic_score: None,
                token_count: 10,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let deduped = pipeline.deduplicate_chunks(chunks);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].score, 0.9); // Higher score kept
    }

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_tokens("hello world") > 0);
        assert!(estimate_tokens("a longer text with more words") > estimate_tokens("short"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a longer string", 10), "this is...");
    }

    #[test]
    fn test_rrf_fusion() {
        let pipeline = RagPipeline::new(RagTierConfig::default());

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "1".to_string(),
                content: "c1".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.9,
                keyword_score: Some(0.9),
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "2".to_string(),
                content: "c2".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.8,
                keyword_score: None,
                semantic_score: Some(0.8),
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let fused = pipeline.reciprocal_rank_fusion(chunks);
        assert!(!fused.is_empty());
    }

    // ========================================================================
    // Phase 2 (v11): Pipeline config variants
    // ========================================================================

    #[test]
    fn test_from_rag_config() {
        let rag_config = RagTierConfig::with_tier(RagTier::Thorough);
        let pipeline_config = RagPipelineConfig::from_rag_config(rag_config);
        assert_eq!(pipeline_config.rag_config.tier, RagTier::Thorough);
        assert_eq!(pipeline_config.timeout_ms, Some(30000));
        assert_eq!(pipeline_config.llm_timeout_ms, 10000);
        assert!(pipeline_config.continue_on_error);
        assert_eq!(pipeline_config.min_chunks, 1);
    }

    #[test]
    fn test_for_tier_all_variants() {
        let tiers = [
            RagTier::Disabled,
            RagTier::Fast,
            RagTier::Semantic,
            RagTier::Enhanced,
            RagTier::Thorough,
            RagTier::Agentic,
            RagTier::Graph,
            RagTier::Full,
            RagTier::Custom,
        ];
        for tier in &tiers {
            let config = RagPipelineConfig::for_tier(*tier);
            assert_eq!(config.rag_config.tier, *tier);
            assert!(config.enable_caching);
            assert!((config.dedup_threshold - 0.9).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_with_config_custom_values() {
        let mut pipeline_config = RagPipelineConfig::default();
        pipeline_config.timeout_ms = Some(5000);
        pipeline_config.llm_timeout_ms = 2000;
        pipeline_config.continue_on_error = false;
        pipeline_config.min_chunks = 3;
        pipeline_config.max_query_variants = 10;
        pipeline_config.max_sub_queries = 8;
        pipeline_config.rerank_top_k = 20;
        pipeline_config.compression_target_tokens = 500;

        let pipeline = RagPipeline::with_config(pipeline_config);
        assert_eq!(pipeline.config().timeout_ms, Some(5000));
        assert_eq!(pipeline.config().llm_timeout_ms, 2000);
        assert!(!pipeline.config().continue_on_error);
        assert_eq!(pipeline.config().min_chunks, 3);
        assert_eq!(pipeline.config().max_query_variants, 10);
    }

    #[test]
    fn test_pipeline_default_config_values() {
        let config = RagPipelineConfig::default();
        assert_eq!(config.timeout_ms, Some(30000));
        assert_eq!(config.llm_timeout_ms, 10000);
        assert!(config.continue_on_error);
        assert_eq!(config.min_chunks, 1);
        assert!(config.enable_caching);
        assert_eq!(config.max_query_variants, 5);
        assert_eq!(config.max_sub_queries, 4);
        assert_eq!(config.rerank_top_k, 10);
        assert_eq!(config.compression_target_tokens, 200);
        assert_eq!(config.sentence_window_size, 2);
        assert_eq!(config.agentic_max_iterations, 5);
        assert_eq!(config.graph_max_depth, 2);
    }

    // ========================================================================
    // Phase 2 (v11): check_requirements
    // ========================================================================

    #[test]
    fn test_check_requirements_fast_satisfied() {
        let pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Fast));
        let missing = pipeline.check_requirements(false, false, false);
        assert!(
            missing.is_empty(),
            "Fast tier should have no missing requirements"
        );
    }

    #[test]
    fn test_check_requirements_semantic_missing_embeddings() {
        let pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Semantic));
        let missing = pipeline.check_requirements(false, false, false);
        assert!(
            missing
                .iter()
                .any(|r| matches!(r, RagRequirement::EmbeddingModel)),
            "Semantic tier needs embedding model"
        );
    }

    #[test]
    fn test_check_requirements_semantic_satisfied() {
        let pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Semantic));
        let missing = pipeline.check_requirements(true, false, false);
        assert!(!missing
            .iter()
            .any(|r| matches!(r, RagRequirement::EmbeddingModel)),);
    }

    #[test]
    fn test_check_requirements_disabled_empty() {
        let pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Disabled));
        let missing = pipeline.check_requirements(false, false, false);
        assert!(missing.is_empty(), "Disabled tier should need nothing");
    }

    // ========================================================================
    // Phase 2 (v11): Query processing
    // ========================================================================

    #[test]
    fn test_synonym_expansion() {
        let pipeline = RagPipeline::new(RagTierConfig::default());
        let expanded = pipeline.expand_synonyms("How fast is this ship?");
        // "ship" should match synonym entries if any exist
        // The function may or may not find matches depending on the synonym map
        let _ = expanded; // Just verify it doesn't panic
    }

    #[test]
    fn test_synonym_expansion_empty_query() {
        let pipeline = RagPipeline::new(RagTierConfig::default());
        let expanded = pipeline.expand_synonyms("");
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_process_disabled_tier() {
        let config = RagTierConfig::with_tier(RagTier::Disabled);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline.process("test query", &llm, None, &retrieval, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_empty_query() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline.process("", &llm, None, &retrieval, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_enhanced_tier() {
        let config = RagTierConfig::with_tier(RagTier::Enhanced);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline.process("What is the cargo capacity?", &llm, None, &retrieval, None);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.context.is_empty());
    }

    // ========================================================================
    // Phase 2 (v11): Post-processing
    // ========================================================================

    #[test]
    fn test_assemble_context_empty_chunks() {
        let pipeline = RagPipeline::new(RagTierConfig::default());
        let features = pipeline.config().rag_config.effective_features();
        let debug_logger = RagDebugLogger::new(RagDebugConfig::default());
        let session = debug_logger.start_query("test");

        let chunks: Vec<RetrievedChunk> = vec![];
        let (context, used, truncated) = pipeline.assemble_context(&chunks, &features, &session);
        assert!(context.is_empty());
        assert!(used.is_empty());
        assert!(!truncated);
    }

    #[test]
    fn test_assemble_context_all_chunks_fit() {
        let mut rag_config = RagTierConfig::with_tier(RagTier::Fast);
        rag_config.max_knowledge_tokens = 10000;
        let pipeline = RagPipeline::new(rag_config);
        let features = pipeline.config().rag_config.effective_features();
        let debug_logger = RagDebugLogger::new(RagDebugConfig::default());
        let session = debug_logger.start_query("test");

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "c1".to_string(),
                content: "Content A".to_string(),
                source: "doc.md".to_string(),
                section: None,
                score: 0.9,
                keyword_score: None,
                semantic_score: None,
                token_count: 10,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "c2".to_string(),
                content: "Content B".to_string(),
                source: "doc.md".to_string(),
                section: None,
                score: 0.8,
                keyword_score: None,
                semantic_score: None,
                token_count: 10,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let (_context, used, truncated) = pipeline.assemble_context(&chunks, &features, &session);
        assert_eq!(used.len(), 2);
        assert!(!truncated);
    }

    #[test]
    fn test_assemble_context_truncation() {
        let mut rag_config = RagTierConfig::with_tier(RagTier::Fast);
        rag_config.max_knowledge_tokens = 20;
        let pipeline = RagPipeline::new(rag_config);
        let features = pipeline.config().rag_config.effective_features();
        let debug_logger = RagDebugLogger::new(RagDebugConfig::default());
        let session = debug_logger.start_query("test");

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "c1".to_string(),
                content: "First chunk content that is reasonably long".to_string(),
                source: "src1.md".to_string(),
                section: None,
                score: 0.9,
                keyword_score: Some(0.9),
                semantic_score: None,
                token_count: 15,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "c2".to_string(),
                content: "Second chunk content that will exceed the budget".to_string(),
                source: "src2.md".to_string(),
                section: None,
                score: 0.8,
                keyword_score: Some(0.8),
                semantic_score: None,
                token_count: 15,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let (context, used, truncated) = pipeline.assemble_context(&chunks, &features, &session);
        assert_eq!(used.len(), 1);
        assert!(truncated);
        assert!(!context.is_empty());
    }

    #[test]
    fn test_weighted_fusion() {
        let pipeline = RagPipeline::new(RagTierConfig::default());

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "shared".to_string(),
                content: "shared content".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.5,
                keyword_score: Some(0.8),
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "shared".to_string(),
                content: "shared content".to_string(),
                source: "test".to_string(),
                section: None,
                score: 0.5,
                keyword_score: None,
                semantic_score: Some(0.9),
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let fused = pipeline.weighted_fusion(chunks);
        assert_eq!(fused.len(), 1, "Duplicate chunk_ids should merge");
        assert!(fused[0].score > 0.0);
    }

    #[test]
    fn test_deduplicate_keeps_highest_score() {
        let pipeline = RagPipeline::new(RagTierConfig::default());

        let chunks = vec![
            RetrievedChunk {
                chunk_id: "dup".to_string(),
                content: "low".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.3,
                keyword_score: None,
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "dup".to_string(),
                content: "high".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.95,
                keyword_score: None,
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "unique".to_string(),
                content: "only".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.5,
                keyword_score: None,
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let deduped = pipeline.deduplicate_chunks(chunks);
        assert_eq!(deduped.len(), 2);
        let dup_entry = deduped.iter().find(|c| c.chunk_id == "dup").unwrap();
        assert!((dup_entry.score - 0.95).abs() < f32::EPSILON);
    }

    // ========================================================================
    // Phase 2 (v11): Stats tracking
    // ========================================================================

    #[test]
    fn test_stats_initial_state() {
        let pipeline = RagPipeline::new(RagTierConfig::default());
        let stats = pipeline.stats();
        assert_eq!(stats.queries_processed, 0);
        assert_eq!(stats.llm_calls, 0);
        assert_eq!(stats.chunks_retrieved, 0);
        assert_eq!(stats.chunks_used, 0);
    }

    #[test]
    fn test_stats_after_process() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let _ = pipeline.process("test query", &llm, None, &retrieval, None);
        let stats = pipeline.stats();
        assert_eq!(stats.queries_processed, 1);
    }

    #[test]
    fn test_stats_accumulate_across_queries() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let _ = pipeline.process("first query", &llm, None, &retrieval, None);
        let _ = pipeline.process("second query", &llm, None, &retrieval, None);
        let stats = pipeline.stats();
        assert_eq!(stats.queries_processed, 2);
    }

    // ========================================================================
    // Phase 2 (v11): Debug logger + edge cases
    // ========================================================================

    #[test]
    fn test_debug_logger_default() {
        let pipeline = RagPipeline::new(RagTierConfig::default());
        let logger = pipeline.debug_logger();
        let _session = logger.start_query("hello");
    }

    #[test]
    fn test_with_debug_logger_custom() {
        let custom_config = RagDebugConfig {
            enabled: true,
            ..Default::default()
        };
        let custom_logger = Arc::new(RagDebugLogger::new(custom_config));
        let pipeline_config = RagPipelineConfig::default();
        let pipeline = RagPipeline::with_debug_logger(pipeline_config, custom_logger.clone());
        assert!(Arc::ptr_eq(&pipeline.debug_logger(), &custom_logger));
    }

    #[test]
    fn test_count_by_score_type() {
        let chunks = vec![
            RetrievedChunk {
                chunk_id: "kw1".to_string(),
                content: "c".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.9,
                keyword_score: Some(0.9),
                semantic_score: None,
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "sem1".to_string(),
                content: "c".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.8,
                keyword_score: None,
                semantic_score: Some(0.8),
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
            RetrievedChunk {
                chunk_id: "both".to_string(),
                content: "c".to_string(),
                source: "s".to_string(),
                section: None,
                score: 0.7,
                keyword_score: Some(0.7),
                semantic_score: Some(0.6),
                token_count: 5,
                position: None,
                metadata: HashMap::new(),
            },
        ];

        let (kw_count, sem_count) = count_by_score_type(&chunks);
        assert_eq!(kw_count, 2);
        assert_eq!(sem_count, 2);
    }

    #[test]
    fn test_estimate_tokens_edge_cases() {
        let empty_tokens = estimate_tokens("");
        assert!(empty_tokens <= 1);
        let single_char = estimate_tokens("x");
        assert_eq!(single_char, 1);
        let long_text = estimate_tokens("hello world this is a test");
        assert!(long_text > 0);
    }

    #[test]
    fn test_pipeline_error_display() {
        let errors = vec![
            RagPipelineError::NoSources,
            RagPipelineError::Timeout,
            RagPipelineError::QueryProcessingError("bad".to_string()),
            RagPipelineError::RetrievalError("fail".to_string()),
            RagPipelineError::PostProcessingError("post".to_string()),
            RagPipelineError::LlmError("llm".to_string()),
            RagPipelineError::ConfigError("cfg".to_string()),
            RagPipelineError::Internal("int".to_string()),
        ];
        for error in &errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_very_long_query() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let long_query = "a ".repeat(5000);
        let result = pipeline.process(&long_query, &llm, None, &retrieval, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_special_characters_query() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline.process(
            "query with <html> & \"quotes\" and \nnewlines",
            &llm,
            None,
            &retrieval,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_result_has_debug_session_id() {
        let config = RagTierConfig::with_tier(RagTier::Fast);
        let mut pipeline = RagPipeline::new(config);
        let llm = MockLlm;
        let retrieval = MockRetrieval;

        let result = pipeline
            .process("test", &llm, None, &retrieval, None)
            .unwrap();
        assert!(result.debug_session_id.is_some());
        assert!(!result.debug_session_id.unwrap().is_empty());
    }
}

#[cfg(test)]
mod tier_survival_tests {
    //! Every tier that retrieves something must still be holding it at the end.
    //!
    //! `RagTier::Semantic` was not: it is the only tier that fuses with RRF and
    //! has no reranker to rewrite the scores afterwards, so its chunks reached
    //! the `min_relevance_score` floor still carrying RRF's rank-scale values
    //! (~0.03 against a 0.1 floor) and every single one was dropped. "Keyword +
    //! semantic search, better recall" returned strictly less than the keyword-
    //! only tier below it: nothing at all.
    //!
    //! The guard is per-tier and not per-bug on purpose — the next scale
    //! mismatch will land on whichever tier lacks the stage that was masking it.
    use super::*;

    struct Llm;
    impl LlmCallback for Llm {
        fn generate(&self, _p: &str, _m: usize) -> Result<String, String> {
            Ok("answer".to_string())
        }
        fn model_name(&self) -> &str {
            "tier-test-llm"
        }
    }

    struct Embed;
    impl EmbeddingCallback for Embed {
        fn embed(&self, _t: &str) -> Result<Vec<f32>, String> {
            Ok(vec![0.1, 0.2, 0.3])
        }
        fn model_name(&self) -> &str {
            "tier-test-embed"
        }
        fn dimension(&self) -> usize {
            3
        }
    }

    /// Two chunks per method, scored the way a healthy retriever scores them.
    struct Retrieval;
    impl Retrieval {
        fn chunk(id: &str, score: f32, keyword: bool) -> RetrievedChunk {
            RetrievedChunk {
                chunk_id: id.to_string(),
                content: format!("passage {id} about testing and retrieval"),
                source: "doc.md".to_string(),
                section: None,
                score,
                keyword_score: if keyword { Some(score) } else { None },
                semantic_score: if keyword { None } else { Some(score) },
                token_count: 8,
                position: None,
                metadata: HashMap::new(),
            }
        }
    }
    impl RetrievalCallback for Retrieval {
        fn keyword_search(&self, _q: &str, _l: usize) -> Result<Vec<RetrievedChunk>, String> {
            Ok(vec![
                Self::chunk("k1", 0.9, true),
                Self::chunk("k2", 0.7, true),
            ])
        }
        fn semantic_search(&self, _e: &[f32], _l: usize) -> Result<Vec<RetrievedChunk>, String> {
            Ok(vec![
                Self::chunk("s1", 0.8, false),
                Self::chunk("k1", 0.6, false),
            ])
        }
        fn get_chunk(&self, id: &str) -> Result<Option<RetrievedChunk>, String> {
            Ok(Some(Self::chunk(id, 0.5, true)))
        }
    }

    #[test]
    fn no_tier_silently_discards_everything_it_retrieved() {
        for tier in [
            RagTier::Fast,
            RagTier::Semantic,
            RagTier::Enhanced,
            RagTier::Thorough,
            RagTier::Agentic,
            RagTier::Graph,
            RagTier::Full,
        ] {
            let mut pipeline = RagPipeline::new(RagTierConfig::with_tier(tier));
            let result = pipeline
                .process("what is testing?", &Llm, Some(&Embed), &Retrieval, None)
                .unwrap_or_else(|e| panic!("{tier:?} failed: {e:?}"));

            assert!(
                result.stats.chunks_retrieved > 0,
                "{tier:?} retrieved nothing, so the test proves nothing"
            );
            assert!(
                !result.chunks.is_empty(),
                "{tier:?} retrieved {} chunks and kept none",
                result.stats.chunks_retrieved
            );
        }
    }

    #[test]
    fn rrf_output_lands_on_the_same_scale_as_every_other_stage() {
        let pipeline = RagPipeline::new(RagTierConfig::with_tier(RagTier::Semantic));
        let fused = pipeline.reciprocal_rank_fusion(vec![
            Retrieval::chunk("a", 0.9, true),
            Retrieval::chunk("b", 0.7, true),
            Retrieval::chunk("a", 0.8, false),
        ]);

        assert!(!fused.is_empty());
        let top = fused[0].score;
        assert!(
            (top - 1.0).abs() < 1e-6,
            "the best fused chunk must sit at 1.0, not at RRF's raw ~0.03: got {top}"
        );
        assert!(
            fused.iter().all(|c| c.score > 0.0 && c.score <= 1.0),
            "every fused score must be inside 0-1: {:?}",
            fused.iter().map(|c| c.score).collect::<Vec<_>>()
        );
        assert_eq!(
            fused[0].chunk_id, "a",
            "normalising must not disturb the ranking RRF computed"
        );
    }
}

/// The reranker must not cost more than the order.
///
/// Every test here corresponds to a way the old implementation failed without
/// saying anything. It overwrote each score with `1.0 - position / total`,
/// which threw away the relevance that the keyword search, the semantic search
/// and the fusion had all contributed to, and put position in its place.
#[cfg(test)]
mod reranking_keeps_what_retrieval_worked_out {
    use super::*;
    use std::sync::Mutex;

    /// Answers with a fixed string, and remembers what it was asked for.
    struct Scripted {
        answer: String,
        asked_for: Mutex<Vec<usize>>,
        prompts: Mutex<Vec<String>>,
    }

    impl Scripted {
        fn saying(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                asked_for: Mutex::new(Vec::new()),
                prompts: Mutex::new(Vec::new()),
            }
        }
        fn last_budget(&self) -> usize {
            self.asked_for
                .lock()
                .ok()
                .and_then(|a| a.last().copied())
                .unwrap_or(0)
        }
        fn last_prompt(&self) -> String {
            self.prompts
                .lock()
                .ok()
                .and_then(|p| p.last().cloned())
                .unwrap_or_default()
        }
    }

    impl LlmCallback for Scripted {
        fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
            if let Ok(mut a) = self.asked_for.lock() {
                a.push(max_tokens);
            }
            if let Ok(mut p) = self.prompts.lock() {
                p.push(prompt.to_string());
            }
            Ok(self.answer.clone())
        }
        fn model_name(&self) -> &str {
            "scripted"
        }
    }

    fn chunk(id: &str, score: f32) -> RetrievedChunk {
        RetrievedChunk {
            chunk_id: id.to_string(),
            content: format!("passage {id}"),
            source: "doc.md".to_string(),
            section: None,
            score,
            keyword_score: Some(score),
            semantic_score: None,
            token_count: 4,
            position: None,
            metadata: HashMap::new(),
        }
    }

    fn pipeline() -> RagPipeline {
        RagPipeline::new(RagTierConfig::with_tier(RagTier::Enhanced))
    }

    fn scores_of(chunks: &[RetrievedChunk]) -> Vec<f32> {
        chunks.iter().map(|c| c.score).collect()
    }

    fn ids_of(chunks: &[RetrievedChunk]) -> Vec<String> {
        chunks.iter().map(|c| c.chunk_id.clone()).collect()
    }

    #[test]
    fn the_model_changes_the_order_and_nothing_else() {
        let chunks = vec![chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.7)];
        let llm = Scripted::saying("3,1,2");

        let out = pipeline()
            .llm_rerank(&chunks, "a query", &llm)
            .expect("the model answered");

        assert_eq!(
            ids_of(&out.chunks),
            vec!["c", "a", "b"],
            "order not applied"
        );
        // Each chunk keeps the relevance it arrived with. This used to be
        // [1.0, 0.667, 0.333] -- position dressed up as relevance.
        assert_eq!(out.chunks[0].score, 0.7, "c kept its own score");
        assert_eq!(out.chunks[1].score, 0.9, "a kept its own score");
        assert_eq!(out.chunks[2].score, 0.8, "b kept its own score");
        assert_eq!(out.ranked, 3);
        assert!(out.note.is_none(), "{:?}", out.note);
    }

    #[test]
    fn an_answer_with_no_ranking_in_it_costs_the_order_and_nothing_more() {
        // FAILURE 1. The old code let `rankings` come out empty, dropped every
        // chunk into the "not mentioned" branch, gave them all the same score
        // and returned Ok. The caller could not tell that from success, and the
        // ordering that keyword search, semantic search and fusion had produced
        // was gone.
        let chunks = vec![chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.7)];
        for nonsense in [
            "I am not able to rank these passages.",
            "",
            "the third one, then the first",
            "99, 100, 250",
        ] {
            let llm = Scripted::saying(nonsense);
            let out = pipeline()
                .llm_rerank(&chunks, "a query", &llm)
                .expect("an unusable answer is not an error");

            assert_eq!(
                ids_of(&out.chunks),
                ids_of(&chunks),
                "order was disturbed by {nonsense:?}"
            );
            assert_eq!(
                scores_of(&out.chunks),
                scores_of(&chunks),
                "scores were flattened by {nonsense:?}"
            );
            assert_eq!(out.ranked, 0);
            assert!(
                out.note.is_some(),
                "silently did nothing for {nonsense:?} -- that is the defect"
            );
        }
    }

    #[test]
    fn no_score_lands_exactly_on_the_relevance_floor() {
        // FAILURE 2. Unplaced chunks used to be given 0.1, and
        // `min_relevance_score` defaults to 0.1: they survived by exact
        // equality of two independent f32 literals written in different files.
        // Raise the floor a hair and the whole tail disappears.
        let floor = RagTierConfig::with_tier(RagTier::Enhanced).min_relevance_score;
        assert!(
            (floor - 0.1).abs() < f32::EPSILON,
            "the floor moved; this test's premise needs re-reading: {floor}"
        );

        let chunks = vec![chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.45)];
        let llm = Scripted::saying("2");
        let out = pipeline()
            .llm_rerank(&chunks, "a query", &llm)
            .expect("the model answered");

        for c in &out.chunks {
            assert!(
                (c.score - floor).abs() > f32::EPSILON,
                "{} sits exactly on the filter that runs two steps later",
                c.chunk_id
            );
        }
        // And the unplaced ones kept what they had, rather than a sentinel.
        let a = out
            .chunks
            .iter()
            .find(|c| c.chunk_id == "a")
            .expect("a survived");
        assert_eq!(a.score, 0.9);
    }

    #[test]
    fn a_partial_ranking_says_so_and_keeps_the_rest_in_order() {
        let chunks = vec![
            chunk("a", 0.9),
            chunk("b", 0.8),
            chunk("c", 0.7),
            chunk("d", 0.6),
        ];
        let llm = Scripted::saying("4,2");
        let out = pipeline()
            .llm_rerank(&chunks, "a query", &llm)
            .expect("the model answered");

        assert_eq!(ids_of(&out.chunks), vec!["d", "b", "a", "c"]);
        assert_eq!(out.ranked, 2);
        assert!(out.note.is_some(), "a partial ranking passed as a full one");
    }

    #[test]
    fn a_repeated_number_does_not_duplicate_a_passage() {
        let chunks = vec![chunk("a", 0.9), chunk("b", 0.8)];
        let llm = Scripted::saying("2,2,1,2");
        let out = pipeline()
            .llm_rerank(&chunks, "a query", &llm)
            .expect("the model answered");
        assert_eq!(ids_of(&out.chunks), vec!["b", "a"]);
        assert_eq!(out.chunks.len(), 2, "a passage came back twice");
    }

    #[test]
    fn a_long_list_is_not_asked_for_in_one_breath() {
        // FAILURE 3. Every chunk went into the prompt while the answer was
        // capped at a flat 100 tokens. Forty numbers do not fit, so the answer
        // was cut and the whole tail became "not mentioned" -- which, with the
        // old sentinel, also flattened it onto the relevance floor.
        let chunks: Vec<RetrievedChunk> = (0..60)
            .map(|i| chunk(&format!("c{i}"), 0.9 - (i as f32) * 0.01))
            .collect();
        let llm = Scripted::saying("1,2,3");

        let out = pipeline()
            .llm_rerank(&chunks, "a query", &llm)
            .expect("the model answered");

        // Nothing is lost: what is not ranked is not discarded.
        assert_eq!(out.chunks.len(), 60, "chunks went missing");

        // The prompt is bounded.
        let prompt = llm.last_prompt();
        assert!(
            !prompt.contains("passage c25"),
            "a passage past the cap was put to the model"
        );
        assert!(
            prompt.contains("passage c0"),
            "the most relevant passage was not offered"
        );

        // And the answer budget is sized to what was asked, not to a constant
        // that happened to be enough for short lists.
        assert!(
            llm.last_budget() > 100,
            "a twenty-passage ranking was asked for in {} tokens",
            llm.last_budget()
        );
    }

    #[test]
    fn the_answer_budget_grows_with_the_question() {
        let short = Scripted::saying("1");
        pipeline()
            .llm_rerank(&[chunk("a", 0.9)], "q", &short)
            .expect("answered");

        let long_chunks: Vec<RetrievedChunk> =
            (0..15).map(|i| chunk(&format!("c{i}"), 0.5)).collect();
        let long = Scripted::saying("1");
        pipeline()
            .llm_rerank(&long_chunks, "q", &long)
            .expect("answered");

        assert!(
            long.last_budget() > short.last_budget(),
            "fifteen passages were given the same room to answer as one: {} vs {}",
            long.last_budget(),
            short.last_budget()
        );
    }

    #[test]
    fn nothing_in_nothing_out() {
        let llm = Scripted::saying("1,2,3");
        let out = pipeline().llm_rerank(&[], "q", &llm).expect("answered");
        assert!(out.chunks.is_empty());
        assert_eq!(out.ranked, 0);
    }
}
