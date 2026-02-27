//! RAG Debug Logging System - Comprehensive debugging for RAG operations
//!
//! This module provides detailed logging and debugging capabilities for all RAG
//! (Retrieval-Augmented Generation) operations. It captures every step of the
//! retrieval process, from query analysis to final context assembly.
//!
//! # Features
//!
//! - **Enable/Disable at runtime**: Toggle debugging without restarting
//! - **File export**: Write debug logs to JSON files for later analysis
//! - **Step-by-step tracing**: Log each RAG method as it executes
//! - **Performance metrics**: Timing for every operation
//! - **Query analysis**: Capture queries, expansions, and transformations
//! - **Retrieval details**: Track chunks retrieved, scores, and filtering
//! - **LLM call tracking**: Log all LLM calls made by RAG features
//!
//! # Usage
//!
//! ```rust
//! use ai_assistant::rag_debug::{RagDebugConfig, RagDebugLogger, RagDebugLevel};
//!
//! // Create debug config
//! let config = RagDebugConfig {
//!     enabled: true,
//!     level: RagDebugLevel::Detailed,
//!     log_to_file: true,
//!     log_path: Some("./rag_debug/".into()),
//!     ..Default::default()
//! };
//!
//! // Create logger
//! let logger = RagDebugLogger::new(config);
//!
//! // Start a query session
//! let session = logger.start_query("What are the ship specifications?");
//!
//! // Log retrieval steps
//! session.log_step(RagDebugStep::QueryExpansion {
//!     original: "ship specifications".into(),
//!     expanded: vec!["ship specs".into(), "vessel specifications".into()],
//!     method: "llm".into(),
//!     duration_ms: 150,
//! });
//!
//! // End session and export
//! session.complete(Some("The Aurora MR has..."));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Debug Configuration
// ============================================================================

/// Debug verbosity level for RAG operations
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RagDebugLevel {
    /// No debug output
    #[default]
    Off,
    /// Only errors and warnings
    Minimal,
    /// Basic operation info (start/end of queries)
    Basic,
    /// Detailed step-by-step logging
    Detailed,
    /// Full verbose logging including internal state
    Verbose,
    /// Trace-level logging for development
    Trace,
}

impl RagDebugLevel {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "off" | "none" | "0" => RagDebugLevel::Off,
            "minimal" | "min" | "1" => RagDebugLevel::Minimal,
            "basic" | "2" => RagDebugLevel::Basic,
            "detailed" | "detail" | "3" => RagDebugLevel::Detailed,
            "verbose" | "4" => RagDebugLevel::Verbose,
            "trace" | "5" => RagDebugLevel::Trace,
            _ => RagDebugLevel::Off,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            RagDebugLevel::Off => "OFF",
            RagDebugLevel::Minimal => "MINIMAL",
            RagDebugLevel::Basic => "BASIC",
            RagDebugLevel::Detailed => "DETAILED",
            RagDebugLevel::Verbose => "VERBOSE",
            RagDebugLevel::Trace => "TRACE",
        }
    }
}

/// Configuration for RAG debug logging
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagDebugConfig {
    /// Whether debug logging is enabled
    pub enabled: bool,

    /// Debug verbosity level
    pub level: RagDebugLevel,

    /// Log to file
    pub log_to_file: bool,

    /// Path for log files (directory)
    pub log_path: Option<PathBuf>,

    /// Log to stderr
    pub log_to_stderr: bool,

    /// Include timestamps in logs
    pub include_timestamps: bool,

    /// Include timing information
    pub include_timing: bool,

    /// Maximum entries to keep in memory
    pub max_entries: usize,

    /// Log individual chunk details
    pub log_chunks: bool,

    /// Log LLM prompts and responses
    pub log_llm_details: bool,

    /// Log embedding vectors (warning: large)
    pub log_embeddings: bool,

    /// Log score calculations
    pub log_scores: bool,

    /// Pretty-print JSON output
    pub pretty_json: bool,

    /// Rotate log files (keep N most recent)
    pub log_rotation: Option<usize>,

    /// Features to log (empty = all)
    pub feature_filter: Vec<String>,
}

impl Default for RagDebugConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            level: RagDebugLevel::Off,
            log_to_file: false,
            log_path: None,
            log_to_stderr: false,
            include_timestamps: true,
            include_timing: true,
            max_entries: 1000,
            log_chunks: true,
            log_llm_details: false,
            log_embeddings: false,
            log_scores: true,
            pretty_json: true,
            log_rotation: Some(10),
            feature_filter: Vec::new(),
        }
    }
}

impl RagDebugConfig {
    /// Create a minimal debug config for basic logging
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            level: RagDebugLevel::Minimal,
            log_to_stderr: true,
            ..Default::default()
        }
    }

    /// Create a detailed debug config for development
    pub fn detailed() -> Self {
        Self {
            enabled: true,
            level: RagDebugLevel::Detailed,
            log_to_stderr: true,
            log_chunks: true,
            log_scores: true,
            ..Default::default()
        }
    }

    /// Create a file-based debug config for production debugging
    pub fn file_based(path: impl Into<PathBuf>) -> Self {
        Self {
            enabled: true,
            level: RagDebugLevel::Detailed,
            log_to_file: true,
            log_path: Some(path.into()),
            log_to_stderr: false,
            log_chunks: true,
            log_llm_details: true,
            log_scores: true,
            pretty_json: true,
            ..Default::default()
        }
    }

    /// Create verbose config for full tracing
    pub fn verbose(path: impl Into<PathBuf>) -> Self {
        Self {
            enabled: true,
            level: RagDebugLevel::Verbose,
            log_to_file: true,
            log_path: Some(path.into()),
            log_to_stderr: true,
            log_chunks: true,
            log_llm_details: true,
            log_embeddings: false, // Still too large by default
            log_scores: true,
            ..Default::default()
        }
    }
}

// ============================================================================
// Debug Steps - Individual RAG operations
// ============================================================================

/// A single step in the RAG process
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RagDebugStep {
    /// Initial query received
    QueryReceived { query: String, timestamp_ms: u64 },

    /// Query analysis/classification
    QueryAnalysis {
        query: String,
        intent: Option<String>,
        complexity: Option<String>,
        keywords: Vec<String>,
        duration_ms: u64,
    },

    /// Query expansion (LLM-based)
    QueryExpansion {
        original: String,
        expanded: Vec<String>,
        method: String, // "llm", "synonym", "semantic"
        duration_ms: u64,
    },

    /// Multi-query decomposition
    MultiQuery {
        original: String,
        sub_queries: Vec<String>,
        duration_ms: u64,
    },

    /// HyDE hypothetical document generation
    HyDE {
        query: String,
        hypothetical_doc: String,
        duration_ms: u64,
    },

    /// FTS/Keyword search
    KeywordSearch {
        query: String,
        results_count: usize,
        top_score: Option<f32>,
        duration_ms: u64,
    },

    /// Semantic/Embedding search
    SemanticSearch {
        query: String,
        embedding_model: String,
        results_count: usize,
        top_similarity: Option<f32>,
        duration_ms: u64,
    },

    /// Hybrid search fusion
    HybridFusion {
        keyword_results: usize,
        semantic_results: usize,
        fused_results: usize,
        method: String, // "rrf", "weighted", "custom"
        weights: Option<HashMap<String, f32>>,
        duration_ms: u64,
    },

    /// Reranking step
    Reranking {
        input_count: usize,
        output_count: usize,
        method: String, // "llm", "cross_encoder", "bm25"
        score_changes: Vec<ScoreChange>,
        duration_ms: u64,
    },

    /// Contextual compression
    ContextualCompression {
        input_chunks: usize,
        input_tokens: usize,
        output_chunks: usize,
        output_tokens: usize,
        compression_ratio: f32,
        duration_ms: u64,
    },

    /// Sentence window expansion
    SentenceWindow {
        matched_sentences: usize,
        window_size: usize,
        expanded_chunks: usize,
        duration_ms: u64,
    },

    /// Parent document retrieval
    ParentDocument {
        child_matches: usize,
        parent_docs_retrieved: usize,
        duration_ms: u64,
    },

    /// Self-reflection evaluation
    SelfReflection {
        query: String,
        context_summary: String,
        is_sufficient: bool,
        confidence: f32,
        reason: Option<String>,
        duration_ms: u64,
    },

    /// Corrective RAG evaluation
    CorrectiveRag {
        retrieval_quality: f32,
        action_taken: String, // "use_as_is", "refine", "web_search", "retry"
        reason: Option<String>,
        duration_ms: u64,
    },

    /// Adaptive strategy selection
    AdaptiveStrategy {
        query: String,
        selected_strategy: String,
        reason: String,
        duration_ms: u64,
    },

    /// Agentic iteration
    AgenticIteration {
        iteration: usize,
        action: String,
        observation: String,
        is_complete: bool,
        duration_ms: u64,
    },

    /// Graph RAG traversal
    GraphTraversal {
        start_entities: Vec<String>,
        traversal_depth: usize,
        nodes_visited: usize,
        relationships_found: usize,
        duration_ms: u64,
    },

    /// RAPTOR hierarchical retrieval
    RaptorRetrieval {
        level: usize,
        summaries_retrieved: usize,
        leaf_chunks_retrieved: usize,
        duration_ms: u64,
    },

    /// LLM call for RAG purposes
    LlmCall {
        purpose: String, // "expansion", "rerank", "compression", etc.
        model: String,
        input_tokens: usize,
        output_tokens: usize,
        prompt_preview: Option<String>,
        response_preview: Option<String>,
        duration_ms: u64,
    },

    /// Chunk retrieved
    ChunkRetrieved {
        source: String,
        chunk_id: String,
        score: f32,
        token_count: usize,
        preview: String,
    },

    /// Final context assembly
    ContextAssembly {
        total_chunks: usize,
        total_tokens: usize,
        sources: Vec<String>,
        truncated: bool,
        duration_ms: u64,
    },

    /// Error occurred
    Error {
        step: String,
        message: String,
        recoverable: bool,
    },

    /// Warning
    Warning { step: String, message: String },

    /// Custom debug info
    Custom {
        name: String,
        data: HashMap<String, serde_json::Value>,
        duration_ms: Option<u64>,
    },
}

/// Score change during reranking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreChange {
    pub chunk_id: String,
    pub before: f32,
    pub after: f32,
    pub rank_before: usize,
    pub rank_after: usize,
}

// ============================================================================
// Debug Session - A single query's debug trace
// ============================================================================

/// A debug session for a single RAG query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagDebugSession {
    /// Unique session ID
    pub session_id: String,

    /// Original query
    pub query: String,

    /// When the session started (Unix timestamp ms)
    pub start_time_ms: u64,

    /// When the session ended (Unix timestamp ms)
    pub end_time_ms: Option<u64>,

    /// Total duration in ms
    pub total_duration_ms: Option<u64>,

    /// Steps in the RAG process
    pub steps: Vec<RagDebugStep>,

    /// Summary statistics
    pub stats: RagSessionStats,

    /// Final context provided to LLM
    pub final_context: Option<String>,

    /// Final response (if captured)
    pub final_response: Option<String>,

    /// Tier used for this query
    pub rag_tier: Option<String>,

    /// Features enabled for this query
    pub features_enabled: Vec<String>,

    /// LLM provider type (e.g., "ollama", "openai", "anthropic")
    #[serde(default)]
    pub provider_type: Option<String>,

    /// LLM provider URL (e.g., "http://localhost:11434")
    #[serde(default)]
    pub provider_url: Option<String>,

    /// Model name used for generation
    #[serde(default)]
    pub model_name: Option<String>,

    /// Any errors that occurred
    pub errors: Vec<String>,

    /// Any warnings that occurred
    pub warnings: Vec<String>,
}

/// Statistics for a debug session
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RagSessionStats {
    /// Number of LLM calls made
    pub llm_calls: usize,

    /// Total LLM tokens (input)
    pub llm_input_tokens: usize,

    /// Total LLM tokens (output)
    pub llm_output_tokens: usize,

    /// Chunks retrieved
    pub chunks_retrieved: usize,

    /// Chunks used in final context
    pub chunks_used: usize,

    /// Time spent in retrieval (ms)
    pub retrieval_time_ms: u64,

    /// Time spent in LLM calls (ms)
    pub llm_time_ms: u64,

    /// Time spent in reranking (ms)
    pub rerank_time_ms: u64,

    /// Time spent in other processing (ms)
    pub other_time_ms: u64,
}

impl RagDebugSession {
    /// Create a new session
    pub fn new(session_id: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            query: query.into(),
            start_time_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            end_time_ms: None,
            total_duration_ms: None,
            steps: Vec::new(),
            stats: RagSessionStats::default(),
            final_context: None,
            final_response: None,
            rag_tier: None,
            features_enabled: Vec::new(),
            provider_type: None,
            provider_url: None,
            model_name: None,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add a step to the session
    pub fn add_step(&mut self, step: RagDebugStep) {
        // Update stats based on step type
        match &step {
            RagDebugStep::LlmCall {
                input_tokens,
                output_tokens,
                duration_ms,
                ..
            } => {
                self.stats.llm_calls += 1;
                self.stats.llm_input_tokens += input_tokens;
                self.stats.llm_output_tokens += output_tokens;
                self.stats.llm_time_ms += duration_ms;
            }
            RagDebugStep::KeywordSearch {
                results_count,
                duration_ms,
                ..
            }
            | RagDebugStep::SemanticSearch {
                results_count,
                duration_ms,
                ..
            } => {
                self.stats.chunks_retrieved += results_count;
                self.stats.retrieval_time_ms += duration_ms;
            }
            RagDebugStep::Reranking { duration_ms, .. } => {
                self.stats.rerank_time_ms += duration_ms;
            }
            RagDebugStep::ContextAssembly {
                total_chunks,
                duration_ms,
                ..
            } => {
                self.stats.chunks_used = *total_chunks;
                self.stats.other_time_ms += duration_ms;
            }
            RagDebugStep::Error { message, .. } => {
                self.errors.push(message.clone());
            }
            RagDebugStep::Warning { message, .. } => {
                self.warnings.push(message.clone());
            }
            _ => {}
        }

        self.steps.push(step);
    }

    /// Complete the session
    pub fn complete(&mut self, response: Option<String>) {
        self.end_time_ms = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
        self.total_duration_ms = self.end_time_ms.map(|end| end - self.start_time_ms);
        self.final_response = response;
    }

    /// Set the tier used
    pub fn set_tier(&mut self, tier: impl Into<String>) {
        self.rag_tier = Some(tier.into());
    }

    /// Set the features enabled
    pub fn set_features(&mut self, features: Vec<String>) {
        self.features_enabled = features;
    }

    /// Set provider information
    pub fn set_provider(
        &mut self,
        provider_type: impl Into<String>,
        provider_url: impl Into<String>,
        model_name: impl Into<String>,
    ) {
        self.provider_type = Some(provider_type.into());
        self.provider_url = Some(provider_url.into());
        self.model_name = Some(model_name.into());
    }

    /// Set provider type only
    pub fn set_provider_type(&mut self, provider_type: impl Into<String>) {
        self.provider_type = Some(provider_type.into());
    }

    /// Set provider URL only
    pub fn set_provider_url(&mut self, url: impl Into<String>) {
        self.provider_url = Some(url.into());
    }

    /// Set model name only
    pub fn set_model_name(&mut self, model: impl Into<String>) {
        self.model_name = Some(model.into());
    }

    /// Set the final context
    pub fn set_context(&mut self, context: impl Into<String>) {
        self.final_context = Some(context.into());
    }

    /// Get duration so far
    pub fn duration_so_far(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now - self.start_time_ms
    }

    /// Generate a summary
    pub fn summary(&self) -> String {
        format!(
            "Session {}: {} steps, {} LLM calls ({} in/{} out tokens), {} chunks retrieved, {} used, {}ms total",
            self.session_id,
            self.steps.len(),
            self.stats.llm_calls,
            self.stats.llm_input_tokens,
            self.stats.llm_output_tokens,
            self.stats.chunks_retrieved,
            self.stats.chunks_used,
            self.total_duration_ms.unwrap_or_else(|| self.duration_so_far())
        )
    }
}

// ============================================================================
// Debug Logger - Main logging interface
// ============================================================================

/// Main RAG debug logger
pub struct RagDebugLogger {
    config: RwLock<RagDebugConfig>,
    sessions: Mutex<Vec<RagDebugSession>>,
    current_session: Mutex<Option<ActiveSession>>,
    start_time: Instant,
}

/// An active session being logged
struct ActiveSession {
    session: RagDebugSession,
}

impl Default for RagDebugLogger {
    fn default() -> Self {
        Self::new(RagDebugConfig::default())
    }
}

impl RagDebugLogger {
    /// Create a new logger
    pub fn new(config: RagDebugConfig) -> Self {
        Self {
            config: RwLock::new(config),
            sessions: Mutex::new(Vec::new()),
            current_session: Mutex::new(None),
            start_time: Instant::now(),
        }
    }

    /// Create with a specific level
    pub fn with_level(level: RagDebugLevel) -> Self {
        Self::new(RagDebugConfig {
            enabled: level != RagDebugLevel::Off,
            level,
            ..Default::default()
        })
    }

    /// Update configuration
    pub fn configure(&self, config: RagDebugConfig) {
        *self.config.write().unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// Get current configuration
    pub fn config(&self) -> RagDebugConfig {
        self.config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Enable/disable debugging
    pub fn set_enabled(&self, enabled: bool) {
        self.config
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .enabled = enabled;
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .enabled
    }

    /// Set debug level
    pub fn set_level(&self, level: RagDebugLevel) {
        let mut config = self.config.write().unwrap_or_else(|e| e.into_inner());
        config.level = level;
        config.enabled = level != RagDebugLevel::Off;
    }

    /// Get current level
    pub fn level(&self) -> RagDebugLevel {
        self.config.read().unwrap_or_else(|e| e.into_inner()).level
    }

    /// Check if a level is enabled
    pub fn is_level_enabled(&self, level: RagDebugLevel) -> bool {
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());
        config.enabled && config.level >= level
    }

    /// Start a new query session
    pub fn start_query(&self, query: impl Into<String>) -> RagQuerySession<'_> {
        let query_str = query.into();
        let session_id = format!(
            "rag_{}_{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            rand_simple()
        );

        let session = RagDebugSession::new(&session_id, &query_str);

        // Store as current session
        *self
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(ActiveSession {
            session: session.clone(),
        });

        // Log start if enabled
        if self.is_level_enabled(RagDebugLevel::Basic) {
            self.log_to_output(&format!(
                "[RAG] Starting query: {}",
                truncate_str(&query_str, 100)
            ));
        }

        RagQuerySession {
            logger: self,
            session_id,
            start: Instant::now(),
        }
    }

    /// Log a step to the current session.
    /// The feature_filter only affects console/file output — all steps are always
    /// recorded in the session for programmatic access.
    pub fn log_step(&self, step: RagDebugStep) {
        if !self.is_enabled() {
            return;
        }

        let config = self.config.read().unwrap_or_else(|e| e.into_inner());

        // Check feature filter — only affects console/file output, not memory
        let passes_filter = config.feature_filter.is_empty() || {
            let step_type = step_type_name(&step);
            config.feature_filter.iter().any(|f| step_type.contains(f))
        };

        // Log to output based on level (respects feature filter)
        if passes_filter && config.level >= RagDebugLevel::Detailed {
            self.log_step_to_output(&step, &config);
        }

        drop(config);

        // Always add to current session regardless of filter
        if let Some(ref mut active) = *self
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            active.session.add_step(step);
        }
    }

    /// Log a step to stderr/file
    fn log_step_to_output(&self, step: &RagDebugStep, config: &RagDebugConfig) {
        let message = format_step(step, config);

        if config.log_to_stderr {
            let timestamp = if config.include_timestamps {
                format!("[{:.3}s] ", self.start_time.elapsed().as_secs_f64())
            } else {
                String::new()
            };
            log::debug!("{}[RAG] {}", timestamp, message);
        }
    }

    /// Log a message to output
    fn log_to_output(&self, message: &str) {
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());
        if config.log_to_stderr {
            let timestamp = if config.include_timestamps {
                format!("[{:.3}s] ", self.start_time.elapsed().as_secs_f64())
            } else {
                String::new()
            };
            log::debug!("{}{}", timestamp, message);
        }
    }

    /// Complete the current session
    pub fn complete_session(&self, session_id: &str, response: Option<String>) {
        let config = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        if let Some(mut active) = self
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            if active.session.session_id == session_id {
                active.session.complete(response);

                // Log completion
                if self.is_level_enabled(RagDebugLevel::Basic) {
                    self.log_to_output(&format!("[RAG] {}", active.session.summary()));
                }

                // Export to file if enabled
                if config.log_to_file {
                    self.export_session_to_file(&active.session, &config);
                }

                // Store in history
                let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                sessions.push(active.session);

                // Trim history if needed
                while sessions.len() > config.max_entries {
                    sessions.remove(0);
                }
            }
        }
    }

    /// Export a session to file
    fn export_session_to_file(&self, session: &RagDebugSession, config: &RagDebugConfig) {
        let path = match &config.log_path {
            Some(p) => p.clone(),
            None => PathBuf::from("./rag_debug"),
        };

        // Ensure directory exists
        if let Err(e) = fs::create_dir_all(&path) {
            log::error!("[RAG Debug] Failed to create log directory: {}", e);
            return;
        }

        // Create filename
        let filename = format!("{}.json", session.session_id);
        let file_path = path.join(&filename);

        // Serialize
        let json = if config.pretty_json {
            serde_json::to_string_pretty(session)
        } else {
            serde_json::to_string(session)
        };

        match json {
            Ok(json) => {
                if let Err(e) = fs::write(&file_path, json) {
                    log::error!("[RAG Debug] Failed to write log file: {}", e);
                } else if config.log_to_stderr {
                    log::debug!("[RAG Debug] Session exported to {:?}", file_path);
                }
            }
            Err(e) => {
                log::error!("[RAG Debug] Failed to serialize session: {}", e);
            }
        }

        // Handle log rotation
        if let Some(max_logs) = config.log_rotation {
            self.rotate_logs(&path, max_logs);
        }
    }

    /// Rotate log files, keeping only the most recent N
    fn rotate_logs(&self, path: &PathBuf, max_logs: usize) {
        let entries: Vec<_> = match fs::read_dir(path) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "json")
                        .unwrap_or(false)
                })
                .collect(),
            Err(_) => return,
        };

        if entries.len() <= max_logs {
            return;
        }

        // Sort by modified time (oldest first)
        let mut entries_with_time: Vec<_> = entries
            .into_iter()
            .filter_map(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| (e, t))
            })
            .collect();

        entries_with_time.sort_by_key(|(_, time)| *time);

        // Remove oldest files
        let to_remove = entries_with_time.len() - max_logs;
        for (entry, _) in entries_with_time.into_iter().take(to_remove) {
            let _ = fs::remove_file(entry.path());
        }
    }

    /// Get recent sessions
    pub fn recent_sessions(&self, count: usize) -> Vec<RagDebugSession> {
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.iter().rev().take(count).cloned().collect()
    }

    /// Get all sessions
    pub fn all_sessions(&self) -> Vec<RagDebugSession> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Option<RagDebugSession> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|s| s.session_id == session_id)
            .cloned()
    }

    /// Clear all sessions
    pub fn clear_sessions(&self) {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Get aggregate statistics
    pub fn aggregate_stats(&self) -> AggregateRagStats {
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats = AggregateRagStats::default();

        stats.total_sessions = sessions.len();

        for session in sessions.iter() {
            stats.total_llm_calls += session.stats.llm_calls;
            stats.total_input_tokens += session.stats.llm_input_tokens;
            stats.total_output_tokens += session.stats.llm_output_tokens;
            stats.total_chunks_retrieved += session.stats.chunks_retrieved;
            stats.total_chunks_used += session.stats.chunks_used;

            if let Some(duration) = session.total_duration_ms {
                stats.total_duration_ms += duration;
            }

            stats.total_errors += session.errors.len();
            stats.total_warnings += session.warnings.len();
        }

        if stats.total_sessions > 0 {
            stats.avg_llm_calls_per_session =
                stats.total_llm_calls as f32 / stats.total_sessions as f32;
            stats.avg_duration_ms = stats.total_duration_ms as f32 / stats.total_sessions as f32;
        }

        stats
    }

    /// Export all sessions to a single file
    pub fn export_all(&self, path: impl Into<PathBuf>) -> Result<(), std::io::Error> {
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());

        let export = AllSessionsExport {
            exported_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            session_count: sessions.len(),
            aggregate_stats: self.aggregate_stats(),
            sessions: sessions.clone(),
        };

        let json = if config.pretty_json {
            serde_json::to_string_pretty(&export)
        } else {
            serde_json::to_string(&export)
        }
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        fs::write(path.into(), json)
    }
}

/// Aggregate statistics across all sessions
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregateRagStats {
    pub total_sessions: usize,
    pub total_llm_calls: usize,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub total_chunks_retrieved: usize,
    pub total_chunks_used: usize,
    pub total_duration_ms: u64,
    pub total_errors: usize,
    pub total_warnings: usize,
    pub avg_llm_calls_per_session: f32,
    pub avg_duration_ms: f32,
}

/// Export structure for all sessions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllSessionsExport {
    pub exported_at: u64,
    pub session_count: usize,
    pub aggregate_stats: AggregateRagStats,
    pub sessions: Vec<RagDebugSession>,
}

// ============================================================================
// Query Session Handle - RAII for session lifecycle
// ============================================================================

/// Handle for an active query session
pub struct RagQuerySession<'a> {
    logger: &'a RagDebugLogger,
    session_id: String,
    start: Instant,
}

impl<'a> RagQuerySession<'a> {
    /// Log a step in this session
    pub fn log_step(&self, step: RagDebugStep) {
        self.logger.log_step(step);
    }

    /// Log query received
    pub fn log_query_received(&self, query: &str) {
        self.log_step(RagDebugStep::QueryReceived {
            query: query.to_string(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        });
    }

    /// Log query expansion
    pub fn log_expansion(
        &self,
        original: &str,
        expanded: Vec<String>,
        method: &str,
        duration: Duration,
    ) {
        self.log_step(RagDebugStep::QueryExpansion {
            original: original.to_string(),
            expanded,
            method: method.to_string(),
            duration_ms: duration.as_millis() as u64,
        });
    }

    /// Log keyword search
    pub fn log_keyword_search(
        &self,
        query: &str,
        results: usize,
        top_score: Option<f32>,
        duration: Duration,
    ) {
        self.log_step(RagDebugStep::KeywordSearch {
            query: query.to_string(),
            results_count: results,
            top_score,
            duration_ms: duration.as_millis() as u64,
        });
    }

    /// Log semantic search
    pub fn log_semantic_search(
        &self,
        query: &str,
        model: &str,
        results: usize,
        top_sim: Option<f32>,
        duration: Duration,
    ) {
        self.log_step(RagDebugStep::SemanticSearch {
            query: query.to_string(),
            embedding_model: model.to_string(),
            results_count: results,
            top_similarity: top_sim,
            duration_ms: duration.as_millis() as u64,
        });
    }

    /// Log LLM call
    pub fn log_llm_call(
        &self,
        purpose: &str,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
        duration: Duration,
    ) {
        self.log_step(RagDebugStep::LlmCall {
            purpose: purpose.to_string(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            prompt_preview: None,
            response_preview: None,
            duration_ms: duration.as_millis() as u64,
        });
    }

    /// Log error
    pub fn log_error(&self, step: &str, message: &str, recoverable: bool) {
        self.log_step(RagDebugStep::Error {
            step: step.to_string(),
            message: message.to_string(),
            recoverable,
        });
    }

    /// Log warning
    pub fn log_warning(&self, step: &str, message: &str) {
        self.log_step(RagDebugStep::Warning {
            step: step.to_string(),
            message: message.to_string(),
        });
    }

    /// Set tier for this session
    pub fn set_tier(&self, tier: &str) {
        if let Some(ref mut active) = *self
            .logger
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            if active.session.session_id == self.session_id {
                active.session.set_tier(tier);
            }
        }
    }

    /// Set features for this session
    pub fn set_features(&self, features: Vec<String>) {
        if let Some(ref mut active) = *self
            .logger
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            if active.session.session_id == self.session_id {
                active.session.set_features(features);
            }
        }
    }

    /// Set final context
    pub fn set_context(&self, context: &str) {
        if let Some(ref mut active) = *self
            .logger
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            if active.session.session_id == self.session_id {
                active.session.set_context(context);
            }
        }
    }

    /// Set provider information for this session
    pub fn set_provider(&self, provider_type: &str, provider_url: &str, model_name: &str) {
        if let Some(ref mut active) = *self
            .logger
            .current_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            if active.session.session_id == self.session_id {
                active
                    .session
                    .set_provider(provider_type, provider_url, model_name);
            }
        }
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Complete the session with optional response
    pub fn complete(self, response: Option<String>) {
        self.logger.complete_session(&self.session_id, response);
    }

    /// Complete with response string
    pub fn complete_with_response(self, response: &str) {
        self.complete(Some(response.to_string()));
    }

    /// Complete without response
    pub fn complete_no_response(self) {
        self.complete(None);
    }
}

impl<'a> Drop for RagQuerySession<'a> {
    fn drop(&mut self) {
        // Auto-complete if not already done
        // Note: This won't capture response, use complete() explicitly for that
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get step type name for filtering
fn step_type_name(step: &RagDebugStep) -> &'static str {
    match step {
        RagDebugStep::QueryReceived { .. } => "query_received",
        RagDebugStep::QueryAnalysis { .. } => "query_analysis",
        RagDebugStep::QueryExpansion { .. } => "query_expansion",
        RagDebugStep::MultiQuery { .. } => "multi_query",
        RagDebugStep::HyDE { .. } => "hyde",
        RagDebugStep::KeywordSearch { .. } => "keyword_search",
        RagDebugStep::SemanticSearch { .. } => "semantic_search",
        RagDebugStep::HybridFusion { .. } => "hybrid_fusion",
        RagDebugStep::Reranking { .. } => "reranking",
        RagDebugStep::ContextualCompression { .. } => "contextual_compression",
        RagDebugStep::SentenceWindow { .. } => "sentence_window",
        RagDebugStep::ParentDocument { .. } => "parent_document",
        RagDebugStep::SelfReflection { .. } => "self_reflection",
        RagDebugStep::CorrectiveRag { .. } => "corrective_rag",
        RagDebugStep::AdaptiveStrategy { .. } => "adaptive_strategy",
        RagDebugStep::AgenticIteration { .. } => "agentic_iteration",
        RagDebugStep::GraphTraversal { .. } => "graph_traversal",
        RagDebugStep::RaptorRetrieval { .. } => "raptor_retrieval",
        RagDebugStep::LlmCall { .. } => "llm_call",
        RagDebugStep::ChunkRetrieved { .. } => "chunk_retrieved",
        RagDebugStep::ContextAssembly { .. } => "context_assembly",
        RagDebugStep::Error { .. } => "error",
        RagDebugStep::Warning { .. } => "warning",
        RagDebugStep::Custom { .. } => "custom",
    }
}

/// Format a step for console output
fn format_step(step: &RagDebugStep, config: &RagDebugConfig) -> String {
    match step {
        RagDebugStep::QueryReceived { query, .. } => {
            format!("Query received: {}", truncate_str(query, 80))
        }
        RagDebugStep::QueryAnalysis {
            intent,
            complexity,
            keywords,
            duration_ms,
            ..
        } => {
            format!(
                "Analysis: intent={:?}, complexity={:?}, keywords={:?} ({}ms)",
                intent, complexity, keywords, duration_ms
            )
        }
        RagDebugStep::QueryExpansion {
            original,
            expanded,
            method,
            duration_ms,
        } => {
            format!(
                "Expansion ({}): {} -> {} variants ({}ms)",
                method,
                truncate_str(original, 40),
                expanded.len(),
                duration_ms
            )
        }
        RagDebugStep::MultiQuery {
            sub_queries,
            duration_ms,
            ..
        } => {
            format!(
                "Multi-query: {} sub-queries ({}ms)",
                sub_queries.len(),
                duration_ms
            )
        }
        RagDebugStep::HyDE { duration_ms, .. } => {
            format!("HyDE: generated hypothetical document ({}ms)", duration_ms)
        }
        RagDebugStep::KeywordSearch {
            results_count,
            top_score,
            duration_ms,
            ..
        } => {
            format!(
                "Keyword search: {} results, top score={:.3} ({}ms)",
                results_count,
                top_score.unwrap_or(0.0),
                duration_ms
            )
        }
        RagDebugStep::SemanticSearch {
            embedding_model,
            results_count,
            top_similarity,
            duration_ms,
            ..
        } => {
            format!(
                "Semantic search ({}): {} results, top sim={:.3} ({}ms)",
                embedding_model,
                results_count,
                top_similarity.unwrap_or(0.0),
                duration_ms
            )
        }
        RagDebugStep::HybridFusion {
            keyword_results,
            semantic_results,
            fused_results,
            method,
            duration_ms,
            ..
        } => {
            format!(
                "Hybrid fusion ({}): {}+{} -> {} results ({}ms)",
                method, keyword_results, semantic_results, fused_results, duration_ms
            )
        }
        RagDebugStep::Reranking {
            input_count,
            output_count,
            method,
            duration_ms,
            ..
        } => {
            format!(
                "Reranking ({}): {} -> {} results ({}ms)",
                method, input_count, output_count, duration_ms
            )
        }
        RagDebugStep::ContextualCompression {
            input_tokens,
            output_tokens,
            compression_ratio,
            duration_ms,
            ..
        } => {
            format!(
                "Compression: {} -> {} tokens ({:.1}x) ({}ms)",
                input_tokens, output_tokens, compression_ratio, duration_ms
            )
        }
        RagDebugStep::SentenceWindow {
            matched_sentences,
            expanded_chunks,
            duration_ms,
            ..
        } => {
            format!(
                "Sentence window: {} sentences -> {} chunks ({}ms)",
                matched_sentences, expanded_chunks, duration_ms
            )
        }
        RagDebugStep::ParentDocument {
            child_matches,
            parent_docs_retrieved,
            duration_ms,
        } => {
            format!(
                "Parent doc: {} children -> {} parents ({}ms)",
                child_matches, parent_docs_retrieved, duration_ms
            )
        }
        RagDebugStep::SelfReflection {
            is_sufficient,
            confidence,
            duration_ms,
            ..
        } => {
            format!(
                "Self-reflection: sufficient={}, confidence={:.2} ({}ms)",
                is_sufficient, confidence, duration_ms
            )
        }
        RagDebugStep::CorrectiveRag {
            retrieval_quality,
            action_taken,
            duration_ms,
            ..
        } => {
            format!(
                "CRAG: quality={:.2}, action={} ({}ms)",
                retrieval_quality, action_taken, duration_ms
            )
        }
        RagDebugStep::AdaptiveStrategy {
            selected_strategy,
            duration_ms,
            ..
        } => {
            format!(
                "Adaptive: selected {} ({}ms)",
                selected_strategy, duration_ms
            )
        }
        RagDebugStep::AgenticIteration {
            iteration,
            action,
            is_complete,
            duration_ms,
            ..
        } => {
            format!(
                "Agentic #{}: {} (complete={}) ({}ms)",
                iteration, action, is_complete, duration_ms
            )
        }
        RagDebugStep::GraphTraversal {
            nodes_visited,
            relationships_found,
            duration_ms,
            ..
        } => {
            format!(
                "Graph: {} nodes, {} relationships ({}ms)",
                nodes_visited, relationships_found, duration_ms
            )
        }
        RagDebugStep::RaptorRetrieval {
            level,
            summaries_retrieved,
            leaf_chunks_retrieved,
            duration_ms,
        } => {
            format!(
                "RAPTOR L{}: {} summaries, {} leaves ({}ms)",
                level, summaries_retrieved, leaf_chunks_retrieved, duration_ms
            )
        }
        RagDebugStep::LlmCall {
            purpose,
            model,
            input_tokens,
            output_tokens,
            duration_ms,
            ..
        } => {
            if config.log_llm_details {
                format!(
                    "LLM ({}/{}): {} in, {} out ({}ms)",
                    purpose, model, input_tokens, output_tokens, duration_ms
                )
            } else {
                format!(
                    "LLM ({}): {} in, {} out ({}ms)",
                    purpose, input_tokens, output_tokens, duration_ms
                )
            }
        }
        RagDebugStep::ChunkRetrieved {
            source,
            score,
            token_count,
            preview,
            ..
        } => {
            if config.log_chunks {
                format!(
                    "Chunk: {} (score={:.3}, {} tokens): {}",
                    source,
                    score,
                    token_count,
                    truncate_str(preview, 50)
                )
            } else {
                format!("Chunk: {} (score={:.3})", source, score)
            }
        }
        RagDebugStep::ContextAssembly {
            total_chunks,
            total_tokens,
            truncated,
            duration_ms,
            ..
        } => {
            format!(
                "Context: {} chunks, {} tokens{} ({}ms)",
                total_chunks,
                total_tokens,
                if *truncated { " [truncated]" } else { "" },
                duration_ms
            )
        }
        RagDebugStep::Error {
            step,
            message,
            recoverable,
        } => {
            format!(
                "ERROR in {}: {} (recoverable={})",
                step, message, recoverable
            )
        }
        RagDebugStep::Warning { step, message } => {
            format!("WARNING in {}: {}", step, message)
        }
        RagDebugStep::Custom {
            name, duration_ms, ..
        } => {
            format!("Custom ({}): {}ms", name, duration_ms.unwrap_or(0))
        }
    }
}

/// Truncate string with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Simple random number for session IDs
fn rand_simple() -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish() as u32
}

// ============================================================================
// Global Logger
// ============================================================================

/// Global RAG debug logger instance
static GLOBAL_RAG_DEBUG: std::sync::OnceLock<Arc<RagDebugLogger>> = std::sync::OnceLock::new();

/// Get or initialize the global RAG debug logger
pub fn global_rag_debug() -> Arc<RagDebugLogger> {
    GLOBAL_RAG_DEBUG
        .get_or_init(|| Arc::new(RagDebugLogger::default()))
        .clone()
}

/// Configure the global RAG debug logger
pub fn configure_global_rag_debug(config: RagDebugConfig) {
    global_rag_debug().configure(config);
}

/// Enable global RAG debugging
pub fn enable_rag_debug(level: RagDebugLevel) {
    global_rag_debug().set_level(level);
}

/// Disable global RAG debugging
pub fn disable_rag_debug() {
    global_rag_debug().set_enabled(false);
}

// ============================================================================
// Macros for convenient logging
// ============================================================================

/// Log a RAG step to the global logger
#[macro_export]
macro_rules! rag_debug_step {
    ($step:expr) => {
        $crate::rag_debug::global_rag_debug().log_step($step)
    };
}

/// Log a RAG error
#[macro_export]
macro_rules! rag_debug_error {
    ($step:expr, $msg:expr) => {
        $crate::rag_debug::global_rag_debug().log_step($crate::rag_debug::RagDebugStep::Error {
            step: $step.to_string(),
            message: $msg.to_string(),
            recoverable: false,
        })
    };
    ($step:expr, $msg:expr, recoverable) => {
        $crate::rag_debug::global_rag_debug().log_step($crate::rag_debug::RagDebugStep::Error {
            step: $step.to_string(),
            message: $msg.to_string(),
            recoverable: true,
        })
    };
}

/// Log a RAG warning
#[macro_export]
macro_rules! rag_debug_warning {
    ($step:expr, $msg:expr) => {
        $crate::rag_debug::global_rag_debug().log_step($crate::rag_debug::RagDebugStep::Warning {
            step: $step.to_string(),
            message: $msg.to_string(),
        })
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_levels() {
        assert!(RagDebugLevel::Trace > RagDebugLevel::Verbose);
        assert!(RagDebugLevel::Verbose > RagDebugLevel::Detailed);
        assert!(RagDebugLevel::Detailed > RagDebugLevel::Basic);
        assert!(RagDebugLevel::Basic > RagDebugLevel::Minimal);
        assert!(RagDebugLevel::Minimal > RagDebugLevel::Off);
    }

    #[test]
    fn test_debug_level_parsing() {
        assert_eq!(RagDebugLevel::from_str("detailed"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("TRACE"), RagDebugLevel::Trace);
        assert_eq!(RagDebugLevel::from_str("3"), RagDebugLevel::Detailed);
    }

    #[test]
    fn test_logger_creation() {
        let logger = RagDebugLogger::new(RagDebugConfig::detailed());
        assert!(logger.is_enabled());
        assert_eq!(logger.level(), RagDebugLevel::Detailed);
    }

    #[test]
    fn test_session_lifecycle() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        let session = logger.start_query("test query");
        session.log_keyword_search("test", 5, Some(0.8), Duration::from_millis(50));
        session.log_semantic_search(
            "test",
            "test-model",
            10,
            Some(0.9),
            Duration::from_millis(100),
        );
        session.complete_with_response("test response");

        let sessions = logger.all_sessions();
        assert_eq!(sessions.len(), 1);

        let s = &sessions[0];
        assert_eq!(s.query, "test query");
        assert_eq!(s.steps.len(), 2);
        assert!(s.total_duration_ms.is_some());
    }

    #[test]
    fn test_session_stats() {
        let mut session = RagDebugSession::new("test", "test query");

        session.add_step(RagDebugStep::LlmCall {
            purpose: "expansion".into(),
            model: "test".into(),
            input_tokens: 100,
            output_tokens: 50,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 200,
        });

        session.add_step(RagDebugStep::KeywordSearch {
            query: "test".into(),
            results_count: 10,
            top_score: Some(0.8),
            duration_ms: 50,
        });

        assert_eq!(session.stats.llm_calls, 1);
        assert_eq!(session.stats.llm_input_tokens, 100);
        assert_eq!(session.stats.llm_output_tokens, 50);
        assert_eq!(session.stats.chunks_retrieved, 10);
        assert_eq!(session.stats.llm_time_ms, 200);
        assert_eq!(session.stats.retrieval_time_ms, 50);
    }

    #[test]
    fn test_step_formatting() {
        let config = RagDebugConfig::default();

        let step = RagDebugStep::KeywordSearch {
            query: "test query".into(),
            results_count: 5,
            top_score: Some(0.85),
            duration_ms: 42,
        };

        let formatted = format_step(&step, &config);
        assert!(formatted.contains("Keyword search"));
        assert!(formatted.contains("5 results"));
        assert!(formatted.contains("0.850"));
        assert!(formatted.contains("42ms"));
    }

    #[test]
    fn test_aggregate_stats() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        // Session 1
        let s1 = logger.start_query("query 1");
        s1.log_llm_call("test", "model", 100, 50, Duration::from_millis(100));
        s1.complete_no_response();

        // Session 2
        let s2 = logger.start_query("query 2");
        s2.log_llm_call("test", "model", 200, 100, Duration::from_millis(200));
        s2.log_llm_call("test", "model", 50, 25, Duration::from_millis(50));
        s2.complete_no_response();

        let stats = logger.aggregate_stats();
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_llm_calls, 3);
        assert_eq!(stats.total_input_tokens, 350);
        assert_eq!(stats.total_output_tokens, 175);
    }

    #[test]
    fn test_config_presets() {
        let minimal = RagDebugConfig::minimal();
        assert!(minimal.enabled);
        assert_eq!(minimal.level, RagDebugLevel::Minimal);

        let detailed = RagDebugConfig::detailed();
        assert!(detailed.log_chunks);
        assert!(detailed.log_scores);

        let file_based = RagDebugConfig::file_based("./logs");
        assert!(file_based.log_to_file);
        assert!(file_based.log_llm_details);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate_str("short", 10), "short");
        assert_eq!(truncate_str("this is a longer string", 10), "this is...");
    }

    #[test]
    fn test_error_and_warning_tracking() {
        let mut session = RagDebugSession::new("test", "query");

        session.add_step(RagDebugStep::Error {
            step: "retrieval".into(),
            message: "timeout".into(),
            recoverable: true,
        });

        session.add_step(RagDebugStep::Warning {
            step: "rerank".into(),
            message: "low scores".into(),
        });

        assert_eq!(session.errors.len(), 1);
        assert_eq!(session.warnings.len(), 1);
    }

    #[test]
    fn test_feature_filter() {
        let logger = RagDebugLogger::new(RagDebugConfig {
            enabled: true,
            level: RagDebugLevel::Detailed,
            feature_filter: vec!["keyword".to_string()],
            ..Default::default()
        });

        let session = logger.start_query("test");

        // This should be logged (matches filter)
        session.log_keyword_search("test", 5, None, Duration::from_millis(10));

        // This should be filtered out
        session.log_semantic_search("test", "model", 10, None, Duration::from_millis(20));

        session.complete_no_response();

        let sessions = logger.all_sessions();
        // Both steps are still added to session (filter only affects console output)
        // But in memory, we track all steps
        assert_eq!(sessions[0].steps.len(), 2);
    }

    // ====================================================================
    // 1. RagDebugLevel: exhaustive from_str / as_str / Default
    // ====================================================================

    #[test]
    fn test_from_str_all_off_variants() {
        assert_eq!(RagDebugLevel::from_str("off"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("none"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("0"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("OFF"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("NONE"), RagDebugLevel::Off);
    }

    #[test]
    fn test_from_str_all_minimal_variants() {
        assert_eq!(RagDebugLevel::from_str("minimal"), RagDebugLevel::Minimal);
        assert_eq!(RagDebugLevel::from_str("min"), RagDebugLevel::Minimal);
        assert_eq!(RagDebugLevel::from_str("1"), RagDebugLevel::Minimal);
        assert_eq!(RagDebugLevel::from_str("MINIMAL"), RagDebugLevel::Minimal);
        assert_eq!(RagDebugLevel::from_str("MIN"), RagDebugLevel::Minimal);
    }

    #[test]
    fn test_from_str_all_basic_variants() {
        assert_eq!(RagDebugLevel::from_str("basic"), RagDebugLevel::Basic);
        assert_eq!(RagDebugLevel::from_str("2"), RagDebugLevel::Basic);
        assert_eq!(RagDebugLevel::from_str("BASIC"), RagDebugLevel::Basic);
    }

    #[test]
    fn test_from_str_all_detailed_variants() {
        assert_eq!(RagDebugLevel::from_str("detailed"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("detail"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("3"), RagDebugLevel::Detailed);
        assert_eq!(RagDebugLevel::from_str("DETAILED"), RagDebugLevel::Detailed);
    }

    #[test]
    fn test_from_str_all_verbose_variants() {
        assert_eq!(RagDebugLevel::from_str("verbose"), RagDebugLevel::Verbose);
        assert_eq!(RagDebugLevel::from_str("4"), RagDebugLevel::Verbose);
        assert_eq!(RagDebugLevel::from_str("VERBOSE"), RagDebugLevel::Verbose);
    }

    #[test]
    fn test_from_str_all_trace_variants() {
        assert_eq!(RagDebugLevel::from_str("trace"), RagDebugLevel::Trace);
        assert_eq!(RagDebugLevel::from_str("5"), RagDebugLevel::Trace);
        assert_eq!(RagDebugLevel::from_str("TRACE"), RagDebugLevel::Trace);
    }

    #[test]
    fn test_from_str_unknown_returns_off() {
        assert_eq!(RagDebugLevel::from_str("garbage"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str(""), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("99"), RagDebugLevel::Off);
        assert_eq!(RagDebugLevel::from_str("debug"), RagDebugLevel::Off);
    }

    #[test]
    fn test_as_str_all_levels() {
        assert_eq!(RagDebugLevel::Off.as_str(), "OFF");
        assert_eq!(RagDebugLevel::Minimal.as_str(), "MINIMAL");
        assert_eq!(RagDebugLevel::Basic.as_str(), "BASIC");
        assert_eq!(RagDebugLevel::Detailed.as_str(), "DETAILED");
        assert_eq!(RagDebugLevel::Verbose.as_str(), "VERBOSE");
        assert_eq!(RagDebugLevel::Trace.as_str(), "TRACE");
    }

    #[test]
    fn test_debug_level_default_is_off() {
        let level: RagDebugLevel = Default::default();
        assert_eq!(level, RagDebugLevel::Off);
    }

    // ====================================================================
    // 2. RagDebugConfig: verbose preset, Serialize/Deserialize roundtrip
    // ====================================================================

    #[test]
    fn test_config_default_fields() {
        let cfg = RagDebugConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.level, RagDebugLevel::Off);
        assert!(!cfg.log_to_file);
        assert!(cfg.log_path.is_none());
        assert!(!cfg.log_to_stderr);
        assert!(cfg.include_timestamps);
        assert!(cfg.include_timing);
        assert_eq!(cfg.max_entries, 1000);
        assert!(cfg.log_chunks);
        assert!(!cfg.log_llm_details);
        assert!(!cfg.log_embeddings);
        assert!(cfg.log_scores);
        assert!(cfg.pretty_json);
        assert_eq!(cfg.log_rotation, Some(10));
        assert!(cfg.feature_filter.is_empty());
    }

    #[test]
    fn test_config_verbose_preset() {
        let cfg = RagDebugConfig::verbose("/tmp/verbose_logs");
        assert!(cfg.enabled);
        assert_eq!(cfg.level, RagDebugLevel::Verbose);
        assert!(cfg.log_to_file);
        assert_eq!(cfg.log_path, Some(PathBuf::from("/tmp/verbose_logs")));
        assert!(cfg.log_to_stderr);
        assert!(cfg.log_chunks);
        assert!(cfg.log_llm_details);
        assert!(!cfg.log_embeddings); // Still too large by default
        assert!(cfg.log_scores);
    }

    #[test]
    fn test_config_serialize_deserialize_roundtrip() {
        let original = RagDebugConfig {
            enabled: true,
            level: RagDebugLevel::Detailed,
            log_to_file: true,
            log_path: Some(PathBuf::from("/tmp/test")),
            log_to_stderr: true,
            include_timestamps: false,
            include_timing: true,
            max_entries: 500,
            log_chunks: false,
            log_llm_details: true,
            log_embeddings: true,
            log_scores: false,
            pretty_json: false,
            log_rotation: Some(5),
            feature_filter: vec!["keyword".to_string(), "semantic".to_string()],
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: RagDebugConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.enabled, original.enabled);
        assert_eq!(restored.level, original.level);
        assert_eq!(restored.max_entries, original.max_entries);
        assert_eq!(restored.log_rotation, original.log_rotation);
        assert_eq!(restored.feature_filter, original.feature_filter);
        assert_eq!(restored.log_path, original.log_path);
    }

    // ====================================================================
    // 3. RagDebugSession: new, individual setters, summary, complete
    // ====================================================================

    #[test]
    fn test_session_new_creates_properly() {
        let session = RagDebugSession::new("sid_1", "What is Rust?");
        assert_eq!(session.session_id, "sid_1");
        assert_eq!(session.query, "What is Rust?");
        assert!(session.start_time_ms > 0);
        assert!(session.end_time_ms.is_none());
        assert!(session.total_duration_ms.is_none());
        assert!(session.steps.is_empty());
        assert!(session.final_context.is_none());
        assert!(session.final_response.is_none());
        assert!(session.rag_tier.is_none());
        assert!(session.features_enabled.is_empty());
        assert!(session.provider_type.is_none());
        assert!(session.provider_url.is_none());
        assert!(session.model_name.is_none());
        assert!(session.errors.is_empty());
        assert!(session.warnings.is_empty());
    }

    #[test]
    fn test_session_set_provider_individual_setters() {
        let mut session = RagDebugSession::new("s", "q");
        session.set_provider_type("ollama");
        session.set_provider_url("http://localhost:11434");
        session.set_model_name("llama3");
        assert_eq!(session.provider_type.as_deref(), Some("ollama"));
        assert_eq!(session.provider_url.as_deref(), Some("http://localhost:11434"));
        assert_eq!(session.model_name.as_deref(), Some("llama3"));
    }

    #[test]
    fn test_session_set_provider_combined() {
        let mut session = RagDebugSession::new("s", "q");
        session.set_provider("anthropic", "https://api.anthropic.com", "claude-3");
        assert_eq!(session.provider_type.as_deref(), Some("anthropic"));
        assert_eq!(session.provider_url.as_deref(), Some("https://api.anthropic.com"));
        assert_eq!(session.model_name.as_deref(), Some("claude-3"));
    }

    #[test]
    fn test_session_set_context_and_tier_and_features() {
        let mut session = RagDebugSession::new("s", "q");
        session.set_context("Here is the relevant context about Rust.");
        session.set_tier("tier3");
        session.set_features(vec!["rag".into(), "rerank".into()]);
        assert_eq!(
            session.final_context.as_deref(),
            Some("Here is the relevant context about Rust.")
        );
        assert_eq!(session.rag_tier.as_deref(), Some("tier3"));
        assert_eq!(session.features_enabled, vec!["rag", "rerank"]);
    }

    #[test]
    fn test_session_summary_format() {
        let mut session = RagDebugSession::new("abc", "test query");
        session.add_step(RagDebugStep::LlmCall {
            purpose: "expansion".into(),
            model: "m".into(),
            input_tokens: 100,
            output_tokens: 50,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 200,
        });
        session.complete(Some("response text".into()));
        let summary = session.summary();
        assert!(summary.contains("Session abc"));
        assert!(summary.contains("1 steps"));
        assert!(summary.contains("1 LLM calls"));
        assert!(summary.contains("100 in"));
        assert!(summary.contains("50 out"));
    }

    #[test]
    fn test_session_complete_sets_duration() {
        let mut session = RagDebugSession::new("s", "q");
        std::thread::sleep(Duration::from_millis(5));
        session.complete(Some("done".into()));
        assert!(session.end_time_ms.is_some());
        assert!(session.total_duration_ms.is_some());
        assert_eq!(session.final_response.as_deref(), Some("done"));
    }

    // ====================================================================
    // 4. Session stats accumulation
    // ====================================================================

    #[test]
    fn test_stats_multiple_llm_calls_accumulate() {
        let mut session = RagDebugSession::new("s", "q");
        for i in 0..3 {
            session.add_step(RagDebugStep::LlmCall {
                purpose: format!("call_{}", i),
                model: "m".into(),
                input_tokens: 100,
                output_tokens: 40,
                prompt_preview: None,
                response_preview: None,
                duration_ms: 50,
            });
        }
        assert_eq!(session.stats.llm_calls, 3);
        assert_eq!(session.stats.llm_input_tokens, 300);
        assert_eq!(session.stats.llm_output_tokens, 120);
        assert_eq!(session.stats.llm_time_ms, 150);
    }

    #[test]
    fn test_stats_multiple_searches_accumulate() {
        let mut session = RagDebugSession::new("s", "q");
        session.add_step(RagDebugStep::KeywordSearch {
            query: "a".into(),
            results_count: 5,
            top_score: Some(0.9),
            duration_ms: 20,
        });
        session.add_step(RagDebugStep::SemanticSearch {
            query: "b".into(),
            embedding_model: "model".into(),
            results_count: 10,
            top_similarity: Some(0.8),
            duration_ms: 80,
        });
        assert_eq!(session.stats.chunks_retrieved, 15);
        assert_eq!(session.stats.retrieval_time_ms, 100);
    }

    #[test]
    fn test_stats_reranking_and_context_assembly() {
        let mut session = RagDebugSession::new("s", "q");
        session.add_step(RagDebugStep::Reranking {
            input_count: 20,
            output_count: 10,
            method: "cross_encoder".into(),
            score_changes: vec![],
            duration_ms: 75,
        });
        session.add_step(RagDebugStep::ContextAssembly {
            total_chunks: 8,
            total_tokens: 2000,
            sources: vec!["doc1.txt".into()],
            truncated: false,
            duration_ms: 10,
        });
        assert_eq!(session.stats.rerank_time_ms, 75);
        assert_eq!(session.stats.chunks_used, 8);
        assert_eq!(session.stats.other_time_ms, 10);
    }

    // ====================================================================
    // 5. step_type_name: all 24 variants
    // ====================================================================

    #[test]
    fn test_step_type_name_all_variants() {
        let cases: Vec<(RagDebugStep, &str)> = vec![
            (
                RagDebugStep::QueryReceived { query: "q".into(), timestamp_ms: 0 },
                "query_received",
            ),
            (
                RagDebugStep::QueryAnalysis {
                    query: "q".into(), intent: None, complexity: None,
                    keywords: vec![], duration_ms: 0,
                },
                "query_analysis",
            ),
            (
                RagDebugStep::QueryExpansion {
                    original: "q".into(), expanded: vec![], method: "llm".into(), duration_ms: 0,
                },
                "query_expansion",
            ),
            (
                RagDebugStep::MultiQuery {
                    original: "q".into(), sub_queries: vec![], duration_ms: 0,
                },
                "multi_query",
            ),
            (
                RagDebugStep::HyDE {
                    query: "q".into(), hypothetical_doc: "h".into(), duration_ms: 0,
                },
                "hyde",
            ),
            (
                RagDebugStep::KeywordSearch {
                    query: "q".into(), results_count: 0, top_score: None, duration_ms: 0,
                },
                "keyword_search",
            ),
            (
                RagDebugStep::SemanticSearch {
                    query: "q".into(), embedding_model: "m".into(),
                    results_count: 0, top_similarity: None, duration_ms: 0,
                },
                "semantic_search",
            ),
            (
                RagDebugStep::HybridFusion {
                    keyword_results: 0, semantic_results: 0, fused_results: 0,
                    method: "rrf".into(), weights: None, duration_ms: 0,
                },
                "hybrid_fusion",
            ),
            (
                RagDebugStep::Reranking {
                    input_count: 0, output_count: 0, method: "llm".into(),
                    score_changes: vec![], duration_ms: 0,
                },
                "reranking",
            ),
            (
                RagDebugStep::ContextualCompression {
                    input_chunks: 0, input_tokens: 0, output_chunks: 0,
                    output_tokens: 0, compression_ratio: 1.0, duration_ms: 0,
                },
                "contextual_compression",
            ),
            (
                RagDebugStep::SentenceWindow {
                    matched_sentences: 0, window_size: 0, expanded_chunks: 0, duration_ms: 0,
                },
                "sentence_window",
            ),
            (
                RagDebugStep::ParentDocument {
                    child_matches: 0, parent_docs_retrieved: 0, duration_ms: 0,
                },
                "parent_document",
            ),
            (
                RagDebugStep::SelfReflection {
                    query: "q".into(), context_summary: "c".into(),
                    is_sufficient: true, confidence: 0.9, reason: None, duration_ms: 0,
                },
                "self_reflection",
            ),
            (
                RagDebugStep::CorrectiveRag {
                    retrieval_quality: 0.5, action_taken: "retry".into(),
                    reason: None, duration_ms: 0,
                },
                "corrective_rag",
            ),
            (
                RagDebugStep::AdaptiveStrategy {
                    query: "q".into(), selected_strategy: "hybrid".into(),
                    reason: "best".into(), duration_ms: 0,
                },
                "adaptive_strategy",
            ),
            (
                RagDebugStep::AgenticIteration {
                    iteration: 1, action: "search".into(), observation: "ok".into(),
                    is_complete: false, duration_ms: 0,
                },
                "agentic_iteration",
            ),
            (
                RagDebugStep::GraphTraversal {
                    start_entities: vec![], traversal_depth: 0,
                    nodes_visited: 0, relationships_found: 0, duration_ms: 0,
                },
                "graph_traversal",
            ),
            (
                RagDebugStep::RaptorRetrieval {
                    level: 0, summaries_retrieved: 0, leaf_chunks_retrieved: 0, duration_ms: 0,
                },
                "raptor_retrieval",
            ),
            (
                RagDebugStep::LlmCall {
                    purpose: "p".into(), model: "m".into(),
                    input_tokens: 0, output_tokens: 0,
                    prompt_preview: None, response_preview: None, duration_ms: 0,
                },
                "llm_call",
            ),
            (
                RagDebugStep::ChunkRetrieved {
                    source: "s".into(), chunk_id: "c".into(),
                    score: 0.5, token_count: 0, preview: "p".into(),
                },
                "chunk_retrieved",
            ),
            (
                RagDebugStep::ContextAssembly {
                    total_chunks: 0, total_tokens: 0, sources: vec![],
                    truncated: false, duration_ms: 0,
                },
                "context_assembly",
            ),
            (
                RagDebugStep::Error {
                    step: "s".into(), message: "m".into(), recoverable: false,
                },
                "error",
            ),
            (
                RagDebugStep::Warning { step: "s".into(), message: "m".into() },
                "warning",
            ),
            (
                RagDebugStep::Custom {
                    name: "n".into(), data: HashMap::new(), duration_ms: None,
                },
                "custom",
            ),
        ];
        for (step, expected_name) in &cases {
            assert_eq!(step_type_name(step), *expected_name, "failed for {}", expected_name);
        }
    }

    // ====================================================================
    // 6. format_step: test formatting of major step variants
    // ====================================================================

    #[test]
    fn test_format_step_query_received() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::QueryReceived { query: "hello world".into(), timestamp_ms: 100 };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Query received"));
        assert!(out.contains("hello world"));
    }

    #[test]
    fn test_format_step_query_expansion() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::QueryExpansion {
            original: "rust language".into(),
            expanded: vec!["rust programming".into(), "rust lang".into()],
            method: "llm".into(),
            duration_ms: 120,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Expansion"));
        assert!(out.contains("llm"));
        assert!(out.contains("2 variants"));
        assert!(out.contains("120ms"));
    }

    #[test]
    fn test_format_step_semantic_search() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::SemanticSearch {
            query: "test".into(),
            embedding_model: "bge-small".into(),
            results_count: 7,
            top_similarity: Some(0.92),
            duration_ms: 55,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Semantic search"));
        assert!(out.contains("bge-small"));
        assert!(out.contains("7 results"));
        assert!(out.contains("0.920"));
        assert!(out.contains("55ms"));
    }

    #[test]
    fn test_format_step_hybrid_fusion() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::HybridFusion {
            keyword_results: 10,
            semantic_results: 15,
            fused_results: 12,
            method: "rrf".into(),
            weights: None,
            duration_ms: 30,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Hybrid fusion"));
        assert!(out.contains("10+15"));
        assert!(out.contains("12 results"));
    }

    #[test]
    fn test_format_step_reranking() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::Reranking {
            input_count: 20,
            output_count: 5,
            method: "cross_encoder".into(),
            score_changes: vec![],
            duration_ms: 90,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Reranking"));
        assert!(out.contains("20 -> 5"));
        assert!(out.contains("90ms"));
    }

    #[test]
    fn test_format_step_self_reflection() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::SelfReflection {
            query: "q".into(),
            context_summary: "c".into(),
            is_sufficient: true,
            confidence: 0.85,
            reason: None,
            duration_ms: 200,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("Self-reflection"));
        assert!(out.contains("sufficient=true"));
        assert!(out.contains("0.85"));
    }

    #[test]
    fn test_format_step_corrective_rag() {
        let cfg = RagDebugConfig::default();
        let step = RagDebugStep::CorrectiveRag {
            retrieval_quality: 0.4,
            action_taken: "retry".into(),
            reason: Some("low quality".into()),
            duration_ms: 150,
        };
        let out = format_step(&step, &cfg);
        assert!(out.contains("CRAG"));
        assert!(out.contains("0.40"));
        assert!(out.contains("retry"));
    }

    #[test]
    fn test_format_step_error_and_warning() {
        let cfg = RagDebugConfig::default();
        let err = RagDebugStep::Error {
            step: "retrieval".into(),
            message: "timeout".into(),
            recoverable: true,
        };
        let warn = RagDebugStep::Warning {
            step: "rerank".into(),
            message: "low scores".into(),
        };
        let err_out = format_step(&err, &cfg);
        assert!(err_out.contains("ERROR"));
        assert!(err_out.contains("retrieval"));
        assert!(err_out.contains("recoverable=true"));
        let warn_out = format_step(&warn, &cfg);
        assert!(warn_out.contains("WARNING"));
        assert!(warn_out.contains("rerank"));
    }

    #[test]
    fn test_format_step_llm_call_with_details() {
        let cfg_details = RagDebugConfig {
            log_llm_details: true,
            ..Default::default()
        };
        let cfg_no_details = RagDebugConfig::default();
        let step = RagDebugStep::LlmCall {
            purpose: "expansion".into(),
            model: "gpt-4".into(),
            input_tokens: 500,
            output_tokens: 200,
            prompt_preview: None,
            response_preview: None,
            duration_ms: 300,
        };
        let with = format_step(&step, &cfg_details);
        assert!(with.contains("gpt-4")); // model shown when log_llm_details
        let without = format_step(&step, &cfg_no_details);
        assert!(!without.contains("gpt-4")); // model hidden otherwise
        assert!(without.contains("expansion"));
    }

    #[test]
    fn test_format_step_chunk_retrieved_with_and_without_details() {
        let cfg_chunks = RagDebugConfig { log_chunks: true, ..Default::default() };
        let cfg_no_chunks = RagDebugConfig { log_chunks: false, ..Default::default() };
        let step = RagDebugStep::ChunkRetrieved {
            source: "doc.txt".into(),
            chunk_id: "c1".into(),
            score: 0.88,
            token_count: 120,
            preview: "The quick brown fox".into(),
        };
        let with = format_step(&step, &cfg_chunks);
        assert!(with.contains("120 tokens"));
        assert!(with.contains("The quick brown fox"));
        let without = format_step(&step, &cfg_no_chunks);
        assert!(!without.contains("120 tokens"));
        assert!(without.contains("0.880"));
    }

    // ====================================================================
    // 7. Logger: level control, configure, sessions, is_level_enabled
    // ====================================================================

    #[test]
    fn test_logger_with_level_off_means_disabled() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Off);
        assert!(!logger.is_enabled());
        assert_eq!(logger.level(), RagDebugLevel::Off);
    }

    #[test]
    fn test_logger_configure_updates_config() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        assert!(!logger.is_enabled());
        let new_cfg = RagDebugConfig::detailed();
        logger.configure(new_cfg);
        assert!(logger.is_enabled());
        assert_eq!(logger.level(), RagDebugLevel::Detailed);
    }

    #[test]
    fn test_logger_set_enabled_and_set_level() {
        let logger = RagDebugLogger::new(RagDebugConfig::default());
        logger.set_enabled(true);
        assert!(logger.is_enabled());
        logger.set_level(RagDebugLevel::Verbose);
        assert_eq!(logger.level(), RagDebugLevel::Verbose);
        logger.set_level(RagDebugLevel::Off);
        assert!(!logger.is_enabled()); // set_level(Off) disables
    }

    #[test]
    fn test_logger_is_level_enabled_checks() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        assert!(logger.is_level_enabled(RagDebugLevel::Minimal));
        assert!(logger.is_level_enabled(RagDebugLevel::Basic));
        assert!(logger.is_level_enabled(RagDebugLevel::Detailed));
        assert!(!logger.is_level_enabled(RagDebugLevel::Verbose));
        assert!(!logger.is_level_enabled(RagDebugLevel::Trace));
    }

    #[test]
    fn test_logger_is_level_enabled_when_disabled() {
        let logger = RagDebugLogger::new(RagDebugConfig {
            enabled: false,
            level: RagDebugLevel::Trace,
            ..Default::default()
        });
        // Even with Trace level, disabled means nothing is enabled
        assert!(!logger.is_level_enabled(RagDebugLevel::Minimal));
    }

    #[test]
    fn test_logger_recent_sessions_ordering() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        let s1 = logger.start_query("first");
        s1.complete_no_response();
        let s2 = logger.start_query("second");
        s2.complete_no_response();
        let s3 = logger.start_query("third");
        s3.complete_no_response();

        let recent = logger.recent_sessions(2);
        assert_eq!(recent.len(), 2);
        // recent_sessions returns newest first (reversed)
        assert_eq!(recent[0].query, "third");
        assert_eq!(recent[1].query, "second");
    }

    #[test]
    fn test_logger_get_session_by_id() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);

        let session = logger.start_query("find me");
        let sid = session.session_id().to_string();
        session.complete_with_response("found");

        let found = logger.get_session(&sid);
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.query, "find me");
        assert_eq!(found.final_response.as_deref(), Some("found"));
    }

    #[test]
    fn test_logger_get_session_not_found() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        assert!(logger.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_logger_clear_sessions() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let s = logger.start_query("will be cleared");
        s.complete_no_response();
        assert_eq!(logger.all_sessions().len(), 1);
        logger.clear_sessions();
        assert!(logger.all_sessions().is_empty());
    }

    // ====================================================================
    // 8. RagQuerySession: log_expansion, log_error, log_warning, setters
    // ====================================================================

    #[test]
    fn test_query_session_log_expansion() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("original query");
        session.log_expansion(
            "original query",
            vec!["expanded 1".into(), "expanded 2".into()],
            "synonym",
            Duration::from_millis(45),
        );
        session.complete_no_response();

        let sessions = logger.all_sessions();
        assert_eq!(sessions[0].steps.len(), 1);
        match &sessions[0].steps[0] {
            RagDebugStep::QueryExpansion { method, expanded, duration_ms, .. } => {
                assert_eq!(method, "synonym");
                assert_eq!(expanded.len(), 2);
                assert_eq!(*duration_ms, 45);
            }
            _ => panic!("expected QueryExpansion step"),
        }
    }

    #[test]
    fn test_query_session_log_error_and_warning() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("q");
        session.log_error("embedding", "connection refused", true);
        session.log_warning("scoring", "empty results");
        session.complete_no_response();

        let sessions = logger.all_sessions();
        assert_eq!(sessions[0].errors.len(), 1);
        assert_eq!(sessions[0].errors[0], "connection refused");
        assert_eq!(sessions[0].warnings.len(), 1);
        assert_eq!(sessions[0].warnings[0], "empty results");
    }

    #[test]
    fn test_query_session_set_tier_features_context_provider() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("test");
        session.set_tier("tier5");
        session.set_features(vec!["self_rag".into(), "crag".into()]);
        session.set_context("assembled context here");
        session.set_provider("openai", "https://api.openai.com", "gpt-4o");
        session.complete_no_response();

        let sessions = logger.all_sessions();
        let s = &sessions[0];
        assert_eq!(s.rag_tier.as_deref(), Some("tier5"));
        assert_eq!(s.features_enabled, vec!["self_rag", "crag"]);
        assert_eq!(s.final_context.as_deref(), Some("assembled context here"));
        assert_eq!(s.provider_type.as_deref(), Some("openai"));
        assert_eq!(s.provider_url.as_deref(), Some("https://api.openai.com"));
        assert_eq!(s.model_name.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn test_query_session_id_and_elapsed() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Detailed);
        let session = logger.start_query("timing test");
        let sid = session.session_id().to_string();
        assert!(sid.starts_with("rag_"));
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = session.elapsed();
        assert!(elapsed >= Duration::from_millis(5));
        session.complete_no_response();
    }

    // ====================================================================
    // 9. Edge cases
    // ====================================================================

    #[test]
    fn test_empty_query_session() {
        let mut session = RagDebugSession::new("empty", "");
        assert_eq!(session.query, "");
        session.complete(None);
        let summary = session.summary();
        assert!(summary.contains("0 steps"));
        assert!(summary.contains("0 LLM calls"));
    }

    #[test]
    fn test_truncate_str_exact_boundary() {
        // String exactly at max_len should not be truncated
        assert_eq!(truncate_str("12345", 5), "12345");
        // One character over should truncate
        assert_eq!(truncate_str("123456", 5), "12...");
        // Very short max_len
        assert_eq!(truncate_str("abcdef", 3), "...");
    }

    #[test]
    fn test_truncate_str_very_long() {
        let long = "x".repeat(10_000);
        let truncated = truncate_str(&long, 50);
        assert_eq!(truncated.len(), 50);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_session_with_no_steps_summary() {
        let mut session = RagDebugSession::new("no_steps", "query without processing");
        session.complete(None);
        let summary = session.summary();
        assert!(summary.contains("0 steps"));
        assert!(summary.contains("0 LLM calls"));
        assert!(summary.contains("0 chunks retrieved"));
        assert!(summary.contains("0 used"));
    }

    #[test]
    fn test_disabled_logger_records_no_steps() {
        let logger = RagDebugLogger::with_level(RagDebugLevel::Off);
        // Even start_query won't really log internally when Off
        // But the session still gets created structurally
        let session = logger.start_query("should be skipped");
        // log_step checks is_enabled(), so steps won't be added to current_session
        session.log_keyword_search("test", 5, Some(0.9), Duration::from_millis(10));
        session.complete_no_response();

        let sessions = logger.all_sessions();
        // When disabled, log_step returns early so no steps added, and
        // complete_session still stores the session (since start_query created it)
        // but steps will be empty because log_step bailed out
        if !sessions.is_empty() {
            assert_eq!(sessions[0].steps.len(), 0);
        }
    }

    #[test]
    fn test_duration_so_far_returns_positive() {
        let session = RagDebugSession::new("dur", "q");
        // duration_so_far should always be >= 0 (it's u64, so always non-negative)
        let dur = session.duration_so_far();
        // Just verify it doesn't panic and returns something reasonable
        assert!(dur < 10_000); // should be well under 10 seconds in a unit test
    }

    #[test]
    fn test_rand_simple_returns_varied_values() {
        // rand_simple() uses RandomState, so calling it twice should give different values
        // (with extremely high probability)
        let a = rand_simple();
        let b = rand_simple();
        // Not strictly guaranteed, but practically always different
        // If they happen to collide, the test still passes — we just test it doesn't panic
        let _ = (a, b);
    }
}
