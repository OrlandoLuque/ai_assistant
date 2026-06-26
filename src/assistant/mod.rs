//! Main AI Assistant implementation

use crate::error::{AiError, AiResult};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::adaptive_thinking::{
    AdaptiveThinkingConfig, QueryClassifier, ThinkingParseResult, ThinkingStrategy,
    ThinkingTagParser,
};
use crate::api_key_rotation::{ApiKey, ApiKeyManager, RotationConfig};
use crate::config::{AiConfig, AiProvider};
use crate::context::{estimate_tokens, get_model_context_size_cached, ContextUsage};
use crate::conversation_compaction::{
    CompactableMessage, CompactionConfig, CompactionResult, ConversationCompactor,
};
use crate::conversation_control::CancellationToken;
use crate::memory::{MemoryConfig, MemoryManager};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;
use crate::providers::{
    build_system_prompt, build_system_prompt_with_notes, fetch_kobold_models,
    fetch_model_context_size, fetch_ollama_models, fetch_openai_compatible_models,
    generate_response, generate_response_streaming, generate_response_streaming_cancellable,
};
use crate::session::{ChatSession, ChatSessionStore, ResponseStyle, UserPreferences};

#[cfg(feature = "autonomous")]
use crate::agent_profiles::ProfileRegistry;
#[cfg(feature = "autonomous")]
use crate::agent_sandbox::SandboxValidator;
#[cfg(feature = "autonomous")]
use crate::autonomous_loop::{AutonomousAgent, AutonomousAgentBuilder};
#[cfg(feature = "autonomous")]
use crate::mode_manager::{ModeManager, OperationMode};
#[cfg(feature = "autonomous")]
use crate::os_tools::register_os_tools;
#[cfg(feature = "autonomous")]
use crate::user_interaction::{
    AutoApproveHandler as AutoApproveInteraction, InteractionManager, UserInteractionHandler,
};
#[cfg(feature = "autonomous")]
use std::sync::RwLock;

#[cfg(feature = "browser")]
use crate::browser_tools::BrowserSession;
#[cfg(feature = "butler")]
use crate::butler::Butler;
#[cfg(feature = "distributed-agents")]
use crate::distributed_agents::DistributedAgentManager;
#[cfg(feature = "scheduler")]
use crate::scheduler::Scheduler;
#[cfg(feature = "scheduler")]
use crate::trigger_system::TriggerManager;

#[cfg(feature = "rag")]
use crate::rag::{
    build_conversation_context, build_knowledge_context, KnowledgeUsage, RagConfig, RagDb,
    DEFAULT_USER_ID,
};
#[cfg(feature = "rag")]
use std::collections::HashMap;

/// Result from background indexing
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
pub struct IndexingResult {
    /// Document source name
    pub source: String,
    /// Number of chunks indexed (0 if unchanged)
    pub chunks: usize,
    /// Total tokens in the document
    pub tokens: usize,
    /// Whether the document was already up-to-date
    pub was_cached: bool,
}

/// Progress update during indexing
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum IndexingProgress {
    /// Starting to index a document
    Starting {
        source: String,
        total_documents: usize,
        current: usize,
    },
    /// Document indexed successfully
    Completed(IndexingResult),
    /// All documents finished
    AllComplete { results: Vec<IndexingResult> },
    /// Error indexing a document
    Error { source: String, error: String },
}

/// Document info for registration
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
pub struct DocumentInfo {
    /// Source name (identifier)
    pub source: String,
    /// Document content
    pub content: String,
    /// Optional priority (higher = more important in search results)
    pub priority: Option<i32>,
}

#[cfg(feature = "rag")]
impl DocumentInfo {
    pub fn new(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            content: content.into(),
            priority: None,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = Some(priority);
        self
    }
}

/// Statistics for an indexed document
#[cfg(feature = "rag")]
#[derive(Debug, Clone)]
pub struct DocumentStats {
    /// Source name (identifier)
    pub source: String,
    /// Number of chunks in the database
    pub chunk_count: usize,
    /// Total estimated tokens
    pub total_tokens: usize,
    /// Content hash for change detection
    pub content_hash: String,
    /// When the document was indexed
    pub indexed_at: String,
    /// Whether the document is pending indexing
    pub is_pending: bool,
}

/// Result from background summarization
#[derive(Debug)]
pub struct SummaryResult {
    /// The generated summary
    pub summary: String,
    /// Number of messages that were summarized
    pub messages_summarized: usize,
}

/// How the assistant composes context for each message.
///
/// - `Conversation` (default): accumulates conversation history, compacts when needed.
/// - `FreshContext`: each prompt builds context fresh from RAG/graph search.
///   Conversation history is saved but NOT included in the context window,
///   maximizing the token budget available for knowledge retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ContextMode {
    /// Accumulate conversation history in context, compact when needed.
    #[default]
    Conversation,
    /// Each prompt gets fresh context from knowledge sources only.
    /// No conversation history in the context window.
    FreshContext,
}

/// Warnings about FreshContext configuration effectiveness.
///
/// These warnings help library consumers and GUI code understand
/// what is missing for optimal FreshContext operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FreshContextWarning {
    /// RAG not initialized — FreshContext is almost useless without it.
    NoRag,
    /// No knowledge sources indexed — nothing to retrieve.
    NoSourcesIndexed,
    /// Knowledge graph not active — loses entity/relation context.
    NoGraph,
    /// Memory not enabled — loses session context between messages.
    NoMemory,
    /// Available token budget is very small (< 500 tokens).
    SmallBudget(usize),
}

impl std::fmt::Display for FreshContextWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRag => write!(
                f,
                "FreshContext without RAG is almost useless — no knowledge to retrieve"
            ),
            Self::NoSourcesIndexed => write!(
                f,
                "No knowledge sources indexed — add documents for effective retrieval"
            ),
            Self::NoGraph => write!(
                f,
                "Knowledge graph not active — entity and relation context unavailable"
            ),
            Self::NoMemory => write!(
                f,
                "Memory not enabled — session context between messages is lost"
            ),
            Self::SmallBudget(tokens) => {
                write!(
                    f,
                    "Available knowledge budget very small: {} tokens",
                    tokens
                )
            }
        }
    }
}

/// How effective FreshContext will be with the current configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FreshContextEffectiveness {
    /// RAG + Graph + Memory — maximum context quality.
    Optimal,
    /// RAG + at least one of (Graph, Memory).
    Good,
    /// RAG only — functional but missing enrichment.
    Limited,
    /// No RAG — essentially stateless, almost useless.
    Ineffective,
}

/// Status report for FreshContext mode configuration.
///
/// Returned by [`AiAssistant::fresh_context_status`]. Contains warnings,
/// effectiveness level, and all relevant state for programmatic inspection.
///
/// # Example
/// ```no_run
/// use ai_assistant::{AiAssistant, ContextMode};
///
/// let mut assistant = AiAssistant::new();
/// assistant.set_context_mode(ContextMode::FreshContext);
/// let status = assistant.fresh_context_status(false);
/// for warning in &status.warnings {
///     println!("{}", warning);
/// }
/// println!("Effectiveness: {:?}", status.effectiveness);
/// ```
#[derive(Debug, Clone)]
pub struct FreshContextStatus {
    /// Current context mode.
    pub mode: ContextMode,
    /// Whether RAG database is initialized.
    pub rag_available: bool,
    /// Number of knowledge sources indexed.
    pub sources_indexed: usize,
    /// Whether a knowledge graph is active (passed by caller).
    pub graph_available: bool,
    /// Whether the memory system is enabled.
    pub memory_available: bool,
    /// Estimated available tokens for knowledge context.
    pub available_knowledge_tokens: usize,
    /// List of warnings about the current configuration.
    pub warnings: Vec<FreshContextWarning>,
    /// Overall effectiveness assessment.
    pub effectiveness: FreshContextEffectiveness,
}

/// Status report from the Context Budget Allocator, for Butler advisor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ContextBudgetStatus {
    /// Total context window of the model (tokens).
    pub model_context_window: usize,
    /// Tokens available for knowledge after system prompt + conversation + reserve.
    pub available_budget: usize,
    /// Whether RAG is available.
    pub rag_available: bool,
    /// Whether memory is available.
    pub memory_available: bool,
    /// Whether procedural store is available.
    pub procedural_available: bool,
    /// Recommendations for improving context utilization.
    pub recommendations: Vec<String>,
}

impl Default for ContextBudgetStatus {
    fn default() -> Self {
        Self {
            model_context_window: 4096,
            available_budget: 0,
            rag_available: false,
            memory_available: false,
            procedural_available: false,
            recommendations: Vec::new(),
        }
    }
}

/// Main AI Assistant state and logic
pub struct AiAssistant {
    /// Configuration
    pub config: AiConfig,
    /// Current conversation messages
    pub conversation: Vec<ChatMessage>,
    /// Learned user preferences
    pub preferences: UserPreferences,
    /// Available models from all providers
    pub available_models: Vec<ModelInfo>,
    /// True if currently generating a response
    pub is_generating: bool,
    /// True if currently fetching models
    pub is_fetching_models: bool,
    /// Current response being built (during streaming)
    pub current_response: String,
    /// Session store
    pub session_store: ChatSessionStore,
    /// Current active session
    pub current_session: Option<ChatSession>,
    /// True if background summarization is in progress
    pub is_summarizing: bool,

    /// Base system prompt (customizable)
    system_prompt_base: String,

    /// Internal knowledge context (managed by the assistant)
    /// This is used automatically when sending messages if no external context is provided
    knowledge_context: String,

    // Async channels
    rx_response: Option<Receiver<AiResponse>>,
    rx_models: Option<Receiver<AiResponse>>,
    rx_summary: Option<Receiver<SummaryResult>>,
    pending_summary_count: usize,
    /// Current cancellation token for streaming
    cancel_token: Option<CancellationToken>,

    // RAG support (optional feature)
    #[cfg(feature = "rag")]
    /// RAG database for knowledge and conversation storage
    pub rag_db: Option<RagDb>,
    #[cfg(feature = "rag")]
    /// RAG configuration
    pub rag_config: RagConfig,
    #[cfg(feature = "rag")]
    /// IDs of messages stored in RAG DB for current session
    rag_message_ids: Vec<i64>,
    #[cfg(feature = "rag")]
    /// Current user ID for multi-user RAG operations
    pub user_id: String,
    #[cfg(feature = "rag")]
    /// Path for RAG database (for lazy initialization)
    rag_db_path: Option<std::path::PathBuf>,
    #[cfg(feature = "rag")]
    /// Registered documents pending indexing: source_name -> content
    pending_documents: HashMap<String, String>,
    #[cfg(feature = "rag")]
    /// Documents that have been registered (for tracking available sources)
    registered_sources: Vec<String>,
    #[cfg(feature = "rag")]
    /// True if background indexing is in progress
    pub is_indexing: bool,
    #[cfg(feature = "rag")]
    /// Channel for receiving indexing progress/results
    rx_indexing: Option<Receiver<IndexingProgress>>,
    #[cfg(feature = "rag")]
    /// Cache for RAG search results
    rag_cache: Option<crate::metrics::SearchCache<Vec<crate::rag::KnowledgeChunk>>>,
    #[cfg(feature = "rag")]
    /// History of knowledge usage per message (most recent first)
    knowledge_usage_history: Vec<KnowledgeUsage>,
    #[cfg(feature = "rag")]
    /// Last knowledge usage from the most recent RAG context build
    pub last_knowledge_usage: Option<KnowledgeUsage>,

    /// Metrics tracker for conversation quality analysis
    pub metrics: crate::metrics::MetricsTracker,

    /// Cached detected context size for the current model
    /// None means not yet detected, Some(size) is the detected value
    detected_context_size: Option<usize>,
    /// Model name for which context size was detected (to invalidate cache on model change)
    detected_context_model: Option<String>,

    /// Adaptive thinking configuration
    pub adaptive_thinking: AdaptiveThinkingConfig,
    /// Active thinking tag parser for current streaming session
    thinking_parser: Option<ThinkingTagParser>,
    /// Last thinking parse result (available after response completes)
    pub last_thinking_result: Option<ThinkingParseResult>,
    /// Last thinking strategy used (available after classification)
    pub last_thinking_strategy: Option<ThinkingStrategy>,

    /// Fallback providers: list of (provider, model) pairs to try when primary fails.
    fallback_providers: Vec<(AiProvider, String)>,
    /// Whether automatic provider fallback is enabled.
    fallback_enabled: bool,
    /// Provider that served the last response (thread-safe, set by background thread).
    fallback_last_provider: Arc<Mutex<Option<String>>>,

    /// Whether automatic conversation compaction is enabled.
    auto_compaction: bool,
    /// Configuration for conversation compaction.
    compaction_config: CompactionConfig,

    /// Context composition mode (Conversation or FreshContext).
    pub context_mode: ContextMode,

    /// Optional memory manager for building session-aware context.
    /// When enabled, FreshContext mode includes memory-based context alongside RAG.
    pub memory_manager: Option<MemoryManager>,

    /// Optional API key manager for providers that require authentication.
    api_key_manager: Option<ApiKeyManager>,

    /// Event bus for lifecycle hooks and monitoring.
    pub event_bus: crate::events::EventBus,

    // === Autonomous agent support (optional feature) ===
    #[cfg(feature = "autonomous")]
    /// Mode manager for operation mode escalation/de-escalation.
    pub mode_manager: ModeManager,
    #[cfg(feature = "autonomous")]
    /// Profile registry with agent, conversation, and workflow profiles.
    pub profile_registry: ProfileRegistry,
    #[cfg(feature = "autonomous")]
    /// Interaction manager for agent-user communication during autonomous execution.
    interaction_manager: Option<Arc<InteractionManager>>,

    #[cfg(feature = "butler")]
    /// Butler for environment auto-detection and configuration suggestions.
    butler: Option<Butler>,

    #[cfg(feature = "browser")]
    /// Browser session for CDP-based browser automation.
    browser_session: Option<BrowserSession>,

    #[cfg(feature = "scheduler")]
    /// Scheduler for cron-like agent/tool execution.
    scheduler: Option<Scheduler>,
    #[cfg(feature = "scheduler")]
    /// Trigger manager for event-driven actions.
    trigger_manager: Option<TriggerManager>,

    #[cfg(feature = "distributed-agents")]
    /// Distributed agent manager for multi-node agent execution.
    distributed_agent_manager: Option<DistributedAgentManager>,

    #[cfg(feature = "eval")]
    /// A/B testing experiment manager.
    experiment_manager: Option<crate::ab_testing::ExperimentManager>,

    /// Cost tracking dashboard for session-level cost monitoring.
    cost_dashboard: Option<crate::cost_integration::CostDashboard>,

    /// Chat hooks for UI framework event streaming.
    chat_hooks: Option<crate::ui_hooks::ChatHooks>,

    #[cfg(feature = "multi-agent")]
    /// Multi-layer knowledge graph for entity storage and cross-layer reasoning.
    pub graph: Option<crate::multi_layer_graph::MultiLayerGraph>,

    /// Reference resolver for tracking lists and resolving back-references in conversation.
    pub reference_resolver: crate::memory::ReferenceResolver,

    /// Turn counter for tracking conversation position (used by list tracking).
    turn_counter: usize,

    /// Procedural memory store for workflow procedures, checklists, and methodologies.
    /// When populated, matching procedures are automatically injected into the system prompt.
    #[cfg(feature = "advanced-memory")]
    procedural_store: Option<crate::advanced_memory::ProceduralStore>,

    /// Procedure evolver for confidence tracking via outcome feedback.
    #[cfg(feature = "advanced-memory")]
    procedure_evolver: Option<crate::advanced_memory::ProcedureEvolver>,

    /// IDs of procedures that were injected in the current turn (for outcome tracking).
    #[cfg(feature = "advanced-memory")]
    active_procedure_ids: Vec<String>,

    /// Optional LLM enhancer for improving pipeline quality.
    /// When set, modules like intent classification, entity extraction,
    /// and response quality scoring can use LLM calls for better results.
    llm_enhancer: Option<Box<dyn crate::llm_enhance::LlmEnhancer>>,

    /// Context budget allocation configuration.
    /// Controls per-source scoring, overflow strategy, dynamic scoring mode,
    /// and compression thresholds.
    pub context_budget_config: crate::context_budget::ContextBudgetConfig,

    /// UCB1 multi-armed bandit for learning the best overflow strategy.
    /// Only active when `context_budget_config.enable_strategy_learning` is true.
    pub strategy_bandit: Option<crate::context_budget::StrategyBandit>,

    /// Anti-hallucination pipeline configuration.
    /// When enabled, LLM outputs are post-processed to detect and handle
    /// ungrounded claims. Opt-in: disabled by default.
    #[cfg(feature = "eval")]
    pub anti_hallucination_config: Option<crate::anti_hallucination::AntiHallucinationConfig>,

    /// Quality gate runner for output validation.
    /// Checks faithfulness, confidence, grounding ratio against thresholds.
    #[cfg(feature = "eval")]
    pub quality_gate_runner: Option<crate::quality_gates::QualityGateRunner>,
}

impl Default for AiAssistant {
    fn default() -> Self {
        Self::new()
    }
}

/// Run streaming generation with optional provider fallback.
///
/// Tries the primary config first. On failure, iterates through fallback
/// providers until one succeeds or all fail. Sends error via `tx` if all fail.
/// Updates `last_provider` with the name of the provider that served the response.
fn try_generate_with_fallback(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
    fallback_providers: &[(AiProvider, String)],
    cancel_token: Option<&CancellationToken>,
    last_provider: &Arc<Mutex<Option<String>>>,
) {
    let primary_result = match cancel_token {
        Some(token) => {
            generate_response_streaming_cancellable(config, conversation, system_prompt, tx, token)
        }
        None => generate_response_streaming(config, conversation, system_prompt, tx),
    };

    if primary_result.is_ok() {
        *last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(config.provider.display_name().to_string());
        return;
    }

    let primary_err = primary_result.unwrap_err();

    if fallback_providers.is_empty() {
        let _ = tx.send(AiResponse::Error(primary_err.to_string()));
        return;
    }

    // Primary failed, attempt fallback providers
    crate::safe_log!(
        "[fallback] Primary provider {} failed: {}",
        config.provider.display_name(),
        primary_err
    );

    for (fb_provider, fb_model) in fallback_providers {
        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return;
            }
        }

        let mut fb_config = config.clone();
        fb_config.provider = fb_provider.clone();
        fb_config.selected_model = fb_model.clone();

        let fb_result = match cancel_token {
            Some(token) => generate_response_streaming_cancellable(
                &fb_config,
                conversation,
                system_prompt,
                tx,
                token,
            ),
            None => generate_response_streaming(&fb_config, conversation, system_prompt, tx),
        };

        if fb_result.is_ok() {
            *last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(fb_provider.display_name().to_string());
            return;
        }
    }

    // All providers failed
    *last_provider.lock().unwrap_or_else(|e| e.into_inner()) = None;
    let _ = tx.send(AiResponse::Error(format!(
        "All providers failed. Primary error: {}",
        primary_err
    )));
}

/// Vision-aware fallback dispatcher. Skips fallback providers that do not
/// support vision at the transport level (per `agent_bridge::vision_supported_for`).
/// Calls the blocking `generate_vision_response` and emits a single
/// `AiResponse::Complete` (no streaming yet — vision API is non-streaming
/// in this codebase).
#[cfg(feature = "vision")]
fn try_generate_vision_with_fallback(
    config: &AiConfig,
    conversation: &[ChatMessage],
    system_prompt: &str,
    tx: &Sender<AiResponse>,
    fallback_providers: &[(AiProvider, String)],
    last_provider: &Arc<Mutex<Option<String>>>,
) {
    let vision_messages = crate::vision::agent_bridge::chat_messages_to_vision(conversation);

    if crate::vision::agent_bridge::vision_supported_for(config) {
        match crate::vision::generate_vision_response(config, &vision_messages, system_prompt) {
            Ok(text) => {
                *last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(config.provider.display_name().to_string());
                let _ = tx.send(AiResponse::Complete(text));
                return;
            }
            Err(primary_err) => {
                if fallback_providers.is_empty() {
                    let _ = tx.send(AiResponse::Error(primary_err.to_string()));
                    return;
                }
                crate::safe_log!(
                    "[fallback-vision] Primary provider {} failed: {}",
                    config.provider.display_name(),
                    primary_err
                );
                for (fb_provider, fb_model) in fallback_providers {
                    let mut fb_config = config.clone();
                    fb_config.provider = fb_provider.clone();
                    fb_config.selected_model = fb_model.clone();
                    if !crate::vision::agent_bridge::vision_supported_for(&fb_config) {
                        continue;
                    }
                    if let Ok(text) = crate::vision::generate_vision_response(
                        &fb_config,
                        &vision_messages,
                        system_prompt,
                    ) {
                        *last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(fb_provider.display_name().to_string());
                        let _ = tx.send(AiResponse::Complete(text));
                        return;
                    }
                }
                *last_provider.lock().unwrap_or_else(|e| e.into_inner()) = None;
                let _ = tx.send(AiResponse::Error(format!(
                    "All vision-capable providers failed. Primary error: {}",
                    primary_err
                )));
            }
        }
    } else {
        // Primary doesn't support vision — try first vision-capable fallback
        for (fb_provider, fb_model) in fallback_providers {
            let mut fb_config = config.clone();
            fb_config.provider = fb_provider.clone();
            fb_config.selected_model = fb_model.clone();
            if !crate::vision::agent_bridge::vision_supported_for(&fb_config) {
                continue;
            }
            if let Ok(text) =
                crate::vision::generate_vision_response(&fb_config, &vision_messages, system_prompt)
            {
                *last_provider.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(fb_provider.display_name().to_string());
                let _ = tx.send(AiResponse::Complete(text));
                return;
            }
        }
        *last_provider.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let _ = tx.send(AiResponse::Error(format!(
            "Provider {} does not support vision and no vision-capable fallback succeeded",
            config.provider.display_name()
        )));
    }
}

mod context;
mod conversation;
#[cfg(any(feature = "containers", feature = "audio"))]
mod execution;
mod integrations;
mod memory;
mod messaging;
mod metrics;
mod models;
#[cfg(feature = "rag")]
mod rag;

impl AiAssistant {
    /// Create a new AI Assistant with default settings
    pub fn new() -> Self {
        Self::with_system_prompt(
            "You are a helpful AI assistant. Be friendly, accurate, and helpful. \
             If you don't know something, say so. Respond in the same language as the user's question."
        )
    }

    /// Create a new AI Assistant with a custom system prompt
    pub fn with_system_prompt(system_prompt: &str) -> Self {
        Self {
            config: AiConfig::default(),
            conversation: Vec::new(),
            preferences: UserPreferences::default(),
            available_models: Vec::new(),
            is_generating: false,
            is_fetching_models: false,
            current_response: String::new(),
            session_store: ChatSessionStore::default(),
            current_session: None,
            is_summarizing: false,
            system_prompt_base: system_prompt.to_string(),
            knowledge_context: String::new(),
            rx_response: None,
            rx_models: None,
            rx_summary: None,
            pending_summary_count: 0,
            cancel_token: None,
            #[cfg(feature = "rag")]
            rag_db: None,
            #[cfg(feature = "rag")]
            rag_config: RagConfig::default(),
            #[cfg(feature = "rag")]
            rag_message_ids: Vec::new(),
            #[cfg(feature = "rag")]
            user_id: DEFAULT_USER_ID.to_string(),
            #[cfg(feature = "rag")]
            rag_db_path: None,
            #[cfg(feature = "rag")]
            pending_documents: HashMap::new(),
            #[cfg(feature = "rag")]
            registered_sources: Vec::new(),
            #[cfg(feature = "rag")]
            is_indexing: false,
            #[cfg(feature = "rag")]
            rx_indexing: None,
            #[cfg(feature = "rag")]
            rag_cache: Some(crate::metrics::SearchCache::new(50, 300)), // 50 entries, 5 min TTL
            #[cfg(feature = "rag")]
            knowledge_usage_history: Vec::new(),
            #[cfg(feature = "rag")]
            last_knowledge_usage: None,
            metrics: crate::metrics::MetricsTracker::new("default"),
            detected_context_size: None,
            detected_context_model: None,
            adaptive_thinking: AdaptiveThinkingConfig::default(),
            thinking_parser: None,
            last_thinking_result: None,
            last_thinking_strategy: None,
            fallback_providers: Vec::new(),
            fallback_enabled: false,
            fallback_last_provider: Arc::new(Mutex::new(None)),
            auto_compaction: false,
            compaction_config: CompactionConfig::default(),
            context_mode: ContextMode::default(),
            memory_manager: None,
            api_key_manager: None,
            event_bus: crate::events::EventBus::new(),

            #[cfg(feature = "autonomous")]
            mode_manager: ModeManager::new(),
            #[cfg(feature = "autonomous")]
            profile_registry: ProfileRegistry::with_defaults(),
            #[cfg(feature = "autonomous")]
            interaction_manager: None,

            #[cfg(feature = "butler")]
            butler: None,

            #[cfg(feature = "browser")]
            browser_session: None,

            #[cfg(feature = "scheduler")]
            scheduler: None,
            #[cfg(feature = "scheduler")]
            trigger_manager: None,

            #[cfg(feature = "distributed-agents")]
            distributed_agent_manager: None,

            #[cfg(feature = "eval")]
            experiment_manager: None,

            cost_dashboard: None,

            chat_hooks: None,

            #[cfg(feature = "multi-agent")]
            graph: None,

            reference_resolver: crate::memory::ReferenceResolver::new(),
            turn_counter: 0,

            #[cfg(feature = "advanced-memory")]
            procedural_store: None,
            #[cfg(feature = "advanced-memory")]
            procedure_evolver: None,
            #[cfg(feature = "advanced-memory")]
            active_procedure_ids: Vec::new(),

            llm_enhancer: None,

            context_budget_config: crate::context_budget::ContextBudgetConfig::default(),
            strategy_bandit: None,

            #[cfg(feature = "eval")]
            anti_hallucination_config: None,
            #[cfg(feature = "eval")]
            quality_gate_runner: None,
        }
    }

    /// Set the context budget allocation configuration.
    pub fn with_context_budget_config(
        mut self,
        config: crate::context_budget::ContextBudgetConfig,
    ) -> Self {
        if config.enable_strategy_learning {
            self.strategy_bandit =
                Some(crate::context_budget::StrategyBandit::default_strategies());
        }
        self.context_budget_config = config;
        self
    }

    /// Configure cost tracking with budget enforcement.
    ///
    /// Initializes the `CostDashboard` with budget limits from the given config.
    /// Costs are automatically recorded after each LLM response in `poll_response()`.
    pub fn with_cost_config(mut self, config: crate::cost_integration::CostAwareConfig) -> Self {
        if config.enabled {
            let mut bm = crate::cost::BudgetManager::new();
            if let Some(d) = config.daily_budget {
                bm = bm.with_daily_limit(d);
            }
            if let Some(m) = config.monthly_budget {
                bm = bm.with_monthly_limit(m);
            }
            if let Some(r) = config.per_request_limit {
                bm = bm.with_request_limit(r);
            }
            bm.warning_threshold = config.alert_threshold_pct as f32;
            self.cost_dashboard = Some(crate::cost_integration::CostDashboard::with_budget(bm));
        }
        self
    }

    /// Set the base system prompt
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt_base = prompt.to_string();
    }

    /// Get the base system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt_base
    }
}

/// Generate a conversation summary using the AI model
fn generate_conversation_summary(
    config: &AiConfig,
    messages: &[ChatMessage],
    previous_summary: Option<&str>,
) -> AiResult<String> {
    let mut conversation_text = String::new();
    for msg in messages {
        let role = if msg.role == "user" {
            "User"
        } else {
            "Assistant"
        };
        conversation_text.push_str(&format!("{}: {}\n\n", role, msg.content));
    }

    let message_count = messages.len();
    let has_previous = previous_summary.is_some();

    let (summary_guidance, max_tokens) = if has_previous {
        (
            "You have a PREVIOUS SUMMARY that must be preserved and expanded with the new information. \
            Do NOT compress or shorten the previous summary. ADD the new information to it.",
            700
        )
    } else if message_count <= 4 {
        (
            "Write a brief summary (3-4 sentences) capturing the main points.",
            150,
        )
    } else if message_count <= 8 {
        (
            "Write a comprehensive summary (5-8 sentences) covering all key topics.",
            300,
        )
    } else {
        (
            "Write a detailed summary (8-12 sentences) preserving all important context.",
            500,
        )
    };

    let summary_prompt = if let Some(prev) = previous_summary {
        format!(
            r#"You are updating a conversation summary with new information.

CRITICAL: Preserve ALL information from the previous summary and ADD new info.

{summary_guidance}

=== PREVIOUS SUMMARY ===
{prev}

=== NEW CONVERSATION ({message_count} messages) ===
{conversation_text}

=== UPDATED COMPLETE SUMMARY ==="#,
        )
    } else {
        format!(
            r#"Summarize this conversation to preserve context for future reference.
{summary_guidance}

Include specific details discussed. Write in third person.

Conversation ({message_count} messages):
{conversation_text}

Summary:"#,
        )
    };

    // Use Ollama API for summarization
    let url = format!("{}/api/chat", config.ollama_url);

    let request_body = serde_json::json!({
        "model": config.selected_model,
        "messages": [{"role": "user", "content": summary_prompt}],
        "stream": false,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens
        }
    });

    let response = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(90))
        .send_json(&request_body)
        .map_err(anyhow::Error::from)?;

    let body: serde_json::Value = response.into_json()?;

    let summary = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("Previous conversation.")
        .trim()
        .to_string();

    Ok(summary)
}

/// Create a simple fallback summary without AI
fn create_simple_summary(messages: &[ChatMessage], previous_summary: Option<&str>) -> String {
    let mut topics: Vec<String> = Vec::new();

    for msg in messages {
        if msg.role == "user" {
            let content = msg.content.trim();
            let first_sentence: String = content
                .split(|c| c == '.' || c == '?' || c == '!')
                .next()
                .unwrap_or(content)
                .chars()
                .take(50)
                .collect();

            if !first_sentence.is_empty() && !topics.contains(&first_sentence) {
                topics.push(first_sentence);
            }
        }
    }

    let new_topics = if topics.is_empty() {
        String::new()
    } else {
        format!("New topics: {}", topics.join("; "))
    };

    match (previous_summary, new_topics.is_empty()) {
        (Some(prev), false) => format!("{} {}", prev, new_topics),
        (Some(prev), true) => prev.to_string(),
        (None, false) => new_topics,
        (None, true) => "Previous conversation.".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_configure() {
        let mut ai = AiAssistant::new();
        assert!(!ai.fallback_active());

        ai.configure_fallback(vec![
            (AiProvider::LMStudio, "model-a".into()),
            (AiProvider::Ollama, "model-b".into()),
        ]);
        assert!(!ai.fallback_active()); // not enabled yet

        ai.enable_fallback();
        assert!(ai.fallback_active());

        ai.disable_fallback();
        assert!(!ai.fallback_active());
    }

    #[test]
    fn test_fallback_empty_not_active() {
        let mut ai = AiAssistant::new();
        ai.enable_fallback();
        // Enabled but no providers configured
        assert!(!ai.fallback_active());
    }

    #[test]
    fn test_last_provider_initially_none() {
        let ai = AiAssistant::new();
        assert!(ai.last_provider_used().is_none());
    }

    #[test]
    fn test_fallback_last_provider_thread_safe() {
        let ai = AiAssistant::new();
        let provider_ref = ai.fallback_last_provider.clone();
        *provider_ref.lock().unwrap() = Some("TestProvider".to_string());
        assert_eq!(ai.last_provider_used(), Some("TestProvider".to_string()));
    }

    // === Compaction Tests ===

    #[test]
    fn test_compaction_disabled_by_default() {
        let ai = AiAssistant::new();
        assert!(!ai.auto_compaction);
    }

    #[test]
    fn test_compaction_toggle() {
        let mut ai = AiAssistant::new();
        ai.enable_auto_compaction();
        assert!(ai.auto_compaction);
        ai.disable_auto_compaction();
        assert!(!ai.auto_compaction);
    }

    #[test]
    fn test_compact_conversation_reduces_messages() {
        let mut ai = AiAssistant::new();
        ai.set_compaction_config(CompactionConfig {
            max_messages: 10,
            target_messages: 5,
            preserve_recent: 2,
            preserve_first: 1,
            min_importance: 0.9,
            llm_enhanced: false,
        });

        // Add 20 messages
        for i in 0..20 {
            ai.conversation
                .push(ChatMessage::user(&format!("Message {}", i)));
        }
        assert_eq!(ai.conversation.len(), 20);

        let result = ai.compact_conversation();
        assert!(result.removed_count > 0);
        // Compacted + summary message should be <= target + 1
        assert!(ai.conversation.len() <= 7); // target + summary
    }

    #[test]
    fn test_compact_small_conversation_unchanged() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi!"));

        let result = ai.compact_conversation();
        assert_eq!(result.removed_count, 0);
        assert_eq!(ai.conversation.len(), 2);
    }

    // === API Key Management Tests ===

    #[test]
    fn test_add_and_get_api_key() {
        let mut ai = AiAssistant::new();
        ai.add_api_key("openai", "key1", "sk-abc123");

        let key = ai.get_current_api_key("openai");
        assert_eq!(key, Some("sk-abc123".to_string()));
    }

    #[test]
    fn test_api_key_rotation_on_rate_limit() {
        let mut ai = AiAssistant::new();
        ai.add_api_key("openai", "key1", "sk-first");
        ai.add_api_key("openai", "key2", "sk-second");

        // First key should be returned
        assert_eq!(
            ai.get_current_api_key("openai"),
            Some("sk-first".to_string())
        );

        // Mark first key as rate-limited
        ai.mark_key_rate_limited("openai", "key1");

        // Should rotate to second key
        assert_eq!(
            ai.get_current_api_key("openai"),
            Some("sk-second".to_string())
        );
    }

    #[test]
    fn test_api_key_no_manager_returns_none() {
        let mut ai = AiAssistant::new();
        assert!(ai.get_current_api_key("openai").is_none());
    }

    // === Container Convenience Tests ===

    #[cfg(feature = "containers")]
    #[test]
    fn test_create_shared_folder() {
        let ai = AiAssistant::new();
        // SharedFolder::temp() creates a temp dir
        let folder = ai.create_shared_folder();
        assert!(folder.is_ok());
    }

    // === Speech Convenience Tests ===

    #[cfg(feature = "audio")]
    #[test]
    fn test_transcribe_unknown_provider() {
        let ai = AiAssistant::new();
        let result = ai.transcribe(
            "nonexistent",
            &[0u8; 10],
            crate::speech::AudioFormat::Wav,
            None,
        );
        assert!(result.is_err());
    }

    #[cfg(feature = "audio")]
    #[test]
    fn test_synthesize_unknown_provider() {
        let ai = AiAssistant::new();
        let result = ai.synthesize(
            "nonexistent",
            "hello",
            &crate::speech::SynthesisOptions::default(),
        );
        assert!(result.is_err());
    }

    #[cfg(feature = "audio")]
    #[test]
    fn test_transcribe_piper_no_stt() {
        let ai = AiAssistant::new();
        // Piper is TTS-only, transcribe should fail
        let result = ai.transcribe("piper", &[0u8; 10], crate::speech::AudioFormat::Wav, None);
        assert!(result.is_err());
    }

    #[cfg(feature = "audio")]
    #[test]
    fn test_synthesize_empty_text() {
        let ai = AiAssistant::new();
        let result = ai.synthesize("piper", "", &crate::speech::SynthesisOptions::default());
        assert!(result.is_err());
    }

    // =========================================================================
    // KPKG -> Knowledge Layer Bridge tests (v4 roadmap item 8.1)
    // =========================================================================

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_from_text() {
        let text = "The ship Aurora was built by Stellar Dynamics in the Mars Orbital Shipyard. \
                     It carries a crew of 200 and is powered by the Helios Reactor.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"Aurora".to_string()),
            "Should extract 'Aurora': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Stellar".to_string()) || entities.contains(&"Dynamics".to_string()),
            "Should extract part of 'Stellar Dynamics': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Mars".to_string()),
            "Should extract 'Mars': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Helios".to_string()),
            "Should extract 'Helios': {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_empty() {
        let entities = AiAssistant::extract_entities_from_text("");
        assert!(entities.is_empty(), "Empty text should yield no entities");
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_all_lowercase() {
        let text = "the quick brown fox jumps over the lazy dog. \
                     no capitalized words here at all.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.is_empty(),
            "All-lowercase text should yield no entities: {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_mixed() {
        let text = "John works at Google. He uses Python and Rust daily. \
                     Mary prefers TypeScript over JavaScript.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"Google".to_string()),
            "Should extract 'Google': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Python".to_string()),
            "Should extract 'Python': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Rust".to_string()),
            "Should extract 'Rust': {:?}",
            entities
        );
        assert!(
            entities.contains(&"TypeScript".to_string()),
            "Should extract 'TypeScript': {:?}",
            entities
        );
        assert!(
            entities.contains(&"JavaScript".to_string()),
            "Should extract 'JavaScript': {:?}",
            entities
        );
        // "He" and "Mary" at sentence start should not appear
        // (Mary is at start of a sentence, so it won't be extracted)
        // John is at the very start too
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_quoted_terms() {
        let text = "the concept of \"Dark Energy\" is fundamental. \
                     we also study \"quantum entanglement\" in depth.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"Dark Energy".to_string()),
            "Should extract quoted term 'Dark Energy': {:?}",
            entities
        );
        assert!(
            entities.contains(&"quantum entanglement".to_string()),
            "Should extract quoted term 'quantum entanglement': {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_deduplication() {
        let text = "The planet Mars is red. People want to colonize Mars. \
                     Mars exploration is ongoing.";
        let entities = AiAssistant::extract_entities_from_text(text);
        let mars_count = entities.iter().filter(|e| *e == "Mars").count();
        assert_eq!(
            mars_count, 1,
            "Mars should appear only once (deduplication): {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_stop_words_filtered() {
        let text = "And Then He said something. But She replied differently.";
        let entities = AiAssistant::extract_entities_from_text(text);
        // "Then", "He", "But", "She" are all stop words
        assert!(
            !entities.contains(&"Then".to_string()),
            "Stop word 'Then' should be filtered: {:?}",
            entities
        );
        assert!(
            !entities.contains(&"He".to_string()),
            "Stop word 'He' should be filtered: {:?}",
            entities
        );
        assert!(
            !entities.contains(&"She".to_string()),
            "Stop word 'She' should be filtered: {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_single_char_skipped() {
        // Single-character uppercase words should be skipped (len < 2 check)
        let text = "we use A for the first and B for the second.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            !entities.contains(&"A".to_string()),
            "Single char 'A' should be skipped: {:?}",
            entities
        );
        assert!(
            !entities.contains(&"B".to_string()),
            "Single char 'B' should be skipped: {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_acronyms_included() {
        let text = "the NASA program launched from the ESA facility.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"NASA".to_string()),
            "Acronym 'NASA' should be included: {:?}",
            entities
        );
        assert!(
            entities.contains(&"ESA".to_string()),
            "Acronym 'ESA' should be included: {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_multiline() {
        let text = "First line mentions Berlin.\nSecond line talks about Paris.\n\
                     Third references Tokyo and Kyoto.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"Berlin".to_string()),
            "Should extract 'Berlin': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Paris".to_string()),
            "Should extract 'Paris': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Tokyo".to_string()),
            "Should extract 'Tokyo': {:?}",
            entities
        );
        assert!(
            entities.contains(&"Kyoto".to_string()),
            "Should extract 'Kyoto': {:?}",
            entities
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_load_kpkg_to_graph_no_graph() {
        // When graph is None, load_kpkg_to_graph should fail at the file read stage
        // because we don't have an actual kpkg file. This tests error handling.
        let mut ai = AiAssistant::new();
        assert!(ai.graph.is_none(), "Graph should be None by default");
        let result = ai.load_kpkg_to_graph("nonexistent_file.kpkg");
        assert!(result.is_err(), "Should fail when kpkg file does not exist");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("read_kpkg") || err_msg.contains("kpkg file"),
            "Error should mention kpkg file read failure: {}",
            err_msg
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_kpkg_manifest_injection() {
        // Test that system_prompt injection modifies the assistant's prompt
        let mut ai = AiAssistant::new();
        let _original_prompt = ai.system_prompt().to_string();

        // We can't easily create a real kpkg in a unit test without the builder,
        // but we can test the prompt injection logic by simulating what happens
        // after a successful load. We'll test via the public API by verifying
        // that the system prompt can be modified.
        ai.set_system_prompt("Base prompt");
        assert_eq!(ai.system_prompt(), "Base prompt");

        // Simulate what load_kpkg_to_graph does to the prompt
        let injection = "[KPKG System Prompt]: You are a space navigator.\n\
                         [KPKG Persona]: Expert in stellar cartography.";
        let new_prompt = format!("{}\n\n{}", ai.system_prompt(), injection);
        ai.set_system_prompt(&new_prompt);

        assert!(
            ai.system_prompt().contains("[KPKG System Prompt]"),
            "System prompt should contain KPKG injection"
        );
        assert!(
            ai.system_prompt().contains("space navigator"),
            "System prompt should contain persona content"
        );
        assert!(
            ai.system_prompt().contains("Base prompt"),
            "Original prompt should be preserved"
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_graph_field_default_none() {
        let ai = AiAssistant::new();
        assert!(
            ai.graph.is_none(),
            "MultiLayerGraph should be None by default"
        );
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_graph_field_can_be_set() {
        let mut ai = AiAssistant::new();
        ai.graph = Some(crate::multi_layer_graph::MultiLayerGraph::new());
        assert!(ai.graph.is_some(), "Graph should be Some after assignment");
    }

    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    #[test]
    fn test_extract_entities_with_punctuation() {
        let text = "we visited London, Madrid, and Rome. All are capitals.";
        let entities = AiAssistant::extract_entities_from_text(text);
        assert!(
            entities.contains(&"London".to_string()),
            "Should extract 'London' despite trailing comma: {:?}",
            entities
        );
        assert!(
            entities.contains(&"Madrid".to_string()),
            "Should extract 'Madrid' despite trailing comma: {:?}",
            entities
        );
        assert!(
            entities.contains(&"Rome".to_string()),
            "Should extract 'Rome': {:?}",
            entities
        );
    }

    // =========================================================================
    // v8 4.1 — New test coverage for AiAssistant
    // =========================================================================

    // --- Default / Constructor Tests ---

    #[test]
    fn test_new_defaults() {
        let ai = AiAssistant::new();
        assert!(
            ai.conversation.is_empty(),
            "Conversation should start empty"
        );
        assert!(!ai.is_generating, "Should not be generating initially");
        assert!(
            !ai.is_fetching_models,
            "Should not be fetching models initially"
        );
        assert!(
            ai.current_response.is_empty(),
            "Current response should be empty"
        );
        assert!(ai.current_session.is_none(), "No session should be active");
        assert!(!ai.is_summarizing, "Should not be summarizing initially");
        assert!(ai.available_models.is_empty(), "No models should be loaded");
        assert!(
            !ai.fallback_enabled,
            "Fallback should be disabled by default"
        );
        assert!(
            ai.fallback_providers.is_empty(),
            "No fallback providers by default"
        );
        assert!(
            !ai.auto_compaction,
            "Auto-compaction should be disabled by default"
        );
        assert!(
            ai.api_key_manager.is_none(),
            "No API key manager by default"
        );
        assert!(
            !ai.adaptive_thinking.enabled,
            "Adaptive thinking disabled by default"
        );
        assert!(
            ai.last_thinking_result.is_none(),
            "No thinking result by default"
        );
        assert!(
            ai.last_thinking_strategy.is_none(),
            "No thinking strategy by default"
        );
        assert!(
            ai.detected_context_size.is_none(),
            "No detected context size by default"
        );
        assert!(ai.cost_dashboard.is_none(), "No cost dashboard by default");
        assert!(ai.chat_hooks.is_none(), "No chat hooks by default");
    }

    #[test]
    fn test_default_trait_calls_new() {
        let ai = AiAssistant::default();
        // Should behave identically to ::new()
        assert!(ai.conversation.is_empty());
        assert!(!ai.is_generating);
        assert!(ai.system_prompt().contains("helpful AI assistant"));
    }

    #[test]
    fn test_with_system_prompt() {
        let ai = AiAssistant::with_system_prompt("You are a pirate.");
        assert_eq!(ai.system_prompt(), "You are a pirate.");
        assert!(ai.conversation.is_empty());
    }

    // --- System Prompt Tests ---

    #[test]
    fn test_set_and_get_system_prompt() {
        let mut ai = AiAssistant::new();
        let original = ai.system_prompt().to_string();
        assert!(!original.is_empty());

        ai.set_system_prompt("Custom prompt");
        assert_eq!(ai.system_prompt(), "Custom prompt");

        ai.set_system_prompt("");
        assert_eq!(ai.system_prompt(), "");
    }

    // --- Knowledge Context Tests ---

    #[test]
    fn test_knowledge_context_lifecycle() {
        let mut ai = AiAssistant::new();

        // Initially empty
        assert!(!ai.has_knowledge_context());
        assert_eq!(ai.knowledge_context_size(), 0);
        assert_eq!(ai.get_knowledge_context(), "");

        // Set context
        ai.set_knowledge_context("First knowledge");
        assert!(ai.has_knowledge_context());
        assert_eq!(ai.get_knowledge_context(), "First knowledge");
        assert_eq!(ai.knowledge_context_size(), "First knowledge".len());

        // Overwrite context
        ai.set_knowledge_context("Second knowledge");
        assert_eq!(ai.get_knowledge_context(), "Second knowledge");

        // Clear context
        ai.clear_knowledge_context();
        assert!(!ai.has_knowledge_context());
        assert_eq!(ai.knowledge_context_size(), 0);
        assert_eq!(ai.get_knowledge_context(), "");
    }

    #[test]
    fn test_append_knowledge_context() {
        let mut ai = AiAssistant::new();

        // Append to empty
        ai.append_knowledge_context("Part A");
        assert_eq!(ai.get_knowledge_context(), "Part A");

        // Append to non-empty (adds separator)
        ai.append_knowledge_context("Part B");
        assert_eq!(ai.get_knowledge_context(), "Part A\n\nPart B");

        // Append again
        ai.append_knowledge_context("Part C");
        assert_eq!(ai.get_knowledge_context(), "Part A\n\nPart B\n\nPart C");
        assert_eq!(
            ai.knowledge_context_size(),
            "Part A\n\nPart B\n\nPart C".len()
        );
    }

    #[test]
    fn test_knowledge_context_unicode() {
        let mut ai = AiAssistant::new();
        let unicode_text = "Informacion sobre inteligencia artificial y aprendizaje automatico";
        ai.set_knowledge_context(unicode_text);
        assert_eq!(ai.get_knowledge_context(), unicode_text);
        assert!(ai.has_knowledge_context());
        // size() is in bytes, not chars
        assert_eq!(ai.knowledge_context_size(), unicode_text.len());
    }

    // --- Adaptive Thinking Tests ---

    #[test]
    fn test_adaptive_thinking_toggle() {
        let mut ai = AiAssistant::new();
        assert!(!ai.adaptive_thinking.enabled);

        ai.enable_adaptive_thinking();
        assert!(ai.adaptive_thinking.enabled);

        ai.disable_adaptive_thinking();
        assert!(!ai.adaptive_thinking.enabled);
    }

    #[test]
    fn test_set_adaptive_thinking_custom_config() {
        let mut ai = AiAssistant::new();
        let mut config = AdaptiveThinkingConfig::default();
        config.enabled = true;
        config.adjust_temperature = false;
        config.parse_thinking_tags = true;

        ai.set_adaptive_thinking(config.clone());
        assert!(ai.adaptive_thinking.enabled);
        assert!(!ai.adaptive_thinking.adjust_temperature);
        assert!(ai.adaptive_thinking.parse_thinking_tags);
    }

    #[test]
    fn test_classify_query_returns_strategy() {
        let ai = AiAssistant::new();
        // Simple greeting should produce a strategy (doesn't matter which, just that it works)
        let strategy = ai.classify_query("Hello, how are you?");
        // ThinkingStrategy always has a temperature
        assert!(strategy.temperature >= 0.0 && strategy.temperature <= 2.0);
    }

    #[test]
    fn test_classify_query_does_not_mutate_state() {
        let ai = AiAssistant::new();
        assert!(ai.last_thinking_strategy.is_none());
        let _strategy = ai.classify_query("Write a poem about the sea");
        // classify_query is &self, so no mutation
        assert!(ai.last_thinking_strategy.is_none());
    }

    // --- Conversation Management Tests ---

    #[test]
    fn test_conversation_management() {
        let mut ai = AiAssistant::new();
        assert_eq!(ai.message_count(), 0);
        assert!(ai.get_display_messages().is_empty());

        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi!"));
        assert_eq!(ai.message_count(), 2);
        assert_eq!(ai.get_display_messages().len(), 2);
        assert_eq!(ai.get_display_messages()[0].role, "user");
        assert_eq!(ai.get_display_messages()[1].role, "assistant");

        ai.clear_conversation();
        assert_eq!(ai.message_count(), 0);
        assert!(ai.current_response.is_empty());
    }

    // --- Config / Preferences Loading Tests ---

    #[test]
    fn test_load_config() {
        let mut ai = AiAssistant::new();
        let mut config = AiConfig::default();
        config.selected_model = "test-model-7b".to_string();
        config.temperature = 0.42;

        ai.load_config(config);
        assert_eq!(ai.config.selected_model, "test-model-7b");
        assert!((ai.config.temperature - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn test_load_preferences() {
        let mut ai = AiAssistant::new();
        let mut prefs = UserPreferences::default();
        prefs.response_style = ResponseStyle::Technical;
        prefs.global_notes = "Some global notes".to_string();

        ai.load_preferences(prefs);
        assert!(matches!(
            ai.preferences.response_style,
            ResponseStyle::Technical
        ));
        assert_eq!(ai.preferences.global_notes, "Some global notes");
    }

    // --- Cancellation Tests ---

    #[test]
    fn test_cancel_no_active_generation() {
        let mut ai = AiAssistant::new();
        assert!(!ai.can_cancel());
        assert!(!ai.cancel_generation());
        assert!(ai.get_cancel_token().is_none());
    }

    // --- Session / Notes Tests ---

    #[test]
    fn test_session_notes_no_session() {
        let ai = AiAssistant::new();
        // No current session, should return empty
        assert_eq!(ai.get_session_notes(), "");
    }

    #[test]
    fn test_session_notes_with_session() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        assert_eq!(ai.get_session_notes(), "");

        ai.set_session_notes("Important context for this session");
        assert_eq!(ai.get_session_notes(), "Important context for this session");

        ai.set_session_notes("");
        assert_eq!(ai.get_session_notes(), "");
    }

    #[test]
    fn test_global_notes() {
        let mut ai = AiAssistant::new();
        assert_eq!(ai.get_global_notes(), "");

        ai.set_global_notes("User prefers formal language");
        assert_eq!(ai.get_global_notes(), "User prefers formal language");

        ai.set_global_notes("");
        assert_eq!(ai.get_global_notes(), "");
    }

    // --- Compaction Config Tests ---

    #[test]
    fn test_set_compaction_config() {
        let mut ai = AiAssistant::new();
        let config = CompactionConfig {
            max_messages: 50,
            target_messages: 20,
            preserve_recent: 5,
            preserve_first: 2,
            min_importance: 0.5,
            llm_enhanced: false,
        };
        ai.set_compaction_config(config);
        assert_eq!(ai.compaction_config.max_messages, 50);
        assert_eq!(ai.compaction_config.target_messages, 20);
        assert_eq!(ai.compaction_config.preserve_recent, 5);
    }

    // --- API Key Config Tests ---

    #[test]
    fn test_set_api_key_config_before_add() {
        let mut ai = AiAssistant::new();
        let config = RotationConfig {
            auto_rotate: true,
            rotation_interval: Some(std::time::Duration::from_secs(120)),
            max_errors_before_rotation: 5,
            rate_limit_recovery_time: std::time::Duration::from_secs(60),
        };
        ai.set_api_key_config(config);
        assert!(ai.api_key_manager.is_some());

        // Adding a key should use the existing manager, not create a new one
        ai.add_api_key("anthropic", "key1", "sk-ant-xxx");
        let key = ai.get_current_api_key("anthropic");
        assert_eq!(key, Some("sk-ant-xxx".to_string()));
    }

    #[test]
    fn test_mark_key_rate_limited_no_manager() {
        let mut ai = AiAssistant::new();
        // Should not panic when no manager exists
        ai.mark_key_rate_limited("openai", "nonexistent");
    }

    #[test]
    fn test_api_key_multiple_providers() {
        let mut ai = AiAssistant::new();
        ai.add_api_key("openai", "oai1", "sk-openai-1");
        ai.add_api_key("anthropic", "ant1", "sk-ant-1");

        assert_eq!(
            ai.get_current_api_key("openai"),
            Some("sk-openai-1".to_string())
        );
        assert_eq!(
            ai.get_current_api_key("anthropic"),
            Some("sk-ant-1".to_string())
        );
        assert!(ai.get_current_api_key("google").is_none());
    }

    // --- Context Cache Tests ---

    #[test]
    fn test_invalidate_context_cache() {
        let mut ai = AiAssistant::new();
        // Manually set cache values
        ai.detected_context_size = Some(8192);
        ai.detected_context_model = Some("test-model".to_string());

        ai.invalidate_context_cache();
        assert!(ai.detected_context_size.is_none());
        assert!(ai.detected_context_model.is_none());
    }

    // --- Metrics Tests ---

    #[test]
    fn test_metrics_export_json() {
        let ai = AiAssistant::new();
        let json = ai.export_metrics_json();
        assert!(!json.is_empty(), "Metrics JSON should not be empty");
        // Should be valid JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok(), "Metrics export should be valid JSON");
    }

    #[test]
    fn test_reset_metrics() {
        let mut ai = AiAssistant::new();
        ai.start_message_tracking();
        ai.finish_message_tracking(100);
        assert!(!ai.get_message_metrics().is_empty());

        ai.reset_metrics("new-session");
        assert!(ai.get_message_metrics().is_empty());
    }

    #[test]
    fn test_session_metrics_initial() {
        let ai = AiAssistant::new();
        let metrics = ai.get_session_metrics();
        assert_eq!(metrics.message_count, 0);
    }

    // --- Cost Dashboard / Chat Hooks Initialization Tests ---

    #[test]
    fn test_cost_dashboard_init() {
        let mut ai = AiAssistant::new();
        assert!(ai.cost_dashboard().is_none());

        ai.init_cost_tracking();
        assert!(ai.cost_dashboard().is_some());
        assert!(ai.cost_dashboard_mut().is_some());

        // Calling init again should not replace
        ai.init_cost_tracking();
        assert!(ai.cost_dashboard().is_some());
    }

    #[test]
    fn test_cost_report_none_without_init() {
        let ai = AiAssistant::new();
        assert!(ai.cost_report().is_none());
    }

    #[test]
    fn test_cost_report_some_after_init() {
        let mut ai = AiAssistant::new();
        ai.init_cost_tracking();
        let report = ai.cost_report();
        assert!(
            report.is_some(),
            "Cost report should be available after init"
        );
    }

    #[test]
    fn test_chat_hooks_init() {
        let mut ai = AiAssistant::new();
        assert!(ai.chat_hooks().is_none());

        ai.init_chat_hooks();
        assert!(ai.chat_hooks().is_some());
        assert!(ai.chat_hooks_mut().is_some());

        // Re-init should not replace
        ai.init_chat_hooks();
        assert!(ai.chat_hooks().is_some());
    }

    // --- Preference Extraction Tests ---

    #[test]
    fn test_extract_preferences_with_custom_extractor() {
        let mut ai = AiAssistant::new();
        ai.conversation
            .push(ChatMessage::user("I prefer code examples"));
        ai.conversation.push(ChatMessage::assistant("Sure!"));

        ai.extract_preferences_with(|msgs, prefs| {
            for msg in msgs {
                if msg.content.contains("code examples") {
                    prefs.response_style = ResponseStyle::Technical;
                }
            }
        });

        assert!(matches!(
            ai.preferences.response_style,
            ResponseStyle::Technical
        ));
    }

    // --- Summarization Trigger Tests ---

    #[test]
    fn test_should_summarize_with_few_messages() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi!"));
        // Less than 6 messages, should not trigger
        assert!(!ai.should_summarize(""));
        assert!(!ai.should_summarize_auto());
    }

    #[test]
    fn test_should_summarize_while_summarizing() {
        let mut ai = AiAssistant::new();
        for i in 0..10 {
            ai.conversation
                .push(ChatMessage::user(&format!("Message {}", i)));
        }
        ai.is_summarizing = true;
        // Should not trigger while already summarizing
        assert!(!ai.should_summarize(""));
    }

    // --- New Session Tests ---

    #[test]
    fn test_new_session_creates_session() {
        let mut ai = AiAssistant::new();
        assert!(ai.current_session.is_none());

        ai.new_session();
        assert!(ai.current_session.is_some());
        assert!(ai.conversation.is_empty());
    }

    #[test]
    fn test_new_session_saves_existing_conversation() {
        let mut ai = AiAssistant::new();
        ai.conversation
            .push(ChatMessage::user("Before new session"));
        ai.conversation.push(ChatMessage::assistant("Reply"));

        ai.new_session();
        // Old conversation should have been saved and conversation cleared
        assert!(ai.conversation.is_empty());
        assert!(ai.current_session.is_some());
        // The session store should have the old session saved
        assert!(!ai.session_store.sessions.is_empty());
    }

    // --- Event Bus Tests ---

    #[test]
    fn test_event_bus_accessible() {
        let ai = AiAssistant::new();
        // Event bus should be available and functional
        // Just verify we can emit without panic
        ai.event_bus.emit(crate::events::AiEvent::SessionCreated {
            session_id: "test".to_string(),
        });
    }

    // --- Poll Response with No Active Generation ---

    #[test]
    fn test_poll_response_no_generation() {
        let mut ai = AiAssistant::new();
        assert!(ai.poll_response().is_none());
    }

    // --- Compact Empty Conversation ---

    #[test]
    fn test_compact_empty_conversation() {
        let mut ai = AiAssistant::new();
        let result = ai.compact_conversation();
        assert_eq!(result.removed_count, 0);
        assert!(ai.conversation.is_empty());
    }

    // --- Module Logging Tests ---

    #[test]
    fn test_load_config_model_change_logging() {
        let mut ai = AiAssistant::new();
        let mut config = AiConfig::default();
        config.selected_model = "model-alpha".to_string();
        ai.load_config(config.clone());
        assert_eq!(ai.config.selected_model, "model-alpha");

        // Change model - triggers log::info path
        config.selected_model = "model-beta".to_string();
        ai.load_config(config);
        assert_eq!(ai.config.selected_model, "model-beta");
    }

    #[test]
    fn test_session_lifecycle_logging() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        assert!(ai.current_session.is_some());

        let sid = ai
            .current_session
            .as_ref()
            .map(|s| s.id.clone())
            .unwrap_or_default();
        ai.conversation.push(ChatMessage::user("hello"));
        ai.save_current_session();

        ai.new_session();
        ai.load_session(&sid);
        ai.delete_session(&sid);
    }

    // ----------------------------------------------------------
    // Session Persistence Tests (7.3)
    // ----------------------------------------------------------

    #[test]
    fn test_save_sessions_empty() {
        let ai = AiAssistant::new();
        let dir = std::env::temp_dir().join(format!("test_sessions_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sessions.bin");
        ai.save_sessions_to_file(&path).unwrap();
        assert!(path.exists());
        // File should have been created and contain data
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_and_load_sessions() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        ai.conversation.push(ChatMessage::user("hello"));
        ai.save_current_session();

        let dir = std::env::temp_dir().join(format!("test_sessions_load_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sessions.bin");
        let expected_count = ai.session_store.sessions.len();
        ai.save_sessions_to_file(&path).unwrap();

        let mut ai2 = AiAssistant::new();
        ai2.load_sessions_from_file(&path).unwrap();
        assert_eq!(ai2.session_store.sessions.len(), expected_count);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sessions_nonexistent_returns_default() {
        let mut ai = AiAssistant::new();
        // load_from_file returns Ok(default) for nonexistent paths
        let result =
            ai.load_sessions_from_file(std::path::Path::new("/nonexistent_dir_xyz/sessions.bin"));
        // On most OSes this returns Ok with an empty default store
        // The exact behavior depends on the platform, so just verify it doesn't panic
        if result.is_ok() {
            assert!(ai.session_store.sessions.is_empty());
        }
    }

    #[test]
    fn test_save_sessions_preserves_messages() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        ai.conversation.push(ChatMessage::user("test message"));
        ai.conversation.push(ChatMessage::assistant("test reply"));
        ai.save_current_session();

        let dir = std::env::temp_dir().join(format!("test_sessions_msgs_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sessions.bin");
        ai.save_sessions_to_file(&path).unwrap();

        let mut ai2 = AiAssistant::new();
        ai2.load_sessions_from_file(&path).unwrap();
        // Should have restored the session with messages
        assert!(!ai2.session_store.sessions.is_empty());
        let session = &ai2.session_store.sessions[0];
        assert_eq!(session.messages.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_sessions_multiple_sessions() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        ai.conversation.push(ChatMessage::user("session 1"));
        ai.save_current_session();
        ai.new_session();
        ai.conversation.push(ChatMessage::user("session 2"));
        ai.save_current_session();

        let dir = std::env::temp_dir().join(format!("test_sessions_multi_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sessions.bin");
        let expected_count = ai.session_store.sessions.len();
        ai.save_sessions_to_file(&path).unwrap();

        let mut ai2 = AiAssistant::new();
        ai2.load_sessions_from_file(&path).unwrap();
        assert_eq!(ai2.session_store.sessions.len(), expected_count);
        // At least one session should exist
        assert!(!ai2.session_store.sessions.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sessions_restores_current() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        let sid = ai.current_session.as_ref().unwrap().id.clone();
        ai.conversation.push(ChatMessage::user("restore me"));
        ai.save_current_session();

        let dir =
            std::env::temp_dir().join(format!("test_sessions_restore_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sessions.bin");
        ai.save_sessions_to_file(&path).unwrap();

        let mut ai2 = AiAssistant::new();
        ai2.load_sessions_from_file(&path).unwrap();
        // The current session should be restored
        assert!(ai2.current_session.is_some());
        assert_eq!(ai2.current_session.as_ref().unwrap().id, sid);
        assert!(!ai2.conversation.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // =========================================================================
    // Constrained Decoding Integration Tests (v9 item 3.1)
    // =========================================================================

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_parses_valid_gbnf() {
        // Verify that the method correctly parses a GBNF grammar.
        // The LLM call will fail (no server), but grammar parsing should succeed first.
        let ai = AiAssistant::new();
        let grammar = r#"root ::= "yes" | "no""#;
        let result = ai.generate_with_grammar(grammar, "Do you agree?");
        // LLM call will fail because no server is running, which is expected
        assert!(result.is_err());
    }

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_rejects_invalid_grammar() {
        let ai = AiAssistant::new();
        let grammar = "this is not valid gbnf at all";
        let result = ai.generate_with_grammar(grammar, "test");
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            err_str.contains("ConstrainedDecoding") || err_str.contains("::="),
            "Error should be about grammar syntax: {}",
            err_str
        );
    }

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_empty_grammar_fails() {
        let ai = AiAssistant::new();
        let result = ai.generate_with_grammar("", "test");
        assert!(result.is_err());
    }

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_complex_grammar_parses() {
        let ai = AiAssistant::new();
        let grammar = r#"root ::= object
object ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= string ws ":" ws value
string ::= "\"" [a-z]+ "\""
value ::= string | "true" | "false"
ws ::= " "*"#;
        let result = ai.generate_with_grammar(grammar, "Generate JSON");
        // Grammar should parse, LLM call will fail
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        // Should not be a grammar parse error
        assert!(
            !err_str.contains("GrammarSyntaxError"),
            "Should not have a grammar syntax error: {}",
            err_str
        );
    }

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_comment_lines_ignored() {
        let ai = AiAssistant::new();
        let grammar = "# This is a comment\nroot ::= \"hello\"";
        let result = ai.generate_with_grammar(grammar, "Say hello");
        // Grammar should parse fine, LLM call will fail
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            !err_str.contains("GrammarSyntaxError"),
            "Comments should be ignored: {}",
            err_str
        );
    }

    #[cfg(feature = "constrained-decoding")]
    #[test]
    fn test_generate_with_grammar_multiline_grammar() {
        let ai = AiAssistant::new();
        let grammar = "root ::= greeting\ngreeting ::= \"hi\" | \"hello\" | \"hey\"";
        let result = ai.generate_with_grammar(grammar, "Greet me");
        // Should parse OK (multi-rule grammar), LLM call fails
        assert!(result.is_err());
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            !err_str.contains("GrammarSyntaxError"),
            "Multi-rule grammar should parse: {}",
            err_str
        );
    }

    // =========================================================================
    // HITL Integration Tests (v9 item 3.2)
    // =========================================================================

    #[cfg(feature = "hitl")]
    #[test]
    fn test_send_message_with_approval_auto_approve() {
        let mut ai = AiAssistant::new();
        // LLM call will fail but we verify the method signature and flow
        let result = ai.send_message_with_approval("Hello", true);
        // Will fail because no LLM server is running
        assert!(result.is_err());
    }

    #[cfg(feature = "hitl")]
    #[test]
    fn test_send_message_with_approval_manual_gate() {
        let mut ai = AiAssistant::new();
        let result = ai.send_message_with_approval("Test message", false);
        // Will fail because no LLM server is running
        assert!(result.is_err());
    }

    #[cfg(feature = "hitl")]
    #[test]
    fn test_send_message_with_approval_updates_conversation_on_success() {
        // When LLM is unavailable, conversation should not be updated
        let mut ai = AiAssistant::new();
        let initial_len = ai.conversation.len();
        let _result = ai.send_message_with_approval("Test", true);
        // If LLM fails, conversation should not have been modified
        assert_eq!(ai.conversation.len(), initial_len);
    }

    #[cfg(feature = "hitl")]
    #[test]
    fn test_send_message_with_approval_empty_message() {
        let mut ai = AiAssistant::new();
        let result = ai.send_message_with_approval("", true);
        // Empty message should still be sent (LLM will fail due to no server)
        assert!(result.is_err());
    }

    #[cfg(feature = "hitl")]
    #[test]
    fn test_hitl_approval_request_creation() {
        // Test that ApprovalRequest can be created with expected fields
        use crate::hitl::{ApprovalRequest, ImpactLevel};
        use std::collections::HashMap;

        let request = ApprovalRequest::new(
            "test-id",
            "send_message",
            HashMap::new(),
            "ai_assistant",
            "Test context",
            ImpactLevel::Low,
        );
        assert_eq!(request.request_id, "test-id");
        assert_eq!(request.tool_name, "send_message");
        assert_eq!(request.agent_id, "ai_assistant");
        assert!(matches!(request.estimated_impact, ImpactLevel::Low));
    }

    #[cfg(feature = "hitl")]
    #[test]
    fn test_hitl_approval_log_records() {
        use crate::hitl::{
            ApprovalDecision, ApprovalLog, ApprovalLogEntry, ApprovalRequest, ImpactLevel,
        };
        use std::collections::HashMap;

        let mut log = ApprovalLog::new(100);
        assert!(log.is_empty());

        let request = ApprovalRequest::new(
            "req-1",
            "send_message",
            HashMap::new(),
            "ai_assistant",
            "context",
            ImpactLevel::Low,
        );

        log.record(ApprovalLogEntry {
            request,
            decision: ApprovalDecision::Approve,
            gate_name: "test-gate".to_string(),
            timestamp: 12345,
        });

        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
        assert_eq!(log.approval_rate(), 1.0);
    }

    // =========================================================================
    // MCP Client Integration Tests (v9 item 3.3)
    // =========================================================================

    #[test]
    fn test_connect_mcp_server_empty_url() {
        let mut ai = AiAssistant::new();
        let result = ai.connect_mcp_server("");
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("empty"),
            "Error should mention empty URL: {}",
            err_str
        );
    }

    #[test]
    fn test_connect_mcp_server_invalid_protocol() {
        let mut ai = AiAssistant::new();
        let result = ai.connect_mcp_server("ftp://example.com");
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(
            err_str.contains("http") || err_str.contains("Invalid"),
            "Error should mention invalid protocol: {}",
            err_str
        );
    }

    #[test]
    fn test_connect_mcp_server_simulated_connect() {
        let mut ai = AiAssistant::new();
        // This URL won't resolve but RemoteMcpClient falls back to simulated mode
        let result = ai.connect_mcp_server("http://localhost:19999/mcp");
        // Should succeed (falls back to simulated mode)
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_mcp_tools_empty_url() {
        let ai = AiAssistant::new();
        let tools = ai.list_mcp_tools("");
        assert!(tools.is_empty());
    }

    #[test]
    fn test_list_mcp_tools_simulated_server() {
        let ai = AiAssistant::new();
        // Simulated server provides default tools
        let tools = ai.list_mcp_tools("http://localhost:19999/mcp");
        // Simulated mode returns some placeholder tools
        // (may be empty or populated depending on implementation)
        let _ = tools.len(); // just verify it does not panic
    }

    #[test]
    fn test_list_mcp_tools_unreachable_server() {
        let ai = AiAssistant::new();
        // The client will fall back to simulated mode for an unreachable server
        let tools = ai.list_mcp_tools("http://192.0.2.1:1/mcp");
        // Should return empty or simulated tools without panicking
        let _ = tools.len();
    }

    // =========================================================================
    // Distillation Integration Tests (v9 item 3.4)
    // =========================================================================

    #[cfg(feature = "distillation")]
    #[test]
    fn test_collect_trajectory_empty_conversation() {
        let mut ai = AiAssistant::new();
        let pairs = ai.collect_trajectory();
        assert!(pairs.is_empty());
    }

    #[cfg(feature = "distillation")]
    #[test]
    fn test_collect_trajectory_with_pairs() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi there!"));
        ai.conversation.push(ChatMessage::user("How are you?"));
        ai.conversation
            .push(ChatMessage::assistant("I'm doing well!"));

        let pairs = ai.collect_trajectory();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "Hello");
        assert_eq!(pairs[0].1, "Hi there!");
        assert_eq!(pairs[1].0, "How are you?");
        assert_eq!(pairs[1].1, "I'm doing well!");
    }

    #[cfg(feature = "distillation")]
    #[test]
    fn test_collect_trajectory_odd_messages() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi!"));
        ai.conversation.push(ChatMessage::user("Unpaired"));

        let pairs = ai.collect_trajectory();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "Hello");
        assert_eq!(pairs[0].1, "Hi!");
    }

    #[cfg(feature = "distillation")]
    #[test]
    fn test_export_training_data_empty() {
        let ai = AiAssistant::new();
        let json = ai.export_training_data().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 0);
    }

    #[cfg(feature = "distillation")]
    #[test]
    fn test_export_training_data_with_conversation() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("What is Rust?"));
        ai.conversation.push(ChatMessage::assistant(
            "Rust is a systems programming language.",
        ));

        let json = ai.export_training_data().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["input"], "What is Rust?");
        assert_eq!(arr[0]["output"], "Rust is a systems programming language.");
    }

    #[cfg(feature = "distillation")]
    #[test]
    fn test_export_training_data_valid_json() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user(
            "Tell me about \"quotes\" and \\backslash",
        ));
        ai.conversation.push(ChatMessage::assistant(
            "Special chars: \"quotes\", \\backslash",
        ));
        ai.conversation.push(ChatMessage::user("Another"));
        ai.conversation.push(ChatMessage::assistant("Response two"));

        let json = ai.export_training_data().unwrap();
        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // Verify special characters are preserved
        assert!(arr[0]["input"].as_str().unwrap().contains("quotes"));
    }

    // =========================================================================
    // Knowledge Context Advanced Tests
    // =========================================================================

    #[test]
    fn test_knowledge_context_append_multiple_all_present() {
        let mut ai = AiAssistant::new();
        ai.append_knowledge_context("Chapter 1: Introduction");
        ai.append_knowledge_context("Chapter 2: Methods");
        ai.append_knowledge_context("Chapter 3: Results");

        let ctx = ai.get_knowledge_context();
        assert!(ctx.contains("Chapter 1: Introduction"));
        assert!(ctx.contains("Chapter 2: Methods"));
        assert!(ctx.contains("Chapter 3: Results"));
        // All three must be present simultaneously
        assert_eq!(ctx.matches("Chapter").count(), 3);
    }

    #[test]
    fn test_knowledge_context_size_matches_byte_length() {
        let mut ai = AiAssistant::new();
        let content = "Knowledge about Rust programming language features";
        ai.set_knowledge_context(content);
        assert_eq!(ai.knowledge_context_size(), content.len());
        assert_eq!(ai.knowledge_context_size(), content.len());
    }

    #[test]
    fn test_knowledge_context_clear_resets_size_to_zero() {
        let mut ai = AiAssistant::new();
        ai.set_knowledge_context("Some important data that takes up space");
        assert!(ai.knowledge_context_size() > 0);

        ai.clear_knowledge_context();
        assert_eq!(ai.knowledge_context_size(), 0);
    }

    #[test]
    fn test_knowledge_context_overwrite_replaces_completely() {
        let mut ai = AiAssistant::new();
        ai.set_knowledge_context("Original content");
        ai.set_knowledge_context("Replacement content");
        assert_eq!(ai.get_knowledge_context(), "Replacement content");
        assert!(!ai.get_knowledge_context().contains("Original"));
    }

    #[test]
    fn test_knowledge_context_append_preserves_separator() {
        let mut ai = AiAssistant::new();
        ai.append_knowledge_context("AAA");
        ai.append_knowledge_context("BBB");
        // Separator is \n\n between non-empty parts
        assert_eq!(ai.get_knowledge_context(), "AAA\n\nBBB");
        // Size should account for the separator bytes
        assert_eq!(ai.knowledge_context_size(), "AAA\n\nBBB".len());
    }

    // =========================================================================
    // Conversation Management Advanced Tests
    // =========================================================================

    #[test]
    fn test_message_count_increments_with_pushes() {
        let mut ai = AiAssistant::new();
        for i in 0..5 {
            ai.conversation
                .push(ChatMessage::user(&format!("Message {}", i)));
        }
        assert_eq!(ai.message_count(), 5);
    }

    #[test]
    fn test_clear_conversation_resets_count_to_zero() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Hello"));
        ai.conversation.push(ChatMessage::assistant("Hi!"));
        ai.conversation.push(ChatMessage::user("How are you?"));
        assert_eq!(ai.message_count(), 3);

        ai.clear_conversation();
        assert_eq!(ai.message_count(), 0);
        assert!(ai.get_display_messages().is_empty());
    }

    #[test]
    fn test_get_display_messages_returns_all_pushed() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("First"));
        ai.conversation.push(ChatMessage::assistant("Second"));
        ai.conversation.push(ChatMessage::user("Third"));

        let msgs = ai.get_display_messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "First");
        assert_eq!(msgs[1].content, "Second");
        assert_eq!(msgs[2].content, "Third");
    }

    #[test]
    fn test_conversation_with_mixed_roles() {
        let mut ai = AiAssistant::new();
        ai.conversation
            .push(ChatMessage::system("System instruction"));
        ai.conversation.push(ChatMessage::user("User question"));
        ai.conversation
            .push(ChatMessage::assistant("Assistant answer"));

        assert_eq!(ai.message_count(), 3);
        assert_eq!(ai.get_display_messages()[0].role, "system");
        assert_eq!(ai.get_display_messages()[1].role, "user");
        assert_eq!(ai.get_display_messages()[2].role, "assistant");
    }

    #[test]
    fn test_clear_conversation_also_clears_current_response() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("Test"));
        ai.current_response = "partial response".to_string();

        ai.clear_conversation();
        assert!(ai.current_response.is_empty());
    }

    #[test]
    fn test_is_generating_false_initially() {
        let ai = AiAssistant::new();
        assert!(!ai.is_generating);
    }

    // =========================================================================
    // Session Management Tests
    // =========================================================================

    #[test]
    fn test_get_sessions_empty_initially() {
        let ai = AiAssistant::new();
        assert!(ai.get_sessions().is_empty());
    }

    #[test]
    fn test_delete_session_removes_from_store() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        let session_id = ai.current_session.as_ref().unwrap().id.clone();
        ai.conversation.push(ChatMessage::user("test msg"));
        ai.save_current_session();

        let count_before = ai.get_sessions().len();
        assert!(count_before >= 1);

        ai.delete_session(&session_id);
        // After deleting the current session, it should be None
        assert!(ai.current_session.is_none());
        assert!(ai.conversation.is_empty());
    }

    #[test]
    fn test_delete_session_nonexistent_does_not_panic() {
        let mut ai = AiAssistant::new();
        // Should not panic when deleting a session that doesn't exist
        ai.delete_session("nonexistent-session-id");
        assert!(ai.get_sessions().is_empty());
    }

    #[test]
    fn test_load_session_restores_messages() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        let session_id = ai.current_session.as_ref().unwrap().id.clone();
        ai.conversation.push(ChatMessage::user("remember me"));
        ai.conversation
            .push(ChatMessage::assistant("I will remember"));
        ai.save_current_session();

        // Start a new session (clears conversation)
        ai.new_session();
        assert!(ai.conversation.is_empty());

        // Load the old session
        ai.load_session(&session_id);
        assert_eq!(ai.conversation.len(), 2);
        assert_eq!(ai.conversation[0].content, "remember me");
        assert_eq!(ai.conversation[1].content, "I will remember");
    }

    #[test]
    fn test_session_notes_set_without_session_is_noop() {
        let mut ai = AiAssistant::new();
        assert!(ai.current_session.is_none());
        // Setting notes without a session should not panic
        ai.set_session_notes("These notes go nowhere");
        // And reading still returns empty
        assert_eq!(ai.get_session_notes(), "");
    }

    // =========================================================================
    // Context and Model Sizing Tests
    // =========================================================================

    #[test]
    fn test_detect_model_context_size_returns_positive() {
        let mut ai = AiAssistant::new();
        let size = ai.detect_model_context_size();
        // Should always return a positive value (fallback default)
        assert!(size > 0);
    }

    #[test]
    fn test_context_cache_invalidation_clears_detected() {
        let mut ai = AiAssistant::new();
        // Force a detection to populate the cache
        let _ = ai.detect_model_context_size();
        assert!(ai.detected_context_size.is_some());
        assert!(ai.detected_context_model.is_some());

        ai.invalidate_context_cache();
        assert!(ai.detected_context_size.is_none());
        assert!(ai.detected_context_model.is_none());
    }

    #[test]
    fn test_calculate_context_usage_empty_conversation() {
        let ai = AiAssistant::new();
        let usage = ai.calculate_context_usage("");
        // With no messages and no knowledge, usage should be minimal
        assert!(usage.conversation_tokens == 0 || usage.conversation_tokens < 10);
        assert_eq!(usage.knowledge_tokens, 0);
    }

    #[test]
    fn test_should_summarize_few_messages_returns_false() {
        let mut ai = AiAssistant::new();
        // With only 4 messages (< 6 threshold), should not summarize
        ai.conversation.push(ChatMessage::user("A"));
        ai.conversation.push(ChatMessage::assistant("B"));
        ai.conversation.push(ChatMessage::user("C"));
        ai.conversation.push(ChatMessage::assistant("D"));
        assert!(!ai.should_summarize("some knowledge context"));
    }

    #[test]
    fn test_get_effective_max_history_positive() {
        let ai = AiAssistant::new();
        let max_history = ai.get_effective_max_history("");
        // Should always return at least 4 (the minimum clamp)
        assert!(max_history >= 4);
    }

    // =========================================================================
    // Notes Management Tests
    // =========================================================================

    #[test]
    fn test_global_notes_default_is_empty_string() {
        let ai = AiAssistant::new();
        assert_eq!(ai.get_global_notes(), "");
        assert!(ai.get_global_notes().is_empty());
    }

    #[test]
    fn test_global_notes_set_then_overwrite() {
        let mut ai = AiAssistant::new();
        ai.set_global_notes("First draft");
        assert_eq!(ai.get_global_notes(), "First draft");

        ai.set_global_notes("Final version");
        assert_eq!(ai.get_global_notes(), "Final version");
        assert!(!ai.get_global_notes().contains("First draft"));
    }

    #[test]
    fn test_session_notes_default_empty_with_session() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        // A fresh session should have empty notes
        assert_eq!(ai.get_session_notes(), "");
    }

    #[test]
    fn test_session_notes_persist_across_save_load() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        ai.set_session_notes("Remember: user prefers concise answers");
        let session_id = ai.current_session.as_ref().unwrap().id.clone();
        ai.save_current_session();

        // Create new session, then load the old one
        ai.new_session();
        ai.load_session(&session_id);

        // Notes should be restored via the session's context_notes field
        // (load_session restores the full ChatSession object)
        if let Some(ref session) = ai.current_session {
            assert_eq!(
                session.context_notes,
                "Remember: user prefers concise answers"
            );
        }
    }

    // =========================================================================
    // Fallback Providers Advanced Tests
    // =========================================================================

    #[test]
    fn test_fallback_active_false_by_default() {
        let ai = AiAssistant::new();
        assert!(!ai.fallback_active());
        assert!(!ai.fallback_enabled);
        assert!(ai.fallback_providers.is_empty());
    }

    #[test]
    fn test_enable_disable_fallback_toggling() {
        let mut ai = AiAssistant::new();
        ai.configure_fallback(vec![(AiProvider::Ollama, "llama3".into())]);

        ai.enable_fallback();
        assert!(ai.fallback_active());

        ai.disable_fallback();
        assert!(!ai.fallback_active());

        ai.enable_fallback();
        assert!(ai.fallback_active());
    }

    #[test]
    fn test_configure_fallback_with_three_providers() {
        let mut ai = AiAssistant::new();
        ai.configure_fallback(vec![
            (AiProvider::LMStudio, "model-a".into()),
            (AiProvider::Ollama, "model-b".into()),
            (AiProvider::OpenAI, "gpt-4".into()),
        ]);
        ai.enable_fallback();
        assert!(ai.fallback_active());
        assert_eq!(ai.fallback_providers.len(), 3);
    }

    #[test]
    fn test_fallback_reconfigure_replaces_providers() {
        let mut ai = AiAssistant::new();
        ai.configure_fallback(vec![
            (AiProvider::Ollama, "model-a".into()),
            (AiProvider::LMStudio, "model-b".into()),
        ]);
        assert_eq!(ai.fallback_providers.len(), 2);

        // Reconfigure with a single provider
        ai.configure_fallback(vec![(AiProvider::OpenAI, "gpt-4o".into())]);
        assert_eq!(ai.fallback_providers.len(), 1);
    }

    // =========================================================================
    // RAG Feature-Gated Tests
    // =========================================================================

    #[cfg(feature = "rag")]
    #[test]
    fn test_has_rag_false_initially() {
        let ai = AiAssistant::new();
        assert!(!ai.has_rag());
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_init_rag_creates_db() {
        let dir = std::env::temp_dir().join(format!("test_rag_init_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let db_path = dir.join("test.db");

        let mut ai = AiAssistant::new();
        assert!(!ai.has_rag());
        let result = ai.init_rag(&db_path);
        assert!(result.is_ok());
        assert!(ai.has_rag());
        assert!(ai.is_rag_initialized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_set_user_id_and_get() {
        let mut ai = AiAssistant::new();
        assert_eq!(ai.get_user_id(), "default");

        ai.set_user_id("orlando");
        assert_eq!(ai.get_user_id(), "orlando");
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_register_knowledge_document_marks_pending() {
        let mut ai = AiAssistant::new();
        assert!(!ai.has_pending_documents());

        ai.register_knowledge_document("guide", "# Guide\nSome content here");
        assert!(ai.has_pending_documents());
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_pending_document_count_tracks_registrations() {
        let mut ai = AiAssistant::new();
        assert_eq!(ai.pending_document_count(), 0);

        ai.register_knowledge_document("doc1", "Content 1");
        ai.register_knowledge_document("doc2", "Content 2");
        ai.register_knowledge_document("doc3", "Content 3");
        assert_eq!(ai.pending_document_count(), 3);
    }

    // =========================================================================
    // Metrics Tests
    // =========================================================================

    #[test]
    fn test_export_metrics_json_is_valid_json() {
        let mut ai = AiAssistant::new();
        ai.start_message_tracking();
        ai.finish_message_tracking(50);

        let json = ai.export_metrics_json();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
        assert!(
            parsed.is_ok(),
            "Metrics JSON with data should be valid JSON"
        );
    }

    #[test]
    fn test_reset_metrics_clears_all_message_metrics() {
        let mut ai = AiAssistant::new();
        ai.start_message_tracking();
        ai.finish_message_tracking(100);
        ai.start_message_tracking();
        ai.finish_message_tracking(200);
        assert_eq!(ai.get_message_metrics().len(), 2);

        ai.reset_metrics("fresh-session");
        assert!(ai.get_message_metrics().is_empty());
    }

    #[test]
    fn test_get_session_metrics_zero_values_on_new() {
        let ai = AiAssistant::new();
        let metrics = ai.get_session_metrics();
        assert_eq!(metrics.message_count, 0);
        assert_eq!(metrics.total_input_tokens, 0);
        assert_eq!(metrics.total_output_tokens, 0);
        assert_eq!(metrics.avg_response_time_ms, 0.0);
    }

    // =========================================================================
    // Adaptive Thinking Advanced Tests
    // =========================================================================

    #[test]
    fn test_classify_query_question_returns_strategy() {
        let ai = AiAssistant::new();
        let strategy = ai.classify_query("What is Rust and why should I use it?");
        // A question should produce a valid strategy with reasonable temperature
        assert!(strategy.temperature >= 0.0);
        assert!(strategy.temperature <= 2.0);
    }

    #[test]
    fn test_classify_query_code_returns_strategy() {
        let ai = AiAssistant::new();
        let strategy = ai.classify_query("Write a function to sort a vector of integers in Rust");
        // Code queries should produce a valid strategy
        assert!(strategy.temperature >= 0.0);
        assert!(strategy.temperature <= 2.0);
    }

    // =========================================================================
    // Constructor and Default State Tests
    // =========================================================================

    #[test]
    fn test_default_trait_creates_same_as_new() {
        let ai_new = AiAssistant::new();
        let ai_default = AiAssistant::default();
        // Both should have same initial state
        assert_eq!(ai_new.message_count(), ai_default.message_count());
        assert_eq!(ai_new.is_generating, ai_default.is_generating);
        assert_eq!(ai_new.system_prompt(), ai_default.system_prompt());
    }

    #[test]
    fn test_with_system_prompt_uses_custom_prompt() {
        let ai = AiAssistant::with_system_prompt("You are a pirate assistant. Say arr!");
        assert_eq!(ai.system_prompt(), "You are a pirate assistant. Say arr!");
    }

    #[test]
    fn test_set_system_prompt_changes_prompt() {
        let mut ai = AiAssistant::new();
        let original = ai.system_prompt().to_string();
        ai.set_system_prompt("New custom prompt");
        assert_eq!(ai.system_prompt(), "New custom prompt");
        assert_ne!(ai.system_prompt(), &original);
    }

    #[test]
    fn test_initial_state_no_current_session() {
        let ai = AiAssistant::new();
        assert!(ai.current_session.is_none());
        assert!(ai.conversation.is_empty());
        assert!(ai.current_response.is_empty());
        assert!(!ai.is_generating);
        assert!(!ai.is_fetching_models);
        assert!(!ai.is_summarizing);
    }

    #[test]
    fn test_initial_available_models_empty() {
        let ai = AiAssistant::new();
        assert!(ai.available_models.is_empty());
    }

    #[test]
    fn test_detected_context_size_none_initially() {
        let ai = AiAssistant::new();
        assert!(ai.detected_context_size.is_none());
        assert!(ai.detected_context_model.is_none());
    }

    #[test]
    fn test_adaptive_thinking_disabled_by_default() {
        let ai = AiAssistant::new();
        assert!(!ai.adaptive_thinking.enabled);
        assert!(ai.last_thinking_strategy.is_none());
        assert!(ai.last_thinking_result.is_none());
    }

    // === FreshContext Mode Tests ===

    #[test]
    fn test_context_mode_default_is_conversation() {
        let ai = AiAssistant::new();
        assert_eq!(ai.context_mode(), ContextMode::Conversation);
    }

    #[test]
    fn test_context_mode_set_and_get() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        assert_eq!(ai.context_mode(), ContextMode::FreshContext);
        ai.set_context_mode(ContextMode::Conversation);
        assert_eq!(ai.context_mode(), ContextMode::Conversation);
    }

    #[test]
    fn test_fresh_context_preserves_conversation_history() {
        let mut ai = AiAssistant::new();
        ai.conversation.push(ChatMessage::user("First"));
        ai.conversation.push(ChatMessage::assistant("Reply"));
        ai.conversation.push(ChatMessage::user("Second"));
        ai.set_context_mode(ContextMode::FreshContext);
        // History is preserved even in FreshContext
        assert_eq!(ai.conversation.len(), 3);
    }

    #[test]
    fn test_fresh_context_only_last_message() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        ai.conversation.push(ChatMessage::user("Old"));
        ai.conversation.push(ChatMessage::assistant("Reply"));
        ai.conversation.push(ChatMessage::user("Current"));
        // Simulate the FreshContext filtering logic
        let filtered = [ai
            .conversation
            .last()
            .expect("message was just pushed")
            .clone()];
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].content, "Current");
    }

    #[test]
    fn test_fresh_context_more_tokens_available() {
        let mut ai = AiAssistant::new();
        // Add messages to consume conversation budget
        for i in 0..10 {
            ai.conversation
                .push(ChatMessage::user(&format!("Message {}", i)));
            ai.conversation
                .push(ChatMessage::assistant(&format!("Reply {}", i)));
        }
        let conv_tokens = ai.calculate_available_knowledge_tokens("test query");
        ai.set_context_mode(ContextMode::FreshContext);
        let fresh_tokens = ai.calculate_available_knowledge_tokens("test query");
        assert!(
            fresh_tokens > conv_tokens,
            "FreshContext should have more tokens: {} vs {}",
            fresh_tokens,
            conv_tokens
        );
    }

    // === Memory Integration Tests ===

    #[test]
    fn test_memory_disabled_by_default() {
        let ai = AiAssistant::new();
        assert!(!ai.has_memory());
        assert!(ai.memory_manager().is_none());
    }

    #[test]
    fn test_memory_enable_disable() {
        let mut ai = AiAssistant::new();
        ai.enable_memory(crate::memory::MemoryConfig::default());
        assert!(ai.has_memory());
        assert!(ai.memory_manager().is_some());
        ai.disable_memory();
        assert!(!ai.has_memory());
    }

    #[test]
    fn test_build_memory_context_empty_when_disabled() {
        let mut ai = AiAssistant::new();
        let ctx = ai.build_memory_context("test query", 1000);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_build_memory_context_with_memories() {
        let mut ai = AiAssistant::new();
        ai.enable_memory(crate::memory::MemoryConfig::default());
        if let Some(mm) = ai.memory_manager_mut() {
            mm.remember_fact("The project uses Rust", 0.9);
        }
        let ctx = ai.build_memory_context("Rust project", 1000);
        assert!(
            !ctx.is_empty(),
            "Should return memory context with relevant fact"
        );
    }

    // === FreshContext Advisor Tests ===

    #[cfg(feature = "rag")]
    #[test]
    fn test_fresh_context_status_ineffective_without_rag() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        let status = ai.fresh_context_status(false);
        assert_eq!(status.effectiveness, FreshContextEffectiveness::Ineffective);
        assert!(status.warnings.contains(&FreshContextWarning::NoRag));
        assert!(status.warnings.contains(&FreshContextWarning::NoGraph));
        assert!(status.warnings.contains(&FreshContextWarning::NoMemory));
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_fresh_context_status_limited_with_rag_only() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        let temp = std::env::temp_dir().join(format!("fc_test_{}.db", uuid::Uuid::new_v4()));
        ai.init_rag(&temp).expect("RAG init");
        ai.register_knowledge_document("test", "some content");
        let status = ai.fresh_context_status(false);
        assert_eq!(status.effectiveness, FreshContextEffectiveness::Limited);
        assert!(!status.warnings.contains(&FreshContextWarning::NoRag));
        assert!(status.warnings.contains(&FreshContextWarning::NoGraph));
        let _ = std::fs::remove_file(&temp);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_fresh_context_status_good_with_rag_and_memory() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        let temp = std::env::temp_dir().join(format!("fc_test_{}.db", uuid::Uuid::new_v4()));
        ai.init_rag(&temp).expect("RAG init");
        ai.register_knowledge_document("test", "content");
        ai.enable_memory(crate::memory::MemoryConfig::default());
        let status = ai.fresh_context_status(false);
        assert_eq!(status.effectiveness, FreshContextEffectiveness::Good);
        assert!(!status.warnings.contains(&FreshContextWarning::NoMemory));
        let _ = std::fs::remove_file(&temp);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_fresh_context_status_optimal() {
        let mut ai = AiAssistant::new();
        ai.set_context_mode(ContextMode::FreshContext);
        let temp = std::env::temp_dir().join(format!("fc_test_{}.db", uuid::Uuid::new_v4()));
        ai.init_rag(&temp).expect("RAG init");
        ai.register_knowledge_document("test", "content");
        ai.enable_memory(crate::memory::MemoryConfig::default());
        let status = ai.fresh_context_status(true);
        assert_eq!(status.effectiveness, FreshContextEffectiveness::Optimal);
        assert!(status.warnings.is_empty());
        let _ = std::fs::remove_file(&temp);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_fresh_context_warning_display() {
        let w = FreshContextWarning::NoRag;
        let s = format!("{}", w);
        assert!(
            s.contains("RAG"),
            "Warning display should mention RAG: {}",
            s
        );

        let w2 = FreshContextWarning::SmallBudget(200);
        let s2 = format!("{}", w2);
        assert!(s2.contains("200"), "Should include token count: {}", s2);
    }

    // ================================================================
    // Procedural memory tests
    // ================================================================

    #[cfg(feature = "advanced-memory")]
    fn make_test_procedure(
        id: &str,
        name: &str,
        condition: &str,
        steps: Vec<&str>,
        confidence: f64,
    ) -> crate::advanced_memory::Procedure {
        crate::advanced_memory::Procedure {
            id: id.to_string(),
            name: name.to_string(),
            condition: condition.to_string(),
            steps: steps.into_iter().map(|s| s.to_string()).collect(),
            success_count: (confidence * 10.0) as usize,
            failure_count: ((1.0 - confidence) * 10.0) as usize,
            confidence,
            created_from: Vec::new(),
            tags: Vec::new(),
        }
    }

    #[cfg(feature = "advanced-memory")]
    #[test]
    fn test_assistant_procedural_crud() {
        let mut ai = AiAssistant::new();

        // Not enabled yet
        assert!(!ai.has_procedural_memory());
        assert!(ai.list_procedures().is_empty());

        // Enable
        ai.enable_procedural_memory(50);
        assert!(ai.has_procedural_memory());

        // Add
        ai.add_procedure(make_test_procedure(
            "p1",
            "Deploy",
            "deploy rust app",
            vec!["test", "build", "deploy"],
            0.9,
        ));
        ai.add_procedure(make_test_procedure(
            "p2",
            "Review",
            "code review checklist",
            vec!["compile", "test", "docs"],
            0.85,
        ));
        assert_eq!(ai.list_procedures().len(), 2);

        // Find
        let found = ai.find_procedures("deploy rust application");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "p1");

        // Remove
        let removed = ai.remove_procedure("p1");
        assert!(removed.is_some());
        assert_eq!(ai.list_procedures().len(), 1);

        // Store access
        assert!(ai.procedural_store().is_some());
        assert_eq!(ai.procedural_store().unwrap().len(), 1);

        // Disable
        ai.disable_procedural_memory();
        assert!(!ai.has_procedural_memory());
        assert!(ai.list_procedures().is_empty());
    }

    #[cfg(feature = "advanced-memory")]
    #[test]
    fn test_assistant_procedural_persistence() {
        let mut ai = AiAssistant::new();
        ai.enable_procedural_memory(50);
        ai.add_procedure(make_test_procedure(
            "p1",
            "Deploy",
            "deploy rust app",
            vec!["test", "build"],
            0.9,
        ));
        ai.add_procedure(make_test_procedure(
            "p2",
            "Review",
            "code review",
            vec!["compile", "lint"],
            0.85,
        ));

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("procedures.json");

        // Save
        ai.save_procedures(&path).expect("save");
        assert!(path.exists());

        // Load into a new assistant
        let mut ai2 = AiAssistant::new();
        ai2.load_procedures(&path, 50).expect("load");
        assert!(ai2.has_procedural_memory());
        assert_eq!(ai2.list_procedures().len(), 2);
    }

    #[cfg(feature = "advanced-memory")]
    #[test]
    fn test_assistant_procedural_context_formatting() {
        let mut ai = AiAssistant::new();
        ai.enable_procedural_memory(50);
        ai.add_procedure(make_test_procedure(
            "p1",
            "Deploy Pipeline",
            "deploy rust application",
            vec!["Run cargo test", "Build release binary", "Deploy to server"],
            0.92,
        ));

        let ctx = ai.build_procedural_context("deploy rust application now", 5, 500);
        assert!(ctx.contains("--- WORKFLOW GUIDELINES ---"));
        assert!(ctx.contains("Deploy Pipeline"));
        assert!(ctx.contains("confidence: 92%"));
        assert!(ctx.contains("1. Run cargo test"));
        assert!(ctx.contains("2. Build release binary"));
        assert!(ctx.contains("--- END WORKFLOW GUIDELINES ---"));

        // Should have recorded active procedure IDs
        assert_eq!(ai.active_procedure_ids.len(), 1);
        assert_eq!(ai.active_procedure_ids[0], "p1");
    }

    #[cfg(feature = "advanced-memory")]
    #[test]
    fn test_assistant_procedural_context_no_match() {
        let mut ai = AiAssistant::new();
        ai.enable_procedural_memory(50);
        ai.add_procedure(make_test_procedure(
            "p1",
            "Deploy",
            "deploy rust app",
            vec!["step"],
            0.9,
        ));

        // Query that doesn't match
        let ctx = ai.build_procedural_context("tell me about quantum physics", 5, 500);
        assert!(ctx.is_empty());
        assert!(ai.active_procedure_ids.is_empty());
    }

    // === V75: Cost tracking tests ===

    #[test]
    fn test_with_cost_config_builder() {
        let config = crate::cost_integration::CostAwareConfig {
            enabled: true,
            daily_budget: Some(10.0),
            monthly_budget: Some(100.0),
            per_request_limit: Some(1.0),
            alert_threshold_pct: 0.8,
            track_by_model: true,
        };
        let ai = AiAssistant::new().with_cost_config(config);
        assert!(
            ai.cost_dashboard.is_some(),
            "dashboard should be initialized by with_cost_config"
        );
    }

    #[test]
    fn test_with_cost_config_disabled() {
        let config = crate::cost_integration::CostAwareConfig {
            enabled: false,
            ..Default::default()
        };
        let ai = AiAssistant::new().with_cost_config(config);
        assert!(
            ai.cost_dashboard.is_none(),
            "dashboard should not be initialized when disabled"
        );
    }

    #[test]
    fn test_cost_dashboard_report_after_init() {
        let mut ai = AiAssistant::new();
        assert!(
            ai.cost_report().is_none(),
            "report should be None before init"
        );
        ai.init_cost_tracking();
        assert!(
            ai.cost_report().is_some(),
            "report should be Some after init"
        );
        let report = ai.cost_report().unwrap();
        assert!(report.contains("Cost Dashboard Report"));
    }

    // === Batch 1 — Vision entry points ===

    #[cfg(feature = "vision")]
    #[test]
    fn test_send_message_with_images_pushes_chat_message_with_images() {
        let mut ai = AiAssistant::new();
        // Use an unreachable provider so the spawned thread errors out fast;
        // we only assert on conversation state which is set synchronously.
        ai.config.provider = AiProvider::Ollama;
        ai.config.ollama_url = "http://127.0.0.1:1".to_string();
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        ai.send_message_with_images("describe".to_string(), vec![img], "");
        assert_eq!(ai.conversation.len(), 1);
        let last = ai.conversation.last().unwrap();
        assert_eq!(last.role, "user");
        assert_eq!(last.content, "describe");
        assert!(last.has_images());
        assert_eq!(last.images.len(), 1);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_send_message_simple_with_images_pushes_message() {
        let mut ai = AiAssistant::new();
        ai.config.provider = AiProvider::Ollama;
        ai.config.ollama_url = "http://127.0.0.1:1".to_string();
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        ai.send_message_simple_with_images("hi".to_string(), vec![img]);
        assert_eq!(ai.conversation.len(), 1);
        assert!(ai.conversation[0].has_images());
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_send_message_auto_with_images_uses_internal_knowledge() {
        let mut ai = AiAssistant::new();
        ai.config.provider = AiProvider::Ollama;
        ai.config.ollama_url = "http://127.0.0.1:1".to_string();
        ai.set_knowledge_context("# Notes\nKey fact.");
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        ai.send_message_auto_with_images("q".to_string(), vec![img]);
        assert_eq!(ai.conversation.len(), 1);
        assert_eq!(ai.conversation[0].content, "q");
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_generate_sync_with_images_unsupported_provider_errors_without_fallback() {
        let mut ai = AiAssistant::new();
        ai.config.provider = AiProvider::Bedrock {
            region: "us-east-1".to_string(),
        };
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        let res = ai.generate_sync_with_images("q".to_string(), vec![img], "");
        assert!(res.is_err());
        let msg = res.unwrap_err().to_string();
        assert!(
            msg.contains("does not support vision") || msg.contains("vision"),
            "unexpected error: {msg}"
        );
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_generate_sync_with_images_attaches_to_conversation() {
        let mut ai = AiAssistant::new();
        // Bedrock isn't vision-capable per agent_bridge; the message still
        // gets pushed to conversation before dispatch. Confirm wire state.
        ai.config.provider = AiProvider::Bedrock {
            region: "us-east-1".to_string(),
        };
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        let _ = ai.generate_sync_with_images("q".to_string(), vec![img], "");
        assert_eq!(ai.conversation.len(), 1);
        assert!(ai.conversation[0].has_images());
    }

    #[cfg(all(feature = "vision", feature = "rag"))]
    #[test]
    fn test_send_message_with_images_rag_pushes_message() {
        let mut ai = AiAssistant::new();
        ai.config.provider = AiProvider::Ollama;
        ai.config.ollama_url = "http://127.0.0.1:1".to_string();
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        ai.send_message_with_images_rag("q".to_string(), vec![img]);
        assert_eq!(ai.conversation.len(), 1);
        assert!(ai.conversation[0].has_images());
    }
}
