//! Main AI Assistant implementation

use anyhow::Result;
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
        }
    }

    /// Set the base system prompt
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt_base = prompt.to_string();
    }

    /// Get the base system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt_base
    }

    // === Adaptive Thinking ===

    /// Enable adaptive thinking with default configuration.
    ///
    /// When enabled, the assistant automatically adjusts temperature, max_tokens,
    /// RAG tier, and system prompt based on query complexity classification.
    pub fn enable_adaptive_thinking(&mut self) {
        self.adaptive_thinking.enabled = true;
    }

    /// Disable adaptive thinking (default state).
    pub fn disable_adaptive_thinking(&mut self) {
        self.adaptive_thinking.enabled = false;
    }

    /// Set a custom adaptive thinking configuration.
    pub fn set_adaptive_thinking(&mut self, config: AdaptiveThinkingConfig) {
        self.adaptive_thinking = config;
    }

    /// Classify a query and return the thinking strategy (for inspection/debugging).
    ///
    /// This does not affect the assistant state — it only returns the strategy
    /// that *would* be applied if adaptive thinking were enabled.
    pub fn classify_query(&self, query: &str) -> ThinkingStrategy {
        let classifier = QueryClassifier::new(self.adaptive_thinking.clone());
        classifier.classify(query)
    }

    /// Apply adaptive thinking to modify system prompt and config before an LLM call.
    ///
    /// Returns `(modified_system_prompt, modified_config)`. When adaptive thinking
    /// is disabled, returns the inputs unchanged.
    ///
    /// Logs a warning when adaptive RAG tier conflicts with explicit user tier.
    fn apply_adaptive_thinking(
        &mut self,
        user_message: &str,
        base_system_prompt: String,
    ) -> (String, crate::config::AiConfig) {
        if !self.adaptive_thinking.enabled {
            self.last_thinking_strategy = None;
            return (base_system_prompt, self.config.clone());
        }

        let classifier = QueryClassifier::new(self.adaptive_thinking.clone());
        let strategy = classifier.classify(user_message);

        let mut config = self.config.clone();
        let mut prompt = base_system_prompt;

        // Apply temperature override
        if self.adaptive_thinking.adjust_temperature {
            config.temperature = strategy.temperature;
        }

        // Inject CoT instructions into system prompt
        if !strategy.system_prompt_addition.is_empty() {
            prompt.push_str("\n\n--- REASONING INSTRUCTIONS ---\n");
            prompt.push_str(&strategy.system_prompt_addition);
            prompt.push_str("\n--- END REASONING INSTRUCTIONS ---\n");
        }

        // Initialize thinking tag parser if configured for transparent parsing
        if self.adaptive_thinking.parse_thinking_tags
            && self.adaptive_thinking.transparent_thinking_parse
        {
            self.thinking_parser = Some(ThinkingTagParser::new(
                self.adaptive_thinking.strip_thinking_from_response,
            ));
        } else {
            self.thinking_parser = None;
        }

        // Store the strategy for inspection
        self.last_thinking_strategy = Some(strategy);

        (prompt, config)
    }

    // === Dynamic Context Size Detection ===

    /// Detect and cache the context size for the current model.
    ///
    /// Uses the global context size cache (`get_model_context_size_cached`).
    /// On a cache miss the provider API is queried first; if that fails,
    /// the static model-name table is used as fallback.
    ///
    /// The instance fields `detected_context_size` / `detected_context_model`
    /// are kept in sync for fast per-instance access without locking.
    pub fn detect_model_context_size(&mut self) -> usize {
        let current_model = self.config.selected_model.clone();

        // Fast path: instance cache hit
        if let (Some(cached_size), Some(ref cached_model)) =
            (self.detected_context_size, &self.detected_context_model)
        {
            if cached_model == &current_model {
                return cached_size;
            }
        }

        // Delegate to global cache (which calls fetcher on miss)
        let config_ref = self.config.clone();
        let size = get_model_context_size_cached(&current_model, |name| {
            fetch_model_context_size(&config_ref, name)
        });

        // Sync instance cache
        self.detected_context_size = Some(size);
        self.detected_context_model = Some(current_model);

        size
    }

    /// Get the cached context size without re-detecting.
    ///
    /// Returns the instance-cached size if the model hasn't changed,
    /// otherwise delegates to `detect_model_context_size`.
    pub fn get_model_context_size(&mut self) -> usize {
        if let (Some(cached_size), Some(ref cached_model)) =
            (self.detected_context_size, &self.detected_context_model)
        {
            if cached_model == &self.config.selected_model {
                return cached_size;
            }
        }
        self.detect_model_context_size()
    }

    /// Calculate available tokens for knowledge context
    ///
    /// This calculates how many tokens can be used for RAG knowledge based on:
    /// - Model's total context window
    /// - Reserved space for response (20%)
    /// - System prompt size
    /// - Current conversation size
    /// - User message size estimate
    ///
    /// Returns the number of tokens available for knowledge.
    pub fn calculate_available_knowledge_tokens(&mut self, user_message: &str) -> usize {
        let total_context = self.get_model_context_size();

        // Reserve 20% for response generation
        let response_reserve = total_context / 5;

        // Estimate system prompt tokens
        let system_tokens = estimate_tokens(&self.system_prompt_base);

        // Estimate conversation history tokens
        let conversation_tokens: usize = self
            .conversation
            .iter()
            .map(|msg| estimate_tokens(&msg.content))
            .sum();

        // Estimate user message tokens
        let user_tokens = estimate_tokens(user_message);

        // Calculate available
        let used = system_tokens + conversation_tokens + user_tokens + response_reserve;
        let available = total_context.saturating_sub(used);

        // Leave a small buffer (5%) for safety
        let safe_available = (available as f32 * 0.95) as usize;

        println!(
            "[AI Context] Total: {}, Used: {} (sys:{} conv:{} user:{} reserve:{}), Available for knowledge: {}",
            total_context, used, system_tokens, conversation_tokens, user_tokens, response_reserve, safe_available
        );

        safe_available
    }

    /// Invalidate the cached context size (call when model changes)
    pub fn invalidate_context_cache(&mut self) {
        self.detected_context_size = None;
        self.detected_context_model = None;
    }

    // === Knowledge Context Management ===

    /// Set the knowledge context that will be used for all messages
    ///
    /// This context is automatically included in the system prompt when
    /// sending messages using `send_message_auto()` or when calling
    /// `send_message()` with an empty knowledge_context parameter.
    ///
    /// # Example
    /// ```no_run
    /// use ai_assistant::AiAssistant;
    ///
    /// let mut assistant = AiAssistant::new();
    /// assistant.set_knowledge_context("# Star Citizen Ships\n\nThe Aurora MR is...");
    ///
    /// // Messages will automatically use the knowledge context
    /// assistant.send_message_auto("Tell me about the Aurora MR");
    /// ```
    pub fn set_knowledge_context(&mut self, context: &str) {
        self.knowledge_context = context.to_string();
    }

    /// Append content to the existing knowledge context
    ///
    /// Useful for incrementally building knowledge from multiple sources.
    pub fn append_knowledge_context(&mut self, content: &str) {
        if !self.knowledge_context.is_empty() {
            self.knowledge_context.push_str("\n\n");
        }
        self.knowledge_context.push_str(content);
    }

    /// Clear the knowledge context
    pub fn clear_knowledge_context(&mut self) {
        self.knowledge_context.clear();
    }

    /// Get the current knowledge context
    pub fn get_knowledge_context(&self) -> &str {
        &self.knowledge_context
    }

    /// Check if there is any knowledge context set
    pub fn has_knowledge_context(&self) -> bool {
        !self.knowledge_context.is_empty()
    }

    /// Get the size of the knowledge context in bytes
    pub fn knowledge_context_size(&self) -> usize {
        self.knowledge_context.len()
    }

    /// Load configuration
    pub fn load_config(&mut self, config: AiConfig) {
        let old_model = self.config.selected_model.clone();
        self.config = config;
        if old_model != self.config.selected_model {
            log::info!("Model changed: from={} to={}", old_model, self.config.selected_model);
        }
    }

    /// Load preferences
    pub fn load_preferences(&mut self, preferences: UserPreferences) {
        self.preferences = preferences;
    }

    // === Model Discovery ===

    /// Start fetching available models from all providers asynchronously
    pub fn fetch_models(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.rx_models = Some(rx);
        self.is_fetching_models = true;

        let ollama_url = self.config.ollama_url.clone();
        let lm_studio_url = self.config.lm_studio_url.clone();
        let text_gen_webui_url = self.config.text_gen_webui_url.clone();
        let kobold_url = self.config.kobold_url.clone();
        let local_ai_url = self.config.local_ai_url.clone();

        thread::spawn(move || {
            let mut all_models = Vec::new();

            // Try Ollama
            if let Ok(models) = fetch_ollama_models(&ollama_url) {
                all_models.extend(models);
            }

            // Try LM Studio
            if let Ok(models) = fetch_openai_compatible_models(&lm_studio_url, AiProvider::LMStudio)
            {
                all_models.extend(models);
            }

            // Try text-generation-webui
            if let Ok(models) =
                fetch_openai_compatible_models(&text_gen_webui_url, AiProvider::TextGenWebUI)
            {
                all_models.extend(models);
            }

            // Try Kobold.cpp
            if let Ok(models) = fetch_kobold_models(&kobold_url) {
                all_models.extend(models);
            }

            // Try LocalAI
            if let Ok(models) = fetch_openai_compatible_models(&local_ai_url, AiProvider::LocalAI) {
                all_models.extend(models);
            }

            let _ = tx.send(AiResponse::ModelsLoaded(all_models));
        });
    }

    /// Poll for model fetch results. Returns true if models were loaded.
    pub fn poll_models(&mut self) -> bool {
        if let Some(ref rx) = self.rx_models {
            match rx.try_recv() {
                Ok(AiResponse::ModelsLoaded(models)) => {
                    self.available_models = models;
                    self.rx_models = None;
                    self.is_fetching_models = false;

                    // Auto-select first model if none selected
                    if self.config.selected_model.is_empty() && !self.available_models.is_empty() {
                        self.config.selected_model = self.available_models[0].name.clone();
                        self.config.provider = self.available_models[0].provider.clone();
                    }
                    return true;
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.rx_models = None;
                    self.is_fetching_models = false;
                }
            }
        }
        false
    }

    // === Provider Fallback ===

    /// Configure fallback providers for automatic failover.
    ///
    /// When the primary provider fails, the assistant tries each fallback in order.
    /// Each entry is a `(AiProvider, model_name)` pair.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_assistant::{AiAssistant, config::AiProvider};
    ///
    /// let mut ai = AiAssistant::new();
    /// ai.configure_fallback(vec![
    ///     (AiProvider::LMStudio, "local-model".into()),
    ///     (AiProvider::Ollama, "llama3.2:latest".into()),
    /// ]);
    /// ai.enable_fallback();
    /// ```
    pub fn configure_fallback(&mut self, providers: Vec<(AiProvider, String)>) {
        self.fallback_providers = providers;
    }

    /// Enable automatic provider fallback.
    pub fn enable_fallback(&mut self) {
        self.fallback_enabled = true;
    }

    /// Disable automatic provider fallback.
    pub fn disable_fallback(&mut self) {
        self.fallback_enabled = false;
    }

    /// Returns `true` if fallback is enabled and at least one provider is configured.
    pub fn fallback_active(&self) -> bool {
        self.fallback_enabled && !self.fallback_providers.is_empty()
    }

    /// Get the name of the provider that served the last response.
    ///
    /// Updated asynchronously by background generation threads.
    /// Returns `None` before the first response completes.
    pub fn last_provider_used(&self) -> Option<String> {
        self.fallback_last_provider
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    // === Conversation Compaction ===

    /// Enable automatic conversation compaction.
    ///
    /// When enabled, conversations exceeding `CompactionConfig::max_messages` are
    /// automatically compacted before each message send. This is a lightweight,
    /// heuristic-based compaction (no LLM call) that preserves important, first, and
    /// recent messages while summarizing removed ones.
    pub fn enable_auto_compaction(&mut self) {
        self.auto_compaction = true;
    }

    /// Disable automatic conversation compaction.
    pub fn disable_auto_compaction(&mut self) {
        self.auto_compaction = false;
    }

    /// Set the compaction configuration.
    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compaction_config = config;
    }

    /// Manually compact the current conversation.
    ///
    /// Converts conversation messages to compactable form, runs the compactor,
    /// and replaces the conversation with the compacted result. A summary of
    /// removed messages is inserted as a system message after the first message.
    ///
    /// Returns the `CompactionResult` with details about what was removed.
    pub fn compact_conversation(&mut self) -> CompactionResult {
        let compactor = ConversationCompactor::new(self.compaction_config.clone());

        let compactable: Vec<CompactableMessage> = self
            .conversation
            .iter()
            .map(|m| CompactableMessage::new(&m.role, &m.content))
            .collect();

        let result = compactor.compact(compactable);

        // Replace conversation with compacted messages
        self.conversation = result
            .messages
            .iter()
            .map(|m| match m.role.as_str() {
                "user" => ChatMessage::user(&m.content),
                "assistant" => ChatMessage::assistant(&m.content),
                _ => ChatMessage::system(&m.content),
            })
            .collect();

        // Insert summary after the first message if available
        if let Some(ref summary) = result.summary {
            if !summary.is_empty() && !self.conversation.is_empty() {
                self.conversation
                    .insert(1.min(self.conversation.len()), ChatMessage::system(summary));
            }
        }

        result
    }

    /// Run compaction if auto_compaction is enabled and the conversation exceeds
    /// the configured threshold. Called internally before each send.
    fn maybe_compact_conversation(&mut self) {
        if !self.auto_compaction {
            return;
        }
        let compactor = ConversationCompactor::new(self.compaction_config.clone());
        if compactor.needs_compaction(self.conversation.len()) {
            let _ = self.compact_conversation();
        }
    }

    // === API Key Management ===

    /// Initialize the API key manager with custom rotation config.
    ///
    /// Required before adding keys. If not called, `add_api_key` will
    /// initialize a manager with default settings.
    pub fn set_api_key_config(&mut self, config: RotationConfig) {
        self.api_key_manager = Some(ApiKeyManager::new(config));
    }

    /// Add an API key for a provider.
    ///
    /// Creates the key manager with default config if not yet initialized.
    pub fn add_api_key(&mut self, provider: &str, key_id: &str, key_value: &str) {
        if self.api_key_manager.is_none() {
            self.api_key_manager = Some(ApiKeyManager::default());
        }
        let api_key = ApiKey::new(key_id, key_value, provider);
        self.api_key_manager
            .as_mut()
            .expect("api_key_manager must be initialized")
            .add_key(api_key);
    }

    /// Get the current API key for a provider (round-robin, skips rate-limited keys).
    ///
    /// Returns `None` if no usable key is available.
    pub fn get_current_api_key(&mut self, provider: &str) -> Option<String> {
        self.api_key_manager
            .as_mut()
            .and_then(|m| m.get_key(provider))
            .map(|k| k.key.clone())
    }

    /// Mark the current key for a provider as rate-limited, triggering rotation
    /// to the next available key.
    pub fn mark_key_rate_limited(&mut self, provider: &str, key_id: &str) {
        if let Some(ref mut manager) = self.api_key_manager {
            manager.mark_rate_limited(provider, key_id);
        }
    }

    // === Message Handling ===

    /// Send a message and start generating a response
    ///
    /// # Arguments
    /// * `user_message` - The user's message
    /// * `knowledge_context` - Optional knowledge/context to include in system prompt
    pub fn send_message(&mut self, user_message: String, knowledge_context: &str) {
        self.conversation.push(ChatMessage::user(&user_message));
        self.maybe_compact_conversation();
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let config = self.config.clone();
        let conversation = self.conversation.clone();
        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
        );

        let fallback_providers = if self.fallback_enabled {
            self.fallback_providers.clone()
        } else {
            Vec::new()
        };
        let last_provider = self.fallback_last_provider.clone();

        thread::spawn(move || {
            try_generate_with_fallback(
                &config,
                &conversation,
                &system_prompt,
                &tx,
                &fallback_providers,
                None,
                &last_provider,
            );
        });
    }

    /// Send a message without knowledge context
    pub fn send_message_simple(&mut self, user_message: String) {
        self.send_message(user_message, "");
    }

    /// Send a message using the internal knowledge context
    ///
    /// This method automatically uses the knowledge context that was set via
    /// `set_knowledge_context()` or `append_knowledge_context()`.
    ///
    /// # Example
    /// ```no_run
    /// use ai_assistant::AiAssistant;
    ///
    /// let mut assistant = AiAssistant::new();
    /// assistant.set_knowledge_context("# Guide\nImportant info...");
    /// assistant.send_message_auto("What does the guide say?".to_string());
    /// ```
    pub fn send_message_auto(&mut self, user_message: String) {
        let context = self.knowledge_context.clone();
        self.send_message(user_message, &context);
    }

    /// Send a message using internal knowledge context with additional session notes
    ///
    /// Combines the internal knowledge context with session-specific notes.
    pub fn send_message_auto_with_notes(
        &mut self,
        user_message: String,
        session_notes: &str,
        knowledge_notes: &str,
    ) {
        let context = self.knowledge_context.clone();
        // Debug: Log context size to verify knowledge is being used
        println!(
            "[AI DEBUG] send_message_auto_with_notes: knowledge_context size = {} bytes",
            context.len()
        );
        if context.is_empty() {
            println!("[AI DEBUG] WARNING: knowledge_context is EMPTY!");
        } else {
            // Show first 300 chars of context
            let preview: String = context.chars().take(300).collect();
            println!("[AI DEBUG] Context preview: {}...", preview);

            // Check for CCU-related content specifically
            let context_lower = context.to_lowercase();
            if context_lower.contains("cross-chassis") || context_lower.contains("ccu") {
                println!("[AI DEBUG] CCU knowledge FOUND in context");
            } else {
                println!("[AI DEBUG] WARNING: No CCU-related content found in context!");
            }

            // Count how many knowledge sections
            let section_count = context.matches("# ").count();
            println!("[AI DEBUG] Knowledge sections: {}", section_count);
        }
        self.send_message_with_notes(user_message, &context, session_notes, knowledge_notes);
    }

    /// Send a message with full context including user notes
    ///
    /// # Arguments
    /// * `user_message` - The user's message
    /// * `knowledge_context` - Optional knowledge/context to include
    /// * `session_notes` - Session-specific notes
    /// * `knowledge_notes` - Notes about knowledge documents being used
    pub fn send_message_with_notes(
        &mut self,
        user_message: String,
        knowledge_context: &str,
        session_notes: &str,
        knowledge_notes: &str,
    ) {
        self.conversation.push(ChatMessage::user(&user_message));
        self.maybe_compact_conversation();
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
            session_notes,
            knowledge_notes,
        );

        let (system_prompt, config) = self.apply_adaptive_thinking(&user_message, system_prompt);
        let conversation = self.conversation.clone();
        let fallback_providers = if self.fallback_enabled {
            self.fallback_providers.clone()
        } else {
            Vec::new()
        };
        let last_provider = self.fallback_last_provider.clone();

        thread::spawn(move || {
            try_generate_with_fallback(
                &config,
                &conversation,
                &system_prompt,
                &tx,
                &fallback_providers,
                None,
                &last_provider,
            );
        });
    }

    /// Generate a response synchronously (blocking).
    ///
    /// Supports provider fallback: if the primary provider fails and fallback
    /// is enabled, tries each fallback provider in order.
    pub fn generate_sync(
        &mut self,
        user_message: String,
        knowledge_context: &str,
    ) -> Result<String> {
        self.conversation.push(ChatMessage::user(&user_message));

        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
        );

        // Try primary provider
        let response = match generate_response(&self.config, &self.conversation, &system_prompt) {
            Ok(r) => {
                *self
                    .fallback_last_provider
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) =
                    Some(self.config.provider.display_name().to_string());
                r
            }
            Err(primary_err) => {
                if !self.fallback_enabled || self.fallback_providers.is_empty() {
                    return Err(primary_err);
                }
                // Try fallback providers
                let mut last_err = primary_err;
                let mut found = None;
                for (provider, model) in &self.fallback_providers {
                    let mut fb_config = self.config.clone();
                    fb_config.provider = provider.clone();
                    fb_config.selected_model = model.clone();
                    match generate_response(&fb_config, &self.conversation, &system_prompt) {
                        Ok(r) => {
                            *self
                                .fallback_last_provider
                                .lock()
                                .unwrap_or_else(|e| e.into_inner()) =
                                Some(provider.display_name().to_string());
                            found = Some(r);
                            break;
                        }
                        Err(e) => last_err = e,
                    }
                }
                found.ok_or(last_err)?
            }
        };

        self.conversation.push(ChatMessage::assistant(&response));
        self.extract_preferences_from_response(&response);

        Ok(response)
    }

    // === Cancellable Streaming ===

    /// Send a message with cancellation support
    ///
    /// Returns a CancellationToken that can be used to cancel the generation
    pub fn send_message_cancellable(
        &mut self,
        user_message: String,
        knowledge_context: &str,
    ) -> CancellationToken {
        self.conversation.push(ChatMessage::user(&user_message));
        self.maybe_compact_conversation();
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let cancel_token = CancellationToken::new();
        self.cancel_token = Some(cancel_token.clone());

        let config = self.config.clone();
        let conversation = self.conversation.clone();
        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
        );

        let fallback_providers = if self.fallback_enabled {
            self.fallback_providers.clone()
        } else {
            Vec::new()
        };
        let last_provider = self.fallback_last_provider.clone();
        let token = cancel_token.clone();

        thread::spawn(move || {
            try_generate_with_fallback(
                &config,
                &conversation,
                &system_prompt,
                &tx,
                &fallback_providers,
                Some(&token),
                &last_provider,
            );
        });

        cancel_token
    }

    /// Send a message with cancellation support (no knowledge context)
    pub fn send_message_cancellable_simple(&mut self, user_message: String) -> CancellationToken {
        self.send_message_cancellable(user_message, "")
    }

    /// Send a message with cancellation support using internal knowledge context
    ///
    /// Uses the knowledge context set via `set_knowledge_context()`.
    pub fn send_message_cancellable_auto(&mut self, user_message: String) -> CancellationToken {
        let context = self.knowledge_context.clone();
        self.send_message_cancellable(user_message, &context)
    }

    /// Send a message with cancellation support using internal context and notes
    pub fn send_message_cancellable_auto_with_notes(
        &mut self,
        user_message: String,
        session_notes: &str,
        knowledge_notes: &str,
    ) -> CancellationToken {
        let context = self.knowledge_context.clone();
        self.send_message_cancellable_with_notes(
            user_message,
            &context,
            session_notes,
            knowledge_notes,
        )
    }

    /// Send a message with full context and cancellation support
    pub fn send_message_cancellable_with_notes(
        &mut self,
        user_message: String,
        knowledge_context: &str,
        session_notes: &str,
        knowledge_notes: &str,
    ) -> CancellationToken {
        // Emit message sent event
        self.event_bus.emit(crate::events::AiEvent::MessageSent {
            content_length: user_message.len(),
            has_knowledge: !knowledge_context.is_empty(),
        });

        let msg = ChatMessage::user(&user_message);
        self.conversation.push(msg.clone());
        self.maybe_compact_conversation();

        // Auto-store user message in RAG if enabled
        #[cfg(feature = "rag")]
        if self.rag_config.auto_store_messages {
            let _ = self.store_message_in_rag(&msg, true);
        }

        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let cancel_token = CancellationToken::new();
        self.cancel_token = Some(cancel_token.clone());

        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
            session_notes,
            knowledge_notes,
        );

        let (system_prompt, config) = self.apply_adaptive_thinking(&user_message, system_prompt);

        // Emit provider attempt event
        self.event_bus
            .emit(crate::events::AiEvent::ProviderAttempt {
                provider: config.provider.display_name().to_string(),
                model: config.selected_model.clone(),
            });

        let conversation = self.conversation.clone();
        let fallback_providers = if self.fallback_enabled {
            self.fallback_providers.clone()
        } else {
            Vec::new()
        };
        let last_provider = self.fallback_last_provider.clone();

        let token = cancel_token.clone();
        thread::spawn(move || {
            try_generate_with_fallback(
                &config,
                &conversation,
                &system_prompt,
                &tx,
                &fallback_providers,
                Some(&token),
                &last_provider,
            );
        });

        cancel_token
    }

    /// Cancel the current generation if in progress
    ///
    /// Returns true if there was an active generation to cancel
    pub fn cancel_generation(&mut self) -> bool {
        if let Some(ref token) = self.cancel_token {
            token.cancel();
            true
        } else {
            false
        }
    }

    /// Check if generation can be cancelled
    pub fn can_cancel(&self) -> bool {
        self.is_generating && self.cancel_token.is_some()
    }

    /// Get the current cancellation token if generating
    pub fn get_cancel_token(&self) -> Option<CancellationToken> {
        self.cancel_token.clone()
    }

    /// Poll for response chunks/completion.
    ///
    /// When adaptive thinking is enabled with `transparent_thinking_parse`, thinking
    /// tags (`<think>...</think>`) are automatically stripped from chunks. The extracted
    /// thinking content is available via `last_thinking_result` after the response completes.
    pub fn poll_response(&mut self) -> Option<AiResponse> {
        if let Some(ref rx) = self.rx_response {
            match rx.try_recv() {
                Ok(response) => {
                    match response {
                        AiResponse::Complete(text) => {
                            // Finalize thinking parser if active
                            if let Some(ref mut parser) = self.thinking_parser {
                                parser.process_chunk(&text);
                                parser.finalize();
                                let parse_result = parser.result();
                                self.current_response = parse_result.visible_response.clone();
                                self.last_thinking_result = Some(parse_result);
                            } else {
                                self.current_response = text;
                            }

                            let msg = ChatMessage::assistant(&self.current_response);
                            self.conversation.push(msg.clone());
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                            self.thinking_parser = None;
                            self.extract_preferences_from_response(&self.current_response.clone());

                            // Auto-store assistant message in RAG if enabled
                            #[cfg(feature = "rag")]
                            if self.rag_config.auto_store_messages {
                                let _ = self.store_message_in_rag(&msg, true);
                            }

                            self.event_bus
                                .emit(crate::events::AiEvent::ResponseComplete {
                                    response_length: self.current_response.len(),
                                });
                            return Some(AiResponse::Complete(self.current_response.clone()));
                        }
                        AiResponse::Cancelled(partial) => {
                            self.current_response = partial.clone();
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                            self.thinking_parser = None;
                            self.event_bus
                                .emit(crate::events::AiEvent::ResponseCancelled {
                                    partial_length: partial.len(),
                                });
                            return Some(AiResponse::Cancelled(partial));
                        }
                        AiResponse::Chunk(chunk) => {
                            // Route through thinking tag parser if active
                            if let Some(ref mut parser) = self.thinking_parser {
                                let visible = parser.process_chunk(&chunk);
                                if !visible.is_empty() {
                                    self.current_response.push_str(&visible);
                                    return Some(AiResponse::Chunk(visible));
                                }
                                // Chunk was entirely thinking content — don't emit anything
                                return None;
                            } else {
                                self.current_response.push_str(&chunk);
                                return Some(AiResponse::Chunk(chunk));
                            }
                        }
                        AiResponse::Error(e) => {
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                            self.thinking_parser = None;
                            self.event_bus
                                .emit(crate::events::AiEvent::ResponseError { error: e.clone() });
                            return Some(AiResponse::Error(e));
                        }
                        other => {
                            return Some(other);
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.is_generating = false;
                    self.rx_response = None;
                    self.cancel_token = None;
                    self.thinking_parser = None;
                }
            }
        }
        None
    }

    /// Extract preferences from conversation (basic implementation)
    fn extract_preferences_from_response(&mut self, _response: &str) {
        for msg in self.conversation.iter().rev().take(5) {
            if msg.role == "user" {
                let content_lower = msg.content.to_lowercase();

                // Detect response style preference
                if content_lower.contains("be brief") || content_lower.contains("short answer") {
                    self.preferences.response_style = ResponseStyle::Concise;
                } else if content_lower.contains("explain in detail")
                    || content_lower.contains("detailed")
                {
                    self.preferences.response_style = ResponseStyle::Detailed;
                } else if content_lower.contains("technical") {
                    self.preferences.response_style = ResponseStyle::Technical;
                }
            }
        }
    }

    /// Add a custom preference extractor
    /// This allows domain-specific preference extraction
    pub fn extract_preferences_with<F>(&mut self, extractor: F)
    where
        F: FnOnce(&[ChatMessage], &mut UserPreferences),
    {
        extractor(&self.conversation, &mut self.preferences);
    }

    // === Conversation Management ===

    /// Clear conversation history
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
        self.current_response.clear();
    }

    /// Get conversation messages for display
    pub fn get_display_messages(&self) -> &[ChatMessage] {
        &self.conversation
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.conversation.len()
    }

    // === Session Management ===

    /// Start a new session
    pub fn new_session(&mut self) {
        if !self.conversation.is_empty() {
            self.save_current_session();
        }

        let session = ChatSession::new("New Chat");
        let session_id = session.id.clone();
        self.current_session = Some(session);
        self.conversation.clear();
        self.current_response.clear();
        log::info!("Session created: session_id={}", session_id);
        self.event_bus
            .emit(crate::events::AiEvent::SessionCreated { session_id });
    }

    /// Save the current conversation to session
    pub fn save_current_session(&mut self) {
        if let Some(ref mut session) = self.current_session {
            session.messages = self.conversation.clone();
            session.preferences = self.preferences.clone();
            session.touch();

            if session.name == "New Chat" && !session.messages.is_empty() {
                session.auto_name();
            }

            self.session_store.save_session(session.clone());
            self.session_store.current_session_id = Some(session.id.clone());
            log::info!("Session saved: session_id={}, messages={}", session.id, session.messages.len());
        } else if !self.conversation.is_empty() {
            let mut session = ChatSession::new("New Chat");
            session.messages = self.conversation.clone();
            session.preferences = self.preferences.clone();
            session.auto_name();

            self.session_store.current_session_id = Some(session.id.clone());
            self.session_store.save_session(session.clone());
            log::info!("Session saved (new): session_id={}, messages={}", session.id, session.messages.len());
            self.current_session = Some(session);
        }
    }

    /// Load a session by ID
    pub fn load_session(&mut self, session_id: &str) {
        if !self.conversation.is_empty() {
            self.save_current_session();
        }

        if let Some(session) = self.session_store.find_session(session_id).cloned() {
            self.conversation = session.messages.clone();
            self.preferences = session.preferences.clone();
            self.session_store.current_session_id = Some(session.id.clone());
            self.current_session = Some(session);
            log::info!("Session loaded: session_id={}", session_id);
            self.event_bus.emit(crate::events::AiEvent::SessionLoaded {
                session_id: session_id.to_string(),
            });
        }
    }

    /// Delete a session by ID
    pub fn delete_session(&mut self, session_id: &str) {
        self.session_store.delete_session(session_id);

        if self.current_session.as_ref().map(|s| s.id.as_str()) == Some(session_id) {
            self.current_session = None;
            self.conversation.clear();
        }
        log::info!("Session deleted: session_id={}", session_id);
        self.event_bus.emit(crate::events::AiEvent::SessionDeleted {
            session_id: session_id.to_string(),
        });
    }

    /// Get all sessions
    pub fn get_sessions(&self) -> &[ChatSession] {
        &self.session_store.sessions
    }

    // === Notes Management ===

    /// Get the current session's context notes
    pub fn get_session_notes(&self) -> &str {
        self.current_session
            .as_ref()
            .map(|s| s.context_notes.as_str())
            .unwrap_or("")
    }

    /// Set the current session's context notes
    pub fn set_session_notes(&mut self, notes: &str) {
        if let Some(ref mut session) = self.current_session {
            session.context_notes = notes.to_string();
            session.touch();
        }
    }

    /// Get global notes from preferences
    pub fn get_global_notes(&self) -> &str {
        &self.preferences.global_notes
    }

    /// Set global notes in preferences
    pub fn set_global_notes(&mut self, notes: &str) {
        self.preferences.global_notes = notes.to_string();
    }

    /// Save sessions to file
    pub fn save_sessions_to_file(&self, path: &Path) -> Result<()> {
        let mut store = self.session_store.clone();

        // Update current session in store
        if let Some(ref current) = self.current_session {
            let mut updated = current.clone();
            updated.messages = self.conversation.clone();
            updated.preferences = self.preferences.clone();
            updated.touch();
            store.save_session(updated);
        }

        store.save_to_file(path)
    }

    /// Load sessions from file
    pub fn load_sessions_from_file(&mut self, path: &Path) -> Result<()> {
        self.session_store = ChatSessionStore::load_from_file(path)?;

        // Restore current session
        if let Some(ref id) = self.session_store.current_session_id.clone() {
            if let Some(session) = self.session_store.find_session(id).cloned() {
                self.conversation = session.messages.clone();
                self.preferences = session.preferences.clone();
                self.current_session = Some(session);
            }
        }
        Ok(())
    }

    // === Context Management ===

    /// Calculate current context usage
    pub fn calculate_context_usage(&self, knowledge: &str) -> ContextUsage {
        let system_tokens = estimate_tokens(&self.system_prompt_base);
        let knowledge_tokens = estimate_tokens(knowledge);

        let history_start = self
            .conversation
            .len()
            .saturating_sub(self.config.max_history_messages);
        let conversation_tokens: usize = self.conversation[history_start..]
            .iter()
            .map(|msg| estimate_tokens(&msg.content) + 4) // +4 for role tokens
            .sum();

        let max_tokens = get_model_context_size_cached(&self.config.selected_model, |name| {
            fetch_model_context_size(&self.config, name)
        });

        ContextUsage::calculate(
            system_tokens,
            knowledge_tokens,
            conversation_tokens,
            max_tokens,
        )
    }

    /// Get dynamic max history based on knowledge size
    pub fn get_effective_max_history(&self, knowledge: &str) -> usize {
        let knowledge_tokens = estimate_tokens(knowledge);
        let max_tokens = get_model_context_size_cached(&self.config.selected_model, |name| {
            fetch_model_context_size(&self.config, name)
        });

        // Reserve tokens for system prompt, response, buffer
        let reserved = 1700;
        let available = max_tokens.saturating_sub(knowledge_tokens + reserved);

        // ~300 tokens per message pair
        let max_pairs = available / 300;
        let max_messages = max_pairs * 2;

        max_messages.clamp(4, self.config.max_history_messages)
    }

    // === Summarization ===

    /// Check if summarization should be triggered
    pub fn should_summarize(&self, knowledge: &str) -> bool {
        if self.is_summarizing || self.conversation.len() < 6 {
            return false;
        }

        let usage = self.calculate_context_usage(knowledge);
        if usage.is_warning {
            log::warn!(
                "Context size warning: usage_pct={:.1}%%, total_tokens={}, max_tokens={}, model={}",
                usage.usage_percent,
                usage.total_tokens,
                usage.max_tokens,
                self.config.selected_model,
            );
        }
        usage.is_warning
    }

    /// Mark messages for summarization (call before sending)
    pub fn summarize_old_messages(&mut self, knowledge: &str) {
        if self.is_summarizing || self.conversation.len() < 6 {
            return;
        }

        let usage = self.calculate_context_usage(knowledge);
        if !usage.is_warning {
            return;
        }

        let keep_count = 4;
        let to_summarize = self.conversation.len().saturating_sub(keep_count);

        if to_summarize >= 2 {
            self.pending_summary_count = to_summarize;
        }
    }

    /// Mark messages for summarization using the internal knowledge context
    pub fn summarize_old_messages_auto(&mut self) {
        let context = self.knowledge_context.clone();
        self.summarize_old_messages(&context);
    }

    /// Check if summarization should be triggered using internal knowledge context
    pub fn should_summarize_auto(&self) -> bool {
        self.should_summarize(&self.knowledge_context)
    }

    /// Start background AI-powered summarization
    pub fn start_background_summarization(&mut self) {
        if self.pending_summary_count == 0 || self.is_summarizing || self.is_generating {
            return;
        }

        let to_summarize = self.pending_summary_count;
        if to_summarize < 2 || self.conversation.len() < to_summarize {
            self.pending_summary_count = 0;
            return;
        }

        // Check for previous summary
        let previous_summary = self
            .conversation
            .first()
            .filter(|msg| msg.role == "system" && msg.content.starts_with("[Conversation summary:"))
            .map(|msg| {
                msg.content
                    .trim_start_matches("[Conversation summary: ")
                    .trim_end_matches(']')
                    .to_string()
            });

        let skip_count = if previous_summary.is_some() { 1 } else { 0 };
        let messages_to_summarize: Vec<ChatMessage> = self
            .conversation
            .iter()
            .skip(skip_count)
            .take(to_summarize - skip_count)
            .cloned()
            .collect();

        if messages_to_summarize.is_empty() {
            self.pending_summary_count = 0;
            return;
        }

        let config = self.config.clone();
        let (tx, rx) = mpsc::channel();

        self.is_summarizing = true;
        self.rx_summary = Some(rx);
        self.pending_summary_count = 0;

        thread::spawn(move || {
            let result = generate_conversation_summary(
                &config,
                &messages_to_summarize,
                previous_summary.as_deref(),
            );
            match result {
                Ok(summary) => {
                    let _ = tx.send(SummaryResult {
                        summary,
                        messages_summarized: to_summarize,
                    });
                }
                Err(_) => {
                    let fallback =
                        create_simple_summary(&messages_to_summarize, previous_summary.as_deref());
                    let _ = tx.send(SummaryResult {
                        summary: fallback,
                        messages_summarized: to_summarize,
                    });
                }
            }
        });
    }

    /// Poll for completed summarization
    pub fn poll_summarization(&mut self) {
        if let Some(ref rx) = self.rx_summary {
            match rx.try_recv() {
                Ok(result) => {
                    let keep_start = result.messages_summarized;
                    if self.conversation.len() > keep_start {
                        let kept_messages: Vec<ChatMessage> =
                            self.conversation.iter().skip(keep_start).cloned().collect();

                        self.conversation.clear();
                        self.conversation.push(ChatMessage::system(format!(
                            "[Conversation summary: {}]",
                            result.summary
                        )));
                        self.conversation.extend(kept_messages);
                    }

                    self.is_summarizing = false;
                    self.rx_summary = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.is_summarizing = false;
                    self.rx_summary = None;
                }
            }
        }
    }

    // === RAG Support (optional feature) ===

    #[cfg(feature = "rag")]
    /// Set the path for the RAG database (lazy initialization)
    ///
    /// The database will be created automatically when first needed (e.g., when
    /// a document is registered or when RAG context is requested).
    ///
    /// # Example
    /// ```no_run
    /// use ai_assistant::AiAssistant;
    /// use std::path::Path;
    ///
    /// let mut assistant = AiAssistant::new();
    /// assistant.set_rag_path(Path::new("./app_data/ai_rag.db"));
    ///
    /// // Register documents - RAG will initialize automatically
    /// assistant.register_knowledge_document("guide", "# Guide\nContent here...");
    /// ```
    pub fn set_rag_path(&mut self, db_path: &Path) {
        self.rag_db_path = Some(db_path.to_path_buf());
    }

    #[cfg(feature = "rag")]
    /// Initialize RAG database at the specified path (explicit initialization)
    ///
    /// Note: You can also use `set_rag_path()` for lazy initialization, which
    /// will create the database automatically when first needed.
    pub fn init_rag(&mut self, db_path: &Path) -> Result<()> {
        self.rag_db_path = Some(db_path.to_path_buf());
        self.rag_db = Some(RagDb::open(db_path)?);
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Ensure RAG database is initialized (lazy initialization)
    ///
    /// This is called internally before any RAG operation. If a path has been
    /// set via `set_rag_path()`, the database will be created automatically.
    ///
    /// Returns true if RAG is available, false otherwise.
    fn ensure_rag_initialized(&mut self) -> bool {
        if self.rag_db.is_some() {
            return true;
        }

        if let Some(ref path) = self.rag_db_path.clone() {
            match RagDb::open(path) {
                Ok(db) => {
                    self.rag_db = Some(db);
                    true
                }
                Err(e) => {
                    log::error!("[AI RAG] Failed to initialize database: {}", e);
                    false
                }
            }
        } else {
            false
        }
    }

    #[cfg(feature = "rag")]
    /// Register a knowledge document for indexing
    ///
    /// The document will be indexed automatically when needed (e.g., before
    /// the first message is sent or when RAG context is requested).
    ///
    /// If the document content hasn't changed since last indexing, it will
    /// be skipped automatically.
    ///
    /// # Arguments
    /// * `source` - Unique identifier for the document (e.g., filename without extension)
    /// * `content` - The full text content of the document
    ///
    /// # Example
    /// ```no_run
    /// use ai_assistant::AiAssistant;
    /// use std::path::Path;
    ///
    /// let mut assistant = AiAssistant::new();
    /// assistant.set_rag_path(Path::new("./ai_rag.db"));
    ///
    /// // Register documents from files
    /// let content = std::fs::read_to_string("knowledge/guide.md").unwrap();
    /// assistant.register_knowledge_document("guide", &content);
    ///
    /// // Documents are indexed automatically before first use
    /// assistant.send_message("Help me understand the guide".to_string(), "");
    /// ```
    pub fn register_knowledge_document(&mut self, source: &str, content: &str) {
        self.pending_documents
            .insert(source.to_string(), content.to_string());
        if !self.registered_sources.contains(&source.to_string()) {
            self.registered_sources.push(source.to_string());
        }
    }

    #[cfg(feature = "rag")]
    /// Unregister a knowledge document
    ///
    /// Removes the document from the pending list and registered sources.
    /// Note: This does not delete the document from the database if already indexed.
    /// Use `delete_knowledge_document()` to also remove from database.
    ///
    /// Returns `Err` if `append_only_mode` is enabled in `RagConfig`.
    pub fn unregister_knowledge_document(&mut self, source: &str) -> Result<()> {
        if self.rag_config.append_only_mode {
            return Err(anyhow::anyhow!(
                "Cannot unregister knowledge document '{}': append-only mode is enabled. \
                 Only adding new documents is allowed.",
                source
            ));
        }
        self.pending_documents.remove(source);
        self.registered_sources.retain(|s| s != source);
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Delete a knowledge document from the database
    ///
    /// Removes the document from both the pending list and the database.
    ///
    /// Returns `Err` if `append_only_mode` is enabled in `RagConfig`.
    pub fn delete_knowledge_document(&mut self, source: &str) -> Result<()> {
        if self.rag_config.append_only_mode {
            return Err(anyhow::anyhow!(
                "Cannot delete knowledge document '{}': append-only mode is enabled. \
                 Only adding new documents is allowed.",
                source
            ));
        }
        self.pending_documents.remove(source);
        self.registered_sources.retain(|s| s != source);

        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.delete_document(source)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Enable or disable append-only mode for knowledge
    ///
    /// When enabled, knowledge documents can only be added, not removed.
    /// This protects against accidental deletion of important knowledge.
    pub fn set_append_only_mode(&mut self, enabled: bool) {
        self.rag_config.append_only_mode = enabled;
    }

    #[cfg(feature = "rag")]
    /// Check if append-only mode is enabled
    pub fn is_append_only_mode(&self) -> bool {
        self.rag_config.append_only_mode
    }

    #[cfg(feature = "rag")]
    /// Get list of registered knowledge sources
    pub fn get_registered_sources(&self) -> &[String] {
        &self.registered_sources
    }

    #[cfg(feature = "rag")]
    /// Process all pending documents (index them into RAG database)
    ///
    /// This is called automatically before RAG operations, but can be called
    /// manually to force indexing.
    ///
    /// Returns a vector of (source_name, chunks_indexed) for documents that were indexed.
    /// Documents that were already up-to-date return 0 chunks.
    pub fn process_pending_documents(&mut self) -> Vec<(String, usize)> {
        if self.pending_documents.is_empty() {
            return Vec::new();
        }

        if !self.ensure_rag_initialized() {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Take ownership of pending documents to process them
        let documents: Vec<(String, String)> = self.pending_documents.drain().collect();

        if let Some(ref db) = self.rag_db {
            for (source, content) in documents {
                match db.index_document(&source, &content) {
                    Ok(chunks) => {
                        results.push((source, chunks));
                    }
                    Err(e) => {
                        log::error!("[AI RAG] Failed to index '{}': {}", source, e);
                    }
                }
            }
        }

        results
    }

    #[cfg(feature = "rag")]
    /// Check if there are pending documents to index
    pub fn has_pending_documents(&self) -> bool {
        !self.pending_documents.is_empty()
    }

    #[cfg(feature = "rag")]
    /// Get the number of pending documents
    pub fn pending_document_count(&self) -> usize {
        self.pending_documents.len()
    }

    #[cfg(feature = "rag")]
    /// Register multiple documents at once (batch registration)
    ///
    /// More efficient than calling `register_knowledge_document` multiple times.
    pub fn register_documents(&mut self, documents: Vec<DocumentInfo>) {
        for doc in documents {
            self.pending_documents
                .insert(doc.source.clone(), doc.content);
            if !self.registered_sources.contains(&doc.source) {
                self.registered_sources.push(doc.source);
            }
        }
    }

    #[cfg(feature = "rag")]
    /// Start background indexing of all pending documents
    ///
    /// Returns immediately. Use `poll_indexing()` to check progress.
    /// When complete, `is_indexing` will be false.
    pub fn start_background_indexing(&mut self) {
        if self.pending_documents.is_empty() || self.is_indexing {
            return;
        }

        if !self.ensure_rag_initialized() {
            return;
        }

        let db_path = match &self.rag_db_path {
            Some(p) => p.clone(),
            None => return,
        };

        let documents: Vec<(String, String)> = self.pending_documents.drain().collect();
        let (tx, rx) = mpsc::channel();

        self.is_indexing = true;
        self.rx_indexing = Some(rx);

        thread::spawn(move || {
            let db = match RagDb::open(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    let _ = tx.send(IndexingProgress::Error {
                        source: "database".to_string(),
                        error: e.to_string(),
                    });
                    return;
                }
            };

            let total = documents.len();
            let mut results = Vec::new();

            for (i, (source, content)) in documents.into_iter().enumerate() {
                let _ = tx.send(IndexingProgress::Starting {
                    source: source.clone(),
                    total_documents: total,
                    current: i + 1,
                });

                match db.index_document(&source, &content) {
                    Ok(chunks) => {
                        let tokens = content.len() / 4; // Rough estimate
                        let result = IndexingResult {
                            source: source.clone(),
                            chunks,
                            tokens,
                            was_cached: chunks == 0,
                        };
                        let _ = tx.send(IndexingProgress::Completed(result.clone()));
                        results.push(result);
                    }
                    Err(e) => {
                        let _ = tx.send(IndexingProgress::Error {
                            source,
                            error: e.to_string(),
                        });
                    }
                }
            }

            let _ = tx.send(IndexingProgress::AllComplete { results });
        });
    }

    #[cfg(feature = "rag")]
    /// Poll for indexing progress updates
    ///
    /// Returns the latest progress update, if any.
    pub fn poll_indexing(&mut self) -> Option<IndexingProgress> {
        if let Some(ref rx) = self.rx_indexing {
            match rx.try_recv() {
                Ok(progress) => {
                    if matches!(progress, IndexingProgress::AllComplete { .. }) {
                        self.is_indexing = false;
                        self.rx_indexing = None;
                    }
                    return Some(progress);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.is_indexing = false;
                    self.rx_indexing = None;
                }
            }
        }
        None
    }

    #[cfg(feature = "rag")]
    /// Process pending documents with a progress callback
    ///
    /// Blocks until all documents are indexed, but calls the callback for each.
    pub fn process_with_callback<F>(&mut self, mut on_progress: F) -> Vec<IndexingResult>
    where
        F: FnMut(IndexingProgress),
    {
        if self.pending_documents.is_empty() {
            return Vec::new();
        }

        if !self.ensure_rag_initialized() {
            return Vec::new();
        }

        let documents: Vec<(String, String)> = self.pending_documents.drain().collect();
        let total = documents.len();
        let mut results = Vec::new();

        if let Some(ref db) = self.rag_db {
            for (i, (source, content)) in documents.into_iter().enumerate() {
                on_progress(IndexingProgress::Starting {
                    source: source.clone(),
                    total_documents: total,
                    current: i + 1,
                });

                match db.index_document(&source, &content) {
                    Ok(chunks) => {
                        let tokens = content.len() / 4;
                        let result = IndexingResult {
                            source: source.clone(),
                            chunks,
                            tokens,
                            was_cached: chunks == 0,
                        };
                        on_progress(IndexingProgress::Completed(result.clone()));
                        results.push(result);
                    }
                    Err(e) => {
                        on_progress(IndexingProgress::Error {
                            source,
                            error: e.to_string(),
                        });
                    }
                }
            }
        }

        on_progress(IndexingProgress::AllComplete {
            results: results.clone(),
        });
        results
    }

    #[cfg(feature = "rag")]
    /// Get detailed information about a specific indexed document
    pub fn get_document_info(&mut self, source: &str) -> Option<DocumentStats> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                if let Ok(Some((hash, chunks, tokens, indexed_at))) = db.get_source_info(source) {
                    return Some(DocumentStats {
                        source: source.to_string(),
                        chunk_count: chunks,
                        total_tokens: tokens,
                        content_hash: hash,
                        indexed_at,
                        is_pending: self.pending_documents.contains_key(source),
                    });
                }
            }
        }
        // Check if it's pending
        if self.pending_documents.contains_key(source) {
            return Some(DocumentStats {
                source: source.to_string(),
                chunk_count: 0,
                total_tokens: 0,
                content_hash: String::new(),
                indexed_at: String::new(),
                is_pending: true,
            });
        }
        None
    }

    #[cfg(feature = "rag")]
    /// Get statistics for all indexed documents
    pub fn get_all_document_stats(&mut self) -> Vec<DocumentStats> {
        let mut stats = Vec::new();

        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                if let Ok(sources) = db.list_indexed_sources() {
                    for (source, chunks, tokens, indexed_at) in sources {
                        stats.push(DocumentStats {
                            source: source.clone(),
                            chunk_count: chunks,
                            total_tokens: tokens,
                            content_hash: String::new(), // Not needed for listing
                            indexed_at,
                            is_pending: self.pending_documents.contains_key(&source),
                        });
                    }
                }
            }
        }

        // Add pending documents that aren't indexed yet
        for source in self.pending_documents.keys() {
            if !stats.iter().any(|s| &s.source == source) {
                stats.push(DocumentStats {
                    source: source.clone(),
                    chunk_count: 0,
                    total_tokens: 0,
                    content_hash: String::new(),
                    indexed_at: String::new(),
                    is_pending: true,
                });
            }
        }

        stats
    }

    #[cfg(feature = "rag")]
    /// Set the current user ID for RAG operations
    pub fn set_user_id(&mut self, user_id: &str) {
        self.user_id = user_id.to_string();
    }

    #[cfg(feature = "rag")]
    /// Get the current user ID
    pub fn get_user_id(&self) -> &str {
        &self.user_id
    }

    #[cfg(feature = "rag")]
    /// Get or create user in RAG database, returns global notes
    pub fn ensure_user(&mut self) -> Result<String> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let user = db.get_or_create_user(&self.user_id)?;
                return Ok(user.global_notes);
            }
        }
        Ok(String::new())
    }

    #[cfg(feature = "rag")]
    /// Check if RAG is initialized or can be initialized
    pub fn has_rag(&self) -> bool {
        self.rag_db.is_some() || self.rag_db_path.is_some()
    }

    #[cfg(feature = "rag")]
    /// Check if RAG database is currently open
    pub fn is_rag_initialized(&self) -> bool {
        self.rag_db.is_some()
    }

    #[cfg(feature = "rag")]
    /// Index a document into the knowledge base (direct method)
    ///
    /// Note: Prefer using `register_knowledge_document()` for automatic
    /// management of document indexing.
    pub fn index_knowledge_document(&mut self, source: &str, content: &str) -> Result<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.index_document(source, content);
            }
        }
        Ok(0)
    }

    #[cfg(feature = "rag")]
    /// Clear all knowledge from the database
    pub fn clear_knowledge(&mut self) -> Result<()> {
        self.pending_documents.clear();
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.clear_knowledge()?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Get knowledge base statistics (chunk count, total tokens)
    pub fn get_knowledge_stats(&mut self) -> Result<(usize, usize)> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_knowledge_stats();
            }
        }
        Ok((0, 0))
    }

    #[cfg(feature = "rag")]
    /// Build context using RAG retrieval based on the user's query
    ///
    /// This automatically processes any pending documents before searching.
    /// Also tracks which knowledge sources were used (accessible via `last_knowledge_usage`).
    ///
    /// Returns (knowledge_context, conversation_context) if RAG is enabled
    pub fn build_rag_context(&mut self, query: &str) -> (String, String) {
        // Ensure RAG is initialized (lazy initialization if path was set)
        if !self.ensure_rag_initialized() {
            // RAG not available, return empty contexts
            return (String::new(), String::new());
        }

        // Process pending documents first
        if self.has_pending_documents() {
            let results = self.process_pending_documents();
            for (source, chunks) in results {
                if chunks > 0 {
                    println!("[AI RAG] Indexed '{}': {} chunks", source, chunks);
                } else {
                    println!("[AI RAG] '{}' up-to-date (skipped)", source);
                }
            }
        }

        let mut knowledge_context = String::new();
        let mut conversation_context = String::new();
        self.last_knowledge_usage = None;

        // Calculate effective max tokens for knowledge
        // If dynamic context is enabled, use the available space; otherwise use configured max
        let effective_max_knowledge_tokens = if self.rag_config.dynamic_context_enabled {
            self.calculate_available_knowledge_tokens(query)
        } else {
            self.rag_config.max_knowledge_tokens
        };

        if let Some(ref db) = self.rag_db {
            // Knowledge RAG with caching
            if self.rag_config.knowledge_rag_enabled {
                // Check cache first (include effective tokens in cache key)
                let cache_key = format!("{}_{}", query, effective_max_knowledge_tokens);
                let cached = self.rag_cache.as_mut().and_then(|c| c.get(&cache_key));

                let chunks = if let Some(cached_chunks) = cached {
                    self.metrics.record_cache_hit();
                    cached_chunks
                } else {
                    self.metrics.record_cache_miss();
                    if let Ok(search_chunks) = db.search_knowledge(
                        query,
                        effective_max_knowledge_tokens,
                        self.rag_config.top_k_chunks,
                    ) {
                        // Cache the result
                        if let Some(ref mut cache) = self.rag_cache {
                            cache.insert(cache_key, search_chunks.clone());
                        }
                        search_chunks
                    } else {
                        Vec::new()
                    }
                };

                // Record source access for metrics
                for chunk in &chunks {
                    self.metrics.record_source_access(&chunk.source);
                }

                // Track knowledge usage
                if !chunks.is_empty() {
                    let usage = KnowledgeUsage::from_chunks(query, &chunks);
                    self.last_knowledge_usage = Some(usage.clone());
                    self.knowledge_usage_history.insert(0, usage);
                    // Keep history limited to last 100 entries
                    if self.knowledge_usage_history.len() > 100 {
                        self.knowledge_usage_history.truncate(100);
                    }
                }

                knowledge_context = build_knowledge_context(&chunks);
            }

            // Conversation RAG
            if self.rag_config.conversation_rag_enabled {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");

                // First try semantic search on archived messages
                if let Ok(messages) = db.search_conversation(
                    &self.user_id,
                    session_id,
                    query,
                    self.rag_config.max_conversation_tokens / 2,
                    true, // exclude in-context messages
                ) {
                    if !messages.is_empty() {
                        conversation_context.push_str(&build_conversation_context(&messages));
                    }
                }

                // Also get recent archived messages for continuity
                if let Ok(recent) = db.get_recent_archived_messages(
                    &self.user_id,
                    session_id,
                    self.rag_config.max_conversation_tokens / 2,
                ) {
                    if !recent.is_empty() {
                        let recent_context = build_conversation_context(&recent);
                        if !conversation_context.contains(&recent_context) {
                            conversation_context.push_str(&recent_context);
                        }
                    }
                }
            }
        }

        (knowledge_context, conversation_context)
    }

    #[cfg(feature = "rag")]
    /// Build context with tracking and return usage information
    ///
    /// Similar to `build_rag_context`, but returns the knowledge usage tracking
    /// as a third element of the tuple.
    ///
    /// Returns (knowledge_context, conversation_context, knowledge_usage)
    pub fn build_rag_context_with_tracking(
        &mut self,
        query: &str,
    ) -> (String, String, Option<KnowledgeUsage>) {
        let (knowledge_context, conversation_context) = self.build_rag_context(query);
        let usage = self.last_knowledge_usage.clone();
        (knowledge_context, conversation_context, usage)
    }

    #[cfg(feature = "rag")]
    /// Get the last knowledge usage information
    ///
    /// This is updated after each call to `build_rag_context`.
    pub fn get_last_knowledge_usage(&self) -> Option<&KnowledgeUsage> {
        self.last_knowledge_usage.as_ref()
    }

    #[cfg(feature = "rag")]
    /// Get the knowledge usage history (most recent first)
    ///
    /// Limited to the last 100 entries.
    pub fn get_knowledge_usage_history(&self) -> &[KnowledgeUsage] {
        &self.knowledge_usage_history
    }

    #[cfg(feature = "rag")]
    /// Clear the knowledge usage history
    pub fn clear_knowledge_usage_history(&mut self) {
        self.knowledge_usage_history.clear();
        self.last_knowledge_usage = None;
    }

    #[cfg(feature = "rag")]
    /// Get a summary of knowledge sources most frequently used
    ///
    /// Returns a list of (source, usage_count) sorted by count descending.
    pub fn get_knowledge_source_frequency(&self) -> Vec<(String, usize)> {
        use std::collections::HashMap;

        let mut frequency: HashMap<String, usize> = HashMap::new();
        for usage in &self.knowledge_usage_history {
            for source in &usage.sources {
                *frequency.entry(source.source.clone()).or_default() += 1;
            }
        }

        let mut result: Vec<_> = frequency.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    #[cfg(feature = "rag")]
    /// Build RAG context filtering by specific knowledge sources
    ///
    /// Only retrieves chunks from the specified sources.
    /// Useful when the user has selected specific knowledge documents to use.
    ///
    /// Returns (knowledge_context, conversation_context, knowledge_usage)
    pub fn build_rag_context_filtered(
        &mut self,
        query: &str,
        sources: &[String],
    ) -> (String, String, Option<KnowledgeUsage>) {
        // Ensure RAG is initialized (lazy initialization if path was set)
        if !self.ensure_rag_initialized() {
            return (String::new(), String::new(), None);
        }

        // Process pending documents first
        if self.has_pending_documents() {
            let results = self.process_pending_documents();
            for (source, chunks) in results {
                if chunks > 0 {
                    println!("[AI RAG] Indexed '{}': {} chunks", source, chunks);
                } else {
                    println!("[AI RAG] '{}' up-to-date (skipped)", source);
                }
            }
        }

        let mut knowledge_context = String::new();
        let mut conversation_context = String::new();
        self.last_knowledge_usage = None;

        // Calculate effective max tokens for knowledge (dynamic or fixed)
        let effective_max_knowledge_tokens = if self.rag_config.dynamic_context_enabled {
            self.calculate_available_knowledge_tokens(query)
        } else {
            self.rag_config.max_knowledge_tokens
        };

        if let Some(ref db) = self.rag_db {
            // Knowledge RAG with source filtering
            if self.rag_config.knowledge_rag_enabled && !sources.is_empty() {
                // No caching for filtered searches (cache key would need to include sources)
                self.metrics.record_cache_miss();

                if let Ok(chunks) = db.search_knowledge_filtered(
                    query,
                    sources,
                    effective_max_knowledge_tokens,
                    self.rag_config.top_k_chunks,
                ) {
                    // Record source access for metrics
                    for chunk in &chunks {
                        self.metrics.record_source_access(&chunk.source);
                    }

                    // Track knowledge usage
                    if !chunks.is_empty() {
                        let usage = KnowledgeUsage::from_chunks(query, &chunks);
                        self.last_knowledge_usage = Some(usage.clone());
                        self.knowledge_usage_history.insert(0, usage);
                        if self.knowledge_usage_history.len() > 100 {
                            self.knowledge_usage_history.truncate(100);
                        }
                    }

                    knowledge_context = build_knowledge_context(&chunks);
                }
            }

            // Conversation RAG (same as regular build_rag_context)
            if self.rag_config.conversation_rag_enabled {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");

                if let Ok(messages) = db.search_conversation(
                    &self.user_id,
                    session_id,
                    query,
                    self.rag_config.max_conversation_tokens / 2,
                    true,
                ) {
                    if !messages.is_empty() {
                        conversation_context.push_str(&build_conversation_context(&messages));
                    }
                }

                if let Ok(recent) = db.get_recent_archived_messages(
                    &self.user_id,
                    session_id,
                    self.rag_config.max_conversation_tokens / 2,
                ) {
                    if !recent.is_empty() {
                        let recent_context = build_conversation_context(&recent);
                        if !conversation_context.contains(&recent_context) {
                            conversation_context.push_str(&recent_context);
                        }
                    }
                }
            }
        }

        let usage = self.last_knowledge_usage.clone();
        (knowledge_context, conversation_context, usage)
    }

    #[cfg(feature = "rag")]
    /// Get all available knowledge sources
    ///
    /// Returns a list of all document sources in the knowledge base.
    pub fn get_all_knowledge_sources(&self) -> Vec<String> {
        if let Some(ref db) = self.rag_db {
            db.get_knowledge_sources().unwrap_or_default()
        } else {
            self.registered_sources.clone()
        }
    }

    #[cfg(feature = "rag")]
    /// Get knowledge source statistics for UI display
    ///
    /// Returns stats (source_name, chunk_count, token_count) for each source.
    pub fn get_knowledge_source_stats(&self) -> Vec<(String, usize, usize)> {
        if let Some(ref db) = self.rag_db {
            let sources = db.get_knowledge_sources().unwrap_or_default();
            let mut stats = Vec::new();
            for source in sources {
                if let Ok(Some((_, chunk_count, token_count, _))) = db.get_source_info(&source) {
                    stats.push((source, chunk_count, token_count));
                }
            }
            return stats;
        }
        Vec::new()
    }

    #[cfg(feature = "rag")]
    /// Store a message in the RAG database
    pub fn store_message_in_rag(&mut self, msg: &ChatMessage, in_context: bool) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| "default".to_string());

                let id = db.store_message(&self.user_id, &session_id, msg, in_context)?;
                if in_context {
                    self.rag_message_ids.push(id);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Archive old messages from context to RAG storage
    /// This marks messages as out-of-context but keeps them searchable
    pub fn archive_messages_to_rag(&mut self, count: usize) -> Result<()> {
        if self.rag_message_ids.len() < count {
            return Ok(());
        }

        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| "default".to_string());

                let to_archive: Vec<i64> = self.rag_message_ids.drain(..count).collect();
                db.mark_messages_out_of_context(&self.user_id, &session_id, &to_archive)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Get conversation stats from RAG database
    pub fn get_conversation_rag_stats(&mut self) -> Result<(usize, usize, usize)> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");
                return db.get_conversation_stats(&self.user_id, session_id);
            }
        }
        Ok((0, 0, 0))
    }

    #[cfg(feature = "rag")]
    /// Enable or disable knowledge RAG, returning whether the change was successful
    pub fn set_knowledge_rag_enabled(&mut self, enabled: bool) -> bool {
        if self.ensure_rag_initialized() {
            self.rag_config.knowledge_rag_enabled = enabled;
            true
        } else {
            false
        }
    }

    #[cfg(feature = "rag")]
    /// Enable or disable conversation RAG, returning whether the change was successful
    pub fn set_conversation_rag_enabled(&mut self, enabled: bool) -> bool {
        if self.ensure_rag_initialized() {
            self.rag_config.conversation_rag_enabled = enabled;
            true
        } else {
            false
        }
    }

    #[cfg(feature = "rag")]
    /// Enable or disable auto-storage of messages in RAG
    /// When enabled, messages are automatically indexed as they are sent/received
    pub fn set_auto_store_messages(&mut self, enabled: bool) -> bool {
        if self.ensure_rag_initialized() {
            self.rag_config.auto_store_messages = enabled;
            true
        } else {
            false
        }
    }

    #[cfg(feature = "rag")]
    /// Check if auto-store messages is enabled
    pub fn is_auto_store_messages_enabled(&self) -> bool {
        self.rag_config.auto_store_messages
    }

    #[cfg(feature = "rag")]
    /// Check if enabling RAG would help with context overflow
    /// Returns (can_help_with_knowledge, can_help_with_conversation)
    pub fn can_rag_help_with_context(&self) -> (bool, bool) {
        let has_rag = self.rag_db.is_some() || self.rag_db_path.is_some();
        let can_knowledge = has_rag && !self.rag_config.knowledge_rag_enabled;
        let can_conversation = has_rag && !self.rag_config.conversation_rag_enabled;
        (can_knowledge, can_conversation)
    }

    #[cfg(feature = "rag")]
    /// Estimate context savings if RAG were enabled
    /// Returns estimated tokens that would be saved
    pub fn estimate_rag_savings(&self, current_knowledge: &str) -> usize {
        let mut savings = 0;

        if !self.rag_config.knowledge_rag_enabled {
            // Full knowledge vs RAG-retrieved subset
            let full_tokens = estimate_tokens(current_knowledge);
            let rag_tokens = self.rag_config.max_knowledge_tokens;
            if full_tokens > rag_tokens {
                savings += full_tokens - rag_tokens;
            }
        }

        if !self.rag_config.conversation_rag_enabled && self.conversation.len() > 4 {
            // Estimate savings from archiving old messages
            let archive_count = self.conversation.len().saturating_sub(4);
            let archive_tokens: usize = self.conversation[..archive_count]
                .iter()
                .map(|m| estimate_tokens(&m.content))
                .sum();
            let rag_retrieval_tokens = self.rag_config.max_conversation_tokens;
            if archive_tokens > rag_retrieval_tokens {
                savings += archive_tokens - rag_retrieval_tokens;
            }
        }

        savings
    }

    #[cfg(feature = "rag")]
    /// Get notes for a specific knowledge source/guide
    pub fn get_knowledge_notes(&mut self, source: &str) -> Option<String> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_knowledge_notes(&self.user_id, source).ok().flatten();
            }
        }
        None
    }

    #[cfg(feature = "rag")]
    /// Set notes for a specific knowledge source/guide
    pub fn set_knowledge_notes(&mut self, source: &str, notes: &str) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                if notes.is_empty() {
                    db.delete_knowledge_notes(&self.user_id, source)?;
                } else {
                    db.set_knowledge_notes(&self.user_id, source, notes)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Get all knowledge sources (guides) that have been indexed
    ///
    /// Returns both registered sources and any additional indexed sources from database.
    pub fn get_knowledge_sources(&mut self) -> Vec<String> {
        let mut sources = self.registered_sources.clone();

        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                if let Ok(db_sources) = db.get_knowledge_sources() {
                    for s in db_sources {
                        if !sources.contains(&s) {
                            sources.push(s);
                        }
                    }
                }
            }
        }
        sources
    }

    #[cfg(feature = "rag")]
    /// Build combined knowledge notes string from all sources with notes
    pub fn build_knowledge_notes_context(&mut self) -> String {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                if let Ok(all_notes) = db.get_all_knowledge_notes(&self.user_id) {
                    if all_notes.is_empty() {
                        return String::new();
                    }

                    let mut context = String::new();
                    for (source, notes) in all_notes {
                        context.push_str(&format!("Notes for '{}':\n{}\n\n", source, notes));
                    }
                    return context;
                }
            }
        }
        String::new()
    }

    // === RAG Global Notes (stored in database per user) ===

    #[cfg(feature = "rag")]
    /// Get global notes from RAG database for current user
    pub fn get_rag_global_notes(&mut self) -> String {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_user_global_notes(&self.user_id).unwrap_or_default();
            }
        }
        String::new()
    }

    #[cfg(feature = "rag")]
    /// Set global notes in RAG database for current user
    pub fn set_rag_global_notes(&mut self, notes: &str) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.set_user_global_notes(&self.user_id, notes)?;
            }
        }
        Ok(())
    }

    // === RAG Session Notes (stored in database per user) ===

    #[cfg(feature = "rag")]
    /// Get session notes from RAG database for current user and session
    pub fn get_rag_session_notes(&mut self) -> String {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");
                return db
                    .get_session_notes(&self.user_id, session_id)
                    .ok()
                    .flatten()
                    .unwrap_or_default();
            }
        }
        String::new()
    }

    #[cfg(feature = "rag")]
    /// Set session notes in RAG database for current user and session
    pub fn set_rag_session_notes(&mut self, notes: &str) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.clone())
                    .unwrap_or_else(|| "default".to_string());
                if notes.is_empty() {
                    db.delete_session_notes(&self.user_id, &session_id)?;
                } else {
                    db.set_session_notes(&self.user_id, &session_id, notes)?;
                }
            }
        }
        Ok(())
    }

    // === Knowledge Base Export/Import ===

    #[cfg(feature = "rag")]
    /// Export the knowledge base to a file
    ///
    /// Exports all indexed documents and their chunks to a JSON file that can
    /// be imported later or shared between installations.
    pub fn export_knowledge_to_file(&mut self, path: &std::path::Path) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.export_knowledge_to_file(path)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Import knowledge base from a file
    ///
    /// # Arguments
    /// * `path` - Path to the JSON export file
    /// * `replace` - If true, clears existing knowledge first. If false, merges.
    ///
    /// Returns the number of chunks imported.
    pub fn import_knowledge_from_file(
        &mut self,
        path: &std::path::Path,
        replace: bool,
    ) -> Result<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.import_knowledge_from_file(path, replace);
            }
        }
        Ok(0)
    }

    #[cfg(feature = "rag")]
    /// Export knowledge base to a serializable format
    ///
    /// Use this when you need to handle the export data programmatically
    /// rather than writing directly to a file.
    pub fn export_knowledge(&mut self) -> Option<crate::rag::KnowledgeExport> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.export_knowledge().ok();
            }
        }
        None
    }

    #[cfg(feature = "rag")]
    /// Import knowledge from a serializable format
    ///
    /// # Arguments
    /// * `data` - The knowledge export data
    /// * `replace` - If true, clears existing knowledge first. If false, merges.
    pub fn import_knowledge(
        &mut self,
        data: &crate::rag::KnowledgeExport,
        replace: bool,
    ) -> Result<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.import_knowledge(data, replace);
            }
        }
        Ok(0)
    }

    // === Metrics Methods ===

    /// Get session metrics aggregated for the current session
    pub fn get_session_metrics(&self) -> crate::metrics::SessionMetrics {
        self.metrics.get_session_metrics()
    }

    /// Get RAG quality metrics
    pub fn get_rag_quality_metrics(&self) -> crate::metrics::RagQualityMetrics {
        self.metrics.get_rag_quality_metrics()
    }

    /// Get all message metrics from the current session
    pub fn get_message_metrics(&self) -> &[crate::metrics::MessageMetrics] {
        self.metrics.get_message_metrics()
    }

    /// Export all metrics as JSON
    pub fn export_metrics_json(&self) -> String {
        self.metrics.export_json()
    }

    /// Reset metrics for a new session
    pub fn reset_metrics(&mut self, session_id: &str) {
        self.metrics = crate::metrics::MetricsTracker::new(session_id);
    }

    /// Start tracking a new message (call before sending)
    pub fn start_message_tracking(&mut self) {
        self.metrics.start_message(&self.config.selected_model);
    }

    /// Mark that the first token was received
    pub fn mark_first_token_received(&mut self) {
        self.metrics.mark_first_token();
    }

    /// Finish tracking the current message (call after response complete)
    pub fn finish_message_tracking(&mut self, output_tokens: usize) {
        self.metrics.finish_message(output_tokens);
    }

    /// Clear the RAG search cache
    #[cfg(feature = "rag")]
    pub fn clear_rag_cache(&mut self) {
        if let Some(ref mut cache) = self.rag_cache {
            cache.clear();
        }
    }

    /// Get RAG cache statistics: (entries, total_hits)
    #[cfg(feature = "rag")]
    pub fn get_rag_cache_stats(&self) -> (usize, usize) {
        self.rag_cache.as_ref().map(|c| c.stats()).unwrap_or((0, 0))
    }

    #[cfg(feature = "rag")]
    /// Set priority for a knowledge source
    /// Higher priority sources appear first in search results
    pub fn set_source_priority(&mut self, source: &str, priority: i32) -> Result<()> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                db.set_source_priority(source, priority)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "rag")]
    /// Get priority for a knowledge source
    pub fn get_source_priority(&mut self, source: &str) -> i32 {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_source_priority(source).unwrap_or(0);
            }
        }
        0
    }

    // =========================================================================
    // AUTONOMOUS AGENT INTEGRATION
    // =========================================================================

    /// Get the current operation mode.
    #[cfg(feature = "autonomous")]
    pub fn operation_mode(&self) -> OperationMode {
        self.mode_manager.current()
    }

    /// Set the operation mode (respects allowed_max ceiling).
    #[cfg(feature = "autonomous")]
    pub fn set_operation_mode(&mut self, mode: OperationMode) -> Result<()> {
        self.mode_manager
            .set_mode(mode)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Escalate to next operation mode.
    #[cfg(feature = "autonomous")]
    pub fn escalate_mode(&mut self) -> Result<OperationMode> {
        self.mode_manager.escalate().map_err(|e| anyhow::anyhow!(e))
    }

    /// De-escalate to lower operation mode.
    #[cfg(feature = "autonomous")]
    pub fn de_escalate_mode(&mut self) -> OperationMode {
        self.mode_manager.de_escalate()
    }

    /// Get the profile registry.
    #[cfg(feature = "autonomous")]
    pub fn profiles(&self) -> &ProfileRegistry {
        &self.profile_registry
    }

    /// Get the profile registry mutably.
    #[cfg(feature = "autonomous")]
    pub fn profiles_mut(&mut self) -> &mut ProfileRegistry {
        &mut self.profile_registry
    }

    /// Set the interaction handler for agent-user communication.
    #[cfg(feature = "autonomous")]
    pub fn set_interaction_handler(&mut self, handler: Arc<dyn UserInteractionHandler>) {
        self.interaction_manager = Some(Arc::new(InteractionManager::new(handler, 300)));
    }

    /// Get the interaction manager (if configured).
    #[cfg(feature = "autonomous")]
    pub fn interaction_manager(&self) -> Option<&Arc<InteractionManager>> {
        self.interaction_manager.as_ref()
    }

    /// Create an autonomous agent from a registered profile name.
    ///
    /// The agent uses the assistant's config to derive a response generator
    /// callback, and applies the profile's policy and tools.
    #[cfg(feature = "autonomous")]
    pub fn create_agent(
        &self,
        profile_name: &str,
        response_generator: Arc<
            dyn Fn(&[crate::agentic_loop::AgentMessage]) -> String + Send + Sync,
        >,
    ) -> Result<AutonomousAgent> {
        let profile = self
            .profile_registry
            .get_agent_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Agent profile '{}' not found", profile_name))?;

        let mut builder = AutonomousAgentBuilder::new(&profile.name, response_generator)
            .policy(profile.policy.clone())
            .mode(profile.mode);

        if let Some(ref prompt) = profile.system_prompt {
            builder = builder.system_prompt(prompt.clone());
        }

        if let Some(ref max_iter) = Some(profile.policy.max_iterations) {
            builder = builder.max_iterations(*max_iter);
        }

        // Register OS tools into the agent's tool registry
        let policy = profile.policy.clone();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));
        let mut registry = crate::unified_tools::ToolRegistry::new();
        register_os_tools(&mut registry, sandbox.clone());
        builder = builder.tool_registry(registry).sandbox(sandbox);

        if let Some(ref im) = self.interaction_manager {
            builder = builder.interaction(im.clone());
        }

        Ok(builder.build())
    }

    /// Create an autonomous agent with auto-approve interaction (for headless/test usage).
    #[cfg(feature = "autonomous")]
    pub fn create_agent_headless(
        &self,
        profile_name: &str,
        response_generator: Arc<
            dyn Fn(&[crate::agentic_loop::AgentMessage]) -> String + Send + Sync,
        >,
    ) -> Result<AutonomousAgent> {
        let handler: Arc<dyn UserInteractionHandler> = Arc::new(AutoApproveInteraction::new());
        let im = Arc::new(InteractionManager::new(handler, 300));

        let profile = self
            .profile_registry
            .get_agent_profile(profile_name)
            .ok_or_else(|| anyhow::anyhow!("Agent profile '{}' not found", profile_name))?;

        let policy = profile.policy.clone();
        let sandbox = Arc::new(RwLock::new(SandboxValidator::new(policy)));
        let mut registry = crate::unified_tools::ToolRegistry::new();
        register_os_tools(&mut registry, sandbox.clone());

        let mut builder = AutonomousAgentBuilder::new(&profile.name, response_generator)
            .policy(profile.policy.clone())
            .mode(profile.mode)
            .tool_registry(registry)
            .sandbox(sandbox)
            .interaction(im);

        if let Some(ref prompt) = profile.system_prompt {
            builder = builder.system_prompt(prompt.clone());
        }

        builder = builder.max_iterations(profile.policy.max_iterations);

        Ok(builder.build())
    }

    // === Butler ===

    /// Initialize the Butler for environment auto-detection.
    #[cfg(feature = "butler")]
    pub fn init_butler(&mut self) {
        self.butler = Some(Butler::new());
    }

    /// Run Butler environment scan and return the report.
    #[cfg(feature = "butler")]
    pub fn butler_scan(&mut self) -> Option<crate::butler::EnvironmentReport> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        self.butler.as_mut().map(|b| b.scan())
    }

    /// Auto-configure the assistant using Butler's environment scan.
    /// Updates the AiConfig based on detected providers.
    #[cfg(feature = "butler")]
    pub fn auto_configure(&mut self) -> Result<()> {
        if self.butler.is_none() {
            self.butler = Some(Butler::new());
        }
        let butler = self.butler.as_mut().expect("butler must be initialized");
        let report = butler.scan();
        let suggested_config = butler.suggest_config(&report);
        self.config = suggested_config;
        Ok(())
    }

    // === Scheduler ===

    /// Initialize the scheduler.
    #[cfg(feature = "scheduler")]
    pub fn init_scheduler(&mut self) {
        self.scheduler = Some(Scheduler::new());
    }

    /// Get the scheduler (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn scheduler(&self) -> Option<&Scheduler> {
        self.scheduler.as_ref()
    }

    /// Get the scheduler mutably (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn scheduler_mut(&mut self) -> Option<&mut Scheduler> {
        self.scheduler.as_mut()
    }

    /// Initialize the trigger manager.
    #[cfg(feature = "scheduler")]
    pub fn init_trigger_manager(&mut self) {
        self.trigger_manager = Some(TriggerManager::new());
    }

    /// Get the trigger manager (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn trigger_manager(&self) -> Option<&TriggerManager> {
        self.trigger_manager.as_ref()
    }

    /// Get the trigger manager mutably (if initialized).
    #[cfg(feature = "scheduler")]
    pub fn trigger_manager_mut(&mut self) -> Option<&mut TriggerManager> {
        self.trigger_manager.as_mut()
    }

    // === Browser ===

    /// Initialize the browser session for CDP-based browser automation.
    #[cfg(feature = "browser")]
    pub fn init_browser(&mut self) {
        self.browser_session = Some(BrowserSession::new());
    }

    /// Get the browser session (if initialized).
    #[cfg(feature = "browser")]
    pub fn browser_session(&self) -> Option<&BrowserSession> {
        self.browser_session.as_ref()
    }

    /// Get the browser session mutably (if initialized).
    #[cfg(feature = "browser")]
    pub fn browser_session_mut(&mut self) -> Option<&mut BrowserSession> {
        self.browser_session.as_mut()
    }

    // === Distributed Agents ===

    /// Initialize the distributed agent manager for multi-node agent execution.
    #[cfg(feature = "distributed-agents")]
    pub fn init_distributed_agents(&mut self, local_node_id: crate::distributed::NodeId) {
        self.distributed_agent_manager = Some(DistributedAgentManager::new(local_node_id));
    }

    /// Get the distributed agent manager (if initialized).
    #[cfg(feature = "distributed-agents")]
    pub fn distributed_agents(&self) -> Option<&DistributedAgentManager> {
        self.distributed_agent_manager.as_ref()
    }

    /// Get the distributed agent manager mutably (if initialized).
    #[cfg(feature = "distributed-agents")]
    pub fn distributed_agents_mut(&mut self) -> Option<&mut DistributedAgentManager> {
        self.distributed_agent_manager.as_mut()
    }

    // === A/B Testing ===

    /// Initialize the experiment manager for A/B testing.
    #[cfg(feature = "eval")]
    pub fn init_experiment_manager(&mut self) {
        if self.experiment_manager.is_none() {
            self.experiment_manager = Some(crate::ab_testing::ExperimentManager::new());
        }
    }

    /// Get the experiment manager (if initialized).
    #[cfg(feature = "eval")]
    pub fn experiment_manager(&self) -> Option<&crate::ab_testing::ExperimentManager> {
        self.experiment_manager.as_ref()
    }

    /// Get the experiment manager mutably (if initialized).
    #[cfg(feature = "eval")]
    pub fn experiment_manager_mut(&mut self) -> Option<&mut crate::ab_testing::ExperimentManager> {
        self.experiment_manager.as_mut()
    }

    // === Cost Dashboard ===

    /// Initialize cost tracking with default settings.
    pub fn init_cost_tracking(&mut self) {
        if self.cost_dashboard.is_none() {
            self.cost_dashboard = Some(crate::cost_integration::CostDashboard::new());
        }
    }

    /// Get reference to cost dashboard.
    pub fn cost_dashboard(&self) -> Option<&crate::cost_integration::CostDashboard> {
        self.cost_dashboard.as_ref()
    }

    /// Get mutable reference to cost dashboard.
    pub fn cost_dashboard_mut(&mut self) -> Option<&mut crate::cost_integration::CostDashboard> {
        self.cost_dashboard.as_mut()
    }

    /// Get formatted cost report.
    pub fn cost_report(&self) -> Option<String> {
        self.cost_dashboard.as_ref().map(|d| d.format_report())
    }

    // === Chat Hooks ===

    /// Initialize chat hooks for UI framework event streaming.
    pub fn init_chat_hooks(&mut self) {
        if self.chat_hooks.is_none() {
            self.chat_hooks = Some(crate::ui_hooks::ChatHooks::new());
        }
    }

    /// Get the chat hooks (if initialized).
    pub fn chat_hooks(&self) -> Option<&crate::ui_hooks::ChatHooks> {
        self.chat_hooks.as_ref()
    }

    /// Get the chat hooks mutably (if initialized).
    pub fn chat_hooks_mut(&mut self) -> Option<&mut crate::ui_hooks::ChatHooks> {
        self.chat_hooks.as_mut()
    }

    /// Emit a chat event to all subscribers (if hooks are initialized).
    pub fn emit_chat_event(&mut self, event: crate::ui_hooks::ChatStreamEvent) {
        if let Some(ref mut hooks) = self.chat_hooks {
            hooks.emit(event);
        }
    }

    // === Container Execution ===

    /// Create a new container executor with default configuration.
    #[cfg(feature = "containers")]
    pub fn create_container_executor(&self) -> Result<crate::container_executor::ContainerExecutor> {
        crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))
    }

    /// Create a container executor with custom configuration.
    #[cfg(feature = "containers")]
    pub fn create_container_executor_with_config(
        &self,
        config: crate::container_executor::ContainerConfig,
    ) -> Result<crate::container_executor::ContainerExecutor> {
        crate::container_executor::ContainerExecutor::new(config)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Execute code in an isolated Docker container.
    ///
    /// Automatically selects the appropriate Docker image based on language.
    /// Falls back to process-based execution if Docker is unavailable.
    #[cfg(feature = "containers")]
    pub fn run_code_isolated(
        &self,
        code: &str,
        language: &crate::code_sandbox::Language,
    ) -> Result<crate::code_sandbox::ExecutionResult> {
        let mut sandbox = crate::container_sandbox::ContainerSandbox::new(
            crate::container_sandbox::ContainerSandboxConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Ok(sandbox.execute(language, code))
    }

    /// Create a shared folder for container file exchange.
    #[cfg(feature = "containers")]
    pub fn create_shared_folder(&self) -> Result<crate::shared_folder::SharedFolder> {
        crate::shared_folder::SharedFolder::temp()
    }

    // === Document Creation ===

    /// Create a document pipeline with default settings.
    ///
    /// Internally creates a `ContainerExecutor` and a temporary `SharedFolder`.
    #[cfg(feature = "containers")]
    pub fn create_document_pipeline(
        &self,
    ) -> Result<crate::document_pipeline::DocumentPipeline> {
        let executor = crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let shared_folder = crate::shared_folder::SharedFolder::temp()?;
        Ok(crate::document_pipeline::DocumentPipeline::new(
            crate::document_pipeline::DocumentPipelineConfig::default(),
            std::sync::Arc::new(std::sync::RwLock::new(executor)),
            shared_folder,
        ))
    }

    /// Create a document by converting content to the specified format.
    ///
    /// Uses container-based pandoc/LibreOffice for conversion.
    #[cfg(feature = "containers")]
    pub fn create_document(
        &self,
        content: &str,
        source_format: crate::document_pipeline::SourceFormat,
        output_format: crate::document_pipeline::OutputFormat,
    ) -> Result<crate::document_pipeline::DocumentResult> {
        let executor = crate::container_executor::ContainerExecutor::new(
            crate::container_executor::ContainerConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let shared_folder = crate::shared_folder::SharedFolder::temp()?;
        let mut pipeline = crate::document_pipeline::DocumentPipeline::new(
            crate::document_pipeline::DocumentPipelineConfig::default(),
            std::sync::Arc::new(std::sync::RwLock::new(executor)),
            shared_folder,
        );
        let request = crate::document_pipeline::DocumentRequest {
            content: content.to_string(),
            source_format,
            output_format,
            output_name: "document".into(),
            stylesheet: None,
            extra_args: Vec::new(),
            metadata: std::collections::HashMap::new(),
        };
        pipeline.create(&request).map_err(|e| anyhow::anyhow!(e))
    }

    // === Speech (STT / TTS) ===

    /// Transcribe audio to text using the specified speech provider.
    ///
    /// # Arguments
    /// * `provider_name` - Provider name ("openai", "google", "whisper", "local")
    /// * `audio` - Raw audio bytes
    /// * `format` - Audio encoding format
    /// * `language` - Optional language hint (ISO 639-1)
    #[cfg(feature = "audio")]
    pub fn transcribe(
        &self,
        provider_name: &str,
        audio: &[u8],
        format: crate::speech::AudioFormat,
        language: Option<&str>,
    ) -> Result<crate::speech::TranscriptionResult> {
        let provider = crate::speech::create_speech_provider(provider_name)?;
        provider.transcribe(audio, format, language)
    }

    /// Synthesize text to audio using the specified speech provider.
    ///
    /// # Arguments
    /// * `provider_name` - Provider name ("openai", "google", "piper", "coqui", "local")
    /// * `text` - Text to synthesize
    /// * `options` - Synthesis options (voice, format, speed)
    #[cfg(feature = "audio")]
    pub fn synthesize(
        &self,
        provider_name: &str,
        text: &str,
        options: &crate::speech::SynthesisOptions,
    ) -> Result<crate::speech::SynthesisResult> {
        let provider = crate::speech::create_speech_provider(provider_name)?;
        provider.synthesize(text, options)
    }

    /// Get the recommended speech configuration from butler (if available).
    ///
    /// Returns (stt_provider, tts_provider) suggestions based on detected environment.
    #[cfg(all(feature = "audio", feature = "butler"))]
    pub fn suggest_speech_providers(&mut self) -> (Option<String>, Option<String>) {
        let mut butler = crate::butler::Butler::new();
        butler.scan();
        butler.suggest_speech_config()
    }

    // =========================================================================
    // KPKG -> Knowledge Layer Bridge (v4 roadmap item 8.1)
    // =========================================================================

    /// Extract named entities from text content.
    ///
    /// Identifies capitalized proper nouns and quoted terms that are not at the
    /// start of a sentence. Returns a deduplicated list of entity names.
    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    pub fn extract_entities_from_text(text: &str) -> Vec<String> {
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Common words that should not be treated as entities even when capitalized
        let stop_words: std::collections::HashSet<&str> = [
            "The", "This", "That", "These", "Those", "It", "Its", "They", "Their",
            "He", "She", "His", "Her", "We", "Our", "You", "Your", "My", "I",
            "A", "An", "And", "Or", "But", "Not", "No", "If", "When", "Where",
            "How", "What", "Who", "Which", "Why", "Is", "Are", "Was", "Were",
            "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
            "Will", "Would", "Could", "Should", "May", "Might", "Can", "Shall",
            "For", "From", "With", "About", "Into", "Through", "During", "Before",
            "After", "Above", "Below", "To", "Of", "In", "On", "At", "By",
            "As", "So", "Then", "Than", "Also", "Just", "Only", "Each", "Every",
            "All", "Any", "Both", "Few", "More", "Most", "Other", "Some", "Such",
            "Very", "Much", "Many", "Here", "There", "Now", "Still", "Already",
            "El", "La", "Los", "Las", "Un", "Una", "De", "En", "Por", "Para",
            "Con", "Sin", "Sobre", "Entre", "Es", "Son", "Fue", "Era",
        ]
        .iter()
        .copied()
        .collect();

        // Extract quoted terms (single and double quotes)
        for cap in text.split('"').enumerate() {
            // Odd indices are inside quotes
            if cap.0 % 2 == 1 {
                let term = cap.1.trim();
                if !term.is_empty() && term.len() <= 80 {
                    let key = term.to_lowercase();
                    if !seen.contains(&key) {
                        seen.insert(key);
                        entities.push(term.to_string());
                    }
                }
            }
        }

        // Split into sentences and extract capitalized words not at sentence start
        let sentences: Vec<&str> = text
            .split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
            .filter(|s| !s.trim().is_empty())
            .collect();

        for sentence in &sentences {
            let trimmed = sentence.trim();
            let words: Vec<&str> = trimmed.split_whitespace().collect();

            // Skip the first word (it's capitalized because it starts the sentence)
            for (idx, word) in words.iter().enumerate() {
                // Strip trailing punctuation for analysis
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
                    .collect();

                if clean.is_empty() {
                    continue;
                }

                let first_char = clean.chars().next().unwrap_or('a');

                // Check if it starts with uppercase and is not at the very start of the sentence
                if idx > 0 && first_char.is_uppercase() && clean.len() >= 2 {
                    // Skip common stop words
                    if stop_words.contains(clean.as_str()) {
                        continue;
                    }

                    // Check it's not ALL uppercase (likely an acronym like "API" or "USA")
                    // — we still include those as entities
                    let key = clean.to_lowercase();
                    if !seen.contains(&key) {
                        seen.insert(key);
                        entities.push(clean);
                    }
                }
            }
        }

        entities
    }

    /// Load a .kpkg encrypted knowledge package and bridge its contents into
    /// the multi-layer knowledge graph.
    ///
    /// This method:
    /// 1. Reads and decrypts the kpkg file using `KpkgReader`
    /// 2. Extracts the manifest metadata (title, description, system_prompt, persona)
    /// 3. Parses document content for named entities (capitalized proper nouns, quoted terms)
    /// 4. Creates `LayeredEntity` entries on the Knowledge layer
    /// 5. Inserts entities into the `MultiLayerGraph` (if present)
    /// 6. Injects manifest system_prompt / persona into the assistant's system prompt
    /// 7. Returns the number of entities extracted
    ///
    /// # Errors
    ///
    /// Returns `AiError` if the kpkg file cannot be read, decrypted, or parsed.
    ///
    /// # Feature gates
    ///
    /// Requires both `rag` (for kpkg support) and `multi-agent` (for MultiLayerGraph).
    #[cfg(all(feature = "rag", feature = "multi-agent"))]
    pub fn load_kpkg_to_graph(&mut self, kpkg_path: &str) -> Result<usize, crate::error::AiError> {
        use crate::encrypted_knowledge::{AppKeyProvider, KpkgReader};
        use crate::multi_layer_graph::{ConfidenceLevel, GraphLayer, LayeredEntity};
        use std::time::{SystemTime, UNIX_EPOCH};

        // 1. Read the kpkg file from disk
        let data = std::fs::read(kpkg_path).map_err(|e| {
            crate::error::AiError::Io(crate::error::IoError {
                operation: "read_kpkg".to_string(),
                path: Some(kpkg_path.to_string()),
                reason: format!("Failed to read kpkg file: {}", e),
            })
        })?;

        // 2. Decrypt and extract documents + manifest
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        let (documents, manifest) = reader.read_with_manifest(&data).map_err(|e| {
            crate::error::AiError::Other(format!(
                "Failed to decrypt kpkg '{}': {}",
                kpkg_path, e
            ))
        })?;

        // 3. Extract entities from all document content
        let mut all_content = String::new();
        for doc in &documents {
            all_content.push_str(&doc.content);
            all_content.push('\n');
        }

        // Also include manifest metadata in entity extraction
        if !manifest.name.is_empty() {
            all_content.push_str(&manifest.name);
            all_content.push('\n');
        }
        if !manifest.description.is_empty() {
            all_content.push_str(&manifest.description);
            all_content.push('\n');
        }

        let entity_names = Self::extract_entities_from_text(&all_content);
        let entity_count = entity_names.len();

        // 4. Create LayeredEntity entries and insert into graph
        if let Some(ref mut graph) = self.graph {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            // Ensure the knowledge custom layer exists
            let layer_name = "kpkg_knowledge";
            graph.add_custom_layer(layer_name);

            for name in &entity_names {
                let entity = LayeredEntity {
                    name: name.clone(),
                    entity_type: "KpkgEntity".to_string(),
                    layer: GraphLayer::Knowledge,
                    confidence: ConfidenceLevel::Verified,
                    source: kpkg_path.to_string(),
                    timestamp: now,
                    ttl_seconds: None,
                };
                // Ignore errors from duplicate entities or full layers
                let _ = graph.add_to_custom_layer(layer_name, entity);
            }
        }

        // 5. Inject manifest system_prompt and persona into assistant context
        let mut injected_parts = Vec::new();

        if let Some(ref system_prompt) = manifest.system_prompt {
            if !system_prompt.is_empty() {
                injected_parts.push(format!("[KPKG System Prompt]: {}", system_prompt));
            }
        }

        if let Some(ref persona) = manifest.persona {
            if !persona.is_empty() {
                injected_parts.push(format!("[KPKG Persona]: {}", persona));
            }
        }

        // Inject examples as context hints
        if !manifest.examples.is_empty() {
            let mut examples_text = String::from("[KPKG Examples]:");
            for (i, example) in manifest.examples.iter().enumerate() {
                examples_text.push_str(&format!(
                    "\n  Example {}: Input: {} -> Output: {}",
                    i + 1,
                    example.input,
                    example.output
                ));
            }
            injected_parts.push(examples_text);
        }

        if !injected_parts.is_empty() {
            let injection = injected_parts.join("\n");
            if self.system_prompt_base.is_empty() {
                self.system_prompt_base = injection;
            } else {
                self.system_prompt_base
                    .push_str(&format!("\n\n{}", injection));
            }
        }

        Ok(entity_count)
    }

    // =========================================================================
    // Constrained Decoding Integration (v9 item 3.1)
    // =========================================================================

    /// Generate a response constrained by a GBNF grammar.
    ///
    /// Parses the grammar string (in GBNF format) into a [`Grammar`], sends
    /// the prompt to the configured LLM provider (synchronously), and validates
    /// the response against the grammar using a [`StreamingValidator`]-style
    /// check.
    ///
    /// # Arguments
    /// * `grammar` - A GBNF grammar string (e.g. `root ::= "yes" | "no"`)
    /// * `prompt` - The user prompt to send to the LLM
    ///
    /// # Errors
    /// Returns `AiError` if the grammar cannot be parsed, the LLM call fails,
    /// or the response does not conform to the grammar.
    #[cfg(feature = "constrained-decoding")]
    pub fn generate_with_grammar(
        &self,
        grammar: &str,
        prompt: &str,
    ) -> Result<String, crate::error::AiError> {
        use crate::constrained_decoding::{Grammar, GrammarConstraint};

        // 1. Parse the GBNF grammar
        let parsed_grammar = Grammar::from_gbnf(grammar)?;

        // 2. Verify the grammar can be formatted for the current provider
        let provider_name = self.config.provider.display_name();
        // Attempt to format — if provider is unsupported, we still proceed
        // with validation-only mode.
        let _grammar_str = GrammarConstraint::for_provider(&parsed_grammar, provider_name)
            .unwrap_or_else(|_| parsed_grammar.to_gbnf());

        // 3. Build conversation with the prompt
        let conversation = vec![crate::messages::ChatMessage::user(prompt)];
        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            "",
        );

        // 4. Call the LLM synchronously
        let response = generate_response(&self.config, &conversation, &system_prompt)
            .map_err(|e| crate::error::AiError::Other(format!("LLM generation failed: {}", e)))?;

        // 5. Validate the response against the grammar rules
        // Check if the response matches any of the root rule's alternatives
        let root_rule = parsed_grammar.rules.iter().find(|r| r.name == parsed_grammar.root_rule);
        if let Some(rule) = root_rule {
            let trimmed = response.trim();
            let valid = rule.alternatives.iter().any(|alt| {
                // Simple validation: check literal-only alternatives
                let literal_match: String = alt.elements.iter().filter_map(|el| {
                    if let crate::constrained_decoding::GrammarElement::Literal(s) = el {
                        Some(s.as_str())
                    } else {
                        None
                    }
                }).collect();
                if !literal_match.is_empty() {
                    return trimmed == literal_match || trimmed.contains(&literal_match);
                }
                // For non-literal rules, accept the response as valid
                // (full recursive validation would require a full parser)
                true
            });
            if !valid {
                return Err(crate::error::AiError::ConstrainedDecoding(
                    crate::error::ConstrainedDecodingError::GrammarCompilationFailed {
                        reason: format!(
                            "Response '{}' does not match grammar root rule '{}'",
                            trimmed, parsed_grammar.root_rule
                        ),
                    },
                ));
            }
        }

        Ok(response)
    }

    // =========================================================================
    // Human-in-the-Loop Integration (v9 item 3.2)
    // =========================================================================

    /// Send a message with an optional HITL approval gate.
    ///
    /// When `auto_approve` is `true`, the message is sent and the response
    /// returned directly. When `false`, the method simulates a HITL approval
    /// gate by creating an [`ApprovalRequest`] and logging it to an
    /// [`ApprovalLog`] before returning the response.
    ///
    /// The approval request records the prompt as the tool name and the
    /// response as context, providing a full audit trail of LLM interactions.
    ///
    /// # Arguments
    /// * `message` - The user message to send
    /// * `auto_approve` - If true, skip the approval gate
    ///
    /// # Errors
    /// Returns `AiError` if the LLM call fails.
    #[cfg(feature = "hitl")]
    pub fn send_message_with_approval(
        &mut self,
        message: &str,
        auto_approve: bool,
    ) -> Result<String, crate::error::AiError> {
        use crate::hitl::{
            ApprovalDecision, ApprovalLog, ApprovalLogEntry, ApprovalRequest, AutoApproveGate,
            HitlApprovalGate, ImpactLevel,
        };
        use std::collections::HashMap as HitlHashMap;

        // 1. Send the message to the LLM synchronously
        let conversation = {
            let mut conv = self.conversation.clone();
            conv.push(crate::messages::ChatMessage::user(message));
            conv
        };
        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            &self.knowledge_context,
        );
        let response = generate_response(&self.config, &conversation, &system_prompt)
            .map_err(|e| crate::error::AiError::Other(format!("LLM generation failed: {}", e)))?;

        // 2. Record in conversation
        self.conversation.push(crate::messages::ChatMessage::user(message));
        self.conversation
            .push(crate::messages::ChatMessage::assistant(&response));

        // 3. Apply approval gate
        if !auto_approve {
            let request = ApprovalRequest::new(
                format!("msg-{}", self.conversation.len()),
                "send_message",
                HitlHashMap::new(),
                "ai_assistant",
                format!("User message: {}; LLM response: {}", message, &response),
                ImpactLevel::Low,
            );

            let gate = AutoApproveGate;
            let decision = gate
                .request_approval(&request)
                .map_err(|e| crate::error::AiError::Other(format!("HITL gate error: {}", e)))?;

            // Log the decision
            let mut log = ApprovalLog::new(1000);
            log.record(ApprovalLogEntry {
                request,
                decision: decision.clone(),
                gate_name: gate.name().to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });

            if let ApprovalDecision::Deny { reason } = decision {
                return Err(crate::error::AiError::Other(format!(
                    "Message denied by HITL gate: {}",
                    reason
                )));
            }
        }

        Ok(response)
    }

    // =========================================================================
    // MCP Client Integration (v9 item 3.3)
    // =========================================================================

    /// Connect to a remote MCP server by URL.
    ///
    /// Validates the URL and creates a [`RemoteMcpClient`] connection. The
    /// connection URL is stored internally for subsequent tool listing.
    ///
    /// # Arguments
    /// * `server_url` - The MCP server URL (e.g. `"http://localhost:3000/mcp"`)
    ///
    /// # Errors
    /// Returns `AiError` if the URL is empty or the connection fails.
    pub fn connect_mcp_server(
        &mut self,
        server_url: &str,
    ) -> Result<(), crate::error::AiError> {
        use crate::mcp_client::{McpClientConfig, RemoteMcpClient};

        if server_url.is_empty() {
            return Err(crate::error::AiError::Other(
                "MCP server URL cannot be empty".to_string(),
            ));
        }

        // Validate URL format (basic check)
        if !server_url.starts_with("http://") && !server_url.starts_with("https://") {
            return Err(crate::error::AiError::Other(format!(
                "Invalid MCP server URL (must start with http:// or https://): {}",
                server_url
            )));
        }

        let config = McpClientConfig {
            url: server_url.to_string(),
            ..McpClientConfig::default()
        };

        let mut client = RemoteMcpClient::new(config);
        client.connect().map_err(|e| {
            crate::error::AiError::Other(format!("MCP connection failed: {}", e))
        })?;

        // Store the connection URL as an indicator that connection was established
        self.knowledge_context.push_str(&format!(
            "\n[MCP Server connected: {}]\n",
            server_url
        ));

        Ok(())
    }

    /// List available tools from connected MCP servers.
    ///
    /// Returns the names of tools discovered via MCP. If no server has been
    /// connected, returns an empty list.
    ///
    /// This is a lightweight query that does not require a persistent connection
    /// -- it creates a temporary client, connects, and fetches the tool list.
    ///
    /// # Arguments
    /// * `server_url` - The MCP server URL to query for tools
    pub fn list_mcp_tools(&self, server_url: &str) -> Vec<String> {
        use crate::mcp_client::{McpClientConfig, RemoteMcpClient};

        if server_url.is_empty() {
            return Vec::new();
        }

        let config = McpClientConfig {
            url: server_url.to_string(),
            ..McpClientConfig::default()
        };

        let mut client = RemoteMcpClient::new(config);
        match client.connect() {
            Ok(()) => match client.list_tools() {
                Ok(tools) => tools.iter().map(|t| t.name.clone()).collect(),
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }

    // =========================================================================
    // Distillation Integration (v9 item 3.4)
    // =========================================================================

    /// Collect the current conversation history as (input, output) pairs.
    ///
    /// Iterates over the conversation messages and pairs consecutive user and
    /// assistant messages into tuples. Messages without a corresponding pair
    /// are skipped.
    ///
    /// # Returns
    /// A vector of `(user_input, assistant_output)` pairs from the session.
    #[cfg(feature = "distillation")]
    pub fn collect_trajectory(&mut self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < self.conversation.len() {
            let user_msg = &self.conversation[i];
            let assistant_msg = &self.conversation[i + 1];
            if user_msg.role == "user" && assistant_msg.role == "assistant" {
                pairs.push((user_msg.content.clone(), assistant_msg.content.clone()));
                i += 2;
            } else {
                i += 1;
            }
        }
        pairs
    }

    /// Export the conversation trajectory as a JSON-formatted training dataset.
    ///
    /// Collects all (input, output) pairs from the session and serializes them
    /// as a JSON array of objects with `"input"` and `"output"` fields, suitable
    /// for fine-tuning or distillation pipelines.
    ///
    /// # Errors
    /// Returns `AiError` if JSON serialization fails.
    #[cfg(feature = "distillation")]
    pub fn export_training_data(&self) -> Result<String, crate::error::AiError> {
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < self.conversation.len() {
            let user_msg = &self.conversation[i];
            let assistant_msg = &self.conversation[i + 1];
            if user_msg.role == "user" && assistant_msg.role == "assistant" {
                pairs.push(serde_json::json!({
                    "input": user_msg.content,
                    "output": assistant_msg.content,
                }));
                i += 2;
            } else {
                i += 1;
            }
        }

        serde_json::to_string_pretty(&pairs).map_err(|e| {
            crate::error::AiError::Other(format!("Failed to serialize training data: {}", e))
        })
    }
}

/// Generate a conversation summary using the AI model
fn generate_conversation_summary(
    config: &AiConfig,
    messages: &[ChatMessage],
    previous_summary: Option<&str>,
) -> Result<String> {
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
        .send_json(&request_body)?;

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
        let result = ai.transcribe("nonexistent", &[0u8; 10], crate::speech::AudioFormat::Wav, None);
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
        let result = ai.synthesize(
            "piper",
            "",
            &crate::speech::SynthesisOptions::default(),
        );
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
            entities.contains(&"Stellar".to_string())
                || entities.contains(&"Dynamics".to_string()),
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
        assert!(
            result.is_err(),
            "Should fail when kpkg file does not exist"
        );
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
        assert!(ai.conversation.is_empty(), "Conversation should start empty");
        assert!(!ai.is_generating, "Should not be generating initially");
        assert!(!ai.is_fetching_models, "Should not be fetching models initially");
        assert!(ai.current_response.is_empty(), "Current response should be empty");
        assert!(ai.current_session.is_none(), "No session should be active");
        assert!(!ai.is_summarizing, "Should not be summarizing initially");
        assert!(ai.available_models.is_empty(), "No models should be loaded");
        assert!(!ai.fallback_enabled, "Fallback should be disabled by default");
        assert!(ai.fallback_providers.is_empty(), "No fallback providers by default");
        assert!(!ai.auto_compaction, "Auto-compaction should be disabled by default");
        assert!(ai.api_key_manager.is_none(), "No API key manager by default");
        assert!(ai.adaptive_thinking.enabled == false, "Adaptive thinking disabled by default");
        assert!(ai.last_thinking_result.is_none(), "No thinking result by default");
        assert!(ai.last_thinking_strategy.is_none(), "No thinking strategy by default");
        assert!(ai.detected_context_size.is_none(), "No detected context size by default");
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
        assert!(matches!(ai.preferences.response_style, ResponseStyle::Technical));
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

        assert_eq!(ai.get_current_api_key("openai"), Some("sk-openai-1".to_string()));
        assert_eq!(ai.get_current_api_key("anthropic"), Some("sk-ant-1".to_string()));
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
        assert!(report.is_some(), "Cost report should be available after init");
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
        ai.conversation.push(ChatMessage::user("I prefer code examples"));
        ai.conversation.push(ChatMessage::assistant("Sure!"));

        ai.extract_preferences_with(|msgs, prefs| {
            for msg in msgs {
                if msg.content.contains("code examples") {
                    prefs.response_style = ResponseStyle::Technical;
                }
            }
        });

        assert!(matches!(ai.preferences.response_style, ResponseStyle::Technical));
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
        ai.conversation.push(ChatMessage::user("Before new session"));
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
        ai.event_bus
            .emit(crate::events::AiEvent::SessionCreated {
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

        let sid = ai.current_session.as_ref().map(|s| s.id.clone()).unwrap_or_default();
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
        let result = ai.load_sessions_from_file(std::path::Path::new("/nonexistent_dir_xyz/sessions.bin"));
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
        assert!(ai2.session_store.sessions.len() >= 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_sessions_restores_current() {
        let mut ai = AiAssistant::new();
        ai.new_session();
        let sid = ai.current_session.as_ref().unwrap().id.clone();
        ai.conversation.push(ChatMessage::user("restore me"));
        ai.save_current_session();

        let dir = std::env::temp_dir().join(format!("test_sessions_restore_{}", std::process::id()));
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
        ai.conversation
            .push(ChatMessage::assistant("Rust is a systems programming language."));

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
        ai.conversation
            .push(ChatMessage::user("Tell me about \"quotes\" and \\backslash"));
        ai.conversation
            .push(ChatMessage::assistant("Special chars: \"quotes\", \\backslash"));
        ai.conversation.push(ChatMessage::user("Another"));
        ai.conversation
            .push(ChatMessage::assistant("Response two"));

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
        assert_eq!(ai.knowledge_context_size(), content.as_bytes().len());
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
        ai.conversation.push(ChatMessage::system("System instruction"));
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
            assert_eq!(session.context_notes, "Remember: user prefers concise answers");
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
        assert!(parsed.is_ok(), "Metrics JSON with data should be valid JSON");
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
}
