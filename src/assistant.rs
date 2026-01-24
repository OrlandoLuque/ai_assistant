//! Main AI Assistant implementation

use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::path::Path;
use anyhow::Result;

use crate::config::{AiConfig, AiProvider};
use crate::messages::{AiResponse, ChatMessage};
use crate::models::ModelInfo;
use crate::session::{ChatSession, ChatSessionStore, UserPreferences, ResponseStyle};
use crate::context::{ContextUsage, estimate_tokens, get_model_context_size};
use crate::providers::{
    fetch_ollama_models, fetch_openai_compatible_models, fetch_kobold_models,
    build_system_prompt, build_system_prompt_with_notes, generate_response_streaming, generate_response,
    generate_response_streaming_cancellable,
};
use crate::conversation_control::CancellationToken;

#[cfg(feature = "rag")]
use crate::rag::{RagDb, RagConfig, KnowledgeUsage, build_knowledge_context, build_conversation_context, DEFAULT_USER_ID};
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
    Starting { source: String, total_documents: usize, current: usize },
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
}

impl Default for AiAssistant {
    fn default() -> Self {
        Self::new()
    }
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

    /// Load configuration
    pub fn load_config(&mut self, config: AiConfig) {
        self.config = config;
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
            if let Ok(models) = fetch_openai_compatible_models(&lm_studio_url, AiProvider::LMStudio) {
                all_models.extend(models);
            }

            // Try text-generation-webui
            if let Ok(models) = fetch_openai_compatible_models(&text_gen_webui_url, AiProvider::TextGenWebUI) {
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

    // === Message Handling ===

    /// Send a message and start generating a response
    ///
    /// # Arguments
    /// * `user_message` - The user's message
    /// * `knowledge_context` - Optional knowledge/context to include in system prompt
    pub fn send_message(&mut self, user_message: String, knowledge_context: &str) {
        self.conversation.push(ChatMessage::user(&user_message));
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

        thread::spawn(move || {
            let result = generate_response_streaming(&config, &conversation, &system_prompt, &tx);
            if let Err(e) = result {
                let _ = tx.send(AiResponse::Error(e.to_string()));
            }
        });
    }

    /// Send a message without knowledge context
    pub fn send_message_simple(&mut self, user_message: String) {
        self.send_message(user_message, "");
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
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let config = self.config.clone();
        let conversation = self.conversation.clone();
        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
            session_notes,
            knowledge_notes,
        );

        thread::spawn(move || {
            let result = generate_response_streaming(&config, &conversation, &system_prompt, &tx);
            if let Err(e) = result {
                let _ = tx.send(AiResponse::Error(e.to_string()));
            }
        });
    }

    /// Generate a response synchronously (blocking)
    pub fn generate_sync(&mut self, user_message: String, knowledge_context: &str) -> Result<String> {
        self.conversation.push(ChatMessage::user(&user_message));

        let system_prompt = build_system_prompt(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
        );

        let response = generate_response(&self.config, &self.conversation, &system_prompt)?;

        self.conversation.push(ChatMessage::assistant(&response));
        self.extract_preferences_from_response(&response);

        Ok(response)
    }

    // === Cancellable Streaming ===

    /// Send a message with cancellation support
    ///
    /// Returns a CancellationToken that can be used to cancel the generation
    pub fn send_message_cancellable(&mut self, user_message: String, knowledge_context: &str) -> CancellationToken {
        self.conversation.push(ChatMessage::user(&user_message));
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

        let token = cancel_token.clone();
        thread::spawn(move || {
            let result = generate_response_streaming_cancellable(&config, &conversation, &system_prompt, &tx, &token);
            if let Err(e) = result {
                let _ = tx.send(AiResponse::Error(e.to_string()));
            }
        });

        cancel_token
    }

    /// Send a message with cancellation support (no knowledge context)
    pub fn send_message_cancellable_simple(&mut self, user_message: String) -> CancellationToken {
        self.send_message_cancellable(user_message, "")
    }

    /// Send a message with full context and cancellation support
    pub fn send_message_cancellable_with_notes(
        &mut self,
        user_message: String,
        knowledge_context: &str,
        session_notes: &str,
        knowledge_notes: &str,
    ) -> CancellationToken {
        self.conversation.push(ChatMessage::user(&user_message));
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let cancel_token = CancellationToken::new();
        self.cancel_token = Some(cancel_token.clone());

        let config = self.config.clone();
        let conversation = self.conversation.clone();
        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_context,
            session_notes,
            knowledge_notes,
        );

        let token = cancel_token.clone();
        thread::spawn(move || {
            let result = generate_response_streaming_cancellable(&config, &conversation, &system_prompt, &tx, &token);
            if let Err(e) = result {
                let _ = tx.send(AiResponse::Error(e.to_string()));
            }
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

    /// Poll for response chunks/completion
    pub fn poll_response(&mut self) -> Option<AiResponse> {
        if let Some(ref rx) = self.rx_response {
            match rx.try_recv() {
                Ok(response) => {
                    match &response {
                        AiResponse::Complete(text) => {
                            self.current_response = text.clone();
                            self.conversation.push(ChatMessage::assistant(text));
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                            self.extract_preferences_from_response(text);
                        }
                        AiResponse::Cancelled(partial) => {
                            // Store partial response but don't add to conversation
                            self.current_response = partial.clone();
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                        }
                        AiResponse::Chunk(chunk) => {
                            self.current_response.push_str(chunk);
                        }
                        AiResponse::Error(_) => {
                            self.is_generating = false;
                            self.rx_response = None;
                            self.cancel_token = None;
                        }
                        _ => {}
                    }
                    return Some(response);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.is_generating = false;
                    self.rx_response = None;
                    self.cancel_token = None;
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
                } else if content_lower.contains("explain in detail") || content_lower.contains("detailed") {
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
        self.current_session = Some(session);
        self.conversation.clear();
        self.current_response.clear();
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
        } else if !self.conversation.is_empty() {
            let mut session = ChatSession::new("New Chat");
            session.messages = self.conversation.clone();
            session.preferences = self.preferences.clone();
            session.auto_name();

            self.session_store.current_session_id = Some(session.id.clone());
            self.session_store.save_session(session.clone());
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
        }
    }

    /// Delete a session by ID
    pub fn delete_session(&mut self, session_id: &str) {
        self.session_store.delete_session(session_id);

        if self.current_session.as_ref().map(|s| s.id.as_str()) == Some(session_id) {
            self.current_session = None;
            self.conversation.clear();
        }
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

        let history_start = self.conversation.len().saturating_sub(self.config.max_history_messages);
        let conversation_tokens: usize = self.conversation[history_start..]
            .iter()
            .map(|msg| estimate_tokens(&msg.content) + 4) // +4 for role tokens
            .sum();

        let max_tokens = get_model_context_size(&self.config.selected_model);

        ContextUsage::calculate(system_tokens, knowledge_tokens, conversation_tokens, max_tokens)
    }

    /// Get dynamic max history based on knowledge size
    pub fn get_effective_max_history(&self, knowledge: &str) -> usize {
        let knowledge_tokens = estimate_tokens(knowledge);
        let max_tokens = get_model_context_size(&self.config.selected_model);

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
        let previous_summary = self.conversation.first()
            .filter(|msg| msg.role == "system" && msg.content.starts_with("[Conversation summary:"))
            .map(|msg| {
                msg.content
                    .trim_start_matches("[Conversation summary: ")
                    .trim_end_matches(']')
                    .to_string()
            });

        let skip_count = if previous_summary.is_some() { 1 } else { 0 };
        let messages_to_summarize: Vec<ChatMessage> = self.conversation
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
            let result = generate_conversation_summary(&config, &messages_to_summarize, previous_summary.as_deref());
            match result {
                Ok(summary) => {
                    let _ = tx.send(SummaryResult {
                        summary,
                        messages_summarized: to_summarize,
                    });
                }
                Err(_) => {
                    let fallback = create_simple_summary(&messages_to_summarize, previous_summary.as_deref());
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
                        let kept_messages: Vec<ChatMessage> = self.conversation
                            .iter()
                            .skip(keep_start)
                            .cloned()
                            .collect();

                        self.conversation.clear();
                        self.conversation.push(ChatMessage::system(
                            format!("[Conversation summary: {}]", result.summary)
                        ));
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
                    eprintln!("[AI RAG] Failed to initialize database: {}", e);
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
        self.pending_documents.insert(source.to_string(), content.to_string());
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
                        eprintln!("[AI RAG] Failed to index '{}': {}", source, e);
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
            self.pending_documents.insert(doc.source.clone(), doc.content);
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

        on_progress(IndexingProgress::AllComplete { results: results.clone() });
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

        if let Some(ref db) = self.rag_db {
            // Knowledge RAG with caching
            if self.rag_config.knowledge_rag_enabled {
                // Check cache first
                let cache_key = format!("{}_{}", query, self.rag_config.max_knowledge_tokens);
                let cached = self.rag_cache.as_mut().and_then(|c| c.get(&cache_key));

                let chunks = if let Some(cached_chunks) = cached {
                    self.metrics.record_cache_hit();
                    cached_chunks
                } else {
                    self.metrics.record_cache_miss();
                    if let Ok(search_chunks) = db.search_knowledge(
                        query,
                        self.rag_config.max_knowledge_tokens,
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
                let session_id = self.current_session
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
    pub fn build_rag_context_with_tracking(&mut self, query: &str) -> (String, String, Option<KnowledgeUsage>) {
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

        if let Some(ref db) = self.rag_db {
            // Knowledge RAG with source filtering
            if self.rag_config.knowledge_rag_enabled && !sources.is_empty() {
                // No caching for filtered searches (cache key would need to include sources)
                self.metrics.record_cache_miss();

                if let Ok(chunks) = db.search_knowledge_filtered(
                    query,
                    sources,
                    self.rag_config.max_knowledge_tokens,
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
                let session_id = self.current_session
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
                let session_id = self.current_session
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
                let session_id = self.current_session
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
                let session_id = self.current_session
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
                let session_id = self.current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");
                return db.get_session_notes(&self.user_id, session_id)
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
                let session_id = self.current_session
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
    pub fn import_knowledge_from_file(&mut self, path: &std::path::Path, replace: bool) -> Result<usize> {
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
    pub fn import_knowledge(&mut self, data: &crate::rag::KnowledgeExport, replace: bool) -> Result<usize> {
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
}

/// Generate a conversation summary using the AI model
fn generate_conversation_summary(
    config: &AiConfig,
    messages: &[ChatMessage],
    previous_summary: Option<&str>,
) -> Result<String> {
    let mut conversation_text = String::new();
    for msg in messages {
        let role = if msg.role == "user" { "User" } else { "Assistant" };
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
        ("Write a brief summary (3-4 sentences) capturing the main points.", 150)
    } else if message_count <= 8 {
        ("Write a comprehensive summary (5-8 sentences) covering all key topics.", 300)
    } else {
        ("Write a detailed summary (8-12 sentences) preserving all important context.", 500)
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
