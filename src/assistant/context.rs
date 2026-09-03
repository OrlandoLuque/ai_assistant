use super::*;

impl AiAssistant {
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

    /// Set an optional LLM enhancer for improving pipeline quality.
    /// When set, modules like intent classification, entity extraction,
    /// and response quality scoring will use LLM calls for better results.
    pub fn set_llm_enhancer(&mut self, enhancer: Box<dyn crate::llm_enhance::LlmEnhancer>) {
        self.llm_enhancer = Some(enhancer);
    }

    /// Get a reference to the current LLM enhancer, if set.
    pub fn llm_enhancer(&self) -> Option<&dyn crate::llm_enhance::LlmEnhancer> {
        self.llm_enhancer.as_deref()
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
    pub(crate) fn apply_adaptive_thinking(
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

        // Reserve based on token precision (dynamic: 10-20% depending on model)
        let precision = self.token_precision();
        let response_reserve = (total_context as f64 * precision.reserve_factor()) as usize;

        // Estimate tokens using model-aware counting
        let system_tokens = self.estimate_tokens_for_current_model(&self.system_prompt_base);

        // Conversation history tokens (FreshContext mode = 0)
        let conversation_tokens: usize = match self.context_mode {
            ContextMode::Conversation => self
                .conversation
                .iter()
                .map(|msg| self.estimate_tokens_for_current_model(&msg.content))
                .sum(),
            ContextMode::FreshContext => 0,
        };

        // User message tokens
        let user_tokens = self.estimate_tokens_for_current_model(user_message);

        // Calculate available
        let used = system_tokens + conversation_tokens + user_tokens + response_reserve;
        let available = total_context.saturating_sub(used);

        // Leave a small buffer (5%) for safety
        let safe_available = (available as f32 * 0.95) as usize;

        log::debug!(
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
            log::info!(
                "Model changed: from={} to={}",
                old_model,
                self.config.selected_model
            );
        }
    }

    /// Load preferences
    pub fn load_preferences(&mut self, preferences: UserPreferences) {
        self.preferences = preferences;
    }

    // === FreshContext Advisor ===

    /// Report the health and effectiveness of the current FreshContext configuration.
    ///
    /// `has_graph` must be passed because `KnowledgeGraph` lives outside `AiAssistant`
    /// (typically on the GUI app struct). Pass `true` if a knowledge graph is active.
    ///
    /// This method is usable from both GUI code and library consumers directly.
    #[cfg(feature = "rag")]
    pub fn fresh_context_status(&mut self, has_graph: bool) -> FreshContextStatus {
        let rag_available = self.has_rag();
        let sources_indexed = self.registered_sources.len();
        let memory_available = self.memory_manager.is_some();
        let available = self.calculate_available_knowledge_tokens("estimate");

        let mut warnings = Vec::new();

        if !rag_available {
            warnings.push(FreshContextWarning::NoRag);
        } else if sources_indexed == 0 {
            warnings.push(FreshContextWarning::NoSourcesIndexed);
        }
        if !has_graph {
            warnings.push(FreshContextWarning::NoGraph);
        }
        if !memory_available {
            warnings.push(FreshContextWarning::NoMemory);
        }
        if available < 500 {
            warnings.push(FreshContextWarning::SmallBudget(available));
        }

        let effectiveness = match (
            rag_available && sources_indexed > 0,
            has_graph,
            memory_available,
        ) {
            (true, true, true) => FreshContextEffectiveness::Optimal,
            (true, true, false) | (true, false, true) => FreshContextEffectiveness::Good,
            (true, false, false) => FreshContextEffectiveness::Limited,
            (false, _, _) => FreshContextEffectiveness::Ineffective,
        };

        FreshContextStatus {
            mode: self.context_mode,
            rag_available,
            sources_indexed,
            graph_available: has_graph,
            memory_available,
            available_knowledge_tokens: available,
            warnings,
            effectiveness,
        }
    }

    /// Estimate tokens using the best available method for the current model.
    ///
    /// Uses tiktoken (if `precise-tokens` feature + OpenAI model), BPE-200
    /// (cloud models), or ~3.5 chars/token heuristic (local/unknown).
    pub fn estimate_tokens_for_current_model(&self, text: &str) -> usize {
        crate::context::estimate_tokens_for_model(text, &self.config.selected_model)
    }

    /// Get the token precision level for the current model.
    pub fn token_precision(&self) -> crate::token_counter::TokenPrecision {
        use crate::token_counter::ProviderTokenCounter;
        thread_local! {
            static COUNTER: ProviderTokenCounter = ProviderTokenCounter::new();
        }
        COUNTER.with(|c| c.precision_for_model(&self.config.selected_model))
    }

    /// Get the context budget status with recommendations from the Butler advisor.
    pub fn context_budget_status(&self) -> ContextBudgetStatus {
        let model_ctx = self.detected_context_size.unwrap_or_else(|| {
            crate::context::get_model_context_size_cached(&self.config.selected_model, |_| None)
        });
        let system_tokens = crate::context::estimate_tokens(&self.system_prompt_base);
        let conversation_tokens: usize = self
            .conversation
            .iter()
            .map(|m| crate::context::estimate_tokens(&m.content))
            .sum();
        let response_reserve = 800;
        let available = model_ctx
            .saturating_sub(system_tokens)
            .saturating_sub(conversation_tokens)
            .saturating_sub(response_reserve);

        #[cfg(feature = "rag")]
        let rag_available = self.rag_db.is_some();
        #[cfg(not(feature = "rag"))]
        let rag_available = false;
        let memory_available = self.memory_manager.is_some();
        #[cfg(feature = "advanced-memory")]
        let procedural_available = self.procedural_store.is_some();
        #[cfg(not(feature = "advanced-memory"))]
        let procedural_available = false;

        let mut recommendations = Vec::new();

        // Budget recommendations
        let utilization_pct = if model_ctx > 0 {
            ((model_ctx - available) as f64 / model_ctx as f64 * 100.0) as usize
        } else {
            0
        };

        if available < 500 {
            recommendations.push(
                "Very low budget — consider using FreshContext mode to free conversation tokens."
                    .to_string(),
            );
        } else if available > model_ctx / 2 && !rag_available {
            recommendations.push(
                "Over 50% of context is unused — enable RAG to fill with relevant knowledge."
                    .to_string(),
            );
        }

        if !rag_available {
            recommendations.push(
                "RAG not enabled — index documents with add_knowledge_source() for better answers."
                    .to_string(),
            );
        }
        if !memory_available {
            recommendations.push(
                "Memory not enabled — enable_memory() to remember user preferences across turns."
                    .to_string(),
            );
        }
        if !procedural_available {
            recommendations.push(
                "Procedural memory not enabled — workflows could improve task-specific responses."
                    .to_string(),
            );
        }

        if utilization_pct > 85 {
            recommendations.push(format!(
                "Context {}% full — consider upgrading to a model with a larger context window.",
                utilization_pct
            ));
        }

        ContextBudgetStatus {
            model_context_window: model_ctx,
            available_budget: available,
            rag_available,
            memory_available,
            procedural_available,
            recommendations,
        }
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
    pub(crate) fn maybe_compact_conversation(&mut self) {
        if !self.auto_compaction {
            return;
        }
        let compactor = ConversationCompactor::new(self.compaction_config.clone());
        if compactor.needs_compaction(self.conversation.len()) {
            let _ = self.compact_conversation();
        }
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
}
