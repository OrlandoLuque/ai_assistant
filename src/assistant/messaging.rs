use super::*;

impl AiAssistant {
    // === Message Handling ===

    /// Send a message and start generating a response
    ///
    /// # Arguments
    /// * `user_message` - The user's message
    /// * `knowledge_context` - Optional knowledge/context to include in system prompt
    pub fn send_message(&mut self, user_message: String, knowledge_context: &str) {
        crate::diag_debug!(
            "[assistant] send_message: mode={:?}, conversation_len={}, knowledge_context_len={} chars",
            self.context_mode, self.conversation.len(), knowledge_context.len()
        );
        self.conversation.push(ChatMessage::user(&user_message));
        self.turn_counter += 1;
        let conversation = match self.context_mode {
            ContextMode::Conversation => {
                self.maybe_compact_conversation();
                self.conversation.clone()
            }
            ContextMode::FreshContext => {
                vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()]
            }
        };
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        // Build context using the adaptive budget allocator
        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let config = self.config.clone();
        let system_prompt =
            build_system_prompt(&self.system_prompt_base, &self.preferences, knowledge_ref);

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

    /// Send a message with automatic RAG context lookup.
    ///
    /// Calls `build_rag_context()` internally before sending, so the caller
    /// doesn't need to manage RAG context manually. If RAG is not configured,
    /// behaves like `send_message_simple()`.
    #[cfg(feature = "rag")]
    pub fn send_message_with_rag(&mut self, user_message: String) {
        let (knowledge_ctx, _conversation_ctx) = self.build_rag_context(&user_message);
        self.send_message(user_message, &knowledge_ctx);
    }

    /// Synchronous response generation with automatic RAG context.
    ///
    /// Like `generate_sync()` but automatically builds RAG context from the query.
    #[cfg(feature = "rag")]
    pub fn generate_sync_with_rag(&mut self, user_message: String) -> AiResult<String> {
        let (knowledge_ctx, _conversation_ctx) = self.build_rag_context(&user_message);
        self.generate_sync(user_message, &knowledge_ctx)
    }

    /// Send a message with image attachments. Streaming-style entry point —
    /// emits a single `AiResponse::Complete` (vision dispatch is currently
    /// non-streaming). Fallback chain skips providers that do not support
    /// vision at the transport level.
    #[cfg(feature = "vision")]
    pub fn send_message_with_images(
        &mut self,
        user_message: String,
        images: Vec<crate::vision::ImageInput>,
        knowledge_context: &str,
    ) {
        self.conversation
            .push(ChatMessage::user(&user_message).with_images(images));
        self.turn_counter += 1;
        let conversation = match self.context_mode {
            ContextMode::Conversation => {
                self.maybe_compact_conversation();
                self.conversation.clone()
            }
            ContextMode::FreshContext => {
                vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()]
            }
        };
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let config = self.config.clone();
        let system_prompt =
            build_system_prompt(&self.system_prompt_base, &self.preferences, knowledge_ref);

        let fallback_providers = if self.fallback_enabled {
            self.fallback_providers.clone()
        } else {
            Vec::new()
        };
        let last_provider = self.fallback_last_provider.clone();

        thread::spawn(move || {
            try_generate_vision_with_fallback(
                &config,
                &conversation,
                &system_prompt,
                &tx,
                &fallback_providers,
                &last_provider,
            );
        });
    }

    /// Send a message with images and no extra knowledge context.
    #[cfg(feature = "vision")]
    pub fn send_message_simple_with_images(
        &mut self,
        user_message: String,
        images: Vec<crate::vision::ImageInput>,
    ) {
        self.send_message_with_images(user_message, images, "");
    }

    /// Send a message with images using the internal knowledge context.
    #[cfg(feature = "vision")]
    pub fn send_message_auto_with_images(
        &mut self,
        user_message: String,
        images: Vec<crate::vision::ImageInput>,
    ) {
        let context = self.knowledge_context.clone();
        self.send_message_with_images(user_message, images, &context);
    }

    /// Synchronous vision response. Validates primary provider supports
    /// vision, falls back to first vision-capable fallback provider on
    /// failure, returns assembled response.
    #[cfg(feature = "vision")]
    pub fn generate_sync_with_images(
        &mut self,
        user_message: String,
        images: Vec<crate::vision::ImageInput>,
        knowledge_context: &str,
    ) -> AiResult<String> {
        self.conversation
            .push(ChatMessage::user(&user_message).with_images(images));

        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };
        let system_prompt =
            build_system_prompt(&self.system_prompt_base, &self.preferences, knowledge_ref);

        let fresh_conv: Vec<ChatMessage>;
        let conversation: &[ChatMessage] = match self.context_mode {
            ContextMode::Conversation => &self.conversation,
            ContextMode::FreshContext => {
                fresh_conv = vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()];
                &fresh_conv
            }
        };

        let vision_messages = crate::vision::agent_bridge::chat_messages_to_vision(conversation);

        let primary_attempt = if crate::vision::agent_bridge::vision_supported_for(&self.config) {
            crate::vision::generate_vision_response(&self.config, &vision_messages, &system_prompt)
        } else {
            Err(anyhow::anyhow!(
                "Provider {} does not support vision",
                self.config.provider.display_name()
            ))
        };

        let response = match primary_attempt {
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
                    return Err(primary_err.into());
                }
                let mut last_err = primary_err;
                let mut found = None;
                for (provider, model) in &self.fallback_providers {
                    let mut fb_config = self.config.clone();
                    fb_config.provider = provider.clone();
                    fb_config.selected_model = model.clone();
                    if !crate::vision::agent_bridge::vision_supported_for(&fb_config) {
                        continue;
                    }
                    match crate::vision::generate_vision_response(
                        &fb_config,
                        &vision_messages,
                        &system_prompt,
                    ) {
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

    /// Send a message with images and automatic RAG context lookup.
    #[cfg(all(feature = "vision", feature = "rag"))]
    pub fn send_message_with_images_rag(
        &mut self,
        user_message: String,
        images: Vec<crate::vision::ImageInput>,
    ) {
        let (knowledge_ctx, _conversation_ctx) = self.build_rag_context(&user_message);
        self.send_message_with_images(user_message, images, &knowledge_ctx);
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
        log::debug!(
            "[AI] send_message_auto_with_notes: knowledge_context size = {} bytes",
            context.len()
        );
        if context.is_empty() {
            log::warn!("[AI] knowledge_context is EMPTY");
        } else {
            // Show first 300 chars of context
            let preview: String = context.chars().take(300).collect();
            log::debug!("[AI] Context preview: {}...", preview);

            // Check for CCU-related content specifically
            let context_lower = context.to_lowercase();
            if context_lower.contains("cross-chassis") || context_lower.contains("ccu") {
                log::debug!("[AI] CCU knowledge FOUND in context");
            } else {
                log::warn!("[AI] No CCU-related content found in context");
            }

            // Count how many knowledge sections
            let section_count = context.matches("# ").count();
            log::debug!("[AI] Knowledge sections: {}", section_count);
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
        crate::diag_debug!(
            "[assistant] send_message_with_notes: mode={:?}, conversation_len={}, knowledge={} chars, session_notes={} chars",
            self.context_mode, self.conversation.len(), knowledge_context.len(), session_notes.len()
        );
        self.conversation.push(ChatMessage::user(&user_message));
        let conversation = match self.context_mode {
            ContextMode::Conversation => {
                self.maybe_compact_conversation();
                self.conversation.clone()
            }
            ContextMode::FreshContext => {
                vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()]
            }
        };
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        // Build context using the adaptive budget allocator
        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_ref,
            session_notes,
            knowledge_notes,
        );

        let (system_prompt, config) = self.apply_adaptive_thinking(&user_message, system_prompt);
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
    ) -> AiResult<String> {
        crate::diag_debug!(
            "[assistant] generate_sync: mode={:?}, conversation_len={}, knowledge={} chars",
            self.context_mode,
            self.conversation.len(),
            knowledge_context.len()
        );
        self.conversation.push(ChatMessage::user(&user_message));

        // Build context using the adaptive budget allocator
        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let system_prompt =
            build_system_prompt(&self.system_prompt_base, &self.preferences, knowledge_ref);

        // In FreshContext mode, only send the current message
        let fresh_conv: Vec<ChatMessage>;
        let conversation: &[ChatMessage] = match self.context_mode {
            ContextMode::Conversation => &self.conversation,
            ContextMode::FreshContext => {
                fresh_conv = vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()];
                &fresh_conv
            }
        };

        // Try primary provider
        let response = match generate_response(&self.config, conversation, &system_prompt) {
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
                    return Err(primary_err.into());
                }
                // Try fallback providers
                let mut last_err = primary_err;
                let mut found = None;
                for (provider, model) in &self.fallback_providers {
                    let mut fb_config = self.config.clone();
                    fb_config.provider = provider.clone();
                    fb_config.selected_model = model.clone();
                    match generate_response(&fb_config, conversation, &system_prompt) {
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
        crate::diag_debug!(
            "[assistant] send_message_cancellable: mode={:?}, conversation_len={}, knowledge={} chars",
            self.context_mode, self.conversation.len(), knowledge_context.len()
        );
        self.conversation.push(ChatMessage::user(&user_message));
        let conversation = match self.context_mode {
            ContextMode::Conversation => {
                self.maybe_compact_conversation();
                self.conversation.clone()
            }
            ContextMode::FreshContext => {
                vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()]
            }
        };
        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let cancel_token = CancellationToken::new();
        self.cancel_token = Some(cancel_token.clone());

        // Build context using the adaptive budget allocator
        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let config = self.config.clone();
        let system_prompt =
            build_system_prompt(&self.system_prompt_base, &self.preferences, knowledge_ref);

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
        crate::diag_debug!(
            "[assistant] send_message_cancellable_with_notes: mode={:?}, conversation_len={}, knowledge={} chars",
            self.context_mode, self.conversation.len(), knowledge_context.len()
        );
        // Emit message sent event
        self.event_bus.emit(crate::events::AiEvent::MessageSent {
            content_length: user_message.len(),
            has_knowledge: !knowledge_context.is_empty(),
        });

        let msg = ChatMessage::user(&user_message);
        self.conversation.push(msg.clone());

        // Auto-store user message in RAG if enabled
        #[cfg(feature = "rag")]
        if self.rag_config.auto_store_messages {
            let _ = self.store_message_in_rag(&msg, true);
        }

        let conversation = match self.context_mode {
            ContextMode::Conversation => {
                self.maybe_compact_conversation();
                self.conversation.clone()
            }
            ContextMode::FreshContext => {
                vec![self
                    .conversation
                    .last()
                    .expect("message was just pushed")
                    .clone()]
            }
        };

        self.is_generating = true;
        self.current_response.clear();

        let (tx, rx) = mpsc::channel();
        self.rx_response = Some(rx);

        let cancel_token = CancellationToken::new();
        self.cancel_token = Some(cancel_token.clone());

        // Build context using the adaptive budget allocator
        let intent = self.classify_intent_for_budget(&user_message);
        let allocated_context =
            self.build_allocated_context(&user_message, knowledge_context, intent.as_ref());
        let effective_knowledge: String;
        let knowledge_ref = if allocated_context.is_empty() {
            knowledge_context
        } else {
            effective_knowledge = allocated_context;
            &effective_knowledge
        };

        let system_prompt = build_system_prompt_with_notes(
            &self.system_prompt_base,
            &self.preferences,
            knowledge_ref,
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
                            crate::diag_debug!(
                                "[assistant] poll_response: complete, response={} chars, conversation now {} messages",
                                self.current_response.len(), self.conversation.len() + 1
                            );
                            crate::safe_diag_trace!(
                                "[assistant] poll_response: response_preview={:.500}",
                                self.current_response
                            );
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

                            // Process messages into memory if enabled
                            if let Some(ref mut mm) = self.memory_manager {
                                if self.conversation.len() >= 2 {
                                    let user_msg =
                                        self.conversation[self.conversation.len() - 2].clone();
                                    mm.process_message(&user_msg);
                                }
                                mm.process_message(&msg);
                            }

                            // Track outcomes for active procedures
                            #[cfg(feature = "advanced-memory")]
                            if !self.active_procedure_ids.is_empty() {
                                let success = self.current_response.len() > 20;
                                if let Some(ref mut store) = self.procedural_store {
                                    for pid in &self.active_procedure_ids {
                                        let _ = store.update_outcome(pid, success);
                                    }
                                }
                                if let Some(ref mut evolver) = self.procedure_evolver {
                                    for pid in &self.active_procedure_ids {
                                        let ctx = self
                                            .conversation
                                            .last()
                                            .map(|m| {
                                                m.content.chars().take(200).collect::<String>()
                                            })
                                            .unwrap_or_default();
                                        evolver.record_feedback(
                                            crate::advanced_memory::ProcedureFeedback {
                                                procedure_id: pid.clone(),
                                                outcome: if success {
                                                    crate::advanced_memory::FeedbackOutcome::Success
                                                } else {
                                                    crate::advanced_memory::FeedbackOutcome::Failure
                                                },
                                                context: ctx,
                                                timestamp: chrono::Utc::now(),
                                            },
                                        );
                                    }
                                }
                                self.active_procedure_ids.clear();
                            }

                            // Auto-track lists in LLM response for reference resolution
                            let topic = self
                                .conversation
                                .iter()
                                .rev()
                                .find(|m| m.role == "user")
                                .map(|m| {
                                    let words: Vec<&str> =
                                        m.content.split_whitespace().take(8).collect();
                                    words.join(" ")
                                })
                                .unwrap_or_default();
                            self.reference_resolver.track_lists_in_message(
                                &self.current_response,
                                &topic,
                                self.turn_counter,
                            );

                            // Auto-record cost to dashboard if enabled
                            if let Some(ref mut dashboard) = self.cost_dashboard {
                                let model = self.config.selected_model.clone();
                                // Estimate input tokens from last user message + system prompt
                                let input_tokens = self
                                    .conversation
                                    .iter()
                                    .rev()
                                    .find(|m| m.role == "user")
                                    .map(|m| crate::context::estimate_tokens(&m.content))
                                    .unwrap_or(0)
                                    + crate::context::estimate_tokens(&self.system_prompt_base);
                                let output_tokens =
                                    crate::context::estimate_tokens(&self.current_response);
                                dashboard.record(
                                    &model,
                                    input_tokens,
                                    output_tokens,
                                    crate::cost_integration::RequestType::Chat,
                                );
                            }

                            self.event_bus
                                .emit(crate::events::AiEvent::ResponseComplete {
                                    response_length: self.current_response.len(),
                                });
                            return Some(AiResponse::Complete(self.current_response.clone()));
                        }
                        AiResponse::Cancelled(partial) => {
                            self.current_response = partial.clone();
                            // Save partial response to conversation so the user
                            // can later send "continue" and the model sees context.
                            if !partial.is_empty() {
                                let partial_msg = ChatMessage::assistant(&format!(
                                    "{}\n\n[... response interrupted]",
                                    partial
                                ));
                                self.conversation.push(partial_msg);
                            }
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
}
