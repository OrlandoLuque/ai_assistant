use super::*;

impl AiAssistant {
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
    pub fn init_rag(&mut self, db_path: &Path) -> AiResult<()> {
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
    pub(crate) fn ensure_rag_initialized(&mut self) -> bool {
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
    pub fn unregister_knowledge_document(&mut self, source: &str) -> AiResult<()> {
        if self.rag_config.append_only_mode {
            return Err(AiError::Other(format!(
                "Cannot unregister knowledge document '{}': append-only mode is enabled. \
                 Only adding new documents is allowed.",
                source
            )));
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
    pub fn delete_knowledge_document(&mut self, source: &str) -> AiResult<()> {
        if self.rag_config.append_only_mode {
            return Err(AiError::Other(format!(
                "Cannot delete knowledge document '{}': append-only mode is enabled. \
                 Only adding new documents is allowed.",
                source
            )));
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
    pub fn ensure_user(&mut self) -> AiResult<String> {
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
    pub fn index_knowledge_document(&mut self, source: &str, content: &str) -> AiResult<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.index_document(source, content).map_err(AiError::from);
            }
        }
        Ok(0)
    }

    #[cfg(feature = "rag")]
    /// Clear all knowledge from the database
    pub fn clear_knowledge(&mut self) -> AiResult<()> {
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
    pub fn get_knowledge_stats(&mut self) -> AiResult<(usize, usize)> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.get_knowledge_stats().map_err(AiError::from);
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
        crate::diag_debug!(
            "[rag-context] build_rag_context: query_len={} chars",
            query.len()
        );
        crate::safe_diag_trace!("[rag-context] query={:.300}", query);

        // Ensure RAG is initialized (lazy initialization if path was set)
        if !self.ensure_rag_initialized() {
            crate::diag_debug!("[rag-context] RAG not available, returning empty contexts");
            // RAG not available, return empty contexts
            return (String::new(), String::new());
        }

        // Process pending documents first
        if self.has_pending_documents() {
            let results = self.process_pending_documents();
            for (source, chunks) in results {
                if chunks > 0 {
                    log::info!("[AI RAG] Indexed '{}': {} chunks", source, chunks);
                } else {
                    log::debug!("[AI RAG] '{}' up-to-date (skipped)", source);
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
        crate::diag_debug!(
            "[rag-context] effective_max_knowledge_tokens={}, dynamic={}",
            effective_max_knowledge_tokens,
            self.rag_config.dynamic_context_enabled
        );

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

                crate::diag_debug!(
                    "[rag-context] retrieved {} knowledge chunks, top_k={}",
                    chunks.len(),
                    self.rag_config.top_k_chunks
                );
                #[cfg(feature = "diagnostic-logging")]
                for (i, chunk) in chunks.iter().enumerate() {
                    crate::diag_trace!(
                        "[rag-context] chunk[{}]: source={}, section={}, tokens={}",
                        i,
                        chunk.source,
                        chunk.section,
                        chunk.token_count
                    );
                }

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

            // Knowledge Graph context — extracted to build_graph_context_string()
            // Graph is now added as a separate ContextItem via build_allocated_context()
            // to prevent double-counting and allow independent scoring.

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

        // Context overflow truncation: if knowledge exceeds budget, trim to fit
        let knowledge_tokens = crate::estimate_tokens(&knowledge_context);
        if knowledge_tokens > effective_max_knowledge_tokens && effective_max_knowledge_tokens > 0 {
            let ratio = effective_max_knowledge_tokens as f64 / knowledge_tokens as f64;
            let target_chars = (knowledge_context.len() as f64 * ratio * 0.95) as usize;
            if target_chars < knowledge_context.len() {
                // Truncate at a clean line boundary
                let truncated = &knowledge_context[..target_chars];
                let last_newline = truncated.rfind('\n').unwrap_or(target_chars);
                knowledge_context.truncate(last_newline);
                knowledge_context.push_str("\n[... truncated to fit context window ...]\n");
                crate::diag_debug!(
                    "[rag-context] overflow truncation: {} -> {} tokens",
                    knowledge_tokens,
                    crate::estimate_tokens(&knowledge_context)
                );
            }
        }

        crate::diag_debug!(
            "[rag-context] result: knowledge={} chars, conversation={} chars",
            knowledge_context.len(),
            conversation_context.len()
        );
        crate::safe_diag_trace!(
            "[rag-context] knowledge_context_preview={:.500}",
            knowledge_context
        );

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
                    log::info!("[AI RAG] Indexed '{}': {} chunks", source, chunks);
                } else {
                    log::debug!("[AI RAG] '{}' up-to-date (skipped)", source);
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
    pub fn store_message_in_rag(&mut self, msg: &ChatMessage, in_context: bool) -> AiResult<()> {
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
    pub fn archive_messages_to_rag(&mut self, count: usize) -> AiResult<()> {
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
    pub fn get_conversation_rag_stats(&mut self) -> AiResult<(usize, usize, usize)> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                let session_id = self
                    .current_session
                    .as_ref()
                    .map(|s| s.id.as_str())
                    .unwrap_or("default");
                return db
                    .get_conversation_stats(&self.user_id, session_id)
                    .map_err(AiError::from);
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
    pub fn set_knowledge_notes(&mut self, source: &str, notes: &str) -> AiResult<()> {
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
    pub fn set_rag_global_notes(&mut self, notes: &str) -> AiResult<()> {
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
    pub fn set_rag_session_notes(&mut self, notes: &str) -> AiResult<()> {
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
    pub fn export_knowledge_to_file(&mut self, path: &std::path::Path) -> AiResult<()> {
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
    ) -> AiResult<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db
                    .import_knowledge_from_file(path, replace)
                    .map_err(AiError::from);
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
    ) -> AiResult<usize> {
        if self.ensure_rag_initialized() {
            if let Some(ref db) = self.rag_db {
                return db.import_knowledge(data, replace).map_err(AiError::from);
            }
        }
        Ok(0)
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
            "The", "This", "That", "These", "Those", "It", "Its", "They", "Their", "He", "She",
            "His", "Her", "We", "Our", "You", "Your", "My", "I", "A", "An", "And", "Or", "But",
            "Not", "No", "If", "When", "Where", "How", "What", "Who", "Which", "Why", "Is", "Are",
            "Was", "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
            "Will", "Would", "Could", "Should", "May", "Might", "Can", "Shall", "For", "From",
            "With", "About", "Into", "Through", "During", "Before", "After", "Above", "Below",
            "To", "Of", "In", "On", "At", "By", "As", "So", "Then", "Than", "Also", "Just", "Only",
            "Each", "Every", "All", "Any", "Both", "Few", "More", "Most", "Other", "Some", "Such",
            "Very", "Much", "Many", "Here", "There", "Now", "Still", "Already", "El", "La", "Los",
            "Las", "Un", "Una", "De", "En", "Por", "Para", "Con", "Sin", "Sobre", "Entre", "Es",
            "Son", "Fue", "Era",
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
            crate::error::AiError::Other(format!("Failed to decrypt kpkg '{}': {}", kpkg_path, e))
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
}
