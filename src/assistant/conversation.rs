use super::*;

impl AiAssistant {
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

    /// Get the current context composition mode.
    pub fn context_mode(&self) -> ContextMode {
        self.context_mode
    }

    /// Set the context composition mode.
    ///
    /// Switching to `FreshContext` does NOT clear existing conversation history.
    /// The history is kept for display and RAG archival, but will not be sent
    /// in the context window for subsequent messages.
    pub fn set_context_mode(&mut self, mode: ContextMode) {
        self.context_mode = mode;
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
            log::info!(
                "Session saved: session_id={}, messages={}",
                session.id,
                session.messages.len()
            );
        } else if !self.conversation.is_empty() {
            let mut session = ChatSession::new("New Chat");
            session.messages = self.conversation.clone();
            session.preferences = self.preferences.clone();
            session.auto_name();

            self.session_store.current_session_id = Some(session.id.clone());
            self.session_store.save_session(session.clone());
            log::info!(
                "Session saved (new): session_id={}, messages={}",
                session.id,
                session.messages.len()
            );
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
    pub fn save_sessions_to_file(&self, path: &Path) -> AiResult<()> {
        let mut store = self.session_store.clone();

        // Update current session in store
        if let Some(ref current) = self.current_session {
            let mut updated = current.clone();
            updated.messages = self.conversation.clone();
            updated.preferences = self.preferences.clone();
            updated.touch();
            store.save_session(updated);
        }

        store.save_to_file(path).map_err(AiError::from)
    }

    /// Load sessions from file
    pub fn load_sessions_from_file(&mut self, path: &Path) -> AiResult<()> {
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
}
