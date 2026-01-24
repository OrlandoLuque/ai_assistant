//! Session and preference management

use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::Result;

use crate::messages::ChatMessage;

/// User preferences learned from conversations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserPreferences {
    /// User's owned items (e.g., ships)
    pub ships_owned: Vec<String>,
    /// Target item user is interested in
    pub target_ship: Option<String>,
    /// Preferred response style
    pub response_style: ResponseStyle,
    /// User's budget (if mentioned)
    pub budget: Option<String>,
    /// Areas of interest
    pub interests: Vec<String>,
    /// Custom notes
    pub notes: Vec<String>,
    /// Global notes that apply to all conversations
    /// These are injected into the system prompt for every message
    #[serde(default)]
    pub global_notes: String,
}

/// Response style preference
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum ResponseStyle {
    #[default]
    Normal,
    Concise,
    Detailed,
    Technical,
}

/// Represents a saved chat session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    /// Unique identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Messages in this session
    pub messages: Vec<ChatMessage>,
    /// Learned preferences from this session
    pub preferences: UserPreferences,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Session-specific context notes
    /// These are injected into the system prompt for this conversation only
    #[serde(default)]
    pub context_notes: String,
}

impl ChatSession {
    /// Create a new session with default name
    pub fn new(name: &str) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: format!("session_{}", now.timestamp_millis()),
            name: name.to_string(),
            messages: Vec::new(),
            preferences: UserPreferences::default(),
            created_at: now,
            updated_at: now,
            context_notes: String::new(),
        }
    }

    /// Auto-generate a name from the first user message
    pub fn auto_name(&mut self) {
        if let Some(first_msg) = self.messages.iter().find(|m| m.role == "user") {
            let summary = if first_msg.content.len() > 40 {
                format!("{}...", &first_msg.content[..40])
            } else {
                first_msg.content.clone()
            };
            self.name = summary;
        }
    }

    /// Update the session timestamp
    pub fn touch(&mut self) {
        self.updated_at = chrono::Utc::now();
    }
}

/// Collection of saved sessions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatSessionStore {
    /// All saved sessions
    pub sessions: Vec<ChatSession>,
    /// ID of the current/last session
    pub current_session_id: Option<String>,
}

impl ChatSessionStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a session by ID
    pub fn find_session(&self, id: &str) -> Option<&ChatSession> {
        self.sessions.iter().find(|s| s.id == id)
    }

    /// Find a session by ID (mutable)
    pub fn find_session_mut(&mut self, id: &str) -> Option<&mut ChatSession> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }

    /// Add or update a session
    pub fn save_session(&mut self, session: ChatSession) {
        if let Some(idx) = self.sessions.iter().position(|s| s.id == session.id) {
            self.sessions[idx] = session;
        } else {
            self.sessions.push(session);
        }
    }

    /// Delete a session by ID
    pub fn delete_session(&mut self, id: &str) {
        self.sessions.retain(|s| s.id != id);
        if self.current_session_id.as_deref() == Some(id) {
            self.current_session_id = None;
        }
    }

    /// Get sessions sorted by last update (newest first)
    pub fn sessions_by_date(&self) -> Vec<&ChatSession> {
        let mut sessions: Vec<_> = self.sessions.iter().collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    /// Save to a JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from a JSON file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let store: Self = serde_json::from_str(&content)?;
            Ok(store)
        } else {
            Ok(Self::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_session_new() {
        let session = ChatSession::new("Test Session");
        assert_eq!(session.name, "Test Session");
        assert!(session.messages.is_empty());
        assert!(session.id.starts_with("session_"));
    }

    #[test]
    fn test_chat_session_auto_name() {
        let mut session = ChatSession::new("Untitled");
        session.messages.push(ChatMessage::user("What ships are in Star Citizen?"));
        session.auto_name();
        assert_eq!(session.name, "What ships are in Star Citizen?");
    }

    #[test]
    fn test_chat_session_auto_name_truncates() {
        let mut session = ChatSession::new("Untitled");
        session.messages.push(ChatMessage::user(
            "This is a very long message that should be truncated to forty characters plus ellipsis"
        ));
        session.auto_name();
        assert!(session.name.ends_with("..."));
        assert!(session.name.len() <= 43); // 40 + "..."
    }

    #[test]
    fn test_session_store_save_and_find() {
        let mut store = ChatSessionStore::new();
        let session = ChatSession::new("My Session");
        let id = session.id.clone();
        store.save_session(session);

        assert!(store.find_session(&id).is_some());
        assert_eq!(store.find_session(&id).unwrap().name, "My Session");
    }

    #[test]
    fn test_session_store_delete() {
        let mut store = ChatSessionStore::new();
        let session = ChatSession::new("To Delete");
        let id = session.id.clone();
        store.save_session(session);
        store.current_session_id = Some(id.clone());

        store.delete_session(&id);
        assert!(store.find_session(&id).is_none());
        assert!(store.current_session_id.is_none());
    }

    #[test]
    fn test_user_preferences_default() {
        let prefs = UserPreferences::default();
        assert!(prefs.ships_owned.is_empty());
        assert!(prefs.target_ship.is_none());
        assert_eq!(prefs.response_style, ResponseStyle::Normal);
    }

    #[test]
    fn test_session_store_update_existing() {
        let mut store = ChatSessionStore::new();
        let mut session = ChatSession::new("Original");
        let id = session.id.clone();
        store.save_session(session.clone());

        session.name = "Updated".to_string();
        store.save_session(session);

        assert_eq!(store.sessions.len(), 1);
        assert_eq!(store.find_session(&id).unwrap().name, "Updated");
    }
}
