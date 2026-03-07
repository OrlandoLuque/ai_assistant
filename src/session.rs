//! Session and preference management
//!
//! Provides two session storage strategies:
//!
//! - **`ChatSessionStore`** — full JSON serialization (read/write entire file).
//! - **`JournalSession`** — JSONL append-only log. Each message is a single
//!   JSON line appended to the file, avoiding a full rewrite on every message.
//!   Supports compaction (rewrite with summary + recent messages).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

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

        #[cfg(feature = "analytics")]
        crate::scalability_monitor::check_scalability(
            crate::scalability_monitor::Subsystem::SessionStore,
            self.sessions.len(),
        );
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

    /// Save to file using internal storage format.
    ///
    /// With the `binary-storage` feature: bincode + gzip (compact, fast).
    /// Without: JSON (human-readable, backward-compatible).
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        crate::internal_storage::save_internal(self, path)
    }

    /// Load from file, auto-detecting format.
    ///
    /// Transparently reads both the new binary format and legacy JSON files,
    /// so existing session files continue to work after the migration.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        if path.exists() {
            crate::internal_storage::load_internal(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Save to a JSON file (explicit JSON format regardless of features).
    ///
    /// Useful for exporting human-readable session data.
    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from an explicit JSON file.
    pub fn load_from_json(path: &Path) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let store: Self = serde_json::from_str(&content)?;
            Ok(store)
        } else {
            Ok(Self::default())
        }
    }
}

// ============================================================================
// Encrypted session storage (AES-256-GCM, feature = "rag")
// ============================================================================

#[cfg(feature = "rag")]
mod encrypted {
    use super::*;
    use aes_gcm::aead::rand_core::RngCore;
    use aes_gcm::{
        aead::{Aead, KeyInit, OsRng},
        Aes256Gcm, Nonce,
    };

    /// Size of AES-256-GCM nonce in bytes (96 bits).
    const NONCE_SIZE: usize = 12;

    impl ChatSessionStore {
        /// Save sessions to an AES-256-GCM encrypted file.
        ///
        /// `key` must be exactly 32 bytes.
        /// Format: `[12-byte nonce][ciphertext+tag]`
        pub fn save_encrypted(&self, path: &Path, key: &[u8; 32]) -> Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let plaintext = serde_json::to_vec(self).context("Failed to serialize sessions")?;

            let cipher = Aes256Gcm::new_from_slice(key)
                .map_err(|e| anyhow::anyhow!("Invalid encryption key: {}", e))?;

            let mut nonce_bytes = [0u8; NONCE_SIZE];
            OsRng.fill_bytes(&mut nonce_bytes);
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = cipher
                .encrypt(nonce, plaintext.as_slice())
                .map_err(|_| anyhow::anyhow!("Encryption failed"))?;

            let mut output = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
            output.extend_from_slice(&nonce_bytes);
            output.extend(ciphertext);

            std::fs::write(path, output)?;
            Ok(())
        }

        /// Load sessions from an AES-256-GCM encrypted file.
        ///
        /// `key` must be exactly 32 bytes. Returns an empty store if the file
        /// does not exist.
        pub fn load_encrypted(path: &Path, key: &[u8; 32]) -> Result<Self> {
            if !path.exists() {
                return Ok(Self::default());
            }
            let data = std::fs::read(path)
                .with_context(|| format!("Failed to read encrypted session: {}", path.display()))?;

            if data.len() < NONCE_SIZE + 16 {
                anyhow::bail!("Encrypted session file too short");
            }

            let nonce_bytes = &data[..NONCE_SIZE];
            let ciphertext = &data[NONCE_SIZE..];

            let cipher = Aes256Gcm::new_from_slice(key)
                .map_err(|e| anyhow::anyhow!("Invalid encryption key: {}", e))?;
            let nonce = Nonce::from_slice(nonce_bytes);

            let plaintext = cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| anyhow::anyhow!("Decryption failed — wrong key or corrupted data"))?;

            let store: Self = serde_json::from_slice(&plaintext)
                .context("Failed to deserialize decrypted sessions")?;
            Ok(store)
        }
    }
}

// ============================================================================
// Journal-based session (append-only JSONL)
// ============================================================================

/// Type of journal entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JournalEntryType {
    /// A regular chat message (user, assistant, or system).
    Message,
    /// A compaction summary that replaces older entries.
    CompactionSummary,
    /// Metadata entry (e.g. session name change, preference update).
    Metadata,
}

/// A single entry in the journal log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// When this entry was written.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Kind of entry.
    pub entry_type: JournalEntryType,
    /// The serialized payload (message content, summary text, or JSON metadata).
    pub data: String,
    /// For `Message` entries: the role (user / assistant / system).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

impl JournalEntry {
    /// Create a message entry from a `ChatMessage`.
    pub fn from_message(msg: &ChatMessage) -> Self {
        Self {
            timestamp: msg.timestamp,
            entry_type: JournalEntryType::Message,
            data: msg.content.clone(),
            role: Some(msg.role.clone()),
        }
    }

    /// Create a compaction summary entry.
    pub fn compaction_summary(summary: &str) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            entry_type: JournalEntryType::CompactionSummary,
            data: summary.to_string(),
            role: None,
        }
    }

    /// Create a metadata entry.
    pub fn metadata(data: &str) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            entry_type: JournalEntryType::Metadata,
            data: data.to_string(),
            role: None,
        }
    }

    /// Convert a `Message` entry back to a `ChatMessage`.
    ///
    /// Returns `None` for non-message entry types.
    pub fn to_chat_message(&self) -> Option<ChatMessage> {
        if self.entry_type != JournalEntryType::Message {
            return None;
        }
        let role = self.role.as_deref().unwrap_or("user");
        Some(ChatMessage {
            role: role.to_string(),
            content: self.data.clone(),
            timestamp: self.timestamp,
        })
    }
}

/// Append-only JSONL session.
///
/// Each operation appends one JSON line to the file. This avoids
/// rewriting the entire session on every message, making it efficient
/// for long conversations.
///
/// # Example
///
/// ```rust,no_run
/// use ai_assistant::session::JournalSession;
/// use ai_assistant::ChatMessage;
///
/// let journal = JournalSession::new("my_session.jsonl");
/// journal.append_message(&ChatMessage::user("Hello")).unwrap();
///
/// let entries = journal.load().unwrap();
/// assert_eq!(entries.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct JournalSession {
    /// Path to the JSONL file.
    path: PathBuf,
}

impl JournalSession {
    /// Create a new journal session at the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Get the path to the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a chat message as a single JSONL line.
    pub fn append_message(&self, msg: &ChatMessage) -> Result<()> {
        let entry = JournalEntry::from_message(msg);
        self.append_entry(&entry)
    }

    /// Append a raw journal entry as a single JSONL line.
    pub fn append_entry(&self, entry: &JournalEntry) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("Failed to open journal: {}", self.path.display()))?;
        let line = serde_json::to_string(entry).context("Failed to serialize journal entry")?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    /// Load all entries from the journal file.
    ///
    /// Silently skips lines that fail to parse (graceful recovery).
    pub fn load(&self) -> Result<Vec<JournalEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Failed to open journal: {}", self.path.display()))?;
        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<JournalEntry>(trimmed) {
                entries.push(entry);
            }
            // Silently skip unparseable lines for forward compatibility
        }
        Ok(entries)
    }

    /// Load only `Message` entries as `ChatMessage`s.
    pub fn load_messages(&self) -> Result<Vec<ChatMessage>> {
        let entries = self.load()?;
        Ok(entries.iter().filter_map(|e| e.to_chat_message()).collect())
    }

    /// Count the number of lines in the journal without loading all data.
    ///
    /// This is faster than `load().len()` for large journals.
    pub fn message_count(&self) -> Result<usize> {
        if !self.path.exists() {
            return Ok(0);
        }
        let file = std::fs::File::open(&self.path)
            .with_context(|| format!("Failed to open journal: {}", self.path.display()))?;
        let reader = std::io::BufReader::new(file);
        let count = reader
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .count();
        Ok(count)
    }

    /// Compact the journal: rewrite with a summary + the most recent messages.
    ///
    /// `keep_recent` controls how many of the newest message entries to preserve.
    /// All older entries are replaced by a single `CompactionSummary` entry.
    pub fn compact(&self, summary: &str, keep_recent: usize) -> Result<usize> {
        let entries = self.load()?;
        let original_count = entries.len();

        // Separate message entries from non-message entries
        let message_entries: Vec<&JournalEntry> = entries
            .iter()
            .filter(|e| e.entry_type == JournalEntryType::Message)
            .collect();

        let non_message_entries: Vec<&JournalEntry> = entries
            .iter()
            .filter(|e| e.entry_type == JournalEntryType::Metadata)
            .collect();

        // Determine which messages to keep
        let keep_start = message_entries.len().saturating_sub(keep_recent);
        let kept_messages = &message_entries[keep_start..];

        // Build compacted journal: metadata + summary + recent messages
        let mut compacted: Vec<JournalEntry> = Vec::new();

        // Preserve metadata entries
        for entry in &non_message_entries {
            compacted.push((*entry).clone());
        }

        // Add compaction summary
        if !summary.is_empty() {
            compacted.push(JournalEntry::compaction_summary(summary));
        }

        // Add kept messages
        for entry in kept_messages {
            compacted.push((*entry).clone());
        }

        // Atomic-ish rewrite: write to temp, then rename
        let tmp_path = self.path.with_extension("jsonl.tmp");
        {
            let mut file =
                std::fs::File::create(&tmp_path).context("Failed to create temp journal file")?;
            for entry in &compacted {
                let line = serde_json::to_string(entry)?;
                writeln!(file, "{}", line)?;
            }
            file.flush()?;
        }
        std::fs::rename(&tmp_path, &self.path).context("Failed to rename temp journal file")?;

        Ok(original_count - compacted.len())
    }
}

impl ChatSession {
    /// Export this session to a JSONL journal file.
    ///
    /// Writes all messages as journal entries. This can be used to migrate
    /// from the full-JSON format to the append-only JSONL format.
    pub fn to_journal(&self, path: impl Into<PathBuf>) -> Result<JournalSession> {
        let journal = JournalSession::new(path);

        // Write session metadata
        let meta = serde_json::json!({
            "session_id": self.id,
            "session_name": self.name,
            "created_at": self.created_at.to_rfc3339(),
        });
        journal.append_entry(&JournalEntry::metadata(&meta.to_string()))?;

        // Write all messages
        for msg in &self.messages {
            journal.append_message(msg)?;
        }

        Ok(journal)
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
        session
            .messages
            .push(ChatMessage::user("What ships are in Star Citizen?"));
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

    // ========================================================================
    // JournalSession tests
    // ========================================================================

    fn temp_journal_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("ai_assistant_tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{}.jsonl", name))
    }

    #[test]
    fn test_journal_append_and_load_roundtrip() {
        let path = temp_journal_path("journal_roundtrip");
        let _ = std::fs::remove_file(&path);

        let journal = JournalSession::new(&path);
        journal.append_message(&ChatMessage::user("Hello")).unwrap();
        journal
            .append_message(&ChatMessage::assistant("Hi there"))
            .unwrap();
        journal
            .append_message(&ChatMessage::user("How are you?"))
            .unwrap();

        let entries = journal.load().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].data, "Hello");
        assert_eq!(entries[0].role.as_deref(), Some("user"));
        assert_eq!(entries[1].data, "Hi there");
        assert_eq!(entries[1].role.as_deref(), Some("assistant"));
        assert_eq!(entries[2].data, "How are you?");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_journal_load_messages() {
        let path = temp_journal_path("journal_load_msgs");
        let _ = std::fs::remove_file(&path);

        let journal = JournalSession::new(&path);
        journal.append_message(&ChatMessage::user("Q")).unwrap();
        journal
            .append_entry(&JournalEntry::metadata("some meta"))
            .unwrap();
        journal
            .append_message(&ChatMessage::assistant("A"))
            .unwrap();

        let messages = journal.load_messages().unwrap();
        assert_eq!(messages.len(), 2); // metadata entry is filtered out
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_journal_message_count() {
        let path = temp_journal_path("journal_count");
        let _ = std::fs::remove_file(&path);

        let journal = JournalSession::new(&path);
        assert_eq!(journal.message_count().unwrap(), 0);

        for i in 0..10 {
            journal
                .append_message(&ChatMessage::user(&format!("msg {}", i)))
                .unwrap();
        }
        assert_eq!(journal.message_count().unwrap(), 10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_journal_compact_reduces_entries() {
        let path = temp_journal_path("journal_compact");
        let _ = std::fs::remove_file(&path);

        let journal = JournalSession::new(&path);

        // Write 20 messages
        for i in 0..20 {
            journal
                .append_message(&ChatMessage::user(&format!("message {}", i)))
                .unwrap();
        }
        assert_eq!(journal.message_count().unwrap(), 20);

        // Compact, keeping last 5
        let removed = journal.compact("Summary of first 15 messages", 5).unwrap();
        assert!(removed > 0);

        let entries = journal.load().unwrap();
        // Should have: 1 summary + 5 messages = 6
        assert_eq!(entries.len(), 6);

        // First entry should be the compaction summary
        assert_eq!(entries[0].entry_type, JournalEntryType::CompactionSummary);
        assert_eq!(entries[0].data, "Summary of first 15 messages");

        // Last 5 messages preserved
        assert_eq!(entries[1].data, "message 15");
        assert_eq!(entries[5].data, "message 19");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_journal_load_nonexistent_file() {
        let journal = JournalSession::new("/tmp/does_not_exist_12345.jsonl");
        let entries = journal.load().unwrap();
        assert!(entries.is_empty());
        assert_eq!(journal.message_count().unwrap(), 0);
    }

    #[test]
    fn test_chat_session_to_journal() {
        let path = temp_journal_path("journal_migration");
        let _ = std::fs::remove_file(&path);

        let mut session = ChatSession::new("Test Migration");
        session.messages.push(ChatMessage::user("Hello"));
        session.messages.push(ChatMessage::assistant("Hi"));
        session.messages.push(ChatMessage::user("Bye"));

        let journal = session.to_journal(&path).unwrap();
        let entries = journal.load().unwrap();

        // 1 metadata + 3 messages = 4 entries
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].entry_type, JournalEntryType::Metadata);
        assert!(entries[0].data.contains(&session.id));
        assert_eq!(entries[1].data, "Hello");
        assert_eq!(entries[2].data, "Hi");
        assert_eq!(entries[3].data, "Bye");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_journal_graceful_on_bad_lines() {
        let path = temp_journal_path("journal_bad_lines");
        let _ = std::fs::remove_file(&path);

        // Write some valid entries
        let journal = JournalSession::new(&path);
        journal.append_message(&ChatMessage::user("Valid")).unwrap();

        // Manually append a bad line
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "this is not valid JSON").unwrap();

        // Append another valid entry
        journal
            .append_message(&ChatMessage::user("Also valid"))
            .unwrap();

        // Load should skip the bad line
        let entries = journal.load().unwrap();
        assert_eq!(entries.len(), 2); // bad line skipped
        assert_eq!(entries[0].data, "Valid");
        assert_eq!(entries[1].data, "Also valid");

        let _ = std::fs::remove_file(&path);
    }

    // ========================================================================
    // Encrypted sessions (feature = "rag")
    // ========================================================================

    #[cfg(feature = "rag")]
    #[test]
    fn test_encrypted_session_roundtrip() {
        let dir = std::env::temp_dir().join("ai_assistant_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("encrypted_session.bin");
        let _ = std::fs::remove_file(&path);

        let key: [u8; 32] = [42u8; 32];

        let mut store = ChatSessionStore::new();
        let mut session = ChatSession::new("Secret Session");
        session
            .messages
            .push(ChatMessage::user("Top secret question"));
        session
            .messages
            .push(ChatMessage::assistant("Classified answer"));
        store.save_session(session);

        // Save encrypted
        store.save_encrypted(&path, &key).unwrap();
        assert!(path.exists());

        // File should NOT be valid JSON (it's encrypted)
        let raw = std::fs::read_to_string(&path);
        assert!(
            raw.is_err()
                || serde_json::from_str::<ChatSessionStore>(&raw.unwrap_or_default()).is_err()
        );

        // Load with correct key
        let loaded = ChatSessionStore::load_encrypted(&path, &key).unwrap();
        assert_eq!(loaded.sessions.len(), 1);
        assert_eq!(loaded.sessions[0].name, "Secret Session");
        assert_eq!(loaded.sessions[0].messages.len(), 2);
        assert_eq!(
            loaded.sessions[0].messages[0].content,
            "Top secret question"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_encrypted_session_wrong_key() {
        let dir = std::env::temp_dir().join("ai_assistant_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("encrypted_session_wrongkey.bin");
        let _ = std::fs::remove_file(&path);

        let key: [u8; 32] = [1u8; 32];
        let wrong_key: [u8; 32] = [2u8; 32];

        let mut store = ChatSessionStore::new();
        store.save_session(ChatSession::new("Test"));
        store.save_encrypted(&path, &key).unwrap();

        // Load with wrong key should fail
        let result = ChatSessionStore::load_encrypted(&path, &wrong_key);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Decryption failed") || err_msg.contains("wrong key"));

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "rag")]
    #[test]
    fn test_encrypted_session_nonexistent_returns_default() {
        let key: [u8; 32] = [0u8; 32];
        let path = Path::new("/tmp/nonexistent_encrypted_session_12345.bin");
        let store = ChatSessionStore::load_encrypted(path, &key).unwrap();
        assert!(store.sessions.is_empty());
    }
}
