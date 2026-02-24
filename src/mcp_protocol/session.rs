//! MCP Session Resources (v4 - item 8.2).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Session resource info for MCP session:// URIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResourceInfo {
    pub session_id: String,
    pub name: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
    pub closed_cleanly: bool,
}

/// Summary of a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub summary: String,
    pub message_count: usize,
    pub key_topics: Vec<String>,
    pub decisions: Vec<String>,
    pub open_questions: Vec<String>,
}

/// Session highlights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHighlights {
    pub session_id: String,
    pub highlights: Vec<String>,
    pub conclusions: Vec<String>,
    pub action_items: Vec<String>,
}

/// Session context (entities/relations extracted)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub session_id: String,
    pub entities: Vec<String>,
    pub relations: Vec<(String, String, String)>, // (source, relation, target)
    pub key_facts: Vec<String>,
}

/// Session beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeliefs {
    pub session_id: String,
    pub beliefs: Vec<SessionBelief>,
}

/// A single belief statement extracted from session messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBelief {
    pub statement: String,
    pub belief_type: String,
    pub confidence: f32,
}

/// Session repair result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRepairResult {
    pub session_id: String,
    pub success: bool,
    pub messages_recovered: usize,
    pub messages_lost: usize,
    pub repair_notes: Vec<String>,
}

/// Manager for session MCP resources
pub struct SessionMcpManager {
    sessions: HashMap<String, SessionResourceInfo>,
}

impl SessionMcpManager {
    /// Create a new empty session manager.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Register a session resource.
    pub fn register_session(&mut self, info: SessionResourceInfo) {
        self.sessions.insert(info.session_id.clone(), info);
    }

    /// Unregister a session by ID. Returns the removed info if it existed.
    pub fn unregister(&mut self, session_id: &str) -> Option<SessionResourceInfo> {
        self.sessions.remove(session_id)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<&SessionResourceInfo> {
        self.sessions.get(session_id)
    }

    /// List all registered sessions.
    pub fn list_sessions(&self) -> Vec<&SessionResourceInfo> {
        self.sessions.values().collect()
    }

    /// Serialize all sessions to a JSON value.
    pub fn sessions_to_json(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .sessions
            .values()
            .map(|info| {
                serde_json::to_value(info).unwrap_or_else(|_| serde_json::Value::Null)
            })
            .collect();
        serde_json::Value::Array(entries)
    }

    /// Generate a summary for a session from its messages.
    ///
    /// Each message is a tuple of (role, content).
    pub fn generate_summary(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionSummary {
        if messages.is_empty() {
            return SessionSummary {
                session_id: session_id.to_string(),
                summary: String::new(),
                message_count: 0,
                key_topics: Vec::new(),
                decisions: Vec::new(),
                open_questions: Vec::new(),
            };
        }

        // Build word frequency map for topic extraction (words >= 4 chars, lowercased)
        let stop_words: &[&str] = &[
            "this", "that", "with", "from", "have", "been", "were", "they",
            "their", "about", "would", "could", "should", "there", "which",
            "will", "what", "when", "where", "some", "into", "also", "then",
            "than", "them", "these", "those", "each", "other", "more",
        ];
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for (_role, content) in messages {
            for word in content.split_whitespace() {
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if cleaned.len() >= 4 && !stop_words.contains(&cleaned.as_str()) {
                    *word_freq.entry(cleaned).or_insert(0) += 1;
                }
            }
        }

        // Top topics by frequency
        let mut freq_vec: Vec<(String, usize)> = word_freq.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let key_topics: Vec<String> = freq_vec
            .into_iter()
            .take(5)
            .map(|(word, _)| word)
            .collect();

        // Decisions: messages containing decision-related keywords
        let decision_keywords = ["decided", "agreed", "chosen", "decision", "we will"];
        let mut decisions = Vec::new();
        for (_role, content) in messages {
            let lower = content.to_lowercase();
            if decision_keywords.iter().any(|kw| lower.contains(kw)) {
                decisions.push(content.clone());
            }
        }

        // Open questions: messages containing "?"
        let mut open_questions = Vec::new();
        for (_role, content) in messages {
            if content.contains('?') {
                open_questions.push(content.clone());
            }
        }

        // Build summary text
        let summary = if key_topics.is_empty() {
            format!("Session with {} messages.", messages.len())
        } else {
            format!(
                "Session with {} messages. Key topics: {}.",
                messages.len(),
                key_topics.join(", ")
            )
        };

        SessionSummary {
            session_id: session_id.to_string(),
            summary,
            message_count: messages.len(),
            key_topics,
            decisions,
            open_questions,
        }
    }

    /// Extract highlights from session messages.
    pub fn extract_highlights(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionHighlights {
        let mut highlights = Vec::new();
        let mut conclusions = Vec::new();
        let mut action_items = Vec::new();

        let highlight_keywords = ["important", "key", "critical", "significant", "essential"];
        let conclusion_keywords = ["in conclusion", "therefore", "finally", "to summarize", "in summary"];
        let action_keywords = ["todo", "need to", "should", "action item", "must", "next step"];

        for (_role, content) in messages {
            let lower = content.to_lowercase();

            if highlight_keywords.iter().any(|kw| lower.contains(kw)) {
                highlights.push(content.clone());
            }
            if conclusion_keywords.iter().any(|kw| lower.contains(kw)) {
                conclusions.push(content.clone());
            }
            if action_keywords.iter().any(|kw| lower.contains(kw)) {
                action_items.push(content.clone());
            }
        }

        SessionHighlights {
            session_id: session_id.to_string(),
            highlights,
            conclusions,
            action_items,
        }
    }

    /// Extract context (entities, relations, key facts) from session messages.
    pub fn extract_context(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionContext {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let mut key_facts = Vec::new();

        for (_role, content) in messages {
            // Entity extraction: capitalized words (simple heuristic)
            let words: Vec<&str> = content.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                if cleaned.len() >= 2 {
                    if let Some(first_char) = cleaned.chars().next() {
                        // Skip if it's the first word in the message (sentence-initial capitalization)
                        if first_char.is_uppercase() && i > 0 && !entities.contains(&cleaned) {
                            entities.push(cleaned);
                        }
                    }
                }
            }

            // Simple subject-verb-object relation extraction
            // Look for patterns like "X is Y", "X uses Y", "X requires Y"
            let relation_verbs = ["is", "uses", "requires", "contains", "provides", "supports"];
            for verb in &relation_verbs {
                let pattern = format!(" {} ", verb);
                if let Some(pos) = content.find(&pattern) {
                    let before = content[..pos].split_whitespace().next_back();
                    let after_start = pos + pattern.len();
                    let after = content.get(after_start..).and_then(|s| s.split_whitespace().next());
                    if let (Some(subj), Some(obj)) = (before, after) {
                        let subj_clean: String = subj.chars().filter(|c| c.is_alphanumeric()).collect();
                        let obj_clean: String = obj.chars().filter(|c| c.is_alphanumeric()).collect();
                        if !subj_clean.is_empty() && !obj_clean.is_empty() {
                            relations.push((
                                subj_clean,
                                verb.to_string(),
                                obj_clean,
                            ));
                        }
                    }
                }
            }

            // Key facts: declarative statements (sentences that don't end with ?)
            let lower = content.to_lowercase();
            if !content.contains('?')
                && content.len() > 20
                && (lower.contains(" is ") || lower.contains(" are ") || lower.contains(" was "))
            {
                key_facts.push(content.clone());
            }
        }

        SessionContext {
            session_id: session_id.to_string(),
            entities,
            relations,
            key_facts,
        }
    }

    /// Extract beliefs from session messages.
    pub fn extract_beliefs(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionBeliefs {
        let belief_patterns: &[(&str, &str, f32)] = &[
            ("i think", "opinion", 0.6),
            ("i believe", "conviction", 0.8),
            ("we should", "recommendation", 0.7),
            ("it seems", "observation", 0.5),
            ("i'm sure", "conviction", 0.9),
            ("probably", "speculation", 0.4),
            ("definitely", "conviction", 0.9),
            ("maybe", "speculation", 0.3),
        ];

        let mut beliefs = Vec::new();

        for (_role, content) in messages {
            let lower = content.to_lowercase();
            for &(pattern, belief_type, confidence) in belief_patterns {
                if lower.contains(pattern) {
                    beliefs.push(SessionBelief {
                        statement: content.clone(),
                        belief_type: belief_type.to_string(),
                        confidence,
                    });
                    break; // One belief per message
                }
            }
        }

        SessionBeliefs {
            session_id: session_id.to_string(),
            beliefs,
        }
    }

    /// Attempt to repair a session from raw data.
    ///
    /// Tries JSON array parsing first, then falls back to line-by-line recovery.
    pub fn repair_session(session_id: &str, raw_data: &str) -> SessionRepairResult {
        // Attempt 1: parse as complete JSON array
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(raw_data) {
            return SessionRepairResult {
                session_id: session_id.to_string(),
                success: true,
                messages_recovered: arr.len(),
                messages_lost: 0,
                repair_notes: vec!["Parsed successfully as JSON array.".to_string()],
            };
        }

        // Attempt 2: line-by-line recovery of partial JSON objects
        let mut recovered = 0usize;
        let mut lost = 0usize;
        let mut notes = Vec::new();

        for (line_idx, line) in raw_data.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Try to parse each line as a JSON object
            if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                recovered += 1;
            } else {
                // Try to fix common JSON issues: trailing comma, missing closing brace
                let mut candidate = trimmed.to_string();
                // Remove trailing comma
                if candidate.ends_with(',') {
                    candidate.pop();
                }
                // Try adding missing closing brace
                if candidate.starts_with('{') && !candidate.ends_with('}') {
                    candidate.push('}');
                }
                if serde_json::from_str::<serde_json::Value>(&candidate).is_ok() {
                    recovered += 1;
                    notes.push(format!("Line {}: repaired (trailing comma or missing brace).", line_idx + 1));
                } else {
                    lost += 1;
                    notes.push(format!("Line {}: unrecoverable.", line_idx + 1));
                }
            }
        }

        let success = recovered > 0;
        if !success {
            notes.push("No messages could be recovered.".to_string());
        }

        SessionRepairResult {
            session_id: session_id.to_string(),
            success,
            messages_recovered: recovered,
            messages_lost: lost,
            repair_notes: notes,
        }
    }
}

impl Default for SessionMcpManager {
    fn default() -> Self {
        Self::new()
    }
}
