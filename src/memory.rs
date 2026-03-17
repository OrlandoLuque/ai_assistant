//! Conversation memory system for long-term context retention
//!
//! This module provides a memory system that maintains long-term context across
//! conversations through periodic summarization and semantic recall.

use crate::ChatMessage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// A memory entry in the long-term memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// The memory content (summary or fact)
    pub content: String,
    /// Memory type
    pub memory_type: MemoryType,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Number of times this memory was recalled
    pub recall_count: u32,
    /// When the memory was created
    pub created_at: DateTime<Utc>,
    /// When the memory was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Source session ID
    pub source_session: Option<String>,
    /// Related memory IDs
    pub related: Vec<String>,
    /// Embedding vector for semantic search (optional)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(content: &str, memory_type: MemoryType) -> Self {
        let now = Utc::now();
        Self {
            id: generate_memory_id(),
            content: content.to_string(),
            memory_type,
            importance: 0.5,
            recall_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec![],
            source_session: None,
            related: vec![],
            embedding: None,
        }
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set source session
    pub fn with_session(mut self, session_id: &str) -> Self {
        self.source_session = Some(session_id.to_string());
        self
    }

    /// Calculate decay factor based on time since last access
    pub fn decay_factor(&self, half_life_days: f64) -> f32 {
        let days_since_access = (Utc::now() - self.last_accessed).num_seconds() as f64 / 86400.0;
        (0.5f64.powf(days_since_access / half_life_days)) as f32
    }

    /// Calculate effective importance (with decay and recall boost)
    pub fn effective_importance(&self, half_life_days: f64) -> f32 {
        let decay = self.decay_factor(half_life_days);
        let recall_boost = 1.0 + (self.recall_count as f32 * 0.1).min(0.5);
        (self.importance * decay * recall_boost).min(1.0)
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Utc::now();
        self.recall_count += 1;
    }
}

/// Type of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemoryType {
    /// Summary of a conversation segment
    Summary,
    /// A specific fact learned
    Fact,
    /// User preference
    Preference,
    /// User goal or objective
    Goal,
    /// Important event or milestone
    Event,
    /// Relationship or connection
    Relationship,
    /// Skill or capability mentioned
    Skill,
    /// Custom memory type
    Custom,
}

/// Configuration for the memory system
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MemoryConfig {
    /// Maximum number of memories to retain
    pub max_memories: usize,
    /// Half-life for memory decay (in days)
    pub decay_half_life_days: f64,
    /// Minimum importance to keep a memory
    pub min_importance: f32,
    /// Number of messages before auto-summarization
    pub summarize_threshold: usize,
    /// Maximum memories to return in a recall
    pub max_recall_results: usize,
    /// Enable automatic consolidation
    pub auto_consolidate: bool,
    /// Consolidation threshold (memories with similarity above this are merged)
    pub consolidation_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memories: 1000,
            decay_half_life_days: 30.0,
            min_importance: 0.1,
            summarize_threshold: 20,
            max_recall_results: 10,
            auto_consolidate: true,
            consolidation_threshold: 0.85,
        }
    }
}

/// Long-term memory store
#[derive(Debug)]
pub struct MemoryStore {
    /// All memories
    memories: HashMap<String, MemoryEntry>,
    /// Configuration
    config: MemoryConfig,
    /// Messages pending summarization
    pending_messages: Vec<ChatMessage>,
    /// Last cleanup time
    last_cleanup: Instant,
}

impl MemoryStore {
    /// Create a new memory store
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            memories: HashMap::new(),
            config,
            pending_messages: Vec::new(),
            last_cleanup: Instant::now(),
        }
    }

    /// Add a memory
    pub fn add(&mut self, memory: MemoryEntry) -> String {
        let id = memory.id.clone();
        self.memories.insert(id.clone(), memory);

        // Check if we need cleanup
        if self.memories.len() > self.config.max_memories {
            self.cleanup();
        }

        id
    }

    /// Get a memory by ID
    pub fn get(&self, id: &str) -> Option<&MemoryEntry> {
        self.memories.get(id)
    }

    /// Get a memory mutably and mark as accessed
    pub fn recall(&mut self, id: &str) -> Option<&MemoryEntry> {
        if let Some(memory) = self.memories.get_mut(id) {
            memory.mark_accessed();
            Some(memory)
        } else {
            None
        }
    }

    /// Remove a memory
    pub fn remove(&mut self, id: &str) -> Option<MemoryEntry> {
        self.memories.remove(id)
    }

    /// Search memories by text (simple keyword matching)
    pub fn search(&mut self, query: &str) -> Vec<&MemoryEntry> {
        crate::diag_debug!("[memory] search: query_len={}, total_memories={}", query.len(), self.memories.len());
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<(f32, &str)> = self
            .memories
            .iter()
            .filter_map(|(id, memory)| {
                let content_lower = memory.content.to_lowercase();
                let mut score = 0.0f32;

                for word in &query_words {
                    if content_lower.contains(word) {
                        score += 1.0;
                    }
                }

                if score > 0.0 {
                    let effective = memory.effective_importance(self.config.decay_half_life_days);
                    Some((score * effective, id.as_str()))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.max_recall_results);

        // Mark accessed and collect results
        let ids: Vec<String> = results.iter().map(|(_, id)| id.to_string()).collect();
        for id in &ids {
            if let Some(memory) = self.memories.get_mut(id) {
                memory.mark_accessed();
            }
        }

        crate::diag_debug!("[memory] search: found {} matching memories", ids.len());
        ids.iter().filter_map(|id| self.memories.get(id)).collect()
    }

    /// Get memories by type
    pub fn get_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryEntry> {
        self.memories
            .values()
            .filter(|m| m.memory_type == memory_type)
            .collect()
    }

    /// Get memories by tag
    pub fn get_by_tag(&self, tag: &str) -> Vec<&MemoryEntry> {
        self.memories
            .values()
            .filter(|m| m.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get most important memories
    pub fn get_top_memories(&self, count: usize) -> Vec<&MemoryEntry> {
        let mut memories: Vec<_> = self.memories.values().collect();
        memories.sort_by(|a, b| {
            let a_imp = a.effective_importance(self.config.decay_half_life_days);
            let b_imp = b.effective_importance(self.config.decay_half_life_days);
            b_imp
                .partial_cmp(&a_imp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        memories.truncate(count);
        memories
    }

    /// Add a message for potential summarization
    pub fn add_message(&mut self, message: ChatMessage) {
        self.pending_messages.push(message);
    }

    /// Check if summarization is needed
    pub fn needs_summarization(&self) -> bool {
        self.pending_messages.len() >= self.config.summarize_threshold
    }

    /// Get pending messages for summarization
    pub fn get_pending_messages(&self) -> &[ChatMessage] {
        &self.pending_messages
    }

    /// Clear pending messages after summarization
    pub fn clear_pending(&mut self) {
        self.pending_messages.clear();
    }

    /// Create a summary memory from pending messages
    pub fn create_summary(&mut self, summary_text: &str, importance: f32) -> String {
        let memory =
            MemoryEntry::new(summary_text, MemoryType::Summary).with_importance(importance);
        let id = self.add(memory);
        self.clear_pending();
        id
    }

    /// Cleanup old/unimportant memories
    pub fn cleanup(&mut self) {
        let min_importance = self.config.min_importance;
        let half_life = self.config.decay_half_life_days;

        // Remove memories below importance threshold
        self.memories
            .retain(|_, m| m.effective_importance(half_life) >= min_importance);

        // If still over limit, remove least important
        if self.memories.len() > self.config.max_memories {
            let mut sorted: Vec<_> = self
                .memories
                .iter()
                .map(|(id, m)| (id.clone(), m.effective_importance(half_life)))
                .collect();

            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_remove = self.memories.len() - self.config.max_memories;
            for (id, _) in sorted.into_iter().take(to_remove) {
                self.memories.remove(&id);
            }
        }

        self.last_cleanup = Instant::now();
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let mut type_counts = HashMap::new();
        let mut total_importance = 0.0f32;

        for memory in self.memories.values() {
            *type_counts.entry(memory.memory_type).or_insert(0) += 1;
            total_importance += memory.effective_importance(self.config.decay_half_life_days);
        }

        MemoryStats {
            total_memories: self.memories.len(),
            pending_messages: self.pending_messages.len(),
            type_counts,
            average_importance: if self.memories.is_empty() {
                0.0
            } else {
                total_importance / self.memories.len() as f32
            },
        }
    }

    /// Export memories to JSON
    pub fn export(&self) -> Result<String, serde_json::Error> {
        let memories: Vec<_> = self.memories.values().collect();
        serde_json::to_string_pretty(&memories)
    }

    /// Import memories from JSON
    pub fn import(&mut self, json: &str) -> Result<usize, serde_json::Error> {
        let memories: Vec<MemoryEntry> = serde_json::from_str(json)?;
        let count = memories.len();
        for memory in memories {
            self.memories.insert(memory.id.clone(), memory);
        }
        Ok(count)
    }

    /// Export memories to internal binary format (bincode+gzip when feature enabled).
    #[cfg(feature = "binary-storage")]
    pub fn export_bytes(&self) -> Result<Vec<u8>, anyhow::Error> {
        let memories: Vec<_> = self.memories.values().collect();
        crate::internal_storage::serialize_internal(&memories)
    }

    /// Import memories from internal binary format (auto-detects binary or JSON).
    #[cfg(feature = "binary-storage")]
    pub fn import_bytes(&mut self, bytes: &[u8]) -> Result<usize, anyhow::Error> {
        let memories: Vec<MemoryEntry> = crate::internal_storage::deserialize_internal(bytes)?;
        let count = memories.len();
        for memory in memories {
            self.memories.insert(memory.id.clone(), memory);
        }
        Ok(count)
    }

    /// Build a context string from relevant memories
    pub fn build_context(&mut self, query: &str, max_tokens: usize) -> String {
        crate::diag_debug!("[memory] build_context: max_tokens={}", max_tokens);
        let relevant = self.search(query);

        let mut context = String::new();
        let mut estimated_tokens = 0;

        for memory in relevant {
            let memory_tokens = crate::estimate_tokens(&memory.content);
            if estimated_tokens + memory_tokens > max_tokens {
                break;
            }

            if !context.is_empty() {
                context.push_str("\n");
            }
            context.push_str(&format!("- {}", memory.content));
            estimated_tokens += memory_tokens;
        }

        crate::diag_debug!("[memory] build_context: result {} chars, ~{} tokens", context.len(), estimated_tokens);
        context
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of memories
    pub total_memories: usize,
    /// Messages pending summarization
    pub pending_messages: usize,
    /// Count by memory type
    pub type_counts: HashMap<MemoryType, usize>,
    /// Average effective importance
    pub average_importance: f32,
}

/// Working memory for current conversation context
#[derive(Debug, Default)]
pub struct WorkingMemory {
    /// Current conversation topic
    pub current_topic: Option<String>,
    /// Active entities mentioned
    pub active_entities: Vec<String>,
    /// Recent facts mentioned
    pub recent_facts: Vec<String>,
    /// User's current intent
    pub current_intent: Option<String>,
    /// Temporary notes
    pub scratch_pad: HashMap<String, String>,
}

impl WorkingMemory {
    /// Create new working memory
    pub fn new() -> Self {
        Self::default()
    }

    /// Update topic
    pub fn set_topic(&mut self, topic: &str) {
        self.current_topic = Some(topic.to_string());
    }

    /// Add an entity
    pub fn add_entity(&mut self, entity: &str) {
        if !self.active_entities.contains(&entity.to_string()) {
            self.active_entities.push(entity.to_string());
            // Keep only last 10 entities
            if self.active_entities.len() > 10 {
                self.active_entities.remove(0);
            }
        }
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: &str) {
        self.recent_facts.push(fact.to_string());
        if self.recent_facts.len() > 5 {
            self.recent_facts.remove(0);
        }
    }

    /// Set intent
    pub fn set_intent(&mut self, intent: &str) {
        self.current_intent = Some(intent.to_string());
    }

    /// Add to scratch pad
    pub fn note(&mut self, key: &str, value: &str) {
        self.scratch_pad.insert(key.to_string(), value.to_string());
    }

    /// Get from scratch pad
    pub fn get_note(&self, key: &str) -> Option<&String> {
        self.scratch_pad.get(key)
    }

    /// Clear working memory
    pub fn clear(&mut self) {
        self.current_topic = None;
        self.active_entities.clear();
        self.recent_facts.clear();
        self.current_intent = None;
        self.scratch_pad.clear();
    }

    /// Build context summary
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if let Some(topic) = &self.current_topic {
            parts.push(format!("Topic: {}", topic));
        }

        if let Some(intent) = &self.current_intent {
            parts.push(format!("Intent: {}", intent));
        }

        if !self.active_entities.is_empty() {
            parts.push(format!("Entities: {}", self.active_entities.join(", ")));
        }

        if !self.recent_facts.is_empty() {
            parts.push(format!("Facts: {}", self.recent_facts.join("; ")));
        }

        parts.join("\n")
    }
}

/// Generate a unique memory ID
fn generate_memory_id() -> String {
    use std::time::SystemTime;
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("mem_{:x}", timestamp)
}

/// Memory manager combining long-term and working memory
pub struct MemoryManager {
    /// Long-term memory store
    pub long_term: MemoryStore,
    /// Working memory
    pub working: WorkingMemory,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            long_term: MemoryStore::new(config),
            working: WorkingMemory::new(),
        }
    }

    /// Process a message and update memories
    pub fn process_message(&mut self, message: &ChatMessage) {
        crate::diag_debug!("[memory] process_message: role={:?}, content_len={}", message.role, message.content.len());
        // Add to pending for summarization
        self.long_term.add_message(message.clone());

        // Extract entities and facts for working memory
        let content = &message.content;

        // Simple entity extraction (URLs, emails)
        for word in content.split_whitespace() {
            if word.contains('@') && word.contains('.') {
                self.working.add_entity(word);
            } else if word.starts_with("http") {
                self.working.add_entity(word);
            }
        }
    }

    /// Store a fact in long-term memory
    pub fn remember_fact(&mut self, fact: &str, importance: f32) -> String {
        crate::diag_debug!("[memory] remember_fact: importance={:.2}, text={:.200}", importance, fact);
        let memory = MemoryEntry::new(fact, MemoryType::Fact).with_importance(importance);
        self.long_term.add(memory)
    }

    /// Store a preference
    pub fn remember_preference(&mut self, preference: &str) -> String {
        let memory = MemoryEntry::new(preference, MemoryType::Preference).with_importance(0.7);
        self.long_term.add(memory)
    }

    /// Store a goal
    pub fn remember_goal(&mut self, goal: &str) -> String {
        let memory = MemoryEntry::new(goal, MemoryType::Goal).with_importance(0.8);
        self.long_term.add(memory)
    }

    /// Recall relevant memories for a query
    pub fn recall(&mut self, query: &str) -> Vec<&MemoryEntry> {
        self.long_term.search(query)
    }

    /// Build context for a query
    pub fn build_context(&mut self, query: &str, max_tokens: usize) -> String {
        crate::diag_debug!("[memory] MemoryManager::build_context: max_tokens={}", max_tokens);
        let mut context = String::new();

        // Add working memory summary
        let working_summary = self.working.summary();
        if !working_summary.is_empty() {
            context.push_str("Current context:\n");
            context.push_str(&working_summary);
            context.push_str("\n\n");
        }

        // Add relevant long-term memories
        let long_term_context = self.long_term.build_context(query, max_tokens);
        if !long_term_context.is_empty() {
            context.push_str("Relevant memories:\n");
            context.push_str(&long_term_context);
        }

        crate::diag_debug!("[memory] MemoryManager::build_context: result {} chars", context.len());
        crate::diag_trace!("[memory] MemoryManager::build_context: content={:.500}", context);
        context
    }

    /// Clear working memory for new conversation
    pub fn new_conversation(&mut self) {
        self.working.clear();
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

// =============================================================================
// LIST TRACKING & REFERENCE RESOLUTION
// =============================================================================

/// A tracked list from LLM or user output, stored for later reference resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrackedList {
    /// Items in the list
    pub items: Vec<String>,
    /// The topic/context the list was about
    pub topic: String,
    /// Timestamp when the list was generated
    pub timestamp: DateTime<Utc>,
    /// Which message produced this list (0-based turn index)
    pub turn_index: usize,
}

/// Tracks lists and resolves back-references to previous conversation content.
#[derive(Debug, Default)]
pub struct ReferenceResolver {
    /// All tracked lists from the conversation, most recent first
    pub tracked_lists: Vec<TrackedList>,
    /// Maximum lists to retain
    pub max_lists: usize,
}

impl ReferenceResolver {
    pub fn new() -> Self {
        Self {
            tracked_lists: Vec::new(),
            max_lists: 50,
        }
    }

    /// Scan a message for numbered/bulleted lists and track them.
    pub fn track_lists_in_message(&mut self, message: &str, topic: &str, turn_index: usize) {
        let items = Self::extract_list_items(message);
        if items.len() >= 2 {
            self.tracked_lists.insert(0, TrackedList {
                items,
                topic: topic.to_string(),
                timestamp: Utc::now(),
                turn_index,
            });
            if self.tracked_lists.len() > self.max_lists {
                self.tracked_lists.truncate(self.max_lists);
            }
        }
    }

    /// Extract list items from a message. Supports:
    /// - Numbered: "1. item", "1) item", "1- item"
    /// - Bulleted: "- item", "* item", "+ item"
    /// - Lettered: "a. item", "a) item"
    pub fn extract_list_items(text: &str) -> Vec<String> {
        let mut items = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            // Numbered: "1. ", "1) ", "1- ", "12. " etc.
            if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
                let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
                if let Some(content) = rest.strip_prefix(". ")
                    .or_else(|| rest.strip_prefix(") "))
                    .or_else(|| rest.strip_prefix("- "))
                {
                    let content = content.trim();
                    if !content.is_empty() {
                        items.push(content.to_string());
                    }
                    continue;
                }
            }
            // Bulleted: "- ", "* ", "+ "
            if let Some(content) = trimmed.strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .or_else(|| trimmed.strip_prefix("+ "))
            {
                let content = content.trim();
                if !content.is_empty() {
                    items.push(content.to_string());
                }
                continue;
            }
            // Lettered: "a. ", "a) ", "b. ", "b) "
            if trimmed.len() >= 3 {
                let first = trimmed.as_bytes()[0];
                if first.is_ascii_lowercase() {
                    if let Some(content) = trimmed[1..].strip_prefix(". ")
                        .or_else(|| trimmed[1..].strip_prefix(") "))
                    {
                        let content = content.trim();
                        if !content.is_empty() {
                            items.push(content.to_string());
                        }
                    }
                }
            }
        }
        items
    }

    /// Try to resolve a user reference like "the second one", "option 3",
    /// "the list about X" against tracked lists.
    /// Returns the resolved context string, or None if no match.
    pub fn resolve_reference(&self, user_message: &str) -> Option<String> {
        let msg_lower = user_message.to_lowercase();

        // Quick check: does the message contain reference patterns?
        let has_reference = Self::has_reference_pattern(&msg_lower);
        if !has_reference {
            return None;
        }

        // Try to extract a numeric reference (ordinal or cardinal)
        let index = Self::extract_item_index(&msg_lower);

        // Try to find which list the user is referring to
        let target_list = if self.tracked_lists.is_empty() {
            return None;
        } else if self.tracked_lists.len() == 1 || index.is_some() {
            // If there's only one list, or we have an index, use the most recent
            &self.tracked_lists[0]
        } else {
            // Try to match by topic keywords
            let best = self.tracked_lists.iter()
                .max_by_key(|list| {
                    let topic_lower = list.topic.to_lowercase();
                    msg_lower.split_whitespace()
                        .filter(|w| w.len() > 2 && topic_lower.contains(w))
                        .count()
                });
            best.unwrap_or(&self.tracked_lists[0])
        };

        if let Some(idx) = index {
            // Specific item reference
            if idx < target_list.items.len() {
                Some(format!(
                    "[Reference resolved: item {} from list about '{}': {}]",
                    idx + 1, target_list.topic, target_list.items[idx]
                ))
            } else {
                Some(format!(
                    "[Reference: the list about '{}' has {} items, but item {} was requested]",
                    target_list.topic, target_list.items.len(), idx + 1
                ))
            }
        } else {
            // General list reference
            let items_str: String = target_list.items.iter()
                .enumerate()
                .map(|(i, item)| format!("{}. {}", i + 1, item))
                .collect::<Vec<_>>()
                .join("\n");
            Some(format!(
                "[Reference resolved: list about '{}' ({} items):\n{}]",
                target_list.topic, target_list.items.len(), items_str
            ))
        }
    }

    /// Check if a message contains reference patterns.
    fn has_reference_pattern(msg: &str) -> bool {
        let patterns = [
            "the first", "the second", "the third", "the fourth", "the fifth",
            "el primer", "el segundo", "el tercer", "el cuarto", "el quinto",
            "la primer", "la segund", "la tercer", "la cuart", "la quint",
            "option ", "opción ", "opcion ",
            "item ", "punto ", "element",
            "number ", "número ", "numero ",
            "that list", "the list", "esa lista", "la lista",
            "lo anterior", "the previous", "lo de antes",
            "which one", "cuál", "cual",
        ];
        patterns.iter().any(|p| msg.contains(p))
    }

    /// Extract a 0-based item index from ordinal/cardinal references.
    fn extract_item_index(msg: &str) -> Option<usize> {
        // Ordinals (English + Spanish)
        let ordinals = [
            ("first", 0), ("primer", 0), ("1st", 0),
            ("second", 1), ("segund", 1), ("2nd", 1),
            ("third", 2), ("tercer", 2), ("3rd", 3),
            ("fourth", 3), ("cuart", 3), ("4th", 3),
            ("fifth", 4), ("quint", 4), ("5th", 4),
            ("sixth", 5), ("sext", 5),
            ("seventh", 6), ("séptim", 6), ("septim", 6),
            ("eighth", 7), ("octav", 7),
            ("ninth", 8), ("noven", 8),
            ("tenth", 9), ("décim", 9), ("decim", 9),
        ];

        for (word, idx) in &ordinals {
            if msg.contains(word) {
                return Some(*idx);
            }
        }

        // Cardinal with context: "option 3", "punto 5", "number 2"
        let cardinal_prefixes = [
            "option ", "opción ", "opcion ", "item ", "punto ",
            "number ", "número ", "numero ", "element ", "elemento ",
        ];
        for prefix in &cardinal_prefixes {
            if let Some(pos) = msg.find(prefix) {
                let after = &msg[pos + prefix.len()..];
                if let Some(num) = after.split_whitespace().next() {
                    if let Ok(n) = num.trim_matches(|c: char| !c.is_ascii_digit()).parse::<usize>() {
                        if n > 0 {
                            return Some(n - 1); // 0-based
                        }
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let memory = MemoryEntry::new("User prefers Rust", MemoryType::Preference)
            .with_importance(0.8)
            .with_tag("programming");

        assert_eq!(memory.memory_type, MemoryType::Preference);
        assert_eq!(memory.importance, 0.8);
        assert!(memory.tags.contains(&"programming".to_string()));
    }

    #[test]
    fn test_memory_store() {
        let mut store = MemoryStore::new(MemoryConfig::default());

        let id1 = store.add(MemoryEntry::new("Fact 1", MemoryType::Fact));
        let id2 = store.add(MemoryEntry::new("Fact 2", MemoryType::Fact));

        assert!(store.get(&id1).is_some());
        assert!(store.get(&id2).is_some());

        let stats = store.stats();
        assert_eq!(stats.total_memories, 2);
    }

    #[test]
    fn test_memory_search() {
        let mut store = MemoryStore::new(MemoryConfig::default());

        store.add(MemoryEntry::new(
            "User likes Rust programming",
            MemoryType::Preference,
        ));
        store.add(MemoryEntry::new(
            "User dislikes Python",
            MemoryType::Preference,
        ));
        store.add(MemoryEntry::new(
            "Goal is to learn systems programming",
            MemoryType::Goal,
        ));

        let results = store.search("Rust");
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_memory_decay() {
        let memory = MemoryEntry::new("Test", MemoryType::Fact).with_importance(1.0);

        // Fresh memory should have high effective importance
        let effective = memory.effective_importance(30.0);
        assert!(effective > 0.9);
    }

    #[test]
    fn test_working_memory() {
        let mut working = WorkingMemory::new();

        working.set_topic("Programming");
        working.add_entity("user@example.com");
        working.add_fact("User wants to learn Rust");

        let summary = working.summary();
        assert!(summary.contains("Programming"));
        assert!(summary.contains("user@example.com"));
    }

    #[test]
    fn test_memory_manager() {
        let mut manager = MemoryManager::default();

        manager.remember_fact("User is learning Rust", 0.8);
        manager.remember_preference("Prefers concise explanations");

        let memories = manager.recall("Rust");
        assert!(!memories.is_empty());
    }

    #[test]
    fn test_memory_export_import() {
        let mut store = MemoryStore::new(MemoryConfig::default());
        store.add(MemoryEntry::new("Test memory", MemoryType::Fact));

        let json = store.export().unwrap();
        assert!(json.contains("Test memory"));

        let mut new_store = MemoryStore::new(MemoryConfig::default());
        let count = new_store.import(&json).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_get_by_type_and_get_by_tag() {
        let mut store = MemoryStore::new(MemoryConfig::default());

        store.add(MemoryEntry::new("Fact alpha", MemoryType::Fact).with_tag("science"));
        store.add(MemoryEntry::new("Goal beta", MemoryType::Goal).with_tag("science"));
        store.add(MemoryEntry::new("Fact gamma", MemoryType::Fact).with_tag("history"));

        // get_by_type should return only Facts
        let facts = store.get_by_type(MemoryType::Fact);
        assert_eq!(facts.len(), 2);
        for f in &facts {
            assert_eq!(f.memory_type, MemoryType::Fact);
        }

        // get_by_type for a type with no entries
        let events = store.get_by_type(MemoryType::Event);
        assert!(events.is_empty());

        // get_by_tag should return matching tag
        let science = store.get_by_tag("science");
        assert_eq!(science.len(), 2);

        let history = store.get_by_tag("history");
        assert_eq!(history.len(), 1);

        let unknown = store.get_by_tag("nonexistent");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_recall_marks_accessed_and_remove() {
        let mut store = MemoryStore::new(MemoryConfig::default());

        let id = store.add(MemoryEntry::new("Recall me", MemoryType::Fact));

        // Initial recall_count is 0
        assert_eq!(store.get(&id).unwrap().recall_count, 0);

        // recall should increment recall_count
        let recalled = store.recall(&id);
        assert!(recalled.is_some());
        assert_eq!(store.get(&id).unwrap().recall_count, 1);

        // recall again
        store.recall(&id);
        assert_eq!(store.get(&id).unwrap().recall_count, 2);

        // recall non-existent id returns None
        assert!(store.recall("nonexistent_id").is_none());

        // remove should return the entry and leave store empty
        let removed = store.remove(&id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().content, "Recall me");
        assert!(store.get(&id).is_none());

        // remove non-existent returns None
        assert!(store.remove("nonexistent_id").is_none());
    }

    #[test]
    fn test_working_memory_clear_and_entity_overflow() {
        let mut wm = WorkingMemory::new();

        // Set various fields
        wm.set_topic("Testing");
        wm.set_intent("verify overflow");
        wm.note("key1", "value1");
        wm.add_fact("fact1");

        assert!(wm.current_topic.is_some());
        assert!(wm.current_intent.is_some());
        assert_eq!(wm.get_note("key1"), Some(&"value1".to_string()));

        // Clear should reset everything
        wm.clear();
        assert!(wm.current_topic.is_none());
        assert!(wm.current_intent.is_none());
        assert!(wm.active_entities.is_empty());
        assert!(wm.recent_facts.is_empty());
        assert!(wm.scratch_pad.is_empty());

        // Entity dedup: adding the same entity twice should not duplicate
        wm.add_entity("duplicate");
        wm.add_entity("duplicate");
        assert_eq!(wm.active_entities.len(), 1);

        // Entity overflow: adding >10 unique entities keeps only last 10
        for i in 0..12 {
            wm.add_entity(&format!("entity_{}", i));
        }
        assert_eq!(wm.active_entities.len(), 10);
        // First entities (including "duplicate") should have been evicted
        assert!(!wm.active_entities.contains(&"duplicate".to_string()));

        // Fact overflow: adding >5 facts keeps only last 5
        wm.recent_facts.clear();
        for i in 0..7 {
            wm.add_fact(&format!("fact_{}", i));
        }
        assert_eq!(wm.recent_facts.len(), 5);
        // The first two facts should have been evicted
        assert!(!wm.recent_facts.contains(&"fact_0".to_string()));
        assert!(!wm.recent_facts.contains(&"fact_1".to_string()));
        assert!(wm.recent_facts.contains(&"fact_6".to_string()));
    }
}
