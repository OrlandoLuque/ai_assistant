//! Context window management
//!
//! Sliding window and intelligent context management for LLMs.

use std::collections::HashMap;
use std::collections::VecDeque;

/// Message in the context window
#[derive(Debug, Clone)]
pub struct ContextMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub token_count: usize,
    pub timestamp: u64,
    pub pinned: bool,
    pub importance: f64,
}

impl ContextMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            token_count: estimate_tokens(content),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            pinned: false,
            importance: 0.5,
        }
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

/// Estimate tokens in text (rough approximation)
fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 chars per token for English
    (text.len() + 3) / 4
}

/// Split text into sentences by punctuation boundaries
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();

    let mut i = 0;
    while i < len {
        if (bytes[i] == b'.' || bytes[i] == b'?' || bytes[i] == b'!')
            && (i + 1 < len && bytes[i + 1] == b' ' || i + 1 == len)
        {
            let sentence = text[start..=i].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = i + 1;
        }
        i += 1;
    }

    // Trailing text after last punctuation
    if start < len {
        let sentence = text[start..].trim();
        if !sentence.is_empty() {
            sentences.push(sentence);
        }
    }

    sentences
}

/// Compute TF-based scores for each sentence in the text
fn compute_sentence_scores(text: &str) -> Vec<(String, f64)> {
    let sentences = split_sentences(text);

    // Count word frequencies across all text
    let mut freq: HashMap<String, usize> = HashMap::new();
    for w in text.split_whitespace() {
        *freq.entry(w.to_lowercase()).or_insert(0) += 1;
    }

    // Score each sentence
    sentences
        .iter()
        .map(|s| {
            let s_words: Vec<&str> = s.split_whitespace().collect();
            if s_words.is_empty() {
                return (s.to_string(), 0.0);
            }
            let score: f64 = s_words
                .iter()
                .map(|w| *freq.get(&w.to_lowercase()).unwrap_or(&0) as f64)
                .sum::<f64>()
                / s_words.len() as f64;
            (s.to_string(), score)
        })
        .collect()
}

/// Context window configuration
#[derive(Debug, Clone)]
pub struct ContextWindowConfig {
    /// Maximum tokens in the context window
    pub max_tokens: usize,
    /// Reserve tokens for response
    pub response_reserve: usize,
    /// Minimum messages to keep (even if over token limit)
    pub min_messages: usize,
    /// Strategy for removing messages when over limit
    pub eviction_strategy: EvictionStrategy,
    /// Whether to keep system message always
    pub preserve_system: bool,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            response_reserve: 1024,
            min_messages: 2,
            eviction_strategy: EvictionStrategy::OldestFirst,
            preserve_system: true,
        }
    }
}

/// Strategy for evicting messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Remove oldest messages first
    OldestFirst,
    /// Remove least important messages first
    LeastImportant,
    /// Summarize old messages instead of removing
    Summarize,
    /// Remove middle messages, keep recent and oldest
    KeepEnds,
}

/// Sliding context window
pub struct ContextWindow {
    messages: VecDeque<ContextMessage>,
    config: ContextWindowConfig,
    total_tokens: usize,
    system_message: Option<ContextMessage>,
}

impl ContextWindow {
    pub fn new(config: ContextWindowConfig) -> Self {
        Self {
            messages: VecDeque::new(),
            config,
            total_tokens: 0,
            system_message: None,
        }
    }

    /// Set the system message
    pub fn set_system(&mut self, content: &str) {
        let msg = ContextMessage::new("system", content).pinned();
        self.total_tokens = self.total_tokens.saturating_sub(
            self.system_message
                .as_ref()
                .map(|m| m.token_count)
                .unwrap_or(0),
        );
        self.total_tokens += msg.token_count;
        self.system_message = Some(msg);
    }

    /// Add a message to the context
    pub fn add(&mut self, message: ContextMessage) {
        self.total_tokens += message.token_count;
        self.messages.push_back(message);
        self.enforce_limits();
    }

    /// Add a user message
    pub fn add_user(&mut self, content: &str) {
        self.add(ContextMessage::new("user", content));
    }

    /// Add an assistant message
    pub fn add_assistant(&mut self, content: &str) {
        self.add(ContextMessage::new("assistant", content));
    }

    /// Pin a message by ID
    pub fn pin(&mut self, id: &str) {
        if let Some(msg) = self.messages.iter_mut().find(|m| m.id == id) {
            msg.pinned = true;
        }
    }

    /// Unpin a message by ID
    pub fn unpin(&mut self, id: &str) {
        if let Some(msg) = self.messages.iter_mut().find(|m| m.id == id) {
            msg.pinned = false;
        }
    }

    /// Get available tokens for response
    pub fn available_tokens(&self) -> usize {
        let used = self.total_tokens;
        let max = self
            .config
            .max_tokens
            .saturating_sub(self.config.response_reserve);
        max.saturating_sub(used)
    }

    /// Get all messages for API call
    pub fn get_messages(&self) -> Vec<&ContextMessage> {
        let mut result = Vec::new();

        if let Some(sys) = &self.system_message {
            result.push(sys);
        }

        for msg in &self.messages {
            result.push(msg);
        }

        result
    }

    /// Get total token count
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.messages.len() + if self.system_message.is_some() { 1 } else { 0 }
    }

    /// Enforce token limits
    fn enforce_limits(&mut self) {
        let max_tokens = self
            .config
            .max_tokens
            .saturating_sub(self.config.response_reserve);

        while self.total_tokens > max_tokens && self.messages.len() > self.config.min_messages {
            match self.config.eviction_strategy {
                EvictionStrategy::OldestFirst => {
                    self.evict_oldest();
                }
                EvictionStrategy::LeastImportant => {
                    self.evict_least_important();
                }
                EvictionStrategy::Summarize => {
                    self.summarize_messages();
                }
                EvictionStrategy::KeepEnds => {
                    self.evict_middle();
                }
            }
        }
    }

    fn evict_oldest(&mut self) {
        // Find first non-pinned message
        if let Some(pos) = self.messages.iter().position(|m| !m.pinned) {
            if let Some(removed) = self.messages.remove(pos) {
                self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
            }
        }
    }

    fn evict_least_important(&mut self) {
        // Find least important non-pinned message
        let pos = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| !m.pinned)
            .min_by(|(_, a), (_, b)| {
                a.importance
                    .partial_cmp(&b.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(pos) = pos {
            if let Some(removed) = self.messages.remove(pos) {
                self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
            }
        }
    }

    fn evict_middle(&mut self) {
        // Remove from the middle, keeping first and last
        let len = self.messages.len();
        if len <= 2 {
            return;
        }

        // Find middle non-pinned message
        let mid_start = len / 4;
        let mid_end = 3 * len / 4;

        let pos = self
            .messages
            .iter()
            .enumerate()
            .filter(|(i, m)| !m.pinned && *i >= mid_start && *i <= mid_end)
            .map(|(i, _)| i)
            .next();

        if let Some(pos) = pos {
            if let Some(removed) = self.messages.remove(pos) {
                self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
            }
        } else {
            // Fall back to oldest
            self.evict_oldest();
        }
    }

    /// Extractive summarization of the oldest messages
    fn summarize_messages(&mut self) {
        let len = self.messages.len();
        if len <= 1 {
            // Nothing to summarize; fall back to evict_oldest
            self.evict_oldest();
            return;
        }

        // Take the oldest 30% of messages (at least 2, keep at least 1 recent)
        let count = ((len as f64 * 0.3).ceil() as usize).max(2).min(len - 1);

        // Collect oldest non-pinned messages to summarize
        let mut to_summarize_indices: Vec<usize> = Vec::new();
        for i in 0..self.messages.len() {
            if to_summarize_indices.len() >= count {
                break;
            }
            if !self.messages[i].pinned {
                to_summarize_indices.push(i);
            }
        }

        if to_summarize_indices.is_empty() {
            // All messages are pinned; fall back
            self.evict_oldest();
            return;
        }

        // Concatenate content from the selected messages
        let combined: String = to_summarize_indices
            .iter()
            .map(|&i| self.messages[i].content.as_str())
            .collect::<Vec<&str>>()
            .join(" ");

        // Score sentences and pick the top 3
        let scored = compute_sentence_scores(&combined);
        let mut indexed: Vec<(usize, String, f64)> = scored
            .into_iter()
            .enumerate()
            .map(|(i, (s, score))| (i, s, score))
            .collect();

        // Sort by score descending to pick top 3
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut top: Vec<(usize, String)> = indexed
            .into_iter()
            .take(3)
            .map(|(i, s, _)| (i, s))
            .collect();

        // Restore original order
        top.sort_by_key(|(i, _)| *i);

        let summary_text = top
            .iter()
            .map(|(_, s)| s.as_str())
            .collect::<Vec<&str>>()
            .join(". ");
        let n = to_summarize_indices.len();
        let summary_content = format!("[Summary of {} earlier messages: {}]", n, summary_text);

        // Remove summarized messages (in reverse to preserve indices)
        let mut removed_tokens = 0usize;
        for &i in to_summarize_indices.iter().rev() {
            if let Some(removed) = self.messages.remove(i) {
                removed_tokens += removed.token_count;
            }
        }
        self.total_tokens = self.total_tokens.saturating_sub(removed_tokens);

        // Insert summary message at the front
        let summary_msg = ContextMessage::new("system", &summary_content);
        self.total_tokens += summary_msg.token_count;
        self.messages.push_front(summary_msg);
    }

    /// Clear all messages except system
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens = self
            .system_message
            .as_ref()
            .map(|m| m.token_count)
            .unwrap_or(0);
    }

    /// Get window statistics
    pub fn stats(&self) -> ContextWindowStats {
        let pinned_count = self.messages.iter().filter(|m| m.pinned).count();
        let pinned_tokens: usize = self
            .messages
            .iter()
            .filter(|m| m.pinned)
            .map(|m| m.token_count)
            .sum();

        ContextWindowStats {
            total_messages: self.messages.len(),
            total_tokens: self.total_tokens,
            pinned_messages: pinned_count,
            pinned_tokens,
            available_tokens: self.available_tokens(),
            utilization: self.total_tokens as f64 / self.config.max_tokens as f64,
        }
    }
}

impl Default for ContextWindow {
    fn default() -> Self {
        Self::new(ContextWindowConfig::default())
    }
}

/// Context window statistics
#[derive(Debug, Clone)]
pub struct ContextWindowStats {
    pub total_messages: usize,
    pub total_tokens: usize,
    pub pinned_messages: usize,
    pub pinned_tokens: usize,
    pub available_tokens: usize,
    pub utilization: f64,
}

/// Builder for context window configuration
pub struct ContextWindowBuilder {
    config: ContextWindowConfig,
}

impl ContextWindowBuilder {
    pub fn new() -> Self {
        Self {
            config: ContextWindowConfig::default(),
        }
    }

    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = tokens;
        self
    }

    pub fn response_reserve(mut self, tokens: usize) -> Self {
        self.config.response_reserve = tokens;
        self
    }

    pub fn min_messages(mut self, count: usize) -> Self {
        self.config.min_messages = count;
        self
    }

    pub fn eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.config.eviction_strategy = strategy;
        self
    }

    pub fn preserve_system(mut self, preserve: bool) -> Self {
        self.config.preserve_system = preserve;
        self
    }

    pub fn build(self) -> ContextWindow {
        ContextWindow::new(self.config)
    }
}

impl Default for ContextWindowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage() {
        let mut window = ContextWindow::default();

        window.add_user("Hello");
        window.add_assistant("Hi there!");

        assert_eq!(window.message_count(), 2);
    }

    #[test]
    fn test_system_message() {
        let mut window = ContextWindow::default();

        window.set_system("You are a helpful assistant.");
        window.add_user("Hello");

        let messages = window.get_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
    }

    #[test]
    fn test_pinning() {
        let mut window = ContextWindow::default();

        window.add_user("Important message");
        let id = window.messages.back().unwrap().id.clone();
        window.pin(&id);

        assert!(window.messages.back().unwrap().pinned);
    }

    #[test]
    fn test_eviction() {
        let config = ContextWindowConfig {
            max_tokens: 100,
            response_reserve: 20,
            min_messages: 1,
            ..Default::default()
        };
        let mut window = ContextWindow::new(config);

        // Add messages until over limit
        for i in 0..20 {
            window.add_user(&format!(
                "Message number {} with some extra text to use tokens",
                i
            ));
        }

        // Should have evicted some messages
        assert!(window.messages.len() < 20);
    }

    #[test]
    fn test_stats() {
        let mut window = ContextWindow::default();

        window.add_user("Hello");
        window.add_assistant("Hi");

        let stats = window.stats();
        assert_eq!(stats.total_messages, 2);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn test_summarize_eviction_reduces_messages() {
        // Use a small token budget so summarize is triggered
        let config = ContextWindowConfig {
            max_tokens: 200,
            response_reserve: 20,
            min_messages: 1,
            eviction_strategy: EvictionStrategy::Summarize,
            preserve_system: false,
        };
        let mut window = ContextWindow::new(config);

        // Add 10 messages with enough text to exceed the token budget
        for i in 0..10 {
            window.add_user(&format!(
                "Message number {} contains important data about the project. We need to track progress.",
                i
            ));
        }

        // Summarize eviction should have reduced the message count
        assert!(window.messages.len() < 10);

        // There should be a summary message present (role "system" at front)
        let has_summary = window
            .messages
            .iter()
            .any(|m| m.role == "system" && m.content.starts_with("[Summary of"));
        assert!(has_summary, "Expected a summary message but found none");
    }

    #[test]
    fn test_summarize_preserves_recent() {
        // Give enough budget so only one round of summarization is needed
        let config = ContextWindowConfig {
            max_tokens: 400,
            response_reserve: 20,
            min_messages: 1,
            eviction_strategy: EvictionStrategy::Summarize,
            preserve_system: false,
        };
        let mut window = ContextWindow::new(config);

        // Add messages; the last one is distinctive
        for i in 0..8 {
            window.add_user(&format!(
                "Old message {} with enough words to occupy tokens in the context window budget.",
                i
            ));
        }
        let recent_text =
            "This is the most recent message and should be preserved after summarization occurs.";
        window.add_user(recent_text);

        // The most recent message must still be present
        let last = window.messages.back().unwrap();
        assert_eq!(
            last.content, recent_text,
            "The most recent message should be preserved"
        );
    }

    #[test]
    fn test_summarize_content_quality() {
        // Manually invoke summarize_messages to inspect the summary content
        let config = ContextWindowConfig {
            max_tokens: 100_000, // large budget so enforce_limits doesn't auto-evict
            response_reserve: 0,
            min_messages: 1,
            eviction_strategy: EvictionStrategy::Summarize,
            preserve_system: false,
        };
        let mut window = ContextWindow::new(config);

        // Add messages with distinctive keywords
        window.add_user("The weather today is sunny and warm. Rust programming is powerful.");
        window
            .add_user("Machine learning models require large datasets. Rust programming is fast.");
        window.add_user("Databases store structured data efficiently. The weather today is cold.");
        window.add_user("Final message that should remain untouched.");

        // Manually trigger summarization
        window.summarize_messages();

        // Should have fewer messages now
        assert!(window.messages.len() < 4);

        // Find the summary message
        let summary = window
            .messages
            .iter()
            .find(|m| m.content.starts_with("[Summary of"))
            .expect("Summary message should exist");

        // The summary should contain key repeated terms from the original messages
        let content_lower = summary.content.to_lowercase();
        assert!(
            content_lower.contains("rust") || content_lower.contains("weather"),
            "Summary should contain high-frequency words from originals, got: {}",
            summary.content
        );
    }
}
