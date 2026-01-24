//! Context window management
//!
//! Sliding window and intelligent context management for LLMs.

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
                .unwrap()
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
            self.system_message.as_ref().map(|m| m.token_count).unwrap_or(0)
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
        let max = self.config.max_tokens.saturating_sub(self.config.response_reserve);
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
        let max_tokens = self.config.max_tokens.saturating_sub(self.config.response_reserve);

        while self.total_tokens > max_tokens && self.messages.len() > self.config.min_messages {
            match self.config.eviction_strategy {
                EvictionStrategy::OldestFirst => {
                    self.evict_oldest();
                }
                EvictionStrategy::LeastImportant => {
                    self.evict_least_important();
                }
                EvictionStrategy::Summarize => {
                    // For now, fall back to oldest first
                    // Full summarization would require LLM call
                    self.evict_oldest();
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
        let pos = self.messages.iter()
            .enumerate()
            .filter(|(_, m)| !m.pinned)
            .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap())
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

        let pos = self.messages.iter()
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

    /// Clear all messages except system
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens = self.system_message.as_ref().map(|m| m.token_count).unwrap_or(0);
    }

    /// Get window statistics
    pub fn stats(&self) -> ContextWindowStats {
        let pinned_count = self.messages.iter().filter(|m| m.pinned).count();
        let pinned_tokens: usize = self.messages.iter()
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
            window.add_user(&format!("Message number {} with some extra text to use tokens", i));
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
}
