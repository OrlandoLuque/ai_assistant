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

/// Estimate tokens in text — delegates to the canonical implementation.
fn estimate_tokens(text: &str) -> usize {
    crate::context::estimate_tokens(text)
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
#[non_exhaustive]
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
#[non_exhaustive]
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

// =============================================================================
// CONTEXT OVERFLOW PREVENTION (v4 - item 8.3)
// =============================================================================

/// Overflow thresholds that define when warning, critical, and emergency levels
/// are triggered based on the fraction of the context window currently in use.
#[derive(Debug, Clone)]
pub struct OverflowThresholds {
    /// Fraction at which a warning is issued (default 0.70)
    pub warning: f64,
    /// Fraction at which usage is considered critical (default 0.85)
    pub critical: f64,
    /// Fraction at which an emergency eviction should occur (default 0.95)
    pub emergency: f64,
}

impl Default for OverflowThresholds {
    fn default() -> Self {
        Self {
            warning: 0.70,
            critical: 0.85,
            emergency: 0.95,
        }
    }
}

/// Severity level of context window overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum OverflowLevel {
    /// Usage is within normal bounds
    Normal,
    /// Usage has crossed the warning threshold
    Warning,
    /// Usage has crossed the critical threshold
    Critical,
    /// Usage has crossed the emergency threshold
    Emergency,
}

impl OverflowLevel {
    /// Human-readable name for this overflow level.
    pub fn display_name(&self) -> &'static str {
        match self {
            OverflowLevel::Normal => "Normal",
            OverflowLevel::Warning => "Warning",
            OverflowLevel::Critical => "Critical",
            OverflowLevel::Emergency => "Emergency",
        }
    }
}

/// Auto-configure max_tokens from model capabilities.
///
/// When the model's context window size is known, this calculates the effective
/// maximum token budget by subtracting a response reserve. When unknown, a
/// fallback value is used.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutoTokenConfig {
    /// Known context window size of the model, if available
    pub model_context_window: Option<usize>,
    /// Tokens reserved for the model's response
    pub response_reserve: usize,
    /// Fallback max tokens when the model context window is unknown
    pub fallback_max_tokens: usize,
}

impl Default for AutoTokenConfig {
    fn default() -> Self {
        Self {
            model_context_window: None,
            response_reserve: 1024,
            fallback_max_tokens: 4096,
        }
    }
}

impl AutoTokenConfig {
    /// Create a config from a known model context window size.
    pub fn from_model(window: usize) -> Self {
        Self {
            model_context_window: Some(window),
            ..Default::default()
        }
    }

    /// Compute the effective maximum tokens available for context.
    ///
    /// If the model window is known, subtracts the response reserve (clamping to
    /// zero rather than underflowing). Otherwise returns the fallback value.
    pub fn effective_max_tokens(&self) -> usize {
        match self.model_context_window {
            Some(window) => window.saturating_sub(self.response_reserve),
            None => self.fallback_max_tokens,
        }
    }
}

/// Monitors context window usage and detects overflow conditions.
///
/// The monitor tracks the current token usage against a configured maximum and
/// fires an optional callback whenever the overflow level increases (i.e.
/// transitions to a more severe level).
pub struct ContextOverflowMonitor {
    thresholds: OverflowThresholds,
    max_tokens: usize,
    current_usage: usize,
    on_overflow: Option<Box<dyn Fn(OverflowLevel, usize, usize) + Send + Sync>>,
    last_level: OverflowLevel,
    overflow_count: usize,
}

impl ContextOverflowMonitor {
    /// Create a new monitor with default thresholds and the given max token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            thresholds: OverflowThresholds::default(),
            max_tokens,
            current_usage: 0,
            on_overflow: None,
            last_level: OverflowLevel::Normal,
            overflow_count: 0,
        }
    }

    /// Create a monitor from an `AutoTokenConfig`, using its effective max tokens.
    pub fn from_auto_config(config: &AutoTokenConfig) -> Self {
        Self::new(config.effective_max_tokens())
    }

    /// Override the default thresholds.
    pub fn with_thresholds(mut self, thresholds: OverflowThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Register a callback that fires when the overflow level increases.
    ///
    /// The callback receives `(new_level, current_usage, max_tokens)`.
    pub fn on_overflow<F>(mut self, callback: F) -> Self
    where
        F: Fn(OverflowLevel, usize, usize) + Send + Sync + 'static,
    {
        self.on_overflow = Some(Box::new(callback));
        self
    }

    /// Update the current usage and return the new overflow level.
    ///
    /// If the level has increased compared to the last update, the overflow
    /// callback (if any) is fired and the overflow counter is incremented.
    pub fn update(&mut self, current_tokens: usize) -> OverflowLevel {
        self.current_usage = current_tokens;
        let new_level = self.compute_level();

        if new_level > self.last_level {
            self.overflow_count += 1;
            if let Some(ref cb) = self.on_overflow {
                cb(new_level, self.current_usage, self.max_tokens);
            }
        }

        self.last_level = new_level;
        new_level
    }

    /// Return the current overflow level without updating usage.
    pub fn check_level(&self) -> OverflowLevel {
        self.last_level
    }

    /// Fraction of the context window currently in use (0.0 to 1.0+).
    pub fn usage_fraction(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.current_usage as f64 / self.max_tokens as f64
    }

    /// Tokens remaining before reaching the maximum.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.current_usage)
    }

    /// The configured maximum token budget.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The current token usage last provided via `update`.
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Number of times the overflow level has increased since creation or last reset.
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// The overflow level as of the most recent `update` call.
    pub fn last_level(&self) -> OverflowLevel {
        self.last_level
    }

    /// A recommended action string based on the current overflow level.
    pub fn recommended_action(&self) -> &'static str {
        match self.last_level {
            OverflowLevel::Normal => "No action needed",
            OverflowLevel::Warning => "Consider summarizing older messages",
            OverflowLevel::Critical => "Evict low-priority messages immediately",
            OverflowLevel::Emergency => "Aggressive eviction required to prevent data loss",
        }
    }

    /// Reset the monitor state (usage, level, and overflow count).
    pub fn reset(&mut self) {
        self.current_usage = 0;
        self.last_level = OverflowLevel::Normal;
        self.overflow_count = 0;
    }

    /// Compute the overflow level from current usage and thresholds.
    fn compute_level(&self) -> OverflowLevel {
        let fraction = self.usage_fraction();
        if fraction >= self.thresholds.emergency {
            OverflowLevel::Emergency
        } else if fraction >= self.thresholds.critical {
            OverflowLevel::Critical
        } else if fraction >= self.thresholds.warning {
            OverflowLevel::Warning
        } else {
            OverflowLevel::Normal
        }
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

    // =========================================================================
    // Context Overflow Prevention tests (v4 - item 8.3)
    // =========================================================================

    #[test]
    fn test_overflow_thresholds_defaults() {
        let t = OverflowThresholds::default();
        assert!((t.warning - 0.70).abs() < f64::EPSILON);
        assert!((t.critical - 0.85).abs() < f64::EPSILON);
        assert!((t.emergency - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overflow_level_ordering() {
        assert!(OverflowLevel::Normal < OverflowLevel::Warning);
        assert!(OverflowLevel::Warning < OverflowLevel::Critical);
        assert!(OverflowLevel::Critical < OverflowLevel::Emergency);
    }

    #[test]
    fn test_overflow_level_display_name() {
        assert_eq!(OverflowLevel::Normal.display_name(), "Normal");
        assert_eq!(OverflowLevel::Warning.display_name(), "Warning");
        assert_eq!(OverflowLevel::Critical.display_name(), "Critical");
        assert_eq!(OverflowLevel::Emergency.display_name(), "Emergency");
    }

    #[test]
    fn test_auto_token_config_with_model() {
        let cfg = AutoTokenConfig {
            model_context_window: Some(8192),
            response_reserve: 1024,
            fallback_max_tokens: 4096,
        };
        assert_eq!(cfg.effective_max_tokens(), 8192 - 1024);
    }

    #[test]
    fn test_auto_token_config_without_model() {
        let cfg = AutoTokenConfig::default();
        assert_eq!(cfg.model_context_window, None);
        assert_eq!(cfg.effective_max_tokens(), 4096);
    }

    #[test]
    fn test_auto_token_from_model() {
        let cfg = AutoTokenConfig::from_model(32_000);
        assert_eq!(cfg.model_context_window, Some(32_000));
        // default response_reserve is 1024
        assert_eq!(cfg.effective_max_tokens(), 32_000 - 1024);
    }

    #[test]
    fn test_monitor_new_normal() {
        let monitor = ContextOverflowMonitor::new(1000);
        assert_eq!(monitor.check_level(), OverflowLevel::Normal);
        assert_eq!(monitor.current_usage(), 0);
        assert_eq!(monitor.max_tokens(), 1000);
        assert_eq!(monitor.remaining_tokens(), 1000);
    }

    #[test]
    fn test_monitor_update_warning() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        let level = monitor.update(750); // 0.75 >= 0.70 warning
        assert_eq!(level, OverflowLevel::Warning);
        assert_eq!(monitor.check_level(), OverflowLevel::Warning);
    }

    #[test]
    fn test_monitor_update_critical() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        let level = monitor.update(870); // 0.87 >= 0.85 critical
        assert_eq!(level, OverflowLevel::Critical);
    }

    #[test]
    fn test_monitor_update_emergency() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        let level = monitor.update(960); // 0.96 >= 0.95 emergency
        assert_eq!(level, OverflowLevel::Emergency);
    }

    #[test]
    fn test_monitor_callback_fires() {
        use std::sync::{Arc, Mutex};
        let fired = Arc::new(Mutex::new(Vec::new()));
        let fired_clone = Arc::clone(&fired);

        let mut monitor = ContextOverflowMonitor::new(1000).on_overflow(move |level, usage, max| {
            if let Ok(mut v) = fired_clone.lock() {
                v.push((level, usage, max));
            }
        });

        // Normal -> no callback
        monitor.update(500);
        // Normal -> Warning -> callback fires
        monitor.update(750);
        // Warning -> Warning -> no callback (same level)
        monitor.update(760);
        // Warning -> Critical -> callback fires
        monitor.update(870);

        let events = fired.lock().expect("lock should not be poisoned");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, OverflowLevel::Warning);
        assert_eq!(events[0].1, 750);
        assert_eq!(events[0].2, 1000);
        assert_eq!(events[1].0, OverflowLevel::Critical);
    }

    #[test]
    fn test_monitor_remaining_tokens() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        monitor.update(300);
        assert_eq!(monitor.remaining_tokens(), 700);
        monitor.update(1200); // over budget
        assert_eq!(monitor.remaining_tokens(), 0);
    }

    #[test]
    fn test_monitor_usage_fraction() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        monitor.update(500);
        assert!((monitor.usage_fraction() - 0.5).abs() < f64::EPSILON);

        // Zero max_tokens edge case
        let zero_monitor = ContextOverflowMonitor::new(0);
        assert!((zero_monitor.usage_fraction() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_recommended_action() {
        let mut monitor = ContextOverflowMonitor::new(1000);

        monitor.update(100);
        assert_eq!(monitor.recommended_action(), "No action needed");

        monitor.update(750);
        assert_eq!(
            monitor.recommended_action(),
            "Consider summarizing older messages"
        );

        monitor.update(870);
        assert_eq!(
            monitor.recommended_action(),
            "Evict low-priority messages immediately"
        );

        monitor.update(960);
        assert_eq!(
            monitor.recommended_action(),
            "Aggressive eviction required to prevent data loss"
        );
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = ContextOverflowMonitor::new(1000);
        monitor.update(960);
        assert_eq!(monitor.check_level(), OverflowLevel::Emergency);
        assert!(monitor.overflow_count() > 0);

        monitor.reset();
        assert_eq!(monitor.check_level(), OverflowLevel::Normal);
        assert_eq!(monitor.current_usage(), 0);
        assert_eq!(monitor.overflow_count(), 0);
    }

    #[test]
    fn test_monitor_overflow_count() {
        let mut monitor = ContextOverflowMonitor::new(1000);

        // Normal -> Warning (count 1)
        monitor.update(750);
        assert_eq!(monitor.overflow_count(), 1);

        // Warning -> Warning (no increase, still 1)
        monitor.update(760);
        assert_eq!(monitor.overflow_count(), 1);

        // Warning -> Critical (count 2)
        monitor.update(870);
        assert_eq!(monitor.overflow_count(), 2);

        // Critical -> Emergency (count 3)
        monitor.update(960);
        assert_eq!(monitor.overflow_count(), 3);

        // Emergency -> Warning (decrease, no increment, still 3)
        monitor.update(750);
        assert_eq!(monitor.overflow_count(), 3);
    }

    #[test]
    fn test_monitor_custom_thresholds() {
        let thresholds = OverflowThresholds {
            warning: 0.50,
            critical: 0.75,
            emergency: 0.90,
        };
        let mut monitor = ContextOverflowMonitor::new(1000).with_thresholds(thresholds);

        // 55% should be warning with the custom 0.50 threshold
        let level = monitor.update(550);
        assert_eq!(level, OverflowLevel::Warning);

        // 80% should be critical with the custom 0.75 threshold
        let level = monitor.update(800);
        assert_eq!(level, OverflowLevel::Critical);

        // 92% should be emergency with the custom 0.90 threshold
        let level = monitor.update(920);
        assert_eq!(level, OverflowLevel::Emergency);
    }

    #[test]
    fn test_monitor_from_auto_config() {
        let cfg = AutoTokenConfig::from_model(16_000);
        let monitor = ContextOverflowMonitor::from_auto_config(&cfg);
        // 16000 - 1024 (default reserve) = 14976
        assert_eq!(monitor.max_tokens(), 14_976);
        assert_eq!(monitor.check_level(), OverflowLevel::Normal);
    }

    #[test]
    fn test_context_message_uses_unified_estimate() {
        // ContextMessage should use the same estimate as context::estimate_tokens
        let text = "This is a test message for verifying unified token estimation.";
        let msg = ContextMessage::new("user", text);
        assert_eq!(msg.token_count, crate::context::estimate_tokens(text));
    }
}
