//! Conversation compaction
//!
//! Intelligent compaction of long conversations to preserve context.

use std::collections::HashMap;

/// Compaction configuration
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Maximum messages before compaction
    pub max_messages: usize,
    /// Target messages after compaction
    pub target_messages: usize,
    /// Preserve recent messages count
    pub preserve_recent: usize,
    /// Preserve first messages count
    pub preserve_first: usize,
    /// Minimum importance score to preserve
    pub min_importance: f64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_messages: 50,
            target_messages: 20,
            preserve_recent: 10,
            preserve_first: 2,
            min_importance: 0.8,
        }
    }
}

/// Message for compaction
#[derive(Debug, Clone)]
pub struct CompactableMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub importance: f64,
    pub topics: Vec<String>,
    pub entities: Vec<String>,
}

impl CompactableMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            importance: 0.5,
            topics: Vec::new(),
            entities: Vec::new(),
        }
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }
}

/// Compaction result
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Compacted messages
    pub messages: Vec<CompactableMessage>,
    /// Summary of removed messages
    pub summary: Option<String>,
    /// Number of messages removed
    pub removed_count: usize,
    /// Topics preserved
    pub preserved_topics: Vec<String>,
    /// Entities preserved
    pub preserved_entities: Vec<String>,
}

/// Conversation compactor
pub struct ConversationCompactor {
    config: CompactionConfig,
}

impl ConversationCompactor {
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self, message_count: usize) -> bool {
        message_count > self.config.max_messages
    }

    /// Compact a conversation
    pub fn compact(&self, messages: Vec<CompactableMessage>) -> CompactionResult {
        if messages.len() <= self.config.target_messages {
            return CompactionResult {
                messages,
                summary: None,
                removed_count: 0,
                preserved_topics: Vec::new(),
                preserved_entities: Vec::new(),
            };
        }

        let mut preserved = Vec::new();
        let mut removed = Vec::new();
        let len = messages.len();

        // Collect all topics and entities first
        let mut all_topics: HashMap<String, usize> = HashMap::new();
        let mut all_entities: HashMap<String, usize> = HashMap::new();

        for msg in &messages {
            for topic in &msg.topics {
                *all_topics.entry(topic.clone()).or_insert(0) += 1;
            }
            for entity in &msg.entities {
                *all_entities.entry(entity.clone()).or_insert(0) += 1;
            }
        }

        for (i, msg) in messages.into_iter().enumerate() {
            let should_preserve =
                // Preserve first N messages
                i < self.config.preserve_first ||
                // Preserve last N messages
                i >= len.saturating_sub(self.config.preserve_recent) ||
                // Preserve high importance messages
                msg.importance >= self.config.min_importance ||
                // Preserve messages with unique important topics
                self.has_unique_important_topic(&msg, &all_topics);

            if should_preserve {
                preserved.push(msg);
            } else {
                removed.push(msg);
            }
        }

        // If still too many, remove lowest importance from middle
        while preserved.len() > self.config.target_messages {
            let mid_start = self.config.preserve_first;
            let mid_end = preserved.len().saturating_sub(self.config.preserve_recent);

            if mid_start >= mid_end {
                break;
            }

            // Find lowest importance in middle section
            let min_idx = preserved[mid_start..mid_end]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i + mid_start);

            if let Some(idx) = min_idx {
                removed.push(preserved.remove(idx));
            } else {
                break;
            }
        }

        // Generate summary of removed messages
        let summary = self.generate_summary(&removed);

        // Collect preserved topics and entities
        let preserved_topics: Vec<String> = all_topics.into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(topic, _)| topic)
            .collect();

        let preserved_entities: Vec<String> = all_entities.into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(entity, _)| entity)
            .collect();

        CompactionResult {
            messages: preserved,
            summary: Some(summary),
            removed_count: removed.len(),
            preserved_topics,
            preserved_entities,
        }
    }

    fn has_unique_important_topic(&self, msg: &CompactableMessage, all_topics: &HashMap<String, usize>) -> bool {
        // Check if message contains a topic that only appears once or twice
        for topic in &msg.topics {
            if let Some(&count) = all_topics.get(topic) {
                if count <= 2 {
                    return true;
                }
            }
        }
        false
    }

    fn generate_summary(&self, removed: &[CompactableMessage]) -> String {
        if removed.is_empty() {
            return String::new();
        }

        let mut summary_parts = Vec::new();

        // Group by role
        let user_count = removed.iter().filter(|m| m.role == "user").count();
        let assistant_count = removed.iter().filter(|m| m.role == "assistant").count();

        if user_count > 0 {
            summary_parts.push(format!("{} user messages", user_count));
        }
        if assistant_count > 0 {
            summary_parts.push(format!("{} assistant responses", assistant_count));
        }

        // Collect unique topics
        let topics: Vec<_> = removed.iter()
            .flat_map(|m| m.topics.iter())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(5)
            .collect();

        if !topics.is_empty() {
            summary_parts.push(format!("discussing: {}", topics.into_iter().cloned().collect::<Vec<_>>().join(", ")));
        }

        format!("[Earlier conversation: {}]", summary_parts.join(", "))
    }

    /// Create a summary message from compaction
    pub fn create_summary_message(&self, result: &CompactionResult) -> Option<CompactableMessage> {
        result.summary.as_ref().map(|summary| {
            CompactableMessage::new("system", summary)
                .with_importance(0.7)
        })
    }
}

impl Default for ConversationCompactor {
    fn default() -> Self {
        Self::new(CompactionConfig::default())
    }
}

/// Builder for compaction config
pub struct CompactionConfigBuilder {
    config: CompactionConfig,
}

impl CompactionConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: CompactionConfig::default(),
        }
    }

    pub fn max_messages(mut self, count: usize) -> Self {
        self.config.max_messages = count;
        self
    }

    pub fn target_messages(mut self, count: usize) -> Self {
        self.config.target_messages = count;
        self
    }

    pub fn preserve_recent(mut self, count: usize) -> Self {
        self.config.preserve_recent = count;
        self
    }

    pub fn preserve_first(mut self, count: usize) -> Self {
        self.config.preserve_first = count;
        self
    }

    pub fn min_importance(mut self, importance: f64) -> Self {
        self.config.min_importance = importance;
        self
    }

    pub fn build(self) -> ConversationCompactor {
        ConversationCompactor::new(self.config)
    }
}

impl Default for CompactionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compaction_needed() {
        let compactor = ConversationCompactor::default();
        let messages: Vec<_> = (0..10)
            .map(|i| CompactableMessage::new("user", &format!("Message {}", i)))
            .collect();

        let result = compactor.compact(messages);
        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn test_compaction() {
        let config = CompactionConfig {
            max_messages: 10,
            target_messages: 5,
            preserve_recent: 2,
            preserve_first: 1,
            min_importance: 0.9,
        };
        let compactor = ConversationCompactor::new(config);

        let messages: Vec<_> = (0..20)
            .map(|i| CompactableMessage::new("user", &format!("Message {}", i)))
            .collect();

        let result = compactor.compact(messages);
        assert!(result.messages.len() <= 5);
        assert!(result.removed_count > 0);
    }

    #[test]
    fn test_preserve_important() {
        let config = CompactionConfig {
            max_messages: 5,
            target_messages: 3,
            preserve_recent: 1,
            preserve_first: 1,
            min_importance: 0.8,
        };
        let compactor = ConversationCompactor::new(config);

        let mut messages: Vec<_> = (0..10)
            .map(|i| CompactableMessage::new("user", &format!("Message {}", i)))
            .collect();

        // Mark one as important
        messages[5] = messages[5].clone().with_importance(0.9);

        let result = compactor.compact(messages);

        // Important message should be preserved
        assert!(result.messages.iter().any(|m| m.importance >= 0.8));
    }
}
