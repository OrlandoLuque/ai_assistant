//! Conversation analytics for AI interactions
//!
//! This module provides analytics and insights for AI conversations,
//! helping understand usage patterns and improve interactions.
//!
//! # Features
//!
//! - **Usage tracking**: Track conversations, messages, tokens
//! - **Pattern analysis**: Identify common query patterns
//! - **Quality metrics**: Track response quality over time
//! - **User insights**: Understand user behavior

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Configuration for analytics
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Enable detailed tracking
    pub detailed_tracking: bool,
    /// Track user patterns
    pub track_patterns: bool,
    /// Track quality metrics
    pub track_quality: bool,
    /// Maximum events to store
    pub max_events: usize,
    /// Aggregation interval
    pub aggregation_interval: Duration,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            detailed_tracking: true,
            track_patterns: true,
            track_quality: true,
            max_events: 10000,
            aggregation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// An analytics event
#[derive(Debug, Clone)]
pub struct AnalyticsEvent {
    /// Event type
    pub event_type: EventType,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Session ID
    pub session_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Model used
    pub model: Option<String>,
    /// Event data
    pub data: HashMap<String, EventValue>,
}

/// Event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Conversation started
    ConversationStart,
    /// Message sent
    MessageSent,
    /// Response received
    ResponseReceived,
    /// Error occurred
    Error,
    /// Conversation ended
    ConversationEnd,
    /// Model changed
    ModelChanged,
    /// Rating/feedback given
    Feedback,
    /// Custom event
    Custom,
}

/// Event value types
#[derive(Debug, Clone)]
pub enum EventValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Duration(Duration),
}

impl EventValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            EventValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            EventValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            EventValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// Conversation analytics tracker
pub struct ConversationAnalytics {
    config: AnalyticsConfig,
    /// All events
    events: Vec<AnalyticsEvent>,
    /// Aggregated stats
    aggregated: AggregatedStats,
    /// Pattern tracker
    patterns: PatternTracker,
}

/// Aggregated statistics
#[derive(Debug, Clone, Default)]
pub struct AggregatedStats {
    /// Total conversations
    pub total_conversations: usize,
    /// Total messages
    pub total_messages: usize,
    /// Total tokens
    pub total_tokens: u64,
    /// Total errors
    pub total_errors: usize,
    /// Average response time
    pub avg_response_time: Duration,
    /// Average conversation length
    pub avg_conversation_length: f64,
    /// Messages per conversation
    pub avg_messages_per_conversation: f64,
    /// Model usage
    pub model_usage: HashMap<String, usize>,
    /// Hourly message counts
    pub hourly_messages: HashMap<u8, usize>,
    /// Error rate
    pub error_rate: f64,
}

/// Pattern tracker
#[derive(Debug, Clone, Default)]
pub struct PatternTracker {
    /// Common query patterns
    pub query_patterns: HashMap<String, usize>,
    /// Common topics
    pub topics: HashMap<String, usize>,
    /// Query length distribution
    pub query_lengths: Vec<usize>,
    /// Response satisfaction (if feedback available)
    pub satisfaction_scores: Vec<f64>,
}

impl ConversationAnalytics {
    /// Create new analytics tracker
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
            aggregated: AggregatedStats::default(),
            patterns: PatternTracker::default(),
        }
    }

    /// Track an event
    pub fn track(&mut self, event: AnalyticsEvent) {
        // Update aggregated stats
        self.update_aggregated(&event);

        // Update patterns
        if self.config.track_patterns {
            self.update_patterns(&event);
        }

        // Store event
        self.events.push(event);

        // Trim if over limit
        while self.events.len() > self.config.max_events {
            self.events.remove(0);
        }
    }

    /// Track conversation start
    pub fn track_conversation_start(&mut self, session_id: &str, user_id: Option<&str>, model: &str) {
        let mut data = HashMap::new();
        data.insert("model".to_string(), EventValue::String(model.to_string()));

        self.track(AnalyticsEvent {
            event_type: EventType::ConversationStart,
            timestamp: SystemTime::now(),
            session_id: Some(session_id.to_string()),
            user_id: user_id.map(|s| s.to_string()),
            model: Some(model.to_string()),
            data,
        });
    }

    /// Track message sent
    pub fn track_message(
        &mut self,
        session_id: &str,
        user_id: Option<&str>,
        model: &str,
        message: &str,
        is_user: bool,
        tokens: u64,
        response_time: Option<Duration>,
    ) {
        let mut data = HashMap::new();
        data.insert("tokens".to_string(), EventValue::Int(tokens as i64));
        data.insert("is_user".to_string(), EventValue::Bool(is_user));
        data.insert("message_length".to_string(), EventValue::Int(message.len() as i64));

        if let Some(rt) = response_time {
            data.insert("response_time".to_string(), EventValue::Duration(rt));
        }

        self.track(AnalyticsEvent {
            event_type: if is_user { EventType::MessageSent } else { EventType::ResponseReceived },
            timestamp: SystemTime::now(),
            session_id: Some(session_id.to_string()),
            user_id: user_id.map(|s| s.to_string()),
            model: Some(model.to_string()),
            data,
        });
    }

    /// Track error
    pub fn track_error(&mut self, session_id: Option<&str>, model: Option<&str>, error: &str) {
        let mut data = HashMap::new();
        data.insert("error".to_string(), EventValue::String(error.to_string()));

        self.track(AnalyticsEvent {
            event_type: EventType::Error,
            timestamp: SystemTime::now(),
            session_id: session_id.map(|s| s.to_string()),
            user_id: None,
            model: model.map(|s| s.to_string()),
            data,
        });
    }

    /// Track feedback
    pub fn track_feedback(&mut self, session_id: &str, rating: f64, comment: Option<&str>) {
        let mut data = HashMap::new();
        data.insert("rating".to_string(), EventValue::Float(rating));
        if let Some(c) = comment {
            data.insert("comment".to_string(), EventValue::String(c.to_string()));
        }

        self.track(AnalyticsEvent {
            event_type: EventType::Feedback,
            timestamp: SystemTime::now(),
            session_id: Some(session_id.to_string()),
            user_id: None,
            model: None,
            data,
        });
    }

    fn update_aggregated(&mut self, event: &AnalyticsEvent) {
        match event.event_type {
            EventType::ConversationStart => {
                self.aggregated.total_conversations += 1;
            }
            EventType::MessageSent | EventType::ResponseReceived => {
                self.aggregated.total_messages += 1;

                if let Some(EventValue::Int(tokens)) = event.data.get("tokens") {
                    self.aggregated.total_tokens += *tokens as u64;
                }

                if let Some(EventValue::Duration(rt)) = event.data.get("response_time") {
                    // Update average response time (simplified)
                    let n = self.aggregated.total_messages as u32;
                    self.aggregated.avg_response_time =
                        (self.aggregated.avg_response_time * (n - 1) + *rt) / n;
                }

                if let Some(model) = &event.model {
                    *self.aggregated.model_usage.entry(model.clone()).or_insert(0) += 1;
                }
            }
            EventType::Error => {
                self.aggregated.total_errors += 1;
                self.aggregated.error_rate =
                    self.aggregated.total_errors as f64 / self.aggregated.total_messages.max(1) as f64;
            }
            _ => {}
        }

        // Update averages
        if self.aggregated.total_conversations > 0 {
            self.aggregated.avg_messages_per_conversation =
                self.aggregated.total_messages as f64 / self.aggregated.total_conversations as f64;
        }
    }

    fn update_patterns(&mut self, event: &AnalyticsEvent) {
        if event.event_type == EventType::MessageSent {
            if let Some(EventValue::Int(len)) = event.data.get("message_length") {
                self.patterns.query_lengths.push(*len as usize);
            }
        }

        if event.event_type == EventType::Feedback {
            if let Some(EventValue::Float(rating)) = event.data.get("rating") {
                self.patterns.satisfaction_scores.push(*rating);
            }
        }
    }

    /// Get aggregated stats
    pub fn stats(&self) -> &AggregatedStats {
        &self.aggregated
    }

    /// Get pattern insights
    pub fn patterns(&self) -> &PatternTracker {
        &self.patterns
    }

    /// Generate analytics report
    pub fn report(&self) -> AnalyticsReport {
        let avg_satisfaction = if self.patterns.satisfaction_scores.is_empty() {
            None
        } else {
            Some(
                self.patterns.satisfaction_scores.iter().sum::<f64>()
                    / self.patterns.satisfaction_scores.len() as f64,
            )
        };

        let avg_query_length = if self.patterns.query_lengths.is_empty() {
            0.0
        } else {
            self.patterns.query_lengths.iter().sum::<usize>() as f64
                / self.patterns.query_lengths.len() as f64
        };

        let top_models: Vec<_> = {
            let mut models: Vec<_> = self.aggregated.model_usage.iter().collect();
            models.sort_by(|a, b| b.1.cmp(a.1));
            models.into_iter().take(5).map(|(k, v)| (k.clone(), *v)).collect()
        };

        AnalyticsReport {
            total_conversations: self.aggregated.total_conversations,
            total_messages: self.aggregated.total_messages,
            total_tokens: self.aggregated.total_tokens,
            avg_messages_per_conversation: self.aggregated.avg_messages_per_conversation,
            avg_response_time: self.aggregated.avg_response_time,
            error_rate: self.aggregated.error_rate,
            avg_satisfaction,
            avg_query_length,
            top_models,
            events_tracked: self.events.len(),
        }
    }

    /// Export events as JSON-like structure
    pub fn export_events(&self) -> Vec<ExportedEvent> {
        self.events.iter().map(|e| ExportedEvent {
            event_type: format!("{:?}", e.event_type),
            timestamp: e.timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            session_id: e.session_id.clone(),
            model: e.model.clone(),
        }).collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.events.clear();
        self.aggregated = AggregatedStats::default();
        self.patterns = PatternTracker::default();
    }
}

impl Default for ConversationAnalytics {
    fn default() -> Self {
        Self::new(AnalyticsConfig::default())
    }
}

/// Analytics report
#[derive(Debug, Clone)]
pub struct AnalyticsReport {
    /// Total conversations
    pub total_conversations: usize,
    /// Total messages
    pub total_messages: usize,
    /// Total tokens
    pub total_tokens: u64,
    /// Average messages per conversation
    pub avg_messages_per_conversation: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Error rate
    pub error_rate: f64,
    /// Average satisfaction score
    pub avg_satisfaction: Option<f64>,
    /// Average query length
    pub avg_query_length: f64,
    /// Top models by usage
    pub top_models: Vec<(String, usize)>,
    /// Events tracked
    pub events_tracked: usize,
}

/// Exported event (simplified)
#[derive(Debug, Clone)]
pub struct ExportedEvent {
    pub event_type: String,
    pub timestamp: u64,
    pub session_id: Option<String>,
    pub model: Option<String>,
}

/// Builder for analytics config
pub struct AnalyticsConfigBuilder {
    config: AnalyticsConfig,
}

impl AnalyticsConfigBuilder {
    pub fn new() -> Self {
        Self { config: AnalyticsConfig::default() }
    }

    pub fn detailed_tracking(mut self, enabled: bool) -> Self {
        self.config.detailed_tracking = enabled;
        self
    }

    pub fn track_patterns(mut self, enabled: bool) -> Self {
        self.config.track_patterns = enabled;
        self
    }

    pub fn track_quality(mut self, enabled: bool) -> Self {
        self.config.track_quality = enabled;
        self
    }

    pub fn max_events(mut self, max: usize) -> Self {
        self.config.max_events = max;
        self
    }

    pub fn build(self) -> AnalyticsConfig {
        self.config
    }
}

impl Default for AnalyticsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_creation() {
        let analytics = ConversationAnalytics::default();
        assert_eq!(analytics.stats().total_conversations, 0);
    }

    #[test]
    fn test_track_conversation() {
        let mut analytics = ConversationAnalytics::default();

        analytics.track_conversation_start("session1", Some("user1"), "gpt-4");
        analytics.track_message(
            "session1",
            Some("user1"),
            "gpt-4",
            "Hello",
            true,
            10,
            None,
        );
        analytics.track_message(
            "session1",
            Some("user1"),
            "gpt-4",
            "Hi there!",
            false,
            20,
            Some(Duration::from_millis(500)),
        );

        assert_eq!(analytics.stats().total_conversations, 1);
        assert_eq!(analytics.stats().total_messages, 2);
        assert_eq!(analytics.stats().total_tokens, 30);
    }

    #[test]
    fn test_error_tracking() {
        let mut analytics = ConversationAnalytics::default();

        analytics.track_message("s1", None, "model", "msg", true, 10, None);
        analytics.track_error(Some("s1"), Some("model"), "Connection failed");

        assert_eq!(analytics.stats().total_errors, 1);
        assert!(analytics.stats().error_rate > 0.0);
    }

    #[test]
    fn test_feedback_tracking() {
        let mut analytics = ConversationAnalytics::default();

        analytics.track_feedback("session1", 0.9, Some("Great response!"));
        analytics.track_feedback("session1", 0.8, None);

        assert_eq!(analytics.patterns().satisfaction_scores.len(), 2);
    }

    #[test]
    fn test_report_generation() {
        let mut analytics = ConversationAnalytics::default();

        analytics.track_conversation_start("s1", None, "gpt-4");
        analytics.track_message("s1", None, "gpt-4", "Hello", true, 10, None);
        analytics.track_message(
            "s1",
            None,
            "gpt-4",
            "Hi!",
            false,
            5,
            Some(Duration::from_millis(100)),
        );

        let report = analytics.report();
        assert_eq!(report.total_conversations, 1);
        assert_eq!(report.total_messages, 2);
    }
}
