//! Audit logging for tracking all AI assistant operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Type of audit event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AuditEventType {
    /// Message sent to AI
    MessageSent,
    /// Response received from AI
    ResponseReceived,
    /// Response cancelled/aborted
    ResponseCancelled,
    /// Response regenerated
    ResponseRegenerated,
    /// Message edited
    MessageEdited,
    /// Session created
    SessionCreated,
    /// Session loaded
    SessionLoaded,
    /// Session deleted
    SessionDeleted,
    /// Knowledge document indexed
    DocumentIndexed,
    /// Knowledge document deleted
    DocumentDeleted,
    /// Rate limit hit
    RateLimitHit,
    /// Input sanitized
    InputSanitized,
    /// Configuration changed
    ConfigChanged,
    /// Error occurred
    Error,
}

/// An audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event ID (auto-incremented)
    pub id: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: AuditEventType,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Session ID (if applicable)
    pub session_id: Option<String>,
    /// Event details
    pub details: HashMap<String, String>,
    /// Success status
    pub success: bool,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl AuditEvent {
    pub fn new(event_type: AuditEventType) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            timestamp: Utc::now(),
            event_type,
            user_id: None,
            session_id: None,
            details: HashMap::new(),
            success: true,
            error: None,
        }
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_error(mut self, error: &str) -> Self {
        self.success = false;
        self.error = Some(error.to_string());
        self
    }
}

/// Configuration for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Maximum events to keep in memory
    pub max_events: usize,
    /// Log message content (privacy consideration)
    pub log_message_content: bool,
    /// Event types to log (empty = all)
    pub event_filter: Vec<AuditEventType>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events: 1000,
            log_message_content: false, // Privacy by default
            event_filter: Vec::new(),   // Log all
        }
    }
}

/// Audit logger for tracking all AI assistant operations
pub struct AuditLogger {
    config: AuditConfig,
    events: VecDeque<AuditEvent>,
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
        }
    }

    /// Log an audit event
    pub fn log(&mut self, event: AuditEvent) {
        if !self.config.enabled {
            return;
        }

        // Check filter
        if !self.config.event_filter.is_empty()
            && !self.config.event_filter.contains(&event.event_type)
        {
            return;
        }

        // Add event
        self.events.push_back(event);

        // Trim if over limit
        while self.events.len() > self.config.max_events {
            self.events.pop_front();
        }
    }

    /// Log a simple event
    pub fn log_simple(&mut self, event_type: AuditEventType) {
        self.log(AuditEvent::new(event_type));
    }

    /// Get all events
    pub fn get_events(&self) -> &VecDeque<AuditEvent> {
        &self.events
    }

    /// Get events of a specific type
    pub fn get_events_by_type(&self, event_type: AuditEventType) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get events for a specific session
    pub fn get_session_events(&self, session_id: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.session_id.as_deref() == Some(session_id))
            .collect()
    }

    /// Get events in a time range
    pub fn get_events_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect()
    }

    /// Get recent events (last N)
    pub fn get_recent(&self, count: usize) -> Vec<&AuditEvent> {
        self.events.iter().rev().take(count).collect()
    }

    /// Get error events
    pub fn get_errors(&self) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| !e.success).collect()
    }

    /// Get event count by type
    pub fn count_by_type(&self) -> HashMap<AuditEventType, usize> {
        let mut counts = HashMap::new();
        for event in &self.events {
            *counts.entry(event.event_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Export events as JSON
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.events.iter().collect::<Vec<_>>())
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Get statistics
    pub fn get_stats(&self) -> AuditStats {
        let total = self.events.len();
        let errors = self.events.iter().filter(|e| !e.success).count();
        let by_type = self.count_by_type();

        let messages_sent = by_type
            .get(&AuditEventType::MessageSent)
            .copied()
            .unwrap_or(0);
        let responses = by_type
            .get(&AuditEventType::ResponseReceived)
            .copied()
            .unwrap_or(0);

        AuditStats {
            total_events: total,
            error_count: errors,
            messages_sent,
            responses_received: responses,
            events_by_type: by_type,
        }
    }
}

/// Statistics from audit log
#[derive(Debug, Clone)]
pub struct AuditStats {
    pub total_events: usize,
    pub error_count: usize,
    pub messages_sent: usize,
    pub responses_received: usize,
    pub events_by_type: HashMap<AuditEventType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_logger() {
        let config = AuditConfig::default();
        let mut logger = AuditLogger::new(config);

        logger.log(
            AuditEvent::new(AuditEventType::MessageSent)
                .with_user("test_user")
                .with_detail("content_length", "100"),
        );

        logger.log(AuditEvent::new(AuditEventType::ResponseReceived).with_user("test_user"));

        let stats = logger.get_stats();
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.messages_sent, 1);
    }

    #[test]
    fn test_event_filtering() {
        let config = AuditConfig {
            enabled: true,
            max_events: 100,
            log_message_content: false,
            event_filter: vec![AuditEventType::Error, AuditEventType::RateLimitHit],
        };
        let mut logger = AuditLogger::new(config);

        // Log events of various types
        logger.log(AuditEvent::new(AuditEventType::MessageSent));
        logger.log(AuditEvent::new(AuditEventType::Error).with_error("something failed"));
        logger.log(AuditEvent::new(AuditEventType::RateLimitHit));
        logger.log(AuditEvent::new(AuditEventType::SessionCreated));

        // Only Error and RateLimitHit should have been stored
        assert_eq!(logger.get_events().len(), 2);
        assert_eq!(logger.get_events_by_type(AuditEventType::Error).len(), 1);
        assert_eq!(
            logger
                .get_events_by_type(AuditEventType::RateLimitHit)
                .len(),
            1
        );
        assert_eq!(
            logger.get_events_by_type(AuditEventType::MessageSent).len(),
            0
        );
    }

    #[test]
    fn test_circular_buffer_eviction() {
        let config = AuditConfig {
            enabled: true,
            max_events: 3,
            log_message_content: false,
            event_filter: Vec::new(),
        };
        let mut logger = AuditLogger::new(config);

        // Log 5 events
        for _ in 0..5 {
            logger.log_simple(AuditEventType::MessageSent);
        }

        // Only 3 should remain (the most recent 3)
        assert_eq!(logger.get_events().len(), 3);
    }

    #[test]
    fn test_stats_aggregation() {
        let config = AuditConfig::default();
        let mut logger = AuditLogger::new(config);

        logger.log_simple(AuditEventType::MessageSent);
        logger.log_simple(AuditEventType::MessageSent);
        logger.log_simple(AuditEventType::ResponseReceived);
        logger.log(AuditEvent::new(AuditEventType::Error).with_error("oops"));
        logger.log(AuditEvent::new(AuditEventType::Error).with_error("again"));

        let stats = logger.get_stats();
        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.error_count, 2);
        assert_eq!(stats.messages_sent, 2);
        assert_eq!(stats.responses_received, 1);
        assert_eq!(
            *stats.events_by_type.get(&AuditEventType::Error).unwrap(),
            2
        );
        assert_eq!(
            *stats
                .events_by_type
                .get(&AuditEventType::MessageSent)
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_disabled_logger() {
        let config = AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let mut logger = AuditLogger::new(config);
        logger.log_simple(AuditEventType::MessageSent);
        logger.log_simple(AuditEventType::Error);
        assert_eq!(logger.get_events().len(), 0);
    }

    #[test]
    fn test_session_events_filtering() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        logger.log(AuditEvent::new(AuditEventType::MessageSent).with_session("s1"));
        logger.log(AuditEvent::new(AuditEventType::ResponseReceived).with_session("s2"));
        logger.log(AuditEvent::new(AuditEventType::MessageSent).with_session("s1"));
        assert_eq!(logger.get_session_events("s1").len(), 2);
        assert_eq!(logger.get_session_events("s2").len(), 1);
        assert_eq!(logger.get_session_events("s3").len(), 0);
    }

    #[test]
    fn test_error_events_filtering() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        logger.log_simple(AuditEventType::MessageSent);
        logger.log(AuditEvent::new(AuditEventType::Error).with_error("fail1"));
        logger.log_simple(AuditEventType::ResponseReceived);
        logger.log(AuditEvent::new(AuditEventType::SessionDeleted).with_error("fail2"));
        let errors = logger.get_errors();
        assert_eq!(errors.len(), 2);
        assert!(!errors[0].success);
        assert!(!errors[1].success);
    }

    #[test]
    fn test_recent_events() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        for _ in 0..10 {
            logger.log_simple(AuditEventType::MessageSent);
        }
        let recent3 = logger.get_recent(3);
        assert_eq!(recent3.len(), 3);
        let recent20 = logger.get_recent(20);
        assert_eq!(recent20.len(), 10);
    }

    #[test]
    fn test_clear_and_export_json() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        logger.log(AuditEvent::new(AuditEventType::SessionCreated).with_user("u1"));
        logger.log_simple(AuditEventType::MessageSent);
        assert_eq!(logger.get_events().len(), 2);

        let json = logger.export_json();
        assert!(json.starts_with('['));
        assert!(json.contains("SessionCreated"));

        logger.clear();
        assert_eq!(logger.get_events().len(), 0);
        assert_eq!(logger.export_json(), "[]");
    }

    #[test]
    fn test_get_events_by_type() {
        let config = AuditConfig::default();
        let mut logger = AuditLogger::new(config);

        logger.log(AuditEvent::new(AuditEventType::SessionCreated).with_session("s1"));
        logger.log(AuditEvent::new(AuditEventType::MessageSent).with_session("s1"));
        logger.log(AuditEvent::new(AuditEventType::SessionCreated).with_session("s2"));
        logger.log(AuditEvent::new(AuditEventType::ResponseReceived).with_session("s1"));

        let created = logger.get_events_by_type(AuditEventType::SessionCreated);
        assert_eq!(created.len(), 2);
        // Verify the sessions match
        assert_eq!(created[0].session_id.as_deref(), Some("s1"));
        assert_eq!(created[1].session_id.as_deref(), Some("s2"));

        let sent = logger.get_events_by_type(AuditEventType::MessageSent);
        assert_eq!(sent.len(), 1);

        let deleted = logger.get_events_by_type(AuditEventType::SessionDeleted);
        assert_eq!(deleted.len(), 0);
    }
}
