//! User engagement metrics
//!
//! Track and analyze user engagement patterns.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Engagement event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EngagementEvent {
    MessageSent,
    MessageReceived,
    QuestionAsked,
    FollowUpQuestion,
    TopicChange,
    ClarificationRequest,
    PositiveFeedback,
    NegativeFeedback,
    SessionStart,
    SessionEnd,
    LongPause,
    QuickResponse,
}

/// User engagement metrics
#[derive(Debug, Clone)]
pub struct EngagementMetrics {
    pub session_duration: Duration,
    pub message_count: usize,
    pub avg_response_time: Duration,
    pub question_ratio: f64,
    pub follow_up_ratio: f64,
    pub topic_changes: usize,
    pub engagement_score: f64,
    pub sentiment_trend: f64,
    pub depth_score: f64,
}

/// Engagement event record
#[derive(Debug, Clone)]
struct EngagementRecord {
    event: EngagementEvent,
    timestamp: Instant,
    metadata: HashMap<String, String>,
}


/// User engagement tracker
pub struct EngagementTracker {
    user_id: String,
    events: Vec<EngagementRecord>,
    session_start: Instant,
    last_activity: Instant,
    message_times: Vec<Duration>,
    topics: Vec<String>,
    feedback_scores: Vec<i32>,
}

impl EngagementTracker {
    pub fn new(user_id: &str) -> Self {
        let now = Instant::now();
        Self {
            user_id: user_id.to_string(),
            events: vec![EngagementRecord {
                event: EngagementEvent::SessionStart,
                timestamp: now,
                metadata: HashMap::new(),
            }],
            session_start: now,
            last_activity: now,
            message_times: Vec::new(),
            topics: Vec::new(),
            feedback_scores: Vec::new(),
        }
    }

    /// Get the user ID for this tracker.
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Get the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get the elapsed time since the most recent event.
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.events.last().map(|r| r.timestamp.elapsed())
    }

    /// Get metadata from the most recent event.
    pub fn last_event_metadata(&self) -> Option<&HashMap<String, String>> {
        self.events.last().map(|r| &r.metadata)
    }

    pub fn record_event(&mut self, event: EngagementEvent) {
        self.record_event_with_metadata(event, HashMap::new());
    }

    pub fn record_event_with_metadata(
        &mut self,
        event: EngagementEvent,
        metadata: HashMap<String, String>,
    ) {
        let now = Instant::now();
        let time_since_last = now.duration_since(self.last_activity);

        // Track response times
        if matches!(
            event,
            EngagementEvent::MessageSent | EngagementEvent::MessageReceived
        ) {
            self.message_times.push(time_since_last);
        }

        // Track long pauses
        if time_since_last > Duration::from_secs(60) {
            self.events.push(EngagementRecord {
                event: EngagementEvent::LongPause,
                timestamp: now,
                metadata: HashMap::new(),
            });
        } else if time_since_last < Duration::from_secs(5) && event == EngagementEvent::MessageSent
        {
            self.events.push(EngagementRecord {
                event: EngagementEvent::QuickResponse,
                timestamp: now,
                metadata: HashMap::new(),
            });
        }

        self.events.push(EngagementRecord {
            event,
            timestamp: now,
            metadata,
        });

        self.last_activity = now;
    }

    pub fn record_topic(&mut self, topic: &str) {
        if self.topics.last().map(|t| t != topic).unwrap_or(true) {
            self.topics.push(topic.to_string());
            self.record_event(EngagementEvent::TopicChange);
        }
    }

    pub fn record_feedback(&mut self, positive: bool) {
        self.feedback_scores.push(if positive { 1 } else { -1 });
        self.record_event(if positive {
            EngagementEvent::PositiveFeedback
        } else {
            EngagementEvent::NegativeFeedback
        });
    }

    pub fn calculate_metrics(&self) -> EngagementMetrics {
        let session_duration = self.last_activity.duration_since(self.session_start);

        let message_count = self
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e.event,
                    EngagementEvent::MessageSent | EngagementEvent::MessageReceived
                )
            })
            .count();

        let avg_response_time = if self.message_times.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.message_times.iter().sum();
            total / self.message_times.len() as u32
        };

        let questions = self.count_event(EngagementEvent::QuestionAsked);
        let question_ratio = if message_count > 0 {
            questions as f64 / message_count as f64
        } else {
            0.0
        };

        let follow_ups = self.count_event(EngagementEvent::FollowUpQuestion);
        let follow_up_ratio = if questions > 0 {
            follow_ups as f64 / questions as f64
        } else {
            0.0
        };

        let topic_changes = self.count_event(EngagementEvent::TopicChange);

        let engagement_score = self.calculate_engagement_score(
            message_count,
            session_duration,
            avg_response_time,
            follow_up_ratio,
        );

        let sentiment_trend = self.calculate_sentiment_trend();
        let depth_score = self.calculate_depth_score();

        EngagementMetrics {
            session_duration,
            message_count,
            avg_response_time,
            question_ratio,
            follow_up_ratio,
            topic_changes,
            engagement_score,
            sentiment_trend,
            depth_score,
        }
    }

    fn count_event(&self, event: EngagementEvent) -> usize {
        self.events.iter().filter(|e| e.event == event).count()
    }

    fn calculate_engagement_score(
        &self,
        message_count: usize,
        session_duration: Duration,
        avg_response_time: Duration,
        follow_up_ratio: f64,
    ) -> f64 {
        let mut score = 0.5;

        // More messages = more engagement
        score += (message_count as f64 / 20.0).min(0.2);

        // Longer sessions = more engagement
        let minutes = session_duration.as_secs_f64() / 60.0;
        score += (minutes / 30.0).min(0.15);

        // Quick responses = more engagement
        let resp_secs = avg_response_time.as_secs_f64();
        if resp_secs < 30.0 {
            score += 0.1;
        } else if resp_secs > 120.0 {
            score -= 0.1;
        }

        // Follow-up questions = deep engagement
        score += follow_up_ratio * 0.15;

        // Quick responses bonus
        let quick_responses = self.count_event(EngagementEvent::QuickResponse);
        score += (quick_responses as f64 / 10.0).min(0.1);

        // Long pauses penalty
        let long_pauses = self.count_event(EngagementEvent::LongPause);
        score -= (long_pauses as f64 / 5.0).min(0.2);

        score.clamp(0.0, 1.0)
    }

    fn calculate_sentiment_trend(&self) -> f64 {
        if self.feedback_scores.is_empty() {
            return 0.0;
        }

        // Calculate weighted average with more recent feedback weighted higher
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (i, score) in self.feedback_scores.iter().enumerate() {
            let weight = (i + 1) as f64;
            weighted_sum += *score as f64 * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }

    fn calculate_depth_score(&self) -> f64 {
        let mut score = 0.0;

        // Follow-up questions indicate depth
        let follow_ups = self.count_event(EngagementEvent::FollowUpQuestion);
        score += (follow_ups as f64 / 5.0).min(0.3);

        // Clarification requests indicate engagement
        let clarifications = self.count_event(EngagementEvent::ClarificationRequest);
        score += (clarifications as f64 / 3.0).min(0.2);

        // Few topic changes = focused conversation
        let topic_changes = self.count_event(EngagementEvent::TopicChange);
        if topic_changes < 3 {
            score += 0.2;
        }

        // Multiple topics explored = breadth
        score += (self.topics.len() as f64 / 5.0).min(0.3);

        score.clamp(0.0, 1.0)
    }

    pub fn end_session(&mut self) {
        self.record_event(EngagementEvent::SessionEnd);
    }
}

/// Multi-user engagement manager
pub struct EngagementManager {
    trackers: HashMap<String, EngagementTracker>,
    historical_metrics: HashMap<String, Vec<EngagementMetrics>>,
}

impl EngagementManager {
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            historical_metrics: HashMap::new(),
        }
    }

    pub fn get_tracker(&mut self, user_id: &str) -> &mut EngagementTracker {
        self.trackers
            .entry(user_id.to_string())
            .or_insert_with(|| EngagementTracker::new(user_id))
    }

    pub fn end_session(&mut self, user_id: &str) {
        if let Some(tracker) = self.trackers.get_mut(user_id) {
            tracker.end_session();
            let metrics = tracker.calculate_metrics();
            self.historical_metrics
                .entry(user_id.to_string())
                .or_default()
                .push(metrics);
        }
        self.trackers.remove(user_id);
    }

    pub fn get_user_history(&self, user_id: &str) -> Option<&Vec<EngagementMetrics>> {
        self.historical_metrics.get(user_id)
    }

    pub fn calculate_user_trends(&self, user_id: &str) -> Option<UserTrends> {
        let history = self.historical_metrics.get(user_id)?;
        if history.len() < 2 {
            return None;
        }

        let recent = &history[history.len() - 1];
        let previous = &history[history.len() - 2];

        Some(UserTrends {
            engagement_delta: recent.engagement_score - previous.engagement_score,
            session_duration_delta: recent.session_duration.as_secs_f64()
                - previous.session_duration.as_secs_f64(),
            message_count_delta: recent.message_count as i32 - previous.message_count as i32,
            avg_engagement: history.iter().map(|m| m.engagement_score).sum::<f64>()
                / history.len() as f64,
            total_sessions: history.len(),
        })
    }
}

impl Default for EngagementManager {
    fn default() -> Self {
        Self::new()
    }
}

/// User engagement trends
#[derive(Debug, Clone)]
pub struct UserTrends {
    pub engagement_delta: f64,
    pub session_duration_delta: f64,
    pub message_count_delta: i32,
    pub avg_engagement: f64,
    pub total_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engagement_tracker() {
        let mut tracker = EngagementTracker::new("user1");

        tracker.record_event(EngagementEvent::MessageSent);
        tracker.record_event(EngagementEvent::MessageReceived);
        tracker.record_event(EngagementEvent::QuestionAsked);

        let metrics = tracker.calculate_metrics();
        assert!(metrics.message_count >= 2);
    }

    #[test]
    fn test_feedback_tracking() {
        let mut tracker = EngagementTracker::new("user1");

        tracker.record_feedback(true);
        tracker.record_feedback(true);
        tracker.record_feedback(true); // All positive

        let metrics = tracker.calculate_metrics();
        assert!(metrics.sentiment_trend > 0.0);
    }

    #[test]
    fn test_engagement_manager() {
        let mut manager = EngagementManager::new();

        {
            let tracker = manager.get_tracker("user1");
            tracker.record_event(EngagementEvent::MessageSent);
        }

        manager.end_session("user1");
        assert!(manager.get_user_history("user1").is_some());
    }

    #[test]
    fn test_all_event_variants_recordable() {
        let mut tracker = EngagementTracker::new("u");
        let events = [
            EngagementEvent::MessageSent,
            EngagementEvent::MessageReceived,
            EngagementEvent::QuestionAsked,
            EngagementEvent::FollowUpQuestion,
            EngagementEvent::TopicChange,
            EngagementEvent::ClarificationRequest,
            EngagementEvent::PositiveFeedback,
            EngagementEvent::NegativeFeedback,
            EngagementEvent::SessionStart,
            EngagementEvent::SessionEnd,
            EngagementEvent::LongPause,
            EngagementEvent::QuickResponse,
        ];
        for e in events {
            tracker.record_event(e);
        }
        // SessionStart is auto-recorded in new(), plus 12 explicit = at least 13 events
        assert!(tracker.events.len() >= 13);
    }

    #[test]
    fn test_event_counts_correct() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_event(EngagementEvent::MessageSent);
        tracker.record_event(EngagementEvent::MessageSent);
        tracker.record_event(EngagementEvent::MessageReceived);
        let metrics = tracker.calculate_metrics();
        // 2 sent + 1 received = 3 messages
        assert_eq!(metrics.message_count, 3);
    }

    #[test]
    fn test_record_event_with_metadata() {
        let mut tracker = EngagementTracker::new("u");
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "keyboard".to_string());
        tracker.record_event_with_metadata(EngagementEvent::MessageSent, meta);
        // Verify event was recorded (should have SessionStart + MessageSent + possibly QuickResponse)
        assert!(tracker.events.len() >= 2);
    }

    #[test]
    fn test_record_topic_new_topic() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_topic("rust");
        tracker.record_topic("python");
        let metrics = tracker.calculate_metrics();
        assert_eq!(metrics.topic_changes, 2);
    }

    #[test]
    fn test_record_topic_repeated_no_change() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_topic("rust");
        tracker.record_topic("rust"); // Same topic — should NOT count
        tracker.record_topic("rust");
        let metrics = tracker.calculate_metrics();
        assert_eq!(metrics.topic_changes, 1); // Only 1 topic change
    }

    #[test]
    fn test_engagement_score_range() {
        let mut tracker = EngagementTracker::new("u");
        for _ in 0..10 {
            tracker.record_event(EngagementEvent::MessageSent);
            tracker.record_event(EngagementEvent::MessageReceived);
        }
        let metrics = tracker.calculate_metrics();
        assert!(metrics.engagement_score >= 0.0);
        assert!(metrics.engagement_score <= 1.0);
    }

    #[test]
    fn test_mixed_feedback_sentiment() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_feedback(true);  // +1
        tracker.record_feedback(false); // -1
        tracker.record_feedback(true);  // +1
        let metrics = tracker.calculate_metrics();
        // Weighted: 1*1 + (-1)*2 + 1*3 = 2, weights = 1+2+3 = 6, result = 0.33...
        assert!(metrics.sentiment_trend > 0.0);
    }

    #[test]
    fn test_all_negative_feedback_sentiment() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_feedback(false);
        tracker.record_feedback(false);
        let metrics = tracker.calculate_metrics();
        assert!(metrics.sentiment_trend < 0.0);
    }

    #[test]
    fn test_manager_multi_user() {
        let mut manager = EngagementManager::new();

        {
            let t1 = manager.get_tracker("alice");
            t1.record_event(EngagementEvent::MessageSent);
        }
        {
            let t2 = manager.get_tracker("bob");
            t2.record_event(EngagementEvent::QuestionAsked);
        }

        manager.end_session("alice");
        manager.end_session("bob");

        assert!(manager.get_user_history("alice").is_some());
        assert!(manager.get_user_history("bob").is_some());
        assert!(manager.get_user_history("charlie").is_none());
    }

    #[test]
    fn test_user_trends_requires_two_sessions() {
        let mut manager = EngagementManager::new();

        // One session only
        manager.get_tracker("u").record_event(EngagementEvent::MessageSent);
        manager.end_session("u");
        assert!(manager.calculate_user_trends("u").is_none());

        // Second session
        manager.get_tracker("u").record_event(EngagementEvent::MessageSent);
        manager.end_session("u");
        let trends = manager.calculate_user_trends("u");
        assert!(trends.is_some());
        assert_eq!(trends.unwrap().total_sessions, 2);
    }

    #[test]
    fn test_user_trends_delta_calculation() {
        let mut manager = EngagementManager::new();

        // Session 1: 2 messages
        {
            let t = manager.get_tracker("u");
            t.record_event(EngagementEvent::MessageSent);
            t.record_event(EngagementEvent::MessageReceived);
        }
        manager.end_session("u");

        // Session 2: 4 messages
        {
            let t = manager.get_tracker("u");
            t.record_event(EngagementEvent::MessageSent);
            t.record_event(EngagementEvent::MessageReceived);
            t.record_event(EngagementEvent::MessageSent);
            t.record_event(EngagementEvent::MessageReceived);
        }
        manager.end_session("u");

        let trends = manager.calculate_user_trends("u").unwrap();
        assert_eq!(trends.message_count_delta, 2); // 4 - 2 = 2
        assert_eq!(trends.total_sessions, 2);
        assert!(trends.avg_engagement > 0.0);
    }

    #[test]
    fn test_empty_tracker_metrics() {
        let tracker = EngagementTracker::new("u");
        let metrics = tracker.calculate_metrics();
        assert_eq!(metrics.message_count, 0);
        assert_eq!(metrics.question_ratio, 0.0);
        assert_eq!(metrics.follow_up_ratio, 0.0);
        assert_eq!(metrics.topic_changes, 0);
        assert_eq!(metrics.sentiment_trend, 0.0);
    }

    #[test]
    fn test_depth_score_with_follow_ups() {
        let mut tracker = EngagementTracker::new("u");
        tracker.record_event(EngagementEvent::FollowUpQuestion);
        tracker.record_event(EngagementEvent::FollowUpQuestion);
        tracker.record_event(EngagementEvent::ClarificationRequest);
        let metrics = tracker.calculate_metrics();
        assert!(metrics.depth_score > 0.0);
    }

    #[test]
    fn test_end_session_records_event() {
        let mut tracker = EngagementTracker::new("u");
        tracker.end_session();
        let has_end = tracker
            .events
            .iter()
            .any(|e| e.event == EngagementEvent::SessionEnd);
        assert!(has_end);
    }

    #[test]
    fn test_manager_default_impl() {
        let _manager = EngagementManager::default();
        // Should not panic
    }
}
