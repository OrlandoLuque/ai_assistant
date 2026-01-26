//! User engagement metrics
//!
//! Track and analyze user engagement patterns.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Engagement event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    metadata: HashMap<String, String>,
}

/// User engagement tracker
pub struct EngagementTracker {
    #[allow(dead_code)]
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

    pub fn record_event(&mut self, event: EngagementEvent) {
        self.record_event_with_metadata(event, HashMap::new());
    }

    pub fn record_event_with_metadata(&mut self, event: EngagementEvent, metadata: HashMap<String, String>) {
        let now = Instant::now();
        let time_since_last = now.duration_since(self.last_activity);

        // Track response times
        if matches!(event, EngagementEvent::MessageSent | EngagementEvent::MessageReceived) {
            self.message_times.push(time_since_last);
        }

        // Track long pauses
        if time_since_last > Duration::from_secs(60) {
            self.events.push(EngagementRecord {
                event: EngagementEvent::LongPause,
                timestamp: now,
                metadata: HashMap::new(),
            });
        } else if time_since_last < Duration::from_secs(5) && event == EngagementEvent::MessageSent {
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

        let message_count = self.events.iter()
            .filter(|e| matches!(e.event, EngagementEvent::MessageSent | EngagementEvent::MessageReceived))
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
            avg_engagement: history.iter().map(|m| m.engagement_score).sum::<f64>() / history.len() as f64,
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
}
