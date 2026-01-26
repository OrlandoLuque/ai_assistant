//! Conversation flow analysis
//!
//! Analyze conversation patterns and flow.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Turn in a conversation
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: Instant,
    pub response_time: Option<Duration>,
    pub token_count: usize,
}

impl ConversationTurn {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Instant::now(),
            response_time: None,
            token_count: content.split_whitespace().count(),
        }
    }
}

/// Flow state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowState {
    Opening,
    InformationGathering,
    ProblemSolving,
    Clarification,
    Elaboration,
    Conclusion,
    OffTopic,
    Stalled,
}

/// Topic transition
#[derive(Debug, Clone)]
pub struct TopicTransition {
    pub from_topic: String,
    pub to_topic: String,
    pub turn_index: usize,
    pub smoothness: f64,
}

/// Flow analysis result
#[derive(Debug, Clone)]
pub struct FlowAnalysis {
    pub current_state: FlowState,
    pub topic_coherence: f64,
    pub engagement_score: f64,
    pub depth_score: f64,
    pub topics: Vec<String>,
    pub transitions: Vec<TopicTransition>,
    pub turn_distribution: HashMap<String, usize>,
    pub average_turn_length: f64,
    pub conversation_balance: f64,
}

/// Conversation flow analyzer
pub struct FlowAnalyzer {
    turns: Vec<ConversationTurn>,
    #[allow(dead_code)]
    topic_keywords: HashMap<String, Vec<String>>,
}

impl FlowAnalyzer {
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            topic_keywords: HashMap::new(),
        }
    }

    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
    }

    pub fn analyze(&self) -> FlowAnalysis {
        let current_state = self.determine_state();
        let topics = self.extract_topics();
        let transitions = self.analyze_transitions(&topics);

        let topic_coherence = self.calculate_coherence(&topics);
        let engagement_score = self.calculate_engagement();
        let depth_score = self.calculate_depth();

        let turn_distribution = self.calculate_turn_distribution();
        let average_turn_length = self.calculate_avg_turn_length();
        let conversation_balance = self.calculate_balance();

        FlowAnalysis {
            current_state,
            topic_coherence,
            engagement_score,
            depth_score,
            topics,
            transitions,
            turn_distribution,
            average_turn_length,
            conversation_balance,
        }
    }

    fn determine_state(&self) -> FlowState {
        if self.turns.is_empty() {
            return FlowState::Opening;
        }

        let recent: Vec<_> = self.turns.iter().rev().take(3).collect();

        // Check for stalled conversation
        if recent.len() >= 2 {
            let avg_length: f64 = recent.iter().map(|t| t.token_count as f64).sum::<f64>() / recent.len() as f64;
            if avg_length < 5.0 {
                return FlowState::Stalled;
            }
        }

        // Analyze recent content
        if let Some(last) = recent.first() {
            let lower = last.content.to_lowercase();

            if lower.contains("thank") || lower.contains("bye") || lower.contains("goodbye") {
                return FlowState::Conclusion;
            }

            if lower.contains("?") && (lower.contains("what do you mean") || lower.contains("clarify")) {
                return FlowState::Clarification;
            }

            if lower.contains("tell me more") || lower.contains("explain") || lower.contains("elaborate") {
                return FlowState::Elaboration;
            }

            if lower.contains("?") {
                return FlowState::InformationGathering;
            }
        }

        FlowState::ProblemSolving
    }

    fn extract_topics(&self) -> Vec<String> {
        let mut topic_counts: HashMap<String, usize> = HashMap::new();

        for turn in &self.turns {
            let content_lower = turn.content.to_lowercase();
            let words: Vec<&str> = content_lower
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .collect();

            for word in words {
                *topic_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Get top topics
        let mut topics: Vec<_> = topic_counts.into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();

        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter()
            .take(5)
            .map(|(word, _)| word)
            .collect()
    }

    fn analyze_transitions(&self, topics: &[String]) -> Vec<TopicTransition> {
        let mut transitions = Vec::new();

        for (i, turn) in self.turns.iter().enumerate().skip(1) {
            let prev_turn = &self.turns[i - 1];

            let prev_topics: Vec<_> = topics.iter()
                .filter(|t| prev_turn.content.to_lowercase().contains(*t))
                .collect();

            let curr_topics: Vec<_> = topics.iter()
                .filter(|t| turn.content.to_lowercase().contains(*t))
                .collect();

            // Check for topic change
            if !prev_topics.is_empty() && !curr_topics.is_empty() {
                let overlap = prev_topics.iter()
                    .filter(|t| curr_topics.contains(t))
                    .count();

                let smoothness = overlap as f64 / prev_topics.len().max(curr_topics.len()) as f64;

                if smoothness < 0.5 {
                    transitions.push(TopicTransition {
                        from_topic: prev_topics.first().map(|s| s.to_string()).unwrap_or_default(),
                        to_topic: curr_topics.first().map(|s| s.to_string()).unwrap_or_default(),
                        turn_index: i,
                        smoothness,
                    });
                }
            }
        }

        transitions
    }

    fn calculate_coherence(&self, topics: &[String]) -> f64 {
        if self.turns.len() < 2 || topics.is_empty() {
            return 1.0;
        }

        let mut coherent_turns = 0;

        for turn in &self.turns {
            let lower = turn.content.to_lowercase();
            if topics.iter().any(|t| lower.contains(t)) {
                coherent_turns += 1;
            }
        }

        coherent_turns as f64 / self.turns.len() as f64
    }

    fn calculate_engagement(&self) -> f64 {
        if self.turns.is_empty() {
            return 0.5;
        }

        let mut score = 0.5;

        // Questions indicate engagement
        let questions = self.turns.iter()
            .filter(|t| t.content.contains("?"))
            .count();

        score += (questions as f64 / self.turns.len() as f64) * 0.3;

        // Longer responses indicate engagement
        let avg_length: f64 = self.turns.iter()
            .map(|t| t.token_count as f64)
            .sum::<f64>() / self.turns.len() as f64;

        if avg_length > 20.0 {
            score += 0.2;
        }

        score.min(1.0)
    }

    fn calculate_depth(&self) -> f64 {
        if self.turns.is_empty() {
            return 0.0;
        }

        let depth_indicators = [
            "because", "therefore", "however", "specifically",
            "in detail", "for example", "furthermore", "moreover",
        ];

        let mut indicator_count = 0;

        for turn in &self.turns {
            let lower = turn.content.to_lowercase();
            for indicator in depth_indicators {
                if lower.contains(indicator) {
                    indicator_count += 1;
                }
            }
        }

        (indicator_count as f64 / self.turns.len() as f64).min(1.0)
    }

    fn calculate_turn_distribution(&self) -> HashMap<String, usize> {
        let mut dist = HashMap::new();

        for turn in &self.turns {
            *dist.entry(turn.role.clone()).or_insert(0) += 1;
        }

        dist
    }

    fn calculate_avg_turn_length(&self) -> f64 {
        if self.turns.is_empty() {
            return 0.0;
        }

        self.turns.iter()
            .map(|t| t.token_count as f64)
            .sum::<f64>() / self.turns.len() as f64
    }

    fn calculate_balance(&self) -> f64 {
        let dist = self.calculate_turn_distribution();

        let user_turns = *dist.get("user").unwrap_or(&0) as f64;
        let assistant_turns = *dist.get("assistant").unwrap_or(&0) as f64;

        if user_turns + assistant_turns == 0.0 {
            return 0.5;
        }

        let ratio = user_turns / (user_turns + assistant_turns);

        // Perfect balance is 0.5, score decreases as it deviates
        1.0 - (ratio - 0.5).abs() * 2.0
    }

    pub fn suggest_next_action(&self) -> String {
        let analysis = self.analyze();

        match analysis.current_state {
            FlowState::Opening => "Gather information about user needs".to_string(),
            FlowState::InformationGathering => "Provide relevant information or ask clarifying questions".to_string(),
            FlowState::ProblemSolving => "Continue working on the solution".to_string(),
            FlowState::Clarification => "Provide clearer explanation".to_string(),
            FlowState::Elaboration => "Expand on the topic with more details".to_string(),
            FlowState::Conclusion => "Summarize and offer further assistance".to_string(),
            FlowState::OffTopic => "Gently redirect to the main topic".to_string(),
            FlowState::Stalled => "Ask an engaging question to restart the conversation".to_string(),
        }
    }
}

impl Default for FlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_creation() {
        let turn = ConversationTurn::new("user", "Hello, how are you?");
        assert_eq!(turn.role, "user");
        assert!(turn.token_count > 0);
    }

    #[test]
    fn test_flow_analysis() {
        let mut analyzer = FlowAnalyzer::new();

        analyzer.add_turn(ConversationTurn::new("user", "Hello!"));
        analyzer.add_turn(ConversationTurn::new("assistant", "Hi! How can I help you today?"));
        analyzer.add_turn(ConversationTurn::new("user", "I need help with Python programming"));

        let analysis = analyzer.analyze();
        assert!(analysis.engagement_score > 0.0);
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = FlowAnalyzer::new();
        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Opening);
    }
}
