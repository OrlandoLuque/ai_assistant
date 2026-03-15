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
#[non_exhaustive]
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

impl FlowAnalysis {
    /// Export the analysis as JSON for visualization.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "current_state": format!("{:?}", self.current_state),
            "topic_coherence": self.topic_coherence,
            "engagement_score": self.engagement_score,
            "depth_score": self.depth_score,
            "topics": self.topics,
            "transitions": self.transitions.iter().map(|t| serde_json::json!({
                "from_topic": t.from_topic,
                "to_topic": t.to_topic,
                "turn_index": t.turn_index,
                "smoothness": t.smoothness,
            })).collect::<Vec<_>>(),
            "turn_distribution": self.turn_distribution,
            "average_turn_length": self.average_turn_length,
            "conversation_balance": self.conversation_balance,
        })
    }
}

/// Conversation flow analyzer
pub struct FlowAnalyzer {
    turns: Vec<ConversationTurn>,
    topic_keywords: HashMap<String, Vec<String>>,
}

impl FlowAnalyzer {
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            topic_keywords: HashMap::new(),
        }
    }

    /// Add a keyword associated with a topic for enriched analysis.
    pub fn add_topic_keyword(&mut self, topic: &str, keyword: &str) {
        self.topic_keywords
            .entry(topic.to_string())
            .or_default()
            .push(keyword.to_string());
    }

    /// Get all registered topic keywords.
    pub fn topic_keywords(&self) -> &HashMap<String, Vec<String>> {
        &self.topic_keywords
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
            let avg_length: f64 =
                recent.iter().map(|t| t.token_count as f64).sum::<f64>() / recent.len() as f64;
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

            if lower.contains("?")
                && (lower.contains("what do you mean") || lower.contains("clarify"))
            {
                return FlowState::Clarification;
            }

            if lower.contains("tell me more")
                || lower.contains("explain")
                || lower.contains("elaborate")
            {
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
        let mut topics: Vec<_> = topic_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();

        topics.sort_by(|a, b| b.1.cmp(&a.1));

        topics.into_iter().take(5).map(|(word, _)| word).collect()
    }

    fn analyze_transitions(&self, topics: &[String]) -> Vec<TopicTransition> {
        let mut transitions = Vec::new();

        for (i, turn) in self.turns.iter().enumerate().skip(1) {
            let prev_turn = &self.turns[i - 1];

            let prev_topics: Vec<_> = topics
                .iter()
                .filter(|t| prev_turn.content.to_lowercase().contains(*t))
                .collect();

            let curr_topics: Vec<_> = topics
                .iter()
                .filter(|t| turn.content.to_lowercase().contains(*t))
                .collect();

            // Check for topic change
            if !prev_topics.is_empty() && !curr_topics.is_empty() {
                let overlap = prev_topics
                    .iter()
                    .filter(|t| curr_topics.contains(t))
                    .count();

                let smoothness = overlap as f64 / prev_topics.len().max(curr_topics.len()) as f64;

                if smoothness < 0.5 {
                    transitions.push(TopicTransition {
                        from_topic: prev_topics
                            .first()
                            .map(|s| s.to_string())
                            .unwrap_or_default(),
                        to_topic: curr_topics
                            .first()
                            .map(|s| s.to_string())
                            .unwrap_or_default(),
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
        let questions = self
            .turns
            .iter()
            .filter(|t| t.content.contains("?"))
            .count();

        score += (questions as f64 / self.turns.len() as f64) * 0.3;

        // Longer responses indicate engagement
        let avg_length: f64 =
            self.turns.iter().map(|t| t.token_count as f64).sum::<f64>() / self.turns.len() as f64;

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
            "because",
            "therefore",
            "however",
            "specifically",
            "in detail",
            "for example",
            "furthermore",
            "moreover",
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

        self.turns.iter().map(|t| t.token_count as f64).sum::<f64>() / self.turns.len() as f64
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
            FlowState::InformationGathering => {
                "Provide relevant information or ask clarifying questions".to_string()
            }
            FlowState::ProblemSolving => "Continue working on the solution".to_string(),
            FlowState::Clarification => "Provide clearer explanation".to_string(),
            FlowState::Elaboration => "Expand on the topic with more details".to_string(),
            FlowState::Conclusion => "Summarize and offer further assistance".to_string(),
            FlowState::OffTopic => "Gently redirect to the main topic".to_string(),
            FlowState::Stalled => {
                "Ask an engaging question to restart the conversation".to_string()
            }
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
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "Hi! How can I help you today?",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "I need help with Python programming",
        ));

        let analysis = analyzer.analyze();
        assert!(analysis.engagement_score > 0.0);
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = FlowAnalyzer::new();
        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Opening);
    }

    #[test]
    fn test_flow_analysis_to_json() {
        let analysis = FlowAnalysis {
            current_state: FlowState::Opening,
            topic_coherence: 0.8,
            engagement_score: 0.7,
            depth_score: 0.6,
            topics: vec!["rust".to_string()],
            transitions: vec![],
            turn_distribution: std::collections::HashMap::new(),
            average_turn_length: 50.0,
            conversation_balance: 0.5,
        };
        let json = analysis.to_json();
        assert_eq!(json["current_state"], "Opening");
        assert_eq!(json["topic_coherence"], 0.8);
        assert_eq!(json["topics"].as_array().unwrap().len(), 1);
    }

    // ====================================================================
    // Additional tests: state transitions, edge cases, coverage expansion
    // ====================================================================

    #[test]
    fn test_state_transition_user_assistant_user() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new("user", "What is Rust programming?"));
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "Rust is a systems programming language focused on safety and performance.",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "Tell me more about its memory safety features.",
        ));

        let analysis = analyzer.analyze();
        // Last message contains "tell me more" -> Elaboration
        assert_eq!(analysis.current_state, FlowState::Elaboration);
    }

    #[test]
    fn test_state_conclusion_on_thank_you() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new("user", "How do I sort a list in Python?"));
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "You can use the sorted() function or the list.sort() method.",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "Thank you so much, that helps!",
        ));

        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Conclusion);
    }

    #[test]
    fn test_state_clarification() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new("user", "Explain monads."));
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "Monads are a design pattern used in functional programming.",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "What do you mean? Can you clarify?",
        ));

        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Clarification);
    }

    #[test]
    fn test_state_information_gathering_with_question() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "I have a project with multiple modules. Which architecture pattern should I use?",
        ));

        let analysis = analyzer.analyze();
        // Contains "?" but not "clarify" or "tell me more" -> InformationGathering
        assert_eq!(analysis.current_state, FlowState::InformationGathering);
    }

    #[test]
    fn test_state_stalled_short_messages() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new("user", "Ok"));
        analyzer.add_turn(ConversationTurn::new("assistant", "Yes"));
        analyzer.add_turn(ConversationTurn::new("user", "Hmm"));

        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Stalled);
    }

    #[test]
    fn test_state_problem_solving_default() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "I am implementing a binary search tree in Rust with generics and lifetime parameters.",
        ));

        let analysis = analyzer.analyze();
        // No question mark, not short, no keywords -> ProblemSolving
        assert_eq!(analysis.current_state, FlowState::ProblemSolving);
    }

    #[test]
    fn test_single_message_conversation() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new("user", "Hello world"));

        let analysis = analyzer.analyze();
        assert_eq!(analysis.turn_distribution.get("user"), Some(&1));
        assert!(analysis.average_turn_length > 0.0);
        // Only user turns -> balance deviates from 0.5
        assert!(analysis.conversation_balance < 1.0);
    }

    #[test]
    fn test_empty_message_turn() {
        let turn = ConversationTurn::new("user", "");
        assert_eq!(turn.token_count, 0);
        assert_eq!(turn.content, "");
        assert_eq!(turn.role, "user");
    }

    #[test]
    fn test_very_long_message() {
        let long_content = "word ".repeat(5000);
        let turn = ConversationTurn::new("user", &long_content);
        assert_eq!(turn.token_count, 5000);
        assert!(turn.content.len() > 20000);
    }

    #[test]
    fn test_turn_distribution_balanced() {
        let mut analyzer = FlowAnalyzer::new();
        for _ in 0..5 {
            analyzer.add_turn(ConversationTurn::new("user", "This is a user message with enough words."));
            analyzer.add_turn(ConversationTurn::new(
                "assistant",
                "This is an assistant reply with enough words too.",
            ));
        }

        let analysis = analyzer.analyze();
        assert_eq!(analysis.turn_distribution.get("user"), Some(&5));
        assert_eq!(analysis.turn_distribution.get("assistant"), Some(&5));
        // Perfect balance should be close to 1.0
        assert!(
            analysis.conversation_balance > 0.9,
            "Expected balanced conversation, got {}",
            analysis.conversation_balance
        );
    }

    #[test]
    fn test_depth_score_with_indicators() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "Because of the complexity, I need a detailed solution. Furthermore, for example, \
             consider the case where the input is invalid. Moreover, specifically the edge cases \
             need to be handled. Therefore the implementation must be robust. However, in detail, \
             we should also consider performance.",
        ));

        let analysis = analyzer.analyze();
        assert!(
            analysis.depth_score > 0.0,
            "Expected positive depth score, got {}",
            analysis.depth_score
        );
    }

    #[test]
    fn test_engagement_score_with_questions() {
        let mut analyzer = FlowAnalyzer::new();
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "How does async work in Rust? What about lifetimes? Can you show an example?",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "Great questions! Async in Rust works through the Future trait. \
             Would you like me to elaborate on any specific aspect?",
        ));

        let analysis = analyzer.analyze();
        // Both turns have questions, so engagement should be > baseline 0.5
        assert!(
            analysis.engagement_score > 0.5,
            "Expected engagement > 0.5, got {}",
            analysis.engagement_score
        );
    }

    #[test]
    fn test_suggest_next_action_per_state() {
        // Opening
        let analyzer = FlowAnalyzer::new();
        let suggestion = analyzer.suggest_next_action();
        assert!(suggestion.contains("Gather"));

        // Conclusion
        let mut analyzer2 = FlowAnalyzer::new();
        analyzer2.add_turn(ConversationTurn::new("user", "Thank you, goodbye!"));
        let suggestion2 = analyzer2.suggest_next_action();
        assert!(suggestion2.contains("Summarize"));

        // Stalled
        let mut analyzer3 = FlowAnalyzer::new();
        analyzer3.add_turn(ConversationTurn::new("user", "Ok"));
        analyzer3.add_turn(ConversationTurn::new("assistant", "Yes"));
        analyzer3.add_turn(ConversationTurn::new("user", "Hmm"));
        let suggestion3 = analyzer3.suggest_next_action();
        assert!(suggestion3.contains("engaging question"));
    }

    #[test]
    fn test_flow_analysis_to_json_with_transitions() {
        let analysis = FlowAnalysis {
            current_state: FlowState::ProblemSolving,
            topic_coherence: 0.9,
            engagement_score: 0.85,
            depth_score: 0.7,
            topics: vec!["rust".to_string(), "async".to_string()],
            transitions: vec![TopicTransition {
                from_topic: "rust".to_string(),
                to_topic: "async".to_string(),
                turn_index: 3,
                smoothness: 0.4,
            }],
            turn_distribution: {
                let mut m = HashMap::new();
                m.insert("user".to_string(), 5);
                m.insert("assistant".to_string(), 4);
                m
            },
            average_turn_length: 42.0,
            conversation_balance: 0.9,
        };
        let json = analysis.to_json();
        assert_eq!(json["current_state"], "ProblemSolving");
        assert_eq!(json["topics"].as_array().unwrap().len(), 2);
        let transitions = json["transitions"].as_array().unwrap();
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0]["from_topic"], "rust");
        assert_eq!(transitions[0]["to_topic"], "async");
        assert_eq!(transitions[0]["turn_index"], 3);
    }

    #[test]
    fn test_default_flow_analyzer() {
        let analyzer = FlowAnalyzer::default();
        let analysis = analyzer.analyze();
        assert_eq!(analysis.current_state, FlowState::Opening);
        assert_eq!(analysis.average_turn_length, 0.0);
        assert_eq!(analysis.depth_score, 0.0);
        assert_eq!(analysis.conversation_balance, 0.5);
    }

    #[test]
    fn test_topic_extraction_with_repeated_words() {
        let mut analyzer = FlowAnalyzer::new();
        // Words must appear at least twice and be > 4 chars to be extracted as topics
        analyzer.add_turn(ConversationTurn::new(
            "user",
            "I want to learn about programming languages and their performance characteristics.",
        ));
        analyzer.add_turn(ConversationTurn::new(
            "assistant",
            "Programming languages differ in performance based on their type system and runtime.",
        ));

        let analysis = analyzer.analyze();
        // "programming" and "performance" appear in both turns and are > 4 chars
        assert!(
            !analysis.topics.is_empty(),
            "Expected topics to be extracted from repeated words"
        );
    }
}
