//! Response effectiveness scoring
//!
//! Measure how effective AI responses are.

use std::collections::HashMap;

/// Response effectiveness metrics
#[derive(Debug, Clone)]
pub struct EffectivenessScore {
    pub overall: f64,
    pub relevance: f64,
    pub completeness: f64,
    pub clarity: f64,
    pub actionability: f64,
    pub user_satisfaction: Option<f64>,
    pub breakdown: HashMap<String, f64>,
}

/// Question-answer pair for evaluation
#[derive(Debug, Clone)]
pub struct QAPair {
    pub question: String,
    pub response: String,
    pub context: Option<String>,
    pub expected_topics: Vec<String>,
    pub user_feedback: Option<UserFeedback>,
}

/// User feedback on response
#[derive(Debug, Clone)]
pub struct UserFeedback {
    pub helpful: bool,
    pub rating: Option<u8>,
    pub follow_up_needed: bool,
    pub comments: Option<String>,
}

/// Effectiveness scorer
pub struct EffectivenessScorer {
    weights: ScoringWeights,
    history: Vec<(QAPair, EffectivenessScore)>,
}

/// Scoring weights
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub relevance: f64,
    pub completeness: f64,
    pub clarity: f64,
    pub actionability: f64,
    pub user_satisfaction: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            relevance: 0.25,
            completeness: 0.25,
            clarity: 0.20,
            actionability: 0.15,
            user_satisfaction: 0.15,
        }
    }
}

impl EffectivenessScorer {
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
            history: Vec::new(),
        }
    }

    pub fn with_weights(weights: ScoringWeights) -> Self {
        Self {
            weights,
            history: Vec::new(),
        }
    }

    pub fn score(&mut self, qa: &QAPair) -> EffectivenessScore {
        let relevance = self.score_relevance(qa);
        let completeness = self.score_completeness(qa);
        let clarity = self.score_clarity(&qa.response);
        let actionability = self.score_actionability(&qa.response);
        let user_satisfaction = qa
            .user_feedback
            .as_ref()
            .map(|f| self.score_user_feedback(f));

        let mut overall = relevance * self.weights.relevance
            + completeness * self.weights.completeness
            + clarity * self.weights.clarity
            + actionability * self.weights.actionability;

        if let Some(satisfaction) = user_satisfaction {
            overall += satisfaction * self.weights.user_satisfaction;
        } else {
            // Redistribute weight if no user feedback
            let factor = 1.0 / (1.0 - self.weights.user_satisfaction);
            overall *= factor;
        }

        let mut breakdown = HashMap::new();
        breakdown.insert("relevance".to_string(), relevance);
        breakdown.insert("completeness".to_string(), completeness);
        breakdown.insert("clarity".to_string(), clarity);
        breakdown.insert("actionability".to_string(), actionability);

        let score = EffectivenessScore {
            overall: overall.clamp(0.0, 1.0),
            relevance,
            completeness,
            clarity,
            actionability,
            user_satisfaction,
            breakdown,
        };

        self.history.push((qa.clone(), score.clone()));

        score
    }

    fn score_relevance(&self, qa: &QAPair) -> f64 {
        let question_lower = qa.question.to_lowercase();
        let response_lower = qa.response.to_lowercase();

        // Extract key terms from question
        let question_terms: Vec<_> = question_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| {
                ![
                    "what", "how", "why", "when", "where", "which", "would", "could", "should",
                ]
                .contains(w)
            })
            .collect();

        if question_terms.is_empty() {
            return 0.5;
        }

        // Check how many question terms appear in response
        let matches = question_terms
            .iter()
            .filter(|term| response_lower.contains(*term))
            .count();

        let term_coverage = matches as f64 / question_terms.len() as f64;

        // Check expected topics
        let topic_coverage = if qa.expected_topics.is_empty() {
            1.0
        } else {
            let topic_matches = qa
                .expected_topics
                .iter()
                .filter(|topic| response_lower.contains(&topic.to_lowercase()))
                .count();
            topic_matches as f64 / qa.expected_topics.len() as f64
        };

        (term_coverage * 0.6 + topic_coverage * 0.4).clamp(0.0, 1.0)
    }

    fn score_completeness(&self, qa: &QAPair) -> f64 {
        let response = &qa.response;
        let mut score: f64 = 0.5;

        // Length-based scoring
        let word_count = response.split_whitespace().count();
        if word_count >= 50 {
            score += 0.2;
        } else if word_count < 20 {
            score -= 0.1;
        }

        // Check for structured content
        if response.contains('\n') {
            score += 0.1;
        }

        // Check for examples
        if response.to_lowercase().contains("example") || response.contains("```") {
            score += 0.1;
        }

        // Check for explanations
        let explanation_markers = ["because", "therefore", "this means", "in other words"];
        if explanation_markers
            .iter()
            .any(|m| response.to_lowercase().contains(m))
        {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    fn score_clarity(&self, response: &str) -> f64 {
        let mut score: f64 = 0.6;

        // Sentence structure
        let sentences: Vec<_> = response
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if !sentences.is_empty() {
            let avg_sentence_length: f64 = sentences
                .iter()
                .map(|s| s.split_whitespace().count() as f64)
                .sum::<f64>()
                / sentences.len() as f64;

            // Optimal sentence length is 15-25 words
            if (15.0..=25.0).contains(&avg_sentence_length) {
                score += 0.2;
            } else if avg_sentence_length > 40.0 {
                score -= 0.2;
            }
        }

        // Check for clear structure
        if response.contains("1.") || response.contains("- ") || response.contains("* ") {
            score += 0.1;
        }

        // Avoid jargon density (simple heuristic)
        let complex_words = response.split_whitespace().filter(|w| w.len() > 12).count();

        let total_words = response.split_whitespace().count().max(1);
        let complex_ratio = complex_words as f64 / total_words as f64;

        if complex_ratio > 0.1 {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    fn score_actionability(&self, response: &str) -> f64 {
        let lower = response.to_lowercase();
        let mut score = 0.3;

        // Check for action words
        let action_words = [
            "can",
            "should",
            "try",
            "use",
            "run",
            "execute",
            "click",
            "open",
            "create",
            "add",
            "remove",
            "install",
            "configure",
        ];

        let action_count = action_words.iter().filter(|w| lower.contains(*w)).count();

        score += (action_count as f64 / 5.0).min(0.3);

        // Check for step-by-step instructions
        if lower.contains("step") || lower.contains("first") || lower.contains("then") {
            score += 0.2;
        }

        // Check for code examples
        if response.contains("```") || response.contains("`") {
            score += 0.2;
        }

        score.clamp(0.0, 1.0)
    }

    fn score_user_feedback(&self, feedback: &UserFeedback) -> f64 {
        let mut score = 0.5;

        if feedback.helpful {
            score += 0.3;
        } else {
            score -= 0.3;
        }

        if let Some(rating) = feedback.rating {
            score += (rating as f64 - 3.0) / 10.0;
        }

        if feedback.follow_up_needed {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    pub fn get_average_scores(&self) -> Option<EffectivenessScore> {
        if self.history.is_empty() {
            return None;
        }

        let n = self.history.len() as f64;
        let mut avg = EffectivenessScore {
            overall: 0.0,
            relevance: 0.0,
            completeness: 0.0,
            clarity: 0.0,
            actionability: 0.0,
            user_satisfaction: None,
            breakdown: HashMap::new(),
        };

        let mut satisfaction_sum = 0.0;
        let mut satisfaction_count = 0;

        for (_, score) in &self.history {
            avg.overall += score.overall;
            avg.relevance += score.relevance;
            avg.completeness += score.completeness;
            avg.clarity += score.clarity;
            avg.actionability += score.actionability;

            if let Some(s) = score.user_satisfaction {
                satisfaction_sum += s;
                satisfaction_count += 1;
            }
        }

        avg.overall /= n;
        avg.relevance /= n;
        avg.completeness /= n;
        avg.clarity /= n;
        avg.actionability /= n;

        if satisfaction_count > 0 {
            avg.user_satisfaction = Some(satisfaction_sum / satisfaction_count as f64);
        }

        Some(avg)
    }

    pub fn get_improvement_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(avg) = self.get_average_scores() {
            if avg.relevance < 0.6 {
                suggestions.push(
                    "Improve response relevance by focusing on key question terms".to_string(),
                );
            }
            if avg.completeness < 0.6 {
                suggestions.push("Provide more comprehensive responses with examples".to_string());
            }
            if avg.clarity < 0.6 {
                suggestions.push("Use clearer language and better structure".to_string());
            }
            if avg.actionability < 0.5 {
                suggestions.push("Include more actionable steps and code examples".to_string());
            }
            if avg.user_satisfaction.map(|s| s < 0.6).unwrap_or(false) {
                suggestions.push("Focus on user needs and provide follow-up options".to_string());
            }
        }

        suggestions
    }
}

impl Default for EffectivenessScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch effectiveness evaluator
pub struct BatchEvaluator {
    scorer: EffectivenessScorer,
}

impl BatchEvaluator {
    pub fn new() -> Self {
        Self {
            scorer: EffectivenessScorer::new(),
        }
    }

    pub fn evaluate(&mut self, pairs: &[QAPair]) -> BatchResult {
        let scores: Vec<_> = pairs.iter().map(|qa| self.scorer.score(qa)).collect();

        let overall_avg =
            scores.iter().map(|s| s.overall).sum::<f64>() / scores.len().max(1) as f64;

        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.overall
                    .partial_cmp(&b.overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let worst_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.overall
                    .partial_cmp(&b.overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        BatchResult {
            scores,
            overall_average: overall_avg,
            best_response_idx: best_idx,
            worst_response_idx: worst_idx,
        }
    }
}

impl Default for BatchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch evaluation result
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub scores: Vec<EffectivenessScore>,
    pub overall_average: f64,
    pub best_response_idx: Option<usize>,
    pub worst_response_idx: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effectiveness_scoring() {
        let mut scorer = EffectivenessScorer::new();

        let qa = QAPair {
            question: "How do I install Python?".to_string(),
            response: "To install Python, you can download it from python.org. \
                       First, go to the website. Then click download. \
                       After downloading, run the installer."
                .to_string(),
            context: None,
            expected_topics: vec!["python".to_string(), "install".to_string()],
            user_feedback: None,
        };

        let score = scorer.score(&qa);
        assert!(score.overall > 0.0);
        assert!(score.relevance > 0.5);
    }

    #[test]
    fn test_with_feedback() {
        let mut scorer = EffectivenessScorer::new();

        let qa = QAPair {
            question: "What is Rust?".to_string(),
            response: "Rust is a systems programming language.".to_string(),
            context: None,
            expected_topics: vec!["rust".to_string()],
            user_feedback: Some(UserFeedback {
                helpful: true,
                rating: Some(4),
                follow_up_needed: false,
                comments: None,
            }),
        };

        let score = scorer.score(&qa);
        assert!(score.user_satisfaction.is_some());
    }

    #[test]
    fn test_batch_evaluation() {
        let mut evaluator = BatchEvaluator::new();

        let pairs = vec![
            QAPair {
                question: "Q1".to_string(),
                response: "A detailed response with examples and code.".to_string(),
                context: None,
                expected_topics: vec![],
                user_feedback: None,
            },
            QAPair {
                question: "Q2".to_string(),
                response: "Short.".to_string(),
                context: None,
                expected_topics: vec![],
                user_feedback: None,
            },
        ];

        let result = evaluator.evaluate(&pairs);
        assert_eq!(result.scores.len(), 2);
    }

    #[test]
    fn test_relevance_scoring() {
        let mut scorer = EffectivenessScorer::new();

        // High overlap: response contains many question terms
        let high_overlap = QAPair {
            question: "How do I configure database connections in Rust?".to_string(),
            response: "To configure database connections in Rust, you need to set up \
                       a connection pool. Database configuration typically involves \
                       specifying the host, port, and credentials for your Rust application."
                .to_string(),
            context: None,
            expected_topics: vec!["database".to_string(), "rust".to_string()],
            user_feedback: None,
        };
        let high_score = scorer.score(&high_overlap);

        // Low overlap: response has almost nothing in common with question
        let low_overlap = QAPair {
            question: "How do I configure database connections in Rust?".to_string(),
            response: "The weather today is sunny with a high of 75 degrees. \
                       Pack sunscreen if you plan to go outside."
                .to_string(),
            context: None,
            expected_topics: vec!["database".to_string(), "rust".to_string()],
            user_feedback: None,
        };
        let low_score = scorer.score(&low_overlap);

        assert!(
            high_score.relevance > low_score.relevance,
            "High overlap relevance ({}) should exceed low overlap relevance ({})",
            high_score.relevance,
            low_score.relevance
        );
    }

    #[test]
    fn test_completeness_long_response() {
        let mut scorer = EffectivenessScorer::new();

        // Build a 50+ word response with structured content and explanation markers
        let long_response = "To set up a Rust project you need to install the Rust toolchain. \
            First download rustup from the official website. Then run the installer. \
            After installation you can create a new project with cargo init. \
            For example, you might run cargo new my_project to scaffold a binary. \
            This means you will have a src/main.rs file ready. \
            Because Cargo handles dependencies, adding crates is straightforward. \
            Simply edit Cargo.toml and run cargo build to fetch everything.";

        let qa = QAPair {
            question: "How do I set up a Rust project?".to_string(),
            response: long_response.to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };

        let score = scorer.score(&qa);

        // 50+ words -> +0.2, contains "example" -> +0.1, has explanation marker "this means"/"because" -> +0.1
        // Base 0.5 + 0.2 + 0.1 + 0.1 = 0.9
        assert!(
            score.completeness >= 0.7,
            "Long detailed response should have high completeness, got {}",
            score.completeness
        );
    }

    #[test]
    fn test_completeness_short_response() {
        let mut scorer = EffectivenessScorer::new();

        let qa = QAPair {
            question: "How do I set up a Rust project?".to_string(),
            response: "Use cargo.".to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };

        let score = scorer.score(&qa);

        // Very short: < 20 words -> base 0.5 - 0.1 = 0.4, no structure/example/explanation
        assert!(
            score.completeness <= 0.5,
            "Very short response should have low completeness, got {}",
            score.completeness
        );
    }

    #[test]
    fn test_clarity_scoring() {
        let mut scorer = EffectivenessScorer::new();

        // Well-structured response with moderate sentence lengths and list markers
        let clear_qa = QAPair {
            question: "What are the benefits of Rust?".to_string(),
            response: "Rust provides several key benefits for developers today. \
                       Memory safety is guaranteed at compile time without garbage collection. \
                       Concurrency is fearless thanks to the ownership system. \
                       - Zero cost abstractions for high performance. \
                       - Great tooling with cargo and rustfmt."
                .to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };
        let clear_score = scorer.score(&clear_qa);

        // Run-on response: one giant sentence with 40+ words and no structure
        let unclear_qa = QAPair {
            question: "What are the benefits of Rust?".to_string(),
            response: "Rust is good because it has memory safety and it also has concurrency \
                       and it has zero cost abstractions and it has great tooling and it compiles \
                       fast and it has a nice community and it integrates well with C code and \
                       it has pattern matching and it has traits and generics and lifetimes and \
                       borrowing and ownership and move semantics and iterators and closures"
                .to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };
        let unclear_score = scorer.score(&unclear_qa);

        assert!(
            clear_score.clarity > unclear_score.clarity,
            "Well-structured text clarity ({}) should exceed run-on clarity ({})",
            clear_score.clarity,
            unclear_score.clarity
        );
    }

    #[test]
    fn test_actionability_scoring() {
        let mut scorer = EffectivenessScorer::new();

        // Response with action words, step-by-step instructions, and code blocks
        let actionable_qa = QAPair {
            question: "How do I create a web server?".to_string(),
            response: "First, you should install the framework. Then create a new file. \
                       Try using the following code:\n\
                       ```rust\nfn main() { println!(\"hello\"); }\n```\n\
                       Run the server with `cargo run`. You can configure the port \
                       by adding an environment variable."
                .to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };
        let actionable_score = scorer.score(&actionable_qa);

        // Response with no actions, no steps, no code
        let passive_qa = QAPair {
            question: "How do I create a web server?".to_string(),
            response: "Web servers are interesting pieces of technology. \
                       They have been around for decades. Many languages support them."
                .to_string(),
            context: None,
            expected_topics: vec![],
            user_feedback: None,
        };
        let passive_score = scorer.score(&passive_qa);

        assert!(
            actionable_score.actionability > passive_score.actionability,
            "Actionable response ({}) should score higher than passive ({})",
            actionable_score.actionability,
            passive_score.actionability
        );
        // The actionable response has action words + step words + code blocks
        assert!(
            actionable_score.actionability >= 0.7,
            "Highly actionable response should be >= 0.7, got {}",
            actionable_score.actionability
        );
    }

    #[test]
    fn test_average_scores() {
        let mut scorer = EffectivenessScorer::new();

        // Score with no history should be None
        assert!(scorer.get_average_scores().is_none());

        let qa1 = QAPair {
            question: "What is Rust programming language used for in production?".to_string(),
            response: "Rust is used for systems programming, web services, and CLI tools. \
                       Because it guarantees memory safety, it is popular in production."
                .to_string(),
            context: None,
            expected_topics: vec!["rust".to_string()],
            user_feedback: None,
        };
        let score1 = scorer.score(&qa1);

        let qa2 = QAPair {
            question: "How does cargo manage dependencies in large projects?".to_string(),
            response: "Cargo reads Cargo.toml to manage dependencies. You should run \
                       cargo update to refresh the lock file. This means reproducible builds."
                .to_string(),
            context: None,
            expected_topics: vec!["cargo".to_string()],
            user_feedback: None,
        };
        let score2 = scorer.score(&qa2);

        let avg = scorer.get_average_scores().expect("Should have averages after scoring");

        let expected_overall = (score1.overall + score2.overall) / 2.0;
        let expected_relevance = (score1.relevance + score2.relevance) / 2.0;
        let expected_completeness = (score1.completeness + score2.completeness) / 2.0;
        let expected_clarity = (score1.clarity + score2.clarity) / 2.0;
        let expected_actionability = (score1.actionability + score2.actionability) / 2.0;

        let eps = 1e-10;
        assert!((avg.overall - expected_overall).abs() < eps, "overall mismatch");
        assert!((avg.relevance - expected_relevance).abs() < eps, "relevance mismatch");
        assert!((avg.completeness - expected_completeness).abs() < eps, "completeness mismatch");
        assert!((avg.clarity - expected_clarity).abs() < eps, "clarity mismatch");
        assert!((avg.actionability - expected_actionability).abs() < eps, "actionability mismatch");
        // No user feedback provided, so satisfaction should be None
        assert!(avg.user_satisfaction.is_none());
    }
}
