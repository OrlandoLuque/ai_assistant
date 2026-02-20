//! Response quality analysis
//!
//! This module provides tools to evaluate the quality of AI responses
//! including coherence, relevance, fluency, and factual consistency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Quality Metrics
// ============================================================================

/// Overall quality score for a response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall score (0.0 - 1.0)
    pub overall: f32,
    /// Relevance to the query
    pub relevance: f32,
    /// Text coherence and flow
    pub coherence: f32,
    /// Language fluency
    pub fluency: f32,
    /// Response completeness
    pub completeness: f32,
    /// Factual consistency with context
    pub consistency: f32,
    /// Quality issues found
    pub issues: Vec<QualityIssue>,
    /// Positive aspects
    pub strengths: Vec<String>,
}

impl Default for QualityScore {
    fn default() -> Self {
        Self {
            overall: 0.5,
            relevance: 0.5,
            coherence: 0.5,
            fluency: 0.5,
            completeness: 0.5,
            consistency: 0.5,
            issues: Vec::new(),
            strengths: Vec::new(),
        }
    }
}

impl QualityScore {
    /// Calculate overall score from components
    pub fn calculate_overall(&mut self) {
        self.overall = (self.relevance * 0.25
            + self.coherence * 0.2
            + self.fluency * 0.2
            + self.completeness * 0.2
            + self.consistency * 0.15)
            .clamp(0.0, 1.0);
    }

    /// Get quality level description
    pub fn quality_level(&self) -> &'static str {
        match self.overall {
            x if x >= 0.9 => "Excellent",
            x if x >= 0.75 => "Good",
            x if x >= 0.5 => "Acceptable",
            x if x >= 0.25 => "Poor",
            _ => "Very Poor",
        }
    }
}

/// A quality issue found in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,
    /// Description of the issue
    pub description: String,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
    /// Position in text (if applicable)
    pub position: Option<usize>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Types of quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Response is off-topic
    Irrelevant,
    /// Response doesn't fully answer the question
    Incomplete,
    /// Response contradicts itself
    SelfContradiction,
    /// Response contradicts context/knowledge
    FactualInconsistency,
    /// Grammar or spelling error
    GrammarError,
    /// Repetitive content
    Repetition,
    /// Too verbose
    Verbose,
    /// Too brief
    TerseResponse,
    /// Unclear or ambiguous
    Unclear,
    /// Missing expected information
    MissingInfo,
    /// Potentially harmful content
    SafetyIssue,
    /// Code error (for code responses)
    CodeError,
    /// Hallucination detected
    Hallucination,
}

impl QualityIssueType {
    pub fn description(&self) -> &'static str {
        match self {
            QualityIssueType::Irrelevant => "Response is off-topic",
            QualityIssueType::Incomplete => "Response doesn't fully answer the question",
            QualityIssueType::SelfContradiction => "Response contradicts itself",
            QualityIssueType::FactualInconsistency => "Response contradicts provided context",
            QualityIssueType::GrammarError => "Grammar or spelling error",
            QualityIssueType::Repetition => "Repetitive content",
            QualityIssueType::Verbose => "Response is unnecessarily long",
            QualityIssueType::TerseResponse => "Response is too brief",
            QualityIssueType::Unclear => "Response is unclear or ambiguous",
            QualityIssueType::MissingInfo => "Missing expected information",
            QualityIssueType::SafetyIssue => "Potentially problematic content",
            QualityIssueType::CodeError => "Code contains errors",
            QualityIssueType::Hallucination => "Response contains fabricated information",
        }
    }
}

// ============================================================================
// Quality Analyzer
// ============================================================================

/// Configuration for quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum expected response length
    pub min_response_length: usize,
    /// Maximum expected response length
    pub max_response_length: usize,
    /// Check for repetition
    pub check_repetition: bool,
    /// Repetition threshold (0.0 - 1.0)
    pub repetition_threshold: f32,
    /// Check relevance to query
    pub check_relevance: bool,
    /// Check code quality (for code responses)
    pub check_code: bool,
    /// Check for safety issues
    pub check_safety: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_response_length: 20,
            max_response_length: 10000,
            check_repetition: true,
            repetition_threshold: 0.3,
            check_relevance: true,
            check_code: true,
            check_safety: true,
        }
    }
}

/// Quality analyzer for AI responses
pub struct QualityAnalyzer {
    config: QualityConfig,
}

impl QualityAnalyzer {
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Analyze response quality
    pub fn analyze(&self, query: &str, response: &str, context: Option<&str>) -> QualityScore {
        let mut score = QualityScore::default();

        // Check relevance
        if self.config.check_relevance {
            score.relevance = self.analyze_relevance(query, response);
        }

        // Check coherence
        score.coherence = self.analyze_coherence(response);

        // Check fluency
        score.fluency = self.analyze_fluency(response);

        // Check completeness
        score.completeness = self.analyze_completeness(query, response);

        // Check consistency with context
        if let Some(ctx) = context {
            score.consistency = self.analyze_consistency(response, ctx);
        } else {
            score.consistency = 0.8; // Default to good if no context
        }

        // Check for issues
        score.issues = self.find_issues(query, response, context);

        // Find strengths
        score.strengths = self.find_strengths(query, response);

        // Apply issue penalties
        let issue_penalty: f32 = score.issues.iter().map(|i| i.severity * 0.1).sum();
        score.overall = (score.overall - issue_penalty).max(0.0);

        score.calculate_overall();
        score
    }

    /// Analyze relevance to the query
    fn analyze_relevance(&self, query: &str, response: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        let response_lower = response.to_lowercase();

        if query_words.is_empty() {
            return 0.8;
        }

        let matches: usize = query_words
            .iter()
            .filter(|w| response_lower.contains(*w))
            .count();

        let ratio = matches as f32 / query_words.len() as f32;

        // Adjust score based on ratio
        (ratio * 0.5 + 0.5).min(1.0)
    }

    /// Analyze text coherence
    fn analyze_coherence(&self, response: &str) -> f32 {
        let sentences: Vec<&str> = response
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.is_empty() {
            return 0.3;
        }

        if sentences.len() == 1 {
            return 0.7;
        }

        // Check for transition words between sentences
        let transition_words = [
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "additionally",
            "also",
            "thus",
            "hence",
            "consequently",
            "because",
            "since",
            "although",
            "while",
            "but",
            "and",
            "so",
            "then",
            "first",
            "second",
            "finally",
            "in conclusion",
            "for example",
            "specifically",
        ];

        let response_lower = response.to_lowercase();
        let transition_count = transition_words
            .iter()
            .filter(|w| response_lower.contains(*w))
            .count();

        let transition_ratio = (transition_count as f32 / sentences.len() as f32).min(1.0);

        0.6 + transition_ratio * 0.4
    }

    /// Analyze fluency
    fn analyze_fluency(&self, response: &str) -> f32 {
        let words: Vec<&str> = response.split_whitespace().collect();

        if words.is_empty() {
            return 0.0;
        }

        // Check average word length (reasonable is 4-8)
        let avg_word_len: f32 =
            words.iter().map(|w| w.len() as f32).sum::<f32>() / words.len() as f32;
        let word_len_score = if avg_word_len >= 3.0 && avg_word_len <= 10.0 {
            1.0
        } else {
            0.5
        };

        // Check average sentence length
        let sentences: Vec<&str> = response
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let avg_sentence_len = if sentences.is_empty() {
            0.0
        } else {
            words.len() as f32 / sentences.len() as f32
        };

        let sentence_len_score = if avg_sentence_len >= 5.0 && avg_sentence_len <= 30.0 {
            1.0
        } else if avg_sentence_len > 30.0 {
            0.7 // Too long
        } else {
            0.6 // Too short
        };

        (word_len_score + sentence_len_score) / 2.0
    }

    /// Analyze completeness
    fn analyze_completeness(&self, query: &str, response: &str) -> f32 {
        // Check for question words in query
        let query_lower = query.to_lowercase();
        let is_question = query_lower.contains('?')
            || query_lower.starts_with("what")
            || query_lower.starts_with("how")
            || query_lower.starts_with("why")
            || query_lower.starts_with("when")
            || query_lower.starts_with("where")
            || query_lower.starts_with("who")
            || query_lower.starts_with("can")
            || query_lower.starts_with("could");

        let response_len = response.len();

        // Length-based completeness
        let len_score = if response_len < self.config.min_response_length {
            0.3
        } else if response_len > self.config.max_response_length {
            0.7
        } else {
            0.9
        };

        // Question response should be substantial
        if is_question && response_len < 50 {
            return 0.4;
        }

        len_score
    }

    /// Analyze consistency with context
    fn analyze_consistency(&self, response: &str, context: &str) -> f32 {
        // Simple keyword overlap check
        let context_lower = context.to_lowercase();
        let context_words: std::collections::HashSet<&str> = context_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let response_lower = response.to_lowercase();

        if context_words.is_empty() {
            return 0.8;
        }

        let matches: usize = context_words
            .iter()
            .filter(|w| response_lower.contains(*w))
            .count();

        let ratio = matches as f32 / context_words.len().min(50) as f32;

        (ratio * 0.5 + 0.5).min(1.0)
    }

    /// Find quality issues
    fn find_issues(
        &self,
        query: &str,
        response: &str,
        _context: Option<&str>,
    ) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check response length
        if response.len() < self.config.min_response_length {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::TerseResponse,
                description: "Response is too brief".to_string(),
                severity: 0.5,
                position: None,
                suggestion: Some("Provide more detailed explanation".to_string()),
            });
        }

        if response.len() > self.config.max_response_length {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::Verbose,
                description: "Response is excessively long".to_string(),
                severity: 0.3,
                position: None,
                suggestion: Some("Consider being more concise".to_string()),
            });
        }

        // Check for repetition
        if self.config.check_repetition {
            if let Some(issue) = self.check_repetition(response) {
                issues.push(issue);
            }
        }

        // Check relevance
        if self.config.check_relevance {
            let relevance = self.analyze_relevance(query, response);
            if relevance < 0.4 {
                issues.push(QualityIssue {
                    issue_type: QualityIssueType::Irrelevant,
                    description: "Response may not address the question".to_string(),
                    severity: 0.6,
                    position: None,
                    suggestion: Some("Focus more on the specific question asked".to_string()),
                });
            }
        }

        // Check for code issues
        if self.config.check_code && self.contains_code(response) {
            issues.extend(self.check_code_issues(response));
        }

        // Check for safety issues
        if self.config.check_safety {
            if let Some(issue) = self.check_safety_issues(response) {
                issues.push(issue);
            }
        }

        issues
    }

    /// Check for repetition
    fn check_repetition(&self, response: &str) -> Option<QualityIssue> {
        let sentences: Vec<&str> = response
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.len() < 3 {
            return None;
        }

        // Check for repeated sentences
        let mut seen: HashMap<String, usize> = HashMap::new();
        for sentence in &sentences {
            let normalized = sentence.to_lowercase().trim().to_string();
            *seen.entry(normalized).or_insert(0) += 1;
        }

        let repetitions = seen.values().filter(|&&c| c > 1).count();
        let ratio = repetitions as f32 / sentences.len() as f32;

        if ratio > self.config.repetition_threshold {
            Some(QualityIssue {
                issue_type: QualityIssueType::Repetition,
                description: format!("Response contains {} repeated sentences", repetitions),
                severity: 0.4,
                position: None,
                suggestion: Some("Avoid repeating the same information".to_string()),
            })
        } else {
            None
        }
    }

    /// Check if response contains code
    fn contains_code(&self, response: &str) -> bool {
        response.contains("```")
            || response.contains("fn ")
            || response.contains("def ")
            || response.contains("function ")
            || response.contains("class ")
            || response.contains("import ")
            || response.contains("let ")
            || response.contains("const ")
    }

    /// Check for code issues
    fn check_code_issues(&self, response: &str) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check for unbalanced brackets
        let open_parens = response.matches('(').count();
        let close_parens = response.matches(')').count();
        let open_braces = response.matches('{').count();
        let close_braces = response.matches('}').count();
        let open_brackets = response.matches('[').count();
        let close_brackets = response.matches(']').count();

        if open_parens != close_parens {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::CodeError,
                description: "Unbalanced parentheses in code".to_string(),
                severity: 0.5,
                position: None,
                suggestion: Some("Check opening and closing parentheses".to_string()),
            });
        }

        if open_braces != close_braces {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::CodeError,
                description: "Unbalanced braces in code".to_string(),
                severity: 0.5,
                position: None,
                suggestion: Some("Check opening and closing braces".to_string()),
            });
        }

        if open_brackets != close_brackets {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::CodeError,
                description: "Unbalanced brackets in code".to_string(),
                severity: 0.5,
                position: None,
                suggestion: Some("Check opening and closing brackets".to_string()),
            });
        }

        // Check for unclosed code blocks
        let code_block_count = response.matches("```").count();
        if code_block_count % 2 != 0 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::CodeError,
                description: "Unclosed code block".to_string(),
                severity: 0.4,
                position: None,
                suggestion: Some("Close the code block with ```".to_string()),
            });
        }

        issues
    }

    /// Check for safety issues
    fn check_safety_issues(&self, response: &str) -> Option<QualityIssue> {
        let response_lower = response.to_lowercase();

        // Check for potentially problematic patterns
        let concerning_patterns = [
            "i cannot help",
            "i can't help",
            "i am unable",
            "as an ai",
            "as a language model",
        ];

        for pattern in concerning_patterns {
            if response_lower.contains(pattern) {
                return Some(QualityIssue {
                    issue_type: QualityIssueType::SafetyIssue,
                    description: "Response contains refusal or disclaimer".to_string(),
                    severity: 0.2,
                    position: response_lower.find(pattern),
                    suggestion: None,
                });
            }
        }

        None
    }

    /// Find strengths in the response
    fn find_strengths(&self, _query: &str, response: &str) -> Vec<String> {
        let mut strengths = Vec::new();

        // Check for structured response
        if response.contains("\n- ") || response.contains("\n* ") || response.contains("\n1.") {
            strengths.push("Well-structured with bullet points or numbered lists".to_string());
        }

        // Check for code examples
        if response.contains("```") {
            strengths.push("Includes code examples".to_string());
        }

        // Check for explanations
        let explanation_words = [
            "because",
            "since",
            "therefore",
            "this means",
            "in other words",
        ];
        let response_lower = response.to_lowercase();
        if explanation_words.iter().any(|w| response_lower.contains(w)) {
            strengths.push("Provides clear explanations".to_string());
        }

        // Check for examples
        if response_lower.contains("for example") || response_lower.contains("e.g.") {
            strengths.push("Includes helpful examples".to_string());
        }

        // Check for good length
        let words = response.split_whitespace().count();
        if words >= 50 && words <= 500 {
            strengths.push("Appropriate response length".to_string());
        }

        strengths
    }
}

// ============================================================================
// Comparative Analysis
// ============================================================================

/// Compare multiple responses to find the best one
#[derive(Debug, Clone)]
pub struct ResponseComparison {
    /// Index of the best response
    pub best_index: usize,
    /// Scores for all responses
    pub scores: Vec<QualityScore>,
    /// Summary of comparison
    pub summary: String,
}

/// Compare multiple candidate responses
pub fn compare_responses(
    query: &str,
    responses: &[&str],
    context: Option<&str>,
) -> ResponseComparison {
    let analyzer = QualityAnalyzer::new(QualityConfig::default());

    let scores: Vec<QualityScore> = responses
        .iter()
        .map(|r| analyzer.analyze(query, r, context))
        .collect();

    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.overall
                .partial_cmp(&b.overall)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_score = scores.get(best_index).map(|s| s.overall).unwrap_or(0.0);

    let summary = format!(
        "Best response is #{} with score {:.2}. Quality: {}",
        best_index + 1,
        best_score,
        scores
            .get(best_index)
            .map(|s| s.quality_level())
            .unwrap_or("Unknown")
    );

    ResponseComparison {
        best_index,
        scores,
        summary,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_analysis() {
        let analyzer = QualityAnalyzer::new(QualityConfig::default());

        let query = "How do I create a function in Rust?";
        let response = "To create a function in Rust, you use the `fn` keyword followed by the function name. For example:\n\n```rust\nfn my_function() {\n    println!(\"Hello!\");\n}\n```\n\nFunctions can also take parameters and return values.";

        let score = analyzer.analyze(query, response, None);

        assert!(score.overall > 0.5);
        assert!(score.relevance > 0.5);
    }

    #[test]
    fn test_short_response_detection() {
        let analyzer = QualityAnalyzer::new(QualityConfig::default());

        let query = "Explain quantum computing in detail";
        let response = "It's complex.";

        let score = analyzer.analyze(query, response, None);

        assert!(score.completeness < 0.5);
        assert!(score
            .issues
            .iter()
            .any(|i| i.issue_type == QualityIssueType::TerseResponse));
    }

    #[test]
    fn test_code_issue_detection() {
        let analyzer = QualityAnalyzer::new(QualityConfig::default());

        let query = "Write a function";
        let response = "Here's the code:\n```rust\nfn test() {\n    println!(\"Hello\");\n// Missing closing brace";

        let score = analyzer.analyze(query, response, None);

        assert!(score
            .issues
            .iter()
            .any(|i| i.issue_type == QualityIssueType::CodeError));
    }

    #[test]
    fn test_response_comparison() {
        let query = "What is Rust?";
        let responses = [
            "Rust is a programming language.",
            "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents common bugs like null pointer dereferences and data races at compile time.",
        ];

        let comparison = compare_responses(query, &responses, None);

        assert_eq!(comparison.best_index, 1);
        assert!(comparison.scores[1].overall > comparison.scores[0].overall);
    }

    #[test]
    fn test_quality_score_levels() {
        let mut score = QualityScore::default();

        score.overall = 0.95;
        assert_eq!(score.quality_level(), "Excellent");

        score.overall = 0.8;
        assert_eq!(score.quality_level(), "Good");

        score.overall = 0.6;
        assert_eq!(score.quality_level(), "Acceptable");

        score.overall = 0.3;
        assert_eq!(score.quality_level(), "Poor");
    }
}
