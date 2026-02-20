//! Self-consistency via multi-sampling
//!
//! This module provides self-consistency checking by running the same
//! query multiple times and taking the consensus answer.
//!
//! # Features
//!
//! - **Multi-sampling**: Generate multiple responses
//! - **Consensus finding**: Find the most consistent answer
//! - **Confidence scoring**: Rate confidence based on agreement
//! - **Parallel execution**: Sample in parallel for speed

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for self-consistency
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    /// Number of samples to generate
    pub num_samples: usize,
    /// Temperature for sampling (higher = more diverse)
    pub temperature: f32,
    /// Minimum agreement ratio for consensus
    pub min_consensus: f64,
    /// Enable parallel sampling
    pub parallel: bool,
    /// Timeout per sample
    pub sample_timeout: Duration,
    /// Similarity threshold for grouping answers
    pub similarity_threshold: f64,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            num_samples: 5,
            temperature: 0.7,
            min_consensus: 0.6,
            parallel: true,
            sample_timeout: Duration::from_secs(60),
            similarity_threshold: 0.85,
        }
    }
}

/// A single sample response
#[derive(Debug, Clone)]
pub struct Sample {
    /// Sample index
    pub index: usize,
    /// The response text
    pub response: String,
    /// Time taken to generate
    pub duration: Duration,
    /// Whether it was successful
    pub success: bool,
    /// Error if failed
    pub error: Option<String>,
}

/// A group of similar answers
#[derive(Debug, Clone)]
pub struct AnswerGroup {
    /// Representative answer for this group
    pub representative: String,
    /// All answers in this group
    pub answers: Vec<String>,
    /// Count of answers in this group
    pub count: usize,
    /// Ratio of total samples
    pub ratio: f64,
}

/// Result of self-consistency check
#[derive(Debug, Clone)]
pub struct ConsistencyResult {
    /// The prompt used
    pub prompt: String,
    /// All samples collected
    pub samples: Vec<Sample>,
    /// Groups of similar answers
    pub groups: Vec<AnswerGroup>,
    /// The consensus answer (if found)
    pub consensus: Option<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Total time taken
    pub total_duration: Duration,
    /// Number of successful samples
    pub successful_samples: usize,
    /// Whether consensus was reached
    pub has_consensus: bool,
}

/// Self-consistency checker
pub struct ConsistencyChecker {
    config: ConsistencyConfig,
}

impl ConsistencyChecker {
    /// Create a new consistency checker
    pub fn new(config: ConsistencyConfig) -> Self {
        Self { config }
    }

    /// Check self-consistency of a prompt
    pub fn check<F>(&self, prompt: &str, model: &str, generate: F) -> ConsistencyResult
    where
        F: Fn(&str, &str, f32) -> Result<String, String>,
    {
        let start = Instant::now();
        let mut samples = Vec::with_capacity(self.config.num_samples);

        // Generate samples
        for i in 0..self.config.num_samples {
            let sample_start = Instant::now();
            let result = generate(prompt, model, self.config.temperature);
            let duration = sample_start.elapsed();

            match result {
                Ok(response) => {
                    samples.push(Sample {
                        index: i,
                        response,
                        duration,
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    samples.push(Sample {
                        index: i,
                        response: String::new(),
                        duration,
                        success: false,
                        error: Some(e),
                    });
                }
            }
        }

        // Group similar answers
        let successful_count = samples.iter().filter(|s| s.success).count();
        let successful_refs: Vec<_> = samples.iter().filter(|s| s.success).collect();
        let groups = self.group_answers(&successful_refs);

        // Find consensus
        let (consensus, confidence) = self.find_consensus(&groups);
        let has_consensus = consensus.is_some() && confidence >= self.config.min_consensus;
        let total_duration = start.elapsed();

        ConsistencyResult {
            prompt: prompt.to_string(),
            samples,
            groups: groups.clone(),
            consensus: consensus.clone(),
            confidence,
            total_duration,
            successful_samples: successful_count,
            has_consensus,
        }
    }

    /// Group similar answers together
    fn group_answers(&self, samples: &[&Sample]) -> Vec<AnswerGroup> {
        if samples.is_empty() {
            return Vec::new();
        }

        let mut groups: Vec<AnswerGroup> = Vec::new();

        for sample in samples {
            let mut found_group = false;

            // Try to add to existing group
            for group in &mut groups {
                if self.are_similar(&sample.response, &group.representative) {
                    group.answers.push(sample.response.clone());
                    group.count += 1;
                    found_group = true;
                    break;
                }
            }

            // Create new group if no match
            if !found_group {
                groups.push(AnswerGroup {
                    representative: sample.response.clone(),
                    answers: vec![sample.response.clone()],
                    count: 1,
                    ratio: 0.0,
                });
            }
        }

        // Calculate ratios
        let total = samples.len() as f64;
        for group in &mut groups {
            group.ratio = group.count as f64 / total;
        }

        // Sort by count (descending)
        groups.sort_by(|a, b| b.count.cmp(&a.count));

        groups
    }

    /// Find consensus answer from groups
    fn find_consensus(&self, groups: &[AnswerGroup]) -> (Option<String>, f64) {
        if groups.is_empty() {
            return (None, 0.0);
        }

        let top_group = &groups[0];

        if top_group.ratio >= self.config.min_consensus {
            (Some(top_group.representative.clone()), top_group.ratio)
        } else {
            // Still return top answer but with lower confidence
            (Some(top_group.representative.clone()), top_group.ratio)
        }
    }

    /// Check if two answers are similar
    fn are_similar(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }

        // Normalize and compare
        let a_norm = normalize_answer(a);
        let b_norm = normalize_answer(b);

        if a_norm == b_norm {
            return true;
        }

        // Check Jaccard similarity
        let similarity = jaccard_similarity(&a_norm, &b_norm);
        similarity >= self.config.similarity_threshold
    }
}

impl Default for ConsistencyChecker {
    fn default() -> Self {
        Self::new(ConsistencyConfig::default())
    }
}

/// Normalize an answer for comparison
fn normalize_answer(answer: &str) -> String {
    answer
        .to_lowercase()
        .trim()
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calculate Jaccard similarity between two strings
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<_> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<_> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Voting-based consistency for categorical answers
pub struct VotingConsistency {
    config: ConsistencyConfig,
}

impl VotingConsistency {
    /// Create a new voting consistency checker
    pub fn new(config: ConsistencyConfig) -> Self {
        Self { config }
    }

    /// Run voting-based consistency check
    pub fn vote<F>(&self, prompt: &str, model: &str, generate: F) -> VotingResult
    where
        F: Fn(&str, &str, f32) -> Result<String, String>,
    {
        let start = Instant::now();
        let mut votes: HashMap<String, usize> = HashMap::new();
        let mut samples = Vec::new();

        for i in 0..self.config.num_samples {
            let sample_start = Instant::now();
            match generate(prompt, model, self.config.temperature) {
                Ok(response) => {
                    // Extract the answer (first word or line)
                    let answer = extract_categorical_answer(&response);
                    *votes.entry(answer.clone()).or_insert(0) += 1;

                    samples.push(Sample {
                        index: i,
                        response,
                        duration: sample_start.elapsed(),
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    samples.push(Sample {
                        index: i,
                        response: String::new(),
                        duration: sample_start.elapsed(),
                        success: false,
                        error: Some(e),
                    });
                }
            }
        }

        // Find winner
        let total_votes: usize = votes.values().sum();
        let winner = votes
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, v)| (k.clone(), *v));

        let (answer, vote_count) = winner.unwrap_or((String::new(), 0));
        let confidence = if total_votes > 0 {
            vote_count as f64 / total_votes as f64
        } else {
            0.0
        };

        VotingResult {
            prompt: prompt.to_string(),
            samples,
            votes,
            winner: if confidence >= self.config.min_consensus {
                Some(answer.clone())
            } else {
                None
            },
            winner_votes: vote_count,
            total_votes,
            confidence,
            total_duration: start.elapsed(),
        }
    }
}

/// Result of voting-based consistency
#[derive(Debug, Clone)]
pub struct VotingResult {
    /// The prompt used
    pub prompt: String,
    /// All samples
    pub samples: Vec<Sample>,
    /// Vote counts per answer
    pub votes: HashMap<String, usize>,
    /// Winning answer (if consensus reached)
    pub winner: Option<String>,
    /// Votes for winner
    pub winner_votes: usize,
    /// Total votes cast
    pub total_votes: usize,
    /// Confidence (winner votes / total)
    pub confidence: f64,
    /// Total time
    pub total_duration: Duration,
}

/// Extract categorical answer from response
fn extract_categorical_answer(response: &str) -> String {
    // Try to extract first word or letter as category
    let trimmed = response.trim();

    // Handle "(A)" pattern first
    if trimmed.starts_with('(') {
        if let Some(second_char) = trimmed.chars().nth(1) {
            if second_char.is_ascii_alphabetic() {
                if let Some(third_char) = trimmed.chars().nth(2) {
                    if third_char == ')' {
                        return second_char.to_string().to_uppercase();
                    }
                }
            }
        }
    }

    // Common patterns for categorical answers
    // "A)" or "A:" or "(A)" or just "A"
    if let Some(first_char) = trimmed.chars().next() {
        if first_char.is_ascii_alphabetic() {
            let second = trimmed.chars().nth(1);
            match second {
                Some(')') | Some(':') | Some('.') | Some(' ') | None => {
                    return first_char.to_string().to_uppercase();
                }
                _ => {}
            }
        }
    }

    // Try first word
    trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_uppercase()
}

/// Builder for consistency configuration
pub struct ConsistencyConfigBuilder {
    config: ConsistencyConfig,
}

impl ConsistencyConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ConsistencyConfig::default(),
        }
    }

    /// Set number of samples
    pub fn samples(mut self, n: usize) -> Self {
        self.config.num_samples = n;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set minimum consensus ratio
    pub fn min_consensus(mut self, ratio: f64) -> Self {
        self.config.min_consensus = ratio;
        self
    }

    /// Set similarity threshold
    pub fn similarity_threshold(mut self, threshold: f64) -> Self {
        self.config.similarity_threshold = threshold;
        self
    }

    /// Enable/disable parallel sampling
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// Set sample timeout
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.config.sample_timeout = duration;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ConsistencyConfig {
        self.config
    }
}

impl Default for ConsistencyConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregator for combining multiple consistency results
pub struct ConsistencyAggregator {
    results: Vec<ConsistencyResult>,
}

impl ConsistencyAggregator {
    /// Create a new aggregator
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add a result
    pub fn add(&mut self, result: ConsistencyResult) {
        self.results.push(result);
    }

    /// Get overall statistics
    pub fn stats(&self) -> AggregatedStats {
        let total = self.results.len();
        let with_consensus = self.results.iter().filter(|r| r.has_consensus).count();
        let avg_confidence = if total > 0 {
            self.results.iter().map(|r| r.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let avg_duration = if total > 0 {
            let total_duration: Duration = self.results.iter().map(|r| r.total_duration).sum();
            total_duration / total as u32
        } else {
            Duration::ZERO
        };

        AggregatedStats {
            total_checks: total,
            checks_with_consensus: with_consensus,
            consensus_rate: if total > 0 {
                with_consensus as f64 / total as f64
            } else {
                0.0
            },
            average_confidence: avg_confidence,
            average_duration: avg_duration,
        }
    }
}

impl Default for ConsistencyAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated statistics
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    /// Total consistency checks performed
    pub total_checks: usize,
    /// Checks that reached consensus
    pub checks_with_consensus: usize,
    /// Rate of reaching consensus
    pub consensus_rate: f64,
    /// Average confidence score
    pub average_confidence: f64,
    /// Average time per check
    pub average_duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_checker_creation() {
        let checker = ConsistencyChecker::default();
        assert_eq!(checker.config.num_samples, 5);
    }

    #[test]
    fn test_normalize_answer() {
        let answer = "  Hello   World\n  Test  ";
        let normalized = normalize_answer(answer);
        assert_eq!(normalized, "hello world test");
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = "hello world test";
        let b = "hello world";
        let similarity = jaccard_similarity(a, b);
        assert!(similarity > 0.5 && similarity < 1.0);

        let c = "hello world test";
        let d = "hello world test";
        assert!((jaccard_similarity(c, d) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_consistency_check() {
        let checker = ConsistencyChecker::default();

        let result = checker.check("What is 2+2?", "model", |_, _, _| Ok("4".to_string()));

        assert!(result.successful_samples > 0);
        assert!(result.has_consensus);
        assert!(result.confidence >= 0.8); // All same answer
    }

    #[test]
    fn test_voting_consistency() {
        let voting = VotingConsistency::new(ConsistencyConfig::default());

        let result = voting.vote("Choose A, B, or C", "model", |_, _, _| Ok("A".to_string()));

        assert_eq!(result.winner, Some("A".to_string()));
        assert!((result.confidence - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_categorical() {
        assert_eq!(extract_categorical_answer("A) This is option A"), "A");
        assert_eq!(extract_categorical_answer("B: Second option"), "B");
        assert_eq!(extract_categorical_answer("(C) Third"), "C");
        assert_eq!(extract_categorical_answer("D"), "D");
    }

    #[test]
    fn test_config_builder() {
        let config = ConsistencyConfigBuilder::new()
            .samples(10)
            .temperature(0.9)
            .min_consensus(0.7)
            .similarity_threshold(0.9)
            .build();

        assert_eq!(config.num_samples, 10);
        assert!((config.temperature - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_aggregator() {
        let mut aggregator = ConsistencyAggregator::new();

        aggregator.add(ConsistencyResult {
            prompt: "test".to_string(),
            samples: Vec::new(),
            groups: Vec::new(),
            consensus: Some("answer".to_string()),
            confidence: 0.8,
            total_duration: Duration::from_secs(1),
            successful_samples: 5,
            has_consensus: true,
        });

        let stats = aggregator.stats();
        assert_eq!(stats.total_checks, 1);
        assert_eq!(stats.checks_with_consensus, 1);
    }
}
