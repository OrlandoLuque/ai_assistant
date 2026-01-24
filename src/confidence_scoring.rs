//! Confidence scoring for AI responses
//!
//! This module provides confidence scoring capabilities to estimate
//! how reliable an AI response is.
//!
//! # Features
//!
//! - **Token probability analysis**: Use logprobs when available
//! - **Linguistic markers**: Detect certainty/uncertainty language
//! - **Response consistency**: Compare multiple samples
//! - **Calibration**: Adjust scores based on historical accuracy

use std::collections::HashMap;

/// Configuration for confidence scoring
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Use token probabilities if available
    pub use_logprobs: bool,
    /// Analyze linguistic markers
    pub analyze_language: bool,
    /// Number of samples for consistency check
    pub consistency_samples: usize,
    /// Enable calibration
    pub enable_calibration: bool,
    /// Minimum confidence to consider reliable
    pub reliability_threshold: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            use_logprobs: true,
            analyze_language: true,
            consistency_samples: 0, // Disabled by default
            enable_calibration: false,
            reliability_threshold: 0.7,
        }
    }
}

/// Confidence score for a response
#[derive(Debug, Clone)]
pub struct ConfidenceScore {
    /// Overall confidence (0-1)
    pub overall: f64,
    /// Token-level confidence if available
    pub token_confidence: Option<f64>,
    /// Linguistic confidence
    pub linguistic_confidence: f64,
    /// Consistency confidence (if multi-sampled)
    pub consistency_confidence: Option<f64>,
    /// Calibrated confidence
    pub calibrated: Option<f64>,
    /// Breakdown by component
    pub breakdown: ConfidenceBreakdown,
    /// Reliability assessment
    pub reliability: Reliability,
}

/// Breakdown of confidence components
#[derive(Debug, Clone)]
pub struct ConfidenceBreakdown {
    /// Certainty indicators found
    pub certainty_indicators: Vec<CertaintyIndicator>,
    /// Uncertainty indicators found
    pub uncertainty_indicators: Vec<UncertaintyIndicator>,
    /// Hedging phrases
    pub hedging_count: usize,
    /// Strong assertions
    pub assertion_count: usize,
    /// Questions/qualifications
    pub qualification_count: usize,
}

/// Certainty indicator
#[derive(Debug, Clone)]
pub struct CertaintyIndicator {
    /// The indicator text
    pub text: String,
    /// Weight (how strong this indicator is)
    pub weight: f64,
    /// Position in text
    pub position: usize,
}

/// Uncertainty indicator
#[derive(Debug, Clone)]
pub struct UncertaintyIndicator {
    /// The indicator text
    pub text: String,
    /// Weight (how strong this indicator is)
    pub weight: f64,
    /// Position in text
    pub position: usize,
    /// Type of uncertainty
    pub uncertainty_type: UncertaintyType,
}

/// Types of uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyType {
    /// Hedging language
    Hedging,
    /// Qualification
    Qualification,
    /// Explicit uncertainty
    ExplicitUncertainty,
    /// Conditional statement
    Conditional,
    /// Approximation
    Approximation,
}

/// Reliability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Highly reliable
    High,
    /// Moderately reliable
    Medium,
    /// Low reliability
    Low,
    /// Unreliable
    Unreliable,
}

impl Reliability {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.85 {
            Reliability::High
        } else if score >= 0.65 {
            Reliability::Medium
        } else if score >= 0.45 {
            Reliability::Low
        } else {
            Reliability::Unreliable
        }
    }
}

/// Confidence scorer
pub struct ConfidenceScorer {
    config: ConfidenceConfig,
    /// Certainty phrases and weights
    certainty_phrases: Vec<(&'static str, f64)>,
    /// Uncertainty phrases and weights
    uncertainty_phrases: Vec<(&'static str, f64, UncertaintyType)>,
    /// Historical accuracy for calibration
    calibration_data: CalibrationData,
}

/// Calibration data
#[derive(Debug, Clone, Default)]
struct CalibrationData {
    /// Total predictions
    total: usize,
    /// Correct predictions by confidence bucket
    correct_by_bucket: HashMap<u8, usize>,
    /// Total by confidence bucket
    total_by_bucket: HashMap<u8, usize>,
}

impl ConfidenceScorer {
    /// Create a new confidence scorer
    pub fn new(config: ConfidenceConfig) -> Self {
        let certainty_phrases = vec![
            ("definitely", 0.9),
            ("certainly", 0.85),
            ("absolutely", 0.9),
            ("undoubtedly", 0.85),
            ("clearly", 0.7),
            ("obviously", 0.7),
            ("without doubt", 0.85),
            ("it is", 0.5),
            ("the answer is", 0.6),
            ("this is", 0.5),
            ("always", 0.8),
            ("never", 0.8),
        ];

        let uncertainty_phrases = vec![
            ("I think", 0.4, UncertaintyType::Hedging),
            ("I believe", 0.4, UncertaintyType::Hedging),
            ("probably", 0.5, UncertaintyType::Hedging),
            ("likely", 0.5, UncertaintyType::Hedging),
            ("possibly", 0.6, UncertaintyType::Hedging),
            ("perhaps", 0.5, UncertaintyType::Hedging),
            ("maybe", 0.6, UncertaintyType::Hedging),
            ("might", 0.5, UncertaintyType::Hedging),
            ("could be", 0.5, UncertaintyType::Hedging),
            ("may be", 0.5, UncertaintyType::Hedging),
            ("I'm not sure", 0.7, UncertaintyType::ExplicitUncertainty),
            ("I don't know", 0.9, UncertaintyType::ExplicitUncertainty),
            ("unclear", 0.6, UncertaintyType::ExplicitUncertainty),
            ("uncertain", 0.7, UncertaintyType::ExplicitUncertainty),
            ("approximately", 0.3, UncertaintyType::Approximation),
            ("about", 0.2, UncertaintyType::Approximation),
            ("around", 0.2, UncertaintyType::Approximation),
            ("roughly", 0.3, UncertaintyType::Approximation),
            ("if", 0.3, UncertaintyType::Conditional),
            ("assuming", 0.4, UncertaintyType::Conditional),
            ("depending on", 0.4, UncertaintyType::Conditional),
            ("however", 0.2, UncertaintyType::Qualification),
            ("although", 0.2, UncertaintyType::Qualification),
            ("but", 0.1, UncertaintyType::Qualification),
        ];

        Self {
            config,
            certainty_phrases,
            uncertainty_phrases,
            calibration_data: CalibrationData::default(),
        }
    }

    /// Score confidence of a response
    pub fn score(&self, text: &str, logprobs: Option<&[f64]>) -> ConfidenceScore {
        let mut breakdown = ConfidenceBreakdown {
            certainty_indicators: Vec::new(),
            uncertainty_indicators: Vec::new(),
            hedging_count: 0,
            assertion_count: 0,
            qualification_count: 0,
        };

        // Analyze linguistic markers
        let linguistic_confidence = if self.config.analyze_language {
            self.analyze_linguistic_markers(text, &mut breakdown)
        } else {
            0.5
        };

        // Analyze token probabilities
        let token_confidence = if self.config.use_logprobs {
            logprobs.map(|lp| self.analyze_logprobs(lp))
        } else {
            None
        };

        // Calculate overall confidence
        let overall = self.calculate_overall(linguistic_confidence, token_confidence);

        // Calibrate if enabled
        let calibrated = if self.config.enable_calibration {
            Some(self.calibrate(overall))
        } else {
            None
        };

        let final_score = calibrated.unwrap_or(overall);
        let reliability = Reliability::from_score(final_score);

        ConfidenceScore {
            overall: final_score,
            token_confidence,
            linguistic_confidence,
            consistency_confidence: None, // Would need multiple samples
            calibrated,
            breakdown,
            reliability,
        }
    }

    /// Analyze linguistic markers
    fn analyze_linguistic_markers(&self, text: &str, breakdown: &mut ConfidenceBreakdown) -> f64 {
        let lower = text.to_lowercase();
        let mut certainty_total = 0.0;
        let mut uncertainty_total = 0.0;

        // Find certainty indicators
        for (phrase, weight) in &self.certainty_phrases {
            if let Some(pos) = lower.find(&phrase.to_lowercase()) {
                breakdown.certainty_indicators.push(CertaintyIndicator {
                    text: phrase.to_string(),
                    weight: *weight,
                    position: pos,
                });
                certainty_total += weight;
                breakdown.assertion_count += 1;
            }
        }

        // Find uncertainty indicators
        for (phrase, weight, utype) in &self.uncertainty_phrases {
            if let Some(pos) = lower.find(&phrase.to_lowercase()) {
                breakdown.uncertainty_indicators.push(UncertaintyIndicator {
                    text: phrase.to_string(),
                    weight: *weight,
                    position: pos,
                    uncertainty_type: *utype,
                });
                uncertainty_total += weight;

                match utype {
                    UncertaintyType::Hedging => breakdown.hedging_count += 1,
                    UncertaintyType::Qualification => breakdown.qualification_count += 1,
                    _ => {}
                }
            }
        }

        // Calculate linguistic confidence
        if certainty_total == 0.0 && uncertainty_total == 0.0 {
            0.5 // Neutral
        } else {
            let confidence = (certainty_total - uncertainty_total * 0.5) / (certainty_total + uncertainty_total).max(1.0);
            (confidence + 1.0) / 2.0 // Normalize to 0-1
        }
    }

    /// Analyze log probabilities
    fn analyze_logprobs(&self, logprobs: &[f64]) -> f64 {
        if logprobs.is_empty() {
            return 0.5;
        }

        // Convert log probs to probabilities and average
        let probs: Vec<f64> = logprobs.iter().map(|lp| lp.exp()).collect();
        let avg_prob = probs.iter().sum::<f64>() / probs.len() as f64;

        // Consider variance
        let variance = probs.iter()
            .map(|p| (p - avg_prob).powi(2))
            .sum::<f64>() / probs.len() as f64;

        // High variance indicates uncertainty
        let variance_penalty = variance.sqrt() * 0.5;

        (avg_prob - variance_penalty).max(0.0).min(1.0)
    }

    /// Calculate overall confidence
    fn calculate_overall(&self, linguistic: f64, token: Option<f64>) -> f64 {
        match token {
            Some(t) => (linguistic * 0.4 + t * 0.6), // Token probs weighted higher
            None => linguistic,
        }
    }

    /// Apply calibration
    fn calibrate(&self, raw_score: f64) -> f64 {
        let bucket = (raw_score * 10.0) as u8;

        if let (Some(correct), Some(total)) = (
            self.calibration_data.correct_by_bucket.get(&bucket),
            self.calibration_data.total_by_bucket.get(&bucket),
        ) {
            if *total > 0 {
                return *correct as f64 / *total as f64;
            }
        }

        raw_score
    }

    /// Record outcome for calibration
    pub fn record_outcome(&mut self, predicted_confidence: f64, was_correct: bool) {
        let bucket = (predicted_confidence * 10.0) as u8;
        self.calibration_data.total += 1;

        *self.calibration_data.total_by_bucket.entry(bucket).or_insert(0) += 1;
        if was_correct {
            *self.calibration_data.correct_by_bucket.entry(bucket).or_insert(0) += 1;
        }
    }

    /// Get calibration statistics
    pub fn calibration_stats(&self) -> CalibrationStats {
        let mut bucket_accuracy = HashMap::new();

        for bucket in 0..=10 {
            if let (Some(correct), Some(total)) = (
                self.calibration_data.correct_by_bucket.get(&bucket),
                self.calibration_data.total_by_bucket.get(&bucket),
            ) {
                if *total > 0 {
                    bucket_accuracy.insert(bucket, *correct as f64 / *total as f64);
                }
            }
        }

        CalibrationStats {
            total_predictions: self.calibration_data.total,
            accuracy_by_bucket: bucket_accuracy,
        }
    }

    /// Score confidence with multiple samples
    pub fn score_with_consistency<F>(
        &self,
        prompt: &str,
        generate: F,
    ) -> ConfidenceScore
    where
        F: Fn(&str) -> (String, Option<Vec<f64>>),
    {
        if self.config.consistency_samples == 0 {
            let (response, logprobs) = generate(prompt);
            return self.score(&response, logprobs.as_deref());
        }

        let mut responses = Vec::new();
        let mut scores = Vec::new();

        for _ in 0..self.config.consistency_samples {
            let (response, logprobs) = generate(prompt);
            let score = self.score(&response, logprobs.as_deref());
            responses.push(response);
            scores.push(score);
        }

        // Calculate consistency
        let consistency = self.calculate_consistency(&responses);
        let avg_score = scores.iter().map(|s| s.overall).sum::<f64>() / scores.len() as f64;

        let mut final_score = scores.into_iter().next().unwrap_or_else(|| {
            ConfidenceScore {
                overall: 0.5,
                token_confidence: None,
                linguistic_confidence: 0.5,
                consistency_confidence: None,
                calibrated: None,
                breakdown: ConfidenceBreakdown {
                    certainty_indicators: Vec::new(),
                    uncertainty_indicators: Vec::new(),
                    hedging_count: 0,
                    assertion_count: 0,
                    qualification_count: 0,
                },
                reliability: Reliability::Medium,
            }
        });

        final_score.consistency_confidence = Some(consistency);
        final_score.overall = (avg_score * 0.7 + consistency * 0.3);
        final_score.reliability = Reliability::from_score(final_score.overall);

        final_score
    }

    /// Calculate consistency between responses
    fn calculate_consistency(&self, responses: &[String]) -> f64 {
        if responses.len() < 2 {
            return 1.0;
        }

        let mut similarity_sum = 0.0;
        let mut count = 0;

        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                similarity_sum += self.text_similarity(&responses[i], &responses[j]);
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            similarity_sum / count as f64
        }
    }

    /// Simple text similarity (Jaccard)
    fn text_similarity(&self, a: &str, b: &str) -> f64 {
        let lower_a = a.to_lowercase();
        let lower_b = b.to_lowercase();
        let words_a: std::collections::HashSet<_> = lower_a.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = lower_b.split_whitespace().collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new(ConfidenceConfig::default())
    }
}

/// Calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Total predictions made
    pub total_predictions: usize,
    /// Accuracy by confidence bucket
    pub accuracy_by_bucket: HashMap<u8, f64>,
}

/// Builder for confidence configuration
pub struct ConfidenceConfigBuilder {
    config: ConfidenceConfig,
}

impl ConfidenceConfigBuilder {
    pub fn new() -> Self {
        Self { config: ConfidenceConfig::default() }
    }

    pub fn use_logprobs(mut self, use_it: bool) -> Self {
        self.config.use_logprobs = use_it;
        self
    }

    pub fn analyze_language(mut self, analyze: bool) -> Self {
        self.config.analyze_language = analyze;
        self
    }

    pub fn consistency_samples(mut self, samples: usize) -> Self {
        self.config.consistency_samples = samples;
        self
    }

    pub fn enable_calibration(mut self, enable: bool) -> Self {
        self.config.enable_calibration = enable;
        self
    }

    pub fn reliability_threshold(mut self, threshold: f64) -> Self {
        self.config.reliability_threshold = threshold;
        self
    }

    pub fn build(self) -> ConfidenceConfig {
        self.config
    }
}

impl Default for ConfidenceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_confidence() {
        let scorer = ConfidenceScorer::default();
        let result = scorer.score("The answer is definitely 42.", None);

        assert!(result.linguistic_confidence > 0.5);
        assert!(!result.breakdown.certainty_indicators.is_empty());
    }

    #[test]
    fn test_low_confidence() {
        let scorer = ConfidenceScorer::default();
        let result = scorer.score("I think the answer might be 42, but I'm not sure.", None);

        assert!(result.linguistic_confidence < 0.5);
        assert!(!result.breakdown.uncertainty_indicators.is_empty());
    }

    #[test]
    fn test_neutral() {
        let scorer = ConfidenceScorer::default();
        let result = scorer.score("The value is 42.", None);

        assert!(result.linguistic_confidence >= 0.4 && result.linguistic_confidence <= 0.6);
    }

    #[test]
    fn test_with_logprobs() {
        let scorer = ConfidenceScorer::default();
        let logprobs = vec![-0.1, -0.2, -0.1, -0.15]; // High confidence logprobs

        let result = scorer.score("The answer is 42.", Some(&logprobs));

        assert!(result.token_confidence.is_some());
        assert!(result.token_confidence.unwrap() > 0.7);
    }

    #[test]
    fn test_reliability_levels() {
        assert_eq!(Reliability::from_score(0.9), Reliability::High);
        assert_eq!(Reliability::from_score(0.7), Reliability::Medium);
        assert_eq!(Reliability::from_score(0.5), Reliability::Low);
        assert_eq!(Reliability::from_score(0.2), Reliability::Unreliable);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfidenceConfigBuilder::new()
            .use_logprobs(true)
            .analyze_language(true)
            .consistency_samples(3)
            .build();

        assert!(config.use_logprobs);
        assert_eq!(config.consistency_samples, 3);
    }
}
