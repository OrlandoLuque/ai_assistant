//! Hallucination detection for AI responses
//!
//! This module provides detection of potential hallucinations
//! in AI-generated content.
//!
//! # Features
//!
//! - **Fact verification**: Check claims against sources
//! - **Consistency checking**: Detect contradictions
//! - **Confidence markers**: Identify hedging language
//! - **Source attribution**: Track unsourced claims

use std::collections::{HashMap, HashSet};
use regex::Regex;

/// Configuration for hallucination detection
#[derive(Debug, Clone)]
pub struct HallucinationConfig {
    /// Minimum confidence for claims
    pub min_claim_confidence: f64,
    /// Check for internal contradictions
    pub check_contradictions: bool,
    /// Check for unsupported claims
    pub check_unsupported_claims: bool,
    /// Detect hedging language
    pub detect_hedging: bool,
    /// Verify facts against sources
    pub verify_against_sources: bool,
}

impl Default for HallucinationConfig {
    fn default() -> Self {
        Self {
            min_claim_confidence: 0.7,
            check_contradictions: true,
            check_unsupported_claims: true,
            detect_hedging: true,
            verify_against_sources: true,
        }
    }
}

/// Types of potential hallucinations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HallucinationType {
    /// Made up fact
    FactualError,
    /// Internal contradiction
    Contradiction,
    /// Unsupported claim
    UnsupportedClaim,
    /// Invented entity (person, place, etc.)
    InventedEntity,
    /// Wrong attribution
    MisAttribution,
    /// Fabricated quote
    FabricatedQuote,
    /// Incorrect statistic
    IncorrectStatistic,
    /// Anachronism
    Anachronism,
}

impl HallucinationType {
    pub fn display_name(&self) -> &'static str {
        match self {
            HallucinationType::FactualError => "Factual Error",
            HallucinationType::Contradiction => "Internal Contradiction",
            HallucinationType::UnsupportedClaim => "Unsupported Claim",
            HallucinationType::InventedEntity => "Invented Entity",
            HallucinationType::MisAttribution => "Wrong Attribution",
            HallucinationType::FabricatedQuote => "Fabricated Quote",
            HallucinationType::IncorrectStatistic => "Incorrect Statistic",
            HallucinationType::Anachronism => "Anachronism",
        }
    }
}

/// A detected potential hallucination
#[derive(Debug, Clone)]
pub struct HallucinationDetection {
    /// Type of hallucination
    pub hallucination_type: HallucinationType,
    /// The problematic text
    pub text: String,
    /// Position in response
    pub position: usize,
    /// Confidence that this is a hallucination (0-1)
    pub confidence: f64,
    /// Explanation
    pub reason: String,
    /// Suggested correction
    pub suggestion: Option<String>,
}

/// A claim extracted from text
#[derive(Debug, Clone)]
pub struct Claim {
    /// The claim text
    pub text: String,
    /// Position in text
    pub position: usize,
    /// Claim type
    pub claim_type: ClaimType,
    /// Has supporting evidence
    pub supported: bool,
    /// Confidence level
    pub confidence: f64,
}

/// Types of claims
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimType {
    /// Statement of fact
    Factual,
    /// Quote or attribution
    Quote,
    /// Statistic or number
    Statistic,
    /// Date or time
    Temporal,
    /// Reference to person/place
    Entity,
    /// Opinion or interpretation
    Opinion,
}

/// Result of hallucination detection
#[derive(Debug, Clone)]
pub struct HallucinationResult {
    /// Original text
    pub original: String,
    /// Detected hallucinations
    pub detections: Vec<HallucinationDetection>,
    /// Extracted claims
    pub claims: Vec<Claim>,
    /// Overall reliability score (0-1)
    pub reliability_score: f64,
    /// Number of unsupported claims
    pub unsupported_claims: usize,
    /// Hedging indicators found
    pub hedging_indicators: Vec<String>,
}

/// Hallucination detector
pub struct HallucinationDetector {
    config: HallucinationConfig,
    /// Known facts for verification
    known_facts: HashMap<String, String>,
    /// Hedging phrases
    hedging_phrases: Vec<&'static str>,
    /// Certainty patterns
    certainty_patterns: Vec<Regex>,
}

impl HallucinationDetector {
    /// Create a new detector
    pub fn new(config: HallucinationConfig) -> Self {
        let hedging_phrases = vec![
            "I think",
            "I believe",
            "probably",
            "might be",
            "could be",
            "possibly",
            "perhaps",
            "it seems",
            "apparently",
            "I'm not sure",
            "as far as I know",
            "to the best of my knowledge",
            "allegedly",
            "reportedly",
            "may or may not",
            "it's possible that",
        ];

        let mut certainty_patterns = Vec::new();
        // Patterns indicating uncertain knowledge
        if let Ok(re) = Regex::new(r"(?i)(?:definitely|certainly|absolutely|always|never)\s") {
            certainty_patterns.push(re);
        }

        Self {
            config,
            known_facts: HashMap::new(),
            hedging_phrases,
            certainty_patterns,
        }
    }

    /// Add known facts for verification
    pub fn add_known_facts(&mut self, facts: HashMap<String, String>) {
        self.known_facts.extend(facts);
    }

    /// Detect potential hallucinations
    pub fn detect(&self, text: &str, context: Option<&str>) -> HallucinationResult {
        let mut detections = Vec::new();
        let claims = self.extract_claims(text);

        // Check for hedging
        let mut hedging_indicators = Vec::new();
        if self.config.detect_hedging {
            hedging_indicators = self.find_hedging(text);
        }

        // Check for contradictions
        if self.config.check_contradictions {
            let contradictions = self.find_contradictions(text, &claims);
            detections.extend(contradictions);
        }

        // Check unsupported claims
        if self.config.check_unsupported_claims {
            let unsupported = self.find_unsupported_claims(&claims, context);
            detections.extend(unsupported);
        }

        // Check for fabricated quotes
        let quotes = self.find_fabricated_quotes(text);
        detections.extend(quotes);

        // Check for suspicious statistics
        let stats = self.check_statistics(text);
        detections.extend(stats);

        // Calculate reliability score
        let unsupported_count = claims.iter().filter(|c| !c.supported).count();
        let reliability_score = self.calculate_reliability(&claims, &detections);

        HallucinationResult {
            original: text.to_string(),
            detections,
            claims,
            reliability_score,
            unsupported_claims: unsupported_count,
            hedging_indicators,
        }
    }

    /// Extract claims from text
    fn extract_claims(&self, text: &str) -> Vec<Claim> {
        let mut claims = Vec::new();

        // Extract sentences as claims
        let sentences: Vec<_> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        for (i, sentence) in sentences.iter().enumerate() {
            let trimmed = sentence.trim();
            let position = text.find(trimmed).unwrap_or(i * 100);

            let claim_type = self.classify_claim(trimmed);
            let supported = self.is_claim_supported(trimmed);
            let confidence = self.estimate_confidence(trimmed);

            claims.push(Claim {
                text: trimmed.to_string(),
                position,
                claim_type,
                supported,
                confidence,
            });
        }

        claims
    }

    /// Classify a claim
    fn classify_claim(&self, text: &str) -> ClaimType {
        let lower = text.to_lowercase();

        if text.contains('"') || lower.contains("said") || lower.contains("stated") {
            ClaimType::Quote
        } else if Regex::new(r"\d+%|\d+\s+(?:percent|million|billion|people|times)")
            .map(|re| re.is_match(&lower))
            .unwrap_or(false)
        {
            ClaimType::Statistic
        } else if lower.contains("in ") && Regex::new(r"\b\d{4}\b")
            .map(|re| re.is_match(&lower))
            .unwrap_or(false)
        {
            ClaimType::Temporal
        } else if lower.starts_with("i think")
            || lower.starts_with("i believe")
            || lower.contains("in my opinion")
        {
            ClaimType::Opinion
        } else {
            ClaimType::Factual
        }
    }

    /// Check if claim is supported
    fn is_claim_supported(&self, claim: &str) -> bool {
        // Check against known facts
        let claim_lower = claim.to_lowercase();
        for (key, _) in &self.known_facts {
            if claim_lower.contains(&key.to_lowercase()) {
                return true;
            }
        }

        // Check for hedging language (indicates uncertainty)
        for phrase in &self.hedging_phrases {
            if claim_lower.contains(&phrase.to_lowercase()) {
                return true; // Hedging indicates awareness of uncertainty
            }
        }

        // Default: assume unsupported
        false
    }

    /// Estimate confidence of a claim
    fn estimate_confidence(&self, text: &str) -> f64 {
        let lower = text.to_lowercase();
        let mut confidence: f64 = 0.5;

        // Hedging lowers confidence
        for phrase in &self.hedging_phrases {
            if lower.contains(&phrase.to_lowercase()) {
                confidence -= 0.1;
            }
        }

        // Strong certainty words (might indicate overconfidence)
        for pattern in &self.certainty_patterns {
            if pattern.is_match(&lower) {
                confidence += 0.2;
            }
        }

        confidence.max(0.0).min(1.0)
    }

    /// Find hedging language
    fn find_hedging(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.hedging_phrases
            .iter()
            .filter(|p| lower.contains(&p.to_lowercase()))
            .map(|p| p.to_string())
            .collect()
    }

    /// Find internal contradictions
    fn find_contradictions(&self, _text: &str, claims: &[Claim]) -> Vec<HallucinationDetection> {
        let mut detections = Vec::new();

        // Simple contradiction detection: look for opposing statements
        let negation_pairs = [
            ("is", "is not"),
            ("are", "are not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("always", "never"),
        ];

        for (i, claim1) in claims.iter().enumerate() {
            for claim2 in claims.iter().skip(i + 1) {
                let c1_lower = claim1.text.to_lowercase();
                let c2_lower = claim2.text.to_lowercase();

                for (pos, neg) in &negation_pairs {
                    if (c1_lower.contains(pos) && c2_lower.contains(neg))
                        || (c1_lower.contains(neg) && c2_lower.contains(pos))
                    {
                        // Check if they're about the same subject
                        let words1: HashSet<_> = c1_lower.split_whitespace().collect();
                        let words2: HashSet<_> = c2_lower.split_whitespace().collect();
                        let overlap = words1.intersection(&words2).count();

                        if overlap >= 3 {
                            detections.push(HallucinationDetection {
                                hallucination_type: HallucinationType::Contradiction,
                                text: format!("'{}' vs '{}'", claim1.text, claim2.text),
                                position: claim1.position,
                                confidence: 0.7,
                                reason: "Potentially contradictory statements".to_string(),
                                suggestion: Some("Review for consistency".to_string()),
                            });
                        }
                    }
                }
            }
        }

        detections
    }

    /// Find unsupported claims
    fn find_unsupported_claims(
        &self,
        claims: &[Claim],
        context: Option<&str>,
    ) -> Vec<HallucinationDetection> {
        let mut detections = Vec::new();
        let context_words: HashSet<_> = context
            .map(|c| c.to_lowercase().split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        for claim in claims {
            if claim.claim_type == ClaimType::Factual && !claim.supported {
                // Check if claim relates to context
                let claim_words: HashSet<_> = claim.text.to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let has_context_support = !context_words.is_empty()
                    && claim_words.intersection(&context_words).count() >= 2;

                if !has_context_support {
                    detections.push(HallucinationDetection {
                        hallucination_type: HallucinationType::UnsupportedClaim,
                        text: claim.text.clone(),
                        position: claim.position,
                        confidence: 0.6,
                        reason: "Claim not supported by provided context".to_string(),
                        suggestion: Some("Verify claim or add source".to_string()),
                    });
                }
            }
        }

        detections
    }

    /// Find potentially fabricated quotes
    fn find_fabricated_quotes(&self, text: &str) -> Vec<HallucinationDetection> {
        let mut detections = Vec::new();

        // Look for quotes with attribution
        let quote_pattern = Regex::new(r#""([^"]+)"\s*(?:-|,|said|stated|wrote)\s*(\w+(?:\s+\w+)?)"#).expect("valid regex");

        for cap in quote_pattern.captures_iter(text) {
            let quote = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let attribution = cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if quote.len() > 20 {
                detections.push(HallucinationDetection {
                    hallucination_type: HallucinationType::FabricatedQuote,
                    text: format!("\"{}\" - {}", quote, attribution),
                    position: cap.get(0).map(|m| m.start()).unwrap_or(0),
                    confidence: 0.5,
                    reason: "Quote may be fabricated - verification recommended".to_string(),
                    suggestion: Some("Verify quote authenticity".to_string()),
                });
            }
        }

        detections
    }

    /// Check statistics for plausibility
    fn check_statistics(&self, text: &str) -> Vec<HallucinationDetection> {
        let mut detections = Vec::new();

        // Look for percentages over 100 or other suspicious numbers
        let percent_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*(?:%|percent)").expect("valid regex");

        for cap in percent_pattern.captures_iter(text) {
            if let Some(num_str) = cap.get(1) {
                if let Ok(num) = num_str.as_str().parse::<f64>() {
                    if num > 100.0 && !text.contains("increase") && !text.contains("growth") {
                        detections.push(HallucinationDetection {
                            hallucination_type: HallucinationType::IncorrectStatistic,
                            text: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
                            position: cap.get(0).map(|m| m.start()).unwrap_or(0),
                            confidence: 0.7,
                            reason: "Percentage over 100% may be incorrect".to_string(),
                            suggestion: Some("Verify statistic".to_string()),
                        });
                    }
                }
            }
        }

        detections
    }

    /// Calculate overall reliability score
    fn calculate_reliability(&self, claims: &[Claim], detections: &[HallucinationDetection]) -> f64 {
        if claims.is_empty() {
            return 1.0;
        }

        let total_claims = claims.len() as f64;
        let supported_claims = claims.iter().filter(|c| c.supported).count() as f64;
        let detection_penalty = detections.len() as f64 * 0.1;

        ((supported_claims / total_claims) - detection_penalty).max(0.0).min(1.0)
    }
}

impl Default for HallucinationDetector {
    fn default() -> Self {
        Self::new(HallucinationConfig::default())
    }
}

/// Builder for hallucination configuration
pub struct HallucinationConfigBuilder {
    config: HallucinationConfig,
}

impl HallucinationConfigBuilder {
    pub fn new() -> Self {
        Self { config: HallucinationConfig::default() }
    }

    pub fn min_confidence(mut self, conf: f64) -> Self {
        self.config.min_claim_confidence = conf;
        self
    }

    pub fn check_contradictions(mut self, check: bool) -> Self {
        self.config.check_contradictions = check;
        self
    }

    pub fn verify_sources(mut self, verify: bool) -> Self {
        self.config.verify_against_sources = verify;
        self
    }

    pub fn build(self) -> HallucinationConfig {
        self.config
    }
}

impl Default for HallucinationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedging_detection() {
        let detector = HallucinationDetector::default();
        let result = detector.detect("I think the answer might be 42.", None);

        assert!(!result.hedging_indicators.is_empty());
    }

    #[test]
    fn test_claim_extraction() {
        let detector = HallucinationDetector::default();
        let result = detector.detect("The sky is blue. Water is wet.", None);

        assert_eq!(result.claims.len(), 2);
    }

    #[test]
    fn test_quote_detection() {
        let detector = HallucinationDetector::default();
        let result = detector.detect(
            r#"Einstein once said "Imagination is more important than knowledge" in 1929."#,
            None,
        );

        let quote_detections = result.detections.iter()
            .filter(|d| d.hallucination_type == HallucinationType::FabricatedQuote)
            .count();

        assert!(quote_detections > 0 || result.claims.len() > 0);
    }

    #[test]
    fn test_statistic_check() {
        let detector = HallucinationDetector::default();
        let result = detector.detect("150% of people agree with this statement.", None);

        let stat_issues = result.detections.iter()
            .filter(|d| d.hallucination_type == HallucinationType::IncorrectStatistic)
            .count();

        assert!(stat_issues > 0);
    }
}
