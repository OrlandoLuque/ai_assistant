//! Fact verification
//!
//! Verify facts against known sources.

use std::collections::HashMap;

/// Verification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Fact is verified true
    Verified,
    /// Fact is verified false
    Contradicted,
    /// Unable to verify
    Unverified,
    /// Partially supported
    PartiallySupported,
    /// Fact is outdated
    Outdated,
}

/// Verified fact
#[derive(Debug, Clone)]
pub struct VerifiedFact {
    pub claim: String,
    pub status: VerificationStatus,
    pub confidence: f64,
    pub sources: Vec<FactSource>,
    pub explanation: Option<String>,
    pub alternatives: Vec<String>,
}

/// Source for fact verification
#[derive(Debug, Clone)]
pub struct FactSource {
    pub id: String,
    pub name: String,
    pub content: String,
    pub reliability: f64,
    pub date: Option<String>,
}

impl FactSource {
    pub fn new(id: &str, name: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            content: content.to_string(),
            reliability: 0.8,
            date: None,
        }
    }

    pub fn with_reliability(mut self, reliability: f64) -> Self {
        self.reliability = reliability.clamp(0.0, 1.0);
        self
    }

    pub fn with_date(mut self, date: &str) -> Self {
        self.date = Some(date.to_string());
        self
    }
}

/// Fact verification configuration
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Minimum confidence for verification
    pub min_confidence: f64,
    /// Minimum sources for verification
    pub min_sources: usize,
    /// Weight for source reliability
    pub reliability_weight: f64,
    /// Consider date for freshness
    pub check_freshness: bool,
    /// Maximum age in days for fresh data
    pub max_age_days: u64,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            min_sources: 1,
            reliability_weight: 0.3,
            check_freshness: true,
            max_age_days: 365,
        }
    }
}

/// Fact verifier
pub struct FactVerifier {
    config: VerificationConfig,
    knowledge_base: HashMap<String, Vec<FactSource>>,
}

impl FactVerifier {
    pub fn new(config: VerificationConfig) -> Self {
        Self {
            config,
            knowledge_base: HashMap::new(),
        }
    }

    /// Add a source to the knowledge base
    pub fn add_source(&mut self, topic: &str, source: FactSource) {
        self.knowledge_base
            .entry(topic.to_lowercase())
            .or_default()
            .push(source);
    }

    /// Add multiple sources for a topic
    pub fn add_sources(&mut self, topic: &str, sources: Vec<FactSource>) {
        for source in sources {
            self.add_source(topic, source);
        }
    }

    /// Verify a claim
    pub fn verify(&self, claim: &str) -> VerifiedFact {
        let claim_lower = claim.to_lowercase();
        let keywords = self.extract_keywords(&claim_lower);

        // Find relevant sources
        let mut relevant_sources: Vec<&FactSource> = Vec::new();

        for (topic, sources) in &self.knowledge_base {
            if keywords.iter().any(|k| topic.contains(k) || k.contains(topic)) {
                relevant_sources.extend(sources.iter());
            }
        }

        if relevant_sources.is_empty() {
            return VerifiedFact {
                claim: claim.to_string(),
                status: VerificationStatus::Unverified,
                confidence: 0.0,
                sources: Vec::new(),
                explanation: Some("No relevant sources found".to_string()),
                alternatives: Vec::new(),
            };
        }

        // Check each source
        let mut supports = 0;
        let mut contradicts = 0;
        let mut total_reliability = 0.0;
        let mut supporting_sources = Vec::new();

        for source in &relevant_sources {
            let (matches, contradicts_claim) = self.check_source(&claim_lower, source);

            if matches {
                supports += 1;
                total_reliability += source.reliability;
                supporting_sources.push((*source).clone());
            } else if contradicts_claim {
                contradicts += 1;
            }
        }

        // Determine status
        let (status, confidence) = self.determine_status(
            supports,
            contradicts,
            relevant_sources.len(),
            total_reliability,
        );

        let explanation = self.generate_explanation(status, supports, contradicts, relevant_sources.len());

        VerifiedFact {
            claim: claim.to_string(),
            status,
            confidence,
            sources: supporting_sources,
            explanation: Some(explanation),
            alternatives: Vec::new(),
        }
    }

    /// Verify multiple claims
    pub fn verify_all(&self, claims: &[&str]) -> Vec<VerifiedFact> {
        claims.iter().map(|c| self.verify(c)).collect()
    }

    /// Extract claims from text
    pub fn extract_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();

        // Split into sentences
        for sentence in text.split(|c| c == '.' || c == '!' || c == '?') {
            let trimmed = sentence.trim();
            if trimmed.len() > 10 && self.is_factual_claim(trimmed) {
                claims.push(trimmed.to_string());
            }
        }

        claims
    }

    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| !["the", "and", "that", "this", "with", "from", "have", "been"].contains(w))
            .map(|w| w.to_string())
            .collect()
    }

    fn check_source(&self, claim: &str, source: &FactSource) -> (bool, bool) {
        let source_lower = source.content.to_lowercase();
        let claim_words: Vec<_> = claim.split_whitespace().collect();
        let source_words: Vec<_> = source_lower.split_whitespace().collect();

        // Count word overlap
        let overlap = claim_words.iter()
            .filter(|w| w.len() > 3)
            .filter(|w| source_words.iter().any(|sw| sw.contains(*w) || w.contains(sw)))
            .count();

        let overlap_ratio = overlap as f64 / claim_words.len().max(1) as f64;

        // Check for negation patterns
        let has_negation = source_lower.contains("not ") || source_lower.contains("never ") ||
                          source_lower.contains("false") || source_lower.contains("incorrect");

        let claim_has_negation = claim.contains("not ") || claim.contains("never ");

        // If both or neither have negation, they might agree
        let contradicts = (has_negation && !claim_has_negation) || (!has_negation && claim_has_negation);

        let matches = overlap_ratio > 0.3 && !contradicts;
        let contradicts_claim = overlap_ratio > 0.3 && contradicts;

        (matches, contradicts_claim)
    }

    fn determine_status(
        &self,
        supports: usize,
        contradicts: usize,
        total: usize,
        total_reliability: f64,
    ) -> (VerificationStatus, f64) {
        if supports == 0 && contradicts == 0 {
            return (VerificationStatus::Unverified, 0.0);
        }

        let support_ratio = supports as f64 / total as f64;
        let contradict_ratio = contradicts as f64 / total as f64;
        let avg_reliability = if supports > 0 {
            total_reliability / supports as f64
        } else {
            0.0
        };

        let confidence = (support_ratio * (1.0 - self.config.reliability_weight) +
                         avg_reliability * self.config.reliability_weight).clamp(0.0, 1.0);

        if supports >= self.config.min_sources && confidence >= self.config.min_confidence {
            if contradict_ratio > 0.3 {
                (VerificationStatus::PartiallySupported, confidence * 0.7)
            } else {
                (VerificationStatus::Verified, confidence)
            }
        } else if contradicts > supports {
            (VerificationStatus::Contradicted, contradict_ratio)
        } else if supports > 0 {
            (VerificationStatus::PartiallySupported, confidence * 0.5)
        } else {
            (VerificationStatus::Unverified, 0.0)
        }
    }

    fn generate_explanation(
        &self,
        status: VerificationStatus,
        supports: usize,
        contradicts: usize,
        total: usize,
    ) -> String {
        match status {
            VerificationStatus::Verified => {
                format!("Verified by {} of {} sources", supports, total)
            }
            VerificationStatus::Contradicted => {
                format!("Contradicted by {} of {} sources", contradicts, total)
            }
            VerificationStatus::PartiallySupported => {
                format!("Partially supported: {} support, {} contradict", supports, contradicts)
            }
            VerificationStatus::Unverified => {
                "Unable to verify - no matching sources".to_string()
            }
            VerificationStatus::Outdated => {
                "Information may be outdated".to_string()
            }
        }
    }

    fn is_factual_claim(&self, text: &str) -> bool {
        let lower = text.to_lowercase();

        // Skip questions
        if text.ends_with('?') {
            return false;
        }

        // Skip opinions
        let opinion_markers = ["i think", "i believe", "in my opinion", "probably", "maybe"];
        if opinion_markers.iter().any(|m| lower.contains(m)) {
            return false;
        }

        // Check for factual indicators
        let factual_patterns = [
            "is ", "are ", "was ", "were ", "has ", "have ",
            "can ", "will ", "the ", "there ",
        ];

        factual_patterns.iter().any(|p| lower.contains(p))
    }
}

impl Default for FactVerifier {
    fn default() -> Self {
        Self::new(VerificationConfig::default())
    }
}

/// Builder for fact verifier
pub struct FactVerifierBuilder {
    config: VerificationConfig,
    sources: Vec<(String, FactSource)>,
}

impl FactVerifierBuilder {
    pub fn new() -> Self {
        Self {
            config: VerificationConfig::default(),
            sources: Vec::new(),
        }
    }

    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.config.min_confidence = confidence;
        self
    }

    pub fn min_sources(mut self, count: usize) -> Self {
        self.config.min_sources = count;
        self
    }

    pub fn add_source(mut self, topic: &str, source: FactSource) -> Self {
        self.sources.push((topic.to_string(), source));
        self
    }

    pub fn build(self) -> FactVerifier {
        let mut verifier = FactVerifier::new(self.config);
        for (topic, source) in self.sources {
            verifier.add_source(&topic, source);
        }
        verifier
    }
}

impl Default for FactVerifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification() {
        let mut verifier = FactVerifier::default();

        verifier.add_source("python", FactSource::new(
            "wiki",
            "Wikipedia",
            "Python is a high-level programming language created by Guido van Rossum."
        ));

        let result = verifier.verify("Python is a programming language");
        assert_eq!(result.status, VerificationStatus::Verified);
    }

    #[test]
    fn test_unverified() {
        let verifier = FactVerifier::default();

        let result = verifier.verify("Unknown random fact");
        assert_eq!(result.status, VerificationStatus::Unverified);
    }

    #[test]
    fn test_claim_extraction() {
        let verifier = FactVerifier::default();

        let text = "Python is a programming language. It was created in 1991. Do you like it?";
        let claims = verifier.extract_claims(text);

        assert!(claims.len() >= 2);
        assert!(claims.iter().any(|c| c.contains("programming")));
    }

    #[test]
    fn test_builder() {
        let verifier = FactVerifierBuilder::new()
            .min_confidence(0.8)
            .min_sources(2)
            .add_source("test", FactSource::new("s1", "Source", "Test content"))
            .build();

        assert_eq!(verifier.config.min_confidence, 0.8);
    }
}
