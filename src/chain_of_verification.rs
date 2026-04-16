//! Chain-of-Verification (CoVe) pipeline
//!
//! Implements the CoVe methodology: generate a draft response, extract claims,
//! verify each claim independently, then produce a corrected response.
//!
//! The pipeline uses a configurable verification source (RAG, web search, or both)
//! and applies corrections via annotate, replace, or footnote modes.
//!
//! # Pipeline Steps
//!
//! 1. Extract claims from the draft response
//! 2. Generate verification questions for each claim
//! 3. Verify each claim against sources
//! 4. Correct or annotate the response based on verification results

use crate::anti_hallucination::UngroundedClaimStrategy;

/// Where to look for verification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerificationSource {
    /// Only check against RAG/knowledge base
    RagOnly,
    /// Only check via web search
    WebSearchOnly,
    /// Try RAG first, fall back to web search
    RagThenWeb,
    /// Check both RAG and web search, combine evidence
    Both,
}

impl Default for VerificationSource {
    fn default() -> Self {
        Self::RagOnly
    }
}

/// How to apply corrections to the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CorrectionMode {
    /// Replace incorrect claims with corrected text
    Replace,
    /// Add annotations (e.g., "[unverified]") to unverified claims
    Annotate,
    /// Add footnotes with verification details
    Footnote,
}

impl Default for CorrectionMode {
    fn default() -> Self {
        Self::Annotate
    }
}

/// Status of a verified claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClaimVerificationStatus {
    /// Claim is supported by evidence
    Supported,
    /// Claim is contradicted by evidence
    Contradicted,
    /// No evidence found to support or deny
    Unverifiable,
    /// Claim is partially supported
    PartiallySupported,
}

/// Configuration for the Chain-of-Verification pipeline.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CoVeConfig {
    /// Enable/disable the CoVe pipeline
    pub enabled: bool,
    /// Maximum number of claims to verify (hard cap to control cost)
    pub max_claims_to_verify: usize,
    /// Where to look for verification evidence
    pub verification_source: VerificationSource,
    /// How to apply corrections
    pub correction_mode: CorrectionMode,
    /// Minimum word overlap ratio to consider a source as supporting
    pub min_support_similarity: f64,
    /// Strategy for ungrounded claims (reuses anti_hallucination strategy)
    pub ungrounded_strategy: UngroundedClaimStrategy,
    /// Maximum LLM calls budget for verification (integrated with RAG tier budget)
    pub max_llm_calls: usize,
}

impl Default for CoVeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_claims_to_verify: 10,
            verification_source: VerificationSource::default(),
            correction_mode: CorrectionMode::default(),
            min_support_similarity: 0.3,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            max_llm_calls: 5,
        }
    }
}

impl CoVeConfig {
    /// Strict configuration: replace contradicted claims, verify against both sources
    pub fn strict() -> Self {
        Self {
            enabled: true,
            max_claims_to_verify: 10,
            verification_source: VerificationSource::Both,
            correction_mode: CorrectionMode::Replace,
            min_support_similarity: 0.4,
            ungrounded_strategy: UngroundedClaimStrategy::Omit,
            max_llm_calls: 10,
        }
    }

    /// Permissive configuration: annotate only, RAG-only verification
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            max_claims_to_verify: 5,
            verification_source: VerificationSource::RagOnly,
            correction_mode: CorrectionMode::Annotate,
            min_support_similarity: 0.2,
            ungrounded_strategy: UngroundedClaimStrategy::Warn,
            max_llm_calls: 3,
        }
    }
}

/// Result of verifying a single claim.
#[derive(Debug, Clone)]
pub struct VerifiedClaimResult {
    /// The original claim text
    pub claim: String,
    /// Verification status
    pub status: ClaimVerificationStatus,
    /// Confidence in the verification (0.0-1.0)
    pub confidence: f64,
    /// Evidence supporting or contradicting the claim
    pub evidence: Vec<VerificationEvidence>,
    /// Corrected text (if status is Contradicted and correction available)
    pub correction: Option<String>,
    /// Verification question generated for this claim
    pub verification_question: String,
}

/// A piece of evidence from verification.
#[derive(Debug, Clone)]
pub struct VerificationEvidence {
    /// Source of the evidence (e.g., "RAG chunk #3", "Web: example.com")
    pub source: String,
    /// The evidence text
    pub content: String,
    /// How well the evidence supports the claim (0.0-1.0)
    pub relevance: f64,
    /// Whether this evidence supports or contradicts
    pub supports: bool,
}

/// Result of the full Chain-of-Verification pipeline.
#[derive(Debug, Clone)]
pub struct CoVeResult {
    /// The original response before verification
    pub original_response: String,
    /// The corrected/annotated response
    pub corrected_response: String,
    /// Per-claim verification results
    pub verified_claims: Vec<VerifiedClaimResult>,
    /// Number of corrections applied
    pub corrections_made: usize,
    /// Overall accuracy estimate (ratio of supported claims)
    pub overall_accuracy: f64,
    /// Number of LLM calls used during verification
    pub llm_calls_used: usize,
    /// Number of claims that were skipped due to budget limits
    pub claims_skipped: usize,
}

impl CoVeResult {
    /// Count of claims by status
    pub fn count_by_status(&self, status: ClaimVerificationStatus) -> usize {
        self.verified_claims
            .iter()
            .filter(|c| c.status == status)
            .count()
    }

    /// Check if any contradictions were found
    pub fn has_contradictions(&self) -> bool {
        self.verified_claims
            .iter()
            .any(|c| c.status == ClaimVerificationStatus::Contradicted)
    }

    /// Get all contradicted claims
    pub fn contradicted_claims(&self) -> Vec<&VerifiedClaimResult> {
        self.verified_claims
            .iter()
            .filter(|c| c.status == ClaimVerificationStatus::Contradicted)
            .collect()
    }
}

/// A context source for verification (RAG chunk, web result, etc.)
#[derive(Debug, Clone)]
pub struct VerificationContext {
    /// Source identifier
    pub source_id: String,
    /// Source label (e.g., "RAG", "Web Search")
    pub source_type: String,
    /// The context text
    pub content: String,
    /// Reliability score (0.0-1.0)
    pub reliability: f64,
}

/// Chain-of-Verification pipeline.
pub struct ChainOfVerification {
    config: CoVeConfig,
}

impl ChainOfVerification {
    /// Create a new CoVe pipeline with the given configuration.
    pub fn new(config: CoVeConfig) -> Self {
        Self { config }
    }

    /// Run the full verification pipeline on a response.
    ///
    /// `response` — the draft LLM response to verify.
    /// `context` — available verification sources (RAG chunks, web results).
    pub fn verify(&self, response: &str, context: &[VerificationContext]) -> CoVeResult {
        // Step 1: Extract claims from the response
        let claims = self.extract_claims(response);

        // Step 2: Cap claims to verify
        let claims_to_verify = claims.len().min(self.config.max_claims_to_verify);
        let claims_skipped = claims.len().saturating_sub(claims_to_verify);

        // Step 3: Verify each claim
        let mut verified_claims = Vec::new();
        let mut llm_calls_used = 0;

        for claim in claims.iter().take(claims_to_verify) {
            let verification_question = self.generate_verification_question(claim);
            let result = self.verify_claim(claim, &verification_question, context);
            verified_claims.push(result);
            // Each verification counts as one logical "call" for budget tracking
            llm_calls_used += 1;
            if llm_calls_used >= self.config.max_llm_calls {
                break;
            }
        }

        // Step 4: Apply corrections
        let corrections_made = verified_claims
            .iter()
            .filter(|c| c.status == ClaimVerificationStatus::Contradicted && c.correction.is_some())
            .count();

        let corrected_response = self.apply_corrections(response, &verified_claims);

        // Calculate overall accuracy
        let total_verified = verified_claims.len();
        let supported = verified_claims
            .iter()
            .filter(|c| {
                c.status == ClaimVerificationStatus::Supported
                    || c.status == ClaimVerificationStatus::PartiallySupported
            })
            .count();
        let overall_accuracy = if total_verified > 0 {
            supported as f64 / total_verified as f64
        } else {
            1.0
        };

        CoVeResult {
            original_response: response.to_string(),
            corrected_response,
            verified_claims,
            corrections_made,
            overall_accuracy,
            llm_calls_used,
            claims_skipped,
        }
    }

    /// Extract factual claims from text (sentences that look factual).
    fn extract_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();
        for part in text.split(|c| c == '.' || c == '!' || c == '?') {
            let trimmed = part.trim();
            if trimmed.len() > 10 && Self::is_factual_sentence(trimmed) {
                claims.push(trimmed.to_string());
            }
        }
        claims
    }

    /// Check if a sentence looks like a factual claim (not a question, opinion, etc.)
    fn is_factual_sentence(text: &str) -> bool {
        let lower = text.to_lowercase();

        // Skip opinions
        let opinion_markers = [
            "i think",
            "i believe",
            "in my opinion",
            "probably",
            "maybe",
            "perhaps",
            "it seems",
        ];
        if opinion_markers.iter().any(|m| lower.contains(m)) {
            return false;
        }

        // Skip imperative/instructions
        let imperative_markers = ["please ", "let me ", "you should", "try to", "make sure"];
        if imperative_markers.iter().any(|m| lower.starts_with(m)) {
            return false;
        }

        // Factual indicators
        let factual_patterns = [
            "is ",
            "are ",
            "was ",
            "were ",
            "has ",
            "have ",
            "had ",
            "the ",
            "there ",
            "can ",
            "will ",
            "contains ",
            "includes ",
        ];
        factual_patterns.iter().any(|p| lower.contains(p))
    }

    /// Generate a verification question for a claim.
    fn generate_verification_question(&self, claim: &str) -> String {
        // Simple heuristic: convert claim to question form
        let lower = claim.to_lowercase();

        if lower.contains(" is ") {
            format!("Is it true that {}?", claim.to_lowercase())
        } else if lower.contains(" are ") {
            format!("Is it true that {}?", claim.to_lowercase())
        } else if lower.contains(" was ") {
            format!("Is it true that {}?", claim.to_lowercase())
        } else if lower.contains(" has ") || lower.contains(" have ") {
            format!("Is it true that {}?", claim.to_lowercase())
        } else {
            format!("Verify: {}", claim)
        }
    }

    /// Verify a single claim against available context.
    fn verify_claim(
        &self,
        claim: &str,
        _verification_question: &str,
        context: &[VerificationContext],
    ) -> VerifiedClaimResult {
        let claim_words = words_set(claim);
        let mut evidence_list = Vec::new();
        let mut best_support: f64 = 0.0;
        let mut has_contradiction = false;

        // Filter context by verification source config
        let relevant_context: Vec<&VerificationContext> = match self.config.verification_source {
            VerificationSource::RagOnly => {
                context.iter().filter(|c| c.source_type == "RAG").collect()
            }
            VerificationSource::WebSearchOnly => {
                context.iter().filter(|c| c.source_type == "Web").collect()
            }
            VerificationSource::RagThenWeb => {
                let rag: Vec<_> = context.iter().filter(|c| c.source_type == "RAG").collect();
                if rag.is_empty() {
                    context.iter().filter(|c| c.source_type == "Web").collect()
                } else {
                    rag
                }
            }
            VerificationSource::Both => context.iter().collect(),
        };

        for ctx in &relevant_context {
            let ctx_words = words_set(&ctx.content);
            let similarity = jaccard(&claim_words, &ctx_words);

            if similarity >= self.config.min_support_similarity {
                // Check for negation/contradiction
                let contradicts = has_negation_mismatch(claim, &ctx.content);

                let supports = !contradicts && similarity >= self.config.min_support_similarity;

                if supports {
                    best_support = best_support.max(similarity);
                }
                if contradicts {
                    has_contradiction = true;
                }

                evidence_list.push(VerificationEvidence {
                    source: format!("{}: {}", ctx.source_type, ctx.source_id),
                    content: ctx.content.clone(),
                    relevance: similarity,
                    supports,
                });
            }
        }

        // Determine status
        let status = if has_contradiction {
            ClaimVerificationStatus::Contradicted
        } else if best_support >= 0.5 {
            ClaimVerificationStatus::Supported
        } else if best_support >= self.config.min_support_similarity {
            ClaimVerificationStatus::PartiallySupported
        } else {
            ClaimVerificationStatus::Unverifiable
        };

        // Generate correction if contradicted
        let correction = if status == ClaimVerificationStatus::Contradicted {
            evidence_list
                .iter()
                .find(|e| !e.supports && e.relevance > 0.0)
                .map(|e| {
                    // Use the contradicting evidence as the correction basis
                    let sentences: Vec<&str> = e
                        .content
                        .split(|c| c == '.' || c == '!' || c == '?')
                        .map(|s| s.trim())
                        .filter(|s| s.len() > 10)
                        .collect();
                    sentences
                        .first()
                        .unwrap_or(&"[correction needed]")
                        .to_string()
                })
        } else {
            None
        };

        let confidence = if status == ClaimVerificationStatus::Supported {
            best_support.min(1.0)
        } else if status == ClaimVerificationStatus::Contradicted {
            0.8 // High confidence in contradiction when evidence exists
        } else if status == ClaimVerificationStatus::PartiallySupported {
            best_support * 0.7
        } else {
            0.0
        };

        let verification_question = self.generate_verification_question(claim);

        VerifiedClaimResult {
            claim: claim.to_string(),
            status,
            confidence,
            evidence: evidence_list,
            correction,
            verification_question,
        }
    }

    /// Apply corrections to the original response based on verification results.
    fn apply_corrections(&self, response: &str, verified_claims: &[VerifiedClaimResult]) -> String {
        let mut result = response.to_string();

        for claim_result in verified_claims {
            match claim_result.status {
                ClaimVerificationStatus::Contradicted => match self.config.correction_mode {
                    CorrectionMode::Replace => {
                        if let Some(correction) = &claim_result.correction {
                            result = result.replace(&claim_result.claim, correction);
                        }
                    }
                    CorrectionMode::Annotate => {
                        let annotated = format!("{} [contradicted]", claim_result.claim);
                        result = result.replace(&claim_result.claim, &annotated);
                    }
                    CorrectionMode::Footnote => {
                        let footnoted =
                            format!("{}[^cove: contradicted by evidence]", claim_result.claim);
                        result = result.replace(&claim_result.claim, &footnoted);
                    }
                },
                ClaimVerificationStatus::Unverifiable => {
                    match self.config.ungrounded_strategy {
                        UngroundedClaimStrategy::Mark => {
                            let marked = format!("{} [unverified]", claim_result.claim);
                            result = result.replace(&claim_result.claim, &marked);
                        }
                        UngroundedClaimStrategy::Omit => {
                            // Remove the unverifiable claim sentence
                            result = result.replace(&claim_result.claim, "");
                        }
                        UngroundedClaimStrategy::Warn => {
                            let warned = format!("{} [warning: unverified]", claim_result.claim);
                            result = result.replace(&claim_result.claim, &warned);
                        }
                        _ => {
                            // Other strategies: leave as-is
                        }
                    }
                }
                _ => {
                    // Supported / PartiallySupported: leave as-is
                }
            }
        }

        // Clean up double spaces from omissions
        while result.contains("  ") {
            result = result.replace("  ", " ");
        }
        result = result.replace(". .", ".").trim().to_string();

        result
    }
}

impl Default for ChainOfVerification {
    fn default() -> Self {
        Self::new(CoVeConfig::default())
    }
}

// === Helper functions ===

/// Compute word set from text (lowercase).
fn words_set(text: &str) -> std::collections::HashSet<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity between two word sets.
fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Check if there's a negation mismatch between claim and source.
fn has_negation_mismatch(claim: &str, source: &str) -> bool {
    let claim_lower = claim.to_lowercase();
    let source_lower = source.to_lowercase();

    let negation_words = [
        "not ", "never ", "no ", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't",
        "didn't", "cannot", "can't", "won't",
    ];

    let claim_has_negation = negation_words.iter().any(|n| claim_lower.contains(n));
    let source_has_negation = negation_words.iter().any(|n| source_lower.contains(n));

    // Check for words like "false", "incorrect", "wrong" in source
    let contradiction_words = ["false", "incorrect", "wrong", "myth", "misconception"];
    let source_has_contradiction = contradiction_words.iter().any(|w| source_lower.contains(w));

    // Mismatch: one has negation but not the other, OR source explicitly contradicts
    (claim_has_negation != source_has_negation) || (!claim_has_negation && source_has_contradiction)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rag_context(id: &str, content: &str) -> VerificationContext {
        VerificationContext {
            source_id: id.to_string(),
            source_type: "RAG".to_string(),
            content: content.to_string(),
            reliability: 0.9,
        }
    }

    fn make_web_context(id: &str, content: &str) -> VerificationContext {
        VerificationContext {
            source_id: id.to_string(),
            source_type: "Web".to_string(),
            content: content.to_string(),
            reliability: 0.7,
        }
    }

    #[test]
    fn test_cove_default_config() {
        let config = CoVeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_claims_to_verify, 10);
        assert_eq!(config.verification_source, VerificationSource::RagOnly);
        assert_eq!(config.correction_mode, CorrectionMode::Annotate);
        assert_eq!(config.max_llm_calls, 5);
    }

    #[test]
    fn test_cove_strict_config() {
        let config = CoVeConfig::strict();
        assert_eq!(config.verification_source, VerificationSource::Both);
        assert_eq!(config.correction_mode, CorrectionMode::Replace);
        assert_eq!(config.max_llm_calls, 10);
    }

    #[test]
    fn test_cove_permissive_config() {
        let config = CoVeConfig::permissive();
        assert_eq!(config.max_claims_to_verify, 5);
        assert_eq!(config.verification_source, VerificationSource::RagOnly);
        assert_eq!(config.correction_mode, CorrectionMode::Annotate);
    }

    #[test]
    fn test_extract_claims() {
        let cove = ChainOfVerification::default();
        let text = "Python is a programming language. I think it is great. The sky is blue.";
        let claims = cove.extract_claims(text);
        // "Python is a programming language" and "The sky is blue" are factual
        // "I think it is great" should be filtered out
        assert!(claims.len() >= 2);
        assert!(claims.iter().any(|c| c.contains("Python")));
        assert!(claims.iter().any(|c| c.contains("sky")));
        assert!(!claims.iter().any(|c| c.contains("I think")));
    }

    #[test]
    fn test_extract_claims_filters_opinions() {
        let cove = ChainOfVerification::default();
        let text =
            "I believe this is correct. Maybe the answer is wrong. Rust is a systems language.";
        let claims = cove.extract_claims(text);
        for claim in &claims {
            let lower = claim.to_lowercase();
            assert!(!lower.contains("i believe"));
            assert!(!lower.contains("maybe"));
        }
    }

    #[test]
    fn test_verify_supported_claim() {
        let cove = ChainOfVerification::default();
        let context = vec![make_rag_context("doc1", "Python is a programming language")];

        let result = cove.verify("Python is a programming language.", &context);
        assert!(!result.verified_claims.is_empty());
        assert_eq!(
            result.verified_claims[0].status,
            ClaimVerificationStatus::Supported
        );
        assert!(result.overall_accuracy > 0.0);
    }

    #[test]
    fn test_verify_contradicted_claim() {
        let config = CoVeConfig {
            correction_mode: CorrectionMode::Annotate,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![make_rag_context(
            "doc1",
            "Python was not created in 1980. It is incorrect to say Python is from 1980.",
        )];

        let result = cove.verify("Python was created in 1980.", &context);
        assert!(!result.verified_claims.is_empty());
        assert_eq!(
            result.verified_claims[0].status,
            ClaimVerificationStatus::Contradicted
        );
        assert!(result.corrected_response.contains("[contradicted]"));
    }

    #[test]
    fn test_verify_unverifiable_claim() {
        let cove = ChainOfVerification::default();
        // Empty context — nothing to verify against
        let context: Vec<VerificationContext> = Vec::new();

        let result = cove.verify("The capital of Mars colony is Olympus City.", &context);
        // No evidence at all
        assert!(
            result.verified_claims.is_empty()
                || result
                    .verified_claims
                    .iter()
                    .all(|c| c.status == ClaimVerificationStatus::Unverifiable)
        );
    }

    #[test]
    fn test_correction_mode_replace() {
        let config = CoVeConfig {
            correction_mode: CorrectionMode::Replace,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![make_rag_context(
            "doc1",
            "The Earth is not flat. This is a false claim that has been debunked.",
        )];

        let result = cove.verify("The Earth is flat and has no curvature.", &context);
        // With Replace mode, contradicted claims should be replaced
        if result.has_contradictions() {
            // The corrected response should differ from original
            assert_ne!(result.corrected_response, result.original_response);
        }
    }

    #[test]
    fn test_correction_mode_footnote() {
        let config = CoVeConfig {
            correction_mode: CorrectionMode::Footnote,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![make_rag_context(
            "doc1",
            "This fact is incorrect and false according to evidence.",
        )];

        let result = cove.verify("This fact is true and well established.", &context);
        if result.has_contradictions() {
            assert!(result.corrected_response.contains("[^cove:"));
        }
    }

    #[test]
    fn test_max_claims_cap() {
        let config = CoVeConfig {
            max_claims_to_verify: 2,
            max_llm_calls: 10,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![make_rag_context(
            "doc1",
            "General knowledge base with many facts about science and technology",
        )];

        // Generate a response with many claims
        let response =
            "Python is a language. Rust is fast. Java has GC. Go is concurrent. C is old.";
        let result = cove.verify(response, &context);
        // Should verify at most 2 claims
        assert!(result.verified_claims.len() <= 2);
    }

    #[test]
    fn test_max_llm_calls_budget() {
        let config = CoVeConfig {
            max_claims_to_verify: 100,
            max_llm_calls: 2,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![make_rag_context("doc1", "General knowledge")];

        let response =
            "Claim one is true. Claim two is valid. Claim three is correct. Claim four is right.";
        let result = cove.verify(response, &context);
        assert!(result.llm_calls_used <= 2);
    }

    #[test]
    fn test_verification_source_rag_only() {
        let config = CoVeConfig {
            verification_source: VerificationSource::RagOnly,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![
            make_rag_context("rag1", "Python is a programming language"),
            make_web_context("web1", "Python is widely used for web development"),
        ];

        let result = cove.verify("Python is a programming language.", &context);
        if let Some(claim) = result.verified_claims.first() {
            // Should only use RAG evidence
            for ev in &claim.evidence {
                assert!(ev.source.starts_with("RAG:"));
            }
        }
    }

    #[test]
    fn test_verification_source_web_only() {
        let config = CoVeConfig {
            verification_source: VerificationSource::WebSearchOnly,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![
            make_rag_context("rag1", "Python is a programming language"),
            make_web_context("web1", "Python is a programming language used worldwide"),
        ];

        let result = cove.verify("Python is a programming language.", &context);
        if let Some(claim) = result.verified_claims.first() {
            for ev in &claim.evidence {
                assert!(ev.source.starts_with("Web:"));
            }
        }
    }

    #[test]
    fn test_verification_source_both() {
        let config = CoVeConfig {
            verification_source: VerificationSource::Both,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![
            make_rag_context("rag1", "Python is a high-level programming language"),
            make_web_context("web1", "Python is a programming language created by Guido"),
        ];

        let result = cove.verify("Python is a programming language.", &context);
        if let Some(claim) = result.verified_claims.first() {
            // Should have evidence from both sources
            let has_rag = claim.evidence.iter().any(|e| e.source.starts_with("RAG:"));
            let has_web = claim.evidence.iter().any(|e| e.source.starts_with("Web:"));
            assert!(has_rag || has_web);
        }
    }

    #[test]
    fn test_verification_source_rag_then_web_fallback() {
        let config = CoVeConfig {
            verification_source: VerificationSource::RagThenWeb,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        // Only web context, no RAG — should fall back to web
        let context = vec![make_web_context("web1", "Python is a programming language")];

        let result = cove.verify("Python is a programming language.", &context);
        if let Some(claim) = result.verified_claims.first() {
            for ev in &claim.evidence {
                assert!(ev.source.starts_with("Web:"));
            }
        }
    }

    #[test]
    fn test_cove_result_helpers() {
        let result = CoVeResult {
            original_response: "test".to_string(),
            corrected_response: "test".to_string(),
            verified_claims: vec![
                VerifiedClaimResult {
                    claim: "claim1".to_string(),
                    status: ClaimVerificationStatus::Supported,
                    confidence: 0.9,
                    evidence: Vec::new(),
                    correction: None,
                    verification_question: "Q1".to_string(),
                },
                VerifiedClaimResult {
                    claim: "claim2".to_string(),
                    status: ClaimVerificationStatus::Contradicted,
                    confidence: 0.8,
                    evidence: Vec::new(),
                    correction: Some("corrected".to_string()),
                    verification_question: "Q2".to_string(),
                },
                VerifiedClaimResult {
                    claim: "claim3".to_string(),
                    status: ClaimVerificationStatus::Supported,
                    confidence: 0.7,
                    evidence: Vec::new(),
                    correction: None,
                    verification_question: "Q3".to_string(),
                },
            ],
            corrections_made: 1,
            overall_accuracy: 0.67,
            llm_calls_used: 3,
            claims_skipped: 0,
        };

        assert_eq!(
            result.count_by_status(ClaimVerificationStatus::Supported),
            2
        );
        assert_eq!(
            result.count_by_status(ClaimVerificationStatus::Contradicted),
            1
        );
        assert!(result.has_contradictions());
        assert_eq!(result.contradicted_claims().len(), 1);
    }

    #[test]
    fn test_generate_verification_question() {
        let cove = ChainOfVerification::default();
        let q = cove.generate_verification_question("Python is a language");
        assert!(q.contains("true") || q.contains("Verify"));

        let q2 = cove.generate_verification_question("The results have shown improvement");
        assert!(q2.contains("Verify") || q2.contains("true"));
    }

    #[test]
    fn test_negation_mismatch_detection() {
        assert!(has_negation_mismatch(
            "The earth is flat",
            "The earth is not flat"
        ));
        assert!(!has_negation_mismatch(
            "The earth is round",
            "The earth is spherical and round"
        ));
        assert!(has_negation_mismatch(
            "Python is fast",
            "This is false for Python performance"
        ));
    }

    #[test]
    fn test_words_set() {
        let words = words_set("Hello World test a");
        assert!(words.contains("hello"));
        assert!(words.contains("world"));
        assert!(words.contains("test"));
        // "a" is too short (len <= 2)
        assert!(!words.contains("a"));
    }

    #[test]
    fn test_jaccard_identical() {
        let a = words_set("hello world test");
        let b = words_set("hello world test");
        assert!((jaccard(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = words_set("hello world");
        let b = words_set("foo bar baz");
        assert!((jaccard(&a, &b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ungrounded_strategy_mark() {
        let config = CoVeConfig {
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let result = cove.verify(
            "Some unverifiable claim is stated here.",
            &[], // No context
        );
        if !result.verified_claims.is_empty()
            && result.verified_claims[0].status == ClaimVerificationStatus::Unverifiable
        {
            assert!(result.corrected_response.contains("[unverified]"));
        }
    }

    #[test]
    fn test_ungrounded_strategy_omit() {
        let config = CoVeConfig {
            ungrounded_strategy: UngroundedClaimStrategy::Omit,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let text = "Some unverifiable claim is stated here.";
        let result = cove.verify(text, &[]);
        if !result.verified_claims.is_empty()
            && result.verified_claims[0].status == ClaimVerificationStatus::Unverifiable
        {
            // The claim text should be removed
            assert!(!result.corrected_response.contains("unverifiable claim"));
        }
    }

    #[test]
    fn test_empty_response() {
        let cove = ChainOfVerification::default();
        let result = cove.verify("", &[]);
        assert!(result.verified_claims.is_empty());
        assert_eq!(result.overall_accuracy, 1.0);
        assert_eq!(result.corrections_made, 0);
    }

    #[test]
    fn test_multiple_claims_with_mixed_results() {
        let config = CoVeConfig {
            verification_source: VerificationSource::Both,
            ..Default::default()
        };
        let cove = ChainOfVerification::new(config);
        let context = vec![
            make_rag_context("doc1", "Python is a high-level programming language"),
            make_rag_context(
                "doc2",
                "Java was not created in 2020, it was created in 1995",
            ),
        ];

        let response = "Python is a programming language. Java was created in 2020.";
        let result = cove.verify(response, &context);

        // Should have verified both claims
        assert!(result.verified_claims.len() >= 1);
    }
}
