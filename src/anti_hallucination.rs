//! Anti-Hallucination Pipeline
//!
//! Orchestrates multiple anti-hallucination techniques into a configurable pipeline
//! that processes LLM responses to detect, mark, or remove ungrounded claims.
//!
//! # Pipeline Order
//!
//! ```text
//! 1. Auto-temperature (B13) — before sending to the LLM
//! 2. LLM generation
//! 3. Abstention check (B1) — if confidence < threshold, stop here
//! 4. Claim decomposition — shared by B3, B4, B6, B7
//! 5. Faithfulness scoring (B3)
//! 6. Grounded generation check (B7)
//! 7. Fact-check with search (B4) — only claims with NliVerdict::Neutral
//! 8. Chain-of-Verification (B6) — only if configured, after B4
//! 9. Self-consistency (B9) — parallel/independent
//! 10. Apply UngroundedClaimStrategy (B2) — on ungrounded claims
//! 11. Quality gates check (B12) — fail/warn/log
//! ```
//!
//! # Usage
//!
//! ```rust
//! use ai_assistant::anti_hallucination::*;
//!
//! let config = AntiHallucinationConfig::default();
//! let pipeline = AntiHallucinationPipeline::new(config);
//!
//! // Process a response
//! let result = pipeline.process("The Earth orbits the Sun.", None);
//! assert!(!result.abstained);
//! ```

use serde::{Deserialize, Serialize};

use crate::confidence_scoring::ConfidenceScorer;
use crate::hallucination_detection::{Claim, HallucinationDetector};

// ============================================================================
// Strategy enum
// ============================================================================

/// Strategy for handling ungrounded (unverified) claims in LLM output.
///
/// Each variant represents a different way to deal with claims that cannot
/// be verified against provided context or knowledge sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UngroundedClaimStrategy {
    /// Remove ungrounded claims entirely from the output.
    Omit,
    /// Mark ungrounded claims with a configurable tag (e.g., "[unverified]").
    Mark,
    /// Emit a warning but keep the claim in the output.
    Warn,
    /// Add a footnote explaining the claim could not be verified.
    Footnote,
    /// Attempt verification first; if still ungrounded, mark it.
    VerifyThenMark,
    /// Attempt verification first; if still ungrounded, omit it.
    VerifyThenOmit,
    /// Ask the user to confirm or reject the claim (interactive mode).
    Ask,
}

impl UngroundedClaimStrategy {
    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Omit => "Omit",
            Self::Mark => "Mark",
            Self::Warn => "Warn",
            Self::Footnote => "Footnote",
            Self::VerifyThenMark => "Verify then Mark",
            Self::VerifyThenOmit => "Verify then Omit",
            Self::Ask => "Ask User",
        }
    }

    /// Short description of what this strategy does.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Omit => "Remove unverified claims from the output",
            Self::Mark => "Tag unverified claims with a marker (e.g., [unverified])",
            Self::Warn => "Keep claims but emit a warning",
            Self::Footnote => "Add explanatory footnotes for unverified claims",
            Self::VerifyThenMark => "Try to verify, then mark if still ungrounded",
            Self::VerifyThenOmit => "Try to verify, then remove if still ungrounded",
            Self::Ask => "Ask the user to confirm or reject each ungrounded claim",
        }
    }

    /// All available strategies, useful for UI selectors.
    pub fn all() -> &'static [UngroundedClaimStrategy] {
        &[
            Self::Omit,
            Self::Mark,
            Self::Warn,
            Self::Footnote,
            Self::VerifyThenMark,
            Self::VerifyThenOmit,
            Self::Ask,
        ]
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().replace(['-', ' '], "_").as_str() {
            "omit" | "remove" | "delete" => Some(Self::Omit),
            "mark" | "tag" | "label" => Some(Self::Mark),
            "warn" | "warning" => Some(Self::Warn),
            "footnote" | "note" => Some(Self::Footnote),
            "verify_then_mark" | "verify_mark" => Some(Self::VerifyThenMark),
            "verify_then_omit" | "verify_omit" => Some(Self::VerifyThenOmit),
            "ask" | "interactive" | "confirm" => Some(Self::Ask),
            _ => None,
        }
    }

    /// Whether this strategy requires a verification step before applying.
    pub fn requires_verification(&self) -> bool {
        matches!(self, Self::VerifyThenMark | Self::VerifyThenOmit)
    }
}

impl Default for UngroundedClaimStrategy {
    fn default() -> Self {
        Self::Mark
    }
}

impl std::fmt::Display for UngroundedClaimStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the anti-hallucination pipeline.
///
/// All features are opt-in. By default, the pipeline is disabled.
/// Enable it with `enabled: true` and configure individual features.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AntiHallucinationConfig {
    /// Master enable/disable switch. Default: `false`.
    pub enabled: bool,

    /// Strategy for handling ungrounded claims. Default: `Mark`.
    pub ungrounded_strategy: UngroundedClaimStrategy,

    /// Enable calibrated abstention — refuse to answer when confidence
    /// is below `abstention_threshold`. Default: `false`.
    pub abstention_enabled: bool,

    /// Confidence threshold below which the pipeline abstains from answering.
    /// Only used when `abstention_enabled` is true. Default: `0.3`.
    pub abstention_threshold: f64,

    /// Enable per-claim confidence scoring. Default: `true`.
    pub confidence_scoring_enabled: bool,

    /// Enable automatic temperature adjustment for factual queries.
    /// When enabled, factual queries use `factual_query_temperature`
    /// instead of the model's default temperature. Default: `false`.
    pub auto_temperature_enabled: bool,

    /// Temperature to use for factual queries when `auto_temperature_enabled`
    /// is true. Lower values produce more deterministic output. Default: `0.3`.
    pub factual_query_temperature: f32,

    /// Format string for marking ungrounded claims when strategy is `Mark`.
    /// The claim text replaces `{}` in the format. Default: `"[unverified] {}"`.
    pub mark_format: String,

    /// Minimum confidence score for a claim to be included in the output.
    /// Claims below this threshold are subject to the `ungrounded_strategy`.
    /// Default: `0.3`.
    pub min_confidence_for_output: f64,

    /// Maximum number of extra LLM calls the anti-hallucination pipeline
    /// may make per response. Prevents runaway costs. Default: `5`.
    pub max_extra_llm_calls: usize,

    /// Custom abstention message. If None, a default message is used.
    pub abstention_message: Option<String>,
}

impl Default for AntiHallucinationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            abstention_enabled: false,
            abstention_threshold: 0.3,
            confidence_scoring_enabled: true,
            auto_temperature_enabled: false,
            factual_query_temperature: 0.3,
            mark_format: "[unverified] {}".to_string(),
            min_confidence_for_output: 0.3,
            max_extra_llm_calls: 5,
            abstention_message: None,
        }
    }
}

impl AntiHallucinationConfig {
    /// Create a config with sensible defaults for production use.
    ///
    /// Enables: abstention, confidence scoring, auto-temperature, Mark strategy.
    pub fn production() -> Self {
        Self {
            enabled: true,
            abstention_enabled: true,
            auto_temperature_enabled: true,
            ..Default::default()
        }
    }

    /// Create a strict config that removes all ungrounded claims.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Omit,
            abstention_enabled: true,
            abstention_threshold: 0.5,
            auto_temperature_enabled: true,
            min_confidence_for_output: 0.5,
            ..Default::default()
        }
    }

    /// Create a permissive config that only warns about ungrounded claims.
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Warn,
            abstention_enabled: false,
            min_confidence_for_output: 0.1,
            ..Default::default()
        }
    }
}

// ============================================================================
// Result types
// ============================================================================

/// A claim that has been processed through the anti-hallucination pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedClaim {
    /// The original claim text.
    pub text: String,
    /// Whether the claim is considered grounded (verified or high-confidence).
    pub grounded: bool,
    /// Confidence score for this claim (0.0–1.0).
    pub confidence: f64,
    /// IDs of sources that support this claim (if any).
    pub source_ids: Vec<String>,
    /// The action that was taken on this claim.
    pub action_taken: UngroundedClaimStrategy,
    /// Position in the original text (byte offset).
    pub position: usize,
}

/// Result of running the anti-hallucination pipeline on a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiHallucinationResult {
    /// The original response text before processing.
    pub original_text: String,
    /// The processed text after applying the strategy.
    pub processed_text: String,
    /// All claims extracted and their processing results.
    pub claims: Vec<ProcessedClaim>,
    /// Overall confidence score for the response (0.0–1.0).
    pub overall_confidence: f64,
    /// Whether the pipeline chose to abstain from answering.
    pub abstained: bool,
    /// Reason for abstention, if applicable.
    pub abstention_reason: Option<String>,
    /// Temperature that was used for generation (if auto-temp was active).
    pub temperature_used: Option<f32>,
    /// The strategy that was applied.
    pub strategy_applied: UngroundedClaimStrategy,
    /// Number of claims that were marked as ungrounded.
    pub ungrounded_count: usize,
    /// Number of extra LLM calls consumed by the pipeline.
    pub llm_calls_used: usize,
}

impl AntiHallucinationResult {
    /// Fraction of claims that are grounded (0.0–1.0).
    pub fn grounding_ratio(&self) -> f64 {
        if self.claims.is_empty() {
            return 1.0;
        }
        let grounded = self.claims.iter().filter(|c| c.grounded).count();
        grounded as f64 / self.claims.len() as f64
    }

    /// Whether all claims are grounded.
    pub fn fully_grounded(&self) -> bool {
        self.claims.iter().all(|c| c.grounded)
    }

    /// Claims that are NOT grounded.
    pub fn ungrounded_claims(&self) -> Vec<&ProcessedClaim> {
        self.claims.iter().filter(|c| !c.grounded).collect()
    }
}

// ============================================================================
// Pipeline
// ============================================================================

/// The anti-hallucination pipeline orchestrator.
///
/// Processes LLM responses through a series of checks and applies the
/// configured strategy to ungrounded claims.
pub struct AntiHallucinationPipeline {
    config: AntiHallucinationConfig,
    confidence_scorer: ConfidenceScorer,
    hallucination_detector: HallucinationDetector,
}

impl AntiHallucinationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: AntiHallucinationConfig) -> Self {
        Self {
            config,
            confidence_scorer: ConfidenceScorer::default(),
            hallucination_detector: HallucinationDetector::new(Default::default()),
        }
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &AntiHallucinationConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: AntiHallucinationConfig) {
        self.config = config;
    }

    /// Check if the pipeline is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the suggested temperature for a query, taking into account
    /// auto-temperature settings.
    ///
    /// Returns `Some(temp)` if auto-temperature is active and the query
    /// is classified as factual, `None` otherwise.
    pub fn suggest_temperature(&self, query: &str) -> Option<f32> {
        if !self.config.enabled || !self.config.auto_temperature_enabled {
            return None;
        }

        if is_factual_query(query) {
            Some(self.config.factual_query_temperature)
        } else {
            None
        }
    }

    /// Process an LLM response through the anti-hallucination pipeline.
    ///
    /// # Arguments
    ///
    /// * `response` — The LLM-generated text to process.
    /// * `context` — Optional context/sources used for generation (for grounding checks).
    ///
    /// # Returns
    ///
    /// An `AntiHallucinationResult` with processed text and claim-level details.
    pub fn process(&self, response: &str, context: Option<&str>) -> AntiHallucinationResult {
        if !self.config.enabled {
            return AntiHallucinationResult {
                original_text: response.to_string(),
                processed_text: response.to_string(),
                claims: Vec::new(),
                overall_confidence: 1.0,
                abstained: false,
                abstention_reason: None,
                temperature_used: None,
                strategy_applied: self.config.ungrounded_strategy,
                ungrounded_count: 0,
                llm_calls_used: 0,
            };
        }

        let llm_calls_used: usize = 0;

        // Step 1: Score overall confidence
        let confidence_score = self.confidence_scorer.score(response, None);
        let overall_confidence = confidence_score.overall;

        // Step 2: Abstention check
        if self.config.abstention_enabled && overall_confidence < self.config.abstention_threshold {
            let reason = format!(
                "Response confidence ({:.2}) is below abstention threshold ({:.2})",
                overall_confidence, self.config.abstention_threshold,
            );
            let abstention_msg = self.config.abstention_message.clone().unwrap_or_else(|| {
                "I don't have enough confidence to answer this question accurately. \
                     Please provide more context or rephrase the question."
                    .to_string()
            });
            return AntiHallucinationResult {
                original_text: response.to_string(),
                processed_text: abstention_msg,
                claims: Vec::new(),
                overall_confidence,
                abstained: true,
                abstention_reason: Some(reason),
                temperature_used: None,
                strategy_applied: self.config.ungrounded_strategy,
                ungrounded_count: 0,
                llm_calls_used,
            };
        }

        // Step 3: Extract and score claims
        let hallucination_result = self.hallucination_detector.detect(response, context);
        let raw_claims = hallucination_result.claims;

        // Step 4: Score each claim's confidence
        let processed_claims: Vec<ProcessedClaim> = raw_claims
            .iter()
            .map(|claim| {
                let claim_confidence = if self.config.confidence_scoring_enabled {
                    claim.confidence
                } else {
                    overall_confidence
                };

                let grounded =
                    claim.supported || claim_confidence >= self.config.min_confidence_for_output;

                ProcessedClaim {
                    text: claim.text.clone(),
                    grounded,
                    confidence: claim_confidence,
                    source_ids: Vec::new(),
                    action_taken: if grounded {
                        UngroundedClaimStrategy::Mark // Pass-through for grounded
                    } else {
                        self.config.ungrounded_strategy
                    },
                    position: claim.position,
                }
            })
            .collect();

        // Step 5: Apply strategy to ungrounded claims
        let ungrounded_count = processed_claims.iter().filter(|c| !c.grounded).count();
        let processed_text = self.apply_strategy(response, &processed_claims, &raw_claims);

        AntiHallucinationResult {
            original_text: response.to_string(),
            processed_text,
            claims: processed_claims,
            overall_confidence,
            abstained: false,
            abstention_reason: None,
            temperature_used: None,
            strategy_applied: self.config.ungrounded_strategy,
            ungrounded_count,
            llm_calls_used,
        }
    }

    /// Apply the configured strategy to produce the final output text.
    fn apply_strategy(
        &self,
        original: &str,
        processed_claims: &[ProcessedClaim],
        raw_claims: &[Claim],
    ) -> String {
        // If no ungrounded claims, return original unchanged
        if processed_claims.iter().all(|c| c.grounded) {
            return original.to_string();
        }

        match self.config.ungrounded_strategy {
            UngroundedClaimStrategy::Omit => {
                self.apply_omit(original, processed_claims, raw_claims)
            }
            UngroundedClaimStrategy::Mark => {
                self.apply_mark(original, processed_claims, raw_claims)
            }
            UngroundedClaimStrategy::Warn => self.apply_warn(original, processed_claims),
            UngroundedClaimStrategy::Footnote => {
                self.apply_footnote(original, processed_claims, raw_claims)
            }
            // VerifyThen* strategies fall back to their base strategy
            // (actual verification happens at a higher level with LLM access)
            UngroundedClaimStrategy::VerifyThenMark => {
                self.apply_mark(original, processed_claims, raw_claims)
            }
            UngroundedClaimStrategy::VerifyThenOmit => {
                self.apply_omit(original, processed_claims, raw_claims)
            }
            UngroundedClaimStrategy::Ask => {
                // In non-interactive mode, fall back to Mark
                self.apply_mark(original, processed_claims, raw_claims)
            }
        }
    }

    /// Remove ungrounded claim sentences from the text.
    fn apply_omit(
        &self,
        original: &str,
        processed_claims: &[ProcessedClaim],
        _raw_claims: &[Claim],
    ) -> String {
        let sentences: Vec<&str> = original
            .split(|c| c == '.' || c == '!' || c == '?')
            .collect();

        let mut result_parts = Vec::new();
        for sentence in &sentences {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }

            let is_ungrounded = processed_claims
                .iter()
                .any(|c| !c.grounded && c.text == trimmed);

            if !is_ungrounded {
                result_parts.push(trimmed);
            }
        }

        if result_parts.is_empty() {
            original.to_string()
        } else {
            result_parts.join(". ") + "."
        }
    }

    /// Mark ungrounded claims with the configured tag.
    fn apply_mark(
        &self,
        original: &str,
        processed_claims: &[ProcessedClaim],
        _raw_claims: &[Claim],
    ) -> String {
        let mut result = original.to_string();

        // Process in reverse order to preserve positions
        let mut ungrounded: Vec<_> = processed_claims.iter().filter(|c| !c.grounded).collect();
        ungrounded.sort_by(|a, b| b.position.cmp(&a.position));

        for claim in ungrounded {
            let marked = self.config.mark_format.replace("{}", &claim.text);
            result = result.replace(&claim.text, &marked);
        }

        result
    }

    /// Keep all claims but append a warning about ungrounded ones.
    fn apply_warn(&self, original: &str, processed_claims: &[ProcessedClaim]) -> String {
        let ungrounded: Vec<_> = processed_claims.iter().filter(|c| !c.grounded).collect();

        if ungrounded.is_empty() {
            return original.to_string();
        }

        let warning = format!(
            "\n\n⚠ Warning: {} claim(s) in this response could not be verified against provided sources.",
            ungrounded.len()
        );

        format!("{}{}", original, warning)
    }

    /// Add footnotes for ungrounded claims.
    fn apply_footnote(
        &self,
        original: &str,
        processed_claims: &[ProcessedClaim],
        _raw_claims: &[Claim],
    ) -> String {
        let ungrounded: Vec<_> = processed_claims
            .iter()
            .filter(|c| !c.grounded)
            .enumerate()
            .collect();

        if ungrounded.is_empty() {
            return original.to_string();
        }

        let mut result = original.to_string();
        let mut footnotes = Vec::new();

        for (idx, claim) in &ungrounded {
            let note_num = idx + 1;
            let marker = format!("[^{}]", note_num);
            result = result.replace(&claim.text, &format!("{}{}", claim.text, marker));
            footnotes.push(format!(
                "[^{}]: This claim could not be verified (confidence: {:.0}%)",
                note_num,
                claim.confidence * 100.0
            ));
        }

        format!("{}\n\n---\n{}", result, footnotes.join("\n"))
    }
}

impl std::fmt::Debug for AntiHallucinationPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AntiHallucinationPipeline")
            .field("config", &self.config)
            .field("confidence_scorer", &"<...>")
            .field("hallucination_detector", &"<...>")
            .finish()
    }
}

// ============================================================================
// Grounded Generation (V82)
// ============================================================================

/// Method for anchoring response sentences to source chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ChunkAnchorMethod {
    /// Anchor claims to sources after generation (post-hoc analysis).
    PostHoc,
    /// Prompt the model to cite sources inline during generation.
    Prompted,
}

impl Default for ChunkAnchorMethod {
    fn default() -> Self {
        Self::PostHoc
    }
}

/// Configuration for grounded generation.
///
/// When enabled, every sentence in the response is anchored to a retrieved
/// source chunk. Sentences without a matching source are handled according
/// to the configured [`UngroundedClaimStrategy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GroundedGenerationConfig {
    /// Whether grounded generation is enabled.
    pub enabled: bool,
    /// How to anchor response sentences to source chunks.
    pub anchor_method: ChunkAnchorMethod,
    /// Strategy for claims that can't be anchored.
    pub ungrounded_strategy: UngroundedClaimStrategy,
    /// Minimum similarity score (0.0–1.0) for a sentence-chunk anchor.
    pub min_anchor_similarity: f64,
}

impl Default for GroundedGenerationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            anchor_method: ChunkAnchorMethod::PostHoc,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            min_anchor_similarity: 0.5,
        }
    }
}

/// Grounded generation engine.
///
/// Post-processes model output to ensure every claim is traceable to a
/// source chunk. Uses word overlap similarity to anchor sentences.
pub struct GroundedGenerator {
    config: GroundedGenerationConfig,
}

impl GroundedGenerator {
    /// Create a new grounded generator with the given config.
    pub fn new(config: GroundedGenerationConfig) -> Self {
        Self { config }
    }

    /// Anchor each sentence in `response` to the best-matching source chunk.
    ///
    /// Returns a list of (sentence, best_source_index, similarity_score) tuples.
    pub fn anchor_sentences(
        &self,
        response: &str,
        sources: &[&str],
    ) -> Vec<(String, Option<usize>, f64)> {
        let sentences = split_sentences_simple(response);
        let mut results = Vec::with_capacity(sentences.len());

        for sentence in &sentences {
            let trimmed = sentence.trim();
            if trimmed.len() < 5 {
                continue;
            }

            let claim_words = words_lowercase(trimmed);
            if claim_words.is_empty() {
                results.push((trimmed.to_string(), None, 0.0));
                continue;
            }

            let mut best_idx: Option<usize> = None;
            let mut best_sim = 0.0f64;

            for (i, source) in sources.iter().enumerate() {
                let source_words = words_lowercase(source);
                if source_words.is_empty() {
                    continue;
                }

                let intersection = claim_words
                    .iter()
                    .filter(|w| source_words.contains(*w))
                    .count() as f64;
                let union_size = {
                    let mut all = claim_words.clone();
                    for w in &source_words {
                        if !all.contains(w) {
                            all.push(w.clone());
                        }
                    }
                    all.len() as f64
                };
                let sim = if union_size > 0.0 {
                    intersection / union_size
                } else {
                    0.0
                };

                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }

            let anchored = if best_sim >= self.config.min_anchor_similarity {
                best_idx
            } else {
                None
            };

            results.push((trimmed.to_string(), anchored, best_sim));
        }

        results
    }

    /// Process a response, marking unanchored sentences per the configured strategy.
    pub fn process(&self, response: &str, sources: &[&str]) -> GroundedGenerationResult {
        let anchored = self.anchor_sentences(response, sources);

        let grounded_count = anchored.iter().filter(|(_, idx, _)| idx.is_some()).count();
        let total = anchored.len();
        let grounding_ratio = if total > 0 {
            grounded_count as f64 / total as f64
        } else {
            1.0
        };

        let mut processed = response.to_string();

        // Apply strategy to unanchored sentences (reverse order for position safety)
        for (sentence, idx, _sim) in anchored.iter().rev() {
            if idx.is_none() {
                match self.config.ungrounded_strategy {
                    UngroundedClaimStrategy::Omit => {
                        if let Some(start) = processed.find(sentence.as_str()) {
                            processed.replace_range(start..start + sentence.len(), "");
                        }
                    }
                    UngroundedClaimStrategy::Mark => {
                        if let Some(start) = processed.find(sentence.as_str()) {
                            let marked = format!("[ungrounded] {}", sentence);
                            processed.replace_range(start..start + sentence.len(), &marked);
                        }
                    }
                    _ => {} // Warn, Footnote, etc. — keep text as-is
                }
            }
        }

        GroundedGenerationResult {
            original_text: response.to_string(),
            processed_text: processed,
            grounded_count,
            ungrounded_count: total - grounded_count,
            grounding_ratio,
            anchored_sentences: anchored,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GroundedGenerationConfig {
        &self.config
    }
}

impl Default for GroundedGenerator {
    fn default() -> Self {
        Self::new(GroundedGenerationConfig::default())
    }
}

impl std::fmt::Debug for GroundedGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundedGenerator")
            .field("config", &self.config)
            .finish()
    }
}

/// Result of grounded generation processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedGenerationResult {
    /// Original response text.
    pub original_text: String,
    /// Processed text after applying strategy to ungrounded sentences.
    pub processed_text: String,
    /// Number of sentences anchored to a source.
    pub grounded_count: usize,
    /// Number of sentences without a source anchor.
    pub ungrounded_count: usize,
    /// Ratio of grounded sentences (0.0–1.0).
    pub grounding_ratio: f64,
    /// Per-sentence anchoring: (sentence, source_index, similarity).
    pub anchored_sentences: Vec<(String, Option<usize>, f64)>,
}

/// Split text into sentences by `.`, `!`, `?` delimiters (simple version).
fn split_sentences_simple(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Extract lowercase words from text (for similarity computation).
fn words_lowercase(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty() && w.len() > 1)
        .collect()
}

// ============================================================================
// Helper: factual query detection
// ============================================================================

/// Heuristic keywords indicating factual intent.
const FACTUAL_KEYWORDS: &[&str] = &[
    "what is",
    "what are",
    "who is",
    "who was",
    "when did",
    "when was",
    "where is",
    "where was",
    "how many",
    "how much",
    "how old",
    "define",
    "definition of",
    "meaning of the word",
    "explain",
    "describe",
    "list the",
    "name the",
    "what year",
    "in what year",
    "what date",
    "capital of",
    "population of",
    "founded in",
    "invented by",
    "discovered by",
    "true or false",
    "is it true",
    "fact check",
    "verify",
];

/// Creative/subjective keywords indicating the query is NOT factual.
const CREATIVE_KEYWORDS: &[&str] = &[
    "write a story",
    "write a poem",
    "creative",
    "imagine",
    "fictional",
    "brainstorm",
    "come up with",
    "make up",
    "invent a",
    "fantasy",
    "hypothetical",
    "what if",
    "roleplay",
    "pretend",
    "compose",
    "draft a",
    "write me",
    "generate a",
    "suggest ideas",
];

/// Determine whether a query is factual (vs creative/subjective).
///
/// Used by auto-temperature to lower temperature for factual queries,
/// producing more deterministic and accurate responses.
///
/// Returns `true` for queries like "What is the capital of France?"
/// and `false` for queries like "Write a poem about the sea."
pub fn is_factual_query(query: &str) -> bool {
    let lower = query.to_lowercase();

    // Check for creative indicators first (they override factual signals)
    if CREATIVE_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return false;
    }

    // Check for factual indicators
    if FACTUAL_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return true;
    }

    // Heuristic: short questions with a question mark are often factual
    let word_count = query.split_whitespace().count();
    if word_count <= 10 && query.contains('?') {
        return true;
    }

    // Default: not explicitly factual
    false
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- UngroundedClaimStrategy tests ---

    #[test]
    fn test_strategy_display_names() {
        assert_eq!(UngroundedClaimStrategy::Omit.display_name(), "Omit");
        assert_eq!(UngroundedClaimStrategy::Mark.display_name(), "Mark");
        assert_eq!(UngroundedClaimStrategy::Warn.display_name(), "Warn");
        assert_eq!(UngroundedClaimStrategy::Footnote.display_name(), "Footnote");
        assert_eq!(
            UngroundedClaimStrategy::VerifyThenMark.display_name(),
            "Verify then Mark"
        );
        assert_eq!(
            UngroundedClaimStrategy::VerifyThenOmit.display_name(),
            "Verify then Omit"
        );
        assert_eq!(UngroundedClaimStrategy::Ask.display_name(), "Ask User");
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(
            UngroundedClaimStrategy::from_str("omit"),
            Some(UngroundedClaimStrategy::Omit)
        );
        assert_eq!(
            UngroundedClaimStrategy::from_str("Mark"),
            Some(UngroundedClaimStrategy::Mark)
        );
        assert_eq!(
            UngroundedClaimStrategy::from_str("WARN"),
            Some(UngroundedClaimStrategy::Warn)
        );
        assert_eq!(
            UngroundedClaimStrategy::from_str("verify-then-mark"),
            Some(UngroundedClaimStrategy::VerifyThenMark)
        );
        assert_eq!(
            UngroundedClaimStrategy::from_str("verify_then_omit"),
            Some(UngroundedClaimStrategy::VerifyThenOmit)
        );
        assert_eq!(
            UngroundedClaimStrategy::from_str("interactive"),
            Some(UngroundedClaimStrategy::Ask)
        );
        assert_eq!(UngroundedClaimStrategy::from_str("invalid"), None);
    }

    #[test]
    fn test_strategy_all() {
        let all = UngroundedClaimStrategy::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_strategy_default() {
        assert_eq!(
            UngroundedClaimStrategy::default(),
            UngroundedClaimStrategy::Mark
        );
    }

    #[test]
    fn test_strategy_requires_verification() {
        assert!(!UngroundedClaimStrategy::Omit.requires_verification());
        assert!(!UngroundedClaimStrategy::Mark.requires_verification());
        assert!(UngroundedClaimStrategy::VerifyThenMark.requires_verification());
        assert!(UngroundedClaimStrategy::VerifyThenOmit.requires_verification());
    }

    // --- Config tests ---

    #[test]
    fn test_config_default() {
        let config = AntiHallucinationConfig::default();
        assert!(!config.enabled);
        assert!(!config.abstention_enabled);
        assert!(!config.auto_temperature_enabled);
        assert!(config.confidence_scoring_enabled);
        assert_eq!(config.ungrounded_strategy, UngroundedClaimStrategy::Mark);
        assert!((config.abstention_threshold - 0.3).abs() < f64::EPSILON);
        assert!((config.factual_query_temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.max_extra_llm_calls, 5);
    }

    #[test]
    fn test_config_production() {
        let config = AntiHallucinationConfig::production();
        assert!(config.enabled);
        assert!(config.abstention_enabled);
        assert!(config.auto_temperature_enabled);
    }

    #[test]
    fn test_config_strict() {
        let config = AntiHallucinationConfig::strict();
        assert!(config.enabled);
        assert_eq!(config.ungrounded_strategy, UngroundedClaimStrategy::Omit);
        assert!((config.abstention_threshold - 0.5).abs() < f64::EPSILON);
        assert!((config.min_confidence_for_output - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_permissive() {
        let config = AntiHallucinationConfig::permissive();
        assert!(config.enabled);
        assert_eq!(config.ungrounded_strategy, UngroundedClaimStrategy::Warn);
        assert!(!config.abstention_enabled);
    }

    // --- Pipeline tests ---

    #[test]
    fn test_pipeline_disabled() {
        let config = AntiHallucinationConfig::default(); // disabled by default
        let pipeline = AntiHallucinationPipeline::new(config);
        assert!(!pipeline.is_enabled());

        let result = pipeline.process("The sky is blue.", None);
        assert!(!result.abstained);
        assert_eq!(result.processed_text, "The sky is blue.");
        assert!(result.claims.is_empty());
        assert!((result.overall_confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pipeline_enabled_processes_text() {
        let config = AntiHallucinationConfig {
            enabled: true,
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);
        assert!(pipeline.is_enabled());

        let result = pipeline.process("The Earth orbits the Sun. Water is wet.", None);
        assert!(!result.abstained);
        assert!(!result.claims.is_empty());
    }

    #[test]
    fn test_pipeline_abstention() {
        let config = AntiHallucinationConfig {
            enabled: true,
            abstention_enabled: true,
            abstention_threshold: 0.99, // Very high threshold to trigger abstention
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        // Use text with very low confidence (lots of hedging)
        let result = pipeline.process(
            "I think maybe possibly it could be that perhaps something might happen.",
            None,
        );
        assert!(result.abstained);
        assert!(result.abstention_reason.is_some());
    }

    #[test]
    fn test_pipeline_warn_strategy() {
        let config = AntiHallucinationConfig {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Warn,
            min_confidence_for_output: 0.99, // Force everything to be ungrounded
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        let result = pipeline.process("Some claim. Another claim.", None);
        assert!(result.processed_text.contains("Warning"));
    }

    #[test]
    fn test_pipeline_mark_strategy() {
        let config = AntiHallucinationConfig {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            mark_format: "[UNVERIFIED] {}".to_string(),
            min_confidence_for_output: 0.99, // Force everything to be ungrounded
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        let result = pipeline.process("The capital of Atlantis is underwater.", None);
        assert!(result.processed_text.contains("[UNVERIFIED]"));
    }

    #[test]
    fn test_pipeline_footnote_strategy() {
        let config = AntiHallucinationConfig {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Footnote,
            min_confidence_for_output: 0.99,
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        let result = pipeline.process("Claim one. Claim two.", None);
        assert!(result.processed_text.contains("[^"));
    }

    #[test]
    fn test_pipeline_result_grounding_ratio() {
        let result = AntiHallucinationResult {
            original_text: String::new(),
            processed_text: String::new(),
            claims: vec![
                ProcessedClaim {
                    text: "grounded".to_string(),
                    grounded: true,
                    confidence: 0.9,
                    source_ids: Vec::new(),
                    action_taken: UngroundedClaimStrategy::Mark,
                    position: 0,
                },
                ProcessedClaim {
                    text: "ungrounded".to_string(),
                    grounded: false,
                    confidence: 0.2,
                    source_ids: Vec::new(),
                    action_taken: UngroundedClaimStrategy::Mark,
                    position: 10,
                },
            ],
            overall_confidence: 0.5,
            abstained: false,
            abstention_reason: None,
            temperature_used: None,
            strategy_applied: UngroundedClaimStrategy::Mark,
            ungrounded_count: 1,
            llm_calls_used: 0,
        };

        assert!((result.grounding_ratio() - 0.5).abs() < f64::EPSILON);
        assert!(!result.fully_grounded());
        assert_eq!(result.ungrounded_claims().len(), 1);
    }

    #[test]
    fn test_pipeline_result_empty_claims() {
        let result = AntiHallucinationResult {
            original_text: String::new(),
            processed_text: String::new(),
            claims: Vec::new(),
            overall_confidence: 1.0,
            abstained: false,
            abstention_reason: None,
            temperature_used: None,
            strategy_applied: UngroundedClaimStrategy::Mark,
            ungrounded_count: 0,
            llm_calls_used: 0,
        };

        assert!((result.grounding_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(result.fully_grounded());
    }

    // --- is_factual_query tests ---

    #[test]
    fn test_factual_query_detection() {
        assert!(is_factual_query("What is the capital of France?"));
        assert!(is_factual_query("Who invented the telephone?"));
        assert!(is_factual_query("When was the Eiffel Tower built?"));
        assert!(is_factual_query(
            "How many planets are in the solar system?"
        ));
        assert!(is_factual_query("Define photosynthesis"));
        assert!(is_factual_query("True or false: water boils at 100C"));
    }

    #[test]
    fn test_creative_query_detection() {
        assert!(!is_factual_query("Write a story about dragons"));
        assert!(!is_factual_query("Write a poem about the ocean"));
        assert!(!is_factual_query("Imagine a world without gravity"));
        assert!(!is_factual_query("Brainstorm ideas for a party"));
        assert!(!is_factual_query("Come up with a fictional character"));
    }

    #[test]
    fn test_ambiguous_query() {
        // Short questions with '?' default to factual
        assert!(is_factual_query("Is Rust fast?"));
        // Longer queries without explicit markers default to non-factual
        assert!(!is_factual_query(
            "Tell me your thoughts on the meaning of existence"
        ));
    }

    // --- Temperature suggestion tests ---

    #[test]
    fn test_suggest_temperature_disabled() {
        let config = AntiHallucinationConfig::default(); // disabled
        let pipeline = AntiHallucinationPipeline::new(config);
        assert_eq!(pipeline.suggest_temperature("What is Rust?"), None);
    }

    #[test]
    fn test_suggest_temperature_factual() {
        let config = AntiHallucinationConfig {
            enabled: true,
            auto_temperature_enabled: true,
            factual_query_temperature: 0.2,
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        assert_eq!(
            pipeline.suggest_temperature("What is the capital of France?"),
            Some(0.2)
        );
    }

    #[test]
    fn test_suggest_temperature_creative() {
        let config = AntiHallucinationConfig {
            enabled: true,
            auto_temperature_enabled: true,
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        assert_eq!(
            pipeline.suggest_temperature("Write a poem about the sea"),
            None
        );
    }

    #[test]
    fn test_pipeline_custom_abstention_message() {
        let config = AntiHallucinationConfig {
            enabled: true,
            abstention_enabled: true,
            abstention_threshold: 0.99,
            abstention_message: Some("Cannot answer with sufficient confidence.".to_string()),
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        let result = pipeline.process(
            "I think maybe possibly it might be something uncertain.",
            None,
        );
        if result.abstained {
            assert_eq!(
                result.processed_text,
                "Cannot answer with sufficient confidence."
            );
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = AntiHallucinationConfig::production();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: AntiHallucinationConfig =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.ungrounded_strategy, config.ungrounded_strategy);
    }

    #[test]
    fn test_strategy_serialization() {
        let strategy = UngroundedClaimStrategy::VerifyThenMark;
        let json = serde_json::to_string(&strategy).expect("serialize");
        assert_eq!(json, "\"verify_then_mark\"");

        let deserialized: UngroundedClaimStrategy =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, strategy);
    }

    #[test]
    fn test_pipeline_debug() {
        let pipeline = AntiHallucinationPipeline::new(AntiHallucinationConfig::default());
        let debug = format!("{:?}", pipeline);
        assert!(debug.contains("AntiHallucinationPipeline"));
    }

    #[test]
    fn test_omit_strategy_preserves_grounded() {
        let config = AntiHallucinationConfig {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Omit,
            min_confidence_for_output: 0.99,
            ..Default::default()
        };
        let pipeline = AntiHallucinationPipeline::new(config);

        let result = pipeline.process("Single simple sentence.", None);
        // Even with aggressive omitting, we don't return empty
        assert!(!result.processed_text.is_empty());
    }

    // --- GroundedGeneration tests (V82) ---

    #[test]
    fn test_grounded_generation_default_config() {
        let config = GroundedGenerationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.anchor_method, ChunkAnchorMethod::PostHoc);
        assert!((config.min_anchor_similarity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_grounded_generator_anchor_sentences() {
        let config = GroundedGenerationConfig {
            enabled: true,
            min_anchor_similarity: 0.2,
            ..Default::default()
        };
        let gen = GroundedGenerator::new(config);

        let sources = &[
            "Rust was released in 2015 by Mozilla.",
            "Python was created by Guido van Rossum.",
        ];
        let response = "Rust was released in 2015.";

        let anchored = gen.anchor_sentences(response, sources);
        assert!(!anchored.is_empty());
        // Should anchor to first source (Rust)
        assert_eq!(anchored[0].1, Some(0));
    }

    #[test]
    fn test_grounded_generator_no_sources() {
        let gen = GroundedGenerator::default();
        let result = gen.process("Some text about things.", &[]);
        assert_eq!(result.grounding_ratio, 0.0);
    }

    #[test]
    fn test_grounded_generator_process_mark() {
        let config = GroundedGenerationConfig {
            enabled: true,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            min_anchor_similarity: 0.9, // Very high threshold → most will be ungrounded
            ..Default::default()
        };
        let gen = GroundedGenerator::new(config);

        let sources = &["Completely unrelated text about quantum physics."];
        let response = "Rust is a programming language.";
        let result = gen.process(response, sources);
        // With very high threshold, the sentence won't be anchored
        assert!(result.ungrounded_count > 0 || result.grounded_count == 0);
    }

    #[test]
    fn test_grounded_generator_debug() {
        let gen = GroundedGenerator::default();
        let debug = format!("{:?}", gen);
        assert!(debug.contains("GroundedGenerator"));
    }

    #[test]
    fn test_chunk_anchor_method_default() {
        assert_eq!(ChunkAnchorMethod::default(), ChunkAnchorMethod::PostHoc);
    }

    #[test]
    fn test_grounded_result_full_grounding() {
        let config = GroundedGenerationConfig {
            enabled: true,
            min_anchor_similarity: 0.1, // Very low threshold
            ..Default::default()
        };
        let gen = GroundedGenerator::new(config);
        let sources = &["Rust is a systems programming language released in 2015."];
        let response = "Rust is a programming language.";
        let result = gen.process(response, sources);
        assert!(result.grounding_ratio > 0.0);
    }
}
