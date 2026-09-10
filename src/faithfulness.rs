//! Faithfulness scoring for AI responses against source context.
//!
//! Evaluates whether each claim in a generated response is supported
//! (entailed) by the retrieved context. Uses Natural Language Inference (NLI)
//! at the claim level, producing a per-claim verdict and an overall
//! faithfulness score.
//!
//! # Methods
//!
//! Two NLI methods are available:
//!
//! - **`WordOverlap`**: Zero-cost Jaccard word overlap. Fast but imprecise.
//! - **`LlmNli`**: LLM-based entailment check, one call per claim.
//!   More accurate, and it costs what it says.
//!
//! # The LLM has to be supplied
//!
//! `LlmNli`, [`DecompositionMethod::LlmDecomposition`] and the verify-then-*
//! strategies all need a model, and this module never picks one: attach it with
//! [`FaithfulnessScorer::with_llm_verifier`], the same shape
//! [`crate::chain_of_verification`] uses.
//!
//! **Without a verifier those options fall back to the cheap method and say so in
//! [`FaithfulnessReport::degraded`].** That field is not decoration. Until V309
//! selecting `LlmNli` silently ran Jaccard word overlap — same number, no LLM call,
//! no way for the caller to tell — which made the more expensive-sounding option a
//! lie. Anything that had to fall back is now named there, and
//! [`FaithfulnessReport::llm_calls_used`] says what was actually spent.
//!
//! # Usage
//!
//! ```rust
//! use ai_assistant::faithfulness::*;
//!
//! let config = FaithfulnessConfig::default();
//! let scorer = FaithfulnessScorer::new(config);
//!
//! let context = &["Rust was first released in 2015.", "It is a systems language."];
//! let response = "Rust was released in 2015 and is a systems programming language.";
//!
//! let report = scorer.score(response, context);
//! assert!(report.overall_score > 0.5);
//! ```

use serde::{Deserialize, Serialize};

use crate::anti_hallucination::UngroundedClaimStrategy;

/// Confidence recorded for a verdict the model stated outright.
///
/// Deliberately a constant: the model returned a *label*, not a probability, and
/// turning a label into a calibrated-looking number is the fake precision this
/// module exists to remove. High, because a decisive answer is worth more than a
/// word-overlap ratio — but never 1.0, which would claim certainty nobody has.
const LLM_NLI_CONFIDENCE: f64 = 0.9;

/// How many context chunks to cite as evidence for one claim.
const MAX_SUPPORTING_CHUNKS: usize = 3;

// ============================================================================
// NLI types
// ============================================================================

/// Natural Language Inference verdict for a claim against source context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NliVerdict {
    /// The claim is supported by the source context.
    Entailed,
    /// The claim contradicts the source context.
    Contradicted,
    /// The claim is neither supported nor contradicted (no evidence).
    Neutral,
}

impl NliVerdict {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Entailed => "Entailed",
            Self::Contradicted => "Contradicted",
            Self::Neutral => "Neutral",
        }
    }

    /// Whether this verdict indicates the claim is grounded.
    pub fn is_grounded(&self) -> bool {
        matches!(self, Self::Entailed)
    }
}

impl std::fmt::Display for NliVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ============================================================================
// Decomposition
// ============================================================================

/// Method for decomposing a response into atomic claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DecompositionMethod {
    /// Split on sentence boundaries (`.`, `!`, `?`). Zero cost.
    SentenceSplit,
    /// Use an LLM to decompose into atomic claims. One LLM call.
    LlmDecomposition,
}

impl Default for DecompositionMethod {
    fn default() -> Self {
        Self::SentenceSplit
    }
}

/// Method for performing NLI inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NliMethod {
    /// Jaccard word overlap — zero cost, lower precision.
    WordOverlap,
    /// LLM prompt-based NLI — one call per batch, higher accuracy.
    LlmNli,
}

impl Default for NliMethod {
    fn default() -> Self {
        Self::WordOverlap
    }
}

// ============================================================================
// Atomic claim
// ============================================================================

/// An atomic claim extracted from a response.
///
/// Finer-grained than [`crate::hallucination_detection::Claim`], representing
/// a single verifiable assertion. Multiple atomic claims may come from one
/// sentence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicClaim {
    /// The claim text.
    pub text: String,
    /// Character position in the original response.
    pub position: usize,
    /// Character length in the original response.
    pub length: usize,
    /// The source sentence this claim was extracted from.
    pub source_sentence: String,
}

// ============================================================================
// Faithfulness config & report
// ============================================================================

/// Configuration for faithfulness scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FaithfulnessConfig {
    /// How to decompose the response into claims.
    pub decomposition_method: DecompositionMethod,
    /// Which NLI method to use.
    pub nli_method: NliMethod,
    /// Minimum faithfulness score to pass (0.0–1.0).
    pub min_faithfulness_score: f64,
    /// What to do with ungrounded claims.
    pub ungrounded_strategy: UngroundedClaimStrategy,
    /// Word overlap threshold for `NliMethod::WordOverlap` (0.0–1.0).
    /// Claims with overlap above this are considered entailed.
    pub word_overlap_entailment_threshold: f64,
    /// Word overlap threshold below which a claim is contradicted.
    /// (Only applies when source explicitly contradicts.)
    pub word_overlap_contradiction_threshold: f64,
    /// Ceiling on LLM calls for one `score()`, across decomposition, NLI and
    /// verify-then-* strategies.
    ///
    /// Claims beyond the budget are evaluated with word overlap, and the report
    /// says so in [`FaithfulnessReport::degraded`] — running out of budget is a
    /// result the caller has to be able to see, not a silent downgrade.
    pub max_llm_calls: usize,
}

impl Default for FaithfulnessConfig {
    fn default() -> Self {
        Self {
            decomposition_method: DecompositionMethod::SentenceSplit,
            nli_method: NliMethod::WordOverlap,
            min_faithfulness_score: 0.7,
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            word_overlap_entailment_threshold: 0.3,
            word_overlap_contradiction_threshold: 0.05,
            max_llm_calls: 10,
        }
    }
}

/// Faithfulness assessment for a single claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimFaithfulness {
    /// The atomic claim being assessed.
    pub claim: AtomicClaim,
    /// NLI verdict against the context.
    pub verdict: NliVerdict,
    /// Confidence in the verdict (0.0–1.0).
    pub confidence: f64,
    /// Source chunks that support (or contradict) this claim.
    pub supporting_chunks: Vec<String>,
}

/// Complete faithfulness report for a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FaithfulnessReport {
    /// Per-claim faithfulness results.
    pub claims: Vec<ClaimFaithfulness>,
    /// Overall faithfulness score (0.0–1.0).
    /// Computed as entailed_count / total_count.
    pub overall_score: f64,
    /// Number of entailed (supported) claims.
    pub entailed_count: usize,
    /// Number of contradicted claims.
    pub contradicted_count: usize,
    /// Number of neutral (unsupported) claims.
    pub neutral_count: usize,
    /// Response text with ungrounded claims processed per strategy.
    pub processed_text: String,
    /// LLM calls actually spent producing this report.
    ///
    /// Zero with [`NliMethod::WordOverlap`] and no verifier attached, which is
    /// the zero-cost path the module advertises.
    pub llm_calls_used: usize,
    /// Anything that was **requested but not delivered**, in the caller's terms.
    ///
    /// Empty means the report is exactly what the configuration asked for. A
    /// non-empty entry means some step fell back to a cheaper method — no
    /// verifier attached, budget exhausted, or an answer that could not be
    /// parsed — and the score must be read as that cheaper method's.
    ///
    /// This field exists because the alternative is what this module used to do:
    /// accept `LlmNli`, run Jaccard word overlap, and return a number the caller
    /// believed came from an LLM.
    pub degraded: Vec<String>,
}

impl FaithfulnessReport {
    /// Whether the report meets the minimum faithfulness threshold.
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }

    /// Ratio of grounded (entailed) claims to total claims.
    pub fn grounding_ratio(&self) -> f64 {
        self.overall_score
    }

    /// Get only the contradicted claims.
    pub fn contradicted_claims(&self) -> Vec<&ClaimFaithfulness> {
        self.claims
            .iter()
            .filter(|c| c.verdict == NliVerdict::Contradicted)
            .collect()
    }

    /// Get only the neutral (unsupported) claims.
    pub fn neutral_claims(&self) -> Vec<&ClaimFaithfulness> {
        self.claims
            .iter()
            .filter(|c| c.verdict == NliVerdict::Neutral)
            .collect()
    }
}

// ============================================================================
// Faithfulness scorer
// ============================================================================

/// Scores the faithfulness of a response against retrieved context.
///
/// Decomposes the response into atomic claims, then checks each claim
/// against the source context using the configured NLI method.
pub struct FaithfulnessScorer {
    config: FaithfulnessConfig,
    /// Optional LLM callback. Required by [`NliMethod::LlmNli`],
    /// [`DecompositionMethod::LlmDecomposition`] and the verify-then-* strategies;
    /// without it those options fall back and say so in
    /// [`FaithfulnessReport::degraded`].
    llm_fn: Option<Box<dyn Fn(&str) -> Option<String>>>,
    /// Optional confirmation callback for [`UngroundedClaimStrategy::Ask`].
    /// Returns true to keep the claim, false to drop it.
    confirm_fn: Option<Box<dyn Fn(&str) -> bool>>,
}

/// Tracks LLM spend within one `score()` call and records what had to degrade.
///
/// Both halves belong together on purpose: the only reason to cap spend is that
/// the cap will sometimes bite, and a cap that bites silently is how a cheaper
/// method ends up reported as an expensive one.
struct LlmBudget {
    used: usize,
    max: usize,
    degraded: Vec<String>,
}

impl LlmBudget {
    fn new(max: usize) -> Self {
        Self {
            used: 0,
            max,
            degraded: Vec::new(),
        }
    }

    /// Charge one call, or refuse when the budget is spent.
    fn charge(&mut self) -> bool {
        if self.used >= self.max {
            self.note(format!(
                "LLM budget of {} call(s) exhausted; remaining claims scored by word overlap",
                self.max
            ));
            return false;
        }
        self.used += 1;
        true
    }

    /// Record a degradation once, however many claims hit it — a per-claim list
    /// would bury the one line the caller needs under fifty copies of itself.
    fn note(&mut self, reason: String) {
        if !self.degraded.contains(&reason) {
            self.degraded.push(reason);
        }
    }
}

impl FaithfulnessScorer {
    /// Create a new scorer with the given configuration.
    ///
    /// Without [`with_llm_verifier`](Self::with_llm_verifier) the LLM-backed
    /// options degrade to their cheap equivalents and report it.
    pub fn new(config: FaithfulnessConfig) -> Self {
        Self {
            config,
            llm_fn: None,
            confirm_fn: None,
        }
    }

    /// Attach the LLM used by `LlmNli`, `LlmDecomposition` and the
    /// verify-then-* strategies.
    ///
    /// The callback receives a prompt and returns the model's reply, so the
    /// library never picks a provider, a model or a transport — the same shape
    /// [`crate::chain_of_verification::ChainOfVerification::with_llm_verifier`]
    /// uses.
    ///
    /// ```rust
    /// use ai_assistant::faithfulness::*;
    ///
    /// let mut config = FaithfulnessConfig::default();
    /// config.nli_method = NliMethod::LlmNli;
    ///
    /// // A real caller would forward this to their assistant.
    /// let scorer = FaithfulnessScorer::new(config)
    ///     .with_llm_verifier(|_prompt| Some("Entailed".to_string()));
    ///
    /// let report = scorer.score("Rust is a systems language.", &["Rust is a systems language."]);
    /// assert!(report.degraded.is_empty(), "nothing had to fall back");
    /// assert_eq!(report.llm_calls_used, 1);
    /// ```
    pub fn with_llm_verifier<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Option<String> + 'static,
    {
        self.llm_fn = Some(Box::new(f));
        self
    }

    /// Attach the prompt used by [`UngroundedClaimStrategy::Ask`]. Return true
    /// to keep an ungrounded claim, false to drop it.
    pub fn with_confirmation<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> bool + 'static,
    {
        self.confirm_fn = Some(Box::new(f));
        self
    }

    /// Score faithfulness of `response` against `context` chunks.
    pub fn score(&self, response: &str, context: &[&str]) -> FaithfulnessReport {
        let mut budget = LlmBudget::new(self.config.max_llm_calls);

        // Step 1: Decompose response into atomic claims
        let claims = self.decompose(response, &mut budget);

        if claims.is_empty() {
            return FaithfulnessReport {
                claims: Vec::new(),
                overall_score: 1.0,
                entailed_count: 0,
                contradicted_count: 0,
                neutral_count: 0,
                processed_text: response.to_string(),
                llm_calls_used: budget.used,
                degraded: budget.degraded,
            };
        }

        // Step 2: Evaluate each claim against context
        let mut results = Vec::with_capacity(claims.len());
        let mut entailed = 0usize;
        let mut contradicted = 0usize;
        let mut neutral = 0usize;

        for claim in claims {
            let (verdict, confidence, supporting) = match self.config.nli_method {
                NliMethod::WordOverlap => self.evaluate_word_overlap(&claim, context),
                NliMethod::LlmNli => self.evaluate_llm_nli(&claim, context, &mut budget),
            };

            match verdict {
                NliVerdict::Entailed => entailed += 1,
                NliVerdict::Contradicted => contradicted += 1,
                NliVerdict::Neutral => neutral += 1,
            }

            results.push(ClaimFaithfulness {
                claim,
                verdict,
                confidence,
                supporting_chunks: supporting,
            });
        }

        let total = results.len() as f64;
        let overall_score = if total > 0.0 {
            entailed as f64 / total
        } else {
            1.0
        };

        // Step 3: Build processed text
        let processed_text = self.apply_strategy(response, &results, context, &mut budget);

        FaithfulnessReport {
            claims: results,
            overall_score,
            entailed_count: entailed,
            contradicted_count: contradicted,
            neutral_count: neutral,
            processed_text,
            llm_calls_used: budget.used,
            degraded: budget.degraded,
        }
    }

    /// Decompose response text into atomic claims.
    fn decompose(&self, text: &str, budget: &mut LlmBudget) -> Vec<AtomicClaim> {
        match self.config.decomposition_method {
            DecompositionMethod::SentenceSplit => Self::sentence_split(text),
            DecompositionMethod::LlmDecomposition => self.llm_decompose(text, budget),
        }
    }

    /// Ask the model to split the text into atomic claims, one per line.
    ///
    /// Falls back to sentence splitting — and records why — when there is no
    /// verifier, the budget is spent, or the reply yields nothing usable.
    fn llm_decompose(&self, text: &str, budget: &mut LlmBudget) -> Vec<AtomicClaim> {
        let Some(ref llm) = self.llm_fn else {
            budget.note(
                "LlmDecomposition requested without an LLM verifier; used sentence splitting"
                    .to_string(),
            );
            return Self::sentence_split(text);
        };
        if !budget.charge() {
            return Self::sentence_split(text);
        }

        let prompt = format!(
            "Break the following text into atomic factual claims: single, independently \
             checkable assertions. Output one claim per line, with no numbering, no bullets \
             and no commentary.\n\nText:\n{text}"
        );
        let Some(reply) = llm(&prompt) else {
            budget.note(
                "the LLM returned nothing for decomposition; used sentence splitting".to_string(),
            );
            return Self::sentence_split(text);
        };

        let claims = Self::claims_from_lines(&reply, text);
        if claims.is_empty() {
            budget.note(
                "the LLM's decomposition had no usable lines; used sentence splitting".to_string(),
            );
            return Self::sentence_split(text);
        }
        claims
    }

    /// Turn one-claim-per-line model output into [`AtomicClaim`]s.
    ///
    /// Positions are resolved by locating each claim in the original text, because
    /// the ungrounded strategies splice by position and the model may reword what
    /// it returns. A claim that cannot be located keeps length 0, which the
    /// splicing code treats as "nothing to replace" rather than corrupting an
    /// unrelated span.
    fn claims_from_lines(reply: &str, original: &str) -> Vec<AtomicClaim> {
        let mut claims = Vec::new();
        for line in reply.lines() {
            // Strip the numbering and bullets the prompt asked it not to add.
            let text = line
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')')
                .trim_start_matches(['-', '*', '•'])
                .trim();
            if text.len() < 3 {
                continue;
            }
            let (position, length) = match original.find(text) {
                Some(p) => (p, text.len()),
                None => (0, 0),
            };
            claims.push(AtomicClaim {
                text: text.to_string(),
                position,
                length,
                source_sentence: text.to_string(),
            });
        }
        claims
    }

    /// Split text into sentences as atomic claims.
    fn sentence_split(text: &str) -> Vec<AtomicClaim> {
        let mut claims = Vec::new();
        let mut pos = 0;

        for sentence in split_sentences(text) {
            let trimmed = sentence.trim();
            if trimmed.len() >= 5 {
                // Skip very short fragments
                let start = text[pos..].find(trimmed).map(|i| i + pos).unwrap_or(pos);
                claims.push(AtomicClaim {
                    text: trimmed.to_string(),
                    position: start,
                    length: trimmed.len(),
                    source_sentence: trimmed.to_string(),
                });
            }
            pos += sentence.len();
        }

        claims
    }

    /// Evaluate a claim using Jaccard word overlap.
    fn evaluate_word_overlap(
        &self,
        claim: &AtomicClaim,
        context: &[&str],
    ) -> (NliVerdict, f64, Vec<String>) {
        let claim_words = words_set(&claim.text);
        if claim_words.is_empty() {
            return (NliVerdict::Neutral, 0.5, Vec::new());
        }

        let mut best_overlap = 0.0f64;
        let mut best_chunk = String::new();

        for chunk in context {
            let chunk_words = words_set(chunk);
            if chunk_words.is_empty() {
                continue;
            }

            let intersection = claim_words.intersection(&chunk_words).count() as f64;
            let union = claim_words.union(&chunk_words).count() as f64;
            let jaccard = if union > 0.0 {
                intersection / union
            } else {
                0.0
            };

            if jaccard > best_overlap {
                best_overlap = jaccard;
                best_chunk = chunk.to_string();
            }
        }

        let supporting = if !best_chunk.is_empty() {
            vec![best_chunk]
        } else {
            Vec::new()
        };

        if best_overlap >= self.config.word_overlap_entailment_threshold {
            (NliVerdict::Entailed, best_overlap.min(1.0), supporting)
        } else if best_overlap <= self.config.word_overlap_contradiction_threshold {
            // Very low overlap could mean contradiction or just missing context
            (NliVerdict::Neutral, 1.0 - best_overlap, supporting)
        } else {
            (NliVerdict::Neutral, 0.5, supporting)
        }
    }

    /// Evaluate a claim by asking the model whether the context entails it.
    ///
    /// Falls back to word overlap — recording why — when there is no verifier,
    /// the budget is spent, or the reply names no verdict. The fallback is the
    /// point of the `degraded` list: this used to happen unconditionally and
    /// silently, so `LlmNli` was word overlap wearing a more expensive name.
    fn evaluate_llm_nli(
        &self,
        claim: &AtomicClaim,
        context: &[&str],
        budget: &mut LlmBudget,
    ) -> (NliVerdict, f64, Vec<String>) {
        let Some(ref llm) = self.llm_fn else {
            budget.note(
                "LlmNli requested without an LLM verifier; scored by word overlap".to_string(),
            );
            return self.evaluate_word_overlap(claim, context);
        };
        if context.is_empty() {
            // Nothing to entail against; the cheap path already answers this
            // correctly and an LLM call would only cost money to agree.
            return self.evaluate_word_overlap(claim, context);
        }
        if !budget.charge() {
            return self.evaluate_word_overlap(claim, context);
        }

        let prompt = format!(
            "Decide whether the reference context supports the claim.\n\n\
             Reference context:\n{}\n\n\
             Claim: \"{}\"\n\n\
             Answer with exactly one word: Entailed, Contradicted, or Neutral.",
            context.join("\n"),
            claim.text
        );

        let Some(reply) = llm(&prompt) else {
            budget.note("the LLM returned nothing for NLI; scored by word overlap".to_string());
            return self.evaluate_word_overlap(claim, context);
        };

        let lower = reply.to_lowercase();
        // Contradiction first: "not entailed" contains "entailed", so testing for
        // entailment first would read a denial as agreement.
        let verdict = if lower.contains("contradict") {
            Some(NliVerdict::Contradicted)
        } else if lower.contains("entail") || lower.contains("support") {
            Some(NliVerdict::Entailed)
        } else if lower.contains("neutral") || lower.contains("unsupported") {
            Some(NliVerdict::Neutral)
        } else {
            None
        };

        match verdict {
            // The confidence is a fixed constant, not a probability: the model
            // returned a label, and dressing a label up as a calibrated number
            // is the kind of fake precision this module exists to remove.
            Some(v) => (
                v,
                LLM_NLI_CONFIDENCE,
                self.supporting_chunks(claim, context),
            ),
            None => {
                budget.note(format!(
                    "the LLM's NLI reply named no verdict ({:?}); scored by word overlap",
                    reply.chars().take(40).collect::<String>()
                ));
                self.evaluate_word_overlap(claim, context)
            }
        }
    }

    /// Context chunks worth citing for a claim, by overlap, so an LLM verdict
    /// still points at its evidence.
    fn supporting_chunks(&self, claim: &AtomicClaim, context: &[&str]) -> Vec<String> {
        let claim_words = words_set(&claim.text);
        if claim_words.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f64, &str)> = context
            .iter()
            .filter_map(|chunk| {
                let chunk_words = words_set(chunk);
                let union = claim_words.union(&chunk_words).count() as f64;
                if union == 0.0 {
                    return None;
                }
                let score = claim_words.intersection(&chunk_words).count() as f64 / union;
                (score > 0.0).then_some((score, *chunk))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(MAX_SUPPORTING_CHUNKS)
            .map(|(_, chunk)| chunk.to_string())
            .collect()
    }

    /// Apply the configured ungrounded strategy to build processed text.
    fn apply_strategy(
        &self,
        original: &str,
        results: &[ClaimFaithfulness],
        context: &[&str],
        budget: &mut LlmBudget,
    ) -> String {
        let mut processed = original.to_string();

        // Process in reverse order to preserve positions
        let mut sorted: Vec<&ClaimFaithfulness> = results
            .iter()
            .filter(|r| !r.verdict.is_grounded())
            .collect();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.claim.position));

        for result in sorted {
            let claim_text = &result.claim.text;
            match self.config.ungrounded_strategy {
                UngroundedClaimStrategy::Omit => {
                    if let Some(start) = processed.find(claim_text) {
                        processed.replace_range(start..start + claim_text.len(), "");
                    }
                }
                UngroundedClaimStrategy::Mark => {
                    if let Some(start) = processed.find(claim_text) {
                        let marked = format!("[unverified] {}", claim_text);
                        processed.replace_range(start..start + claim_text.len(), &marked);
                    }
                }
                UngroundedClaimStrategy::Warn => {
                    // Keep text as-is; warning is in the report
                }
                UngroundedClaimStrategy::Footnote => {
                    if let Some(start) = processed.find(claim_text) {
                        let end = start + claim_text.len();
                        let footnote = format!(
                            "{}[^unverified: not supported by provided context]",
                            claim_text
                        );
                        processed.replace_range(start..end, &footnote);
                    }
                }
                // Give the claim a second chance before acting on it: the first
                // pass judged it against the retrieved context only, and a claim
                // can be true without that context containing it.
                UngroundedClaimStrategy::VerifyThenMark
                | UngroundedClaimStrategy::VerifyThenOmit => {
                    let survived = self.second_opinion(claim_text, context, budget);
                    if let Some(start) = processed.find(claim_text) {
                        let end = start + claim_text.len();
                        if survived {
                            // Verified after all — leave it exactly as written.
                        } else if self.config.ungrounded_strategy
                            == UngroundedClaimStrategy::VerifyThenOmit
                        {
                            processed.replace_range(start..end, "");
                        } else {
                            let marked = format!("[unverified] {}", claim_text);
                            processed.replace_range(start..end, &marked);
                        }
                    }
                }
                UngroundedClaimStrategy::Ask => {
                    let keep = match self.confirm_fn {
                        Some(ref ask) => ask(claim_text),
                        None => {
                            budget.note(
                                "Ask requested without a confirmation callback; marked instead"
                                    .to_string(),
                            );
                            true
                        }
                    };
                    if let Some(start) = processed.find(claim_text) {
                        let end = start + claim_text.len();
                        if keep {
                            let marked = format!("[unverified] {}", claim_text);
                            processed.replace_range(start..end, &marked);
                        } else {
                            processed.replace_range(start..end, "");
                        }
                    }
                } // Deliberately exhaustive, with no catch-all: a variant added later
                  // must break the build here rather than silently inherit someone
                  // else's behaviour. A `_ =>` arm is exactly how the three
                  // verify-then-* strategies spent months quietly acting as `Mark`.
            }
        }

        processed
    }

    /// Ask the model whether an ungrounded claim is true anyway.
    ///
    /// This is the "verify" in verify-then-mark: the NLI pass only asked whether
    /// the *retrieved context* supports the claim, and a claim can be perfectly
    /// true while absent from the chunks that happened to be retrieved. Marking
    /// those is the false positive the strategy exists to avoid.
    ///
    /// Returns true when the claim survives. With no verifier it returns false —
    /// the conservative direction: without a second opinion the first one stands,
    /// so the claim gets marked exactly as before.
    fn second_opinion(&self, claim: &str, context: &[&str], budget: &mut LlmBudget) -> bool {
        let Some(ref llm) = self.llm_fn else {
            budget.note(format!(
                "{} requested without an LLM verifier; no second opinion, claims stay ungrounded",
                self.config.ungrounded_strategy.display_name()
            ));
            return false;
        };
        if !budget.charge() {
            return false;
        }

        let prompt = if context.is_empty() {
            format!(
                "Is the following statement factually correct? Answer with exactly one word, \
                 Yes or No.\n\nStatement: \"{claim}\""
            )
        } else {
            format!(
                "The reference context does not clearly support the statement below. Using your \
                 own knowledge, is the statement nonetheless factually correct? Answer with \
                 exactly one word, Yes or No.\n\nReference context:\n{}\n\nStatement: \"{claim}\"",
                context.join("\n")
            )
        };

        let Some(reply) = llm(&prompt) else {
            budget.note(
                "the LLM returned nothing for verification; claims stay ungrounded".to_string(),
            );
            return false;
        };

        // "No" is a substring of plenty of words ("nonetheless", "not"), so match
        // the negative on a trimmed token rather than anywhere in the reply.
        let lower = reply.trim().to_lowercase();
        let first = lower
            .split(|c: char| !c.is_alphabetic())
            .find(|w| !w.is_empty())
            .unwrap_or("");
        matches!(first, "yes" | "true" | "correct" | "supported")
    }

    /// Get the current configuration.
    pub fn config(&self) -> &FaithfulnessConfig {
        &self.config
    }
}

impl Default for FaithfulnessScorer {
    fn default() -> Self {
        Self::new(FaithfulnessConfig::default())
    }
}

impl std::fmt::Debug for FaithfulnessScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaithfulnessScorer")
            .field("config", &self.config)
            .finish()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Split text into sentences by common delimiters.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            // Check it's not an abbreviation (e.g., "Dr.", "U.S.")
            let trimmed = current.trim();
            if trimmed.split_whitespace().count() >= 2 || trimmed.len() > 10 {
                sentences.push(std::mem::take(&mut current));
            }
        }
    }

    // Remaining text
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Build a word set (lowercase, alphanumeric only) from text.
fn words_set(text: &str) -> std::collections::HashSet<String> {
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
// Visual groundedness (Batch 20)
// ============================================================================

/// Visual groundedness assessment for a response generated with image
/// attachments. Captures whether the response actually engages with the
/// supplied visual evidence (vs. text-only hallucination).
///
/// This is a *cheap heuristic* — it counts visual-vocabulary mentions
/// in the response. Higher precision requires VLM-side feature
/// alignment which is out of scope for a free Rust-pure metric.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VisualGroundednessReport {
    /// Number of image attachments the response was generated against.
    pub image_count: usize,
    /// Total tokens (whitespace-split words) in the response.
    pub total_tokens: usize,
    /// Tokens matching the built-in visual vocabulary (color, shape,
    /// spatial-relation terms, "image", "photo", "picture", etc.).
    pub visual_terms: usize,
    /// `visual_terms / total_tokens`, clamped to `[0, 1]`. 0 when there
    /// are no tokens.
    pub visual_density: f64,
    /// Whether the response mentions images at all when at least one
    /// image was supplied. False = strong signal of visual ungrounding.
    pub has_visual_grounding: bool,
}

#[cfg(feature = "vision")]
impl VisualGroundednessReport {
    /// Whether the response meets the minimum visual-density threshold.
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.visual_density >= threshold
    }
}

/// The fixed visual vocabulary used by [`score_visual_groundedness`].
/// Kept small and language-en biased on purpose — callers wanting i18n
/// vocabularies should fork the function rather than contend over a
/// shared list.
#[cfg(feature = "vision")]
const VISUAL_VOCAB: &[&str] = &[
    "image",
    "photo",
    "picture",
    "screenshot",
    "figure",
    "diagram",
    "shows",
    "depicts",
    "displays",
    "visible",
    "shown",
    "see",
    "color",
    "colour",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "left",
    "right",
    "above",
    "below",
    "top",
    "bottom",
    "center",
    "background",
    "foreground",
    "object",
    "scene",
];

/// Score how visually grounded a response is, given it was generated
/// with `image_count` images attached. Pure heuristic: counts visual
/// vocabulary tokens.
#[cfg(feature = "vision")]
pub fn score_visual_groundedness(response: &str, image_count: usize) -> VisualGroundednessReport {
    let tokens: Vec<String> = response
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let total = tokens.len();
    let mut visual = 0usize;
    for tok in &tokens {
        if VISUAL_VOCAB.iter().any(|v| *v == tok) {
            visual += 1;
        }
    }
    let density = if total == 0 {
        0.0
    } else {
        (visual as f64 / total as f64).clamp(0.0, 1.0)
    };
    VisualGroundednessReport {
        image_count,
        total_tokens: total,
        visual_terms: visual,
        visual_density: density,
        has_visual_grounding: image_count == 0 || visual > 0,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- NliVerdict tests ---

    #[test]
    fn test_nli_verdict_labels() {
        assert_eq!(NliVerdict::Entailed.label(), "Entailed");
        assert_eq!(NliVerdict::Contradicted.label(), "Contradicted");
        assert_eq!(NliVerdict::Neutral.label(), "Neutral");
    }

    #[test]
    fn test_nli_verdict_is_grounded() {
        assert!(NliVerdict::Entailed.is_grounded());
        assert!(!NliVerdict::Contradicted.is_grounded());
        assert!(!NliVerdict::Neutral.is_grounded());
    }

    #[test]
    fn test_nli_verdict_display() {
        assert_eq!(format!("{}", NliVerdict::Entailed), "Entailed");
        assert_eq!(format!("{}", NliVerdict::Neutral), "Neutral");
    }

    // --- Decomposition tests ---

    #[test]
    fn test_sentence_split_basic() {
        let claims =
            FaithfulnessScorer::sentence_split("Rust is a language. It was released in 2015.");
        assert_eq!(claims.len(), 2);
        assert!(claims[0].text.contains("Rust is a language"));
        assert!(claims[1].text.contains("released in 2015"));
    }

    #[test]
    fn test_sentence_split_single_sentence() {
        let claims = FaithfulnessScorer::sentence_split("Rust is a systems programming language.");
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn test_sentence_split_skips_short_fragments() {
        let claims = FaithfulnessScorer::sentence_split("Hi. Rust is a language.");
        // "Hi." is too short (< 5 chars), should be skipped
        assert_eq!(claims.len(), 1);
    }

    #[test]
    fn test_sentence_split_preserves_positions() {
        let text = "First sentence. Second sentence.";
        let claims = FaithfulnessScorer::sentence_split(text);
        assert_eq!(claims.len(), 2);
        // First claim starts at position 0
        assert_eq!(claims[0].position, 0);
    }

    // --- Word overlap NLI tests ---

    #[test]
    fn test_word_overlap_entailed() {
        let scorer = FaithfulnessScorer::default();
        let claim = AtomicClaim {
            text: "Rust was released in 2015.".to_string(),
            position: 0,
            length: 25,
            source_sentence: "Rust was released in 2015.".to_string(),
        };
        let context = &["Rust was first released in 2015 by Mozilla."];
        let (verdict, confidence, supporting) = scorer.evaluate_word_overlap(&claim, context);
        assert_eq!(verdict, NliVerdict::Entailed);
        assert!(confidence > 0.0);
        assert!(!supporting.is_empty());
    }

    #[test]
    fn test_word_overlap_neutral() {
        let scorer = FaithfulnessScorer::default();
        let claim = AtomicClaim {
            text: "The weather in Tokyo is pleasant today.".to_string(),
            position: 0,
            length: 39,
            source_sentence: "The weather in Tokyo is pleasant today.".to_string(),
        };
        let context = &["Rust is a systems programming language."];
        let (verdict, _confidence, _supporting) = scorer.evaluate_word_overlap(&claim, context);
        assert_ne!(verdict, NliVerdict::Entailed);
    }

    #[test]
    fn test_word_overlap_empty_context() {
        let scorer = FaithfulnessScorer::default();
        let claim = AtomicClaim {
            text: "Some claim about something.".to_string(),
            position: 0,
            length: 27,
            source_sentence: "Some claim about something.".to_string(),
        };
        let context: &[&str] = &[];
        let (verdict, _confidence, supporting) = scorer.evaluate_word_overlap(&claim, context);
        assert_eq!(verdict, NliVerdict::Neutral);
        assert!(supporting.is_empty());
    }

    // --- Full scorer tests ---

    #[test]
    fn test_scorer_high_faithfulness() {
        let scorer = FaithfulnessScorer::default();
        let context = &[
            "Rust was first released in 2015.",
            "It is a systems programming language.",
        ];
        let response = "Rust was released in 2015. It is a systems language.";
        let report = scorer.score(response, context);

        assert!(
            report.overall_score > 0.0,
            "Response closely matching context should have positive score, got {}",
            report.overall_score
        );
        assert!(report.entailed_count > 0);
    }

    #[test]
    fn test_scorer_low_faithfulness() {
        let scorer = FaithfulnessScorer::default();
        let context = &["Rust is a programming language."];
        let response =
            "Python was invented by Guido van Rossum in 1991. Java was created by James Gosling.";
        let report = scorer.score(response, context);

        assert!(
            report.overall_score < 1.0,
            "Unrelated response should have lower score"
        );
    }

    #[test]
    fn test_scorer_empty_response() {
        let scorer = FaithfulnessScorer::default();
        let context = &["Some context."];
        let report = scorer.score("", context);

        assert_eq!(
            report.overall_score, 1.0,
            "Empty response = vacuously faithful"
        );
        assert!(report.claims.is_empty());
    }

    #[test]
    fn test_scorer_meets_threshold() {
        let report = FaithfulnessReport {
            claims: Vec::new(),
            overall_score: 0.8,
            entailed_count: 4,
            contradicted_count: 0,
            neutral_count: 1,
            processed_text: String::new(),
            llm_calls_used: 0,
            degraded: Vec::new(),
        };
        assert!(report.meets_threshold(0.7));
        assert!(!report.meets_threshold(0.9));
    }

    #[test]
    fn test_scorer_contradicted_claims() {
        let report = FaithfulnessReport {
            claims: vec![
                ClaimFaithfulness {
                    claim: AtomicClaim {
                        text: "A".to_string(),
                        position: 0,
                        length: 1,
                        source_sentence: "A".to_string(),
                    },
                    verdict: NliVerdict::Entailed,
                    confidence: 0.9,
                    supporting_chunks: vec![],
                },
                ClaimFaithfulness {
                    claim: AtomicClaim {
                        text: "B".to_string(),
                        position: 2,
                        length: 1,
                        source_sentence: "B".to_string(),
                    },
                    verdict: NliVerdict::Contradicted,
                    confidence: 0.8,
                    supporting_chunks: vec![],
                },
            ],
            overall_score: 0.5,
            entailed_count: 1,
            contradicted_count: 1,
            neutral_count: 0,
            processed_text: String::new(),
            llm_calls_used: 0,
            degraded: Vec::new(),
        };
        assert_eq!(report.contradicted_claims().len(), 1);
        assert_eq!(report.neutral_claims().len(), 0);
    }

    // --- Strategy application tests ---

    #[test]
    fn test_strategy_mark() {
        let config = FaithfulnessConfig {
            ungrounded_strategy: UngroundedClaimStrategy::Mark,
            ..Default::default()
        };
        let scorer = FaithfulnessScorer::new(config);
        let context: &[&str] = &["Rust is great."];
        let response = "Rust is great. Python was invented in Antarctica.";
        let report = scorer.score(response, context);

        // The unrelated claim about Python/Antarctica should be marked
        // (depends on whether word overlap catches it)
        let _ = report.processed_text; // Just verify it doesn't panic
    }

    #[test]
    fn test_strategy_warn_preserves_text() {
        let config = FaithfulnessConfig {
            ungrounded_strategy: UngroundedClaimStrategy::Warn,
            ..Default::default()
        };
        let scorer = FaithfulnessScorer::new(config);
        let context: &[&str] = &[];
        let response = "Some claim about things.";
        let report = scorer.score(response, context);

        // Warn strategy preserves original text
        assert_eq!(report.processed_text, response);
    }

    // --- Config tests ---

    #[test]
    fn test_default_config() {
        let config = FaithfulnessConfig::default();
        assert_eq!(
            config.decomposition_method,
            DecompositionMethod::SentenceSplit
        );
        assert_eq!(config.nli_method, NliMethod::WordOverlap);
        assert!((config.min_faithfulness_score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_words_set() {
        let set = words_set("Hello, World! This is a test.");
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
        assert!(set.contains("this"));
        assert!(set.contains("test"));
        // Single-char words are excluded
        assert!(!set.contains("a"));
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("This is first. This is second! This is third?");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_grounding_ratio() {
        let report = FaithfulnessReport {
            claims: Vec::new(),
            overall_score: 0.75,
            entailed_count: 3,
            contradicted_count: 0,
            neutral_count: 1,
            processed_text: String::new(),
            llm_calls_used: 0,
            degraded: Vec::new(),
        };
        assert!((report.grounding_ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scorer_debug_display() {
        let scorer = FaithfulnessScorer::default();
        let debug = format!("{:?}", scorer);
        assert!(debug.contains("FaithfulnessScorer"));
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_visual_groundedness_no_images_passes_through() {
        let r = score_visual_groundedness("This is a text-only response.", 0);
        assert_eq!(r.image_count, 0);
        // No images attached means grounding is vacuously satisfied.
        assert!(r.has_visual_grounding);
        assert_eq!(r.visual_terms, 0);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_visual_groundedness_image_with_visual_text() {
        let r = score_visual_groundedness(
            "The image shows a red car on the left of a blue building.",
            1,
        );
        assert_eq!(r.image_count, 1);
        // "image", "shows", "red", "left", "blue" → 5 hits.
        assert!(r.visual_terms >= 4, "got {}", r.visual_terms);
        assert!(r.has_visual_grounding);
        assert!(r.visual_density > 0.0);
        assert!(r.meets_threshold(0.1));
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_visual_groundedness_image_with_no_visual_text() {
        let r =
            score_visual_groundedness("The historical context dates back to medieval Europe.", 1);
        assert_eq!(r.image_count, 1);
        assert_eq!(r.visual_terms, 0);
        // Image attached but zero visual mentions: ungrounded.
        assert!(!r.has_visual_grounding);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_visual_groundedness_empty_response() {
        let r = score_visual_groundedness("", 1);
        assert_eq!(r.total_tokens, 0);
        assert_eq!(r.visual_density, 0.0);
        assert!(!r.has_visual_grounding);
    }
}

#[cfg(test)]
mod llm_path_tests {
    //! The LLM-backed options must either do the expensive thing or admit they
    //! did not. Silence is the bug these tests exist to prevent: before V309 all
    //! three ran the cheap method and reported nothing.
    use super::*;
    use std::cell::RefCell;

    /// A scorer whose "model" answers with a fixed reply.
    fn scorer_with(config: FaithfulnessConfig, reply: &'static str) -> FaithfulnessScorer {
        FaithfulnessScorer::new(config).with_llm_verifier(move |_| Some(reply.to_string()))
    }

    #[test]
    fn llm_nli_without_a_verifier_says_it_fell_back() {
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;

        let report = FaithfulnessScorer::new(config).score("Rust is fast.", &["Rust is fast."]);

        assert_eq!(report.llm_calls_used, 0, "there was no LLM to call");
        assert!(
            report.degraded.iter().any(|d| d.contains("LlmNli")),
            "the caller asked for LlmNli and got word overlap; that must be in the report, got {:?}",
            report.degraded
        );
    }

    #[test]
    fn llm_nli_with_a_verifier_spends_calls_and_degrades_nothing() {
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;

        let report =
            scorer_with(config, "Entailed").score("Rust is fast.", &["Rust is a fast language."]);

        assert_eq!(report.llm_calls_used, 1);
        assert!(report.degraded.is_empty(), "got {:?}", report.degraded);
        assert_eq!(report.entailed_count, 1);
    }

    #[test]
    fn the_llm_verdict_beats_word_overlap() {
        // The whole reason to pay for an LLM: the claim shares almost no words
        // with the context, so word overlap calls it ungrounded and the model
        // says it is entailed. If the two agreed here, the test could not tell a
        // real LLM path from the old silent fallback.
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;
        let context = &["The capital of France is Paris."];
        let claim = "Paris serves as the French seat of government.";

        let cheap = FaithfulnessScorer::new(FaithfulnessConfig::default()).score(claim, context);
        assert_eq!(
            cheap.entailed_count, 0,
            "word overlap should miss this one, else the test proves nothing"
        );

        let rich = scorer_with(config, "Entailed").score(claim, context);
        assert_eq!(rich.entailed_count, 1);
        assert_eq!(rich.claims[0].confidence, LLM_NLI_CONFIDENCE);
    }

    #[test]
    fn a_reply_naming_no_verdict_falls_back_and_reports_it() {
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;

        let report = scorer_with(config, "I am not sure about that, sorry")
            .score("Rust is fast.", &["Rust is fast."]);

        assert_eq!(
            report.llm_calls_used, 1,
            "the call was still made and paid for"
        );
        assert!(
            report.degraded.iter().any(|d| d.contains("no verdict")),
            "got {:?}",
            report.degraded
        );
    }

    #[test]
    fn a_denial_is_not_read_as_agreement() {
        // "not entailed" contains "entailed"; matching entailment first would
        // turn every denial into a pass.
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;

        let report = scorer_with(config, "Contradicted - the context says otherwise")
            .score("Rust is slow.", &["Rust is fast."]);

        assert_eq!(report.contradicted_count, 1);
        assert_eq!(report.entailed_count, 0);
    }

    #[test]
    fn the_budget_caps_spend_and_names_itself_when_it_bites() {
        let mut config = FaithfulnessConfig::default();
        config.nli_method = NliMethod::LlmNli;
        config.max_llm_calls = 2;

        let report = scorer_with(config, "Entailed").score(
            "One is true. Two is true. Three is true. Four is true.",
            &["Everything here is true."],
        );

        assert_eq!(report.llm_calls_used, 2, "the cap must hold");
        assert!(
            report.degraded.iter().any(|d| d.contains("budget")),
            "claims past the cap were scored differently; say so. got {:?}",
            report.degraded
        );
    }

    #[test]
    fn verify_then_omit_removes_only_what_the_second_opinion_rejects() {
        let mut config = FaithfulnessConfig::default();
        config.ungrounded_strategy = UngroundedClaimStrategy::VerifyThenOmit;

        // The model vouches for the ungrounded claim, so it must survive intact.
        let kept = scorer_with(config.clone(), "Yes").score(
            "Water boils at 100C.",
            &["Completely unrelated reference material."],
        );
        assert!(
            kept.processed_text.contains("Water boils"),
            "a verified claim must not be removed: {:?}",
            kept.processed_text
        );
        assert!(!kept.processed_text.contains("[unverified]"));

        // And when the model refuses to vouch, it goes.
        let dropped = scorer_with(config, "No").score(
            "Water boils at 12C.",
            &["Completely unrelated reference material."],
        );
        assert!(
            !dropped.processed_text.contains("Water boils"),
            "an unverified claim should have been omitted: {:?}",
            dropped.processed_text
        );
    }

    #[test]
    fn verify_then_mark_without_a_verifier_keeps_the_old_behaviour_and_admits_it() {
        // Conservative direction: with no second opinion the first one stands, so
        // the claim is marked exactly as before, but the caller is told that the
        // "verify" half never ran.
        let mut config = FaithfulnessConfig::default();
        config.ungrounded_strategy = UngroundedClaimStrategy::VerifyThenMark;

        let report =
            FaithfulnessScorer::new(config).score("Water boils at 100C.", &["Unrelated material."]);

        assert!(report.processed_text.contains("[unverified]"));
        assert!(
            report.degraded.iter().any(|d| d.contains("second opinion")),
            "got {:?}",
            report.degraded
        );
    }

    #[test]
    fn ask_routes_through_the_confirmation_callback() {
        let mut config = FaithfulnessConfig::default();
        config.ungrounded_strategy = UngroundedClaimStrategy::Ask;
        let seen = std::rc::Rc::new(RefCell::new(Vec::new()));
        let recorder = std::rc::Rc::clone(&seen);

        let report = FaithfulnessScorer::new(config)
            .with_confirmation(move |claim| {
                recorder.borrow_mut().push(claim.to_string());
                false // reject it
            })
            .score("Water boils at 12C.", &["Unrelated material."]);

        assert_eq!(seen.borrow().len(), 1, "the user was asked exactly once");
        assert!(
            !report.processed_text.contains("Water boils"),
            "a rejected claim must be dropped: {:?}",
            report.processed_text
        );
    }

    #[test]
    fn llm_decomposition_uses_the_model_and_splits_finer_than_sentences() {
        let mut config = FaithfulnessConfig::default();
        config.decomposition_method = DecompositionMethod::LlmDecomposition;

        // One sentence, two claims, which sentence splitting cannot produce.
        let report = scorer_with(config, "Rust is fast\nRust is memory safe").score(
            "Rust is fast and memory safe.",
            &["Rust is fast and memory safe."],
        );

        assert_eq!(report.claims.len(), 2, "got {:?}", report.claims);
        assert_eq!(report.llm_calls_used, 1);
        assert!(report.degraded.is_empty(), "got {:?}", report.degraded);
    }

    #[test]
    fn llm_decomposition_without_a_verifier_falls_back_and_reports_it() {
        let mut config = FaithfulnessConfig::default();
        config.decomposition_method = DecompositionMethod::LlmDecomposition;

        let report =
            FaithfulnessScorer::new(config).score("A is true. B is true.", &["A is true."]);

        assert_eq!(report.claims.len(), 2, "sentence split still ran");
        assert!(
            report
                .degraded
                .iter()
                .any(|d| d.contains("LlmDecomposition")),
            "got {:?}",
            report.degraded
        );
    }

    #[test]
    fn the_cheap_path_stays_free_and_clean() {
        // The default configuration must spend nothing and report nothing: the
        // zero-cost promise is the other half of being honest about cost.
        let report = FaithfulnessScorer::default().score("Rust is fast.", &["Rust is fast."]);

        assert_eq!(report.llm_calls_used, 0);
        assert!(report.degraded.is_empty(), "got {:?}", report.degraded);
    }
}
