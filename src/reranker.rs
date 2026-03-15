//! Neural cross-encoder reranking for RAG pipelines.
//!
//! Provides multiple reranking strategies: cross-encoder, reciprocal rank fusion,
//! maximal marginal relevance (diversity), and cascade pipelines.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for reranking operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerankerConfig {
    /// Maximum number of documents to return.
    pub top_k: usize,
    /// Diversity weight for MMR (0.0 = max diversity, 1.0 = max relevance).
    pub diversity_lambda: f64,
    /// Minimum score threshold; documents below this are filtered out.
    pub min_score: f64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            diversity_lambda: 0.5,
            min_score: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ScoredDocument
// ---------------------------------------------------------------------------

/// A document with an associated relevance score and metadata.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    /// The document text.
    pub content: String,
    /// Current relevance score (may be updated by reranking stages).
    pub score: f64,
    /// The rank position from the original retrieval.
    pub original_rank: usize,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ScoredDocument {
    /// Create a new scored document.
    pub fn new(content: &str, score: f64, original_rank: usize) -> Self {
        Self {
            content: content.to_string(),
            score,
            original_rank,
            metadata: HashMap::new(),
        }
    }

    /// Attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredDocument {}

impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordering is *descending* by score so that sorting yields highest-score first.
impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Reranker trait
// ---------------------------------------------------------------------------

/// Common interface for all reranking strategies.
pub trait Reranker: Send + Sync {
    /// Rerank the given documents with respect to `query` and return the
    /// reranked (and possibly filtered) list.
    fn rerank(
        &self,
        query: &str,
        docs: &[ScoredDocument],
        config: &RerankerConfig,
    ) -> Vec<ScoredDocument>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a string to a set of normalised word tokens (lowercase, trimmed of
/// non-alphanumeric edges).  Mirrors the pattern in `smart_suggestions.rs`.
fn word_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity between two word sets.
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let union_size = a.union(b).count();
    if union_size == 0 {
        return 0.0;
    }
    let intersection_size = a.intersection(b).count();
    intersection_size as f64 / union_size as f64
}

/// Apply `min_score` filter and `top_k` truncation to a scored list.
fn apply_filters(mut docs: Vec<ScoredDocument>, config: &RerankerConfig) -> Vec<ScoredDocument> {
    docs.retain(|d| d.score >= config.min_score);
    docs.truncate(config.top_k);
    docs
}

// ---------------------------------------------------------------------------
// CrossEncoderReranker
// ---------------------------------------------------------------------------

/// Reranks documents using a pluggable scoring function that evaluates
/// (query, document) pairs.  The default scorer computes term-overlap
/// (shared words / total unique words), similar to TF-IDF bag-of-words.
pub struct CrossEncoderReranker {
    scorer: Arc<dyn Fn(&str, &str) -> f64 + Send + Sync>,
}

impl fmt::Debug for CrossEncoderReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CrossEncoderReranker")
            .field("scorer", &"...")
            .finish()
    }
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker with a custom scoring function.
    pub fn new(scorer: Arc<dyn Fn(&str, &str) -> f64 + Send + Sync>) -> Self {
        Self { scorer }
    }

    /// Create a cross-encoder reranker with the default term-overlap scorer.
    pub fn default_scorer() -> Self {
        Self {
            scorer: Arc::new(|query, doc| {
                let q = word_set(query);
                let d = word_set(doc);
                jaccard_similarity(&q, &d)
            }),
        }
    }
}

impl Reranker for CrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        docs: &[ScoredDocument],
        config: &RerankerConfig,
    ) -> Vec<ScoredDocument> {
        let mut scored: Vec<ScoredDocument> = docs
            .iter()
            .map(|d| {
                let new_score = (self.scorer)(query, &d.content);
                ScoredDocument {
                    content: d.content.clone(),
                    score: new_score,
                    original_rank: d.original_rank,
                    metadata: d.metadata.clone(),
                }
            })
            .collect();

        scored.sort();
        apply_filters(scored, config)
    }
}

// ---------------------------------------------------------------------------
// ReciprocalRankFusion
// ---------------------------------------------------------------------------

/// Merges multiple ranked lists using the Reciprocal Rank Fusion algorithm.
///
/// RRF score for document *d*: `sum over lists of 1 / (k + rank(d))`.
/// Documents are matched across lists by their `content` string.
#[derive(Debug)]
pub struct ReciprocalRankFusion {
    /// The *k* constant that dampens the contribution of low ranks.
    k: usize,
}

impl ReciprocalRankFusion {
    /// Create a new RRF merger. The default `k` is 60.
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Fuse multiple ranked lists into a single merged list sorted by RRF score.
    pub fn fuse(&self, ranked_lists: &[Vec<ScoredDocument>]) -> Vec<ScoredDocument> {
        // Map from content -> (accumulated_rrf_score, first-seen ScoredDocument)
        let mut scores: HashMap<String, (f64, ScoredDocument)> = HashMap::new();

        for list in ranked_lists {
            for (rank_idx, doc) in list.iter().enumerate() {
                let rrf_contribution = 1.0 / (self.k as f64 + (rank_idx + 1) as f64);
                let entry = scores.entry(doc.content.clone()).or_insert_with(|| {
                    (
                        0.0,
                        ScoredDocument {
                            content: doc.content.clone(),
                            score: 0.0,
                            original_rank: doc.original_rank,
                            metadata: doc.metadata.clone(),
                        },
                    )
                });
                entry.0 += rrf_contribution;
            }
        }

        let mut result: Vec<ScoredDocument> = scores
            .into_values()
            .map(|(rrf_score, mut doc)| {
                doc.score = rrf_score;
                doc
            })
            .collect();

        result.sort();
        result
    }
}

impl Default for ReciprocalRankFusion {
    fn default() -> Self {
        Self::new(60)
    }
}

// ---------------------------------------------------------------------------
// DiversityReranker (Maximal Marginal Relevance)
// ---------------------------------------------------------------------------

/// Reranks documents using Maximal Marginal Relevance (MMR) to balance
/// relevance to the query with diversity among selected documents.
///
/// MMR formula:
/// `argmax[ lambda * sim(query, d) - (1 - lambda) * max_over_selected( sim(d, d_sel) ) ]`
///
/// Similarity is Jaccard over word sets.
#[derive(Debug)]
pub struct DiversityReranker {
    /// Balance between relevance (1.0) and diversity (0.0).
    lambda: f64,
}

impl DiversityReranker {
    /// Create a new MMR diversity reranker.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }
}

impl Reranker for DiversityReranker {
    fn rerank(
        &self,
        query: &str,
        docs: &[ScoredDocument],
        config: &RerankerConfig,
    ) -> Vec<ScoredDocument> {
        if docs.is_empty() {
            return Vec::new();
        }

        let query_words = word_set(query);
        let doc_word_sets: Vec<HashSet<String>> =
            docs.iter().map(|d| word_set(&d.content)).collect();

        // Pre-compute relevance to query for each doc
        let relevance: Vec<f64> = doc_word_sets
            .iter()
            .map(|ws| jaccard_similarity(&query_words, ws))
            .collect();

        let mut selected: Vec<usize> = Vec::new();
        let mut remaining: Vec<usize> = (0..docs.len()).collect();
        let mut result: Vec<ScoredDocument> = Vec::new();

        let target = config.top_k.min(docs.len());

        while result.len() < target && !remaining.is_empty() {
            let mut best_idx_in_remaining = 0;
            let mut best_mmr = f64::NEG_INFINITY;

            for (ri, &doc_idx) in remaining.iter().enumerate() {
                let rel = relevance[doc_idx];

                // Maximum similarity to any already-selected document
                let max_sim_selected = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .map(|&si| jaccard_similarity(&doc_word_sets[doc_idx], &doc_word_sets[si]))
                        .fold(0.0_f64, f64::max)
                };

                let mmr = self.lambda * rel - (1.0 - self.lambda) * max_sim_selected;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx_in_remaining = ri;
                }
            }

            let chosen = remaining.remove(best_idx_in_remaining);
            selected.push(chosen);

            let mut doc = docs[chosen].clone();
            doc.score = best_mmr;
            result.push(doc);
        }

        result.retain(|d| d.score >= config.min_score);
        result
    }
}

// ---------------------------------------------------------------------------
// CascadeReranker
// ---------------------------------------------------------------------------

/// Two-stage reranker: a cheap first pass narrows the candidate set, then an
/// expensive second pass reranks the survivors.
pub struct CascadeReranker {
    first_pass_top_k: usize,
    first_pass: Option<Box<dyn Reranker>>,
    second_pass: Option<Box<dyn Reranker>>,
}

impl fmt::Debug for CascadeReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CascadeReranker")
            .field("first_pass_top_k", &self.first_pass_top_k)
            .field("has_first_pass", &self.first_pass.is_some())
            .field("has_second_pass", &self.second_pass.is_some())
            .finish()
    }
}

impl CascadeReranker {
    /// Create a cascade reranker that retains `first_pass_top_k` documents
    /// after the first stage.
    pub fn new(first_pass_top_k: usize) -> Self {
        Self {
            first_pass_top_k,
            first_pass: None,
            second_pass: None,
        }
    }

    /// Set the first-pass (cheap) reranker.
    pub fn with_first_pass(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.first_pass = Some(reranker);
        self
    }

    /// Set the second-pass (expensive) reranker.
    pub fn with_second_pass(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.second_pass = Some(reranker);
        self
    }

    /// Run the cascade: first pass with `first_pass_top_k`, then second pass
    /// with the caller's `config`.
    pub fn rerank(
        &self,
        query: &str,
        docs: &[ScoredDocument],
        config: &RerankerConfig,
    ) -> Vec<ScoredDocument> {
        // Stage 1 — cheap filter
        let first_config = RerankerConfig {
            top_k: self.first_pass_top_k,
            min_score: 0.0,
            ..*config
        };

        let stage1 = match &self.first_pass {
            Some(r) => r.rerank(query, docs, &first_config),
            None => {
                let mut v = docs.to_vec();
                v.sort();
                v.truncate(self.first_pass_top_k);
                v
            }
        };

        // Stage 2 — expensive rerank
        match &self.second_pass {
            Some(r) => r.rerank(query, &stage1, config),
            None => {
                let mut v = stage1;
                v.sort();
                apply_filters(v, config)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RerankerPipeline
// ---------------------------------------------------------------------------

/// Chains multiple `Reranker` stages sequentially, passing the output of each
/// stage as input to the next.
pub struct RerankerPipeline {
    stages: Vec<Box<dyn Reranker>>,
}

impl fmt::Debug for RerankerPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankerPipeline")
            .field("stages_count", &self.stages.len())
            .finish()
    }
}

impl RerankerPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a reranking stage (builder pattern).
    pub fn add_stage(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.stages.push(reranker);
        self
    }

    /// Run all stages sequentially.
    pub fn run(
        &self,
        query: &str,
        docs: Vec<ScoredDocument>,
        config: &RerankerConfig,
    ) -> Vec<ScoredDocument> {
        let mut current = docs;
        for stage in &self.stages {
            current = stage.rerank(query, &current, config);
        }
        current
    }
}

impl Default for RerankerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_doc(content: &str, score: f64, rank: usize) -> ScoredDocument {
        ScoredDocument::new(content, score, rank)
    }

    fn default_config() -> RerankerConfig {
        RerankerConfig::default()
    }

    // -----------------------------------------------------------------------
    // RRF tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rrf_basic() {
        let rrf = ReciprocalRankFusion::new(60);

        // List 1: A at rank 1, B at rank 2
        let list1 = vec![make_doc("A", 0.9, 0), make_doc("B", 0.8, 1)];
        // List 2: B at rank 1, A at rank 2
        let list2 = vec![make_doc("B", 0.95, 0), make_doc("A", 0.7, 1)];

        let fused = rrf.fuse(&[list1, list2]);

        assert_eq!(fused.len(), 2);
        // Both A and B appear in both lists.
        // A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        // B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        // Scores are equal, so order may vary — just verify scores match.
        let score_a = fused.iter().find(|d| d.content == "A").unwrap().score;
        let score_b = fused.iter().find(|d| d.content == "B").unwrap().score;
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((score_a - expected).abs() < 1e-10);
        assert!((score_b - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_disjoint() {
        let rrf = ReciprocalRankFusion::new(60);

        let list1 = vec![make_doc("A", 1.0, 0)];
        let list2 = vec![make_doc("B", 1.0, 0)];

        let fused = rrf.fuse(&[list1, list2]);

        assert_eq!(fused.len(), 2);
        // Both should have the same RRF score (1/(60+1))
        let score_a = fused.iter().find(|d| d.content == "A").unwrap().score;
        let score_b = fused.iter().find(|d| d.content == "B").unwrap().score;
        assert!((score_a - 1.0 / 61.0).abs() < 1e-10);
        assert!((score_b - 1.0 / 61.0).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_single_list() {
        let rrf = ReciprocalRankFusion::new(60);

        let list = vec![
            make_doc("A", 1.0, 0),
            make_doc("B", 0.8, 1),
            make_doc("C", 0.6, 2),
        ];

        let fused = rrf.fuse(&[list]);

        assert_eq!(fused.len(), 3);
        // Order should be preserved: A > B > C by RRF score
        assert_eq!(fused[0].content, "A");
        assert_eq!(fused[1].content, "B");
        assert_eq!(fused[2].content, "C");
    }

    // -----------------------------------------------------------------------
    // MMR / Diversity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mmr_diversity() {
        let mmr = DiversityReranker::new(0.3); // favour diversity

        // Three very similar docs about cats, one different doc about cars
        let docs = vec![
            make_doc("the cat sat on the mat", 0.9, 0),
            make_doc("the cat lay on the mat", 0.85, 1),
            make_doc("the cat slept on the mat", 0.8, 2),
            make_doc("fast cars race on the track", 0.5, 3),
        ];

        let config = RerankerConfig {
            top_k: 4,
            min_score: f64::NEG_INFINITY,
            ..default_config()
        };

        let result = mmr.rerank("cat mat", &docs, &config);

        assert_eq!(result.len(), 4);
        // The "cars" doc should be promoted relative to its original rank (3rd
        // or better) because it is diverse compared to the cat docs.
        let car_pos = result
            .iter()
            .position(|d| d.content.contains("cars"))
            .unwrap();
        assert!(
            car_pos < 3,
            "Diverse 'cars' doc should be promoted above position 3, found at {}",
            car_pos
        );
    }

    #[test]
    fn test_mmr_pure_relevance() {
        let mmr = DiversityReranker::new(1.0); // pure relevance

        let docs = vec![
            make_doc("rust programming language", 0.9, 0),
            make_doc("python programming language", 0.7, 1),
            make_doc("cooking recipes for dinner", 0.3, 2),
        ];

        let config = RerankerConfig {
            top_k: 3,
            min_score: f64::NEG_INFINITY,
            ..default_config()
        };

        let result = mmr.rerank("programming language", &docs, &config);

        // With lambda=1.0 the diversity penalty is zero, so the order should
        // follow relevance to the query.  "rust programming language" has the
        // highest Jaccard similarity to the query.
        assert_eq!(result[0].content, "rust programming language");
    }

    // -----------------------------------------------------------------------
    // CrossEncoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_encoder_scoring() {
        // Custom scorer: score = number of query words found in doc / query word count
        let scorer = Arc::new(|query: &str, doc: &str| {
            let qw: HashSet<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();
            let dw: HashSet<String> = doc.split_whitespace().map(|w| w.to_lowercase()).collect();
            if qw.is_empty() {
                return 0.0;
            }
            qw.intersection(&dw).count() as f64 / qw.len() as f64
        });

        let ce = CrossEncoderReranker::new(scorer);

        let docs = vec![
            make_doc("rust is great", 0.5, 0),
            make_doc("python is great", 0.6, 1),
            make_doc("rust python both", 0.4, 2),
        ];

        let config = RerankerConfig {
            top_k: 10,
            min_score: 0.0,
            ..default_config()
        };

        let result = ce.rerank("rust python", &docs, &config);

        // "rust python both" contains both query terms -> score 1.0
        assert_eq!(result[0].content, "rust python both");
        assert!((result[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_encoder_top_k() {
        let ce = CrossEncoderReranker::default_scorer();

        let docs: Vec<ScoredDocument> = (0..10)
            .map(|i| make_doc(&format!("document number {}", i), 0.5, i))
            .collect();

        let config = RerankerConfig {
            top_k: 3,
            ..default_config()
        };

        let result = ce.rerank("document", &docs, &config);

        assert_eq!(result.len(), 3);
    }

    // -----------------------------------------------------------------------
    // CascadeReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cascade_two_stages() {
        // First pass: keep top 3 by default ordering
        // Second pass: custom scorer that boosts docs containing "important"
        let scorer = Arc::new(|_query: &str, doc: &str| {
            if doc.contains("important") {
                1.0
            } else {
                0.1
            }
        });

        let cascade = CascadeReranker::new(3)
            .with_second_pass(Box::new(CrossEncoderReranker::new(scorer)));

        let docs = vec![
            make_doc("first result", 0.9, 0),
            make_doc("important second result", 0.85, 1),
            make_doc("third result", 0.8, 2),
            make_doc("important fourth result but low rank", 0.3, 3),
            make_doc("fifth result", 0.2, 4),
        ];

        let config = default_config();
        let result = cascade.rerank("query", &docs, &config);

        // First pass keeps top 3 by input score order (0.9, 0.85, 0.8)
        // (the 4th "important" doc with score 0.3 is cut)
        // Second pass: "important second result" scores 1.0, others 0.1
        assert!(result.len() <= 3);
        assert_eq!(result[0].content, "important second result");
    }

    // -----------------------------------------------------------------------
    // Pipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_chaining() {
        // Stage 1: cross-encoder default scorer
        // Stage 2: diversity reranker
        // Stage 3: another cross-encoder with a trivial scorer (pass-through scores)
        let pipeline = RerankerPipeline::new()
            .add_stage(Box::new(CrossEncoderReranker::default_scorer()))
            .add_stage(Box::new(DiversityReranker::new(0.5)))
            .add_stage(Box::new(CrossEncoderReranker::default_scorer()));

        let docs = vec![
            make_doc("rust programming", 0.5, 0),
            make_doc("rust systems language", 0.6, 1),
            make_doc("cooking recipes", 0.3, 2),
        ];

        let config = RerankerConfig {
            top_k: 10,
            min_score: f64::NEG_INFINITY,
            ..default_config()
        };

        let result = pipeline.run("rust programming", docs, &config);

        // All 3 docs should survive, and "rust programming" should be first
        // (highest Jaccard to query).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "rust programming");
    }

    // -----------------------------------------------------------------------
    // Config & filtering tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let config = RerankerConfig::default();
        assert_eq!(config.top_k, 10);
        assert!((config.diversity_lambda - 0.5).abs() < f64::EPSILON);
        assert!((config.min_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_score_filter() {
        let ce = CrossEncoderReranker::default_scorer();

        let docs = vec![
            make_doc("rust programming language", 0.9, 0),
            make_doc("the", 0.1, 1), // very low overlap with query
            make_doc("cooking recipes for dinner", 0.3, 2),
        ];

        let config = RerankerConfig {
            top_k: 10,
            min_score: 0.3,
            ..default_config()
        };

        let result = ce.rerank("rust programming", &docs, &config);

        // Only docs whose reranked score >= 0.3 should survive.
        for doc in &result {
            assert!(
                doc.score >= 0.3,
                "Doc '{}' has score {} below min_score 0.3",
                doc.content,
                doc.score
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_input() {
        let ce = CrossEncoderReranker::default_scorer();
        let result = ce.rerank("query", &[], &default_config());
        assert!(result.is_empty());

        let mmr = DiversityReranker::new(0.5);
        let result = mmr.rerank("query", &[], &default_config());
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_doc() {
        let ce = CrossEncoderReranker::default_scorer();
        let docs = vec![make_doc("the only document", 0.5, 0)];
        let config = RerankerConfig {
            top_k: 10,
            min_score: 0.0,
            ..default_config()
        };
        let result = ce.rerank("document", &docs, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "the only document");
    }

    #[test]
    fn test_scored_document_ordering() {
        let mut docs = vec![
            make_doc("low", 0.1, 2),
            make_doc("high", 0.9, 0),
            make_doc("mid", 0.5, 1),
        ];

        docs.sort();

        assert_eq!(docs[0].content, "high");
        assert_eq!(docs[1].content, "mid");
        assert_eq!(docs[2].content, "low");
    }

    #[test]
    fn test_rrf_k_parameter() {
        let rrf_small_k = ReciprocalRankFusion::new(1);
        let rrf_large_k = ReciprocalRankFusion::new(100);

        let list = vec![make_doc("A", 1.0, 0), make_doc("B", 0.5, 1)];

        let fused_small = rrf_small_k.fuse(&[list.clone()]);
        let fused_large = rrf_large_k.fuse(&[list]);

        // With k=1: A gets 1/(1+1) = 0.5, B gets 1/(1+2) = 0.333
        // The gap is 0.5 - 0.333 = 0.167
        let gap_small = fused_small[0].score - fused_small[1].score;

        // With k=100: A gets 1/(100+1) ~ 0.0099, B gets 1/(100+2) ~ 0.0098
        // The gap is much smaller
        let gap_large = fused_large[0].score - fused_large[1].score;

        assert!(
            gap_small > gap_large,
            "Smaller k should produce larger score gaps: {} vs {}",
            gap_small,
            gap_large
        );
    }
}
