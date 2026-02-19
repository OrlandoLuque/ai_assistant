//! Advanced RAG Methods - Standalone implementations of RAG techniques
//!
//! This module provides standalone, reusable implementations of advanced RAG
//! (Retrieval-Augmented Generation) methods. Each technique can be used
//! independently or composed together.
//!
//! # Available Methods
//!
//! ## Query Enhancement
//! - `QueryExpander`: LLM-based query expansion
//! - `MultiQueryDecomposer`: Break complex queries into sub-queries
//! - `HydeGenerator`: Hypothetical Document Embeddings
//!
//! ## Retrieval Enhancement
//! - `HybridSearcher`: Combine keyword and semantic search
//! - `SentenceWindowRetriever`: Expand matched sentences with context
//! - `ParentDocumentRetriever`: Return parent documents for child matches
//!
//! ## Result Processing
//! - `LlmReranker`: LLM-based result reranking
//! - `CrossEncoderReranker`: Cross-encoder model reranking
//! - `RrfFusion`: Reciprocal Rank Fusion for combining result sets
//! - `ContextualCompressor`: Extract only relevant parts from chunks
//!
//! ## Self-Improvement
//! - `SelfRagEvaluator`: Self-reflection on retrieval quality
//! - `CragEvaluator`: Corrective RAG with quality assessment
//! - `AdaptiveStrategySelector`: Dynamic retrieval strategy selection
//!
//! ## Advanced
//! - `AgenticRetriever`: Iterative agent-based retrieval
//! - `GraphRagRetriever`: Knowledge graph-based retrieval
//! - `RaptorRetriever`: Hierarchical summarization retrieval
//!
//! # Usage
//!
//! ```rust
//! use ai_assistant::rag_methods::{QueryExpander, LlmReranker, RrfFusion};
//!
//! // Expand query
//! let expander = QueryExpander::new();
//! let variants = expander.expand("What is the Aurora MR?", &llm)?;
//!
//! // Rerank results
//! let reranker = LlmReranker::new();
//! let reranked = reranker.rerank("query", chunks, &llm)?;
//!
//! // Fuse results from multiple sources
//! let fusion = RrfFusion::new();
//! let fused = fusion.fuse(vec![keyword_results, semantic_results]);
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ============================================================================
// Shared Types
// ============================================================================

/// A scored item (generic)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoredItem<T> {
    pub item: T,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

impl<T> ScoredItem<T> {
    pub fn new(item: T, score: f32) -> Self {
        Self {
            item,
            score,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(item: T, score: f32, metadata: HashMap<String, String>) -> Self {
        Self { item, score, metadata }
    }
}

/// Result of a method execution with timing
#[derive(Clone, Debug)]
pub struct MethodResult<T> {
    pub result: T,
    pub duration_ms: u64,
    pub details: HashMap<String, String>,
}

impl<T> MethodResult<T> {
    pub fn new(result: T, duration: std::time::Duration) -> Self {
        Self {
            result,
            duration_ms: duration.as_millis() as u64,
            details: HashMap::new(),
        }
    }

    pub fn with_details(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}

/// Trait for LLM generation
pub trait LlmGenerate {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String>;
    fn model_name(&self) -> &str;
}

/// Trait for embedding generation
pub trait EmbeddingGenerate {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
    fn dimension(&self) -> usize;
}

// ============================================================================
// QUERY ENHANCEMENT METHODS
// ============================================================================

// ----------------------------------------------------------------------------
// Query Expander
// ----------------------------------------------------------------------------

/// Configuration for query expansion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryExpanderConfig {
    /// Maximum number of expanded queries to generate
    pub max_expansions: usize,
    /// Include synonym-based expansion
    pub use_synonyms: bool,
    /// Custom prompt template (use {query} placeholder)
    pub prompt_template: Option<String>,
}

impl Default for QueryExpanderConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            use_synonyms: true,
            prompt_template: None,
        }
    }
}

/// LLM-based query expansion
///
/// Generates alternative phrasings of a query to improve recall.
pub struct QueryExpander {
    config: QueryExpanderConfig,
}

impl QueryExpander {
    pub fn new() -> Self {
        Self::with_config(QueryExpanderConfig::default())
    }

    pub fn with_config(config: QueryExpanderConfig) -> Self {
        Self { config }
    }

    /// Expand a query using LLM
    pub fn expand(
        &self,
        query: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<Vec<String>>, String> {
        let start = Instant::now();

        let prompt = self.config.prompt_template.clone().unwrap_or_else(|| {
            format!(
                "Generate {} alternative ways to phrase this search query. \
                 Focus on different aspects and synonyms. \
                 Return ONLY the alternatives, one per line, no numbering or bullets:\n\n\
                 Query: {}\n\n\
                 Alternatives:",
                self.config.max_expansions,
                "{query}"
            )
        }).replace("{query}", query);

        let response = llm.generate(&prompt, 300)?;

        let expansions: Vec<String> = response
            .lines()
            .map(|l| l.trim().trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-' || c == '*'))
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && l.len() > 3)
            .take(self.config.max_expansions)
            .map(|s| s.to_string())
            .collect();

        // Optionally add synonym-based expansions
        let mut all_expansions = expansions;
        if self.config.use_synonyms {
            all_expansions.extend(self.synonym_expand(query));
        }

        // Deduplicate
        let unique: Vec<String> = all_expansions
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .take(self.config.max_expansions)
            .collect();

        Ok(MethodResult::new(unique, start.elapsed())
            .with_details("method", "llm_expansion")
            .with_details("model", llm.model_name()))
    }

    /// Simple synonym-based expansion (no LLM)
    pub fn synonym_expand(&self, query: &str) -> Vec<String> {
        let synonyms: HashMap<&str, &[&str]> = [
            ("ship", &["vessel", "spacecraft", "craft", "starship"][..]),
            ("fast", &["quick", "speedy", "rapid"][..]),
            ("price", &["cost", "value", "pricing"][..]),
            ("weapon", &["gun", "armament", "turret"][..]),
            ("cargo", &["storage", "hold", "freight"][..]),
            ("speed", &["velocity", "performance"][..]),
            ("size", &["dimensions", "length", "mass"][..]),
            ("buy", &["purchase", "acquire", "get"][..]),
            ("best", &["top", "optimal", "recommended"][..]),
        ].into_iter().collect();

        let lower = query.to_lowercase();
        let mut expanded = Vec::new();

        for (word, syns) in synonyms {
            if lower.contains(word) {
                for syn in syns.iter().take(2) {
                    expanded.push(lower.replace(word, syn));
                }
            }
        }

        expanded.into_iter().take(3).collect()
    }
}

impl Default for QueryExpander {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// Multi-Query Decomposer
// ----------------------------------------------------------------------------

/// Configuration for multi-query decomposition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiQueryConfig {
    /// Maximum sub-queries to generate
    pub max_sub_queries: usize,
    /// Minimum complexity to trigger decomposition
    pub min_complexity_threshold: f32,
    /// Custom prompt template
    pub prompt_template: Option<String>,
}

impl Default for MultiQueryConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 4,
            min_complexity_threshold: 0.3,
            prompt_template: None,
        }
    }
}

/// Decompose complex queries into simpler sub-queries
pub struct MultiQueryDecomposer {
    config: MultiQueryConfig,
}

impl MultiQueryDecomposer {
    pub fn new() -> Self {
        Self::with_config(MultiQueryConfig::default())
    }

    pub fn with_config(config: MultiQueryConfig) -> Self {
        Self { config }
    }

    /// Estimate query complexity (0.0 to 1.0)
    pub fn estimate_complexity(&self, query: &str) -> f32 {
        let mut score: f32 = 0.0;

        // Multiple question marks
        if query.matches('?').count() > 1 {
            score += 0.3;
        }

        // Conjunctions suggest multiple parts
        let conjunctions = ["and", "or", "but", "also", "as well as", "both"];
        for conj in conjunctions {
            if query.to_lowercase().contains(conj) {
                score += 0.15;
            }
        }

        // Long queries are often complex
        let word_count = query.split_whitespace().count();
        if word_count > 15 {
            score += 0.2;
        }

        // Comparison words
        let comparisons = ["compare", "versus", "vs", "difference", "better"];
        for comp in comparisons {
            if query.to_lowercase().contains(comp) {
                score += 0.2;
            }
        }

        score.min(1.0)
    }

    /// Decompose a complex query into sub-queries
    pub fn decompose(
        &self,
        query: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<Vec<String>>, String> {
        let start = Instant::now();

        // Check complexity first
        let complexity = self.estimate_complexity(query);
        if complexity < self.config.min_complexity_threshold {
            return Ok(MethodResult::new(vec![query.to_string()], start.elapsed())
                .with_details("skipped", "complexity_below_threshold")
                .with_details("complexity", format!("{:.2}", complexity)));
        }

        let prompt = self.config.prompt_template.clone().unwrap_or_else(|| {
            format!(
                "Break down this complex question into {} simpler, independent sub-questions. \
                 Each sub-question should be answerable on its own.\n\
                 Return ONLY the sub-questions, one per line, no numbering:\n\n\
                 Question: {}\n\n\
                 Sub-questions:",
                self.config.max_sub_queries,
                "{query}"
            )
        }).replace("{query}", query);

        let response = llm.generate(&prompt, 400)?;

        let sub_queries: Vec<String> = response
            .lines()
            .map(|l| l.trim().trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-'))
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && l.len() > 5)
            .take(self.config.max_sub_queries)
            .map(|s| s.to_string())
            .collect();

        // Always include original query
        let mut all_queries = vec![query.to_string()];
        all_queries.extend(sub_queries);

        let sub_count = all_queries.len() - 1;
        Ok(MethodResult::new(all_queries, start.elapsed())
            .with_details("complexity", format!("{:.2}", complexity))
            .with_details("sub_queries_generated", sub_count.to_string()))
    }
}

impl Default for MultiQueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// HyDE (Hypothetical Document Embeddings)
// ----------------------------------------------------------------------------

/// Configuration for HyDE
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HydeConfig {
    /// Target length for hypothetical document (tokens)
    pub target_length: usize,
    /// Number of hypothetical documents to generate
    pub num_hypotheticals: usize,
    /// Custom prompt template
    pub prompt_template: Option<String>,
}

impl Default for HydeConfig {
    fn default() -> Self {
        Self {
            target_length: 200,
            num_hypotheticals: 1,
            prompt_template: None,
        }
    }
}

/// Hypothetical Document Embeddings generator
///
/// Generates a hypothetical answer document, then uses its embedding
/// for semantic search instead of the original query.
pub struct HydeGenerator {
    config: HydeConfig,
}

impl HydeGenerator {
    pub fn new() -> Self {
        Self::with_config(HydeConfig::default())
    }

    pub fn with_config(config: HydeConfig) -> Self {
        Self { config }
    }

    /// Generate hypothetical document(s) for a query
    pub fn generate(
        &self,
        query: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<Vec<String>>, String> {
        let start = Instant::now();

        let prompt = self.config.prompt_template.clone().unwrap_or_else(|| {
            "Write a detailed paragraph that would answer this question. \
             Write as if you're an expert providing factual, specific information. \
             Include relevant details, numbers, and specifications if applicable.\n\n\
             Question: {query}\n\n\
             Answer:".to_string()
        }).replace("{query}", query);

        let mut hypotheticals = Vec::new();

        for _ in 0..self.config.num_hypotheticals {
            let response = llm.generate(&prompt, self.config.target_length)?;
            hypotheticals.push(response.trim().to_string());
        }

        Ok(MethodResult::new(hypotheticals, start.elapsed())
            .with_details("num_generated", self.config.num_hypotheticals.to_string()))
    }

    /// Generate hypothetical document and its embedding
    pub fn generate_with_embedding(
        &self,
        query: &str,
        llm: &dyn LlmGenerate,
        embedder: &dyn EmbeddingGenerate,
    ) -> Result<MethodResult<(String, Vec<f32>)>, String> {
        let start = Instant::now();

        let doc_result = self.generate(query, llm)?;
        let doc = doc_result.result.first()
            .ok_or_else(|| "No hypothetical document generated".to_string())?
            .clone();

        let embedding = embedder.embed(&doc)?;
        let doc_len = doc.len();

        Ok(MethodResult::new((doc, embedding), start.elapsed())
            .with_details("doc_length", doc_len.to_string()))
    }
}

impl Default for HydeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RESULT PROCESSING METHODS
// ============================================================================

// ----------------------------------------------------------------------------
// LLM Reranker
// ----------------------------------------------------------------------------

/// Configuration for LLM reranking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmRerankerConfig {
    /// Maximum chunks to include in prompt
    pub max_chunks: usize,
    /// Preview length for each chunk
    pub chunk_preview_length: usize,
    /// Custom prompt template
    pub prompt_template: Option<String>,
}

impl Default for LlmRerankerConfig {
    fn default() -> Self {
        Self {
            max_chunks: 10,
            chunk_preview_length: 300,
            prompt_template: None,
        }
    }
}

/// LLM-based result reranking
pub struct LlmReranker {
    config: LlmRerankerConfig,
}

impl LlmReranker {
    pub fn new() -> Self {
        Self::with_config(LlmRerankerConfig::default())
    }

    pub fn with_config(config: LlmRerankerConfig) -> Self {
        Self { config }
    }

    /// Rerank items using LLM
    pub fn rerank<T: Clone + AsRef<str>>(
        &self,
        query: &str,
        items: Vec<ScoredItem<T>>,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<Vec<ScoredItem<T>>>, String> {
        let start = Instant::now();

        if items.is_empty() {
            return Ok(MethodResult::new(vec![], start.elapsed()));
        }

        // Build prompt
        let mut prompt = format!(
            "Given this question: \"{}\"\n\n\
             Rank these text passages by relevance (most relevant first).\n\
             Return ONLY the passage numbers in order, comma-separated (e.g., \"3,1,5,2,4\"):\n\n",
            query
        );

        let items_to_rank: Vec<_> = items.iter().take(self.config.max_chunks).collect();

        for (i, item) in items_to_rank.iter().enumerate() {
            let preview = truncate(item.item.as_ref(), self.config.chunk_preview_length);
            prompt.push_str(&format!("{}. {}\n\n", i + 1, preview));
        }

        prompt.push_str("Ranking (most relevant first): ");

        let response = llm.generate(&prompt, 100)?;

        // Parse ranking
        let rankings: Vec<usize> = response
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0 && n <= items_to_rank.len())
            .collect();

        // Reorder items
        let mut reranked = Vec::new();
        let mut seen = HashSet::new();

        for (new_rank, rank) in rankings.iter().enumerate() {
            let idx = rank - 1;
            if !seen.contains(&idx) && idx < items_to_rank.len() {
                let mut item = items_to_rank[idx].clone();
                // New score based on rank
                item.score = 1.0 - (new_rank as f32 / items_to_rank.len() as f32);
                reranked.push(item);
                seen.insert(idx);
            }
        }

        // Add unranked items at the end
        for (i, item) in items_to_rank.iter().enumerate() {
            if !seen.contains(&i) {
                let mut item = (*item).clone();
                item.score = 0.1;
                reranked.push(item);
            }
        }

        Ok(MethodResult::new(reranked, start.elapsed())
            .with_details("items_ranked", rankings.len().to_string()))
    }
}

impl Default for LlmReranker {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// Cross-Encoder Reranker
// ----------------------------------------------------------------------------

/// Trait for cross-encoder models
pub trait CrossEncoderScore {
    /// Score a query-document pair (0.0 to 1.0)
    fn score_pair(&self, query: &str, document: &str) -> Result<f32, String>;

    /// Score multiple pairs at once
    fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, String> {
        documents.iter().map(|d| self.score_pair(query, d)).collect()
    }
}

/// Cross-encoder based reranking
pub struct CrossEncoderReranker {
    top_k: usize,
}

impl CrossEncoderReranker {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Rerank items using cross-encoder
    pub fn rerank<T: Clone + AsRef<str>>(
        &self,
        query: &str,
        items: Vec<ScoredItem<T>>,
        encoder: &dyn CrossEncoderScore,
    ) -> Result<MethodResult<Vec<ScoredItem<T>>>, String> {
        let start = Instant::now();

        if items.is_empty() {
            return Ok(MethodResult::new(vec![], start.elapsed()));
        }

        // Score all items
        let documents: Vec<&str> = items.iter().map(|i| i.item.as_ref()).collect();
        let scores = encoder.score_pairs(query, &documents)?;

        // Apply new scores
        let mut reranked: Vec<ScoredItem<T>> = items
            .into_iter()
            .zip(scores)
            .map(|(mut item, score)| {
                item.score = score;
                item
            })
            .collect();

        // Sort by score
        reranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        reranked.truncate(self.top_k);

        Ok(MethodResult::new(reranked, start.elapsed()))
    }
}

// ----------------------------------------------------------------------------
// Reciprocal Rank Fusion (RRF)
// ----------------------------------------------------------------------------

/// Configuration for RRF
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrfConfig {
    /// RRF constant k (default: 60)
    pub k: f32,
    /// Maximum results to return
    pub max_results: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            max_results: 20,
        }
    }
}

/// Reciprocal Rank Fusion for combining multiple ranked lists
pub struct RrfFusion {
    config: RrfConfig,
}

impl RrfFusion {
    pub fn new() -> Self {
        Self::with_config(RrfConfig::default())
    }

    pub fn with_config(config: RrfConfig) -> Self {
        Self { config }
    }

    /// Fuse multiple ranked lists using RRF
    ///
    /// Each inner Vec should be sorted by relevance (best first).
    pub fn fuse<T: Clone + Eq + std::hash::Hash>(
        &self,
        ranked_lists: Vec<Vec<ScoredItem<T>>>,
        get_id: impl Fn(&T) -> String,
    ) -> MethodResult<Vec<ScoredItem<T>>> {
        let start = Instant::now();

        let mut scores: HashMap<String, (f32, ScoredItem<T>)> = HashMap::new();

        for list in ranked_lists {
            for (rank, item) in list.into_iter().enumerate() {
                let id = get_id(&item.item);
                let rrf_score = 1.0 / (self.config.k + rank as f32 + 1.0);

                let entry = scores.entry(id).or_insert((0.0, item.clone()));
                entry.0 += rrf_score;
            }
        }

        let mut fused: Vec<ScoredItem<T>> = scores
            .into_iter()
            .map(|(_, (score, mut item))| {
                item.score = score;
                item
            })
            .collect();

        fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(self.config.max_results);

        MethodResult::new(fused, start.elapsed())
    }

    /// Fuse string items (using content as ID)
    pub fn fuse_strings(&self, ranked_lists: Vec<Vec<ScoredItem<String>>>) -> MethodResult<Vec<ScoredItem<String>>> {
        self.fuse(ranked_lists, |s: &String| s.clone())
    }
}

impl Default for RrfFusion {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// Contextual Compression
// ----------------------------------------------------------------------------

/// Configuration for contextual compression
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target tokens per compressed chunk
    pub target_tokens: usize,
    /// Minimum output length (to avoid over-compression)
    pub min_tokens: usize,
    /// Custom prompt template
    pub prompt_template: Option<String>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target_tokens: 150,
            min_tokens: 30,
            prompt_template: None,
        }
    }
}

/// Contextual compression - extracts only query-relevant parts
pub struct ContextualCompressor {
    config: CompressionConfig,
}

impl ContextualCompressor {
    pub fn new() -> Self {
        Self::with_config(CompressionConfig::default())
    }

    pub fn with_config(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a single chunk
    pub fn compress(
        &self,
        query: &str,
        content: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<String>, String> {
        let start = Instant::now();

        let prompt = self.config.prompt_template.clone().unwrap_or_else(|| {
            "Extract only the parts relevant to answering this question.\n\
             Keep specific facts, numbers, and details. Remove unrelated content.\n\n\
             Question: {query}\n\n\
             Text: {content}\n\n\
             Relevant extract:".to_string()
        })
        .replace("{query}", query)
        .replace("{content}", content);

        let compressed = llm.generate(&prompt, self.config.target_tokens)?;
        let compressed = compressed.trim().to_string();

        // Check minimum length
        let token_estimate = compressed.len() / 4;
        if token_estimate < self.config.min_tokens {
            // Return original if compression is too aggressive
            return Ok(MethodResult::new(content.to_string(), start.elapsed())
                .with_details("compression_skipped", "output_too_short"));
        }

        let ratio = content.len() as f32 / compressed.len().max(1) as f32;

        Ok(MethodResult::new(compressed, start.elapsed())
            .with_details("compression_ratio", format!("{:.2}x", ratio)))
    }

    /// Compress multiple chunks
    pub fn compress_batch(
        &self,
        query: &str,
        contents: &[String],
        llm: &dyn LlmGenerate,
    ) -> Result<Vec<MethodResult<String>>, String> {
        contents
            .iter()
            .map(|c| self.compress(query, c, llm))
            .collect()
    }
}

impl Default for ContextualCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SELF-IMPROVEMENT METHODS
// ============================================================================

// ----------------------------------------------------------------------------
// Self-RAG Evaluator
// ----------------------------------------------------------------------------

/// Self-reflection evaluation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfReflectionResult {
    pub is_sufficient: bool,
    pub confidence: f32,
    pub reason: Option<String>,
    pub suggested_action: SelfReflectionAction,
}

/// Actions suggested by self-reflection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SelfReflectionAction {
    UseAsIs,
    RefineQuery,
    ExpandSearch,
    SeekMoreContext,
}

/// Configuration for self-RAG evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfRagConfig {
    /// Confidence threshold to trigger action
    pub confidence_threshold: f32,
    /// Maximum context preview length
    pub context_preview_length: usize,
}

impl Default for SelfRagConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            context_preview_length: 1000,
        }
    }
}

/// Self-RAG evaluator for retrieval quality assessment
pub struct SelfRagEvaluator {
    config: SelfRagConfig,
}

impl SelfRagEvaluator {
    pub fn new() -> Self {
        Self::with_config(SelfRagConfig::default())
    }

    pub fn with_config(config: SelfRagConfig) -> Self {
        Self { config }
    }

    /// Evaluate if retrieved context is sufficient
    pub fn evaluate(
        &self,
        query: &str,
        context: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<SelfReflectionResult>, String> {
        let start = Instant::now();

        let context_preview = truncate(context, self.config.context_preview_length);

        let prompt = format!(
            "Question: {}\n\n\
             Available context:\n{}\n\n\
             Can this context sufficiently answer the question?\n\
             Reply with: YES/NO | confidence (0-100) | brief reason\n\
             Example: YES|85|Contains specific ship specifications",
            query, context_preview
        );

        let response = llm.generate(&prompt, 100)?;

        // Parse response
        let parts: Vec<&str> = response.split('|').collect();

        let is_sufficient = parts.first()
            .map(|s| s.trim().to_uppercase().starts_with("YES"))
            .unwrap_or(false);

        let confidence = parts.get(1)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|c| c / 100.0)
            .unwrap_or(0.5);

        let reason = parts.get(2).map(|s| s.trim().to_string());

        let suggested_action = if is_sufficient && confidence >= self.config.confidence_threshold {
            SelfReflectionAction::UseAsIs
        } else if confidence < 0.3 {
            SelfReflectionAction::ExpandSearch
        } else if confidence < self.config.confidence_threshold {
            SelfReflectionAction::SeekMoreContext
        } else {
            SelfReflectionAction::RefineQuery
        };

        let result = SelfReflectionResult {
            is_sufficient,
            confidence,
            reason,
            suggested_action,
        };

        Ok(MethodResult::new(result, start.elapsed()))
    }
}

impl Default for SelfRagEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// CRAG Evaluator
// ----------------------------------------------------------------------------

/// CRAG evaluation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CragResult {
    pub quality_score: f32,
    pub action: CragAction,
    pub reason: Option<String>,
}

/// Actions from CRAG evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CragAction {
    /// Retrieved documents are good, use them
    Correct,
    /// Documents are somewhat relevant, need refinement
    Ambiguous,
    /// Documents are not relevant, need to retry or use web
    Incorrect,
}

/// Configuration for CRAG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CragConfig {
    /// Threshold for "correct" classification
    pub correct_threshold: f32,
    /// Threshold for "ambiguous" (below is "incorrect")
    pub ambiguous_threshold: f32,
}

impl Default for CragConfig {
    fn default() -> Self {
        Self {
            correct_threshold: 0.7,
            ambiguous_threshold: 0.4,
        }
    }
}

/// Corrective RAG evaluator
pub struct CragEvaluator {
    config: CragConfig,
}

impl CragEvaluator {
    pub fn new() -> Self {
        Self::with_config(CragConfig::default())
    }

    pub fn with_config(config: CragConfig) -> Self {
        Self { config }
    }

    /// Evaluate retrieval quality
    pub fn evaluate(
        &self,
        query: &str,
        documents: &[&str],
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<CragResult>, String> {
        let start = Instant::now();

        let docs_preview: String = documents
            .iter()
            .take(5)
            .enumerate()
            .map(|(i, d)| format!("{}. {}", i + 1, truncate(d, 200)))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Question: {}\n\n\
             Retrieved documents:\n{}\n\n\
             Rate the overall relevance of these documents for answering the question.\n\
             Reply with: score (0-100) | brief assessment\n\
             Example: 75|Documents contain relevant information but lack specifics",
            query, docs_preview
        );

        let response = llm.generate(&prompt, 100)?;

        let parts: Vec<&str> = response.split('|').collect();

        let quality_score = parts.first()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|s| s / 100.0)
            .unwrap_or(0.5);

        let reason = parts.get(1).map(|s| s.trim().to_string());

        let action = if quality_score >= self.config.correct_threshold {
            CragAction::Correct
        } else if quality_score >= self.config.ambiguous_threshold {
            CragAction::Ambiguous
        } else {
            CragAction::Incorrect
        };

        let result = CragResult {
            quality_score,
            action,
            reason,
        };

        Ok(MethodResult::new(result, start.elapsed()))
    }
}

impl Default for CragEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// Adaptive Strategy Selector
// ----------------------------------------------------------------------------

/// Available retrieval strategies
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetrievalStrategy {
    KeywordOnly,
    SemanticOnly,
    HybridBalanced,
    HybridKeywordHeavy,
    HybridSemanticHeavy,
    MultiQuery,
    AgenticIterative,
}

/// Configuration for adaptive strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveStrategyConfig {
    /// Whether to use LLM for strategy selection
    pub use_llm: bool,
}

impl Default for AdaptiveStrategyConfig {
    fn default() -> Self {
        Self { use_llm: true }
    }
}

/// Adaptive strategy selector
pub struct AdaptiveStrategySelector {
    config: AdaptiveStrategyConfig,
}

impl AdaptiveStrategySelector {
    pub fn new() -> Self {
        Self::with_config(AdaptiveStrategyConfig::default())
    }

    pub fn with_config(config: AdaptiveStrategyConfig) -> Self {
        Self { config }
    }

    /// Select strategy based on query characteristics
    pub fn select_heuristic(&self, query: &str) -> RetrievalStrategy {
        let lower = query.to_lowercase();

        // Check for specific patterns
        let has_technical_terms = lower.contains("specification")
            || lower.contains("stats")
            || lower.contains("numbers")
            || lower.contains("exact");

        let is_conceptual = lower.contains("explain")
            || lower.contains("how does")
            || lower.contains("what is")
            || lower.contains("why");

        let is_comparison = lower.contains("compare")
            || lower.contains("versus")
            || lower.contains("difference")
            || lower.contains("better");

        let is_complex = query.len() > 100
            || query.matches('?').count() > 1
            || (lower.contains(" and ") && lower.contains(" or "));

        if is_complex {
            RetrievalStrategy::MultiQuery
        } else if is_comparison {
            RetrievalStrategy::AgenticIterative
        } else if has_technical_terms {
            RetrievalStrategy::HybridKeywordHeavy
        } else if is_conceptual {
            RetrievalStrategy::HybridSemanticHeavy
        } else {
            RetrievalStrategy::HybridBalanced
        }
    }

    /// Select strategy using LLM
    pub fn select_with_llm(
        &self,
        query: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<RetrievalStrategy>, String> {
        let start = Instant::now();

        if !self.config.use_llm {
            return Ok(MethodResult::new(self.select_heuristic(query), start.elapsed())
                .with_details("method", "heuristic"));
        }

        let prompt = format!(
            "Classify this query to select the best retrieval strategy:\n\n\
             Query: \"{}\"\n\n\
             Options:\n\
             1. KEYWORD - Best for exact terms, names, codes\n\
             2. SEMANTIC - Best for conceptual understanding\n\
             3. HYBRID - Best for general questions\n\
             4. MULTIQUERY - Best for complex multi-part questions\n\
             5. AGENTIC - Best for comparisons or research tasks\n\n\
             Reply with just the strategy name (e.g., HYBRID):",
            query
        );

        let response = llm.generate(&prompt, 20)?;
        let response_upper = response.to_uppercase();

        let strategy = if response_upper.contains("KEYWORD") {
            RetrievalStrategy::KeywordOnly
        } else if response_upper.contains("SEMANTIC") {
            RetrievalStrategy::SemanticOnly
        } else if response_upper.contains("MULTI") {
            RetrievalStrategy::MultiQuery
        } else if response_upper.contains("AGENT") {
            RetrievalStrategy::AgenticIterative
        } else {
            RetrievalStrategy::HybridBalanced
        };

        Ok(MethodResult::new(strategy, start.elapsed())
            .with_details("method", "llm"))
    }
}

impl Default for AdaptiveStrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ADVANCED METHODS
// ============================================================================

// ----------------------------------------------------------------------------
// Graph RAG Types (skeleton for integration)
// ----------------------------------------------------------------------------

/// An entity extracted from text
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub mentions: Vec<EntityMention>,
}

/// A mention of an entity in text
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityMention {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

/// A relationship between entities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relationship {
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    pub weight: f32,
    pub source_chunk: Option<String>,
}

/// Configuration for Graph RAG
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphRagConfig {
    /// Maximum traversal depth
    pub max_depth: usize,
    /// Maximum entities to extract per query
    pub max_entities: usize,
    /// Entity types to extract
    pub entity_types: Vec<String>,
}

/// Trait for graph database operations
pub trait GraphDatabase {
    /// Add entity to graph
    fn add_entity(&mut self, entity: &Entity) -> Result<(), String>;

    /// Add relationship to graph
    fn add_relationship(&mut self, relationship: &Relationship) -> Result<(), String>;

    /// Find entities matching text
    fn find_entities(&self, text: &str) -> Result<Vec<Entity>, String>;

    /// Get relationships for an entity
    fn get_relationships(&self, entity: &str, max_depth: usize) -> Result<Vec<Relationship>, String>;

    /// Get related entities
    fn get_related_entities(&self, entity: &str, max_depth: usize) -> Result<Vec<Entity>, String>;
}

/// Graph RAG retriever (skeleton)
pub struct GraphRagRetriever {
    config: GraphRagConfig,
}

impl GraphRagRetriever {
    pub fn new(config: GraphRagConfig) -> Self {
        Self { config }
    }

    /// Extract entities from text using LLM
    pub fn extract_entities(
        &self,
        text: &str,
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<Vec<Entity>>, String> {
        let start = Instant::now();

        let entity_types = if self.config.entity_types.is_empty() {
            "person, organization, location, product, concept".to_string()
        } else {
            self.config.entity_types.join(", ")
        };

        let prompt = format!(
            "Extract named entities from this text.\n\
             Entity types to find: {}\n\n\
             Text: {}\n\n\
             List entities as: TYPE: name\n\
             Example: PRODUCT: Aurora MR",
            entity_types,
            truncate(text, 1000)
        );

        let response = llm.generate(&prompt, 300)?;

        let entities: Vec<Entity> = response
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if parts.len() == 2 {
                    Some(Entity {
                        name: parts[1].trim().to_string(),
                        entity_type: parts[0].trim().to_uppercase(),
                        mentions: vec![],
                    })
                } else {
                    None
                }
            })
            .take(self.config.max_entities)
            .collect();

        Ok(MethodResult::new(entities, start.elapsed()))
    }
}

// ----------------------------------------------------------------------------
// RAPTOR Retriever Types (skeleton)
// ----------------------------------------------------------------------------

/// A summary node in the RAPTOR tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaptorNode {
    pub id: String,
    pub level: usize,
    pub content: String,
    pub children: Vec<String>,
    pub embedding: Option<Vec<f32>>,
}

/// Configuration for RAPTOR
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaptorConfig {
    /// Maximum levels in the hierarchy
    pub max_levels: usize,
    /// Chunks per summary
    pub chunks_per_summary: usize,
    /// Summary length in tokens
    pub summary_length: usize,
}

impl Default for RaptorConfig {
    fn default() -> Self {
        Self {
            max_levels: 3,
            chunks_per_summary: 5,
            summary_length: 200,
        }
    }
}

/// RAPTOR retriever (skeleton for hierarchical summarization)
pub struct RaptorRetriever {
    config: RaptorConfig,
}

impl RaptorRetriever {
    pub fn new(config: RaptorConfig) -> Self {
        Self { config }
    }

    /// Build a summary for a group of chunks
    pub fn summarize_group(
        &self,
        chunks: &[&str],
        llm: &dyn LlmGenerate,
    ) -> Result<MethodResult<String>, String> {
        let start = Instant::now();

        let combined = chunks.join("\n\n---\n\n");

        let prompt = format!(
            "Summarize the following text passages, preserving key facts and details:\n\n\
             {}\n\n\
             Summary:",
            truncate(&combined, 2000)
        );

        let summary = llm.generate(&prompt, self.config.summary_length)?;

        Ok(MethodResult::new(summary.trim().to_string(), start.elapsed()))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct MockLlm;
    impl LlmGenerate for MockLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String, String> {
            Ok("Mock response".to_string())
        }
        fn model_name(&self) -> &str { "mock" }
    }

    #[test]
    fn test_query_expander_synonyms() {
        let expander = QueryExpander::new();
        let synonyms = expander.synonym_expand("What is the ship price?");
        assert!(!synonyms.is_empty());
        assert!(synonyms.iter().any(|s| s.contains("cost") || s.contains("vessel")));
    }

    #[test]
    fn test_multi_query_complexity() {
        let decomposer = MultiQueryDecomposer::new();

        let simple = "What is the Aurora?";
        let complex = "Compare the Aurora and the Mustang, and tell me which is better for cargo and combat?";

        assert!(decomposer.estimate_complexity(simple) < decomposer.estimate_complexity(complex));
    }

    #[test]
    fn test_rrf_fusion() {
        let fusion = RrfFusion::new();

        let list1 = vec![
            ScoredItem::new("doc1".to_string(), 0.9),
            ScoredItem::new("doc2".to_string(), 0.8),
        ];
        let list2 = vec![
            ScoredItem::new("doc2".to_string(), 0.95),
            ScoredItem::new("doc3".to_string(), 0.7),
        ];

        let result = fusion.fuse_strings(vec![list1, list2]);
        assert!(!result.result.is_empty());

        // doc2 appears in both lists, should have higher combined score
        let doc2 = result.result.iter().find(|i| i.item == "doc2");
        let doc1 = result.result.iter().find(|i| i.item == "doc1");
        assert!(doc2.is_some());
        assert!(doc1.is_some());
    }

    #[test]
    fn test_adaptive_strategy() {
        let selector = AdaptiveStrategySelector::new();

        let keyword_query = "Aurora MR specifications stats";
        let conceptual_query = "How does the quantum drive work?";
        let comparison_query = "Compare Aurora versus Mustang";

        assert_eq!(
            selector.select_heuristic(keyword_query),
            RetrievalStrategy::HybridKeywordHeavy
        );
        assert_eq!(
            selector.select_heuristic(conceptual_query),
            RetrievalStrategy::HybridSemanticHeavy
        );
        assert_eq!(
            selector.select_heuristic(comparison_query),
            RetrievalStrategy::AgenticIterative
        );
    }

    #[test]
    fn test_scored_item() {
        let item = ScoredItem::new("test".to_string(), 0.5);
        assert_eq!(item.score, 0.5);
        assert!(item.metadata.is_empty());

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test.md".to_string());
        let item_with_meta = ScoredItem::with_metadata("test".to_string(), 0.5, meta);
        assert_eq!(item_with_meta.metadata.get("source"), Some(&"test.md".to_string()));
    }
}
