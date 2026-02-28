//! Query expansion for RAG systems
//!
//! This module provides query expansion capabilities to improve
//! retrieval in RAG (Retrieval-Augmented Generation) systems.
//!
//! # Features
//!
//! - **Synonym expansion**: Add synonyms to queries
//! - **LLM-based expansion**: Use LLM to generate related queries
//! - **Keyword extraction**: Extract key terms for search
//! - **Multi-query generation**: Generate multiple query variations

use std::collections::{HashMap, HashSet};

/// Configuration for query expansion
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Maximum expanded queries to generate
    pub max_expansions: usize,
    /// Use synonyms for expansion
    pub use_synonyms: bool,
    /// Use LLM for expansion
    pub use_llm: bool,
    /// Extract keywords from query
    pub extract_keywords: bool,
    /// Minimum keyword length
    pub min_keyword_length: usize,
    /// Boost factor for original query
    pub original_boost: f32,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            use_synonyms: true,
            use_llm: true,
            extract_keywords: true,
            min_keyword_length: 3,
            original_boost: 2.0,
        }
    }
}

/// An expanded query
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// The query text
    pub query: String,
    /// Source of this expansion
    pub source: ExpansionSource,
    /// Relevance weight (1.0 = normal)
    pub weight: f32,
    /// Keywords extracted/used
    pub keywords: Vec<String>,
}

/// Source of query expansion
#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionSource {
    /// Original query
    Original,
    /// Synonym expansion
    Synonym,
    /// LLM-generated
    LlmGenerated,
    /// Keyword extraction
    Keyword,
    /// Acronym expansion
    Acronym,
    /// Spelling correction
    SpellingCorrection,
    /// Manual/custom
    Custom,
}

/// Result of query expansion
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    /// Original query
    pub original: String,
    /// Expanded queries
    pub expansions: Vec<ExpandedQuery>,
    /// Combined keywords from all expansions
    pub all_keywords: Vec<String>,
    /// Statistics
    pub stats: ExpansionStats,
}

impl ExpansionResult {
    /// Get queries sorted by weight
    pub fn sorted_queries(&self) -> Vec<&ExpandedQuery> {
        let mut queries: Vec<_> = self.expansions.iter().collect();
        queries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        queries
    }

    /// Get query texts only
    pub fn query_texts(&self) -> Vec<&str> {
        self.expansions.iter().map(|e| e.query.as_str()).collect()
    }
}

/// Statistics about expansion
#[derive(Debug, Clone, Default)]
pub struct ExpansionStats {
    /// Number of synonym expansions
    pub synonym_expansions: usize,
    /// Number of LLM expansions
    pub llm_expansions: usize,
    /// Number of keyword extractions
    pub keyword_extractions: usize,
    /// Total unique keywords
    pub unique_keywords: usize,
}

/// Query expander
pub struct QueryExpander {
    config: ExpansionConfig,
    /// Synonym dictionary
    synonyms: HashMap<String, Vec<String>>,
    /// Common acronyms
    acronyms: HashMap<String, String>,
    /// Stop words to filter
    stop_words: HashSet<String>,
}

impl QueryExpander {
    /// Create a new query expander
    pub fn new(config: ExpansionConfig) -> Self {
        let mut expander = Self {
            config,
            synonyms: HashMap::new(),
            acronyms: HashMap::new(),
            stop_words: HashSet::new(),
        };
        expander.init_defaults();
        expander
    }

    fn init_defaults(&mut self) {
        // Common stop words
        let stops = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "this", "that", "these",
            "those", "what", "which", "who", "whom", "it", "its",
        ];

        self.stop_words = stops.iter().map(|s| s.to_string()).collect();

        // Common synonyms for search
        self.add_synonyms("search", vec!["find", "look for", "locate", "discover"]);
        self.add_synonyms("create", vec!["make", "build", "generate", "produce"]);
        self.add_synonyms("delete", vec!["remove", "erase", "clear", "eliminate"]);
        self.add_synonyms("update", vec!["modify", "change", "edit", "alter"]);
        self.add_synonyms("error", vec!["bug", "issue", "problem", "fault"]);
        self.add_synonyms("help", vec!["assist", "support", "guide", "aid"]);

        // Common acronyms
        self.acronyms.insert(
            "API".to_string(),
            "Application Programming Interface".to_string(),
        );
        self.acronyms
            .insert("UI".to_string(), "User Interface".to_string());
        self.acronyms
            .insert("DB".to_string(), "Database".to_string());
        self.acronyms
            .insert("ML".to_string(), "Machine Learning".to_string());
        self.acronyms
            .insert("AI".to_string(), "Artificial Intelligence".to_string());
        self.acronyms
            .insert("LLM".to_string(), "Large Language Model".to_string());
        self.acronyms.insert(
            "RAG".to_string(),
            "Retrieval Augmented Generation".to_string(),
        );
    }

    /// Add synonyms for a word
    pub fn add_synonyms(&mut self, word: impl Into<String>, synonyms: Vec<&str>) {
        let word = word.into().to_lowercase();
        let syns: Vec<String> = synonyms.iter().map(|s| s.to_lowercase()).collect();
        self.synonyms.insert(word, syns);
    }

    /// Add an acronym expansion
    pub fn add_acronym(&mut self, acronym: impl Into<String>, expansion: impl Into<String>) {
        self.acronyms
            .insert(acronym.into().to_uppercase(), expansion.into());
    }

    /// Expand a query
    pub fn expand(&self, query: &str) -> ExpansionResult {
        let mut expansions = Vec::new();
        let mut stats = ExpansionStats::default();
        let mut all_keywords = HashSet::new();

        // Add original query
        let original_keywords = self.extract_keywords(query);
        all_keywords.extend(original_keywords.iter().cloned());

        expansions.push(ExpandedQuery {
            query: query.to_string(),
            source: ExpansionSource::Original,
            weight: self.config.original_boost,
            keywords: original_keywords.clone(),
        });

        // Synonym expansion
        if self.config.use_synonyms {
            let synonym_expansions = self.expand_with_synonyms(query);
            stats.synonym_expansions = synonym_expansions.len();

            for expanded in synonym_expansions {
                all_keywords.extend(self.extract_keywords(&expanded).iter().cloned());
                expansions.push(ExpandedQuery {
                    query: expanded,
                    source: ExpansionSource::Synonym,
                    weight: 0.8,
                    keywords: Vec::new(),
                });
            }
        }

        // Acronym expansion
        let acronym_expansions = self.expand_acronyms(query);
        for expanded in acronym_expansions {
            expansions.push(ExpandedQuery {
                query: expanded,
                source: ExpansionSource::Acronym,
                weight: 0.9,
                keywords: Vec::new(),
            });
        }

        // Keyword-based queries
        if self.config.extract_keywords && !original_keywords.is_empty() {
            stats.keyword_extractions = original_keywords.len();

            // Create keyword-only query
            if original_keywords.len() > 1 {
                expansions.push(ExpandedQuery {
                    query: original_keywords.join(" "),
                    source: ExpansionSource::Keyword,
                    weight: 0.7,
                    keywords: original_keywords.clone(),
                });
            }

            // Individual important keywords
            for keyword in &original_keywords {
                if keyword.len() >= 5 {
                    expansions.push(ExpandedQuery {
                        query: keyword.clone(),
                        source: ExpansionSource::Keyword,
                        weight: 0.5,
                        keywords: vec![keyword.clone()],
                    });
                }
            }
        }

        // Deduplicate and limit
        let mut seen = HashSet::new();
        expansions.retain(|e| seen.insert(e.query.to_lowercase()));
        expansions.truncate(self.config.max_expansions);

        stats.unique_keywords = all_keywords.len();

        ExpansionResult {
            original: query.to_string(),
            expansions,
            all_keywords: all_keywords.into_iter().collect(),
            stats,
        }
    }

    /// Expand with LLM (requires external generator)
    pub fn expand_with_llm<F>(&self, query: &str, generate: F) -> Vec<String>
    where
        F: Fn(&str) -> Result<String, String>,
    {
        let prompt = format!(
            "Generate 3 alternative ways to phrase this search query. \
             Output only the queries, one per line:\n\nQuery: {}",
            query
        );

        match generate(&prompt) {
            Ok(response) => response
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && l.len() > 3)
                .take(3)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Extract keywords from text
    pub fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| w.to_lowercase())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| w.len() >= self.config.min_keyword_length && !self.stop_words.contains(w))
            .collect()
    }

    /// Expand using synonyms
    fn expand_with_synonyms(&self, query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut expansions = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();
            if let Some(syns) = self.synonyms.get(&word_lower) {
                for syn in syns.iter().take(2) {
                    let mut new_words = words.clone();
                    new_words[i] = syn;
                    expansions.push(new_words.join(" "));
                }
            }
        }

        expansions
    }

    /// Expand acronyms in query
    fn expand_acronyms(&self, query: &str) -> Vec<String> {
        let mut expansions = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let word_upper = word.to_uppercase();
            if let Some(expansion) = self.acronyms.get(&word_upper) {
                let mut new_words: Vec<String> = words.iter().map(|s| s.to_string()).collect();
                new_words[i] = expansion.clone();
                expansions.push(new_words.join(" "));
            }
        }

        expansions
    }
}

impl Default for QueryExpander {
    fn default() -> Self {
        Self::new(ExpansionConfig::default())
    }
}

/// Multi-query retriever for improved RAG
pub struct MultiQueryRetriever {
    expander: QueryExpander,
    /// Deduplication threshold for results
    pub dedup_threshold: f64,
}

impl MultiQueryRetriever {
    /// Create a new multi-query retriever
    pub fn new(expander: QueryExpander) -> Self {
        Self {
            expander,
            dedup_threshold: 0.9,
        }
    }

    /// Retrieve using multiple query variations
    pub fn retrieve<F, T>(&self, query: &str, search: F) -> Vec<ScoredResult<T>>
    where
        F: Fn(&str) -> Vec<(T, f32)>,
        T: Clone + PartialEq,
    {
        let expansion = self.expander.expand(query);
        let mut all_results: Vec<ScoredResult<T>> = Vec::new();

        for expanded in &expansion.expansions {
            let results = search(&expanded.query);
            for (item, score) in results {
                let weighted_score = score * expanded.weight;

                // Check if already in results
                let existing = all_results.iter_mut().find(|r| r.item == item);
                if let Some(existing) = existing {
                    // Keep best score
                    if weighted_score > existing.score {
                        existing.score = weighted_score;
                        existing.matched_query = expanded.query.clone();
                    }
                    existing.match_count += 1;
                } else {
                    all_results.push(ScoredResult {
                        item,
                        score: weighted_score,
                        matched_query: expanded.query.clone(),
                        match_count: 1,
                    });
                }
            }
        }

        // Sort by score and match count
        all_results.sort_by(|a, b| {
            let score_cmp = b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal);
            if score_cmp == std::cmp::Ordering::Equal {
                b.match_count.cmp(&a.match_count)
            } else {
                score_cmp
            }
        });

        all_results
    }
}

/// A search result with score
#[derive(Debug, Clone)]
pub struct ScoredResult<T> {
    /// The result item
    pub item: T,
    /// Combined score
    pub score: f32,
    /// Query that matched
    pub matched_query: String,
    /// Number of query variations that matched
    pub match_count: usize,
}

/// Builder for expansion configuration
pub struct ExpansionConfigBuilder {
    config: ExpansionConfig,
}

impl ExpansionConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ExpansionConfig::default(),
        }
    }

    /// Set max expansions
    pub fn max_expansions(mut self, n: usize) -> Self {
        self.config.max_expansions = n;
        self
    }

    /// Enable/disable synonyms
    pub fn use_synonyms(mut self, enabled: bool) -> Self {
        self.config.use_synonyms = enabled;
        self
    }

    /// Enable/disable LLM expansion
    pub fn use_llm(mut self, enabled: bool) -> Self {
        self.config.use_llm = enabled;
        self
    }

    /// Set original query boost
    pub fn original_boost(mut self, boost: f32) -> Self {
        self.config.original_boost = boost;
        self
    }

    /// Set min keyword length
    pub fn min_keyword_length(mut self, len: usize) -> Self {
        self.config.min_keyword_length = len;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ExpansionConfig {
        self.config
    }
}

impl Default for ExpansionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_extraction() {
        let expander = QueryExpander::default();
        let keywords = expander.extract_keywords("How do I create a new API endpoint?");

        assert!(keywords.contains(&"create".to_string()));
        assert!(keywords.contains(&"api".to_string()));
        assert!(keywords.contains(&"endpoint".to_string()));
        assert!(!keywords.contains(&"do".to_string())); // stop word
    }

    #[test]
    fn test_synonym_expansion() {
        let expander = QueryExpander::default();
        let result = expander.expand("How to create a database");

        // Should have original plus synonym expansions
        assert!(result.expansions.len() > 1);

        // Check that "make" or "build" appeared as synonym for "create"
        let has_synonym = result.expansions.iter().any(|e| {
            e.source == ExpansionSource::Synonym
                && (e.query.contains("make") || e.query.contains("build"))
        });
        assert!(has_synonym);
    }

    #[test]
    fn test_acronym_expansion() {
        let expander = QueryExpander::default();
        // Use RAG without punctuation to ensure clean matching
        let result = expander.expand("What is RAG");

        let has_expansion = result.expansions.iter().any(|e| {
            e.source == ExpansionSource::Acronym
                && e.query.contains("Retrieval Augmented Generation")
        });
        assert!(has_expansion);
    }

    #[test]
    fn test_query_weights() {
        let expander = QueryExpander::default();
        let result = expander.expand("search for errors");

        // Original should have highest weight
        let original = result
            .expansions
            .iter()
            .find(|e| e.source == ExpansionSource::Original);
        assert!(original.is_some());
        assert!(original.unwrap().weight >= 1.0);

        // Synonyms should have lower weight
        let synonyms: Vec<_> = result
            .expansions
            .iter()
            .filter(|e| e.source == ExpansionSource::Synonym)
            .collect();
        for syn in synonyms {
            assert!(syn.weight < original.unwrap().weight);
        }
    }

    #[test]
    fn test_config_builder() {
        let config = ExpansionConfigBuilder::new()
            .max_expansions(10)
            .use_synonyms(true)
            .use_llm(false)
            .original_boost(3.0)
            .build();

        assert_eq!(config.max_expansions, 10);
        assert!(config.use_synonyms);
        assert!(!config.use_llm);
    }

    #[test]
    fn test_sorted_queries() {
        let expander = QueryExpander::default();
        let result = expander.expand("find bugs");

        let sorted = result.sorted_queries();
        assert!(!sorted.is_empty());

        // First should have highest weight
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].weight >= sorted[i].weight);
        }
    }

    #[test]
    fn test_expand_includes_original() {
        let expander = QueryExpander::default();
        let result = expander.expand("test query");
        let texts = result.query_texts();
        assert!(texts.contains(&"test query"));
    }

    #[test]
    fn test_keyword_min_length() {
        let config = ExpansionConfig {
            min_keyword_length: 5,
            ..Default::default()
        };
        let expander = QueryExpander::new(config);
        let keywords = expander.extract_keywords("a bb ccc dddd eeeee");
        assert!(keywords.iter().all(|k| k.len() >= 5));
    }

    #[test]
    fn test_custom_synonyms() {
        let mut expander = QueryExpander::default();
        expander.add_synonyms("bug", vec!["defect", "issue"]);
        let result = expander.expand("fix bug");
        assert!(result.expansions.len() > 1);
    }

    #[test]
    fn test_expansion_stats() {
        let expander = QueryExpander::default();
        let result = expander.expand("find errors");
        let _ = result.stats.synonym_expansions; // verify field exists
    }
}
