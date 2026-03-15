//! Memory search optimization: hybrid search combining keyword matching, recency,
//! and access frequency to rank memory results.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Weights for the hybrid memory search scoring function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchWeights {
    /// Weight for keyword-match score component.
    pub keyword_weight: f64,
    /// Weight for embedding similarity score component.
    pub embedding_weight: f64,
    /// Weight for recency score component.
    pub recency_weight: f64,
    /// Weight for access-frequency score component.
    pub access_frequency_weight: f64,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            keyword_weight: 0.25,
            embedding_weight: 0.25,
            recency_weight: 0.25,
            access_frequency_weight: 0.25,
        }
    }
}

/// A scored memory search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    /// The content of the matched memory.
    pub content: String,
    /// Overall relevance score (higher = more relevant).
    pub relevance_score: f64,
    /// Reasons this result matched the query.
    pub match_reasons: Vec<MatchReason>,
    /// The type of memory store this result came from.
    pub source_type: MemorySourceType,
    /// Timestamp of the original memory entry.
    pub timestamp: u64,
}

/// A reason why a memory matched a search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MatchReason {
    /// Matched via keyword overlap.
    KeywordMatch { keyword: String, count: usize },
    /// Matched via embedding similarity.
    EmbeddingSimilarity { score: f64 },
    /// Matched due to recent access.
    RecentAccess { age_secs: u64 },
    /// Matched due to frequent access.
    FrequentAccess { access_count: usize },
}

/// The type of memory store a search result originated from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemorySourceType {
    /// From the episodic memory store.
    Episodic,
    /// From the procedural memory store.
    Procedural,
    /// From the entity memory store.
    Entity,
    /// From the fact store.
    Fact,
}

/// An entry in the memory search index.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// The content text of this memory.
    pub content: String,
    /// Which memory store this entry came from.
    pub source_type: MemorySourceType,
    /// Keywords associated with this entry.
    pub keywords: Vec<String>,
    /// Timestamp of the original memory.
    pub timestamp: u64,
    /// Number of times this memory has been accessed.
    pub access_count: usize,
}

/// An inverted keyword index over memory entries for fast lookup.
pub struct MemoryIndex {
    pub(crate) keyword_index: HashMap<String, Vec<usize>>,
    pub(crate) entries: Vec<IndexEntry>,
}

impl MemoryIndex {
    /// Create an empty memory index.
    pub fn new() -> Self {
        Self {
            keyword_index: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Add an entry to the index, updating the keyword map.
    pub fn add_entry(&mut self, entry: IndexEntry) {
        let idx = self.entries.len();
        for kw in &entry.keywords {
            let kw_lower = kw.to_lowercase();
            self.keyword_index
                .entry(kw_lower)
                .or_default()
                .push(idx);
        }
        self.entries.push(entry);
    }

    /// Search for entries matching a keyword (case-insensitive). Returns the
    /// indices of matching entries.
    pub fn search_keyword(&self, keyword: &str) -> Vec<usize> {
        let kw_lower = keyword.to_lowercase();
        self.keyword_index
            .get(&kw_lower)
            .cloned()
            .unwrap_or_default()
    }

    /// Return the total number of indexed entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get a reference to an entry by index.
    pub fn get_entry(&self, index: usize) -> Option<&IndexEntry> {
        self.entries.get(index)
    }

    /// Clear all entries and the keyword index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.keyword_index.clear();
    }

    /// Rebuild the keyword index from the current entries. Useful after bulk
    /// modifications.
    pub fn rebuild_index(&mut self) {
        self.keyword_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            for kw in &entry.keywords {
                let kw_lower = kw.to_lowercase();
                self.keyword_index
                    .entry(kw_lower)
                    .or_default()
                    .push(idx);
            }
        }
    }
}

/// Hybrid memory search engine that combines keyword matching, recency, and
/// access frequency to rank results.
pub struct MemorySearchEngine {
    index: MemoryIndex,
    weights: SearchWeights,
}

impl MemorySearchEngine {
    /// Create a search engine with the given weights.
    pub fn new(weights: SearchWeights) -> Self {
        Self {
            index: MemoryIndex::new(),
            weights,
        }
    }

    /// Create a search engine with default weights (all 0.25).
    pub fn with_default_weights() -> Self {
        Self::new(SearchWeights::default())
    }

    /// Add a memory to the search index.
    pub fn add_memory(
        &mut self,
        content: String,
        source_type: MemorySourceType,
        keywords: Vec<String>,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.index.add_entry(IndexEntry {
            content,
            source_type,
            keywords,
            timestamp: now,
            access_count: 0,
        });
    }

    /// Perform a hybrid search combining keyword matching, recency, and access
    /// frequency. Returns the top `max_results` results sorted by relevance.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<MemorySearchResult> {
        if query.is_empty() || self.index.entry_count() == 0 {
            return Vec::new();
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Tokenize query into keywords
        let query_keywords: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Find all candidate entries via keyword index
        let mut entry_scores: HashMap<usize, (f64, Vec<MatchReason>)> = HashMap::new();

        for qk in &query_keywords {
            let matching_indices = self.index.search_keyword(qk);
            for idx in matching_indices {
                let entry = entry_scores.entry(idx).or_insert_with(|| (0.0, Vec::new()));
                // Count keyword occurrences in the entry's keywords
                if let Some(index_entry) = self.index.get_entry(idx) {
                    let count = index_entry
                        .keywords
                        .iter()
                        .filter(|k| k.to_lowercase() == *qk)
                        .count();
                    entry.0 += count as f64;
                    entry.1.push(MatchReason::KeywordMatch {
                        keyword: qk.clone(),
                        count,
                    });
                }
            }
        }

        // If no keyword matches, try content-level word matching as fallback
        if entry_scores.is_empty() {
            for (idx, entry) in self.index.entries.iter().enumerate() {
                let content_lower = entry.content.to_lowercase();
                let mut matched = false;
                for qk in &query_keywords {
                    if content_lower.contains(qk.as_str()) {
                        let count = content_lower.matches(qk.as_str()).count();
                        let scores = entry_scores
                            .entry(idx)
                            .or_insert_with(|| (0.0, Vec::new()));
                        scores.0 += count as f64;
                        scores.1.push(MatchReason::KeywordMatch {
                            keyword: qk.clone(),
                            count,
                        });
                        matched = true;
                    }
                }
                if !matched {
                    continue;
                }
            }
        }

        // Score each candidate
        // Find max values for normalization
        let max_keyword_score = entry_scores
            .values()
            .map(|(s, _)| *s)
            .fold(0.0_f64, f64::max);
        let max_access_count = self
            .index
            .entries
            .iter()
            .map(|e| e.access_count)
            .max()
            .unwrap_or(1)
            .max(1) as f64;
        let max_age = self
            .index
            .entries
            .iter()
            .map(|e| now.saturating_sub(e.timestamp))
            .max()
            .unwrap_or(1)
            .max(1) as f64;

        let mut results: Vec<MemorySearchResult> = entry_scores
            .into_iter()
            .filter_map(|(idx, (kw_score, mut reasons))| {
                let entry = self.index.get_entry(idx)?;

                // Normalize keyword score to [0, 1]
                let norm_kw = if max_keyword_score > 0.0 {
                    kw_score / max_keyword_score
                } else {
                    0.0
                };

                // Recency score: newer = higher (1.0 for newest, 0.0 for oldest)
                let age_secs = now.saturating_sub(entry.timestamp);
                let recency = if max_age > 0.0 {
                    1.0 - (age_secs as f64 / max_age)
                } else {
                    1.0
                };
                if age_secs > 0 {
                    reasons.push(MatchReason::RecentAccess { age_secs });
                }

                // Access frequency score
                let freq = entry.access_count as f64 / max_access_count;
                if entry.access_count > 0 {
                    reasons.push(MatchReason::FrequentAccess {
                        access_count: entry.access_count,
                    });
                }

                let score = norm_kw * self.weights.keyword_weight
                    + recency * self.weights.recency_weight
                    + freq * self.weights.access_frequency_weight;

                Some(MemorySearchResult {
                    content: entry.content.clone(),
                    relevance_score: score,
                    match_reasons: reasons,
                    source_type: entry.source_type.clone(),
                    timestamp: entry.timestamp,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(max_results);
        results
    }

    /// Return the total number of memories in the index.
    pub fn memory_count(&self) -> usize {
        self.index.entry_count()
    }

    /// Get a reference to the current search weights.
    pub fn weights(&self) -> &SearchWeights {
        &self.weights
    }
}
