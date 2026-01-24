//! Response caching with TTL and intelligent invalidation
//!
//! This module provides caching for AI responses to avoid redundant API calls
//! for similar queries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// A cached response entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// The response content
    pub content: String,
    /// When the entry was created
    #[serde(skip)]
    pub created_at: Option<Instant>,
    /// Time-to-live
    #[serde(skip)]
    pub ttl: Duration,
    /// Number of cache hits
    pub hit_count: u32,
    /// The model used
    pub model: String,
    /// Token count
    pub tokens: usize,
    /// Query fingerprint
    pub fingerprint: u64,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl CachedResponse {
    /// Check if the entry has expired
    pub fn is_expired(&self) -> bool {
        self.created_at
            .map(|t| t.elapsed() > self.ttl)
            .unwrap_or(true)
    }

    /// Get remaining TTL
    pub fn remaining_ttl(&self) -> Duration {
        self.created_at
            .map(|t| self.ttl.saturating_sub(t.elapsed()))
            .unwrap_or(Duration::ZERO)
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Default TTL for entries
    pub default_ttl: Duration,
    /// TTL for different query types
    pub ttl_by_type: HashMap<String, Duration>,
    /// Enable fuzzy matching
    pub fuzzy_matching: bool,
    /// Similarity threshold for fuzzy matching (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        let mut ttl_by_type = HashMap::new();
        ttl_by_type.insert("factual".to_string(), Duration::from_secs(86400)); // 24 hours
        ttl_by_type.insert("creative".to_string(), Duration::from_secs(3600)); // 1 hour
        ttl_by_type.insert("code".to_string(), Duration::from_secs(43200)); // 12 hours
        ttl_by_type.insert("translation".to_string(), Duration::from_secs(86400 * 7)); // 1 week

        Self {
            max_entries: 1000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            ttl_by_type,
            fuzzy_matching: true,
            similarity_threshold: 0.85,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Response cache
#[derive(Debug)]
pub struct ResponseCache {
    /// Cached entries by fingerprint
    entries: HashMap<u64, CachedResponse>,
    /// Index from normalized query to fingerprint
    query_index: HashMap<String, u64>,
    /// Configuration
    config: CacheConfig,
    /// Last cleanup time
    last_cleanup: Instant,
    /// Statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total entries
    pub total_entries: usize,
    /// Expired entries removed
    pub expired_removed: u64,
    /// Evicted entries (due to capacity)
    pub evicted: u64,
    /// Total tokens saved
    pub tokens_saved: usize,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl ResponseCache {
    /// Create a new cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            query_index: HashMap::new(),
            config,
            last_cleanup: Instant::now(),
            stats: CacheStats::default(),
        }
    }

    /// Get a cached response (returns cloned content to avoid borrow issues)
    pub fn get(&mut self, query: &str, model: &str) -> Option<CachedResponse> {
        // Try exact match first
        let fingerprint = self.compute_fingerprint(query, model);

        if let Some(entry) = self.entries.get_mut(&fingerprint) {
            if !entry.is_expired() {
                entry.record_hit();
                self.stats.hits += 1;
                self.stats.tokens_saved += entry.tokens;
                return Some(entry.clone());
            } else {
                // Remove expired entry
                self.entries.remove(&fingerprint);
                self.stats.expired_removed += 1;
            }
        }

        // Try fuzzy match if enabled
        if self.config.fuzzy_matching {
            if let Some(fp) = self.find_similar_fp(query, model) {
                if let Some(entry) = self.entries.get_mut(&fp) {
                    entry.record_hit();
                    self.stats.hits += 1;
                    return Some(entry.clone());
                }
            }
        }

        self.stats.misses += 1;
        None
    }

    /// Store a response in the cache
    pub fn put(
        &mut self,
        query: &str,
        model: &str,
        response: &str,
        tokens: usize,
        query_type: Option<&str>,
    ) {
        // Run cleanup if needed
        if self.config.auto_cleanup && self.last_cleanup.elapsed() > self.config.cleanup_interval {
            self.cleanup();
        }

        // Check capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_oldest();
        }

        let fingerprint = self.compute_fingerprint(query, model);
        let ttl = query_type
            .and_then(|t| self.config.ttl_by_type.get(t))
            .copied()
            .unwrap_or(self.config.default_ttl);

        let entry = CachedResponse {
            content: response.to_string(),
            created_at: Some(Instant::now()),
            ttl,
            hit_count: 0,
            model: model.to_string(),
            tokens,
            fingerprint,
            tags: query_type.map(|t| vec![t.to_string()]).unwrap_or_default(),
        };

        self.entries.insert(fingerprint, entry);
        self.query_index.insert(self.normalize_query(query), fingerprint);
        self.stats.total_entries = self.entries.len();
    }

    /// Remove a specific entry
    pub fn remove(&mut self, query: &str, model: &str) -> Option<CachedResponse> {
        let fingerprint = self.compute_fingerprint(query, model);
        let entry = self.entries.remove(&fingerprint);
        if entry.is_some() {
            self.query_index.remove(&self.normalize_query(query));
            self.stats.total_entries = self.entries.len();
        }
        entry
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.query_index.clear();
        self.stats.total_entries = 0;
    }

    /// Clear entries for a specific model
    pub fn clear_model(&mut self, model: &str) {
        self.entries.retain(|_, e| e.model != model);
        self.stats.total_entries = self.entries.len();
    }

    /// Clear entries by tag
    pub fn clear_by_tag(&mut self, tag: &str) {
        self.entries.retain(|_, e| !e.tags.contains(&tag.to_string()));
        self.stats.total_entries = self.entries.len();
    }

    /// Clean up expired entries
    pub fn cleanup(&mut self) {
        let before_count = self.entries.len();
        self.entries.retain(|_, e| !e.is_expired());
        let removed = before_count - self.entries.len();
        self.stats.expired_removed += removed as u64;
        self.stats.total_entries = self.entries.len();
        self.last_cleanup = Instant::now();
    }

    /// Evict the oldest entry
    fn evict_oldest(&mut self) {
        if let Some((&oldest_fp, _)) = self.entries
            .iter()
            .min_by_key(|(_, e)| e.created_at.map(|t| t.elapsed()).unwrap_or(Duration::MAX))
        {
            self.entries.remove(&oldest_fp);
            self.stats.evicted += 1;
        }
    }

    /// Find a similar cached entry fingerprint
    fn find_similar_fp(&self, query: &str, model: &str) -> Option<u64> {
        let normalized = self.normalize_query(query);

        for (cached_query, fingerprint) in &self.query_index {
            if let Some(entry) = self.entries.get(fingerprint) {
                if entry.model == model && !entry.is_expired() {
                    let similarity = self.compute_similarity(&normalized, cached_query);
                    if similarity >= self.config.similarity_threshold {
                        return Some(*fingerprint);
                    }
                }
            }
        }

        None
    }

    /// Compute fingerprint for a query
    fn compute_fingerprint(&self, query: &str, model: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.normalize_query(query).hash(&mut hasher);
        model.hash(&mut hasher);
        hasher.finish()
    }

    /// Normalize a query for comparison
    fn normalize_query(&self, query: &str) -> String {
        query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute similarity between two normalized queries
    fn compute_similarity(&self, a: &str, b: &str) -> f32 {
        let words_a: std::collections::HashSet<_> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = b.split_whitespace().collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries (for export)
    pub fn entries(&self) -> &HashMap<u64, CachedResponse> {
        &self.entries
    }
}

impl Default for ResponseCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Cache key builder for more control over cache keys
#[derive(Debug, Clone)]
pub struct CacheKey {
    query: String,
    model: String,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    extra: HashMap<String, String>,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(query: &str, model: &str) -> Self {
        Self {
            query: query.to_string(),
            model: model.to_string(),
            system_prompt: None,
            temperature: None,
            extra: HashMap::new(),
        }
    }

    /// Include system prompt in key
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// Include temperature in key
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Add extra key component
    pub fn with_extra(mut self, key: &str, value: &str) -> Self {
        self.extra.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute the fingerprint
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        self.query.to_lowercase().hash(&mut hasher);
        self.model.hash(&mut hasher);

        if let Some(ref prompt) = self.system_prompt {
            prompt.hash(&mut hasher);
        }

        if let Some(temp) = self.temperature {
            ((temp * 100.0) as u32).hash(&mut hasher);
        }

        for (k, v) in &self.extra {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Semantic cache that uses embeddings for similarity
#[derive(Debug)]
pub struct SemanticCache {
    /// Base response cache
    cache: ResponseCache,
    /// Embeddings for cached queries
    embeddings: HashMap<u64, Vec<f32>>,
    /// Similarity threshold for semantic matching
    similarity_threshold: f32,
}

impl SemanticCache {
    /// Create a new semantic cache
    pub fn new(config: CacheConfig) -> Self {
        let threshold = config.similarity_threshold;
        Self {
            cache: ResponseCache::new(config),
            embeddings: HashMap::new(),
            similarity_threshold: threshold,
        }
    }

    /// Store with embedding
    pub fn put_with_embedding(
        &mut self,
        query: &str,
        model: &str,
        response: &str,
        tokens: usize,
        embedding: Vec<f32>,
        query_type: Option<&str>,
    ) {
        self.cache.put(query, model, response, tokens, query_type);
        let fingerprint = self.cache.compute_fingerprint(query, model);
        self.embeddings.insert(fingerprint, embedding);
    }

    /// Get with semantic matching
    pub fn get_semantic(
        &mut self,
        query: &str,
        model: &str,
        query_embedding: &[f32],
    ) -> Option<CachedResponse> {
        // Try exact match first
        if let Some(entry) = self.cache.get(query, model) {
            return Some(entry);
        }

        // Try semantic matching
        let mut best_match: Option<(u64, f32)> = None;

        for (fp, embedding) in &self.embeddings {
            let similarity = cosine_similarity(query_embedding, embedding);
            if similarity >= self.similarity_threshold {
                if let Some((_, best_sim)) = best_match {
                    if similarity > best_sim {
                        best_match = Some((*fp, similarity));
                    }
                } else {
                    best_match = Some((*fp, similarity));
                }
            }
        }

        if let Some((fp, _)) = best_match {
            if let Some(entry) = self.cache.entries.get_mut(&fp) {
                if !entry.is_expired() {
                    entry.record_hit();
                    self.cache.stats.hits += 1;
                    return Some(entry.clone());
                }
            }
        }

        self.cache.stats.misses += 1;
        None
    }

    /// Get base cache
    pub fn cache(&self) -> &ResponseCache {
        &self.cache
    }

    /// Get mutable base cache
    pub fn cache_mut(&mut self) -> &mut ResponseCache {
        &mut self.cache
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_caching() {
        let mut cache = ResponseCache::default();

        cache.put("What is Rust?", "llama-3", "Rust is a programming language...", 50, None);

        let result = cache.get("What is Rust?", "llama-3");
        assert!(result.is_some());
        assert!(result.unwrap().content.contains("Rust"));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ResponseCache::default();

        let result = cache.get("Unknown query", "llama-3");
        assert!(result.is_none());

        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_different_models() {
        let mut cache = ResponseCache::default();

        cache.put("Hello", "llama-3", "Response from Llama", 20, None);
        cache.put("Hello", "gpt-4", "Response from GPT", 20, None);

        let result1 = cache.get("Hello", "llama-3");
        let result2 = cache.get("Hello", "gpt-4");

        assert!(result1.is_some());
        assert!(result2.is_some());
        assert_ne!(result1.unwrap().content, result2.unwrap().content);
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut cache = ResponseCache::new(CacheConfig {
            fuzzy_matching: true,
            similarity_threshold: 0.7,
            ..Default::default()
        });

        cache.put("What is the Rust programming language", "llama-3", "Rust is...", 50, None);

        // Similar query should match
        let result = cache.get("What is Rust programming language", "llama-3");
        assert!(result.is_some());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ResponseCache::default();

        cache.put("Query 1", "model", "Response 1", 100, None);
        cache.get("Query 1", "model"); // hit
        cache.get("Query 2", "model"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_key_builder() {
        let key1 = CacheKey::new("query", "model")
            .with_temperature(0.7)
            .fingerprint();

        let key2 = CacheKey::new("query", "model")
            .with_temperature(0.8)
            .fingerprint();

        let key3 = CacheKey::new("query", "model")
            .with_temperature(0.7)
            .fingerprint();

        assert_ne!(key1, key2);
        assert_eq!(key1, key3);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = vec![0.707, 0.707, 0.0];
        let sim = cosine_similarity(&a, &d);
        assert!(sim > 0.7 && sim < 0.71);
    }
}
