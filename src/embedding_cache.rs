//! Embedding Cache - Caching for embedding vectors
//!
//! This module provides caching functionality for embedding vectors
//! to avoid redundant computation of embeddings.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Time-to-live for entries
    pub ttl: Duration,
    /// Enable LRU eviction
    pub enable_lru: bool,
    /// Persist to disk
    pub persist: bool,
    /// Cache file path (if persist is true)
    pub cache_path: Option<String>,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl: Duration::from_secs(3600 * 24), // 24 hours
            enable_lru: true,
            persist: false,
            cache_path: None,
        }
    }
}

/// A cached embedding entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The embedding vector
    embedding: Vec<f32>,
    /// When this entry was created
    created_at: Instant,
    /// Last access time (for LRU)
    last_accessed: Instant,
    /// Access count
    access_count: usize,
    /// Model used to generate this embedding
    model: String,
}

impl CacheEntry {
    fn new(embedding: Vec<f32>, model: String) -> Self {
        let now = Instant::now();
        Self {
            embedding,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            model,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Statistics for the embedding cache
#[derive(Debug, Clone, Default)]
pub struct EmbeddingCacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of entries
    pub entries: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Total memory used (estimated bytes)
    pub memory_bytes: usize,
}

impl EmbeddingCacheStats {
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

/// In-memory embedding cache
pub struct EmbeddingCache {
    /// Configuration
    config: EmbeddingCacheConfig,
    /// Cache entries (key is hash of text + model)
    entries: HashMap<u64, CacheEntry>,
    /// Statistics
    stats: EmbeddingCacheStats,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: EmbeddingCacheStats::default(),
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(EmbeddingCacheConfig::default())
    }

    /// Compute cache key for text + model
    fn compute_key(text: &str, model: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        model.hash(&mut hasher);
        hasher.finish()
    }

    /// Get an embedding from cache
    pub fn get(&mut self, text: &str, model: &str) -> Option<Vec<f32>> {
        let key = Self::compute_key(text, model);

        if let Some(entry) = self.entries.get_mut(&key) {
            // Check if expired
            if entry.is_expired(self.config.ttl) {
                self.entries.remove(&key);
                self.stats.misses += 1;
                self.stats.entries = self.entries.len();
                return None;
            }

            // Update LRU info
            entry.touch();
            self.stats.hits += 1;
            Some(entry.embedding.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store an embedding in cache
    pub fn set(&mut self, text: &str, model: &str, embedding: Vec<f32>) {
        let key = Self::compute_key(text, model);

        // Check if we need to evict
        if self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }

        let entry = CacheEntry::new(embedding, model.to_string());
        self.stats.memory_bytes += entry.embedding.len() * std::mem::size_of::<f32>();
        self.entries.insert(key, entry);
        self.stats.entries = self.entries.len();
    }

    /// Evict one entry based on LRU
    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let key_to_remove = if self.config.enable_lru {
            // Find least recently used
            self.entries.iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(k, _)| *k)
        } else {
            // Remove first entry
            self.entries.keys().next().copied()
        };

        if let Some(key) = key_to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.stats.memory_bytes = self.stats.memory_bytes
                    .saturating_sub(entry.embedding.len() * std::mem::size_of::<f32>());
                self.stats.evictions += 1;
            }
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.entries = 0;
        self.stats.memory_bytes = 0;
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) -> usize {
        let ttl = self.config.ttl;
        let before_len = self.entries.len();

        self.entries.retain(|_, entry| !entry.is_expired(ttl));

        let removed = before_len - self.entries.len();
        self.stats.entries = self.entries.len();
        self.stats.evictions += removed;

        // Recalculate memory
        self.stats.memory_bytes = self.entries.values()
            .map(|e| e.embedding.len() * std::mem::size_of::<f32>())
            .sum();

        removed
    }

    /// Get statistics
    pub fn stats(&self) -> &EmbeddingCacheStats {
        &self.stats
    }

    /// Check if text is cached
    pub fn contains(&self, text: &str, model: &str) -> bool {
        let key = Self::compute_key(text, model);
        if let Some(entry) = self.entries.get(&key) {
            !entry.is_expired(self.config.ttl)
        } else {
            false
        }
    }

    /// Get all cached texts (for debugging)
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Thread-safe embedding cache
pub struct SharedEmbeddingCache {
    inner: Arc<RwLock<EmbeddingCache>>,
}

impl SharedEmbeddingCache {
    /// Create a new shared cache
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(EmbeddingCache::new(config))),
        }
    }

    /// Create with defaults
    pub fn with_defaults() -> Self {
        Self::new(EmbeddingCacheConfig::default())
    }

    /// Get an embedding
    pub fn get(&self, text: &str, model: &str) -> Option<Vec<f32>> {
        self.inner.write().ok()?.get(text, model)
    }

    /// Set an embedding
    pub fn set(&self, text: &str, model: &str, embedding: Vec<f32>) {
        if let Ok(mut cache) = self.inner.write() {
            cache.set(text, model, embedding);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.inner.write() {
            cache.clear();
        }
    }

    /// Cleanup expired entries
    pub fn cleanup_expired(&self) -> usize {
        self.inner.write().ok().map(|mut c| c.cleanup_expired()).unwrap_or(0)
    }

    /// Get statistics
    pub fn stats(&self) -> Option<EmbeddingCacheStats> {
        self.inner.read().ok().map(|c| c.stats().clone())
    }

    /// Check if cached
    pub fn contains(&self, text: &str, model: &str) -> bool {
        self.inner.read().ok().map(|c| c.contains(text, model)).unwrap_or(false)
    }

    /// Clone the Arc
    pub fn clone_ref(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Clone for SharedEmbeddingCache {
    fn clone(&self) -> Self {
        self.clone_ref()
    }
}

/// Batch embedding cache for efficient batch operations
pub struct BatchEmbeddingCache {
    cache: SharedEmbeddingCache,
    pending: Vec<(String, String)>, // (text, model)
}

impl BatchEmbeddingCache {
    /// Create a new batch cache
    pub fn new(cache: SharedEmbeddingCache) -> Self {
        Self {
            cache,
            pending: Vec::new(),
        }
    }

    /// Queue a text for embedding (returns cached if available)
    pub fn queue(&mut self, text: &str, model: &str) -> Option<Vec<f32>> {
        if let Some(embedding) = self.cache.get(text, model) {
            Some(embedding)
        } else {
            self.pending.push((text.to_string(), model.to_string()));
            None
        }
    }

    /// Get pending texts that need embedding
    pub fn get_pending(&self) -> Vec<(String, String)> {
        self.pending.clone()
    }

    /// Store batch results
    pub fn store_batch(&mut self, results: Vec<(String, String, Vec<f32>)>) {
        for (text, model, embedding) in results {
            self.cache.set(&text, &model, embedding);
        }
        self.pending.clear();
    }

    /// Clear pending
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Number of pending texts
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Embedding cache with similarity search
pub struct SimilarityEmbeddingCache {
    /// Base cache
    cache: EmbeddingCache,
    /// Store text alongside hash for similarity search
    text_index: HashMap<u64, String>,
    /// Similarity threshold
    similarity_threshold: f32,
}

impl SimilarityEmbeddingCache {
    /// Create a new similarity cache
    pub fn new(config: EmbeddingCacheConfig, similarity_threshold: f32) -> Self {
        Self {
            cache: EmbeddingCache::new(config),
            text_index: HashMap::new(),
            similarity_threshold,
        }
    }

    /// Get embedding, or find similar cached embedding
    pub fn get_or_similar(&mut self, text: &str, model: &str) -> Option<(Vec<f32>, bool)> {
        // Try exact match first
        if let Some(embedding) = self.cache.get(text, model) {
            return Some((embedding, true)); // exact match
        }

        // No exact match - would need to compute embedding to find similar
        // Return None to indicate computation needed
        None
    }

    /// Store embedding with text index
    pub fn set(&mut self, text: &str, model: &str, embedding: Vec<f32>) {
        let key = EmbeddingCache::compute_key(text, model);
        self.text_index.insert(key, text.to_string());
        self.cache.set(text, model, embedding);
    }

    /// Find similar cached embeddings
    pub fn find_similar(&self, embedding: &[f32], model: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = Vec::new();

        for (key, entry) in &self.cache.entries {
            if entry.model != model {
                continue;
            }

            let similarity = cosine_similarity(embedding, &entry.embedding);
            if similarity >= self.similarity_threshold {
                if let Some(text) = self.text_index.get(key) {
                    similarities.push((text.clone(), similarity));
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Get stats
    pub fn stats(&self) -> &EmbeddingCacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.text_index.clear();
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Normalize a vector to unit length
pub fn normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cache_basic() {
        let mut cache = EmbeddingCache::with_defaults();

        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        cache.set("hello world", "test-model", embedding.clone());

        let retrieved = cache.get("hello world", "test-model");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = EmbeddingCache::with_defaults();

        let result = cache.get("not cached", "test-model");
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_different_models() {
        let mut cache = EmbeddingCache::with_defaults();

        let embedding1 = vec![0.1, 0.2, 0.3];
        let embedding2 = vec![0.4, 0.5, 0.6];

        cache.set("text", "model1", embedding1.clone());
        cache.set("text", "model2", embedding2.clone());

        assert_eq!(cache.get("text", "model1").unwrap(), embedding1);
        assert_eq!(cache.get("text", "model2").unwrap(), embedding2);
    }

    #[test]
    fn test_cache_eviction() {
        let config = EmbeddingCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = EmbeddingCache::new(config);

        cache.set("text1", "model", vec![0.1]);
        cache.set("text2", "model", vec![0.2]);
        cache.set("text3", "model", vec![0.3]); // Should evict one

        assert_eq!(cache.entry_count(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = EmbeddingCache::with_defaults();

        cache.set("text", "model", vec![0.1, 0.2, 0.3]);
        cache.get("text", "model"); // hit
        cache.get("other", "model"); // miss

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_shared_cache() {
        let cache = SharedEmbeddingCache::with_defaults();

        cache.set("text", "model", vec![0.1, 0.2]);
        let result = cache.get("text", "model");

        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![0.1, 0.2]);
    }

    #[test]
    fn test_batch_cache() {
        let shared = SharedEmbeddingCache::with_defaults();
        shared.set("cached", "model", vec![0.1]);

        let mut batch = BatchEmbeddingCache::new(shared);

        // This should return cached
        let result = batch.queue("cached", "model");
        assert!(result.is_some());

        // This should add to pending
        let result = batch.queue("not_cached", "model");
        assert!(result.is_none());

        assert_eq!(batch.pending_count(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);

        // Should be unit length
        let length: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_cache() {
        let mut cache = SimilarityEmbeddingCache::new(
            EmbeddingCacheConfig::default(),
            0.9,
        );

        cache.set("hello", "model", vec![1.0, 0.0, 0.0]);
        cache.set("world", "model", vec![0.0, 1.0, 0.0]);

        let similar = cache.find_similar(&[0.9, 0.1, 0.0], "model", 5);
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0, "hello");
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = EmbeddingCache::with_defaults();

        assert!(!cache.contains("text", "model"));
        cache.set("text", "model", vec![0.1]);
        assert!(cache.contains("text", "model"));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = EmbeddingCache::with_defaults();

        cache.set("text1", "model", vec![0.1]);
        cache.set("text2", "model", vec![0.2]);

        assert_eq!(cache.entry_count(), 2);
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }
}
