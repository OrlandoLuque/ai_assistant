//! Request coalescing for combining similar AI requests
//!
//! This module provides request coalescing to reduce API calls by combining
//! similar or identical requests that arrive within a short time window.
//!
//! # Features
//!
//! - **Request deduplication**: Identical requests share a single API call
//! - **Similarity-based coalescing**: Similar requests can be batched
//! - **Time-window batching**: Requests within a window are grouped
//! - **Result broadcasting**: One result is shared among all waiters

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for request coalescing
#[derive(Debug, Clone)]
pub struct CoalescingConfig {
    /// Maximum time to wait for similar requests
    pub coalescing_window: Duration,
    /// Maximum requests to coalesce together
    pub max_batch_size: usize,
    /// Whether to use semantic similarity for coalescing
    pub use_semantic_matching: bool,
    /// Similarity threshold for semantic matching (0.0-1.0)
    pub similarity_threshold: f64,
    /// Cache coalesced results for reuse
    pub cache_results: bool,
    /// TTL for cached results
    pub cache_ttl: Duration,
}

impl Default for CoalescingConfig {
    fn default() -> Self {
        Self {
            coalescing_window: Duration::from_millis(100),
            max_batch_size: 10,
            use_semantic_matching: false,
            similarity_threshold: 0.95,
            cache_results: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// A request that can be coalesced
#[derive(Debug, Clone)]
pub struct CoalescableRequest {
    /// Unique request ID
    pub id: String,
    /// The prompt/query text
    pub prompt: String,
    /// Model to use
    pub model: String,
    /// Additional parameters hash
    pub params_hash: u64,
    /// When the request was submitted
    pub submitted_at: Instant,
    /// Optional semantic embedding for similarity matching
    pub embedding: Option<Vec<f32>>,
}

impl CoalescableRequest {
    /// Create a new coalescable request
    pub fn new(prompt: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: uuid_v4(),
            prompt: prompt.into(),
            model: model.into(),
            params_hash: 0,
            submitted_at: Instant::now(),
            embedding: None,
        }
    }

    /// Set parameters hash
    pub fn with_params_hash(mut self, hash: u64) -> Self {
        self.params_hash = hash;
        self
    }

    /// Set embedding for semantic matching
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Compute a coalescing key for exact matching
    pub fn coalescing_key(&self) -> CoalescingKey {
        CoalescingKey {
            prompt: self.prompt.clone(),
            model: self.model.clone(),
            params_hash: self.params_hash,
        }
    }
}

/// Key for exact-match coalescing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoalescingKey {
    prompt: String,
    model: String,
    params_hash: u64,
}

/// Result of a coalesced request
#[derive(Debug, Clone)]
pub struct CoalescedResult {
    /// The generated response
    pub response: String,
    /// IDs of requests that shared this result
    pub request_ids: Vec<String>,
    /// Whether this was from cache
    pub from_cache: bool,
    /// Time taken (if not cached)
    pub generation_time: Option<Duration>,
}

/// Pending request group waiting for coalescing
struct PendingGroup {
    requests: Vec<CoalescableRequest>,
    created_at: Instant,
    result_sender: Option<std::sync::mpsc::Sender<CoalescedResult>>,
}

/// Request coalescer
pub struct RequestCoalescer {
    config: CoalescingConfig,
    /// Pending requests grouped by key
    pending: Arc<RwLock<HashMap<CoalescingKey, PendingGroup>>>,
    /// Cached results
    cache: Arc<RwLock<HashMap<CoalescingKey, CachedResult>>>,
    /// Statistics
    stats: Arc<Mutex<CoalescingStats>>,
}

struct CachedResult {
    response: String,
    created_at: Instant,
}

/// Statistics for request coalescing
#[derive(Debug, Clone, Default)]
pub struct CoalescingStats {
    /// Total requests received
    pub total_requests: usize,
    /// Requests that were coalesced with others
    pub coalesced_requests: usize,
    /// Requests served from cache
    pub cache_hits: usize,
    /// Actual API calls made
    pub api_calls: usize,
    /// Average requests per batch
    pub avg_batch_size: f64,
    /// Total time saved by coalescing (estimated)
    pub time_saved: Duration,
}

impl std::fmt::Debug for RequestCoalescer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestCoalescer")
            .field("config", &self.config)
            .field("pending_groups", &self.pending_count())
            .finish()
    }
}

impl RequestCoalescer {
    /// Create a new request coalescer
    pub fn new(config: CoalescingConfig) -> Self {
        Self {
            config,
            pending: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CoalescingStats::default())),
        }
    }

    /// Submit a request for potential coalescing
    /// Returns immediately with a receiver for the result
    pub fn submit(&self, request: CoalescableRequest) -> CoalescingHandle {
        let key = request.coalescing_key();

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_requests += 1;
        }

        // Check cache first
        if self.config.cache_results {
            if let Some(cached) = self.get_cached(&key) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.cache_hits += 1;
                }
                return CoalescingHandle::immediate(CoalescedResult {
                    response: cached,
                    request_ids: vec![request.id.clone()],
                    from_cache: true,
                    generation_time: None,
                });
            }
        }

        // Add to pending group
        let (tx, rx) = std::sync::mpsc::channel();

        {
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            let group = pending.entry(key.clone()).or_insert_with(|| PendingGroup {
                requests: Vec::new(),
                created_at: Instant::now(),
                result_sender: None,
            });

            group.requests.push(request);

            // First request in group owns the sender
            if group.result_sender.is_none() {
                group.result_sender = Some(tx.clone());
            }
        }

        CoalescingHandle::pending(rx)
    }

    /// Process pending requests that have waited long enough
    pub fn process_pending<F>(&self, generate: F) -> Vec<CoalescedResult>
    where
        F: Fn(&str, &str) -> Result<String, String>,
    {
        let now = Instant::now();
        let mut ready_groups: Vec<(CoalescingKey, PendingGroup)> = Vec::new();

        // Find groups that are ready to process
        {
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            let keys_to_remove: Vec<_> = pending
                .iter()
                .filter(|(_, group)| {
                    now.duration_since(group.created_at) >= self.config.coalescing_window
                        || group.requests.len() >= self.config.max_batch_size
                })
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                if let Some(group) = pending.remove(&key) {
                    ready_groups.push((key, group));
                }
            }
        }

        // Process each group
        let mut results = Vec::new();
        for (key, group) in ready_groups {
            let start = Instant::now();

            // Get first request as representative
            if let Some(first) = group.requests.first() {
                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.api_calls += 1;
                    stats.coalesced_requests += group.requests.len().saturating_sub(1);

                    // Update average batch size
                    let total_batches = stats.api_calls as f64;
                    stats.avg_batch_size = (stats.avg_batch_size * (total_batches - 1.0)
                        + group.requests.len() as f64)
                        / total_batches;
                }

                // Generate response
                let response = match generate(&first.prompt, &first.model) {
                    Ok(r) => r,
                    Err(e) => format!("Error: {}", e),
                };

                let generation_time = start.elapsed();

                // Cache result
                if self.config.cache_results {
                    self.cache_result(&key, &response);
                }

                let result = CoalescedResult {
                    response: response.clone(),
                    request_ids: group.requests.iter().map(|r| r.id.clone()).collect(),
                    from_cache: false,
                    generation_time: Some(generation_time),
                };

                // Broadcast result to all waiters
                if let Some(sender) = group.result_sender {
                    let _ = sender.send(result.clone());
                }

                results.push(result);
            }
        }

        results
    }

    /// Check if there are pending requests
    pub fn has_pending(&self) -> bool {
        self.pending.read().map(|p| !p.is_empty()).unwrap_or(false)
    }

    /// Get count of pending request groups
    pub fn pending_count(&self) -> usize {
        self.pending.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Get statistics
    pub fn stats(&self) -> CoalescingStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Clean up expired cache entries
    pub fn cleanup_cache(&self) {
        let now = Instant::now();
        if let Ok(mut cache) = self.cache.write() {
            cache.retain(|_, v| now.duration_since(v.created_at) < self.config.cache_ttl);
        }
    }

    fn get_cached(&self, key: &CoalescingKey) -> Option<String> {
        let cache = self.cache.read().ok()?;
        let cached = cache.get(key)?;

        if Instant::now().duration_since(cached.created_at) < self.config.cache_ttl {
            Some(cached.response.clone())
        } else {
            None
        }
    }

    fn cache_result(&self, key: &CoalescingKey, response: &str) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(
                key.clone(),
                CachedResult {
                    response: response.to_string(),
                    created_at: Instant::now(),
                },
            );
        }
    }
}

impl Default for RequestCoalescer {
    fn default() -> Self {
        Self::new(CoalescingConfig::default())
    }
}

/// Handle for a coalescing request
#[derive(Debug)]
pub struct CoalescingHandle {
    receiver: Option<std::sync::mpsc::Receiver<CoalescedResult>>,
    immediate_result: Option<CoalescedResult>,
}

impl CoalescingHandle {
    fn immediate(result: CoalescedResult) -> Self {
        Self {
            receiver: None,
            immediate_result: Some(result),
        }
    }

    fn pending(receiver: std::sync::mpsc::Receiver<CoalescedResult>) -> Self {
        Self {
            receiver: Some(receiver),
            immediate_result: None,
        }
    }

    /// Wait for the result (blocking)
    pub fn wait(self) -> Option<CoalescedResult> {
        if let Some(result) = self.immediate_result {
            return Some(result);
        }

        self.receiver?.recv().ok()
    }

    /// Try to get result without blocking
    pub fn try_recv(&self) -> Option<CoalescedResult> {
        if let Some(ref result) = self.immediate_result {
            return Some(result.clone());
        }

        self.receiver.as_ref()?.try_recv().ok()
    }

    /// Check if result is immediately available
    pub fn is_ready(&self) -> bool {
        self.immediate_result.is_some()
    }
}

/// Semantic coalescer that groups requests by similarity
#[derive(Debug)]
pub struct SemanticCoalescer {
    config: CoalescingConfig,
    pending: Arc<RwLock<Vec<CoalescableRequest>>>,
    stats: Arc<Mutex<CoalescingStats>>,
}

impl SemanticCoalescer {
    /// Create a new semantic coalescer
    pub fn new(config: CoalescingConfig) -> Self {
        Self {
            config,
            pending: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(Mutex::new(CoalescingStats::default())),
        }
    }

    /// Submit a request with embedding
    pub fn submit(&self, request: CoalescableRequest) {
        if let Ok(mut pending) = self.pending.write() {
            if let Ok(mut stats) = self.stats.lock() {
                stats.total_requests += 1;
            }
            pending.push(request);
        }
    }

    /// Find similar requests and group them
    pub fn find_groups(&self) -> Vec<Vec<CoalescableRequest>> {
        let pending = match self.pending.read() {
            Ok(p) => p.clone(),
            Err(_) => return Vec::new(),
        };

        if pending.is_empty() {
            return Vec::new();
        }

        let mut groups: Vec<Vec<CoalescableRequest>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; pending.len()];

        for i in 0..pending.len() {
            if assigned[i] {
                continue;
            }

            let mut group = vec![pending[i].clone()];
            assigned[i] = true;

            // Find similar requests
            for j in (i + 1)..pending.len() {
                if assigned[j] {
                    continue;
                }

                if self.are_similar(&pending[i], &pending[j]) {
                    group.push(pending[j].clone());
                    assigned[j] = true;

                    if group.len() >= self.config.max_batch_size {
                        break;
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Process groups and clear pending
    pub fn process_and_clear<F>(&self, generate: F) -> Vec<CoalescedResult>
    where
        F: Fn(&[CoalescableRequest]) -> Result<String, String>,
    {
        let groups = self.find_groups();

        // Clear pending
        if let Ok(mut pending) = self.pending.write() {
            pending.clear();
        }

        let mut results = Vec::new();
        for group in groups {
            if let Ok(mut stats) = self.stats.lock() {
                stats.api_calls += 1;
                stats.coalesced_requests += group.len().saturating_sub(1);
            }

            let response = match generate(&group) {
                Ok(r) => r,
                Err(e) => format!("Error: {}", e),
            };

            results.push(CoalescedResult {
                response,
                request_ids: group.iter().map(|r| r.id.clone()).collect(),
                from_cache: false,
                generation_time: None,
            });
        }

        results
    }

    fn are_similar(&self, a: &CoalescableRequest, b: &CoalescableRequest) -> bool {
        // Must be same model
        if a.model != b.model {
            return false;
        }

        // Check semantic similarity if embeddings available
        if let (Some(ref emb_a), Some(ref emb_b)) = (&a.embedding, &b.embedding) {
            let similarity = cosine_similarity(emb_a, emb_b);
            return similarity >= self.config.similarity_threshold;
        }

        // Fallback to exact match
        a.prompt == b.prompt && a.params_hash == b.params_hash
    }

    /// Get statistics
    pub fn stats(&self) -> CoalescingStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)) as f64
}

/// Generate a simple UUID v4-like string
fn uuid_v4() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:016x}{:016x}", now.as_nanos(), rand_u64())
}

fn rand_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    std::time::Instant::now().hash(&mut hasher);
    hasher.finish()
}

/// Builder for creating coalescing configurations
#[derive(Debug)]
pub struct CoalescingConfigBuilder {
    config: CoalescingConfig,
}

impl CoalescingConfigBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: CoalescingConfig::default(),
        }
    }

    /// Set coalescing window
    pub fn coalescing_window(mut self, duration: Duration) -> Self {
        self.config.coalescing_window = duration;
        self
    }

    /// Set max batch size
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Enable semantic matching
    pub fn semantic_matching(mut self, threshold: f64) -> Self {
        self.config.use_semantic_matching = true;
        self.config.similarity_threshold = threshold;
        self
    }

    /// Configure caching
    pub fn cache(mut self, enabled: bool, ttl: Duration) -> Self {
        self.config.cache_results = enabled;
        self.config.cache_ttl = ttl;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CoalescingConfig {
        self.config
    }
}

impl Default for CoalescingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_coalescing_key() {
        let req1 = CoalescableRequest::new("Hello", "gpt-4");
        let req2 = CoalescableRequest::new("Hello", "gpt-4");
        let req3 = CoalescableRequest::new("Hello", "gpt-3.5");

        assert_eq!(req1.coalescing_key(), req2.coalescing_key());
        assert_ne!(req1.coalescing_key(), req3.coalescing_key());
    }

    #[test]
    fn test_coalescer_submission() {
        let coalescer = RequestCoalescer::default();

        let req1 = CoalescableRequest::new("Test prompt", "model");
        let _handle = coalescer.submit(req1);

        assert_eq!(coalescer.pending_count(), 1);

        let req2 = CoalescableRequest::new("Test prompt", "model");
        let _handle = coalescer.submit(req2);

        // Same key, should be in same group
        assert_eq!(coalescer.pending_count(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_cache_hit() {
        let config = CoalescingConfig {
            cache_results: true,
            cache_ttl: Duration::from_secs(60),
            coalescing_window: Duration::from_millis(0), // Process immediately
            ..Default::default()
        };
        let coalescer = RequestCoalescer::new(config);

        // First request - should be pending
        let req1 = CoalescableRequest::new("Test", "model");
        let handle1 = coalescer.submit(req1);
        assert!(!handle1.is_ready());

        // Process it (needs a short sleep to ensure window elapsed)
        std::thread::sleep(Duration::from_millis(1));
        coalescer.process_pending(|_, _| Ok("Response".to_string()));

        // Second identical request - should hit cache
        let req2 = CoalescableRequest::new("Test", "model");
        let handle2 = coalescer.submit(req2);
        assert!(handle2.is_ready());
    }

    #[test]
    fn test_stats() {
        let coalescer = RequestCoalescer::default();

        for _ in 0..5 {
            let req = CoalescableRequest::new("Same prompt", "model");
            let _ = coalescer.submit(req);
        }

        let stats = coalescer.stats();
        assert_eq!(stats.total_requests, 5);
    }

    #[test]
    fn test_config_builder() {
        let config = CoalescingConfigBuilder::new()
            .coalescing_window(Duration::from_millis(200))
            .max_batch_size(20)
            .semantic_matching(0.9)
            .cache(true, Duration::from_secs(600))
            .build();

        assert_eq!(config.coalescing_window, Duration::from_millis(200));
        assert_eq!(config.max_batch_size, 20);
        assert!(config.use_semantic_matching);
        assert_eq!(config.similarity_threshold, 0.9);
    }
}
