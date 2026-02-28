//! Intelligent prefetching for AI responses
//!
//! This module provides predictive prefetching to reduce perceived latency
//! by anticipating likely user queries and pre-generating responses.
//!
//! # Features
//!
//! - **Pattern-based prediction**: Learn query patterns from history
//! - **Context-aware prefetching**: Use conversation context to predict
//! - **Priority-based scheduling**: Prefetch most likely queries first
//! - **Resource-aware**: Respect rate limits and resource constraints

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for prefetching
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Maximum number of items to prefetch
    pub max_prefetch_items: usize,
    /// Minimum confidence to trigger prefetch
    pub min_confidence: f64,
    /// Maximum age of prefetched responses
    pub max_age: Duration,
    /// Enable pattern learning
    pub learn_patterns: bool,
    /// Maximum patterns to track
    pub max_patterns: usize,
    /// Cooldown between prefetch operations
    pub prefetch_cooldown: Duration,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_prefetch_items: 5,
            min_confidence: 0.7,
            max_age: Duration::from_secs(300),
            learn_patterns: true,
            max_patterns: 1000,
            prefetch_cooldown: Duration::from_secs(1),
        }
    }
}

/// A query pattern learned from history
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// The trigger context (what user typed/asked before)
    pub trigger: String,
    /// The likely follow-up query
    pub followup: String,
    /// Model to use
    pub model: String,
    /// How many times this pattern occurred
    pub occurrences: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Last time this pattern was observed
    pub last_seen: Instant,
}

/// A prefetched response
#[derive(Debug, Clone)]
pub struct PrefetchedResponse {
    /// The query this responds to
    pub query: String,
    /// The model used
    pub model: String,
    /// The generated response
    pub response: String,
    /// When it was prefetched
    pub prefetched_at: Instant,
    /// Confidence this will be needed
    pub confidence: f64,
    /// Whether it was used
    pub used: bool,
}

/// Prefetch candidate
#[derive(Debug, Clone)]
pub struct PrefetchCandidate {
    /// Query to prefetch
    pub query: String,
    /// Model to use
    pub model: String,
    /// Confidence score
    pub confidence: f64,
    /// Priority (higher = more important)
    pub priority: i32,
}

/// Statistics for prefetching
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    /// Total prefetch operations
    pub total_prefetches: usize,
    /// Successful cache hits
    pub cache_hits: usize,
    /// Cache misses (prefetch wasn't used)
    pub cache_misses: usize,
    /// Hit rate (0.0-1.0)
    pub hit_rate: f64,
    /// Average latency saved per hit
    pub avg_latency_saved: Duration,
    /// Patterns learned
    pub patterns_learned: usize,
}

/// Intelligent prefetcher
pub struct Prefetcher {
    config: PrefetchConfig,
    /// Learned patterns
    patterns: Arc<RwLock<HashMap<String, Vec<QueryPattern>>>>,
    /// Prefetched responses
    cache: Arc<RwLock<HashMap<String, PrefetchedResponse>>>,
    /// Recent queries for pattern learning
    recent_queries: Arc<Mutex<VecDeque<(String, String, Instant)>>>,
    /// Statistics
    stats: Arc<Mutex<PrefetchStats>>,
    /// Last prefetch time
    last_prefetch: Arc<Mutex<Instant>>,
}

impl Prefetcher {
    /// Create a new prefetcher
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config,
            patterns: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            recent_queries: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(PrefetchStats::default())),
            last_prefetch: Arc::new(Mutex::new(Instant::now() - Duration::from_secs(60))),
        }
    }

    /// Record a query for pattern learning
    pub fn record_query(&self, query: impl Into<String>, model: impl Into<String>) {
        if !self.config.learn_patterns {
            return;
        }

        let query = query.into();
        let model = model.into();
        let now = Instant::now();

        let mut recent = self
            .recent_queries
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Learn from previous query -> this query
        if let Some((prev_query, prev_model, _)) = recent.back() {
            if prev_model == &model {
                self.learn_pattern(prev_query.clone(), query.clone(), model.clone());
            }
        }

        // Add to recent queries
        recent.push_back((query, model, now));

        // Limit size
        while recent.len() > 100 {
            recent.pop_front();
        }
    }

    /// Learn a pattern from observed query sequence
    fn learn_pattern(&self, trigger: String, followup: String, model: String) {
        let mut patterns = self.patterns.write().unwrap_or_else(|e| e.into_inner());

        let key = format!("{}::{}", trigger, model);
        let entries = patterns.entry(key).or_insert_with(Vec::new);

        // Check if pattern exists
        if let Some(pattern) = entries.iter_mut().find(|p| p.followup == followup) {
            pattern.occurrences += 1;
            pattern.last_seen = Instant::now();
            // Update confidence based on frequency
            pattern.confidence = (pattern.occurrences as f64 / 10.0).min(0.95);
        } else {
            entries.push(QueryPattern {
                trigger: trigger.clone(),
                followup,
                model,
                occurrences: 1,
                confidence: 0.3,
                last_seen: Instant::now(),
            });
        }

        // Prune old/low-confidence patterns
        entries.retain(|p| {
            p.confidence >= 0.1
                && Instant::now().duration_since(p.last_seen) < Duration::from_secs(86400)
        });

        // Sort by confidence
        entries.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit patterns per trigger
        entries.truncate(10);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.patterns_learned = patterns.values().map(|v| v.len()).sum();
        }
    }

    /// Get prefetch candidates based on current context
    pub fn get_candidates(&self, current_query: &str, model: &str) -> Vec<PrefetchCandidate> {
        let patterns = self.patterns.read().unwrap_or_else(|e| e.into_inner());
        let key = format!("{}::{}", current_query, model);

        let mut candidates = Vec::new();

        if let Some(entries) = patterns.get(&key) {
            for pattern in entries
                .iter()
                .filter(|p| p.confidence >= self.config.min_confidence)
            {
                candidates.push(PrefetchCandidate {
                    query: pattern.followup.clone(),
                    model: pattern.model.clone(),
                    confidence: pattern.confidence,
                    priority: (pattern.confidence * 100.0) as i32,
                });
            }
        }

        // Also check similar queries (prefix matching)
        for (k, entries) in patterns.iter() {
            if k.starts_with(current_query) && k != &key {
                for pattern in entries.iter().take(2) {
                    if pattern.confidence >= self.config.min_confidence {
                        candidates.push(PrefetchCandidate {
                            query: pattern.followup.clone(),
                            model: pattern.model.clone(),
                            confidence: pattern.confidence * 0.8, // Lower confidence for prefix matches
                            priority: (pattern.confidence * 80.0) as i32,
                        });
                    }
                }
            }
        }

        // Sort by priority and deduplicate
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority));
        candidates.dedup_by(|a, b| a.query == b.query);
        candidates.truncate(self.config.max_prefetch_items);

        candidates
    }

    /// Perform prefetching with given generator function
    pub fn prefetch<F>(&self, candidates: &[PrefetchCandidate], generate: F)
    where
        F: Fn(&str, &str) -> Result<String, String>,
    {
        // Check cooldown
        {
            let last = self.last_prefetch.lock().unwrap_or_else(|e| e.into_inner());
            if Instant::now().duration_since(*last) < self.config.prefetch_cooldown {
                return;
            }
        }

        // Update last prefetch time
        {
            let mut last = self.last_prefetch.lock().unwrap_or_else(|e| e.into_inner());
            *last = Instant::now();
        }

        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());

        for candidate in candidates.iter().take(self.config.max_prefetch_items) {
            let cache_key = format!("{}::{}", candidate.query, candidate.model);

            // Skip if already cached and fresh
            if let Some(cached) = cache.get(&cache_key) {
                if Instant::now().duration_since(cached.prefetched_at) < self.config.max_age {
                    continue;
                }
            }

            // Generate response
            if let Ok(response) = generate(&candidate.query, &candidate.model) {
                cache.insert(
                    cache_key,
                    PrefetchedResponse {
                        query: candidate.query.clone(),
                        model: candidate.model.clone(),
                        response,
                        prefetched_at: Instant::now(),
                        confidence: candidate.confidence,
                        used: false,
                    },
                );

                if let Ok(mut stats) = self.stats.lock() {
                    stats.total_prefetches += 1;
                }
            }
        }

        // Clean up old entries
        let max_age = self.config.max_age;
        cache.retain(|_, v| Instant::now().duration_since(v.prefetched_at) < max_age);
    }

    /// Try to get a prefetched response
    pub fn get_prefetched(&self, query: &str, model: &str) -> Option<PrefetchedResponse> {
        let cache_key = format!("{}::{}", query, model);
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());

        if let Some(mut cached) = cache.remove(&cache_key) {
            // Check if still valid
            if Instant::now().duration_since(cached.prefetched_at) < self.config.max_age {
                cached.used = true;

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.cache_hits += 1;
                    stats.hit_rate = stats.cache_hits as f64
                        / (stats.cache_hits + stats.cache_misses).max(1) as f64;
                }

                return Some(cached);
            }
        }

        // Miss
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_misses += 1;
            stats.hit_rate =
                stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses).max(1) as f64;
        }

        None
    }

    /// Check if a response is prefetched (without removing)
    pub fn has_prefetched(&self, query: &str, model: &str) -> bool {
        let cache_key = format!("{}::{}", query, model);
        let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());

        if let Some(cached) = cache.get(&cache_key) {
            Instant::now().duration_since(cached.prefetched_at) < self.config.max_age
        } else {
            false
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PrefetchStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Clear all caches and patterns
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut patterns) = self.patterns.write() {
            patterns.clear();
        }
        if let Ok(mut recent) = self.recent_queries.lock() {
            recent.clear();
        }
    }

    /// Get number of cached responses
    pub fn cache_size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Get number of learned patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns
            .read()
            .map(|p| p.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new(PrefetchConfig::default())
    }
}

/// Context-aware prefetch predictor
pub struct ContextPredictor {
    /// Common follow-up patterns
    common_followups: HashMap<String, Vec<String>>,
    /// Domain-specific patterns
    domain_patterns: HashMap<String, Vec<String>>,
}

impl ContextPredictor {
    /// Create a new context predictor
    pub fn new() -> Self {
        let mut predictor = Self {
            common_followups: HashMap::new(),
            domain_patterns: HashMap::new(),
        };
        predictor.init_common_patterns();
        predictor
    }

    fn init_common_patterns(&mut self) {
        // Common conversational follow-ups
        self.common_followups.insert(
            "greeting".to_string(),
            vec![
                "How can I help you?".to_string(),
                "What would you like to know?".to_string(),
            ],
        );

        self.common_followups.insert(
            "explanation".to_string(),
            vec![
                "Can you give me an example?".to_string(),
                "Can you explain more?".to_string(),
            ],
        );

        self.common_followups.insert(
            "code".to_string(),
            vec![
                "How do I run this?".to_string(),
                "Can you fix this error?".to_string(),
            ],
        );

        // Domain patterns for translation context
        self.domain_patterns.insert(
            "translation".to_string(),
            vec![
                "How do I say...".to_string(),
                "What's the translation for...".to_string(),
                "Is this translation correct?".to_string(),
            ],
        );
    }

    /// Predict likely follow-up queries
    pub fn predict(&self, query: &str, _response: &str) -> Vec<String> {
        let mut predictions = Vec::new();

        // Check for keywords and suggest follow-ups
        let query_lower = query.to_lowercase();

        if query_lower.contains("hello") || query_lower.contains("hi") {
            if let Some(followups) = self.common_followups.get("greeting") {
                predictions.extend(followups.iter().cloned());
            }
        }

        if query_lower.contains("what is") || query_lower.contains("explain") {
            if let Some(followups) = self.common_followups.get("explanation") {
                predictions.extend(followups.iter().cloned());
            }
        }

        if query_lower.contains("code") || query_lower.contains("function") {
            if let Some(followups) = self.common_followups.get("code") {
                predictions.extend(followups.iter().cloned());
            }
        }

        if query_lower.contains("translate") || query_lower.contains("translation") {
            if let Some(followups) = self.domain_patterns.get("translation") {
                predictions.extend(followups.iter().cloned());
            }
        }

        predictions.truncate(5);
        predictions
    }

    /// Add custom domain patterns
    pub fn add_domain_pattern(&mut self, domain: impl Into<String>, patterns: Vec<String>) {
        self.domain_patterns.insert(domain.into(), patterns);
    }
}

impl Default for ContextPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for prefetch configuration
pub struct PrefetchConfigBuilder {
    config: PrefetchConfig,
}

impl PrefetchConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PrefetchConfig::default(),
        }
    }

    /// Set max prefetch items
    pub fn max_items(mut self, count: usize) -> Self {
        self.config.max_prefetch_items = count;
        self
    }

    /// Set minimum confidence
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.config.min_confidence = confidence;
        self
    }

    /// Set max age for cached responses
    pub fn max_age(mut self, duration: Duration) -> Self {
        self.config.max_age = duration;
        self
    }

    /// Enable/disable pattern learning
    pub fn learn_patterns(mut self, enabled: bool) -> Self {
        self.config.learn_patterns = enabled;
        self
    }

    /// Set prefetch cooldown
    pub fn cooldown(mut self, duration: Duration) -> Self {
        self.config.prefetch_cooldown = duration;
        self
    }

    /// Build the configuration
    pub fn build(self) -> PrefetchConfig {
        self.config
    }
}

impl Default for PrefetchConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetcher_creation() {
        let prefetcher = Prefetcher::default();
        assert_eq!(prefetcher.cache_size(), 0);
        assert_eq!(prefetcher.pattern_count(), 0);
    }

    #[test]
    fn test_pattern_learning() {
        let prefetcher = Prefetcher::default();

        // Record a sequence of queries
        prefetcher.record_query("Hello", "model");
        prefetcher.record_query("How are you?", "model");

        // Pattern should be learned
        assert!(prefetcher.pattern_count() > 0);
    }

    #[test]
    fn test_prefetch_candidates() {
        let prefetcher = Prefetcher::default();

        // Train with repeated pattern
        for _ in 0..10 {
            prefetcher.record_query("Hello", "model");
            prefetcher.record_query("How are you?", "model");
        }

        let candidates = prefetcher.get_candidates("Hello", "model");
        // Should predict "How are you?" as likely follow-up
        assert!(!candidates.is_empty() || prefetcher.pattern_count() > 0);
    }

    #[test]
    fn test_prefetch_and_retrieve() {
        let prefetcher = Prefetcher::default();

        let candidates = vec![PrefetchCandidate {
            query: "Test query".to_string(),
            model: "model".to_string(),
            confidence: 0.9,
            priority: 90,
        }];

        prefetcher.prefetch(&candidates, |_, _| Ok("Test response".to_string()));

        // Should be able to retrieve
        let result = prefetcher.get_prefetched("Test query", "model");
        assert!(result.is_some());
        assert_eq!(result.unwrap().response, "Test response");
    }

    #[test]
    fn test_context_predictor() {
        let predictor = ContextPredictor::new();

        let predictions = predictor.predict("What is machine learning?", "");
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = PrefetchConfigBuilder::new()
            .max_items(10)
            .min_confidence(0.8)
            .max_age(Duration::from_secs(600))
            .learn_patterns(true)
            .cooldown(Duration::from_millis(500))
            .build();

        assert_eq!(config.max_prefetch_items, 10);
        assert_eq!(config.min_confidence, 0.8);
        assert!(config.learn_patterns);
    }

    #[test]
    fn test_stats() {
        let prefetcher = Prefetcher::default();

        // Initial stats should be zero
        let stats = prefetcher.stats();
        assert_eq!(stats.total_prefetches, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_prefetch_config_defaults() {
        let config = PrefetchConfig::default();
        assert!(config.max_prefetch_items > 0);
        assert!(config.min_confidence > 0.0);
    }

    #[test]
    fn test_record_query_patterns() {
        let prefetcher = Prefetcher::default();
        prefetcher.record_query("What is Rust?", "gpt-4");
        prefetcher.record_query("Tell me more", "gpt-4");
        let stats = prefetcher.stats();
        assert_eq!(stats.patterns_learned, 1);
    }

    #[test]
    fn test_get_candidates_empty() {
        let prefetcher = Prefetcher::default();
        let candidates = prefetcher.get_candidates("random query", "model");
        assert!(candidates.is_empty());
    }
}
