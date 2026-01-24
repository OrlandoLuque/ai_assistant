//! Conversation metrics and quality tracking
//!
//! This module provides tools to measure and analyze conversation quality,
//! including response times, token usage, context efficiency, and RAG retrieval quality.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Metrics for a single message exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetrics {
    /// Timestamp when the message was sent
    pub timestamp: String,
    /// Time to first token (streaming latency)
    pub time_to_first_token_ms: Option<u64>,
    /// Total response generation time
    pub total_response_time_ms: u64,
    /// Tokens in the user message
    pub input_tokens: usize,
    /// Tokens in the assistant response
    pub output_tokens: usize,
    /// Total context tokens used for this message
    pub context_tokens: usize,
    /// Tokens from knowledge RAG
    pub knowledge_tokens: usize,
    /// Tokens from conversation RAG
    pub conversation_tokens: usize,
    /// Number of knowledge chunks retrieved
    pub knowledge_chunks_retrieved: usize,
    /// Number of conversation messages retrieved
    pub conversation_messages_retrieved: usize,
    /// Whether context was near limit (>80%)
    pub context_near_limit: bool,
    /// Model used for this response
    pub model: String,
}

impl Default for MessageMetrics {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            time_to_first_token_ms: None,
            total_response_time_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
            context_tokens: 0,
            knowledge_tokens: 0,
            conversation_tokens: 0,
            knowledge_chunks_retrieved: 0,
            conversation_messages_retrieved: 0,
            context_near_limit: false,
            model: String::new(),
        }
    }
}

/// Aggregated session metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    /// Session identifier
    pub session_id: String,
    /// Total messages in session
    pub message_count: usize,
    /// Total input tokens
    pub total_input_tokens: usize,
    /// Total output tokens
    pub total_output_tokens: usize,
    /// Average response time in ms
    pub avg_response_time_ms: f64,
    /// Average time to first token in ms
    pub avg_time_to_first_token_ms: f64,
    /// Total knowledge chunks retrieved
    pub total_knowledge_chunks: usize,
    /// Average context usage percentage
    pub avg_context_usage_percent: f64,
    /// Number of times context was near limit
    pub context_limit_warnings: usize,
    /// Session duration in seconds
    pub session_duration_secs: u64,
}

/// RAG retrieval quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagQualityMetrics {
    /// Total queries made
    pub total_queries: usize,
    /// Queries that returned results
    pub queries_with_results: usize,
    /// Average chunks per query
    pub avg_chunks_per_query: f64,
    /// Average tokens per query
    pub avg_tokens_per_query: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Most frequently accessed sources
    pub top_sources: Vec<(String, usize)>,
}

/// Live metrics tracker
pub struct MetricsTracker {
    /// Metrics for each message in current session
    message_metrics: Vec<MessageMetrics>,
    /// Current message being tracked
    current_message: Option<MessageMetricsBuilder>,
    /// Session start time
    session_start: Instant,
    /// Session ID
    session_id: String,
    /// RAG query cache stats
    cache_hits: usize,
    cache_misses: usize,
    /// Source access counts
    source_access_counts: HashMap<String, usize>,
}

/// Builder for tracking a single message's metrics
pub struct MessageMetricsBuilder {
    start_time: Instant,
    first_token_time: Option<Instant>,
    input_tokens: usize,
    model: String,
    knowledge_tokens: usize,
    conversation_tokens: usize,
    knowledge_chunks: usize,
    conversation_messages: usize,
    context_tokens: usize,
    context_near_limit: bool,
}

impl MessageMetricsBuilder {
    pub fn new(model: &str) -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            input_tokens: 0,
            model: model.to_string(),
            knowledge_tokens: 0,
            conversation_tokens: 0,
            knowledge_chunks: 0,
            conversation_messages: 0,
            context_tokens: 0,
            context_near_limit: false,
        }
    }

    pub fn set_input_tokens(&mut self, tokens: usize) {
        self.input_tokens = tokens;
    }

    pub fn set_context_info(&mut self, total: usize, knowledge: usize, conversation: usize, near_limit: bool) {
        self.context_tokens = total;
        self.knowledge_tokens = knowledge;
        self.conversation_tokens = conversation;
        self.context_near_limit = near_limit;
    }

    pub fn set_rag_info(&mut self, knowledge_chunks: usize, conversation_messages: usize) {
        self.knowledge_chunks = knowledge_chunks;
        self.conversation_messages = conversation_messages;
    }

    pub fn mark_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
    }

    pub fn finish(self, output_tokens: usize) -> MessageMetrics {
        let total_time = self.start_time.elapsed();
        let ttft = self.first_token_time.map(|t| {
            t.duration_since(self.start_time).as_millis() as u64
        });

        MessageMetrics {
            timestamp: chrono::Utc::now().to_rfc3339(),
            time_to_first_token_ms: ttft,
            total_response_time_ms: total_time.as_millis() as u64,
            input_tokens: self.input_tokens,
            output_tokens,
            context_tokens: self.context_tokens,
            knowledge_tokens: self.knowledge_tokens,
            conversation_tokens: self.conversation_tokens,
            knowledge_chunks_retrieved: self.knowledge_chunks,
            conversation_messages_retrieved: self.conversation_messages,
            context_near_limit: self.context_near_limit,
            model: self.model,
        }
    }
}

impl MetricsTracker {
    pub fn new(session_id: &str) -> Self {
        Self {
            message_metrics: Vec::new(),
            current_message: None,
            session_start: Instant::now(),
            session_id: session_id.to_string(),
            cache_hits: 0,
            cache_misses: 0,
            source_access_counts: HashMap::new(),
        }
    }

    /// Start tracking a new message
    pub fn start_message(&mut self, model: &str) {
        self.current_message = Some(MessageMetricsBuilder::new(model));
    }

    /// Get mutable reference to current message builder
    pub fn current_message_mut(&mut self) -> Option<&mut MessageMetricsBuilder> {
        self.current_message.as_mut()
    }

    /// Mark first token received
    pub fn mark_first_token(&mut self) {
        if let Some(ref mut builder) = self.current_message {
            builder.mark_first_token();
        }
    }

    /// Finish tracking current message
    pub fn finish_message(&mut self, output_tokens: usize) {
        if let Some(builder) = self.current_message.take() {
            self.message_metrics.push(builder.finish(output_tokens));
        }
    }

    /// Cancel current message tracking
    pub fn cancel_message(&mut self) {
        self.current_message = None;
    }

    /// Record a cache hit for RAG
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss for RAG
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Record source access
    pub fn record_source_access(&mut self, source: &str) {
        *self.source_access_counts.entry(source.to_string()).or_insert(0) += 1;
    }

    /// Get all message metrics
    pub fn get_message_metrics(&self) -> &[MessageMetrics] {
        &self.message_metrics
    }

    /// Get aggregated session metrics
    pub fn get_session_metrics(&self) -> SessionMetrics {
        let count = self.message_metrics.len();
        if count == 0 {
            return SessionMetrics {
                session_id: self.session_id.clone(),
                ..Default::default()
            };
        }

        let total_input: usize = self.message_metrics.iter().map(|m| m.input_tokens).sum();
        let total_output: usize = self.message_metrics.iter().map(|m| m.output_tokens).sum();
        let total_response_time: u64 = self.message_metrics.iter().map(|m| m.total_response_time_ms).sum();
        let total_ttft: u64 = self.message_metrics.iter()
            .filter_map(|m| m.time_to_first_token_ms)
            .sum();
        let ttft_count = self.message_metrics.iter()
            .filter(|m| m.time_to_first_token_ms.is_some())
            .count();
        let total_chunks: usize = self.message_metrics.iter().map(|m| m.knowledge_chunks_retrieved).sum();
        let context_warnings = self.message_metrics.iter().filter(|m| m.context_near_limit).count();

        SessionMetrics {
            session_id: self.session_id.clone(),
            message_count: count,
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            avg_response_time_ms: total_response_time as f64 / count as f64,
            avg_time_to_first_token_ms: if ttft_count > 0 {
                total_ttft as f64 / ttft_count as f64
            } else {
                0.0
            },
            total_knowledge_chunks: total_chunks,
            avg_context_usage_percent: 0.0, // Would need context limit to calculate
            context_limit_warnings: context_warnings,
            session_duration_secs: self.session_start.elapsed().as_secs(),
        }
    }

    /// Get RAG quality metrics
    pub fn get_rag_quality_metrics(&self) -> RagQualityMetrics {
        let total_queries = self.cache_hits + self.cache_misses;
        let queries_with_results = self.message_metrics.iter()
            .filter(|m| m.knowledge_chunks_retrieved > 0)
            .count();

        let total_chunks: usize = self.message_metrics.iter()
            .map(|m| m.knowledge_chunks_retrieved)
            .sum();
        let total_tokens: usize = self.message_metrics.iter()
            .map(|m| m.knowledge_tokens)
            .sum();

        let mut top_sources: Vec<(String, usize)> = self.source_access_counts
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        top_sources.sort_by(|a, b| b.1.cmp(&a.1));
        top_sources.truncate(5);

        RagQualityMetrics {
            total_queries,
            queries_with_results,
            avg_chunks_per_query: if total_queries > 0 {
                total_chunks as f64 / total_queries as f64
            } else {
                0.0
            },
            avg_tokens_per_query: if total_queries > 0 {
                total_tokens as f64 / total_queries as f64
            } else {
                0.0
            },
            cache_hit_rate: if total_queries > 0 {
                self.cache_hits as f64 / total_queries as f64
            } else {
                0.0
            },
            top_sources,
        }
    }

    /// Export all metrics to JSON
    pub fn export_json(&self) -> String {
        let export = MetricsExport {
            session_metrics: self.get_session_metrics(),
            rag_quality: self.get_rag_quality_metrics(),
            message_metrics: self.message_metrics.clone(),
        };
        serde_json::to_string_pretty(&export).unwrap_or_default()
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.message_metrics.clear();
        self.current_message = None;
        self.session_start = Instant::now();
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.source_access_counts.clear();
    }
}

/// Export format for metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsExport {
    pub session_metrics: SessionMetrics,
    pub rag_quality: RagQualityMetrics,
    pub message_metrics: Vec<MessageMetrics>,
}

// === RAG Search Cache ===

/// Cache entry for RAG search results
#[derive(Clone)]
pub struct CacheEntry<T> {
    pub data: T,
    pub created_at: Instant,
    pub hits: usize,
}

/// LRU cache for RAG search results
pub struct SearchCache<T: Clone> {
    entries: HashMap<String, CacheEntry<T>>,
    max_entries: usize,
    ttl: Duration,
}

impl<T: Clone> SearchCache<T> {
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Get a cached entry if it exists and is not expired
    pub fn get(&mut self, key: &str) -> Option<T> {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.created_at.elapsed() < self.ttl {
                entry.hits += 1;
                return Some(entry.data.clone());
            } else {
                // Expired, remove it
                self.entries.remove(key);
            }
        }
        None
    }

    /// Insert a new entry, evicting oldest if at capacity
    pub fn insert(&mut self, key: String, data: T) {
        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }

        self.entries.insert(key, CacheEntry {
            data,
            created_at: Instant::now(),
            hits: 0,
        });
    }

    /// Remove expired entries
    pub fn cleanup(&mut self) {
        self.entries.retain(|_, entry| entry.created_at.elapsed() < self.ttl);
    }

    /// Evict the oldest entry
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self.entries
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    /// Get cache stats
    pub fn stats(&self) -> (usize, usize) {
        let total_hits: usize = self.entries.values().map(|e| e.hits).sum();
        (self.entries.len(), total_hits)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// === Quality Tests ===

/// Test case for conversation quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTestCase {
    /// Test name/description
    pub name: String,
    /// User query
    pub query: String,
    /// Expected keywords that should appear in response
    pub expected_keywords: Vec<String>,
    /// Keywords that should NOT appear
    pub forbidden_keywords: Vec<String>,
    /// Expected sources to be retrieved (if using RAG)
    pub expected_sources: Vec<String>,
    /// Maximum acceptable response time in ms
    pub max_response_time_ms: Option<u64>,
    /// Minimum expected output tokens
    pub min_output_tokens: Option<usize>,
    /// Maximum expected output tokens
    pub max_output_tokens: Option<usize>,
}

/// Result of a test case execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    /// Test case name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Response from the AI
    pub response: String,
    /// Metrics from the response
    pub metrics: MessageMetrics,
    /// Which expected keywords were found
    pub found_keywords: Vec<String>,
    /// Which expected keywords were missing
    pub missing_keywords: Vec<String>,
    /// Which forbidden keywords were found
    pub found_forbidden: Vec<String>,
    /// Which expected sources were retrieved
    pub found_sources: Vec<String>,
    /// Which expected sources were missing
    pub missing_sources: Vec<String>,
    /// Failure reasons
    pub failure_reasons: Vec<String>,
}

impl ConversationTestCase {
    /// Evaluate a response against this test case
    pub fn evaluate(&self, response: &str, metrics: &MessageMetrics, retrieved_sources: &[String]) -> TestCaseResult {
        let response_lower = response.to_lowercase();
        let mut failure_reasons = Vec::new();

        // Check keywords
        let found_keywords: Vec<String> = self.expected_keywords
            .iter()
            .filter(|kw| response_lower.contains(&kw.to_lowercase()))
            .cloned()
            .collect();

        let missing_keywords: Vec<String> = self.expected_keywords
            .iter()
            .filter(|kw| !response_lower.contains(&kw.to_lowercase()))
            .cloned()
            .collect();

        if !missing_keywords.is_empty() {
            failure_reasons.push(format!("Missing keywords: {:?}", missing_keywords));
        }

        // Check forbidden keywords
        let found_forbidden: Vec<String> = self.forbidden_keywords
            .iter()
            .filter(|kw| response_lower.contains(&kw.to_lowercase()))
            .cloned()
            .collect();

        if !found_forbidden.is_empty() {
            failure_reasons.push(format!("Found forbidden keywords: {:?}", found_forbidden));
        }

        // Check sources
        let found_sources: Vec<String> = self.expected_sources
            .iter()
            .filter(|s| retrieved_sources.iter().any(|rs| rs.contains(*s)))
            .cloned()
            .collect();

        let missing_sources: Vec<String> = self.expected_sources
            .iter()
            .filter(|s| !retrieved_sources.iter().any(|rs| rs.contains(*s)))
            .cloned()
            .collect();

        if !missing_sources.is_empty() && !self.expected_sources.is_empty() {
            failure_reasons.push(format!("Missing sources: {:?}", missing_sources));
        }

        // Check response time
        if let Some(max_time) = self.max_response_time_ms {
            if metrics.total_response_time_ms > max_time {
                failure_reasons.push(format!(
                    "Response time {} ms > max {} ms",
                    metrics.total_response_time_ms, max_time
                ));
            }
        }

        // Check token counts
        if let Some(min_tokens) = self.min_output_tokens {
            if metrics.output_tokens < min_tokens {
                failure_reasons.push(format!(
                    "Output tokens {} < min {}",
                    metrics.output_tokens, min_tokens
                ));
            }
        }

        if let Some(max_tokens) = self.max_output_tokens {
            if metrics.output_tokens > max_tokens {
                failure_reasons.push(format!(
                    "Output tokens {} > max {}",
                    metrics.output_tokens, max_tokens
                ));
            }
        }

        TestCaseResult {
            name: self.name.clone(),
            passed: failure_reasons.is_empty(),
            response: response.to_string(),
            metrics: metrics.clone(),
            found_keywords,
            missing_keywords,
            found_forbidden,
            found_sources,
            missing_sources,
            failure_reasons,
        }
    }
}

/// Test suite for running multiple test cases
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub description: String,
    pub test_cases: Vec<ConversationTestCase>,
}

/// Results from running a test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResults {
    pub suite_name: String,
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub avg_response_time_ms: f64,
    pub results: Vec<TestCaseResult>,
}

impl TestSuite {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            test_cases: Vec::new(),
        }
    }

    pub fn add_test(&mut self, test: ConversationTestCase) {
        self.test_cases.push(test);
    }

    /// Create results summary from individual results
    pub fn summarize_results(&self, results: Vec<TestCaseResult>) -> TestSuiteResults {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let total_time: u64 = results.iter().map(|r| r.metrics.total_response_time_ms).sum();

        TestSuiteResults {
            suite_name: self.name.clone(),
            total_tests: total,
            passed,
            failed,
            pass_rate: if total > 0 { passed as f64 / total as f64 } else { 0.0 },
            avg_response_time_ms: if total > 0 { total_time as f64 / total as f64 } else { 0.0 },
            results,
        }
    }

    /// Load test suite from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Export test suite to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_cache() {
        let mut cache: SearchCache<String> = SearchCache::new(3, 60);

        cache.insert("key1".to_string(), "value1".to_string());
        cache.insert("key2".to_string(), "value2".to_string());

        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key3"), None);

        // Test eviction
        cache.insert("key3".to_string(), "value3".to_string());
        cache.insert("key4".to_string(), "value4".to_string());

        // key2 should be evicted (oldest without hits)
        assert_eq!(cache.entries.len(), 3);
    }

    #[test]
    fn test_conversation_test_case() {
        let test = ConversationTestCase {
            name: "Test greeting".to_string(),
            query: "Hello".to_string(),
            expected_keywords: vec!["hello".to_string(), "help".to_string()],
            forbidden_keywords: vec!["error".to_string()],
            expected_sources: vec![],
            max_response_time_ms: Some(5000),
            min_output_tokens: Some(5),
            max_output_tokens: Some(100),
        };

        let metrics = MessageMetrics {
            output_tokens: 20,
            total_response_time_ms: 1000,
            ..Default::default()
        };

        let result = test.evaluate("Hello! How can I help you today?", &metrics, &[]);
        assert!(result.passed);
        assert!(result.found_keywords.contains(&"hello".to_string()));
        assert!(result.found_keywords.contains(&"help".to_string()));
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new("test_session");

        tracker.start_message("test-model");
        if let Some(builder) = tracker.current_message_mut() {
            builder.set_input_tokens(50);
            builder.set_context_info(1000, 200, 100, false);
        }
        tracker.mark_first_token();
        tracker.finish_message(100);

        let session = tracker.get_session_metrics();
        assert_eq!(session.message_count, 1);
        assert_eq!(session.total_input_tokens, 50);
        assert_eq!(session.total_output_tokens, 100);
    }
}
