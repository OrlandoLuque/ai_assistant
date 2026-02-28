//! Batch processing for parallel message handling
//!
//! This module provides utilities for processing multiple messages
//! in parallel with configurable concurrency.
//!
//! # Features
//!
//! - **Parallel execution**: Process multiple requests concurrently
//! - **Rate limiting**: Respect provider limits
//! - **Progress tracking**: Monitor batch progress
//! - **Error handling**: Handle partial failures gracefully
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::batch::{BatchProcessor, BatchConfig, BatchRequest};
//!
//! let config = BatchConfig {
//!     max_concurrent: 4,
//!     ..Default::default()
//! };
//!
//! let processor = BatchProcessor::new(config);
//!
//! let requests = vec![
//!     BatchRequest::new("id1", "What is 2+2?"),
//!     BatchRequest::new("id2", "What is the capital of France?"),
//! ];
//!
//! // Process would be: let results = processor.process(requests, generate_fn);
//! ```

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Delay between starting requests
    pub request_delay: Duration,
    /// Maximum retries per request
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Overall timeout for batch
    pub batch_timeout: Option<Duration>,
    /// Individual request timeout
    pub request_timeout: Duration,
    /// Continue on error
    pub continue_on_error: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            request_delay: Duration::from_millis(100),
            max_retries: 2,
            retry_delay: Duration::from_secs(1),
            batch_timeout: None,
            request_timeout: Duration::from_secs(120),
            continue_on_error: true,
        }
    }
}

/// A batch request
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request ID
    pub id: String,
    /// Message content
    pub message: String,
    /// Optional system prompt
    pub system_prompt: Option<String>,
    /// Optional model override
    pub model: Option<String>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl BatchRequest {
    /// Create a new batch request
    pub fn new(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            message: message.into(),
            system_prompt: None,
            model: None,
            metadata: HashMap::new(),
        }
    }

    /// Add system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add model override
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Result of a batch request
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Request ID
    pub id: String,
    /// Response content (if successful)
    pub response: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Processing duration
    pub duration: Duration,
    /// Number of retries used
    pub retries: usize,
    /// Whether request succeeded
    pub success: bool,
}

impl BatchResult {
    fn success(id: String, response: String, duration: Duration, retries: usize) -> Self {
        Self {
            id,
            response: Some(response),
            error: None,
            duration,
            retries,
            success: true,
        }
    }

    fn failure(id: String, error: String, duration: Duration, retries: usize) -> Self {
        Self {
            id,
            response: None,
            error: Some(error),
            duration,
            retries,
            success: false,
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total requests
    pub total: usize,
    /// Completed requests
    pub completed: usize,
    /// Successful requests
    pub successful: usize,
    /// Failed requests
    pub failed: usize,
    /// Total retries
    pub total_retries: usize,
    /// Average duration
    pub avg_duration: Duration,
    /// Total duration
    pub total_duration: Duration,
    /// Start time
    pub started_at: Option<Instant>,
    /// End time
    pub completed_at: Option<Instant>,
}

impl BatchStats {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.completed == 0 {
            0.0
        } else {
            self.successful as f64 / self.completed as f64
        }
    }

    /// Get requests per second
    pub fn requests_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs > 0.0 {
            self.completed as f64 / secs
        } else {
            0.0
        }
    }

    /// Get estimated time remaining
    pub fn eta(&self) -> Option<Duration> {
        if self.completed == 0 || self.completed >= self.total {
            return None;
        }

        let remaining = self.total - self.completed;
        let avg_per_request = self.total_duration.as_secs_f64() / self.completed as f64;
        Some(Duration::from_secs_f64(remaining as f64 * avg_per_request))
    }
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(&BatchStats) + Send + Sync>;

/// Batch processor
pub struct BatchProcessor {
    config: BatchConfig,
    stats: Arc<Mutex<BatchStats>>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(BatchStats::default())),
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Process a batch of requests
    pub fn process<F>(&self, requests: Vec<BatchRequest>, generator: F) -> BatchResults
    where
        F: Fn(&BatchRequest) -> Result<String, String> + Send + Sync + 'static,
    {
        self.process_with_progress(requests, generator, None)
    }

    /// Process with progress callback
    pub fn process_with_progress<F>(
        &self,
        requests: Vec<BatchRequest>,
        generator: F,
        progress: Option<ProgressCallback>,
    ) -> BatchResults
    where
        F: Fn(&BatchRequest) -> Result<String, String> + Send + Sync + 'static,
    {
        let total = requests.len();

        // Initialize stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            *stats = BatchStats {
                total,
                started_at: Some(Instant::now()),
                ..Default::default()
            };
        }

        // Reset cancelled flag
        self.cancelled.store(false, Ordering::SeqCst);

        let results = Arc::new(Mutex::new(Vec::with_capacity(total)));
        let generator = Arc::new(generator);
        let progress = progress.map(Arc::new);

        // Use a thread pool approach with channels
        let (tx, rx) = std::sync::mpsc::channel::<BatchRequest>();
        let rx = Arc::new(Mutex::new(rx));
        let request_count = Arc::new(AtomicUsize::new(0));

        // Spawn worker threads
        let mut handles = Vec::new();
        for _ in 0..self.config.max_concurrent {
            let rx = rx.clone();
            let generator = generator.clone();
            let results = results.clone();
            let stats = self.stats.clone();
            let cancelled = self.cancelled.clone();
            let config = self.config.clone();
            let progress = progress.clone();

            let handle = thread::spawn(move || {
                loop {
                    let request = {
                        let rx = rx.lock().unwrap_or_else(|e| e.into_inner());
                        rx.recv().ok()
                    };
                    let Some(request) = request else { break };
                    if cancelled.load(Ordering::SeqCst) {
                        break;
                    }

                    let start = Instant::now();
                    let mut retries = 0;
                    #[allow(unused_assignments)]
                    let mut last_error = String::new();

                    let result = loop {
                        match generator(&request) {
                            Ok(response) => {
                                break BatchResult::success(
                                    request.id.clone(),
                                    response,
                                    start.elapsed(),
                                    retries,
                                );
                            }
                            Err(e) => {
                                last_error = e;
                                retries += 1;

                                if retries > config.max_retries {
                                    break BatchResult::failure(
                                        request.id.clone(),
                                        last_error,
                                        start.elapsed(),
                                        retries - 1,
                                    );
                                }

                                thread::sleep(config.retry_delay);
                            }
                        }
                    };

                    // Update stats
                    {
                        let mut stats = stats.lock().unwrap_or_else(|e| e.into_inner());
                        stats.completed += 1;
                        if result.success {
                            stats.successful += 1;
                        } else {
                            stats.failed += 1;
                        }
                        stats.total_retries += result.retries;
                        stats.total_duration =
                            stats.started_at.map(|s| s.elapsed()).unwrap_or_default();

                        if stats.completed > 0 {
                            stats.avg_duration = Duration::from_nanos(
                                stats.total_duration.as_nanos() as u64 / stats.completed as u64,
                            );
                        }

                        // Call progress callback
                        if let Some(ref progress) = progress {
                            progress(&stats);
                        }
                    }

                    results
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(result);
                }
            });

            handles.push(handle);
        }

        // Drop our copy of rx so workers can detect completion
        drop(rx);

        // Send requests with delay
        for request in requests {
            if self.cancelled.load(Ordering::SeqCst) {
                break;
            }

            tx.send(request).ok();
            request_count.fetch_add(1, Ordering::SeqCst);

            if !self.config.request_delay.is_zero() {
                thread::sleep(self.config.request_delay);
            }
        }

        // Drop sender to signal completion
        drop(tx);

        // Wait for all workers
        for handle in handles {
            handle.join().ok();
        }

        // Finalize stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.completed_at = Some(Instant::now());
        }

        // Collect results
        let mut final_results = results.lock().unwrap_or_else(|e| e.into_inner()).clone();
        final_results.sort_by(|a, b| a.id.cmp(&b.id));

        BatchResults {
            results: final_results,
            stats: self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }

    /// Cancel ongoing batch processing
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if processing is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get current stats
    pub fn stats(&self) -> BatchStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

/// Results from batch processing
#[derive(Debug, Clone)]
pub struct BatchResults {
    /// Individual results
    pub results: Vec<BatchResult>,
    /// Overall statistics
    pub stats: BatchStats,
}

impl BatchResults {
    /// Get successful results
    pub fn successful(&self) -> Vec<&BatchResult> {
        self.results.iter().filter(|r| r.success).collect()
    }

    /// Get failed results
    pub fn failed(&self) -> Vec<&BatchResult> {
        self.results.iter().filter(|r| !r.success).collect()
    }

    /// Get result by ID
    pub fn get(&self, id: &str) -> Option<&BatchResult> {
        self.results.iter().find(|r| r.id == id)
    }

    /// Convert to HashMap
    pub fn to_map(&self) -> HashMap<String, &BatchResult> {
        self.results.iter().map(|r| (r.id.clone(), r)).collect()
    }
}

/// Builder for batch requests
pub struct BatchBuilder {
    requests: Vec<BatchRequest>,
    default_system_prompt: Option<String>,
    default_model: Option<String>,
}

impl BatchBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            default_system_prompt: None,
            default_model: None,
        }
    }

    /// Set default system prompt
    pub fn default_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.default_system_prompt = Some(prompt.into());
        self
    }

    /// Set default model
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Add a request
    pub fn add(mut self, id: impl Into<String>, message: impl Into<String>) -> Self {
        let mut request = BatchRequest::new(id, message);

        if request.system_prompt.is_none() {
            request.system_prompt = self.default_system_prompt.clone();
        }
        if request.model.is_none() {
            request.model = self.default_model.clone();
        }

        self.requests.push(request);
        self
    }

    /// Add multiple requests
    pub fn add_all<I, S1, S2>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        for (id, message) in items {
            let mut request = BatchRequest::new(id, message);

            if request.system_prompt.is_none() {
                request.system_prompt = self.default_system_prompt.clone();
            }
            if request.model.is_none() {
                request.model = self.default_model.clone();
            }

            self.requests.push(request);
        }
        self
    }

    /// Build the request list
    pub fn build(self) -> Vec<BatchRequest> {
        self.requests
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_request() {
        let req = BatchRequest::new("id1", "Hello")
            .with_system_prompt("Be helpful")
            .with_model("gpt-4")
            .with_metadata("key", "value");

        assert_eq!(req.id, "id1");
        assert_eq!(req.message, "Hello");
        assert_eq!(req.system_prompt, Some("Be helpful".to_string()));
        assert_eq!(req.model, Some("gpt-4".to_string()));
        assert_eq!(req.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_batch_builder() {
        let requests = BatchBuilder::new()
            .default_system_prompt("System")
            .add("1", "Hello")
            .add("2", "World")
            .build();

        assert_eq!(requests.len(), 2);
        assert!(requests
            .iter()
            .all(|r| r.system_prompt == Some("System".to_string())));
    }

    #[test]
    fn test_batch_processor() {
        let config = BatchConfig {
            max_concurrent: 2,
            request_delay: Duration::ZERO,
            ..Default::default()
        };

        let processor = BatchProcessor::new(config);

        let requests = vec![
            BatchRequest::new("1", "a"),
            BatchRequest::new("2", "b"),
            BatchRequest::new("3", "c"),
        ];

        let results =
            processor.process(requests, |req| Ok(format!("Response to: {}", req.message)));

        assert_eq!(results.stats.total, 3);
        assert_eq!(results.stats.successful, 3);
        assert_eq!(results.stats.failed, 0);
    }

    #[test]
    fn test_batch_with_failures() {
        let config = BatchConfig {
            max_concurrent: 1,
            max_retries: 0,
            request_delay: Duration::ZERO,
            ..Default::default()
        };

        let processor = BatchProcessor::new(config);

        let requests = vec![BatchRequest::new("1", "ok"), BatchRequest::new("2", "fail")];

        let results = processor.process(requests, |req| {
            if req.message == "fail" {
                Err("Failed!".to_string())
            } else {
                Ok("OK".to_string())
            }
        });

        assert_eq!(results.stats.successful, 1);
        assert_eq!(results.stats.failed, 1);
    }

    #[test]
    fn test_batch_stats() {
        let stats = BatchStats {
            total: 10,
            completed: 8,
            successful: 7,
            failed: 1,
            total_duration: Duration::from_secs(4),
            ..Default::default()
        };

        assert!((stats.success_rate() - 0.875).abs() < 0.001);
        assert!((stats.requests_per_second() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_stats_eta() {
        let stats = BatchStats {
            total: 10,
            completed: 5,
            total_duration: Duration::from_secs(10),
            ..Default::default()
        };
        // 5 remaining, avg 2s each => 10s ETA
        let eta = stats.eta().unwrap();
        assert!((eta.as_secs_f64() - 10.0).abs() < 0.1);

        // When all completed, no ETA
        let done = BatchStats { total: 5, completed: 5, ..Default::default() };
        assert!(done.eta().is_none());
    }

    #[test]
    fn test_batch_results_accessors() {
        let config = BatchConfig {
            max_concurrent: 1,
            max_retries: 0,
            request_delay: Duration::ZERO,
            ..Default::default()
        };
        let processor = BatchProcessor::new(config);
        let requests = vec![
            BatchRequest::new("ok1", "good"),
            BatchRequest::new("fail1", "bad"),
            BatchRequest::new("ok2", "good"),
        ];
        let results = processor.process(requests, |req| {
            if req.message == "bad" { Err("err".into()) } else { Ok("ok".into()) }
        });
        assert_eq!(results.successful().len(), 2);
        assert_eq!(results.failed().len(), 1);
        assert!(results.get("ok1").unwrap().success);
        assert!(!results.get("fail1").unwrap().success);
        let map = results.to_map();
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.max_concurrent, 4);
        assert_eq!(config.max_retries, 2);
        assert!(config.continue_on_error);
        assert!(config.batch_timeout.is_none());
    }

    #[test]
    fn test_batch_builder_add_all() {
        let requests = BatchBuilder::new()
            .default_model("llama3")
            .add_all(vec![("1", "a"), ("2", "b"), ("3", "c")])
            .build();
        assert_eq!(requests.len(), 3);
        assert!(requests.iter().all(|r| r.model == Some("llama3".into())));
    }

    #[test]
    fn test_cancel_batch() {
        let processor = BatchProcessor::default();
        assert!(!processor.is_cancelled());
        processor.cancel();
        assert!(processor.is_cancelled());
    }
}
