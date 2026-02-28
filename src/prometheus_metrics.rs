//! Prometheus-compatible metrics for AI operations
//!
//! This module provides Prometheus-style metrics collection for
//! monitoring AI operations in production environments.
//!
//! # Features
//!
//! - **Counters**: Request counts, error counts
//! - **Histograms**: Latency distributions
//! - **Gauges**: Active connections, queue sizes
//! - **Labels**: Per-model, per-provider metrics

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// A Prometheus-style counter
#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    help: String,
    values: Arc<RwLock<HashMap<Vec<(String, String)>, u64>>>,
}

impl Counter {
    /// Create a new counter
    pub fn new(name: impl Into<String>, help: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            values: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Increment counter with labels
    pub fn inc_with_labels(&self, labels: &[(&str, &str)]) {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if let Ok(mut values) = self.values.write() {
            *values.entry(key).or_insert(0) += 1;
        }
    }

    /// Increment counter without labels
    pub fn inc(&self) {
        self.inc_with_labels(&[]);
    }

    /// Add to counter
    pub fn add(&self, n: u64) {
        self.add_with_labels(n, &[]);
    }

    /// Add to counter with labels
    pub fn add_with_labels(&self, n: u64, labels: &[(&str, &str)]) {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if let Ok(mut values) = self.values.write() {
            *values.entry(key).or_insert(0) += n;
        }
    }

    /// Get counter value
    pub fn get(&self, labels: &[(&str, &str)]) -> u64 {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        self.values
            .read()
            .map(|v| *v.get(&key).unwrap_or(&0))
            .unwrap_or(0)
    }

    /// Export to Prometheus format
    pub fn export(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} counter\n", self.name));

        if let Ok(values) = self.values.read() {
            for (labels, value) in values.iter() {
                let label_str = self.format_labels(labels);
                output.push_str(&format!("{}{} {}\n", self.name, label_str, value));
            }
        }

        output
    }

    fn format_labels(&self, labels: &[(String, String)]) -> String {
        if labels.is_empty() {
            String::new()
        } else {
            let parts: Vec<_> = labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// A Prometheus-style gauge
#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    help: String,
    values: Arc<RwLock<HashMap<Vec<(String, String)>, f64>>>,
}

impl Gauge {
    /// Create a new gauge
    pub fn new(name: impl Into<String>, help: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            values: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set gauge value
    pub fn set(&self, value: f64) {
        self.set_with_labels(value, &[]);
    }

    /// Set gauge value with labels
    pub fn set_with_labels(&self, value: f64, labels: &[(&str, &str)]) {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if let Ok(mut values) = self.values.write() {
            values.insert(key, value);
        }
    }

    /// Increment gauge
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Decrement gauge
    pub fn dec(&self) {
        self.add(-1.0);
    }

    /// Add to gauge
    pub fn add(&self, n: f64) {
        self.add_with_labels(n, &[]);
    }

    /// Add to gauge with labels
    pub fn add_with_labels(&self, n: f64, labels: &[(&str, &str)]) {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if let Ok(mut values) = self.values.write() {
            *values.entry(key).or_insert(0.0) += n;
        }
    }

    /// Get gauge value
    pub fn get(&self, labels: &[(&str, &str)]) -> f64 {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        self.values
            .read()
            .map(|v| *v.get(&key).unwrap_or(&0.0))
            .unwrap_or(0.0)
    }

    /// Export to Prometheus format
    pub fn export(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} gauge\n", self.name));

        if let Ok(values) = self.values.read() {
            for (labels, value) in values.iter() {
                let label_str = self.format_labels(labels);
                output.push_str(&format!("{}{} {}\n", self.name, label_str, value));
            }
        }

        output
    }

    fn format_labels(&self, labels: &[(String, String)]) -> String {
        if labels.is_empty() {
            String::new()
        } else {
            let parts: Vec<_> = labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// A Prometheus-style histogram
#[derive(Debug, Clone)]
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
    observations: Arc<Mutex<HashMap<Vec<(String, String)>, HistogramData>>>,
}

#[derive(Debug, Clone, Default)]
struct HistogramData {
    bucket_counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    /// Create a new histogram with default buckets
    pub fn new(name: impl Into<String>, help: impl Into<String>) -> Self {
        Self::with_buckets(
            name,
            help,
            vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        )
    }

    /// Create a histogram with custom buckets
    pub fn with_buckets(
        name: impl Into<String>,
        help: impl Into<String>,
        buckets: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            buckets,
            observations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Observe a value
    pub fn observe(&self, value: f64) {
        self.observe_with_labels(value, &[]);
    }

    /// Observe a value with labels
    pub fn observe_with_labels(&self, value: f64, labels: &[(&str, &str)]) {
        let key: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if let Ok(mut obs) = self.observations.lock() {
            let data = obs.entry(key).or_insert_with(|| HistogramData {
                bucket_counts: vec![0; self.buckets.len() + 1],
                sum: 0.0,
                count: 0,
            });

            data.sum += value;
            data.count += 1;

            // Update buckets
            for (i, &bucket) in self.buckets.iter().enumerate() {
                if value <= bucket {
                    data.bucket_counts[i] += 1;
                }
            }
            // +Inf bucket
            data.bucket_counts[self.buckets.len()] += 1;
        }
    }

    /// Start a timer
    pub fn start_timer(&self) -> HistogramTimer {
        HistogramTimer {
            histogram: self.clone(),
            labels: Vec::new(),
            start: Instant::now(),
        }
    }

    /// Start a timer with labels
    pub fn start_timer_with_labels(&self, labels: &[(&str, &str)]) -> HistogramTimer {
        let label_vec: Vec<_> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        HistogramTimer {
            histogram: self.clone(),
            labels: label_vec,
            start: Instant::now(),
        }
    }

    /// Export to Prometheus format
    pub fn export(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} histogram\n", self.name));

        if let Ok(obs) = self.observations.lock() {
            for (labels, data) in obs.iter() {
                let label_str = self.format_labels(labels);

                // Export buckets
                for (i, &bucket) in self.buckets.iter().enumerate() {
                    let bucket_label = if labels.is_empty() {
                        format!("{{le=\"{}\"}}", bucket)
                    } else {
                        let parts: Vec<_> = labels
                            .iter()
                            .map(|(k, v)| format!("{}=\"{}\"", k, v))
                            .chain(std::iter::once(format!("le=\"{}\"", bucket)))
                            .collect();
                        format!("{{{}}}", parts.join(","))
                    };
                    output.push_str(&format!(
                        "{}_bucket{} {}\n",
                        self.name, bucket_label, data.bucket_counts[i]
                    ));
                }

                // +Inf bucket
                let inf_label = if labels.is_empty() {
                    "{le=\"+Inf\"}".to_string()
                } else {
                    let parts: Vec<_> = labels
                        .iter()
                        .map(|(k, v)| format!("{}=\"{}\"", k, v))
                        .chain(std::iter::once("le=\"+Inf\"".to_string()))
                        .collect();
                    format!("{{{}}}", parts.join(","))
                };
                output.push_str(&format!(
                    "{}_bucket{} {}\n",
                    self.name,
                    inf_label,
                    data.bucket_counts[self.buckets.len()]
                ));

                // Sum and count
                output.push_str(&format!("{}_sum{} {}\n", self.name, label_str, data.sum));
                output.push_str(&format!(
                    "{}_count{} {}\n",
                    self.name, label_str, data.count
                ));
            }
        }

        output
    }

    fn format_labels(&self, labels: &[(String, String)]) -> String {
        if labels.is_empty() {
            String::new()
        } else {
            let parts: Vec<_> = labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// Timer for histogram observations
pub struct HistogramTimer {
    histogram: Histogram,
    labels: Vec<(String, String)>,
    start: Instant,
}

impl HistogramTimer {
    /// Stop the timer and record observation
    pub fn observe(self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let labels: Vec<_> = self
            .labels
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.histogram.observe_with_labels(elapsed, &labels);
    }
}

impl Drop for HistogramTimer {
    fn drop(&mut self) {
        // Auto-observe on drop if not explicitly called
        // Note: This is a simplified implementation
    }
}

/// AI-specific metrics registry
pub struct AiMetricsRegistry {
    /// Total requests counter
    pub requests_total: Counter,
    /// Request errors counter
    pub request_errors_total: Counter,
    /// Active requests gauge
    pub active_requests: Gauge,
    /// Request latency histogram
    pub request_duration_seconds: Histogram,
    /// Tokens processed counter
    pub tokens_processed_total: Counter,
    /// Token generation rate histogram
    pub tokens_per_second: Histogram,
    /// Cache hit/miss counters
    pub cache_hits_total: Counter,
    pub cache_misses_total: Counter,
    /// Model loading time histogram
    pub model_load_duration_seconds: Histogram,
    /// Queue size gauge
    pub queue_size: Gauge,
}

impl AiMetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            requests_total: Counter::new("ai_requests_total", "Total number of AI requests"),
            request_errors_total: Counter::new(
                "ai_request_errors_total",
                "Total number of AI request errors",
            ),
            active_requests: Gauge::new(
                "ai_active_requests",
                "Number of currently active AI requests",
            ),
            request_duration_seconds: Histogram::with_buckets(
                "ai_request_duration_seconds",
                "AI request duration in seconds",
                vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            ),
            tokens_processed_total: Counter::new(
                "ai_tokens_processed_total",
                "Total tokens processed",
            ),
            tokens_per_second: Histogram::with_buckets(
                "ai_tokens_per_second",
                "Token generation rate",
                vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            ),
            cache_hits_total: Counter::new("ai_cache_hits_total", "Total cache hits"),
            cache_misses_total: Counter::new("ai_cache_misses_total", "Total cache misses"),
            model_load_duration_seconds: Histogram::with_buckets(
                "ai_model_load_duration_seconds",
                "Model loading duration in seconds",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            ),
            queue_size: Gauge::new("ai_queue_size", "Current request queue size"),
        }
    }

    /// Record a request
    pub fn record_request(&self, model: &str, provider: &str) {
        self.requests_total
            .inc_with_labels(&[("model", model), ("provider", provider)]);
        self.active_requests.inc();
    }

    /// Record request completion
    pub fn record_completion(
        &self,
        model: &str,
        provider: &str,
        duration: Duration,
        tokens: u64,
        success: bool,
    ) {
        self.active_requests.dec();

        let labels = &[("model", model), ("provider", provider)];

        self.request_duration_seconds
            .observe_with_labels(duration.as_secs_f64(), labels);

        self.tokens_processed_total.add_with_labels(tokens, labels);

        if !success {
            self.request_errors_total.inc_with_labels(labels);
        }

        if duration.as_secs_f64() > 0.0 {
            let rate = tokens as f64 / duration.as_secs_f64();
            self.tokens_per_second.observe_with_labels(rate, labels);
        }
    }

    /// Record cache access
    pub fn record_cache(&self, hit: bool) {
        if hit {
            self.cache_hits_total.inc();
        } else {
            self.cache_misses_total.inc();
        }
    }

    /// Export all metrics
    pub fn export(&self) -> String {
        let mut output = String::new();
        output.push_str(&self.requests_total.export());
        output.push_str(&self.request_errors_total.export());
        output.push_str(&self.active_requests.export());
        output.push_str(&self.request_duration_seconds.export());
        output.push_str(&self.tokens_processed_total.export());
        output.push_str(&self.tokens_per_second.export());
        output.push_str(&self.cache_hits_total.export());
        output.push_str(&self.cache_misses_total.export());
        output.push_str(&self.model_load_duration_seconds.export());
        output.push_str(&self.queue_size.export());
        output
    }
}

impl Default for AiMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter", "Test counter");
        counter.inc();
        counter.inc();
        assert_eq!(counter.get(&[]), 2);

        counter.inc_with_labels(&[("model", "gpt-4")]);
        assert_eq!(counter.get(&[("model", "gpt-4")]), 1);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge", "Test gauge");
        gauge.set(10.0);
        assert_eq!(gauge.get(&[]), 10.0);

        gauge.inc();
        assert_eq!(gauge.get(&[]), 11.0);

        gauge.dec();
        assert_eq!(gauge.get(&[]), 10.0);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new("test_histogram", "Test histogram");
        histogram.observe(0.1);
        histogram.observe(0.5);
        histogram.observe(1.0);

        let export = histogram.export();
        assert!(export.contains("test_histogram_bucket"));
        assert!(export.contains("test_histogram_sum"));
        assert!(export.contains("test_histogram_count"));
    }

    #[test]
    fn test_metrics_registry() {
        let registry = AiMetricsRegistry::new();

        registry.record_request("gpt-4", "openai");
        registry.record_completion("gpt-4", "openai", Duration::from_secs(1), 100, true);
        registry.record_cache(true);
        registry.record_cache(false);

        let export = registry.export();
        assert!(export.contains("ai_requests_total"));
        assert!(export.contains("ai_cache_hits_total"));
    }

    #[test]
    fn test_export_format() {
        let counter = Counter::new("my_counter", "My counter help");
        counter.inc_with_labels(&[("status", "200")]);

        let export = counter.export();
        assert!(export.contains("# HELP my_counter My counter help"));
        assert!(export.contains("# TYPE my_counter counter"));
        assert!(export.contains("my_counter{status=\"200\"} 1"));
    }

    #[test]
    fn test_counter_add() {
        let counter = Counter::new("requests", "Total requests");
        counter.add(5);
        assert_eq!(counter.get(&[]), 5);
        counter.add(3);
        assert_eq!(counter.get(&[]), 8);
    }

    #[test]
    fn test_counter_multiple_labels() {
        let counter = Counter::new("http_requests", "HTTP requests");
        counter.inc_with_labels(&[("method", "GET"), ("status", "200")]);
        counter.inc_with_labels(&[("method", "POST"), ("status", "201")]);
        counter.inc_with_labels(&[("method", "GET"), ("status", "200")]);

        assert_eq!(counter.get(&[("method", "GET"), ("status", "200")]), 2);
        assert_eq!(counter.get(&[("method", "POST"), ("status", "201")]), 1);
    }

    #[test]
    fn test_gauge_set_and_export() {
        let gauge = Gauge::new("active_connections", "Active connections");
        gauge.set(42.0);

        let export = gauge.export();
        assert!(export.contains("# HELP active_connections"));
        assert!(export.contains("# TYPE active_connections gauge"));
        assert!(export.contains("42"));
    }

    #[test]
    fn test_histogram_buckets() {
        let histogram = Histogram::new("latency", "Request latency");
        histogram.observe(0.05);
        histogram.observe(0.25);
        histogram.observe(2.5);

        let export = histogram.export();
        assert!(export.contains("latency_count 3"));
    }

    #[test]
    fn test_registry_multiple_operations() {
        let registry = AiMetricsRegistry::new();
        registry.record_request("llama", "ollama");
        registry.record_request("llama", "ollama");
        registry.record_request("gpt-4", "openai");
        registry.record_completion("llama", "ollama", Duration::from_millis(500), 50, true);
        registry.record_completion("gpt-4", "openai", Duration::from_millis(200), 100, false);

        let export = registry.export();
        assert!(export.contains("ai_requests_total"));
        assert!(export.contains("ai_request_duration_seconds"));
    }
}
