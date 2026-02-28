//! Latency metrics tracking per provider
//!
//! This module provides detailed latency tracking and analysis
//! for AI provider requests.
//!
//! # Features
//!
//! - **Per-provider tracking**: Separate metrics for each provider
//! - **Percentile calculations**: P50, P90, P95, P99 latencies
//! - **Time series**: Track latency over time
//! - **Anomaly detection**: Identify unusual latency spikes
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::latency_metrics::{LatencyTracker, LatencyRecord};
//! use std::time::Duration;
//!
//! let mut tracker = LatencyTracker::new();
//!
//! // Record request latencies
//! tracker.record("ollama", Duration::from_millis(150), true);
//! tracker.record("ollama", Duration::from_millis(200), true);
//!
//! // Get statistics
//! let stats = tracker.stats("ollama").unwrap();
//! println!("Average: {:?}", stats.avg_latency);
//! println!("P95: {:?}", stats.p95);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// A single latency record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRecord {
    /// Timestamp of the request
    pub timestamp: u64,
    /// Request latency
    pub latency_ms: u64,
    /// Whether request succeeded
    pub success: bool,
    /// Optional model name
    pub model: Option<String>,
    /// Optional token count
    pub tokens: Option<usize>,
    /// Time to first token (for streaming)
    pub ttft_ms: Option<u64>,
}

impl LatencyRecord {
    /// Create a new record
    pub fn new(latency: Duration, success: bool) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            latency_ms: latency.as_millis() as u64,
            success,
            model: None,
            tokens: None,
            ttft_ms: None,
        }
    }

    /// Add model info
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add token count
    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Add time to first token
    pub fn with_ttft(mut self, ttft: Duration) -> Self {
        self.ttft_ms = Some(ttft.as_millis() as u64);
        self
    }

    /// Get latency as Duration
    pub fn latency(&self) -> Duration {
        Duration::from_millis(self.latency_ms)
    }

    /// Get tokens per second
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.tokens.map(|t| {
            let secs = self.latency_ms as f64 / 1000.0;
            if secs > 0.0 {
                t as f64 / secs
            } else {
                0.0
            }
        })
    }
}

/// Latency statistics for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Provider name
    pub provider: String,
    /// Total requests
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Success rate
    pub success_rate: f64,
    /// Minimum latency
    pub min_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
    /// Average latency
    pub avg_latency: Duration,
    /// Median latency (P50)
    pub p50: Duration,
    /// P90 latency
    pub p90: Duration,
    /// P95 latency
    pub p95: Duration,
    /// P99 latency
    pub p99: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Average tokens per second
    pub avg_tokens_per_second: Option<f64>,
    /// Average time to first token
    pub avg_ttft: Option<Duration>,
    /// Requests in the last minute
    pub requests_last_minute: usize,
    /// Requests in the last hour
    pub requests_last_hour: usize,
}

/// Provider-specific metrics storage
struct ProviderMetrics {
    records: VecDeque<LatencyRecord>,
    max_records: usize,
}

impl ProviderMetrics {
    fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
        }
    }

    fn add(&mut self, record: LatencyRecord) {
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    fn calculate_stats(&self, provider: &str) -> LatencyStats {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let one_minute_ago = now.saturating_sub(60);
        let one_hour_ago = now.saturating_sub(3600);

        let total = self.records.len();
        let successful: Vec<_> = self.records.iter().filter(|r| r.success).collect();
        let failed = total - successful.len();

        // Calculate latencies from successful requests only
        let mut latencies: Vec<u64> = successful.iter().map(|r| r.latency_ms).collect();
        latencies.sort_unstable();

        let min = latencies.first().copied().unwrap_or(0);
        let max = latencies.last().copied().unwrap_or(0);

        let avg = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        } else {
            0
        };

        let p50 = percentile(&latencies, 50);
        let p90 = percentile(&latencies, 90);
        let p95 = percentile(&latencies, 95);
        let p99 = percentile(&latencies, 99);

        // Standard deviation
        let std_dev = if latencies.len() > 1 {
            let mean = avg as f64;
            let variance: f64 = latencies
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / latencies.len() as f64;
            variance.sqrt() as u64
        } else {
            0
        };

        // Token stats
        let tps: Vec<f64> = successful
            .iter()
            .filter_map(|r| r.tokens_per_second())
            .collect();
        let avg_tps = if !tps.is_empty() {
            Some(tps.iter().sum::<f64>() / tps.len() as f64)
        } else {
            None
        };

        let ttfts: Vec<u64> = successful.iter().filter_map(|r| r.ttft_ms).collect();
        let avg_ttft = if !ttfts.is_empty() {
            Some(Duration::from_millis(
                ttfts.iter().sum::<u64>() / ttfts.len() as u64,
            ))
        } else {
            None
        };

        // Time-based counts
        let last_minute = self
            .records
            .iter()
            .filter(|r| r.timestamp >= one_minute_ago)
            .count();
        let last_hour = self
            .records
            .iter()
            .filter(|r| r.timestamp >= one_hour_ago)
            .count();

        LatencyStats {
            provider: provider.to_string(),
            total_requests: total,
            successful_requests: successful.len(),
            failed_requests: failed,
            success_rate: if total > 0 {
                successful.len() as f64 / total as f64
            } else {
                0.0
            },
            min_latency: Duration::from_millis(min),
            max_latency: Duration::from_millis(max),
            avg_latency: Duration::from_millis(avg),
            p50: Duration::from_millis(p50),
            p90: Duration::from_millis(p90),
            p95: Duration::from_millis(p95),
            p99: Duration::from_millis(p99),
            std_dev: Duration::from_millis(std_dev),
            avg_tokens_per_second: avg_tps,
            avg_ttft,
            requests_last_minute: last_minute,
            requests_last_hour: last_hour,
        }
    }
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// Latency tracker for multiple providers
pub struct LatencyTracker {
    providers: HashMap<String, ProviderMetrics>,
    max_records_per_provider: usize,
    anomaly_threshold: f64,
}

impl LatencyTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create with custom capacity
    pub fn with_capacity(max_records_per_provider: usize) -> Self {
        Self {
            providers: HashMap::new(),
            max_records_per_provider,
            anomaly_threshold: 3.0, // 3 standard deviations
        }
    }

    /// Record a request latency
    pub fn record(&mut self, provider: &str, latency: Duration, success: bool) -> &LatencyRecord {
        self.record_full(provider, LatencyRecord::new(latency, success))
    }

    /// Record with full details
    pub fn record_full(&mut self, provider: &str, record: LatencyRecord) -> &LatencyRecord {
        let metrics = self
            .providers
            .entry(provider.to_string())
            .or_insert_with(|| ProviderMetrics::new(self.max_records_per_provider));

        metrics.add(record);
        metrics.records.back().expect("record just added")
    }

    /// Get statistics for a provider
    pub fn stats(&self, provider: &str) -> Option<LatencyStats> {
        self.providers
            .get(provider)
            .map(|m| m.calculate_stats(provider))
    }

    /// Get statistics for all providers
    pub fn all_stats(&self) -> Vec<LatencyStats> {
        self.providers
            .iter()
            .map(|(name, metrics)| metrics.calculate_stats(name))
            .collect()
    }

    /// Check if a latency is anomalous
    pub fn is_anomalous(&self, provider: &str, latency: Duration) -> bool {
        if let Some(stats) = self.stats(provider) {
            let avg = stats.avg_latency.as_millis() as f64;
            let std = stats.std_dev.as_millis() as f64;
            let lat = latency.as_millis() as f64;

            if std > 0.0 {
                let z_score = (lat - avg) / std;
                return z_score.abs() > self.anomaly_threshold;
            }
        }
        false
    }

    /// Get recent latency trend (increasing, decreasing, stable)
    pub fn latency_trend(&self, provider: &str, window: usize) -> Option<LatencyTrend> {
        let metrics = self.providers.get(provider)?;
        if metrics.records.len() < window * 2 {
            return None;
        }

        let recent: Vec<_> = metrics
            .records
            .iter()
            .rev()
            .take(window)
            .filter(|r| r.success)
            .map(|r| r.latency_ms)
            .collect();

        let older: Vec<_> = metrics
            .records
            .iter()
            .rev()
            .skip(window)
            .take(window)
            .filter(|r| r.success)
            .map(|r| r.latency_ms)
            .collect();

        if recent.is_empty() || older.is_empty() {
            return None;
        }

        let recent_avg: f64 = recent.iter().sum::<u64>() as f64 / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<u64>() as f64 / older.len() as f64;

        let change_percent = (recent_avg - older_avg) / older_avg * 100.0;

        Some(if change_percent > 20.0 {
            LatencyTrend::Increasing { change_percent }
        } else if change_percent < -20.0 {
            LatencyTrend::Decreasing {
                change_percent: change_percent.abs(),
            }
        } else {
            LatencyTrend::Stable
        })
    }

    /// Get the fastest provider
    pub fn fastest_provider(&self) -> Option<String> {
        self.all_stats()
            .into_iter()
            .filter(|s| s.successful_requests > 0)
            .min_by_key(|s| s.avg_latency)
            .map(|s| s.provider)
    }

    /// Get the most reliable provider
    pub fn most_reliable_provider(&self) -> Option<String> {
        self.all_stats()
            .into_iter()
            .filter(|s| s.total_requests >= 10) // Minimum sample size
            .max_by(|a, b| {
                a.success_rate
                    .partial_cmp(&b.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.provider)
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.providers.clear();
    }

    /// Clear metrics for a specific provider
    pub fn clear_provider(&mut self, provider: &str) {
        self.providers.remove(provider);
    }

    /// Set anomaly detection threshold (in standard deviations)
    pub fn set_anomaly_threshold(&mut self, threshold: f64) {
        self.anomaly_threshold = threshold;
    }

    /// Get list of tracked providers
    pub fn providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum LatencyTrend {
    /// Latency is increasing
    Increasing { change_percent: f64 },
    /// Latency is decreasing
    Decreasing { change_percent: f64 },
    /// Latency is stable
    Stable,
}

/// Request timing helper
pub struct RequestTimer {
    start: Instant,
    first_byte: Option<Instant>,
}

impl RequestTimer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            first_byte: None,
        }
    }

    /// Mark when first byte was received
    pub fn mark_first_byte(&mut self) {
        if self.first_byte.is_none() {
            self.first_byte = Some(Instant::now());
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get time to first byte
    pub fn ttft(&self) -> Option<Duration> {
        self.first_byte.map(|fb| fb.duration_since(self.start))
    }

    /// Create a latency record
    pub fn finish(self, success: bool) -> LatencyRecord {
        let mut record = LatencyRecord::new(self.elapsed(), success);
        if let Some(ttft) = self.ttft() {
            record = record.with_ttft(ttft);
        }
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tracking() {
        let mut tracker = LatencyTracker::new();

        tracker.record("ollama", Duration::from_millis(100), true);
        tracker.record("ollama", Duration::from_millis(150), true);
        tracker.record("ollama", Duration::from_millis(200), true);

        let stats = tracker.stats("ollama").unwrap();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.success_rate, 1.0);
    }

    #[test]
    fn test_percentiles() {
        let mut tracker = LatencyTracker::new();

        for i in 1..=100 {
            tracker.record("test", Duration::from_millis(i), true);
        }

        let stats = tracker.stats("test").unwrap();
        // p50 can be 50 or 51 depending on implementation
        assert!(stats.p50.as_millis() >= 50 && stats.p50.as_millis() <= 51);
        assert!(stats.p90.as_millis() >= 90);
        assert!(stats.p99.as_millis() >= 99);
    }

    #[test]
    fn test_failure_tracking() {
        let mut tracker = LatencyTracker::new();

        tracker.record("test", Duration::from_millis(100), true);
        tracker.record("test", Duration::from_millis(200), false);
        tracker.record("test", Duration::from_millis(150), true);

        let stats = tracker.stats("test").unwrap();
        assert_eq!(stats.failed_requests, 1);
        assert!(stats.success_rate < 1.0);
    }

    #[test]
    fn test_multiple_providers() {
        let mut tracker = LatencyTracker::new();

        tracker.record("ollama", Duration::from_millis(100), true);
        tracker.record("lmstudio", Duration::from_millis(200), true);

        assert!(tracker.stats("ollama").is_some());
        assert!(tracker.stats("lmstudio").is_some());
        assert_eq!(tracker.all_stats().len(), 2);
    }

    #[test]
    fn test_fastest_provider() {
        let mut tracker = LatencyTracker::new();

        tracker.record("slow", Duration::from_millis(500), true);
        tracker.record("fast", Duration::from_millis(100), true);
        tracker.record("medium", Duration::from_millis(250), true);

        assert_eq!(tracker.fastest_provider(), Some("fast".to_string()));
    }

    #[test]
    fn test_request_timer() {
        let mut timer = RequestTimer::start();
        std::thread::sleep(Duration::from_millis(10));
        timer.mark_first_byte();
        std::thread::sleep(Duration::from_millis(10));

        let record = timer.finish(true);
        assert!(record.latency_ms >= 20);
        assert!(record.ttft_ms.is_some());
        assert!(record.ttft_ms.unwrap() <= record.latency_ms);
    }

    #[test]
    fn test_tokens_per_second() {
        let record = LatencyRecord::new(Duration::from_secs(2), true).with_tokens(100);

        assert_eq!(record.tokens_per_second(), Some(50.0));
    }

    #[test]
    fn test_record_with_model() {
        let record = LatencyRecord::new(Duration::from_millis(100), true)
            .with_model("gpt-4")
            .with_tokens(50);
        assert_eq!(record.latency(), Duration::from_millis(100));
        assert!(record.tokens_per_second().unwrap() > 0.0);
    }

    #[test]
    fn test_stats_percentile_ordering() {
        let mut tracker = LatencyTracker::new();
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            tracker.record("provider", Duration::from_millis(ms), true);
        }
        let stats = tracker.stats("provider").unwrap();
        assert!(stats.p50 <= stats.p90);
        assert!(stats.p90 <= stats.p99);
    }

    #[test]
    fn test_fastest_provider_multi() {
        let mut tracker = LatencyTracker::new();
        tracker.record("slow", Duration::from_millis(500), true);
        tracker.record("fast", Duration::from_millis(10), true);
        tracker.record("medium", Duration::from_millis(200), true);
        assert_eq!(tracker.fastest_provider(), Some("fast".to_string()));
    }
}
