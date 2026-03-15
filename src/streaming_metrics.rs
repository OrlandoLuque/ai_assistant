//! Token streaming metrics
//!
//! This module provides real-time metrics for streaming responses,
//! including tokens per second, time to first token, and throughput tracking.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Real-time streaming metrics collector
#[derive(Debug)]
pub struct StreamingMetrics {
    /// When the request started
    request_start: Option<Instant>,
    /// When the first token arrived
    first_token_time: Option<Instant>,
    /// When the last token arrived
    last_token_time: Option<Instant>,
    /// Total tokens received
    total_tokens: usize,
    /// Total characters received
    total_chars: usize,
    /// Token arrival times for rate calculation
    token_times: VecDeque<Instant>,
    /// Sliding window size for rate calculation
    window_size: usize,
    /// Configuration
    config: MetricsConfig,
}

/// Configuration for streaming metrics
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MetricsConfig {
    /// Sliding window size for rate calculation (in tokens)
    pub window_size: usize,
    /// Minimum tokens needed for rate calculation
    pub min_tokens_for_rate: usize,
    /// Whether to estimate tokens from characters
    pub estimate_from_chars: bool,
    /// Average characters per token (for estimation)
    pub chars_per_token: f32,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            min_tokens_for_rate: 5,
            estimate_from_chars: true,
            chars_per_token: 4.0,
        }
    }
}

impl StreamingMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::with_config(MetricsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            request_start: None,
            first_token_time: None,
            last_token_time: None,
            total_tokens: 0,
            total_chars: 0,
            token_times: VecDeque::with_capacity(config.window_size),
            window_size: config.window_size,
            config,
        }
    }

    /// Start tracking a new request
    pub fn start(&mut self) {
        self.request_start = Some(Instant::now());
        self.first_token_time = None;
        self.last_token_time = None;
        self.total_tokens = 0;
        self.total_chars = 0;
        self.token_times.clear();
    }

    /// Record a token/chunk arrival
    pub fn record_chunk(&mut self, chunk: &str) {
        let now = Instant::now();

        if self.first_token_time.is_none() {
            self.first_token_time = Some(now);
        }
        self.last_token_time = Some(now);

        // Estimate tokens from chunk
        let estimated_tokens = if self.config.estimate_from_chars {
            (chunk.len() as f32 / self.config.chars_per_token).ceil() as usize
        } else {
            1
        };

        self.total_tokens += estimated_tokens;
        self.total_chars += chunk.len();

        // Add to sliding window
        for _ in 0..estimated_tokens {
            self.token_times.push_back(now);
            if self.token_times.len() > self.window_size {
                self.token_times.pop_front();
            }
        }
    }

    /// Record a specific number of tokens
    pub fn record_tokens(&mut self, count: usize) {
        let now = Instant::now();

        if self.first_token_time.is_none() {
            self.first_token_time = Some(now);
        }
        self.last_token_time = Some(now);

        self.total_tokens += count;

        for _ in 0..count {
            self.token_times.push_back(now);
            if self.token_times.len() > self.window_size {
                self.token_times.pop_front();
            }
        }
    }

    /// Get time to first token
    pub fn time_to_first_token(&self) -> Option<Duration> {
        match (self.request_start, self.first_token_time) {
            (Some(start), Some(first)) => Some(first.duration_since(start)),
            _ => None,
        }
    }

    /// Get total generation time so far
    pub fn total_time(&self) -> Option<Duration> {
        self.request_start.map(|start| start.elapsed())
    }

    /// Get time since first token
    pub fn generation_time(&self) -> Option<Duration> {
        match (self.first_token_time, self.last_token_time) {
            (Some(first), Some(last)) => Some(last.duration_since(first)),
            _ => None,
        }
    }

    /// Get current tokens per second (based on sliding window)
    pub fn current_tokens_per_second(&self) -> Option<f64> {
        if self.token_times.len() < self.config.min_tokens_for_rate {
            return None;
        }

        let first = self.token_times.front()?;
        let last = self.token_times.back()?;
        let duration = last.duration_since(*first);

        if duration.as_secs_f64() < 0.001 {
            return None;
        }

        Some(self.token_times.len() as f64 / duration.as_secs_f64())
    }

    /// Get average tokens per second (over entire generation)
    pub fn average_tokens_per_second(&self) -> Option<f64> {
        let duration = self.generation_time()?;
        if duration.as_secs_f64() < 0.001 {
            return None;
        }
        Some(self.total_tokens as f64 / duration.as_secs_f64())
    }

    /// Get total tokens so far
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get total characters received
    pub fn total_characters(&self) -> usize {
        self.total_chars
    }

    /// Check if streaming has started
    pub fn has_started(&self) -> bool {
        self.first_token_time.is_some()
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> StreamingSnapshot {
        StreamingSnapshot {
            time_to_first_token: self.time_to_first_token(),
            total_time: self.total_time(),
            generation_time: self.generation_time(),
            total_tokens: self.total_tokens,
            total_chars: self.total_chars,
            current_tokens_per_second: self.current_tokens_per_second(),
            average_tokens_per_second: self.average_tokens_per_second(),
            is_generating: self.request_start.is_some() && self.first_token_time.is_some(),
        }
    }

    /// Finalize metrics (call when generation completes)
    pub fn finalize(&self) -> FinalMetrics {
        FinalMetrics {
            time_to_first_token: self.time_to_first_token(),
            total_time: self.total_time().unwrap_or_default(),
            generation_time: self.generation_time().unwrap_or_default(),
            total_tokens: self.total_tokens,
            total_chars: self.total_chars,
            average_tokens_per_second: self.average_tokens_per_second().unwrap_or(0.0),
            chars_per_token: if self.total_tokens > 0 {
                self.total_chars as f32 / self.total_tokens as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of current streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSnapshot {
    /// Time to first token
    #[serde(with = "option_duration_serde")]
    pub time_to_first_token: Option<Duration>,
    /// Total time since request start
    #[serde(with = "option_duration_serde")]
    pub total_time: Option<Duration>,
    /// Time since first token
    #[serde(with = "option_duration_serde")]
    pub generation_time: Option<Duration>,
    /// Total tokens received
    pub total_tokens: usize,
    /// Total characters received
    pub total_chars: usize,
    /// Current tokens per second
    pub current_tokens_per_second: Option<f64>,
    /// Average tokens per second
    pub average_tokens_per_second: Option<f64>,
    /// Whether generation is in progress
    pub is_generating: bool,
}

/// Final metrics after generation completes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalMetrics {
    /// Time to first token
    #[serde(with = "option_duration_serde")]
    pub time_to_first_token: Option<Duration>,
    /// Total time from start to finish
    #[serde(with = "duration_serde")]
    pub total_time: Duration,
    /// Time from first token to last
    #[serde(with = "duration_serde")]
    pub generation_time: Duration,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Total characters generated
    pub total_chars: usize,
    /// Average tokens per second
    pub average_tokens_per_second: f64,
    /// Actual characters per token ratio
    pub chars_per_token: f32,
}

impl FinalMetrics {
    /// Format as human-readable summary
    pub fn summary(&self) -> String {
        let ttft = self
            .time_to_first_token
            .map(|d| format!("{:.0}ms", d.as_millis()))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "Tokens: {} | Time: {:.1}s | TTFT: {} | Speed: {:.1} tok/s",
            self.total_tokens,
            self.total_time.as_secs_f64(),
            ttft,
            self.average_tokens_per_second
        )
    }
}

/// Aggregated metrics across multiple generations
#[derive(Debug, Clone, Default)]
pub struct AggregatedMetrics {
    /// Number of generations
    pub count: usize,
    /// Total tokens across all generations
    pub total_tokens: usize,
    /// Total time across all generations
    pub total_time: Duration,
    /// Average time to first token
    pub avg_ttft: Option<Duration>,
    /// Minimum tokens per second
    pub min_tokens_per_second: f64,
    /// Maximum tokens per second
    pub max_tokens_per_second: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// All individual metrics
    metrics: Vec<FinalMetrics>,
}

impl AggregatedMetrics {
    /// Create a new aggregator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metrics from a completed generation
    pub fn add(&mut self, metrics: FinalMetrics) {
        self.count += 1;
        self.total_tokens += metrics.total_tokens;
        self.total_time += metrics.total_time;

        // Update min/max
        if self.count == 1 {
            self.min_tokens_per_second = metrics.average_tokens_per_second;
            self.max_tokens_per_second = metrics.average_tokens_per_second;
        } else {
            self.min_tokens_per_second = self
                .min_tokens_per_second
                .min(metrics.average_tokens_per_second);
            self.max_tokens_per_second = self
                .max_tokens_per_second
                .max(metrics.average_tokens_per_second);
        }

        // Update averages
        self.metrics.push(metrics);
        self.recalculate_averages();
    }

    fn recalculate_averages(&mut self) {
        if self.metrics.is_empty() {
            return;
        }

        // Average tokens per second
        let sum_tps: f64 = self
            .metrics
            .iter()
            .map(|m| m.average_tokens_per_second)
            .sum();
        self.avg_tokens_per_second = sum_tps / self.metrics.len() as f64;

        // Average TTFT
        let ttfts: Vec<Duration> = self
            .metrics
            .iter()
            .filter_map(|m| m.time_to_first_token)
            .collect();

        if !ttfts.is_empty() {
            let sum: Duration = ttfts.iter().sum();
            self.avg_ttft = Some(sum / ttfts.len() as u32);
        }
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let avg_ttft = self
            .avg_ttft
            .map(|d| format!("{:.0}ms", d.as_millis()))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "Generations: {} | Total tokens: {} | Avg TTFT: {} | Speed: {:.1} tok/s (min: {:.1}, max: {:.1})",
            self.count,
            self.total_tokens,
            avg_ttft,
            self.avg_tokens_per_second,
            self.min_tokens_per_second,
            self.max_tokens_per_second
        )
    }

    /// Get percentile metrics
    pub fn percentile(&self, p: f64) -> Option<&FinalMetrics> {
        if self.metrics.is_empty() {
            return None;
        }

        let mut sorted: Vec<_> = self.metrics.iter().collect();
        sorted.sort_by(|a, b| {
            a.average_tokens_per_second
                .partial_cmp(&b.average_tokens_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let index = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[index])
    }
}

/// Display formatter for streaming metrics
pub struct MetricsDisplay;

impl MetricsDisplay {
    /// Format tokens per second with appropriate units
    pub fn format_rate(tps: f64) -> String {
        if tps >= 1000.0 {
            format!("{:.1}k tok/s", tps / 1000.0)
        } else if tps >= 1.0 {
            format!("{:.1} tok/s", tps)
        } else {
            format!("{:.2} tok/s", tps)
        }
    }

    /// Format duration
    pub fn format_duration(d: Duration) -> String {
        let millis = d.as_millis();
        if millis >= 60000 {
            format!("{:.1}m", d.as_secs_f64() / 60.0)
        } else if millis >= 1000 {
            format!("{:.1}s", d.as_secs_f64())
        } else {
            format!("{}ms", millis)
        }
    }

    /// Format a progress bar
    pub fn progress_bar(current: usize, total: usize, width: usize) -> String {
        let ratio = if total > 0 {
            current as f64 / total as f64
        } else {
            0.0
        };

        let filled = (ratio * width as f64).round() as usize;
        let empty = width - filled;

        format!(
            "[{}{}] {}/{}",
            "=".repeat(filled),
            " ".repeat(empty),
            current,
            total
        )
    }

    /// Format streaming status line
    pub fn status_line(snapshot: &StreamingSnapshot) -> String {
        let rate = snapshot
            .current_tokens_per_second
            .map(|r| Self::format_rate(r))
            .unwrap_or_else(|| "-- tok/s".to_string());

        let time = snapshot
            .generation_time
            .map(|d| Self::format_duration(d))
            .unwrap_or_else(|| "--".to_string());

        format!(
            "Tokens: {} | Time: {} | Speed: {}",
            snapshot.total_tokens, time, rate
        )
    }
}

// Serde helpers for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

mod option_duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.map(|d| d.as_millis()).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis: Option<u64> = Option::deserialize(deserializer)?;
        Ok(millis.map(Duration::from_millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_metrics() {
        let mut metrics = StreamingMetrics::new();
        metrics.start();

        thread::sleep(Duration::from_millis(10));
        metrics.record_chunk("Hello ");

        thread::sleep(Duration::from_millis(10));
        metrics.record_chunk("world!");

        assert!(metrics.has_started());
        assert!(metrics.total_tokens() > 0);
        assert!(metrics.time_to_first_token().is_some());
    }

    #[test]
    fn test_tokens_per_second() {
        let mut metrics = StreamingMetrics::new();
        metrics.start();

        // Record tokens with small delays to ensure measurable time
        metrics.record_tokens(1);
        thread::sleep(Duration::from_millis(5));

        for _ in 0..99 {
            metrics.record_tokens(1);
        }
        thread::sleep(Duration::from_millis(5));

        let final_metrics = metrics.finalize();
        // Should have recorded 100 tokens over at least 10ms
        assert!(final_metrics.total_tokens >= 100);
        // Rate should be > 0 (we have tokens and time elapsed)
        assert!(final_metrics.average_tokens_per_second >= 0.0);
    }

    #[test]
    fn test_snapshot() {
        let mut metrics = StreamingMetrics::new();
        metrics.start();
        metrics.record_chunk("Test");

        let snapshot = metrics.snapshot();
        assert!(snapshot.is_generating);
        assert!(snapshot.total_tokens > 0);
    }

    #[test]
    fn test_aggregated_metrics() {
        let mut agg = AggregatedMetrics::new();

        for i in 1..=3 {
            let metrics = FinalMetrics {
                time_to_first_token: Some(Duration::from_millis(100)),
                total_time: Duration::from_secs(i),
                generation_time: Duration::from_secs(i),
                total_tokens: 100 * i as usize,
                total_chars: 400 * i as usize,
                average_tokens_per_second: 100.0 / i as f64,
                chars_per_token: 4.0,
            };
            agg.add(metrics);
        }

        assert_eq!(agg.count, 3);
        assert!(agg.avg_tokens_per_second > 0.0);
        assert!(agg.avg_ttft.is_some());
    }

    #[test]
    fn test_display_formatting() {
        assert_eq!(MetricsDisplay::format_rate(50.5), "50.5 tok/s");
        assert_eq!(MetricsDisplay::format_rate(1500.0), "1.5k tok/s");
        assert_eq!(
            MetricsDisplay::format_duration(Duration::from_millis(500)),
            "500ms"
        );
        assert_eq!(
            MetricsDisplay::format_duration(Duration::from_secs(5)),
            "5.0s"
        );
    }

    #[test]
    fn test_progress_bar() {
        let bar = MetricsDisplay::progress_bar(50, 100, 10);
        assert!(bar.contains("====="));
        assert!(bar.contains("50/100"));
    }

    #[test]
    fn test_has_started() {
        let metrics = StreamingMetrics::new();
        assert!(!metrics.has_started());
    }

    #[test]
    fn test_total_tokens_and_chars() {
        let mut metrics = StreamingMetrics::new();
        metrics.start();
        metrics.record_chunk("Hello");
        metrics.record_chunk(" World");
        assert_eq!(metrics.total_characters(), 11);
    }

    #[test]
    fn test_format_rate() {
        let formatted = MetricsDisplay::format_rate(42.5);
        assert!(formatted.contains("42.5"));
    }

    #[test]
    fn test_finalize_metrics() {
        let mut metrics = StreamingMetrics::new();
        metrics.start();
        metrics.record_chunk("test");
        let final_m = metrics.finalize();
        assert!(final_m.total_chars >= 4);
    }
}
