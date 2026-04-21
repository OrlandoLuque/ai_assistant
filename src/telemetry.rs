//! Optional telemetry for monitoring and analytics
//!
//! Opt-in metrics collection for performance monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub collect_latency: bool,
    pub collect_tokens: bool,
    pub collect_errors: bool,
    pub sample_rate: f64,
    pub flush_interval: Duration,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            collect_latency: true,
            collect_tokens: true,
            collect_errors: true,
            sample_rate: 1.0,
            flush_interval: Duration::from_secs(60),
        }
    }
}

/// A telemetry event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub name: String,
    pub timestamp: u64,
    pub duration_ms: Option<u64>,
    pub properties: HashMap<String, String>,
    pub metrics: HashMap<String, f64>,
}

impl TelemetryEvent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms: None,
            properties: HashMap::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }
}

/// Telemetry collector
pub struct TelemetryCollector {
    config: TelemetryConfig,
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
    aggregated: Arc<Mutex<AggregatedMetrics>>,
}

#[derive(Debug, Clone, Default)]
pub struct AggregatedMetrics {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_errors: usize,
    pub avg_latency_ms: f64,
    pub latency_samples: usize,
    /// Cumulative count of user-stall events detected by the in-crate
    /// [`crate::stall_detection`] heuristic. Only incremented when the
    /// `stall-detection` feature is enabled; otherwise remains zero.
    pub user_stall_events_total: u64,
    /// Cumulative count of sub-agent spawn requests recorded via
    /// [`TelemetryCollector::record_sub_agent_spawn`]. Only incremented when
    /// the `sub-agents` feature is enabled; otherwise remains zero.
    pub sub_agents_spawned_total: u64,
    /// Cumulative count of sub-agent completions recorded via
    /// [`TelemetryCollector::record_sub_agent_complete`] with a successful
    /// status. Only incremented when the `sub-agents` feature is enabled.
    pub sub_agents_completed_total: u64,
}

impl TelemetryCollector {
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            aggregated: Arc::new(Mutex::new(AggregatedMetrics::default())),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn record(&self, event: TelemetryEvent) {
        if !self.config.enabled {
            return;
        }

        // Deterministic sample rate check based on event name + timestamp
        if self.config.sample_rate < 1.0 {
            let trace_id = format!("{}:{}", event.name, event.timestamp);
            if !should_sample(&trace_id, self.config.sample_rate) {
                return;
            }
        }

        // Update aggregated metrics
        {
            let mut agg = self.aggregated.lock().unwrap_or_else(|e| e.into_inner());
            agg.total_requests += 1;

            if let Some(tokens) = event.metrics.get("tokens") {
                agg.total_tokens += *tokens as usize;
            }

            if event.properties.get("error").is_some() {
                agg.total_errors += 1;
            }

            if let Some(dur) = event.duration_ms {
                let n = agg.latency_samples as f64;
                agg.avg_latency_ms = (agg.avg_latency_ms * n + dur as f64) / (n + 1.0);
                agg.latency_samples += 1;
            }
        }

        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    }

    pub fn record_request(&self, provider: &str, model: &str, duration: Duration, tokens: usize) {
        self.record(
            TelemetryEvent::new("request")
                .with_duration(duration)
                .with_property("provider", provider)
                .with_property("model", model)
                .with_metric("tokens", tokens as f64),
        );
    }

    pub fn record_error(&self, provider: &str, error: &str) {
        self.record(
            TelemetryEvent::new("error")
                .with_property("provider", provider)
                .with_property("error", error),
        );
    }

    /// Record a user-stall event detected by the in-crate
    /// [`crate::stall_detection`] heuristic. Increments
    /// `user_stall_events_total` on the aggregated metrics and logs a
    /// structured event with the signal name (e.g. `"Frustrated"` or
    /// `"RepeatedToolCall"`).
    ///
    /// The signal is intentionally a `&str` instead of a typed enum so this
    /// method is callable whether or not the `stall-detection` feature is
    /// compiled in — telemetry stays a thin collector, the heuristic stays
    /// behind the feature.
    pub fn record_user_stall(&self, signal: &str) {
        {
            let mut agg = self.aggregated.lock().unwrap_or_else(|e| e.into_inner());
            agg.user_stall_events_total = agg.user_stall_events_total.saturating_add(1);
        }
        self.record(TelemetryEvent::new("user_stall_detected").with_property("signal", signal));
    }

    /// Record that a sub-agent was spawned (see [`crate::sub_agents`]).
    /// Increments `sub_agents_spawned_total` and emits a
    /// `sub_agent_spawned` event with `kind` and `isolation` properties.
    ///
    /// `kind` and `isolation` are `&str` (not typed enums) for the same
    /// reason as [`Self::record_user_stall`]: telemetry stays callable
    /// without the `sub-agents` feature compiled in.
    pub fn record_sub_agent_spawn(&self, kind: &str, isolation: &str) {
        {
            let mut agg = self.aggregated.lock().unwrap_or_else(|e| e.into_inner());
            agg.sub_agents_spawned_total = agg.sub_agents_spawned_total.saturating_add(1);
        }
        self.record(
            TelemetryEvent::new("sub_agent_spawned")
                .with_property("kind", kind)
                .with_property("isolation", isolation),
        );
    }

    /// Record that a sub-agent finished. Increments
    /// `sub_agents_completed_total` only when `success` is true and emits a
    /// `sub_agent_completed` event with `kind` + `status` properties.
    pub fn record_sub_agent_complete(&self, kind: &str, status: &str, success: bool) {
        if success {
            let mut agg = self.aggregated.lock().unwrap_or_else(|e| e.into_inner());
            agg.sub_agents_completed_total = agg.sub_agents_completed_total.saturating_add(1);
        }
        self.record(
            TelemetryEvent::new("sub_agent_completed")
                .with_property("kind", kind)
                .with_property("status", status),
        );
    }

    /// Record a faithfulness check event.
    pub fn record_faithfulness_check(&self, score: f64, claims_count: usize, duration: Duration) {
        self.record(
            TelemetryEvent::new("faithfulness_check")
                .with_duration(duration)
                .with_metric("score", score)
                .with_metric("claims_count", claims_count as f64),
        );
    }

    /// Record an academic search event.
    pub fn record_academic_search(&self, provider: &str, query: &str, results_count: usize) {
        self.record(
            TelemetryEvent::new("academic_search")
                .with_property("provider", provider)
                .with_property("query", query)
                .with_metric("results_count", results_count as f64),
        );
    }

    /// Record a quality gate run event.
    pub fn record_quality_gate_run(&self, passed: bool, score: f64) {
        self.record(
            TelemetryEvent::new("quality_gate_run")
                .with_property("passed", if passed { "true" } else { "false" })
                .with_metric("score", score),
        );
    }

    /// Record a Chain-of-Verification event.
    pub fn record_cove_verification(&self, corrections: usize, accuracy: f64, duration: Duration) {
        self.record(
            TelemetryEvent::new("cove_verification")
                .with_duration(duration)
                .with_metric("corrections", corrections as f64)
                .with_metric("accuracy", accuracy),
        );
    }

    /// Record an abstention event.
    pub fn record_abstention(&self, reason: &str, confidence: f64) {
        self.record(
            TelemetryEvent::new("abstention")
                .with_property("reason", reason)
                .with_metric("confidence", confidence),
        );
    }

    pub fn get_aggregated(&self) -> AggregatedMetrics {
        self.aggregated
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn flush(&self) -> Vec<TelemetryEvent> {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *events)
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new(TelemetryConfig::default())
    }
}

/// Timed operation helper
pub struct TimedOperation {
    start: Instant,
    collector: Arc<TelemetryCollector>,
    event_name: String,
}

impl TimedOperation {
    pub fn start(collector: Arc<TelemetryCollector>, name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            collector,
            event_name: name.into(),
        }
    }

    pub fn finish(self) -> Duration {
        let duration = self.start.elapsed();
        self.collector
            .record(TelemetryEvent::new(&self.event_name).with_duration(duration));
        duration
    }
}

/// Compute a deterministic FNV-1a hash of the given bytes, returning a u64.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministic sampling decision based on a trace identifier and a rate in [0.0, 1.0].
///
/// Hashes the `trace_id` with FNV-1a and checks whether the hash falls within
/// the acceptance range.  The same `trace_id` always produces the same decision.
fn should_sample(trace_id: &str, rate: f64) -> bool {
    if rate >= 1.0 {
        return true;
    }
    if rate <= 0.0 {
        return false;
    }
    let hash = fnv1a_hash(trace_id.as_bytes());
    (hash % 10000) < (rate * 10000.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_disabled() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());
        collector.record(TelemetryEvent::new("test"));
        assert!(collector.flush().is_empty());
    }

    #[test]
    fn test_telemetry_enabled() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);

        collector.record_request("ollama", "llama2", Duration::from_millis(100), 50);

        let agg = collector.get_aggregated();
        assert_eq!(agg.total_requests, 1);
        assert_eq!(agg.total_tokens, 50);
    }

    #[test]
    fn test_deterministic_sampling() {
        // The same trace_id with the same rate must always produce the same decision
        let trace = "request-abc-12345";
        let rate = 0.5;

        let first = should_sample(trace, rate);
        for _ in 0..100 {
            assert_eq!(
                should_sample(trace, rate),
                first,
                "should_sample must be deterministic for the same trace_id"
            );
        }
    }

    #[test]
    fn test_event_builder() {
        let event = TelemetryEvent::new("my_event")
            .with_duration(Duration::from_millis(42))
            .with_property("key", "value")
            .with_metric("score", 0.95);
        assert_eq!(event.name, "my_event");
        assert_eq!(event.duration_ms, Some(42));
        assert_eq!(event.properties.get("key").unwrap(), "value");
        assert!((event.metrics["score"] - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_error() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);
        collector.record_error("openai", "timeout");
        let agg = collector.get_aggregated();
        assert_eq!(agg.total_requests, 1);
        assert_eq!(agg.total_errors, 1);
    }

    #[test]
    fn test_record_user_stall_increments_counter() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);

        collector.record_user_stall("Frustrated");
        collector.record_user_stall("RepeatedToolCall");

        let agg = collector.get_aggregated();
        assert_eq!(agg.user_stall_events_total, 2);
        assert_eq!(agg.total_requests, 2);
    }

    #[test]
    fn test_record_sub_agent_spawn_increments_counter() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);

        collector.record_sub_agent_spawn("Fork", "InProcess");
        collector.record_sub_agent_spawn("Teammate", "ContextIsolated");
        collector.record_sub_agent_spawn("Explore", "InProcess");

        let agg = collector.get_aggregated();
        assert_eq!(agg.sub_agents_spawned_total, 3);
        assert_eq!(agg.sub_agents_completed_total, 0);
    }

    #[test]
    fn test_record_sub_agent_complete_counts_only_success() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);

        collector.record_sub_agent_complete("Fork", "Completed", true);
        collector.record_sub_agent_complete("Fork", "Failed", false);
        collector.record_sub_agent_complete("Teammate", "Cancelled", false);
        collector.record_sub_agent_complete("Teammate", "Completed", true);

        let agg = collector.get_aggregated();
        assert_eq!(agg.sub_agents_completed_total, 2);
    }

    #[test]
    fn test_flush() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = TelemetryCollector::new(config);
        collector.record(TelemetryEvent::new("a"));
        collector.record(TelemetryEvent::new("b"));
        let events = collector.flush();
        assert_eq!(events.len(), 2);
        assert!(collector.flush().is_empty());
    }

    #[test]
    fn test_timed_operation() {
        let config = TelemetryConfig {
            enabled: true,
            ..Default::default()
        };
        let collector = Arc::new(TelemetryCollector::new(config));
        let op = TimedOperation::start(collector.clone(), "op1");
        let duration = op.finish();
        assert!(duration.as_nanos() > 0);
        let events = collector.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "op1");
    }

    #[test]
    fn test_is_enabled() {
        let default = TelemetryCollector::default();
        assert!(!default.is_enabled());
        let enabled = TelemetryCollector::new(TelemetryConfig {
            enabled: true,
            ..Default::default()
        });
        assert!(enabled.is_enabled());
    }

    #[test]
    fn test_sampling_edge_cases() {
        assert!(!should_sample("anything", 0.0));
        assert!(should_sample("anything", 1.0));
        assert!(should_sample("test", 1.5));
    }

    #[test]
    fn test_approximate_sampling_rate() {
        let rate = 0.5;
        let total = 1000;
        let sampled = (0..total)
            .filter(|i| should_sample(&format!("trace-{}", i), rate))
            .count();

        // With 1000 unique trace IDs at rate 0.5, expect roughly 40-60% sampled
        assert!(
            sampled >= 400 && sampled <= 600,
            "Expected ~50% sampling rate, got {}/1000 = {:.1}%",
            sampled,
            sampled as f64 / 10.0
        );
    }
}
