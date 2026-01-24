//! Optional telemetry for monitoring and analytics
//!
//! Opt-in metrics collection for performance monitoring.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
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

        // Sample rate check
        if self.config.sample_rate < 1.0 {
            if rand::random::<f64>() > self.config.sample_rate {
                return;
            }
        }

        // Update aggregated metrics
        {
            let mut agg = self.aggregated.lock().unwrap();
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

        self.events.lock().unwrap().push(event);
    }

    pub fn record_request(&self, provider: &str, model: &str, duration: Duration, tokens: usize) {
        self.record(TelemetryEvent::new("request")
            .with_duration(duration)
            .with_property("provider", provider)
            .with_property("model", model)
            .with_metric("tokens", tokens as f64));
    }

    pub fn record_error(&self, provider: &str, error: &str) {
        self.record(TelemetryEvent::new("error")
            .with_property("provider", provider)
            .with_property("error", error));
    }

    pub fn get_aggregated(&self) -> AggregatedMetrics {
        self.aggregated.lock().unwrap().clone()
    }

    pub fn flush(&self) -> Vec<TelemetryEvent> {
        let mut events = self.events.lock().unwrap();
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
        self.collector.record(TelemetryEvent::new(&self.event_name).with_duration(duration));
        duration
    }
}

mod rand {
    pub fn random<T>() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
        (nanos as f64) / 1_000_000_000.0
    }
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
        let config = TelemetryConfig { enabled: true, ..Default::default() };
        let collector = TelemetryCollector::new(config);

        collector.record_request("ollama", "llama2", Duration::from_millis(100), 50);

        let agg = collector.get_aggregated();
        assert_eq!(agg.total_requests, 1);
        assert_eq!(agg.total_tokens, 50);
    }
}
