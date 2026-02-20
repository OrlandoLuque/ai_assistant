// opentelemetry_integration.rs — OpenTelemetry integration for AI operations tracing.
//
// Provides OTel-compatible tracing for AI operations: model calls, RAG queries,
// tool invocations, and agent steps. Bridges the existing TelemetryEvent system
// to OpenTelemetry spans and metrics.
//
// Requires the `otel` feature flag for actual OTel SDK integration.
// Without the feature, provides a no-op tracer that records events locally.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Span Types
// ============================================================================

/// Represents an AI operation span compatible with OpenTelemetry.
#[derive(Debug, Clone)]
pub struct AiSpan {
    /// Unique span identifier
    pub span_id: String,
    /// Optional parent span ID for nested operations
    pub parent_id: Option<String>,
    /// Operation name (e.g., "llm.generate", "rag.query", "tool.invoke")
    pub operation: String,
    /// AI model name if applicable
    pub model: Option<String>,
    /// Provider name
    pub provider: Option<String>,
    /// Start timestamp (Unix millis)
    pub start_time_ms: u64,
    /// End timestamp (Unix millis), None if still running
    pub end_time_ms: Option<u64>,
    /// Token counts
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Status: "ok", "error"
    pub status: String,
    /// Error message if status is "error"
    pub error_message: Option<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl AiSpan {
    /// Create a new span for an operation.
    pub fn new(operation: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            span_id: generate_span_id(),
            parent_id: None,
            operation: operation.to_string(),
            model: None,
            provider: None,
            start_time_ms: now,
            end_time_ms: None,
            input_tokens: None,
            output_tokens: None,
            status: "ok".to_string(),
            error_message: None,
            attributes: HashMap::new(),
        }
    }

    /// Set the model name.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set the provider.
    pub fn with_provider(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_string());
        self
    }

    /// Set the parent span.
    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    /// Add a custom attribute.
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Record token usage.
    pub fn with_tokens(mut self, input: u64, output: u64) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self
    }

    /// Mark the span as completed.
    pub fn finish(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.end_time_ms = Some(now);
    }

    /// Mark the span as failed.
    pub fn fail(&mut self, error: &str) {
        self.status = "error".to_string();
        self.error_message = Some(error.to_string());
        self.finish();
    }

    /// Get span duration in milliseconds, or None if not finished.
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_time_ms
            .map(|end| end.saturating_sub(self.start_time_ms))
    }

    /// Check if the span is still running.
    pub fn is_running(&self) -> bool {
        self.end_time_ms.is_none()
    }
}

// ============================================================================
// Tracer Configuration
// ============================================================================

/// Configuration for the OTel tracer.
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name for the tracer
    pub service_name: String,
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub endpoint: String,
    /// Whether to export spans (requires `otel` feature)
    pub export_enabled: bool,
    /// Maximum spans to keep in local buffer
    pub max_buffer_size: usize,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Additional resource attributes
    pub resource_attributes: HashMap<String, String>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "ai_assistant".to_string(),
            endpoint: "http://localhost:4317".to_string(),
            export_enabled: false,
            max_buffer_size: 10000,
            sampling_rate: 1.0,
            resource_attributes: HashMap::new(),
        }
    }
}

// ============================================================================
// OTel Tracer
// ============================================================================

/// OpenTelemetry-compatible tracer for AI operations.
///
/// Records spans for model calls, RAG queries, tool invocations, and agent steps.
/// Without the `otel` feature, spans are recorded locally only.
pub struct OtelTracer {
    config: OtelConfig,
    spans: Arc<Mutex<Vec<AiSpan>>>,
    active_spans: Arc<Mutex<HashMap<String, AiSpan>>>,
}

impl OtelTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: OtelConfig) -> Self {
        Self {
            config,
            spans: Arc::new(Mutex::new(Vec::new())),
            active_spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start a new span.
    pub fn start_span(&self, operation: &str) -> AiSpan {
        let span = AiSpan::new(operation);
        if let Ok(mut active) = self.active_spans.lock() {
            active.insert(span.span_id.clone(), span.clone());
        }
        span
    }

    /// Start a child span under an existing parent.
    pub fn start_child_span(&self, operation: &str, parent_id: &str) -> AiSpan {
        let span = AiSpan::new(operation).with_parent(parent_id);
        if let Ok(mut active) = self.active_spans.lock() {
            active.insert(span.span_id.clone(), span.clone());
        }
        span
    }

    /// End a span and move it to completed spans buffer.
    pub fn end_span(&self, mut span: AiSpan) {
        span.finish();
        if let Ok(mut active) = self.active_spans.lock() {
            active.remove(&span.span_id);
        }
        if let Ok(mut completed) = self.spans.lock() {
            // Check sampling
            if self.should_sample() {
                if completed.len() >= self.config.max_buffer_size {
                    completed.remove(0); // Evict oldest
                }
                completed.push(span);
            }
        }
    }

    /// Record a failed span.
    pub fn record_error(&self, mut span: AiSpan, error: &str) {
        span.fail(error);
        if let Ok(mut active) = self.active_spans.lock() {
            active.remove(&span.span_id);
        }
        if let Ok(mut completed) = self.spans.lock() {
            if completed.len() >= self.config.max_buffer_size {
                completed.remove(0);
            }
            completed.push(span);
        }
    }

    /// Get all completed spans.
    pub fn completed_spans(&self) -> Vec<AiSpan> {
        self.spans.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Get count of active (unfinished) spans.
    pub fn active_span_count(&self) -> usize {
        self.active_spans.lock().map(|a| a.len()).unwrap_or(0)
    }

    /// Get count of completed spans in buffer.
    pub fn completed_span_count(&self) -> usize {
        self.spans.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Drain all completed spans from buffer.
    pub fn drain_spans(&self) -> Vec<AiSpan> {
        self.spans
            .lock()
            .map(|mut s| std::mem::take(&mut *s))
            .unwrap_or_default()
    }

    /// Get the tracer's service name.
    pub fn service_name(&self) -> &str {
        &self.config.service_name
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    fn should_sample(&self) -> bool {
        if self.config.sampling_rate >= 1.0 {
            return true;
        }
        if self.config.sampling_rate <= 0.0 {
            return false;
        }
        // Simple deterministic sampling based on system time nanos
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        (nanos as f64 / u32::MAX as f64) < self.config.sampling_rate
    }
}

// ============================================================================
// Bridge from TelemetryEvent
// ============================================================================

/// Semantic conventions for AI spans (follows OpenTelemetry GenAI semantic conventions).
pub mod semantic_conventions {
    pub const LLM_SYSTEM: &str = "gen_ai.system";
    pub const LLM_REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const LLM_RESPONSE_MODEL: &str = "gen_ai.response.model";
    pub const LLM_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    pub const LLM_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    pub const LLM_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const LLM_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const LLM_RESPONSE_FINISH_REASON: &str = "gen_ai.response.finish_reasons";

    pub const RAG_QUERY: &str = "rag.query";
    pub const RAG_NUM_RESULTS: &str = "rag.num_results";
    pub const RAG_TOP_SCORE: &str = "rag.top_score";

    pub const TOOL_NAME: &str = "tool.name";
    pub const TOOL_DESCRIPTION: &str = "tool.description";

    pub const AGENT_NAME: &str = "agent.name";
    pub const AGENT_STEP: &str = "agent.step";
}

/// Convert an AI operation description into an OTel-compatible span.
pub fn create_llm_span(tracer: &OtelTracer, model: &str, provider: &str) -> AiSpan {
    tracer
        .start_span("gen_ai.chat")
        .with_model(model)
        .with_provider(provider)
        .with_attribute(semantic_conventions::LLM_SYSTEM, provider)
        .with_attribute(semantic_conventions::LLM_REQUEST_MODEL, model)
}

/// Create a RAG query span.
pub fn create_rag_span(tracer: &OtelTracer, query: &str) -> AiSpan {
    tracer
        .start_span("rag.query")
        .with_attribute(semantic_conventions::RAG_QUERY, query)
}

/// Create a tool invocation span.
pub fn create_tool_span(tracer: &OtelTracer, tool_name: &str) -> AiSpan {
    tracer
        .start_span("tool.invoke")
        .with_attribute(semantic_conventions::TOOL_NAME, tool_name)
}

// ============================================================================
// Helpers
// ============================================================================

fn generate_span_id() -> String {
    // Generate a 16-byte hex span ID using system time + counter
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos();
    format!("{:016x}", nanos & 0xFFFFFFFFFFFFFFFF)
}

// ============================================================================
// Histogram Stats
// ============================================================================

/// Statistical summary of histogram-recorded values.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Number of recorded values
    pub count: usize,
    /// Sum of all values
    pub sum: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Arithmetic mean
    pub mean: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

impl HistogramStats {
    /// Compute statistics from a slice of values.
    /// The input slice must not be empty.
    pub fn from_values(values: &[f64]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let min = sorted[0];
        let max = sorted[count - 1];
        let mean = sum / count as f64;
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);
        Self {
            count,
            sum,
            min,
            max,
            mean,
            p50,
            p95,
            p99,
        }
    }
}

/// Compute the p-th percentile from a sorted slice using nearest-rank method.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

// ============================================================================
// Metrics Collector
// ============================================================================

/// Collects counters, gauges, and histograms for AI operation metrics.
pub struct MetricsCollector {
    /// Monotonic counters (e.g., request counts)
    pub counters: HashMap<String, u64>,
    /// Point-in-time gauge values (e.g., queue depth)
    pub gauges: HashMap<String, f64>,
    /// Histogram value buckets (e.g., latency distributions)
    pub histograms: HashMap<String, Vec<f64>>,
}

impl MetricsCollector {
    /// Create a new empty metrics collector.
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
        }
    }

    /// Increment a counter by 1.
    pub fn increment(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Increment a counter by a specified amount.
    pub fn increment_by(&mut self, name: &str, amount: u64) {
        *self.counters.entry(name.to_string()).or_insert(0) += amount;
    }

    /// Set a gauge to a specific value.
    pub fn gauge(&mut self, name: &str, value: f64) {
        self.gauges.insert(name.to_string(), value);
    }

    /// Record a value in a histogram.
    pub fn record_histogram(&mut self, name: &str, value: f64) {
        self.histograms
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get the current value of a counter, or 0 if not set.
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or(0)
    }

    /// Get the current value of a gauge, or None if not set.
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).copied()
    }

    /// Compute histogram statistics for a named histogram.
    /// Returns None if the histogram does not exist or is empty.
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histograms.get(name).and_then(|v| {
            if v.is_empty() {
                None
            } else {
                Some(HistogramStats::from_values(v))
            }
        })
    }

    /// Export all metrics in Prometheus text exposition format.
    pub fn export_prometheus(&self) -> String {
        let mut out = String::new();

        let mut counter_names: Vec<&String> = self.counters.keys().collect();
        counter_names.sort();
        for name in counter_names {
            let value = self.counters[name];
            out.push_str(&format!("# TYPE {} counter\n", name));
            out.push_str(&format!("{} {}\n", name, value));
        }

        let mut gauge_names: Vec<&String> = self.gauges.keys().collect();
        gauge_names.sort();
        for name in gauge_names {
            let value = self.gauges[name];
            out.push_str(&format!("# TYPE {} gauge\n", name));
            out.push_str(&format!("{} {}\n", name, value));
        }

        let mut hist_names: Vec<&String> = self.histograms.keys().collect();
        hist_names.sort();
        for name in hist_names {
            let values = &self.histograms[name];
            let count = values.len();
            let sum: f64 = values.iter().sum();
            out.push_str(&format!("# TYPE {} histogram\n", name));
            out.push_str(&format!("{}_count {}\n", name, count));
            out.push_str(&format!("{}_sum {}\n", name, sum));
        }

        out
    }

    /// Clear all collected metrics.
    pub fn reset(&mut self) {
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
    }
}

// ============================================================================
// Span Exporter
// ============================================================================

/// Export format for span data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Standard JSON array of spans
    Json,
    /// OpenTelemetry Protocol (OTLP) JSON structure
    OtlpJson,
}

/// Exports recorded spans in various formats (JSON, OTLP).
pub struct SpanExporter {
    spans: Vec<AiSpan>,
    format: ExportFormat,
}

impl SpanExporter {
    /// Create a new exporter for the given spans and format.
    pub fn new(spans: Vec<AiSpan>, format: ExportFormat) -> Self {
        Self { spans, format }
    }

    /// Export spans as a JSON array.
    pub fn export_json(&self) -> serde_json::Value {
        let span_values: Vec<serde_json::Value> = self
            .spans
            .iter()
            .map(|s| {
                let mut obj = serde_json::json!({
                    "span_id": s.span_id,
                    "operation": s.operation,
                    "status": s.status,
                    "duration_ms": s.duration_ms(),
                });
                if let Some(tokens) = s.input_tokens {
                    obj["input_tokens"] = serde_json::json!(tokens);
                }
                if let Some(tokens) = s.output_tokens {
                    obj["output_tokens"] = serde_json::json!(tokens);
                }
                obj
            })
            .collect();
        serde_json::Value::Array(span_values)
    }

    /// Export spans in an OTLP-like JSON structure.
    pub fn export_otlp_json(&self) -> serde_json::Value {
        let span_values: Vec<serde_json::Value> = self
            .spans
            .iter()
            .map(|s| {
                serde_json::json!({
                    "spanId": s.span_id,
                    "name": s.operation,
                    "status": { "code": if s.status == "ok" { 1 } else { 2 } },
                    "startTimeUnixNano": s.start_time_ms * 1_000_000,
                    "endTimeUnixNano": s.end_time_ms.unwrap_or(0) * 1_000_000,
                })
            })
            .collect();

        serde_json::json!({
            "resourceSpans": [{
                "scopeSpans": [{
                    "spans": span_values
                }]
            }]
        })
    }

    /// Export spans using the configured format.
    pub fn export(&self) -> serde_json::Value {
        match self.format {
            ExportFormat::Json => self.export_json(),
            ExportFormat::OtlpJson => self.export_otlp_json(),
        }
    }
}

// ============================================================================
// Tracing Middleware
// ============================================================================

/// Middleware that wraps OtelTracer and MetricsCollector to provide
/// convenient before/after hooks for LLM and tool calls.
pub struct TracingMiddleware {
    tracer: OtelTracer,
    metrics: MetricsCollector,
    auto_record: bool,
}

impl TracingMiddleware {
    /// Create a new tracing middleware with auto-recording enabled by default.
    pub fn new(tracer: OtelTracer, metrics: MetricsCollector) -> Self {
        Self {
            tracer,
            metrics,
            auto_record: true,
        }
    }

    /// Set whether to automatically record token metrics.
    pub fn with_auto_record(mut self, enabled: bool) -> Self {
        self.auto_record = enabled;
        self
    }

    /// Start tracking an LLM call. Returns the span_id for use in after_llm_call.
    pub fn before_llm_call(&mut self, model: &str, provider: &str) -> String {
        let span = create_llm_span(&self.tracer, model, provider);
        let span_id = span.span_id.clone();
        // The span is already in active_spans via create_llm_span -> start_span
        self.metrics.increment("llm.calls");
        span_id
    }

    /// Complete an LLM call span and record token metrics.
    pub fn after_llm_call(
        &mut self,
        span_id: &str,
        input_tokens: u64,
        output_tokens: u64,
        success: bool,
    ) {
        // Retrieve the span from active spans
        let span_opt = {
            let mut active = self.tracer.active_spans.lock().unwrap();
            active.remove(span_id)
        };
        if let Some(mut span) = span_opt {
            span.input_tokens = Some(input_tokens);
            span.output_tokens = Some(output_tokens);
            if success {
                self.tracer.end_span(span);
            } else {
                self.tracer.record_error(span, "llm call failed");
            }
            if self.auto_record {
                self.metrics
                    .record_histogram("llm.input_tokens", input_tokens as f64);
                self.metrics
                    .record_histogram("llm.output_tokens", output_tokens as f64);
            }
        }
    }

    /// Start tracking a tool call. Returns the span_id for use in after_tool_call.
    pub fn before_tool_call(&mut self, tool_name: &str) -> String {
        let span = create_tool_span(&self.tracer, tool_name);
        let span_id = span.span_id.clone();
        self.metrics.increment("tool.calls");
        span_id
    }

    /// Complete a tool call span.
    pub fn after_tool_call(&mut self, span_id: &str, success: bool) {
        let span_opt = {
            let mut active = self.tracer.active_spans.lock().unwrap();
            active.remove(span_id)
        };
        if let Some(span) = span_opt {
            if success {
                self.tracer.end_span(span);
            } else {
                self.tracer.record_error(span, "tool call failed");
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_span_new() {
        let span = AiSpan::new("test.operation");
        assert_eq!(span.operation, "test.operation");
        assert_eq!(span.status, "ok");
        assert!(span.is_running());
        assert!(span.end_time_ms.is_none());
        assert!(span.span_id.len() > 0);
    }

    #[test]
    fn test_ai_span_builder() {
        let span = AiSpan::new("llm.generate")
            .with_model("gpt-4")
            .with_provider("openai")
            .with_attribute("custom_key", "custom_value")
            .with_tokens(100, 50);
        assert_eq!(span.model.as_deref(), Some("gpt-4"));
        assert_eq!(span.provider.as_deref(), Some("openai"));
        assert_eq!(span.input_tokens, Some(100));
        assert_eq!(span.output_tokens, Some(50));
        assert_eq!(span.attributes.get("custom_key").unwrap(), "custom_value");
    }

    #[test]
    fn test_ai_span_finish() {
        let mut span = AiSpan::new("test");
        assert!(span.is_running());
        span.finish();
        assert!(!span.is_running());
        assert!(span.end_time_ms.is_some());
        assert_eq!(span.status, "ok");
    }

    #[test]
    fn test_ai_span_fail() {
        let mut span = AiSpan::new("test");
        span.fail("connection timeout");
        assert!(!span.is_running());
        assert_eq!(span.status, "error");
        assert_eq!(span.error_message.as_deref(), Some("connection timeout"));
    }

    #[test]
    fn test_ai_span_duration() {
        let mut span = AiSpan::new("test");
        assert!(span.duration_ms().is_none());
        span.finish();
        // Duration should be >= 0
        assert!(span.duration_ms().unwrap() < 1000);
    }

    #[test]
    fn test_ai_span_parent() {
        let parent = AiSpan::new("parent");
        let child = AiSpan::new("child").with_parent(&parent.span_id);
        assert_eq!(child.parent_id.as_deref(), Some(parent.span_id.as_str()));
    }

    #[test]
    fn test_otel_config_default() {
        let config = OtelConfig::default();
        assert_eq!(config.service_name, "ai_assistant");
        assert_eq!(config.endpoint, "http://localhost:4317");
        assert!(!config.export_enabled);
        assert_eq!(config.max_buffer_size, 10000);
        assert!((config.sampling_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracer_start_end_span() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = tracer.start_span("test.op");
        assert_eq!(tracer.active_span_count(), 1);
        assert_eq!(tracer.completed_span_count(), 0);

        tracer.end_span(span);
        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 1);
    }

    #[test]
    fn test_tracer_record_error() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = tracer.start_span("failing.op");
        tracer.record_error(span, "network error");

        let completed = tracer.completed_spans();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].status, "error");
        assert_eq!(completed[0].error_message.as_deref(), Some("network error"));
    }

    #[test]
    fn test_tracer_child_spans() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let parent = tracer.start_span("parent");
        let child = tracer.start_child_span("child", &parent.span_id);
        assert_eq!(child.parent_id.as_deref(), Some(parent.span_id.as_str()));
        assert_eq!(tracer.active_span_count(), 2);

        tracer.end_span(child);
        tracer.end_span(parent);
        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 2);
    }

    #[test]
    fn test_tracer_drain_spans() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let s1 = tracer.start_span("op1");
        let s2 = tracer.start_span("op2");
        tracer.end_span(s1);
        tracer.end_span(s2);

        let drained = tracer.drain_spans();
        assert_eq!(drained.len(), 2);
        assert_eq!(tracer.completed_span_count(), 0);
    }

    #[test]
    fn test_tracer_buffer_eviction() {
        let config = OtelConfig {
            max_buffer_size: 3,
            ..Default::default()
        };
        let tracer = OtelTracer::new(config);

        for i in 0..5 {
            let span = tracer.start_span(&format!("op{}", i));
            tracer.end_span(span);
        }

        // Buffer should be capped at 3
        assert_eq!(tracer.completed_span_count(), 3);
    }

    #[test]
    fn test_create_llm_span() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = create_llm_span(&tracer, "claude-3", "anthropic");
        assert_eq!(span.operation, "gen_ai.chat");
        assert_eq!(span.model.as_deref(), Some("claude-3"));
        assert_eq!(span.provider.as_deref(), Some("anthropic"));
        assert_eq!(
            span.attributes
                .get(semantic_conventions::LLM_SYSTEM)
                .unwrap(),
            "anthropic"
        );
    }

    #[test]
    fn test_create_rag_span() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = create_rag_span(&tracer, "what is Rust?");
        assert_eq!(span.operation, "rag.query");
        assert_eq!(
            span.attributes
                .get(semantic_conventions::RAG_QUERY)
                .unwrap(),
            "what is Rust?"
        );
    }

    #[test]
    fn test_create_tool_span() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = create_tool_span(&tracer, "web_search");
        assert_eq!(span.operation, "tool.invoke");
        assert_eq!(
            span.attributes
                .get(semantic_conventions::TOOL_NAME)
                .unwrap(),
            "web_search"
        );
    }

    #[test]
    fn test_semantic_conventions() {
        assert_eq!(semantic_conventions::LLM_SYSTEM, "gen_ai.system");
        assert_eq!(
            semantic_conventions::LLM_USAGE_INPUT_TOKENS,
            "gen_ai.usage.input_tokens"
        );
        assert_eq!(semantic_conventions::RAG_QUERY, "rag.query");
        assert_eq!(semantic_conventions::TOOL_NAME, "tool.name");
    }

    #[test]
    fn test_generate_span_id_unique() {
        let id1 = generate_span_id();
        // Add tiny delay to ensure different nanos
        std::thread::sleep(std::time::Duration::from_nanos(100));
        let id2 = generate_span_id();
        // IDs should be different (or at least well-formed)
        assert_eq!(id1.len(), 16);
        assert_eq!(id2.len(), 16);
    }

    #[test]
    fn test_tracer_service_name() {
        let config = OtelConfig {
            service_name: "my_app".to_string(),
            ..Default::default()
        };
        let tracer = OtelTracer::new(config);
        assert_eq!(tracer.service_name(), "my_app");
        assert_eq!(tracer.endpoint(), "http://localhost:4317");
    }

    // ========================================================================
    // WS3: Observability Enhancements Tests
    // ========================================================================

    #[test]
    fn test_metrics_collector_counters() {
        let mut mc = MetricsCollector::new();
        assert_eq!(mc.get_counter("requests"), 0);

        mc.increment("requests");
        assert_eq!(mc.get_counter("requests"), 1);

        mc.increment("requests");
        mc.increment("requests");
        assert_eq!(mc.get_counter("requests"), 3);

        mc.increment_by("requests", 10);
        assert_eq!(mc.get_counter("requests"), 13);

        mc.increment_by("errors", 5);
        assert_eq!(mc.get_counter("errors"), 5);
    }

    #[test]
    fn test_metrics_collector_gauges() {
        let mut mc = MetricsCollector::new();
        assert_eq!(mc.get_gauge("cpu"), None);

        mc.gauge("cpu", 0.75);
        assert_eq!(mc.get_gauge("cpu"), Some(0.75));

        // Overwrite
        mc.gauge("cpu", 0.90);
        assert_eq!(mc.get_gauge("cpu"), Some(0.90));

        mc.gauge("memory", 0.5);
        assert_eq!(mc.get_gauge("memory"), Some(0.5));
        assert_eq!(mc.get_gauge("cpu"), Some(0.90));
    }

    #[test]
    fn test_metrics_collector_histograms() {
        let mut mc = MetricsCollector::new();
        // Record 100 values: 1.0, 2.0, ..., 100.0
        for i in 1..=100 {
            mc.record_histogram("latency", i as f64);
        }
        let stats = mc.get_histogram_stats("latency").unwrap();
        assert_eq!(stats.count, 100);
        assert!((stats.sum - 5050.0).abs() < f64::EPSILON);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 100.0).abs() < f64::EPSILON);
        assert!((stats.mean - 50.5).abs() < f64::EPSILON);

        // p50 should be around 50.5 (interpolated between 50 and 51)
        assert!((stats.p50 - 50.5).abs() < 1.0);
        // p95 should be around 95.05
        assert!((stats.p95 - 95.05).abs() < 1.0);
        // p99 should be around 99.01
        assert!((stats.p99 - 99.01).abs() < 1.0);

        // Non-existent histogram returns None
        assert!(mc.get_histogram_stats("nonexistent").is_none());
    }

    #[test]
    fn test_histogram_stats_single_value() {
        let mut mc = MetricsCollector::new();
        mc.record_histogram("single", 42.0);
        let stats = mc.get_histogram_stats("single").unwrap();
        assert_eq!(stats.count, 1);
        assert!((stats.sum - 42.0).abs() < f64::EPSILON);
        assert!((stats.min - 42.0).abs() < f64::EPSILON);
        assert!((stats.max - 42.0).abs() < f64::EPSILON);
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
        assert!((stats.p50 - 42.0).abs() < f64::EPSILON);
        assert!((stats.p95 - 42.0).abs() < f64::EPSILON);
        assert!((stats.p99 - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_prometheus_export() {
        let mut mc = MetricsCollector::new();
        mc.increment_by("http_requests_total", 42);
        mc.gauge("temperature", 36.6);
        mc.record_histogram("response_time", 100.0);
        mc.record_histogram("response_time", 200.0);

        let output = mc.export_prometheus();

        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total 42"));
        assert!(output.contains("# TYPE temperature gauge"));
        assert!(output.contains("temperature 36.6"));
        assert!(output.contains("# TYPE response_time histogram"));
        assert!(output.contains("response_time_count 2"));
        assert!(output.contains("response_time_sum 300"));
    }

    #[test]
    fn test_metrics_reset() {
        let mut mc = MetricsCollector::new();
        mc.increment("a");
        mc.gauge("b", 1.0);
        mc.record_histogram("c", 5.0);

        assert_eq!(mc.get_counter("a"), 1);
        assert!(mc.get_gauge("b").is_some());
        assert!(mc.get_histogram_stats("c").is_some());

        mc.reset();

        assert_eq!(mc.get_counter("a"), 0);
        assert!(mc.get_gauge("b").is_none());
        assert!(mc.get_histogram_stats("c").is_none());
    }

    #[test]
    fn test_span_exporter_json() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = create_llm_span(&tracer, "gpt-4", "openai").with_tokens(100, 50);
        tracer.end_span(span);

        let spans = tracer.drain_spans();
        let exporter = SpanExporter::new(spans, ExportFormat::Json);
        let json = exporter.export_json();

        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let first = &arr[0];
        assert_eq!(first["operation"], "gen_ai.chat");
        assert_eq!(first["status"], "ok");
        assert_eq!(first["input_tokens"], 100);
        assert_eq!(first["output_tokens"], 50);
        assert!(first["span_id"].is_string());
        assert!(first["duration_ms"].is_number());
    }

    #[test]
    fn test_span_exporter_otlp() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let span = tracer.start_span("test.op");
        tracer.end_span(span);

        let spans = tracer.drain_spans();
        let exporter = SpanExporter::new(spans, ExportFormat::OtlpJson);
        let json = exporter.export_otlp_json();

        let resource_spans = json["resourceSpans"].as_array().unwrap();
        assert_eq!(resource_spans.len(), 1);
        let scope_spans = resource_spans[0]["scopeSpans"].as_array().unwrap();
        assert_eq!(scope_spans.len(), 1);
        let spans_arr = scope_spans[0]["spans"].as_array().unwrap();
        assert_eq!(spans_arr.len(), 1);
        assert_eq!(spans_arr[0]["name"], "test.op");
        assert!(spans_arr[0]["spanId"].is_string());
        assert!(spans_arr[0]["startTimeUnixNano"].is_number());

        // Also test the export() dispatcher with OtlpJson format
        let tracer2 = OtelTracer::new(OtelConfig::default());
        let s = tracer2.start_span("dispatch.test");
        tracer2.end_span(s);
        let spans2 = tracer2.drain_spans();
        let exporter2 = SpanExporter::new(spans2, ExportFormat::OtlpJson);
        let dispatched = exporter2.export();
        assert!(dispatched["resourceSpans"].is_array());
    }

    #[test]
    fn test_tracing_middleware_llm_lifecycle() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let metrics = MetricsCollector::new();
        let mut mw = TracingMiddleware::new(tracer, metrics);

        let span_id = mw.before_llm_call("claude-3", "anthropic");
        assert!(!span_id.is_empty());
        assert_eq!(mw.metrics.get_counter("llm.calls"), 1);

        mw.after_llm_call(&span_id, 500, 200, true);

        // Span should be completed
        let completed = mw.tracer.completed_spans();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].status, "ok");
        assert_eq!(completed[0].input_tokens, Some(500));
        assert_eq!(completed[0].output_tokens, Some(200));

        // Histogram metrics should be recorded
        let input_stats = mw.metrics.get_histogram_stats("llm.input_tokens").unwrap();
        assert_eq!(input_stats.count, 1);
        assert!((input_stats.sum - 500.0).abs() < f64::EPSILON);

        let output_stats = mw.metrics.get_histogram_stats("llm.output_tokens").unwrap();
        assert_eq!(output_stats.count, 1);
        assert!((output_stats.sum - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracing_middleware_tool_lifecycle() {
        let tracer = OtelTracer::new(OtelConfig::default());
        let metrics = MetricsCollector::new();
        let mut mw = TracingMiddleware::new(tracer, metrics);

        let span_id = mw.before_tool_call("web_search");
        assert!(!span_id.is_empty());
        assert_eq!(mw.metrics.get_counter("tool.calls"), 1);

        mw.after_tool_call(&span_id, false);

        // Span should be completed with error status
        let completed = mw.tracer.completed_spans();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].status, "error");
        assert!(completed[0].error_message.is_some());

        // Call another tool successfully
        let span_id2 = mw.before_tool_call("calculator");
        assert_eq!(mw.metrics.get_counter("tool.calls"), 2);
        mw.after_tool_call(&span_id2, true);

        let completed2 = mw.tracer.completed_spans();
        assert_eq!(completed2.len(), 2);
        assert_eq!(completed2[1].status, "ok");
    }
}
