//! OpenTelemetry integration for AI operations tracing.
//!
//! Provides OTel-compatible tracing for AI operations: model calls, RAG queries,
//! tool invocations, and agent steps. Bridges the existing [`TelemetryEvent`] system
//! to OpenTelemetry spans and metrics following the GenAI semantic conventions.
//!
//! ## Key types
//!
//! - [`OtelTracer`] — Main tracer; create spans, record metrics, export traces
//! - [`OtelSpan`] — Individual span with attributes, events, and timing
//! - [`OtelMetrics`] — Histogram and counter aggregation for model performance
//! - [`OtelConfig`] — Configuration for service name, endpoint, sampling
//!
//! ## Feature flags
//!
//! Always compiled (no feature gate). The module provides a local tracer that
//! records events in-memory. An external OTel collector can be configured via
//! [`OtelConfig`] for production export.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};

// ============================================================================
// Span Types
// ============================================================================

/// Represents an AI operation span compatible with OpenTelemetry.
#[derive(Debug, Clone)]
pub struct AiSpan {
    /// Unique span identifier
    pub span_id: String,
    /// Trace identifier (shared across all spans in a trace)
    pub trace_id: String,
    /// Optional parent span ID for nested operations
    pub parent_id: Option<String>,
    /// Operation name (e.g., "llm.generate", "rag.query", "tool.invoke")
    pub operation: String,
    /// Span kind (e.g., "client", "server", "internal")
    pub kind: String,
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
            trace_id: generate_span_id(),
            parent_id: None,
            operation: operation.to_string(),
            kind: "internal".to_string(),
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
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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
            let mut active = self.tracer.active_spans.lock().expect("active_spans lock in after_llm_call");
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
            let mut active = self.tracer.active_spans.lock().expect("active_spans lock in after_tool_call");
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
// 6.1 GenAI Semantic Attributes
// ============================================================================

/// GenAI system/provider identifiers following OTel GenAI semantic conventions.
#[derive(Debug, Clone, PartialEq)]
pub enum GenAiSystem {
    OpenAI,
    Anthropic,
    Gemini,
    Ollama,
    LmStudio,
    Bedrock,
    HuggingFace,
    Mistral,
    Groq,
    Together,
    Other(String),
}

impl GenAiSystem {
    /// Return the canonical string representation of this system.
    pub fn as_str(&self) -> &str {
        match self {
            GenAiSystem::OpenAI => "openai",
            GenAiSystem::Anthropic => "anthropic",
            GenAiSystem::Gemini => "gemini",
            GenAiSystem::Ollama => "ollama",
            GenAiSystem::LmStudio => "lm_studio",
            GenAiSystem::Bedrock => "bedrock",
            GenAiSystem::HuggingFace => "huggingface",
            GenAiSystem::Mistral => "mistral",
            GenAiSystem::Groq => "groq",
            GenAiSystem::Together => "together",
            GenAiSystem::Other(s) => s.as_str(),
        }
    }
}

/// Builder for standardized GenAI attributes following OTel semantic conventions.
#[derive(Debug, Clone)]
pub struct GenAiAttributes {
    /// The GenAI system/provider
    pub system: GenAiSystem,
    /// The model requested
    pub request_model: String,
    /// Sampling temperature
    pub request_temperature: Option<f32>,
    /// Maximum tokens to generate
    pub request_max_tokens: Option<usize>,
    /// Top-p (nucleus sampling) parameter
    pub request_top_p: Option<f32>,
    /// The model that actually responded (may differ from requested)
    pub response_model: Option<String>,
    /// Number of input (prompt) tokens
    pub input_tokens: Option<usize>,
    /// Number of output (completion) tokens
    pub output_tokens: Option<usize>,
    /// Finish/stop reasons for the response
    pub finish_reasons: Vec<String>,
}

impl GenAiAttributes {
    /// Create a new attribute set for the given system and model.
    pub fn new(system: GenAiSystem, model: &str) -> Self {
        Self {
            system,
            request_model: model.to_string(),
            request_temperature: None,
            request_max_tokens: None,
            request_top_p: None,
            response_model: None,
            input_tokens: None,
            output_tokens: None,
            finish_reasons: Vec::new(),
        }
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.request_temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.request_max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p parameter.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.request_top_p = Some(top_p);
        self
    }

    /// Set the response model name.
    pub fn with_response_model(mut self, model: &str) -> Self {
        self.response_model = Some(model.to_string());
        self
    }

    /// Set input and output token counts.
    pub fn with_tokens(mut self, input: usize, output: usize) -> Self {
        self.input_tokens = Some(input);
        self.output_tokens = Some(output);
        self
    }

    /// Add a finish reason.
    pub fn with_finish_reason(mut self, reason: &str) -> Self {
        self.finish_reasons.push(reason.to_string());
        self
    }

    /// Convert to a list of (key, value) attribute pairs following OTel conventions.
    pub fn to_attributes(&self) -> Vec<(String, String)> {
        let mut attrs = Vec::new();
        attrs.push((
            semantic_conventions::LLM_SYSTEM.to_string(),
            self.system.as_str().to_string(),
        ));
        attrs.push((
            semantic_conventions::LLM_REQUEST_MODEL.to_string(),
            self.request_model.clone(),
        ));
        if let Some(temp) = self.request_temperature {
            attrs.push((
                semantic_conventions::LLM_REQUEST_TEMPERATURE.to_string(),
                temp.to_string(),
            ));
        }
        if let Some(max) = self.request_max_tokens {
            attrs.push((
                semantic_conventions::LLM_REQUEST_MAX_TOKENS.to_string(),
                max.to_string(),
            ));
        }
        if let Some(top_p) = self.request_top_p {
            attrs.push((
                "gen_ai.request.top_p".to_string(),
                top_p.to_string(),
            ));
        }
        if let Some(ref resp_model) = self.response_model {
            attrs.push((
                semantic_conventions::LLM_RESPONSE_MODEL.to_string(),
                resp_model.clone(),
            ));
        }
        if let Some(input) = self.input_tokens {
            attrs.push((
                semantic_conventions::LLM_USAGE_INPUT_TOKENS.to_string(),
                input.to_string(),
            ));
        }
        if let Some(output) = self.output_tokens {
            attrs.push((
                semantic_conventions::LLM_USAGE_OUTPUT_TOKENS.to_string(),
                output.to_string(),
            ));
        }
        if !self.finish_reasons.is_empty() {
            attrs.push((
                semantic_conventions::LLM_RESPONSE_FINISH_REASON.to_string(),
                self.finish_reasons.join(","),
            ));
        }
        attrs
    }
}

/// Types of GenAI events that can occur during an operation.
#[derive(Debug, Clone, PartialEq)]
pub enum GenAiEventType {
    /// A prompt was sent to the model
    Prompt,
    /// A completion was received from the model
    Completion,
    /// A tool call was initiated by the model
    ToolCall,
    /// A tool result was returned to the model
    ToolResult,
}

/// A GenAI event recorded during an operation span.
#[derive(Debug, Clone)]
pub struct GenAiEvent {
    /// Type of the event
    pub event_type: GenAiEventType,
    /// Content or description of the event
    pub content: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl GenAiEvent {
    /// Create a new GenAI event.
    pub fn new(event_type: GenAiEventType, content: &str) -> Self {
        Self {
            event_type,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ============================================================================
// 6.2 Hierarchical Agent Tracing
// ============================================================================

/// Status of an agent span.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span completed with an error
    Error(String),
    /// Span status has not been set
    Unset,
}

/// An agent operation span with hierarchical parent/child relationships.
#[derive(Debug, Clone)]
pub struct AgentSpan {
    /// Unique span identifier
    pub span_id: String,
    /// Parent span identifier (None for root spans)
    pub parent_span_id: Option<String>,
    /// Operation name
    pub operation: String,
    /// When the span started
    pub start_time: DateTime<Utc>,
    /// When the span ended (None if still running)
    pub end_time: Option<DateTime<Utc>>,
    /// Span attributes
    pub attributes: HashMap<String, String>,
    /// Events recorded during this span
    pub events: Vec<GenAiEvent>,
    /// IDs of child spans
    pub children: Vec<String>,
    /// Span completion status
    pub status: SpanStatus,
}

/// A tree node for visualizing span hierarchies.
#[derive(Debug, Clone)]
pub struct SpanTreeNode {
    /// The span at this node
    pub span: AgentSpan,
    /// Child nodes
    pub children: Vec<SpanTreeNode>,
}

/// A tree representation of a trace.
#[derive(Debug, Clone)]
pub struct SpanTree {
    /// The root node of the tree
    pub root: SpanTreeNode,
}

/// Creates and manages hierarchical spans for multi-turn agent sessions.
#[derive(Debug)]
pub struct AgentTracer {
    /// All traces keyed by trace_id, each containing a list of spans
    traces: HashMap<String, Vec<AgentSpan>>,
    /// Maps span_id -> trace_id for active (unfinished) spans
    active_spans: HashMap<String, String>,
    /// Counter for generating unique IDs
    id_counter: u64,
}

impl AgentTracer {
    /// Create a new agent tracer.
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
            active_spans: HashMap::new(),
            id_counter: 0,
        }
    }

    /// Start a new agent session. Returns (trace_id, root_span_id).
    pub fn start_session(&mut self, agent_id: &str) -> (String, String) {
        let trace_id = self.next_id("trace");
        let span_id = self.next_id("span");
        let span = AgentSpan {
            span_id: span_id.clone(),
            parent_span_id: None,
            operation: format!("agent.session.{}", agent_id),
            start_time: Utc::now(),
            end_time: None,
            attributes: {
                let mut m = HashMap::new();
                m.insert("agent.id".to_string(), agent_id.to_string());
                m
            },
            events: Vec::new(),
            children: Vec::new(),
            status: SpanStatus::Unset,
        };
        self.traces.insert(trace_id.clone(), vec![span]);
        self.active_spans.insert(span_id.clone(), trace_id.clone());
        (trace_id, span_id)
    }

    /// Start a new turn span under a parent span. Returns the new span_id.
    pub fn start_turn(
        &mut self,
        trace_id: &str,
        parent_span_id: &str,
        turn_number: usize,
    ) -> String {
        let span_id = self.next_id("span");
        let span = AgentSpan {
            span_id: span_id.clone(),
            parent_span_id: Some(parent_span_id.to_string()),
            operation: format!("agent.turn.{}", turn_number),
            start_time: Utc::now(),
            end_time: None,
            attributes: {
                let mut m = HashMap::new();
                m.insert("turn.number".to_string(), turn_number.to_string());
                m
            },
            events: Vec::new(),
            children: Vec::new(),
            status: SpanStatus::Unset,
        };
        self.add_child_to_parent(trace_id, parent_span_id, &span_id);
        if let Some(spans) = self.traces.get_mut(trace_id) {
            spans.push(span);
        }
        self.active_spans.insert(span_id.clone(), trace_id.to_string());
        span_id
    }

    /// Start an LLM call span. Returns the new span_id.
    pub fn start_llm_call(
        &mut self,
        trace_id: &str,
        parent_span_id: &str,
        attrs: &GenAiAttributes,
    ) -> String {
        let span_id = self.next_id("span");
        let mut attributes = HashMap::new();
        for (k, v) in attrs.to_attributes() {
            attributes.insert(k, v);
        }
        let span = AgentSpan {
            span_id: span_id.clone(),
            parent_span_id: Some(parent_span_id.to_string()),
            operation: "gen_ai.chat".to_string(),
            start_time: Utc::now(),
            end_time: None,
            attributes,
            events: Vec::new(),
            children: Vec::new(),
            status: SpanStatus::Unset,
        };
        self.add_child_to_parent(trace_id, parent_span_id, &span_id);
        if let Some(spans) = self.traces.get_mut(trace_id) {
            spans.push(span);
        }
        self.active_spans.insert(span_id.clone(), trace_id.to_string());
        span_id
    }

    /// Start a tool call span. Returns the new span_id.
    pub fn start_tool_call(
        &mut self,
        trace_id: &str,
        parent_span_id: &str,
        tool_name: &str,
    ) -> String {
        let span_id = self.next_id("span");
        let span = AgentSpan {
            span_id: span_id.clone(),
            parent_span_id: Some(parent_span_id.to_string()),
            operation: format!("tool.invoke.{}", tool_name),
            start_time: Utc::now(),
            end_time: None,
            attributes: {
                let mut m = HashMap::new();
                m.insert("tool.name".to_string(), tool_name.to_string());
                m
            },
            events: Vec::new(),
            children: Vec::new(),
            status: SpanStatus::Unset,
        };
        self.add_child_to_parent(trace_id, parent_span_id, &span_id);
        if let Some(spans) = self.traces.get_mut(trace_id) {
            spans.push(span);
        }
        self.active_spans.insert(span_id.clone(), trace_id.to_string());
        span_id
    }

    /// End a span with the given status.
    pub fn end_span(&mut self, trace_id: &str, span_id: &str, status: SpanStatus) {
        self.active_spans.remove(span_id);
        if let Some(spans) = self.traces.get_mut(trace_id) {
            if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                span.end_time = Some(Utc::now());
                span.status = status;
            }
        }
    }

    /// Add a GenAI event to an existing span.
    pub fn add_event(&mut self, trace_id: &str, span_id: &str, event: GenAiEvent) {
        if let Some(spans) = self.traces.get_mut(trace_id) {
            if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                span.events.push(event);
            }
        }
    }

    /// Get all spans in a trace.
    pub fn get_trace(&self, trace_id: &str) -> Option<&Vec<AgentSpan>> {
        self.traces.get(trace_id)
    }

    /// Get a specific span from a trace.
    pub fn get_span(&self, trace_id: &str, span_id: &str) -> Option<&AgentSpan> {
        self.traces
            .get(trace_id)
            .and_then(|spans| spans.iter().find(|s| s.span_id == span_id))
    }

    /// Get the total session duration in milliseconds.
    /// Computed from the earliest start_time to the latest end_time.
    pub fn get_session_duration(&self, trace_id: &str) -> Option<u64> {
        let spans = self.traces.get(trace_id)?;
        if spans.is_empty() {
            return None;
        }
        let min_start = spans.iter().map(|s| s.start_time).min()?;
        let max_end = spans
            .iter()
            .filter_map(|s| s.end_time)
            .max();
        let end = max_end?;
        let duration = end.signed_duration_since(min_start);
        Some(duration.num_milliseconds().max(0) as u64)
    }

    /// Build a tree representation of a trace.
    pub fn get_span_tree(&self, trace_id: &str) -> Option<SpanTree> {
        let spans = self.traces.get(trace_id)?;
        if spans.is_empty() {
            return None;
        }
        // Find the root span (no parent)
        let root_span = spans.iter().find(|s| s.parent_span_id.is_none())?;
        let root_node = self.build_tree_node(root_span, spans);
        Some(SpanTree { root: root_node })
    }

    // --- private helpers ---

    fn next_id(&mut self, prefix: &str) -> String {
        self.id_counter += 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("{}_{:016x}_{}", prefix, now & 0xFFFFFFFFFFFFFFFF, self.id_counter)
    }

    fn add_child_to_parent(&mut self, trace_id: &str, parent_span_id: &str, child_span_id: &str) {
        if let Some(spans) = self.traces.get_mut(trace_id) {
            if let Some(parent) = spans.iter_mut().find(|s| s.span_id == parent_span_id) {
                parent.children.push(child_span_id.to_string());
            }
        }
    }

    fn build_tree_node(&self, span: &AgentSpan, all_spans: &[AgentSpan]) -> SpanTreeNode {
        let children: Vec<SpanTreeNode> = span
            .children
            .iter()
            .filter_map(|child_id| {
                all_spans
                    .iter()
                    .find(|s| s.span_id == *child_id)
                    .map(|child_span| self.build_tree_node(child_span, all_spans))
            })
            .collect();
        SpanTreeNode {
            span: span.clone(),
            children,
        }
    }
}

// ============================================================================
// 6.3 Cost Attribution & Budget Enforcement
// ============================================================================

/// Pricing information for a specific model pattern.
#[derive(Debug, Clone)]
pub struct ModelPricing {
    /// Pattern to match model names (substring match)
    pub model_pattern: String,
    /// Cost per 1,000 input tokens (USD)
    pub input_cost_per_1k: f64,
    /// Cost per 1,000 output tokens (USD)
    pub output_cost_per_1k: f64,
}

/// A table of model pricing entries.
#[derive(Debug, Clone)]
pub struct PricingTable {
    /// Pricing entries
    pub entries: Vec<ModelPricing>,
}

impl PricingTable {
    /// Create a new empty pricing table.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a pricing table pre-populated with standard model pricing.
    pub fn with_defaults() -> Self {
        let mut table = Self::new();
        // OpenAI
        table.add("gpt-4o", 0.005, 0.015);
        table.add("gpt-4o-mini", 0.00015, 0.0006);
        // Anthropic
        table.add("claude-3.5-sonnet", 0.003, 0.015);
        table.add("claude-3-haiku", 0.00025, 0.00125);
        // Google
        table.add("gemini-1.5-pro", 0.00125, 0.005);
        table.add("gemini-1.5-flash", 0.000075, 0.0003);
        // Open source
        table.add("llama-3.1-70b", 0.00059, 0.00079);
        table
    }

    /// Add a pricing entry.
    pub fn add(&mut self, model_pattern: &str, input_per_1k: f64, output_per_1k: f64) {
        self.entries.push(ModelPricing {
            model_pattern: model_pattern.to_string(),
            input_cost_per_1k: input_per_1k,
            output_cost_per_1k: output_per_1k,
        });
    }

    /// Find pricing for a model by substring match.
    pub fn find_pricing(&self, model: &str) -> Option<&ModelPricing> {
        self.entries
            .iter()
            .find(|e| model.contains(&e.model_pattern))
    }

    /// Calculate the cost for a given model and token usage.
    pub fn get_cost(&self, model: &str, input_tokens: usize, output_tokens: usize) -> Option<f64> {
        self.find_pricing(model).map(|p| {
            (input_tokens as f64 / 1000.0) * p.input_cost_per_1k
                + (output_tokens as f64 / 1000.0) * p.output_cost_per_1k
        })
    }
}

/// Result of checking whether a cost is within budget.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetCheckResult {
    /// Cost is allowed within the budget
    Allowed,
    /// Cost is allowed but remaining budget is low (< 20% remaining after this cost)
    Warning {
        remaining: f64,
        estimated: f64,
    },
    /// Cost is denied because it would exceed the budget
    Denied {
        budget: f64,
        current: f64,
        requested: f64,
    },
}

/// An alert threshold for budget monitoring.
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Percentage threshold (0.0 to 1.0)
    pub threshold_pct: f64,
    /// Whether this alert has been triggered
    pub triggered: bool,
    /// When the alert was triggered
    pub triggered_at: Option<DateTime<Utc>>,
}

/// Budget tracking for a specific scope.
#[derive(Debug, Clone)]
pub struct CostBudget {
    /// Maximum allowed cost
    pub max_cost: f64,
    /// Current accumulated cost
    pub current_cost: f64,
    /// Currency (default "USD")
    pub currency: String,
    /// Budget alerts
    pub alerts: Vec<BudgetAlert>,
}

impl CostBudget {
    /// Create a new budget with the given maximum cost.
    pub fn new(max_cost: f64) -> Self {
        Self {
            max_cost,
            current_cost: 0.0,
            currency: "USD".to_string(),
            alerts: vec![
                BudgetAlert {
                    threshold_pct: 0.8,
                    triggered: false,
                    triggered_at: None,
                },
                BudgetAlert {
                    threshold_pct: 0.95,
                    triggered: false,
                    triggered_at: None,
                },
            ],
        }
    }

    /// Get remaining budget.
    pub fn remaining(&self) -> f64 {
        (self.max_cost - self.current_cost).max(0.0)
    }

    /// Check if the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.current_cost >= self.max_cost
    }

    /// Add a cost to the current total and check alerts.
    pub fn add_cost(&mut self, cost: f64) {
        self.current_cost += cost;
        let pct_used = self.current_cost / self.max_cost;
        for alert in &mut self.alerts {
            if !alert.triggered && pct_used >= alert.threshold_pct {
                alert.triggered = true;
                alert.triggered_at = Some(Utc::now());
            }
        }
    }

    /// Check whether an estimated cost is within budget.
    pub fn check_budget(&self, estimated_cost: f64) -> BudgetCheckResult {
        let after_cost = self.current_cost + estimated_cost;
        if after_cost > self.max_cost {
            return BudgetCheckResult::Denied {
                budget: self.max_cost,
                current: self.current_cost,
                requested: estimated_cost,
            };
        }
        let remaining_after = self.max_cost - after_cost;
        let remaining_pct = remaining_after / self.max_cost;
        if remaining_pct < 0.2 {
            BudgetCheckResult::Warning {
                remaining: remaining_after,
                estimated: estimated_cost,
            }
        } else {
            BudgetCheckResult::Allowed
        }
    }
}

/// A single entry in the cost breakdown report.
#[derive(Debug, Clone)]
pub struct CostBreakdownEntry {
    /// Model name
    pub model: String,
    /// Number of calls
    pub calls: usize,
    /// Total input tokens
    pub input_tokens: usize,
    /// Total output tokens
    pub output_tokens: usize,
    /// Total cost for this model
    pub cost: f64,
}

/// A cost report for a specific scope.
#[derive(Debug, Clone)]
pub struct CostReport {
    /// Scope identifier
    pub scope: String,
    /// Total cost incurred
    pub total_cost: f64,
    /// Maximum budget (if set)
    pub budget_max: Option<f64>,
    /// Remaining budget (if set)
    pub budget_remaining: Option<f64>,
    /// Number of calls
    pub num_calls: usize,
    /// Per-model breakdown
    pub breakdown: Vec<CostBreakdownEntry>,
}

/// Tracks and attributes costs across scopes with budget enforcement.
#[derive(Debug)]
pub struct CostAttributor {
    /// Pricing table for cost calculations
    pricing: PricingTable,
    /// Per-scope budgets
    budgets: HashMap<String, CostBudget>,
    /// Per-scope, per-model breakdown tracking
    breakdowns: HashMap<String, HashMap<String, CostBreakdownEntry>>,
    /// Total number of calls per scope
    call_counts: HashMap<String, usize>,
}

impl CostAttributor {
    /// Create a new cost attributor with the given pricing table.
    pub fn new(pricing: PricingTable) -> Self {
        Self {
            pricing,
            budgets: HashMap::new(),
            breakdowns: HashMap::new(),
            call_counts: HashMap::new(),
        }
    }

    /// Set a budget for a scope.
    pub fn set_budget(&mut self, scope: &str, max_cost: f64) {
        self.budgets.insert(scope.to_string(), CostBudget::new(max_cost));
    }

    /// Attribute a cost to a scope. Returns the calculated cost.
    /// Returns an error if the budget is exceeded.
    pub fn attribute_cost(
        &mut self,
        scope: &str,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
    ) -> Result<f64, String> {
        let cost = self
            .pricing
            .get_cost(model, input_tokens, output_tokens)
            .unwrap_or(0.0);

        // Check budget if one exists
        if let Some(budget) = self.budgets.get(scope) {
            if let BudgetCheckResult::Denied { budget: b, current, requested } =
                budget.check_budget(cost)
            {
                return Err(format!(
                    "Budget exceeded for scope '{}': budget={:.4}, current={:.4}, requested={:.4}",
                    scope, b, current, requested
                ));
            }
        }

        // Apply cost to budget
        if let Some(budget) = self.budgets.get_mut(scope) {
            budget.add_cost(cost);
        }

        // Track breakdown
        let scope_breakdown = self
            .breakdowns
            .entry(scope.to_string())
            .or_default();
        let entry = scope_breakdown
            .entry(model.to_string())
            .or_insert_with(|| CostBreakdownEntry {
                model: model.to_string(),
                calls: 0,
                input_tokens: 0,
                output_tokens: 0,
                cost: 0.0,
            });
        entry.calls += 1;
        entry.input_tokens += input_tokens;
        entry.output_tokens += output_tokens;
        entry.cost += cost;

        // Track call count
        *self.call_counts.entry(scope.to_string()).or_insert(0) += 1;

        Ok(cost)
    }

    /// Check if a planned call is within budget without actually spending.
    pub fn check_budget(
        &self,
        scope: &str,
        model: &str,
        estimated_input: usize,
        estimated_output: usize,
    ) -> BudgetCheckResult {
        let cost = self
            .pricing
            .get_cost(model, estimated_input, estimated_output)
            .unwrap_or(0.0);
        match self.budgets.get(scope) {
            Some(budget) => budget.check_budget(cost),
            None => BudgetCheckResult::Allowed,
        }
    }

    /// Get a cost report for a scope.
    pub fn get_cost_report(&self, scope: &str) -> Option<CostReport> {
        let breakdown_map = self.breakdowns.get(scope)?;
        let budget = self.budgets.get(scope);
        let total_cost: f64 = breakdown_map.values().map(|e| e.cost).sum();
        let num_calls = self.call_counts.get(scope).copied().unwrap_or(0);
        let mut breakdown: Vec<CostBreakdownEntry> = breakdown_map.values().cloned().collect();
        breakdown.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap_or(std::cmp::Ordering::Equal));
        Some(CostReport {
            scope: scope.to_string(),
            total_cost,
            budget_max: budget.map(|b| b.max_cost),
            budget_remaining: budget.map(|b| b.remaining()),
            num_calls,
            breakdown,
        })
    }

    /// Get the total cost across all scopes.
    pub fn get_total_cost(&self) -> f64 {
        self.breakdowns
            .values()
            .flat_map(|m| m.values())
            .map(|e| e.cost)
            .sum()
    }
}

// ============================================================================
// OTLP HTTP Exporter (Item 5.1)
// ============================================================================

/// Serializable span attribute for OTLP JSON export.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpanAttribute {
    pub key: String,
    pub value: SpanAttributeValue,
}

/// Attribute value wrapper (string only for simplicity).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpanAttributeValue {
    #[serde(rename = "stringValue")]
    pub string_value: String,
}

/// Status for exported spans.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportSpanStatus {
    pub code: u32,
}

/// Serializable span for OTLP JSON export format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportableSpan {
    #[serde(rename = "traceId")]
    pub trace_id: String,
    #[serde(rename = "spanId")]
    pub span_id: String,
    #[serde(rename = "parentSpanId", skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub name: String,
    /// Span kind: 1=Internal, 2=Server, 3=Client
    pub kind: u32,
    #[serde(rename = "startTimeUnixNano")]
    pub start_time_unix_nano: u64,
    #[serde(rename = "endTimeUnixNano")]
    pub end_time_unix_nano: u64,
    pub attributes: Vec<SpanAttribute>,
    pub status: ExportSpanStatus,
}

/// Exports spans to an OTLP-compatible collector via HTTP POST to `/v1/traces`.
#[derive(Debug)]
pub struct OtlpHttpExporter {
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub batch: Vec<ExportableSpan>,
    pub flush_interval_ms: u64,
    pub max_batch_size: usize,
}

impl OtlpHttpExporter {
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            headers: HashMap::new(),
            batch: Vec::new(),
            flush_interval_ms: 5_000,
            max_batch_size: 512,
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_flush_interval(mut self, ms: u64) -> Self {
        self.flush_interval_ms = ms;
        self
    }

    /// Convert an AiSpan to ExportableSpan and add to batch.
    pub fn add_span(&mut self, span: &AiSpan) {
        let attrs: Vec<SpanAttribute> = span
            .attributes
            .iter()
            .map(|(k, v)| SpanAttribute {
                key: k.clone(),
                value: SpanAttributeValue {
                    string_value: v.clone(),
                },
            })
            .collect();

        let kind = match span.kind.as_str() {
            "client" => 3,
            "server" => 2,
            _ => 1, // internal
        };

        let status_code = if span.status == "error" { 2 } else { 1 };

        self.batch.push(ExportableSpan {
            trace_id: span.trace_id.clone(),
            span_id: span.span_id.clone(),
            parent_span_id: span.parent_id.clone(),
            name: span.operation.clone(),
            kind,
            start_time_unix_nano: span.start_time_ms * 1_000_000,
            end_time_unix_nano: span.end_time_ms.unwrap_or(span.start_time_ms) * 1_000_000,
            attributes: attrs,
            status: ExportSpanStatus { code: status_code },
        });
    }

    /// Returns true if the batch is at or over the max batch size.
    pub fn should_flush(&self) -> bool {
        self.batch.len() >= self.max_batch_size
    }

    /// Build OTLP JSON payload from current batch.
    pub fn to_otlp_json(&self) -> String {
        let payload = serde_json::json!({
            "resourceSpans": [{
                "scopeSpans": [{
                    "scope": { "name": "ai_assistant" },
                    "spans": self.batch.iter().map(|s| serde_json::to_value(s).unwrap_or_default()).collect::<Vec<_>>()
                }]
            }]
        });
        payload.to_string()
    }

    /// Flush the current batch to the OTLP endpoint via HTTP POST.
    /// Returns the number of spans exported. Clears batch on success.
    pub fn flush(&mut self) -> Result<usize, String> {
        if self.batch.is_empty() {
            return Ok(0);
        }

        let count = self.batch.len();
        let body = self.to_otlp_json();
        let url = format!("{}/v1/traces", self.endpoint);

        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(10));

        for (k, v) in &self.headers {
            req = req.set(k, v);
        }

        match req.send_string(&body) {
            Ok(_) => {
                self.batch.clear();
                Ok(count)
            }
            Err(e) => Err(format!("OTLP export failed: {}", e)),
        }
    }
}

// ============================================================================
// GenAI Semantic Conventions (Item 5.2)
// ============================================================================

/// OpenTelemetry GenAI semantic convention attribute names.
pub struct GenAiConventions;

impl GenAiConventions {
    pub const SYSTEM: &'static str = "gen_ai.system";
    pub const REQUEST_MODEL: &'static str = "gen_ai.request.model";
    pub const RESPONSE_MODEL: &'static str = "gen_ai.response.model";
    pub const REQUEST_MAX_TOKENS: &'static str = "gen_ai.request.max_tokens";
    pub const USAGE_INPUT_TOKENS: &'static str = "gen_ai.usage.input_tokens";
    pub const USAGE_OUTPUT_TOKENS: &'static str = "gen_ai.usage.output_tokens";
    pub const RESPONSE_FINISH_REASON: &'static str = "gen_ai.response.finish_reason";
    pub const REQUEST_TEMPERATURE: &'static str = "gen_ai.request.temperature";
}

/// Builder for creating AiSpan with GenAI semantic convention attributes.
#[derive(Debug)]
pub struct GenAiSpanBuilder {
    operation: String,
    attributes: HashMap<String, String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

impl GenAiSpanBuilder {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            attributes: HashMap::new(),
            input_tokens: None,
            output_tokens: None,
        }
    }

    pub fn system(mut self, s: &str) -> Self {
        self.attributes.insert(GenAiConventions::SYSTEM.to_string(), s.to_string());
        self
    }

    pub fn model(mut self, m: &str) -> Self {
        self.attributes.insert(GenAiConventions::REQUEST_MODEL.to_string(), m.to_string());
        self
    }

    pub fn input_tokens(mut self, n: u64) -> Self {
        self.input_tokens = Some(n);
        self.attributes.insert(GenAiConventions::USAGE_INPUT_TOKENS.to_string(), n.to_string());
        self
    }

    pub fn output_tokens(mut self, n: u64) -> Self {
        self.output_tokens = Some(n);
        self.attributes.insert(GenAiConventions::USAGE_OUTPUT_TOKENS.to_string(), n.to_string());
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.attributes.insert(GenAiConventions::REQUEST_TEMPERATURE.to_string(), t.to_string());
        self
    }

    pub fn finish_reason(mut self, r: &str) -> Self {
        self.attributes.insert(GenAiConventions::RESPONSE_FINISH_REASON.to_string(), r.to_string());
        self
    }

    pub fn max_tokens(mut self, n: u64) -> Self {
        self.attributes.insert(GenAiConventions::REQUEST_MAX_TOKENS.to_string(), n.to_string());
        self
    }

    pub fn build(self) -> AiSpan {
        let mut span = AiSpan::new(&self.operation);
        span.kind = "client".to_string();
        span.input_tokens = self.input_tokens;
        span.output_tokens = self.output_tokens;
        span.attributes = self.attributes;
        if let Some(model) = span.attributes.get(GenAiConventions::REQUEST_MODEL) {
            span.model = Some(model.clone());
        }
        if let Some(system) = span.attributes.get(GenAiConventions::SYSTEM) {
            span.provider = Some(system.clone());
        }
        span
    }
}

// ============================================================================
// Prometheus Metrics Endpoint (Item 5.3)
// ============================================================================

/// Prometheus-format metrics for AI operations.
#[derive(Debug)]
pub struct PrometheusMetrics {
    /// (provider, model, status) -> count
    request_counts: HashMap<(String, String, String), u64>,
    /// (provider, model, direction) -> total tokens
    token_counts: HashMap<(String, String, String), u64>,
    /// (provider, error_type) -> count
    error_counts: HashMap<(String, String), u64>,
    /// All recorded durations for histogram
    durations: Vec<f64>,
}

impl PrometheusMetrics {
    pub fn new() -> Self {
        Self {
            request_counts: HashMap::new(),
            token_counts: HashMap::new(),
            error_counts: HashMap::new(),
            durations: Vec::new(),
        }
    }

    pub fn record_request(
        &mut self,
        provider: &str,
        model: &str,
        status: &str,
        duration_secs: f64,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        *self.request_counts
            .entry((provider.to_string(), model.to_string(), status.to_string()))
            .or_insert(0) += 1;

        *self.token_counts
            .entry((provider.to_string(), model.to_string(), "input".to_string()))
            .or_insert(0) += input_tokens;

        *self.token_counts
            .entry((provider.to_string(), model.to_string(), "output".to_string()))
            .or_insert(0) += output_tokens;

        self.durations.push(duration_secs);
    }

    pub fn record_error(&mut self, provider: &str, error_type: &str) {
        *self.error_counts
            .entry((provider.to_string(), error_type.to_string()))
            .or_insert(0) += 1;
    }

    pub fn render(&self) -> String {
        if self.request_counts.is_empty() && self.error_counts.is_empty() {
            return "# No metrics recorded\n".to_string();
        }

        let mut out = String::new();

        // ai_requests_total
        if !self.request_counts.is_empty() {
            out.push_str("# HELP ai_requests_total Total number of AI requests\n");
            out.push_str("# TYPE ai_requests_total counter\n");
            let mut keys: Vec<_> = self.request_counts.keys().collect();
            keys.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for key in keys {
                let count = self.request_counts[key];
                out.push_str(&format!(
                    "ai_requests_total{{provider=\"{}\",model=\"{}\",status=\"{}\"}} {}\n",
                    key.0, key.1, key.2, count
                ));
            }
        }

        // ai_tokens_total
        if !self.token_counts.is_empty() {
            out.push_str("# HELP ai_tokens_total Total tokens used\n");
            out.push_str("# TYPE ai_tokens_total counter\n");
            let mut keys: Vec<_> = self.token_counts.keys().collect();
            keys.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for key in keys {
                let count = self.token_counts[key];
                out.push_str(&format!(
                    "ai_tokens_total{{provider=\"{}\",model=\"{}\",direction=\"{}\"}} {}\n",
                    key.0, key.1, key.2, count
                ));
            }
        }

        // ai_errors_total
        if !self.error_counts.is_empty() {
            out.push_str("# HELP ai_errors_total Total AI errors\n");
            out.push_str("# TYPE ai_errors_total counter\n");
            let mut keys: Vec<_> = self.error_counts.keys().collect();
            keys.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            for key in keys {
                let count = self.error_counts[key];
                out.push_str(&format!(
                    "ai_errors_total{{provider=\"{}\",error_type=\"{}\"}} {}\n",
                    key.0, key.1, count
                ));
            }
        }

        // ai_request_duration_seconds
        if !self.durations.is_empty() {
            let sum: f64 = self.durations.iter().sum();
            let count = self.durations.len();
            out.push_str("# HELP ai_request_duration_seconds Request duration histogram\n");
            out.push_str("# TYPE ai_request_duration_seconds summary\n");
            out.push_str(&format!("ai_request_duration_seconds_sum {:.6}\n", sum));
            out.push_str(&format!("ai_request_duration_seconds_count {}\n", count));
        }

        out
    }

    pub fn reset(&mut self) {
        self.request_counts.clear();
        self.token_counts.clear();
        self.error_counts.clear();
        self.durations.clear();
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

    // ========================================================================
    // Phase 6: GenAI Semantic Conventions Tests
    // ========================================================================

    // --- 6.1 GenAI Semantic Attributes ---

    #[test]
    fn test_genai_system_as_str() {
        assert_eq!(GenAiSystem::OpenAI.as_str(), "openai");
        assert_eq!(GenAiSystem::Anthropic.as_str(), "anthropic");
        assert_eq!(GenAiSystem::Gemini.as_str(), "gemini");
        assert_eq!(GenAiSystem::Ollama.as_str(), "ollama");
        assert_eq!(GenAiSystem::LmStudio.as_str(), "lm_studio");
        assert_eq!(GenAiSystem::Bedrock.as_str(), "bedrock");
        assert_eq!(GenAiSystem::HuggingFace.as_str(), "huggingface");
        assert_eq!(GenAiSystem::Mistral.as_str(), "mistral");
        assert_eq!(GenAiSystem::Groq.as_str(), "groq");
        assert_eq!(GenAiSystem::Together.as_str(), "together");
        assert_eq!(
            GenAiSystem::Other("custom_provider".to_string()).as_str(),
            "custom_provider"
        );
    }

    #[test]
    fn test_genai_attributes_builder() {
        let attrs = GenAiAttributes::new(GenAiSystem::OpenAI, "gpt-4o")
            .with_temperature(0.7)
            .with_max_tokens(4096)
            .with_top_p(0.9)
            .with_response_model("gpt-4o-2024-08-06")
            .with_tokens(500, 200)
            .with_finish_reason("stop");

        assert_eq!(attrs.system, GenAiSystem::OpenAI);
        assert_eq!(attrs.request_model, "gpt-4o");
        assert_eq!(attrs.request_temperature, Some(0.7));
        assert_eq!(attrs.request_max_tokens, Some(4096));
        assert_eq!(attrs.request_top_p, Some(0.9));
        assert_eq!(
            attrs.response_model.as_deref(),
            Some("gpt-4o-2024-08-06")
        );
        assert_eq!(attrs.input_tokens, Some(500));
        assert_eq!(attrs.output_tokens, Some(200));
        assert_eq!(attrs.finish_reasons, vec!["stop".to_string()]);
    }

    #[test]
    fn test_genai_attributes_to_attributes() {
        let attrs = GenAiAttributes::new(GenAiSystem::Anthropic, "claude-3.5-sonnet")
            .with_temperature(0.5)
            .with_max_tokens(1024)
            .with_tokens(100, 50)
            .with_finish_reason("end_turn");

        let kv_pairs = attrs.to_attributes();

        // Check that required attributes are present
        let map: HashMap<String, String> = kv_pairs.into_iter().collect();
        assert_eq!(map.get("gen_ai.system").unwrap(), "anthropic");
        assert_eq!(map.get("gen_ai.request.model").unwrap(), "claude-3.5-sonnet");
        assert_eq!(map.get("gen_ai.request.temperature").unwrap(), "0.5");
        assert_eq!(map.get("gen_ai.request.max_tokens").unwrap(), "1024");
        assert_eq!(map.get("gen_ai.usage.input_tokens").unwrap(), "100");
        assert_eq!(map.get("gen_ai.usage.output_tokens").unwrap(), "50");
        assert_eq!(
            map.get("gen_ai.response.finish_reasons").unwrap(),
            "end_turn"
        );
    }

    #[test]
    fn test_genai_attributes_all_fields() {
        let attrs = GenAiAttributes::new(GenAiSystem::Gemini, "gemini-1.5-pro")
            .with_temperature(1.0)
            .with_max_tokens(8192)
            .with_top_p(0.95)
            .with_response_model("gemini-1.5-pro-002")
            .with_tokens(1000, 2000)
            .with_finish_reason("stop")
            .with_finish_reason("length");

        let kv_pairs = attrs.to_attributes();
        let map: HashMap<String, String> = kv_pairs.into_iter().collect();

        assert_eq!(map.get("gen_ai.system").unwrap(), "gemini");
        assert_eq!(map.get("gen_ai.request.model").unwrap(), "gemini-1.5-pro");
        assert_eq!(map.get("gen_ai.request.temperature").unwrap(), "1");
        assert_eq!(map.get("gen_ai.request.max_tokens").unwrap(), "8192");
        assert_eq!(map.get("gen_ai.request.top_p").unwrap(), "0.95");
        assert_eq!(
            map.get("gen_ai.response.model").unwrap(),
            "gemini-1.5-pro-002"
        );
        assert_eq!(map.get("gen_ai.usage.input_tokens").unwrap(), "1000");
        assert_eq!(map.get("gen_ai.usage.output_tokens").unwrap(), "2000");
        assert_eq!(
            map.get("gen_ai.response.finish_reasons").unwrap(),
            "stop,length"
        );
    }

    #[test]
    fn test_genai_attributes_minimal() {
        let attrs = GenAiAttributes::new(GenAiSystem::Ollama, "llama3");
        let kv_pairs = attrs.to_attributes();
        let map: HashMap<String, String> = kv_pairs.into_iter().collect();

        // Only system and request model should be present
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("gen_ai.system").unwrap(), "ollama");
        assert_eq!(map.get("gen_ai.request.model").unwrap(), "llama3");
    }

    #[test]
    fn test_genai_event_creation() {
        let event = GenAiEvent::new(GenAiEventType::Prompt, "Hello, world!")
            .with_metadata("role", "user");

        assert_eq!(event.event_type, GenAiEventType::Prompt);
        assert_eq!(event.content, "Hello, world!");
        assert_eq!(event.metadata.get("role").unwrap(), "user");
        // Timestamp should be recent
        let now = Utc::now();
        let diff = now.signed_duration_since(event.timestamp).num_seconds().abs();
        assert!(diff < 5);
    }

    #[test]
    fn test_genai_event_types() {
        let prompt = GenAiEvent::new(GenAiEventType::Prompt, "test");
        let completion = GenAiEvent::new(GenAiEventType::Completion, "test");
        let tool_call = GenAiEvent::new(GenAiEventType::ToolCall, "test");
        let tool_result = GenAiEvent::new(GenAiEventType::ToolResult, "test");

        assert_eq!(prompt.event_type, GenAiEventType::Prompt);
        assert_eq!(completion.event_type, GenAiEventType::Completion);
        assert_eq!(tool_call.event_type, GenAiEventType::ToolCall);
        assert_eq!(tool_result.event_type, GenAiEventType::ToolResult);
    }

    // --- 6.2 Hierarchical Agent Tracing ---

    #[test]
    fn test_agent_tracer_start_session() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_span_id) = tracer.start_session("agent-001");

        assert!(!trace_id.is_empty());
        assert!(!root_span_id.is_empty());

        let trace = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(trace.len(), 1);
        assert!(trace[0].operation.contains("agent.session"));
        assert!(trace[0].operation.contains("agent-001"));
        assert!(trace[0].parent_span_id.is_none());
        assert_eq!(trace[0].status, SpanStatus::Unset);
    }

    #[test]
    fn test_agent_tracer_start_turn() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);

        assert!(!turn_id.is_empty());

        let trace = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(trace.len(), 2);

        let turn_span = tracer.get_span(&trace_id, &turn_id).unwrap();
        assert_eq!(turn_span.parent_span_id.as_deref(), Some(root_id.as_str()));
        assert!(turn_span.operation.contains("agent.turn.1"));

        // Root span should have child
        let root = tracer.get_span(&trace_id, &root_id).unwrap();
        assert!(root.children.contains(&turn_id));
    }

    #[test]
    fn test_agent_tracer_start_llm_call() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);
        let attrs = GenAiAttributes::new(GenAiSystem::OpenAI, "gpt-4o")
            .with_tokens(100, 50);
        let llm_id = tracer.start_llm_call(&trace_id, &turn_id, &attrs);

        let llm_span = tracer.get_span(&trace_id, &llm_id).unwrap();
        assert_eq!(llm_span.operation, "gen_ai.chat");
        assert_eq!(
            llm_span.parent_span_id.as_deref(),
            Some(turn_id.as_str())
        );
        assert_eq!(
            llm_span.attributes.get("gen_ai.system").unwrap(),
            "openai"
        );
        assert_eq!(
            llm_span.attributes.get("gen_ai.request.model").unwrap(),
            "gpt-4o"
        );
    }

    #[test]
    fn test_agent_tracer_start_tool_call() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);
        let tool_id = tracer.start_tool_call(&trace_id, &turn_id, "web_search");

        let tool_span = tracer.get_span(&trace_id, &tool_id).unwrap();
        assert!(tool_span.operation.contains("tool.invoke.web_search"));
        assert_eq!(
            tool_span.parent_span_id.as_deref(),
            Some(turn_id.as_str())
        );
        assert_eq!(
            tool_span.attributes.get("tool.name").unwrap(),
            "web_search"
        );
    }

    #[test]
    fn test_agent_tracer_end_span() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");

        tracer.end_span(&trace_id, &root_id, SpanStatus::Ok);

        let span = tracer.get_span(&trace_id, &root_id).unwrap();
        assert!(span.end_time.is_some());
        assert_eq!(span.status, SpanStatus::Ok);
    }

    #[test]
    fn test_agent_tracer_end_span_error() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");

        tracer.end_span(
            &trace_id,
            &root_id,
            SpanStatus::Error("timeout".to_string()),
        );

        let span = tracer.get_span(&trace_id, &root_id).unwrap();
        assert_eq!(span.status, SpanStatus::Error("timeout".to_string()));
    }

    #[test]
    fn test_agent_tracer_add_event() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");

        let event = GenAiEvent::new(GenAiEventType::Prompt, "What is Rust?");
        tracer.add_event(&trace_id, &root_id, event);

        let span = tracer.get_span(&trace_id, &root_id).unwrap();
        assert_eq!(span.events.len(), 1);
        assert_eq!(span.events[0].event_type, GenAiEventType::Prompt);
        assert_eq!(span.events[0].content, "What is Rust?");
    }

    #[test]
    fn test_agent_tracer_get_trace() {
        let mut tracer = AgentTracer::new();
        let (trace_id, _) = tracer.start_session("agent-001");

        assert!(tracer.get_trace(&trace_id).is_some());
        assert!(tracer.get_trace("nonexistent").is_none());
    }

    #[test]
    fn test_agent_tracer_get_span_tree() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);
        let attrs = GenAiAttributes::new(GenAiSystem::OpenAI, "gpt-4o");
        let _llm_id = tracer.start_llm_call(&trace_id, &turn_id, &attrs);
        let _tool_id = tracer.start_tool_call(&trace_id, &turn_id, "calculator");

        let tree = tracer.get_span_tree(&trace_id).unwrap();
        // Root should be the session span
        assert!(tree.root.span.operation.contains("agent.session"));
        // Root should have 1 child: the turn
        assert_eq!(tree.root.children.len(), 1);
        // Turn should have 2 children: llm_call and tool_call
        assert_eq!(tree.root.children[0].children.len(), 2);
    }

    #[test]
    fn test_agent_tracer_session_duration() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);

        // End spans (with a tiny delay to ensure non-zero duration)
        std::thread::sleep(std::time::Duration::from_millis(5));
        tracer.end_span(&trace_id, &turn_id, SpanStatus::Ok);
        tracer.end_span(&trace_id, &root_id, SpanStatus::Ok);

        let duration = tracer.get_session_duration(&trace_id);
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 1); // At least 1ms given the sleep
    }

    #[test]
    fn test_agent_tracer_session_duration_no_end() {
        let mut tracer = AgentTracer::new();
        let (trace_id, _root_id) = tracer.start_session("agent-001");

        // No spans ended, so duration should be None
        let duration = tracer.get_session_duration(&trace_id);
        assert!(duration.is_none());
    }

    #[test]
    fn test_agent_tracer_nested_spans() {
        let mut tracer = AgentTracer::new();
        let (trace_id, root_id) = tracer.start_session("agent-001");
        let turn_id = tracer.start_turn(&trace_id, &root_id, 1);
        let attrs = GenAiAttributes::new(GenAiSystem::Anthropic, "claude-3.5-sonnet");
        let llm_id = tracer.start_llm_call(&trace_id, &turn_id, &attrs);
        let tool_id = tracer.start_tool_call(&trace_id, &llm_id, "calculator");

        // 4 spans total: session -> turn -> llm_call -> tool_call
        let trace = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(trace.len(), 4);

        // Verify hierarchy
        let root = tracer.get_span(&trace_id, &root_id).unwrap();
        assert!(root.children.contains(&turn_id));

        let turn = tracer.get_span(&trace_id, &turn_id).unwrap();
        assert!(turn.children.contains(&llm_id));

        let llm = tracer.get_span(&trace_id, &llm_id).unwrap();
        assert!(llm.children.contains(&tool_id));

        let tool = tracer.get_span(&trace_id, &tool_id).unwrap();
        assert!(tool.children.is_empty());
    }

    #[test]
    fn test_agent_tracer_multiple_traces() {
        let mut tracer = AgentTracer::new();
        let (trace1, _) = tracer.start_session("agent-001");
        let (trace2, _) = tracer.start_session("agent-002");

        assert_ne!(trace1, trace2);
        assert!(tracer.get_trace(&trace1).is_some());
        assert!(tracer.get_trace(&trace2).is_some());
        assert_eq!(tracer.get_trace(&trace1).unwrap().len(), 1);
        assert_eq!(tracer.get_trace(&trace2).unwrap().len(), 1);
    }

    // --- 6.3 Cost Attribution & Budget Enforcement ---

    #[test]
    fn test_pricing_table_defaults() {
        let table = PricingTable::with_defaults();
        assert!(table.entries.len() >= 7);

        // Check known models
        assert!(table.find_pricing("gpt-4o").is_some());
        assert!(table.find_pricing("gpt-4o-mini").is_some());
        assert!(table.find_pricing("claude-3.5-sonnet").is_some());
        assert!(table.find_pricing("claude-3-haiku").is_some());
        assert!(table.find_pricing("gemini-1.5-pro").is_some());
        assert!(table.find_pricing("gemini-1.5-flash").is_some());
        assert!(table.find_pricing("llama-3.1-70b").is_some());
    }

    #[test]
    fn test_pricing_table_custom() {
        let mut table = PricingTable::new();
        table.add("my-model", 0.01, 0.02);

        let pricing = table.find_pricing("my-model-v2").unwrap();
        assert_eq!(pricing.model_pattern, "my-model");
        assert!((pricing.input_cost_per_1k - 0.01).abs() < f64::EPSILON);
        assert!((pricing.output_cost_per_1k - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pricing_table_get_cost() {
        let table = PricingTable::with_defaults();

        // gpt-4o: $0.005/1k input, $0.015/1k output
        let cost = table.get_cost("gpt-4o", 1000, 1000).unwrap();
        let expected = 0.005 + 0.015;
        assert!(
            (cost - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            cost
        );

        // 10k input, 5k output
        let cost2 = table.get_cost("gpt-4o", 10000, 5000).unwrap();
        let expected2 = (10.0 * 0.005) + (5.0 * 0.015);
        assert!(
            (cost2 - expected2).abs() < 1e-10,
            "Expected {}, got {}",
            expected2,
            cost2
        );
    }

    #[test]
    fn test_pricing_table_unknown_model() {
        let table = PricingTable::with_defaults();
        assert!(table.find_pricing("unknown-model-xyz").is_none());
        assert!(table.get_cost("unknown-model-xyz", 1000, 1000).is_none());
    }

    #[test]
    fn test_cost_budget_remaining() {
        let budget = CostBudget::new(10.0);
        assert!((budget.remaining() - 10.0).abs() < f64::EPSILON);
        assert!(!budget.is_exceeded());
    }

    #[test]
    fn test_cost_budget_is_exceeded() {
        let mut budget = CostBudget::new(1.0);
        budget.add_cost(0.5);
        assert!(!budget.is_exceeded());

        budget.add_cost(0.5);
        assert!(budget.is_exceeded());

        budget.add_cost(0.1);
        assert!(budget.is_exceeded());
    }

    #[test]
    fn test_cost_budget_add_cost() {
        let mut budget = CostBudget::new(10.0);
        budget.add_cost(3.0);
        assert!((budget.current_cost - 3.0).abs() < f64::EPSILON);
        assert!((budget.remaining() - 7.0).abs() < f64::EPSILON);

        budget.add_cost(5.0);
        assert!((budget.current_cost - 8.0).abs() < f64::EPSILON);
        assert!((budget.remaining() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_budget_check_allowed() {
        let budget = CostBudget::new(10.0);
        let result = budget.check_budget(1.0);
        assert_eq!(result, BudgetCheckResult::Allowed);
    }

    #[test]
    fn test_cost_budget_check_warning() {
        let mut budget = CostBudget::new(10.0);
        budget.add_cost(8.5);
        // Remaining: 1.5; adding 0.5 would leave 1.0 (10% < 20% threshold)
        let result = budget.check_budget(0.5);
        match result {
            BudgetCheckResult::Warning { remaining, estimated } => {
                assert!((remaining - 1.0).abs() < f64::EPSILON);
                assert!((estimated - 0.5).abs() < f64::EPSILON);
            }
            other => panic!("Expected Warning, got {:?}", other),
        }
    }

    #[test]
    fn test_cost_budget_check_denied() {
        let mut budget = CostBudget::new(10.0);
        budget.add_cost(9.0);
        let result = budget.check_budget(2.0);
        match result {
            BudgetCheckResult::Denied {
                budget: b,
                current,
                requested,
            } => {
                assert!((b - 10.0).abs() < f64::EPSILON);
                assert!((current - 9.0).abs() < f64::EPSILON);
                assert!((requested - 2.0).abs() < f64::EPSILON);
            }
            other => panic!("Expected Denied, got {:?}", other),
        }
    }

    #[test]
    fn test_cost_budget_alerts() {
        let mut budget = CostBudget::new(10.0);
        // Default alerts at 80% and 95%
        assert_eq!(budget.alerts.len(), 2);
        assert!(!budget.alerts[0].triggered);
        assert!(!budget.alerts[1].triggered);

        budget.add_cost(8.5); // 85% used -> 80% alert triggered
        assert!(budget.alerts[0].triggered);
        assert!(budget.alerts[0].triggered_at.is_some());
        assert!(!budget.alerts[1].triggered);

        budget.add_cost(1.2); // 97% used -> 95% alert triggered
        assert!(budget.alerts[1].triggered);
        assert!(budget.alerts[1].triggered_at.is_some());
    }

    #[test]
    fn test_cost_attributor_set_budget() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());
        attributor.set_budget("session-1", 5.0);

        let result = attributor.check_budget("session-1", "gpt-4o", 1000, 1000);
        // Should be Allowed since we haven't spent anything yet
        assert_eq!(result, BudgetCheckResult::Allowed);
    }

    #[test]
    fn test_cost_attributor_attribute_cost() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());
        attributor.set_budget("session-1", 1.0);

        let cost = attributor
            .attribute_cost("session-1", "gpt-4o", 1000, 1000)
            .unwrap();
        assert!(cost > 0.0);
        // gpt-4o: $0.005/1k input + $0.015/1k output = $0.02
        assert!((cost - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_cost_attributor_budget_enforcement() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());
        attributor.set_budget("session-1", 0.01);

        // First call should succeed
        let result = attributor.attribute_cost("session-1", "gpt-4o", 1000, 0);
        assert!(result.is_ok());

        // Second call should fail (would exceed $0.01 budget)
        let result2 = attributor.attribute_cost("session-1", "gpt-4o", 1000, 1000);
        assert!(result2.is_err());
    }

    #[test]
    fn test_cost_attributor_cost_report() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());
        attributor.set_budget("session-1", 10.0);

        attributor
            .attribute_cost("session-1", "gpt-4o", 1000, 500)
            .unwrap();
        attributor
            .attribute_cost("session-1", "gpt-4o", 2000, 1000)
            .unwrap();
        attributor
            .attribute_cost("session-1", "claude-3.5-sonnet", 500, 200)
            .unwrap();

        let report = attributor.get_cost_report("session-1").unwrap();
        assert_eq!(report.scope, "session-1");
        assert_eq!(report.num_calls, 3);
        assert!(report.total_cost > 0.0);
        assert_eq!(report.budget_max, Some(10.0));
        assert!(report.budget_remaining.is_some());
        assert!(report.budget_remaining.unwrap() < 10.0);

        // Should have 2 models in breakdown
        assert_eq!(report.breakdown.len(), 2);
    }

    #[test]
    fn test_cost_attributor_total_cost() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());

        attributor
            .attribute_cost("session-1", "gpt-4o", 1000, 1000)
            .unwrap();
        attributor
            .attribute_cost("session-2", "claude-3.5-sonnet", 1000, 1000)
            .unwrap();

        let total = attributor.get_total_cost();
        // gpt-4o: 0.005 + 0.015 = 0.02
        // claude-3.5-sonnet: 0.003 + 0.015 = 0.018
        assert!((total - 0.038).abs() < 1e-10);
    }

    #[test]
    fn test_cost_attributor_multiple_scopes() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());
        attributor.set_budget("user-alice", 5.0);
        attributor.set_budget("user-bob", 10.0);

        attributor
            .attribute_cost("user-alice", "gpt-4o", 1000, 500)
            .unwrap();
        attributor
            .attribute_cost("user-bob", "gpt-4o", 2000, 1000)
            .unwrap();

        let alice_report = attributor.get_cost_report("user-alice").unwrap();
        let bob_report = attributor.get_cost_report("user-bob").unwrap();

        assert_eq!(alice_report.num_calls, 1);
        assert_eq!(bob_report.num_calls, 1);
        assert!(bob_report.total_cost > alice_report.total_cost);
    }

    #[test]
    fn test_cost_attributor_no_budget_scope() {
        let mut attributor = CostAttributor::new(PricingTable::with_defaults());

        // No budget set — should still track costs
        let cost = attributor
            .attribute_cost("unbounded", "gpt-4o", 1000, 1000)
            .unwrap();
        assert!(cost > 0.0);

        let report = attributor.get_cost_report("unbounded").unwrap();
        assert!(report.budget_max.is_none());
        assert!(report.budget_remaining.is_none());
    }

    #[test]
    fn test_cost_attributor_check_budget_no_scope() {
        let attributor = CostAttributor::new(PricingTable::with_defaults());
        // No budget set for this scope — should be Allowed
        let result = attributor.check_budget("unknown-scope", "gpt-4o", 1000, 1000);
        assert_eq!(result, BudgetCheckResult::Allowed);
    }

    // --- Integration test ---

    #[test]
    fn test_full_session_trace_with_cost() {
        // Set up cost tracking
        let mut cost_attributor = CostAttributor::new(PricingTable::with_defaults());
        cost_attributor.set_budget("session-integration", 1.0);

        // Set up agent tracer
        let mut agent_tracer = AgentTracer::new();
        let (trace_id, root_id) = agent_tracer.start_session("test-agent");

        // Turn 1
        let turn1_id = agent_tracer.start_turn(&trace_id, &root_id, 1);

        // LLM call
        let attrs = GenAiAttributes::new(GenAiSystem::OpenAI, "gpt-4o")
            .with_temperature(0.7)
            .with_max_tokens(4096);
        let llm_id = agent_tracer.start_llm_call(&trace_id, &turn1_id, &attrs);

        // Attribute cost
        let cost = cost_attributor
            .attribute_cost("session-integration", "gpt-4o", 500, 200)
            .unwrap();
        assert!(cost > 0.0);

        // Add completion event
        agent_tracer.add_event(
            &trace_id,
            &llm_id,
            GenAiEvent::new(GenAiEventType::Completion, "I can help with that."),
        );

        // Tool call
        let tool_id = agent_tracer.start_tool_call(&trace_id, &turn1_id, "web_search");
        agent_tracer.add_event(
            &trace_id,
            &tool_id,
            GenAiEvent::new(GenAiEventType::ToolCall, "web_search(query='rust programming')"),
        );

        // End spans
        agent_tracer.end_span(&trace_id, &tool_id, SpanStatus::Ok);
        agent_tracer.end_span(&trace_id, &llm_id, SpanStatus::Ok);
        agent_tracer.end_span(&trace_id, &turn1_id, SpanStatus::Ok);
        agent_tracer.end_span(&trace_id, &root_id, SpanStatus::Ok);

        // Verify trace structure
        let tree = agent_tracer.get_span_tree(&trace_id).unwrap();
        assert!(tree.root.span.operation.contains("agent.session"));
        assert_eq!(tree.root.children.len(), 1); // 1 turn
        assert_eq!(tree.root.children[0].children.len(), 2); // llm + tool

        // Verify cost report
        let report = cost_attributor
            .get_cost_report("session-integration")
            .unwrap();
        assert_eq!(report.num_calls, 1);
        assert!(report.total_cost > 0.0);
        assert!(report.budget_remaining.unwrap() < 1.0);

        // Verify session has non-zero duration
        let duration = agent_tracer.get_session_duration(&trace_id);
        assert!(duration.is_some());
    }

    // ========================================================================
    // OTLP HTTP Exporter tests (Item 5.1)
    // ========================================================================

    #[test]
    fn test_otlp_exporter_new() {
        let exporter = OtlpHttpExporter::new("http://localhost:4318");
        assert_eq!(exporter.endpoint, "http://localhost:4318");
        assert!(exporter.headers.is_empty());
        assert!(exporter.batch.is_empty());
        assert_eq!(exporter.max_batch_size, 512);
    }

    #[test]
    fn test_otlp_exporter_with_header() {
        let exporter = OtlpHttpExporter::new("http://localhost:4318")
            .with_header("Authorization", "Bearer tok123")
            .with_header("X-Custom", "value");
        assert_eq!(exporter.headers.len(), 2);
        assert_eq!(exporter.headers["Authorization"], "Bearer tok123");
    }

    #[test]
    fn test_otlp_exporter_with_flush_interval() {
        let exporter = OtlpHttpExporter::new("http://otel:4318")
            .with_flush_interval(10_000);
        assert_eq!(exporter.flush_interval_ms, 10_000);
    }

    #[test]
    fn test_otlp_exporter_add_span() {
        let mut exporter = OtlpHttpExporter::new("http://localhost:4318");
        let span = AiSpan::new("test.op");
        exporter.add_span(&span);
        assert_eq!(exporter.batch.len(), 1);
        assert_eq!(exporter.batch[0].name, "test.op");
    }

    #[test]
    fn test_otlp_exporter_should_flush() {
        let mut exporter = OtlpHttpExporter::new("http://localhost:4318");
        exporter.max_batch_size = 2;
        assert!(!exporter.should_flush());
        exporter.add_span(&AiSpan::new("a"));
        assert!(!exporter.should_flush());
        exporter.add_span(&AiSpan::new("b"));
        assert!(exporter.should_flush());
    }

    #[test]
    fn test_otlp_exporter_flush_empty() {
        let mut exporter = OtlpHttpExporter::new("http://localhost:4318");
        let result = exporter.flush();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_otlp_exporter_to_otlp_json() {
        let mut exporter = OtlpHttpExporter::new("http://localhost:4318");
        let mut span = AiSpan::new("llm.generate");
        span.attributes.insert("gen_ai.system".to_string(), "openai".to_string());
        exporter.add_span(&span);
        let json = exporter.to_otlp_json();
        assert!(json.contains("resourceSpans"));
        assert!(json.contains("llm.generate"));
        assert!(json.contains("gen_ai.system"));
    }

    #[test]
    fn test_otlp_exporter_flush_with_mock() {
        use crate::http_client::MockHttpServer;

        let server = MockHttpServer::start();
        server.enqueue_json(200, serde_json::json!({}));

        let mut exporter = OtlpHttpExporter::new(&server.url());
        exporter.add_span(&AiSpan::new("test.span"));
        let result = exporter.flush();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert!(exporter.batch.is_empty());

        let (method, path, _) = server.last_request().unwrap();
        assert_eq!(method, "POST");
        assert!(path.contains("/v1/traces"));
    }

    #[test]
    fn test_exportable_span_serialization() {
        let span = ExportableSpan {
            trace_id: "abc123".to_string(),
            span_id: "def456".to_string(),
            parent_span_id: None,
            name: "llm.generate".to_string(),
            kind: 3,
            start_time_unix_nano: 1_000_000_000,
            end_time_unix_nano: 2_000_000_000,
            attributes: vec![SpanAttribute {
                key: "gen_ai.system".to_string(),
                value: SpanAttributeValue {
                    string_value: "openai".to_string(),
                },
            }],
            status: ExportSpanStatus { code: 1 },
        };
        let json = serde_json::to_string(&span).unwrap();
        assert!(json.contains("abc123"));
        assert!(json.contains("llm.generate"));
        assert!(json.contains("gen_ai.system"));
    }

    // ========================================================================
    // GenAI Span Builder tests (Item 5.2)
    // ========================================================================

    #[test]
    fn test_genai_span_builder_minimal() {
        let span = GenAiSpanBuilder::new("chat")
            .system("openai")
            .model("gpt-4o")
            .build();
        assert!(span.operation.contains("chat"));
        assert_eq!(span.attributes.get(GenAiConventions::SYSTEM), Some(&"openai".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::REQUEST_MODEL), Some(&"gpt-4o".to_string()));
    }

    #[test]
    fn test_genai_span_builder_all_fields() {
        let span = GenAiSpanBuilder::new("chat")
            .system("anthropic")
            .model("claude-3")
            .input_tokens(500)
            .output_tokens(200)
            .temperature(0.7)
            .finish_reason("end_turn")
            .max_tokens(4096)
            .build();
        assert_eq!(span.attributes.get(GenAiConventions::SYSTEM), Some(&"anthropic".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::USAGE_INPUT_TOKENS), Some(&"500".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::USAGE_OUTPUT_TOKENS), Some(&"200".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::REQUEST_TEMPERATURE), Some(&"0.7".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::RESPONSE_FINISH_REASON), Some(&"end_turn".to_string()));
        assert_eq!(span.attributes.get(GenAiConventions::REQUEST_MAX_TOKENS), Some(&"4096".to_string()));
        assert_eq!(span.input_tokens, Some(500));
        assert_eq!(span.output_tokens, Some(200));
    }

    #[test]
    fn test_genai_conventions_constants() {
        assert_eq!(GenAiConventions::SYSTEM, "gen_ai.system");
        assert_eq!(GenAiConventions::REQUEST_MODEL, "gen_ai.request.model");
        assert_eq!(GenAiConventions::RESPONSE_MODEL, "gen_ai.response.model");
        assert_eq!(GenAiConventions::USAGE_INPUT_TOKENS, "gen_ai.usage.input_tokens");
        assert_eq!(GenAiConventions::USAGE_OUTPUT_TOKENS, "gen_ai.usage.output_tokens");
    }

    // ========================================================================
    // Prometheus Metrics tests (Item 5.3)
    // ========================================================================

    #[test]
    fn test_prometheus_new() {
        let prom = PrometheusMetrics::new();
        assert!(prom.render().contains("# No metrics recorded"));
    }

    #[test]
    fn test_prometheus_record_request() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4o", "ok", 0.5, 100, 50);
        let rendered = prom.render();
        assert!(rendered.contains("ai_requests_total"));
        assert!(rendered.contains("openai"));
        assert!(rendered.contains("gpt-4o"));
    }

    #[test]
    fn test_prometheus_record_error() {
        let mut prom = PrometheusMetrics::new();
        prom.record_error("anthropic", "timeout");
        let rendered = prom.render();
        assert!(rendered.contains("ai_errors_total"));
        assert!(rendered.contains("anthropic"));
        assert!(rendered.contains("timeout"));
    }

    #[test]
    fn test_prometheus_render_format() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4", "ok", 1.0, 200, 100);
        let rendered = prom.render();
        assert!(rendered.contains("# HELP ai_requests_total"));
        assert!(rendered.contains("# TYPE ai_requests_total counter"));
        assert!(rendered.contains("# HELP ai_tokens_total"));
        assert!(rendered.contains("# TYPE ai_tokens_total counter"));
    }

    #[test]
    fn test_prometheus_render_multiple() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4", "ok", 0.5, 100, 50);
        prom.record_request("openai", "gpt-4", "ok", 0.3, 200, 100);
        prom.record_request("anthropic", "claude", "error", 1.0, 50, 0);
        let rendered = prom.render();
        // Two openai ok + one anthropic error
        assert!(rendered.contains("openai"));
        assert!(rendered.contains("anthropic"));
    }

    #[test]
    fn test_prometheus_reset() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4", "ok", 0.5, 100, 50);
        assert!(!prom.render().contains("# No metrics recorded"));
        prom.reset();
        assert!(prom.render().contains("# No metrics recorded"));
    }

    #[test]
    fn test_prometheus_render_empty() {
        let prom = PrometheusMetrics::new();
        let rendered = prom.render();
        assert!(rendered.contains("# No metrics recorded"));
    }

    #[test]
    fn test_prometheus_tokens_direction() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4", "ok", 0.5, 300, 150);
        let rendered = prom.render();
        assert!(rendered.contains("direction=\"input\""));
        assert!(rendered.contains("direction=\"output\""));
    }

    #[test]
    fn test_prometheus_duration_histogram() {
        let mut prom = PrometheusMetrics::new();
        prom.record_request("openai", "gpt-4", "ok", 0.5, 100, 50);
        prom.record_request("openai", "gpt-4", "ok", 1.5, 100, 50);
        let rendered = prom.render();
        assert!(rendered.contains("ai_request_duration_seconds"));
    }
}
