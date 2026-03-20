//! Distributed log correlation and remote log collection.
//!
//! Provides a unified logging format that correlates log entries across
//! distributed nodes using a shared `trace_id`. When node A distributes
//! work to nodes B and C, each node logs with the same `trace_id` and
//! can return its logs alongside the response. Node A merges all logs
//! into a single, time-sorted unified view.
//!
//! # Architecture
//!
//! ```text
//! User → Node A (generates trace_id)
//!   ├─ logs locally [trace=abc, node=A]
//!   ├─ sends to B with TraceContext{trace_id: abc, collect_logs: true}
//!   │   └─ B logs [trace=abc, node=B], returns logs in response
//!   ├─ sends to C with TraceContext{trace_id: abc, collect_logs: true}
//!   │   └─ C logs [trace=abc, node=C], returns logs in response
//!   └─ A merges all → unified log sorted by timestamp
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Log entry types
// ============================================================================

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// A log entry that can be correlated across distributed nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DistributedLogEntry {
    /// Global trace ID shared across all nodes for this request.
    pub trace_id: String,
    /// ID of the node that generated this entry.
    pub node_id: String,
    /// Span ID within this node (local operation identifier).
    pub span_id: String,
    /// Parent span ID (for nested operations).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// Log severity level.
    pub level: LogLevel,
    /// Human-readable log message.
    pub message: String,
    /// Operation being performed (e.g., "llm.generate", "rag.query").
    pub operation: String,
    /// Duration in milliseconds (set when operation completes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Additional key-value attributes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, String>,
}

impl DistributedLogEntry {
    /// Create a new log entry with the current timestamp.
    pub fn new(
        trace_id: &str,
        node_id: &str,
        level: LogLevel,
        operation: &str,
        message: &str,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let span_id = format!(
            "{:08x}",
            timestamp_ms.wrapping_mul(0x100000001b3) as u32
        );

        Self {
            trace_id: trace_id.to_string(),
            node_id: node_id.to_string(),
            span_id,
            parent_span_id: None,
            timestamp_ms,
            level,
            message: message.to_string(),
            operation: operation.to_string(),
            duration_ms: None,
            attributes: HashMap::new(),
        }
    }

    /// Add a key-value attribute.
    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Set the parent span ID.
    pub fn with_parent(mut self, parent_span_id: &str) -> Self {
        self.parent_span_id = Some(parent_span_id.to_string());
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Format as a human-readable single line.
    pub fn to_text(&self) -> String {
        let dur = self
            .duration_ms
            .map(|d| format!(" ({}ms)", d))
            .unwrap_or_default();
        format!(
            "[{}] {} | {} | {} | {}{}",
            self.timestamp_ms, self.level, self.node_id, self.operation, self.message, dur
        )
    }
}

// ============================================================================
// Trace context (propagated across nodes)
// ============================================================================

/// Context that travels with every distributed request to enable log correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TraceContext {
    /// Global trace ID (generated at the entry point, propagated to all nodes).
    pub trace_id: String,
    /// Node that originated the request.
    pub origin_node: String,
    /// Parent span ID in the calling node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Whether remote nodes should include their logs in the response.
    pub collect_logs: bool,
    /// Maximum log entries to return (0 = unlimited).
    pub max_log_entries: usize,
    /// Minimum log level to collect from remote nodes.
    pub min_log_level: LogLevel,
}

impl TraceContext {
    /// Create a new trace context with a generated trace_id.
    pub fn new(origin_node: &str) -> Self {
        Self {
            trace_id: generate_trace_id(),
            origin_node: origin_node.to_string(),
            parent_span_id: None,
            collect_logs: false,
            max_log_entries: 0,
            min_log_level: LogLevel::Info,
        }
    }

    /// Create with log collection enabled.
    pub fn with_log_collection(origin_node: &str) -> Self {
        Self {
            trace_id: generate_trace_id(),
            origin_node: origin_node.to_string(),
            parent_span_id: None,
            collect_logs: true,
            max_log_entries: 100,
            min_log_level: LogLevel::Info,
        }
    }

    /// Create from an existing trace_id (for child requests).
    pub fn child(trace_id: &str, origin_node: &str, parent_span_id: &str) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            origin_node: origin_node.to_string(),
            parent_span_id: Some(parent_span_id.to_string()),
            collect_logs: true,
            max_log_entries: 100,
            min_log_level: LogLevel::Info,
        }
    }

    /// Set maximum log entries to collect.
    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_log_entries = max;
        self
    }

    /// Set minimum log level.
    pub fn min_level(mut self, level: LogLevel) -> Self {
        self.min_log_level = level;
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new("unknown")
    }
}

/// Generate a trace ID (hex-encoded timestamp + random component).
pub fn generate_trace_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos() as u64;
    let hash = nanos.wrapping_mul(0x517cc1b727220a95);
    format!("{:016x}{:016x}", nanos, hash)
}

// ============================================================================
// Log collector
// ============================================================================

/// Configuration for the distributed log collector.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LogCollectorConfig {
    /// Enable log collection.
    pub enabled: bool,
    /// Maximum entries per trace.
    pub max_entries_per_trace: usize,
    /// Maximum active traces to track simultaneously.
    pub max_active_traces: usize,
    /// Trace retention in seconds (after last entry).
    pub retention_secs: u64,
    /// Share logs with remote nodes when they request them.
    pub share_logs: bool,
    /// Minimum level to collect.
    pub min_level: LogLevel,
}

impl Default for LogCollectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries_per_trace: 500,
            max_active_traces: 100,
            retention_secs: 300, // 5 minutes
            share_logs: true,
            min_level: LogLevel::Info,
        }
    }
}

/// Collects and stores log entries indexed by trace_id.
///
/// Supports merging logs from remote nodes to create a unified view
/// of a distributed operation.
pub struct LogCollector {
    /// Entries indexed by trace_id.
    entries: HashMap<String, Vec<DistributedLogEntry>>,
    /// Last activity timestamp per trace (for expiration).
    last_activity: HashMap<String, u64>,
    /// Configuration.
    config: LogCollectorConfig,
    /// This node's ID.
    node_id: String,
}

impl std::fmt::Debug for LogCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogCollector")
            .field("node_id", &self.node_id)
            .field("active_traces", &self.entries.len())
            .field("config", &self.config)
            .finish()
    }
}

impl LogCollector {
    /// Create a new log collector for the given node.
    pub fn new(node_id: &str, config: LogCollectorConfig) -> Self {
        Self {
            entries: HashMap::new(),
            last_activity: HashMap::new(),
            config,
            node_id: node_id.to_string(),
        }
    }

    /// Log an entry for a trace.
    pub fn log(
        &mut self,
        trace_id: &str,
        level: LogLevel,
        operation: &str,
        message: &str,
    ) {
        if !self.config.enabled || level < self.config.min_level {
            return;
        }

        let entry = DistributedLogEntry::new(trace_id, &self.node_id, level, operation, message);
        self.add_entry(trace_id, entry);
    }

    /// Log an entry with attributes.
    pub fn log_with_attrs(
        &mut self,
        trace_id: &str,
        level: LogLevel,
        operation: &str,
        message: &str,
        attrs: &[(&str, &str)],
    ) {
        if !self.config.enabled || level < self.config.min_level {
            return;
        }

        let mut entry =
            DistributedLogEntry::new(trace_id, &self.node_id, level, operation, message);
        for (k, v) in attrs {
            entry.attributes.insert(k.to_string(), v.to_string());
        }
        self.add_entry(trace_id, entry);
    }

    /// Add a pre-built entry.
    pub fn add_entry(&mut self, trace_id: &str, entry: DistributedLogEntry) {
        if !self.config.enabled {
            return;
        }

        // Enforce max active traces
        if !self.entries.contains_key(trace_id)
            && self.entries.len() >= self.config.max_active_traces
        {
            self.evict_oldest_trace();
        }

        let entries = self.entries.entry(trace_id.to_string()).or_default();

        // Enforce max entries per trace
        if entries.len() >= self.config.max_entries_per_trace {
            entries.remove(0); // Remove oldest
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_activity.insert(trace_id.to_string(), now_ms);

        entries.push(entry);
    }

    /// Get all log entries for a trace (this node only).
    pub fn get_trace_logs(&self, trace_id: &str) -> Vec<&DistributedLogEntry> {
        self.entries
            .get(trace_id)
            .map(|e| e.iter().collect())
            .unwrap_or_default()
    }

    /// Get entries for a trace, filtered for sharing with a remote node.
    ///
    /// Returns `None` if log sharing is disabled.
    pub fn get_shareable_logs(
        &self,
        trace_id: &str,
        max_entries: usize,
        min_level: LogLevel,
    ) -> Option<Vec<DistributedLogEntry>> {
        if !self.config.share_logs {
            return None;
        }

        let entries = self.entries.get(trace_id)?;
        let filtered: Vec<DistributedLogEntry> = entries
            .iter()
            .filter(|e| e.level >= min_level)
            .take(if max_entries > 0 {
                max_entries
            } else {
                usize::MAX
            })
            .cloned()
            .collect();

        Some(filtered)
    }

    /// Merge log entries received from a remote node.
    pub fn merge_remote_logs(
        &mut self,
        trace_id: &str,
        remote_logs: Vec<DistributedLogEntry>,
    ) {
        if !self.config.enabled || remote_logs.is_empty() {
            return;
        }

        let entries = self.entries.entry(trace_id.to_string()).or_default();
        entries.extend(remote_logs);

        // Sort by timestamp after merge
        entries.sort_by_key(|e| e.timestamp_ms);

        // Trim if over limit
        while entries.len() > self.config.max_entries_per_trace {
            entries.remove(0);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_activity.insert(trace_id.to_string(), now_ms);
    }

    /// Get the unified log for a trace (all nodes, sorted by timestamp).
    pub fn get_unified_log(&self, trace_id: &str) -> Vec<&DistributedLogEntry> {
        let mut entries: Vec<&DistributedLogEntry> = self
            .entries
            .get(trace_id)
            .map(|e| e.iter().collect())
            .unwrap_or_default();
        entries.sort_by_key(|e| e.timestamp_ms);
        entries
    }

    /// Remove expired traces based on retention_secs.
    pub fn cleanup_expired(&mut self) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let retention_ms = self.config.retention_secs * 1000;

        let expired: Vec<String> = self
            .last_activity
            .iter()
            .filter(|(_, &last)| now_ms.saturating_sub(last) > retention_ms)
            .map(|(k, _)| k.clone())
            .collect();

        for trace_id in &expired {
            self.entries.remove(trace_id);
            self.last_activity.remove(trace_id);
        }
    }

    /// Export a trace's unified log in the given format.
    pub fn export_trace(&self, trace_id: &str, format: ExportFormat) -> String {
        let entries = self.get_unified_log(trace_id);

        match format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(&entries.iter().cloned().collect::<Vec<_>>())
                    .unwrap_or_else(|_| "[]".to_string())
            }
            ExportFormat::Text => entries.iter().map(|e| e.to_text()).collect::<Vec<_>>().join("\n"),
            ExportFormat::Csv => {
                let mut csv = String::from(
                    "timestamp_ms,level,node_id,operation,message,duration_ms,trace_id,span_id\n",
                );
                for e in &entries {
                    csv.push_str(&format!(
                        "{},{},{},{},{},{},{},{}\n",
                        e.timestamp_ms,
                        e.level,
                        e.node_id,
                        e.operation,
                        e.message.replace(',', ";"),
                        e.duration_ms.map(|d| d.to_string()).unwrap_or_default(),
                        e.trace_id,
                        e.span_id,
                    ));
                }
                csv
            }
        }
    }

    /// Number of active traces being tracked.
    pub fn active_trace_count(&self) -> usize {
        self.entries.len()
    }

    /// Total number of log entries across all traces.
    pub fn total_entry_count(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Evict the oldest trace to make room for a new one.
    fn evict_oldest_trace(&mut self) {
        if let Some(oldest) = self
            .last_activity
            .iter()
            .min_by_key(|(_, &ts)| ts)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest);
            self.last_activity.remove(&oldest);
        }
    }
}

/// Export format for trace logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportFormat {
    /// Pretty-printed JSON array.
    Json,
    /// Human-readable text lines.
    Text,
    /// Comma-separated values.
    Csv,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(trace_id: &str, node: &str, level: LogLevel, op: &str, msg: &str, ts: u64) -> DistributedLogEntry {
        DistributedLogEntry {
            trace_id: trace_id.to_string(),
            node_id: node.to_string(),
            span_id: format!("span-{}", ts),
            parent_span_id: None,
            timestamp_ms: ts,
            level,
            message: msg.to_string(),
            operation: op.to_string(),
            duration_ms: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn test_distributed_log_entry_serialize() {
        let entry = DistributedLogEntry::new("trace-1", "node-A", LogLevel::Info, "test.op", "hello")
            .with_attr("key", "value")
            .with_duration(42);

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: DistributedLogEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.trace_id, "trace-1");
        assert_eq!(parsed.node_id, "node-A");
        assert_eq!(parsed.level, LogLevel::Info);
        assert_eq!(parsed.duration_ms, Some(42));
        assert_eq!(parsed.attributes.get("key").unwrap(), "value");
    }

    #[test]
    fn test_trace_context_propagation() {
        let ctx = TraceContext::with_log_collection("node-A");
        assert!(!ctx.trace_id.is_empty());
        assert_eq!(ctx.origin_node, "node-A");
        assert!(ctx.collect_logs);
        assert_eq!(ctx.max_log_entries, 100);

        let child = TraceContext::child(&ctx.trace_id, "node-B", "parent-span");
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_eq!(child.origin_node, "node-B");
        assert_eq!(child.parent_span_id.as_deref(), Some("parent-span"));
    }

    #[test]
    fn test_log_collector_basic() {
        let mut config = LogCollectorConfig::default();
        config.min_level = LogLevel::Debug; // include Debug entries
        let mut collector = LogCollector::new("node-A", config);

        collector.log("trace-1", LogLevel::Info, "test.op", "message 1");
        collector.log("trace-1", LogLevel::Debug, "test.op", "message 2");
        collector.log("trace-1", LogLevel::Warn, "test.op", "message 3");

        let logs = collector.get_trace_logs("trace-1");
        assert_eq!(logs.len(), 3);
        assert_eq!(logs[0].message, "message 1");

        assert_eq!(collector.active_trace_count(), 1);
        assert_eq!(collector.total_entry_count(), 3);
    }

    #[test]
    fn test_log_collector_merge_remote() {
        let mut collector = LogCollector::new("node-A", LogCollectorConfig::default());

        // Local log
        collector.add_entry("trace-1", make_entry("trace-1", "node-A", LogLevel::Info, "op", "local", 100));

        // Remote logs from node B
        let remote = vec![
            make_entry("trace-1", "node-B", LogLevel::Info, "op", "remote 1", 50),
            make_entry("trace-1", "node-B", LogLevel::Info, "op", "remote 2", 150),
        ];
        collector.merge_remote_logs("trace-1", remote);

        let logs = collector.get_trace_logs("trace-1");
        assert_eq!(logs.len(), 3);
    }

    #[test]
    fn test_log_collector_unified_sorted() {
        let mut collector = LogCollector::new("node-A", LogCollectorConfig::default());

        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Info, "op", "third", 300));
        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Info, "op", "first", 100));

        let remote = vec![make_entry("t1", "B", LogLevel::Info, "op", "second", 200)];
        collector.merge_remote_logs("t1", remote);

        let unified = collector.get_unified_log("t1");
        assert_eq!(unified.len(), 3);
        assert_eq!(unified[0].message, "first");
        assert_eq!(unified[1].message, "second");
        assert_eq!(unified[2].message, "third");
    }

    #[test]
    fn test_log_collector_max_entries() {
        let mut config = LogCollectorConfig::default();
        config.max_entries_per_trace = 3;
        let mut collector = LogCollector::new("A", config);

        for i in 0..5 {
            collector.add_entry("t1", make_entry("t1", "A", LogLevel::Info, "op", &format!("msg {}", i), i as u64));
        }

        let logs = collector.get_trace_logs("t1");
        assert_eq!(logs.len(), 3);
        // Oldest should have been evicted
        assert_eq!(logs[0].message, "msg 2");
    }

    #[test]
    fn test_log_collector_retention() {
        let mut config = LogCollectorConfig::default();
        config.retention_secs = 0; // Expire immediately
        let mut collector = LogCollector::new("A", config);

        collector.log("t1", LogLevel::Info, "op", "old");
        // Force last_activity to be in the past
        collector.last_activity.insert("t1".to_string(), 0);

        collector.cleanup_expired();
        assert_eq!(collector.active_trace_count(), 0);
    }

    #[test]
    fn test_log_collector_share_disabled() {
        let mut config = LogCollectorConfig::default();
        config.share_logs = false;
        let mut collector = LogCollector::new("A", config);

        collector.log("t1", LogLevel::Info, "op", "private");

        let shareable = collector.get_shareable_logs("t1", 10, LogLevel::Info);
        assert!(shareable.is_none());
    }

    #[test]
    fn test_log_collector_share_enabled_filtered() {
        let mut collector = LogCollector::new("A", LogCollectorConfig::default());

        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Debug, "op", "debug msg", 100));
        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Info, "op", "info msg", 200));
        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Error, "op", "error msg", 300));

        // Request only Warn+ level
        let shareable = collector.get_shareable_logs("t1", 10, LogLevel::Warn).unwrap();
        assert_eq!(shareable.len(), 1);
        assert_eq!(shareable[0].message, "error msg");
    }

    #[test]
    fn test_export_json() {
        let mut collector = LogCollector::new("A", LogCollectorConfig::default());
        collector.add_entry("t1", make_entry("t1", "A", LogLevel::Info, "op", "test", 100));

        let json = collector.export_trace("t1", ExportFormat::Json);
        assert!(json.contains("\"trace_id\""));
        assert!(json.contains("\"test\""));

        // Should be valid JSON
        let parsed: Vec<DistributedLogEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_export_text() {
        let mut collector = LogCollector::new("A", LogCollectorConfig::default());
        collector.add_entry("t1", make_entry("t1", "node-A", LogLevel::Warn, "rag.query", "slow query", 100));

        let text = collector.export_trace("t1", ExportFormat::Text);
        assert!(text.contains("WARN"));
        assert!(text.contains("node-A"));
        assert!(text.contains("rag.query"));
        assert!(text.contains("slow query"));
    }

    #[test]
    fn test_generate_trace_id_unique() {
        let id1 = generate_trace_id();
        let id2 = generate_trace_id();
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 32); // 2x 16 hex chars
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }
}
