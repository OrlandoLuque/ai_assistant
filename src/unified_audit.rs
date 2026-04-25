//! Unified audit bus (V104.14)
//!
//! A single, schema-stable place where every subsystem (security, sandbox,
//! plugins, autonomous loop, plan mode, conventions trust, slash commands,
//! ...) can publish audit events. Events are dispatched to any number of
//! [`AuditSink`] implementations: in-memory ring buffer, JSON-Lines file
//! with rotation, callback, ...
//!
//! ## Why a bus on top of existing per-subsystem loggers?
//!
//! Existing modules ([`crate::security::audit`], [`crate::agent_sandbox`])
//! keep their own private logs. That's fine for unit-level debugging, but
//! makes it hard to answer cross-cutting questions like "what did the
//! agent touch yesterday?" V104.14 doesn't *replace* those loggers — it
//! gives them (and any new subsystem) a shared sink so the operator has
//! one stream to tail.
//!
//! ## Schema
//!
//! Every event carries:
//! - `id`: monotonic per-process counter,
//! - `timestamp`: UTC,
//! - `severity`: Info / Notice / Warning / Error / Critical,
//! - `subsystem`: free-form string (e.g. `"agent_sandbox"`, `"plugins/lua"`),
//! - `event_kind`: subsystem-defined (e.g. `"file_write"`, `"command_blocked"`),
//! - `correlation_id`: optional — group events from one logical operation,
//! - `actor`: optional — user/agent who triggered it,
//! - `payload`: JSON object for free-form structured data.
//!
//! ## Security baseline
//!
//! - Sinks run *synchronously* on the publishing thread by default —
//!   slow sinks slow down the publisher (predictable). Async fan-out is
//!   the caller's choice.
//! - The JSON-file sink validates path on open, refuses symlinks, caps
//!   per-line size (16 KiB default), and rotates by size.
//! - PII redaction is opt-in via [`UnifiedAuditConfig::redactor`] — the
//!   redactor sees the full event before any sink does.

use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Severity
// ============================================================================

/// Severity level for an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditSeverity {
    Info,
    Notice,
    Warning,
    Error,
    Critical,
}

impl AuditSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Notice => "notice",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
}

// ============================================================================
// Event
// ============================================================================

/// A single audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAuditEvent {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub severity: AuditSeverity,
    pub subsystem: String,
    pub event_kind: String,
    pub correlation_id: Option<String>,
    pub actor: Option<String>,
    pub payload: Value,
}

impl UnifiedAuditEvent {
    pub fn new(subsystem: &str, event_kind: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self {
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
            timestamp: Utc::now(),
            severity: AuditSeverity::Info,
            subsystem: subsystem.to_string(),
            event_kind: event_kind.to_string(),
            correlation_id: None,
            actor: None,
            payload: Value::Null,
        }
    }

    pub fn severity(mut self, s: AuditSeverity) -> Self {
        self.severity = s;
        self
    }
    pub fn correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }
    pub fn actor(mut self, a: impl Into<String>) -> Self {
        self.actor = Some(a.into());
        self
    }
    pub fn payload(mut self, p: Value) -> Self {
        self.payload = p;
        self
    }
}

// ============================================================================
// Sink trait + built-in sinks
// ============================================================================

/// A consumer of audit events. Sinks must be `Send + Sync` because the
/// bus is shared across threads.
pub trait AuditSink: Send + Sync {
    fn publish(&self, event: &UnifiedAuditEvent) -> Result<(), AuditSinkError>;
    fn name(&self) -> &str;
}

/// Errors a sink can return. Bus catches these and continues with the
/// next sink — one bad sink does not kill the whole pipeline.
#[derive(Debug)]
pub enum AuditSinkError {
    Io(io::Error),
    Other(String),
}

impl std::fmt::Display for AuditSinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Other(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for AuditSinkError {}

// ---------- in-memory sink ----------

/// Bounded in-memory ring buffer; oldest events drop when full.
pub struct InMemorySink {
    name: String,
    inner: Mutex<VecDeque<UnifiedAuditEvent>>,
    capacity: usize,
}

impl InMemorySink {
    pub fn new(capacity: usize) -> Self {
        Self {
            name: "in-memory".into(),
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn snapshot(&self) -> Vec<UnifiedAuditEvent> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AuditSink for InMemorySink {
    fn publish(&self, event: &UnifiedAuditEvent) -> Result<(), AuditSinkError> {
        let mut q = self.inner.lock().unwrap();
        if q.len() >= self.capacity {
            q.pop_front();
        }
        q.push_back(event.clone());
        Ok(())
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ---------- JSON-lines file sink ----------

/// Append-only newline-delimited JSON sink with size-based rotation.
pub struct JsonLinesFileSink {
    name: String,
    path: PathBuf,
    inner: Mutex<JsonLinesState>,
    max_bytes_per_file: u64,
    max_line_bytes: usize,
    keep_rotated: usize,
}

struct JsonLinesState {
    bytes_in_current: u64,
}

impl JsonLinesFileSink {
    /// Open or create `path`. The parent must exist; symlinks are rejected.
    pub fn open(path: PathBuf) -> Result<Self, AuditSinkError> {
        if let Ok(meta) = fs::symlink_metadata(&path) {
            if meta.file_type().is_symlink() {
                return Err(AuditSinkError::Other(format!(
                    "audit log path is a symlink: {}",
                    path.display()
                )));
            }
        }
        let bytes_in_current = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        // Touch the file so we fail fast if the path is bad.
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(AuditSinkError::Io)?;
        Ok(Self {
            name: format!("jsonl: {}", path.display()),
            path,
            inner: Mutex::new(JsonLinesState { bytes_in_current }),
            max_bytes_per_file: 16 * 1024 * 1024,
            max_line_bytes: 16 * 1024,
            keep_rotated: 5,
        })
    }

    pub fn with_max_bytes_per_file(mut self, n: u64) -> Self {
        self.max_bytes_per_file = n;
        self
    }
    pub fn with_max_line_bytes(mut self, n: usize) -> Self {
        self.max_line_bytes = n;
        self
    }
    pub fn with_keep_rotated(mut self, n: usize) -> Self {
        self.keep_rotated = n;
        self
    }

    fn rotate(&self, st: &mut JsonLinesState) -> Result<(), AuditSinkError> {
        // Shift .N → .N+1, drop oldest.
        for i in (1..self.keep_rotated).rev() {
            let from = self.path.with_extension(format!("log.{}", i));
            let to = self.path.with_extension(format!("log.{}", i + 1));
            let _ = fs::rename(&from, &to);
        }
        let first = self.path.with_extension("log.1");
        let _ = fs::rename(&self.path, &first);
        // Truncate by re-creating.
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&self.path)
            .map_err(AuditSinkError::Io)?;
        st.bytes_in_current = 0;
        Ok(())
    }
}

impl AuditSink for JsonLinesFileSink {
    fn publish(&self, event: &UnifiedAuditEvent) -> Result<(), AuditSinkError> {
        let mut line =
            serde_json::to_string(event).map_err(|e| AuditSinkError::Other(e.to_string()))?;
        if line.len() > self.max_line_bytes {
            line.truncate(self.max_line_bytes);
            line.push_str("...\"_truncated\":true}");
        }
        line.push('\n');

        let mut st = self.inner.lock().unwrap();
        if st.bytes_in_current + line.len() as u64 > self.max_bytes_per_file {
            self.rotate(&mut st)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(AuditSinkError::Io)?;
        f.write_all(line.as_bytes()).map_err(AuditSinkError::Io)?;
        st.bytes_in_current += line.len() as u64;
        Ok(())
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ---------- callback sink ----------

/// Trivial sink that calls a closure for each event. Useful for tests
/// or for sending events to other in-process channels.
pub struct CallbackSink {
    name: String,
    cb: Box<dyn Fn(&UnifiedAuditEvent) + Send + Sync>,
}

impl CallbackSink {
    pub fn new<F>(cb: F) -> Self
    where
        F: Fn(&UnifiedAuditEvent) + Send + Sync + 'static,
    {
        Self {
            name: "callback".into(),
            cb: Box::new(cb),
        }
    }
}

impl AuditSink for CallbackSink {
    fn publish(&self, event: &UnifiedAuditEvent) -> Result<(), AuditSinkError> {
        (self.cb)(event);
        Ok(())
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Bus
// ============================================================================

/// Optional event redactor. Receives a mutable event before any sink sees it.
pub type Redactor = Arc<dyn Fn(&mut UnifiedAuditEvent) + Send + Sync>;

/// Configuration for the bus.
#[derive(Default)]
pub struct UnifiedAuditConfig {
    /// Optional minimum severity. Events below it are dropped.
    pub min_severity: Option<AuditSeverity>,
    /// Optional subsystem allow-list (empty = all allowed).
    pub allowed_subsystems: Vec<String>,
    /// Optional redactor invoked before sinks. Useful for stripping PII.
    pub redactor: Option<Redactor>,
}

/// The bus itself. Cheap to clone (`Arc` inside).
pub struct UnifiedAuditBus {
    inner: Arc<BusInner>,
}

struct BusInner {
    sinks: Mutex<Vec<Arc<dyn AuditSink>>>,
    cfg: UnifiedAuditConfig,
    sink_failures: AtomicU64,
}

impl Clone for UnifiedAuditBus {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl UnifiedAuditBus {
    pub fn new(cfg: UnifiedAuditConfig) -> Self {
        Self {
            inner: Arc::new(BusInner {
                sinks: Mutex::new(Vec::new()),
                cfg,
                sink_failures: AtomicU64::new(0),
            }),
        }
    }

    pub fn add_sink(&self, sink: Arc<dyn AuditSink>) {
        self.inner.sinks.lock().unwrap().push(sink);
    }

    pub fn publish(&self, mut event: UnifiedAuditEvent) {
        if let Some(min) = self.inner.cfg.min_severity {
            if (event.severity as u8) < (min as u8) {
                return;
            }
        }
        if !self.inner.cfg.allowed_subsystems.is_empty()
            && !self
                .inner
                .cfg
                .allowed_subsystems
                .iter()
                .any(|s| s == &event.subsystem)
        {
            return;
        }
        if let Some(red) = &self.inner.cfg.redactor {
            red(&mut event);
        }
        let sinks = self.inner.sinks.lock().unwrap().clone();
        for sink in sinks {
            if sink.publish(&event).is_err() {
                self.inner.sink_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn sink_failure_count(&self) -> u64 {
        self.inner.sink_failures.load(Ordering::Relaxed)
    }

    pub fn sink_count(&self) -> usize {
        self.inner.sinks.lock().unwrap().len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(subsystem: &str, kind: &str, sev: AuditSeverity) -> UnifiedAuditEvent {
        UnifiedAuditEvent::new(subsystem, kind).severity(sev)
    }

    fn tmpfile(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ai_assistant_audit_{}_{}.log",
            name,
            std::process::id()
        ));
        let _ = fs::remove_file(&p);
        p
    }

    // ---------- in-memory sink ----------

    #[test]
    fn in_memory_drops_oldest_when_full() {
        let sink = InMemorySink::new(2);
        sink.publish(&ev("a", "k1", AuditSeverity::Info)).unwrap();
        sink.publish(&ev("a", "k2", AuditSeverity::Info)).unwrap();
        sink.publish(&ev("a", "k3", AuditSeverity::Info)).unwrap();
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].event_kind, "k2");
        assert_eq!(snap[1].event_kind, "k3");
    }

    // ---------- jsonl sink ----------

    #[test]
    fn jsonl_appends_lines() {
        let path = tmpfile("append");
        let sink = JsonLinesFileSink::open(path.clone()).unwrap();
        sink.publish(&ev("s", "k", AuditSeverity::Info)).unwrap();
        sink.publish(&ev("s", "k2", AuditSeverity::Warning))
            .unwrap();
        let body = fs::read_to_string(&path).unwrap();
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        let parsed: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["event_kind"], "k");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn jsonl_rotates_on_size() {
        let path = tmpfile("rotate");
        let sink = JsonLinesFileSink::open(path.clone())
            .unwrap()
            .with_max_bytes_per_file(200)
            .with_keep_rotated(2);
        for i in 0..30 {
            sink.publish(&ev("s", &format!("k{}", i), AuditSeverity::Info))
                .unwrap();
        }
        // The current log should exist; at least one rotated copy too.
        assert!(path.exists());
        let rot1 = path.with_extension("log.1");
        assert!(rot1.exists(), "expected rotated log .1 at {:?}", rot1);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&rot1);
    }

    #[test]
    fn jsonl_rejects_symlink() {
        let path = tmpfile("symlink_target");
        fs::write(&path, "").unwrap();
        let link = tmpfile("symlink_link");
        // Best-effort symlink creation — skip the assertion if not supported.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&path, &link).unwrap();
        }
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_file(&path, &link).is_err() {
                let _ = fs::remove_file(&path);
                return;
            }
        }
        let res = JsonLinesFileSink::open(link.clone());
        assert!(res.is_err());
        let _ = fs::remove_file(&link);
        let _ = fs::remove_file(&path);
    }

    // ---------- callback sink ----------

    #[test]
    fn callback_invokes_per_event() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let count = Arc::new(AtomicUsize::new(0));
        let c2 = count.clone();
        let sink = CallbackSink::new(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });
        sink.publish(&ev("s", "k", AuditSeverity::Info)).unwrap();
        sink.publish(&ev("s", "k", AuditSeverity::Info)).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    // ---------- bus ----------

    #[test]
    fn bus_fans_out_to_all_sinks() {
        let bus = UnifiedAuditBus::new(UnifiedAuditConfig::default());
        let s1 = Arc::new(InMemorySink::new(10));
        let s2 = Arc::new(InMemorySink::new(10));
        bus.add_sink(s1.clone());
        bus.add_sink(s2.clone());
        bus.publish(ev("x", "y", AuditSeverity::Info));
        assert_eq!(s1.len(), 1);
        assert_eq!(s2.len(), 1);
    }

    #[test]
    fn bus_filters_by_min_severity() {
        let cfg = UnifiedAuditConfig {
            min_severity: Some(AuditSeverity::Warning),
            ..Default::default()
        };
        let bus = UnifiedAuditBus::new(cfg);
        let s = Arc::new(InMemorySink::new(10));
        bus.add_sink(s.clone());
        bus.publish(ev("x", "info", AuditSeverity::Info));
        bus.publish(ev("x", "warn", AuditSeverity::Warning));
        bus.publish(ev("x", "err", AuditSeverity::Error));
        let snap = s.snapshot();
        assert_eq!(snap.len(), 2);
        assert!(snap.iter().all(|e| e.event_kind != "info"));
    }

    #[test]
    fn bus_filters_by_subsystem_allow_list() {
        let cfg = UnifiedAuditConfig {
            allowed_subsystems: vec!["plugins".into(), "agent".into()],
            ..Default::default()
        };
        let bus = UnifiedAuditBus::new(cfg);
        let s = Arc::new(InMemorySink::new(10));
        bus.add_sink(s.clone());
        bus.publish(ev("plugins", "k", AuditSeverity::Info));
        bus.publish(ev("rag", "k", AuditSeverity::Info));
        bus.publish(ev("agent", "k", AuditSeverity::Info));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn bus_runs_redactor_before_sinks() {
        let cfg = UnifiedAuditConfig {
            redactor: Some(Arc::new(|e| {
                e.payload = serde_json::json!({"redacted": true});
            })),
            ..Default::default()
        };
        let bus = UnifiedAuditBus::new(cfg);
        let s = Arc::new(InMemorySink::new(10));
        bus.add_sink(s.clone());
        let mut e = ev("x", "k", AuditSeverity::Info);
        e.payload = serde_json::json!({"secret": "hunter2"});
        bus.publish(e);
        let snap = s.snapshot();
        assert_eq!(snap[0].payload, serde_json::json!({"redacted": true}));
    }

    #[test]
    fn bus_continues_after_sink_error() {
        struct BrokenSink;
        impl AuditSink for BrokenSink {
            fn publish(&self, _: &UnifiedAuditEvent) -> Result<(), AuditSinkError> {
                Err(AuditSinkError::Other("boom".into()))
            }
            fn name(&self) -> &str {
                "broken"
            }
        }
        let bus = UnifiedAuditBus::new(UnifiedAuditConfig::default());
        bus.add_sink(Arc::new(BrokenSink));
        let good = Arc::new(InMemorySink::new(10));
        bus.add_sink(good.clone());
        bus.publish(ev("x", "k", AuditSeverity::Info));
        assert_eq!(good.len(), 1);
        assert_eq!(bus.sink_failure_count(), 1);
    }

    #[test]
    fn event_builders_chain() {
        let e = UnifiedAuditEvent::new("subsys", "k")
            .severity(AuditSeverity::Critical)
            .correlation("op-1")
            .actor("alice")
            .payload(serde_json::json!({"x": 1}));
        assert_eq!(e.severity, AuditSeverity::Critical);
        assert_eq!(e.correlation_id.as_deref(), Some("op-1"));
        assert_eq!(e.actor.as_deref(), Some("alice"));
        assert_eq!(e.payload["x"], 1);
    }
}
