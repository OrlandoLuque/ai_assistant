//! Verbose debug mode for detailed logging and inspection
//!
//! This module provides comprehensive debugging utilities for
//! inspecting AI assistant operations in detail.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fmt::Write as FmtWrite;

/// Debug verbosity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DebugLevel {
    /// No debug output
    Off = 0,
    /// Error messages only
    Error = 1,
    /// Errors and warnings
    Warn = 2,
    /// General info messages
    Info = 3,
    /// Detailed debug information
    Debug = 4,
    /// Very verbose trace output
    Trace = 5,
}

impl Default for DebugLevel {
    fn default() -> Self {
        DebugLevel::Off
    }
}

impl DebugLevel {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "off" | "none" | "0" => DebugLevel::Off,
            "error" | "1" => DebugLevel::Error,
            "warn" | "warning" | "2" => DebugLevel::Warn,
            "info" | "3" => DebugLevel::Info,
            "debug" | "4" => DebugLevel::Debug,
            "trace" | "verbose" | "5" => DebugLevel::Trace,
            _ => DebugLevel::Off,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            DebugLevel::Off => "OFF",
            DebugLevel::Error => "ERROR",
            DebugLevel::Warn => "WARN",
            DebugLevel::Info => "INFO",
            DebugLevel::Debug => "DEBUG",
            DebugLevel::Trace => "TRACE",
        }
    }
}

/// A debug log entry
#[derive(Debug, Clone)]
pub struct DebugEntry {
    /// Timestamp
    pub timestamp: u64,
    /// Level
    pub level: DebugLevel,
    /// Component that logged this
    pub component: String,
    /// Message
    pub message: String,
    /// Additional context
    pub context: Option<String>,
    /// Duration if timing operation
    pub duration: Option<Duration>,
}

impl DebugEntry {
    /// Create a new entry
    pub fn new(level: DebugLevel, component: &str, message: impl Into<String>) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            level,
            component: component.to_string(),
            message: message.into(),
            context: None,
            duration: None,
        }
    }

    /// Add context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Add duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        let mut s = format!(
            "[{:>5}] {} - {}",
            self.level.as_str(),
            self.component,
            self.message
        );

        if let Some(ref ctx) = self.context {
            write!(s, " | {}", ctx).ok();
        }

        if let Some(dur) = self.duration {
            write!(s, " ({:.2}ms)", dur.as_secs_f64() * 1000.0).ok();
        }

        s
    }
}

/// Debug configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Global debug level
    pub level: DebugLevel,
    /// Maximum entries to keep in memory
    pub max_entries: usize,
    /// Log to stderr
    pub log_to_stderr: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include component names
    pub include_components: bool,
    /// Components to filter (empty = all)
    pub component_filter: Vec<String>,
    /// Enable request/response logging
    pub log_requests: bool,
    /// Enable token counting in logs
    pub log_tokens: bool,
    /// Enable timing for all operations
    pub enable_timing: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            level: DebugLevel::Off,
            max_entries: 1000,
            log_to_stderr: false,
            include_timestamps: true,
            include_components: true,
            component_filter: Vec::new(),
            log_requests: false,
            log_tokens: false,
            enable_timing: false,
        }
    }
}

/// Debug logger
pub struct DebugLogger {
    config: RwLock<DebugConfig>,
    entries: Mutex<VecDeque<DebugEntry>>,
    start_time: Instant,
}

impl Default for DebugLogger {
    fn default() -> Self {
        Self::new(DebugConfig::default())
    }
}

impl DebugLogger {
    /// Create a new debug logger
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config: RwLock::new(config),
            entries: Mutex::new(VecDeque::new()),
            start_time: Instant::now(),
        }
    }

    /// Create with debug level
    pub fn with_level(level: DebugLevel) -> Self {
        Self::new(DebugConfig {
            level,
            ..Default::default()
        })
    }

    /// Update configuration
    pub fn configure(&self, config: DebugConfig) {
        *self.config.write().unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// Set debug level
    pub fn set_level(&self, level: DebugLevel) {
        self.config.write().unwrap_or_else(|e| e.into_inner()).level = level;
    }

    /// Get current level
    pub fn level(&self) -> DebugLevel {
        self.config.read().unwrap_or_else(|e| e.into_inner()).level
    }

    /// Check if level is enabled
    pub fn is_enabled(&self, level: DebugLevel) -> bool {
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());
        config.level >= level
    }

    /// Log an entry
    pub fn log(&self, entry: DebugEntry) {
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());

        // Check level
        if config.level < entry.level {
            return;
        }

        // Check component filter
        if !config.component_filter.is_empty()
            && !config.component_filter.iter().any(|c| entry.component.contains(c)) {
            return;
        }

        // Log to stderr if enabled
        if config.log_to_stderr {
            let formatted = if config.include_timestamps {
                format!("[{:.3}s] {}", self.start_time.elapsed().as_secs_f64(), entry.format())
            } else {
                entry.format()
            };
            log::debug!("{}", formatted);
        }

        drop(config);

        // Store in memory
        let config = self.config.read().unwrap_or_else(|e| e.into_inner());
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.push_back(entry);

        // Trim if needed
        while entries.len() > config.max_entries {
            entries.pop_front();
        }
    }

    /// Log at specific levels
    pub fn error(&self, component: &str, message: impl Into<String>) {
        self.log(DebugEntry::new(DebugLevel::Error, component, message));
    }

    pub fn warn(&self, component: &str, message: impl Into<String>) {
        self.log(DebugEntry::new(DebugLevel::Warn, component, message));
    }

    pub fn info(&self, component: &str, message: impl Into<String>) {
        self.log(DebugEntry::new(DebugLevel::Info, component, message));
    }

    pub fn debug(&self, component: &str, message: impl Into<String>) {
        self.log(DebugEntry::new(DebugLevel::Debug, component, message));
    }

    pub fn trace(&self, component: &str, message: impl Into<String>) {
        self.log(DebugEntry::new(DebugLevel::Trace, component, message));
    }

    /// Start a timed operation
    pub fn start_timer(&self, component: &str, operation: &str) -> TimerGuard<'_> {
        if self.config.read().unwrap_or_else(|e| e.into_inner()).enable_timing {
            self.trace(component, format!("Starting: {}", operation));
        }
        TimerGuard {
            logger: self,
            component: component.to_string(),
            operation: operation.to_string(),
            start: Instant::now(),
        }
    }

    /// Get recent entries
    pub fn recent_entries(&self, count: usize) -> Vec<DebugEntry> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.iter().rev().take(count).cloned().collect()
    }

    /// Get all entries
    pub fn all_entries(&self) -> Vec<DebugEntry> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect()
    }

    /// Get entries by level
    pub fn entries_by_level(&self, level: DebugLevel) -> Vec<DebugEntry> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }

    /// Get entries by component
    pub fn entries_by_component(&self, component: &str) -> Vec<DebugEntry> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|e| e.component.contains(component))
            .cloned()
            .collect()
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Get entry count
    pub fn entry_count(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Generate debug report
    pub fn generate_report(&self) -> DebugReport {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());

        let mut errors = 0;
        let mut warnings = 0;
        let mut by_component: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut total_duration = Duration::ZERO;
        let mut timed_ops = 0;

        for entry in entries.iter() {
            match entry.level {
                DebugLevel::Error => errors += 1,
                DebugLevel::Warn => warnings += 1,
                _ => {}
            }

            *by_component.entry(entry.component.clone()).or_insert(0) += 1;

            if let Some(dur) = entry.duration {
                total_duration += dur;
                timed_ops += 1;
            }
        }

        DebugReport {
            total_entries: entries.len(),
            error_count: errors,
            warning_count: warnings,
            entries_by_component: by_component,
            uptime: self.start_time.elapsed(),
            total_timed_duration: total_duration,
            timed_operations: timed_ops,
        }
    }

    /// Format as text report
    pub fn format_report(&self) -> String {
        let report = self.generate_report();
        let mut s = String::new();

        writeln!(s, "=== Debug Report ===").ok();
        writeln!(s, "Uptime: {:.2}s", report.uptime.as_secs_f64()).ok();
        writeln!(s, "Total entries: {}", report.total_entries).ok();
        writeln!(s, "Errors: {}", report.error_count).ok();
        writeln!(s, "Warnings: {}", report.warning_count).ok();

        if report.timed_operations > 0 {
            writeln!(s, "\nTimed Operations:").ok();
            writeln!(s, "  Count: {}", report.timed_operations).ok();
            writeln!(s, "  Total: {:.2}ms", report.total_timed_duration.as_secs_f64() * 1000.0).ok();
            writeln!(s, "  Avg: {:.2}ms",
                report.total_timed_duration.as_secs_f64() * 1000.0 / report.timed_operations as f64).ok();
        }

        if !report.entries_by_component.is_empty() {
            writeln!(s, "\nBy Component:").ok();
            let mut sorted: Vec<_> = report.entries_by_component.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            for (comp, count) in sorted.iter().take(10) {
                writeln!(s, "  {}: {}", comp, count).ok();
            }
        }

        s
    }
}

/// Guard for timed operations
pub struct TimerGuard<'a> {
    logger: &'a DebugLogger,
    component: String,
    operation: String,
    start: Instant,
}

impl<'a> Drop for TimerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let entry = DebugEntry::new(DebugLevel::Debug, &self.component, format!("Completed: {}", self.operation))
            .with_duration(duration);
        self.logger.log(entry);
    }
}

/// Debug report summary
#[derive(Debug, Clone)]
pub struct DebugReport {
    pub total_entries: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub entries_by_component: std::collections::HashMap<String, usize>,
    pub uptime: Duration,
    pub total_timed_duration: Duration,
    pub timed_operations: usize,
}

/// Request/response inspector
pub struct RequestInspector {
    logger: Arc<DebugLogger>,
    #[allow(dead_code)]
    capture_requests: bool,
    #[allow(dead_code)]
    capture_responses: bool,
    max_body_length: usize,
    requests: Mutex<Vec<CapturedRequest>>,
}

#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub timestamp: u64,
    pub provider: String,
    pub model: String,
    pub prompt_preview: String,
    pub prompt_tokens: Option<usize>,
    pub response_preview: Option<String>,
    pub response_tokens: Option<usize>,
    pub duration: Option<Duration>,
    pub error: Option<String>,
}

impl RequestInspector {
    /// Create a new request inspector
    pub fn new(logger: Arc<DebugLogger>) -> Self {
        Self {
            logger,
            capture_requests: true,
            capture_responses: true,
            max_body_length: 500,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Capture a request
    pub fn capture_request(&self, provider: &str, model: &str, prompt: &str) -> RequestHandle<'_> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let prompt_preview = if prompt.len() > self.max_body_length {
            format!("{}...", &prompt[..self.max_body_length])
        } else {
            prompt.to_string()
        };

        self.logger.debug("RequestInspector",
            format!("Request to {} ({}): {}", provider, model,
                if prompt.len() > 100 { &prompt[..100] } else { prompt }));

        RequestHandle {
            inspector: self,
            request: CapturedRequest {
                timestamp,
                provider: provider.to_string(),
                model: model.to_string(),
                prompt_preview,
                prompt_tokens: Some(prompt.split_whitespace().count()), // Rough estimate
                response_preview: None,
                response_tokens: None,
                duration: None,
                error: None,
            },
            start: Instant::now(),
        }
    }

    /// Get captured requests
    pub fn get_requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Clear captured requests
    pub fn clear_requests(&self) {
        self.requests.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    fn store_request(&self, request: CapturedRequest) {
        let mut requests = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        requests.push(request);

        // Keep last 100 requests
        if requests.len() > 100 {
            requests.remove(0);
        }
    }
}

/// Handle for tracking a request
pub struct RequestHandle<'a> {
    inspector: &'a RequestInspector,
    request: CapturedRequest,
    start: Instant,
}

impl<'a> RequestHandle<'a> {
    /// Complete with success
    pub fn complete(mut self, response: &str) {
        self.request.duration = Some(self.start.elapsed());
        self.request.response_preview = Some(
            if response.len() > self.inspector.max_body_length {
                format!("{}...", &response[..self.inspector.max_body_length])
            } else {
                response.to_string()
            }
        );
        self.request.response_tokens = Some(response.split_whitespace().count());

        self.inspector.logger.debug("RequestInspector",
            format!("Response from {} ({:.2}ms): {} tokens",
                self.request.provider,
                self.request.duration.unwrap_or_default().as_secs_f64() * 1000.0,
                self.request.response_tokens.unwrap_or(0)));

        self.inspector.store_request(self.request);
    }

    /// Complete with error
    pub fn error(mut self, error: &str) {
        self.request.duration = Some(self.start.elapsed());
        self.request.error = Some(error.to_string());

        self.inspector.logger.error("RequestInspector",
            format!("Request to {} failed: {}", self.request.provider, error));

        self.inspector.store_request(self.request);
    }
}

/// Global debug instance
static GLOBAL_DEBUG: std::sync::OnceLock<Arc<DebugLogger>> = std::sync::OnceLock::new();

/// Get or initialize the global debug logger
pub fn global_debug() -> Arc<DebugLogger> {
    GLOBAL_DEBUG.get_or_init(|| Arc::new(DebugLogger::default())).clone()
}

/// Configure the global debug logger
pub fn configure_global_debug(config: DebugConfig) {
    global_debug().configure(config);
}

/// Set global debug level
pub fn set_debug_level(level: DebugLevel) {
    global_debug().set_level(level);
}

/// Convenience macros for debug logging
#[macro_export]
macro_rules! debug_error {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug::global_debug().error($component, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! debug_warn {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug::global_debug().warn($component, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! debug_info {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug::global_debug().info($component, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! debug_debug {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug::global_debug().debug($component, format!($($arg)*))
    };
}

#[macro_export]
macro_rules! debug_trace {
    ($component:expr, $($arg:tt)*) => {
        $crate::debug::global_debug().trace($component, format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_levels() {
        assert!(DebugLevel::Trace > DebugLevel::Debug);
        assert!(DebugLevel::Debug > DebugLevel::Info);
        assert!(DebugLevel::Info > DebugLevel::Warn);
        assert!(DebugLevel::Warn > DebugLevel::Error);
        assert!(DebugLevel::Error > DebugLevel::Off);
    }

    #[test]
    fn test_debug_level_parsing() {
        assert_eq!(DebugLevel::from_str("debug"), DebugLevel::Debug);
        assert_eq!(DebugLevel::from_str("TRACE"), DebugLevel::Trace);
        assert_eq!(DebugLevel::from_str("3"), DebugLevel::Info);
    }

    #[test]
    fn test_debug_logger() {
        let logger = DebugLogger::with_level(DebugLevel::Debug);

        logger.info("test", "Test message");
        logger.debug("test", "Debug message");
        logger.trace("test", "Trace message"); // Should not be logged

        assert_eq!(logger.entry_count(), 2);
    }

    #[test]
    fn test_debug_entry() {
        let entry = DebugEntry::new(DebugLevel::Info, "component", "message")
            .with_context("extra info")
            .with_duration(Duration::from_millis(100));

        let formatted = entry.format();
        assert!(formatted.contains("INFO"));
        assert!(formatted.contains("component"));
        assert!(formatted.contains("message"));
        assert!(formatted.contains("100"));
    }

    #[test]
    fn test_timer_guard() {
        let logger = DebugLogger::new(DebugConfig {
            level: DebugLevel::Debug,
            enable_timing: true,
            ..Default::default()
        });

        {
            let _timer = logger.start_timer("test", "operation");
            std::thread::sleep(Duration::from_millis(10));
        }

        let entries = logger.all_entries();
        assert!(!entries.is_empty());

        let timed = entries.iter().find(|e| e.duration.is_some());
        assert!(timed.is_some());
    }

    #[test]
    fn test_debug_report() {
        let logger = DebugLogger::with_level(DebugLevel::Trace);

        logger.error("comp1", "Error 1");
        logger.warn("comp1", "Warning 1");
        logger.info("comp2", "Info 1");
        logger.debug("comp2", "Debug 1");

        let report = logger.generate_report();
        assert_eq!(report.total_entries, 4);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.warning_count, 1);
        assert_eq!(report.entries_by_component.len(), 2);
    }

    #[test]
    fn test_component_filter() {
        let logger = DebugLogger::new(DebugConfig {
            level: DebugLevel::Debug,
            component_filter: vec!["allowed".to_string()],
            ..Default::default()
        });

        logger.info("allowed", "Should log");
        logger.info("blocked", "Should not log");

        assert_eq!(logger.entry_count(), 1);
    }

    #[test]
    fn test_entries_by_level() {
        let logger = DebugLogger::with_level(DebugLevel::Trace);

        logger.error("test", "Error");
        logger.warn("test", "Warning");
        logger.info("test", "Info");

        let errors = logger.entries_by_level(DebugLevel::Error);
        assert_eq!(errors.len(), 1);

        let warnings = logger.entries_by_level(DebugLevel::Warn);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_max_entries() {
        let logger = DebugLogger::new(DebugConfig {
            level: DebugLevel::Debug,
            max_entries: 5,
            ..Default::default()
        });

        for i in 0..10 {
            logger.info("test", format!("Message {}", i));
        }

        assert_eq!(logger.entry_count(), 5);
    }
}
