//! Progress reporting system for long-running operations
//!
//! This module provides a unified way to report progress for:
//! - Document indexing
//! - Model loading
//! - Batch operations
//! - Response generation
//! - Export/import operations
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::progress::{Progress, ProgressCallback, ProgressReporter};
//!
//! // Create a callback
//! let callback: ProgressCallback = Box::new(|progress| {
//!     println!("{}% - {}", progress.percentage(), progress.message);
//! });
//!
//! // Create a reporter
//! let reporter = ProgressReporter::new(Some(callback));
//!
//! // Report progress
//! reporter.report(Progress::new("Indexing", 0, 100).with_message("Starting..."));
//! reporter.report(Progress::new("Indexing", 50, 100).with_message("Halfway done"));
//! reporter.report(Progress::complete("Indexing").with_message("Done!"));
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Progress information for a long-running operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    /// Name/identifier of the operation
    pub operation: String,
    /// Current step/item being processed
    pub current: usize,
    /// Total steps/items to process (0 if unknown)
    pub total: usize,
    /// Human-readable message about current state
    pub message: String,
    /// Detailed sub-message (e.g., current file name)
    pub detail: Option<String>,
    /// Whether the operation is complete
    pub is_complete: bool,
    /// Whether the operation failed
    pub is_error: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Estimated remaining time in milliseconds (if calculable)
    pub remaining_ms: Option<u64>,
    /// Bytes processed (for I/O operations)
    pub bytes_processed: Option<u64>,
    /// Total bytes (for I/O operations)
    pub bytes_total: Option<u64>,
    /// Items per second (throughput)
    pub items_per_second: Option<f64>,
}

impl Progress {
    /// Create a new progress update
    pub fn new(operation: impl Into<String>, current: usize, total: usize) -> Self {
        Self {
            operation: operation.into(),
            current,
            total,
            message: String::new(),
            detail: None,
            is_complete: false,
            is_error: false,
            error_message: None,
            elapsed_ms: 0,
            remaining_ms: None,
            bytes_processed: None,
            bytes_total: None,
            items_per_second: None,
        }
    }

    /// Create a progress update for an indeterminate operation
    pub fn indeterminate(operation: impl Into<String>) -> Self {
        Self::new(operation, 0, 0)
    }

    /// Create a completed progress update
    pub fn complete(operation: impl Into<String>) -> Self {
        let mut p = Self::new(operation, 1, 1);
        p.is_complete = true;
        p
    }

    /// Create an error progress update
    pub fn error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        let mut p = Self::new(operation, 0, 0);
        p.is_error = true;
        p.error_message = Some(message.into());
        p
    }

    /// Add a message
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Add a detail message
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Set elapsed time
    pub fn with_elapsed(mut self, elapsed: Duration) -> Self {
        self.elapsed_ms = elapsed.as_millis() as u64;
        self
    }

    /// Set bytes information
    pub fn with_bytes(mut self, processed: u64, total: u64) -> Self {
        self.bytes_processed = Some(processed);
        self.bytes_total = Some(total);
        self
    }

    /// Calculate percentage complete (0-100)
    pub fn percentage(&self) -> u8 {
        if self.is_complete {
            return 100;
        }
        if self.total == 0 {
            return 0;
        }
        ((self.current as f64 / self.total as f64) * 100.0).min(100.0) as u8
    }

    /// Calculate remaining time based on elapsed time and progress
    pub fn estimate_remaining(&mut self) {
        if self.elapsed_ms > 0 && self.current > 0 && self.total > self.current {
            let ms_per_item = self.elapsed_ms as f64 / self.current as f64;
            let remaining_items = self.total - self.current;
            self.remaining_ms = Some((ms_per_item * remaining_items as f64) as u64);
            self.items_per_second = Some(1000.0 / ms_per_item);
        }
    }

    /// Format remaining time as human-readable string
    pub fn remaining_human(&self) -> Option<String> {
        self.remaining_ms.map(|ms| {
            let secs = ms / 1000;
            if secs < 60 {
                format!("{}s", secs)
            } else if secs < 3600 {
                format!("{}m {}s", secs / 60, secs % 60)
            } else {
                format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
            }
        })
    }

    /// Format elapsed time as human-readable string
    pub fn elapsed_human(&self) -> String {
        let secs = self.elapsed_ms / 1000;
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }

    /// Format bytes as human-readable string
    pub fn bytes_human(&self) -> Option<String> {
        self.bytes_processed.map(|bytes| {
            if bytes < 1024 {
                format!("{} B", bytes)
            } else if bytes < 1024 * 1024 {
                format!("{:.1} KB", bytes as f64 / 1024.0)
            } else if bytes < 1024 * 1024 * 1024 {
                format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
            } else {
                format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            }
        })
    }
}

/// Callback type for progress updates
pub type ProgressCallback = Box<dyn Fn(Progress) + Send + Sync>;

/// Progress reporter that tracks an operation and reports updates
pub struct ProgressReporter {
    operation: String,
    total: usize,
    current: usize,
    start_time: Instant,
    callback: Option<Arc<ProgressCallback>>,
    last_report: Instant,
    min_report_interval: Duration,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(callback: Option<ProgressCallback>) -> Self {
        Self {
            operation: String::new(),
            total: 0,
            current: 0,
            start_time: Instant::now(),
            callback: callback.map(Arc::new),
            last_report: Instant::now(),
            min_report_interval: Duration::from_millis(100),
        }
    }

    /// Start a new operation
    pub fn start(&mut self, operation: impl Into<String>, total: usize) {
        self.operation = operation.into();
        self.total = total;
        self.current = 0;
        self.start_time = Instant::now();
        self.last_report = Instant::now() - self.min_report_interval; // Allow immediate first report

        self.report(Progress::new(&self.operation, 0, total)
            .with_message("Starting..."));
    }

    /// Update progress
    pub fn update(&mut self, current: usize, message: impl Into<String>) {
        self.current = current;

        // Rate limit reports
        if self.last_report.elapsed() < self.min_report_interval {
            return;
        }
        self.last_report = Instant::now();

        let mut progress = Progress::new(&self.operation, current, self.total)
            .with_message(message)
            .with_elapsed(self.start_time.elapsed());
        progress.estimate_remaining();

        self.report(progress);
    }

    /// Update with detail
    pub fn update_detail(&mut self, current: usize, message: impl Into<String>, detail: impl Into<String>) {
        self.current = current;

        if self.last_report.elapsed() < self.min_report_interval {
            return;
        }
        self.last_report = Instant::now();

        let mut progress = Progress::new(&self.operation, current, self.total)
            .with_message(message)
            .with_detail(detail)
            .with_elapsed(self.start_time.elapsed());
        progress.estimate_remaining();

        self.report(progress);
    }

    /// Increment progress by one
    pub fn increment(&mut self, message: impl Into<String>) {
        self.update(self.current + 1, message);
    }

    /// Mark operation as complete
    pub fn complete(&mut self, message: impl Into<String>) {
        let progress = Progress::complete(&self.operation)
            .with_message(message)
            .with_elapsed(self.start_time.elapsed());
        self.report(progress);
    }

    /// Mark operation as failed
    pub fn error(&mut self, error: impl Into<String>) {
        let progress = Progress::error(&self.operation, error)
            .with_elapsed(self.start_time.elapsed());
        self.report(progress);
    }

    /// Report a progress update
    pub fn report(&self, progress: Progress) {
        if let Some(ref callback) = self.callback {
            callback(progress);
        }
    }

    /// Set minimum interval between reports
    pub fn set_min_interval(&mut self, interval: Duration) {
        self.min_report_interval = interval;
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Multi-operation progress tracker
pub struct MultiProgressTracker {
    operations: Arc<Mutex<Vec<Progress>>>,
    callback: Option<Arc<ProgressCallback>>,
}

impl MultiProgressTracker {
    /// Create a new multi-operation tracker
    pub fn new(callback: Option<ProgressCallback>) -> Self {
        Self {
            operations: Arc::new(Mutex::new(Vec::new())),
            callback: callback.map(Arc::new),
        }
    }

    /// Create a reporter for a specific operation
    pub fn create_reporter(&self, operation: impl Into<String>, total: usize) -> OperationHandle {
        let operation = operation.into();
        let progress = Progress::new(&operation, 0, total);

        let mut ops = self.operations.lock().unwrap_or_else(|e| e.into_inner());
        let index = ops.len();
        ops.push(progress);

        OperationHandle {
            tracker: Arc::clone(&self.operations),
            callback: self.callback.clone(),
            index,
            operation,
            total,
            start_time: Instant::now(),
        }
    }

    /// Get all current operation progresses
    pub fn get_all(&self) -> Vec<Progress> {
        self.operations.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get overall progress across all operations
    pub fn overall_progress(&self) -> Progress {
        let ops = self.operations.lock().unwrap_or_else(|e| e.into_inner());
        if ops.is_empty() {
            return Progress::complete("All operations");
        }

        let total_current: usize = ops.iter().map(|p| p.current).sum();
        let total_total: usize = ops.iter().map(|p| p.total).sum();
        let all_complete = ops.iter().all(|p| p.is_complete);
        let any_error = ops.iter().any(|p| p.is_error);

        let mut progress = Progress::new("All operations", total_current, total_total);
        progress.is_complete = all_complete;
        progress.is_error = any_error;
        progress.message = format!("{} operations", ops.len());
        progress
    }
}

/// Handle for a specific operation in a multi-tracker
pub struct OperationHandle {
    tracker: Arc<Mutex<Vec<Progress>>>,
    callback: Option<Arc<ProgressCallback>>,
    index: usize,
    operation: String,
    total: usize,
    start_time: Instant,
}

impl OperationHandle {
    /// Update progress
    pub fn update(&self, current: usize, message: impl Into<String>) {
        let mut progress = Progress::new(&self.operation, current, self.total)
            .with_message(message)
            .with_elapsed(self.start_time.elapsed());
        progress.estimate_remaining();

        let mut ops = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        if self.index < ops.len() {
            ops[self.index] = progress.clone();
        }

        if let Some(ref callback) = self.callback {
            callback(progress);
        }
    }

    /// Mark complete
    pub fn complete(&self, message: impl Into<String>) {
        let progress = Progress::complete(&self.operation)
            .with_message(message)
            .with_elapsed(self.start_time.elapsed());

        let mut ops = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        if self.index < ops.len() {
            ops[self.index] = progress.clone();
        }

        if let Some(ref callback) = self.callback {
            callback(progress);
        }
    }

    /// Mark as error
    pub fn error(&self, error: impl Into<String>) {
        let progress = Progress::error(&self.operation, error)
            .with_elapsed(self.start_time.elapsed());

        let mut ops = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        if self.index < ops.len() {
            ops[self.index] = progress.clone();
        }

        if let Some(ref callback) = self.callback {
            callback(progress);
        }
    }
}

/// Progress aggregator for parallel operations
#[derive(Clone)]
pub struct ProgressAggregator {
    inner: Arc<Mutex<AggregatorInner>>,
    callback: Option<Arc<ProgressCallback>>,
}

struct AggregatorInner {
    operation: String,
    total_items: usize,
    completed_items: usize,
    failed_items: usize,
    start_time: Instant,
}

impl ProgressAggregator {
    /// Create a new aggregator
    pub fn new(operation: impl Into<String>, total_items: usize, callback: Option<ProgressCallback>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AggregatorInner {
                operation: operation.into(),
                total_items,
                completed_items: 0,
                failed_items: 0,
                start_time: Instant::now(),
            })),
            callback: callback.map(Arc::new),
        }
    }

    /// Record a successful completion
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.completed_items += 1;
        self.report_progress(&inner);
    }

    /// Record a failure
    pub fn record_failure(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.failed_items += 1;
        self.report_progress(&inner);
    }

    /// Get current progress
    pub fn get_progress(&self) -> Progress {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.build_progress(&inner)
    }

    fn report_progress(&self, inner: &AggregatorInner) {
        if let Some(ref callback) = self.callback {
            callback(self.build_progress(inner));
        }
    }

    fn build_progress(&self, inner: &AggregatorInner) -> Progress {
        let current = inner.completed_items + inner.failed_items;
        let mut progress = Progress::new(&inner.operation, current, inner.total_items)
            .with_elapsed(inner.start_time.elapsed());

        progress.is_complete = current >= inner.total_items;
        progress.is_error = inner.failed_items > 0;
        progress.message = format!(
            "{}/{} completed ({} failed)",
            inner.completed_items,
            inner.total_items,
            inner.failed_items
        );
        progress.estimate_remaining();
        progress
    }
}

/// Builder for creating progress callbacks with common patterns
pub struct ProgressCallbackBuilder {
    on_start: Option<Box<dyn Fn(&str) + Send + Sync>>,
    on_progress: Option<Box<dyn Fn(u8, &str) + Send + Sync>>,
    on_complete: Option<Box<dyn Fn(&str, Duration) + Send + Sync>>,
    on_error: Option<Box<dyn Fn(&str, &str) + Send + Sync>>,
}

impl Default for ProgressCallbackBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressCallbackBuilder {
    pub fn new() -> Self {
        Self {
            on_start: None,
            on_progress: None,
            on_complete: None,
            on_error: None,
        }
    }

    /// Set callback for operation start
    pub fn on_start<F: Fn(&str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_start = Some(Box::new(f));
        self
    }

    /// Set callback for progress updates (percentage, message)
    pub fn on_progress<F: Fn(u8, &str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_progress = Some(Box::new(f));
        self
    }

    /// Set callback for completion (operation, duration)
    pub fn on_complete<F: Fn(&str, Duration) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_complete = Some(Box::new(f));
        self
    }

    /// Set callback for errors (operation, error message)
    pub fn on_error<F: Fn(&str, &str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_error = Some(Box::new(f));
        self
    }

    /// Build the callback
    pub fn build(self) -> ProgressCallback {
        Box::new(move |progress: Progress| {
            if progress.is_error {
                if let (Some(ref f), Some(ref err)) = (&self.on_error, &progress.error_message) {
                    f(&progress.operation, err);
                }
            } else if progress.is_complete {
                if let Some(ref f) = self.on_complete {
                    f(&progress.operation, Duration::from_millis(progress.elapsed_ms));
                }
            } else if progress.current == 0 && progress.total > 0 {
                if let Some(ref f) = self.on_start {
                    f(&progress.operation);
                }
            } else if let Some(ref f) = self.on_progress {
                f(progress.percentage(), &progress.message);
            }
        })
    }
}

/// Create a simple logging progress callback
pub fn logging_callback(prefix: &'static str) -> ProgressCallback {
    Box::new(move |progress: Progress| {
        if progress.is_error {
            log::error!("[{}] ERROR {}: {:?}", prefix, progress.operation, progress.error_message);
        } else if progress.is_complete {
            println!("[{}] {} completed in {}", prefix, progress.operation, progress.elapsed_human());
        } else {
            let remaining = progress.remaining_human().map(|r| format!(" (~{} remaining)", r)).unwrap_or_default();
            println!(
                "[{}] {} {}% - {}{}",
                prefix,
                progress.operation,
                progress.percentage(),
                progress.message,
                remaining
            );
        }
    })
}

/// Create a silent callback that does nothing (for testing/benchmarking)
pub fn silent_callback() -> ProgressCallback {
    Box::new(|_| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_progress_percentage() {
        let p = Progress::new("test", 50, 100);
        assert_eq!(p.percentage(), 50);

        let p = Progress::new("test", 0, 0);
        assert_eq!(p.percentage(), 0);

        let p = Progress::complete("test");
        assert_eq!(p.percentage(), 100);
    }

    #[test]
    fn test_progress_reporter() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let callback: ProgressCallback = Box::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let mut reporter = ProgressReporter::new(Some(callback));
        reporter.set_min_interval(Duration::from_millis(0)); // No rate limiting for test
        reporter.start("test", 3);
        reporter.update(1, "step 1");
        reporter.update(2, "step 2");
        reporter.complete("done");

        assert!(counter.load(Ordering::SeqCst) >= 3);
    }

    #[test]
    fn test_aggregator() {
        let agg = ProgressAggregator::new("batch", 3, None);

        agg.record_success();
        agg.record_success();
        agg.record_failure();

        let progress = agg.get_progress();
        assert!(progress.is_complete);
        assert!(progress.is_error);
        assert!(progress.message.contains("1 failed"));
    }

    #[test]
    fn test_remaining_time_format() {
        let mut p = Progress::new("test", 50, 100);
        p.elapsed_ms = 5000; // 5 seconds for 50 items
        p.estimate_remaining();

        assert!(p.remaining_ms.is_some());
        assert!(p.remaining_human().is_some());
    }
}
