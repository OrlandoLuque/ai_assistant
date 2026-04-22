//! Feedback sinks — adapters between `FeedbackDispatcher` and subsystems.
//!
//! The dispatcher doesn't know how to update a memory store or a bandit on
//! its own. Instead it fans out the `TrajectoryRecord` to every registered
//! `FeedbackSink` and lets each one do its thing. Sinks MUST be idempotent
//! on `TrajectoryId` — the dispatcher will retry on partial failure and may
//! deliver duplicates after a crash/restart.

use std::sync::Mutex;

use super::trajectory::{TrajectoryId, TrajectoryRecord};

/// Error returned by a sink. A failed dispatch is recorded in the ledger
/// and may be retried, but the overall dispatch does not abort.
#[derive(Debug)]
pub struct SinkError {
    pub sink: String,
    pub reason: String,
}

impl std::fmt::Display for SinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.sink, self.reason)
    }
}

impl std::error::Error for SinkError {}

/// Adapter to a downstream subsystem. Implementors must be idempotent.
pub trait FeedbackSink: Send + Sync {
    fn name(&self) -> &str;
    fn deliver(&self, record: &TrajectoryRecord) -> Result<(), SinkError>;
    /// Called when a retraction is issued for a trajectory. Sinks that store
    /// the record MUST tombstone it. Sinks that don't store anything can
    /// return `Ok(())` to signal "nothing to do".
    fn retract(&self, _id: &TrajectoryId) -> Result<(), SinkError> {
        Ok(())
    }
}

// =============================================================================
// CollectorSink — for tests and auditing
// =============================================================================

/// In-memory sink that captures everything. Useful for tests and for the
/// CLI auditor's `--dry-run` mode.
pub struct CollectorSink {
    name: String,
    inner: Mutex<CollectorInner>,
}

#[derive(Default)]
struct CollectorInner {
    delivered: Vec<TrajectoryRecord>,
    retracted: Vec<TrajectoryId>,
}

impl CollectorSink {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inner: Mutex::new(CollectorInner::default()),
        }
    }

    pub fn delivered(&self) -> Vec<TrajectoryRecord> {
        self.inner
            .lock()
            .map(|g| g.delivered.clone())
            .unwrap_or_default()
    }

    pub fn retracted(&self) -> Vec<TrajectoryId> {
        self.inner
            .lock()
            .map(|g| g.retracted.clone())
            .unwrap_or_default()
    }

    pub fn delivered_count(&self) -> usize {
        self.inner.lock().map(|g| g.delivered.len()).unwrap_or(0)
    }
}

impl FeedbackSink for CollectorSink {
    fn name(&self) -> &str {
        &self.name
    }
    fn deliver(&self, record: &TrajectoryRecord) -> Result<(), SinkError> {
        let mut g = self.inner.lock().map_err(|_| SinkError {
            sink: self.name.clone(),
            reason: "poisoned".into(),
        })?;
        if g.delivered.iter().any(|r| r.id == record.id) {
            // Idempotency — silent no-op on duplicate delivery.
            return Ok(());
        }
        g.delivered.push(record.clone());
        Ok(())
    }
    fn retract(&self, id: &TrajectoryId) -> Result<(), SinkError> {
        let mut g = self.inner.lock().map_err(|_| SinkError {
            sink: self.name.clone(),
            reason: "poisoned".into(),
        })?;
        g.delivered.retain(|r| r.id != *id);
        if !g.retracted.iter().any(|t| t == id) {
            g.retracted.push(id.clone());
        }
        Ok(())
    }
}

// =============================================================================
// FailingSink — for tests (simulate partial failure)
// =============================================================================

/// Sink that always fails. Lets us test dispatcher ledger + retry behavior
/// without mocking a subsystem.
pub struct FailingSink {
    name: String,
    reason: String,
}

impl FailingSink {
    pub fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }
}

impl FeedbackSink for FailingSink {
    fn name(&self) -> &str {
        &self.name
    }
    fn deliver(&self, _record: &TrajectoryRecord) -> Result<(), SinkError> {
        Err(SinkError {
            sink: self.name.clone(),
            reason: self.reason.clone(),
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TrajectoryRecord {
        TrajectoryRecord::new("alice")
    }

    #[test]
    fn collector_sink_stores_delivery() {
        let s = CollectorSink::new("mem");
        let r = sample();
        s.deliver(&r).unwrap();
        assert_eq!(s.delivered_count(), 1);
    }

    #[test]
    fn collector_sink_is_idempotent_on_duplicate_delivery() {
        let s = CollectorSink::new("mem");
        let r = sample();
        s.deliver(&r).unwrap();
        s.deliver(&r).unwrap();
        assert_eq!(s.delivered_count(), 1);
    }

    #[test]
    fn collector_sink_retract_removes_and_records_tombstone() {
        let s = CollectorSink::new("mem");
        let r = sample();
        let id = r.id.clone();
        s.deliver(&r).unwrap();
        s.retract(&id).unwrap();
        assert_eq!(s.delivered_count(), 0);
        assert_eq!(s.retracted().len(), 1);
    }

    #[test]
    fn failing_sink_always_errors() {
        let s = FailingSink::new("bad", "down");
        let e = s.deliver(&sample()).unwrap_err();
        assert_eq!(e.sink, "bad");
        assert_eq!(e.reason, "down");
    }
}
