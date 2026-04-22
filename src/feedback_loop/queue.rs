//! Bounded FIFO queue with drop-oldest overflow + a lightweight priority lane.
//!
//! The dispatcher uses this to decouple producers (Butler thread) from the
//! async worker that fans out to sinks. We keep two lanes: `normal` and
//! `priority`. On `push_normal`, the oldest item is dropped if the queue is
//! full (to keep producers fast and bound memory). `priority` is never dropped
//! silently — it errors on overflow so the caller can react.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::trajectory::TrajectoryRecord;

/// Drop-or-overflow policy for `push_normal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OverflowAction {
    /// A record was dropped to make room; caller may want to increment a metric.
    DroppedOldest,
    /// Queue had room; no drop.
    NoDrop,
}

#[derive(Debug)]
pub enum QueueError {
    /// Priority lane is full — we do not drop-oldest on priority.
    PriorityFull,
    /// Lock was poisoned — treat as fatal.
    Poisoned,
    /// Capacity zero is invalid.
    ZeroCapacity,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PriorityFull => f.write_str("priority queue full"),
            Self::Poisoned => f.write_str("queue lock poisoned"),
            Self::ZeroCapacity => f.write_str("queue capacity must be > 0"),
        }
    }
}

impl std::error::Error for QueueError {}

#[derive(Debug)]
struct Inner {
    normal: VecDeque<TrajectoryRecord>,
    priority: VecDeque<TrajectoryRecord>,
    dropped_oldest: u64,
}

/// Bounded queue with priority. Thread-safe.
#[derive(Debug)]
pub struct FeedbackQueue {
    capacity_normal: usize,
    capacity_priority: usize,
    inner: Mutex<Inner>,
}

impl FeedbackQueue {
    pub fn new(capacity_normal: usize, capacity_priority: usize) -> Result<Self, QueueError> {
        if capacity_normal == 0 || capacity_priority == 0 {
            return Err(QueueError::ZeroCapacity);
        }
        Ok(Self {
            capacity_normal,
            capacity_priority,
            inner: Mutex::new(Inner {
                normal: VecDeque::with_capacity(capacity_normal),
                priority: VecDeque::with_capacity(capacity_priority),
                dropped_oldest: 0,
            }),
        })
    }

    pub fn push_normal(&self, r: TrajectoryRecord) -> Result<OverflowAction, QueueError> {
        let mut g = self.inner.lock().map_err(|_| QueueError::Poisoned)?;
        let action = if g.normal.len() >= self.capacity_normal {
            g.normal.pop_front();
            g.dropped_oldest += 1;
            OverflowAction::DroppedOldest
        } else {
            OverflowAction::NoDrop
        };
        g.normal.push_back(r);
        Ok(action)
    }

    pub fn push_priority(&self, r: TrajectoryRecord) -> Result<(), QueueError> {
        let mut g = self.inner.lock().map_err(|_| QueueError::Poisoned)?;
        if g.priority.len() >= self.capacity_priority {
            return Err(QueueError::PriorityFull);
        }
        g.priority.push_back(r);
        Ok(())
    }

    /// Dequeue priority first, then normal.
    pub fn pop(&self) -> Option<TrajectoryRecord> {
        let mut g = self.inner.lock().ok()?;
        if let Some(r) = g.priority.pop_front() {
            return Some(r);
        }
        g.normal.pop_front()
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.normal.len() + g.priority.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dropped_oldest_count(&self) -> u64 {
        self.inner.lock().map(|g| g.dropped_oldest).unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(label: &str) -> TrajectoryRecord {
        let mut t = TrajectoryRecord::new(label);
        t.notes = label.to_string();
        t
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            FeedbackQueue::new(0, 1),
            Err(QueueError::ZeroCapacity)
        ));
        assert!(matches!(
            FeedbackQueue::new(1, 0),
            Err(QueueError::ZeroCapacity)
        ));
    }

    #[test]
    fn push_normal_under_capacity_does_not_drop() {
        let q = FeedbackQueue::new(4, 2).unwrap();
        let a = q.push_normal(sample("a")).unwrap();
        let b = q.push_normal(sample("b")).unwrap();
        assert_eq!(a, OverflowAction::NoDrop);
        assert_eq!(b, OverflowAction::NoDrop);
        assert_eq!(q.len(), 2);
        assert_eq!(q.dropped_oldest_count(), 0);
    }

    #[test]
    fn push_normal_overflow_drops_oldest() {
        let q = FeedbackQueue::new(2, 1).unwrap();
        q.push_normal(sample("a")).unwrap();
        q.push_normal(sample("b")).unwrap();
        let action = q.push_normal(sample("c")).unwrap();
        assert_eq!(action, OverflowAction::DroppedOldest);
        assert_eq!(q.len(), 2);
        assert_eq!(q.dropped_oldest_count(), 1);
        // Oldest ("a") should be gone: next pop is "b".
        let got = q.pop().unwrap();
        assert_eq!(got.notes, "b");
    }

    #[test]
    fn priority_preempts_normal_on_pop() {
        let q = FeedbackQueue::new(4, 4).unwrap();
        q.push_normal(sample("n1")).unwrap();
        q.push_priority(sample("p1")).unwrap();
        let a = q.pop().unwrap();
        assert_eq!(a.notes, "p1");
        let b = q.pop().unwrap();
        assert_eq!(b.notes, "n1");
        assert!(q.is_empty());
    }

    #[test]
    fn priority_full_returns_error_no_drop() {
        let q = FeedbackQueue::new(4, 1).unwrap();
        q.push_priority(sample("p1")).unwrap();
        let err = q.push_priority(sample("p2"));
        assert!(matches!(err, Err(QueueError::PriorityFull)));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn pop_from_empty_returns_none() {
        let q = FeedbackQueue::new(2, 2).unwrap();
        assert!(q.pop().is_none());
    }
}
