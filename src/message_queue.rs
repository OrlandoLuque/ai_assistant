//! Message queue integration
//!
//! Queue-based processing for AI requests.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Queue message
#[derive(Debug, Clone)]
pub struct QueueMessage {
    pub id: String,
    pub payload: String,
    pub priority: u8,
    pub created_at: Instant,
    pub metadata: std::collections::HashMap<String, String>,
}

impl QueueMessage {
    pub fn new(payload: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            payload: payload.to_string(),
            priority: 5,
            created_at: Instant::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Simple in-memory message queue
pub struct MemoryQueue {
    messages: Arc<Mutex<VecDeque<QueueMessage>>>,
    signal: Arc<Condvar>,
    max_size: usize,
}

impl MemoryQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: Arc::new(Mutex::new(VecDeque::new())),
            signal: Arc::new(Condvar::new()),
            max_size,
        }
    }

    pub fn push(&self, message: QueueMessage) -> Result<(), QueueError> {
        let mut queue = self.messages.lock().unwrap_or_else(|e| e.into_inner());
        if queue.len() >= self.max_size {
            return Err(QueueError::Full);
        }
        queue.push_back(message);
        self.signal.notify_one();
        Ok(())
    }

    pub fn pop(&self) -> Option<QueueMessage> {
        let mut queue = self.messages.lock().unwrap_or_else(|e| e.into_inner());
        queue.pop_front()
    }

    pub fn pop_blocking(&self, timeout: Duration) -> Option<QueueMessage> {
        let mut queue = self.messages.lock().unwrap_or_else(|e| e.into_inner());

        let deadline = Instant::now() + timeout;
        while queue.is_empty() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let result = self
                .signal
                .wait_timeout(queue, remaining)
                .unwrap_or_else(|e| e.into_inner());
            queue = result.0;
            if result.1.timed_out() {
                return None;
            }
        }

        queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

impl Default for MemoryQueue {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Queue errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    Full,
    Empty,
    Timeout,
    ConnectionFailed,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "Queue is full"),
            Self::Empty => write!(f, "Queue is empty"),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::ConnectionFailed => write!(f, "Connection failed"),
        }
    }
}

impl std::error::Error for QueueError {}

/// Statistics from batch processing operations
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    pub processed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub duration_ms: u64,
}

/// Queue processor for handling messages
pub struct QueueProcessor<F>
where
    F: Fn(QueueMessage) -> Result<String, String> + Send + Sync + 'static,
{
    queue: Arc<MemoryQueue>,
    handler: Arc<F>,
    running: Arc<Mutex<bool>>,
}

impl<F> QueueProcessor<F>
where
    F: Fn(QueueMessage) -> Result<String, String> + Send + Sync + 'static,
{
    pub fn new(queue: Arc<MemoryQueue>, handler: F) -> Self {
        Self {
            queue,
            handler: Arc::new(handler),
            running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start(&self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }

    pub fn stop(&self) {
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }

    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn process_one(&self) -> Option<Result<String, String>> {
        let message = self.queue.pop()?;
        Some((self.handler)(message))
    }

    /// Drain the queue, processing each message. Returns all results.
    pub fn process_all(&self) -> Vec<Result<String, String>> {
        let mut results = Vec::new();
        while let Some(result) = self.process_one() {
            results.push(result);
        }
        results
    }

    /// Process up to `max` messages from the queue. Returns results for processed messages.
    pub fn process_batch(&self, max: usize) -> Vec<Result<String, String>> {
        let mut results = Vec::new();
        for _ in 0..max {
            match self.process_one() {
                Some(result) => results.push(result),
                None => break,
            }
        }
        results
    }

    /// Loop processing messages until the queue is empty, tracking statistics.
    /// Returns empty stats if the processor is not running.
    /// Stops early if `stop()` is called mid-processing.
    pub fn process_until_empty(&self) -> ProcessingStats {
        if !self.is_running() {
            return ProcessingStats::default();
        }

        let start = Instant::now();
        let mut stats = ProcessingStats::default();

        while self.is_running() {
            match self.process_one() {
                Some(Ok(_)) => {
                    stats.processed += 1;
                    stats.succeeded += 1;
                }
                Some(Err(_)) => {
                    stats.processed += 1;
                    stats.failed += 1;
                }
                None => break,
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        stats
    }
}

// ============================================================================
// Dead Letter Queue — Enhanced
// ============================================================================

/// Category of failure that caused a message to be dead-lettered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureCategory {
    /// Request timed out.
    Timeout,
    /// Rate limit was exceeded.
    RateLimited,
    /// Provider was unavailable.
    ProviderUnavailable,
    /// Request was invalid (non-retryable).
    InvalidRequest,
    /// Unknown or uncategorized failure.
    Unknown,
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "Timeout"),
            Self::RateLimited => write!(f, "RateLimited"),
            Self::ProviderUnavailable => write!(f, "ProviderUnavailable"),
            Self::InvalidRequest => write!(f, "InvalidRequest"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A message in the dead letter queue with rich failure metadata.
#[derive(Debug, Clone)]
pub struct DeadLetterEntry {
    /// The original message that failed.
    pub message: QueueMessage,
    /// Human-readable failure reason.
    pub reason: String,
    /// Categorized failure type.
    pub failure_category: FailureCategory,
    /// Number of delivery attempts before dead-lettering.
    pub attempt_count: u32,
    /// Timestamp (ms since epoch) of first failure.
    pub first_failed_at_ms: u64,
    /// Timestamp (ms since epoch) of last failure.
    pub last_failed_at_ms: u64,
    /// Most recent error messages (capped at 5).
    pub error_history: Vec<String>,
}

/// Aggregated statistics for the dead letter queue.
#[derive(Debug, Clone, Default)]
pub struct DlqStats {
    /// Total entries in the queue.
    pub total: usize,
    /// Count of entries per failure category.
    pub by_category: HashMap<FailureCategory, usize>,
    /// Age in milliseconds of the oldest entry (None if empty).
    pub oldest_age_ms: Option<u64>,
}

/// Dead letter queue for failed messages.
///
/// Stores messages that have exhausted their retry budget with
/// rich metadata for debugging and selective replay.
pub struct DeadLetterQueue {
    entries: Mutex<Vec<DeadLetterEntry>>,
    max_size: usize,
}

impl DeadLetterQueue {
    /// Create a new DLQ with the given maximum capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            max_size,
        }
    }

    /// Add a message with a simple reason string (backward-compatible).
    pub fn add(&self, message: QueueMessage, reason: String) {
        let now_ms = current_time_ms();
        let entry = DeadLetterEntry {
            message,
            reason,
            failure_category: FailureCategory::Unknown,
            attempt_count: 1,
            first_failed_at_ms: now_ms,
            last_failed_at_ms: now_ms,
            error_history: Vec::new(),
        };
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.len() >= self.max_size {
            entries.remove(0);
        }
        entries.push(entry);
    }

    /// Add a message with detailed failure information.
    pub fn add_detailed(
        &self,
        message: QueueMessage,
        reason: String,
        category: FailureCategory,
        attempt_count: u32,
        errors: Vec<String>,
    ) {
        let now_ms = current_time_ms();
        let error_history = if errors.len() > 5 {
            errors[errors.len() - 5..].to_vec()
        } else {
            errors
        };
        let entry = DeadLetterEntry {
            message,
            reason,
            failure_category: category,
            attempt_count,
            first_failed_at_ms: now_ms,
            last_failed_at_ms: now_ms,
            error_history,
        };
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.len() >= self.max_size {
            entries.remove(0);
        }
        entries.push(entry);
    }

    /// Pop the most recently added entry (backward-compatible with old `pop()`).
    pub fn pop(&self) -> Option<(QueueMessage, String)> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.pop().map(|e| (e.message, e.reason))
    }

    /// Pop the oldest entry for replay/reprocessing.
    pub fn replay_one(&self) -> Option<DeadLetterEntry> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.is_empty() {
            None
        } else {
            Some(entries.remove(0))
        }
    }

    /// Remove and return all entries matching the given failure category.
    pub fn replay_by_category(&self, category: FailureCategory) -> Vec<DeadLetterEntry> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut matched = Vec::new();
        let mut remaining = Vec::new();

        for entry in entries.drain(..) {
            if entry.failure_category == category {
                matched.push(entry);
            } else {
                remaining.push(entry);
            }
        }

        *entries = remaining;
        matched
    }

    /// Remove and return entries older than the given age in milliseconds.
    pub fn drain_older_than_ms(&self, age_ms: u64) -> Vec<DeadLetterEntry> {
        let now_ms = current_time_ms();
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut drained = Vec::new();
        let mut remaining = Vec::new();

        for entry in entries.drain(..) {
            if now_ms.saturating_sub(entry.first_failed_at_ms) >= age_ms {
                drained.push(entry);
            } else {
                remaining.push(entry);
            }
        }

        *entries = remaining;
        drained
    }

    /// Peek at all entries without consuming them.
    pub fn peek_all(&self) -> Vec<DeadLetterEntry> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get aggregated statistics.
    pub fn stats(&self) -> DlqStats {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let now_ms = current_time_ms();

        let mut by_category: HashMap<FailureCategory, usize> = HashMap::new();
        let mut oldest_age_ms = None;

        for entry in entries.iter() {
            *by_category.entry(entry.failure_category).or_insert(0) += 1;
            let age = now_ms.saturating_sub(entry.first_failed_at_ms);
            match oldest_age_ms {
                None => oldest_age_ms = Some(age),
                Some(current) if age > current => oldest_age_ms = Some(age),
                _ => {}
            }
        }

        DlqStats {
            total: entries.len(),
            by_category,
            oldest_age_ms,
        }
    }

    /// Number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Returns true if the queue has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries from the queue.
    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

impl Default for DeadLetterQueue {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Get current time in milliseconds since UNIX epoch.
fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue() {
        let queue = MemoryQueue::new(10);

        queue.push(QueueMessage::new("test1")).unwrap();
        queue.push(QueueMessage::new("test2")).unwrap();

        assert_eq!(queue.len(), 2);

        let msg = queue.pop().unwrap();
        assert_eq!(msg.payload, "test1");
    }

    #[test]
    fn test_queue_full() {
        let queue = MemoryQueue::new(2);

        queue.push(QueueMessage::new("1")).unwrap();
        queue.push(QueueMessage::new("2")).unwrap();

        let result = queue.push(QueueMessage::new("3"));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_all_drains_queue() {
        let queue = Arc::new(MemoryQueue::new(100));
        for i in 0..5 {
            queue.push(QueueMessage::new(&format!("msg{}", i))).unwrap();
        }

        let processor = QueueProcessor::new(Arc::clone(&queue), |msg| {
            Ok(format!("processed: {}", msg.payload))
        });

        let results = processor.process_all();
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn test_process_batch_limits() {
        let queue = Arc::new(MemoryQueue::new(100));
        for i in 0..10 {
            queue.push(QueueMessage::new(&format!("msg{}", i))).unwrap();
        }

        let processor = QueueProcessor::new(Arc::clone(&queue), |msg| {
            Ok(format!("processed: {}", msg.payload))
        });

        let results = processor.process_batch(3);
        assert_eq!(results.len(), 3);
        assert_eq!(queue.len(), 7);
    }

    #[test]
    fn test_process_until_empty_stats() {
        let queue = Arc::new(MemoryQueue::new(100));
        // Enqueue 4 messages: 3 will succeed, 1 will fail
        queue.push(QueueMessage::new("ok1")).unwrap();
        queue.push(QueueMessage::new("ok2")).unwrap();
        queue.push(QueueMessage::new("fail")).unwrap();
        queue.push(QueueMessage::new("ok3")).unwrap();

        let processor = QueueProcessor::new(Arc::clone(&queue), |msg| {
            if msg.payload == "fail" {
                Err("processing failed".to_string())
            } else {
                Ok(format!("done: {}", msg.payload))
            }
        });

        // Without start(), should return empty stats
        let stats = processor.process_until_empty();
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.succeeded, 0);
        assert_eq!(stats.failed, 0);
        // Queue should still have all messages
        assert_eq!(queue.len(), 4);

        // Now start and process
        processor.start();
        let stats = processor.process_until_empty();
        assert_eq!(stats.processed, 4);
        assert_eq!(stats.succeeded, 3);
        assert_eq!(stats.failed, 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_message_builders() {
        let msg = QueueMessage::new("hello")
            .with_priority(10)
            .with_metadata("key", "value");
        assert_eq!(msg.payload, "hello");
        assert_eq!(msg.priority, 10);
        assert_eq!(msg.metadata.get("key").unwrap(), "value");
        // Age should be very small (just created)
        assert!(msg.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_queue_clear() {
        let queue = MemoryQueue::new(100);
        queue.push(QueueMessage::new("a")).unwrap();
        queue.push(QueueMessage::new("b")).unwrap();
        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());
        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_dead_letter_queue() {
        let dlq = DeadLetterQueue::new(2);
        assert!(dlq.is_empty());
        dlq.add(QueueMessage::new("fail1"), "timeout".to_string());
        dlq.add(QueueMessage::new("fail2"), "error".to_string());
        assert_eq!(dlq.len(), 2);
        // Adding a third should evict the oldest
        dlq.add(QueueMessage::new("fail3"), "crash".to_string());
        assert_eq!(dlq.len(), 2);
        // Pop returns most recent last (it's a Vec, pop removes from end)
        let (msg, reason) = dlq.pop().unwrap();
        assert_eq!(msg.payload, "fail3");
        assert_eq!(reason, "crash");
    }

    #[test]
    fn test_dlq_add_detailed() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add_detailed(
            QueueMessage::new("msg1"),
            "timeout after 3 attempts".to_string(),
            FailureCategory::Timeout,
            3,
            vec!["err1".to_string(), "err2".to_string(), "err3".to_string()],
        );
        assert_eq!(dlq.len(), 1);

        let entry = dlq.replay_one().unwrap();
        assert_eq!(entry.message.payload, "msg1");
        assert_eq!(entry.failure_category, FailureCategory::Timeout);
        assert_eq!(entry.attempt_count, 3);
        assert_eq!(entry.error_history.len(), 3);
    }

    #[test]
    fn test_dlq_error_history_capped() {
        let dlq = DeadLetterQueue::new(100);
        let errors: Vec<String> = (0..10).map(|i| format!("error_{}", i)).collect();
        dlq.add_detailed(
            QueueMessage::new("msg"),
            "many errors".to_string(),
            FailureCategory::Unknown,
            10,
            errors,
        );

        let entry = dlq.replay_one().unwrap();
        assert_eq!(entry.error_history.len(), 5);
        assert_eq!(entry.error_history[0], "error_5");
        assert_eq!(entry.error_history[4], "error_9");
    }

    #[test]
    fn test_dlq_replay_one_fifo() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add(QueueMessage::new("first"), "r1".to_string());
        dlq.add(QueueMessage::new("second"), "r2".to_string());
        dlq.add(QueueMessage::new("third"), "r3".to_string());

        // replay_one returns oldest first (FIFO)
        let entry = dlq.replay_one().unwrap();
        assert_eq!(entry.message.payload, "first");
        assert_eq!(dlq.len(), 2);
    }

    #[test]
    fn test_dlq_replay_by_category() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add_detailed(
            QueueMessage::new("t1"),
            "timeout".to_string(),
            FailureCategory::Timeout,
            1,
            vec![],
        );
        dlq.add_detailed(
            QueueMessage::new("r1"),
            "rate limited".to_string(),
            FailureCategory::RateLimited,
            1,
            vec![],
        );
        dlq.add_detailed(
            QueueMessage::new("t2"),
            "timeout again".to_string(),
            FailureCategory::Timeout,
            2,
            vec![],
        );

        let timeouts = dlq.replay_by_category(FailureCategory::Timeout);
        assert_eq!(timeouts.len(), 2);
        assert_eq!(timeouts[0].message.payload, "t1");
        assert_eq!(timeouts[1].message.payload, "t2");
        // Only rate limited entry remains
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_dlq_peek_all() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add(QueueMessage::new("a"), "reason_a".to_string());
        dlq.add(QueueMessage::new("b"), "reason_b".to_string());

        let peeked = dlq.peek_all();
        assert_eq!(peeked.len(), 2);
        // peek doesn't consume
        assert_eq!(dlq.len(), 2);
    }

    #[test]
    fn test_dlq_clear() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add(QueueMessage::new("a"), "r".to_string());
        dlq.add(QueueMessage::new("b"), "r".to_string());
        assert_eq!(dlq.len(), 2);

        dlq.clear();
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_stats() {
        let dlq = DeadLetterQueue::new(100);
        dlq.add_detailed(
            QueueMessage::new("t1"),
            "timeout".to_string(),
            FailureCategory::Timeout,
            1,
            vec![],
        );
        dlq.add_detailed(
            QueueMessage::new("t2"),
            "timeout".to_string(),
            FailureCategory::Timeout,
            2,
            vec![],
        );
        dlq.add_detailed(
            QueueMessage::new("r1"),
            "rate".to_string(),
            FailureCategory::RateLimited,
            1,
            vec![],
        );

        let stats = dlq.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(*stats.by_category.get(&FailureCategory::Timeout).unwrap_or(&0), 2);
        assert_eq!(*stats.by_category.get(&FailureCategory::RateLimited).unwrap_or(&0), 1);
        assert!(stats.oldest_age_ms.is_some());
    }

    #[test]
    fn test_dlq_backward_compat_add() {
        // Ensure the old add() method still works and creates Unknown category
        let dlq = DeadLetterQueue::new(100);
        dlq.add(QueueMessage::new("old_style"), "old reason".to_string());

        let entries = dlq.peek_all();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].failure_category, FailureCategory::Unknown);
        assert_eq!(entries[0].attempt_count, 1);
    }

    #[test]
    fn test_dlq_max_size_eviction_detailed() {
        let dlq = DeadLetterQueue::new(2);
        dlq.add_detailed(QueueMessage::new("m1"), "r1".to_string(), FailureCategory::Timeout, 1, vec![]);
        dlq.add_detailed(QueueMessage::new("m2"), "r2".to_string(), FailureCategory::RateLimited, 1, vec![]);
        dlq.add_detailed(QueueMessage::new("m3"), "r3".to_string(), FailureCategory::Unknown, 1, vec![]);
        assert_eq!(dlq.len(), 2);

        // m1 was evicted (oldest), m2 and m3 remain
        let entries = dlq.peek_all();
        assert_eq!(entries[0].message.payload, "m2");
        assert_eq!(entries[1].message.payload, "m3");
    }

    #[test]
    fn test_dlq_failure_category_display() {
        assert_eq!(FailureCategory::Timeout.to_string(), "Timeout");
        assert_eq!(FailureCategory::RateLimited.to_string(), "RateLimited");
        assert_eq!(FailureCategory::ProviderUnavailable.to_string(), "ProviderUnavailable");
        assert_eq!(FailureCategory::InvalidRequest.to_string(), "InvalidRequest");
        assert_eq!(FailureCategory::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_queue_error_display() {
        assert_eq!(QueueError::Full.to_string(), "Queue is full");
        assert_eq!(QueueError::Empty.to_string(), "Queue is empty");
        assert_eq!(QueueError::Timeout.to_string(), "Operation timed out");
        assert_eq!(QueueError::ConnectionFailed.to_string(), "Connection failed");
    }

    #[test]
    fn test_processor_start_stop() {
        let queue = Arc::new(MemoryQueue::new(100));
        let processor = QueueProcessor::new(Arc::clone(&queue), |_| Ok("ok".to_string()));
        assert!(!processor.is_running());
        processor.start();
        assert!(processor.is_running());
        processor.stop();
        assert!(!processor.is_running());
    }
}
