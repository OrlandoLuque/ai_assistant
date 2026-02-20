//! Message queue integration
//!
//! Queue-based processing for AI requests.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

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

/// Dead letter queue for failed messages
pub struct DeadLetterQueue {
    messages: Mutex<Vec<(QueueMessage, String)>>,
    max_size: usize,
}

impl DeadLetterQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            max_size,
        }
    }

    pub fn add(&self, message: QueueMessage, reason: String) {
        let mut messages = self.messages.lock().unwrap_or_else(|e| e.into_inner());
        if messages.len() >= self.max_size {
            messages.remove(0);
        }
        messages.push((message, reason));
    }

    pub fn pop(&self) -> Option<(QueueMessage, String)> {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
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
}

impl Default for DeadLetterQueue {
    fn default() -> Self {
        Self::new(1000)
    }
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
}
