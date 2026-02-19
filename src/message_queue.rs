//! Message queue integration
//!
//! Queue-based processing for AI requests.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar};
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
            let result = self.signal.wait_timeout(queue, remaining).unwrap_or_else(|e| e.into_inner());
            queue = result.0;
            if result.1.timed_out() {
                return None;
            }
        }

        queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.messages.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.messages.lock().unwrap_or_else(|e| e.into_inner()).clear();
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
        self.messages.lock().unwrap_or_else(|e| e.into_inner()).pop()
    }

    pub fn len(&self) -> usize {
        self.messages.lock().unwrap_or_else(|e| e.into_inner()).len()
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
}
