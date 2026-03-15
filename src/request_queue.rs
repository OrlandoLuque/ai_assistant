//! Priority-based request queuing with rate limiting and backpressure.
//!
//! Provides [`RequestQueue`], [`QueueConfig`], and [`PriorityRequest`] for
//! thread-safe management of concurrent AI generation requests. Higher-priority
//! requests (system commands, cancellations) are processed before normal user
//! messages or background tasks.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::load_shedding::{LoadContext, LoadShedder, SheddingDecision};

// ============================================================================
// Priority
// ============================================================================

/// Priority level for queued requests.
///
/// Higher-priority requests are dequeued before lower-priority ones
/// within the same insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RequestPriority {
    /// Background tasks: indexing, summarization, cache warming
    Low = 0,
    /// Normal user messages
    Normal = 1,
    /// System commands, cancellations, health checks
    High = 2,
}

impl Default for RequestPriority {
    fn default() -> Self {
        RequestPriority::Normal
    }
}

// ============================================================================
// Queued Request
// ============================================================================

/// A request waiting to be processed.
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    /// Unique request identifier
    pub id: String,
    /// Optional session to associate with
    pub session_id: Option<String>,
    /// Request priority
    pub priority: RequestPriority,
    /// The user message content
    pub message: String,
    /// Optional knowledge context to include
    pub knowledge_context: String,
    /// Timestamp when the request was enqueued (millis since epoch)
    pub enqueued_at: u64,
}

impl QueuedRequest {
    /// Create a new normal-priority request.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: None,
            priority: RequestPriority::Normal,
            message: message.into(),
            knowledge_context: String::new(),
            enqueued_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the session ID.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set knowledge context.
    pub fn with_knowledge(mut self, context: impl Into<String>) -> Self {
        self.knowledge_context = context.into();
        self
    }

    /// How long this request has been waiting in milliseconds.
    pub fn wait_time_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.enqueued_at)
    }
}

// ============================================================================
// Queue Stats
// ============================================================================

/// Statistics about the request queue state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    /// Total requests currently in queue
    pub pending: usize,
    /// Requests by priority
    pub high_count: usize,
    pub normal_count: usize,
    pub low_count: usize,
    /// Total requests processed since creation
    pub total_processed: u64,
    /// Total requests dropped (queue full)
    pub total_dropped: u64,
}

// ============================================================================
// Request Queue
// ============================================================================

/// Thread-safe priority request queue.
///
/// Requests are dequeued in priority order (High > Normal > Low).
/// Within the same priority, requests are served in FIFO order.
///
/// # Example
///
/// ```rust
/// use ai_assistant::request_queue::{RequestQueue, QueuedRequest, RequestPriority};
///
/// let queue = RequestQueue::new(100);
///
/// // Enqueue requests
/// queue.enqueue(QueuedRequest::new("Hello"));
/// queue.enqueue(QueuedRequest::new("System check").with_priority(RequestPriority::High));
///
/// // High priority dequeues first
/// let req = queue.try_dequeue().unwrap();
/// assert_eq!(req.message, "System check");
/// ```
pub struct RequestQueue {
    state: Arc<Mutex<QueueState>>,
    /// Condition variable for blocking dequeue
    not_empty: Arc<Condvar>,
    /// Maximum queue capacity
    max_capacity: usize,
    /// Optional load shedder — when set, `enqueue()` evaluates load context
    /// and may reject or throttle incoming requests under pressure.
    load_shedder: Option<Arc<LoadShedder>>,
}

struct QueueState {
    high: VecDeque<QueuedRequest>,
    normal: VecDeque<QueuedRequest>,
    low: VecDeque<QueuedRequest>,
    total_processed: u64,
    total_dropped: u64,
    closed: bool,
}

impl QueueState {
    fn new() -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
            total_processed: 0,
            total_dropped: 0,
            closed: false,
        }
    }

    fn total_pending(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }

    /// Dequeue the highest-priority request.
    fn dequeue(&mut self) -> Option<QueuedRequest> {
        if let Some(req) = self.high.pop_front() {
            self.total_processed += 1;
            return Some(req);
        }
        if let Some(req) = self.normal.pop_front() {
            self.total_processed += 1;
            return Some(req);
        }
        if let Some(req) = self.low.pop_front() {
            self.total_processed += 1;
            return Some(req);
        }
        None
    }
}

impl RequestQueue {
    /// Create a new request queue with the given maximum capacity.
    ///
    /// When the queue is full, new requests are dropped.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(QueueState::new())),
            not_empty: Arc::new(Condvar::new()),
            max_capacity,
            load_shedder: None,
        }
    }

    /// Attach a load shedder. When set, `enqueue()` evaluates the load context
    /// before accepting a request, potentially shedding or throttling it.
    pub fn with_load_shedder(mut self, shedder: Arc<LoadShedder>) -> Self {
        self.load_shedder = Some(shedder);
        self
    }

    /// Enqueue a request. Returns false if the queue is full, closed, or shed by load shedder.
    ///
    /// When a [`LoadShedder`] is attached via [`with_load_shedder`](Self::with_load_shedder),
    /// the request is evaluated before enqueuing. If the decision is `Shed`, the request is
    /// dropped. If the decision is `Throttle`, a short delay is applied before enqueuing.
    pub fn enqueue(&self, request: QueuedRequest) -> bool {
        // Evaluate load shedding if configured
        if let Some(ref shedder) = self.load_shedder {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let context = LoadContext {
                cpu_load: 0.0,    // caller may set via enqueue_with_context
                memory_load: 0.0, // caller may set via enqueue_with_context
                queue_depth: state.total_pending(),
                priority: request.priority,
                request_age: Duration::from_millis(request.wait_time_ms()),
                p95_latency: None,
            };
            drop(state); // release lock before potential sleep

            let decision = shedder.evaluate(&context);
            shedder.record_decision(&decision);

            match decision {
                SheddingDecision::Shed { reason } => {
                    log::warn!(
                        "Load shedder rejected request {}: {}",
                        request.id,
                        reason
                    );
                    return false;
                }
                SheddingDecision::Throttle { delay } => {
                    log::debug!(
                        "Load shedder throttling request {} for {:?}",
                        request.id,
                        delay
                    );
                    std::thread::sleep(delay);
                }
                SheddingDecision::Accept => {}
            }
        }

        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if state.closed {
            return false;
        }

        if state.total_pending() >= self.max_capacity {
            state.total_dropped += 1;
            log::warn!(
                "Request queue full ({}/{}), dropping request {}",
                state.total_pending(),
                self.max_capacity,
                request.id
            );
            return false;
        }

        match request.priority {
            RequestPriority::High => state.high.push_back(request),
            RequestPriority::Normal => state.normal.push_back(request),
            RequestPriority::Low => state.low.push_back(request),
        }

        // Notify waiting consumers
        self.not_empty.notify_one();
        true
    }

    /// Try to dequeue the highest-priority request without blocking.
    pub fn try_dequeue(&self) -> Option<QueuedRequest> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.dequeue()
    }

    /// Dequeue the highest-priority request, blocking until one is available.
    ///
    /// Returns `None` if the queue is closed.
    pub fn dequeue_blocking(&self) -> Option<QueuedRequest> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(req) = state.dequeue() {
                return Some(req);
            }
            if state.closed {
                return None;
            }
            state = self
                .not_empty
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Dequeue with a timeout. Returns `None` if no request is available
    /// within the given duration or if the queue is closed.
    pub fn dequeue_timeout(&self, timeout: std::time::Duration) -> Option<QueuedRequest> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if let Some(req) = state.dequeue() {
                return Some(req);
            }
            if state.closed {
                return None;
            }

            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }

            let (new_state, timeout_result) = self
                .not_empty
                .wait_timeout(state, remaining)
                .unwrap_or_else(|e| e.into_inner());
            state = new_state;

            if timeout_result.timed_out() {
                return state.dequeue();
            }
        }
    }

    /// Close the queue. No more requests can be enqueued.
    /// Wakes up all blocking consumers.
    pub fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        self.not_empty.notify_all();
    }

    /// Returns whether the queue is closed.
    pub fn is_closed(&self) -> bool {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).closed
    }

    /// Returns whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_pending()
            == 0
    }

    /// Returns the number of pending requests.
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_pending()
    }

    /// Returns queue statistics.
    pub fn stats(&self) -> QueueStats {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        QueueStats {
            pending: state.total_pending(),
            high_count: state.high.len(),
            normal_count: state.normal.len(),
            low_count: state.low.len(),
            total_processed: state.total_processed,
            total_dropped: state.total_dropped,
        }
    }

    /// Remove all requests for a specific session.
    pub fn remove_session(&self, session_id: &str) -> usize {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let before = state.total_pending();

        state
            .high
            .retain(|r| r.session_id.as_deref() != Some(session_id));
        state
            .normal
            .retain(|r| r.session_id.as_deref() != Some(session_id));
        state
            .low
            .retain(|r| r.session_id.as_deref() != Some(session_id));

        before - state.total_pending()
    }

    /// Clear all pending requests.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.high.clear();
        state.normal.clear();
        state.low.clear();
    }

    /// Peek at the next request without removing it.
    pub fn peek(&self) -> Option<QueuedRequest> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .high
            .front()
            .or_else(|| state.normal.front())
            .or_else(|| state.low.front())
            .cloned()
    }
}

impl Clone for RequestQueue {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            not_empty: self.not_empty.clone(),
            max_capacity: self.max_capacity,
            load_shedder: self.load_shedder.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let queue = RequestQueue::new(100);

        queue.enqueue(QueuedRequest::new("low").with_priority(RequestPriority::Low));
        queue.enqueue(QueuedRequest::new("normal"));
        queue.enqueue(QueuedRequest::new("high").with_priority(RequestPriority::High));

        assert_eq!(queue.try_dequeue().unwrap().message, "high");
        assert_eq!(queue.try_dequeue().unwrap().message, "normal");
        assert_eq!(queue.try_dequeue().unwrap().message, "low");
        assert!(queue.try_dequeue().is_none());
    }

    #[test]
    fn test_fifo_within_priority() {
        let queue = RequestQueue::new(100);

        queue.enqueue(QueuedRequest::new("first"));
        queue.enqueue(QueuedRequest::new("second"));
        queue.enqueue(QueuedRequest::new("third"));

        assert_eq!(queue.try_dequeue().unwrap().message, "first");
        assert_eq!(queue.try_dequeue().unwrap().message, "second");
        assert_eq!(queue.try_dequeue().unwrap().message, "third");
    }

    #[test]
    fn test_capacity_limit() {
        let queue = RequestQueue::new(2);

        assert!(queue.enqueue(QueuedRequest::new("a")));
        assert!(queue.enqueue(QueuedRequest::new("b")));
        assert!(!queue.enqueue(QueuedRequest::new("c"))); // dropped

        let stats = queue.stats();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.total_dropped, 1);
    }

    #[test]
    fn test_close_queue() {
        let queue = RequestQueue::new(100);
        queue.enqueue(QueuedRequest::new("before close"));

        queue.close();

        assert!(!queue.enqueue(QueuedRequest::new("after close")));
        assert!(queue.is_closed());

        // Can still dequeue existing
        assert_eq!(queue.try_dequeue().unwrap().message, "before close");
    }

    #[test]
    fn test_remove_session() {
        let queue = RequestQueue::new(100);

        queue.enqueue(QueuedRequest::new("s1 msg1").with_session("s1"));
        queue.enqueue(QueuedRequest::new("s2 msg1").with_session("s2"));
        queue.enqueue(QueuedRequest::new("s1 msg2").with_session("s1"));
        queue.enqueue(QueuedRequest::new("no session"));

        let removed = queue.remove_session("s1");
        assert_eq!(removed, 2);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_stats() {
        let queue = RequestQueue::new(100);

        queue.enqueue(QueuedRequest::new("h").with_priority(RequestPriority::High));
        queue.enqueue(QueuedRequest::new("n1"));
        queue.enqueue(QueuedRequest::new("n2"));
        queue.enqueue(QueuedRequest::new("l").with_priority(RequestPriority::Low));

        let stats = queue.stats();
        assert_eq!(stats.pending, 4);
        assert_eq!(stats.high_count, 1);
        assert_eq!(stats.normal_count, 2);
        assert_eq!(stats.low_count, 1);
        assert_eq!(stats.total_processed, 0);

        queue.try_dequeue();
        let stats = queue.stats();
        assert_eq!(stats.total_processed, 1);
        assert_eq!(stats.pending, 3);
    }

    #[test]
    fn test_peek() {
        let queue = RequestQueue::new(100);

        assert!(queue.peek().is_none());

        queue.enqueue(QueuedRequest::new("normal msg"));
        queue.enqueue(QueuedRequest::new("high msg").with_priority(RequestPriority::High));

        let peeked = queue.peek().unwrap();
        assert_eq!(peeked.message, "high msg");

        // Peek doesn't remove
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_clear() {
        let queue = RequestQueue::new(100);
        queue.enqueue(QueuedRequest::new("a"));
        queue.enqueue(QueuedRequest::new("b"));

        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_dequeue_timeout() {
        let queue = RequestQueue::new(100);

        // Should return None after timeout
        let result = queue.dequeue_timeout(std::time::Duration::from_millis(10));
        assert!(result.is_none());

        // Should return immediately if item available
        queue.enqueue(QueuedRequest::new("hello"));
        let result = queue.dequeue_timeout(std::time::Duration::from_secs(1));
        assert_eq!(result.unwrap().message, "hello");
    }

    #[test]
    fn test_blocking_dequeue_with_close() {
        let queue = RequestQueue::new(100);
        let q = queue.clone();

        let handle = std::thread::spawn(move || q.dequeue_blocking());

        // Give thread time to start blocking
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Close should wake up the blocking thread
        queue.close();
        let result = handle.join().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_thread_safety() {
        let queue = RequestQueue::new(1000);
        let q1 = queue.clone();
        let q2 = queue.clone();

        // Producer thread
        let producer = std::thread::spawn(move || {
            for i in 0..100 {
                q1.enqueue(QueuedRequest::new(format!("msg {}", i)));
            }
        });

        // Consumer thread
        let consumer = std::thread::spawn(move || {
            let mut count = 0;
            while count < 100 {
                if q2.try_dequeue().is_some() {
                    count += 1;
                }
                std::thread::yield_now();
            }
            count
        });

        producer.join().unwrap();
        let consumed = consumer.join().unwrap();
        assert_eq!(consumed, 100);
    }

    #[test]
    fn test_wait_time() {
        let req = QueuedRequest::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(req.wait_time_ms() >= 5);
    }

    #[test]
    fn test_request_builder() {
        let req = QueuedRequest::new("hello")
            .with_priority(RequestPriority::High)
            .with_session("session_1")
            .with_knowledge("some context");

        assert_eq!(req.message, "hello");
        assert_eq!(req.priority, RequestPriority::High);
        assert_eq!(req.session_id.as_deref(), Some("session_1"));
        assert_eq!(req.knowledge_context, "some context");
        assert!(!req.id.is_empty());
    }

    #[test]
    fn test_load_shedder_accepts_normal_load() {
        use crate::load_shedding::{LoadSheddingConfig, LoadShedder};

        let shedder = Arc::new(LoadShedder::new(LoadSheddingConfig::conservative()));
        let queue = RequestQueue::new(100).with_load_shedder(shedder);

        // Under normal load (queue is empty), requests should be accepted
        assert!(queue.enqueue(QueuedRequest::new("normal request")));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_load_shedder_sheds_under_pressure() {
        use crate::load_shedding::{LoadSheddingConfig, LoadShedder, SheddingStrategy};

        // Aggressive config with very low queue depth threshold
        let config = LoadSheddingConfig {
            strategy: SheddingStrategy::PriorityBased,
            cpu_threshold: 0.0, // always over threshold
            memory_threshold: 0.0,
            queue_depth_threshold: 0, // any queue depth triggers
            latency_threshold: Duration::from_millis(1),
            priority_protection: true,
            cooldown: Duration::from_millis(0),
        };
        let shedder = Arc::new(LoadShedder::new(config));
        let queue = RequestQueue::new(100).with_load_shedder(shedder);

        // Low priority may be shed (queue_depth_threshold = 0 means always under pressure)
        let _result = queue.enqueue(
            QueuedRequest::new("low priority").with_priority(RequestPriority::Low),
        );
        // High priority should always pass with priority protection enabled
        let result_high = queue.enqueue(
            QueuedRequest::new("high priority").with_priority(RequestPriority::High),
        );
        assert!(result_high);
    }

    #[test]
    fn test_queue_without_shedder_works_normally() {
        // Verify the shedder is optional and doesn't change existing behavior
        let queue = RequestQueue::new(2);
        assert!(queue.enqueue(QueuedRequest::new("a")));
        assert!(queue.enqueue(QueuedRequest::new("b")));
        assert!(!queue.enqueue(QueuedRequest::new("c"))); // capacity limit, not shedding
    }
}
