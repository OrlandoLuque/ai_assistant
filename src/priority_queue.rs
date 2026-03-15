//! Request priority queue
//!
//! This module provides a priority-based queue for processing AI requests,
//! allowing urgent requests to be processed before others.
//!
//! # Features
//!
//! - **Priority levels**: Critical, High, Normal, Low, Background
//! - **Fair scheduling**: Prevents starvation of low-priority requests
//! - **Deadline support**: Time-sensitive requests get promoted
//! - **Cancellation**: Cancel pending requests
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::priority_queue::{PriorityQueue, PriorityRequest, Priority};
//!
//! let queue = PriorityQueue::new(100);
//!
//! // Add requests with different priorities
//! queue.enqueue(PriorityRequest::new("urgent task", Priority::Critical));
//! queue.enqueue(PriorityRequest::new("background task", Priority::Background));
//!
//! // Critical request will be dequeued first
//! let next = queue.dequeue();
//! assert_eq!(next.unwrap().priority, Priority::Critical);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Priority levels for requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Priority {
    /// System critical - immediate processing
    Critical = 0,
    /// User-initiated urgent request
    High = 1,
    /// Standard priority
    Normal = 2,
    /// Lower priority, can wait
    Low = 3,
    /// Background task, process when idle
    Background = 4,
}

impl Priority {
    /// Get numeric value (lower = higher priority)
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value
    pub fn from_value(v: u8) -> Self {
        match v {
            0 => Self::Critical,
            1 => Self::High,
            2 => Self::Normal,
            3 => Self::Low,
            _ => Self::Background,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Critical => "Critical",
            Self::High => "High",
            Self::Normal => "Normal",
            Self::Low => "Low",
            Self::Background => "Background",
        }
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower value = higher priority, so reverse the comparison
        other.value().cmp(&self.value())
    }
}

/// A request with priority information
#[derive(Debug, Clone)]
pub struct PriorityRequest {
    /// Unique request ID
    pub id: String,
    /// Request content/prompt
    pub content: String,
    /// Priority level
    pub priority: Priority,
    /// When the request was created
    pub created_at: Instant,
    /// Optional deadline for the request
    pub deadline: Option<Instant>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
    /// Whether this request can be cancelled
    pub cancellable: bool,
    /// User/session ID for fairness
    pub user_id: Option<String>,
}

impl PriorityRequest {
    /// Create a new priority request
    pub fn new(content: impl Into<String>, priority: Priority) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            priority,
            created_at: Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
            cancellable: true,
            user_id: None,
        }
    }

    /// Set a deadline for this request
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set deadline as duration from now
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.deadline = Some(Instant::now() + timeout);
        self
    }

    /// Set user ID for fairness tracking
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Mark as non-cancellable
    pub fn non_cancellable(mut self) -> Self {
        self.cancellable = false;
        self
    }

    /// Check if deadline has passed
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Instant::now() > d).unwrap_or(false)
    }

    /// Get time until deadline (if any)
    pub fn time_until_deadline(&self) -> Option<Duration> {
        self.deadline.and_then(|d| {
            let now = Instant::now();
            if d > now {
                Some(d - now)
            } else {
                None
            }
        })
    }

    /// Get age of request
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Calculate effective priority (considers age and deadline)
    fn effective_priority(&self, config: &QueueConfig) -> i64 {
        let base = (4 - self.priority.value() as i64) * 1000;

        // Boost based on age (anti-starvation)
        let age_boost = if config.anti_starvation {
            (self.age().as_secs() / config.age_boost_interval.as_secs()) as i64 * 100
        } else {
            0
        };

        // Boost based on deadline proximity
        let deadline_boost = self
            .time_until_deadline()
            .map(|remaining| {
                if remaining < config.deadline_urgent_threshold {
                    500 // Urgent boost
                } else if remaining < config.deadline_urgent_threshold * 2 {
                    200 // Moderate boost
                } else {
                    0
                }
            })
            .unwrap_or(0);

        base + age_boost + deadline_boost
    }
}

/// Internal entry for the heap
struct QueueEntry {
    request: PriorityRequest,
    effective_priority: i64,
    sequence: u64, // For FIFO within same priority
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.effective_priority == other.effective_priority && self.sequence == other.sequence
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.effective_priority.cmp(&other.effective_priority) {
            Ordering::Equal => other.sequence.cmp(&self.sequence), // Lower sequence = older = higher priority
            other => other,
        }
    }
}

/// Queue configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QueueConfig {
    /// Maximum queue size
    pub max_size: usize,
    /// Enable anti-starvation (boost old requests)
    pub anti_starvation: bool,
    /// Interval for age-based priority boost
    pub age_boost_interval: Duration,
    /// Threshold for deadline urgency boost
    pub deadline_urgent_threshold: Duration,
    /// Maximum requests per user in queue
    pub max_per_user: Option<usize>,
    /// Enable fairness across users
    pub fair_scheduling: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            anti_starvation: true,
            age_boost_interval: Duration::from_secs(60),
            deadline_urgent_threshold: Duration::from_secs(5),
            max_per_user: Some(10),
            fair_scheduling: true,
        }
    }
}

/// Priority queue for AI requests
pub struct PriorityQueue {
    config: QueueConfig,
    heap: Mutex<BinaryHeap<QueueEntry>>,
    sequence: Mutex<u64>,
    pending: RwLock<HashMap<String, PriorityRequest>>,
    user_counts: Mutex<HashMap<String, usize>>,
    stats: Mutex<QueueStats>,
}

impl PriorityQueue {
    /// Create a new priority queue
    pub fn new(max_size: usize) -> Self {
        Self::with_config(QueueConfig {
            max_size,
            ..Default::default()
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: QueueConfig) -> Self {
        Self {
            config,
            heap: Mutex::new(BinaryHeap::new()),
            sequence: Mutex::new(0),
            pending: RwLock::new(HashMap::new()),
            user_counts: Mutex::new(HashMap::new()),
            stats: Mutex::new(QueueStats::default()),
        }
    }

    /// Enqueue a request
    pub fn enqueue(&self, request: PriorityRequest) -> Result<String, QueueError> {
        // Check capacity
        {
            let heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
            if heap.len() >= self.config.max_size {
                return Err(QueueError::QueueFull);
            }
        }

        // Check per-user limit
        if let Some(user_id) = &request.user_id {
            if let Some(max) = self.config.max_per_user {
                let mut user_counts = self.user_counts.lock().unwrap_or_else(|e| e.into_inner());
                let count = user_counts.entry(user_id.clone()).or_insert(0);
                if *count >= max {
                    return Err(QueueError::UserLimitExceeded);
                }
                *count += 1;
            }
        }

        let request_id = request.id.clone();

        // Calculate effective priority
        let effective_priority = request.effective_priority(&self.config);

        // Get sequence number
        let sequence = {
            let mut seq = self.sequence.lock().unwrap_or_else(|e| e.into_inner());
            *seq += 1;
            *seq
        };

        // Add to pending map
        {
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            pending.insert(request_id.clone(), request.clone());
        }

        // Add to heap
        {
            let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
            heap.push(QueueEntry {
                request,
                effective_priority,
                sequence,
            });
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_enqueued += 1;
        }

        Ok(request_id)
    }

    /// Dequeue the highest priority request
    pub fn dequeue(&self) -> Option<PriorityRequest> {
        let entry = {
            let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
            heap.pop()
        }?;

        let request = entry.request;

        // Remove from pending
        {
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            pending.remove(&request.id);
        }

        // Update user count
        if let Some(user_id) = &request.user_id {
            let mut user_counts = self.user_counts.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(count) = user_counts.get_mut(user_id) {
                *count = count.saturating_sub(1);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_dequeued += 1;
            stats.total_wait_time += request.age();
        }

        Some(request)
    }

    /// Dequeue with timeout
    pub fn dequeue_timeout(&self, timeout: Duration) -> Option<PriorityRequest> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if let Some(request) = self.dequeue() {
                return Some(request);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    /// Peek at the highest priority request without removing
    pub fn peek(&self) -> Option<PriorityRequest> {
        let heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
        heap.peek().map(|e| e.request.clone())
    }

    /// Cancel a pending request
    pub fn cancel(&self, request_id: &str) -> Result<PriorityRequest, QueueError> {
        // Remove from pending
        let request = {
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            pending.remove(request_id)
        };

        let request = request.ok_or(QueueError::NotFound)?;

        if !request.cancellable {
            // Put it back
            let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
            pending.insert(request_id.to_string(), request);
            return Err(QueueError::NotCancellable);
        }

        // Rebuild heap without this request
        {
            let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
            let entries: Vec<_> = std::mem::take(&mut *heap)
                .into_iter()
                .filter(|e| e.request.id != request_id)
                .collect();
            *heap = entries.into_iter().collect();
        }

        // Update user count
        if let Some(user_id) = &request.user_id {
            let mut user_counts = self.user_counts.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(count) = user_counts.get_mut(user_id) {
                *count = count.saturating_sub(1);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_cancelled += 1;
        }

        Ok(request)
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.heap.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get length by priority
    pub fn len_by_priority(&self, priority: Priority) -> usize {
        let pending = self.pending.read().unwrap_or_else(|e| e.into_inner());
        pending.values().filter(|r| r.priority == priority).count()
    }

    /// Get all pending requests for a user
    pub fn get_user_requests(&self, user_id: &str) -> Vec<PriorityRequest> {
        let pending = self.pending.read().unwrap_or_else(|e| e.into_inner());
        pending
            .values()
            .filter(|r| r.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect()
    }

    /// Clear all requests
    pub fn clear(&self) {
        let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
        let mut pending = self.pending.write().unwrap_or_else(|e| e.into_inner());
        let mut user_counts = self.user_counts.lock().unwrap_or_else(|e| e.into_inner());

        heap.clear();
        pending.clear();
        user_counts.clear();
    }

    /// Remove expired requests
    pub fn remove_expired(&self) -> Vec<PriorityRequest> {
        let mut expired = Vec::new();

        {
            let pending = self.pending.read().unwrap_or_else(|e| e.into_inner());
            for request in pending.values() {
                if request.is_expired() {
                    expired.push(request.clone());
                }
            }
        }

        for request in &expired {
            let _ = self.cancel(&request.id);
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_expired += expired.len() as u64;
        }

        expired
    }

    /// Rebalance priorities (recalculate effective priorities)
    pub fn rebalance(&self) {
        let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());

        let entries: Vec<_> = std::mem::take(&mut *heap)
            .into_iter()
            .map(|mut e| {
                e.effective_priority = e.request.effective_priority(&self.config);
                e
            })
            .collect();

        *heap = entries.into_iter().collect();
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        let heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());

        QueueStats {
            current_size: heap.len(),
            total_enqueued: stats.total_enqueued,
            total_dequeued: stats.total_dequeued,
            total_cancelled: stats.total_cancelled,
            total_expired: stats.total_expired,
            total_wait_time: stats.total_wait_time,
            avg_wait_time: if stats.total_dequeued > 0 {
                stats.total_wait_time / stats.total_dequeued as u32
            } else {
                Duration::ZERO
            },
        }
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Queue statistics
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Current queue size
    pub current_size: usize,
    /// Total requests enqueued
    pub total_enqueued: u64,
    /// Total requests dequeued
    pub total_dequeued: u64,
    /// Total requests cancelled
    pub total_cancelled: u64,
    /// Total requests expired
    pub total_expired: u64,
    /// Total wait time across all requests
    pub total_wait_time: Duration,
    /// Average wait time
    pub avg_wait_time: Duration,
}

/// Queue errors
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueError {
    /// Queue is at capacity
    QueueFull,
    /// User has too many pending requests
    UserLimitExceeded,
    /// Request not found
    NotFound,
    /// Request cannot be cancelled
    NotCancellable,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "Queue is full"),
            Self::UserLimitExceeded => write!(f, "User limit exceeded"),
            Self::NotFound => write!(f, "Request not found"),
            Self::NotCancellable => write!(f, "Request cannot be cancelled"),
        }
    }
}

impl std::error::Error for QueueError {}

/// Thread-safe queue handle
pub type SharedPriorityQueue = Arc<PriorityQueue>;

/// Create a shared priority queue
pub fn create_shared_queue(max_size: usize) -> SharedPriorityQueue {
    Arc::new(PriorityQueue::new(max_size))
}

/// Priority queue with multiple worker support
pub struct WorkerQueue {
    queue: SharedPriorityQueue,
    worker_count: usize,
}

impl WorkerQueue {
    /// Create a new worker queue
    pub fn new(max_size: usize, workers: usize) -> Self {
        Self {
            queue: create_shared_queue(max_size),
            worker_count: workers,
        }
    }

    /// Get the underlying queue
    pub fn queue(&self) -> &SharedPriorityQueue {
        &self.queue
    }

    /// Get worker count
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Enqueue a request
    pub fn submit(&self, request: PriorityRequest) -> Result<String, QueueError> {
        self.queue.enqueue(request)
    }

    /// Take the next request (for workers)
    pub fn take(&self) -> Option<PriorityRequest> {
        self.queue.dequeue()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Low > Priority::Background);
    }

    #[test]
    fn test_basic_queue() {
        let queue = PriorityQueue::new(100);

        queue
            .enqueue(PriorityRequest::new("low", Priority::Low))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("critical", Priority::Critical))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("normal", Priority::Normal))
            .unwrap();

        // Should get critical first
        let first = queue.dequeue().unwrap();
        assert_eq!(first.priority, Priority::Critical);

        let second = queue.dequeue().unwrap();
        assert_eq!(second.priority, Priority::Normal);

        let third = queue.dequeue().unwrap();
        assert_eq!(third.priority, Priority::Low);
    }

    #[test]
    fn test_queue_full() {
        let queue = PriorityQueue::new(2);

        queue
            .enqueue(PriorityRequest::new("1", Priority::Normal))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("2", Priority::Normal))
            .unwrap();

        let result = queue.enqueue(PriorityRequest::new("3", Priority::Normal));
        assert_eq!(result, Err(QueueError::QueueFull));
    }

    #[test]
    fn test_cancel() {
        let queue = PriorityQueue::new(100);

        let id = queue
            .enqueue(PriorityRequest::new("test", Priority::Normal))
            .unwrap();
        assert_eq!(queue.len(), 1);

        let cancelled = queue.cancel(&id).unwrap();
        assert_eq!(cancelled.content, "test");
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_user_limit() {
        let config = QueueConfig {
            max_per_user: Some(2),
            ..Default::default()
        };
        let queue = PriorityQueue::with_config(config);

        queue
            .enqueue(PriorityRequest::new("1", Priority::Normal).with_user("user1"))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("2", Priority::Normal).with_user("user1"))
            .unwrap();

        let result = queue.enqueue(PriorityRequest::new("3", Priority::Normal).with_user("user1"));
        assert_eq!(result, Err(QueueError::UserLimitExceeded));

        // Different user should work
        queue
            .enqueue(PriorityRequest::new("4", Priority::Normal).with_user("user2"))
            .unwrap();
    }

    #[test]
    fn test_deadline() {
        let request =
            PriorityRequest::new("test", Priority::Normal).with_timeout(Duration::from_secs(10));

        assert!(!request.is_expired());
        assert!(request.time_until_deadline().is_some());
    }

    #[test]
    fn test_peek_and_len_by_priority() {
        let queue = PriorityQueue::new(100);

        assert!(queue.is_empty());
        assert!(queue.peek().is_none());

        queue
            .enqueue(PriorityRequest::new("bg1", Priority::Background))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("high1", Priority::High))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("bg2", Priority::Background))
            .unwrap();

        // Peek should return highest priority without removing
        let peeked = queue.peek().unwrap();
        assert_eq!(peeked.priority, Priority::High);
        assert_eq!(queue.len(), 3); // still 3, nothing removed

        // Count by priority
        assert_eq!(queue.len_by_priority(Priority::Background), 2);
        assert_eq!(queue.len_by_priority(Priority::High), 1);
        assert_eq!(queue.len_by_priority(Priority::Critical), 0);
    }

    #[test]
    fn test_non_cancellable_request() {
        let queue = PriorityQueue::new(100);

        let id = queue
            .enqueue(PriorityRequest::new("important", Priority::Critical).non_cancellable())
            .unwrap();

        // Attempting to cancel a non-cancellable request should fail
        let result = queue.cancel(&id);
        assert!(matches!(result, Err(QueueError::NotCancellable)));

        // Request should still be in the queue
        assert_eq!(queue.len(), 1);

        // Cancelling non-existent request should fail with NotFound
        let result = queue.cancel("nonexistent-id");
        assert!(matches!(result, Err(QueueError::NotFound)));
    }

    #[test]
    fn test_queue_stats_tracking() {
        let queue = PriorityQueue::new(100);

        let id1 = queue
            .enqueue(PriorityRequest::new("a", Priority::Normal))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("b", Priority::High))
            .unwrap();
        queue
            .enqueue(PriorityRequest::new("c", Priority::Low))
            .unwrap();

        // Dequeue one
        let _dequeued = queue.dequeue().unwrap();
        // Cancel one
        let _cancelled = queue.cancel(&id1).unwrap();

        let stats = queue.stats();
        assert_eq!(stats.total_enqueued, 3);
        assert_eq!(stats.total_dequeued, 1);
        assert_eq!(stats.total_cancelled, 1);
        assert_eq!(stats.current_size, 1); // 3 - 1 dequeued - 1 cancelled
    }

    #[test]
    fn test_worker_queue_and_clear() {
        let wq = WorkerQueue::new(50, 4);
        assert_eq!(wq.worker_count(), 4);

        wq.submit(PriorityRequest::new("task1", Priority::Normal))
            .unwrap();
        wq.submit(PriorityRequest::new("task2", Priority::High))
            .unwrap();
        wq.submit(PriorityRequest::new("task3", Priority::Low))
            .unwrap();

        assert_eq!(wq.queue().len(), 3);

        // take returns highest priority first
        let taken = wq.take().unwrap();
        assert_eq!(taken.priority, Priority::High);

        // Clear the queue
        wq.queue().clear();
        assert!(wq.queue().is_empty());
        assert_eq!(wq.queue().len(), 0);

        // Priority::from_value and name coverage
        assert_eq!(Priority::from_value(0), Priority::Critical);
        assert_eq!(Priority::from_value(2), Priority::Normal);
        assert_eq!(Priority::from_value(99), Priority::Background);
        assert_eq!(Priority::Normal.name(), "Normal");
        assert_eq!(Priority::Critical.name(), "Critical");
    }
}
