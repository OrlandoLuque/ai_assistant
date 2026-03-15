//! Enhanced streaming with backpressure support
//!
//! This module provides advanced streaming capabilities with flow control
//! to prevent memory exhaustion during long-running generations.
//!
//! # Features
//!
//! - **Backpressure**: Automatic flow control when consumer is slow
//! - **Buffering**: Configurable buffer sizes for smooth streaming
//! - **Chunking**: Efficient chunk-based processing
//! - **Metrics**: Real-time streaming statistics
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_assistant::streaming::{StreamingConfig, BackpressureStream, StreamBuffer};
//!
//! let config = StreamingConfig::default();
//! let mut buffer = StreamBuffer::new(config.buffer_size);
//!
//! // Producer adds chunks
//! buffer.push("Hello ".to_string());
//! buffer.push("World!".to_string());
//!
//! // Consumer reads with backpressure
//! while let Some(chunk) = buffer.try_pop() {
//!     println!("{}", chunk);
//! }
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Configuration for streaming behavior
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamingConfig {
    /// Maximum buffer size in bytes before applying backpressure
    pub buffer_size: usize,
    /// High water mark - start slowing down at this level
    pub high_water_mark: usize,
    /// Low water mark - resume normal speed at this level
    pub low_water_mark: usize,
    /// Maximum time to wait when buffer is full
    pub backpressure_timeout: Duration,
    /// Chunk size for optimal processing
    pub chunk_size: usize,
    /// Enable automatic chunking of large responses
    pub auto_chunk: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,     // 64KB
            high_water_mark: 48 * 1024, // 75%
            low_water_mark: 16 * 1024,  // 25%
            backpressure_timeout: Duration::from_secs(30),
            chunk_size: 4096,
            auto_chunk: true,
        }
    }
}

/// Streaming buffer with backpressure support
///
/// Uses manual Debug impl because inner buffer contains Condvar which doesn't implement Debug.
pub struct StreamBuffer {
    inner: Arc<StreamBufferInner>,
}

struct StreamBufferInner {
    data: Mutex<StreamBufferData>,
    not_full: Condvar,
    not_empty: Condvar,
}

struct StreamBufferData {
    chunks: VecDeque<String>,
    current_size: usize,
    max_size: usize,
    high_water: usize,
    low_water: usize,
    closed: bool,
    metrics: StreamMetrics,
}

impl StreamBuffer {
    /// Create a new stream buffer with default configuration
    pub fn new(max_size: usize) -> Self {
        Self::with_config(StreamingConfig {
            buffer_size: max_size,
            high_water_mark: max_size * 3 / 4,
            low_water_mark: max_size / 4,
            ..Default::default()
        })
    }

    /// Create with full configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            inner: Arc::new(StreamBufferInner {
                data: Mutex::new(StreamBufferData {
                    chunks: VecDeque::new(),
                    current_size: 0,
                    max_size: config.buffer_size,
                    high_water: config.high_water_mark,
                    low_water: config.low_water_mark,
                    closed: false,
                    metrics: StreamMetrics::new(),
                }),
                not_full: Condvar::new(),
                not_empty: Condvar::new(),
            }),
        }
    }

    /// Push a chunk to the buffer, blocking if full
    pub fn push(&self, chunk: String) -> Result<(), StreamError> {
        self.push_timeout(chunk, Duration::from_secs(30))
    }

    /// Push with timeout
    pub fn push_timeout(&self, chunk: String, timeout: Duration) -> Result<(), StreamError> {
        let chunk_size = chunk.len();
        let mut data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());

        // Wait if buffer is full
        let deadline = Instant::now() + timeout;
        while data.current_size + chunk_size > data.max_size && !data.closed {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(StreamError::Timeout);
            }

            let result = self
                .inner
                .not_full
                .wait_timeout(data, remaining)
                .unwrap_or_else(|e| e.into_inner());
            data = result.0;

            if result.1.timed_out() {
                return Err(StreamError::Timeout);
            }
        }

        if data.closed {
            return Err(StreamError::Closed);
        }

        // Add chunk
        data.current_size += chunk_size;
        data.metrics.bytes_produced += chunk_size;
        data.metrics.chunks_produced += 1;
        data.chunks.push_back(chunk);

        // Notify waiting consumers
        self.inner.not_empty.notify_one();

        Ok(())
    }

    /// Try to push without blocking
    pub fn try_push(&self, chunk: String) -> Result<(), StreamError> {
        let chunk_size = chunk.len();
        let mut data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());

        if data.closed {
            return Err(StreamError::Closed);
        }

        if data.current_size + chunk_size > data.max_size {
            return Err(StreamError::BufferFull);
        }

        data.current_size += chunk_size;
        data.metrics.bytes_produced += chunk_size;
        data.metrics.chunks_produced += 1;
        data.chunks.push_back(chunk);

        self.inner.not_empty.notify_one();
        Ok(())
    }

    /// Pop a chunk from the buffer, blocking if empty
    pub fn pop(&self) -> Option<String> {
        self.pop_timeout(Duration::from_secs(30))
    }

    /// Pop with timeout
    pub fn pop_timeout(&self, timeout: Duration) -> Option<String> {
        let mut data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());

        let deadline = Instant::now() + timeout;
        while data.chunks.is_empty() && !data.closed {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }

            let result = self
                .inner
                .not_empty
                .wait_timeout(data, remaining)
                .unwrap_or_else(|e| e.into_inner());
            data = result.0;

            if result.1.timed_out() {
                return None;
            }
        }

        self.pop_internal(&mut data)
    }

    /// Try to pop without blocking
    pub fn try_pop(&self) -> Option<String> {
        let mut data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());
        self.pop_internal(&mut data)
    }

    fn pop_internal(&self, data: &mut StreamBufferData) -> Option<String> {
        if let Some(chunk) = data.chunks.pop_front() {
            let chunk_size = chunk.len();
            data.current_size -= chunk_size;
            data.metrics.bytes_consumed += chunk_size;
            data.metrics.chunks_consumed += 1;

            // Notify if we went below low water mark
            if data.current_size <= data.low_water {
                self.inner.not_full.notify_all();
            }

            Some(chunk)
        } else {
            None
        }
    }

    /// Close the buffer (no more pushes allowed)
    pub fn close(&self) {
        let mut data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());
        data.closed = true;
        self.inner.not_empty.notify_all();
        self.inner.not_full.notify_all();
    }

    /// Check if buffer is closed
    pub fn is_closed(&self) -> bool {
        self.inner
            .data
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closed
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.inner
            .data
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .current_size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        let data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());
        data.current_size >= data.high_water
    }

    /// Get streaming metrics
    pub fn metrics(&self) -> StreamMetrics {
        self.inner
            .data
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .metrics
            .clone()
    }

    /// Get fill percentage
    pub fn fill_percentage(&self) -> f32 {
        let data = self.inner.data.lock().unwrap_or_else(|e| e.into_inner());
        (data.current_size as f32 / data.max_size as f32) * 100.0
    }
}

impl Clone for StreamBuffer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for StreamBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamBuffer")
            .field("len", &self.len())
            .field("is_closed", &self.is_closed())
            .finish()
    }
}

/// Errors during streaming
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamError {
    /// Buffer is full
    BufferFull,
    /// Stream is closed
    Closed,
    /// Operation timed out
    Timeout,
    /// Producer cancelled
    Cancelled,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::BufferFull => write!(f, "Stream buffer is full"),
            StreamError::Closed => write!(f, "Stream is closed"),
            StreamError::Timeout => write!(f, "Stream operation timed out"),
            StreamError::Cancelled => write!(f, "Stream was cancelled"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Metrics for streaming operations
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    /// Total bytes produced
    pub bytes_produced: usize,
    /// Total bytes consumed
    pub bytes_consumed: usize,
    /// Total chunks produced
    pub chunks_produced: usize,
    /// Total chunks consumed
    pub chunks_consumed: usize,
    /// Number of backpressure events
    pub backpressure_events: usize,
    /// Start time
    start_time: Option<Instant>,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self {
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    /// Get throughput in bytes per second
    pub fn throughput_bps(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.bytes_consumed as f64 / elapsed;
            }
        }
        0.0
    }

    /// Get average chunk size
    pub fn avg_chunk_size(&self) -> usize {
        if self.chunks_produced > 0 {
            self.bytes_produced / self.chunks_produced
        } else {
            0
        }
    }
}

/// A stream that applies backpressure
pub struct BackpressureStream<T> {
    buffer: StreamBuffer,
    config: StreamingConfig,
    _marker: std::marker::PhantomData<T>,
}

impl BackpressureStream<String> {
    /// Create a new backpressure stream
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            buffer: StreamBuffer::with_config(config.clone()),
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get a producer handle
    pub fn producer(&self) -> StreamProducer {
        StreamProducer {
            buffer: self.buffer.clone(),
            chunk_size: self.config.chunk_size,
            auto_chunk: self.config.auto_chunk,
        }
    }

    /// Get a consumer handle
    pub fn consumer(&self) -> StreamConsumer {
        StreamConsumer {
            buffer: self.buffer.clone(),
        }
    }

    /// Close the stream
    pub fn close(&self) {
        self.buffer.close();
    }

    /// Get metrics
    pub fn metrics(&self) -> StreamMetrics {
        self.buffer.metrics()
    }
}

impl<T> std::fmt::Debug for BackpressureStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackpressureStream")
            .field("buffer", &self.buffer)
            .field("config", &self.config)
            .finish()
    }
}

/// Producer side of the stream
#[derive(Debug)]
pub struct StreamProducer {
    buffer: StreamBuffer,
    chunk_size: usize,
    auto_chunk: bool,
}

impl StreamProducer {
    /// Send data to the stream
    pub fn send(&self, data: String) -> Result<(), StreamError> {
        if self.auto_chunk && data.len() > self.chunk_size {
            // Split into chunks
            for chunk in data.as_bytes().chunks(self.chunk_size) {
                let s = String::from_utf8_lossy(chunk).to_string();
                self.buffer.push(s)?;
            }
            Ok(())
        } else {
            self.buffer.push(data)
        }
    }

    /// Try to send without blocking
    pub fn try_send(&self, data: String) -> Result<(), StreamError> {
        self.buffer.try_push(data)
    }

    /// Close the producer side
    pub fn close(&self) {
        self.buffer.close();
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.buffer.is_backpressure_active()
    }
}

/// Consumer side of the stream
#[derive(Debug)]
pub struct StreamConsumer {
    buffer: StreamBuffer,
}

impl StreamConsumer {
    /// Receive data from the stream
    pub fn recv(&self) -> Option<String> {
        self.buffer.pop()
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Option<String> {
        self.buffer.try_pop()
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Option<String> {
        self.buffer.pop_timeout(timeout)
    }

    /// Check if stream is closed and empty
    pub fn is_done(&self) -> bool {
        self.buffer.is_closed() && self.buffer.is_empty()
    }

    /// Collect all remaining chunks
    pub fn collect_all(&self) -> String {
        let mut result = String::new();
        while let Some(chunk) = self.try_recv() {
            result.push_str(&chunk);
        }
        result
    }
}

impl Iterator for StreamConsumer {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.recv()
    }
}

/// Chunker for splitting large strings efficiently
#[derive(Debug)]
pub struct Chunker {
    chunk_size: usize,
    preserve_words: bool,
}

impl Chunker {
    /// Create a new chunker
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            preserve_words: true,
        }
    }

    /// Disable word preservation for exact chunking
    pub fn exact(mut self) -> Self {
        self.preserve_words = false;
        self
    }

    /// Chunk a string
    pub fn chunk(&self, input: &str) -> Vec<String> {
        if input.len() <= self.chunk_size {
            return vec![input.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current = String::new();

        if self.preserve_words {
            for word in input.split_inclusive(char::is_whitespace) {
                if current.len() + word.len() > self.chunk_size && !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                current.push_str(word);
            }
        } else {
            for ch in input.chars() {
                if current.len() >= self.chunk_size {
                    chunks.push(std::mem::take(&mut current));
                }
                current.push(ch);
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }
}

/// Rate-limited stream wrapper
#[derive(Debug)]
pub struct RateLimitedStream {
    buffer: StreamBuffer,
    tokens_per_second: f64,
    last_emit: Mutex<Instant>,
}

impl RateLimitedStream {
    /// Create a new rate-limited stream
    pub fn new(buffer: StreamBuffer, tokens_per_second: f64) -> Self {
        Self {
            buffer,
            tokens_per_second,
            last_emit: Mutex::new(Instant::now()),
        }
    }

    /// Get next chunk with rate limiting
    pub fn next_rate_limited(&self) -> Option<String> {
        let chunk = self.buffer.try_pop()?;

        // Calculate delay based on chunk size
        let tokens = chunk.split_whitespace().count();
        let delay_secs = tokens as f64 / self.tokens_per_second;

        let mut last = self.last_emit.lock().unwrap_or_else(|e| e.into_inner());
        let elapsed = last.elapsed().as_secs_f64();

        if elapsed < delay_secs {
            std::thread::sleep(Duration::from_secs_f64(delay_secs - elapsed));
        }

        *last = Instant::now();
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_stream_buffer_basic() {
        let buffer = StreamBuffer::new(1024);

        buffer.try_push("Hello".to_string()).unwrap();
        buffer.try_push(" World".to_string()).unwrap();

        assert_eq!(buffer.try_pop(), Some("Hello".to_string()));
        assert_eq!(buffer.try_pop(), Some(" World".to_string()));
        assert_eq!(buffer.try_pop(), None);
    }

    #[test]
    fn test_stream_buffer_backpressure() {
        let buffer = StreamBuffer::new(10);

        // Fill buffer
        buffer.try_push("12345".to_string()).unwrap();
        buffer.try_push("12345".to_string()).unwrap();

        // Should fail - buffer full
        assert!(matches!(
            buffer.try_push("more".to_string()),
            Err(StreamError::BufferFull)
        ));

        // Consume some
        buffer.try_pop();

        // Should work now
        assert!(buffer.try_push("ok".to_string()).is_ok());
    }

    #[test]
    fn test_backpressure_stream() {
        let config = StreamingConfig {
            buffer_size: 1024,
            ..Default::default()
        };
        let stream = BackpressureStream::new(config);

        let producer = stream.producer();
        let consumer = stream.consumer();

        producer.send("Hello World!".to_string()).unwrap();
        stream.close();

        let received = consumer.collect_all();
        assert_eq!(received, "Hello World!");
    }

    #[test]
    fn test_chunker() {
        let chunker = Chunker::new(10);
        let chunks = chunker.chunk("Hello World, how are you today?");

        assert!(chunks.len() > 1);
        let reassembled: String = chunks.concat();
        assert_eq!(reassembled, "Hello World, how are you today?");
    }

    #[test]
    fn test_stream_metrics() {
        let buffer = StreamBuffer::new(1024);

        buffer.try_push("Hello".to_string()).unwrap();
        buffer.try_push("World".to_string()).unwrap();
        buffer.try_pop();

        let metrics = buffer.metrics();
        assert_eq!(metrics.chunks_produced, 2);
        assert_eq!(metrics.chunks_consumed, 1);
        assert_eq!(metrics.bytes_produced, 10);
        assert_eq!(metrics.bytes_consumed, 5);
    }

    #[test]
    fn test_concurrent_access() {
        let buffer = StreamBuffer::new(10240);
        let buffer_clone = buffer.clone();

        let producer = thread::spawn(move || {
            for i in 0..100 {
                buffer_clone.push(format!("msg{}", i)).unwrap();
            }
            buffer_clone.close();
        });

        let mut received = Vec::new();
        while !buffer.is_closed() || !buffer.is_empty() {
            if let Some(chunk) = buffer.try_pop() {
                received.push(chunk);
            }
            thread::yield_now();
        }

        producer.join().unwrap();
        assert_eq!(received.len(), 100);
    }

    #[test]
    fn test_push_after_close_returns_closed_error() {
        let buffer = StreamBuffer::new(1024);

        // Push succeeds before close
        assert!(buffer.try_push("before".to_string()).is_ok());
        assert!(!buffer.is_closed());

        // Close the buffer
        buffer.close();
        assert!(buffer.is_closed());

        // Push should return Closed error
        assert_eq!(
            buffer.try_push("after".to_string()),
            Err(StreamError::Closed)
        );

        // Existing data should still be readable
        assert_eq!(buffer.try_pop(), Some("before".to_string()));
        assert_eq!(buffer.try_pop(), None);
    }

    #[test]
    fn test_chunker_exact_mode() {
        let chunker = Chunker::new(5).exact();

        let chunks = chunker.chunk("HelloWorld!");
        // Exact mode splits at exactly 5 chars: "Hello", "World", "!"
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Hello");
        assert_eq!(chunks[1], "World");
        assert_eq!(chunks[2], "!");

        // Input shorter than chunk_size returns single chunk
        let small = chunker.chunk("Hi");
        assert_eq!(small.len(), 1);
        assert_eq!(small[0], "Hi");

        // Empty input
        let empty = chunker.chunk("");
        assert_eq!(empty.len(), 1);
        assert_eq!(empty[0], "");
    }

    #[test]
    fn test_fill_percentage_and_backpressure_active() {
        // Buffer of 100 bytes with 75/25 water marks
        let config = StreamingConfig {
            buffer_size: 100,
            high_water_mark: 75,
            low_water_mark: 25,
            ..Default::default()
        };
        let buffer = StreamBuffer::with_config(config);

        // Empty buffer: 0% fill, no backpressure
        assert!((buffer.fill_percentage() - 0.0).abs() < 0.01);
        assert!(!buffer.is_backpressure_active());

        // Push 50 bytes (50%) - below high water mark
        buffer.try_push("a".repeat(50)).unwrap();
        assert!((buffer.fill_percentage() - 50.0).abs() < 0.01);
        assert!(!buffer.is_backpressure_active());

        // Push 30 more bytes (80%) - above high water mark (75)
        buffer.try_push("b".repeat(30)).unwrap();
        assert!((buffer.fill_percentage() - 80.0).abs() < 0.01);
        assert!(buffer.is_backpressure_active());
    }

    #[test]
    fn test_stream_consumer_is_done_and_error_display() {
        let config = StreamingConfig {
            buffer_size: 1024,
            ..Default::default()
        };
        let stream = BackpressureStream::new(config);
        let producer = stream.producer();
        let consumer = stream.consumer();

        // Not done: stream is open
        assert!(!consumer.is_done());

        // Send data
        producer.try_send("chunk1".to_string()).unwrap();
        producer.try_send("chunk2".to_string()).unwrap();

        // Not done: stream is open with data
        assert!(!consumer.is_done());

        // Close producer side
        producer.close();

        // Not done yet: stream is closed but has data
        assert!(!consumer.is_done());

        // Drain data
        assert_eq!(consumer.try_recv(), Some("chunk1".to_string()));
        assert_eq!(consumer.try_recv(), Some("chunk2".to_string()));

        // Now done: closed and empty
        assert!(consumer.is_done());

        // Verify StreamError Display implementations
        assert_eq!(StreamError::BufferFull.to_string(), "Stream buffer is full");
        assert_eq!(StreamError::Closed.to_string(), "Stream is closed");
        assert_eq!(StreamError::Timeout.to_string(), "Stream operation timed out");
        assert_eq!(StreamError::Cancelled.to_string(), "Stream was cancelled");
    }
}
