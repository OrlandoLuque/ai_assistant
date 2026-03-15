//! Resumable streaming with checkpoint and replay support
//!
//! Provides stream checkpointing and resume-from capabilities for
//! long-running LLM generations. Integrates with SSE via `Last-Event-ID`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A checkpoint capturing the state of a stream at a given sequence.
#[derive(Debug, Clone)]
pub struct StreamCheckpoint {
    /// Monotonically increasing sequence ID for this chunk.
    pub sequence_id: u64,
    /// Unix timestamp (milliseconds) when this checkpoint was created.
    pub timestamp_ms: u64,
    /// Accumulated text up to (and including) this chunk.
    pub accumulated_text: String,
    /// Approximate token count at this checkpoint.
    pub token_count: usize,
}

/// A single chunk in a resumable stream.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Monotonically increasing sequence ID.
    pub sequence_id: u64,
    /// The text content of this chunk.
    pub text: String,
    /// Unix timestamp (milliseconds) when this chunk was produced.
    pub timestamp_ms: u64,
}

/// Configuration for resumable streaming.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ResumableStreamConfig {
    /// How often to create a checkpoint (every N chunks).
    pub checkpoint_interval: u64,
    /// Maximum number of checkpoints to retain.
    pub max_checkpoints: usize,
    /// Maximum number of chunks to retain for replay.
    pub max_replay_buffer: usize,
    /// Timeout for considering a stream stale.
    pub stale_timeout: Duration,
}

impl Default for ResumableStreamConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10,
            max_checkpoints: 100,
            max_replay_buffer: 10000,
            stale_timeout: Duration::from_secs(300),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn estimate_tokens(text: &str) -> usize {
    crate::context::estimate_tokens(text)
}

/// State shared between producer and consumer.
#[derive(Debug)]
struct StreamState {
    /// All chunks in order.
    chunks: BTreeMap<u64, StreamChunk>,
    /// Checkpoints indexed by sequence_id.
    checkpoints: BTreeMap<u64, StreamCheckpoint>,
    /// Current accumulated text.
    accumulated_text: String,
    /// Next sequence ID to assign.
    next_sequence_id: u64,
    /// Whether the producer has finished.
    finished: bool,
    /// When the last chunk was produced.
    last_activity: Instant,
}

/// A resumable stream that wraps a chunk producer.
///
/// The producer pushes text chunks via [`push`], and consumers can
/// read chunks or resume from a given sequence ID.
pub struct ResumableStream {
    config: ResumableStreamConfig,
    state: Arc<Mutex<StreamState>>,
}

impl ResumableStream {
    /// Create a new resumable stream with the given configuration.
    pub fn new(config: ResumableStreamConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(StreamState {
                chunks: BTreeMap::new(),
                checkpoints: BTreeMap::new(),
                accumulated_text: String::new(),
                next_sequence_id: 1,
                finished: false,
                last_activity: Instant::now(),
            })),
        }
    }

    /// Push a new text chunk into the stream.
    ///
    /// Returns the assigned sequence ID.
    pub fn push(&self, text: &str) -> u64 {
        let mut state = self.state.lock().expect("stream state lock in push");
        let seq_id = state.next_sequence_id;
        state.next_sequence_id += 1;
        state.last_activity = Instant::now();

        let chunk = StreamChunk {
            sequence_id: seq_id,
            text: text.to_string(),
            timestamp_ms: now_ms(),
        };
        state.chunks.insert(seq_id, chunk);
        state.accumulated_text.push_str(text);

        // Create checkpoint at configured interval
        if seq_id % self.config.checkpoint_interval == 0 {
            let checkpoint = StreamCheckpoint {
                sequence_id: seq_id,
                timestamp_ms: now_ms(),
                accumulated_text: state.accumulated_text.clone(),
                token_count: estimate_tokens(&state.accumulated_text),
            };
            state.checkpoints.insert(seq_id, checkpoint);

            // Evict old checkpoints if over limit
            while state.checkpoints.len() > self.config.max_checkpoints {
                let oldest = *state.checkpoints.keys().next().expect("non-empty: while loop guarantees len > max");
                state.checkpoints.remove(&oldest);
            }
        }

        // Evict old chunks if over replay buffer limit
        while state.chunks.len() > self.config.max_replay_buffer {
            let oldest = *state.chunks.keys().next().expect("non-empty: while loop guarantees len > max");
            state.chunks.remove(&oldest);
        }

        seq_id
    }

    /// Signal that the producer has finished.
    pub fn finish(&self) {
        let mut state = self.state.lock().expect("stream state lock in finish");
        state.finished = true;

        // Create a final checkpoint
        let seq_id = state.next_sequence_id.saturating_sub(1);
        if seq_id > 0 {
            let checkpoint = StreamCheckpoint {
                sequence_id: seq_id,
                timestamp_ms: now_ms(),
                accumulated_text: state.accumulated_text.clone(),
                token_count: estimate_tokens(&state.accumulated_text),
            };
            state.checkpoints.insert(seq_id, checkpoint);
        }
    }

    /// Whether the producer has finished.
    pub fn is_finished(&self) -> bool {
        self.state.lock().expect("stream state lock in is_finished").finished
    }

    /// Whether the stream is stale (no activity for `stale_timeout`).
    pub fn is_stale(&self) -> bool {
        let state = self.state.lock().expect("stream state lock in is_stale");
        state.last_activity.elapsed() > self.config.stale_timeout
    }

    /// Get all chunks after (exclusive) the given sequence ID.
    ///
    /// Used to resume from the last received chunk (e.g. SSE `Last-Event-ID`).
    pub fn resume_from(&self, last_sequence_id: u64) -> Vec<StreamChunk> {
        let state = self.state.lock().expect("stream state lock in resume_from");
        state
            .chunks
            .range((
                std::ops::Bound::Excluded(last_sequence_id),
                std::ops::Bound::Unbounded,
            ))
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// Get a specific chunk by sequence ID.
    pub fn get_chunk(&self, sequence_id: u64) -> Option<StreamChunk> {
        let state = self.state.lock().expect("stream state lock in get_chunk");
        state.chunks.get(&sequence_id).cloned()
    }

    /// Get the latest checkpoint at or before the given sequence ID.
    pub fn checkpoint_at(&self, sequence_id: u64) -> Option<StreamCheckpoint> {
        let state = self.state.lock().expect("stream state lock in checkpoint_at");
        state
            .checkpoints
            .range(..=sequence_id)
            .next_back()
            .map(|(_, cp)| cp.clone())
    }

    /// Get the latest checkpoint.
    pub fn latest_checkpoint(&self) -> Option<StreamCheckpoint> {
        let state = self.state.lock().expect("stream state lock in latest_checkpoint");
        state.checkpoints.values().next_back().cloned()
    }

    /// Get the current sequence ID (the last assigned).
    pub fn current_sequence_id(&self) -> u64 {
        self.state
            .lock()
            .expect("stream state lock in current_sequence_id")
            .next_sequence_id
            .saturating_sub(1)
    }

    /// Get the total accumulated text so far.
    pub fn accumulated_text(&self) -> String {
        self.state.lock().expect("stream state lock in accumulated_text").accumulated_text.clone()
    }

    /// Get the number of chunks in the replay buffer.
    pub fn chunk_count(&self) -> usize {
        self.state.lock().expect("stream state lock in chunk_count").chunks.len()
    }

    /// Get the number of checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.state.lock().expect("stream state lock in checkpoint_count").checkpoints.len()
    }

    /// Format a chunk as an SSE event.
    ///
    /// ```text
    /// id: 42
    /// data: Hello world
    ///
    /// ```
    pub fn format_sse_event(chunk: &StreamChunk) -> String {
        format_sse_event(chunk)
    }

    /// Parse an SSE `Last-Event-ID` header value to a sequence ID.
    pub fn parse_last_event_id(header_value: &str) -> Option<u64> {
        header_value.trim().parse().ok()
    }
}

/// Format a single chunk as SSE event text.
pub fn format_sse_event(chunk: &StreamChunk) -> String {
    let mut event = String::new();
    event.push_str(&format!("id: {}\n", chunk.sequence_id));
    for line in chunk.text.lines() {
        event.push_str(&format!("data: {}\n", line));
    }
    // SSE events end with a blank line
    event.push('\n');
    event
}

// ============================================================================
// Resilient SSE Stream — Auto-reconnect with checkpoint-based resumption
// ============================================================================

/// Configuration for SSE auto-reconnection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SseReconnectConfig {
    /// Maximum number of reconnection attempts before giving up.
    pub max_attempts: u32,
    /// Initial retry delay in milliseconds.
    pub initial_retry_ms: u64,
    /// Maximum retry delay in milliseconds.
    pub max_retry_ms: u64,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
    /// Whether to honor the server's `retry:` field.
    pub respect_server_retry: bool,
}

impl Default for SseReconnectConfig {
    fn default() -> Self {
        Self {
            max_attempts: 10,
            initial_retry_ms: 1000,
            max_retry_ms: 60_000,
            backoff_multiplier: 2.0,
            respect_server_retry: true,
        }
    }
}

impl SseReconnectConfig {
    /// Aggressive: many attempts, shorter initial delay.
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 20,
            initial_retry_ms: 500,
            max_retry_ms: 120_000,
            backoff_multiplier: 2.0,
            respect_server_retry: true,
        }
    }

    /// Quick: few attempts, short delays.
    pub fn quick() -> Self {
        Self {
            max_attempts: 3,
            initial_retry_ms: 200,
            max_retry_ms: 5000,
            backoff_multiplier: 2.0,
            respect_server_retry: true,
        }
    }
}

/// Statistics for SSE reconnection.
#[derive(Debug, Clone, Default)]
pub struct SseReconnectStats {
    /// The last event ID received before disconnect.
    pub last_event_id: Option<u64>,
    /// Number of reconnection attempts in current cycle.
    pub reconnect_attempts: u32,
    /// Total successful reconnections over the lifetime.
    pub total_reconnects: u64,
    /// Total chunks replayed after reconnections.
    pub chunks_replayed: u64,
}

/// A resilient SSE stream that wraps [`ResumableStream`] with
/// auto-reconnection and checkpoint-based resumption.
///
/// On disconnect, it saves the `last_event_id` (current sequence ID).
/// On reconnection, it uses `resume_from()` to replay missed chunks.
///
/// # Usage Pattern
/// ```ignore
/// let mut sse = ResilientSseStream::new(
///     ResumableStreamConfig::default(),
///     SseReconnectConfig::default(),
/// );
///
/// // On disconnect:
/// sse.handle_disconnect();
///
/// // Reconnection loop:
/// while sse.should_reconnect() {
///     std::thread::sleep(sse.next_retry_delay());
///     match try_reconnect_sse() {
///         Ok(_) => {
///             let missed = sse.get_resume_chunks();
///             sse.mark_reconnected();
///             // Re-send missed chunks to the consumer
///         }
///         Err(_) => { sse.mark_attempt_failed(); }
///     }
/// }
/// ```
pub struct ResilientSseStream {
    stream: ResumableStream,
    config: SseReconnectConfig,
    last_event_id: Option<u64>,
    reconnect_attempts: u32,
    total_reconnects: u64,
    server_retry_ms: Option<u64>,
    chunks_replayed: u64,
    current_retry_ms: u64,
    gave_up: bool,
    on_reconnect: Option<Box<dyn Fn(u64, u32) + Send + Sync>>,
    on_chunks_lost: Option<Box<dyn Fn(u64) + Send + Sync>>,
}

impl ResilientSseStream {
    /// Create a new resilient SSE stream.
    pub fn new(stream_config: ResumableStreamConfig, reconnect_config: SseReconnectConfig) -> Self {
        let initial_ms = reconnect_config.initial_retry_ms;
        Self {
            stream: ResumableStream::new(stream_config),
            config: reconnect_config,
            last_event_id: None,
            reconnect_attempts: 0,
            total_reconnects: 0,
            server_retry_ms: None,
            chunks_replayed: 0,
            current_retry_ms: initial_ms,
            gave_up: false,
            on_reconnect: None,
            on_chunks_lost: None,
        }
    }

    /// Access the underlying resumable stream.
    pub fn stream(&self) -> &ResumableStream {
        &self.stream
    }

    /// Access the underlying resumable stream mutably.
    pub fn stream_mut(&mut self) -> &mut ResumableStream {
        &mut self.stream
    }

    /// Push a chunk to the underlying stream.
    pub fn push(&self, text: &str) -> u64 {
        self.stream.push(text)
    }

    /// Handle a disconnect: save current sequence as last_event_id.
    pub fn handle_disconnect(&mut self) {
        self.last_event_id = Some(self.stream.current_sequence_id());
        self.reconnect_attempts = 0;
        self.current_retry_ms = self.config.initial_retry_ms;
        self.gave_up = false;
    }

    /// Get chunks that need to be replayed after reconnection.
    ///
    /// Calls `ResumableStream::resume_from(last_event_id)`.
    /// Returns an empty vec if no last_event_id is set.
    pub fn get_resume_chunks(&self) -> Vec<StreamChunk> {
        match self.last_event_id {
            Some(id) => self.stream.resume_from(id),
            None => Vec::new(),
        }
    }

    /// Mark a reconnection as successful.
    pub fn mark_reconnected(&mut self) {
        let replayed = self.get_resume_chunks().len() as u64;
        self.chunks_replayed += replayed;
        self.total_reconnects += 1;
        let last_id = self.last_event_id.unwrap_or(0);
        let attempts = self.reconnect_attempts;
        self.reconnect_attempts = 0;
        self.current_retry_ms = self.config.initial_retry_ms;

        if let Some(ref cb) = self.on_reconnect {
            cb(last_id, attempts);
        }

        self.last_event_id = None;
    }

    /// Mark a reconnection attempt as failed.
    pub fn mark_attempt_failed(&mut self) {
        self.reconnect_attempts += 1;

        if self.reconnect_attempts >= self.config.max_attempts {
            self.gave_up = true;
            // Check if chunks were lost (replay buffer may have evicted them)
            if let Some(ref cb) = self.on_chunks_lost {
                let lost = self.estimate_lost_chunks();
                if lost > 0 {
                    cb(lost);
                }
            }
        } else {
            // Exponential backoff
            self.current_retry_ms = ((self.current_retry_ms as f64
                * self.config.backoff_multiplier) as u64)
                .min(self.config.max_retry_ms);
        }
    }

    /// Returns true if reconnection should be attempted.
    pub fn should_reconnect(&self) -> bool {
        !self.gave_up && self.last_event_id.is_some()
    }

    /// Calculate the delay before the next retry.
    pub fn next_retry_delay(&self) -> Duration {
        if self.config.respect_server_retry {
            if let Some(server_ms) = self.server_retry_ms {
                return Duration::from_millis(server_ms);
            }
        }
        Duration::from_millis(self.current_retry_ms)
    }

    /// Set the server-indicated retry delay from SSE `retry:` field.
    pub fn set_server_retry(&mut self, ms: u64) {
        self.server_retry_ms = Some(ms);
    }

    /// Set callback invoked on successful reconnection.
    /// Receives (last_event_id, attempt_count).
    pub fn on_reconnect<F: Fn(u64, u32) + Send + Sync + 'static>(&mut self, f: F) {
        self.on_reconnect = Some(Box::new(f));
    }

    /// Set callback invoked when chunks are estimated to be lost.
    /// Receives the estimated count of lost chunks.
    pub fn on_chunks_lost<F: Fn(u64) + Send + Sync + 'static>(&mut self, f: F) {
        self.on_chunks_lost = Some(Box::new(f));
    }

    /// Get reconnection statistics.
    pub fn stats(&self) -> SseReconnectStats {
        SseReconnectStats {
            last_event_id: self.last_event_id,
            reconnect_attempts: self.reconnect_attempts,
            total_reconnects: self.total_reconnects,
            chunks_replayed: self.chunks_replayed,
        }
    }

    /// Estimate chunks lost due to replay buffer eviction.
    fn estimate_lost_chunks(&self) -> u64 {
        if let Some(last_id) = self.last_event_id {
            let current = self.stream.current_sequence_id();
            let available = self.stream.resume_from(last_id).len() as u64;
            let expected = current.saturating_sub(last_id);
            expected.saturating_sub(available)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_and_read() {
        let stream = ResumableStream::new(ResumableStreamConfig::default());
        let id1 = stream.push("Hello ");
        let id2 = stream.push("world!");

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(stream.accumulated_text(), "Hello world!");
        assert_eq!(stream.chunk_count(), 2);
    }

    #[test]
    fn test_resume_from() {
        let stream = ResumableStream::new(ResumableStreamConfig::default());
        stream.push("A");
        stream.push("B");
        stream.push("C");
        stream.push("D");

        // Resume from sequence 2 — should get C, D
        let resumed = stream.resume_from(2);
        assert_eq!(resumed.len(), 2);
        assert_eq!(resumed[0].text, "C");
        assert_eq!(resumed[1].text, "D");
    }

    #[test]
    fn test_resume_from_zero() {
        let stream = ResumableStream::new(ResumableStreamConfig::default());
        stream.push("A");
        stream.push("B");

        // Resume from 0 — should get all chunks
        let resumed = stream.resume_from(0);
        assert_eq!(resumed.len(), 2);
    }

    #[test]
    fn test_checkpoint_creation() {
        let config = ResumableStreamConfig {
            checkpoint_interval: 3,
            ..Default::default()
        };
        let stream = ResumableStream::new(config);

        for i in 0..9 {
            stream.push(&format!("chunk{} ", i));
        }

        // Checkpoints at sequences 3, 6, 9
        assert_eq!(stream.checkpoint_count(), 3);

        let cp = stream.checkpoint_at(6).unwrap();
        assert_eq!(cp.sequence_id, 6);
        assert!(cp.accumulated_text.contains("chunk5"));
    }

    #[test]
    fn test_checkpoint_at_before() {
        let config = ResumableStreamConfig {
            checkpoint_interval: 5,
            ..Default::default()
        };
        let stream = ResumableStream::new(config);
        for i in 0..12 {
            stream.push(&format!("{}", i));
        }

        // Checkpoints at 5, 10
        let cp = stream.checkpoint_at(7).unwrap();
        assert_eq!(cp.sequence_id, 5);

        let cp = stream.checkpoint_at(11).unwrap();
        assert_eq!(cp.sequence_id, 10);
    }

    #[test]
    fn test_finish_creates_final_checkpoint() {
        let config = ResumableStreamConfig {
            checkpoint_interval: 100, // high so normal checkpoints don't fire
            ..Default::default()
        };
        let stream = ResumableStream::new(config);
        stream.push("Hello");
        stream.push(" World");
        stream.finish();

        assert!(stream.is_finished());
        let cp = stream.latest_checkpoint().unwrap();
        assert_eq!(cp.sequence_id, 2);
        assert_eq!(cp.accumulated_text, "Hello World");
    }

    #[test]
    fn test_sequence_ordering() {
        let stream = ResumableStream::new(ResumableStreamConfig::default());
        let ids: Vec<u64> = (0..20).map(|i| stream.push(&format!("{}", i))).collect();

        for i in 1..ids.len() {
            assert!(ids[i] > ids[i - 1], "IDs must be strictly increasing");
        }
    }

    #[test]
    fn test_get_chunk() {
        let stream = ResumableStream::new(ResumableStreamConfig::default());
        stream.push("first");
        stream.push("second");
        stream.push("third");

        let chunk = stream.get_chunk(2).unwrap();
        assert_eq!(chunk.text, "second");

        assert!(stream.get_chunk(999).is_none());
    }

    #[test]
    fn test_max_replay_buffer_eviction() {
        let config = ResumableStreamConfig {
            max_replay_buffer: 5,
            ..Default::default()
        };
        let stream = ResumableStream::new(config);

        for i in 0..10 {
            stream.push(&format!("{}", i));
        }

        assert_eq!(stream.chunk_count(), 5);
        // Oldest chunks should be evicted
        assert!(stream.get_chunk(1).is_none());
        assert!(stream.get_chunk(5).is_none());
        assert!(stream.get_chunk(6).is_some());
    }

    #[test]
    fn test_max_checkpoints_eviction() {
        let config = ResumableStreamConfig {
            checkpoint_interval: 1,
            max_checkpoints: 3,
            ..Default::default()
        };
        let stream = ResumableStream::new(config);

        for i in 0..10 {
            stream.push(&format!("{}", i));
        }

        assert!(stream.checkpoint_count() <= 3);
    }

    #[test]
    fn test_stale_detection() {
        let config = ResumableStreamConfig {
            stale_timeout: Duration::from_millis(1),
            ..Default::default()
        };
        let stream = ResumableStream::new(config);
        stream.push("data");

        std::thread::sleep(Duration::from_millis(10));
        assert!(stream.is_stale());
    }

    #[test]
    fn test_sse_format() {
        let chunk = StreamChunk {
            sequence_id: 42,
            text: "Hello".to_string(),
            timestamp_ms: 1000,
        };
        let sse = format_sse_event(&chunk);
        assert!(sse.contains("id: 42"));
        assert!(sse.contains("data: Hello"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_parse_last_event_id() {
        assert_eq!(ResumableStream::parse_last_event_id("42"), Some(42));
        assert_eq!(ResumableStream::parse_last_event_id(" 100 "), Some(100));
        assert_eq!(ResumableStream::parse_last_event_id("abc"), None);
    }

    #[test]
    fn test_token_count_in_checkpoint() {
        let config = ResumableStreamConfig {
            checkpoint_interval: 1,
            ..Default::default()
        };
        let stream = ResumableStream::new(config);
        // 100 chars ≈ 29 tokens (ceil(100/3.5))
        let text = "a".repeat(100);
        stream.push(&text);

        let cp = stream.latest_checkpoint().unwrap();
        assert_eq!(cp.token_count, 29);
    }

    // ---- ResilientSseStream tests ----

    #[test]
    fn test_sse_reconnect_initial_state() {
        let sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        assert!(!sse.should_reconnect());
        assert_eq!(sse.stats().total_reconnects, 0);
    }

    #[test]
    fn test_sse_reconnect_disconnect_saves_id() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        sse.push("chunk1");
        sse.push("chunk2");
        sse.push("chunk3");

        sse.handle_disconnect();

        assert_eq!(sse.stats().last_event_id, Some(3));
        assert!(sse.should_reconnect());
    }

    #[test]
    fn test_sse_reconnect_resume_chunks() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        sse.push("A");
        sse.push("B");
        sse.push("C");

        // Simulate disconnect after chunk 2
        sse.last_event_id = Some(2);

        let resumed = sse.get_resume_chunks();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].text, "C");
    }

    #[test]
    fn test_sse_reconnect_mark_reconnected() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        sse.push("data");
        sse.handle_disconnect();
        sse.mark_attempt_failed();
        sse.mark_reconnected();

        assert!(!sse.should_reconnect());
        assert_eq!(sse.stats().total_reconnects, 1);
    }

    #[test]
    fn test_sse_reconnect_max_attempts_gives_up() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::quick(), // max 3
        );
        sse.push("data");
        sse.handle_disconnect();

        for _ in 0..3 {
            sse.mark_attempt_failed();
        }

        assert!(!sse.should_reconnect());
    }

    #[test]
    fn test_sse_reconnect_server_retry_override() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig {
                respect_server_retry: true,
                initial_retry_ms: 1000,
                ..SseReconnectConfig::default()
            },
        );

        sse.set_server_retry(3000);
        let delay = sse.next_retry_delay();
        assert_eq!(delay, Duration::from_millis(3000));
    }

    #[test]
    fn test_sse_reconnect_backoff_increases() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig {
                initial_retry_ms: 100,
                max_retry_ms: 10_000,
                backoff_multiplier: 2.0,
                respect_server_retry: false,
                ..SseReconnectConfig::default()
            },
        );

        sse.push("data");
        sse.handle_disconnect();

        let delay1 = sse.next_retry_delay();
        sse.mark_attempt_failed();
        let delay2 = sse.next_retry_delay();
        sse.mark_attempt_failed();
        let delay3 = sse.next_retry_delay();

        assert!(delay2 > delay1);
        assert!(delay3 > delay2);
    }

    #[test]
    fn test_sse_reconnect_stats() {
        let mut sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        sse.push("data1");
        sse.push("data2");
        sse.handle_disconnect();

        let stats = sse.stats();
        assert_eq!(stats.last_event_id, Some(2));
        assert_eq!(stats.reconnect_attempts, 0);
        assert_eq!(stats.total_reconnects, 0);
    }

    #[test]
    fn test_sse_reconnect_config_presets() {
        let default = SseReconnectConfig::default();
        assert_eq!(default.max_attempts, 10);

        let aggressive = SseReconnectConfig::aggressive();
        assert_eq!(aggressive.max_attempts, 20);

        let quick = SseReconnectConfig::quick();
        assert_eq!(quick.max_attempts, 3);
    }

    #[test]
    fn test_sse_stream_push_delegates() {
        let sse = ResilientSseStream::new(
            ResumableStreamConfig::default(),
            SseReconnectConfig::default(),
        );
        let id1 = sse.push("hello");
        let id2 = sse.push("world");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(sse.stream().accumulated_text(), "helloworld");
    }
}
