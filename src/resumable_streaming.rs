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
    text.len() / 4
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
        let mut state = self.state.lock().unwrap();
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
                let oldest = *state.checkpoints.keys().next().unwrap();
                state.checkpoints.remove(&oldest);
            }
        }

        // Evict old chunks if over replay buffer limit
        while state.chunks.len() > self.config.max_replay_buffer {
            let oldest = *state.chunks.keys().next().unwrap();
            state.chunks.remove(&oldest);
        }

        seq_id
    }

    /// Signal that the producer has finished.
    pub fn finish(&self) {
        let mut state = self.state.lock().unwrap();
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
        self.state.lock().unwrap().finished
    }

    /// Whether the stream is stale (no activity for `stale_timeout`).
    pub fn is_stale(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.last_activity.elapsed() > self.config.stale_timeout
    }

    /// Get all chunks after (exclusive) the given sequence ID.
    ///
    /// Used to resume from the last received chunk (e.g. SSE `Last-Event-ID`).
    pub fn resume_from(&self, last_sequence_id: u64) -> Vec<StreamChunk> {
        let state = self.state.lock().unwrap();
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
        let state = self.state.lock().unwrap();
        state.chunks.get(&sequence_id).cloned()
    }

    /// Get the latest checkpoint at or before the given sequence ID.
    pub fn checkpoint_at(&self, sequence_id: u64) -> Option<StreamCheckpoint> {
        let state = self.state.lock().unwrap();
        state
            .checkpoints
            .range(..=sequence_id)
            .next_back()
            .map(|(_, cp)| cp.clone())
    }

    /// Get the latest checkpoint.
    pub fn latest_checkpoint(&self) -> Option<StreamCheckpoint> {
        let state = self.state.lock().unwrap();
        state.checkpoints.values().next_back().cloned()
    }

    /// Get the current sequence ID (the last assigned).
    pub fn current_sequence_id(&self) -> u64 {
        self.state
            .lock()
            .unwrap()
            .next_sequence_id
            .saturating_sub(1)
    }

    /// Get the total accumulated text so far.
    pub fn accumulated_text(&self) -> String {
        self.state.lock().unwrap().accumulated_text.clone()
    }

    /// Get the number of chunks in the replay buffer.
    pub fn chunk_count(&self) -> usize {
        self.state.lock().unwrap().chunks.len()
    }

    /// Get the number of checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.state.lock().unwrap().checkpoints.len()
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
        // 100 chars ≈ 25 tokens
        let text = "a".repeat(100);
        stream.push(&text);

        let cp = stream.latest_checkpoint().unwrap();
        assert_eq!(cp.token_count, 25);
    }
}
