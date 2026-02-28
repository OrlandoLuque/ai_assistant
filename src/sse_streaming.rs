//! Server-Sent Events (SSE) streaming support
//!
//! Implements SSE protocol for real-time streaming responses from AI models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

/// SSE Event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

impl SseEvent {
    pub fn new(data: &str) -> Self {
        Self {
            event: None,
            data: data.to_string(),
            id: None,
            retry: None,
        }
    }

    pub fn with_event(mut self, event: &str) -> Self {
        self.event = Some(event.to_string());
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_retry(mut self, retry: u64) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Serialize to SSE wire format
    pub fn to_wire_format(&self) -> String {
        let mut output = String::new();

        if let Some(ref event) = self.event {
            output.push_str(&format!("event: {}\n", event));
        }

        if let Some(ref id) = self.id {
            output.push_str(&format!("id: {}\n", id));
        }

        if let Some(retry) = self.retry {
            output.push_str(&format!("retry: {}\n", retry));
        }

        // Handle multi-line data
        for line in self.data.lines() {
            output.push_str(&format!("data: {}\n", line));
        }

        output.push('\n'); // End of event
        output
    }

    /// Parse from SSE wire format
    pub fn from_wire_format(text: &str) -> Option<Self> {
        let mut event = None;
        let mut data_lines = Vec::new();
        let mut id = None;
        let mut retry = None;

        for line in text.lines() {
            if line.starts_with("event:") {
                event = Some(line[6..].trim().to_string());
            } else if line.starts_with("data:") {
                data_lines.push(line[5..].trim().to_string());
            } else if line.starts_with("id:") {
                id = Some(line[3..].trim().to_string());
            } else if line.starts_with("retry:") {
                retry = line[6..].trim().parse().ok();
            }
        }

        if data_lines.is_empty() {
            return None;
        }

        Some(Self {
            event,
            data: data_lines.join("\n"),
            id,
            retry,
        })
    }
}

/// SSE Stream reader
#[derive(Debug)]
pub struct SseReader<R: Read> {
    reader: BufReader<R>,
    last_event_id: Option<String>,
    buffer: String,
}

impl<R: Read> SseReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            last_event_id: None,
            buffer: String::new(),
        }
    }

    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }
}

impl<R: Read> Iterator for SseReader<R> {
    type Item = Result<SseEvent, SseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        let mut has_data = false;

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF
                    if has_data {
                        if let Some(event) = SseEvent::from_wire_format(&self.buffer) {
                            if let Some(ref id) = event.id {
                                self.last_event_id = Some(id.clone());
                            }
                            return Some(Ok(event));
                        }
                    }
                    return None;
                }
                Ok(_) => {
                    if line.trim().is_empty() {
                        // Empty line marks end of event
                        if has_data {
                            if let Some(event) = SseEvent::from_wire_format(&self.buffer) {
                                if let Some(ref id) = event.id {
                                    self.last_event_id = Some(id.clone());
                                }
                                return Some(Ok(event));
                            }
                        }
                        self.buffer.clear();
                        has_data = false;
                    } else if !line.starts_with(':') {
                        // Ignore comments (lines starting with :)
                        self.buffer.push_str(&line);
                        has_data = true;
                    }
                }
                Err(e) => return Some(Err(SseError::IoError(e.to_string()))),
            }
        }
    }
}

/// SSE Stream writer
#[derive(Debug)]
pub struct SseWriter {
    events: Vec<SseEvent>,
    event_counter: u64,
}

impl SseWriter {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            event_counter: 0,
        }
    }

    pub fn write_event(&mut self, event: SseEvent) -> String {
        self.event_counter += 1;
        let event = if event.id.is_none() {
            event.with_id(&self.event_counter.to_string())
        } else {
            event
        };
        let wire = event.to_wire_format();
        self.events.push(event);
        wire
    }

    pub fn write_data(&mut self, data: &str) -> String {
        self.write_event(SseEvent::new(data))
    }

    pub fn write_comment(&self, comment: &str) -> String {
        format!(": {}\n\n", comment)
    }

    pub fn write_keepalive(&self) -> String {
        ":\n\n".to_string()
    }
}

impl Default for SseWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// SSE Error types
#[derive(Debug, Clone)]
pub enum SseError {
    IoError(String),
    ParseError(String),
    ConnectionClosed,
    Timeout,
}

impl std::fmt::Display for SseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::ConnectionClosed => write!(f, "Connection closed"),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

impl std::error::Error for SseError {}

/// SSE Client for consuming SSE streams
#[derive(Debug)]
pub struct SseClient {
    url: String,
    headers: HashMap<String, String>,
    last_event_id: Option<String>,
    retry_ms: u64,
}

impl SseClient {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            headers: HashMap::new(),
            last_event_id: None,
            retry_ms: 3000,
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_last_event_id(mut self, id: &str) -> Self {
        self.last_event_id = Some(id.to_string());
        self
    }

    /// Get the retry interval in milliseconds.
    pub fn retry_ms(&self) -> u64 {
        self.retry_ms
    }

    /// Connect and return an event iterator
    pub fn connect(&self) -> Result<SseConnection, SseError> {
        let mut request = ureq::get(&self.url)
            .set("Accept", "text/event-stream")
            .set("Cache-Control", "no-cache");

        for (key, value) in &self.headers {
            request = request.set(key, value);
        }

        if let Some(ref id) = self.last_event_id {
            request = request.set("Last-Event-ID", id);
        }

        let response = request
            .call()
            .map_err(|e| SseError::IoError(e.to_string()))?;

        Ok(SseConnection {
            reader: response.into_reader(),
            buffer: String::new(),
            last_event_id: self.last_event_id.clone(),
        })
    }
}

/// Active SSE connection
pub struct SseConnection {
    reader: Box<dyn Read + Send + Sync>,
    buffer: String,
    last_event_id: Option<String>,
}

impl std::fmt::Debug for SseConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SseConnection")
            .field("reader", &"<dyn Read + Send + Sync>")
            .field("buffer", &self.buffer)
            .field("last_event_id", &self.last_event_id)
            .finish()
    }
}

impl SseConnection {
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    pub fn read_event(&mut self) -> Result<Option<SseEvent>, SseError> {
        let mut buf = [0u8; 1024];
        self.buffer.clear();

        loop {
            let n = self
                .reader
                .read(&mut buf)
                .map_err(|e| SseError::IoError(e.to_string()))?;

            if n == 0 {
                // EOF
                if !self.buffer.is_empty() {
                    if let Some(event) = SseEvent::from_wire_format(&self.buffer) {
                        if let Some(ref id) = event.id {
                            self.last_event_id = Some(id.clone());
                        }
                        return Ok(Some(event));
                    }
                }
                return Ok(None);
            }

            let text = String::from_utf8_lossy(&buf[..n]);
            self.buffer.push_str(&text);

            // Check for complete event (double newline)
            if self.buffer.contains("\n\n") {
                let buffer_owned = self.buffer.clone();
                let parts: Vec<&str> = buffer_owned.splitn(2, "\n\n").collect();
                if parts.len() == 2 {
                    let event_text = parts[0].to_string();
                    self.buffer = parts[1].to_string();

                    if let Some(event) = SseEvent::from_wire_format(&event_text) {
                        if let Some(ref id) = event.id {
                            self.last_event_id = Some(id.clone());
                        }
                        return Ok(Some(event));
                    }
                }
            }
        }
    }
}

/// Stream chunk for AI responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

/// Stream choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

/// Stream delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl StreamChunk {
    /// Parse from SSE data (OpenAI format)
    pub fn from_sse_data(data: &str) -> Option<Self> {
        if data == "[DONE]" {
            return None;
        }
        serde_json::from_str(data).ok()
    }

    /// Get the text content if any
    pub fn get_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// Check if this is the final chunk
    pub fn is_done(&self) -> bool {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .is_some()
    }
}

/// Stream text aggregator
#[derive(Debug)]
pub struct StreamAggregator {
    chunks: Vec<StreamChunk>,
    full_text: String,
}

impl StreamAggregator {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            full_text: String::new(),
        }
    }

    pub fn add_chunk(&mut self, chunk: StreamChunk) {
        if let Some(content) = chunk.get_content() {
            self.full_text.push_str(content);
        }
        self.chunks.push(chunk);
    }

    pub fn get_text(&self) -> &str {
        &self.full_text
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_complete(&self) -> bool {
        self.chunks.last().map(|c| c.is_done()).unwrap_or(false)
    }
}

impl Default for StreamAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_wire_format() {
        let event = SseEvent::new("Hello, World!")
            .with_event("message")
            .with_id("1");

        let wire = event.to_wire_format();
        assert!(wire.contains("event: message"));
        assert!(wire.contains("id: 1"));
        assert!(wire.contains("data: Hello, World!"));
    }

    #[test]
    fn test_sse_event_parse() {
        let wire = "event: update\nid: 42\ndata: test data\n\n";
        let event = SseEvent::from_wire_format(wire).unwrap();

        assert_eq!(event.event, Some("update".to_string()));
        assert_eq!(event.id, Some("42".to_string()));
        assert_eq!(event.data, "test data");
    }

    #[test]
    fn test_sse_multiline_data() {
        let event = SseEvent::new("Line 1\nLine 2\nLine 3");
        let wire = event.to_wire_format();

        assert!(wire.contains("data: Line 1"));
        assert!(wire.contains("data: Line 2"));
        assert!(wire.contains("data: Line 3"));
    }

    #[test]
    fn test_sse_writer() {
        let mut writer = SseWriter::new();
        let output = writer.write_data("test");

        assert!(output.contains("data: test"));
        assert!(output.contains("id: 1"));
    }

    #[test]
    fn test_stream_chunk_parse() {
        let json = r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk = StreamChunk::from_sse_data(json).unwrap();
        assert_eq!(chunk.get_content(), Some("Hello"));
        assert!(!chunk.is_done());
    }

    #[test]
    fn test_stream_aggregator() {
        let mut aggregator = StreamAggregator::new();

        let chunk1 = StreamChunk {
            id: "1".to_string(),
            object: "chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };

        let chunk2 = StreamChunk {
            id: "2".to_string(),
            object: "chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: Some(" World".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        aggregator.add_chunk(chunk1);
        aggregator.add_chunk(chunk2);

        assert_eq!(aggregator.get_text(), "Hello World");
        assert!(aggregator.is_complete());
    }

    #[test]
    fn test_event_with_id_and_retry() {
        let event = SseEvent::new("hello")
            .with_event("message")
            .with_id("42")
            .with_retry(5000);
        let wire = event.to_wire_format();
        assert!(wire.contains("event: message"));
        assert!(wire.contains("id: 42"));
        assert!(wire.contains("retry: 5000"));
    }

    #[test]
    fn test_aggregator_chunk_count() {
        let mut agg = StreamAggregator::new();
        assert_eq!(agg.chunk_count(), 0);
        agg.add_chunk(StreamChunk {
            id: "1".to_string(),
            object: "chunk".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta { role: None, content: Some("hi".to_string()) },
                finish_reason: None,
            }],
        });
        assert_eq!(agg.chunk_count(), 1);
    }

    #[test]
    fn test_empty_aggregator() {
        let agg = StreamAggregator::new();
        assert_eq!(agg.get_text(), "");
        assert!(!agg.is_complete());
    }

    #[test]
    fn test_sse_error_display() {
        let err = SseError::ConnectionClosed;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
    }
}
