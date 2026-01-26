//! WebSocket streaming support
//!
//! Implements WebSocket protocol for bidirectional real-time AI communication.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// WebSocket message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsOpcode {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
}

impl WsOpcode {
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte & 0x0F {
            0x0 => Some(Self::Continuation),
            0x1 => Some(Self::Text),
            0x2 => Some(Self::Binary),
            0x8 => Some(Self::Close),
            0x9 => Some(Self::Ping),
            0xA => Some(Self::Pong),
            _ => None,
        }
    }
}

/// WebSocket frame
#[derive(Debug, Clone)]
pub struct WsFrame {
    pub fin: bool,
    pub opcode: WsOpcode,
    pub masked: bool,
    pub payload: Vec<u8>,
}

impl WsFrame {
    pub fn text(data: &str) -> Self {
        Self {
            fin: true,
            opcode: WsOpcode::Text,
            masked: true, // Client frames must be masked
            payload: data.as_bytes().to_vec(),
        }
    }

    pub fn binary(data: &[u8]) -> Self {
        Self {
            fin: true,
            opcode: WsOpcode::Binary,
            masked: true,
            payload: data.to_vec(),
        }
    }

    pub fn ping(data: &[u8]) -> Self {
        Self {
            fin: true,
            opcode: WsOpcode::Ping,
            masked: true,
            payload: data.to_vec(),
        }
    }

    pub fn pong(data: &[u8]) -> Self {
        Self {
            fin: true,
            opcode: WsOpcode::Pong,
            masked: true,
            payload: data.to_vec(),
        }
    }

    pub fn close(code: u16, reason: &str) -> Self {
        let mut payload = Vec::new();
        payload.extend_from_slice(&code.to_be_bytes());
        payload.extend_from_slice(reason.as_bytes());
        Self {
            fin: true,
            opcode: WsOpcode::Close,
            masked: true,
            payload,
        }
    }

    /// Encode frame to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // First byte: FIN + opcode
        let first_byte = if self.fin { 0x80 } else { 0x00 } | (self.opcode as u8);
        bytes.push(first_byte);

        // Second byte: MASK + payload length
        let len = self.payload.len();
        let mask_bit = if self.masked { 0x80 } else { 0x00 };

        if len < 126 {
            bytes.push(mask_bit | len as u8);
        } else if len < 65536 {
            bytes.push(mask_bit | 126);
            bytes.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            bytes.push(mask_bit | 127);
            bytes.extend_from_slice(&(len as u64).to_be_bytes());
        }

        // Masking key (if masked)
        if self.masked {
            let mask: [u8; 4] = rand_mask();
            bytes.extend_from_slice(&mask);

            // Masked payload
            for (i, byte) in self.payload.iter().enumerate() {
                bytes.push(byte ^ mask[i % 4]);
            }
        } else {
            bytes.extend_from_slice(&self.payload);
        }

        bytes
    }

    /// Get payload as string (for text frames)
    pub fn as_text(&self) -> Option<&str> {
        if self.opcode == WsOpcode::Text {
            std::str::from_utf8(&self.payload).ok()
        } else {
            None
        }
    }
}

fn rand_mask() -> [u8; 4] {
    // Simple pseudo-random mask
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(0);

    [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
        ((seed >> 24) & 0xFF) as u8,
    ]
}

/// WebSocket close codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsCloseCode {
    Normal = 1000,
    GoingAway = 1001,
    ProtocolError = 1002,
    UnsupportedData = 1003,
    NoStatus = 1005,
    Abnormal = 1006,
    InvalidPayload = 1007,
    PolicyViolation = 1008,
    MessageTooBig = 1009,
    MandatoryExtension = 1010,
    InternalError = 1011,
    ServiceRestart = 1012,
    TryAgainLater = 1013,
}

/// WebSocket message for AI streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsAiMessage {
    #[serde(rename = "chat")]
    Chat {
        id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    #[serde(rename = "stream_start")]
    StreamStart {
        id: String,
        model: String,
    },
    #[serde(rename = "stream_chunk")]
    StreamChunk {
        id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
    },
    #[serde(rename = "stream_end")]
    StreamEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<WsUsage>,
    },
    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        id: String,
        result: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
    },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
}

/// Token usage in WebSocket response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsState {
    Connecting,
    Open,
    Closing,
    Closed,
}

/// WebSocket stream handler
pub struct WsStreamHandler {
    state: WsState,
    message_id: u64,
    pending_chunks: HashMap<String, Vec<String>>,
    callbacks: WsCallbacks,
}

/// Callbacks for WebSocket events
pub struct WsCallbacks {
    pub on_message: Option<Box<dyn Fn(&WsAiMessage) + Send + Sync>>,
    pub on_chunk: Option<Box<dyn Fn(&str, &str) + Send + Sync>>, // (id, content)
    pub on_complete: Option<Box<dyn Fn(&str, &str) + Send + Sync>>, // (id, full_text)
    pub on_error: Option<Box<dyn Fn(&str, &str) + Send + Sync>>, // (code, message)
    pub on_close: Option<Box<dyn Fn(u16, &str) + Send + Sync>>, // (code, reason)
}

impl Default for WsCallbacks {
    fn default() -> Self {
        Self {
            on_message: None,
            on_chunk: None,
            on_complete: None,
            on_error: None,
            on_close: None,
        }
    }
}

impl WsStreamHandler {
    pub fn new() -> Self {
        Self {
            state: WsState::Connecting,
            message_id: 0,
            pending_chunks: HashMap::new(),
            callbacks: WsCallbacks::default(),
        }
    }

    pub fn set_on_message<F>(&mut self, callback: F)
    where
        F: Fn(&WsAiMessage) + Send + Sync + 'static,
    {
        self.callbacks.on_message = Some(Box::new(callback));
    }

    pub fn set_on_chunk<F>(&mut self, callback: F)
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        self.callbacks.on_chunk = Some(Box::new(callback));
    }

    pub fn set_on_complete<F>(&mut self, callback: F)
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        self.callbacks.on_complete = Some(Box::new(callback));
    }

    pub fn set_on_error<F>(&mut self, callback: F)
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        self.callbacks.on_error = Some(Box::new(callback));
    }

    pub fn state(&self) -> WsState {
        self.state
    }

    pub fn set_open(&mut self) {
        self.state = WsState::Open;
    }

    pub fn next_message_id(&mut self) -> String {
        self.message_id += 1;
        format!("msg_{}", self.message_id)
    }

    /// Handle incoming message
    pub fn handle_message(&mut self, text: &str) -> Result<(), WsError> {
        let message: WsAiMessage = serde_json::from_str(text)
            .map_err(|e| WsError::ParseError(e.to_string()))?;

        // Invoke general callback
        if let Some(ref callback) = self.callbacks.on_message {
            callback(&message);
        }

        match &message {
            WsAiMessage::StreamStart { id, .. } => {
                self.pending_chunks.insert(id.clone(), Vec::new());
            }
            WsAiMessage::StreamChunk { id, content, .. } => {
                if let Some(chunks) = self.pending_chunks.get_mut(id) {
                    chunks.push(content.clone());
                }
                if let Some(ref callback) = self.callbacks.on_chunk {
                    callback(id, content);
                }
            }
            WsAiMessage::StreamEnd { id, .. } => {
                if let Some(chunks) = self.pending_chunks.remove(id) {
                    let full_text = chunks.join("");
                    if let Some(ref callback) = self.callbacks.on_complete {
                        callback(id, &full_text);
                    }
                }
            }
            WsAiMessage::Error { code, message } => {
                if let Some(ref callback) = self.callbacks.on_error {
                    callback(code, message);
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Create a chat message frame
    pub fn create_chat_message(&mut self, content: &str, model: Option<&str>) -> WsFrame {
        let id = self.next_message_id();
        let message = WsAiMessage::Chat {
            id,
            content: content.to_string(),
            model: model.map(|s| s.to_string()),
        };
        let json = serde_json::to_string(&message).unwrap_or_default();
        WsFrame::text(&json)
    }

    /// Create a tool result frame
    pub fn create_tool_result(&self, id: &str, result: serde_json::Value, error: Option<&str>) -> WsFrame {
        let message = WsAiMessage::ToolResult {
            id: id.to_string(),
            result,
            error: error.map(|s| s.to_string()),
        };
        let json = serde_json::to_string(&message).unwrap_or_default();
        WsFrame::text(&json)
    }

    /// Create a ping frame
    pub fn create_ping(&self) -> WsFrame {
        let message = WsAiMessage::Ping;
        let json = serde_json::to_string(&message).unwrap_or_default();
        WsFrame::text(&json)
    }
}

impl Default for WsStreamHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket errors
#[derive(Debug, Clone)]
pub enum WsError {
    ConnectionFailed(String),
    ParseError(String),
    ProtocolError(String),
    Closed(u16, String),
    Timeout,
}

impl std::fmt::Display for WsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(e) => write!(f, "Connection failed: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::ProtocolError(e) => write!(f, "Protocol error: {}", e),
            Self::Closed(code, reason) => write!(f, "Closed: {} - {}", code, reason),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

impl std::error::Error for WsError {}

/// WebSocket handshake builder
pub struct WsHandshake {
    host: String,
    path: String,
    key: String,
    protocols: Vec<String>,
    extensions: Vec<String>,
    headers: HashMap<String, String>,
}

impl WsHandshake {
    pub fn new(host: &str, path: &str) -> Self {
        Self {
            host: host.to_string(),
            path: path.to_string(),
            key: generate_ws_key(),
            protocols: Vec::new(),
            extensions: Vec::new(),
            headers: HashMap::new(),
        }
    }

    pub fn with_protocol(mut self, protocol: &str) -> Self {
        self.protocols.push(protocol.to_string());
        self
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build_request(&self) -> String {
        let mut request = format!(
            "GET {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\n\
             Sec-WebSocket-Version: 13\r\n",
            self.path, self.host, self.key
        );

        if !self.protocols.is_empty() {
            request.push_str(&format!(
                "Sec-WebSocket-Protocol: {}\r\n",
                self.protocols.join(", ")
            ));
        }

        if !self.extensions.is_empty() {
            request.push_str(&format!(
                "Sec-WebSocket-Extensions: {}\r\n",
                self.extensions.join(", ")
            ));
        }

        for (key, value) in &self.headers {
            request.push_str(&format!("{}: {}\r\n", key, value));
        }

        request.push_str("\r\n");
        request
    }

    pub fn expected_accept(&self) -> String {
        // In real implementation, compute SHA-1 of key + magic string
        // For now, return placeholder
        format!("accept_{}", self.key)
    }
}

fn generate_ws_key() -> String {
    // Generate 16 random bytes and base64 encode
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    // Simple base64-like encoding for demo
    format!("key_{:016x}", timestamp)
}

/// Bidirectional stream manager
pub struct BidirectionalStream {
    handler: WsStreamHandler,
    outgoing: Vec<WsFrame>,
    #[allow(dead_code)]
    incoming_buffer: String,
}

impl BidirectionalStream {
    pub fn new() -> Self {
        Self {
            handler: WsStreamHandler::new(),
            outgoing: Vec::new(),
            incoming_buffer: String::new(),
        }
    }

    pub fn send_message(&mut self, content: &str, model: Option<&str>) {
        let frame = self.handler.create_chat_message(content, model);
        self.outgoing.push(frame);
    }

    pub fn send_tool_result(&mut self, id: &str, result: serde_json::Value) {
        let frame = self.handler.create_tool_result(id, result, None);
        self.outgoing.push(frame);
    }

    pub fn receive_text(&mut self, text: &str) -> Result<(), WsError> {
        self.handler.handle_message(text)
    }

    pub fn drain_outgoing(&mut self) -> Vec<WsFrame> {
        std::mem::take(&mut self.outgoing)
    }

    pub fn handler_mut(&mut self) -> &mut WsStreamHandler {
        &mut self.handler
    }
}

impl Default for BidirectionalStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_frame_text() {
        let frame = WsFrame::text("Hello");
        assert_eq!(frame.opcode, WsOpcode::Text);
        assert!(frame.fin);
        assert!(frame.masked);
    }

    #[test]
    fn test_ws_frame_encode() {
        let frame = WsFrame::text("Hi");
        let bytes = frame.encode();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0] & 0x0F, WsOpcode::Text as u8);
    }

    #[test]
    fn test_ws_message_serialize() {
        let msg = WsAiMessage::StreamChunk {
            id: "1".to_string(),
            content: "Hello".to_string(),
            index: Some(0),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("stream_chunk"));
    }

    #[test]
    fn test_ws_handler() {
        let mut handler = WsStreamHandler::new();
        handler.set_open();

        assert_eq!(handler.state(), WsState::Open);

        let id = handler.next_message_id();
        assert!(id.starts_with("msg_"));
    }

    #[test]
    fn test_ws_handshake() {
        let handshake = WsHandshake::new("example.com", "/ws")
            .with_protocol("chat")
            .with_header("Authorization", "Bearer token");

        let request = handshake.build_request();
        assert!(request.contains("GET /ws HTTP/1.1"));
        assert!(request.contains("Upgrade: websocket"));
        assert!(request.contains("Sec-WebSocket-Protocol: chat"));
    }

    #[test]
    fn test_bidirectional_stream() {
        let mut stream = BidirectionalStream::new();
        stream.send_message("Hello", Some("gpt-4"));

        let frames = stream.drain_outgoing();
        assert_eq!(frames.len(), 1);
    }
}
