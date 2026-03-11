//! WebSocket streaming support
//!
//! Implements WebSocket protocol for bidirectional real-time AI communication.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    // Use RandomState which is seeded from OS entropy (unlike DefaultHasher
    // which uses fixed seeds). Each RandomState::new() call produces a hasher
    // with unique random keys, making the output unpredictable.
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let random_state = RandomState::new();
    let mut hasher = random_state.build_hasher();
    // Hash a second RandomState to mix additional OS entropy
    let extra_state = RandomState::new();
    let extra_hasher = extra_state.build_hasher();
    hasher.write_u64(extra_hasher.finish());
    let h = hasher.finish();
    // Take 4 bytes from the 8-byte hash
    (h as u32).to_ne_bytes()
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
    StreamStart { id: String, model: String },
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
    Error { code: String, message: String },
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
    pub on_close: Option<Box<dyn Fn(u16, &str) + Send + Sync>>,  // (code, reason)
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
        let message: WsAiMessage =
            serde_json::from_str(text).map_err(|e| WsError::ParseError(e.to_string()))?;

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
    pub fn create_tool_result(
        &self,
        id: &str,
        result: serde_json::Value,
        error: Option<&str>,
    ) -> WsFrame {
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

    /// Compute the expected Sec-WebSocket-Accept value per RFC 6455 Section 4.2.2.
    /// SHA-1( key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11" ) → base64
    pub fn expected_accept(&self) -> String {
        let concat = format!("{}258EAFA5-E914-47DA-95CA-C5AB0DC85B11", self.key);
        let hash = sha1_hash(concat.as_bytes());
        base64_encode(&hash)
    }
}

/// Generate a random WebSocket key: 16 random bytes -> base64 (RFC 6455 Section 4.1)
fn generate_ws_key() -> String {
    // Use RandomState which is seeded from OS entropy (unlike DefaultHasher
    // which uses fixed seeds). Each RandomState::new() call produces a hasher
    // with unique random keys, making the output unpredictable.
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut bytes = [0u8; 16];
    // Fill 16 bytes using two 8-byte hashes from independently-seeded RandomStates
    for chunk in bytes.chunks_mut(8) {
        let state = RandomState::new();
        let mut hasher = state.build_hasher();
        // Mix in a second RandomState for additional entropy
        let extra = RandomState::new();
        hasher.write_u64(extra.build_hasher().finish());
        let h = hasher.finish().to_ne_bytes();
        chunk.copy_from_slice(&h[..chunk.len()]);
    }
    base64_encode(&bytes)
}

/// SHA-1 hash (RFC 3174). Returns 20-byte digest.
pub(crate) fn sha1_hash(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    // Pre-processing: pad message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

/// Standard base64 encoding (RFC 4648)
pub(crate) fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Bidirectional stream manager
pub struct BidirectionalStream {
    handler: WsStreamHandler,
    outgoing: Vec<WsFrame>,
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

    /// Get the accumulated incoming text buffer.
    pub fn incoming_buffer(&self) -> &str {
        &self.incoming_buffer
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

    #[test]
    fn test_sha1_known_vectors() {
        // RFC 3174 test vectors
        let hash = sha1_hash(b"abc");
        assert_eq!(
            hash,
            [
                0xA9, 0x99, 0x3E, 0x36, 0x47, 0x06, 0x81, 0x6A, 0xBA, 0x3E, 0x25, 0x71, 0x78, 0x50,
                0xC2, 0x6C, 0x9C, 0xD0, 0xD8, 0x9D
            ]
        );

        let hash2 = sha1_hash(b"");
        assert_eq!(
            hash2,
            [
                0xDA, 0x39, 0xA3, 0xEE, 0x5E, 0x6B, 0x4B, 0x0D, 0x32, 0x55, 0xBF, 0xEF, 0x95, 0x60,
                0x18, 0x90, 0xAF, 0xD8, 0x07, 0x09
            ]
        );
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_websocket_accept_rfc6455() {
        // RFC 6455 Section 4.2.2 example
        let handshake = WsHandshake {
            host: "example.com".to_string(),
            path: "/chat".to_string(),
            key: "dGhlIHNhbXBsZSBub25jZQ==".to_string(),
            protocols: Vec::new(),
            extensions: Vec::new(),
            headers: HashMap::new(),
        };

        assert_eq!(handshake.expected_accept(), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn test_generate_ws_key_format() {
        let key = generate_ws_key();
        // WebSocket key should be base64-encoded 16 bytes = 24 chars with padding
        assert_eq!(key.len(), 24);
        // Should be valid base64
        assert!(key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='));
    }
}
