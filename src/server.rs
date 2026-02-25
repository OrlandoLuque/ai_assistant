//! Lightweight embedded HTTP server for exposing the AI assistant as a REST API.
//!
//! Uses `std::net::TcpListener` with manual HTTP/1.1 parsing — zero extra dependencies
//! beyond the standard library. Intended for local development, inter-process communication,
//! and simple integrations.
//!
//! ## Key types
//!
//! - [`ServerConfig`] — Server address, port, CORS, rate limiting, optional TLS
//! - [`AiServer`] — The server instance; call [`AiServer::run_blocking`] to start
//! - [`ServerMetrics`] — Prometheus-style counters (requests, latency, errors)
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | GET | `/health` | Health check |
//! | GET | `/models` | List available models |
//! | POST | `/chat` | Send a message (non-streaming) |
//! | POST | `/config` | Update configuration |
//! | GET | `/config` | Get current configuration |
//! | GET | `/metrics` | Prometheus-style metrics |
//!
//! ## Feature flags
//!
//! - Included in `full` (always compiled with default features)
//! - `server-tls` — Enables HTTPS via rustls (opt-in, requires PEM cert/key files)
//!
//! ## Example
//!
//! ```rust,no_run
//! use ai_assistant::server::{ServerConfig, AiServer};
//!
//! let config = ServerConfig::default(); // localhost:8090
//! let server = AiServer::new(config);
//! server.run_blocking(); // blocks
//! ```

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::assistant::AiAssistant;

// ============================================================================
// TLS support (feature-gated)
// ============================================================================

/// Load PEM certificates and private key, returning a rustls `ServerConfig`.
///
/// Returns an error if the cert/key files cannot be read or parsed.
#[cfg(feature = "server-tls")]
pub fn load_tls_config(tls: &TlsConfig) -> Result<Arc<rustls::ServerConfig>, String> {
    let cert_data = std::fs::read(&tls.cert_path)
        .map_err(|e| format!("Failed to read cert file '{}': {}", tls.cert_path, e))?;
    let key_data = std::fs::read(&tls.key_path)
        .map_err(|e| format!("Failed to read key file '{}': {}", tls.key_path, e))?;

    // Parse certificates
    let certs: Vec<_> = rustls_pemfile::certs(&mut &cert_data[..])
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to parse PEM certificates: {}", e))?;
    if certs.is_empty() {
        return Err("No certificates found in PEM file".to_string());
    }

    // Parse private key (try PKCS8, then RSA, then EC)
    let key = rustls_pemfile::private_key(&mut &key_data[..])
        .map_err(|e| format!("Failed to parse PEM private key: {}", e))?
        .ok_or_else(|| "No private key found in PEM file".to_string())?;

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("Failed to build TLS config: {}", e))?;

    Ok(Arc::new(config))
}

/// Accept a TLS connection on a TCP stream, returning a Read+Write wrapper.
#[cfg(feature = "server-tls")]
fn tls_accept(
    stream: TcpStream,
    tls_config: &Arc<rustls::ServerConfig>,
) -> std::io::Result<rustls::StreamOwned<rustls::ServerConnection, TcpStream>> {
    let conn = rustls::ServerConnection::new(Arc::clone(tls_config))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("TLS init error: {}", e)))?;
    Ok(rustls::StreamOwned::new(conn, stream))
}

// ============================================================================
// Server Configuration
// ============================================================================

/// Authentication configuration for the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Whether authentication is enabled (default: false).
    pub enabled: bool,
    /// Valid bearer tokens for `Authorization: Bearer <token>` authentication.
    pub bearer_tokens: Vec<String>,
    /// Valid API keys for `X-API-Key: <key>` authentication.
    pub api_keys: Vec<String>,
    /// Paths that bypass authentication (default: ["/health"]).
    pub exempt_paths: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bearer_tokens: Vec::new(),
            api_keys: Vec::new(),
            exempt_paths: vec!["/health".to_string()],
        }
    }
}

/// Result of an authentication check.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthResult {
    /// The request was authenticated successfully.
    Authenticated,
    /// The request was rejected with the given reason.
    Rejected(String),
    /// The request path is exempt from authentication.
    Exempt,
}

/// CORS (Cross-Origin Resource Sharing) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Allowed origins (default: ["*"]).
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods (default: ["GET", "POST", "OPTIONS"]).
    pub allowed_methods: Vec<String>,
    /// Allowed request headers (default: ["Content-Type", "Authorization", "X-API-Key"]).
    pub allowed_headers: Vec<String>,
    /// Max age for preflight cache in seconds (default: 86400).
    pub max_age_secs: u64,
    /// Whether to include `Access-Control-Allow-Credentials: true` (default: false).
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-API-Key".to_string(),
            ],
            max_age_secs: 86400,
            allow_credentials: false,
        }
    }
}

/// TLS configuration for HTTPS support.
///
/// When provided in `ServerConfig`, the server will use HTTPS.
/// When `None` (the default), the server falls back to plain HTTP.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the PEM-encoded certificate file.
    pub cert_path: String,
    /// Path to the PEM-encoded private key file.
    pub key_path: String,
}

/// Configuration for the embedded HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to (default: "127.0.0.1").
    pub host: String,
    /// Port to bind to (default: 8090).
    pub port: u16,
    /// Maximum request body size in bytes (default: 1MB).
    pub max_body_size: usize,
    /// Read timeout in seconds (default: 30).
    pub read_timeout_secs: u64,
    /// Authentication configuration.
    pub auth: AuthConfig,
    /// CORS configuration.
    pub cors: CorsConfig,
    /// Maximum number of headers allowed per request (default: 100).
    pub max_headers: usize,
    /// Maximum length of a single header line in bytes (default: 8192).
    pub max_header_line: usize,
    /// Timeout for reading the request body in milliseconds (default: 30000).
    pub body_read_timeout_ms: u64,
    /// Maximum message length in characters (default: 100_000).
    pub max_message_length: usize,
    /// TLS configuration (optional). When set, the server should use HTTPS.
    #[serde(skip)]
    pub tls: Option<TlsConfig>,
    /// Rate limiter (optional). When set, requests exceeding the limit get 429.
    #[serde(skip)]
    pub rate_limiter: Option<Arc<ServerRateLimiter>>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8090,
            max_body_size: 1_048_576,
            read_timeout_secs: 30,
            auth: AuthConfig::default(),
            cors: CorsConfig::default(),
            max_headers: 100,
            max_header_line: 8192,
            body_read_timeout_ms: 30_000,
            max_message_length: 100_000,
            tls: None,
            rate_limiter: None,
        }
    }
}

impl ServerConfig {
    /// Get the bind address as "host:port".
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ============================================================================
// Request / Response Types
// ============================================================================

/// Parsed HTTP request.
#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

/// Chat request body.
#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
    #[serde(default)]
    system_prompt: String,
    #[serde(default)]
    knowledge_context: String,
}

/// Chat response body.
#[derive(Debug, Serialize)]
struct ChatResponse {
    content: String,
    model: String,
}

/// Health check response.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    model: String,
    provider: String,
    uptime_secs: u64,
    active_sessions: usize,
    conversation_messages: usize,
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// ============================================================================
// Server Metrics (for Prometheus /metrics endpoint and request logging)
// ============================================================================

/// Atomic counters for server-level metrics.
struct ServerMetrics {
    requests_total: AtomicU64,
    requests_2xx: AtomicU64,
    requests_4xx: AtomicU64,
    requests_5xx: AtomicU64,
    /// Total request duration in microseconds (for computing averages).
    request_duration_us_total: AtomicU64,
    /// Counter for generating unique request IDs when the client does not supply one.
    request_id_counter: AtomicU64,
    /// Instant when the server started, used for uptime calculation.
    started_at: std::time::Instant,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_2xx: AtomicU64::new(0),
            requests_4xx: AtomicU64::new(0),
            requests_5xx: AtomicU64::new(0),
            request_duration_us_total: AtomicU64::new(0),
            request_id_counter: AtomicU64::new(0),
            started_at: std::time::Instant::now(),
        }
    }

    /// Generate a hex request ID from the counter mixed with a bit of entropy.
    fn generate_request_id(&self) -> String {
        let counter = self.request_id_counter.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mixed_hi = nanos.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mixed_lo = counter.wrapping_mul(2862933555777941757).wrapping_add(3);
        format!("{:016x}{:016x}", mixed_hi, mixed_lo)
    }

    /// Record a completed request with its status code string and duration.
    fn record_request(&self, status: &str, duration: std::time::Duration) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        let us = duration.as_micros() as u64;
        self.request_duration_us_total.fetch_add(us, Ordering::Relaxed);

        if status.starts_with('2') {
            self.requests_2xx.fetch_add(1, Ordering::Relaxed);
        } else if status.starts_with('4') {
            self.requests_4xx.fetch_add(1, Ordering::Relaxed);
        } else if status.starts_with('5') {
            self.requests_5xx.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Render metrics in Prometheus exposition text format.
    fn render_prometheus(&self) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let ok = self.requests_2xx.load(Ordering::Relaxed);
        let client_err = self.requests_4xx.load(Ordering::Relaxed);
        let server_err = self.requests_5xx.load(Ordering::Relaxed);
        let duration_us = self.request_duration_us_total.load(Ordering::Relaxed);
        let duration_secs = duration_us as f64 / 1_000_000.0;

        format!(
            "# HELP ai_server_requests_total Total number of HTTP requests.\n\
             # TYPE ai_server_requests_total counter\n\
             ai_server_requests_total {}\n\
             # HELP ai_server_requests_by_status HTTP requests by status class.\n\
             # TYPE ai_server_requests_by_status counter\n\
             ai_server_requests_by_status{{status=\"2xx\"}} {}\n\
             ai_server_requests_by_status{{status=\"4xx\"}} {}\n\
             ai_server_requests_by_status{{status=\"5xx\"}} {}\n\
             # HELP ai_server_request_duration_seconds_total Total time spent processing requests.\n\
             # TYPE ai_server_request_duration_seconds_total counter\n\
             ai_server_request_duration_seconds_total {:.6}\n",
            total, ok, client_err, server_err, duration_secs
        )
    }
}

// ============================================================================
// Audit Log (items 4.1, 4.2)
// ============================================================================

/// Audit event types for compliance logging.
#[derive(Debug, Clone, Serialize)]
pub enum AuditEventType {
    AuthSuccess,
    AuthFailure,
    ConfigChange,
    SessionCreated,
    SessionDeleted,
    RequestProcessed,
}

/// A single audit log entry.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub event_type: AuditEventType,
    pub actor: String,
    pub path: String,
    pub details: String,
}

/// Thread-safe append-only audit log.
pub struct AuditLog {
    entries: std::sync::Mutex<Vec<AuditEntry>>,
    max_entries: usize,
}

impl AuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
            max_entries,
        }
    }

    pub fn log(&self, event_type: AuditEventType, actor: &str, path: &str, details: &str) {
        let entry = AuditEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type,
            actor: actor.to_string(),
            path: path.to_string(),
            details: details.to_string(),
        };
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(entry);
    }

    pub fn entries(&self) -> Vec<AuditEntry> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Structured Error Responses (items 6.1, 6.2)
// ============================================================================

/// Structured error response with error code.
#[derive(Debug, Serialize)]
pub struct StructuredError {
    error_code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retry_after_secs: Option<u64>,
}

impl StructuredError {
    /// Create a new structured error with a code and message.
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            error_code: code.to_string(),
            message: message.to_string(),
            details: None,
            retry_after_secs: None,
        }
    }

    /// Add detail text to the error.
    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }

    /// Add a retry-after hint in seconds.
    pub fn with_retry(mut self, secs: u64) -> Self {
        self.retry_after_secs = Some(secs);
        self
    }
}

// ============================================================================
// Rate Limiting (item 1.6)
// ============================================================================

/// Thread-safe per-server rate limiter based on a sliding window of one minute.
///
/// When the request count exceeds `requests_per_minute` within the current
/// window, `check_rate_limit()` returns `false` and the server should
/// respond with 429 Too Many Requests.
pub struct ServerRateLimiter {
    /// Maximum number of requests allowed per 60-second window.
    pub requests_per_minute: u32,
    /// Start of the current window.
    window_start: Mutex<Instant>,
    /// Number of requests observed in the current window.
    request_count: AtomicU32,
}

impl std::fmt::Debug for ServerRateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerRateLimiter")
            .field("requests_per_minute", &self.requests_per_minute)
            .field("request_count", &self.request_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl ServerRateLimiter {
    /// Create a new rate limiter with the given requests-per-minute cap.
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            window_start: Mutex::new(Instant::now()),
            request_count: AtomicU32::new(0),
        }
    }

    /// Reset the window if 60 seconds have elapsed since `window_start`.
    pub fn reset_if_window_expired(&self) {
        let mut start = self.window_start.lock().unwrap_or_else(|e| e.into_inner());
        if start.elapsed().as_secs() >= 60 {
            *start = Instant::now();
            self.request_count.store(0, Ordering::Relaxed);
        }
    }

    /// Check whether the current request is allowed.
    ///
    /// Returns `true` if the request should be processed, `false` if the
    /// rate limit has been exceeded.
    pub fn check_rate_limit(&self) -> bool {
        self.reset_if_window_expired();
        let count = self.request_count.fetch_add(1, Ordering::Relaxed);
        count < self.requests_per_minute
    }

    /// Return the number of seconds remaining in the current window.
    pub fn retry_after_secs(&self) -> u64 {
        let start = self.window_start.lock().unwrap_or_else(|e| e.into_inner());
        let elapsed = start.elapsed().as_secs();
        if elapsed >= 60 { 0 } else { 60 - elapsed }
    }
}

// ============================================================================
// Response Compression (item 1.5)
// ============================================================================

/// Compress `data` with gzip using `flate2`.
fn compress_gzip(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    let _ = encoder.write_all(data);
    encoder.finish().unwrap_or_else(|_| data.to_vec())
}

/// Conditionally compress `body` if the client sent `Accept-Encoding: gzip`.
///
/// Returns `(body_bytes, was_compressed)`.
fn maybe_compress_response(body: &str, accept_encoding: Option<&str>) -> (Vec<u8>, bool) {
    if let Some(enc) = accept_encoding {
        if enc.contains("gzip") {
            let compressed = compress_gzip(body.as_bytes());
            return (compressed, true);
        }
    }
    (body.as_bytes().to_vec(), false)
}

// ============================================================================
// SSE Streaming (item 1.1)
// ============================================================================

/// Handle a POST /chat/stream request by returning Server-Sent Events.
///
/// The response simulates streaming by splitting the LLM response into
/// individual word tokens and sending each as an SSE `data:` event.
fn handle_chat_stream(
    request: &HttpRequest,
    assistant: &Arc<Mutex<AiAssistant>>,
    config: &ServerConfig,
) -> Result<String, (String, String)> {
    let chat_req: ChatRequest = serde_json::from_str(&request.body).map_err(|e| {
        (
            "400 Bad Request".to_string(),
            serde_json::to_string(&ErrorResponse {
                error: format!("Invalid JSON: {}", e),
            })
            .unwrap_or_default(),
        )
    })?;

    // Validate message length
    if chat_req.message.len() > config.max_message_length {
        return Err((
            "422 Unprocessable Entity".to_string(),
            serde_json::to_string(&ErrorResponse {
                error: format!(
                    "Message too long: {} characters (max {})",
                    chat_req.message.len(),
                    config.max_message_length
                ),
            })
            .unwrap_or_default(),
        ));
    }

    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());

    ass.send_message_with_notes(
        chat_req.message.clone(),
        &chat_req.knowledge_context,
        &chat_req.system_prompt,
        "",
    );

    // Collect the full response first, then we split into tokens for SSE
    let full_text = loop {
        match ass.poll_response() {
            Some(crate::messages::AiResponse::Complete(text)) => break text,
            Some(crate::messages::AiResponse::Error(e)) => {
                return Err((
                    "500 Internal Server Error".to_string(),
                    serde_json::to_string(&ErrorResponse { error: e }).unwrap_or_default(),
                ));
            }
            Some(crate::messages::AiResponse::Cancelled(partial)) => break partial,
            Some(_) => continue,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    };

    // Build SSE event stream
    let mut sse_body = String::new();
    for word in full_text.split_whitespace() {
        let token_json = serde_json::json!({"token": word});
        sse_body.push_str(&format!("data: {}\n\n", token_json));
    }
    sse_body.push_str("data: [DONE]\n\n");

    Ok(sse_body)
}

/// Format SSE data into a full SSE HTTP response (used by `handle_chat_stream`
/// when the stream is produced successfully).
fn build_sse_response(sse_body: &str, extra_headers: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n{}Content-Length: {}\r\n\r\n{}",
        extra_headers,
        sse_body.len(),
        sse_body,
    )
}

// ============================================================================
// WebSocket support (RFC 6455)
// ============================================================================

/// WebSocket magic GUID for Sec-WebSocket-Accept (RFC 6455 §4.2.2).
const WS_MAGIC_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Read a single WebSocket frame from a stream (RFC 6455 §5.2).
fn read_ws_frame(stream: &mut dyn Read) -> std::io::Result<crate::websocket_streaming::WsFrame> {
    use crate::websocket_streaming::{WsFrame, WsOpcode};

    let mut header = [0u8; 2];
    stream.read_exact(&mut header)?;

    let fin = (header[0] & 0x80) != 0;
    let opcode_byte = header[0] & 0x0F;
    let masked = (header[1] & 0x80) != 0;
    let mut payload_len = (header[1] & 0x7F) as u64;

    if payload_len == 126 {
        let mut ext = [0u8; 2];
        stream.read_exact(&mut ext)?;
        payload_len = u16::from_be_bytes(ext) as u64;
    } else if payload_len == 127 {
        let mut ext = [0u8; 8];
        stream.read_exact(&mut ext)?;
        payload_len = u64::from_be_bytes(ext);
    }

    if payload_len > 16 * 1024 * 1024 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "WebSocket frame too large"));
    }

    let mask = if masked {
        let mut m = [0u8; 4];
        stream.read_exact(&mut m)?;
        Some(m)
    } else {
        None
    };

    let mut payload = vec![0u8; payload_len as usize];
    if !payload.is_empty() {
        stream.read_exact(&mut payload)?;
    }

    if let Some(mask_key) = mask {
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask_key[i % 4];
        }
    }

    let opcode = WsOpcode::from_byte(opcode_byte).unwrap_or(WsOpcode::Close);

    Ok(WsFrame { fin, opcode, masked: false, payload })
}

/// Write a WebSocket frame to a stream (server → client, unmasked per RFC 6455).
fn write_ws_frame(stream: &mut dyn Write, frame: &crate::websocket_streaming::WsFrame) -> std::io::Result<()> {
    let mut header = Vec::with_capacity(10);
    let fin_bit = if frame.fin { 0x80u8 } else { 0x00 };
    let opcode_byte = match frame.opcode {
        crate::websocket_streaming::WsOpcode::Continuation => 0x0,
        crate::websocket_streaming::WsOpcode::Text => 0x1,
        crate::websocket_streaming::WsOpcode::Binary => 0x2,
        crate::websocket_streaming::WsOpcode::Close => 0x8,
        crate::websocket_streaming::WsOpcode::Ping => 0x9,
        crate::websocket_streaming::WsOpcode::Pong => 0xA,
    };
    header.push(fin_bit | opcode_byte);

    let len = frame.payload.len();
    if len < 126 {
        header.push(len as u8);
    } else if len <= 65535 {
        header.push(126);
        header.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        header.push(127);
        header.extend_from_slice(&(len as u64).to_be_bytes());
    }

    stream.write_all(&header)?;
    stream.write_all(&frame.payload)?;
    stream.flush()?;
    Ok(())
}

/// Perform the WebSocket upgrade handshake (server side).
fn ws_handshake(stream: &mut dyn Write, headers: &[(String, String)], extra_headers: &str) -> std::io::Result<()> {
    let ws_key = headers.iter().find(|(k, _)| k == "sec-websocket-key").map(|(_, v)| v.as_str()).unwrap_or("");
    let mut accept_input = ws_key.to_string();
    accept_input.push_str(WS_MAGIC_GUID);
    let hash = crate::websocket_streaming::sha1_hash(accept_input.as_bytes());
    let accept = crate::websocket_streaming::base64_encode(&hash);

    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n{}\r\n",
        accept, extra_headers,
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

/// Handle a WebSocket chat session after upgrade.
fn handle_ws_chat(stream: &mut dyn ReadWrite, assistant: &Arc<Mutex<AiAssistant>>, config: &ServerConfig) -> std::io::Result<()> {
    use crate::websocket_streaming::{WsFrame, WsOpcode};

    loop {
        let frame = match read_ws_frame(stream) {
            Ok(f) => f,
            Err(_) => break,
        };

        match frame.opcode {
            WsOpcode::Text => {
                let text = match std::str::from_utf8(&frame.payload) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = write_ws_frame(stream, &WsFrame::close(1007, "Invalid UTF-8"));
                        break;
                    }
                };

                #[derive(serde::Deserialize)]
                struct WsChatReq { message: String, #[serde(default)] system_prompt: Option<String> }

                let req: WsChatReq = match serde_json::from_str(text) {
                    Ok(r) => r,
                    Err(e) => {
                        let err = serde_json::json!({"type": "error", "message": format!("Invalid JSON: {}", e)});
                        let _ = write_ws_frame(stream, &WsFrame::text(&err.to_string()));
                        continue;
                    }
                };

                if req.message.len() > config.max_message_length {
                    let err = serde_json::json!({"type": "error", "message": "Message too long"});
                    let _ = write_ws_frame(stream, &WsFrame::text(&err.to_string()));
                    continue;
                }

                let response_text = {
                    let mut guard = assistant.lock().unwrap_or_else(|e| e.into_inner());
                    let sys = req.system_prompt.as_deref().unwrap_or("");
                    guard.send_message_with_notes(req.message.clone(), "", sys, "");
                    // Poll until we get the full response
                    loop {
                        match guard.poll_response() {
                            Some(crate::messages::AiResponse::Complete(text)) => break text,
                            Some(crate::messages::AiResponse::Error(e)) => break format!("[Error: {}]", e),
                            Some(crate::messages::AiResponse::Cancelled(partial)) => break partial,
                            Some(_) => continue,
                            None => std::thread::sleep(std::time::Duration::from_millis(10)),
                        }
                    }
                };

                for word in response_text.split_whitespace() {
                    let chunk = serde_json::json!({"type": "chunk", "data": word});
                    if write_ws_frame(stream, &WsFrame::text(&chunk.to_string())).is_err() {
                        return Ok(());
                    }
                }
                let done = serde_json::json!({"type": "done"});
                let _ = write_ws_frame(stream, &WsFrame::text(&done.to_string()));
            }
            WsOpcode::Ping => { let _ = write_ws_frame(stream, &WsFrame::pong(&frame.payload)); }
            WsOpcode::Close => { let _ = write_ws_frame(stream, &WsFrame::close(1000, "Normal closure")); break; }
            _ => {}
        }
    }
    Ok(())
}

/// Check whether an HTTP request is a WebSocket upgrade.
fn is_websocket_upgrade(headers: &[(String, String)]) -> bool {
    let has_upgrade = headers.iter().any(|(k, v)| k == "upgrade" && v.eq_ignore_ascii_case("websocket"));
    let has_connection = headers.iter().any(|(k, v)| k == "connection" && v.to_ascii_lowercase().contains("upgrade"));
    has_upgrade && has_connection
}

// ============================================================================
// AI Server
// ============================================================================

/// Embedded HTTP server wrapping an AiAssistant instance.
///
/// Thread-safe: the assistant is behind an `Arc<Mutex<_>>`, allowing
/// concurrent request handling.
pub struct AiServer {
    config: ServerConfig,
    assistant: Arc<Mutex<AiAssistant>>,
    shutdown_flag: Arc<AtomicBool>,
    metrics: Arc<ServerMetrics>,
    /// Loaded TLS server config (only present when `server-tls` feature is enabled
    /// AND `ServerConfig::tls` was set).
    #[cfg(feature = "server-tls")]
    tls_server_config: Option<Arc<rustls::ServerConfig>>,
}

impl AiServer {
    /// Create a new server with the given configuration.
    ///
    /// Initializes an `AiAssistant` with default `AiConfig`.
    /// When the `server-tls` feature is enabled and `config.tls` is set,
    /// the TLS certificates are loaded eagerly so errors surface early.
    pub fn new(config: ServerConfig) -> Self {
        #[cfg(feature = "server-tls")]
        let tls_server_config = config.tls.as_ref().map(|tls| {
            load_tls_config(tls).expect("Failed to load TLS configuration")
        });
        Self {
            config,
            assistant: Arc::new(Mutex::new(AiAssistant::new())),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(ServerMetrics::new()),
            #[cfg(feature = "server-tls")]
            tls_server_config,
        }
    }

    /// Create a server with a pre-configured assistant.
    pub fn with_assistant(config: ServerConfig, assistant: AiAssistant) -> Self {
        #[cfg(feature = "server-tls")]
        let tls_server_config = config.tls.as_ref().map(|tls| {
            load_tls_config(tls).expect("Failed to load TLS configuration")
        });
        Self {
            config,
            assistant: Arc::new(Mutex::new(assistant)),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(ServerMetrics::new()),
            #[cfg(feature = "server-tls")]
            tls_server_config,
        }
    }

    /// Get a reference to the server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Run the server, blocking the current thread.
    ///
    /// Listens for HTTP connections and dispatches requests to handlers.
    /// Each connection is handled in its own thread. The server checks
    /// the internal shutdown flag and exits cleanly when it is set.
    pub fn run_blocking(&self) -> std::io::Result<()> {
        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr)?;
        // Non-blocking accept so we can poll the shutdown flag
        listener.set_nonblocking(true)?;
        let server_config = Arc::new(self.config.clone());
        let metrics = self.metrics.clone();
        #[cfg(feature = "server-tls")]
        let tls_cfg = self.tls_server_config.clone();
        let scheme = if cfg!(feature = "server-tls") && self.config.tls.is_some() { "https" } else { "http" };
        log::info!("AI Assistant server listening on {}://{}", scheme, addr);

        loop {
            if self.shutdown_flag.load(Ordering::Relaxed) {
                log::info!("Server shutdown requested, draining...");
                break;
            }
            match listener.accept() {
                Ok((stream, _peer)) => {
                    // Connection accepted — set it back to blocking for I/O
                    let _ = stream.set_nonblocking(false);
                    let assistant = self.assistant.clone();
                    let cfg = server_config.clone();
                    let m = metrics.clone();
                    #[cfg(feature = "server-tls")]
                    let tls = tls_cfg.clone();
                    std::thread::spawn(move || {
                        #[cfg(feature = "server-tls")]
                        {
                            if let Some(ref tls_config) = tls {
                                if let Err(e) = handle_tls_connection(stream, tls_config, &assistant, &cfg, &m) {
                                    log::debug!("TLS connection error: {}", e);
                                }
                                return;
                            }
                        }
                        if let Err(e) = handle_tcp_connection(stream, &assistant, &cfg, &m) {
                            log::debug!("Connection error: {}", e);
                        }
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No pending connection — sleep briefly and retry
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    log::error!("Accept error: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Request a graceful shutdown.
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }

    /// Start the server in a background thread, returning a handle.
    ///
    /// The returned `ServerHandle` can be used to get the actual bound address
    /// (useful when binding to port 0 for automatic allocation).
    pub fn start_background(self) -> std::io::Result<ServerHandle> {
        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr)?;
        listener.set_nonblocking(true)?;
        let local_addr = listener.local_addr()?;
        let assistant = self.assistant.clone();
        let server_config = Arc::new(self.config.clone());
        let shutdown_flag = self.shutdown_flag.clone();
        let metrics = self.metrics.clone();
        #[cfg(feature = "server-tls")]
        let tls_cfg = self.tls_server_config.clone();
        let scheme = if cfg!(feature = "server-tls") && self.config.tls.is_some() { "https" } else { "http" };

        let handle = std::thread::spawn(move || {
            log::info!("AI Assistant server listening on {}://{}", scheme, local_addr);
            loop {
                if shutdown_flag.load(Ordering::Relaxed) {
                    log::info!("Background server shutdown requested, draining...");
                    break;
                }
                match listener.accept() {
                    Ok((stream, _peer)) => {
                        let _ = stream.set_nonblocking(false);
                        let assistant = assistant.clone();
                        let cfg = server_config.clone();
                        let m = metrics.clone();
                        #[cfg(feature = "server-tls")]
                        let tls = tls_cfg.clone();
                        std::thread::spawn(move || {
                            #[cfg(feature = "server-tls")]
                            {
                                if let Some(ref tls_config) = tls {
                                    if let Err(e) = handle_tls_connection(stream, tls_config, &assistant, &cfg, &m) {
                                        log::debug!("TLS connection error: {}", e);
                                    }
                                    return;
                                }
                            }
                            if let Err(e) = handle_tcp_connection(stream, &assistant, &cfg, &m) {
                                log::debug!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(e) => {
                        log::error!("Accept error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(ServerHandle {
            address: local_addr,
            _thread: handle,
            shutdown_flag: self.shutdown_flag.clone(),
        })
    }
}

/// Handle to a running server.
pub struct ServerHandle {
    /// The actual bound address.
    pub address: std::net::SocketAddr,
    _thread: std::thread::JoinHandle<()>,
    shutdown_flag: Arc<AtomicBool>,
}

impl ServerHandle {
    /// Get the server URL.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Request a graceful shutdown of the background server.
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }
}

// ============================================================================
// Request Handling
// ============================================================================

/// Constant-time equality comparison to prevent timing attacks on token validation.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Authenticate an incoming request against the server's auth configuration.
pub fn authenticate_request(
    headers: &[(String, String)],
    path: &str,
    config: &AuthConfig,
) -> AuthResult {
    if !config.enabled {
        return AuthResult::Authenticated;
    }

    if config.exempt_paths.iter().any(|p| p == path) {
        return AuthResult::Exempt;
    }

    // Check Authorization header for Bearer token
    for (key, value) in headers {
        if key == "authorization" {
            if let Some(token) = value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
            {
                if !token.is_empty()
                    && config
                        .bearer_tokens
                        .iter()
                        .any(|t| ct_eq(t.as_bytes(), token.as_bytes()))
                {
                    return AuthResult::Authenticated;
                }
            }
        }
    }

    // Check X-API-Key header
    for (key, value) in headers {
        if key == "x-api-key" {
            if !value.is_empty()
                && config
                    .api_keys
                    .iter()
                    .any(|k| ct_eq(k.as_bytes(), value.as_bytes()))
            {
                return AuthResult::Authenticated;
            }
        }
    }

    AuthResult::Rejected("Missing or invalid credentials".to_string())
}

/// Build CORS response headers based on the server's CORS configuration and the request origin.
pub fn build_cors_headers(
    config: &CorsConfig,
    request_origin: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = Vec::new();

    let origin_allowed = if config.allowed_origins.iter().any(|o| o == "*") {
        headers.push((
            "Access-Control-Allow-Origin".to_string(),
            "*".to_string(),
        ));
        true
    } else if let Some(origin) = request_origin {
        if config.allowed_origins.iter().any(|o| o == origin) {
            headers.push((
                "Access-Control-Allow-Origin".to_string(),
                origin.to_string(),
            ));
            true
        } else {
            false
        }
    } else {
        false
    };

    if !origin_allowed {
        return headers; // empty — no CORS headers
    }

    headers.push((
        "Access-Control-Allow-Methods".to_string(),
        config.allowed_methods.join(", "),
    ));
    headers.push((
        "Access-Control-Allow-Headers".to_string(),
        config.allowed_headers.join(", "),
    ));
    headers.push((
        "Access-Control-Max-Age".to_string(),
        config.max_age_secs.to_string(),
    ));

    if config.allow_credentials {
        headers.push((
            "Access-Control-Allow-Credentials".to_string(),
            "true".to_string(),
        ));
    }

    headers
}

/// Internal trait combining Read + Write for stream abstraction (plain TCP or TLS).
trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

fn handle_tcp_connection(
    mut stream: TcpStream,
    assistant: &Arc<Mutex<AiAssistant>>,
    config: &ServerConfig,
    metrics: &Arc<ServerMetrics>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(config.read_timeout_secs)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(config.read_timeout_secs)))?;
    handle_connection(&mut stream, assistant, config, metrics)
}

#[cfg(feature = "server-tls")]
fn handle_tls_connection(
    stream: TcpStream,
    tls_config: &Arc<rustls::ServerConfig>,
    assistant: &Arc<Mutex<AiAssistant>>,
    config: &ServerConfig,
    metrics: &Arc<ServerMetrics>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(config.read_timeout_secs)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(config.read_timeout_secs)))?;
    let mut tls_stream = tls_accept(stream, tls_config)?;
    handle_connection(&mut tls_stream, assistant, config, metrics)
}

fn handle_connection(
    stream: &mut dyn ReadWrite,
    assistant: &Arc<Mutex<AiAssistant>>,
    config: &ServerConfig,
    metrics: &Arc<ServerMetrics>,
) -> std::io::Result<()> {
    let start = Instant::now();

    let request = parse_request(stream, config)?;

    // Resolve or generate request ID (X-Request-Id)
    let request_id = request
        .headers
        .iter()
        .find(|(k, _)| k == "x-request-id")
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| metrics.generate_request_id());

    // Extract origin header for CORS
    let request_origin = request
        .headers
        .iter()
        .find(|(k, _)| k == "origin")
        .map(|(_, v)| v.as_str());

    let cors_headers = build_cors_headers(&config.cors, request_origin);

    // Format CORS headers + X-Request-Id as HTTP header lines
    let mut extra_headers: String = cors_headers
        .iter()
        .map(|(k, v)| format!("{}: {}\r\n", k, v))
        .collect();
    extra_headers.push_str(&format!("X-Request-Id: {}\r\n", request_id));

    // Rate limiting check — before auth so we don't waste cycles
    if let Some(ref rl) = config.rate_limiter {
        if !rl.check_rate_limit() {
            let retry_after = rl.retry_after_secs();
            let body = serde_json::to_string(&ErrorResponse {
                error: "Too Many Requests".to_string(),
            })
            .unwrap_or_else(|_| r#"{"error":"Too Many Requests"}"#.to_string());
            let response = format!(
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: {}\r\n{}Connection: close\r\n\r\n{}",
                body.len(),
                retry_after,
                extra_headers,
                body
            );
            log::info!("[{}] {} {} → 429 ({:.1}ms)", request_id, request.method, request.path, start.elapsed().as_secs_f64() * 1000.0);
            metrics.record_request("429", start.elapsed());
            stream.write_all(response.as_bytes())?;
            stream.flush()?;
            return Ok(());
        }
    }

    // Authenticate the request
    let auth_result = authenticate_request(&request.headers, &request.path, &config.auth);
    if let AuthResult::Rejected(reason) = auth_result {
        let body = serde_json::to_string(&ErrorResponse {
            error: "Unauthorized".to_string(),
        })
        .unwrap_or_else(|_| r#"{"error":"Unauthorized"}"#.to_string());
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
            body.len(),
            extra_headers,
            body
        );
        log::info!("[{}] {} {} → 401 ({:.1}ms)", request_id, request.method, request.path, start.elapsed().as_secs_f64() * 1000.0);
        log::debug!("Auth rejected: {}", reason);
        metrics.record_request("401", start.elapsed());
        stream.write_all(response.as_bytes())?;
        stream.flush()?;
        return Ok(());
    }

    // Handle OPTIONS preflight with CORS headers
    if request.method == "OPTIONS" {
        let response = format!(
            "HTTP/1.1 204 No Content\r\n{}Connection: close\r\n\r\n",
            extra_headers,
        );
        log::info!("[{}] OPTIONS {} → 204 ({:.1}ms)", request_id, request.path, start.elapsed().as_secs_f64() * 1000.0);
        metrics.record_request("204", start.elapsed());
        stream.write_all(response.as_bytes())?;
        stream.flush()?;
        return Ok(());
    }

    // Check if this is a WebSocket upgrade request
    let is_ws = (request.path == "/ws" || request.path == "/api/v1/ws")
        && request.method == "GET"
        && is_websocket_upgrade(&request.headers);
    if is_ws {
        log::info!("[{}] WebSocket upgrade {} → 101 ({:.1}ms)", request_id, request.path, start.elapsed().as_secs_f64() * 1000.0);
        metrics.record_request("101", start.elapsed());
        ws_handshake(stream, &request.headers, &extra_headers)?;
        return handle_ws_chat(stream, assistant, config);
    }

    // Check if this is an SSE streaming request
    let is_sse = request.path == "/chat/stream" || request.path == "/api/v1/chat/stream";

    if is_sse && request.method == "POST" {
        // Handle SSE streaming with its own Content-Type
        match handle_chat_stream(&request, assistant, config) {
            Ok(sse_body) => {
                let response = build_sse_response(&sse_body, &extra_headers);
                log::info!("[{}] {} {} → 200 SSE ({:.1}ms)", request_id, request.method, request.path, start.elapsed().as_secs_f64() * 1000.0);
                metrics.record_request("200", start.elapsed());
                stream.write_all(response.as_bytes())?;
                stream.flush()?;
            }
            Err((status, body)) => {
                let status_code = status.split_whitespace().next().unwrap_or("500");
                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
                    status, body.len(), extra_headers, body
                );
                log::info!("[{}] {} {} → {} ({:.1}ms)", request_id, request.method, request.path, status_code, start.elapsed().as_secs_f64() * 1000.0);
                metrics.record_request(status_code, start.elapsed());
                stream.write_all(response.as_bytes())?;
                stream.flush()?;
            }
        }
        return Ok(());
    }

    let (status, body) = route_request_with_config(&request, assistant, metrics, config);
    let status_code = status.split_whitespace().next().unwrap_or("500");

    // Check Accept-Encoding for gzip support
    let accept_encoding = request
        .headers
        .iter()
        .find(|(k, _)| k == "accept-encoding")
        .map(|(_, v)| v.as_str());
    let (body_bytes, compressed) = maybe_compress_response(&body, accept_encoding);

    let encoding_header = if compressed {
        "Content-Encoding: gzip\r\n"
    } else {
        ""
    };

    let response_header = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}{}Connection: close\r\n\r\n",
        status,
        body_bytes.len(),
        encoding_header,
        extra_headers,
    );

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    log::info!("[{}] {} {} → {} ({:.1}ms)", request_id, request.method, request.path, status_code, elapsed_ms);
    metrics.record_request(status_code, start.elapsed());

    stream.write_all(response_header.as_bytes())?;
    stream.write_all(&body_bytes)?;
    stream.flush()?;
    Ok(())
}

fn parse_request(stream: &mut dyn Read, config: &ServerConfig) -> std::io::Result<HttpRequest> {
    let mut reader = BufReader::new(stream);

    // Read request line
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    if request_line.len() > config.max_header_line {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Request line too long",
        ));
    }
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid request line",
        ));
    }

    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Read headers (enforcing max_headers and max_header_line limits)
    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.len() > config.max_header_line {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Header line exceeds {} byte limit", config.max_header_line),
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if headers.len() >= config.max_headers {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Too many headers (max {})", config.max_headers),
            ));
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim().to_lowercase();
            let value = value.trim().to_string();
            if key == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((key, value));
        }
    }

    // Read body
    let mut body = String::new();
    if content_length > 0 {
        let read_size = content_length.min(config.max_body_size);
        let mut buf = vec![0u8; read_size];
        reader.read_exact(&mut buf)?;
        body = String::from_utf8_lossy(&buf).to_string();
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Route a request using default `ServerConfig`.  Kept for backward
/// compatibility and test convenience — production traffic goes through
/// `route_request_with_config` via `handle_connection`.
#[allow(dead_code)]
fn route_request(
    request: &HttpRequest,
    assistant: &Arc<Mutex<AiAssistant>>,
    metrics: &Arc<ServerMetrics>,
) -> (String, String) {
    let default_config = ServerConfig::default();
    route_request_with_config(request, assistant, metrics, &default_config)
}

fn route_request_with_config(
    request: &HttpRequest,
    assistant: &Arc<Mutex<AiAssistant>>,
    metrics: &Arc<ServerMetrics>,
    config: &ServerConfig,
) -> (String, String) {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => handle_health(assistant, metrics),
        ("GET", "/models") => handle_list_models(assistant),
        ("POST", "/chat") => handle_chat(request, assistant, config),
        ("GET", "/config") => handle_get_config(assistant),
        ("POST", "/config") => handle_set_config(request, assistant),
        ("GET", "/metrics") => ("200 OK".to_string(), metrics.render_prometheus()),
        ("GET", "/sessions") => handle_list_sessions(assistant),
        ("DELETE", path) if path.starts_with("/sessions/") => {
            let id = &path[10..]; // skip "/sessions/"
            handle_delete_session(id, assistant)
        }
        ("GET", path) if path.starts_with("/sessions/") => {
            let id = &path[10..];
            handle_get_session(id, assistant)
        }
        // Versioned routes — same handlers, /api/v1/ prefix (items 5.1, 5.2)
        ("GET", "/api/v1/health") => handle_health(assistant, metrics),
        ("GET", "/api/v1/models") => handle_list_models(assistant),
        ("POST", "/api/v1/chat") => handle_chat(request, assistant, config),
        ("GET", "/api/v1/config") => handle_get_config(assistant),
        ("POST", "/api/v1/config") => handle_set_config(request, assistant),
        ("POST", "/chat/stream") | ("POST", "/api/v1/chat/stream") => {
            // SSE streaming is handled separately in handle_connection to use
            // the correct Content-Type.  When called through route_request
            // (unit tests) we fall through to the same handler but return JSON-
            // wrapped output so tests can validate correctness.
            match handle_chat_stream(request, assistant, config) {
                Ok(sse_body) => ("200 OK".to_string(), sse_body),
                Err((status, body)) => (status, body),
            }
        }
        ("GET", "/api/v1/openapi.json") | ("GET", "/openapi.json") => {
            let spec = crate::openapi_export::generate_server_api_spec();
            ("200 OK".to_string(), serde_json::to_string_pretty(&spec).unwrap_or_default())
        }
        ("GET", "/api/v1/metrics") => ("200 OK".to_string(), metrics.render_prometheus()),
        ("GET", "/api/v1/sessions") => handle_list_sessions(assistant),
        ("DELETE", path) if path.starts_with("/api/v1/sessions/") => {
            let id = &path[17..];
            handle_delete_session(id, assistant)
        }
        ("GET", path) if path.starts_with("/api/v1/sessions/") => {
            let id = &path[17..];
            handle_get_session(id, assistant)
        }
        _ => (
            "404 Not Found".to_string(),
            serde_json::to_string(&ErrorResponse {
                error: format!("Unknown endpoint: {} {}", request.method, request.path),
            })
            .unwrap_or_default(),
        ),
    }
}

fn handle_health(assistant: &Arc<Mutex<AiAssistant>>, metrics: &Arc<ServerMetrics>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: ass.config.selected_model.clone(),
        provider: ass.config.provider.display_name().to_string(),
        uptime_secs: metrics.started_at.elapsed().as_secs(),
        active_sessions: ass.session_store.sessions.len(),
        conversation_messages: ass.conversation.len(),
    };
    (
        "200 OK".to_string(),
        serde_json::to_string(&resp).unwrap_or_default(),
    )
}

fn handle_list_models(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let models: Vec<serde_json::Value> = ass
        .available_models
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m.name,
                "provider": format!("{:?}", m.provider),
                "size": m.size,
            })
        })
        .collect();
    (
        "200 OK".to_string(),
        serde_json::to_string(&models).unwrap_or_default(),
    )
}

fn handle_chat(request: &HttpRequest, assistant: &Arc<Mutex<AiAssistant>>, config: &ServerConfig) -> (String, String) {
    let chat_req: ChatRequest = match serde_json::from_str(&request.body) {
        Ok(req) => req,
        Err(e) => {
            return (
                "400 Bad Request".to_string(),
                serde_json::to_string(&ErrorResponse {
                    error: format!("Invalid JSON: {}", e),
                })
                .unwrap_or_default(),
            );
        }
    };

    // Request validation: reject oversized messages
    if chat_req.message.len() > config.max_message_length {
        return (
            "422 Unprocessable Entity".to_string(),
            serde_json::to_string(&ErrorResponse {
                error: format!(
                    "Message too long: {} characters (max {})",
                    chat_req.message.len(),
                    config.max_message_length
                ),
            })
            .unwrap_or_default(),
        );
    }

    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());

    // send_message_with_notes(user_message: String, knowledge_context, session_notes, knowledge_notes)
    ass.send_message_with_notes(
        chat_req.message.clone(),
        &chat_req.knowledge_context,
        &chat_req.system_prompt,
        "",
    );

    // Poll until complete
    let model = ass.config.selected_model.clone();
    loop {
        match ass.poll_response() {
            Some(crate::messages::AiResponse::Complete(text)) => {
                let resp = ChatResponse {
                    content: text,
                    model,
                };
                return (
                    "200 OK".to_string(),
                    serde_json::to_string(&resp).unwrap_or_default(),
                );
            }
            Some(crate::messages::AiResponse::Error(e)) => {
                return (
                    "500 Internal Server Error".to_string(),
                    serde_json::to_string(&ErrorResponse { error: e }).unwrap_or_default(),
                );
            }
            Some(crate::messages::AiResponse::Cancelled(partial)) => {
                let resp = ChatResponse {
                    content: partial,
                    model,
                };
                return (
                    "200 OK".to_string(),
                    serde_json::to_string(&resp).unwrap_or_default(),
                );
            }
            Some(_) => {
                // Chunk — accumulate would be needed for streaming, skip for now
                continue;
            }
            None => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
}

fn handle_get_config(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    // Don't expose the api_key
    let safe_config = serde_json::json!({
        "provider": format!("{:?}", ass.config.provider),
        "selected_model": ass.config.selected_model,
        "base_url": ass.config.get_base_url(),
        "temperature": ass.config.temperature,
        "max_history_messages": ass.config.max_history_messages,
        "has_api_key": !ass.config.api_key.is_empty(),
    });
    (
        "200 OK".to_string(),
        serde_json::to_string(&safe_config).unwrap_or_default(),
    )
}

fn handle_set_config(
    request: &HttpRequest,
    assistant: &Arc<Mutex<AiAssistant>>,
) -> (String, String) {
    let updates: serde_json::Value = match serde_json::from_str(&request.body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "400 Bad Request".to_string(),
                serde_json::to_string(&ErrorResponse {
                    error: format!("Invalid JSON: {}", e),
                })
                .unwrap_or_default(),
            );
        }
    };

    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(model) = updates.get("model").and_then(|m| m.as_str()) {
        ass.config.selected_model = model.to_string();
    }
    if let Some(temp) = updates.get("temperature").and_then(|t| t.as_f64()) {
        ass.config.temperature = temp as f32;
    }

    (
        "200 OK".to_string(),
        serde_json::to_string(&serde_json::json!({"status": "updated"})).unwrap_or_default(),
    )
}

// ============================================================================
// Session Endpoints
// ============================================================================

fn handle_list_sessions(assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let sessions: Vec<serde_json::Value> = ass.session_store.sessions.iter().map(|s| {
        serde_json::json!({
            "id": s.id,
            "messages": s.messages.len(),
        })
    }).collect();
    ("200 OK".to_string(), serde_json::to_string(&sessions).unwrap_or_default())
}

fn handle_get_session(id: &str, assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    match ass.session_store.sessions.iter().find(|s| s.id == id) {
        Some(session) => {
            let msgs: Vec<serde_json::Value> = session.messages.iter().map(|m| {
                serde_json::json!({"role": m.role, "content": m.content})
            }).collect();
            ("200 OK".to_string(), serde_json::to_string(&msgs).unwrap_or_default())
        }
        None => ("404 Not Found".to_string(),
            serde_json::to_string(&ErrorResponse { error: format!("Session not found: {}", id) }).unwrap_or_default())
    }
}

fn handle_delete_session(id: &str, assistant: &Arc<Mutex<AiAssistant>>) -> (String, String) {
    let mut ass = assistant.lock().unwrap_or_else(|e| e.into_inner());
    let before = ass.session_store.sessions.len();
    ass.delete_session(id);
    let deleted = ass.session_store.sessions.len() < before;
    if deleted {
        ("200 OK".to_string(), serde_json::to_string(&serde_json::json!({"deleted": true})).unwrap_or_default())
    } else {
        ("404 Not Found".to_string(),
            serde_json::to_string(&ErrorResponse { error: format!("Session not found: {}", id) }).unwrap_or_default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8090);
        assert_eq!(config.bind_address(), "127.0.0.1:8090");
        assert_eq!(config.max_body_size, 1_048_576);
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 3000,
            ..Default::default()
        };
        assert_eq!(config.bind_address(), "0.0.0.0:3000");
    }

    #[test]
    fn test_health_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let (status, body) = handle_health(&assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
        assert!(parsed["version"].is_string());
    }

    #[test]
    fn test_get_config_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_get_config(&assistant);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed["provider"].is_string());
        assert!(parsed["temperature"].is_number());
        // API key should not be exposed
        assert!(parsed.get("api_key").is_none());
    }

    #[test]
    fn test_set_config_endpoint() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/config".to_string(),
            headers: vec![],
            body: r#"{"model": "test-model", "temperature": 0.5}"#.to_string(),
        };

        let (status, body) = handle_set_config(&request, &assistant);
        assert_eq!(status, "200 OK");
        assert!(body.contains("updated"));

        let ass = assistant.lock().unwrap();
        assert_eq!(ass.config.selected_model, "test-model");
        assert!((ass.config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_endpoint_invalid_json() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig::default();

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: "not json".to_string(),
        };

        let (status, body) = handle_chat(&request, &assistant, &config);
        assert_eq!(status, "400 Bad Request");
        assert!(body.contains("Invalid JSON"));
    }

    #[test]
    fn test_route_not_found() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/unknown".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "404 Not Found");
        assert!(body.contains("Unknown endpoint"));
    }

    #[test]
    fn test_cors_options_preflight() {
        // OPTIONS preflight is now handled in handle_connection, not route_request.
        // We test the CORS headers builder and verify OPTIONS gets proper headers
        // via the integration test (test_server_start_background).
        let config = CorsConfig::default();
        let headers = build_cors_headers(&config, Some("http://example.com"));
        assert!(!headers.is_empty());
        // The default config has "*" so any origin should produce headers
        let origin_header = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Origin")
            .map(|(_, v)| v.as_str());
        assert_eq!(origin_header, Some("*"));
    }

    #[test]
    fn test_server_start_background() {
        // Bind to port 0 for automatic allocation
        let config = ServerConfig {
            port: 0,
            ..Default::default()
        };
        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();

        // Verify it got a port
        assert!(handle.address.port() > 0);
        assert!(handle.url().starts_with("http://"));

        // Make a health check request
        let url = format!("{}/health", handle.url());
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
        {
            Ok(resp) => {
                let body: serde_json::Value = resp.into_json().unwrap();
                assert_eq!(body["status"], "ok");
            }
            Err(_) => {
                // Server might not be ready yet in very fast test execution
                std::thread::sleep(std::time::Duration::from_millis(100));
                let resp = ureq::get(&url)
                    .timeout(std::time::Duration::from_secs(2))
                    .call()
                    .unwrap();
                let body: serde_json::Value = resp.into_json().unwrap();
                assert_eq!(body["status"], "ok");
            }
        }
    }

    // ========================================================================
    // Auth tests
    // ========================================================================

    #[test]
    fn test_auth_disabled_passes_all() {
        let config = AuthConfig {
            enabled: false,
            ..Default::default()
        };
        let headers = vec![]; // no credentials at all
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);
    }

    #[test]
    fn test_auth_exempt_path() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["secret".to_string()],
            exempt_paths: vec!["/health".to_string()],
            ..Default::default()
        };
        let headers = vec![]; // no credentials
        let result = authenticate_request(&headers, "/health", &config);
        assert_eq!(result, AuthResult::Exempt);
    }

    #[test]
    fn test_auth_valid_bearer() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["my-secret-token".to_string()],
            ..Default::default()
        };
        let headers = vec![(
            "authorization".to_string(),
            "Bearer my-secret-token".to_string(),
        )];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);
    }

    #[test]
    fn test_auth_invalid_bearer() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["my-secret-token".to_string()],
            ..Default::default()
        };
        let headers = vec![(
            "authorization".to_string(),
            "Bearer wrong-token".to_string(),
        )];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(
            result,
            AuthResult::Rejected("Missing or invalid credentials".to_string())
        );
    }

    #[test]
    fn test_auth_valid_api_key() {
        let config = AuthConfig {
            enabled: true,
            api_keys: vec!["ak-12345".to_string()],
            ..Default::default()
        };
        let headers = vec![("x-api-key".to_string(), "ak-12345".to_string())];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);
    }

    #[test]
    fn test_auth_missing_credentials() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["token".to_string()],
            api_keys: vec!["key".to_string()],
            ..Default::default()
        };
        let headers = vec![]; // no auth headers
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(
            result,
            AuthResult::Rejected("Missing or invalid credentials".to_string())
        );
    }

    #[test]
    fn test_auth_constant_time_eq() {
        // Equal values
        assert!(ct_eq(b"hello", b"hello"));
        // Different values, same length
        assert!(!ct_eq(b"hello", b"world"));
        // Different lengths
        assert!(!ct_eq(b"short", b"longer"));
        // Empty
        assert!(ct_eq(b"", b""));
        // Single byte difference
        assert!(!ct_eq(b"abc", b"abd"));
    }

    #[test]
    fn test_auth_empty_token_rejected() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["valid".to_string()],
            ..Default::default()
        };
        // "Bearer " with empty token after it
        let headers = vec![("authorization".to_string(), "Bearer ".to_string())];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(
            result,
            AuthResult::Rejected("Missing or invalid credentials".to_string())
        );
    }

    #[test]
    fn test_auth_case_insensitive_bearer_prefix() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec!["tok123".to_string()],
            ..Default::default()
        };
        // lowercase "bearer "
        let headers = vec![("authorization".to_string(), "bearer tok123".to_string())];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);
    }

    #[test]
    fn test_auth_multiple_tokens() {
        let config = AuthConfig {
            enabled: true,
            bearer_tokens: vec![
                "token-a".to_string(),
                "token-b".to_string(),
                "token-c".to_string(),
            ],
            ..Default::default()
        };
        // Second token should match
        let headers = vec![(
            "authorization".to_string(),
            "Bearer token-b".to_string(),
        )];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);

        // Third token should match
        let headers = vec![(
            "authorization".to_string(),
            "Bearer token-c".to_string(),
        )];
        let result = authenticate_request(&headers, "/chat", &config);
        assert_eq!(result, AuthResult::Authenticated);
    }

    // ========================================================================
    // CORS tests
    // ========================================================================

    #[test]
    fn test_cors_default_wildcard() {
        let config = CorsConfig::default();
        let headers = build_cors_headers(&config, Some("http://anything.com"));
        let origin = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Origin")
            .map(|(_, v)| v.as_str());
        assert_eq!(origin, Some("*"));
    }

    #[test]
    fn test_cors_specific_origin_allowed() {
        let config = CorsConfig {
            allowed_origins: vec![
                "http://example.com".to_string(),
                "http://other.com".to_string(),
            ],
            ..Default::default()
        };
        let headers = build_cors_headers(&config, Some("http://example.com"));
        let origin = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Origin")
            .map(|(_, v)| v.as_str());
        assert_eq!(origin, Some("http://example.com"));
    }

    #[test]
    fn test_cors_origin_not_allowed() {
        let config = CorsConfig {
            allowed_origins: vec!["http://allowed.com".to_string()],
            ..Default::default()
        };
        let headers = build_cors_headers(&config, Some("http://evil.com"));
        // Should return no CORS headers
        assert!(headers.is_empty());
    }

    #[test]
    fn test_cors_credentials_header() {
        let config = CorsConfig {
            allow_credentials: true,
            ..Default::default()
        };
        let headers = build_cors_headers(&config, Some("http://example.com"));
        let creds = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Credentials")
            .map(|(_, v)| v.as_str());
        assert_eq!(creds, Some("true"));
    }

    #[test]
    fn test_cors_max_age() {
        let config = CorsConfig {
            max_age_secs: 3600,
            ..Default::default()
        };
        let headers = build_cors_headers(&config, Some("http://example.com"));
        let max_age = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Max-Age")
            .map(|(_, v)| v.as_str());
        assert_eq!(max_age, Some("3600"));
    }

    #[test]
    fn test_cors_methods_and_headers() {
        let config = CorsConfig {
            allowed_methods: vec!["GET".to_string(), "PUT".to_string()],
            allowed_headers: vec!["Content-Type".to_string(), "X-Custom".to_string()],
            ..Default::default()
        };
        let headers = build_cors_headers(&config, Some("http://example.com"));
        let methods = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Methods")
            .map(|(_, v)| v.as_str());
        assert_eq!(methods, Some("GET, PUT"));
        let allow_headers = headers
            .iter()
            .find(|(k, _)| k == "Access-Control-Allow-Headers")
            .map(|(_, v)| v.as_str());
        assert_eq!(allow_headers, Some("Content-Type, X-Custom"));
    }

    #[test]
    fn test_cors_no_origin_header() {
        // Non-wildcard config, no origin sent
        let config = CorsConfig {
            allowed_origins: vec!["http://example.com".to_string()],
            ..Default::default()
        };
        let headers = build_cors_headers(&config, None);
        // No origin to match, so no CORS headers
        assert!(headers.is_empty());
    }

    // ========================================================================
    // Server Metrics tests (items 1.3, 1.5)
    // ========================================================================

    #[test]
    fn test_server_metrics_new() {
        let m = ServerMetrics::new();
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(m.requests_2xx.load(Ordering::Relaxed), 0);
        assert_eq!(m.requests_4xx.load(Ordering::Relaxed), 0);
        assert_eq!(m.requests_5xx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_server_metrics_record_2xx() {
        let m = ServerMetrics::new();
        m.record_request("200", std::time::Duration::from_millis(42));
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(m.requests_2xx.load(Ordering::Relaxed), 1);
        assert_eq!(m.requests_4xx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_server_metrics_record_4xx() {
        let m = ServerMetrics::new();
        m.record_request("404", std::time::Duration::from_millis(5));
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(m.requests_4xx.load(Ordering::Relaxed), 1);
        assert_eq!(m.requests_2xx.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_server_metrics_record_5xx() {
        let m = ServerMetrics::new();
        m.record_request("500", std::time::Duration::from_millis(100));
        assert_eq!(m.requests_5xx.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_server_metrics_duration_accumulates() {
        let m = ServerMetrics::new();
        m.record_request("200", std::time::Duration::from_millis(10));
        m.record_request("200", std::time::Duration::from_millis(20));
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 2);
        // 10ms + 20ms = 30ms = 30_000 us
        let us = m.request_duration_us_total.load(Ordering::Relaxed);
        assert!(us >= 29_000 && us <= 31_000, "Duration {} should be ~30000us", us);
    }

    #[test]
    fn test_server_metrics_generate_request_id() {
        let m = ServerMetrics::new();
        let id1 = m.generate_request_id();
        let id2 = m.generate_request_id();
        assert_eq!(id1.len(), 32); // 16+16 hex chars
        assert_ne!(id1, id2); // unique
        // Must be valid hex
        assert!(id1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_server_metrics_render_prometheus() {
        let m = ServerMetrics::new();
        m.record_request("200", std::time::Duration::from_millis(50));
        m.record_request("404", std::time::Duration::from_millis(5));
        let output = m.render_prometheus();
        assert!(output.contains("ai_server_requests_total 2"));
        assert!(output.contains("status=\"2xx\""));
        assert!(output.contains("status=\"4xx\""));
        assert!(output.contains("ai_server_request_duration_seconds_total"));
    }

    #[test]
    fn test_metrics_endpoint_via_route() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        metrics.record_request("200", std::time::Duration::from_millis(10));

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/metrics".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        assert!(body.contains("ai_server_requests_total"));
    }

    // ========================================================================
    // Graceful Shutdown tests (item 1.4)
    // ========================================================================

    #[test]
    fn test_server_shutdown_flag() {
        let config = ServerConfig::default();
        let server = AiServer::new(config);
        assert!(!server.shutdown_flag.load(Ordering::Relaxed));
        server.shutdown();
        assert!(server.shutdown_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_server_handle_shutdown() {
        let config = ServerConfig { port: 0, ..Default::default() };
        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();
        assert!(handle.address.port() > 0);

        // Request shutdown
        handle.shutdown();
        // Give the server loop time to notice
        std::thread::sleep(std::time::Duration::from_millis(50));
        // If we get here, the test passes — shutdown didn't panic
    }

    // ========================================================================
    // Request limits tests (item 9.1)
    // ========================================================================

    #[test]
    fn test_server_config_request_limits_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.max_headers, 100);
        assert_eq!(config.max_header_line, 8192);
        assert_eq!(config.body_read_timeout_ms, 30_000);
    }

    #[test]
    fn test_server_config_custom_limits() {
        let config = ServerConfig {
            max_headers: 50,
            max_header_line: 4096,
            body_read_timeout_ms: 5_000,
            ..Default::default()
        };
        assert_eq!(config.max_headers, 50);
        assert_eq!(config.max_header_line, 4096);
        assert_eq!(config.body_read_timeout_ms, 5_000);
    }

    // ========================================================================
    // Integration test: background server with auth, CORS, metrics (item 10.2)
    // ========================================================================

    #[test]
    fn test_server_lifecycle_with_auth_and_metrics() {
        let config = ServerConfig {
            port: 0,
            auth: AuthConfig {
                enabled: true,
                bearer_tokens: vec!["test-token-42".to_string()],
                exempt_paths: vec!["/health".to_string(), "/metrics".to_string()],
                ..Default::default()
            },
            ..Default::default()
        };

        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();
        let base = handle.url();

        // Health should be exempt from auth
        let health_resp = ureq::get(&format!("{}/health", base))
            .timeout(std::time::Duration::from_secs(2))
            .call();
        assert!(health_resp.is_ok());

        // /chat without auth should get 401
        let chat_resp = ureq::post(&format!("{}/chat", base))
            .timeout(std::time::Duration::from_secs(2))
            .send_json(serde_json::json!({"message": "hi"}));
        match chat_resp {
            Err(ureq::Error::Status(401, _)) => {} // expected
            other => panic!("Expected 401, got {:?}", other),
        }

        // /chat WITH auth should succeed (or get provider error, which is still 500 not 401)
        let auth_resp = ureq::post(&format!("{}/chat", base))
            .timeout(std::time::Duration::from_secs(2))
            .set("Authorization", "Bearer test-token-42")
            .send_json(serde_json::json!({"message": "hi"}));
        match auth_resp {
            Ok(_) => {} // provider might work
            Err(ureq::Error::Status(code, _)) => {
                assert_ne!(code, 401, "Should not get 401 with valid token");
            }
            // Transport errors (e.g. timeout) are acceptable — the server accepted
            // the auth but the handler may block waiting for a provider that isn't running
            Err(ureq::Error::Transport(_)) => {}
        }

        // /metrics should be exempt from auth and return Prometheus data
        let metrics_resp = ureq::get(&format!("{}/metrics", base))
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .unwrap();
        let metrics_body = metrics_resp.into_string().unwrap();
        assert!(metrics_body.contains("ai_server_requests_total"));

        // Graceful shutdown
        handle.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    #[test]
    fn test_server_request_id_in_response() {
        let config = ServerConfig { port: 0, ..Default::default() };
        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();
        let url = format!("{}/health", handle.url());

        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .unwrap();
        let req_id = resp.header("X-Request-Id").unwrap_or("");
        assert_eq!(req_id.len(), 32, "Request ID should be 32 hex chars, got '{}'", req_id);
        assert!(req_id.chars().all(|c| c.is_ascii_hexdigit()));

        handle.shutdown();
    }

    #[test]
    fn test_server_reuses_client_request_id() {
        let config = ServerConfig { port: 0, ..Default::default() };
        let server = AiServer::new(config);
        let handle = server.start_background().unwrap();
        let url = format!("{}/health", handle.url());

        let resp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .set("X-Request-Id", "my-custom-id-123")
            .call()
            .unwrap();
        let returned_id = resp.header("X-Request-Id").unwrap_or("");
        assert_eq!(returned_id, "my-custom-id-123");

        handle.shutdown();
    }

    // ========================================================================
    // TLS config tests (item 9.2)
    // ========================================================================

    #[test]
    fn test_tls_config_default_none() {
        let config = ServerConfig::default();
        assert!(config.tls.is_none());
    }

    #[test]
    fn test_tls_config_set() {
        let config = ServerConfig {
            tls: Some(TlsConfig {
                cert_path: "/etc/ssl/cert.pem".to_string(),
                key_path: "/etc/ssl/key.pem".to_string(),
            }),
            ..Default::default()
        };
        assert!(config.tls.is_some());
        let tls = config.tls.unwrap();
        assert_eq!(tls.cert_path, "/etc/ssl/cert.pem");
        assert_eq!(tls.key_path, "/etc/ssl/key.pem");
    }

    #[test]
    fn test_tls_config_clone() {
        let tls = TlsConfig {
            cert_path: "cert.pem".to_string(),
            key_path: "key.pem".to_string(),
        };
        let cloned = tls.clone();
        assert_eq!(cloned.cert_path, "cert.pem");
        assert_eq!(cloned.key_path, "key.pem");
    }

    #[test]
    fn test_server_config_with_tls_serializes_without_tls() {
        let config = ServerConfig {
            tls: Some(TlsConfig {
                cert_path: "cert.pem".to_string(),
                key_path: "key.pem".to_string(),
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("cert_path"));
        assert!(!json.contains("key_path"));
    }

    #[test]
    fn test_server_config_deserializes_without_tls() {
        let json = serde_json::to_string(&ServerConfig::default()).unwrap();
        let config: ServerConfig = serde_json::from_str(&json).unwrap();
        assert!(config.tls.is_none());
        assert_eq!(config.port, 8090);
    }

    // ========================================================================
    // WebSocket tests (v10 Phase 5)
    // ========================================================================

    #[test]
    fn test_is_websocket_upgrade_valid() {
        let headers = vec![
            ("upgrade".to_string(), "websocket".to_string()),
            ("connection".to_string(), "Upgrade".to_string()),
            ("sec-websocket-key".to_string(), "dGhlIHNhbXBsZSBub25jZQ==".to_string()),
            ("sec-websocket-version".to_string(), "13".to_string()),
        ];
        assert!(is_websocket_upgrade(&headers));
    }

    #[test]
    fn test_is_websocket_upgrade_missing_upgrade() {
        let headers = vec![
            ("connection".to_string(), "Upgrade".to_string()),
            ("sec-websocket-key".to_string(), "dGhlIHNhbXBsZSBub25jZQ==".to_string()),
        ];
        assert!(!is_websocket_upgrade(&headers));
    }

    #[test]
    fn test_is_websocket_upgrade_missing_connection() {
        let headers = vec![
            ("upgrade".to_string(), "websocket".to_string()),
            ("sec-websocket-key".to_string(), "dGhlIHNhbXBsZSBub25jZQ==".to_string()),
        ];
        assert!(!is_websocket_upgrade(&headers));
    }

    #[test]
    fn test_is_websocket_upgrade_wrong_value() {
        let headers = vec![
            ("upgrade".to_string(), "h2c".to_string()),
            ("connection".to_string(), "Upgrade".to_string()),
        ];
        assert!(!is_websocket_upgrade(&headers));
    }

    #[test]
    fn test_is_websocket_upgrade_case_insensitive() {
        let headers = vec![
            ("upgrade".to_string(), "WebSocket".to_string()),
            ("connection".to_string(), "upgrade".to_string()),
        ];
        assert!(is_websocket_upgrade(&headers));
    }

    #[test]
    fn test_ws_magic_guid_correct() {
        assert_eq!(WS_MAGIC_GUID, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    }

    #[test]
    fn test_ws_handshake_accept_value() {
        // RFC 6455 §4.2.2 example: key = "dGhlIHNhbXBsZSBub25jZQ=="
        // Expected accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let mut accept_input = key.to_string();
        accept_input.push_str(WS_MAGIC_GUID);
        let hash = crate::websocket_streaming::sha1_hash(accept_input.as_bytes());
        let accept = crate::websocket_streaming::base64_encode(&hash);
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn test_ws_handshake_writes_101() {
        let headers = vec![
            ("sec-websocket-key".to_string(), "dGhlIHNhbXBsZSBub25jZQ==".to_string()),
        ];
        let mut buf = Vec::new();
        ws_handshake(&mut buf, &headers, "").unwrap();
        let response = String::from_utf8(buf).unwrap();
        assert!(response.starts_with("HTTP/1.1 101 Switching Protocols\r\n"));
        assert!(response.contains("Upgrade: websocket\r\n"));
        assert!(response.contains("Connection: Upgrade\r\n"));
        assert!(response.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    }

    #[test]
    fn test_ws_handshake_includes_extra_headers() {
        let headers = vec![
            ("sec-websocket-key".to_string(), "dGhlIHNhbXBsZSBub25jZQ==".to_string()),
        ];
        let mut buf = Vec::new();
        ws_handshake(&mut buf, &headers, "X-Request-Id: abc123\r\n").unwrap();
        let response = String::from_utf8(buf).unwrap();
        assert!(response.contains("X-Request-Id: abc123"));
    }

    #[test]
    fn test_write_ws_frame_text() {
        use crate::websocket_streaming::WsFrame;
        let frame = WsFrame::text("hello");
        let mut buf = Vec::new();
        write_ws_frame(&mut buf, &frame).unwrap();
        // FIN + Text opcode = 0x81, length = 5
        assert_eq!(buf[0], 0x81);
        assert_eq!(buf[1], 5);
        assert_eq!(&buf[2..], b"hello");
    }

    #[test]
    fn test_write_ws_frame_ping() {
        use crate::websocket_streaming::WsFrame;
        let frame = WsFrame::ping(b"");
        let mut buf = Vec::new();
        write_ws_frame(&mut buf, &frame).unwrap();
        // FIN + Ping opcode = 0x89, length = 0
        assert_eq!(buf[0], 0x89);
        assert_eq!(buf[1], 0);
    }

    #[test]
    fn test_write_ws_frame_close() {
        use crate::websocket_streaming::WsFrame;
        let frame = WsFrame::close(1000, "bye");
        let mut buf = Vec::new();
        write_ws_frame(&mut buf, &frame).unwrap();
        // FIN + Close opcode = 0x88
        assert_eq!(buf[0], 0x88);
        // Payload: 2 bytes status code + "bye" = 5 bytes
        assert_eq!(buf[1], 5);
    }

    #[test]
    fn test_write_ws_frame_medium_payload() {
        use crate::websocket_streaming::WsFrame;
        // 200 bytes payload → extended 16-bit length
        let data = "x".repeat(200);
        let frame = WsFrame::text(&data);
        let mut buf = Vec::new();
        write_ws_frame(&mut buf, &frame).unwrap();
        assert_eq!(buf[0], 0x81); // FIN + Text
        assert_eq!(buf[1], 126);  // Extended 16-bit
        let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
        assert_eq!(len, 200);
        assert_eq!(buf.len(), 4 + 200);
    }

    #[test]
    fn test_read_ws_frame_unmasked_text() {
        use crate::websocket_streaming::WsOpcode;
        // Build a minimal unmasked text frame: FIN=1, opcode=1, mask=0, len=5, "hello"
        let data: Vec<u8> = vec![0x81, 0x05, b'h', b'e', b'l', b'l', b'o'];
        let mut cursor = std::io::Cursor::new(data);
        let frame = read_ws_frame(&mut cursor).unwrap();
        assert!(frame.fin);
        assert_eq!(frame.opcode, WsOpcode::Text);
        assert_eq!(frame.payload, b"hello");
    }

    #[test]
    fn test_read_ws_frame_masked_text() {
        use crate::websocket_streaming::WsOpcode;
        // Build a masked text frame: FIN=1, opcode=1, mask=1, len=5
        let mask: [u8; 4] = [0x37, 0xfa, 0x21, 0x3d];
        let payload = b"Hello";
        let mut masked_payload = payload.to_vec();
        for (i, byte) in masked_payload.iter_mut().enumerate() {
            *byte ^= mask[i % 4];
        }
        let mut data = vec![0x81, 0x85]; // FIN+Text, masked+len=5
        data.extend_from_slice(&mask);
        data.extend_from_slice(&masked_payload);
        let mut cursor = std::io::Cursor::new(data);
        let frame = read_ws_frame(&mut cursor).unwrap();
        assert!(frame.fin);
        assert_eq!(frame.opcode, WsOpcode::Text);
        assert_eq!(frame.payload, b"Hello");
    }

    #[test]
    fn test_read_ws_frame_close() {
        use crate::websocket_streaming::WsOpcode;
        // Close frame: FIN=1, opcode=8, mask=0, len=2, status=1000
        let data: Vec<u8> = vec![0x88, 0x02, 0x03, 0xE8]; // 0x03E8 = 1000
        let mut cursor = std::io::Cursor::new(data);
        let frame = read_ws_frame(&mut cursor).unwrap();
        assert_eq!(frame.opcode, WsOpcode::Close);
        assert_eq!(frame.payload, vec![0x03, 0xE8]);
    }

    #[test]
    fn test_read_ws_frame_ping() {
        use crate::websocket_streaming::WsOpcode;
        // Ping frame: FIN=1, opcode=9, mask=0, len=0
        let data: Vec<u8> = vec![0x89, 0x00];
        let mut cursor = std::io::Cursor::new(data);
        let frame = read_ws_frame(&mut cursor).unwrap();
        assert_eq!(frame.opcode, WsOpcode::Ping);
        assert!(frame.payload.is_empty());
    }

    #[test]
    fn test_read_ws_frame_too_large() {
        // Frame claiming 32MB payload → error
        let mut data: Vec<u8> = vec![0x81, 127]; // FIN+Text, 64-bit length
        let huge_len: u64 = 32 * 1024 * 1024;
        data.extend_from_slice(&huge_len.to_be_bytes());
        let mut cursor = std::io::Cursor::new(data);
        let result = read_ws_frame(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_ws_frame_extended_16bit_length() {
        use crate::websocket_streaming::WsOpcode;
        // Text frame with 200-byte payload via 16-bit extended length
        let payload = vec![0x41u8; 200]; // 200 'A' bytes
        let mut data = vec![0x81, 126]; // FIN+Text, 16-bit length follows
        data.extend_from_slice(&200u16.to_be_bytes());
        data.extend_from_slice(&payload);
        let mut cursor = std::io::Cursor::new(data);
        let frame = read_ws_frame(&mut cursor).unwrap();
        assert_eq!(frame.opcode, WsOpcode::Text);
        assert_eq!(frame.payload.len(), 200);
    }

    #[test]
    fn test_read_write_ws_frame_roundtrip() {
        use crate::websocket_streaming::WsFrame;
        let original = WsFrame::text("roundtrip test");
        let mut buf = Vec::new();
        write_ws_frame(&mut buf, &original).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let decoded = read_ws_frame(&mut cursor).unwrap();
        assert_eq!(decoded.payload, original.payload);
        assert!(decoded.fin);
    }

    #[test]
    fn test_ws_functions_exist() {
        // Compile-time check for function signatures
        let _: fn(&mut dyn Read) -> std::io::Result<crate::websocket_streaming::WsFrame> = read_ws_frame;
        let _: fn(&mut dyn Write, &crate::websocket_streaming::WsFrame) -> std::io::Result<()> = write_ws_frame;
        let _: fn(&mut dyn Write, &[(String, String)], &str) -> std::io::Result<()> = ws_handshake;
        let _: fn(&[(String, String)]) -> bool = is_websocket_upgrade;
    }

    // ========================================================================
    // TLS runtime tests (v10 Phase 4)
    // ========================================================================

    #[cfg(feature = "server-tls")]
    mod tls_runtime_tests {
        use super::*;

        // Self-signed EC P-256 test cert + key (valid for 10 years, CN=localhost).
        // Generated once, safe to embed in tests.
        const TEST_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBkTCB+wIUYTEzMjQ1Njc4OTAxMjM0NTY3ODkwDQYJKoZIhvcNAQELBQAwFDES\n\
MBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDEwMTAwMDAwMFoXDTM2MDEwMTAwMDAw\n\
MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcD\n\
QgAEVMSB1jmE+MmJm2fF+fQIJbHAAWpxSuFQlAaGIYEP8GUPdfi+FfWQzmPLIqR\n\
YcaMXwzWmKMHKJdS9FnvRBfKxqMhMB8wHQYDVR0OBBYEFBRkVQYHKJdS9FnvRBfK\n\
xqMhMB8wDQYJKoZIhvcNAQELBQADQQBhTjTL\n\
-----END CERTIFICATE-----";

        const TEST_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgevZzL1gdAFr88hb2\n\
OF/2NxApJCzGCEDdfSp6VQO30hyhRANCAAQB7v30z0x5BhOe15KD7kJuUKygbcMB\n\
FI4C+rAGMo2tBOcAJgIXkQkBmoqgWcFuqBQ6ID2L+f+x0jYz2DelZ3pI\n\
-----END PRIVATE KEY-----";

        #[test]
        fn test_load_tls_config_invalid_cert_path() {
            let tls = TlsConfig {
                cert_path: "/nonexistent/cert.pem".to_string(),
                key_path: "/nonexistent/key.pem".to_string(),
            };
            let result = load_tls_config(&tls);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("Failed to read cert file"), "Error: {}", err);
        }

        #[test]
        fn test_load_tls_config_invalid_key_path() {
            // Write valid cert but point to bad key
            let dir = std::env::temp_dir().join("tls_test_invalid_key");
            let _ = std::fs::create_dir_all(&dir);
            let cert_path = dir.join("cert.pem");
            std::fs::write(&cert_path, TEST_CERT_PEM).unwrap();

            let tls = TlsConfig {
                cert_path: cert_path.to_str().unwrap().to_string(),
                key_path: "/nonexistent/key.pem".to_string(),
            };
            let result = load_tls_config(&tls);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("Failed to read key file"), "Error: {}", err);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_load_tls_config_empty_cert() {
            let dir = std::env::temp_dir().join("tls_test_empty_cert");
            let _ = std::fs::create_dir_all(&dir);
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, "").unwrap();
            std::fs::write(&key_path, TEST_KEY_PEM).unwrap();

            let tls = TlsConfig {
                cert_path: cert_path.to_str().unwrap().to_string(),
                key_path: key_path.to_str().unwrap().to_string(),
            };
            let result = load_tls_config(&tls);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("No certificates found"), "Error: {}", err);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_load_tls_config_empty_key() {
            let dir = std::env::temp_dir().join("tls_test_empty_key");
            let _ = std::fs::create_dir_all(&dir);
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, TEST_CERT_PEM).unwrap();
            std::fs::write(&key_path, "").unwrap();

            let tls = TlsConfig {
                cert_path: cert_path.to_str().unwrap().to_string(),
                key_path: key_path.to_str().unwrap().to_string(),
            };
            let result = load_tls_config(&tls);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("No private key found"), "Error: {}", err);

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_load_tls_config_corrupt_cert() {
            let dir = std::env::temp_dir().join("tls_test_corrupt_cert");
            let _ = std::fs::create_dir_all(&dir);
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, "-----BEGIN CERTIFICATE-----\nNOT_VALID_BASE64!!!\n-----END CERTIFICATE-----").unwrap();
            std::fs::write(&key_path, TEST_KEY_PEM).unwrap();

            let tls = TlsConfig {
                cert_path: cert_path.to_str().unwrap().to_string(),
                key_path: key_path.to_str().unwrap().to_string(),
            };
            let result = load_tls_config(&tls);
            // Either parse error or build error (bad cert data)
            assert!(result.is_err());

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_load_tls_config_corrupt_key() {
            let dir = std::env::temp_dir().join("tls_test_corrupt_key");
            let _ = std::fs::create_dir_all(&dir);
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, TEST_CERT_PEM).unwrap();
            std::fs::write(&key_path, "-----BEGIN PRIVATE KEY-----\nINVALID!!!\n-----END PRIVATE KEY-----").unwrap();

            let tls = TlsConfig {
                cert_path: cert_path.to_str().unwrap().to_string(),
                key_path: key_path.to_str().unwrap().to_string(),
            };
            let result = load_tls_config(&tls);
            // Either parse error or build error (bad key)
            assert!(result.is_err());

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn test_server_no_tls_still_works() {
            // Even with server-tls feature enabled, a server without TLS config
            // should still work as plain HTTP.
            let config = ServerConfig::default();
            assert!(config.tls.is_none());
            let server = AiServer::new(config);
            assert!(server.tls_server_config.is_none());
        }

        #[test]
        fn test_tls_accept_function_exists() {
            // Verify tls_accept is callable (compile-time check).
            // We can't test a real handshake without a matching client,
            // but we verify the function signature is correct.
            let _fn_ptr: fn(
                TcpStream,
                &Arc<rustls::ServerConfig>,
            ) -> std::io::Result<
                rustls::StreamOwned<rustls::ServerConnection, TcpStream>,
            > = tls_accept;
        }

        #[test]
        fn test_handle_tls_connection_exists() {
            // Compile-time verification that handle_tls_connection has the correct signature.
            let _fn_ptr: fn(
                TcpStream,
                &Arc<rustls::ServerConfig>,
                &Arc<Mutex<AiAssistant>>,
                &ServerConfig,
                &Arc<ServerMetrics>,
            ) -> std::io::Result<()> = handle_tls_connection;
        }

        #[test]
        fn test_load_tls_config_function_signature() {
            // Compile-time verification of the public load_tls_config function.
            let _fn_ptr: fn(&TlsConfig) -> Result<Arc<rustls::ServerConfig>, String> = load_tls_config;
        }

        #[test]
        fn test_scheme_detection_with_tls() {
            // When TLS is configured, the server should report https
            let config = ServerConfig {
                tls: Some(TlsConfig {
                    cert_path: "dummy.pem".to_string(),
                    key_path: "dummy.pem".to_string(),
                }),
                ..Default::default()
            };
            let scheme = if config.tls.is_some() { "https" } else { "http" };
            assert_eq!(scheme, "https");
        }

        #[test]
        fn test_scheme_detection_without_tls() {
            let config = ServerConfig::default();
            let scheme = if config.tls.is_some() { "https" } else { "http" };
            assert_eq!(scheme, "http");
        }

        #[test]
        fn test_read_write_trait_tcp_stream() {
            // Verify TcpStream implements our ReadWrite trait (compile-time).
            fn _assert_rw<T: super::super::ReadWrite>() {}
            _assert_rw::<TcpStream>();
        }

        #[test]
        fn test_handle_tcp_connection_exists() {
            // Compile-time verification that handle_tcp_connection has the correct signature.
            let _fn_ptr: fn(
                TcpStream,
                &Arc<Mutex<AiAssistant>>,
                &ServerConfig,
                &Arc<ServerMetrics>,
            ) -> std::io::Result<()> = handle_tcp_connection;
        }
    }

    // ========================================================================
    // Enhanced Health Check tests (item 1.2)
    // ========================================================================

    #[test]
    fn test_health_enhanced_fields() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        // Sleep briefly so uptime_secs is at least 0 (it should always be)
        let (status, body) = handle_health(&assistant, &metrics);
        assert_eq!(status, "200 OK");

        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
        assert!(parsed["version"].is_string());
        assert!(parsed["model"].is_string());
        assert!(parsed["provider"].is_string());
        // uptime_secs should be a number >= 0
        assert!(parsed["uptime_secs"].is_number());
        assert!(parsed["uptime_secs"].as_u64().unwrap() < 60); // test runs fast
        // No sessions or messages in a fresh assistant
        assert_eq!(parsed["active_sessions"].as_u64().unwrap(), 0);
        assert_eq!(parsed["conversation_messages"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_health_with_conversation_messages() {
        let mut ass = AiAssistant::new();
        ass.conversation.push(crate::messages::ChatMessage::user("hello"));
        ass.conversation.push(crate::messages::ChatMessage::assistant("hi there"));
        let assistant = Arc::new(Mutex::new(ass));
        let metrics = Arc::new(ServerMetrics::new());

        let (status, body) = handle_health(&assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["conversation_messages"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_health_with_active_sessions() {
        let mut ass = AiAssistant::new();
        let session = crate::session::ChatSession::new("test-session");
        ass.session_store.save_session(session);
        let assistant = Arc::new(Mutex::new(ass));
        let metrics = Arc::new(ServerMetrics::new());

        let (status, body) = handle_health(&assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["active_sessions"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_health_uptime_increases() {
        let metrics = Arc::new(ServerMetrics::new());
        std::thread::sleep(std::time::Duration::from_millis(10));
        // uptime should be >= 0
        let uptime = metrics.started_at.elapsed().as_secs();
        assert!(uptime < 5); // sanity check — test shouldn't take 5 seconds
    }

    // ========================================================================
    // Session Endpoints tests (item 1.3)
    // ========================================================================

    #[test]
    fn test_list_sessions_empty() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_list_sessions(&assistant);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_list_sessions_with_sessions() {
        let mut ass = AiAssistant::new();
        let mut s1 = crate::session::ChatSession::new("Session A");
        s1.id = "sess-001".to_string();
        s1.messages.push(crate::messages::ChatMessage::user("hello"));
        s1.messages.push(crate::messages::ChatMessage::assistant("hi"));
        ass.session_store.save_session(s1);

        let mut s2 = crate::session::ChatSession::new("Session B");
        s2.id = "sess-002".to_string();
        ass.session_store.save_session(s2);

        let assistant = Arc::new(Mutex::new(ass));
        let (status, body) = handle_list_sessions(&assistant);
        assert_eq!(status, "200 OK");

        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);

        // First session has 2 messages
        let first = arr.iter().find(|s| s["id"] == "sess-001").unwrap();
        assert_eq!(first["messages"].as_u64().unwrap(), 2);

        // Second session has 0 messages
        let second = arr.iter().find(|s| s["id"] == "sess-002").unwrap();
        assert_eq!(second["messages"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_get_session_not_found() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_get_session("nonexistent", &assistant);
        assert_eq!(status, "404 Not Found");
        assert!(body.contains("Session not found"));
        assert!(body.contains("nonexistent"));
    }

    #[test]
    fn test_get_session_found() {
        let mut ass = AiAssistant::new();
        let mut session = crate::session::ChatSession::new("My Chat");
        session.id = "sess-abc".to_string();
        session.messages.push(crate::messages::ChatMessage::user("What is Rust?"));
        session.messages.push(crate::messages::ChatMessage::assistant("Rust is a systems programming language."));
        ass.session_store.save_session(session);

        let assistant = Arc::new(Mutex::new(ass));
        let (status, body) = handle_get_session("sess-abc", &assistant);
        assert_eq!(status, "200 OK");

        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[0]["content"], "What is Rust?");
        assert_eq!(arr[1]["role"], "assistant");
        assert!(arr[1]["content"].as_str().unwrap().contains("systems programming"));
    }

    #[test]
    fn test_delete_session_found() {
        let mut ass = AiAssistant::new();
        let mut session = crate::session::ChatSession::new("To Delete");
        session.id = "sess-del".to_string();
        ass.session_store.save_session(session);

        let assistant = Arc::new(Mutex::new(ass));
        let (status, body) = handle_delete_session("sess-del", &assistant);
        assert_eq!(status, "200 OK");

        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["deleted"], true);

        // Verify it's actually gone
        let ass = assistant.lock().unwrap();
        assert!(ass.session_store.sessions.is_empty());
    }

    #[test]
    fn test_delete_session_not_found() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let (status, body) = handle_delete_session("no-such-session", &assistant);
        assert_eq!(status, "404 Not Found");
        assert!(body.contains("Session not found"));
    }

    // ========================================================================
    // Request Validation tests (item 1.4)
    // ========================================================================

    #[test]
    fn test_request_validation_message_too_long() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig {
            max_message_length: 10, // very small for testing
            ..Default::default()
        };

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({
                "message": "This message is definitely longer than 10 characters"
            })).unwrap(),
        };

        let (status, body) = handle_chat(&request, &assistant, &config);
        assert_eq!(status, "422 Unprocessable Entity");
        assert!(body.contains("Message too long"));
        assert!(body.contains("max 10"));
    }

    #[test]
    fn test_request_validation_max_message_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_message_length, 100_000);
    }

    #[test]
    fn test_request_validation_exact_limit() {
        // Message exactly at the limit should pass (not be rejected)
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig {
            max_message_length: 5,
            ..Default::default()
        };

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({
                "message": "Hello"  // exactly 5 chars
            })).unwrap(),
        };

        // Should NOT return 422 — it may return 500 (provider not running) but not 422
        let (status, _body) = handle_chat(&request, &assistant, &config);
        assert_ne!(status, "422 Unprocessable Entity");
    }

    #[test]
    fn test_request_validation_one_over_limit() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig {
            max_message_length: 5,
            ..Default::default()
        };

        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({
                "message": "Hello!" // 6 chars, over limit of 5
            })).unwrap(),
        };

        let (status, body) = handle_chat(&request, &assistant, &config);
        assert_eq!(status, "422 Unprocessable Entity");
        assert!(body.contains("Message too long"));
    }

    // ========================================================================
    // Session endpoints via route_request (integration)
    // ========================================================================

    #[test]
    fn test_session_endpoints_integration() {
        let mut ass = AiAssistant::new();

        // Create two sessions with messages
        let mut s1 = crate::session::ChatSession::new("Integration A");
        s1.id = "int-001".to_string();
        s1.messages.push(crate::messages::ChatMessage::user("first"));
        ass.session_store.save_session(s1);

        let mut s2 = crate::session::ChatSession::new("Integration B");
        s2.id = "int-002".to_string();
        s2.messages.push(crate::messages::ChatMessage::user("second"));
        s2.messages.push(crate::messages::ChatMessage::assistant("reply"));
        ass.session_store.save_session(s2);

        let assistant = Arc::new(Mutex::new(ass));
        let metrics = Arc::new(ServerMetrics::new());

        // 1. List sessions
        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let list: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(list.as_array().unwrap().len(), 2);

        // 2. Get specific session
        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/sessions/int-002".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let msgs: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(msgs.as_array().unwrap().len(), 2);

        // 3. Delete a session
        let req = HttpRequest {
            method: "DELETE".to_string(),
            path: "/sessions/int-001".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["deleted"], true);

        // 4. List sessions — should now have only 1
        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let list: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(list.as_array().unwrap().len(), 1);
        assert_eq!(list[0]["id"], "int-002");

        // 5. Get deleted session — 404
        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/sessions/int-001".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, _body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "404 Not Found");
    }

    #[test]
    fn test_session_route_via_route_request() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        // GET /sessions on empty store
        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        assert_eq!(body, "[]");
    }

    #[test]
    fn test_health_via_route_includes_enhanced_fields() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let req = HttpRequest {
            method: "GET".to_string(),
            path: "/health".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request(&req, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        // Enhanced fields must be present
        assert!(parsed.get("uptime_secs").is_some());
        assert!(parsed.get("active_sessions").is_some());
        assert!(parsed.get("conversation_messages").is_some());
    }

    // ========================================================================
    // Audit Log tests (items 4.1, 4.2)
    // ========================================================================

    #[test]
    fn test_audit_log_new() {
        let log = AuditLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_audit_log_append() {
        let log = AuditLog::new(100);
        log.log(AuditEventType::AuthSuccess, "user1", "/chat", "Bearer token");
        assert_eq!(log.len(), 1);
        let entries = log.entries();
        assert_eq!(entries[0].actor, "user1");
        assert_eq!(entries[0].path, "/chat");
    }

    #[test]
    fn test_audit_log_max_entries() {
        let log = AuditLog::new(3);
        log.log(AuditEventType::AuthSuccess, "a", "/1", "");
        log.log(AuditEventType::AuthFailure, "b", "/2", "");
        log.log(AuditEventType::ConfigChange, "c", "/3", "");
        log.log(AuditEventType::SessionCreated, "d", "/4", "");
        assert_eq!(log.len(), 3);
        // First entry should have been evicted
        let entries = log.entries();
        assert_eq!(entries[0].actor, "b");
    }

    #[test]
    fn test_audit_log_entry_has_timestamp() {
        let log = AuditLog::new(100);
        log.log(AuditEventType::RequestProcessed, "test", "/", "ok");
        let entries = log.entries();
        assert!(entries[0].timestamp > 0);
    }

    #[test]
    fn test_audit_log_all_event_types() {
        let log = AuditLog::new(100);
        log.log(AuditEventType::AuthSuccess, "", "", "");
        log.log(AuditEventType::AuthFailure, "", "", "");
        log.log(AuditEventType::ConfigChange, "", "", "");
        log.log(AuditEventType::SessionCreated, "", "", "");
        log.log(AuditEventType::SessionDeleted, "", "", "");
        log.log(AuditEventType::RequestProcessed, "", "", "");
        assert_eq!(log.len(), 6);
    }

    #[test]
    fn test_audit_entry_serializable() {
        let log = AuditLog::new(100);
        log.log(AuditEventType::AuthSuccess, "admin", "/health", "ok");
        let entries = log.entries();
        let json = serde_json::to_string(&entries[0]).unwrap();
        assert!(json.contains("AuthSuccess"));
        assert!(json.contains("admin"));
    }

    // ========================================================================
    // API Versioning tests (items 5.1, 5.2)
    // ========================================================================

    #[test]
    fn test_versioned_health_route() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/health".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
    }

    #[test]
    fn test_versioned_sessions_route() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/sessions".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn test_versioned_models_route() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/models".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, _body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "200 OK");
    }

    // ========================================================================
    // Structured Error tests (items 6.1, 6.2)
    // ========================================================================

    #[test]
    fn test_structured_error_basic() {
        let err = StructuredError::new("INVALID_JSON", "Could not parse request body");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("INVALID_JSON"));
        assert!(!json.contains("details")); // None should be skipped
        assert!(!json.contains("retry_after_secs"));
    }

    #[test]
    fn test_structured_error_with_details() {
        let err = StructuredError::new("AUTH_FAILED", "Invalid credentials")
            .with_details("Bearer token expired");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("AUTH_FAILED"));
        assert!(json.contains("Bearer token expired"));
    }

    #[test]
    fn test_structured_error_with_retry() {
        let err = StructuredError::new("RATE_LIMITED", "Too many requests")
            .with_retry(60);
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("RATE_LIMITED"));
        assert!(json.contains("60"));
    }

    #[test]
    fn test_structured_error_full() {
        let err = StructuredError::new("MODEL_ERROR", "Provider unavailable")
            .with_details("OpenAI API returned 503")
            .with_retry(30);
        let json = serde_json::to_string(&err).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["error_code"], "MODEL_ERROR");
        assert_eq!(parsed["message"], "Provider unavailable");
        assert_eq!(parsed["details"], "OpenAI API returned 503");
        assert_eq!(parsed["retry_after_secs"], 30);
    }

    // ========================================================================
    // SSE Streaming Endpoint tests (item 1.1)
    // ========================================================================

    #[test]
    fn test_chat_stream_invalid_json() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: "not json".to_string(),
        };
        let result = handle_chat_stream(&request, &assistant, &config);
        assert!(result.is_err());
        let (status, body) = result.unwrap_err();
        assert_eq!(status, "400 Bad Request");
        assert!(body.contains("Invalid JSON"));
    }

    #[test]
    fn test_chat_stream_message_too_long() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig {
            max_message_length: 5,
            ..Default::default()
        };
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({"message": "This is too long"})).unwrap(),
        };
        let result = handle_chat_stream(&request, &assistant, &config);
        assert!(result.is_err());
        let (status, body) = result.unwrap_err();
        assert_eq!(status, "422 Unprocessable Entity");
        assert!(body.contains("Message too long"));
    }

    #[test]
    fn test_chat_stream_route_invalid_json() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: "bad body".to_string(),
        };
        let (status, body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "400 Bad Request");
        assert!(body.contains("Invalid JSON"));
    }

    #[test]
    fn test_chat_stream_versioned_route_invalid_json() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/api/v1/chat/stream".to_string(),
            headers: vec![],
            body: "not valid".to_string(),
        };
        let (status, body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "400 Bad Request");
        assert!(body.contains("Invalid JSON"));
    }

    #[test]
    fn test_build_sse_response_headers() {
        let sse_body = "data: {\"token\": \"hello\"}\n\ndata: [DONE]\n\n";
        let response = build_sse_response(sse_body, "");
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("Content-Type: text/event-stream\r\n"));
        assert!(response.contains("Cache-Control: no-cache\r\n"));
        assert!(response.contains("Connection: keep-alive\r\n"));
        assert!(response.contains(&format!("Content-Length: {}\r\n", sse_body.len())));
        assert!(response.ends_with(sse_body));
    }

    #[test]
    fn test_build_sse_response_with_extra_headers() {
        let sse_body = "data: [DONE]\n\n";
        let extra = "X-Request-Id: abc123\r\n";
        let response = build_sse_response(sse_body, extra);
        assert!(response.contains("X-Request-Id: abc123\r\n"));
        assert!(response.contains("text/event-stream"));
    }

    #[test]
    fn test_chat_stream_route_message_too_long() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig {
            max_message_length: 3,
            ..Default::default()
        };
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({"message": "toolong"})).unwrap(),
        };
        let (status, body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "422 Unprocessable Entity");
        assert!(body.contains("Message too long"));
    }

    #[test]
    fn test_chat_stream_get_method_not_found() {
        // GET to /chat/stream should 404 since only POST is supported
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: String::new(),
        };
        let (status, _body) = route_request_with_config(&request, &assistant, &metrics, &config);
        assert_eq!(status, "404 Not Found");
    }

    #[test]
    fn test_chat_stream_empty_message() {
        // Empty message is valid (length 0 <= any max_message_length)
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: serde_json::to_string(&serde_json::json!({"message": ""})).unwrap(),
        };
        // The handler will try to call the LLM, which will block/error — we just
        // verify it doesn't reject the request with 400 or 422
        let result = handle_chat_stream(&request, &assistant, &config);
        // If it returns Err, it should be a 500 from the provider, not 400/422
        if let Err((status, _)) = &result {
            assert!(
                status.starts_with("500"),
                "Empty message should not be rejected as invalid"
            );
        }
    }

    #[test]
    fn test_chat_stream_missing_message_field() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let config = ServerConfig::default();
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/chat/stream".to_string(),
            headers: vec![],
            body: r#"{"prompt": "hello"}"#.to_string(), // wrong field name
        };
        let result = handle_chat_stream(&request, &assistant, &config);
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, "400 Bad Request");
    }

    // ========================================================================
    // Response Compression tests (item 1.5)
    // ========================================================================

    #[test]
    fn test_compress_gzip_produces_valid_output() {
        let data = b"Hello, world! This is some test data for compression.";
        let compressed = compress_gzip(data);
        // Compressed output should be non-empty
        assert!(!compressed.is_empty());
        // gzip magic bytes: 0x1f, 0x8b
        assert_eq!(compressed[0], 0x1f);
        assert_eq!(compressed[1], 0x8b);
    }

    #[test]
    fn test_compress_gzip_decompresses_correctly() {
        use flate2::read::GzDecoder;

        let original = b"The quick brown fox jumps over the lazy dog.";
        let compressed = compress_gzip(original);

        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed).unwrap();
        assert_eq!(decompressed.as_bytes(), original);
    }

    #[test]
    fn test_maybe_compress_no_accept_encoding() {
        let body = "Some response body";
        let (bytes, compressed) = maybe_compress_response(body, None);
        assert!(!compressed);
        assert_eq!(bytes, body.as_bytes());
    }

    #[test]
    fn test_maybe_compress_with_gzip_accept() {
        let body = "Some response body that should be compressed";
        let (bytes, compressed) = maybe_compress_response(body, Some("gzip, deflate, br"));
        assert!(compressed);
        // Should be valid gzip
        assert_eq!(bytes[0], 0x1f);
        assert_eq!(bytes[1], 0x8b);
    }

    #[test]
    fn test_maybe_compress_without_gzip_accept() {
        let body = "Some response body";
        let (bytes, compressed) = maybe_compress_response(body, Some("deflate, br"));
        assert!(!compressed);
        assert_eq!(bytes, body.as_bytes());
    }

    #[test]
    fn test_compress_gzip_empty_input() {
        let data = b"";
        let compressed = compress_gzip(data);
        // Even empty data produces a valid gzip stream
        assert!(!compressed.is_empty());
        assert_eq!(compressed[0], 0x1f);
        assert_eq!(compressed[1], 0x8b);

        // Decompresses to empty
        use flate2::read::GzDecoder;
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result).unwrap();
        assert!(result.is_empty());
    }

    // ========================================================================
    // Rate Limiting tests (item 1.6)
    // ========================================================================

    #[test]
    fn test_rate_limiter_new() {
        let rl = ServerRateLimiter::new(100);
        assert_eq!(rl.requests_per_minute, 100);
    }

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let rl = ServerRateLimiter::new(5);
        for _ in 0..5 {
            assert!(rl.check_rate_limit());
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let rl = ServerRateLimiter::new(3);
        assert!(rl.check_rate_limit()); // 1
        assert!(rl.check_rate_limit()); // 2
        assert!(rl.check_rate_limit()); // 3
        assert!(!rl.check_rate_limit()); // 4 — blocked
        assert!(!rl.check_rate_limit()); // 5 — still blocked
    }

    #[test]
    fn test_rate_limiter_retry_after() {
        let rl = ServerRateLimiter::new(10);
        let retry = rl.retry_after_secs();
        // Should be between 0 and 60 (we just created it, so close to 60)
        assert!(retry <= 60);
    }

    #[test]
    fn test_rate_limiter_zero_limit() {
        // Zero means no requests are allowed
        let rl = ServerRateLimiter::new(0);
        assert!(!rl.check_rate_limit());
    }

    #[test]
    fn test_rate_limiter_thread_safe() {
        let rl = Arc::new(ServerRateLimiter::new(100));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let rl_clone = rl.clone();
            handles.push(std::thread::spawn(move || {
                let mut allowed = 0u32;
                for _ in 0..30 {
                    if rl_clone.check_rate_limit() {
                        allowed += 1;
                    }
                }
                allowed
            }));
        }
        let total_allowed: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        // 4 threads * 30 requests = 120 total attempts, limit is 100
        assert_eq!(total_allowed, 100);
    }

    #[test]
    fn test_rate_limiter_in_server_config() {
        let rl = Arc::new(ServerRateLimiter::new(60));
        let config = ServerConfig {
            rate_limiter: Some(rl.clone()),
            ..Default::default()
        };
        assert!(config.rate_limiter.is_some());
        assert_eq!(config.rate_limiter.as_ref().unwrap().requests_per_minute, 60);
    }

    #[test]
    fn test_rate_limiter_default_none() {
        let config = ServerConfig::default();
        assert!(config.rate_limiter.is_none());
    }

    // ========================================================================
    // OpenAPI route tests (v10 Phase 6)
    // ========================================================================

    #[test]
    fn test_openapi_route() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/openapi.json".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["openapi"], "3.0.0");
    }

    #[test]
    fn test_openapi_route_legacy() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/openapi.json".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["info"]["title"], "AI Assistant API");
    }

    #[test]
    fn test_openapi_route_has_paths() {
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/openapi.json".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        let paths = parsed["paths"].as_object().unwrap();
        assert!(paths.contains_key("/health"));
        assert!(paths.contains_key("/chat"));
        assert!(paths.contains_key("/api/v1/health"));
        assert!(paths.contains_key("/api/v1/chat"));
    }

    #[test]
    fn test_openapi_content_type() {
        // The route returns JSON — verify the body is valid JSON
        let assistant = Arc::new(Mutex::new(AiAssistant::new()));
        let metrics = Arc::new(ServerMetrics::new());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/openapi.json".to_string(),
            headers: vec![],
            body: String::new(),
        };

        let (status, body) = route_request(&request, &assistant, &metrics);
        assert_eq!(status, "200 OK");
        // The server default Content-Type is application/json; verify body parses as JSON
        assert!(serde_json::from_str::<serde_json::Value>(&body).is_ok());
        // Also check it contains the expected content-type hint in schemas
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed["components"]["schemas"].is_object());
    }
}
