//! MCP v2 Streamable HTTP Transport, Session, and Session Store.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::*;

// ---------------------------------------------------------------------------
// 2.1 -- Streamable HTTP Transport
// ---------------------------------------------------------------------------

/// Transport mode for MCP v2: StdIO, SSE (legacy), or Streamable HTTP (v2 default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportMode {
    /// Standard I/O (stdin/stdout).
    #[serde(rename = "std_io")]
    StdIO,
    /// Legacy Server-Sent Events transport.
    #[serde(rename = "sse")]
    SSE,
    /// Streamable HTTP: server may respond with JSON or SSE depending on the operation.
    #[serde(rename = "streamable_http")]
    StreamableHTTP,
}

/// Streamable HTTP transport client (MCP v2).
///
/// Sends POST to a single endpoint and auto-detects whether the response
/// is immediate JSON or an SSE stream based on the Content-Type header.
pub struct StreamableHttpTransport {
    base_url: String,
    session_id: Option<String>,
    mode: TransportMode,
    /// Optional Bearer token for authenticated requests.
    auth_token: Option<String>,
    /// HTTP timeout in seconds (default: 30).
    timeout_secs: Option<u64>,
}

impl StreamableHttpTransport {
    /// Create a new Streamable HTTP transport pointing at `base_url`.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            session_id: None,
            mode: TransportMode::StreamableHTTP,
            auth_token: None,
            timeout_secs: None,
        }
    }

    /// Set the Bearer token for authenticated requests.
    pub fn set_auth_token(&mut self, token: String) {
        self.auth_token = Some(token);
    }

    /// Set the HTTP request timeout in seconds.
    pub fn set_timeout_secs(&mut self, secs: u64) {
        self.timeout_secs = Some(secs);
    }

    /// Send a JSON-RPC request via real HTTP POST to `base_url`.
    ///
    /// The response Content-Type is inspected to determine whether the server
    /// replied with immediate JSON or an SSE stream. For SSE responses the
    /// first `data:` line carrying a JSON object is parsed and returned.
    pub fn send_request(&mut self, request: &McpRequest) -> Result<McpResponse, String> {
        let body = serde_json::to_value(request)
            .map_err(|e| format!("Serialization error: {}", e))?;

        let timeout = std::time::Duration::from_secs(self.timeout_secs.unwrap_or(30));

        let mut req = ureq::post(&self.base_url)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream")
            .timeout(timeout);

        // Attach session ID header if present.
        if let Some(ref sid) = self.session_id {
            req = req.set("Mcp-Session-Id", sid);
        }

        // Attach Bearer token if configured.
        if let Some(ref token) = self.auth_token {
            req = req.set("Authorization", &format!("Bearer {}", token));
        }

        match req.send_json(&body) {
            Ok(resp) => {
                // Capture session ID from response header if the server provides one.
                if let Some(sid) = resp.header("Mcp-Session-Id") {
                    self.session_id = Some(sid.to_string());
                }

                let content_type = resp.content_type().to_string();
                self.mode = Self::detect_transport(&content_type);

                if content_type.contains("text/event-stream") {
                    // Parse SSE: read the body as text and extract the first
                    // JSON object from `data:` lines.
                    let body_text = resp.into_string()
                        .map_err(|e| format!("Failed to read SSE body: {}", e))?;
                    Self::parse_sse_to_response(&body_text)
                } else {
                    // Direct JSON response.
                    let json_str = resp.into_string()
                        .map_err(|e| format!("Failed to read response body: {}", e))?;
                    serde_json::from_str::<McpResponse>(&json_str)
                        .map_err(|e| format!("Failed to parse JSON response: {}", e))
                }
            }
            Err(e) => Err(format!("HTTP request failed: {}", e)),
        }
    }

    /// Parse an SSE response body and extract the first JSON-RPC response.
    fn parse_sse_to_response(body: &str) -> Result<McpResponse, String> {
        for line in body.lines() {
            let trimmed = line.trim();
            if let Some(data) = trimmed.strip_prefix("data:") {
                let data = data.trim();
                if data.starts_with('{') {
                    return serde_json::from_str::<McpResponse>(data)
                        .map_err(|e| format!("Failed to parse SSE data as JSON-RPC: {}", e));
                }
            }
        }
        Err("No JSON-RPC response found in SSE stream".to_string())
    }

    /// Detect the transport mode from a response Content-Type header value.
    pub fn detect_transport(response_content_type: &str) -> TransportMode {
        let ct = response_content_type.to_lowercase();
        if ct.contains("text/event-stream") {
            TransportMode::SSE
        } else if ct.contains("application/json") {
            TransportMode::StreamableHTTP
        } else {
            // Unknown content type -- default to StreamableHTTP (JSON mode)
            TransportMode::StreamableHTTP
        }
    }

    /// Get the current session ID, if any.
    pub fn get_session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Set the session ID (typically received from an `Mcp-Session-Id` header).
    pub fn set_session_id(&mut self, id: String) {
        self.session_id = Some(id);
    }

    /// Get the current transport mode.
    pub fn mode(&self) -> TransportMode {
        self.mode
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// An MCP session with timestamps and metadata (MCP v2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSession {
    pub session_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Trait for managing MCP sessions.
pub trait McpSessionStore {
    /// Create a new session and return it.
    fn create_session(&mut self) -> McpSession;
    /// Retrieve a session by ID.
    fn get_session(&self, id: &str) -> Option<&McpSession>;
    /// Update the `last_active` timestamp of a session.
    fn touch_session(&mut self, id: &str);
    /// Delete a session by ID.
    fn delete_session(&mut self, id: &str);
    /// List all sessions.
    fn list_sessions(&self) -> Vec<&McpSession>;
    /// Remove sessions older than `max_age_secs` seconds.
    fn cleanup_expired(&mut self, max_age_secs: u64);
}

/// In-memory implementation of `McpSessionStore`.
pub struct InMemorySessionStore {
    pub(crate) sessions: HashMap<String, McpSession>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl McpSessionStore for InMemorySessionStore {
    fn create_session(&mut self) -> McpSession {
        let id = super::transport::generate_secure_session_id("mcp-session");
        let now = chrono::Utc::now();
        let session = McpSession {
            session_id: id.clone(),
            created_at: now,
            last_active: now,
            metadata: HashMap::new(),
        };
        self.sessions.insert(id, session.clone());
        session
    }

    fn get_session(&self, id: &str) -> Option<&McpSession> {
        self.sessions.get(id)
    }

    fn touch_session(&mut self, id: &str) {
        if let Some(session) = self.sessions.get_mut(id) {
            session.last_active = chrono::Utc::now();
        }
    }

    fn delete_session(&mut self, id: &str) {
        self.sessions.remove(id);
    }

    fn list_sessions(&self) -> Vec<&McpSession> {
        self.sessions.values().collect()
    }

    fn cleanup_expired(&mut self, max_age_secs: u64) {
        let now = chrono::Utc::now();
        self.sessions.retain(|_, session| {
            let age = now
                .signed_duration_since(session.last_active)
                .num_seconds();
            age >= 0 && (age as u64) < max_age_secs
        });
    }
}
