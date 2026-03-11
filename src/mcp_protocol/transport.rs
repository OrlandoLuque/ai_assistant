//! MCP Transport types and Streamable HTTP session.

use serde::{Deserialize, Serialize};

use super::server::McpServer;
use super::types::*;

/// MCP Transport type (2025-03-26 spec adds Streamable HTTP)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// Standard HTTP POST (JSON-RPC over HTTP).
    Http,
    /// Server-Sent Events for server->client streaming.
    Sse,
    /// Streamable HTTP: POST for client->server, SSE for server->client.
    /// Replaces the legacy SSE-only transport (2025-03-26 spec).
    StreamableHttp,
    /// Standard I/O (stdin/stdout).
    Stdio,
}

impl Default for McpTransport {
    fn default() -> Self {
        Self::StreamableHttp
    }
}

/// A Streamable HTTP session (2025-03-26 spec).
///
/// Supports bidirectional communication via POST (client->server)
/// and SSE (server->client). Maintains session state via `Mcp-Session-Id`.
pub struct McpStreamableSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Whether the session has been initialized.
    pub initialized: bool,
    /// The underlying MCP server.
    server: McpServer,
    /// Pending SSE events to send to the client.
    pending_events: Vec<String>,
}

impl McpStreamableSession {
    /// Create a new streamable HTTP session wrapping a server.
    pub fn new(server: McpServer) -> Self {
        let session_id = generate_secure_session_id("mcp");
        Self {
            session_id,
            initialized: false,
            server,
            pending_events: Vec::new(),
        }
    }

    /// Handle an incoming POST request body (JSON-RPC).
    ///
    /// Returns the JSON-RPC response and optionally queues SSE events.
    pub fn handle_post(&mut self, body: &str) -> Result<String, McpError> {
        let request: McpRequest =
            serde_json::from_str(body).map_err(|e| McpError::parse_error(&e.to_string()))?;

        if request.method == "initialize" {
            self.initialized = true;
        }

        let response = self.server.handle_request(request);
        let response_json = serde_json::to_string(&response)
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        Ok(response_json)
    }

    /// Get the session ID header value.
    pub fn session_header(&self) -> (&str, &str) {
        ("Mcp-Session-Id", &self.session_id)
    }

    /// Get any pending SSE events.
    pub fn drain_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_events)
    }

    /// Queue an SSE event for the client.
    pub fn push_event(&mut self, event_type: &str, data: &str) {
        let mut event = String::new();
        event.push_str(&format!("event: {}\n", event_type));
        for line in data.lines() {
            event.push_str(&format!("data: {}\n", line));
        }
        event.push('\n');
        self.pending_events.push(event);
    }
}

/// Generate a cryptographically unpredictable session ID.
///
/// Uses multiple `RandomState` instances (each seeded from OS entropy) to
/// harvest 128 bits of randomness, then hex-encodes them. The resulting ID
/// has the form `{prefix}-{32 hex chars}` (e.g. `mcp-a1b2c3d4...`).
///
/// This avoids depending on external crates: `std::collections::hash_map::RandomState`
/// is guaranteed to be randomly seeded on construction (used for HashDoS resistance).
pub(crate) fn generate_secure_session_id(prefix: &str) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    // Harvest 128 bits (16 bytes) from 4 independent RandomState instances.
    // Each RandomState::new() call reads fresh entropy from the OS.
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(4) {
        let state = RandomState::new();
        let mut h = state.build_hasher();
        // Mix in a second RandomState for additional entropy
        let extra = RandomState::new();
        h.write_u64(extra.build_hasher().finish());
        let val = h.finish();
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = (val >> (i * 8)) as u8;
        }
    }

    // Hex-encode to produce a 32-character string
    let mut hex = String::with_capacity(prefix.len() + 1 + 32);
    hex.push_str(prefix);
    hex.push('-');
    for b in &bytes {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}
