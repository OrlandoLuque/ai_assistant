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
        let session_id = format!(
            "mcp-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );
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
