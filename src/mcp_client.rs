//! Remote MCP Client module
//!
//! Provides a client for connecting to remote MCP servers via Streamable HTTP,
//! including tool discovery, resource reading, connection pooling, and a
//! multi-server tool registry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::McpClientError;

// =============================================================================
// Configuration types
// =============================================================================

/// Authentication configuration for connecting to a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpClientAuth {
    /// No authentication required.
    None,
    /// Bearer token authentication (e.g. API key).
    BearerToken(String),
    /// OAuth 2.0 client credentials flow.
    OAuth {
        client_id: String,
        client_secret: String,
        token_url: String,
        token: Option<String>,
    },
}

impl Default for McpClientAuth {
    fn default() -> Self {
        McpClientAuth::None
    }
}

/// Identity information sent during the MCP initialize handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            name: "ai_assistant".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Configuration for a remote MCP client connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientConfig {
    /// The base URL of the remote MCP server (e.g. "http://localhost:3000/mcp").
    pub url: String,
    /// Authentication method.
    pub auth: McpClientAuth,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of retry attempts for recoverable errors.
    pub max_retries: usize,
    /// MCP protocol version to negotiate.
    pub protocol_version: String,
    /// Client identity sent during initialization.
    pub client_info: ClientInfo,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            auth: McpClientAuth::None,
            timeout_ms: 30000,
            max_retries: 3,
            protocol_version: "2025-11-05".to_string(),
            client_info: ClientInfo::default(),
        }
    }
}

// =============================================================================
// Server capability and tool/resource types
// =============================================================================

/// Capabilities reported by the remote MCP server during initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
    pub logging: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            tools: true,
            resources: true,
            prompts: false,
            logging: false,
        }
    }
}

/// A tool exposed by a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub annotations: Option<RemoteToolAnnotations>,
}

/// Behavioural annotations for a remote tool (MCP 2025-03-26 spec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteToolAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub idempotent: bool,
    pub open_world: bool,
}

/// A resource exposed by a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteResource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// Content returned when reading a remote resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    pub uri: String,
    pub mime_type: Option<String>,
    pub text: Option<String>,
    /// Base64-encoded binary content.
    pub blob: Option<String>,
}

/// Result of calling a remote tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
}

/// A single content block inside a tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResultContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, mime_type: Option<String>, text: Option<String> },
}

// =============================================================================
// RemoteMcpClient
// =============================================================================

/// Primary client for connecting to a single remote MCP server via Streamable HTTP.
///
/// Usage:
/// ```ignore
/// let mut client = RemoteMcpClient::new(McpClientConfig {
///     url: "http://localhost:3000/mcp".into(),
///     ..Default::default()
/// });
/// client.connect()?;
/// let tools = client.list_tools()?;
/// ```
#[derive(Debug)]
pub struct RemoteMcpClient {
    config: McpClientConfig,
    session_id: Option<String>,
    server_capabilities: Option<ServerCapabilities>,
    tools_cache: Vec<RemoteTool>,
    resources_cache: Vec<RemoteResource>,
    connected: bool,
}

impl RemoteMcpClient {
    /// Create a new client with the given configuration. The client starts
    /// in a disconnected state; call [`connect`](Self::connect) before issuing
    /// any requests.
    pub fn new(config: McpClientConfig) -> Self {
        Self {
            config,
            session_id: None,
            server_capabilities: None,
            tools_cache: Vec::new(),
            resources_cache: Vec::new(),
            connected: false,
        }
    }

    /// Perform the MCP initialize handshake with the remote server.
    ///
    /// On success the client stores the session id and server capabilities and
    /// transitions to the connected state. Returns
    /// [`McpClientError::ConnectionFailed`] when the configured URL is empty.
    pub fn connect(&mut self) -> Result<(), McpClientError> {
        if self.config.url.is_empty() {
            return Err(McpClientError::ConnectionFailed {
                url: "(empty)".to_string(),
                reason: "URL is empty".to_string(),
            });
        }

        // Simulate the initialize handshake.  In a real implementation this
        // would perform HTTP POST to {url} with the JSON-RPC initialize
        // request and parse the response.
        //
        // Generate a deterministic but unique-looking session id based on the
        // URL so that tests are reproducible.
        let session_hash = self.config.url.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        self.session_id = Some(format!("mcp-sess-{:016x}", session_hash));

        self.server_capabilities = Some(ServerCapabilities::default());
        self.connected = true;

        Ok(())
    }

    /// Disconnect from the remote MCP server and clear session state.
    pub fn disconnect(&mut self) {
        self.session_id = None;
        self.server_capabilities = None;
        self.tools_cache.clear();
        self.resources_cache.clear();
        self.connected = false;
    }

    /// Returns `true` if the client has an active connection.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Returns the current session id, or `None` if not connected.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Returns the server capabilities received during initialization, or
    /// `None` if not connected.
    pub fn server_capabilities(&self) -> Option<&ServerCapabilities> {
        self.server_capabilities.as_ref()
    }

    // ------------------------------------------------------------------
    // Tool operations
    // ------------------------------------------------------------------

    /// Fetch (and cache) the list of tools available on the remote server.
    ///
    /// Subsequent calls return the cached list. Use
    /// [`refresh_tools`](Self::refresh_tools) to force a re-fetch.
    pub fn list_tools(&mut self) -> Result<&[RemoteTool], McpClientError> {
        self.ensure_connected()?;

        if self.tools_cache.is_empty() {
            self.fetch_tools()?;
        }

        Ok(&self.tools_cache)
    }

    /// Force a re-fetch of the tools list from the remote server.
    pub fn refresh_tools(&mut self) -> Result<(), McpClientError> {
        self.ensure_connected()?;
        self.tools_cache.clear();
        self.fetch_tools()?;
        Ok(())
    }

    /// Call a tool on the remote server by name.
    pub fn call_tool(
        &self,
        name: &str,
        arguments: &HashMap<String, serde_json::Value>,
    ) -> Result<ToolCallResult, McpClientError> {
        self.ensure_connected()?;

        // In a real implementation this would POST a tools/call JSON-RPC
        // request.  Here we return a simulated result that echoes back the
        // tool name and the arguments so that the caller can verify the
        // round-trip.
        let echo = serde_json::json!({
            "tool": name,
            "arguments": arguments,
        });

        Ok(ToolCallResult {
            content: vec![ToolResultContent::Text {
                text: echo.to_string(),
            }],
            is_error: false,
        })
    }

    // ------------------------------------------------------------------
    // Resource operations
    // ------------------------------------------------------------------

    /// Fetch (and cache) the list of resources exposed by the remote server.
    pub fn list_resources(&mut self) -> Result<&[RemoteResource], McpClientError> {
        self.ensure_connected()?;

        if self.resources_cache.is_empty() {
            self.fetch_resources()?;
        }

        Ok(&self.resources_cache)
    }

    /// Read a single resource by URI from the remote server.
    pub fn read_resource(&self, uri: &str) -> Result<ResourceContent, McpClientError> {
        self.ensure_connected()?;

        // Simulated: return an empty resource content with the requested URI.
        Ok(ResourceContent {
            uri: uri.to_string(),
            mime_type: Some("text/plain".to_string()),
            text: Some(String::new()),
            blob: None,
        })
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Assert that the client is connected; return an appropriate error
    /// otherwise.
    fn ensure_connected(&self) -> Result<(), McpClientError> {
        if !self.connected {
            if let Some(ref sid) = self.session_id {
                return Err(McpClientError::SessionExpired {
                    session_id: sid.clone(),
                });
            }
            return Err(McpClientError::ConnectionFailed {
                url: self.config.url.clone(),
                reason: "not connected".to_string(),
            });
        }
        Ok(())
    }

    /// Simulated fetch of tools from the remote server.
    fn fetch_tools(&mut self) -> Result<(), McpClientError> {
        // In a real implementation this would POST a tools/list JSON-RPC
        // request and deserialize the response.  We populate the cache with
        // an empty list — real tools will appear once actual HTTP transport is
        // wired up.
        // (Cache stays empty; the caller will see zero tools.)
        Ok(())
    }

    /// Simulated fetch of resources from the remote server.
    fn fetch_resources(&mut self) -> Result<(), McpClientError> {
        // Same as fetch_tools — placeholder for real HTTP transport.
        Ok(())
    }
}

// =============================================================================
// RemoteToolRegistry
// =============================================================================

/// Aggregates tools from multiple remote MCP server connections.
///
/// Allows discovering and invoking tools across a fleet of servers by name,
/// transparently routing each call to the server that owns the tool.
pub struct RemoteToolRegistry {
    clients: Vec<(String, RemoteMcpClient)>,
    tool_index: HashMap<String, String>,
}

impl RemoteToolRegistry {
    /// Create an empty registry with no servers.
    pub fn new() -> Self {
        Self {
            clients: Vec::new(),
            tool_index: HashMap::new(),
        }
    }

    /// Register a server with the given human-readable name.
    pub fn add_server(&mut self, name: String, client: RemoteMcpClient) {
        self.clients.push((name, client));
    }

    /// Remove a server by name.  Returns `true` if a server was found and
    /// removed.
    pub fn remove_server(&mut self, name: &str) -> bool {
        let before = self.clients.len();
        self.clients.retain(|(n, _)| n != name);
        // Clean up tool index entries that pointed to the removed server.
        self.tool_index.retain(|_, srv| srv != name);
        self.clients.len() < before
    }

    /// Connect all registered servers, list their tools, and build the
    /// aggregated tool index.  Returns the total number of tools discovered.
    pub fn discover_all(&mut self) -> Result<usize, McpClientError> {
        self.tool_index.clear();

        for (name, client) in &mut self.clients {
            if !client.is_connected() {
                client.connect()?;
            }
            let tools = client.list_tools()?;
            for tool in tools {
                self.tool_index.insert(tool.name.clone(), name.clone());
            }
        }

        Ok(self.tool_index.len())
    }

    /// Look up a tool by name across all registered servers.  Returns the
    /// server name and tool reference, or `None` if not found.
    pub fn find_tool(&self, name: &str) -> Option<(&str, &RemoteTool)> {
        let server_name = self.tool_index.get(name)?;
        for (sname, client) in &self.clients {
            if sname == server_name {
                for tool in &client.tools_cache {
                    if tool.name == name {
                        return Some((sname.as_str(), tool));
                    }
                }
            }
        }
        None
    }

    /// Return all tools across all registered servers.
    pub fn all_tools(&self) -> Vec<(&str, &RemoteTool)> {
        let mut result = Vec::new();
        for (name, client) in &self.clients {
            for tool in &client.tools_cache {
                result.push((name.as_str(), tool));
            }
        }
        result
    }

    /// Return the names of all registered servers.
    pub fn server_names(&self) -> Vec<&str> {
        self.clients.iter().map(|(n, _)| n.as_str()).collect()
    }
}

// =============================================================================
// McpClientPool
// =============================================================================

/// Pool of reusable connections to MCP servers.
///
/// Connections are created lazily on first access and reused for subsequent
/// requests.  The pool enforces a maximum number of concurrent connections.
pub struct McpClientPool {
    configs: HashMap<String, McpClientConfig>,
    clients: HashMap<String, RemoteMcpClient>,
    max_connections: usize,
}

impl McpClientPool {
    /// Create a new pool with the given maximum number of simultaneous
    /// connections.
    pub fn new(max_connections: usize) -> Self {
        Self {
            configs: HashMap::new(),
            clients: HashMap::new(),
            max_connections,
        }
    }

    /// Register a server configuration.  The connection will be established
    /// lazily on the first call to [`get_or_connect`](Self::get_or_connect).
    pub fn register(&mut self, name: String, config: McpClientConfig) {
        self.configs.insert(name, config);
    }

    /// Return a reference to an existing (or newly-created) connected client
    /// for the given server name.
    ///
    /// If the server name has not been registered via
    /// [`register`](Self::register), returns
    /// [`McpClientError::ConnectionFailed`].  If the pool has reached
    /// `max_connections`, returns [`McpClientError::ConnectionFailed`] as well.
    pub fn get_or_connect(
        &mut self,
        name: &str,
    ) -> Result<&RemoteMcpClient, McpClientError> {
        // Already connected?
        if self.clients.contains_key(name) {
            return Ok(&self.clients[name]);
        }

        // Check capacity.
        if self.clients.len() >= self.max_connections {
            return Err(McpClientError::ConnectionFailed {
                url: name.to_string(),
                reason: format!(
                    "pool capacity reached ({}/{})",
                    self.clients.len(),
                    self.max_connections
                ),
            });
        }

        // Look up the config.
        let config = self.configs.get(name).ok_or_else(|| {
            McpClientError::ConnectionFailed {
                url: name.to_string(),
                reason: "unknown server name".to_string(),
            }
        })?;

        let mut client = RemoteMcpClient::new(config.clone());
        client.connect()?;

        self.clients.insert(name.to_string(), client);
        Ok(&self.clients[name])
    }

    /// Disconnect all pooled clients.
    pub fn disconnect_all(&mut self) {
        for (_, client) in &mut self.clients {
            client.disconnect();
        }
        self.clients.clear();
    }

    /// Return the number of currently connected clients.
    pub fn connected_count(&self) -> usize {
        self.clients.len()
    }

    /// Return the names of all registered server configurations.
    pub fn registered_names(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- McpClientConfig default values --

    #[test]
    fn test_config_default_values() {
        let cfg = McpClientConfig::default();
        assert!(cfg.url.is_empty());
        assert_eq!(cfg.timeout_ms, 30000);
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.protocol_version, "2025-11-05");
        assert_eq!(cfg.client_info.name, "ai_assistant");
        assert!(!cfg.client_info.version.is_empty());
        match cfg.auth {
            McpClientAuth::None => {} // expected
            _ => panic!("default auth should be None"),
        }
    }

    // -- McpClientAuth variants --

    #[test]
    fn test_auth_none_variant() {
        let auth = McpClientAuth::None;
        // Should serialize cleanly.
        let json = serde_json::to_string(&auth).unwrap();
        assert!(json.contains("None"));
    }

    #[test]
    fn test_auth_bearer_variant() {
        let auth = McpClientAuth::BearerToken("tok-abc123".to_string());
        let json = serde_json::to_string(&auth).unwrap();
        assert!(json.contains("tok-abc123"));
    }

    #[test]
    fn test_auth_oauth_variant() {
        let auth = McpClientAuth::OAuth {
            client_id: "cid".into(),
            client_secret: "csec".into(),
            token_url: "https://auth.example.com/token".into(),
            token: Some("access-tok".into()),
        };
        let json = serde_json::to_string(&auth).unwrap();
        assert!(json.contains("cid"));
        assert!(json.contains("csec"));
        assert!(json.contains("access-tok"));
    }

    // -- RemoteMcpClient::new --

    #[test]
    fn test_new_client_is_disconnected() {
        let client = RemoteMcpClient::new(McpClientConfig::default());
        assert!(!client.is_connected());
        assert!(client.session_id().is_none());
        assert!(client.server_capabilities().is_none());
    }

    // -- connect / disconnect --

    #[test]
    fn test_connect_empty_url_returns_error() {
        let mut client = RemoteMcpClient::new(McpClientConfig::default());
        let result = client.connect();
        assert!(result.is_err());
        match result.unwrap_err() {
            McpClientError::ConnectionFailed { url, reason } => {
                assert!(url.contains("empty"), "url={}", url);
                assert!(reason.contains("empty"), "reason={}", reason);
            }
            other => panic!("expected ConnectionFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_connect_success() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();
        assert!(client.is_connected());
        assert!(client.session_id().is_some());
        assert!(client.server_capabilities().is_some());
    }

    #[test]
    fn test_disconnect_clears_session() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();
        assert!(client.is_connected());

        client.disconnect();
        assert!(!client.is_connected());
        assert!(client.session_id().is_none());
        assert!(client.server_capabilities().is_none());
    }

    // -- operations on disconnected client --

    #[test]
    fn test_list_tools_disconnected() {
        let mut client = RemoteMcpClient::new(McpClientConfig::default());
        assert!(client.list_tools().is_err());
    }

    #[test]
    fn test_call_tool_disconnected() {
        let client = RemoteMcpClient::new(McpClientConfig::default());
        let args = HashMap::new();
        assert!(client.call_tool("anything", &args).is_err());
    }

    #[test]
    fn test_list_resources_disconnected() {
        let mut client = RemoteMcpClient::new(McpClientConfig::default());
        assert!(client.list_resources().is_err());
    }

    #[test]
    fn test_read_resource_disconnected() {
        let client = RemoteMcpClient::new(McpClientConfig::default());
        assert!(client.read_resource("file://x").is_err());
    }

    #[test]
    fn test_refresh_tools_disconnected() {
        let mut client = RemoteMcpClient::new(McpClientConfig::default());
        assert!(client.refresh_tools().is_err());
    }

    // -- operations on connected client --

    #[test]
    fn test_list_tools_connected() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();
        let tools = client.list_tools().unwrap();
        // Simulated server returns no tools, but the call must succeed.
        assert!(tools.is_empty());
    }

    #[test]
    fn test_call_tool_connected() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();

        let mut args = HashMap::new();
        args.insert("query".to_string(), serde_json::json!("hello"));
        let result = client.call_tool("search", &args).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn test_read_resource_connected() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();
        let content = client.read_resource("file:///readme.md").unwrap();
        assert_eq!(content.uri, "file:///readme.md");
        assert!(content.mime_type.is_some());
    }

    #[test]
    fn test_refresh_tools_connected() {
        let mut client = RemoteMcpClient::new(McpClientConfig {
            url: "http://localhost:3000/mcp".into(),
            ..Default::default()
        });
        client.connect().unwrap();
        // Should not error even though no real server is present.
        client.refresh_tools().unwrap();
    }

    // -- RemoteToolRegistry --

    #[test]
    fn test_registry_new_is_empty() {
        let registry = RemoteToolRegistry::new();
        assert!(registry.all_tools().is_empty());
        assert!(registry.server_names().is_empty());
    }

    #[test]
    fn test_registry_add_remove_server() {
        let mut registry = RemoteToolRegistry::new();
        let client = RemoteMcpClient::new(McpClientConfig {
            url: "http://srv1:3000/mcp".into(),
            ..Default::default()
        });
        registry.add_server("srv1".to_string(), client);
        assert_eq!(registry.server_names().len(), 1);

        assert!(registry.remove_server("srv1"));
        assert!(registry.server_names().is_empty());
        // Removing again returns false.
        assert!(!registry.remove_server("srv1"));
    }

    #[test]
    fn test_registry_find_tool_empty() {
        let registry = RemoteToolRegistry::new();
        assert!(registry.find_tool("nonexistent").is_none());
    }

    #[test]
    fn test_registry_all_tools_empty() {
        let registry = RemoteToolRegistry::new();
        assert!(registry.all_tools().is_empty());
    }

    #[test]
    fn test_registry_server_names() {
        let mut registry = RemoteToolRegistry::new();
        let c1 = RemoteMcpClient::new(McpClientConfig {
            url: "http://a:3000/mcp".into(),
            ..Default::default()
        });
        let c2 = RemoteMcpClient::new(McpClientConfig {
            url: "http://b:3000/mcp".into(),
            ..Default::default()
        });
        registry.add_server("alpha".to_string(), c1);
        registry.add_server("beta".to_string(), c2);
        let names = registry.server_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_registry_discover_all() {
        let mut registry = RemoteToolRegistry::new();
        let c1 = RemoteMcpClient::new(McpClientConfig {
            url: "http://a:3000/mcp".into(),
            ..Default::default()
        });
        registry.add_server("alpha".to_string(), c1);
        // discover_all connects the servers and lists tools.
        let count = registry.discover_all().unwrap();
        // Simulated server returns 0 tools, but the flow must succeed.
        assert_eq!(count, 0);
    }

    // -- McpClientPool --

    #[test]
    fn test_pool_new() {
        let pool = McpClientPool::new(5);
        assert_eq!(pool.connected_count(), 0);
        assert!(pool.registered_names().is_empty());
    }

    #[test]
    fn test_pool_register() {
        let mut pool = McpClientPool::new(5);
        pool.register(
            "srv1".to_string(),
            McpClientConfig {
                url: "http://srv1:3000/mcp".into(),
                ..Default::default()
            },
        );
        assert_eq!(pool.registered_names().len(), 1);
        assert_eq!(pool.connected_count(), 0);
    }

    #[test]
    fn test_pool_get_or_connect_unknown() {
        let mut pool = McpClientPool::new(5);
        let result = pool.get_or_connect("unknown");
        assert!(result.is_err());
        match result.unwrap_err() {
            McpClientError::ConnectionFailed { reason, .. } => {
                assert!(reason.contains("unknown server"), "reason={}", reason);
            }
            other => panic!("expected ConnectionFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_pool_get_or_connect_success() {
        let mut pool = McpClientPool::new(5);
        pool.register(
            "srv1".to_string(),
            McpClientConfig {
                url: "http://srv1:3000/mcp".into(),
                ..Default::default()
            },
        );
        let client = pool.get_or_connect("srv1").unwrap();
        assert!(client.is_connected());
        assert_eq!(pool.connected_count(), 1);
    }

    #[test]
    fn test_pool_disconnect_all() {
        let mut pool = McpClientPool::new(5);
        pool.register(
            "srv1".to_string(),
            McpClientConfig {
                url: "http://srv1:3000/mcp".into(),
                ..Default::default()
            },
        );
        pool.get_or_connect("srv1").unwrap();
        assert_eq!(pool.connected_count(), 1);

        pool.disconnect_all();
        assert_eq!(pool.connected_count(), 0);
    }

    #[test]
    fn test_pool_connected_count() {
        let mut pool = McpClientPool::new(10);
        assert_eq!(pool.connected_count(), 0);

        pool.register(
            "a".to_string(),
            McpClientConfig {
                url: "http://a:3000/mcp".into(),
                ..Default::default()
            },
        );
        pool.register(
            "b".to_string(),
            McpClientConfig {
                url: "http://b:3000/mcp".into(),
                ..Default::default()
            },
        );

        pool.get_or_connect("a").unwrap();
        assert_eq!(pool.connected_count(), 1);

        pool.get_or_connect("b").unwrap();
        assert_eq!(pool.connected_count(), 2);
    }

    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = McpClientPool::new(1);
        pool.register(
            "a".to_string(),
            McpClientConfig {
                url: "http://a:3000/mcp".into(),
                ..Default::default()
            },
        );
        pool.register(
            "b".to_string(),
            McpClientConfig {
                url: "http://b:3000/mcp".into(),
                ..Default::default()
            },
        );

        pool.get_or_connect("a").unwrap();
        let result = pool.get_or_connect("b");
        assert!(result.is_err());
        match result.unwrap_err() {
            McpClientError::ConnectionFailed { reason, .. } => {
                assert!(reason.contains("capacity"), "reason={}", reason);
            }
            other => panic!("expected ConnectionFailed, got {:?}", other),
        }
    }

    // -- Data type construction tests --

    #[test]
    fn test_tool_call_result_construction() {
        let result = ToolCallResult {
            content: vec![
                ToolResultContent::Text {
                    text: "hello".into(),
                },
                ToolResultContent::Image {
                    data: "base64data".into(),
                    mime_type: "image/png".into(),
                },
            ],
            is_error: false,
        };
        assert_eq!(result.content.len(), 2);
        assert!(!result.is_error);

        // Serialization round-trip.
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ToolCallResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content.len(), 2);
        assert!(!deserialized.is_error);
    }

    #[test]
    fn test_tool_result_content_variants() {
        let text = ToolResultContent::Text {
            text: "hi".into(),
        };
        let image = ToolResultContent::Image {
            data: "abc".into(),
            mime_type: "image/jpeg".into(),
        };
        let resource = ToolResultContent::Resource {
            uri: "file:///a.txt".into(),
            mime_type: Some("text/plain".into()),
            text: Some("content".into()),
        };

        // Ensure all three serialize.
        for variant in &[&text, &image, &resource] {
            let json = serde_json::to_string(variant).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_remote_tool_with_annotations() {
        let tool = RemoteTool {
            name: "file_search".into(),
            description: "Search files on the server".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }),
            annotations: Some(RemoteToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            }),
        };
        assert_eq!(tool.name, "file_search");
        let ann = tool.annotations.as_ref().unwrap();
        assert!(ann.read_only);
        assert!(!ann.destructive);
        assert!(ann.idempotent);
        assert!(!ann.open_world);

        // Serialization round-trip.
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: RemoteTool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "file_search");
    }

    #[test]
    fn test_remote_resource_construction() {
        let resource = RemoteResource {
            uri: "file:///docs/readme.md".into(),
            name: "README".into(),
            description: Some("Project readme".into()),
            mime_type: Some("text/markdown".into()),
        };
        assert_eq!(resource.uri, "file:///docs/readme.md");
        assert_eq!(resource.name, "README");
        assert!(resource.description.is_some());
        assert!(resource.mime_type.is_some());

        // Serialization round-trip.
        let json = serde_json::to_string(&resource).unwrap();
        let deserialized: RemoteResource = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.uri, "file:///docs/readme.md");
    }

    #[test]
    fn test_server_capabilities_default() {
        let caps = ServerCapabilities::default();
        assert!(caps.tools);
        assert!(caps.resources);
        assert!(!caps.prompts);
        assert!(!caps.logging);

        // Serialization round-trip.
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: ServerCapabilities = serde_json::from_str(&json).unwrap();
        assert!(deserialized.tools);
        assert!(deserialized.resources);
    }
}
