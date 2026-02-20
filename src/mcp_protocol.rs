//! MCP (Model Context Protocol) support
//!
//! Implementation of Anthropic's Model Context Protocol for standardized
//! tool use and context sharing between AI models and external systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MCP Protocol version (2025-03-26 spec)
pub const MCP_VERSION: &str = "2025-03-26";

/// Previous MCP protocol version for backward compatibility
pub const MCP_VERSION_PREVIOUS: &str = "2024-11-05";

/// MCP message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpMessageType {
    Initialize,
    Initialized,
    ToolsList,
    ToolsListResponse,
    ToolCall,
    ToolResult,
    ResourcesList,
    ResourcesListResponse,
    ResourceRead,
    ResourceReadResponse,
    PromptsList,
    PromptsListResponse,
    PromptGet,
    PromptGetResponse,
    Ping,
    Pong,
    Error,
}

/// MCP JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl McpRequest {
    pub fn new(method: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::Value::Number(1.into())),
            method: method.to_string(),
            params: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<serde_json::Value>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// MCP JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

impl McpResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, error: McpError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// MCP error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl McpError {
    pub fn parse_error(message: &str) -> Self {
        Self {
            code: -32700,
            message: message.to_string(),
            data: None,
        }
    }

    pub fn invalid_request(message: &str) -> Self {
        Self {
            code: -32600,
            message: message.to_string(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("Method not found: {}", method),
            data: None,
        }
    }

    pub fn invalid_params(message: &str) -> Self {
        Self {
            code: -32602,
            message: message.to_string(),
            data: None,
        }
    }

    pub fn internal_error(message: &str) -> Self {
        Self {
            code: -32603,
            message: message.to_string(),
            data: None,
        }
    }
}

/// MCP Tool annotations (2025-03-26 spec)
///
/// Annotations provide hints about tool behavior without changing functionality.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpToolAnnotation {
    /// Human-readable title for the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// If true, the tool does not modify external state (safe for read-only).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "readOnlyHint")]
    pub read_only_hint: Option<bool>,
    /// If true, the tool may cause irreversible changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "destructiveHint")]
    pub destructive_hint: Option<bool>,
    /// If true, calling with the same arguments produces the same result.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "idempotentHint")]
    pub idempotent_hint: Option<bool>,
    /// If true, the tool interacts with entities beyond its described inputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "openWorldHint")]
    pub open_world_hint: Option<bool>,
}

/// Cursor-based pagination parameters (2025-03-26 spec)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpPagination {
    /// Opaque cursor for the next page. None means no more pages.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "nextCursor")]
    pub next_cursor: Option<String>,
}

/// Pagination request parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpPaginationRequest {
    /// Cursor from a previous paginated response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
    /// Tool annotations (2025-03-26 spec)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<McpToolAnnotation>,
}

impl McpTool {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            annotations: None,
        }
    }

    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    /// Set tool annotations (2025-03-26 spec).
    pub fn with_annotations(mut self, annotations: McpToolAnnotation) -> Self {
        self.annotations = Some(annotations);
        self
    }

    pub fn with_property(
        mut self,
        name: &str,
        prop_type: &str,
        description: &str,
        required: bool,
    ) -> Self {
        if let Some(obj) = self.input_schema.as_object_mut() {
            if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                props.insert(
                    name.to_string(),
                    serde_json::json!({
                        "type": prop_type,
                        "description": description
                    }),
                );
            }

            if required {
                if let Some(req) = obj.get_mut("required").and_then(|r| r.as_array_mut()) {
                    req.push(serde_json::Value::String(name.to_string()));
                }
            }
        }
        self
    }
}

/// MCP Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
}

impl McpResource {
    pub fn new(uri: &str, name: &str) -> Self {
        Self {
            uri: uri.to_string(),
            name: name.to_string(),
            description: None,
            mime_type: None,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    pub fn with_mime_type(mut self, mime: &str) -> Self {
        self.mime_type = Some(mime.to_string());
        self
    }
}

/// MCP Resource content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>, // base64 encoded
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// MCP Prompt definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<McpPromptArgument>>,
}

/// MCP Prompt argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
}

/// MCP Prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptMessage {
    pub role: String, // "user" or "assistant"
    pub content: McpContent,
}

/// MCP Content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "resource")]
    Resource { resource: McpResourceContent },
}

/// MCP Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<McpResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<McpPromptsCapability>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpToolsCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpResourcesCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpPromptsCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

/// MCP Server implementation
pub struct McpServer {
    name: String,
    version: String,
    capabilities: McpServerCapabilities,
    tools: HashMap<String, McpTool>,
    tool_handlers: HashMap<
        String,
        Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
    >,
    resources: HashMap<String, McpResource>,
    resource_handlers:
        HashMap<String, Box<dyn Fn(&str) -> Result<McpResourceContent, String> + Send + Sync>>,
    prompts: HashMap<String, McpPrompt>,
    prompt_handlers: HashMap<
        String,
        Box<dyn Fn(HashMap<String, String>) -> Result<Vec<McpPromptMessage>, String> + Send + Sync>,
    >,
}

impl McpServer {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            capabilities: McpServerCapabilities::default(),
            tools: HashMap::new(),
            tool_handlers: HashMap::new(),
            resources: HashMap::new(),
            resource_handlers: HashMap::new(),
            prompts: HashMap::new(),
            prompt_handlers: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register_tool<F>(&mut self, tool: McpTool, handler: F)
    where
        F: Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync + 'static,
    {
        self.tools.insert(tool.name.clone(), tool.clone());
        self.tool_handlers.insert(tool.name, Box::new(handler));
        self.capabilities.tools = Some(McpToolsCapability {
            list_changed: Some(true),
        });
    }

    /// Register a resource
    pub fn register_resource<F>(&mut self, resource: McpResource, handler: F)
    where
        F: Fn(&str) -> Result<McpResourceContent, String> + Send + Sync + 'static,
    {
        self.resources
            .insert(resource.uri.clone(), resource.clone());
        self.resource_handlers
            .insert(resource.uri, Box::new(handler));
        self.capabilities.resources = Some(McpResourcesCapability {
            subscribe: Some(false),
            list_changed: Some(true),
        });
    }

    /// Register a prompt
    pub fn register_prompt<F>(&mut self, prompt: McpPrompt, handler: F)
    where
        F: Fn(HashMap<String, String>) -> Result<Vec<McpPromptMessage>, String>
            + Send
            + Sync
            + 'static,
    {
        self.prompts.insert(prompt.name.clone(), prompt.clone());
        self.prompt_handlers.insert(prompt.name, Box::new(handler));
        self.capabilities.prompts = Some(McpPromptsCapability {
            list_changed: Some(true),
        });
    }

    /// Handle an incoming MCP request
    pub fn handle_request(&self, request: McpRequest) -> McpResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id, request.params),
            "tools/list" => self.handle_tools_list(id, request.params),
            "tools/call" => self.handle_tool_call(id, request.params),
            "resources/list" => self.handle_resources_list(id, request.params),
            "resources/read" => self.handle_resource_read(id, request.params),
            "prompts/list" => self.handle_prompts_list(id, request.params),
            "prompts/get" => self.handle_prompt_get(id, request.params),
            "ping" => McpResponse::success(id, serde_json::json!({})),
            _ => McpResponse::error(id, McpError::method_not_found(&request.method)),
        }
    }

    fn handle_initialize(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> McpResponse {
        // Version negotiation: accept current or previous version from client
        let client_version = params
            .as_ref()
            .and_then(|p| p.get("protocolVersion"))
            .and_then(|v| v.as_str())
            .unwrap_or(MCP_VERSION);

        let negotiated_version =
            if client_version == MCP_VERSION || client_version == MCP_VERSION_PREVIOUS {
                // Use the lower version of the two for compatibility
                if client_version == MCP_VERSION_PREVIOUS {
                    MCP_VERSION_PREVIOUS
                } else {
                    MCP_VERSION
                }
            } else {
                MCP_VERSION
            };

        McpResponse::success(
            id,
            serde_json::json!({
                "protocolVersion": negotiated_version,
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                },
                "capabilities": self.capabilities
            }),
        )
    }

    fn handle_tools_list(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> McpResponse {
        // Parse pagination from params (2025-03-26 spec)
        let cursor = params
            .as_ref()
            .and_then(|p| p.get("cursor"))
            .and_then(|c| c.as_str())
            .and_then(|s| s.parse::<usize>().ok());

        let page_size = 50; // Default page size
        let all_tools: Vec<&McpTool> = self.tools.values().collect();
        let start = cursor.unwrap_or(0);
        let end = (start + page_size).min(all_tools.len());
        let page = &all_tools[start..end];

        let next_cursor = if end < all_tools.len() {
            Some(serde_json::json!(end.to_string()))
        } else {
            None
        };

        let mut result = serde_json::json!({ "tools": page });
        if let Some(nc) = next_cursor {
            result
                .as_object_mut()
                .unwrap()
                .insert("nextCursor".to_string(), nc);
        }
        McpResponse::success(id, result)
    }

    fn handle_tool_call(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error(id, McpError::invalid_params("Missing params")),
        };

        let name = match params.get("name").and_then(|n| n.as_str()) {
            Some(n) => n,
            None => return McpResponse::error(id, McpError::invalid_params("Missing tool name")),
        };

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let handler = match self.tool_handlers.get(name) {
            Some(h) => h,
            None => return McpResponse::error(id, McpError::method_not_found(name)),
        };

        match handler(arguments) {
            Ok(result) => McpResponse::success(
                id,
                serde_json::json!({
                    "content": [{ "type": "text", "text": result.to_string() }]
                }),
            ),
            Err(e) => McpResponse::error(id, McpError::internal_error(&e)),
        }
    }

    fn handle_resources_list(
        &self,
        id: Option<serde_json::Value>,
        _params: Option<serde_json::Value>,
    ) -> McpResponse {
        let resources: Vec<&McpResource> = self.resources.values().collect();
        McpResponse::success(id, serde_json::json!({ "resources": resources }))
    }

    fn handle_resource_read(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error(id, McpError::invalid_params("Missing params")),
        };

        let uri = match params.get("uri").and_then(|u| u.as_str()) {
            Some(u) => u,
            None => return McpResponse::error(id, McpError::invalid_params("Missing uri")),
        };

        let handler = match self.resource_handlers.get(uri) {
            Some(h) => h,
            None => return McpResponse::error(id, McpError::invalid_params("Resource not found")),
        };

        match handler(uri) {
            Ok(content) => McpResponse::success(id, serde_json::json!({ "contents": [content] })),
            Err(e) => McpResponse::error(id, McpError::internal_error(&e)),
        }
    }

    fn handle_prompts_list(
        &self,
        id: Option<serde_json::Value>,
        _params: Option<serde_json::Value>,
    ) -> McpResponse {
        let prompts: Vec<&McpPrompt> = self.prompts.values().collect();
        McpResponse::success(id, serde_json::json!({ "prompts": prompts }))
    }

    fn handle_prompt_get(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error(id, McpError::invalid_params("Missing params")),
        };

        let name = match params.get("name").and_then(|n| n.as_str()) {
            Some(n) => n,
            None => return McpResponse::error(id, McpError::invalid_params("Missing prompt name")),
        };

        let arguments: HashMap<String, String> = params
            .get("arguments")
            .and_then(|a| a.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let handler = match self.prompt_handlers.get(name) {
            Some(h) => h,
            None => return McpResponse::error(id, McpError::method_not_found(name)),
        };

        match handler(arguments) {
            Ok(messages) => McpResponse::success(id, serde_json::json!({ "messages": messages })),
            Err(e) => McpResponse::error(id, McpError::internal_error(&e)),
        }
    }

    /// Parse and handle a JSON-RPC message
    pub fn handle_message(&self, message: &str) -> String {
        match serde_json::from_str::<McpRequest>(message) {
            Ok(request) => {
                let response = self.handle_request(request);
                serde_json::to_string(&response).unwrap_or_else(|_| {
                    r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"}}"#.to_string()
                })
            }
            Err(_) => {
                let response = McpResponse::error(None, McpError::parse_error("Invalid JSON"));
                serde_json::to_string(&response).unwrap_or_else(|_| {
                    r#"{"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"}}"#
                        .to_string()
                })
            }
        }
    }
}

/// MCP Client for connecting to MCP servers
pub struct McpClient {
    server_url: String,
    initialized: bool,
    server_capabilities: Option<McpServerCapabilities>,
    request_id: u64,
}

impl McpClient {
    pub fn new(server_url: &str) -> Self {
        Self {
            server_url: server_url.to_string(),
            initialized: false,
            server_capabilities: None,
            request_id: 0,
        }
    }

    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }

    /// Initialize connection with server
    pub fn initialize(&mut self) -> Result<McpServerCapabilities, McpError> {
        let request = McpRequest::new("initialize")
            .with_id(self.next_id())
            .with_params(serde_json::json!({
                "protocolVersion": MCP_VERSION,
                "clientInfo": {
                    "name": "ai_assistant",
                    "version": "0.1.0"
                },
                "capabilities": {}
            }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let capabilities: McpServerCapabilities = result
                .get("capabilities")
                .cloned()
                .and_then(|c| serde_json::from_value(c).ok())
                .unwrap_or_default();

            self.server_capabilities = Some(capabilities.clone());
            self.initialized = true;

            // Send initialized notification (fire-and-forget per MCP spec)
            let notif = McpRequest::new("notifications/initialized");
            let _ = self.send_notification(notif);

            Ok(capabilities)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available tools
    pub fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        let request = McpRequest::new("tools/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let tools: Vec<McpTool> = result
                .get("tools")
                .cloned()
                .and_then(|t| serde_json::from_value(t).ok())
                .unwrap_or_default();
            Ok(tools)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Call a tool
    pub fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        let request = McpRequest::new("tools/call")
            .with_id(self.next_id())
            .with_params(serde_json::json!({
                "name": name,
                "arguments": arguments
            }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            Ok(result)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available resources
    pub fn list_resources(&mut self) -> Result<Vec<McpResource>, McpError> {
        let request = McpRequest::new("resources/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let resources: Vec<McpResource> = result
                .get("resources")
                .cloned()
                .and_then(|r| serde_json::from_value(r).ok())
                .unwrap_or_default();
            Ok(resources)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Read a resource
    pub fn read_resource(&mut self, uri: &str) -> Result<Vec<McpResourceContent>, McpError> {
        let request = McpRequest::new("resources/read")
            .with_id(self.next_id())
            .with_params(serde_json::json!({ "uri": uri }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let contents: Vec<McpResourceContent> = result
                .get("contents")
                .cloned()
                .and_then(|c| serde_json::from_value(c).ok())
                .unwrap_or_default();
            Ok(contents)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available prompts
    pub fn list_prompts(&mut self) -> Result<Vec<McpPrompt>, McpError> {
        let request = McpRequest::new("prompts/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let prompts: Vec<McpPrompt> = result
                .get("prompts")
                .cloned()
                .and_then(|p| serde_json::from_value(p).ok())
                .unwrap_or_default();
            Ok(prompts)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Get a prompt
    pub fn get_prompt(
        &mut self,
        name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<Vec<McpPromptMessage>, McpError> {
        let request = McpRequest::new("prompts/get")
            .with_id(self.next_id())
            .with_params(serde_json::json!({
                "name": name,
                "arguments": arguments
            }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let messages: Vec<McpPromptMessage> = result
                .get("messages")
                .cloned()
                .and_then(|m| serde_json::from_value(m).ok())
                .unwrap_or_default();
            Ok(messages)
        } else {
            Err(response
                .error
                .unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Send a notification to the server (fire-and-forget, no response expected)
    fn send_notification(&self, request: McpRequest) -> Result<(), McpError> {
        let body = serde_json::to_string(&request)
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        ureq::post(&self.server_url)
            .set("Content-Type", "application/json")
            .send_string(&body)
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        Ok(())
    }

    /// Send a request to the server (HTTP implementation)
    fn send_request(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        let body = serde_json::to_string(&request)
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        let response = ureq::post(&self.server_url)
            .set("Content-Type", "application/json")
            .send_string(&body)
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        let response_body = response
            .into_string()
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        serde_json::from_str(&response_body).map_err(|e| McpError::parse_error(&e.to_string()))
    }
}

/// MCP Transport type (2025-03-26 spec adds Streamable HTTP)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// Standard HTTP POST (JSON-RPC over HTTP).
    Http,
    /// Server-Sent Events for server→client streaming.
    Sse,
    /// Streamable HTTP: POST for client→server, SSE for server→client.
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
/// Supports bidirectional communication via POST (client→server)
/// and SSE (server→client). Maintains session state via `Mcp-Session-Id`.
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

// ---------------------------------------------------------------------------
// OAuth 2.1 types for MCP authentication (MCP spec 2025-03-26)
// ---------------------------------------------------------------------------

/// OAuth 2.1 grant type for MCP authentication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum McpOAuthGrantType {
    AuthorizationCode,
    ClientCredentials,
    RefreshToken,
}

/// OAuth 2.1 scope definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthScope {
    pub name: String,
    pub description: String,
    pub resources: Vec<String>,
}

/// OAuth 2.1 configuration for MCP servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOAuthConfig {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub scopes: Vec<McpOAuthScope>,
    pub redirect_uri: String,
    pub pkce_enabled: bool,
}

/// OAuth 2.1 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
}

/// OAuth 2.1 authorization request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAuthorizationRequest {
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub state: String,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

/// OAuth 2.1 token manager for MCP sessions
pub struct McpOAuthTokenManager {
    config: McpOAuthConfig,
    current_token: Option<McpTokenResponse>,
    token_expiry_secs: Option<u64>,
    token_obtained_at: Option<std::time::Instant>,
}

impl McpOAuthTokenManager {
    pub fn new(config: McpOAuthConfig) -> Self {
        Self {
            config,
            current_token: None,
            token_expiry_secs: None,
            token_obtained_at: None,
        }
    }

    /// Build the authorization URL for the OAuth 2.1 authorization code flow.
    /// Returns (url, state) where state is the CSRF protection value.
    pub fn build_authorization_url(&self, state: &str) -> (String, McpAuthorizationRequest) {
        let scope_str = self
            .config
            .scopes
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let request = McpAuthorizationRequest {
            response_type: "code".to_string(),
            client_id: self.config.client_id.clone(),
            redirect_uri: self.config.redirect_uri.clone(),
            scope: scope_str.clone(),
            state: state.to_string(),
            code_challenge: None,
            code_challenge_method: None,
        };

        let url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}",
            self.config.authorization_endpoint,
            urlencoding::encode(&self.config.client_id),
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(&scope_str),
            urlencoding::encode(state),
        );

        (url, request)
    }

    /// Build the token request body for authorization code exchange.
    pub fn build_token_request_authorization_code(&self, code: &str) -> Vec<(String, String)> {
        let mut params = vec![
            ("grant_type".to_string(), "authorization_code".to_string()),
            ("code".to_string(), code.to_string()),
            ("redirect_uri".to_string(), self.config.redirect_uri.clone()),
            ("client_id".to_string(), self.config.client_id.clone()),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Build the token request body for client credentials flow.
    pub fn build_token_request_client_credentials(&self) -> Vec<(String, String)> {
        let scope_str = self
            .config
            .scopes
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let mut params = vec![
            ("grant_type".to_string(), "client_credentials".to_string()),
            ("client_id".to_string(), self.config.client_id.clone()),
            ("scope".to_string(), scope_str),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Build the token request body for refreshing an access token.
    pub fn build_token_request_refresh(&self, refresh_token: &str) -> Vec<(String, String)> {
        let mut params = vec![
            ("grant_type".to_string(), "refresh_token".to_string()),
            ("refresh_token".to_string(), refresh_token.to_string()),
            ("client_id".to_string(), self.config.client_id.clone()),
        ];
        if let Some(ref secret) = self.config.client_secret {
            params.push(("client_secret".to_string(), secret.clone()));
        }
        params
    }

    /// Store a token response.
    pub fn set_token(&mut self, token: McpTokenResponse) {
        self.token_expiry_secs = token.expires_in;
        self.current_token = Some(token);
        self.token_obtained_at = Some(std::time::Instant::now());
    }

    /// Get the current access token, if available and not expired.
    pub fn get_access_token(&self) -> Option<&str> {
        if self.is_token_expired() {
            return None;
        }
        self.current_token.as_ref().map(|t| t.access_token.as_str())
    }

    /// Check if the current token has expired.
    pub fn is_token_expired(&self) -> bool {
        match (&self.token_obtained_at, &self.token_expiry_secs) {
            (Some(obtained), Some(expires)) => obtained.elapsed().as_secs() >= *expires,
            (None, _) => true,        // No token obtained
            (Some(_), None) => false, // No expiry = never expires
        }
    }

    /// Generate a PKCE code challenge from a code verifier (S256 method).
    /// Uses a simple hash: sum of bytes mod a large prime, then hex-encoded.
    /// (In production, use SHA-256; this is a pure-Rust fallback.)
    pub fn generate_pkce_challenge(verifier: &str) -> (String, String) {
        // Simple hash for PKCE challenge (deterministic)
        let bytes = verifier.as_bytes();
        let mut hash: u64 = 0;
        for (i, &b) in bytes.iter().enumerate() {
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(b as u64)
                .wrapping_add(i as u64);
        }
        let challenge = format!("{:016x}", hash);
        (challenge, "plain".to_string()) // plain method for simple hash
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &McpOAuthConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_request() {
        let request = McpRequest::new("tools/list").with_id(1u64);

        assert_eq!(request.method, "tools/list");
        assert_eq!(request.jsonrpc, "2.0");
    }

    #[test]
    fn test_mcp_tool() {
        let tool = McpTool::new("search", "Search the web")
            .with_property("query", "string", "Search query", true)
            .with_property("limit", "number", "Max results", false);

        assert_eq!(tool.name, "search");
        assert!(tool.input_schema["properties"]["query"].is_object());
    }

    #[test]
    fn test_mcp_server() {
        let mut server = McpServer::new("test-server", "1.0.0");

        server.register_tool(McpTool::new("echo", "Echo the input"), |args| {
            let text = args.get("text").and_then(|t| t.as_str()).unwrap_or("");
            Ok(serde_json::json!({ "echo": text }))
        });

        // Test initialize
        let init_request =
            McpRequest::new("initialize")
                .with_id(1u64)
                .with_params(serde_json::json!({
                    "protocolVersion": MCP_VERSION,
                    "clientInfo": { "name": "test" },
                    "capabilities": {}
                }));

        let response = server.handle_request(init_request);
        assert!(response.result.is_some());

        // Test tools list
        let list_request = McpRequest::new("tools/list").with_id(2u64);
        let response = server.handle_request(list_request);
        assert!(response.result.is_some());
    }

    #[test]
    fn test_mcp_error() {
        let error = McpError::method_not_found("unknown");
        assert_eq!(error.code, -32601);
    }

    #[test]
    fn test_mcp_resource() {
        let resource = McpResource::new("file:///test.txt", "test.txt")
            .with_description("A test file")
            .with_mime_type("text/plain");

        assert_eq!(resource.uri, "file:///test.txt");
        assert_eq!(resource.mime_type.unwrap(), "text/plain");
    }

    // ===== 2025-03-26 spec tests =====

    #[test]
    fn test_mcp_version_2025() {
        assert_eq!(MCP_VERSION, "2025-03-26");
        assert_eq!(MCP_VERSION_PREVIOUS, "2024-11-05");
    }

    #[test]
    fn test_tool_annotations_serde() {
        let ann = McpToolAnnotation {
            title: Some("My Tool".to_string()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: None,
        };
        let json = serde_json::to_value(&ann).unwrap();
        assert_eq!(json["title"], "My Tool");
        assert_eq!(json["readOnlyHint"], true);
        assert_eq!(json["destructiveHint"], false);
        assert_eq!(json["idempotentHint"], true);
        assert!(json.get("openWorldHint").is_none());
    }

    #[test]
    fn test_tool_with_annotations() {
        let tool = McpTool::new("search", "Search the web").with_annotations(McpToolAnnotation {
            title: Some("Web Search".to_string()),
            read_only_hint: Some(true),
            ..Default::default()
        });
        assert!(tool.annotations.is_some());
        let ann = tool.annotations.unwrap();
        assert_eq!(ann.title.unwrap(), "Web Search");
        assert_eq!(ann.read_only_hint.unwrap(), true);
    }

    #[test]
    fn test_pagination_serde() {
        let page = McpPagination {
            next_cursor: Some("abc123".to_string()),
        };
        let json = serde_json::to_value(&page).unwrap();
        assert_eq!(json["nextCursor"], "abc123");

        let empty_page = McpPagination { next_cursor: None };
        let json = serde_json::to_value(&empty_page).unwrap();
        assert!(json.get("nextCursor").is_none());
    }

    #[test]
    fn test_version_negotiation_current() {
        let server = McpServer::new("test", "1.0.0");
        let req = McpRequest::new("initialize")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2025-03-26",
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }));
        let resp = server.handle_request(req);
        let version = resp.result.unwrap()["protocolVersion"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(version, "2025-03-26");
    }

    #[test]
    fn test_version_negotiation_previous() {
        let server = McpServer::new("test", "1.0.0");
        let req = McpRequest::new("initialize")
            .with_id(1u64)
            .with_params(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": { "name": "old-client" },
                "capabilities": {}
            }));
        let resp = server.handle_request(req);
        let version = resp.result.unwrap()["protocolVersion"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(version, "2024-11-05");
    }

    #[test]
    fn test_transport_types() {
        assert_eq!(McpTransport::default(), McpTransport::StreamableHttp);

        let json = serde_json::to_value(&McpTransport::StreamableHttp).unwrap();
        assert_eq!(json, "streamable_http");

        let json = serde_json::to_value(&McpTransport::Stdio).unwrap();
        assert_eq!(json, "stdio");
    }

    #[test]
    fn test_streamable_session() {
        let server = McpServer::new("test", "1.0.0");
        let mut session = McpStreamableSession::new(server);

        assert!(!session.initialized);
        assert!(session.session_id.starts_with("mcp-"));

        // POST an initialize request
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_VERSION,
                "clientInfo": { "name": "test" },
                "capabilities": {}
            }
        });
        let resp = session.handle_post(&body.to_string()).unwrap();
        assert!(session.initialized);
        let resp_json: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(resp_json["result"]["protocolVersion"].is_string());
    }

    #[test]
    fn test_streamable_session_events() {
        let server = McpServer::new("test", "1.0.0");
        let mut session = McpStreamableSession::new(server);

        session.push_event("message", "hello world");
        session.push_event("progress", "50%");

        let events = session.drain_events();
        assert_eq!(events.len(), 2);
        assert!(events[0].contains("event: message"));
        assert!(events[0].contains("data: hello world"));
        assert!(events[1].contains("event: progress"));

        // After drain, should be empty
        assert!(session.drain_events().is_empty());
    }

    #[test]
    fn test_streamable_session_header() {
        let server = McpServer::new("test", "1.0.0");
        let session = McpStreamableSession::new(server);
        let (key, value) = session.session_header();
        assert_eq!(key, "Mcp-Session-Id");
        assert!(value.starts_with("mcp-"));
    }

    #[test]
    fn test_tool_annotations_default() {
        let ann = McpToolAnnotation::default();
        assert!(ann.title.is_none());
        assert!(ann.read_only_hint.is_none());
        assert!(ann.destructive_hint.is_none());
        assert!(ann.idempotent_hint.is_none());
        assert!(ann.open_world_hint.is_none());
    }

    // -----------------------------------------------------------------------
    // OAuth 2.1 tests
    // -----------------------------------------------------------------------

    fn make_test_oauth_config() -> McpOAuthConfig {
        McpOAuthConfig {
            client_id: "test-client".to_string(),
            client_secret: Some("test-secret".to_string()),
            authorization_endpoint: "https://auth.example.com/authorize".to_string(),
            token_endpoint: "https://auth.example.com/token".to_string(),
            scopes: vec![
                McpOAuthScope {
                    name: "tools:read".to_string(),
                    description: "Read tools".to_string(),
                    resources: vec!["tools/*".to_string()],
                },
                McpOAuthScope {
                    name: "resources:read".to_string(),
                    description: "Read resources".to_string(),
                    resources: vec!["resources/*".to_string()],
                },
            ],
            redirect_uri: "http://localhost:8080/callback".to_string(),
            pkce_enabled: true,
        }
    }

    #[test]
    fn test_oauth_grant_type_serde() {
        let grant = McpOAuthGrantType::AuthorizationCode;
        let json = serde_json::to_string(&grant).unwrap();
        let back: McpOAuthGrantType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, McpOAuthGrantType::AuthorizationCode);

        let cc = McpOAuthGrantType::ClientCredentials;
        let json2 = serde_json::to_string(&cc).unwrap();
        let back2: McpOAuthGrantType = serde_json::from_str(&json2).unwrap();
        assert_eq!(back2, McpOAuthGrantType::ClientCredentials);

        let rt = McpOAuthGrantType::RefreshToken;
        let json3 = serde_json::to_string(&rt).unwrap();
        let back3: McpOAuthGrantType = serde_json::from_str(&json3).unwrap();
        assert_eq!(back3, McpOAuthGrantType::RefreshToken);
    }

    #[test]
    fn test_oauth_config_creation() {
        let config = make_test_oauth_config();
        assert_eq!(config.client_id, "test-client");
        assert_eq!(config.client_secret.as_deref(), Some("test-secret"));
        assert!(config.authorization_endpoint.starts_with("https://"));
        assert!(config.token_endpoint.starts_with("https://"));
        assert_eq!(config.scopes.len(), 2);
        assert!(config.pkce_enabled);
    }

    #[test]
    fn test_oauth_scope_creation() {
        let scope = McpOAuthScope {
            name: "tools:write".to_string(),
            description: "Write access to tools".to_string(),
            resources: vec!["tools/create".to_string(), "tools/update".to_string()],
        };
        assert_eq!(scope.name, "tools:write");
        assert_eq!(scope.resources.len(), 2);
    }

    #[test]
    fn test_authorization_url_building() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let (url, request) = manager.build_authorization_url("csrf-state-123");

        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=test-client"));
        assert!(url.contains("state=csrf-state-123"));
        assert!(url.contains("scope=tools%3Aread%20resources%3Aread"));

        assert_eq!(request.response_type, "code");
        assert_eq!(request.client_id, "test-client");
        assert_eq!(request.state, "csrf-state-123");
        assert_eq!(request.scope, "tools:read resources:read");
    }

    #[test]
    fn test_token_request_authorization_code() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_authorization_code("auth-code-xyz");

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "authorization_code"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "code" && v == "auth-code-xyz"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "redirect_uri" && v == "http://localhost:8080/callback"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
        assert_eq!(params.len(), 5);
    }

    #[test]
    fn test_token_request_client_credentials() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_client_credentials();

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "client_credentials"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "scope" && v == "tools:read resources:read"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
    }

    #[test]
    fn test_token_request_refresh() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        let params = manager.build_token_request_refresh("refresh-tok-abc");

        assert!(params
            .iter()
            .any(|(k, v)| k == "grant_type" && v == "refresh_token"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "refresh_token" && v == "refresh-tok-abc"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_id" && v == "test-client"));
        assert!(params
            .iter()
            .any(|(k, v)| k == "client_secret" && v == "test-secret"));
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_pkce_challenge_generation() {
        let (challenge1, method1) = McpOAuthTokenManager::generate_pkce_challenge("my-verifier");
        let (challenge2, method2) = McpOAuthTokenManager::generate_pkce_challenge("my-verifier");
        // Deterministic
        assert_eq!(challenge1, challenge2);
        assert_eq!(method1, "plain");
        assert_eq!(method2, "plain");
        // 16 hex chars
        assert_eq!(challenge1.len(), 16);

        // Different verifiers produce different challenges
        let (challenge3, _) = McpOAuthTokenManager::generate_pkce_challenge("other-verifier");
        assert_ne!(challenge1, challenge3);
    }

    #[test]
    fn test_token_manager_set_and_get() {
        let config = make_test_oauth_config();
        let mut manager = McpOAuthTokenManager::new(config);

        // No token initially
        assert!(manager.get_access_token().is_none());

        let token = McpTokenResponse {
            access_token: "access-123".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(3600),
            refresh_token: Some("refresh-456".to_string()),
            scope: Some("tools:read".to_string()),
        };

        manager.set_token(token);
        assert_eq!(manager.get_access_token(), Some("access-123"));
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_token_expiry_no_token() {
        let config = make_test_oauth_config();
        let manager = McpOAuthTokenManager::new(config);
        // No token => considered expired
        assert!(manager.is_token_expired());
        assert!(manager.get_access_token().is_none());
    }

    #[test]
    fn test_token_no_expiry_never_expires() {
        let config = make_test_oauth_config();
        let mut manager = McpOAuthTokenManager::new(config);

        let token = McpTokenResponse {
            access_token: "permanent-token".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: None, // No expiry
            refresh_token: None,
            scope: None,
        };

        manager.set_token(token);
        // Token with no expiry should never be considered expired
        assert!(!manager.is_token_expired());
        assert_eq!(manager.get_access_token(), Some("permanent-token"));
    }

    #[test]
    fn test_token_response_serde() {
        let token = McpTokenResponse {
            access_token: "abc".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(7200),
            refresh_token: Some("def".to_string()),
            scope: Some("tools:read resources:read".to_string()),
        };

        let json = serde_json::to_string(&token).unwrap();
        let back: McpTokenResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.access_token, "abc");
        assert_eq!(back.token_type, "Bearer");
        assert_eq!(back.expires_in, Some(7200));
        assert_eq!(back.refresh_token.as_deref(), Some("def"));
        assert_eq!(back.scope.as_deref(), Some("tools:read resources:read"));
    }
}
