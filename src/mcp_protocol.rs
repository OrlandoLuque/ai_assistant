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

// =============================================================================
// MCP SESSION RESOURCES (v4 - item 8.2)
// =============================================================================

/// Session resource info for MCP session:// URIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResourceInfo {
    pub session_id: String,
    pub name: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
    pub closed_cleanly: bool,
}

/// Summary of a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub summary: String,
    pub message_count: usize,
    pub key_topics: Vec<String>,
    pub decisions: Vec<String>,
    pub open_questions: Vec<String>,
}

/// Session highlights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHighlights {
    pub session_id: String,
    pub highlights: Vec<String>,
    pub conclusions: Vec<String>,
    pub action_items: Vec<String>,
}

/// Session context (entities/relations extracted)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    pub session_id: String,
    pub entities: Vec<String>,
    pub relations: Vec<(String, String, String)>, // (source, relation, target)
    pub key_facts: Vec<String>,
}

/// Session beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBeliefs {
    pub session_id: String,
    pub beliefs: Vec<SessionBelief>,
}

/// A single belief statement extracted from session messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBelief {
    pub statement: String,
    pub belief_type: String,
    pub confidence: f32,
}

/// Session repair result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRepairResult {
    pub session_id: String,
    pub success: bool,
    pub messages_recovered: usize,
    pub messages_lost: usize,
    pub repair_notes: Vec<String>,
}

/// Manager for session MCP resources
pub struct SessionMcpManager {
    sessions: HashMap<String, SessionResourceInfo>,
}

impl SessionMcpManager {
    /// Create a new empty session manager.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Register a session resource.
    pub fn register_session(&mut self, info: SessionResourceInfo) {
        self.sessions.insert(info.session_id.clone(), info);
    }

    /// Unregister a session by ID. Returns the removed info if it existed.
    pub fn unregister(&mut self, session_id: &str) -> Option<SessionResourceInfo> {
        self.sessions.remove(session_id)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Option<&SessionResourceInfo> {
        self.sessions.get(session_id)
    }

    /// List all registered sessions.
    pub fn list_sessions(&self) -> Vec<&SessionResourceInfo> {
        self.sessions.values().collect()
    }

    /// Serialize all sessions to a JSON value.
    pub fn sessions_to_json(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .sessions
            .values()
            .map(|info| {
                serde_json::to_value(info).unwrap_or_else(|_| serde_json::Value::Null)
            })
            .collect();
        serde_json::Value::Array(entries)
    }

    /// Generate a summary for a session from its messages.
    ///
    /// Each message is a tuple of (role, content).
    pub fn generate_summary(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionSummary {
        if messages.is_empty() {
            return SessionSummary {
                session_id: session_id.to_string(),
                summary: String::new(),
                message_count: 0,
                key_topics: Vec::new(),
                decisions: Vec::new(),
                open_questions: Vec::new(),
            };
        }

        // Build word frequency map for topic extraction (words >= 4 chars, lowercased)
        let stop_words: &[&str] = &[
            "this", "that", "with", "from", "have", "been", "were", "they",
            "their", "about", "would", "could", "should", "there", "which",
            "will", "what", "when", "where", "some", "into", "also", "then",
            "than", "them", "these", "those", "each", "other", "more",
        ];
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for (_role, content) in messages {
            for word in content.split_whitespace() {
                let cleaned: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if cleaned.len() >= 4 && !stop_words.contains(&cleaned.as_str()) {
                    *word_freq.entry(cleaned).or_insert(0) += 1;
                }
            }
        }

        // Top topics by frequency
        let mut freq_vec: Vec<(String, usize)> = word_freq.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let key_topics: Vec<String> = freq_vec
            .into_iter()
            .take(5)
            .map(|(word, _)| word)
            .collect();

        // Decisions: messages containing decision-related keywords
        let decision_keywords = ["decided", "agreed", "chosen", "decision", "we will"];
        let mut decisions = Vec::new();
        for (_role, content) in messages {
            let lower = content.to_lowercase();
            if decision_keywords.iter().any(|kw| lower.contains(kw)) {
                decisions.push(content.clone());
            }
        }

        // Open questions: messages containing "?"
        let mut open_questions = Vec::new();
        for (_role, content) in messages {
            if content.contains('?') {
                open_questions.push(content.clone());
            }
        }

        // Build summary text
        let summary = if key_topics.is_empty() {
            format!("Session with {} messages.", messages.len())
        } else {
            format!(
                "Session with {} messages. Key topics: {}.",
                messages.len(),
                key_topics.join(", ")
            )
        };

        SessionSummary {
            session_id: session_id.to_string(),
            summary,
            message_count: messages.len(),
            key_topics,
            decisions,
            open_questions,
        }
    }

    /// Extract highlights from session messages.
    pub fn extract_highlights(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionHighlights {
        let mut highlights = Vec::new();
        let mut conclusions = Vec::new();
        let mut action_items = Vec::new();

        let highlight_keywords = ["important", "key", "critical", "significant", "essential"];
        let conclusion_keywords = ["in conclusion", "therefore", "finally", "to summarize", "in summary"];
        let action_keywords = ["todo", "need to", "should", "action item", "must", "next step"];

        for (_role, content) in messages {
            let lower = content.to_lowercase();

            if highlight_keywords.iter().any(|kw| lower.contains(kw)) {
                highlights.push(content.clone());
            }
            if conclusion_keywords.iter().any(|kw| lower.contains(kw)) {
                conclusions.push(content.clone());
            }
            if action_keywords.iter().any(|kw| lower.contains(kw)) {
                action_items.push(content.clone());
            }
        }

        SessionHighlights {
            session_id: session_id.to_string(),
            highlights,
            conclusions,
            action_items,
        }
    }

    /// Extract context (entities, relations, key facts) from session messages.
    pub fn extract_context(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionContext {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let mut key_facts = Vec::new();

        for (_role, content) in messages {
            // Entity extraction: capitalized words (simple heuristic)
            let words: Vec<&str> = content.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let cleaned: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                if cleaned.len() >= 2 {
                    if let Some(first_char) = cleaned.chars().next() {
                        // Skip if it's the first word in the message (sentence-initial capitalization)
                        if first_char.is_uppercase() && i > 0 && !entities.contains(&cleaned) {
                            entities.push(cleaned);
                        }
                    }
                }
            }

            // Simple subject-verb-object relation extraction
            // Look for patterns like "X is Y", "X uses Y", "X requires Y"
            let relation_verbs = ["is", "uses", "requires", "contains", "provides", "supports"];
            for verb in &relation_verbs {
                let pattern = format!(" {} ", verb);
                if let Some(pos) = content.find(&pattern) {
                    let before = content[..pos].split_whitespace().next_back();
                    let after_start = pos + pattern.len();
                    let after = content.get(after_start..).and_then(|s| s.split_whitespace().next());
                    if let (Some(subj), Some(obj)) = (before, after) {
                        let subj_clean: String = subj.chars().filter(|c| c.is_alphanumeric()).collect();
                        let obj_clean: String = obj.chars().filter(|c| c.is_alphanumeric()).collect();
                        if !subj_clean.is_empty() && !obj_clean.is_empty() {
                            relations.push((
                                subj_clean,
                                verb.to_string(),
                                obj_clean,
                            ));
                        }
                    }
                }
            }

            // Key facts: declarative statements (sentences that don't end with ?)
            let lower = content.to_lowercase();
            if !content.contains('?')
                && content.len() > 20
                && (lower.contains(" is ") || lower.contains(" are ") || lower.contains(" was "))
            {
                key_facts.push(content.clone());
            }
        }

        SessionContext {
            session_id: session_id.to_string(),
            entities,
            relations,
            key_facts,
        }
    }

    /// Extract beliefs from session messages.
    pub fn extract_beliefs(
        session_id: &str,
        messages: &[(String, String)],
    ) -> SessionBeliefs {
        let belief_patterns: &[(&str, &str, f32)] = &[
            ("i think", "opinion", 0.6),
            ("i believe", "conviction", 0.8),
            ("we should", "recommendation", 0.7),
            ("it seems", "observation", 0.5),
            ("i'm sure", "conviction", 0.9),
            ("probably", "speculation", 0.4),
            ("definitely", "conviction", 0.9),
            ("maybe", "speculation", 0.3),
        ];

        let mut beliefs = Vec::new();

        for (_role, content) in messages {
            let lower = content.to_lowercase();
            for &(pattern, belief_type, confidence) in belief_patterns {
                if lower.contains(pattern) {
                    beliefs.push(SessionBelief {
                        statement: content.clone(),
                        belief_type: belief_type.to_string(),
                        confidence,
                    });
                    break; // One belief per message
                }
            }
        }

        SessionBeliefs {
            session_id: session_id.to_string(),
            beliefs,
        }
    }

    /// Attempt to repair a session from raw data.
    ///
    /// Tries JSON array parsing first, then falls back to line-by-line recovery.
    pub fn repair_session(session_id: &str, raw_data: &str) -> SessionRepairResult {
        // Attempt 1: parse as complete JSON array
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(raw_data) {
            return SessionRepairResult {
                session_id: session_id.to_string(),
                success: true,
                messages_recovered: arr.len(),
                messages_lost: 0,
                repair_notes: vec!["Parsed successfully as JSON array.".to_string()],
            };
        }

        // Attempt 2: line-by-line recovery of partial JSON objects
        let mut recovered = 0usize;
        let mut lost = 0usize;
        let mut notes = Vec::new();

        for (line_idx, line) in raw_data.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Try to parse each line as a JSON object
            if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                recovered += 1;
            } else {
                // Try to fix common JSON issues: trailing comma, missing closing brace
                let mut candidate = trimmed.to_string();
                // Remove trailing comma
                if candidate.ends_with(',') {
                    candidate.pop();
                }
                // Try adding missing closing brace
                if candidate.starts_with('{') && !candidate.ends_with('}') {
                    candidate.push('}');
                }
                if serde_json::from_str::<serde_json::Value>(&candidate).is_ok() {
                    recovered += 1;
                    notes.push(format!("Line {}: repaired (trailing comma or missing brace).", line_idx + 1));
                } else {
                    lost += 1;
                    notes.push(format!("Line {}: unrecoverable.", line_idx + 1));
                }
            }
        }

        let success = recovered > 0;
        if !success {
            notes.push("No messages could be recovered.".to_string());
        }

        SessionRepairResult {
            session_id: session_id.to_string(),
            success,
            messages_recovered: recovered,
            messages_lost: lost,
            repair_notes: notes,
        }
    }
}

impl Default for SessionMcpManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// MCP v2 PROTOCOL — Phase 2 (v5 roadmap: items 2.1, 2.2, 2.3)
// =============================================================================

// ---------------------------------------------------------------------------
// 2.1 — Streamable HTTP Transport
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
}

impl StreamableHttpTransport {
    /// Create a new Streamable HTTP transport pointing at `base_url`.
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            session_id: None,
            mode: TransportMode::StreamableHTTP,
        }
    }

    /// Send a JSON-RPC request and return the response.
    ///
    /// In a real implementation this would POST to `base_url` and parse the
    /// response; here we build the serialized body and return an error because
    /// no real HTTP round-trip is performed (that requires an async runtime +
    /// network). Integration tests should use a live server.
    pub fn send_request(&mut self, request: &McpRequest) -> Result<McpResponse, String> {
        let body = serde_json::to_string(request)
            .map_err(|e| format!("Serialization error: {}", e))?;

        // In production this would be:
        //   let resp = http_post(&self.base_url, &body)?;
        //   let content_type = resp.header("content-type");
        //   self.mode = Self::detect_transport(content_type);
        //   if mode == SSE { stream... } else { parse JSON }
        //
        // For unit-testability we accept the serialized body and return a
        // synthetic error so callers know no network call was made.
        let _ = body;
        Err("StreamableHttpTransport: no real HTTP backend configured (use integration tests)".to_string())
    }

    /// Detect the transport mode from a response Content-Type header value.
    pub fn detect_transport(response_content_type: &str) -> TransportMode {
        let ct = response_content_type.to_lowercase();
        if ct.contains("text/event-stream") {
            TransportMode::SSE
        } else if ct.contains("application/json") {
            TransportMode::StreamableHTTP
        } else {
            // Unknown content type — default to StreamableHTTP (JSON mode)
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
    sessions: HashMap<String, McpSession>,
    next_id: u64,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 1,
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
        let id = format!("mcp-session-{}", self.next_id);
        self.next_id += 1;
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

// ---------------------------------------------------------------------------
// 2.2 — OAuth 2.1 + PKCE + Dynamic Client Registration
// ---------------------------------------------------------------------------

/// SHA-256 hash (FIPS 180-4). Returns 32-byte digest.
///
/// Pure-Rust implementation so we do not require the `sha2` crate (which is
/// only available behind the `distributed-network` feature flag).
fn sha256_hash(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of the fractional parts of the
    // square roots of the first 8 primes 2..19).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants (first 32 bits of the fractional parts of the cube
    // roots of the first 64 primes 2..311).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: pad message to a multiple of 512 bits (64 bytes).
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block.
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i * 4..(i + 1) * 4].copy_from_slice(&h[i].to_be_bytes());
    }
    result
}

/// Base64url-encode (RFC 4648 section 5) without padding.
fn base64url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize] as char);
        }
    }
    out
}

/// MCP v2 OAuth configuration (simplified for the v2 flow with PKCE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpV2OAuthConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

/// OAuth 2.1 token with expiry tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    pub access_token: String,
    pub token_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// PKCE challenge (RFC 7636) with S256 method.
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
    pub method: String,
}

impl PkceChallenge {
    /// Generate a PKCE challenge pair.
    ///
    /// The verifier is a 43-character random base64url string (matching the
    /// RFC 7636 minimum of 43 characters). The challenge is the base64url-
    /// encoded SHA-256 hash of the verifier.
    pub fn generate() -> Self {
        // Generate a pseudo-random verifier using system time + counter.
        // In production you would use a CSPRNG; this is sufficient for the
        // library's test/demo purposes and avoids adding a `rand` dependency.
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        // Simple LCG to generate 32 bytes of pseudo-random data
        let mut state = seed as u64;
        let mut raw = Vec::with_capacity(32);
        for _ in 0..32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            raw.push((state >> 33) as u8);
        }
        let verifier = base64url_encode(&raw);

        // S256: challenge = BASE64URL(SHA256(ASCII(verifier)))
        let hash = sha256_hash(verifier.as_bytes());
        let challenge = base64url_encode(&hash);

        Self {
            verifier,
            challenge,
            method: "S256".to_string(),
        }
    }

    /// Create a PKCE challenge from a known verifier (useful for testing).
    pub fn from_verifier(verifier: &str) -> Self {
        let hash = sha256_hash(verifier.as_bytes());
        let challenge = base64url_encode(&hash);
        Self {
            verifier: verifier.to_string(),
            challenge,
            method: "S256".to_string(),
        }
    }
}

/// OAuth 2.1 token manager for MCP v2.
pub struct OAuthTokenManager {
    config: McpV2OAuthConfig,
    current_token: Option<OAuthToken>,
}

impl OAuthTokenManager {
    pub fn new(config: McpV2OAuthConfig) -> Self {
        Self {
            config,
            current_token: None,
        }
    }

    /// Build the authorization URL with PKCE parameters.
    ///
    /// Returns `(url, pkce_challenge)`.
    pub fn get_authorization_url(&self) -> (String, PkceChallenge) {
        let pkce = PkceChallenge::generate();
        let scope_str = self.config.scopes.join(" ");

        let mut url = format!(
            "{}?response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method={}",
            self.config.authorization_endpoint,
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(&scope_str),
            urlencoding::encode(&pkce.challenge),
            urlencoding::encode(&pkce.method),
        );

        if let Some(ref client_id) = self.config.client_id {
            url.push_str(&format!("&client_id={}", urlencoding::encode(client_id)));
        }

        (url, pkce)
    }

    /// Exchange an authorization code for a token (mock — returns a synthetic token).
    ///
    /// In production this would POST to the token endpoint; here we simulate
    /// the exchange for testability.
    pub fn exchange_code(&mut self, code: &str, pkce: &PkceChallenge) -> Result<OAuthToken, String> {
        if code.is_empty() {
            return Err("Authorization code is empty".to_string());
        }
        if pkce.verifier.is_empty() {
            return Err("PKCE verifier is empty".to_string());
        }

        // In production: POST to token_endpoint with grant_type=authorization_code,
        // code, redirect_uri, code_verifier.
        let token = OAuthToken {
            access_token: format!("access-{}", code),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some(format!("refresh-{}", code)),
            scope: Some(self.config.scopes.join(" ")),
        };
        self.current_token = Some(token.clone());
        Ok(token)
    }

    /// Refresh the current token (mock — returns a new synthetic token).
    pub fn refresh_token(&mut self) -> Result<OAuthToken, String> {
        let refresh = self
            .current_token
            .as_ref()
            .and_then(|t| t.refresh_token.clone())
            .ok_or_else(|| "No refresh token available".to_string())?;

        // In production: POST to token_endpoint with grant_type=refresh_token.
        let token = OAuthToken {
            access_token: format!("refreshed-{}", refresh),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some(refresh),
            scope: self.current_token.as_ref().and_then(|t| t.scope.clone()),
        };
        self.current_token = Some(token.clone());
        Ok(token)
    }

    /// Check if the current token is expired.
    pub fn is_token_expired(&self) -> bool {
        match &self.current_token {
            None => true,
            Some(token) => match token.expires_at {
                Some(expires) => chrono::Utc::now() >= expires,
                None => false, // No expiry = never expires
            },
        }
    }

    /// Get the current token if valid, or try to refresh if expired.
    pub fn get_valid_token(&mut self) -> Result<&OAuthToken, String> {
        if self.current_token.is_none() {
            return Err("No token available — authorization required".to_string());
        }

        if self.is_token_expired() {
            // Try to refresh
            self.refresh_token()?;
        }

        self.current_token
            .as_ref()
            .ok_or_else(|| "Token unavailable after refresh".to_string())
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &McpV2OAuthConfig {
        &self.config
    }

    /// Get the current token without refresh.
    pub fn current_token(&self) -> Option<&OAuthToken> {
        self.current_token.as_ref()
    }

    /// Manually set a token (e.g. after external exchange).
    pub fn set_token(&mut self, token: OAuthToken) {
        self.current_token = Some(token);
    }
}

/// OAuth 2.1 Authorization Server Metadata (RFC 8414).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registration_endpoint: Option<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

impl AuthorizationServerMetadata {
    /// Discover metadata from `/.well-known/oauth-authorization-server`.
    ///
    /// In production this would fetch the URL; here we return a mock for the
    /// given base_url so the API can be exercised in unit tests.
    pub fn discover(base_url: &str) -> Result<Self, String> {
        // In production: GET {base_url}/.well-known/oauth-authorization-server
        // and parse the JSON response.
        Ok(Self {
            issuer: base_url.to_string(),
            authorization_endpoint: format!("{}/authorize", base_url),
            token_endpoint: format!("{}/token", base_url),
            registration_endpoint: Some(format!("{}/register", base_url)),
            scopes_supported: vec![
                "mcp:tools".to_string(),
                "mcp:resources".to_string(),
                "mcp:prompts".to_string(),
            ],
        })
    }
}

/// Dynamic Client Registration (RFC 7591).
pub struct DynamicClientRegistration;

impl DynamicClientRegistration {
    /// Register a client dynamically at the given registration endpoint.
    ///
    /// Returns `(client_id, Option<client_secret>)`.
    /// In production this would POST JSON to the endpoint; here we return a
    /// mock response.
    pub fn register(
        registration_endpoint: &str,
        client_name: &str,
        redirect_uris: &[String],
    ) -> Result<(String, Option<String>), String> {
        if registration_endpoint.is_empty() {
            return Err("Registration endpoint is empty".to_string());
        }
        if client_name.is_empty() {
            return Err("Client name is empty".to_string());
        }
        if redirect_uris.is_empty() {
            return Err("At least one redirect URI is required".to_string());
        }

        // In production: POST to registration_endpoint with
        // { client_name, redirect_uris, grant_types, ... }
        let client_id = format!("dyn-client-{}", client_name);
        let client_secret = Some(format!("dyn-secret-{}", client_name));
        Ok((client_id, client_secret))
    }
}

// ---------------------------------------------------------------------------
// 2.3 — Tool Annotations
// ---------------------------------------------------------------------------

/// Tool annotations indicating behavior characteristics (MCP v2).
///
/// These complement the existing `McpToolAnnotation` (hint-based, v4 spec) with
/// a more structured, boolean-based model suited for programmatic policy checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAnnotations {
    /// If true, the tool only reads data and does not modify any state.
    #[serde(default)]
    pub read_only: bool,
    /// If true, the tool may cause destructive / irreversible side effects.
    #[serde(default = "default_true")]
    pub destructive: bool,
    /// If true, calling with the same arguments always produces the same result.
    #[serde(default)]
    pub idempotent: bool,
    /// If true, the tool may interact with external systems not described in its schema.
    #[serde(default = "default_true")]
    pub open_world: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        }
    }
}

impl ToolAnnotations {
    /// A tool is considered "safe" if it is read-only and not destructive.
    pub fn is_safe(&self) -> bool {
        self.read_only && !self.destructive
    }

    /// A tool "needs confirmation" if it is destructive or interacts with the open world.
    pub fn needs_confirmation(&self) -> bool {
        self.destructive || self.open_world
    }
}

/// Wrapper that pairs an `McpTool` with `ToolAnnotations`.
///
/// This avoids modifying the existing `McpTool` struct while still allowing
/// annotation data to travel alongside tool definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedTool {
    pub tool: McpTool,
    pub annotations: ToolAnnotations,
}

impl AnnotatedTool {
    /// Create an `AnnotatedTool` with default annotations.
    pub fn from_tool(tool: McpTool) -> Self {
        Self {
            tool,
            annotations: ToolAnnotations::default(),
        }
    }

    /// Create an `AnnotatedTool` with explicit annotations.
    pub fn with_annotations(tool: McpTool, annotations: ToolAnnotations) -> Self {
        Self { tool, annotations }
    }
}

/// Registry that maps tool names to their `ToolAnnotations`.
pub struct ToolAnnotationRegistry {
    annotations: HashMap<String, ToolAnnotations>,
}

impl ToolAnnotationRegistry {
    pub fn new() -> Self {
        Self {
            annotations: HashMap::new(),
        }
    }

    /// Register annotations for a tool by name.
    pub fn register(&mut self, tool_name: &str, annotations: ToolAnnotations) {
        self.annotations.insert(tool_name.to_string(), annotations);
    }

    /// Get the annotations for a tool by name.
    pub fn get(&self, tool_name: &str) -> Option<&ToolAnnotations> {
        self.annotations.get(tool_name)
    }

    /// Check if a tool needs human approval based on its annotations.
    ///
    /// Returns `true` if annotations exist and `needs_confirmation()` is true,
    /// or if no annotations are registered (conservative default).
    pub fn needs_approval(&self, tool_name: &str) -> bool {
        match self.annotations.get(tool_name) {
            Some(ann) => ann.needs_confirmation(),
            None => true, // Unknown tool — require approval by default
        }
    }
}

impl Default for ToolAnnotationRegistry {
    fn default() -> Self {
        Self::new()
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

    // =========================================================================
    // Session MCP resource tests (v4 - item 8.2)
    // =========================================================================

    fn make_test_session_info(id: &str) -> SessionResourceInfo {
        SessionResourceInfo {
            session_id: id.to_string(),
            name: Some(format!("Session {}", id)),
            created_at: 1000,
            updated_at: 2000,
            message_count: 5,
            closed_cleanly: true,
        }
    }

    #[test]
    fn test_session_manager_new() {
        let mgr = SessionMcpManager::new();
        assert!(mgr.list_sessions().is_empty());
    }

    #[test]
    fn test_register_session() {
        let mut mgr = SessionMcpManager::new();
        let info = make_test_session_info("s1");
        mgr.register_session(info);
        assert_eq!(mgr.list_sessions().len(), 1);
    }

    #[test]
    fn test_unregister_session() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("s1"));
        mgr.register_session(make_test_session_info("s2"));
        assert_eq!(mgr.list_sessions().len(), 2);

        let removed = mgr.unregister("s1");
        assert!(removed.is_some());
        assert_eq!(removed.as_ref().map(|r| r.session_id.as_str()), Some("s1"));
        assert_eq!(mgr.list_sessions().len(), 1);

        // Removing non-existent returns None
        assert!(mgr.unregister("s999").is_none());
    }

    #[test]
    fn test_get_session() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("s1"));

        let s = mgr.get_session("s1");
        assert!(s.is_some());
        assert_eq!(s.map(|s| s.message_count), Some(5));

        assert!(mgr.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_list_sessions() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("a"));
        mgr.register_session(make_test_session_info("b"));
        mgr.register_session(make_test_session_info("c"));

        let list = mgr.list_sessions();
        assert_eq!(list.len(), 3);

        let ids: Vec<&str> = list.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_sessions_to_json() {
        let mut mgr = SessionMcpManager::new();
        mgr.register_session(make_test_session_info("j1"));

        let json = mgr.sessions_to_json();
        assert!(json.is_array());
        let arr = json.as_array().expect("should be array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["session_id"], "j1");
        assert_eq!(arr[0]["message_count"], 5);
    }

    #[test]
    fn test_generate_summary_empty() {
        let summary = SessionMcpManager::generate_summary("s-empty", &[]);
        assert_eq!(summary.session_id, "s-empty");
        assert_eq!(summary.message_count, 0);
        assert!(summary.key_topics.is_empty());
        assert!(summary.decisions.is_empty());
        assert!(summary.open_questions.is_empty());
        assert!(summary.summary.is_empty());
    }

    #[test]
    fn test_generate_summary_with_content() {
        let messages = vec![
            ("user".to_string(), "The architecture uses microservices for deployment".to_string()),
            ("assistant".to_string(), "Microservices architecture provides scalability benefits".to_string()),
            ("user".to_string(), "We need more microservices testing coverage".to_string()),
        ];
        let summary = SessionMcpManager::generate_summary("s-content", &messages);
        assert_eq!(summary.message_count, 3);
        assert!(!summary.key_topics.is_empty());
        // "microservices" should be the top topic (appears 3 times)
        assert!(summary.key_topics.contains(&"microservices".to_string()));
        assert!(summary.summary.contains("3 messages"));
    }

    #[test]
    fn test_generate_summary_decisions_and_questions() {
        let messages = vec![
            ("user".to_string(), "We decided to use Rust for the backend".to_string()),
            ("assistant".to_string(), "Agreed, Rust is a good choice".to_string()),
            ("user".to_string(), "What about the frontend framework?".to_string()),
            ("user".to_string(), "Have we chosen a database yet?".to_string()),
        ];
        let summary = SessionMcpManager::generate_summary("s-dq", &messages);
        assert_eq!(summary.decisions.len(), 3); // "decided", "agreed", "chosen"
        assert_eq!(summary.open_questions.len(), 2); // two questions with "?"
    }

    #[test]
    fn test_extract_highlights_conclusions() {
        let messages = vec![
            ("user".to_string(), "This is an important finding about performance".to_string()),
            ("assistant".to_string(), "A critical issue was identified in the pipeline".to_string()),
            ("assistant".to_string(), "In conclusion, we should refactor the module".to_string()),
            ("user".to_string(), "Therefore, the next step is clear".to_string()),
        ];
        let hl = SessionMcpManager::extract_highlights("s-hl", &messages);
        assert_eq!(hl.highlights.len(), 2); // "important" and "critical"
        assert_eq!(hl.conclusions.len(), 2); // "in conclusion" and "therefore"
    }

    #[test]
    fn test_extract_highlights_action_items() {
        let messages = vec![
            ("user".to_string(), "We need to update the dependencies".to_string()),
            ("assistant".to_string(), "TODO: add error handling for edge cases".to_string()),
            ("user".to_string(), "The weather is nice today".to_string()),
            ("assistant".to_string(), "Action item: review the PR before merging".to_string()),
        ];
        let hl = SessionMcpManager::extract_highlights("s-ai", &messages);
        assert_eq!(hl.action_items.len(), 3); // "need to", "todo", "action item"
        assert!(hl.highlights.is_empty());
    }

    #[test]
    fn test_extract_context_entities() {
        let messages = vec![
            ("user".to_string(), "The system uses Rust and PostgreSQL for storage".to_string()),
            ("assistant".to_string(), "Indeed, Rust provides great performance with PostgreSQL".to_string()),
        ];
        let ctx = SessionMcpManager::extract_context("s-ctx", &messages);
        // Entities should include capitalized words (not sentence-initial)
        assert!(ctx.entities.contains(&"Rust".to_string()));
        assert!(ctx.entities.contains(&"PostgreSQL".to_string()));
    }

    #[test]
    fn test_extract_beliefs() {
        let messages = vec![
            ("user".to_string(), "I think Rust is the best language for this project".to_string()),
            ("assistant".to_string(), "I believe the architecture is sound and well-designed".to_string()),
            ("user".to_string(), "We should add more comprehensive tests to the suite".to_string()),
            ("user".to_string(), "The sky is blue".to_string()), // no belief pattern
        ];
        let beliefs = SessionMcpManager::extract_beliefs("s-bel", &messages);
        assert_eq!(beliefs.beliefs.len(), 3);
        assert_eq!(beliefs.beliefs[0].belief_type, "opinion"); // "i think"
        assert!((beliefs.beliefs[0].confidence - 0.6).abs() < f32::EPSILON);
        assert_eq!(beliefs.beliefs[1].belief_type, "conviction"); // "i believe"
        assert_eq!(beliefs.beliefs[2].belief_type, "recommendation"); // "we should"
    }

    #[test]
    fn test_repair_session_valid_json() {
        let raw = r#"[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]"#;
        let result = SessionMcpManager::repair_session("s-ok", raw);
        assert!(result.success);
        assert_eq!(result.messages_recovered, 2);
        assert_eq!(result.messages_lost, 0);
        assert_eq!(result.repair_notes.len(), 1);
        assert!(result.repair_notes[0].contains("JSON array"));
    }

    #[test]
    fn test_repair_session_partial() {
        let raw = r#"{"role":"user","content":"hello"}
{"role":"assistant","content":"hi"}
{"role":"user","content":"broken"
this is garbage"#;
        let result = SessionMcpManager::repair_session("s-partial", raw);
        assert!(result.success);
        assert_eq!(result.messages_recovered, 3); // 2 valid + 1 repaired (missing brace)
        assert_eq!(result.messages_lost, 1); // "this is garbage"
    }

    #[test]
    fn test_repair_session_corrupted() {
        let raw = "not json at all\njust plain text\nnothing useful";
        let result = SessionMcpManager::repair_session("s-bad", raw);
        assert!(!result.success);
        assert_eq!(result.messages_recovered, 0);
        assert_eq!(result.messages_lost, 3);
        assert!(result.repair_notes.iter().any(|n| n.contains("No messages could be recovered")));
    }

    #[test]
    fn test_session_beliefs_serde() {
        let beliefs = SessionBeliefs {
            session_id: "s-serde".to_string(),
            beliefs: vec![
                SessionBelief {
                    statement: "I think this works".to_string(),
                    belief_type: "opinion".to_string(),
                    confidence: 0.6,
                },
            ],
        };
        let json = serde_json::to_string(&beliefs).unwrap();
        let back: SessionBeliefs = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-serde");
        assert_eq!(back.beliefs.len(), 1);
        assert_eq!(back.beliefs[0].statement, "I think this works");
        assert!((back.beliefs[0].confidence - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_session_context_serde() {
        let ctx = SessionContext {
            session_id: "s-ctx-ser".to_string(),
            entities: vec!["Rust".to_string(), "PostgreSQL".to_string()],
            relations: vec![
                ("Rust".to_string(), "is".to_string(), "fast".to_string()),
            ],
            key_facts: vec!["Rust is a systems programming language with good safety".to_string()],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let back: SessionContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-ctx-ser");
        assert_eq!(back.entities.len(), 2);
        assert_eq!(back.relations.len(), 1);
        assert_eq!(back.relations[0].0, "Rust");
        assert_eq!(back.relations[0].1, "is");
        assert_eq!(back.relations[0].2, "fast");
    }

    #[test]
    fn test_session_repair_result_fields() {
        let result = SessionRepairResult {
            session_id: "s-fields".to_string(),
            success: true,
            messages_recovered: 10,
            messages_lost: 2,
            repair_notes: vec!["note1".to_string(), "note2".to_string()],
        };
        assert_eq!(result.session_id, "s-fields");
        assert!(result.success);
        assert_eq!(result.messages_recovered, 10);
        assert_eq!(result.messages_lost, 2);
        assert_eq!(result.repair_notes.len(), 2);

        // Also test serde round-trip
        let json = serde_json::to_string(&result).unwrap();
        let back: SessionRepairResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages_recovered, 10);
        assert_eq!(back.messages_lost, 2);
    }

    // =========================================================================
    // MCP v2 Phase 2 tests (v5 roadmap: items 2.1, 2.2, 2.3)
    // =========================================================================

    // --- 2.1 Streamable HTTP Transport ---

    #[test]
    fn test_transport_mode_detect_json() {
        let mode = StreamableHttpTransport::detect_transport("application/json");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_json_charset() {
        let mode = StreamableHttpTransport::detect_transport("application/json; charset=utf-8");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_sse() {
        let mode = StreamableHttpTransport::detect_transport("text/event-stream");
        assert_eq!(mode, TransportMode::SSE);
    }

    #[test]
    fn test_transport_mode_detect_unknown() {
        let mode = StreamableHttpTransport::detect_transport("text/html");
        assert_eq!(mode, TransportMode::StreamableHTTP);
    }

    #[test]
    fn test_transport_mode_detect_case_insensitive() {
        let mode = StreamableHttpTransport::detect_transport("TEXT/EVENT-STREAM");
        assert_eq!(mode, TransportMode::SSE);
    }

    #[test]
    fn test_transport_mode_serde() {
        let json = serde_json::to_value(TransportMode::StreamableHTTP).unwrap();
        assert_eq!(json, "streamable_http");
        let json2 = serde_json::to_value(TransportMode::SSE).unwrap();
        assert_eq!(json2, "sse");
        let json3 = serde_json::to_value(TransportMode::StdIO).unwrap();
        assert_eq!(json3, "std_io");
    }

    #[test]
    fn test_streamable_http_transport_new() {
        let transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        assert_eq!(transport.base_url(), "http://localhost:8080/mcp");
        assert_eq!(transport.mode(), TransportMode::StreamableHTTP);
        assert!(transport.get_session_id().is_none());
    }

    #[test]
    fn test_streamable_http_transport_session_id() {
        let mut transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        assert!(transport.get_session_id().is_none());

        transport.set_session_id("sess-abc-123".to_string());
        assert_eq!(transport.get_session_id(), Some("sess-abc-123"));
    }

    #[test]
    fn test_streamable_http_transport_send_request_returns_error() {
        let mut transport = StreamableHttpTransport::new("http://localhost:8080/mcp");
        let request = McpRequest::new("tools/list").with_id(1u64);
        let result = transport.send_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no real HTTP backend"));
    }

    // --- Session Store ---

    #[test]
    fn test_in_memory_session_store_create() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        assert!(session.session_id.starts_with("mcp-session-"));
        assert!(!session.session_id.is_empty());
    }

    #[test]
    fn test_in_memory_session_store_create_multiple() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let s2 = store.create_session();
        let s3 = store.create_session();
        assert_ne!(s1.session_id, s2.session_id);
        assert_ne!(s2.session_id, s3.session_id);
        assert_eq!(store.list_sessions().len(), 3);
    }

    #[test]
    fn test_in_memory_session_store_get() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        let id = session.session_id.clone();

        let retrieved = store.get_session(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().session_id, id);

        assert!(store.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_in_memory_session_store_touch() {
        let mut store = InMemorySessionStore::new();
        let session = store.create_session();
        let id = session.session_id.clone();
        let original_last_active = session.last_active;

        // Small sleep to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        store.touch_session(&id);
        let touched = store.get_session(&id).unwrap();
        assert!(touched.last_active >= original_last_active);
    }

    #[test]
    fn test_in_memory_session_store_delete() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let s2 = store.create_session();
        let id1 = s1.session_id.clone();

        assert_eq!(store.list_sessions().len(), 2);
        store.delete_session(&id1);
        assert_eq!(store.list_sessions().len(), 1);
        assert!(store.get_session(&id1).is_none());
        assert!(store.get_session(&s2.session_id).is_some());
    }

    #[test]
    fn test_in_memory_session_store_list() {
        let mut store = InMemorySessionStore::new();
        assert!(store.list_sessions().is_empty());

        store.create_session();
        store.create_session();
        assert_eq!(store.list_sessions().len(), 2);
    }

    #[test]
    fn test_in_memory_session_store_cleanup_expired() {
        let mut store = InMemorySessionStore::new();
        let s1 = store.create_session();
        let id1 = s1.session_id.clone();

        // Manually backdate the session to make it "expired"
        if let Some(session) = store.sessions.get_mut(&id1) {
            session.last_active = chrono::Utc::now() - chrono::Duration::seconds(120);
        }

        // Create a fresh session
        let _s2 = store.create_session();
        assert_eq!(store.list_sessions().len(), 2);

        // Cleanup sessions older than 60 seconds
        store.cleanup_expired(60);
        assert_eq!(store.list_sessions().len(), 1);
        assert!(store.get_session(&id1).is_none());
    }

    #[test]
    fn test_mcp_session_serde() {
        let session = McpSession {
            session_id: "s-test".to_string(),
            created_at: chrono::Utc::now(),
            last_active: chrono::Utc::now(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".to_string(), "value".to_string());
                m
            },
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: McpSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, "s-test");
        assert_eq!(back.metadata.get("key").map(|v| v.as_str()), Some("value"));
    }

    // --- 2.2 OAuth 2.1 + PKCE ---

    #[test]
    fn test_sha256_known_vectors() {
        // SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
        let hash = sha256_hash(b"abc");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256_hash(b"");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_longer_input() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let hash = sha256_hash(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_base64url_encode_basic() {
        // Known value: base64url of [0, 1, 2, 3] should produce "AAECAT"
        // Standard base64: AAECAw== → base64url without padding: AAECAw
        let encoded = base64url_encode(&[0, 1, 2, 3]);
        assert_eq!(encoded, "AAECAw");
    }

    #[test]
    fn test_base64url_no_plus_or_slash() {
        // base64url must not contain + or / (unlike standard base64)
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64url_encode(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_pkce_challenge_generate() {
        let pkce = PkceChallenge::generate();
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
        assert_eq!(pkce.method, "S256");
        // Verifier should be at least 43 chars per RFC 7636
        assert!(pkce.verifier.len() >= 43);
    }

    #[test]
    fn test_pkce_from_verifier_deterministic() {
        let pkce1 = PkceChallenge::from_verifier("test-verifier-12345678901234567890123456789");
        let pkce2 = PkceChallenge::from_verifier("test-verifier-12345678901234567890123456789");
        assert_eq!(pkce1.challenge, pkce2.challenge);
        assert_eq!(pkce1.verifier, pkce2.verifier);
        assert_eq!(pkce1.method, "S256");
    }

    #[test]
    fn test_pkce_different_verifiers_different_challenges() {
        let pkce1 = PkceChallenge::from_verifier("verifier-aaa");
        let pkce2 = PkceChallenge::from_verifier("verifier-bbb");
        assert_ne!(pkce1.challenge, pkce2.challenge);
    }

    #[test]
    fn test_pkce_challenge_is_base64url() {
        let pkce = PkceChallenge::from_verifier("my-test-verifier");
        // base64url characters only: A-Z a-z 0-9 - _
        assert!(pkce
            .challenge
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }

    fn make_test_v2_oauth_config() -> McpV2OAuthConfig {
        McpV2OAuthConfig {
            authorization_endpoint: "https://auth.example.com/authorize".to_string(),
            token_endpoint: "https://auth.example.com/token".to_string(),
            client_id: Some("test-client-v2".to_string()),
            client_secret: Some("test-secret-v2".to_string()),
            scopes: vec!["mcp:tools".to_string(), "mcp:resources".to_string()],
            redirect_uri: "http://localhost:9090/callback".to_string(),
        }
    }

    #[test]
    fn test_oauth_token_manager_new() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        assert!(manager.current_token().is_none());
        assert!(manager.is_token_expired());
    }

    #[test]
    fn test_get_authorization_url() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        let (url, pkce) = manager.get_authorization_url();

        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("client_id=test-client-v2"));
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
    }

    #[test]
    fn test_exchange_code() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");

        let token = manager.exchange_code("auth-code-xyz", &pkce).unwrap();
        assert_eq!(token.access_token, "access-auth-code-xyz");
        assert_eq!(token.token_type, "Bearer");
        assert!(token.expires_at.is_some());
        assert!(token.refresh_token.is_some());
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_exchange_code_empty_code() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");
        let result = manager.exchange_code("", &pkce);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_exchange_code_empty_verifier() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge {
            verifier: String::new(),
            challenge: "abc".to_string(),
            method: "S256".to_string(),
        };
        let result = manager.exchange_code("code", &pkce);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("verifier"));
    }

    #[test]
    fn test_refresh_token_flow() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let pkce = PkceChallenge::from_verifier("test-verifier");

        // First get a token
        manager.exchange_code("code-1", &pkce).unwrap();

        // Now refresh
        let refreshed = manager.refresh_token().unwrap();
        assert!(refreshed.access_token.starts_with("refreshed-"));
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_refresh_token_no_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let result = manager.refresh_token();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No refresh token"));
    }

    #[test]
    fn test_is_token_expired_no_token() {
        let config = make_test_v2_oauth_config();
        let manager = OAuthTokenManager::new(config);
        assert!(manager.is_token_expired());
    }

    #[test]
    fn test_is_token_expired_valid_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "valid".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: None,
            scope: None,
        });
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_is_token_expired_no_expiry() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "permanent".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: None,
            refresh_token: None,
            scope: None,
        });
        assert!(!manager.is_token_expired());
    }

    #[test]
    fn test_get_valid_token_no_token() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        let result = manager.get_valid_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_valid_token_with_valid() {
        let config = make_test_v2_oauth_config();
        let mut manager = OAuthTokenManager::new(config);
        manager.set_token(OAuthToken {
            access_token: "good-token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::hours(1)),
            refresh_token: Some("refresh-1".to_string()),
            scope: None,
        });
        let token = manager.get_valid_token().unwrap();
        assert_eq!(token.access_token, "good-token");
    }

    #[test]
    fn test_oauth_token_serde() {
        let token = OAuthToken {
            access_token: "tok-abc".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Some(chrono::Utc::now()),
            refresh_token: Some("ref-def".to_string()),
            scope: Some("mcp:tools".to_string()),
        };
        let json = serde_json::to_string(&token).unwrap();
        let back: OAuthToken = serde_json::from_str(&json).unwrap();
        assert_eq!(back.access_token, "tok-abc");
        assert_eq!(back.token_type, "Bearer");
        assert!(back.expires_at.is_some());
        assert_eq!(back.refresh_token.as_deref(), Some("ref-def"));
        assert_eq!(back.scope.as_deref(), Some("mcp:tools"));
    }

    // --- Authorization Server Metadata ---

    #[test]
    fn test_authorization_server_metadata_discover() {
        let metadata = AuthorizationServerMetadata::discover("https://auth.example.com").unwrap();
        assert_eq!(metadata.issuer, "https://auth.example.com");
        assert_eq!(
            metadata.authorization_endpoint,
            "https://auth.example.com/authorize"
        );
        assert_eq!(
            metadata.token_endpoint,
            "https://auth.example.com/token"
        );
        assert!(metadata.registration_endpoint.is_some());
        assert!(!metadata.scopes_supported.is_empty());
    }

    #[test]
    fn test_authorization_server_metadata_serde() {
        let metadata = AuthorizationServerMetadata {
            issuer: "https://example.com".to_string(),
            authorization_endpoint: "https://example.com/auth".to_string(),
            token_endpoint: "https://example.com/token".to_string(),
            registration_endpoint: None,
            scopes_supported: vec!["scope1".to_string()],
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let back: AuthorizationServerMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.issuer, "https://example.com");
        assert!(back.registration_endpoint.is_none());
        assert_eq!(back.scopes_supported.len(), 1);
    }

    // --- Dynamic Client Registration ---

    #[test]
    fn test_dynamic_client_registration() {
        let (client_id, client_secret) = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "my-app",
            &["http://localhost:8080/callback".to_string()],
        )
        .unwrap();
        assert!(client_id.contains("my-app"));
        assert!(client_secret.is_some());
    }

    #[test]
    fn test_dynamic_client_registration_empty_endpoint() {
        let result = DynamicClientRegistration::register(
            "",
            "my-app",
            &["http://localhost/cb".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_client_registration_empty_name() {
        let result = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "",
            &["http://localhost/cb".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_client_registration_no_redirects() {
        let result = DynamicClientRegistration::register(
            "https://auth.example.com/register",
            "my-app",
            &[],
        );
        assert!(result.is_err());
    }

    // --- 2.3 Tool Annotations ---

    #[test]
    fn test_tool_annotations_v2_defaults() {
        let ann = ToolAnnotations::default();
        assert!(!ann.read_only);
        assert!(ann.destructive);
        assert!(!ann.idempotent);
        assert!(ann.open_world);
    }

    #[test]
    fn test_tool_annotations_is_safe() {
        let safe = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        assert!(safe.is_safe());

        let not_safe = ToolAnnotations {
            read_only: true,
            destructive: true,
            ..Default::default()
        };
        assert!(!not_safe.is_safe());

        let not_readonly = ToolAnnotations {
            read_only: false,
            destructive: false,
            ..Default::default()
        };
        assert!(!not_readonly.is_safe());
    }

    #[test]
    fn test_tool_annotations_needs_confirmation() {
        // Default: destructive=true, open_world=true => needs confirmation
        let default_ann = ToolAnnotations::default();
        assert!(default_ann.needs_confirmation());

        // Only destructive
        let destructive_only = ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: false,
        };
        assert!(destructive_only.needs_confirmation());

        // Only open_world
        let open_only = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: true,
        };
        assert!(open_only.needs_confirmation());

        // Neither destructive nor open_world
        let no_confirmation = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        assert!(!no_confirmation.needs_confirmation());
    }

    #[test]
    fn test_tool_annotations_serde_v2() {
        let ann = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        let json = serde_json::to_string(&ann).unwrap();
        let back: ToolAnnotations = serde_json::from_str(&json).unwrap();
        assert!(back.read_only);
        assert!(!back.destructive);
        assert!(back.idempotent);
        assert!(!back.open_world);
    }

    #[test]
    fn test_tool_annotations_serde_defaults_on_missing_fields() {
        // When fields are missing, defaults should apply
        let json = r#"{"read_only": true}"#;
        let ann: ToolAnnotations = serde_json::from_str(json).unwrap();
        assert!(ann.read_only);
        assert!(ann.destructive); // default true
        assert!(!ann.idempotent); // default false
        assert!(ann.open_world); // default true
    }

    #[test]
    fn test_annotated_tool_from_tool() {
        let tool = McpTool::new("search", "Search the web");
        let annotated = AnnotatedTool::from_tool(tool);
        assert_eq!(annotated.tool.name, "search");
        // Default annotations
        assert!(!annotated.annotations.read_only);
        assert!(annotated.annotations.destructive);
    }

    #[test]
    fn test_annotated_tool_with_annotations() {
        let tool = McpTool::new("read_file", "Read a file");
        let ann = ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: true,
            open_world: false,
        };
        let annotated = AnnotatedTool::with_annotations(tool, ann);
        assert_eq!(annotated.tool.name, "read_file");
        assert!(annotated.annotations.is_safe());
        assert!(!annotated.annotations.needs_confirmation());
    }

    #[test]
    fn test_annotated_tool_serde() {
        let tool = McpTool::new("deploy", "Deploy to production");
        let ann = ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            open_world: true,
        };
        let annotated = AnnotatedTool::with_annotations(tool, ann);
        let json = serde_json::to_string(&annotated).unwrap();
        let back: AnnotatedTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tool.name, "deploy");
        assert!(back.annotations.destructive);
        assert!(back.annotations.open_world);
    }

    #[test]
    fn test_tool_annotation_registry_register_and_get() {
        let mut registry = ToolAnnotationRegistry::new();
        registry.register(
            "search",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );

        let ann = registry.get("search");
        assert!(ann.is_some());
        let ann = ann.unwrap();
        assert!(ann.read_only);
        assert!(!ann.destructive);

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_tool_annotation_registry_needs_approval() {
        let mut registry = ToolAnnotationRegistry::new();

        // Safe tool — no approval needed
        registry.register(
            "read_file",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );
        assert!(!registry.needs_approval("read_file"));

        // Destructive tool — approval needed
        registry.register(
            "delete_file",
            ToolAnnotations {
                read_only: false,
                destructive: true,
                idempotent: false,
                open_world: false,
            },
        );
        assert!(registry.needs_approval("delete_file"));

        // Unknown tool — approval needed (conservative default)
        assert!(registry.needs_approval("unknown_tool"));
    }

    #[test]
    fn test_tool_annotation_registry_overwrite() {
        let mut registry = ToolAnnotationRegistry::new();
        registry.register("tool1", ToolAnnotations::default());
        assert!(registry.get("tool1").unwrap().destructive);

        // Overwrite with safe annotations
        registry.register(
            "tool1",
            ToolAnnotations {
                read_only: true,
                destructive: false,
                idempotent: true,
                open_world: false,
            },
        );
        assert!(!registry.get("tool1").unwrap().destructive);
    }

    #[test]
    fn test_mcpv2_oauth_config_serde() {
        let config = make_test_v2_oauth_config();
        let json = serde_json::to_string(&config).unwrap();
        let back: McpV2OAuthConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.client_id.as_deref(), Some("test-client-v2"));
        assert_eq!(back.scopes.len(), 2);
        assert_eq!(back.redirect_uri, "http://localhost:9090/callback");
    }
}
