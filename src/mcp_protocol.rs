//! MCP (Model Context Protocol) support
//!
//! Implementation of Anthropic's Model Context Protocol for standardized
//! tool use and context sharing between AI models and external systems.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// MCP Protocol version
pub const MCP_VERSION: &str = "2024-11-05";

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
        Self { code: -32700, message: message.to_string(), data: None }
    }

    pub fn invalid_request(message: &str) -> Self {
        Self { code: -32600, message: message.to_string(), data: None }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self { code: -32601, message: format!("Method not found: {}", method), data: None }
    }

    pub fn invalid_params(message: &str) -> Self {
        Self { code: -32602, message: message.to_string(), data: None }
    }

    pub fn internal_error(message: &str) -> Self {
        Self { code: -32603, message: message.to_string(), data: None }
    }
}

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
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
        }
    }

    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    pub fn with_property(mut self, name: &str, prop_type: &str, description: &str, required: bool) -> Self {
        if let Some(obj) = self.input_schema.as_object_mut() {
            if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                props.insert(name.to_string(), serde_json::json!({
                    "type": prop_type,
                    "description": description
                }));
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
    Image { data: String, #[serde(rename = "mimeType")] mime_type: String },
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
    tool_handlers: HashMap<String, Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>>,
    resources: HashMap<String, McpResource>,
    resource_handlers: HashMap<String, Box<dyn Fn(&str) -> Result<McpResourceContent, String> + Send + Sync>>,
    prompts: HashMap<String, McpPrompt>,
    prompt_handlers: HashMap<String, Box<dyn Fn(HashMap<String, String>) -> Result<Vec<McpPromptMessage>, String> + Send + Sync>>,
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
        self.capabilities.tools = Some(McpToolsCapability { list_changed: Some(true) });
    }

    /// Register a resource
    pub fn register_resource<F>(&mut self, resource: McpResource, handler: F)
    where
        F: Fn(&str) -> Result<McpResourceContent, String> + Send + Sync + 'static,
    {
        self.resources.insert(resource.uri.clone(), resource.clone());
        self.resource_handlers.insert(resource.uri, Box::new(handler));
        self.capabilities.resources = Some(McpResourcesCapability {
            subscribe: Some(false),
            list_changed: Some(true),
        });
    }

    /// Register a prompt
    pub fn register_prompt<F>(&mut self, prompt: McpPrompt, handler: F)
    where
        F: Fn(HashMap<String, String>) -> Result<Vec<McpPromptMessage>, String> + Send + Sync + 'static,
    {
        self.prompts.insert(prompt.name.clone(), prompt.clone());
        self.prompt_handlers.insert(prompt.name, Box::new(handler));
        self.capabilities.prompts = Some(McpPromptsCapability { list_changed: Some(true) });
    }

    /// Handle an incoming MCP request
    pub fn handle_request(&self, request: McpRequest) -> McpResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tool_call(id, request.params),
            "resources/list" => self.handle_resources_list(id),
            "resources/read" => self.handle_resource_read(id, request.params),
            "prompts/list" => self.handle_prompts_list(id),
            "prompts/get" => self.handle_prompt_get(id, request.params),
            "ping" => McpResponse::success(id, serde_json::json!({})),
            _ => McpResponse::error(id, McpError::method_not_found(&request.method)),
        }
    }

    fn handle_initialize(&self, id: Option<serde_json::Value>) -> McpResponse {
        McpResponse::success(id, serde_json::json!({
            "protocolVersion": MCP_VERSION,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            },
            "capabilities": self.capabilities
        }))
    }

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> McpResponse {
        let tools: Vec<&McpTool> = self.tools.values().collect();
        McpResponse::success(id, serde_json::json!({ "tools": tools }))
    }

    fn handle_tool_call(&self, id: Option<serde_json::Value>, params: Option<serde_json::Value>) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error(id, McpError::invalid_params("Missing params")),
        };

        let name = match params.get("name").and_then(|n| n.as_str()) {
            Some(n) => n,
            None => return McpResponse::error(id, McpError::invalid_params("Missing tool name")),
        };

        let arguments = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

        let handler = match self.tool_handlers.get(name) {
            Some(h) => h,
            None => return McpResponse::error(id, McpError::method_not_found(name)),
        };

        match handler(arguments) {
            Ok(result) => McpResponse::success(id, serde_json::json!({
                "content": [{ "type": "text", "text": result.to_string() }]
            })),
            Err(e) => McpResponse::error(id, McpError::internal_error(&e)),
        }
    }

    fn handle_resources_list(&self, id: Option<serde_json::Value>) -> McpResponse {
        let resources: Vec<&McpResource> = self.resources.values().collect();
        McpResponse::success(id, serde_json::json!({ "resources": resources }))
    }

    fn handle_resource_read(&self, id: Option<serde_json::Value>, params: Option<serde_json::Value>) -> McpResponse {
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

    fn handle_prompts_list(&self, id: Option<serde_json::Value>) -> McpResponse {
        let prompts: Vec<&McpPrompt> = self.prompts.values().collect();
        McpResponse::success(id, serde_json::json!({ "prompts": prompts }))
    }

    fn handle_prompt_get(&self, id: Option<serde_json::Value>, params: Option<serde_json::Value>) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => return McpResponse::error(id, McpError::invalid_params("Missing params")),
        };

        let name = match params.get("name").and_then(|n| n.as_str()) {
            Some(n) => n,
            None => return McpResponse::error(id, McpError::invalid_params("Missing prompt name")),
        };

        let arguments: HashMap<String, String> = params.get("arguments")
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
                    r#"{"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"}}"#.to_string()
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
            let capabilities: McpServerCapabilities = result.get("capabilities")
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
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available tools
    pub fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        let request = McpRequest::new("tools/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let tools: Vec<McpTool> = result.get("tools")
                .cloned()
                .and_then(|t| serde_json::from_value(t).ok())
                .unwrap_or_default();
            Ok(tools)
        } else {
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Call a tool
    pub fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<serde_json::Value, McpError> {
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
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available resources
    pub fn list_resources(&mut self) -> Result<Vec<McpResource>, McpError> {
        let request = McpRequest::new("resources/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let resources: Vec<McpResource> = result.get("resources")
                .cloned()
                .and_then(|r| serde_json::from_value(r).ok())
                .unwrap_or_default();
            Ok(resources)
        } else {
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Read a resource
    pub fn read_resource(&mut self, uri: &str) -> Result<Vec<McpResourceContent>, McpError> {
        let request = McpRequest::new("resources/read")
            .with_id(self.next_id())
            .with_params(serde_json::json!({ "uri": uri }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let contents: Vec<McpResourceContent> = result.get("contents")
                .cloned()
                .and_then(|c| serde_json::from_value(c).ok())
                .unwrap_or_default();
            Ok(contents)
        } else {
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// List available prompts
    pub fn list_prompts(&mut self) -> Result<Vec<McpPrompt>, McpError> {
        let request = McpRequest::new("prompts/list").with_id(self.next_id());
        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let prompts: Vec<McpPrompt> = result.get("prompts")
                .cloned()
                .and_then(|p| serde_json::from_value(p).ok())
                .unwrap_or_default();
            Ok(prompts)
        } else {
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
        }
    }

    /// Get a prompt
    pub fn get_prompt(&mut self, name: &str, arguments: HashMap<String, String>) -> Result<Vec<McpPromptMessage>, McpError> {
        let request = McpRequest::new("prompts/get")
            .with_id(self.next_id())
            .with_params(serde_json::json!({
                "name": name,
                "arguments": arguments
            }));

        let response = self.send_request(request)?;

        if let Some(result) = response.result {
            let messages: Vec<McpPromptMessage> = result.get("messages")
                .cloned()
                .and_then(|m| serde_json::from_value(m).ok())
                .unwrap_or_default();
            Ok(messages)
        } else {
            Err(response.error.unwrap_or(McpError::internal_error("No result")))
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

        let response_body = response.into_string()
            .map_err(|e| McpError::internal_error(&e.to_string()))?;

        serde_json::from_str(&response_body)
            .map_err(|e| McpError::parse_error(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_request() {
        let request = McpRequest::new("tools/list")
            .with_id(1u64);

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

        server.register_tool(
            McpTool::new("echo", "Echo the input"),
            |args| {
                let text = args.get("text").and_then(|t| t.as_str()).unwrap_or("");
                Ok(serde_json::json!({ "echo": text }))
            }
        );

        // Test initialize
        let init_request = McpRequest::new("initialize")
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
}
