//! MCP Server implementation.

use std::collections::HashMap;

use super::types::*;

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
