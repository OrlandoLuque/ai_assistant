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
        // Clamp the client-supplied cursor: an out-of-range value would make
        // `start > end` and panic the slice below.
        let start = cursor.unwrap_or(0).min(all_tools.len());
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

    /// Handle one message from a stream transport (stdio, pipes, sockets).
    ///
    /// Returns `None` when the message is a **notification** — a well-formed request that
    /// carries no `id`. JSON-RPC 2.0 forbids answering those, and [`Self::handle_message`]
    /// cannot express it because it always returns a string. A stdio server that echoes a
    /// reply to `notifications/initialized` is putting a frame on the wire that the client
    /// never asked for and, depending on the client, is not prepared to read.
    ///
    /// The notification still *runs*: the spec says no reply, not no effect.
    ///
    /// A message that fails to parse is **not** treated as a notification. It cannot be —
    /// deciding it was one requires having parsed it. Per the spec it gets an error
    /// response with a null `id`, which is what this returns.
    pub fn handle_stream_message(&self, message: &str) -> Option<String> {
        let is_notification = serde_json::from_str::<McpRequest>(message)
            .map(|r| r.id.is_none())
            .unwrap_or(false);
        let reply = self.handle_message(message);
        if is_notification {
            None
        } else {
            Some(reply)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_notification_gets_no_reply_but_still_runs() {
        // JSON-RPC 2.0: a request without `id` is a notification and MUST NOT be
        // answered. `handle_message` cannot say that — it always returns a string — so a
        // stdio transport built on it would write an unrequested frame.
        let server = McpServer::new("test", "1.0");
        let notification = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert_eq!(server.handle_stream_message(notification), None);
        // The same message through the older entry point does produce a reply, which is
        // exactly the difference this method exists for.
        assert!(!server.handle_message(notification).is_empty());
    }

    #[test]
    fn a_request_with_an_id_is_still_answered() {
        let server = McpServer::new("test", "1.0");
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let reply = server
            .handle_stream_message(request)
            .expect("a request with an id must be answered");
        assert!(reply.contains("\"id\":1"), "reply was {reply}");
    }

    #[test]
    fn unparseable_input_is_answered_not_swallowed() {
        // Not a notification: deciding it was one would require having parsed it. The
        // spec says an error response with a null id, and swallowing it would leave the
        // client waiting for a reply that never comes.
        let server = McpServer::new("test", "1.0");
        let reply = server
            .handle_stream_message("{ this is not json")
            .expect("a parse error must be reported, not swallowed");
        assert!(reply.contains("-32700"), "reply was {reply}");
    }

    #[test]
    fn tools_list_out_of_range_cursor_no_panic() {
        // Regression: an out-of-range client-supplied cursor must not panic the
        // page slice (would make `start > end`).
        let server = McpServer::new("test", "1.0");
        let params = Some(serde_json::json!({ "cursor": "999999" }));
        let _ = server.handle_tools_list(Some(serde_json::json!(1)), params); // must not panic
    }
}
