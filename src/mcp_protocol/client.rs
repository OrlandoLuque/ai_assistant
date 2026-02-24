//! MCP Client for connecting to MCP servers.

use std::collections::HashMap;

use super::types::*;

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
