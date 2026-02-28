//! Model integration
//!
//! Unified interface for integrating any LLM (local or cloud) with
//! tool calling, web search, and agentic capabilities.

use crate::tool_calling::{Tool, ToolCall, ToolRegistry};
use crate::web_search::{SearchConfig, WebSearchManager};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unified message format for any model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Tool call information in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

/// Function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Model provider trait
pub trait ModelProvider: Send + Sync {
    /// Send messages and get a response
    fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<ChatResponse, ModelError>;

    /// Get provider name
    fn name(&self) -> &str;

    /// Check if provider supports tool calling
    fn supports_tools(&self) -> bool;

    /// Check if provider supports streaming
    fn supports_streaming(&self) -> bool;
}

/// Chat response from model
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallInfo>,
    pub finish_reason: FinishReason,
    pub usage: Option<TokenUsage>,
}

/// Reason for response completion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Error,
}

/// Token usage information
#[derive(Debug, Clone, Copy)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model errors
#[derive(Debug, Clone)]
pub enum ModelError {
    ConnectionError(String),
    RateLimitError,
    AuthenticationError,
    InvalidRequest(String),
    ContextLengthExceeded,
    ContentFilterTriggered,
    ProviderError(String),
    ToolExecutionError(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            Self::RateLimitError => write!(f, "Rate limit exceeded"),
            Self::AuthenticationError => write!(f, "Authentication failed"),
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::ContextLengthExceeded => write!(f, "Context length exceeded"),
            Self::ContentFilterTriggered => write!(f, "Content filter triggered"),
            Self::ProviderError(msg) => write!(f, "Provider error: {}", msg),
            Self::ToolExecutionError(msg) => write!(f, "Tool execution error: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}

/// Ollama provider implementation
pub struct OllamaProvider {
    base_url: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: model.to_string(),
        }
    }

    pub fn with_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }
}

impl ModelProvider for OllamaProvider {
    fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[serde_json::Value]>,
    ) -> Result<ChatResponse, ModelError> {
        let ollama_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": format!("{:?}", m.role).to_lowercase(),
                    "content": m.content
                })
            })
            .collect();

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": ollama_messages,
            "stream": false
        });

        let response = ureq::post(&format!("{}/api/chat", self.base_url))
            .send_json(&request_body)
            .map_err(|e| ModelError::ConnectionError(e.to_string()))?;

        let response_json: serde_json::Value = response
            .into_json()
            .map_err(|e| ModelError::ProviderError(e.to_string()))?;

        let content = response_json["message"]["content"]
            .as_str()
            .map(|s| s.to_string());

        Ok(ChatResponse {
            content,
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: None,
        })
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn supports_tools(&self) -> bool {
        false // Basic Ollama doesn't support native tool calling
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

/// LM Studio provider implementation
pub struct LMStudioProvider {
    base_url: String,
}

impl LMStudioProvider {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:1234/v1".to_string(),
        }
    }

    pub fn with_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }
}

impl Default for LMStudioProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelProvider for LMStudioProvider {
    fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<ChatResponse, ModelError> {
        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut msg = serde_json::json!({
                    "role": format!("{:?}", m.role).to_lowercase(),
                    "content": m.content
                });

                if let Some(ref tc) = m.tool_calls {
                    msg["tool_calls"] = serde_json::to_value(tc).unwrap_or_default();
                }

                if let Some(ref id) = m.tool_call_id {
                    msg["tool_call_id"] = serde_json::Value::String(id.clone());
                }

                msg
            })
            .collect();

        let mut request_body = serde_json::json!({
            "messages": openai_messages
        });

        if let Some(tools) = tools {
            request_body["tools"] = serde_json::Value::Array(tools.to_vec());
        }

        let response = ureq::post(&format!("{}/chat/completions", self.base_url))
            .send_json(&request_body)
            .map_err(|e| ModelError::ConnectionError(e.to_string()))?;

        let response_json: serde_json::Value = response
            .into_json()
            .map_err(|e| ModelError::ProviderError(e.to_string()))?;

        let choice = &response_json["choices"][0];
        let message = &choice["message"];

        let content = message["content"].as_str().map(|s| s.to_string());

        let tool_calls: Vec<ToolCallInfo> = message["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        Some(ToolCallInfo {
                            id: tc["id"].as_str()?.to_string(),
                            r#type: tc["type"].as_str().unwrap_or("function").to_string(),
                            function: FunctionCall {
                                name: tc["function"]["name"].as_str()?.to_string(),
                                arguments: tc["function"]["arguments"].as_str()?.to_string(),
                            },
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let finish_reason = match choice["finish_reason"].as_str() {
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        };

        let usage = response_json["usage"].as_object().map(|u| TokenUsage {
            prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as usize,
            total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as usize,
        });

        Ok(ChatResponse {
            content,
            tool_calls,
            finish_reason,
            usage,
        })
    }

    fn name(&self) -> &str {
        "lm_studio"
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

/// Integrated model client with tools and search
pub struct IntegratedModelClient {
    provider: Box<dyn ModelProvider>,
    tools: ToolRegistry,
    search_manager: Option<WebSearchManager>,
    conversation: Vec<ChatMessage>,
    system_prompt: Option<String>,
}

impl IntegratedModelClient {
    pub fn new<P: ModelProvider + 'static>(provider: P) -> Self {
        Self {
            provider: Box::new(provider),
            tools: ToolRegistry::new(),
            search_manager: None,
            conversation: Vec::new(),
            system_prompt: None,
        }
    }

    /// Enable web search
    pub fn with_web_search(mut self, config: SearchConfig) -> Self {
        self.search_manager = Some(WebSearchManager::new(config));

        // Register search tool
        let search_tool = Tool::new("web_search", "Search the web for current information")
            .with_parameter(crate::tool_calling::ToolParameter::new(
                "query",
                "The search query",
                crate::tool_calling::ParameterType::String,
            ));
        self.tools.register(search_tool);

        self
    }

    /// Register a tool
    pub fn register_tool(&mut self, tool: Tool) {
        self.tools.register(tool);
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// Send a message and get response with automatic tool handling
    pub fn chat(&mut self, user_message: &str) -> Result<String, ModelError> {
        // Add system prompt if first message
        if self.conversation.is_empty() {
            if let Some(ref system) = self.system_prompt {
                self.conversation.push(ChatMessage {
                    role: ChatRole::System,
                    content: system.clone(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        // Add user message
        self.conversation.push(ChatMessage {
            role: ChatRole::User,
            content: user_message.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        // Get tools schema if provider supports them
        let tools_schema = if self.provider.supports_tools() {
            Some(self.tools.to_json_schema())
        } else {
            None
        };

        // Call model
        let response = self.provider.chat(
            &self.conversation,
            tools_schema.as_ref().map(|t| t.as_slice()),
        )?;

        // Handle tool calls if any
        if !response.tool_calls.is_empty() {
            return self.handle_tool_calls(response);
        }

        // Add assistant response to conversation
        if let Some(ref content) = response.content {
            self.conversation.push(ChatMessage {
                role: ChatRole::Assistant,
                content: content.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        response.content.ok_or(ModelError::ProviderError(
            "No content in response".to_string(),
        ))
    }

    /// Handle tool calls from model response
    fn handle_tool_calls(&mut self, response: ChatResponse) -> Result<String, ModelError> {
        // Add assistant message with tool calls
        self.conversation.push(ChatMessage {
            role: ChatRole::Assistant,
            content: response.content.clone().unwrap_or_default(),
            name: None,
            tool_calls: Some(response.tool_calls.clone()),
            tool_call_id: None,
        });

        // Execute each tool call
        for tool_call in &response.tool_calls {
            let result = self.execute_tool_call(tool_call)?;

            // Add tool result to conversation
            self.conversation.push(ChatMessage {
                role: ChatRole::Tool,
                content: result,
                name: Some(tool_call.function.name.clone()),
                tool_calls: None,
                tool_call_id: Some(tool_call.id.clone()),
            });
        }

        // Get tools schema
        let tools_schema = if self.provider.supports_tools() {
            Some(self.tools.to_json_schema())
        } else {
            None
        };

        // Call model again with tool results
        let final_response = self.provider.chat(
            &self.conversation,
            tools_schema.as_ref().map(|t| t.as_slice()),
        )?;

        // If more tool calls, recurse (with depth limit in real implementation)
        if !final_response.tool_calls.is_empty() {
            return self.handle_tool_calls(final_response);
        }

        // Add final response to conversation
        if let Some(ref content) = final_response.content {
            self.conversation.push(ChatMessage {
                role: ChatRole::Assistant,
                content: content.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        final_response.content.ok_or(ModelError::ProviderError(
            "No content in response".to_string(),
        ))
    }

    /// Execute a single tool call
    fn execute_tool_call(&mut self, tool_call: &ToolCallInfo) -> Result<String, ModelError> {
        let name = &tool_call.function.name;
        let args: HashMap<String, serde_json::Value> =
            serde_json::from_str(&tool_call.function.arguments)
                .map_err(|e| ModelError::ToolExecutionError(format!("Invalid arguments: {}", e)))?;

        // Special handling for web search
        if name == "web_search" {
            if let Some(ref mut search_manager) = self.search_manager {
                let query = args.get("query").and_then(|v| v.as_str()).ok_or(
                    ModelError::ToolExecutionError("Missing query parameter".to_string()),
                )?;

                return search_manager
                    .search_for_context(query, 2500)
                    .map_err(|e| ModelError::ToolExecutionError(e.to_string()));
            }
        }

        // Execute through registry
        let call = ToolCall {
            id: tool_call.id.clone(),
            tool_name: name.clone(),
            arguments: args,
        };

        let result = self.tools.execute(&call);

        if result.success {
            Ok(result.output)
        } else {
            Err(ModelError::ToolExecutionError(
                result.error.unwrap_or_default(),
            ))
        }
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation.clear();
    }

    /// Get conversation history
    pub fn get_history(&self) -> &[ChatMessage] {
        &self.conversation
    }

    /// Get provider name
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

/// Convenience function to create an Ollama client with tools
pub fn create_ollama_client(model: &str) -> IntegratedModelClient {
    IntegratedModelClient::new(OllamaProvider::new(model))
}

/// Convenience function to create an LM Studio client with tools
pub fn create_lm_studio_client() -> IntegratedModelClient {
    IntegratedModelClient::new(LMStudioProvider::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new("llama2").with_url("http://localhost:11434");
        assert_eq!(provider.name(), "ollama");
        assert!(!provider.supports_tools());
    }

    #[test]
    fn test_lm_studio_provider_creation() {
        let provider = LMStudioProvider::new();
        assert_eq!(provider.name(), "lm_studio");
        assert!(provider.supports_tools());
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_integrated_client_creation() {
        let client = create_ollama_client("llama2");
        assert_eq!(client.provider_name(), "ollama");
    }

    #[test]
    fn test_model_error_display() {
        assert_eq!(ModelError::RateLimitError.to_string(), "Rate limit exceeded");
        assert_eq!(ModelError::AuthenticationError.to_string(), "Authentication failed");
        assert_eq!(ModelError::ContextLengthExceeded.to_string(), "Context length exceeded");
        assert_eq!(ModelError::ContentFilterTriggered.to_string(), "Content filter triggered");
        assert!(ModelError::ConnectionError("timeout".into()).to_string().contains("timeout"));
        assert!(ModelError::ToolExecutionError("fail".into()).to_string().contains("fail"));
    }

    #[test]
    fn test_chat_role_serialization_all() {
        let roles = [ChatRole::System, ChatRole::User, ChatRole::Assistant, ChatRole::Tool];
        let expected = ["system", "user", "assistant", "tool"];
        for (role, exp) in roles.iter().zip(expected.iter()) {
            let json = serde_json::to_string(role).unwrap();
            assert!(json.contains(exp), "Role {:?} should serialize to {}", role, exp);
        }
    }

    #[test]
    fn test_lm_studio_default() {
        let provider = LMStudioProvider::default();
        assert_eq!(provider.name(), "lm_studio");
        assert!(provider.supports_tools());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::ToolCalls);
        assert_ne!(FinishReason::Length, FinishReason::Error);
    }

    #[test]
    fn test_client_history_and_clear() {
        let client = create_lm_studio_client();
        assert!(client.get_history().is_empty());
        // Can't call chat without a real server, but we can test clear
        let mut client = client.with_system_prompt("You are helpful.");
        client.clear_history();
        assert!(client.get_history().is_empty());
    }

    #[test]
    fn test_tool_call_info_serialization() {
        let tc = ToolCallInfo {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Madrid"}"#.to_string(),
            },
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("get_weather"));
        assert!(json.contains("Madrid"));
        let restored: ToolCallInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.function.name, "get_weather");
    }
}
