//! Function calling / Tool use support
//!
//! This module provides support for LLM function calling (tool use),
//! allowing models to invoke external functions.
//!
//! # Features
//!
//! - **Tool definitions**: Define tools with JSON schema
//! - **Tool execution**: Handle tool calls from model responses
//! - **Built-in tools**: Common utilities like date/time, math
//! - **Custom tools**: Register your own tools
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::tool_use::{Tool, ToolRegistry, ToolParameter, ParameterType};
//!
//! let mut registry = ToolRegistry::new();
//!
//! // Register a simple tool
//! registry.register(Tool::new("get_time", "Get current time")
//!     .with_handler(|_| Ok(serde_json::json!({"time": "12:00"}))));
//!
//! // Execute a tool call
//! let result = registry.execute("get_time", serde_json::json!({}));
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// A tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: ParameterType,
    /// Whether parameter is required
    pub required: bool,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<JsonValue>,
    /// Enum values (for string types)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "enum")]
    pub enum_values: Option<Vec<String>>,
}

impl ToolParameter {
    /// Create a required string parameter
    pub fn string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParameterType::String,
            required: true,
            default: None,
            enum_values: None,
        }
    }

    /// Create a required number parameter
    pub fn number(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParameterType::Number,
            required: true,
            default: None,
            enum_values: None,
        }
    }

    /// Create a required boolean parameter
    pub fn boolean(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParameterType::Boolean,
            required: true,
            default: None,
            enum_values: None,
        }
    }

    /// Make parameter optional
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set default value
    pub fn with_default(mut self, value: JsonValue) -> Self {
        self.default = Some(value);
        self.required = false;
        self
    }

    /// Set enum values
    pub fn with_enum(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }
}

/// Parameter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParameterType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

/// Tool handler function type
pub type ToolHandler = Box<dyn Fn(JsonValue) -> Result<JsonValue, ToolError> + Send + Sync>;

/// A tool definition
pub struct Tool {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters
    pub parameters: Vec<ToolParameter>,
    /// Handler function
    handler: Option<ToolHandler>,
}

impl Tool {
    /// Create a new tool
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            handler: None,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set handler function
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(JsonValue) -> Result<JsonValue, ToolError> + Send + Sync + 'static,
    {
        self.handler = Some(Box::new(handler));
        self
    }

    /// Execute the tool
    pub fn execute(&self, args: JsonValue) -> Result<JsonValue, ToolError> {
        // Validate required parameters
        for param in &self.parameters {
            if param.required {
                if args.get(&param.name).is_none() {
                    return Err(ToolError::MissingParameter(param.name.clone()));
                }
            }
        }

        // Call handler
        if let Some(ref handler) = self.handler {
            handler(args)
        } else {
            Err(ToolError::NoHandler(self.name.clone()))
        }
    }

    /// Convert to JSON schema format
    pub fn to_schema(&self) -> JsonValue {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert("type".to_string(), serde_json::json!(param.param_type));
            prop.insert("description".to_string(), serde_json::json!(param.description));

            if let Some(ref enum_values) = param.enum_values {
                prop.insert("enum".to_string(), serde_json::json!(enum_values));
            }

            properties.insert(param.name.clone(), JsonValue::Object(prop));

            if param.required {
                required.push(param.name.clone());
            }
        }

        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    }
}

/// Tool execution error
#[derive(Debug, Clone)]
pub enum ToolError {
    /// Missing required parameter
    MissingParameter(String),
    /// Invalid parameter value
    InvalidParameter { name: String, message: String },
    /// No handler registered
    NoHandler(String),
    /// Tool not found
    NotFound(String),
    /// Execution failed
    ExecutionFailed(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::MissingParameter(name) => write!(f, "Missing required parameter: {}", name),
            ToolError::InvalidParameter { name, message } => write!(f, "Invalid parameter '{}': {}", name, message),
            ToolError::NoHandler(name) => write!(f, "No handler for tool: {}", name),
            ToolError::NotFound(name) => write!(f, "Tool not found: {}", name),
            ToolError::ExecutionFailed(msg) => write!(f, "Tool execution failed: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

/// A tool call from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Arguments as JSON
    pub arguments: JsonValue,
}

/// Result of a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Call ID
    pub call_id: String,
    /// Tool name
    pub name: String,
    /// Result (success)
    pub result: Option<JsonValue>,
    /// Error (failure)
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a success result
    pub fn success(call_id: impl Into<String>, name: impl Into<String>, result: JsonValue) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            result: Some(result),
            error: None,
        }
    }

    /// Create an error result
    pub fn error(call_id: impl Into<String>, name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            result: None,
            error: Some(error.into()),
        }
    }
}

/// Registry for managing tools
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create with built-in tools
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }

    /// Register a tool
    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Unregister a tool
    pub fn unregister(&mut self, name: &str) -> Option<Tool> {
        self.tools.remove(name)
    }

    /// Get a tool
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// List all tools
    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Execute a tool call
    pub fn execute(&self, name: &str, args: JsonValue) -> Result<JsonValue, ToolError> {
        let tool = self.tools.get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        tool.execute(args)
    }

    /// Execute a ToolCall
    pub fn execute_call(&self, call: &ToolCall) -> ToolResult {
        match self.execute(&call.name, call.arguments.clone()) {
            Ok(result) => ToolResult::success(&call.id, &call.name, result),
            Err(e) => ToolResult::error(&call.id, &call.name, e.to_string()),
        }
    }

    /// Execute multiple tool calls
    pub fn execute_calls(&self, calls: &[ToolCall]) -> Vec<ToolResult> {
        calls.iter().map(|c| self.execute_call(c)).collect()
    }

    /// Get all tools as JSON schema (for API calls)
    pub fn to_schema(&self) -> Vec<JsonValue> {
        self.tools.values().map(|t| t.to_schema()).collect()
    }

    /// Register built-in tools
    pub fn register_builtins(&mut self) {
        // Get current time
        self.register(Tool::new("get_current_time", "Get the current date and time")
            .with_param(ToolParameter::string("timezone", "Timezone (e.g., 'UTC', 'America/New_York')").optional())
            .with_handler(|_args| {
                let now = chrono::Utc::now();
                Ok(serde_json::json!({
                    "datetime": now.to_rfc3339(),
                    "unix_timestamp": now.timestamp(),
                    "date": now.format("%Y-%m-%d").to_string(),
                    "time": now.format("%H:%M:%S").to_string()
                }))
            }));

        // Simple calculator
        self.register(Tool::new("calculate", "Perform basic math calculations")
            .with_param(ToolParameter::string("expression", "Math expression to evaluate"))
            .with_handler(|args| {
                let expr = args.get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolError::MissingParameter("expression".to_string()))?;

                // Simple evaluation (basic operations only)
                let result = Self::evaluate_simple_math(expr)?;
                Ok(serde_json::json!({
                    "expression": expr,
                    "result": result
                }))
            }));

        // String length
        self.register(Tool::new("string_length", "Get the length of a string")
            .with_param(ToolParameter::string("text", "The string to measure"))
            .with_handler(|args| {
                let text = args.get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolError::MissingParameter("text".to_string()))?;

                Ok(serde_json::json!({
                    "length": text.len(),
                    "characters": text.chars().count(),
                    "words": text.split_whitespace().count()
                }))
            }));

        // JSON validator
        self.register(Tool::new("validate_json", "Validate a JSON string")
            .with_param(ToolParameter::string("json_string", "JSON string to validate"))
            .with_handler(|args| {
                let json_str = args.get("json_string")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolError::MissingParameter("json_string".to_string()))?;

                match serde_json::from_str::<JsonValue>(json_str) {
                    Ok(parsed) => Ok(serde_json::json!({
                        "valid": true,
                        "type": match parsed {
                            JsonValue::Object(_) => "object",
                            JsonValue::Array(_) => "array",
                            JsonValue::String(_) => "string",
                            JsonValue::Number(_) => "number",
                            JsonValue::Bool(_) => "boolean",
                            JsonValue::Null => "null"
                        }
                    })),
                    Err(e) => Ok(serde_json::json!({
                        "valid": false,
                        "error": e.to_string()
                    }))
                }
            }));
    }

    fn evaluate_simple_math(expr: &str) -> Result<f64, ToolError> {
        // Very simple math evaluator for basic operations
        let expr = expr.replace(' ', "");

        // Try to parse as a simple number first
        if let Ok(n) = expr.parse::<f64>() {
            return Ok(n);
        }

        // Look for operators
        for (i, c) in expr.char_indices().rev() {
            if c == '+' || c == '-' {
                if i > 0 {
                    let left = Self::evaluate_simple_math(&expr[..i])?;
                    let right = Self::evaluate_simple_math(&expr[i + 1..])?;
                    return Ok(if c == '+' { left + right } else { left - right });
                }
            }
        }

        for (i, c) in expr.char_indices().rev() {
            if c == '*' || c == '/' {
                let left = Self::evaluate_simple_math(&expr[..i])?;
                let right = Self::evaluate_simple_math(&expr[i + 1..])?;
                return Ok(if c == '*' {
                    left * right
                } else {
                    if right == 0.0 {
                        return Err(ToolError::ExecutionFailed("Division by zero".to_string()));
                    }
                    left / right
                });
            }
        }

        Err(ToolError::ExecutionFailed(format!("Cannot evaluate: {}", expr)))
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse tool calls from model response
pub fn parse_tool_calls(response: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Try to parse as JSON array of tool calls
    if let Ok(parsed) = serde_json::from_str::<Vec<ToolCall>>(response) {
        return parsed;
    }

    // Try to find tool calls in markdown code blocks
    for block in response.split("```") {
        let block = block.trim();
        if block.starts_with("json") || block.starts_with("tool") {
            let json_content = block
                .trim_start_matches("json")
                .trim_start_matches("tool")
                .trim();

            if let Ok(call) = serde_json::from_str::<ToolCall>(json_content) {
                calls.push(call);
            }
        }
    }

    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        let tool = Tool::new("test", "A test tool")
            .with_param(ToolParameter::string("input", "Input value"))
            .with_handler(|args| {
                let input = args.get("input").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({"result": input.to_uppercase()}))
            });

        assert_eq!(tool.name, "test");
        assert_eq!(tool.parameters.len(), 1);
    }

    #[test]
    fn test_tool_execution() {
        let tool = Tool::new("echo", "Echo input")
            .with_param(ToolParameter::string("message", "Message"))
            .with_handler(|args| Ok(args));

        let result = tool.execute(serde_json::json!({"message": "hello"})).unwrap();
        assert_eq!(result["message"], "hello");
    }

    #[test]
    fn test_missing_parameter() {
        let tool = Tool::new("test", "Test")
            .with_param(ToolParameter::string("required", "Required param"))
            .with_handler(|_| Ok(serde_json::json!({})));

        let result = tool.execute(serde_json::json!({}));
        assert!(matches!(result, Err(ToolError::MissingParameter(_))));
    }

    #[test]
    fn test_registry() {
        let mut registry = ToolRegistry::new();

        registry.register(Tool::new("test", "Test tool")
            .with_handler(|_| Ok(serde_json::json!({"ok": true}))));

        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());

        let result = registry.execute("test", serde_json::json!({})).unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn test_builtin_tools() {
        let registry = ToolRegistry::with_builtins();

        assert!(registry.get("get_current_time").is_some());
        assert!(registry.get("calculate").is_some());
        assert!(registry.get("string_length").is_some());
    }

    #[test]
    fn test_calculate() {
        let registry = ToolRegistry::with_builtins();

        let result = registry.execute("calculate", serde_json::json!({
            "expression": "2 + 3 * 4"
        })).unwrap();

        assert_eq!(result["result"], 14.0);
    }

    #[test]
    fn test_tool_schema() {
        let tool = Tool::new("search", "Search for something")
            .with_param(ToolParameter::string("query", "Search query"))
            .with_param(ToolParameter::number("limit", "Max results").optional());

        let schema = tool.to_schema();
        assert_eq!(schema["function"]["name"], "search");
    }
}
