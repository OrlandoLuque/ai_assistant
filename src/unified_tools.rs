//! Unified tool system for LLM function calling and tool invocation.
//!
//! Merges capabilities from multiple subsystems into a single coherent API:
//! typed error handling, nested parameter schemas with ranges and enums,
//! multi-format response parsing (JSON, `[TOOL:]`, XML), provider plugins,
//! and a fluent builder pattern for tool registration.
//!
//! ## Key types
//!
//! - [`ToolRegistry`] — Central registry for available tools
//! - [`ToolCall`] — A parsed tool invocation (name + arguments)
//! - [`ToolResult`] — Execution result (success/error) returned to the LLM
//! - [`ToolSchema`] — JSON Schema description of a tool's parameters
//! - [`ToolError`] — Typed error enum for tool execution failures
//!
//! ## Feature flags
//!
//! Requires the `tools` feature flag (included in `full`).

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// Parameter Types
// ============================================================================

/// JSON Schema parameter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

impl fmt::Display for ParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamType::String => write!(f, "string"),
            ParamType::Number => write!(f, "number"),
            ParamType::Integer => write!(f, "integer"),
            ParamType::Boolean => write!(f, "boolean"),
            ParamType::Array => write!(f, "array"),
            ParamType::Object => write!(f, "object"),
        }
    }
}

// ============================================================================
// Parameter Schema
// ============================================================================

/// Schema for a single tool parameter, supporting nested types and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSchema {
    /// Parameter name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Type of the parameter.
    pub param_type: ParamType,
    /// Whether the parameter is required.
    pub required: bool,
    /// Default value if not provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<JsonValue>,
    /// Allowed enum values (for string enums).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Schema for array items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParamSchema>>,
    /// Minimum value (for numbers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value (for numbers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

impl ParamSchema {
    /// Create a required string parameter.
    pub fn string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::String,
            required: true,
            default: None,
            enum_values: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a required number parameter.
    pub fn number(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::Number,
            required: true,
            default: None,
            enum_values: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a required integer parameter.
    pub fn integer(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::Integer,
            required: true,
            default: None,
            enum_values: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a required boolean parameter.
    pub fn boolean(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::Boolean,
            required: true,
            default: None,
            enum_values: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a required enum parameter (string with allowed values).
    pub fn enum_type(
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::String,
            required: true,
            default: None,
            enum_values: Some(values),
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a required array parameter with item schema.
    pub fn array(
        name: impl Into<String>,
        description: impl Into<String>,
        items: ParamSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::Array,
            required: true,
            default: None,
            enum_values: None,
            items: Some(Box::new(items)),
            minimum: None,
            maximum: None,
        }
    }

    /// Mark this parameter as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, default: JsonValue) -> Self {
        self.default = Some(default);
        self.required = false;
        self
    }

    /// Set min/max range for numeric parameters.
    pub fn with_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.minimum = min;
        self.maximum = max;
        self
    }

    /// Convert to JSON Schema format.
    pub fn to_json_schema(&self) -> JsonValue {
        let mut schema = serde_json::json!({
            "type": self.param_type.to_string(),
            "description": self.description,
        });
        if let Some(ref enums) = self.enum_values {
            schema["enum"] = serde_json::json!(enums);
        }
        if let Some(ref items) = self.items {
            schema["items"] = items.to_json_schema();
        }
        if let Some(min) = self.minimum {
            schema["minimum"] = serde_json::json!(min);
        }
        if let Some(max) = self.maximum {
            schema["maximum"] = serde_json::json!(max);
        }
        if let Some(ref default) = self.default {
            schema["default"] = default.clone();
        }
        schema
    }
}

// ============================================================================
// Tool Definition
// ============================================================================

/// A tool the AI model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Unique tool name (must match handler registration).
    pub name: String,
    /// Human-readable description for the model.
    pub description: String,
    /// Parameter schemas.
    pub parameters: Vec<ParamSchema>,
    /// Optional category for grouping (e.g. "math", "system", "search").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Whether this tool is currently enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl ToolDef {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            category: None,
            enabled: true,
        }
    }

    /// Add a parameter.
    pub fn with_param(mut self, param: ParamSchema) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Convert to OpenAI function format.
    pub fn to_openai_function(&self) -> JsonValue {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            properties.insert(param.name.clone(), param.to_json_schema());
            if param.required {
                required.push(serde_json::json!(param.name));
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
                    "required": required,
                }
            }
        })
    }

    /// Convert to JSON Schema format (simpler, without function wrapper).
    pub fn to_json_schema(&self) -> JsonValue {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            properties.insert(param.name.clone(), param.to_json_schema());
            if param.required {
                required.push(serde_json::json!(param.name));
            }
        }

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    }
}

// ============================================================================
// Tool Builder (fluent API)
// ============================================================================

/// Fluent builder for constructing tool definitions.
///
/// # Example
///
/// ```rust
/// use ai_assistant::unified_tools::ToolBuilder;
///
/// let tool = ToolBuilder::new("search", "Search the web")
///     .required_string("query", "Search query")
///     .optional_number("limit", "Max results")
///     .category("search")
///     .build();
/// ```
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: Vec<ParamSchema>,
    category: Option<String>,
}

impl ToolBuilder {
    /// Start building a new tool.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
            category: None,
        }
    }

    /// Add a custom parameter schema.
    pub fn param(mut self, param: ParamSchema) -> Self {
        self.parameters.push(param);
        self
    }

    /// Add a required string parameter.
    pub fn required_string(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters.push(ParamSchema::string(name, desc));
        self
    }

    /// Add an optional string parameter.
    pub fn optional_string(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters
            .push(ParamSchema::string(name, desc).optional());
        self
    }

    /// Add a required number parameter.
    pub fn required_number(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters.push(ParamSchema::number(name, desc));
        self
    }

    /// Add an optional number parameter.
    pub fn optional_number(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters
            .push(ParamSchema::number(name, desc).optional());
        self
    }

    /// Add a required integer parameter.
    pub fn required_integer(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters.push(ParamSchema::integer(name, desc));
        self
    }

    /// Add a required boolean parameter.
    pub fn required_bool(mut self, name: impl Into<String>, desc: impl Into<String>) -> Self {
        self.parameters.push(ParamSchema::boolean(name, desc));
        self
    }

    /// Add a required enum parameter.
    pub fn required_enum(
        mut self,
        name: impl Into<String>,
        desc: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        self.parameters
            .push(ParamSchema::enum_type(name, desc, values));
        self
    }

    /// Set category.
    pub fn category(mut self, cat: impl Into<String>) -> Self {
        self.category = Some(cat.into());
        self
    }

    /// Build the tool definition.
    pub fn build(self) -> ToolDef {
        ToolDef {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            category: self.category,
            enabled: true,
        }
    }
}

// ============================================================================
// Tool Call
// ============================================================================

/// A tool call from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID (for matching results).
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments as key-value pairs.
    pub arguments: HashMap<String, JsonValue>,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(name: impl Into<String>, arguments: HashMap<String, JsonValue>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            arguments,
        }
    }

    /// Create with a specific ID.
    pub fn with_id(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: HashMap<String, JsonValue>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Get a string argument.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    /// Get a number argument.
    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.arguments.get(key).and_then(|v| v.as_f64())
    }

    /// Get an integer argument.
    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.arguments.get(key).and_then(|v| v.as_i64())
    }

    /// Get a boolean argument.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key).and_then(|v| v.as_bool())
    }

    /// Get a raw JSON value argument.
    pub fn get_value(&self, key: &str) -> Option<&JsonValue> {
        self.arguments.get(key)
    }

    /// Get arguments as a JSON Value object.
    pub fn arguments_json(&self) -> JsonValue {
        serde_json::to_value(&self.arguments).unwrap_or(JsonValue::Null)
    }
}

// ============================================================================
// Tool Output & Error
// ============================================================================

/// Successful output from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text content of the result.
    pub content: String,
    /// Optional structured data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

impl ToolOutput {
    /// Create a text-only output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            data: None,
        }
    }

    /// Create output with text and structured data.
    pub fn with_data(content: impl Into<String>, data: JsonValue) -> Self {
        Self {
            content: content.into(),
            data: Some(data),
        }
    }
}

/// Error type for tool execution failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolError {
    /// Tool not found in registry.
    NotFound(String),
    /// Required parameter is missing.
    MissingParameter(String),
    /// Parameter value is invalid.
    InvalidParameter { name: String, message: String },
    /// Tool exists but has no handler registered.
    NoHandler(String),
    /// Handler execution failed.
    ExecutionFailed(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::NotFound(name) => write!(f, "tool not found: {}", name),
            ToolError::MissingParameter(name) => write!(f, "missing required parameter: {}", name),
            ToolError::InvalidParameter { name, message } => {
                write!(f, "invalid parameter '{}': {}", name, message)
            }
            ToolError::NoHandler(name) => write!(f, "no handler for tool: {}", name),
            ToolError::ExecutionFailed(msg) => write!(f, "execution failed: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

// ============================================================================
// Tool Choice
// ============================================================================

/// Controls how the model selects tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model decides whether to call a tool.
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call at least one tool.
    Required,
    /// Model must call a specific tool.
    Specific { name: String },
}

impl ToolChoice {
    /// Convert to API format for OpenAI-compatible endpoints.
    pub fn to_api_value(&self) -> JsonValue {
        match self {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Required => serde_json::json!("required"),
            ToolChoice::Specific { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }
}

// ============================================================================
// Handler Type
// ============================================================================

/// Handler function for tool execution.
///
/// Receives the parsed ToolCall and returns either a ToolOutput or ToolError.
/// Handlers must be Send + Sync for thread-safe registration.
pub type ToolHandler = Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync>;

// ============================================================================
// Tool Registry
// ============================================================================

struct RegisteredTool {
    definition: ToolDef,
    handler: ToolHandler,
}

impl fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("definition", &self.definition)
            .field("has_handler", &true)
            .finish()
    }
}

/// Registry for managing tools and executing tool calls.
///
/// Provides registration, lookup, validation, execution, and schema export
/// for tools. Thread-safe through Arc-wrapped handlers.
///
/// # Example
///
/// ```rust
/// use ai_assistant::unified_tools::*;
/// use std::sync::Arc;
///
/// let mut registry = ToolRegistry::new();
///
/// let def = ToolBuilder::new("greet", "Greet a user")
///     .required_string("name", "User name")
///     .build();
///
/// registry.register(def, Arc::new(|call: &ToolCall| {
///     let name = call.get_string("name").unwrap_or("world");
///     Ok(ToolOutput::text(format!("Hello, {}!", name)))
/// }));
///
/// let call = ToolCall::new("greet", [("name".into(), serde_json::json!("Alice"))].into());
/// let result = registry.execute(&call).unwrap();
/// assert_eq!(result.content, "Hello, Alice!");
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create a registry with built-in tools pre-registered.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        for (def, handler) in builtin_tools() {
            reg.register(def, handler);
        }
        reg
    }

    /// Register a tool with its handler.
    pub fn register(&mut self, definition: ToolDef, handler: ToolHandler) {
        self.tools.insert(
            definition.name.clone(),
            RegisteredTool {
                definition,
                handler,
            },
        );
    }

    /// Unregister a tool by name.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Get a tool definition by name.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.get(name).map(|t| &t.definition)
    }

    /// List all registered tool definitions.
    pub fn list(&self) -> Vec<&ToolDef> {
        self.tools.values().map(|t| &t.definition).collect()
    }

    /// List only enabled tools.
    pub fn list_enabled(&self) -> Vec<&ToolDef> {
        self.tools
            .values()
            .filter(|t| t.definition.enabled)
            .map(|t| &t.definition)
            .collect()
    }

    /// Enable or disable a tool.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(tool) = self.tools.get_mut(name) {
            tool.definition.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Validate a tool call against its definition.
    ///
    /// Checks: tool exists, tool enabled, required params present, enum values valid.
    pub fn validate_call(&self, call: &ToolCall) -> Result<(), ToolError> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| ToolError::NotFound(call.name.clone()))?;

        if !tool.definition.enabled {
            return Err(ToolError::NotFound(format!("{} (disabled)", call.name)));
        }

        for param in &tool.definition.parameters {
            if param.required && !call.arguments.contains_key(&param.name) {
                if param.default.is_none() {
                    return Err(ToolError::MissingParameter(param.name.clone()));
                }
            }

            // Validate enum values
            if let (Some(ref enums), Some(value)) =
                (&param.enum_values, call.arguments.get(&param.name))
            {
                if let Some(s) = value.as_str() {
                    if !enums.contains(&s.to_string()) {
                        return Err(ToolError::InvalidParameter {
                            name: param.name.clone(),
                            message: format!("value '{}' not in allowed values: {:?}", s, enums),
                        });
                    }
                }
            }

            // Validate numeric range
            if let Some(value) = call.arguments.get(&param.name) {
                if let Some(num) = value.as_f64() {
                    if let Some(min) = param.minimum {
                        if num < min {
                            return Err(ToolError::InvalidParameter {
                                name: param.name.clone(),
                                message: format!("value {} below minimum {}", num, min),
                            });
                        }
                    }
                    if let Some(max) = param.maximum {
                        if num > max {
                            return Err(ToolError::InvalidParameter {
                                name: param.name.clone(),
                                message: format!("value {} above maximum {}", num, max),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a tool call: validate then run the handler.
    pub fn execute(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        self.validate_call(call)?;

        let tool = self
            .tools
            .get(&call.name)
            .expect("tool exists: validate_call verified");
        (tool.handler)(call)
    }

    /// Execute multiple tool calls, returning results paired with calls.
    pub fn execute_all<'a>(
        &self,
        calls: &'a [ToolCall],
    ) -> Vec<(&'a ToolCall, Result<ToolOutput, ToolError>)> {
        calls.iter().map(|c| (c, self.execute(c))).collect()
    }

    /// Export all enabled tools as OpenAI function format.
    pub fn to_openai_functions(&self) -> Vec<JsonValue> {
        self.list_enabled()
            .iter()
            .map(|t| t.to_openai_function())
            .collect()
    }

    /// Export all enabled tools as JSON Schema format.
    pub fn to_json_schema(&self) -> Vec<JsonValue> {
        self.list_enabled()
            .iter()
            .map(|t| t.to_json_schema())
            .collect()
    }

    /// Format tool results into text context suitable for injecting into a conversation.
    pub fn format_results_for_context(
        results: &[(&ToolCall, Result<ToolOutput, ToolError>)],
    ) -> String {
        let mut out = String::new();
        for (call, result) in results {
            out.push_str(&format!("[Tool: {}]\n", call.name));
            match result {
                Ok(output) => {
                    out.push_str(&output.content);
                }
                Err(e) => {
                    out.push_str(&format!("Error: {}", e));
                }
            }
            out.push_str("\n\n");
        }
        out.trim_end().to_string()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tool Call Parsing (multi-format)
// ============================================================================

/// Parse tool calls from model output text.
///
/// Supports multiple formats:
/// 1. JSON: `[{"name": "...", "arguments": {...}}]` or `{"tool_calls": [...]}`
/// 2. Text: `[TOOL:name({"arg": "val"})]`
/// 3. XML: `<tool name="..."><param name="k">v</param></tool>`
/// 4. OpenAI: `{"function_call": {"name": "...", "arguments": "..."}}`
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try JSON array format first
    if let Some(calls) = try_parse_json_array(trimmed) {
        return calls;
    }

    // Try OpenAI response format
    if let Some(calls) = try_parse_openai_response(trimmed) {
        return calls;
    }

    // Try text format: [TOOL:name(args)]
    let mut calls = try_parse_text_format(trimmed);

    // Try XML format: <tool name="...">
    calls.extend(try_parse_xml_format(trimmed));

    calls
}

fn try_parse_json_array(text: &str) -> Option<Vec<ToolCall>> {
    // Try as direct array
    if let Ok(arr) = serde_json::from_str::<Vec<JsonValue>>(text) {
        let calls: Vec<ToolCall> = arr
            .iter()
            .filter_map(|v| parse_single_call_json(v))
            .collect();
        if !calls.is_empty() {
            return Some(calls);
        }
    }

    // Try as object with tool_calls field
    if let Ok(obj) = serde_json::from_str::<JsonValue>(text) {
        if let Some(arr) = obj.get("tool_calls").and_then(|v| v.as_array()) {
            let calls: Vec<ToolCall> = arr
                .iter()
                .filter_map(|v| parse_single_call_json(v))
                .collect();
            if !calls.is_empty() {
                return Some(calls);
            }
        }
        // Single tool call object
        if let Some(call) = parse_single_call_json(&obj) {
            return Some(vec![call]);
        }
    }

    None
}

fn parse_single_call_json(v: &JsonValue) -> Option<ToolCall> {
    // OpenAI format: {"function": {"name": ..., "arguments": ...}}
    if let Some(func) = v.get("function") {
        let name = func.get("name")?.as_str()?.to_string();
        let args_val = func.get("arguments")?;
        let arguments = parse_arguments_value(args_val);
        let id = v
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let id = if id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            id
        };
        return Some(ToolCall {
            id,
            name,
            arguments,
        });
    }

    // Direct format: {"name": ..., "arguments": ...}
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .or_else(|| v.get("tool_name").and_then(|n| n.as_str()))?
        .to_string();
    let args_val = v
        .get("arguments")
        .or_else(|| v.get("args"))
        .or_else(|| v.get("parameters"))?;
    let arguments = parse_arguments_value(args_val);
    let id = v
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let id = if id.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        id
    };
    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

fn parse_arguments_value(v: &JsonValue) -> HashMap<String, JsonValue> {
    match v {
        JsonValue::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        JsonValue::String(s) => {
            // Arguments might be a JSON string
            if let Ok(parsed) = serde_json::from_str::<HashMap<String, JsonValue>>(s) {
                parsed
            } else {
                HashMap::new()
            }
        }
        _ => HashMap::new(),
    }
}

fn try_parse_openai_response(text: &str) -> Option<Vec<ToolCall>> {
    let obj: JsonValue = serde_json::from_str(text).ok()?;

    // {"function_call": {"name": "...", "arguments": "..."}}
    if let Some(fc) = obj.get("function_call") {
        let name = fc.get("name")?.as_str()?.to_string();
        let args_val = fc.get("arguments")?;
        let arguments = parse_arguments_value(args_val);
        return Some(vec![ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            arguments,
        }]);
    }

    // {"choices": [{"message": {"tool_calls": [...]}}]}
    if let Some(choices) = obj.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            if let Some(tool_calls) = choice
                .get("message")
                .and_then(|m| m.get("tool_calls"))
                .and_then(|t| t.as_array())
            {
                let calls: Vec<ToolCall> = tool_calls
                    .iter()
                    .filter_map(|tc| parse_single_call_json(tc))
                    .collect();
                if !calls.is_empty() {
                    return Some(calls);
                }
            }
        }
    }

    None
}

fn try_parse_text_format(text: &str) -> Vec<ToolCall> {
    // Match [TOOL:name(args)] pattern
    let re = regex::Regex::new(r"\[TOOL:(\w+)\((\{.*?\})\)\]").expect("valid regex");
    re.captures_iter(text)
        .filter_map(|cap| {
            let name = cap[1].to_string();
            let args_str = &cap[2];
            let arguments: HashMap<String, JsonValue> =
                serde_json::from_str(args_str).unwrap_or_default();
            Some(ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                arguments,
            })
        })
        .collect()
}

fn try_parse_xml_format(text: &str) -> Vec<ToolCall> {
    // Match <tool name="...">...</tool> or <tool_call>...</tool_call>
    let re = regex::Regex::new(
        r#"<tool(?:_call)?\s+name\s*=\s*"(\w+)"[^>]*>([\s\S]*?)</tool(?:_call)?>"#,
    )
    .expect("valid regex");

    let param_re = regex::Regex::new(r#"<param\s+name\s*=\s*"(\w+)"[^>]*>([^<]*)</param>"#)
        .expect("valid regex");

    re.captures_iter(text)
        .map(|cap| {
            let name = cap[1].to_string();
            let body = &cap[2];
            let mut arguments = HashMap::new();

            for param_cap in param_re.captures_iter(body) {
                let key = param_cap[1].to_string();
                let val = param_cap[2].to_string();
                // Try to parse as JSON, fallback to string
                let json_val =
                    serde_json::from_str(&val).unwrap_or_else(|_| JsonValue::String(val));
                arguments.insert(key, json_val);
            }

            ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                arguments,
            }
        })
        .collect()
}

// ============================================================================
// Provider Plugin System (from tools.rs)
// ============================================================================

/// Capability flags for a provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tool_calling: bool,
    pub vision: bool,
    pub embeddings: bool,
    pub json_mode: bool,
    pub system_prompt: bool,
}

/// Trait for provider plugins that supply model inference.
///
/// Implementations provide model listing, text generation, streaming,
/// tool-augmented generation, and optional embeddings.
pub trait ProviderPlugin: Send + Sync {
    /// Provider name (e.g. "ollama", "openai").
    fn name(&self) -> &str;

    /// Supported capabilities.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Whether the provider is currently reachable.
    fn is_available(&self) -> bool;

    /// List available models.
    fn list_models(&self) -> Result<Vec<String>, String>;

    /// Generate a response (non-streaming).
    fn generate(
        &self,
        model: &str,
        messages: &[JsonValue],
        options: &JsonValue,
    ) -> Result<String, String>;

    /// Generate a streaming response. Default: falls back to non-streaming.
    fn generate_streaming(
        &self,
        model: &str,
        messages: &[JsonValue],
        options: &JsonValue,
    ) -> Result<Box<dyn std::io::Read + Send>, String> {
        let response = self.generate(model, messages, options)?;
        Ok(Box::new(std::io::Cursor::new(response.into_bytes())))
    }

    /// Generate with tools. Default: falls back to regular generate.
    fn generate_with_tools(
        &self,
        model: &str,
        messages: &[JsonValue],
        tools: &[JsonValue],
        options: &JsonValue,
    ) -> Result<String, String> {
        let _ = tools;
        self.generate(model, messages, options)
    }

    /// Generate embeddings. Default: not supported.
    fn generate_embeddings(&self, _model: &str, _texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        Err("embeddings not supported by this provider".to_string())
    }
}

/// Registry for provider plugins.
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn ProviderPlugin>>,
    default_provider: Option<String>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: None,
        }
    }

    /// Register a provider plugin.
    pub fn register(&mut self, provider: Box<dyn ProviderPlugin>) {
        let name = provider.name().to_string();
        self.providers.insert(name, provider);
    }

    /// Unregister a provider.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.providers.remove(name).is_some()
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Option<&dyn ProviderPlugin> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// Get the default provider.
    pub fn get_default(&self) -> Option<&dyn ProviderPlugin> {
        self.default_provider
            .as_ref()
            .and_then(|name| self.get(name))
    }

    /// Set the default provider.
    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_provider = Some(name.into());
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// List only available providers.
    pub fn list_available(&self) -> Vec<&str> {
        self.providers
            .iter()
            .filter(|(_, p)| p.is_available())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Combined capabilities of all registered providers.
    pub fn combined_capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::default();
        for provider in self.providers.values() {
            let pc = provider.capabilities();
            caps.streaming |= pc.streaming;
            caps.tool_calling |= pc.tool_calling;
            caps.vision |= pc.vision;
            caps.embeddings |= pc.embeddings;
            caps.json_mode |= pc.json_mode;
            caps.system_prompt |= pc.system_prompt;
        }
        caps
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Tools
// ============================================================================

/// Create the standard set of built-in tools.
///
/// Includes: `get_current_time`, `calculate`, `string_length`, `validate_json`.
pub fn builtin_tools() -> Vec<(ToolDef, ToolHandler)> {
    vec![
        // get_current_time
        (
            ToolBuilder::new("get_current_time", "Get the current date and time")
                .optional_string("format", "Output format: iso8601, unix, or human")
                .category("system")
                .build(),
            Arc::new(|call: &ToolCall| {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let format = call.get_string("format").unwrap_or("iso8601");
                let content = match format {
                    "unix" => now.as_secs().to_string(),
                    "human" => {
                        let secs = now.as_secs();
                        let days = secs / 86400;
                        let hours = (secs % 86400) / 3600;
                        let mins = (secs % 3600) / 60;
                        let s = secs % 60;
                        format!(
                            "Day {} since epoch, {:02}:{:02}:{:02} UTC",
                            days, hours, mins, s
                        )
                    }
                    _ => {
                        // iso8601-ish
                        let secs = now.as_secs();
                        let days = secs / 86400;
                        let time_secs = secs % 86400;
                        let hours = time_secs / 3600;
                        let mins = (time_secs % 3600) / 60;
                        let s = time_secs % 60;
                        // Simple epoch-based date (not calendar-accurate, but functional)
                        format!("epoch+{}d {:02}:{:02}:{:02}Z", days, hours, mins, s)
                    }
                };
                Ok(ToolOutput::text(content))
            }),
        ),
        // calculate
        (
            ToolBuilder::new("calculate", "Evaluate a mathematical expression")
                .required_string("expression", "Math expression (e.g. '2 + 3 * 4')")
                .category("math")
                .build(),
            Arc::new(|call: &ToolCall| {
                let expr = call
                    .get_string("expression")
                    .ok_or_else(|| ToolError::MissingParameter("expression".to_string()))?;
                match evaluate_math(expr) {
                    Ok(result) => Ok(ToolOutput::with_data(
                        result.to_string(),
                        serde_json::json!({ "result": result }),
                    )),
                    Err(e) => Err(ToolError::ExecutionFailed(e)),
                }
            }),
        ),
        // string_length
        (
            ToolBuilder::new("string_length", "Get the length of a string")
                .required_string("text", "The string to measure")
                .category("text")
                .build(),
            Arc::new(|call: &ToolCall| {
                let text = call
                    .get_string("text")
                    .ok_or_else(|| ToolError::MissingParameter("text".to_string()))?;
                let len = text.len();
                let chars = text.chars().count();
                Ok(ToolOutput::with_data(
                    format!("{} bytes, {} characters", len, chars),
                    serde_json::json!({ "bytes": len, "characters": chars }),
                ))
            }),
        ),
        // validate_json
        (
            ToolBuilder::new("validate_json", "Validate a JSON string")
                .required_string("json", "JSON string to validate")
                .category("text")
                .build(),
            Arc::new(|call: &ToolCall| {
                let json_str = call
                    .get_string("json")
                    .ok_or_else(|| ToolError::MissingParameter("json".to_string()))?;
                match serde_json::from_str::<JsonValue>(json_str) {
                    Ok(_) => Ok(ToolOutput::text("Valid JSON")),
                    Err(e) => Ok(ToolOutput::text(format!("Invalid JSON: {}", e))),
                }
            }),
        ),
    ]
}

/// Simple math expression evaluator supporting +, -, *, /, %, ** and parentheses.
pub fn evaluate_math(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("empty expression".to_string());
    }

    // Tokenize
    let tokens = tokenize_math(expr)?;
    // Parse and evaluate using recursive descent
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected token at position {}", pos));
    }
    Ok(result)
}

#[derive(Debug, Clone)]
enum MathToken {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    LParen,
    RParen,
}

fn tokenize_math(expr: &str) -> Result<Vec<MathToken>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => {
                i += 1;
            }
            '+' => {
                tokens.push(MathToken::Plus);
                i += 1;
            }
            '-' => {
                // Unary minus: if at start, after operator, or after '('
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(
                            MathToken::Plus
                                | MathToken::Minus
                                | MathToken::Star
                                | MathToken::Slash
                                | MathToken::Percent
                                | MathToken::Power
                                | MathToken::LParen
                        )
                    );
                if is_unary {
                    // Parse the number with sign
                    i += 1;
                    let start = i;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    if i == start {
                        // It's just a minus operator with no immediate number
                        tokens.push(MathToken::Number(0.0));
                        tokens.push(MathToken::Minus);
                    } else {
                        let num_str: String = chars[start..i].iter().collect();
                        let num: f64 = num_str
                            .parse()
                            .map_err(|_| format!("invalid number: -{}", num_str))?;
                        tokens.push(MathToken::Number(-num));
                    }
                } else {
                    tokens.push(MathToken::Minus);
                    i += 1;
                }
            }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    tokens.push(MathToken::Power);
                    i += 2;
                } else {
                    tokens.push(MathToken::Star);
                    i += 1;
                }
            }
            '/' => {
                tokens.push(MathToken::Slash);
                i += 1;
            }
            '%' => {
                tokens.push(MathToken::Percent);
                i += 1;
            }
            '(' => {
                tokens.push(MathToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(MathToken::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("invalid number: {}", num_str))?;
                tokens.push(MathToken::Number(num));
            }
            c => {
                return Err(format!("unexpected character: '{}'", c));
            }
        }
    }

    Ok(tokens)
}

// Recursive descent parser: expr -> term ((+|-) term)*
fn parse_expr(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Plus => {
                *pos += 1;
                left += parse_term(tokens, pos)?;
            }
            MathToken::Minus => {
                *pos += 1;
                left -= parse_term(tokens, pos)?;
            }
            _ => break,
        }
    }
    Ok(left)
}

// term -> power ((*|/|%) power)*
fn parse_term(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let mut left = parse_power(tokens, pos)?;
    while *pos < tokens.len() {
        match tokens[*pos] {
            MathToken::Star => {
                *pos += 1;
                left *= parse_power(tokens, pos)?;
            }
            MathToken::Slash => {
                *pos += 1;
                let right = parse_power(tokens, pos)?;
                if right == 0.0 {
                    return Err("division by zero".to_string());
                }
                left /= right;
            }
            MathToken::Percent => {
                *pos += 1;
                let right = parse_power(tokens, pos)?;
                if right == 0.0 {
                    return Err("modulo by zero".to_string());
                }
                left %= right;
            }
            _ => break,
        }
    }
    Ok(left)
}

// power -> atom (** atom)*
fn parse_power(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    let base = parse_atom(tokens, pos)?;
    if *pos < tokens.len() && matches!(tokens[*pos], MathToken::Power) {
        *pos += 1;
        let exp = parse_power(tokens, pos)?; // right-associative
        Ok(base.powf(exp))
    } else {
        Ok(base)
    }
}

// atom -> NUMBER | '(' expr ')'
fn parse_atom(tokens: &[MathToken], pos: &mut usize) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".to_string());
    }
    match &tokens[*pos] {
        MathToken::Number(n) => {
            let val = *n;
            *pos += 1;
            Ok(val)
        }
        MathToken::LParen => {
            *pos += 1;
            let val = parse_expr(tokens, pos)?;
            if *pos >= tokens.len() || !matches!(tokens[*pos], MathToken::RParen) {
                return Err("missing closing parenthesis".to_string());
            }
            *pos += 1;
            Ok(val)
        }
        _ => Err(format!("unexpected token: {:?}", tokens[*pos])),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- ParamSchema ---

    #[test]
    fn test_param_schema_constructors() {
        let s = ParamSchema::string("query", "Search query");
        assert_eq!(s.name, "query");
        assert_eq!(s.param_type, ParamType::String);
        assert!(s.required);

        let n = ParamSchema::number("limit", "Max results").optional();
        assert!(!n.required);

        let e =
            ParamSchema::enum_type("format", "Output format", vec!["json".into(), "xml".into()]);
        assert_eq!(e.enum_values.as_ref().unwrap().len(), 2);

        let r = ParamSchema::integer("count", "Count").with_range(Some(1.0), Some(100.0));
        assert_eq!(r.minimum, Some(1.0));
        assert_eq!(r.maximum, Some(100.0));

        let d = ParamSchema::boolean("verbose", "Verbose").with_default(serde_json::json!(false));
        assert!(!d.required);
        assert_eq!(d.default, Some(serde_json::json!(false)));
    }

    #[test]
    fn test_param_schema_json_schema() {
        let p = ParamSchema::string("name", "User name");
        let schema = p.to_json_schema();
        assert_eq!(schema["type"], "string");
        assert_eq!(schema["description"], "User name");
    }

    #[test]
    fn test_array_param_schema() {
        let items = ParamSchema::string("item", "An item");
        let arr = ParamSchema::array("tags", "List of tags", items);
        assert_eq!(arr.param_type, ParamType::Array);
        assert!(arr.items.is_some());
        let schema = arr.to_json_schema();
        assert_eq!(schema["items"]["type"], "string");
    }

    // --- ToolDef ---

    #[test]
    fn test_tool_def_builder() {
        let tool = ToolBuilder::new("search", "Search the web")
            .required_string("query", "Search query")
            .optional_number("limit", "Max results")
            .required_enum(
                "format",
                "Output format",
                vec!["json".into(), "text".into()],
            )
            .category("search")
            .build();

        assert_eq!(tool.name, "search");
        assert_eq!(tool.parameters.len(), 3);
        assert_eq!(tool.category, Some("search".to_string()));
        assert!(tool.enabled);
    }

    #[test]
    fn test_tool_def_openai_format() {
        let tool = ToolBuilder::new("greet", "Greet someone")
            .required_string("name", "Name to greet")
            .build();

        let func = tool.to_openai_function();
        assert_eq!(func["type"], "function");
        assert_eq!(func["function"]["name"], "greet");
        assert!(func["function"]["parameters"]["properties"]
            .get("name")
            .is_some());
        assert_eq!(func["function"]["parameters"]["required"][0], "name");
    }

    #[test]
    fn test_tool_def_json_schema() {
        let tool = ToolBuilder::new("test", "Test tool")
            .required_string("input", "Input data")
            .build();

        let schema = tool.to_json_schema();
        assert_eq!(schema["name"], "test");
        assert!(schema["parameters"]["properties"]["input"].is_object());
    }

    // --- ToolCall ---

    #[test]
    fn test_tool_call_accessors() {
        let mut args = HashMap::new();
        args.insert("name".to_string(), serde_json::json!("Alice"));
        args.insert("age".to_string(), serde_json::json!(30));
        args.insert("active".to_string(), serde_json::json!(true));

        let call = ToolCall::new("test", args);
        assert_eq!(call.get_string("name"), Some("Alice"));
        assert_eq!(call.get_number("age"), Some(30.0));
        assert_eq!(call.get_integer("age"), Some(30));
        assert_eq!(call.get_bool("active"), Some(true));
        assert!(call.get_string("missing").is_none());
    }

    #[test]
    fn test_tool_call_with_id() {
        let call = ToolCall::with_id("call_123", "test", HashMap::new());
        assert_eq!(call.id, "call_123");
    }

    // --- ToolOutput & ToolError ---

    #[test]
    fn test_tool_output() {
        let out = ToolOutput::text("hello");
        assert_eq!(out.content, "hello");
        assert!(out.data.is_none());

        let out = ToolOutput::with_data("result", serde_json::json!(42));
        assert_eq!(out.content, "result");
        assert_eq!(out.data, Some(serde_json::json!(42)));
    }

    #[test]
    fn test_tool_error_display() {
        assert_eq!(
            ToolError::NotFound("foo".into()).to_string(),
            "tool not found: foo"
        );
        assert_eq!(
            ToolError::MissingParameter("bar".into()).to_string(),
            "missing required parameter: bar"
        );
        assert_eq!(
            ToolError::InvalidParameter {
                name: "x".into(),
                message: "bad".into()
            }
            .to_string(),
            "invalid parameter 'x': bad"
        );
    }

    // --- ToolChoice ---

    #[test]
    fn test_tool_choice_api_value() {
        assert_eq!(ToolChoice::Auto.to_api_value(), "auto");
        assert_eq!(ToolChoice::None.to_api_value(), "none");
        assert_eq!(ToolChoice::Required.to_api_value(), "required");

        let specific = ToolChoice::Specific {
            name: "search".into(),
        };
        let val = specific.to_api_value();
        assert_eq!(val["function"]["name"], "search");
    }

    // --- ToolRegistry ---

    #[test]
    fn test_registry_register_and_execute() {
        let mut reg = ToolRegistry::new();

        let def = ToolBuilder::new("echo", "Echo back input")
            .required_string("text", "Text to echo")
            .build();

        reg.register(
            def,
            Arc::new(|call: &ToolCall| {
                let text = call.get_string("text").unwrap_or("");
                Ok(ToolOutput::text(text.to_string()))
            }),
        );

        assert_eq!(reg.len(), 1);
        assert!(reg.get("echo").is_some());

        let call = ToolCall::new("echo", [("text".into(), serde_json::json!("hello"))].into());
        let result = reg.execute(&call).unwrap();
        assert_eq!(result.content, "hello");
    }

    #[test]
    fn test_registry_not_found() {
        let reg = ToolRegistry::new();
        let call = ToolCall::new("missing", HashMap::new());
        assert!(matches!(reg.execute(&call), Err(ToolError::NotFound(_))));
    }

    #[test]
    fn test_registry_missing_required_param() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("test", "Test")
            .required_string("name", "Required name")
            .build();
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        let call = ToolCall::new("test", HashMap::new());
        assert!(matches!(
            reg.execute(&call),
            Err(ToolError::MissingParameter(_))
        ));
    }

    #[test]
    fn test_registry_enum_validation() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("test", "Test")
            .required_enum("format", "Format", vec!["json".into(), "xml".into()])
            .build();
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        // Valid enum value
        let call = ToolCall::new(
            "test",
            [("format".into(), serde_json::json!("json"))].into(),
        );
        assert!(reg.execute(&call).is_ok());

        // Invalid enum value
        let call = ToolCall::new(
            "test",
            [("format".into(), serde_json::json!("yaml"))].into(),
        );
        assert!(matches!(
            reg.execute(&call),
            Err(ToolError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn test_registry_range_validation() {
        let mut reg = ToolRegistry::new();
        let def = ToolDef::new("test", "Test")
            .with_param(ParamSchema::number("value", "A value").with_range(Some(0.0), Some(100.0)));
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        // In range
        let call = ToolCall::new("test", [("value".into(), serde_json::json!(50.0))].into());
        assert!(reg.execute(&call).is_ok());

        // Below min
        let call = ToolCall::new("test", [("value".into(), serde_json::json!(-1.0))].into());
        assert!(matches!(
            reg.execute(&call),
            Err(ToolError::InvalidParameter { .. })
        ));

        // Above max
        let call = ToolCall::new("test", [("value".into(), serde_json::json!(200.0))].into());
        assert!(matches!(
            reg.execute(&call),
            Err(ToolError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn test_registry_enable_disable() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("test", "Test").build();
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        assert_eq!(reg.list_enabled().len(), 1);
        reg.set_enabled("test", false);
        assert_eq!(reg.list_enabled().len(), 0);

        let call = ToolCall::new("test", HashMap::new());
        assert!(matches!(reg.execute(&call), Err(ToolError::NotFound(_))));

        reg.set_enabled("test", true);
        assert!(reg.execute(&call).is_ok());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("test", "Test").build();
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        assert!(reg.unregister("test"));
        assert!(!reg.unregister("test"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_builtins() {
        let reg = ToolRegistry::with_builtins();
        assert!(reg.get("calculate").is_some());
        assert!(reg.get("get_current_time").is_some());
        assert!(reg.get("string_length").is_some());
        assert!(reg.get("validate_json").is_some());
    }

    #[test]
    fn test_registry_execute_all() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("echo", "Echo")
            .required_string("t", "text")
            .build();
        reg.register(
            def,
            Arc::new(|c: &ToolCall| {
                Ok(ToolOutput::text(
                    c.get_string("t").unwrap_or("").to_string(),
                ))
            }),
        );

        let calls = vec![
            ToolCall::new("echo", [("t".into(), serde_json::json!("a"))].into()),
            ToolCall::new("echo", [("t".into(), serde_json::json!("b"))].into()),
            ToolCall::new("missing", HashMap::new()),
        ];

        let results = reg.execute_all(&calls);
        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_ok());
        assert!(results[2].1.is_err());
    }

    #[test]
    fn test_registry_openai_export() {
        let mut reg = ToolRegistry::new();
        let def = ToolBuilder::new("search", "Search")
            .required_string("q", "query")
            .build();
        reg.register(def, Arc::new(|_| Ok(ToolOutput::text("ok"))));

        let funcs = reg.to_openai_functions();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0]["function"]["name"], "search");
    }

    #[test]
    fn test_format_results_for_context() {
        let call1 = ToolCall::new("search", HashMap::new());
        let call2 = ToolCall::new("calc", HashMap::new());

        let results: Vec<(&ToolCall, Result<ToolOutput, ToolError>)> = vec![
            (&call1, Ok(ToolOutput::text("found 5 results"))),
            (&call2, Err(ToolError::ExecutionFailed("oops".into()))),
        ];

        let ctx = ToolRegistry::format_results_for_context(&results);
        assert!(ctx.contains("[Tool: search]"));
        assert!(ctx.contains("found 5 results"));
        assert!(ctx.contains("[Tool: calc]"));
        assert!(ctx.contains("Error: execution failed: oops"));
    }

    // --- Parsing ---

    #[test]
    fn test_parse_json_array() {
        let json = r#"[{"name": "search", "arguments": {"query": "rust"}}]"#;
        let calls = parse_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].get_string("query"), Some("rust"));
    }

    #[test]
    fn test_parse_openai_format() {
        let json = r#"{"tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": "{\"expression\": \"2+2\"}"}}]}"#;
        let calls = parse_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calc");
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].get_string("expression"), Some("2+2"));
    }

    #[test]
    fn test_parse_function_call_format() {
        let json = r#"{"function_call": {"name": "greet", "arguments": "{\"name\": \"Alice\"}"}}"#;
        let calls = parse_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "greet");
        assert_eq!(calls[0].get_string("name"), Some("Alice"));
    }

    #[test]
    fn test_parse_text_format() {
        let text = r#"I'll search for that. [TOOL:search({"query": "rust lang"})]"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].get_string("query"), Some("rust lang"));
    }

    #[test]
    fn test_parse_xml_format() {
        let text = r#"<tool name="search"><param name="query">rust</param><param name="limit">5</param></tool>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].get_string("query"), Some("rust"));
    }

    #[test]
    fn test_parse_empty() {
        assert!(parse_tool_calls("").is_empty());
        assert!(parse_tool_calls("Hello world").is_empty());
    }

    #[test]
    fn test_parse_choices_format() {
        let json = r#"{"choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "test", "arguments": "{}"}}]}}]}"#;
        let calls = parse_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test");
    }

    // --- Math evaluator ---

    #[test]
    fn test_evaluate_math_basic() {
        assert_eq!(evaluate_math("2 + 3").unwrap(), 5.0);
        assert_eq!(evaluate_math("10 - 4").unwrap(), 6.0);
        assert_eq!(evaluate_math("3 * 4").unwrap(), 12.0);
        assert_eq!(evaluate_math("15 / 3").unwrap(), 5.0);
        assert_eq!(evaluate_math("7 % 3").unwrap(), 1.0);
    }

    #[test]
    fn test_evaluate_math_precedence() {
        assert_eq!(evaluate_math("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(evaluate_math("(2 + 3) * 4").unwrap(), 20.0);
    }

    #[test]
    fn test_evaluate_math_power() {
        assert_eq!(evaluate_math("2 ** 3").unwrap(), 8.0);
        assert_eq!(evaluate_math("2 ** 3 ** 2").unwrap(), 512.0); // right-associative
    }

    #[test]
    fn test_evaluate_math_negative() {
        assert_eq!(evaluate_math("-5 + 3").unwrap(), -2.0);
        assert_eq!(evaluate_math("(-5) * 2").unwrap(), -10.0);
    }

    #[test]
    fn test_evaluate_math_errors() {
        assert!(evaluate_math("").is_err());
        assert!(evaluate_math("1 / 0").is_err());
        assert!(evaluate_math("1 % 0").is_err());
        assert!(evaluate_math("abc").is_err());
    }

    // --- Builtins ---

    #[test]
    fn test_builtin_calculate() {
        let reg = ToolRegistry::with_builtins();
        let call = ToolCall::new(
            "calculate",
            [("expression".into(), serde_json::json!("(2 + 3) * 4"))].into(),
        );
        let result = reg.execute(&call).unwrap();
        assert_eq!(result.content, "20");
    }

    #[test]
    fn test_builtin_string_length() {
        let reg = ToolRegistry::with_builtins();
        let call = ToolCall::new(
            "string_length",
            [("text".into(), serde_json::json!("hello"))].into(),
        );
        let result = reg.execute(&call).unwrap();
        assert!(result.content.contains("5 bytes"));
        assert!(result.content.contains("5 characters"));
    }

    #[test]
    fn test_builtin_validate_json() {
        let reg = ToolRegistry::with_builtins();

        let call = ToolCall::new(
            "validate_json",
            [("json".into(), serde_json::json!("{\"a\": 1}"))].into(),
        );
        let result = reg.execute(&call).unwrap();
        assert_eq!(result.content, "Valid JSON");

        let call = ToolCall::new(
            "validate_json",
            [("json".into(), serde_json::json!("{bad}"))].into(),
        );
        let result = reg.execute(&call).unwrap();
        assert!(result.content.starts_with("Invalid JSON"));
    }

    #[test]
    fn test_builtin_get_current_time() {
        let reg = ToolRegistry::with_builtins();

        let call = ToolCall::new(
            "get_current_time",
            [("format".into(), serde_json::json!("unix"))].into(),
        );
        let result = reg.execute(&call).unwrap();
        let secs: u64 = result.content.parse().unwrap();
        assert!(secs > 1_700_000_000); // after 2023

        let call = ToolCall::new("get_current_time", HashMap::new());
        let result = reg.execute(&call).unwrap();
        assert!(result.content.contains("epoch+"));
    }

    // --- ProviderRegistry ---

    #[test]
    fn test_provider_registry() {
        struct TestProvider;
        impl ProviderPlugin for TestProvider {
            fn name(&self) -> &str {
                "test"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities {
                    streaming: true,
                    tool_calling: true,
                    ..Default::default()
                }
            }
            fn is_available(&self) -> bool {
                true
            }
            fn list_models(&self) -> Result<Vec<String>, String> {
                Ok(vec!["model-1".into()])
            }
            fn generate(
                &self,
                _model: &str,
                _messages: &[JsonValue],
                _options: &JsonValue,
            ) -> Result<String, String> {
                Ok("response".into())
            }
        }

        let mut reg = ProviderRegistry::new();
        reg.register(Box::new(TestProvider));
        reg.set_default("test");

        assert_eq!(reg.list(), vec!["test"]);
        assert_eq!(reg.list_available(), vec!["test"]);
        assert!(reg.get("test").is_some());
        assert!(reg.get_default().is_some());

        let caps = reg.combined_capabilities();
        assert!(caps.streaming);
        assert!(caps.tool_calling);
        assert!(!caps.vision);

        assert!(reg.unregister("test"));
        assert!(reg.list().is_empty());
    }
}
