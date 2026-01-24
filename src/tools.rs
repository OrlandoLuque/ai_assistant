//! Tool calling support and provider plugins

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Tool Definitions
// ============================================================================

/// Type of parameter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParameterType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

/// A parameter for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: ParameterType,
    /// Description of the parameter
    pub description: String,
    /// Is this parameter required
    #[serde(default)]
    pub required: bool,
    /// Default value if not provided
    pub default: Option<Value>,
    /// Enum of allowed values
    #[serde(rename = "enum")]
    pub enum_values: Option<Vec<String>>,
}

/// Definition of a tool that the AI can call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (unique identifier)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Parameters the tool accepts
    pub parameters: Vec<ToolParameter>,
    /// Category for organization
    pub category: Option<String>,
    /// Whether this tool is currently enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: Vec::new(),
            category: None,
            enabled: true,
        }
    }

    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }

    /// Convert to OpenAI-compatible function schema
    pub fn to_openai_function(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert("type".to_string(), Value::String(format!("{:?}", param.param_type).to_lowercase()));
            prop.insert("description".to_string(), Value::String(param.description.clone()));

            if let Some(ref enums) = param.enum_values {
                prop.insert("enum".to_string(), Value::Array(
                    enums.iter().map(|e| Value::String(e.clone())).collect()
                ));
            }

            properties.insert(param.name.clone(), Value::Object(prop));

            if param.required {
                required.push(Value::String(param.name.clone()));
            }
        }

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    }
}

/// A call to a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name
    pub name: String,
    /// Arguments provided
    pub arguments: HashMap<String, Value>,
    /// Call ID (for tracking)
    pub id: String,
}

impl ToolCall {
    pub fn new(name: &str, arguments: HashMap<String, Value>) -> Self {
        Self {
            name: name.to_string(),
            arguments,
            id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Get a string argument
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.arguments.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// Get a number argument
    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.arguments.get(key).and_then(|v| v.as_f64())
    }

    /// Get an integer argument
    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.arguments.get(key).and_then(|v| v.as_i64())
    }

    /// Get a boolean argument
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key).and_then(|v| v.as_bool())
    }
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID
    pub call_id: String,
    /// Tool name
    pub tool_name: String,
    /// Was execution successful
    pub success: bool,
    /// Result content (for display to AI)
    pub content: String,
    /// Structured data (optional)
    pub data: Option<Value>,
    /// Error message if failed
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(call_id: &str, tool_name: &str, content: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            success: true,
            content: content.to_string(),
            data: None,
            error: None,
        }
    }

    pub fn success_with_data(call_id: &str, tool_name: &str, content: &str, data: Value) -> Self {
        Self {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            success: true,
            content: content.to_string(),
            data: Some(data),
            error: None,
        }
    }

    pub fn error(call_id: &str, tool_name: &str, error: &str) -> Self {
        Self {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            success: false,
            content: format!("Error: {}", error),
            data: None,
            error: Some(error.to_string()),
        }
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Handler function type
pub type ToolHandler = Box<dyn Fn(&ToolCall) -> ToolResult + Send + Sync>;

/// Registry for tools and their handlers
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
    handlers: HashMap<String, ToolHandler>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool with its handler
    pub fn register<F>(&mut self, tool: ToolDefinition, handler: F)
    where
        F: Fn(&ToolCall) -> ToolResult + Send + Sync + 'static,
    {
        let name = tool.name.clone();
        self.tools.insert(name.clone(), tool);
        self.handlers.insert(name, Box::new(handler));
    }

    /// Register a tool definition only (handler registered separately)
    pub fn register_tool(&mut self, tool: ToolDefinition) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register a handler for an existing tool
    pub fn register_handler<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&ToolCall) -> ToolResult + Send + Sync + 'static,
    {
        self.handlers.insert(name.to_string(), Box::new(handler));
    }

    /// Unregister a tool
    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
        self.handlers.remove(name);
    }

    /// Get a tool definition
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Get all tool definitions
    pub fn get_tools(&self) -> Vec<&ToolDefinition> {
        self.tools.values().collect()
    }

    /// Get enabled tools only
    pub fn get_enabled_tools(&self) -> Vec<&ToolDefinition> {
        self.tools.values().filter(|t| t.enabled).collect()
    }

    /// Enable/disable a tool
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(tool) = self.tools.get_mut(name) {
            tool.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Execute a tool call
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        // Check if tool exists
        let tool = match self.tools.get(&call.name) {
            Some(t) => t,
            None => return ToolResult::error(&call.id, &call.name, "Tool not found"),
        };

        // Check if enabled
        if !tool.enabled {
            return ToolResult::error(&call.id, &call.name, "Tool is disabled");
        }

        // Get handler
        let handler = match self.handlers.get(&call.name) {
            Some(h) => h,
            None => return ToolResult::error(&call.id, &call.name, "No handler registered"),
        };

        // Execute
        handler(call)
    }

    /// Get tools as OpenAI-compatible function list
    pub fn to_openai_functions(&self) -> Vec<Value> {
        self.get_enabled_tools()
            .iter()
            .map(|t| t.to_openai_function())
            .collect()
    }

    /// Parse tool calls from OpenAI-style response
    pub fn parse_tool_calls(response: &Value) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        if let Some(tool_calls) = response.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc.get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(function) = tc.get("function") {
                    let name = function.get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();

                    let arguments: HashMap<String, Value> = function.get("arguments")
                        .and_then(|a| a.as_str())
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or_default();

                    calls.push(ToolCall {
                        name,
                        arguments,
                        id,
                    });
                }
            }
        }

        calls
    }
}

// ============================================================================
// Provider Plugin System
// ============================================================================

/// Capability flags for providers
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tool_calling: bool,
    pub vision: bool,
    pub embeddings: bool,
    pub json_mode: bool,
    pub system_prompt: bool,
}

/// A provider plugin interface
pub trait ProviderPlugin: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities;

    /// Check if provider is available/connected
    fn is_available(&self) -> bool;

    /// List available models
    fn list_models(&self) -> anyhow::Result<Vec<crate::models::ModelInfo>>;

    /// Generate a response (non-streaming)
    fn generate(
        &self,
        config: &crate::config::AiConfig,
        messages: &[crate::messages::ChatMessage],
        system_prompt: &str,
    ) -> anyhow::Result<String>;

    /// Generate a streaming response
    fn generate_streaming(
        &self,
        config: &crate::config::AiConfig,
        messages: &[crate::messages::ChatMessage],
        system_prompt: &str,
        tx: &std::sync::mpsc::Sender<crate::messages::AiResponse>,
    ) -> anyhow::Result<()>;

    /// Generate with tools
    fn generate_with_tools(
        &self,
        config: &crate::config::AiConfig,
        messages: &[crate::messages::ChatMessage],
        system_prompt: &str,
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<(String, Vec<ToolCall>)> {
        // Default implementation ignores tools
        let response = self.generate(config, messages, system_prompt)?;
        Ok((response, Vec::new()))
    }

    /// Generate embeddings for text
    fn generate_embeddings(&self, _texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::bail!("Embeddings not supported by this provider")
    }
}

/// Registry for provider plugins
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn ProviderPlugin>>,
    default_provider: Option<String>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: None,
        }
    }

    /// Register a provider plugin
    pub fn register(&mut self, provider: Box<dyn ProviderPlugin>) {
        let name = provider.name().to_string();
        if self.default_provider.is_none() {
            self.default_provider = Some(name.clone());
        }
        self.providers.insert(name, provider);
    }

    /// Unregister a provider
    pub fn unregister(&mut self, name: &str) {
        self.providers.remove(name);
        if self.default_provider.as_deref() == Some(name) {
            self.default_provider = self.providers.keys().next().cloned();
        }
    }

    /// Get a provider by name
    pub fn get(&self, name: &str) -> Option<&dyn ProviderPlugin> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// Get the default provider
    pub fn get_default(&self) -> Option<&dyn ProviderPlugin> {
        self.default_provider.as_ref()
            .and_then(|name| self.get(name))
    }

    /// Set the default provider
    pub fn set_default(&mut self, name: &str) -> bool {
        if self.providers.contains_key(name) {
            self.default_provider = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// List all registered providers
    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// List available providers (that respond to health check)
    pub fn list_available(&self) -> Vec<&str> {
        self.providers.iter()
            .filter(|(_, p)| p.is_available())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// Get combined capabilities from all providers
    pub fn combined_capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::default();

        for provider in self.providers.values() {
            let p_caps = provider.capabilities();
            caps.streaming |= p_caps.streaming;
            caps.tool_calling |= p_caps.tool_calling;
            caps.vision |= p_caps.vision;
            caps.embeddings |= p_caps.embeddings;
            caps.json_mode |= p_caps.json_mode;
            caps.system_prompt |= p_caps.system_prompt;
        }

        caps
    }
}

// ============================================================================
// Built-in Tools
// ============================================================================

/// Create common built-in tools
pub fn create_builtin_tools() -> Vec<(ToolDefinition, ToolHandler)> {
    vec![
        // Calculator tool
        (
            ToolDefinition::new("calculator", "Perform mathematical calculations")
                .with_parameter(ToolParameter {
                    name: "expression".to_string(),
                    param_type: ParameterType::String,
                    description: "Mathematical expression to evaluate".to_string(),
                    required: true,
                    default: None,
                    enum_values: None,
                })
                .with_category("utility"),
            Box::new(|call: &ToolCall| {
                let expr = call.get_string("expression").unwrap_or_default();
                // Simple evaluation (in production, use a proper math parser)
                match simple_eval(&expr) {
                    Ok(result) => ToolResult::success(&call.id, &call.name, &format!("{}", result)),
                    Err(e) => ToolResult::error(&call.id, &call.name, &e),
                }
            }),
        ),
        // Current time tool
        (
            ToolDefinition::new("get_current_time", "Get the current date and time")
                .with_parameter(ToolParameter {
                    name: "timezone".to_string(),
                    param_type: ParameterType::String,
                    description: "Timezone (e.g., 'UTC', 'local')".to_string(),
                    required: false,
                    default: Some(Value::String("local".to_string())),
                    enum_values: Some(vec!["UTC".to_string(), "local".to_string()]),
                })
                .with_category("utility"),
            Box::new(|call: &ToolCall| {
                let tz = call.get_string("timezone").unwrap_or_else(|| "local".to_string());
                let time = if tz.to_lowercase() == "utc" {
                    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string()
                } else {
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string()
                };
                ToolResult::success(&call.id, &call.name, &time)
            }),
        ),
    ]
}

/// Simple expression evaluator (basic implementation)
fn simple_eval(expr: &str) -> Result<f64, String> {
    // Very basic: only handles simple arithmetic
    // In production, use a library like `meval`
    let expr = expr.replace(' ', "");

    // Try to parse as number
    if let Ok(n) = expr.parse::<f64>() {
        return Ok(n);
    }

    // Simple operations
    for op in ['+', '-', '*', '/'] {
        if let Some(pos) = expr.rfind(op) {
            if pos > 0 {
                let left = simple_eval(&expr[..pos])?;
                let right = simple_eval(&expr[pos + 1..])?;
                return match op {
                    '+' => Ok(left + right),
                    '-' => Ok(left - right),
                    '*' => Ok(left * right),
                    '/' => {
                        if right == 0.0 {
                            Err("Division by zero".to_string())
                        } else {
                            Ok(left / right)
                        }
                    }
                    _ => Err("Unknown operator".to_string()),
                };
            }
        }
    }

    Err(format!("Cannot evaluate: {}", expr))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new("test_tool", "A test tool")
            .with_parameter(ToolParameter {
                name: "input".to_string(),
                param_type: ParameterType::String,
                description: "Input text".to_string(),
                required: true,
                default: None,
                enum_values: None,
            });

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.parameters.len(), 1);
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();

        let tool = ToolDefinition::new("echo", "Echo back the input");
        registry.register(tool, |call| {
            ToolResult::success(&call.id, &call.name, "echoed")
        });

        assert!(registry.get_tool("echo").is_some());

        let call = ToolCall::new("echo", HashMap::new());
        let result = registry.execute(&call);
        assert!(result.success);
    }

    #[test]
    fn test_simple_eval() {
        assert_eq!(simple_eval("2+3").unwrap(), 5.0);
        assert_eq!(simple_eval("10-4").unwrap(), 6.0);
        assert_eq!(simple_eval("3*4").unwrap(), 12.0);
        assert_eq!(simple_eval("15/3").unwrap(), 5.0);
    }

    #[test]
    fn test_builtin_tools() {
        let tools = create_builtin_tools();
        assert!(!tools.is_empty());

        // Test calculator
        let (calc_def, calc_handler) = &tools[0];
        assert_eq!(calc_def.name, "calculator");

        let mut args = HashMap::new();
        args.insert("expression".to_string(), Value::String("2+2".to_string()));
        let call = ToolCall { name: "calculator".to_string(), arguments: args, id: "test".to_string() };
        let result = calc_handler(&call);
        assert!(result.success);
        assert_eq!(result.content, "4");
    }
}
