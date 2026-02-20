//! Tool calling support and provider plugins

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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
            prop.insert(
                "type".to_string(),
                Value::String(format!("{:?}", param.param_type).to_lowercase()),
            );
            prop.insert(
                "description".to_string(),
                Value::String(param.description.clone()),
            );

            if let Some(ref enums) = param.enum_values {
                prop.insert(
                    "enum".to_string(),
                    Value::Array(enums.iter().map(|e| Value::String(e.clone())).collect()),
                );
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
        self.arguments
            .get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
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
                let id = tc
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(function) = tc.get("function") {
                    let name = function
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();

                    let arguments: HashMap<String, Value> = function
                        .get("arguments")
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
        self.default_provider
            .as_ref()
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
        self.providers
            .iter()
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
                let tz = call
                    .get_string("timezone")
                    .unwrap_or_else(|| "local".to_string());
                let time = if tz.to_lowercase() == "utc" {
                    chrono::Utc::now()
                        .format("%Y-%m-%d %H:%M:%S UTC")
                        .to_string()
                } else {
                    chrono::Local::now()
                        .format("%Y-%m-%d %H:%M:%S %Z")
                        .to_string()
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
// Advanced Tool Calling — Chaining, Validation, Retry, Approval
// ============================================================================

/// Where an argument value comes from in a tool chain step
#[derive(Debug, Clone)]
pub enum ArgumentSource {
    /// Hardcoded value
    Literal(Value),
    /// Extract field from previous step's output
    FromPrevious(String),
    /// Extract field from initial chain input
    FromInput(String),
}

/// Transformation to apply to a step's output before passing to next step
#[derive(Debug, Clone)]
pub enum OutputTransform {
    /// Pass output as-is
    Identity,
    /// Extract a specific JSON field
    ExtractField(String),
    /// Format string with {result} placeholder
    Template(String),
}

/// A single step in a tool chain
#[derive(Debug, Clone)]
pub struct ToolChainStep {
    pub tool_name: String,
    pub argument_mapping: HashMap<String, ArgumentSource>,
    pub transform: Option<OutputTransform>,
}

impl ToolChainStep {
    pub fn new(tool_name: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            argument_mapping: HashMap::new(),
            transform: None,
        }
    }

    pub fn with_arg(mut self, name: &str, source: ArgumentSource) -> Self {
        self.argument_mapping.insert(name.to_string(), source);
        self
    }

    pub fn with_transform(mut self, transform: OutputTransform) -> Self {
        self.transform = Some(transform);
        self
    }
}

/// Result of a single tool chain step execution
#[derive(Debug, Clone)]
pub struct ToolStepResult {
    pub step_index: usize,
    pub tool_name: String,
    pub input: Value,
    pub output: ToolResult,
    pub duration_ms: u64,
}

/// Result of executing a full tool chain
#[derive(Debug, Clone)]
pub struct ToolChainResult {
    pub steps: Vec<ToolStepResult>,
    pub final_output: Option<Value>,
    pub success: bool,
    pub total_duration_ms: u64,
}

/// A chain of tool calls that execute sequentially, passing data between steps
pub struct ToolChain {
    steps: Vec<ToolChainStep>,
    pub name: String,
    pub description: String,
}

impl ToolChain {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            steps: Vec::new(),
            name: name.to_string(),
            description: description.to_string(),
        }
    }

    pub fn add_step(&mut self, step: ToolChainStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Validate that all tools referenced in steps exist in the registry
    pub fn validate(&self, registry: &ToolRegistry) -> Result<(), String> {
        for step in &self.steps {
            if registry.get_tool(&step.tool_name).is_none() {
                return Err(format!("Tool '{}' not found in registry", step.tool_name));
            }
        }
        Ok(())
    }

    /// Execute the chain, passing data between steps
    pub fn execute(&self, registry: &ToolRegistry, initial_input: Value) -> ToolChainResult {
        let chain_start = std::time::Instant::now();
        let mut step_results: Vec<ToolStepResult> = Vec::new();
        let mut last_output_data: Option<Value> = None;

        for (idx, step) in self.steps.iter().enumerate() {
            let step_start = std::time::Instant::now();

            // Resolve arguments
            let mut resolved_args: HashMap<String, Value> = HashMap::new();
            for (arg_name, source) in &step.argument_mapping {
                let value = match source {
                    ArgumentSource::Literal(v) => v.clone(),
                    ArgumentSource::FromInput(field) => {
                        initial_input.get(field).cloned().unwrap_or(Value::Null)
                    }
                    ArgumentSource::FromPrevious(field) => last_output_data
                        .as_ref()
                        .and_then(|d| d.get(field).cloned())
                        .unwrap_or(Value::Null),
                };
                resolved_args.insert(arg_name.clone(), value);
            }

            let call = ToolCall {
                name: step.tool_name.clone(),
                arguments: resolved_args.clone(),
                id: format!("chain-{}-step-{}", self.name, idx),
            };

            let input_value = serde_json::to_value(&resolved_args).unwrap_or(Value::Null);
            let result = registry.execute(&call);
            let duration_ms = step_start.elapsed().as_millis() as u64;

            let step_failed = !result.success;

            // Apply output transform
            let transformed_data = match &step.transform {
                Some(OutputTransform::Identity) | None => result.data.clone(),
                Some(OutputTransform::ExtractField(field)) => {
                    result.data.as_ref().and_then(|d| d.get(field).cloned())
                }
                Some(OutputTransform::Template(template)) => {
                    let content = &result.content;
                    Some(Value::String(template.replace("{result}", content)))
                }
            };

            last_output_data = transformed_data.clone().or_else(|| result.data.clone());

            step_results.push(ToolStepResult {
                step_index: idx,
                tool_name: step.tool_name.clone(),
                input: input_value,
                output: result,
                duration_ms,
            });

            if step_failed {
                return ToolChainResult {
                    steps: step_results,
                    final_output: None,
                    success: false,
                    total_duration_ms: chain_start.elapsed().as_millis() as u64,
                };
            }
        }

        let final_output = last_output_data;
        ToolChainResult {
            steps: step_results,
            final_output,
            success: true,
            total_duration_ms: chain_start.elapsed().as_millis() as u64,
        }
    }
}

/// Type of validation to apply to a tool argument
#[derive(Debug, Clone)]
pub enum ValidationType {
    /// Field must exist and not be null
    Required,
    /// String value length must be <= max
    MaxLength(usize),
    /// Number must be >= min
    MinValue(f64),
    /// Number must be <= max
    MaxValue(f64),
    /// String must contain this substring (simple match, no regex)
    Pattern(String),
    /// String must be one of these values
    OneOf(Vec<String>),
}

/// A validation rule for a specific field
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub field: String,
    pub rule_type: ValidationType,
}

/// Validates tool arguments against a set of rules
pub struct ToolValidator {
    rules: Vec<ValidationRule>,
}

impl ToolValidator {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, field: &str, rule_type: ValidationType) -> &mut Self {
        self.rules.push(ValidationRule {
            field: field.to_string(),
            rule_type,
        });
        self
    }

    /// Validate arguments against all rules, collecting all violations
    pub fn validate(&self, args: &HashMap<String, Value>) -> Result<(), Vec<String>> {
        let mut violations = Vec::new();

        for rule in &self.rules {
            let value = args.get(&rule.field);

            match &rule.rule_type {
                ValidationType::Required => match value {
                    None => violations.push(format!("Field '{}' is required", rule.field)),
                    Some(Value::Null) => {
                        violations.push(format!("Field '{}' must not be null", rule.field))
                    }
                    _ => {}
                },
                ValidationType::MaxLength(max) => {
                    if let Some(Value::String(s)) = value {
                        if s.len() > *max {
                            violations.push(format!(
                                "Field '{}' exceeds max length {} (got {})",
                                rule.field,
                                max,
                                s.len()
                            ));
                        }
                    }
                }
                ValidationType::MinValue(min) => {
                    if let Some(v) = value.and_then(|v| v.as_f64()) {
                        if v < *min {
                            violations.push(format!(
                                "Field '{}' is below minimum {} (got {})",
                                rule.field, min, v
                            ));
                        }
                    }
                }
                ValidationType::MaxValue(max) => {
                    if let Some(v) = value.and_then(|v| v.as_f64()) {
                        if v > *max {
                            violations.push(format!(
                                "Field '{}' exceeds maximum {} (got {})",
                                rule.field, max, v
                            ));
                        }
                    }
                }
                ValidationType::Pattern(pattern) => {
                    if let Some(Value::String(s)) = value {
                        if !s.contains(pattern.as_str()) {
                            violations.push(format!(
                                "Field '{}' does not match pattern '{}'",
                                rule.field, pattern
                            ));
                        }
                    }
                }
                ValidationType::OneOf(options) => {
                    if let Some(Value::String(s)) = value {
                        if !options.contains(s) {
                            violations.push(format!(
                                "Field '{}' must be one of {:?} (got '{}')",
                                rule.field, options, s
                            ));
                        }
                    }
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

impl Default for ToolValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for retrying failed tool calls
#[derive(Debug, Clone)]
pub struct ToolRetryConfig {
    pub max_retries: u32,
    pub backoff_base_ms: u64,
    pub backoff_multiplier: f64,
    pub retry_on_error: bool,
}

impl Default for ToolRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_base_ms: 100,
            backoff_multiplier: 2.0,
            retry_on_error: true,
        }
    }
}

/// A tool call wrapped with retry logic
pub struct RetryableToolCall {
    pub tool_call: ToolCall,
    pub config: ToolRetryConfig,
}

impl RetryableToolCall {
    pub fn new(tool_call: ToolCall, config: ToolRetryConfig) -> Self {
        Self { tool_call, config }
    }

    /// Execute the tool call with retry logic
    pub fn execute_with_retry(&self, registry: &ToolRegistry) -> ToolResult {
        let mut last_result = registry.execute(&self.tool_call);

        if last_result.success || !self.config.retry_on_error {
            return last_result;
        }

        for attempt in 0..self.config.max_retries {
            let backoff_ms = (self.config.backoff_base_ms as f64
                * self.config.backoff_multiplier.powi(attempt as i32))
                as u64;
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));

            last_result = registry.execute(&self.tool_call);
            if last_result.success {
                return last_result;
            }
        }

        last_result
    }
}

/// Status of an approval request
#[derive(Debug, Clone)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Denied(String),
}

/// A tool call pending approval
#[derive(Debug, Clone)]
pub struct PendingApproval {
    pub id: String,
    pub tool_call: ToolCall,
    pub requested_at_ms: u64,
    pub status: ApprovalStatus,
}

/// Gate that requires approval before tool execution
pub struct ApprovalGate {
    pending: Vec<PendingApproval>,
    auto_approve_tools: Vec<String>,
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            auto_approve_tools: Vec::new(),
        }
    }

    /// Add a tool name to the auto-approve list
    pub fn auto_approve(&mut self, tool_name: &str) -> &mut Self {
        self.auto_approve_tools.push(tool_name.to_string());
        self
    }

    /// Request approval for a tool call. Returns the approval id.
    pub fn request_approval(&mut self, call: ToolCall) -> String {
        let id = format!("approval-{}", self.pending.len());
        let status = if self.auto_approve_tools.contains(&call.name) {
            ApprovalStatus::Approved
        } else {
            ApprovalStatus::Pending
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.pending.push(PendingApproval {
            id: id.clone(),
            tool_call: call,
            requested_at_ms: now,
            status,
        });

        id
    }

    /// Approve a pending request by id. Returns true if found.
    pub fn approve(&mut self, id: &str) -> bool {
        if let Some(entry) = self.pending.iter_mut().find(|p| p.id == id) {
            entry.status = ApprovalStatus::Approved;
            true
        } else {
            false
        }
    }

    /// Deny a pending request by id. Returns true if found.
    pub fn deny(&mut self, id: &str, reason: &str) -> bool {
        if let Some(entry) = self.pending.iter_mut().find(|p| p.id == id) {
            entry.status = ApprovalStatus::Denied(reason.to_string());
            true
        } else {
            false
        }
    }

    /// Count items with Pending status
    pub fn pending_count(&self) -> usize {
        self.pending
            .iter()
            .filter(|p| matches!(p.status, ApprovalStatus::Pending))
            .count()
    }

    /// Return tool calls that have been approved
    pub fn get_approved(&self) -> Vec<&ToolCall> {
        self.pending
            .iter()
            .filter(|p| matches!(p.status, ApprovalStatus::Approved))
            .map(|p| &p.tool_call)
            .collect()
    }
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new("test_tool", "A test tool").with_parameter(ToolParameter {
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
        let call = ToolCall {
            name: "calculator".to_string(),
            arguments: args,
            id: "test".to_string(),
        };
        let result = calc_handler(&call);
        assert!(result.success);
        assert_eq!(result.content, "4");
    }

    // ====================================================================
    // Advanced Tool Calling Tests
    // ====================================================================

    /// Helper: create a registry with "add" and "format" tools for chain tests
    fn create_chain_test_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();

        // "add" tool: adds two numbers, returns result in data
        let add_tool = ToolDefinition::new("add", "Add two numbers");
        registry.register(add_tool, |call| {
            let a = call.get_number("a").unwrap_or(0.0);
            let b = call.get_number("b").unwrap_or(0.0);
            let sum = a + b;
            ToolResult::success_with_data(
                &call.id,
                &call.name,
                &format!("{}", sum),
                serde_json::json!({ "result": sum }),
            )
        });

        // "format" tool: formats a number as a string
        let fmt_tool = ToolDefinition::new("format", "Format a value");
        registry.register(fmt_tool, |call| {
            let value = call.get_number("value").unwrap_or(0.0);
            let formatted = format!("Result is: {}", value);
            ToolResult::success_with_data(
                &call.id,
                &call.name,
                &formatted,
                serde_json::json!({ "formatted": formatted }),
            )
        });

        registry
    }

    #[test]
    fn test_tool_chain_validate() {
        let registry = create_chain_test_registry();

        let mut chain = ToolChain::new("test_chain", "A test chain");
        chain.add_step(ToolChainStep::new("add"));
        chain.add_step(ToolChainStep::new("format"));

        assert_eq!(chain.step_count(), 2);
        assert!(chain.validate(&registry).is_ok());
    }

    #[test]
    fn test_tool_chain_validate_missing() {
        let registry = create_chain_test_registry();

        let mut chain = ToolChain::new("bad_chain", "Chain with missing tool");
        chain.add_step(ToolChainStep::new("add"));
        chain.add_step(ToolChainStep::new("nonexistent_tool"));

        let result = chain.validate(&registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nonexistent_tool"));
    }

    #[test]
    fn test_tool_chain_execute_two_steps() {
        let registry = create_chain_test_registry();

        let step1 = ToolChainStep::new("add")
            .with_arg("a", ArgumentSource::FromInput("x".to_string()))
            .with_arg("b", ArgumentSource::FromInput("y".to_string()));

        let step2 = ToolChainStep::new("format")
            .with_arg("value", ArgumentSource::FromPrevious("result".to_string()));

        let mut chain = ToolChain::new("add_and_format", "Add then format");
        chain.add_step(step1);
        chain.add_step(step2);

        let input = serde_json::json!({ "x": 3.0, "y": 7.0 });
        let result = chain.execute(&registry, input);

        assert!(result.success);
        assert_eq!(result.steps.len(), 2);
        assert!(result.steps[0].output.success);
        assert!(result.steps[1].output.success);
        assert!(result.steps[1].output.content.contains("10"));
    }

    #[test]
    fn test_tool_chain_from_previous() {
        let registry = create_chain_test_registry();

        // Step 1: add 5 + 10 = 15 (output data has "result": 15)
        let step1 = ToolChainStep::new("add")
            .with_arg("a", ArgumentSource::Literal(serde_json::json!(5.0)))
            .with_arg("b", ArgumentSource::Literal(serde_json::json!(10.0)));

        // Step 2: add previous result (15) + 20 = 35
        let step2 = ToolChainStep::new("add")
            .with_arg("a", ArgumentSource::FromPrevious("result".to_string()))
            .with_arg("b", ArgumentSource::Literal(serde_json::json!(20.0)));

        let mut chain = ToolChain::new("chained_add", "Two additions");
        chain.add_step(step1);
        chain.add_step(step2);

        let result = chain.execute(&registry, serde_json::json!({}));

        assert!(result.success);
        assert_eq!(result.steps.len(), 2);
        // Second step output should show 35
        assert!(result.steps[1].output.content.contains("35"));
        // Final output should have result: 35
        let final_out = result.final_output.unwrap();
        assert_eq!(final_out.get("result").and_then(|v| v.as_f64()), Some(35.0));
    }

    #[test]
    fn test_validator_required_field() {
        let mut validator = ToolValidator::new();
        validator.add_rule("name", ValidationType::Required);

        // Present and non-null: should pass
        let mut args = HashMap::new();
        args.insert("name".to_string(), Value::String("Alice".to_string()));
        assert!(validator.validate(&args).is_ok());

        // Missing: should fail
        let empty_args: HashMap<String, Value> = HashMap::new();
        let err = validator.validate(&empty_args).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].contains("required"));

        // Null: should fail
        let mut null_args = HashMap::new();
        null_args.insert("name".to_string(), Value::Null);
        let err = validator.validate(&null_args).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].contains("null"));
    }

    #[test]
    fn test_validator_max_length() {
        let mut validator = ToolValidator::new();
        validator.add_rule("username", ValidationType::MaxLength(10));

        // Within limit
        let mut args = HashMap::new();
        args.insert("username".to_string(), Value::String("alice".to_string()));
        assert!(validator.validate(&args).is_ok());

        // Exceeds limit
        let mut args2 = HashMap::new();
        args2.insert(
            "username".to_string(),
            Value::String("a]very_long_username".to_string()),
        );
        let err = validator.validate(&args2).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].contains("max length"));
    }

    #[test]
    fn test_validator_min_max_value() {
        let mut validator = ToolValidator::new();
        validator.add_rule("age", ValidationType::MinValue(0.0));
        validator.add_rule("age", ValidationType::MaxValue(150.0));

        // Valid
        let mut args = HashMap::new();
        args.insert("age".to_string(), serde_json::json!(25));
        assert!(validator.validate(&args).is_ok());

        // Below min
        let mut args2 = HashMap::new();
        args2.insert("age".to_string(), serde_json::json!(-1));
        let err = validator.validate(&args2).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].contains("below minimum"));

        // Above max
        let mut args3 = HashMap::new();
        args3.insert("age".to_string(), serde_json::json!(200));
        let err = validator.validate(&args3).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(err[0].contains("exceeds maximum"));
    }

    #[test]
    fn test_retry_succeeds_on_second() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut registry = ToolRegistry::new();
        let tool = ToolDefinition::new("flaky", "A flaky tool");
        registry.register(tool, move |call| {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                ToolResult::error(&call.id, &call.name, "Temporary failure")
            } else {
                ToolResult::success(&call.id, &call.name, "OK")
            }
        });

        let call = ToolCall::new("flaky", HashMap::new());
        let config = ToolRetryConfig {
            max_retries: 3,
            backoff_base_ms: 10, // keep tests fast
            backoff_multiplier: 1.0,
            retry_on_error: true,
        };

        let retryable = RetryableToolCall::new(call, config);
        let result = retryable.execute_with_retry(&registry);

        assert!(result.success);
        assert_eq!(result.content, "OK");
        // Counter should be 2: first attempt (fail) + first retry (success)
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_approval_gate_approve_deny() {
        let mut gate = ApprovalGate::new();

        let call1 = ToolCall::new("dangerous_tool", HashMap::new());
        let call2 = ToolCall::new("another_tool", HashMap::new());

        let id1 = gate.request_approval(call1);
        let id2 = gate.request_approval(call2);

        // Both should be pending
        assert_eq!(gate.pending_count(), 2);
        assert!(gate.get_approved().is_empty());

        // Approve first
        assert!(gate.approve(&id1));
        assert_eq!(gate.pending_count(), 1);
        assert_eq!(gate.get_approved().len(), 1);
        assert_eq!(gate.get_approved()[0].name, "dangerous_tool");

        // Deny second
        assert!(gate.deny(&id2, "Not allowed"));
        assert_eq!(gate.pending_count(), 0);
        assert_eq!(gate.get_approved().len(), 1); // only the approved one

        // Approve nonexistent id
        assert!(!gate.approve("approval-999"));
    }

    #[test]
    fn test_approval_gate_auto_approve() {
        let mut gate = ApprovalGate::new();
        gate.auto_approve("safe_tool");

        let safe_call = ToolCall::new("safe_tool", HashMap::new());
        let unsafe_call = ToolCall::new("unsafe_tool", HashMap::new());

        let _id1 = gate.request_approval(safe_call);
        let _id2 = gate.request_approval(unsafe_call);

        // safe_tool should be auto-approved, unsafe_tool pending
        assert_eq!(gate.pending_count(), 1);
        assert_eq!(gate.get_approved().len(), 1);
        assert_eq!(gate.get_approved()[0].name, "safe_tool");
    }
}
