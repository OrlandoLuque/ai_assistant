//! Tool calling system
//!
//! Enables LLM models to use tools like web search, calculations, etc.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Tool handler function type
pub type ToolHandlerFn =
    dyn Fn(&HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync;

/// Tool definition that can be used by an LLM
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub handler: Option<Arc<ToolHandlerFn>>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("handler", &self.handler.is_some())
            .finish()
    }
}

/// Tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default: Option<String>,
}

/// Parameter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

/// Tool call request from model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl Tool {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: Vec::new(),
            handler: None,
        }
    }

    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(&HashMap<String, serde_json::Value>) -> Result<String, String>
            + Send
            + Sync
            + 'static,
    {
        self.handler = Some(Arc::new(handler));
        self
    }

    /// Generate JSON schema for this tool (OpenAI function calling format)
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let type_str = match param.param_type {
                ParameterType::String => "string",
                ParameterType::Integer => "integer",
                ParameterType::Float => "number",
                ParameterType::Boolean => "boolean",
                ParameterType::Array => "array",
                ParameterType::Object => "object",
            };

            let mut prop = serde_json::Map::new();
            prop.insert(
                "type".to_string(),
                serde_json::Value::String(type_str.to_string()),
            );
            prop.insert(
                "description".to_string(),
                serde_json::Value::String(param.description.clone()),
            );

            properties.insert(param.name.clone(), serde_json::Value::Object(prop));

            if param.required {
                required.push(serde_json::Value::String(param.name.clone()));
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

impl ToolParameter {
    pub fn new(name: &str, description: &str, param_type: ParameterType) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            param_type,
            required: true,
            default: None,
        }
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn with_default(mut self, default: &str) -> Self {
        self.default = Some(default.to_string());
        self.required = false;
        self
    }
}

/// Tool registry and executor
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        let tool = match self.tools.get(&call.tool_name) {
            Some(t) => t,
            None => {
                return ToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("Tool '{}' not found", call.tool_name)),
                };
            }
        };

        let handler = match &tool.handler {
            Some(h) => h,
            None => {
                return ToolResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some("Tool has no handler".to_string()),
                };
            }
        };

        match handler(&call.arguments) {
            Ok(output) => ToolResult {
                call_id: call.id.clone(),
                tool_name: call.tool_name.clone(),
                success: true,
                output,
                error: None,
            },
            Err(e) => ToolResult {
                call_id: call.id.clone(),
                tool_name: call.tool_name.clone(),
                success: false,
                output: String::new(),
                error: Some(e),
            },
        }
    }

    /// Generate tools array for API calls (OpenAI format)
    pub fn to_json_schema(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|t| t.to_json_schema()).collect()
    }

    /// Parse tool calls from model response (supports multiple formats)
    pub fn parse_tool_calls(&self, response: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Try JSON format first
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(tool_calls) = parsed.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    if let (Some(name), Some(args)) = (
                        tc.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str()),
                        tc.get("function").and_then(|f| f.get("arguments")),
                    ) {
                        let arguments: HashMap<String, serde_json::Value> =
                            if let Some(s) = args.as_str() {
                                serde_json::from_str(s).unwrap_or_default()
                            } else if let Some(obj) = args.as_object() {
                                obj.clone().into_iter().collect()
                            } else {
                                HashMap::new()
                            };

                        calls.push(ToolCall {
                            id: tc
                                .get("id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string(),
                            tool_name: name.to_string(),
                            arguments,
                        });
                    }
                }
            }
        }

        // Try text-based format: [TOOL:name(arg1="value1", arg2="value2")]
        let re = regex::Regex::new(r"\[TOOL:(\w+)\((.*?)\)\]").ok();
        if let Some(re) = re {
            for cap in re.captures_iter(response) {
                let name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
                let args_str = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                let mut arguments = HashMap::new();
                let arg_re = regex::Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).ok();
                if let Some(arg_re) = arg_re {
                    for arg_cap in arg_re.captures_iter(args_str) {
                        let key = arg_cap.get(1).map(|m| m.as_str()).unwrap_or("");
                        let value = arg_cap.get(2).map(|m| m.as_str()).unwrap_or("");
                        arguments.insert(
                            key.to_string(),
                            serde_json::Value::String(value.to_string()),
                        );
                    }
                }

                calls.push(ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    tool_name: name.to_string(),
                    arguments,
                });
            }
        }

        // Try XML-like format: <tool name="search"><query>text</query></tool>
        let xml_re = regex::Regex::new(r#"<tool\s+name="(\w+)">(.*?)</tool>"#).ok();
        if let Some(xml_re) = xml_re {
            for cap in xml_re.captures_iter(response) {
                let name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
                let content = cap.get(2).map(|m| m.as_str()).unwrap_or("");

                let mut arguments = HashMap::new();
                let param_re = regex::Regex::new(r"<(\w+)>(.*?)</\1>").ok();
                if let Some(param_re) = param_re {
                    for param_cap in param_re.captures_iter(content) {
                        let key = param_cap.get(1).map(|m| m.as_str()).unwrap_or("");
                        let value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");
                        arguments.insert(
                            key.to_string(),
                            serde_json::Value::String(value.to_string()),
                        );
                    }
                }

                // If no params found, use content as default "query" param
                if arguments.is_empty() && !content.trim().is_empty() {
                    arguments.insert(
                        "query".to_string(),
                        serde_json::Value::String(content.trim().to_string()),
                    );
                }

                calls.push(ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    tool_name: name.to_string(),
                    arguments,
                });
            }
        }

        calls
    }

    /// Format tool results for injection into context
    pub fn format_results_for_context(&self, results: &[ToolResult]) -> String {
        let mut output = String::new();

        for result in results {
            if result.success {
                output.push_str(&format!(
                    "[Tool: {}]\n{}\n\n",
                    result.tool_name, result.output
                ));
            } else {
                output.push_str(&format!(
                    "[Tool: {} - Error]\n{}\n\n",
                    result.tool_name,
                    result.error.as_deref().unwrap_or("Unknown error")
                ));
            }
        }

        output
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for common tools
pub struct CommonTools;

impl CommonTools {
    /// Create a web search tool
    pub fn web_search() -> Tool {
        Tool::new("web_search", "Search the web for current information")
            .with_parameter(ToolParameter::new(
                "query",
                "The search query",
                ParameterType::String,
            ))
            .with_parameter(
                ToolParameter::new(
                    "num_results",
                    "Number of results to return",
                    ParameterType::Integer,
                )
                .with_default("5"),
            )
    }

    /// Create a calculator tool
    pub fn calculator() -> Tool {
        Tool::new("calculator", "Perform mathematical calculations")
            .with_parameter(ToolParameter::new(
                "expression",
                "The mathematical expression to evaluate",
                ParameterType::String,
            ))
            .with_handler(|args| {
                let expr = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing expression")?;

                // Simple expression evaluator
                evaluate_expression(expr)
            })
    }

    /// Create a date/time tool
    pub fn datetime() -> Tool {
        Tool::new("datetime", "Get current date and time information")
            .with_parameter(
                ToolParameter::new(
                    "format",
                    "Date format (iso, human, timestamp)",
                    ParameterType::String,
                )
                .with_default("human"),
            )
            .with_handler(|args| {
                let format = args
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("human");

                let now = chrono::Local::now();

                Ok(match format {
                    "iso" => now.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    "timestamp" => now.timestamp().to_string(),
                    _ => now.format("%A, %B %d, %Y at %H:%M").to_string(),
                })
            })
    }

    /// Create a text length tool
    pub fn text_length() -> Tool {
        Tool::new("text_length", "Count characters and words in text")
            .with_parameter(ToolParameter::new(
                "text",
                "The text to analyze",
                ParameterType::String,
            ))
            .with_handler(|args| {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing text")?;

                let chars = text.chars().count();
                let words = text.split_whitespace().count();
                let lines = text.lines().count();

                Ok(format!(
                    "Characters: {}, Words: {}, Lines: {}",
                    chars, words, lines
                ))
            })
    }
}

/// Simple expression evaluator
fn evaluate_expression(expr: &str) -> Result<String, String> {
    // Remove whitespace
    let expr = expr.replace(' ', "");

    // Simple parser for basic operations
    if let Some(pos) = expr.rfind('+') {
        let (left, right) = expr.split_at(pos);
        let left_val: f64 = evaluate_expression(left)?
            .parse()
            .map_err(|_| "Invalid number")?;
        let right_val: f64 = evaluate_expression(&right[1..])?
            .parse()
            .map_err(|_| "Invalid number")?;
        return Ok((left_val + right_val).to_string());
    }

    if let Some(pos) = expr.rfind('-') {
        if pos > 0 {
            let (left, right) = expr.split_at(pos);
            let left_val: f64 = evaluate_expression(left)?
                .parse()
                .map_err(|_| "Invalid number")?;
            let right_val: f64 = evaluate_expression(&right[1..])?
                .parse()
                .map_err(|_| "Invalid number")?;
            return Ok((left_val - right_val).to_string());
        }
    }

    if let Some(pos) = expr.rfind('*') {
        let (left, right) = expr.split_at(pos);
        let left_val: f64 = evaluate_expression(left)?
            .parse()
            .map_err(|_| "Invalid number")?;
        let right_val: f64 = evaluate_expression(&right[1..])?
            .parse()
            .map_err(|_| "Invalid number")?;
        return Ok((left_val * right_val).to_string());
    }

    if let Some(pos) = expr.rfind('/') {
        let (left, right) = expr.split_at(pos);
        let left_val: f64 = evaluate_expression(left)?
            .parse()
            .map_err(|_| "Invalid number")?;
        let right_val: f64 = evaluate_expression(&right[1..])?
            .parse()
            .map_err(|_| "Invalid number")?;
        if right_val == 0.0 {
            return Err("Division by zero".to_string());
        }
        return Ok((left_val / right_val).to_string());
    }

    // Try parsing as number
    expr.parse::<f64>()
        .map(|n| n.to_string())
        .map_err(|_| format!("Invalid expression: {}", expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_creation() {
        let tool = CommonTools::web_search();
        assert_eq!(tool.name, "web_search");
        assert_eq!(tool.parameters.len(), 2);
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(CommonTools::calculator());
        registry.register(CommonTools::datetime());

        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("calculator").is_some());
    }

    #[test]
    fn test_calculator_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(CommonTools::calculator());

        let call = ToolCall {
            id: "1".to_string(),
            tool_name: "calculator".to_string(),
            arguments: [(
                "expression".to_string(),
                serde_json::Value::String("2+3*4".to_string()),
            )]
            .into_iter()
            .collect(),
        };

        let result = registry.execute(&call);
        assert!(result.success);
    }

    #[test]
    fn test_parse_tool_calls_text_format() {
        let registry = ToolRegistry::new();
        let response = r#"I'll search for that. [TOOL:web_search(query="rust programming")]"#;

        let calls = registry.parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name, "web_search");
    }

    #[test]
    fn test_parse_tool_calls_xml_format() {
        let registry = ToolRegistry::new();
        let response = r#"<tool name="search"><query>latest news</query></tool>"#;

        let calls = registry.parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name, "search");
    }

    #[test]
    fn test_json_schema_generation() {
        let tool = CommonTools::web_search();
        let schema = tool.to_json_schema();

        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "web_search");
    }

    #[test]
    fn test_common_tools_calculator() {
        let calc = CommonTools::calculator();
        assert_eq!(calc.name, "calculator");
    }

    #[test]
    fn test_common_tools_datetime() {
        let dt = CommonTools::datetime();
        assert_eq!(dt.name, "datetime");
    }

    #[test]
    fn test_evaluate_expression() {
        assert_eq!(evaluate_expression("2+3").unwrap(), "5");
        assert_eq!(evaluate_expression("10/2").unwrap(), "5");
    }

    #[test]
    fn test_format_results_for_context() {
        let registry = ToolRegistry::new();
        let results = vec![ToolResult {
            call_id: "1".to_string(),
            tool_name: "test".to_string(),
            success: true,
            output: "result: 42".to_string(),
            error: None,
        }];
        let formatted = registry.format_results_for_context(&results);
        assert!(formatted.contains("test"));
    }
}
