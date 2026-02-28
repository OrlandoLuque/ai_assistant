//! Function calling support for OpenAI-compatible APIs
//!
//! This module provides function calling capabilities compatible with
//! OpenAI's function calling API and similar implementations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// A function definition for the AI to call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Function description (helps the AI understand when to use it)
    pub description: String,
    /// Parameter schema (JSON Schema format)
    pub parameters: FunctionParameters,
}

/// Parameters schema for a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameters {
    /// Type (always "object" for function parameters)
    #[serde(rename = "type")]
    pub param_type: String,
    /// Properties (parameter definitions)
    pub properties: HashMap<String, ParameterProperty>,
    /// Required parameters
    #[serde(default)]
    pub required: Vec<String>,
}

impl Default for FunctionParameters {
    fn default() -> Self {
        Self {
            param_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        }
    }
}

/// A single parameter property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterProperty {
    /// Parameter type (string, number, boolean, array, object)
    #[serde(rename = "type")]
    pub param_type: String,
    /// Parameter description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enum values (if type is string with fixed options)
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Items schema (for arrays)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParameterProperty>>,
    /// Minimum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

impl ParameterProperty {
    /// Create a string parameter
    pub fn string(description: &str) -> Self {
        Self {
            param_type: "string".to_string(),
            description: Some(description.to_string()),
            enum_values: None,
            default: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a number parameter
    pub fn number(description: &str) -> Self {
        Self {
            param_type: "number".to_string(),
            description: Some(description.to_string()),
            enum_values: None,
            default: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an integer parameter
    pub fn integer(description: &str) -> Self {
        Self {
            param_type: "integer".to_string(),
            description: Some(description.to_string()),
            enum_values: None,
            default: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a boolean parameter
    pub fn boolean(description: &str) -> Self {
        Self {
            param_type: "boolean".to_string(),
            description: Some(description.to_string()),
            enum_values: None,
            default: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an enum parameter (string with fixed values)
    pub fn enum_type(description: &str, values: Vec<&str>) -> Self {
        Self {
            param_type: "string".to_string(),
            description: Some(description.to_string()),
            enum_values: Some(values.into_iter().map(|s| s.to_string()).collect()),
            default: None,
            items: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an array parameter
    pub fn array(description: &str, item_type: ParameterProperty) -> Self {
        Self {
            param_type: "array".to_string(),
            description: Some(description.to_string()),
            enum_values: None,
            default: None,
            items: Some(Box::new(item_type)),
            minimum: None,
            maximum: None,
        }
    }

    /// Set default value
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Set min/max for numbers
    pub fn with_range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.minimum = min;
        self.maximum = max;
        self
    }
}

/// A function call requested by the AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name
    pub name: String,
    /// Arguments (JSON string or parsed object)
    pub arguments: Value,
}

impl FunctionCall {
    /// Parse arguments as a specific type
    pub fn parse_arguments<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        // Handle both string and object arguments
        let args = match &self.arguments {
            Value::String(s) => serde_json::from_str(s)?,
            other => other.clone(),
        };
        serde_json::from_value(args).map_err(|e| anyhow!("Failed to parse arguments: {}", e))
    }

    /// Get a string argument
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get_value(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// Get a number argument
    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.get_value(key).and_then(|v| v.as_f64())
    }

    /// Get a boolean argument
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get_value(key).and_then(|v| v.as_bool())
    }

    /// Get a raw value
    pub fn get_value(&self, key: &str) -> Option<&Value> {
        match &self.arguments {
            Value::Object(map) => map.get(key),
            Value::String(s) => {
                // Try to parse if it's a JSON string
                serde_json::from_str::<Value>(s)
                    .ok()
                    .and_then(|v| v.get(key).cloned())
                    .as_ref()
                    .map(|_| &Value::Null) // Can't return reference to temporary
            }
            _ => None,
        }
    }
}

/// Result of a function execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResult {
    /// The function name that was called
    pub name: String,
    /// The result content
    pub content: String,
    /// Whether the call was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl FunctionResult {
    /// Create a successful result
    pub fn success(name: &str, content: &str) -> Self {
        Self {
            name: name.to_string(),
            content: content.to_string(),
            success: true,
            error: None,
        }
    }

    /// Create an error result
    pub fn error(name: &str, error: &str) -> Self {
        Self {
            name: name.to_string(),
            content: String::new(),
            success: false,
            error: Some(error.to_string()),
        }
    }

    /// Create from JSON value
    pub fn success_json(name: &str, value: &Value) -> Self {
        Self {
            name: name.to_string(),
            content: serde_json::to_string(value).unwrap_or_default(),
            success: true,
            error: None,
        }
    }
}

/// Builder for function definitions
pub struct FunctionBuilder {
    function: FunctionDefinition,
}

impl FunctionBuilder {
    /// Start building a function
    pub fn new(name: &str) -> Self {
        Self {
            function: FunctionDefinition {
                name: name.to_string(),
                description: String::new(),
                parameters: FunctionParameters::default(),
            },
        }
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.function.description = desc.to_string();
        self
    }

    /// Add a parameter
    pub fn param(mut self, name: &str, property: ParameterProperty, required: bool) -> Self {
        self.function
            .parameters
            .properties
            .insert(name.to_string(), property);
        if required {
            self.function.parameters.required.push(name.to_string());
        }
        self
    }

    /// Add a required string parameter
    pub fn required_string(self, name: &str, description: &str) -> Self {
        self.param(name, ParameterProperty::string(description), true)
    }

    /// Add an optional string parameter
    pub fn optional_string(self, name: &str, description: &str) -> Self {
        self.param(name, ParameterProperty::string(description), false)
    }

    /// Add a required number parameter
    pub fn required_number(self, name: &str, description: &str) -> Self {
        self.param(name, ParameterProperty::number(description), true)
    }

    /// Add an optional number parameter
    pub fn optional_number(self, name: &str, description: &str) -> Self {
        self.param(name, ParameterProperty::number(description), false)
    }

    /// Add a required boolean parameter
    pub fn required_bool(self, name: &str, description: &str) -> Self {
        self.param(name, ParameterProperty::boolean(description), true)
    }

    /// Add an enum parameter
    pub fn required_enum(self, name: &str, description: &str, values: Vec<&str>) -> Self {
        self.param(
            name,
            ParameterProperty::enum_type(description, values),
            true,
        )
    }

    /// Build the function definition
    pub fn build(self) -> FunctionDefinition {
        self.function
    }
}

/// Function registry for managing available functions
#[derive(Debug, Default)]
pub struct FunctionRegistry {
    functions: HashMap<String, RegisteredFunction>,
}

/// A registered function with its handler
pub struct RegisteredFunction {
    /// The function definition
    pub definition: FunctionDefinition,
    /// The handler function
    handler: Box<dyn Fn(FunctionCall) -> FunctionResult + Send + Sync>,
}

impl std::fmt::Debug for RegisteredFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredFunction")
            .field("definition", &self.definition)
            .finish()
    }
}

impl FunctionRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a function with its handler
    pub fn register<F>(&mut self, definition: FunctionDefinition, handler: F)
    where
        F: Fn(FunctionCall) -> FunctionResult + Send + Sync + 'static,
    {
        self.functions.insert(
            definition.name.clone(),
            RegisteredFunction {
                definition,
                handler: Box::new(handler),
            },
        );
    }

    /// Get a function definition by name
    pub fn get(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.get(name).map(|f| &f.definition)
    }

    /// Execute a function call
    pub fn execute(&self, call: &FunctionCall) -> FunctionResult {
        match self.functions.get(&call.name) {
            Some(func) => (func.handler)(call.clone()),
            None => FunctionResult::error(&call.name, "Function not found"),
        }
    }

    /// Get all function definitions (for sending to API)
    pub fn definitions(&self) -> Vec<&FunctionDefinition> {
        self.functions.values().map(|f| &f.definition).collect()
    }

    /// Get function names
    pub fn names(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a function exists
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Remove a function
    pub fn remove(&mut self, name: &str) -> bool {
        self.functions.remove(name).is_some()
    }

    /// Convert to OpenAI API format
    pub fn to_openai_format(&self) -> Vec<Value> {
        self.functions
            .values()
            .map(|f| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": f.definition.name,
                        "description": f.definition.description,
                        "parameters": f.definition.parameters,
                    }
                })
            })
            .collect()
    }
}

/// Tool choice configuration for the API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide
    Auto,
    /// Don't use any tools
    None,
    /// Force a specific function
    Function { name: String },
}

impl ToolChoice {
    /// Convert to API format
    pub fn to_api_value(&self) -> Value {
        match self {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::None => serde_json::json!("none"),
            ToolChoice::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }
}

/// Parse function calls from API response
pub fn parse_function_calls(response: &Value) -> Vec<FunctionCall> {
    let mut calls = Vec::new();

    // Try OpenAI format (tool_calls)
    if let Some(tool_calls) = response.get("tool_calls").and_then(|v| v.as_array()) {
        for call in tool_calls {
            if let (Some(name), Some(arguments)) = (
                call.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str()),
                call.get("function").and_then(|f| f.get("arguments")),
            ) {
                calls.push(FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.clone(),
                });
            }
        }
    }

    // Try older format (function_call)
    if let Some(function_call) = response.get("function_call") {
        if let (Some(name), Some(arguments)) = (
            function_call.get("name").and_then(|n| n.as_str()),
            function_call.get("arguments"),
        ) {
            calls.push(FunctionCall {
                name: name.to_string(),
                arguments: arguments.clone(),
            });
        }
    }

    calls
}

/// Common built-in functions
pub mod builtins {
    use super::*;

    /// Get current time function
    pub fn get_current_time() -> FunctionDefinition {
        FunctionBuilder::new("get_current_time")
            .description("Get the current date and time")
            .optional_string("timezone", "Timezone (e.g., 'UTC', 'America/New_York')")
            .build()
    }

    /// Calculator function
    pub fn calculate() -> FunctionDefinition {
        FunctionBuilder::new("calculate")
            .description("Perform a mathematical calculation")
            .required_string("expression", "The mathematical expression to evaluate")
            .build()
    }

    /// Web search function
    pub fn web_search() -> FunctionDefinition {
        FunctionBuilder::new("web_search")
            .description("Search the web for information")
            .required_string("query", "The search query")
            .optional_number("num_results", "Number of results to return")
            .build()
    }

    /// File read function
    pub fn read_file() -> FunctionDefinition {
        FunctionBuilder::new("read_file")
            .description("Read the contents of a file")
            .required_string("path", "The file path to read")
            .build()
    }

    /// File write function
    pub fn write_file() -> FunctionDefinition {
        FunctionBuilder::new("write_file")
            .description("Write content to a file")
            .required_string("path", "The file path to write to")
            .required_string("content", "The content to write")
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_builder() {
        let func = FunctionBuilder::new("test_function")
            .description("A test function")
            .required_string("name", "The name parameter")
            .optional_number("count", "Optional count")
            .build();

        assert_eq!(func.name, "test_function");
        assert_eq!(func.parameters.required, vec!["name"]);
        assert!(func.parameters.properties.contains_key("name"));
        assert!(func.parameters.properties.contains_key("count"));
    }

    #[test]
    fn test_function_call_parsing() {
        let call = FunctionCall {
            name: "test".to_string(),
            arguments: serde_json::json!({
                "name": "John",
                "age": 30
            }),
        };

        assert_eq!(call.get_string("name"), Some("John".to_string()));
        assert_eq!(call.get_number("age"), Some(30.0));
    }

    #[test]
    fn test_function_registry() {
        let mut registry = FunctionRegistry::new();

        let func = FunctionBuilder::new("greet")
            .description("Greet someone")
            .required_string("name", "Name to greet")
            .build();

        registry.register(func, |call| {
            let name = call.get_string("name").unwrap_or_default();
            FunctionResult::success("greet", &format!("Hello, {}!", name))
        });

        assert!(registry.has("greet"));

        let call = FunctionCall {
            name: "greet".to_string(),
            arguments: serde_json::json!({"name": "World"}),
        };

        let result = registry.execute(&call);
        assert!(result.success);
        assert!(result.content.contains("Hello, World!"));
    }

    #[test]
    fn test_openai_format() {
        let mut registry = FunctionRegistry::new();

        registry.register(builtins::get_current_time(), |_| {
            FunctionResult::success("get_current_time", "2024-01-01 00:00:00")
        });

        let format = registry.to_openai_format();
        assert!(!format.is_empty());
        assert_eq!(format[0]["type"], "function");
    }

    #[test]
    fn test_parse_function_calls() {
        let response = serde_json::json!({
            "tool_calls": [{
                "function": {
                    "name": "test_func",
                    "arguments": "{\"key\": \"value\"}"
                }
            }]
        });

        let calls = parse_function_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test_func");
    }

    #[test]
    fn test_parameter_property() {
        let param = ParameterProperty::number("A number")
            .with_range(Some(0.0), Some(100.0))
            .with_default(serde_json::json!(50));

        assert_eq!(param.param_type, "number");
        assert_eq!(param.minimum, Some(0.0));
        assert_eq!(param.maximum, Some(100.0));
        assert_eq!(param.default, Some(serde_json::json!(50)));
    }

    #[test]
    fn test_enum_parameter() {
        let param = ParameterProperty::enum_type("Color", vec!["red", "green", "blue"]);

        assert_eq!(param.param_type, "string");
        assert_eq!(
            param.enum_values,
            Some(vec![
                "red".to_string(),
                "green".to_string(),
                "blue".to_string()
            ])
        );
    }

    #[test]
    fn test_function_result_error_and_success_json() {
        let err_result = FunctionResult::error("broken_fn", "something went wrong");
        assert!(!err_result.success);
        assert_eq!(err_result.name, "broken_fn");
        assert_eq!(
            err_result.error,
            Some("something went wrong".to_string())
        );
        assert!(err_result.content.is_empty());

        let json_val = serde_json::json!({"status": "ok", "count": 42});
        let ok_result = FunctionResult::success_json("json_fn", &json_val);
        assert!(ok_result.success);
        assert!(ok_result.error.is_none());
        // Content should be valid JSON
        let parsed: Value = serde_json::from_str(&ok_result.content).unwrap();
        assert_eq!(parsed["count"], 42);
    }

    #[test]
    fn test_registry_remove_and_execute_unknown() {
        let mut registry = FunctionRegistry::new();

        let func = FunctionBuilder::new("temp_fn")
            .description("Temporary function")
            .build();

        registry.register(func, |_| FunctionResult::success("temp_fn", "done"));
        assert!(registry.has("temp_fn"));
        assert_eq!(registry.names().len(), 1);

        // Remove the function
        assert!(registry.remove("temp_fn"));
        assert!(!registry.has("temp_fn"));
        assert!(registry.get("temp_fn").is_none());

        // Removing again returns false
        assert!(!registry.remove("temp_fn"));

        // Execute an unknown function returns error result
        let call = FunctionCall {
            name: "nonexistent".to_string(),
            arguments: serde_json::json!({}),
        };
        let result = registry.execute(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("not found"));
    }

    #[test]
    fn test_parse_legacy_function_call_format_and_get_bool() {
        // Test the older function_call format (not tool_calls)
        let response = serde_json::json!({
            "function_call": {
                "name": "toggle_feature",
                "arguments": {"enabled": true, "label": "dark_mode"}
            }
        });

        let calls = parse_function_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "toggle_feature");

        // Test get_bool accessor
        assert_eq!(calls[0].get_bool("enabled"), Some(true));
        assert_eq!(calls[0].get_string("label"), Some("dark_mode".to_string()));
        assert_eq!(calls[0].get_bool("nonexistent"), None);

        // Test array parameter property construction
        let items = ParameterProperty::string("item name");
        let arr = ParameterProperty::array("List of items", items);
        assert_eq!(arr.param_type, "array");
        assert!(arr.items.is_some());
        assert_eq!(arr.items.as_ref().unwrap().param_type, "string");
    }
}
