//! Structured Output - JSON generation with schema validation
//!
//! This module provides tools for generating structured JSON outputs
//! from LLM responses with schema validation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// JSON Schema types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

/// Property definition in a schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaProperty {
    /// Property type
    #[serde(rename = "type")]
    pub schema_type: SchemaType,
    /// Property description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Enum values (for string type)
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Minimum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Minimum length (for strings/arrays)
    #[serde(rename = "minLength", skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,
    /// Maximum length (for strings/arrays)
    #[serde(rename = "maxLength", skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    /// Pattern (for strings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Items schema (for arrays)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<SchemaProperty>>,
    /// Nested properties (for objects)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, SchemaProperty>>,
    /// Required properties (for objects)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl SchemaProperty {
    /// Create a string property
    pub fn string() -> Self {
        Self {
            schema_type: SchemaType::String,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a number property
    pub fn number() -> Self {
        Self {
            schema_type: SchemaType::Number,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an integer property
    pub fn integer() -> Self {
        Self {
            schema_type: SchemaType::Integer,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a boolean property
    pub fn boolean() -> Self {
        Self {
            schema_type: SchemaType::Boolean,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an array property
    pub fn array(items: SchemaProperty) -> Self {
        Self {
            schema_type: SchemaType::Array,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: Some(Box::new(items)),
            properties: None,
            required: None,
        }
    }

    /// Create an object property
    pub fn object() -> Self {
        Self {
            schema_type: SchemaType::Object,
            description: None,
            enum_values: None,
            default: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            properties: Some(HashMap::new()),
            required: Some(Vec::new()),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set enum values
    pub fn with_enum(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }

    /// Set default value
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Set minimum value
    pub fn with_minimum(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    /// Set maximum value
    pub fn with_maximum(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    /// Set min length
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Set max length
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Set pattern
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// Add a property (for objects)
    pub fn with_property(mut self, name: impl Into<String>, prop: SchemaProperty) -> Self {
        if let Some(ref mut props) = self.properties {
            props.insert(name.into(), prop);
        }
        self
    }

    /// Mark a property as required (for objects)
    pub fn with_required(mut self, name: impl Into<String>) -> Self {
        if let Some(ref mut req) = self.required {
            req.push(name.into());
        }
        self
    }
}

/// JSON Schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema name
    pub name: String,
    /// Schema description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Root property (usually an object)
    #[serde(flatten)]
    pub root: SchemaProperty,
}

impl JsonSchema {
    /// Create a new schema
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            root: SchemaProperty::object(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a property
    pub fn with_property(mut self, name: impl Into<String>, prop: SchemaProperty) -> Self {
        self.root = self.root.with_property(name, prop);
        self
    }

    /// Mark a property as required
    pub fn with_required(mut self, name: impl Into<String>) -> Self {
        self.root = self.root.with_required(name);
        self
    }

    /// Generate a prompt for the schema
    pub fn to_prompt(&self) -> String {
        let schema_json = serde_json::to_string_pretty(self).unwrap_or_default();
        format!(
            "Respond with a JSON object that matches this schema:\n\n```json\n{}\n```\n\nRespond ONLY with valid JSON, no other text.",
            schema_json
        )
    }

    /// Convert to OpenAI response_format
    pub fn to_openai_format(&self) -> Value {
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.root,
                "strict": true
            }
        })
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Path to the invalid field
    pub path: String,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validated value (possibly with defaults applied)
    pub value: Option<Value>,
}

impl ValidationResult {
    fn success(value: Value) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            value: Some(value),
        }
    }

    fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
            value: None,
        }
    }
}

/// JSON Schema validator
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate a value against a schema
    pub fn validate(value: &Value, schema: &JsonSchema) -> ValidationResult {
        let mut errors = Vec::new();
        let validated = Self::validate_property(value, &schema.root, "$", &mut errors);

        if errors.is_empty() {
            ValidationResult::success(validated)
        } else {
            ValidationResult::failure(errors)
        }
    }

    fn validate_property(
        value: &Value,
        prop: &SchemaProperty,
        path: &str,
        errors: &mut Vec<ValidationError>,
    ) -> Value {
        // Check type
        let type_valid = match (&prop.schema_type, value) {
            (SchemaType::String, Value::String(_)) => true,
            (SchemaType::Number, Value::Number(_)) => true,
            (SchemaType::Integer, Value::Number(n)) => n.is_i64() || n.is_u64(),
            (SchemaType::Boolean, Value::Bool(_)) => true,
            (SchemaType::Array, Value::Array(_)) => true,
            (SchemaType::Object, Value::Object(_)) => true,
            (SchemaType::Null, Value::Null) => true,
            _ => false,
        };

        if !type_valid {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!(
                    "Expected {:?}, got {:?}",
                    prop.schema_type,
                    value_type(value)
                ),
            });
            return value.clone();
        }

        // Type-specific validation
        match value {
            Value::String(s) => {
                // Check enum
                if let Some(ref enum_values) = prop.enum_values {
                    if !enum_values.contains(s) {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("Value '{}' not in enum {:?}", s, enum_values),
                        });
                    }
                }
                // Check min length
                if let Some(min) = prop.min_length {
                    if s.len() < min {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("String length {} < minimum {}", s.len(), min),
                        });
                    }
                }
                // Check max length
                if let Some(max) = prop.max_length {
                    if s.len() > max {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("String length {} > maximum {}", s.len(), max),
                        });
                    }
                }
                // Check pattern (simple validation - patterns stored for schema export only)
                if let Some(ref _pattern) = prop.pattern {
                    // Pattern validation requires regex crate - skipping for now
                    // Patterns are still exported for schema consumers to validate
                }
            }
            Value::Number(n) => {
                let num = n.as_f64().unwrap_or(0.0);
                // Check minimum
                if let Some(min) = prop.minimum {
                    if num < min {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("Number {} < minimum {}", num, min),
                        });
                    }
                }
                // Check maximum
                if let Some(max) = prop.maximum {
                    if num > max {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("Number {} > maximum {}", num, max),
                        });
                    }
                }
            }
            Value::Array(arr) => {
                // Check min length
                if let Some(min) = prop.min_length {
                    if arr.len() < min {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("Array length {} < minimum {}", arr.len(), min),
                        });
                    }
                }
                // Check max length
                if let Some(max) = prop.max_length {
                    if arr.len() > max {
                        errors.push(ValidationError {
                            path: path.to_string(),
                            message: format!("Array length {} > maximum {}", arr.len(), max),
                        });
                    }
                }
                // Validate items
                if let Some(ref items_schema) = prop.items {
                    for (i, item) in arr.iter().enumerate() {
                        let item_path = format!("{}[{}]", path, i);
                        Self::validate_property(item, items_schema, &item_path, errors);
                    }
                }
            }
            Value::Object(obj) => {
                // Check required properties
                if let Some(ref required) = prop.required {
                    for req_prop in required {
                        if !obj.contains_key(req_prop) {
                            errors.push(ValidationError {
                                path: format!("{}.{}", path, req_prop),
                                message: "Required property missing".to_string(),
                            });
                        }
                    }
                }
                // Validate properties
                if let Some(ref props) = prop.properties {
                    for (key, value) in obj {
                        let prop_path = format!("{}.{}", path, key);
                        if let Some(prop_schema) = props.get(key) {
                            Self::validate_property(value, prop_schema, &prop_path, errors);
                        }
                    }
                }
            }
            _ => {}
        }

        value.clone()
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Structured output generator
pub struct StructuredOutputGenerator {
    /// Schemas registry
    schemas: HashMap<String, JsonSchema>,
}

impl StructuredOutputGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a schema
    pub fn register_schema(&mut self, schema: JsonSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    /// Get a schema by name
    pub fn get_schema(&self, name: &str) -> Option<&JsonSchema> {
        self.schemas.get(name)
    }

    /// Generate prompt for a schema
    pub fn generate_prompt(&self, schema_name: &str, user_prompt: &str) -> Option<String> {
        self.schemas
            .get(schema_name)
            .map(|schema| format!("{}\n\n{}", user_prompt, schema.to_prompt()))
    }

    /// Parse and validate a response
    pub fn parse_response(
        &self,
        schema_name: &str,
        response: &str,
    ) -> Result<ValidationResult, String> {
        let schema = self
            .schemas
            .get(schema_name)
            .ok_or_else(|| format!("Schema '{}' not found", schema_name))?;

        // Try to extract JSON from response
        let json_str = extract_json(response);

        // Parse JSON
        let value: Value =
            serde_json::from_str(&json_str).map_err(|e| format!("JSON parse error: {}", e))?;

        Ok(SchemaValidator::validate(&value, schema))
    }

    /// Parse response with retry hints
    pub fn parse_with_hints(&self, schema_name: &str, response: &str) -> StructuredParseResult {
        let schema = match self.schemas.get(schema_name) {
            Some(s) => s,
            None => {
                return StructuredParseResult {
                    success: false,
                    value: None,
                    errors: vec!["Schema not found".to_string()],
                    retry_prompt: None,
                }
            }
        };

        let json_str = extract_json(response);

        match serde_json::from_str::<Value>(&json_str) {
            Ok(value) => {
                let validation = SchemaValidator::validate(&value, schema);
                if validation.valid {
                    StructuredParseResult {
                        success: true,
                        value: validation.value,
                        errors: Vec::new(),
                        retry_prompt: None,
                    }
                } else {
                    let error_msgs: Vec<String> =
                        validation.errors.iter().map(|e| e.to_string()).collect();
                    let retry = format!(
                        "The JSON response had validation errors:\n{}\n\nPlease fix these issues and respond again with valid JSON.",
                        error_msgs.join("\n")
                    );
                    StructuredParseResult {
                        success: false,
                        value: None,
                        errors: error_msgs,
                        retry_prompt: Some(retry),
                    }
                }
            }
            Err(e) => {
                let retry = format!(
                    "Could not parse your response as JSON: {}\n\nPlease respond with ONLY valid JSON, no other text.",
                    e
                );
                StructuredParseResult {
                    success: false,
                    value: None,
                    errors: vec![e.to_string()],
                    retry_prompt: Some(retry),
                }
            }
        }
    }
}

impl Default for StructuredOutputGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of parsing structured output
#[derive(Debug, Clone)]
pub struct StructuredParseResult {
    /// Whether parsing and validation succeeded
    pub success: bool,
    /// Parsed and validated value
    pub value: Option<Value>,
    /// Error messages
    pub errors: Vec<String>,
    /// Prompt to retry if failed
    pub retry_prompt: Option<String>,
}

/// Extract JSON from a response that might contain other text
fn extract_json(response: &str) -> String {
    // Try to find JSON in code blocks
    if let Some(start) = response.find("```json") {
        if let Some(end) = response[start + 7..].find("```") {
            return response[start + 7..start + 7 + end].trim().to_string();
        }
    }

    // Try to find JSON in generic code blocks
    if let Some(start) = response.find("```") {
        let after_start = &response[start + 3..];
        if let Some(end) = after_start.find("```") {
            let content = &after_start[..end];
            // Skip language identifier if present
            let content = if let Some(newline) = content.find('\n') {
                &content[newline + 1..]
            } else {
                content
            };
            if content.trim().starts_with('{') || content.trim().starts_with('[') {
                return content.trim().to_string();
            }
        }
    }

    // Try to find raw JSON object
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            if end > start {
                return response[start..=end].to_string();
            }
        }
    }

    // Try to find raw JSON array
    if let Some(start) = response.find('[') {
        if let Some(end) = response.rfind(']') {
            if end > start {
                return response[start..=end].to_string();
            }
        }
    }

    response.to_string()
}

/// Schema builder for common patterns
pub struct SchemaBuilder;

impl SchemaBuilder {
    /// Create a sentiment analysis schema
    pub fn sentiment_analysis() -> JsonSchema {
        JsonSchema::new("sentiment_analysis")
            .with_description("Sentiment analysis result")
            .with_property(
                "sentiment",
                SchemaProperty::string()
                    .with_description("Overall sentiment")
                    .with_enum(vec![
                        "positive".to_string(),
                        "negative".to_string(),
                        "neutral".to_string(),
                        "mixed".to_string(),
                    ]),
            )
            .with_property(
                "confidence",
                SchemaProperty::number()
                    .with_description("Confidence score from 0 to 1")
                    .with_minimum(0.0)
                    .with_maximum(1.0),
            )
            .with_property(
                "reasons",
                SchemaProperty::array(SchemaProperty::string())
                    .with_description("Reasons for the sentiment"),
            )
            .with_required("sentiment")
            .with_required("confidence")
    }

    /// Create an entity extraction schema
    pub fn entity_extraction() -> JsonSchema {
        let entity = SchemaProperty::object()
            .with_property(
                "text",
                SchemaProperty::string().with_description("The extracted text"),
            )
            .with_property(
                "type",
                SchemaProperty::string()
                    .with_description("Entity type")
                    .with_enum(vec![
                        "person".to_string(),
                        "organization".to_string(),
                        "location".to_string(),
                        "date".to_string(),
                        "email".to_string(),
                        "phone".to_string(),
                        "url".to_string(),
                        "other".to_string(),
                    ]),
            )
            .with_property(
                "confidence",
                SchemaProperty::number().with_minimum(0.0).with_maximum(1.0),
            )
            .with_required("text")
            .with_required("type");

        JsonSchema::new("entity_extraction")
            .with_description("Entity extraction result")
            .with_property(
                "entities",
                SchemaProperty::array(entity).with_description("List of extracted entities"),
            )
            .with_required("entities")
    }

    /// Create a classification schema
    pub fn classification(categories: Vec<String>) -> JsonSchema {
        JsonSchema::new("classification")
            .with_description("Text classification result")
            .with_property(
                "category",
                SchemaProperty::string()
                    .with_description("The classified category")
                    .with_enum(categories),
            )
            .with_property(
                "confidence",
                SchemaProperty::number()
                    .with_description("Confidence score")
                    .with_minimum(0.0)
                    .with_maximum(1.0),
            )
            .with_property(
                "explanation",
                SchemaProperty::string().with_description("Explanation for the classification"),
            )
            .with_required("category")
            .with_required("confidence")
    }

    /// Create a summary schema
    pub fn summary() -> JsonSchema {
        JsonSchema::new("summary")
            .with_description("Text summary result")
            .with_property(
                "title",
                SchemaProperty::string()
                    .with_description("A short title")
                    .with_max_length(100),
            )
            .with_property(
                "summary",
                SchemaProperty::string().with_description("The summary text"),
            )
            .with_property(
                "key_points",
                SchemaProperty::array(SchemaProperty::string())
                    .with_description("Key points from the text")
                    .with_min_length(1)
                    .with_max_length(10),
            )
            .with_required("summary")
            .with_required("key_points")
    }

    /// Create a translation schema
    pub fn translation() -> JsonSchema {
        JsonSchema::new("translation")
            .with_description("Translation result")
            .with_property(
                "original_language",
                SchemaProperty::string().with_description("Detected source language"),
            )
            .with_property(
                "target_language",
                SchemaProperty::string().with_description("Target language"),
            )
            .with_property(
                "translation",
                SchemaProperty::string().with_description("The translated text"),
            )
            .with_property(
                "alternatives",
                SchemaProperty::array(SchemaProperty::string())
                    .with_description("Alternative translations"),
            )
            .with_required("translation")
    }

    /// Create a Q&A schema
    pub fn question_answer() -> JsonSchema {
        JsonSchema::new("question_answer")
            .with_description("Question answering result")
            .with_property(
                "answer",
                SchemaProperty::string().with_description("The answer to the question"),
            )
            .with_property(
                "confidence",
                SchemaProperty::number()
                    .with_description("Confidence in the answer")
                    .with_minimum(0.0)
                    .with_maximum(1.0),
            )
            .with_property(
                "sources",
                SchemaProperty::array(SchemaProperty::string())
                    .with_description("Sources for the answer"),
            )
            .with_property(
                "follow_up_questions",
                SchemaProperty::array(SchemaProperty::string())
                    .with_description("Suggested follow-up questions"),
            )
            .with_required("answer")
    }
}

/// Strategy for enforcing structured output based on provider capabilities.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredOutputStrategy {
    /// Use OpenAI's native response_format with json_schema
    OpenAiNative,
    /// Use Anthropic's forced tool use pattern
    AnthropicToolUse,
    /// Use prompt engineering + validation + retry
    PromptEngineering,
}

/// Errors that can occur during structured output enforcement.
#[derive(Debug)]
pub enum StructuredOutputError {
    /// The LLM generation call itself failed.
    GenerationFailed(String),
    /// Could not parse the LLM response as JSON.
    ParseFailed(String),
    /// The parsed JSON did not pass schema validation.
    ValidationFailed(String),
    /// Could not extract structured data from a provider-specific response format.
    ExtractionFailed(String),
}

impl std::fmt::Display for StructuredOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuredOutputError::GenerationFailed(msg) => {
                write!(f, "Generation failed: {}", msg)
            }
            StructuredOutputError::ParseFailed(msg) => write!(f, "Parse failed: {}", msg),
            StructuredOutputError::ValidationFailed(msg) => {
                write!(f, "Validation failed: {}", msg)
            }
            StructuredOutputError::ExtractionFailed(msg) => {
                write!(f, "Extraction failed: {}", msg)
            }
        }
    }
}

/// Builder for creating structured output requests with provider-specific optimization.
pub struct StructuredOutputRequest {
    schema: JsonSchema,
    strategy: StructuredOutputStrategy,
    max_retries: usize,
    strict: bool,
}

impl StructuredOutputRequest {
    /// Create a new structured output request with the given schema.
    /// Defaults to PromptEngineering strategy, 3 retries, strict mode enabled.
    pub fn new(schema: JsonSchema) -> Self {
        Self {
            schema,
            strategy: StructuredOutputStrategy::PromptEngineering,
            max_retries: 3,
            strict: true,
        }
    }

    /// Set the structured output strategy.
    pub fn with_strategy(mut self, strategy: StructuredOutputStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the maximum number of retries for prompt engineering strategy.
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set whether to use strict mode for OpenAI response_format.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Get a reference to the strategy.
    pub fn strategy(&self) -> &StructuredOutputStrategy {
        &self.strategy
    }

    /// Generate the OpenAI response_format parameter for the API request body.
    ///
    /// Produces a JSON value matching the OpenAI `response_format` spec:
    /// ```json
    /// {
    ///   "type": "json_schema",
    ///   "json_schema": {
    ///     "name": "<schema_name>",
    ///     "strict": true|false,
    ///     "schema": { ... }
    ///   }
    /// }
    /// ```
    pub fn to_openai_response_format(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": self.schema.name,
                "strict": self.strict,
                "schema": self.schema.to_openai_format()
            }
        })
    }

    /// Generate Anthropic tool_use parameters.
    ///
    /// Returns a tuple of (tools_array, tool_choice) for inclusion in the API request.
    /// The tool definition is derived from the schema, and tool_choice forces the model
    /// to use that specific tool, ensuring structured output.
    pub fn to_anthropic_tool_params(&self) -> (serde_json::Value, serde_json::Value) {
        let tool_name = self.schema.name.clone();

        let tool_def = serde_json::json!([{
            "name": tool_name,
            "description": self.schema.description.as_deref()
                .unwrap_or("Generate structured output matching the schema"),
            "input_schema": self.schema.to_openai_format()
        }]);

        let tool_choice = serde_json::json!({
            "type": "tool",
            "name": tool_name
        });

        (tool_def, tool_choice)
    }

    /// Extract structured output from an Anthropic tool_use response.
    ///
    /// Expects a response JSON with a `content` array containing at least one block
    /// of type `tool_use`. Extracts the `input` field from the first matching tool_use
    /// block and validates it against the schema.
    pub fn extract_from_anthropic_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<serde_json::Value, StructuredOutputError> {
        // Look for content array
        let content = response
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                StructuredOutputError::ExtractionFailed(
                    "Response missing 'content' array".to_string(),
                )
            })?;

        // Find the tool_use block
        let tool_use_block = content
            .iter()
            .find(|block| block.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
            .ok_or_else(|| {
                StructuredOutputError::ExtractionFailed(
                    "No tool_use block found in response content".to_string(),
                )
            })?;

        // Extract the input field
        let input = tool_use_block.get("input").ok_or_else(|| {
            StructuredOutputError::ExtractionFailed(
                "tool_use block missing 'input' field".to_string(),
            )
        })?;

        // Validate against schema
        self.validate_output(input)?;

        Ok(input.clone())
    }

    /// Automatically select the best strategy based on the provider name.
    ///
    /// OpenAI-compatible providers (openai, groq, together, fireworks, deepseek,
    /// mistral, openrouter, perplexity) use native JSON mode. Anthropic uses
    /// forced tool use. All others fall back to prompt engineering.
    pub fn auto_strategy(mut self, provider_name: &str) -> Self {
        self.strategy = match provider_name.to_lowercase().as_str() {
            "openai" | "groq" | "together" | "fireworks" | "deepseek" | "mistral"
            | "openrouter" | "perplexity" => StructuredOutputStrategy::OpenAiNative,
            "anthropic" => StructuredOutputStrategy::AnthropicToolUse,
            _ => StructuredOutputStrategy::PromptEngineering,
        };
        self
    }

    /// Execute structured output with the selected strategy.
    ///
    /// The `generate_fn` closure receives:
    /// - `Option<&Value>`: OpenAI response_format parameter (for OpenAiNative)
    /// - `Option<(&Value, &Value)>`: Anthropic (tools, tool_choice) parameters (for AnthropicToolUse)
    /// - `&str`: the prompt text
    ///
    /// It should return the raw response string from the LLM, or an error string.
    pub fn execute(
        &self,
        generate_fn: &dyn Fn(
            Option<&serde_json::Value>,
            Option<(&serde_json::Value, &serde_json::Value)>,
            &str,
        ) -> Result<String, String>,
        prompt: &str,
    ) -> Result<serde_json::Value, StructuredOutputError> {
        match self.strategy {
            StructuredOutputStrategy::OpenAiNative => {
                let response_format = self.to_openai_response_format();
                let raw = generate_fn(Some(&response_format), None, prompt)
                    .map_err(|e| StructuredOutputError::GenerationFailed(e))?;
                let parsed: serde_json::Value = serde_json::from_str(&raw)
                    .map_err(|e| StructuredOutputError::ParseFailed(e.to_string()))?;
                self.validate_output(&parsed)?;
                Ok(parsed)
            }
            StructuredOutputStrategy::AnthropicToolUse => {
                let (tools, tool_choice) = self.to_anthropic_tool_params();
                let raw = generate_fn(None, Some((&tools, &tool_choice)), prompt)
                    .map_err(|e| StructuredOutputError::GenerationFailed(e))?;
                let response: serde_json::Value = serde_json::from_str(&raw)
                    .map_err(|e| StructuredOutputError::ParseFailed(e.to_string()))?;
                self.extract_from_anthropic_response(&response)
            }
            StructuredOutputStrategy::PromptEngineering => {
                let schema_json = serde_json::to_string_pretty(
                    &self.schema.to_openai_format(),
                )
                .unwrap_or_default();

                for attempt in 0..=self.max_retries {
                    let enhanced_prompt = format!(
                        "{}\n\nRespond with valid JSON matching this schema:\n{}",
                        prompt, schema_json
                    );
                    let raw = generate_fn(None, None, &enhanced_prompt)
                        .map_err(|e| StructuredOutputError::GenerationFailed(e))?;
                    // Try to extract JSON from response (may be in markdown code block)
                    if let Some(json_str) = extract_json_from_text(&raw) {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                            if self.validate_output(&parsed).is_ok() {
                                return Ok(parsed);
                            }
                        }
                    }
                    if attempt == self.max_retries {
                        return Err(StructuredOutputError::ValidationFailed(format!(
                            "Failed after {} attempts",
                            self.max_retries + 1
                        )));
                    }
                }
                unreachable!()
            }
        }
    }

    /// Validate a parsed JSON value against the schema.
    fn validate_output(&self, value: &serde_json::Value) -> Result<(), StructuredOutputError> {
        let result = SchemaValidator::validate(value, &self.schema);
        if result.valid {
            Ok(())
        } else {
            let error_msgs: Vec<String> = result.errors.iter().map(|e| e.to_string()).collect();
            Err(StructuredOutputError::ValidationFailed(
                error_msgs.join("; "),
            ))
        }
    }
}

/// Extract JSON from text that may contain markdown code blocks.
///
/// Tries in order:
/// 1. Direct parse as JSON
/// 2. ```json ... ``` fenced blocks
/// 3. Raw `{...}` or `[...]` substrings
fn extract_json_from_text(text: &str) -> Option<String> {
    // Try raw parse first
    let trimmed = text.trim();
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    // Then look for ```json ... ``` blocks
    if let Some(start) = text.find("```json") {
        let content_start = start + 7;
        let content_start = if text[content_start..].starts_with('\n') {
            content_start + 1
        } else {
            content_start
        };
        if let Some(end_offset) = text[content_start..].find("```") {
            let json_str = text[content_start..content_start + end_offset].trim();
            if !json_str.is_empty() {
                return Some(json_str.to_string());
            }
        }
    }

    // Then look for { ... } or [ ... ]
    if let Some(result) = find_balanced(text, '{', '}') {
        return Some(result.to_string());
    }
    if let Some(result) = find_balanced(text, '[', ']') {
        return Some(result.to_string());
    }

    None
}

/// Structured output request builder
pub struct StructuredRequest {
    schema: JsonSchema,
    prompt: String,
    examples: Vec<(String, Value)>,
    max_retries: usize,
}

impl StructuredRequest {
    /// Create a new structured request
    pub fn new(schema: JsonSchema) -> Self {
        Self {
            schema,
            prompt: String::new(),
            examples: Vec::new(),
            max_retries: 3,
        }
    }

    /// Set the user prompt
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Add an example
    pub fn with_example(mut self, input: impl Into<String>, output: Value) -> Self {
        self.examples.push((input.into(), output));
        self
    }

    /// Set max retries
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Build the full prompt
    pub fn build_prompt(&self) -> String {
        let mut prompt = self.prompt.clone();

        // Add examples if any
        if !self.examples.is_empty() {
            prompt.push_str("\n\nExamples:\n");
            for (i, (input, output)) in self.examples.iter().enumerate() {
                let output_str = serde_json::to_string_pretty(output).unwrap_or_default();
                prompt.push_str(&format!(
                    "\nExample {}:\nInput: {}\nOutput:\n```json\n{}\n```\n",
                    i + 1,
                    input,
                    output_str
                ));
            }
        }

        // Add schema
        prompt.push_str(&format!("\n{}", self.schema.to_prompt()));

        prompt
    }

    /// Get the schema
    pub fn schema(&self) -> &JsonSchema {
        &self.schema
    }

    /// Get max retries
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }
}

/// Configuration for the structured output enforcement loop
#[derive(Debug, Clone)]
pub struct EnforcementConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Whether to include response_format parameter for compatible APIs
    pub include_response_format: bool,
    /// Whether to include the full schema in the prompt
    pub include_schema_in_prompt: bool,
    /// Whether to include error feedback in retry prompts
    pub feedback_on_error: bool,
}

impl Default for EnforcementConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            include_response_format: true,
            include_schema_in_prompt: true,
            feedback_on_error: true,
        }
    }
}

/// Result of an enforcement loop execution
#[derive(Debug, Clone)]
pub struct EnforcementResult {
    /// The successfully parsed JSON value, if any
    pub value: Option<serde_json::Value>,
    /// Number of attempts made
    pub attempts: u32,
    /// Errors collected across all attempts
    pub errors: Vec<String>,
    /// Whether enforcement succeeded
    pub success: bool,
}

/// Extract JSON from an LLM response that may contain markdown fences or surrounding text.
///
/// Tries in order:
/// 1. ```json\n...\n``` fenced blocks
/// 2. ```\n...\n``` generic fenced blocks (if content looks like JSON)
/// 3. Raw JSON object `{...}` with brace-nesting tracking
/// 4. Raw JSON array `[...]` with bracket-nesting tracking
pub fn extract_json_from_response(response: &str) -> Option<&str> {
    // Try ```json ... ```
    if let Some(start_marker) = response.find("```json") {
        let content_start = start_marker + 7; // length of "```json"
                                              // Skip optional newline after ```json
        let content_start = if response[content_start..].starts_with('\n') {
            content_start + 1
        } else {
            content_start
        };
        if let Some(end_offset) = response[content_start..].find("```") {
            let json_str = response[content_start..content_start + end_offset].trim();
            if !json_str.is_empty() {
                return Some(json_str);
            }
        }
    }

    // Try ``` ... ``` (generic code block)
    if let Some(start_marker) = response.find("```") {
        let after_backticks = start_marker + 3;
        if let Some(end_offset) = response[after_backticks..].find("```") {
            let block = &response[after_backticks..after_backticks + end_offset];
            // Skip language identifier line if present
            let content = if let Some(newline_pos) = block.find('\n') {
                &block[newline_pos + 1..]
            } else {
                block
            };
            let trimmed = content.trim();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                return Some(trimmed);
            }
        }
    }

    // Try raw JSON object with nesting tracking
    if let Some(result) = find_balanced(response, '{', '}') {
        return Some(result);
    }

    // Try raw JSON array with nesting tracking
    if let Some(result) = find_balanced(response, '[', ']') {
        return Some(result);
    }

    None
}

/// Find a balanced pair of open/close characters in the string, tracking nesting depth.
fn find_balanced(s: &str, open: char, close: char) -> Option<&str> {
    let start = s.find(open)?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(&s[start..start + i + ch.len_utf8()]);
            }
        }
    }
    None
}

/// Enforces structured JSON output from LLM responses with a retry loop,
/// schema validation, and error feedback.
pub struct StructuredOutputEnforcer {
    generator: StructuredOutputGenerator,
    config: EnforcementConfig,
}

impl StructuredOutputEnforcer {
    /// Create a new enforcer wrapping a generator and enforcement configuration.
    pub fn new(generator: StructuredOutputGenerator, config: EnforcementConfig) -> Self {
        Self { generator, config }
    }

    /// Build a prompt that includes JSON constraint instructions.
    ///
    /// Uses the generator's `generate_prompt` if the schema exists, otherwise
    /// builds a basic constrained prompt from the user text alone.
    pub fn build_constrained_prompt(&self, schema_name: &str, user_prompt: &str) -> Option<String> {
        let base = if self.config.include_schema_in_prompt {
            self.generator.generate_prompt(schema_name, user_prompt)?
        } else {
            // Schema exists but caller chose not to embed it
            self.generator.get_schema(schema_name)?;
            user_prompt.to_string()
        };

        Some(format!(
            "{}\n\nYou MUST respond with valid JSON matching this schema. Do not include any text outside the JSON.",
            base
        ))
    }

    /// Validate a response string against the named schema, extracting JSON first.
    ///
    /// Returns `Ok(Value)` on success or `Err(Vec<String>)` with all error messages.
    pub fn validate_and_extract(
        &self,
        schema_name: &str,
        response: &str,
    ) -> Result<serde_json::Value, Vec<String>> {
        // Step 1: extract JSON substring
        let json_str = match extract_json_from_response(response) {
            Some(s) => s,
            None => {
                return Err(vec![
                    "No JSON found in response. Expected a JSON object or array.".to_string(),
                ]);
            }
        };

        // Step 2: parse
        let value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| vec![format!("JSON parse error: {}", e)])?;

        // Step 3: validate against schema if available
        if let Some(schema) = self.generator.get_schema(schema_name) {
            let result = SchemaValidator::validate(&value, schema);
            if result.valid {
                Ok(result.value.unwrap_or(value))
            } else {
                Err(result.errors.iter().map(|e| e.to_string()).collect())
            }
        } else {
            // No schema registered under that name — just return parsed value
            Ok(value)
        }
    }

    /// Build a retry prompt that includes error feedback from the previous attempt.
    pub fn build_retry_prompt(
        &self,
        schema_name: &str,
        user_prompt: &str,
        previous_response: &str,
        errors: &[String],
    ) -> String {
        let mut prompt = String::new();

        // Include original constrained prompt
        if let Some(constrained) = self.build_constrained_prompt(schema_name, user_prompt) {
            prompt.push_str(&constrained);
        } else {
            prompt.push_str(user_prompt);
        }

        if self.config.feedback_on_error && !errors.is_empty() {
            prompt.push_str("\n\nYour previous response was invalid:\n");
            prompt.push_str(&format!("```\n{}\n```\n", previous_response));
            prompt.push_str("\nErrors found:\n");
            for err in errors {
                prompt.push_str(&format!("- {}\n", err));
            }
            prompt.push_str("\nPlease correct these errors and respond with valid JSON only.");
        }

        prompt
    }

    /// Main enforcement loop: prompt -> validate -> retry if needed.
    ///
    /// The `response_generator` closure is called with a prompt string and should
    /// return the LLM's response text. The loop retries up to `config.max_retries`
    /// times on validation failure.
    pub fn enforce<F>(
        &self,
        schema_name: &str,
        user_prompt: &str,
        response_generator: &F,
    ) -> EnforcementResult
    where
        F: Fn(&str) -> String,
    {
        let mut all_errors: Vec<String> = Vec::new();
        let mut attempts = 0u32;
        let mut last_response = String::new();

        let max = self.config.max_retries.max(1); // at least one attempt

        for attempt in 0..max {
            attempts = attempt + 1;

            // Build the prompt
            let prompt = if attempt == 0 {
                match self.build_constrained_prompt(schema_name, user_prompt) {
                    Some(p) => p,
                    None => {
                        all_errors.push(format!("Schema '{}' not found", schema_name));
                        return EnforcementResult {
                            value: None,
                            attempts,
                            errors: all_errors,
                            success: false,
                        };
                    }
                }
            } else {
                self.build_retry_prompt(schema_name, user_prompt, &last_response, &all_errors)
            };

            // Generate response
            last_response = response_generator(&prompt);

            // Validate
            match self.validate_and_extract(schema_name, &last_response) {
                Ok(value) => {
                    return EnforcementResult {
                        value: Some(value),
                        attempts,
                        errors: all_errors,
                        success: true,
                    };
                }
                Err(errs) => {
                    all_errors.extend(errs);
                }
            }
        }

        // Exhausted retries
        EnforcementResult {
            value: None,
            attempts,
            errors: all_errors,
            success: false,
        }
    }

    /// Build an OpenAI-compatible `response_format` parameter for the named schema.
    ///
    /// If the schema is registered and `include_response_format` is enabled, returns
    /// a `json_schema` format with the full schema. Otherwise returns a basic
    /// `json_object` format, or `None` if response_format is disabled.
    pub fn build_response_format_param(&self, schema_name: &str) -> Option<serde_json::Value> {
        if !self.config.include_response_format {
            return None;
        }

        if let Some(schema) = self.generator.get_schema(schema_name) {
            Some(schema.to_openai_format())
        } else {
            Some(serde_json::json!({
                "type": "json_object"
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_property_builder() {
        let prop = SchemaProperty::string()
            .with_description("A name")
            .with_min_length(1)
            .with_max_length(100);

        assert_eq!(prop.schema_type, SchemaType::String);
        assert_eq!(prop.description, Some("A name".to_string()));
        assert_eq!(prop.min_length, Some(1));
        assert_eq!(prop.max_length, Some(100));
    }

    #[test]
    fn test_json_schema_builder() {
        let schema = JsonSchema::new("person")
            .with_description("A person object")
            .with_property(
                "name",
                SchemaProperty::string().with_description("Person's name"),
            )
            .with_property(
                "age",
                SchemaProperty::integer()
                    .with_minimum(0.0)
                    .with_maximum(150.0),
            )
            .with_required("name");

        assert_eq!(schema.name, "person");
        assert!(schema.description.is_some());
        assert!(schema
            .root
            .properties
            .as_ref()
            .unwrap()
            .contains_key("name"));
        assert!(schema
            .root
            .required
            .as_ref()
            .unwrap()
            .contains(&"name".to_string()));
    }

    #[test]
    fn test_validation_valid() {
        let schema = JsonSchema::new("test")
            .with_property("name", SchemaProperty::string())
            .with_property("age", SchemaProperty::integer())
            .with_required("name");

        let value = serde_json::json!({
            "name": "John",
            "age": 30
        });

        let result = SchemaValidator::validate(&value, &schema);
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_missing_required() {
        let schema = JsonSchema::new("test")
            .with_property("name", SchemaProperty::string())
            .with_required("name");

        let value = serde_json::json!({
            "age": 30
        });

        let result = SchemaValidator::validate(&value, &schema);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validation_wrong_type() {
        let schema = JsonSchema::new("test").with_property("age", SchemaProperty::integer());

        let value = serde_json::json!({
            "age": "thirty"
        });

        let result = SchemaValidator::validate(&value, &schema);
        assert!(!result.valid);
    }

    #[test]
    fn test_validation_enum() {
        let schema = JsonSchema::new("test").with_property(
            "color",
            SchemaProperty::string().with_enum(vec![
                "red".to_string(),
                "green".to_string(),
                "blue".to_string(),
            ]),
        );

        let valid = serde_json::json!({ "color": "red" });
        let invalid = serde_json::json!({ "color": "yellow" });

        assert!(SchemaValidator::validate(&valid, &schema).valid);
        assert!(!SchemaValidator::validate(&invalid, &schema).valid);
    }

    #[test]
    fn test_extract_json() {
        let with_code_block = "Here's the result:\n```json\n{\"name\": \"test\"}\n```\nDone!";
        assert_eq!(extract_json(with_code_block), "{\"name\": \"test\"}");

        let raw_json = "Sure! {\"name\": \"test\"} is the answer.";
        assert_eq!(extract_json(raw_json), "{\"name\": \"test\"}");
    }

    #[test]
    fn test_schema_builder_sentiment() {
        let schema = SchemaBuilder::sentiment_analysis();
        assert_eq!(schema.name, "sentiment_analysis");
        assert!(schema
            .root
            .properties
            .as_ref()
            .unwrap()
            .contains_key("sentiment"));
        assert!(schema
            .root
            .properties
            .as_ref()
            .unwrap()
            .contains_key("confidence"));
    }

    #[test]
    fn test_structured_output_generator() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(SchemaBuilder::sentiment_analysis());

        let prompt = generator
            .generate_prompt("sentiment_analysis", "Analyze this text")
            .unwrap();
        assert!(prompt.contains("Analyze this text"));
        assert!(prompt.contains("sentiment_analysis"));
    }

    #[test]
    fn test_structured_request() {
        let schema =
            SchemaBuilder::classification(vec!["spam".to_string(), "not_spam".to_string()]);
        let request = StructuredRequest::new(schema)
            .with_prompt("Classify this email")
            .with_example(
                "Buy now!",
                serde_json::json!({
                    "category": "spam",
                    "confidence": 0.95
                }),
            )
            .with_max_retries(5);

        let prompt = request.build_prompt();
        assert!(prompt.contains("Classify this email"));
        assert!(prompt.contains("Buy now!"));
        assert!(prompt.contains("spam"));
        assert_eq!(request.max_retries(), 5);
    }

    #[test]
    fn test_constrained_prompt_includes_json_instruction() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(
            JsonSchema::new("person")
                .with_property("name", SchemaProperty::string())
                .with_required("name"),
        );
        let enforcer = StructuredOutputEnforcer::new(generator, EnforcementConfig::default());

        let prompt = enforcer
            .build_constrained_prompt("person", "Describe a person")
            .unwrap();
        assert!(
            prompt.contains("JSON"),
            "Constrained prompt must contain JSON instruction"
        );
        assert!(prompt.contains("Describe a person"));
    }

    #[test]
    fn test_validate_valid_json() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(
            JsonSchema::new("item")
                .with_property("name", SchemaProperty::string())
                .with_property("count", SchemaProperty::integer())
                .with_required("name")
                .with_required("count"),
        );
        let enforcer = StructuredOutputEnforcer::new(generator, EnforcementConfig::default());

        let result = enforcer.validate_and_extract("item", r#"{"name": "apple", "count": 5}"#);
        assert!(result.is_ok(), "Valid JSON should pass validation");
        let val = result.unwrap();
        assert_eq!(val["name"], "apple");
        assert_eq!(val["count"], 5);
    }

    #[test]
    fn test_validate_invalid_json() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(
            JsonSchema::new("item")
                .with_property("name", SchemaProperty::string())
                .with_required("name"),
        );
        let enforcer = StructuredOutputEnforcer::new(generator, EnforcementConfig::default());

        // Completely non-JSON string
        let result = enforcer.validate_and_extract("item", "This is not JSON at all");
        assert!(result.is_err(), "Non-JSON string should fail validation");
        let errors = result.unwrap_err();
        assert!(!errors.is_empty(), "Should have error messages");
    }

    #[test]
    fn test_extract_json_from_markdown() {
        // Test ```json ... ``` fenced block
        let markdown = "Here is the result:\n```json\n{\"key\":\"value\"}\n```\nDone.";
        let extracted = extract_json_from_response(markdown);
        assert!(extracted.is_some());
        let parsed: serde_json::Value = serde_json::from_str(extracted.unwrap()).unwrap();
        assert_eq!(parsed["key"], "value");

        // Test raw JSON extraction
        let raw = "Sure, here: {\"key\":\"value\"} that's it.";
        let extracted_raw = extract_json_from_response(raw);
        assert!(extracted_raw.is_some());
        let parsed_raw: serde_json::Value = serde_json::from_str(extracted_raw.unwrap()).unwrap();
        assert_eq!(parsed_raw["key"], "value");
    }

    #[test]
    fn test_enforce_succeeds_first_try() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(
            JsonSchema::new("greeting")
                .with_property("message", SchemaProperty::string())
                .with_required("message"),
        );
        let enforcer = StructuredOutputEnforcer::new(generator, EnforcementConfig::default());

        let result = enforcer.enforce("greeting", "Say hello", &|_prompt: &str| {
            r#"{"message": "Hello, world!"}"#.to_string()
        });

        assert!(result.success);
        assert_eq!(result.attempts, 1);
        assert!(result.value.is_some());
        assert_eq!(result.value.unwrap()["message"], "Hello, world!");
    }

    #[test]
    fn test_enforce_retries_on_invalid() {
        let mut generator = StructuredOutputGenerator::new();
        generator.register_schema(
            JsonSchema::new("greeting")
                .with_property("message", SchemaProperty::string())
                .with_required("message"),
        );
        let config = EnforcementConfig {
            max_retries: 3,
            ..Default::default()
        };
        let enforcer = StructuredOutputEnforcer::new(generator, config);

        let call_count = std::cell::Cell::new(0u32);
        let result = enforcer.enforce("greeting", "Say hello", &|_prompt: &str| {
            let count = call_count.get() + 1;
            call_count.set(count);
            if count == 1 {
                // First attempt: invalid response (not JSON)
                "I don't know what to say".to_string()
            } else {
                // Second attempt: valid JSON
                r#"{"message": "Hello!"}"#.to_string()
            }
        });

        assert!(result.success);
        assert_eq!(result.attempts, 2);
        assert!(result.value.is_some());
        assert_eq!(result.value.unwrap()["message"], "Hello!");
        assert!(
            !result.errors.is_empty(),
            "Should have collected errors from first attempt"
        );
    }

    // ---- NEW TESTS (24+) ----

    #[test]
    fn test_schema_property_number_constructor() {
        let prop = SchemaProperty::number();
        assert_eq!(prop.schema_type, SchemaType::Number);
        assert!(prop.description.is_none());
        assert!(prop.minimum.is_none());
        assert!(prop.maximum.is_none());
        assert!(prop.items.is_none());
        assert!(prop.properties.is_none());
    }

    #[test]
    fn test_schema_property_integer_constructor() {
        let prop = SchemaProperty::integer();
        assert_eq!(prop.schema_type, SchemaType::Integer);
        assert!(prop.enum_values.is_none());
        assert!(prop.default.is_none());
    }

    #[test]
    fn test_schema_property_boolean_constructor() {
        let prop = SchemaProperty::boolean();
        assert_eq!(prop.schema_type, SchemaType::Boolean);
        assert!(prop.pattern.is_none());
        assert!(prop.items.is_none());
    }

    #[test]
    fn test_schema_property_array_constructor() {
        let items = SchemaProperty::string();
        let prop = SchemaProperty::array(items);
        assert_eq!(prop.schema_type, SchemaType::Array);
        assert!(prop.items.is_some());
        assert_eq!(prop.items.as_ref().unwrap().schema_type, SchemaType::String);
        assert!(prop.properties.is_none());
    }

    #[test]
    fn test_schema_property_object_constructor() {
        let prop = SchemaProperty::object();
        assert_eq!(prop.schema_type, SchemaType::Object);
        assert!(prop.properties.is_some());
        assert!(prop.properties.as_ref().unwrap().is_empty());
        assert!(prop.required.is_some());
        assert!(prop.required.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_schema_property_with_enum() {
        let prop =
            SchemaProperty::string().with_enum(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            prop.enum_values,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_schema_property_with_default() {
        let prop = SchemaProperty::string().with_default(serde_json::json!("hello"));
        assert_eq!(prop.default, Some(serde_json::json!("hello")));
    }

    #[test]
    fn test_schema_property_with_pattern() {
        let prop = SchemaProperty::string().with_pattern(r"^\d{3}-\d{4}$");
        assert_eq!(prop.pattern, Some(r"^\d{3}-\d{4}$".to_string()));
    }

    #[test]
    fn test_schema_property_with_property_on_non_object() {
        // with_property on a string property should be a no-op since properties is None
        let prop =
            SchemaProperty::string().with_property("child", SchemaProperty::integer());
        assert!(prop.properties.is_none());
    }

    #[test]
    fn test_schema_property_with_required_on_non_object() {
        // with_required on a non-object should be a no-op since required is None
        let prop = SchemaProperty::string().with_required("field");
        assert!(prop.required.is_none());
    }

    #[test]
    fn test_validation_number_min_max() {
        let schema = JsonSchema::new("test").with_property(
            "score",
            SchemaProperty::number().with_minimum(0.0).with_maximum(100.0),
        );

        let valid = serde_json::json!({ "score": 50.5 });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        let too_low = serde_json::json!({ "score": -1.0 });
        let result = SchemaValidator::validate(&too_low, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("minimum"));

        let too_high = serde_json::json!({ "score": 150.0 });
        let result = SchemaValidator::validate(&too_high, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("maximum"));
    }

    #[test]
    fn test_validation_string_length_constraints() {
        let schema = JsonSchema::new("test").with_property(
            "code",
            SchemaProperty::string().with_min_length(3).with_max_length(10),
        );

        let valid = serde_json::json!({ "code": "ABC123" });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        let too_short = serde_json::json!({ "code": "AB" });
        let result = SchemaValidator::validate(&too_short, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("minimum"));

        let too_long = serde_json::json!({ "code": "ABCDEFGHIJK" });
        let result = SchemaValidator::validate(&too_long, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("maximum"));
    }

    #[test]
    fn test_validation_array_length_constraints() {
        let schema = JsonSchema::new("test").with_property(
            "tags",
            SchemaProperty::array(SchemaProperty::string())
                .with_min_length(1)
                .with_max_length(3),
        );

        let valid = serde_json::json!({ "tags": ["a", "b"] });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        let too_few = serde_json::json!({ "tags": [] });
        let result = SchemaValidator::validate(&too_few, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("minimum"));

        let too_many = serde_json::json!({ "tags": ["a", "b", "c", "d"] });
        let result = SchemaValidator::validate(&too_many, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].message.contains("maximum"));
    }

    #[test]
    fn test_validation_array_item_types() {
        let schema = JsonSchema::new("test").with_property(
            "nums",
            SchemaProperty::array(SchemaProperty::integer()),
        );

        let valid = serde_json::json!({ "nums": [1, 2, 3] });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        // Array with a wrong-typed item
        let invalid = serde_json::json!({ "nums": [1, "two", 3] });
        let result = SchemaValidator::validate(&invalid, &schema);
        assert!(!result.valid);
        assert!(result.errors[0].path.contains("[1]"));
    }

    #[test]
    fn test_validation_boolean_type() {
        let schema = JsonSchema::new("test")
            .with_property("active", SchemaProperty::boolean());

        let valid = serde_json::json!({ "active": true });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        let invalid = serde_json::json!({ "active": "yes" });
        assert!(!SchemaValidator::validate(&invalid, &schema).valid);
    }

    #[test]
    fn test_validation_null_type() {
        let schema = JsonSchema::new("test").with_property(
            "nothing",
            SchemaProperty {
                schema_type: SchemaType::Null,
                description: None,
                enum_values: None,
                default: None,
                minimum: None,
                maximum: None,
                min_length: None,
                max_length: None,
                pattern: None,
                items: None,
                properties: None,
                required: None,
            },
        );

        let valid = serde_json::json!({ "nothing": null });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        let invalid = serde_json::json!({ "nothing": 0 });
        assert!(!SchemaValidator::validate(&invalid, &schema).valid);
    }

    #[test]
    fn test_validation_nested_object() {
        let inner = SchemaProperty::object()
            .with_property("street", SchemaProperty::string())
            .with_required("street");

        let schema = JsonSchema::new("test")
            .with_property("address", inner)
            .with_required("address");

        let valid = serde_json::json!({
            "address": { "street": "123 Main St" }
        });
        assert!(SchemaValidator::validate(&valid, &schema).valid);

        // Missing required nested field
        let invalid = serde_json::json!({
            "address": { "city": "Boston" }
        });
        let result = SchemaValidator::validate(&invalid, &schema);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.path.contains("street")));
    }

    #[test]
    fn test_validation_integer_rejects_float() {
        let schema = JsonSchema::new("test")
            .with_property("count", SchemaProperty::integer());

        // 3.5 is a float, not an integer
        let invalid = serde_json::json!({ "count": 3.5 });
        let result = SchemaValidator::validate(&invalid, &schema);
        assert!(!result.valid);
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError {
            path: "$.name".to_string(),
            message: "Required property missing".to_string(),
        };
        let display = format!("{}", err);
        assert_eq!(display, "$.name: Required property missing");
    }

    #[test]
    fn test_structured_output_error_display() {
        let gen = StructuredOutputError::GenerationFailed("timeout".to_string());
        assert_eq!(format!("{}", gen), "Generation failed: timeout");

        let parse = StructuredOutputError::ParseFailed("unexpected token".to_string());
        assert_eq!(format!("{}", parse), "Parse failed: unexpected token");

        let val = StructuredOutputError::ValidationFailed("missing field".to_string());
        assert_eq!(format!("{}", val), "Validation failed: missing field");

        let ext = StructuredOutputError::ExtractionFailed("no content".to_string());
        assert_eq!(format!("{}", ext), "Extraction failed: no content");
    }

    #[test]
    fn test_schema_type_serde_roundtrip() {
        let types = vec![
            SchemaType::String,
            SchemaType::Number,
            SchemaType::Integer,
            SchemaType::Boolean,
            SchemaType::Array,
            SchemaType::Object,
            SchemaType::Null,
        ];
        for st in &types {
            let json = serde_json::to_string(st).unwrap();
            let parsed: SchemaType = serde_json::from_str(&json).unwrap();
            assert_eq!(*st, parsed);
        }
    }

    #[test]
    fn test_extract_json_from_generic_code_block() {
        let input = "Result:\n```\n{\"a\": 1}\n```\nEnd";
        let result = extract_json(input);
        assert_eq!(result, r#"{"a": 1}"#);
    }

    #[test]
    fn test_extract_json_raw_array() {
        let input = "Here is the list: [1, 2, 3] done";
        let result = extract_json(input);
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn test_extract_json_no_json_returns_original() {
        let input = "No JSON here at all";
        let result = extract_json(input);
        assert_eq!(result, "No JSON here at all");
    }

    #[test]
    fn test_extract_json_from_response_array() {
        let input = "Here: [1, 2, 3] done.";
        let extracted = extract_json_from_response(input);
        assert!(extracted.is_some());
        assert_eq!(extracted.unwrap(), "[1, 2, 3]");
    }

    #[test]
    fn test_extract_json_from_response_no_json() {
        let input = "This text has no JSON at all.";
        assert!(extract_json_from_response(input).is_none());
    }

    #[test]
    fn test_extract_json_from_text_direct_parse() {
        let input = r#"{"key": "value"}"#;
        let result = extract_json_from_text(input);
        assert_eq!(result, Some(r#"{"key": "value"}"#.to_string()));
    }

    #[test]
    fn test_extract_json_from_text_no_json() {
        let input = "plain text without any brackets";
        assert!(extract_json_from_text(input).is_none());
    }

    #[test]
    fn test_find_balanced_nested_braces() {
        let input = r#"prefix {"a": {"b": "c"}} suffix"#;
        let result = find_balanced(input, '{', '}');
        assert_eq!(result, Some(r#"{"a": {"b": "c"}}"#));
    }

    #[test]
    fn test_find_balanced_with_escaped_quotes() {
        let input = r#"{"msg": "say \"hello\""}"#;
        let result = find_balanced(input, '{', '}');
        assert_eq!(result, Some(r#"{"msg": "say \"hello\""}"#));
    }

    #[test]
    fn test_find_balanced_no_match() {
        assert!(find_balanced("no braces here", '{', '}').is_none());
    }

    #[test]
    fn test_find_balanced_unbalanced() {
        // Opening brace without matching close
        assert!(find_balanced("{unclosed", '{', '}').is_none());
    }

    #[test]
    fn test_schema_builder_entity_extraction() {
        let schema = SchemaBuilder::entity_extraction();
        assert_eq!(schema.name, "entity_extraction");
        let props = schema.root.properties.as_ref().unwrap();
        assert!(props.contains_key("entities"));
        let entities_prop = &props["entities"];
        assert_eq!(entities_prop.schema_type, SchemaType::Array);
        assert!(entities_prop.items.is_some());
    }

    #[test]
    fn test_schema_builder_summary() {
        let schema = SchemaBuilder::summary();
        assert_eq!(schema.name, "summary");
        let props = schema.root.properties.as_ref().unwrap();
        assert!(props.contains_key("summary"));
        assert!(props.contains_key("key_points"));
        assert!(props.contains_key("title"));
        let req = schema.root.required.as_ref().unwrap();
        assert!(req.contains(&"summary".to_string()));
        assert!(req.contains(&"key_points".to_string()));
    }

    #[test]
    fn test_schema_builder_translation() {
        let schema = SchemaBuilder::translation();
        assert_eq!(schema.name, "translation");
        let props = schema.root.properties.as_ref().unwrap();
        assert!(props.contains_key("original_language"));
        assert!(props.contains_key("target_language"));
        assert!(props.contains_key("translation"));
        assert!(props.contains_key("alternatives"));
        let req = schema.root.required.as_ref().unwrap();
        assert!(req.contains(&"translation".to_string()));
    }

    #[test]
    fn test_schema_builder_question_answer() {
        let schema = SchemaBuilder::question_answer();
        assert_eq!(schema.name, "question_answer");
        let props = schema.root.properties.as_ref().unwrap();
        assert!(props.contains_key("answer"));
        assert!(props.contains_key("confidence"));
        assert!(props.contains_key("sources"));
        assert!(props.contains_key("follow_up_questions"));
    }

    #[test]
    fn test_schema_builder_classification() {
        let cats = vec!["tech".to_string(), "sports".to_string()];
        let schema = SchemaBuilder::classification(cats.clone());
        assert_eq!(schema.name, "classification");
        let props = schema.root.properties.as_ref().unwrap();
        let cat_prop = &props["category"];
        assert_eq!(cat_prop.enum_values.as_ref().unwrap(), &cats);
    }

    #[test]
    fn test_generator_default() {
        let gen = StructuredOutputGenerator::default();
        assert!(gen.get_schema("nonexistent").is_none());
    }

    #[test]
    fn test_generator_get_schema_not_found() {
        let gen = StructuredOutputGenerator::new();
        assert!(gen.get_schema("missing").is_none());
        assert!(gen.generate_prompt("missing", "hello").is_none());
    }

    #[test]
    fn test_generator_parse_response_valid() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("name", SchemaProperty::string())
                .with_required("name"),
        );

        let result = gen
            .parse_response("item", r#"{"name": "widget"}"#)
            .unwrap();
        assert!(result.valid);
        assert_eq!(result.value.unwrap()["name"], "widget");
    }

    #[test]
    fn test_generator_parse_response_schema_not_found() {
        let gen = StructuredOutputGenerator::new();
        let result = gen.parse_response("missing", r#"{"a":1}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_generator_parse_response_invalid_json() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(JsonSchema::new("item"));
        let result = gen.parse_response("item", "not json at all");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON parse error"));
    }

    #[test]
    fn test_parse_with_hints_success() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("x", SchemaProperty::integer())
                .with_required("x"),
        );

        let result = gen.parse_with_hints("item", r#"{"x": 42}"#);
        assert!(result.success);
        assert!(result.value.is_some());
        assert!(result.retry_prompt.is_none());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_parse_with_hints_schema_not_found() {
        let gen = StructuredOutputGenerator::new();
        let result = gen.parse_with_hints("missing", r#"{"x": 1}"#);
        assert!(!result.success);
        assert!(result.errors[0].contains("Schema not found"));
        assert!(result.retry_prompt.is_none());
    }

    #[test]
    fn test_parse_with_hints_invalid_json() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(JsonSchema::new("item"));
        let result = gen.parse_with_hints("item", "garbage text");
        assert!(!result.success);
        assert!(result.retry_prompt.is_some());
        let retry = result.retry_prompt.unwrap();
        assert!(retry.contains("Could not parse"));
    }

    #[test]
    fn test_parse_with_hints_validation_failure() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("x", SchemaProperty::integer())
                .with_required("x"),
        );

        // Valid JSON but missing required field
        let result = gen.parse_with_hints("item", r#"{"y": 1}"#);
        assert!(!result.success);
        assert!(result.retry_prompt.is_some());
        let retry = result.retry_prompt.unwrap();
        assert!(retry.contains("validation errors"));
    }

    #[test]
    fn test_json_schema_to_prompt() {
        let schema = JsonSchema::new("demo")
            .with_description("A demo schema")
            .with_property("val", SchemaProperty::string());

        let prompt = schema.to_prompt();
        assert!(prompt.contains("Respond with a JSON object"));
        assert!(prompt.contains("```json"));
        assert!(prompt.contains("Respond ONLY with valid JSON"));
        assert!(prompt.contains("demo"));
    }

    #[test]
    fn test_json_schema_to_openai_format() {
        let schema = JsonSchema::new("test_fmt")
            .with_property("a", SchemaProperty::string());

        let fmt = schema.to_openai_format();
        assert_eq!(fmt["type"], "json_schema");
        assert_eq!(fmt["json_schema"]["name"], "test_fmt");
        assert_eq!(fmt["json_schema"]["strict"], true);
    }

    #[test]
    fn test_structured_output_request_defaults() {
        let schema = JsonSchema::new("req_test");
        let req = StructuredOutputRequest::new(schema);
        assert_eq!(
            *req.strategy(),
            StructuredOutputStrategy::PromptEngineering
        );
    }

    #[test]
    fn test_structured_output_request_builder_methods() {
        let schema = JsonSchema::new("req_test");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::OpenAiNative)
            .with_max_retries(5)
            .strict(false);

        assert_eq!(*req.strategy(), StructuredOutputStrategy::OpenAiNative);
        let fmt = req.to_openai_response_format();
        assert_eq!(fmt["json_schema"]["strict"], false);
    }

    #[test]
    fn test_structured_output_request_auto_strategy() {
        let schema = JsonSchema::new("s");
        let req = StructuredOutputRequest::new(schema).auto_strategy("openai");
        assert_eq!(*req.strategy(), StructuredOutputStrategy::OpenAiNative);

        let schema2 = JsonSchema::new("s2");
        let req2 = StructuredOutputRequest::new(schema2).auto_strategy("anthropic");
        assert_eq!(*req2.strategy(), StructuredOutputStrategy::AnthropicToolUse);

        let schema3 = JsonSchema::new("s3");
        let req3 = StructuredOutputRequest::new(schema3).auto_strategy("ollama");
        assert_eq!(
            *req3.strategy(),
            StructuredOutputStrategy::PromptEngineering
        );

        // Case-insensitive
        let schema4 = JsonSchema::new("s4");
        let req4 = StructuredOutputRequest::new(schema4).auto_strategy("GROQ");
        assert_eq!(*req4.strategy(), StructuredOutputStrategy::OpenAiNative);
    }

    #[test]
    fn test_anthropic_tool_params() {
        let schema = JsonSchema::new("my_tool").with_description("Do stuff");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::AnthropicToolUse);

        let (tools, tool_choice) = req.to_anthropic_tool_params();
        assert!(tools.is_array());
        assert_eq!(tools[0]["name"], "my_tool");
        assert_eq!(tools[0]["description"], "Do stuff");
        assert_eq!(tool_choice["type"], "tool");
        assert_eq!(tool_choice["name"], "my_tool");
    }

    #[test]
    fn test_anthropic_tool_params_no_description() {
        let schema = JsonSchema::new("no_desc");
        let req = StructuredOutputRequest::new(schema);
        let (tools, _) = req.to_anthropic_tool_params();
        assert_eq!(
            tools[0]["description"],
            "Generate structured output matching the schema"
        );
    }

    #[test]
    fn test_extract_from_anthropic_response_success() {
        let schema = JsonSchema::new("t")
            .with_property("val", SchemaProperty::string())
            .with_required("val");
        let req = StructuredOutputRequest::new(schema);

        let response = serde_json::json!({
            "content": [
                {
                    "type": "tool_use",
                    "name": "t",
                    "input": { "val": "hello" }
                }
            ]
        });

        let result = req.extract_from_anthropic_response(&response);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["val"], "hello");
    }

    #[test]
    fn test_extract_from_anthropic_response_no_content() {
        let schema = JsonSchema::new("t");
        let req = StructuredOutputRequest::new(schema);

        let response = serde_json::json!({"data": "nope"});
        let result = req.extract_from_anthropic_response(&response);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Extraction failed"));
    }

    #[test]
    fn test_extract_from_anthropic_response_no_tool_use_block() {
        let schema = JsonSchema::new("t");
        let req = StructuredOutputRequest::new(schema);

        let response = serde_json::json!({
            "content": [
                { "type": "text", "text": "Hello" }
            ]
        });
        let result = req.extract_from_anthropic_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_from_anthropic_response_missing_input() {
        let schema = JsonSchema::new("t");
        let req = StructuredOutputRequest::new(schema);

        let response = serde_json::json!({
            "content": [
                { "type": "tool_use", "name": "t" }
            ]
        });
        let result = req.extract_from_anthropic_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_enforcement_config_default() {
        let cfg = EnforcementConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert!(cfg.include_response_format);
        assert!(cfg.include_schema_in_prompt);
        assert!(cfg.feedback_on_error);
    }

    #[test]
    fn test_enforcer_build_response_format_param_with_schema() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("a", SchemaProperty::string()),
        );
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        let param = enforcer.build_response_format_param("item");
        assert!(param.is_some());
        let val = param.unwrap();
        assert_eq!(val["type"], "json_schema");
    }

    #[test]
    fn test_enforcer_build_response_format_param_no_schema() {
        let gen = StructuredOutputGenerator::new();
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        let param = enforcer.build_response_format_param("missing");
        assert!(param.is_some());
        assert_eq!(param.unwrap()["type"], "json_object");
    }

    #[test]
    fn test_enforcer_build_response_format_param_disabled() {
        let gen = StructuredOutputGenerator::new();
        let config = EnforcementConfig {
            include_response_format: false,
            ..Default::default()
        };
        let enforcer = StructuredOutputEnforcer::new(gen, config);

        assert!(enforcer.build_response_format_param("any").is_none());
    }

    #[test]
    fn test_enforcer_constrained_prompt_missing_schema() {
        let gen = StructuredOutputGenerator::new();
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        assert!(enforcer
            .build_constrained_prompt("nonexistent", "hello")
            .is_none());
    }

    #[test]
    fn test_enforcer_constrained_prompt_schema_not_in_prompt() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("x", SchemaProperty::integer()),
        );
        let config = EnforcementConfig {
            include_schema_in_prompt: false,
            ..Default::default()
        };
        let enforcer = StructuredOutputEnforcer::new(gen, config);

        let prompt = enforcer.build_constrained_prompt("item", "Give me an item");
        assert!(prompt.is_some());
        let p = prompt.unwrap();
        // Should still contain JSON instruction but is based on user prompt only
        assert!(p.contains("Give me an item"));
        assert!(p.contains("JSON"));
    }

    #[test]
    fn test_enforcer_build_retry_prompt() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("name", SchemaProperty::string())
                .with_required("name"),
        );
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        let retry = enforcer.build_retry_prompt(
            "item",
            "Give me an item",
            "bad response",
            &["missing name".to_string()],
        );
        assert!(retry.contains("Give me an item"));
        assert!(retry.contains("bad response"));
        assert!(retry.contains("missing name"));
        assert!(retry.contains("correct these errors"));
    }

    #[test]
    fn test_enforcer_enforce_schema_not_found() {
        let gen = StructuredOutputGenerator::new();
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        let result = enforcer.enforce("missing", "test", &|_| "{}".to_string());
        assert!(!result.success);
        assert!(result.errors.iter().any(|e| e.contains("not found")));
        assert_eq!(result.attempts, 1);
    }

    #[test]
    fn test_enforcer_enforce_exhausts_retries() {
        let mut gen = StructuredOutputGenerator::new();
        gen.register_schema(
            JsonSchema::new("item")
                .with_property("name", SchemaProperty::string())
                .with_required("name"),
        );
        let config = EnforcementConfig {
            max_retries: 2,
            ..Default::default()
        };
        let enforcer = StructuredOutputEnforcer::new(gen, config);

        // Always return invalid response (missing required 'name')
        let result = enforcer.enforce("item", "test", &|_| r#"{"other": 1}"#.to_string());
        assert!(!result.success);
        assert!(result.value.is_none());
        assert_eq!(result.attempts, 2);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_structured_request_prompt_no_examples() {
        let schema = JsonSchema::new("simple")
            .with_property("a", SchemaProperty::string());
        let req = StructuredRequest::new(schema).with_prompt("Do something");

        let prompt = req.build_prompt();
        assert!(prompt.contains("Do something"));
        assert!(!prompt.contains("Examples:"));
        assert!(prompt.contains("Respond ONLY with valid JSON"));
    }

    #[test]
    fn test_structured_request_schema_accessor() {
        let schema = JsonSchema::new("accessor_test");
        let req = StructuredRequest::new(schema);
        assert_eq!(req.schema().name, "accessor_test");
        assert_eq!(req.max_retries(), 3); // default
    }

    #[test]
    fn test_structured_output_strategy_equality() {
        assert_eq!(
            StructuredOutputStrategy::OpenAiNative,
            StructuredOutputStrategy::OpenAiNative
        );
        assert_ne!(
            StructuredOutputStrategy::OpenAiNative,
            StructuredOutputStrategy::AnthropicToolUse
        );
        assert_ne!(
            StructuredOutputStrategy::AnthropicToolUse,
            StructuredOutputStrategy::PromptEngineering
        );
    }

    #[test]
    fn test_execute_openai_native_strategy() {
        let schema = JsonSchema::new("exec")
            .with_property("x", SchemaProperty::integer())
            .with_required("x");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::OpenAiNative);

        let result = req.execute(
            &|response_format, _anthropic, _prompt| {
                // Verify response_format is provided
                assert!(response_format.is_some());
                Ok(r#"{"x": 42}"#.to_string())
            },
            "Give me x",
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap()["x"], 42);
    }

    #[test]
    fn test_execute_anthropic_strategy() {
        let schema = JsonSchema::new("exec_ant")
            .with_property("y", SchemaProperty::string())
            .with_required("y");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::AnthropicToolUse);

        let result = req.execute(
            &|_rf, anthropic_params, _prompt| {
                // Verify anthropic params are provided
                assert!(anthropic_params.is_some());
                // Return an Anthropic-style tool_use response
                Ok(serde_json::json!({
                    "content": [{
                        "type": "tool_use",
                        "name": "exec_ant",
                        "input": { "y": "hello" }
                    }]
                })
                .to_string())
            },
            "Give me y",
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap()["y"], "hello");
    }

    #[test]
    fn test_execute_prompt_engineering_strategy() {
        let schema = JsonSchema::new("exec_pe")
            .with_property("z", SchemaProperty::boolean())
            .with_required("z");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::PromptEngineering)
            .with_max_retries(1);

        let result = req.execute(
            &|rf, ant, _prompt| {
                // Neither response_format nor anthropic params
                assert!(rf.is_none());
                assert!(ant.is_none());
                Ok(r#"{"z": true}"#.to_string())
            },
            "Give me z",
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap()["z"], true);
    }

    #[test]
    fn test_execute_generation_failure() {
        let schema = JsonSchema::new("fail");
        let req = StructuredOutputRequest::new(schema)
            .with_strategy(StructuredOutputStrategy::OpenAiNative);

        let result = req.execute(
            &|_, _, _| Err("network timeout".to_string()),
            "prompt",
        );

        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Generation failed"));
    }

    #[test]
    fn test_validation_result_success_and_failure() {
        let success = ValidationResult::success(serde_json::json!({"ok": true}));
        assert!(success.valid);
        assert!(success.errors.is_empty());
        assert!(success.value.is_some());

        let failure = ValidationResult::failure(vec![ValidationError {
            path: "$".to_string(),
            message: "bad".to_string(),
        }]);
        assert!(!failure.valid);
        assert_eq!(failure.errors.len(), 1);
        assert!(failure.value.is_none());
    }

    #[test]
    fn test_enforcer_validate_and_extract_no_schema() {
        let gen = StructuredOutputGenerator::new();
        let enforcer = StructuredOutputEnforcer::new(gen, EnforcementConfig::default());

        // Schema not registered — should still return parsed value
        let result = enforcer.validate_and_extract("unknown", r#"{"a": 1}"#);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["a"], 1);
    }

    #[test]
    fn test_find_balanced_braces_inside_strings() {
        // Braces inside JSON string values should be ignored
        let input = r#"{"msg": "use {braces} freely"}"#;
        let result = find_balanced(input, '{', '}');
        assert_eq!(result, Some(r#"{"msg": "use {braces} freely"}"#));
    }

    #[test]
    fn test_extract_json_from_response_generic_code_block() {
        let markdown = "```python\n{\"key\": \"val\"}\n```";
        let extracted = extract_json_from_response(markdown);
        assert!(extracted.is_some());
        let parsed: serde_json::Value =
            serde_json::from_str(extracted.unwrap()).unwrap();
        assert_eq!(parsed["key"], "val");
    }
}
