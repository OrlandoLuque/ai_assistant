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
}
