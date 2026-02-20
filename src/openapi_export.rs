//! OpenAPI and JSON Schema export functionality
//!
//! This module provides utilities for exporting API definitions
//! in OpenAPI 3.0 and JSON Schema formats.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAPI 3.0 specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiSpec {
    pub openapi: String,
    pub info: OpenApiInfo,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub servers: Vec<OpenApiServer>,
    pub paths: HashMap<String, OpenApiPathItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub components: Option<OpenApiComponents>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tags: Vec<OpenApiTag>,
}

impl Default for OpenApiSpec {
    fn default() -> Self {
        Self {
            openapi: "3.0.3".to_string(),
            info: OpenApiInfo::default(),
            servers: Vec::new(),
            paths: HashMap::new(),
            components: None,
            tags: Vec::new(),
        }
    }
}

/// API info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiInfo {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact: Option<OpenApiContact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<OpenApiLicense>,
}

impl Default for OpenApiInfo {
    fn default() -> Self {
        Self {
            title: "AI Assistant API".to_string(),
            description: Some("Local LLM provider API".to_string()),
            version: "1.0.0".to_string(),
            contact: None,
            license: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiContact {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiLicense {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiServer {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiTag {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenApiPathItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get: Option<OpenApiOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post: Option<OpenApiOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub put: Option<OpenApiOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delete: Option<OpenApiOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch: Option<OpenApiOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiOperation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "operationId")]
    pub operation_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub parameters: Vec<OpenApiParameter>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "requestBody")]
    pub request_body: Option<OpenApiRequestBody>,
    pub responses: HashMap<String, OpenApiResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiParameter {
    pub name: String,
    #[serde(rename = "in")]
    pub location: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
    pub schema: JsonSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiRequestBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub content: HashMap<String, OpenApiMediaType>,
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiMediaType {
    pub schema: JsonSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiResponse {
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<HashMap<String, OpenApiMediaType>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenApiComponents {
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub schemas: HashMap<String, JsonSchema>,
    #[serde(
        skip_serializing_if = "HashMap::is_empty",
        default,
        rename = "securitySchemes"
    )]
    pub security_schemes: HashMap<String, OpenApiSecurityScheme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenApiSecurityScheme {
    #[serde(rename = "type")]
    pub scheme_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "in")]
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
}

/// JSON Schema definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JsonSchema {
    #[serde(skip_serializing_if = "Option::is_none", rename = "$ref")]
    pub reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub schema_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "default")]
    pub default_value: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "minLength")]
    pub min_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxLength")]
    pub max_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "additionalProperties"
    )]
    pub additional_properties: Option<Box<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "oneOf")]
    pub one_of: Option<Vec<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "anyOf")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "allOf")]
    pub all_of: Option<Vec<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
}

impl JsonSchema {
    /// Create a string schema
    pub fn string() -> Self {
        Self {
            schema_type: Some("string".to_string()),
            ..Default::default()
        }
    }

    /// Create an integer schema
    pub fn integer() -> Self {
        Self {
            schema_type: Some("integer".to_string()),
            ..Default::default()
        }
    }

    /// Create a number schema
    pub fn number() -> Self {
        Self {
            schema_type: Some("number".to_string()),
            ..Default::default()
        }
    }

    /// Create a boolean schema
    pub fn boolean() -> Self {
        Self {
            schema_type: Some("boolean".to_string()),
            ..Default::default()
        }
    }

    /// Create an array schema
    pub fn array(items: JsonSchema) -> Self {
        Self {
            schema_type: Some("array".to_string()),
            items: Some(Box::new(items)),
            ..Default::default()
        }
    }

    /// Create an object schema
    pub fn object() -> Self {
        Self {
            schema_type: Some("object".to_string()),
            properties: Some(HashMap::new()),
            ..Default::default()
        }
    }

    /// Create a reference
    pub fn reference(path: impl Into<String>) -> Self {
        Self {
            reference: Some(path.into()),
            ..Default::default()
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add property to object
    pub fn with_property(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        if self.properties.is_none() {
            self.properties = Some(HashMap::new());
        }
        self.properties
            .as_mut()
            .expect("properties must be initialized")
            .insert(name.into(), schema);
        self
    }

    /// Mark property as required
    pub fn with_required(mut self, names: Vec<&str>) -> Self {
        self.required = Some(names.into_iter().map(|s| s.to_string()).collect());
        self
    }

    /// Add format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add enum values
    pub fn with_enum(mut self, values: Vec<serde_json::Value>) -> Self {
        self.enum_values = Some(values);
        self
    }
}

/// Builder for OpenAPI specs
pub struct OpenApiBuilder {
    spec: OpenApiSpec,
}

impl Default for OpenApiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenApiBuilder {
    pub fn new() -> Self {
        Self {
            spec: OpenApiSpec::default(),
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.spec.info.title = title.into();
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.spec.info.description = Some(desc.into());
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.spec.info.version = version.into();
        self
    }

    pub fn server(mut self, url: impl Into<String>, description: Option<&str>) -> Self {
        self.spec.servers.push(OpenApiServer {
            url: url.into(),
            description: description.map(|s| s.to_string()),
        });
        self
    }

    pub fn tag(mut self, name: impl Into<String>, description: Option<&str>) -> Self {
        self.spec.tags.push(OpenApiTag {
            name: name.into(),
            description: description.map(|s| s.to_string()),
        });
        self
    }

    pub fn path(mut self, path: impl Into<String>, item: OpenApiPathItem) -> Self {
        self.spec.paths.insert(path.into(), item);
        self
    }

    pub fn schema(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        if self.spec.components.is_none() {
            self.spec.components = Some(OpenApiComponents::default());
        }
        self.spec
            .components
            .as_mut()
            .expect("components must be initialized")
            .schemas
            .insert(name.into(), schema);
        self
    }

    pub fn build(self) -> OpenApiSpec {
        self.spec
    }
}

/// Operation builder
pub struct OperationBuilder {
    operation: OpenApiOperation,
}

impl Default for OperationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationBuilder {
    pub fn new() -> Self {
        Self {
            operation: OpenApiOperation {
                summary: None,
                description: None,
                operation_id: None,
                tags: Vec::new(),
                parameters: Vec::new(),
                request_body: None,
                responses: HashMap::new(),
            },
        }
    }

    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.operation.summary = Some(summary.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.operation.description = Some(desc.into());
        self
    }

    pub fn operation_id(mut self, id: impl Into<String>) -> Self {
        self.operation.operation_id = Some(id.into());
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.operation.tags.push(tag.into());
        self
    }

    pub fn query_param(
        mut self,
        name: impl Into<String>,
        schema: JsonSchema,
        required: bool,
    ) -> Self {
        self.operation.parameters.push(OpenApiParameter {
            name: name.into(),
            location: "query".to_string(),
            description: schema.description.clone(),
            required,
            schema,
        });
        self
    }

    pub fn path_param(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        self.operation.parameters.push(OpenApiParameter {
            name: name.into(),
            location: "path".to_string(),
            description: schema.description.clone(),
            required: true,
            schema,
        });
        self
    }

    pub fn request_body(mut self, schema: JsonSchema, required: bool) -> Self {
        let mut content = HashMap::new();
        content.insert(
            "application/json".to_string(),
            OpenApiMediaType {
                schema,
                example: None,
            },
        );
        self.operation.request_body = Some(OpenApiRequestBody {
            description: None,
            content,
            required,
        });
        self
    }

    pub fn response(
        mut self,
        status: &str,
        description: impl Into<String>,
        schema: Option<JsonSchema>,
    ) -> Self {
        let content = schema.map(|s| {
            let mut map = HashMap::new();
            map.insert(
                "application/json".to_string(),
                OpenApiMediaType {
                    schema: s,
                    example: None,
                },
            );
            map
        });
        self.operation.responses.insert(
            status.to_string(),
            OpenApiResponse {
                description: description.into(),
                content,
            },
        );
        self
    }

    pub fn build(self) -> OpenApiOperation {
        self.operation
    }
}

/// Generate OpenAPI spec for AI Assistant
pub fn generate_ai_assistant_spec() -> OpenApiSpec {
    OpenApiBuilder::new()
        .title("AI Assistant API")
        .description("Local LLM provider API for chat completions, embeddings, and more")
        .version("1.0.0")
        .server("http://localhost:11434", Some("Ollama"))
        .server("http://localhost:1234", Some("LM Studio"))
        .tag("chat", Some("Chat completion endpoints"))
        .tag("models", Some("Model management"))
        .tag("embeddings", Some("Text embeddings"))
        // Chat completion endpoint
        .path(
            "/api/chat",
            OpenApiPathItem {
                post: Some(
                    OperationBuilder::new()
                        .summary("Generate chat completion")
                        .description("Send messages and receive AI responses")
                        .operation_id("createChatCompletion")
                        .tag("chat")
                        .request_body(
                            JsonSchema::object()
                                .with_property(
                                    "model",
                                    JsonSchema::string().with_description("Model name"),
                                )
                                .with_property(
                                    "messages",
                                    JsonSchema::array(
                                        JsonSchema::object()
                                            .with_property("role", JsonSchema::string())
                                            .with_property("content", JsonSchema::string()),
                                    ),
                                )
                                .with_property("stream", JsonSchema::boolean())
                                .with_property("temperature", JsonSchema::number())
                                .with_required(vec!["model", "messages"]),
                            true,
                        )
                        .response(
                            "200",
                            "Successful response",
                            Some(
                                JsonSchema::object()
                                    .with_property("model", JsonSchema::string())
                                    .with_property(
                                        "message",
                                        JsonSchema::object()
                                            .with_property("role", JsonSchema::string())
                                            .with_property("content", JsonSchema::string()),
                                    )
                                    .with_property("done", JsonSchema::boolean()),
                            ),
                        )
                        .response("400", "Bad request", None)
                        .response("500", "Server error", None)
                        .build(),
                ),
                ..Default::default()
            },
        )
        // List models endpoint
        .path(
            "/api/tags",
            OpenApiPathItem {
                get: Some(
                    OperationBuilder::new()
                        .summary("List available models")
                        .operation_id("listModels")
                        .tag("models")
                        .response(
                            "200",
                            "List of models",
                            Some(
                                JsonSchema::object().with_property(
                                    "models",
                                    JsonSchema::array(
                                        JsonSchema::object()
                                            .with_property("name", JsonSchema::string())
                                            .with_property("size", JsonSchema::integer())
                                            .with_property(
                                                "modified_at",
                                                JsonSchema::string().with_format("date-time"),
                                            ),
                                    ),
                                ),
                            ),
                        )
                        .build(),
                ),
                ..Default::default()
            },
        )
        // Embeddings endpoint
        .path(
            "/api/embeddings",
            OpenApiPathItem {
                post: Some(
                    OperationBuilder::new()
                        .summary("Generate embeddings")
                        .operation_id("createEmbeddings")
                        .tag("embeddings")
                        .request_body(
                            JsonSchema::object()
                                .with_property("model", JsonSchema::string())
                                .with_property("prompt", JsonSchema::string())
                                .with_required(vec!["model", "prompt"]),
                            true,
                        )
                        .response(
                            "200",
                            "Embedding vector",
                            Some(JsonSchema::object().with_property(
                                "embedding",
                                JsonSchema::array(JsonSchema::number()),
                            )),
                        )
                        .build(),
                ),
                ..Default::default()
            },
        )
        // Common schemas
        .schema(
            "Message",
            JsonSchema::object()
                .with_property(
                    "role",
                    JsonSchema::string().with_enum(vec![
                        serde_json::json!("system"),
                        serde_json::json!("user"),
                        serde_json::json!("assistant"),
                    ]),
                )
                .with_property("content", JsonSchema::string())
                .with_required(vec!["role", "content"]),
        )
        .schema(
            "Model",
            JsonSchema::object()
                .with_property("name", JsonSchema::string())
                .with_property("size", JsonSchema::integer())
                .with_property("modified_at", JsonSchema::string().with_format("date-time")),
        )
        .schema(
            "Error",
            JsonSchema::object()
                .with_property("error", JsonSchema::string())
                .with_property("message", JsonSchema::string())
                .with_required(vec!["error"]),
        )
        .build()
}

/// Export spec to JSON
pub fn export_to_json(spec: &OpenApiSpec) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(spec)
}

/// Export spec to YAML
pub fn export_to_yaml(spec: &OpenApiSpec) -> String {
    // Simple YAML-like format (without full YAML dependency)
    let json = serde_json::to_value(spec).unwrap_or_default();
    json_to_yaml(&json, 0)
}

fn json_to_yaml(value: &serde_json::Value, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => {
            if s.contains('\n') || s.contains(':') || s.contains('#') {
                format!("|\n{}{}", prefix, s.replace('\n', &format!("\n{}", prefix)))
            } else {
                format!("\"{}\"", s.replace('\"', "\\\""))
            }
        }
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                "[]".to_string()
            } else {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| format!("{}- {}", prefix, json_to_yaml(v, indent + 1).trim_start()))
                    .collect();
                format!("\n{}", items.join("\n"))
            }
        }
        serde_json::Value::Object(obj) => {
            if obj.is_empty() {
                "{}".to_string()
            } else {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| {
                        let val_str = json_to_yaml(v, indent + 1);
                        if val_str.starts_with('\n') {
                            format!("{}{}:{}", prefix, k, val_str)
                        } else {
                            format!("{}{}: {}", prefix, k, val_str)
                        }
                    })
                    .collect();
                format!("\n{}", items.join("\n"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_schema_string() {
        let schema = JsonSchema::string().with_description("A test string");
        assert_eq!(schema.schema_type, Some("string".to_string()));
        assert_eq!(schema.description, Some("A test string".to_string()));
    }

    #[test]
    fn test_json_schema_object() {
        let schema = JsonSchema::object()
            .with_property("name", JsonSchema::string())
            .with_property("age", JsonSchema::integer())
            .with_required(vec!["name"]);

        assert_eq!(schema.schema_type, Some("object".to_string()));
        assert!(schema.properties.is_some());
        assert_eq!(schema.properties.as_ref().unwrap().len(), 2);
        assert_eq!(schema.required, Some(vec!["name".to_string()]));
    }

    #[test]
    fn test_json_schema_array() {
        let schema = JsonSchema::array(JsonSchema::string());
        assert_eq!(schema.schema_type, Some("array".to_string()));
        assert!(schema.items.is_some());
    }

    #[test]
    fn test_openapi_builder() {
        let spec = OpenApiBuilder::new()
            .title("Test API")
            .version("1.0.0")
            .server("http://localhost:8080", Some("Local"))
            .tag("test", Some("Test endpoints"))
            .build();

        assert_eq!(spec.info.title, "Test API");
        assert_eq!(spec.info.version, "1.0.0");
        assert_eq!(spec.servers.len(), 1);
        assert_eq!(spec.tags.len(), 1);
    }

    #[test]
    fn test_operation_builder() {
        let op = OperationBuilder::new()
            .summary("Test operation")
            .operation_id("testOp")
            .tag("test")
            .query_param("limit", JsonSchema::integer(), false)
            .response("200", "Success", Some(JsonSchema::object()))
            .build();

        assert_eq!(op.summary, Some("Test operation".to_string()));
        assert_eq!(op.operation_id, Some("testOp".to_string()));
        assert_eq!(op.tags.len(), 1);
        assert_eq!(op.parameters.len(), 1);
        assert!(op.responses.contains_key("200"));
    }

    #[test]
    fn test_generate_spec() {
        let spec = generate_ai_assistant_spec();

        assert!(!spec.paths.is_empty());
        assert!(spec.paths.contains_key("/api/chat"));
        assert!(spec.paths.contains_key("/api/tags"));
        assert!(spec.components.is_some());
    }

    #[test]
    fn test_export_json() {
        let spec = generate_ai_assistant_spec();
        let json = export_to_json(&spec);

        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("openapi"));
        assert!(json_str.contains("paths"));
    }

    #[test]
    fn test_json_schema_enum() {
        let schema = JsonSchema::string().with_enum(vec![
            serde_json::json!("option1"),
            serde_json::json!("option2"),
        ]);

        assert!(schema.enum_values.is_some());
        assert_eq!(schema.enum_values.as_ref().unwrap().len(), 2);
    }
}
