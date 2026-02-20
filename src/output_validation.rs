//! Output validation for AI responses
//!
//! This module provides comprehensive validation of AI-generated outputs
//! against various constraints and schemas.
//!
//! # Features
//!
//! - **Schema validation**: Validate against JSON schemas
//! - **Format validation**: Check for expected formats
//! - **Content validation**: Verify content constraints
//! - **Safety validation**: Check for unsafe content

use std::collections::HashMap;

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict mode (fail on any issue)
    pub strict: bool,
    /// Maximum output length
    pub max_length: Option<usize>,
    /// Minimum output length
    pub min_length: Option<usize>,
    /// Required format
    pub format: Option<OutputFormat>,
    /// Content must contain
    pub must_contain: Vec<String>,
    /// Content must not contain
    pub must_not_contain: Vec<String>,
    /// Custom validators
    pub custom_validators: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict: false,
            max_length: None,
            min_length: None,
            format: None,
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
            custom_validators: Vec::new(),
        }
    }
}

/// Expected output formats
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    /// Plain text
    PlainText,
    /// JSON
    Json,
    /// Markdown
    Markdown,
    /// XML
    Xml,
    /// YAML
    Yaml,
    /// Code (with optional language)
    Code(Option<String>),
    /// List (numbered or bulleted)
    List,
    /// Table
    Table,
    /// Custom format name
    Custom(String),
}

/// A single validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue type
    pub issue_type: IssueType,
    /// Human-readable message
    pub message: String,
    /// Position in output (if applicable)
    pub position: Option<usize>,
    /// Suggestion for fixing
    pub suggestion: Option<String>,
}

/// Severity levels for validation issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational only
    Info,
    /// Warning but valid
    Warning,
    /// Error, validation failed
    Error,
    /// Critical error
    Critical,
}

/// Types of validation issues
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    /// Output too long
    TooLong,
    /// Output too short
    TooShort,
    /// Invalid format
    InvalidFormat,
    /// Missing required content
    MissingContent,
    /// Contains forbidden content
    ForbiddenContent,
    /// Schema validation failed
    SchemaViolation,
    /// Safety concern
    SafetyConcern,
    /// Custom validation failed
    CustomValidation(String),
}

/// Result of validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<ValidationIssue>,
    /// Original output
    pub output: String,
    /// Corrected output (if auto-correction applied)
    pub corrected: Option<String>,
    /// Validation score (0-1)
    pub score: f64,
}

impl ValidationResult {
    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity >= IssueSeverity::Error)
    }

    /// Get error messages
    pub fn error_messages(&self) -> Vec<String> {
        self.issues
            .iter()
            .filter(|i| i.severity >= IssueSeverity::Error)
            .map(|i| i.message.clone())
            .collect()
    }
}

/// Output validator
pub struct OutputValidator {
    config: ValidationConfig,
    /// Custom validation functions
    custom_validators: HashMap<String, Box<dyn Fn(&str) -> Option<ValidationIssue> + Send + Sync>>,
}

impl OutputValidator {
    /// Create a new output validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            custom_validators: HashMap::new(),
        }
    }

    /// Register a custom validator
    pub fn register_validator<F>(&mut self, name: impl Into<String>, f: F)
    where
        F: Fn(&str) -> Option<ValidationIssue> + Send + Sync + 'static,
    {
        self.custom_validators.insert(name.into(), Box::new(f));
    }

    /// Validate an output string
    pub fn validate(&self, output: &str) -> ValidationResult {
        let mut issues = Vec::new();

        // Length validation
        if let Some(max) = self.config.max_length {
            if output.len() > max {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::TooLong,
                    message: format!("Output length {} exceeds maximum {}", output.len(), max),
                    position: Some(max),
                    suggestion: Some("Truncate or summarize the output".to_string()),
                });
            }
        }

        if let Some(min) = self.config.min_length {
            if output.len() < min {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::TooShort,
                    message: format!("Output length {} below minimum {}", output.len(), min),
                    position: None,
                    suggestion: Some("Provide more detailed output".to_string()),
                });
            }
        }

        // Format validation
        if let Some(ref format) = self.config.format {
            if let Some(issue) = self.validate_format(output, format) {
                issues.push(issue);
            }
        }

        // Required content
        for required in &self.config.must_contain {
            if !output.contains(required) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::MissingContent,
                    message: format!("Output missing required content: '{}'", required),
                    position: None,
                    suggestion: Some(format!("Include '{}' in the output", required)),
                });
            }
        }

        // Forbidden content
        for forbidden in &self.config.must_not_contain {
            if let Some(pos) = output.find(forbidden) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::ForbiddenContent,
                    message: format!("Output contains forbidden content: '{}'", forbidden),
                    position: Some(pos),
                    suggestion: Some(format!("Remove '{}' from the output", forbidden)),
                });
            }
        }

        // Custom validators
        for name in &self.config.custom_validators {
            if let Some(validator) = self.custom_validators.get(name) {
                if let Some(issue) = validator(output) {
                    issues.push(issue);
                }
            }
        }

        // Calculate score
        let score = self.calculate_score(&issues);

        let valid = if self.config.strict {
            issues.is_empty()
        } else {
            !issues.iter().any(|i| i.severity >= IssueSeverity::Error)
        };

        ValidationResult {
            valid,
            issues,
            output: output.to_string(),
            corrected: None,
            score,
        }
    }

    /// Validate output format
    fn validate_format(&self, output: &str, format: &OutputFormat) -> Option<ValidationIssue> {
        match format {
            OutputFormat::Json => {
                if serde_json::from_str::<serde_json::Value>(output).is_err() {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Error,
                        issue_type: IssueType::InvalidFormat,
                        message: "Invalid JSON format".to_string(),
                        position: None,
                        suggestion: Some("Ensure output is valid JSON".to_string()),
                    });
                }
            }
            OutputFormat::Markdown => {
                // Basic markdown validation (check for common markdown elements)
                let has_markdown = output.contains('#')
                    || output.contains('*')
                    || output.contains('`')
                    || output.contains('-');
                if !has_markdown && output.len() > 100 {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::InvalidFormat,
                        message: "Output doesn't appear to use Markdown formatting".to_string(),
                        position: None,
                        suggestion: Some("Add Markdown formatting elements".to_string()),
                    });
                }
            }
            OutputFormat::Xml => {
                if !output.trim().starts_with('<') || !output.trim().ends_with('>') {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Error,
                        issue_type: IssueType::InvalidFormat,
                        message: "Output doesn't appear to be valid XML".to_string(),
                        position: None,
                        suggestion: Some("Ensure output is valid XML".to_string()),
                    });
                }
            }
            OutputFormat::Yaml => {
                // Simple YAML validation (check for key: value patterns)
                let has_yaml_pattern = output.lines().any(|line| {
                    let trimmed = line.trim();
                    !trimmed.is_empty()
                        && (trimmed.contains(": ")
                            || trimmed.starts_with("- ")
                            || trimmed.starts_with("#"))
                });
                if !has_yaml_pattern {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::InvalidFormat,
                        message: "Output doesn't appear to be valid YAML".to_string(),
                        position: None,
                        suggestion: Some(
                            "Ensure output is valid YAML with key: value pairs".to_string(),
                        ),
                    });
                }
            }
            OutputFormat::Code(lang) => {
                // Check for code block
                let has_code_block = output.contains("```");
                if !has_code_block {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::InvalidFormat,
                        message: "Expected code block not found".to_string(),
                        position: None,
                        suggestion: Some("Wrap code in ```code``` blocks".to_string()),
                    });
                }
                // Check language if specified
                if let Some(ref expected_lang) = lang {
                    if !output.contains(&format!("```{}", expected_lang)) {
                        return Some(ValidationIssue {
                            severity: IssueSeverity::Warning,
                            issue_type: IssueType::InvalidFormat,
                            message: format!(
                                "Expected {} code block, found different language",
                                expected_lang
                            ),
                            position: None,
                            suggestion: Some(format!("Use ```{} for code blocks", expected_lang)),
                        });
                    }
                }
            }
            OutputFormat::List => {
                let has_list = output.contains("- ")
                    || output.contains("* ")
                    || regex::Regex::new(r"^\d+\.")
                        .map(|re| re.is_match(output))
                        .unwrap_or(false);
                if !has_list {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::InvalidFormat,
                        message: "Expected list format not found".to_string(),
                        position: None,
                        suggestion: Some("Use bullet points (- ) or numbers (1.)".to_string()),
                    });
                }
            }
            OutputFormat::Table => {
                let has_table = output.contains('|') && output.contains('-');
                if !has_table {
                    return Some(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        issue_type: IssueType::InvalidFormat,
                        message: "Expected table format not found".to_string(),
                        position: None,
                        suggestion: Some("Use Markdown table format with | and -".to_string()),
                    });
                }
            }
            OutputFormat::PlainText => {
                // No special validation for plain text
            }
            OutputFormat::Custom(_) => {
                // Custom format validation would need to be registered
            }
        }
        None
    }

    /// Calculate validation score
    fn calculate_score(&self, issues: &[ValidationIssue]) -> f64 {
        if issues.is_empty() {
            return 1.0;
        }

        let mut penalty = 0.0;
        for issue in issues {
            penalty += match issue.severity {
                IssueSeverity::Info => 0.0,
                IssueSeverity::Warning => 0.1,
                IssueSeverity::Error => 0.3,
                IssueSeverity::Critical => 0.5,
            };
        }

        (1.0_f64 - penalty).max(0.0)
    }
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

/// JSON Schema validator
pub struct SchemaValidator {
    schema: serde_json::Value,
}

impl SchemaValidator {
    /// Create validator from JSON schema
    pub fn new(schema: serde_json::Value) -> Self {
        Self { schema }
    }

    /// Create validator from schema string
    pub fn from_str(schema: &str) -> Result<Self, String> {
        let schema = serde_json::from_str(schema).map_err(|e| e.to_string())?;
        Ok(Self { schema })
    }

    /// Validate JSON against schema
    pub fn validate(&self, json: &str) -> ValidationResult {
        let mut issues = Vec::new();

        // Parse JSON first
        let value: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::InvalidFormat,
                    message: format!("Invalid JSON: {}", e),
                    position: None,
                    suggestion: Some("Fix JSON syntax errors".to_string()),
                });

                return ValidationResult {
                    valid: false,
                    issues,
                    output: json.to_string(),
                    corrected: None,
                    score: 0.0,
                };
            }
        };

        // Validate against schema
        issues.extend(self.validate_value(&value, &self.schema, ""));

        let valid = !issues.iter().any(|i| i.severity >= IssueSeverity::Error);
        let score = if valid { 1.0 } else { 0.0 };

        ValidationResult {
            valid,
            issues,
            output: json.to_string(),
            corrected: None,
            score,
        }
    }

    fn validate_value(
        &self,
        value: &serde_json::Value,
        schema: &serde_json::Value,
        path: &str,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if let Some(schema_type) = schema.get("type").and_then(|t| t.as_str()) {
            let type_valid = match schema_type {
                "string" => value.is_string(),
                "number" => value.is_number(),
                "integer" => value.is_i64(),
                "boolean" => value.is_boolean(),
                "array" => value.is_array(),
                "object" => value.is_object(),
                "null" => value.is_null(),
                _ => true,
            };

            if !type_valid {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    issue_type: IssueType::SchemaViolation,
                    message: format!(
                        "Type mismatch at {}: expected {}, got {}",
                        path,
                        schema_type,
                        value_type(value)
                    ),
                    position: None,
                    suggestion: Some(format!("Change to {} type", schema_type)),
                });
            }
        }

        // Check required properties
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            if let Some(obj) = value.as_object() {
                for req in required {
                    if let Some(req_str) = req.as_str() {
                        if !obj.contains_key(req_str) {
                            issues.push(ValidationIssue {
                                severity: IssueSeverity::Error,
                                issue_type: IssueType::SchemaViolation,
                                message: format!("Missing required property: {}", req_str),
                                position: None,
                                suggestion: Some(format!("Add '{}' property", req_str)),
                            });
                        }
                    }
                }
            }
        }

        // Validate object properties
        if let Some(props) = schema.get("properties") {
            if let Some(obj) = value.as_object() {
                for (key, prop_schema) in props.as_object().unwrap_or(&serde_json::Map::new()) {
                    if let Some(prop_value) = obj.get(key) {
                        let prop_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };
                        issues.extend(self.validate_value(prop_value, prop_schema, &prop_path));
                    }
                }
            }
        }

        // Validate array items
        if let Some(items) = schema.get("items") {
            if let Some(arr) = value.as_array() {
                for (i, item) in arr.iter().enumerate() {
                    let item_path = format!("{}[{}]", path, i);
                    issues.extend(self.validate_value(item, items, &item_path));
                }
            }
        }

        issues
    }
}

fn value_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Builder for validation configuration
pub struct ValidationConfigBuilder {
    config: ValidationConfig,
}

impl ValidationConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Enable strict mode
    pub fn strict(mut self, enabled: bool) -> Self {
        self.config.strict = enabled;
        self
    }

    /// Set max length
    pub fn max_length(mut self, len: usize) -> Self {
        self.config.max_length = Some(len);
        self
    }

    /// Set min length
    pub fn min_length(mut self, len: usize) -> Self {
        self.config.min_length = Some(len);
        self
    }

    /// Set required format
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.config.format = Some(format);
        self
    }

    /// Add required content
    pub fn must_contain(mut self, content: impl Into<String>) -> Self {
        self.config.must_contain.push(content.into());
        self
    }

    /// Add forbidden content
    pub fn must_not_contain(mut self, content: impl Into<String>) -> Self {
        self.config.must_not_contain.push(content.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> ValidationConfig {
        self.config
    }
}

impl Default for ValidationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_validation() {
        let config = ValidationConfig {
            max_length: Some(10),
            min_length: Some(5),
            ..Default::default()
        };
        let validator = OutputValidator::new(config);

        let result = validator.validate("hello");
        assert!(result.valid);

        let result = validator.validate("hi");
        assert!(!result.valid);

        let result = validator.validate("this is too long for the limit");
        assert!(!result.valid);
    }

    #[test]
    fn test_format_validation_json() {
        let config = ValidationConfig {
            format: Some(OutputFormat::Json),
            ..Default::default()
        };
        let validator = OutputValidator::new(config);

        let result = validator.validate(r#"{"key": "value"}"#);
        assert!(result.valid);

        let result = validator.validate("not json");
        assert!(!result.valid);
    }

    #[test]
    fn test_content_validation() {
        let config = ValidationConfig {
            must_contain: vec!["required".to_string()],
            must_not_contain: vec!["forbidden".to_string()],
            ..Default::default()
        };
        let validator = OutputValidator::new(config);

        let result = validator.validate("This is required content");
        assert!(result.valid);

        let result = validator.validate("This is forbidden content");
        assert!(!result.valid);
    }

    #[test]
    fn test_schema_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        });

        let validator = SchemaValidator::new(schema);

        let result = validator.validate(r#"{"name": "Alice", "age": 30}"#);
        assert!(result.valid);

        let result = validator.validate(r#"{"age": 30}"#);
        assert!(!result.valid); // Missing required "name"
    }

    #[test]
    fn test_config_builder() {
        let config = ValidationConfigBuilder::new()
            .strict(true)
            .max_length(1000)
            .min_length(10)
            .format(OutputFormat::Markdown)
            .must_contain("summary")
            .must_not_contain("TODO")
            .build();

        assert!(config.strict);
        assert_eq!(config.max_length, Some(1000));
    }
}
