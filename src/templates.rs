//! Prompt templates with variable substitution
//!
//! This module provides a templating system for creating reusable prompts with
//! placeholders that can be filled in at runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

/// A prompt template with variable placeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// The template content with {{variable}} placeholders
    pub content: String,
    /// Variable definitions with descriptions and defaults
    pub variables: Vec<TemplateVariable>,
    /// Category for organization
    pub category: Option<String>,
    /// Tags for searching
    pub tags: Vec<String>,
}

/// Definition of a template variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name (used in {{name}} placeholder)
    pub name: String,
    /// Description of what the variable is for
    pub description: String,
    /// Default value if not provided
    pub default: Option<String>,
    /// Whether this variable is required
    pub required: bool,
    /// Validation type
    pub var_type: VariableType,
    /// Example values
    pub examples: Vec<String>,
}

/// Type of variable for validation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VariableType {
    /// Free-form text
    Text,
    /// Single line text
    SingleLine,
    /// Must be a number
    Number,
    /// Must be one of predefined choices
    Choice,
    /// Must be a valid language code
    Language,
    /// Must be a valid file path
    FilePath,
    /// Code snippet
    Code,
    /// List of items (comma-separated)
    List,
}

impl PromptTemplate {
    /// Create a new template
    pub fn new(name: &str, content: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            content: content.to_string(),
            variables: Self::extract_variables(content),
            category: None,
            tags: vec![],
        }
    }

    /// Extract variable names from template content
    fn extract_variables(content: &str) -> Vec<TemplateVariable> {
        let mut variables = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let mut chars = content.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second {
                let mut var_name = String::new();

                while let Some(&next) = chars.peek() {
                    if next == '}' {
                        chars.next();
                        if chars.peek() == Some(&'}') {
                            chars.next();
                            break;
                        }
                    } else {
                        var_name.push(chars.next().expect("char verified by peek"));
                    }
                }

                let var_name = var_name.trim().to_string();
                if !var_name.is_empty() && !seen.contains(&var_name) {
                    seen.insert(var_name.clone());
                    variables.push(TemplateVariable {
                        name: var_name,
                        description: String::new(),
                        default: None,
                        required: true,
                        var_type: VariableType::Text,
                        examples: vec![],
                    });
                }
            }
        }

        variables
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set category
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Configure a variable
    pub fn configure_variable(
        mut self,
        name: &str,
        description: &str,
        default: Option<&str>,
        required: bool,
    ) -> Self {
        if let Some(var) = self.variables.iter_mut().find(|v| v.name == name) {
            var.description = description.to_string();
            var.default = default.map(|s| s.to_string());
            var.required = required;
        }
        self
    }

    /// Set variable type
    pub fn set_variable_type(mut self, name: &str, var_type: VariableType) -> Self {
        if let Some(var) = self.variables.iter_mut().find(|v| v.name == name) {
            var.var_type = var_type;
        }
        self
    }

    /// Add example for a variable
    pub fn add_variable_example(mut self, name: &str, example: &str) -> Self {
        if let Some(var) = self.variables.iter_mut().find(|v| v.name == name) {
            var.examples.push(example.to_string());
        }
        self
    }

    /// Render the template with provided values
    pub fn render(&self, values: &HashMap<String, String>) -> Result<String> {
        let mut result = self.content.clone();

        for var in &self.variables {
            let placeholder = format!("{{{{{}}}}}", var.name);
            let value = values
                .get(&var.name)
                .or(var.default.as_ref())
                .ok_or_else(|| anyhow!("Missing required variable: {}", var.name))?;

            // Validate the value
            self.validate_value(&var.name, value, var.var_type)?;

            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Validate a value against its type
    fn validate_value(&self, name: &str, value: &str, var_type: VariableType) -> Result<()> {
        match var_type {
            VariableType::Number => {
                value.parse::<f64>()
                    .map_err(|_| anyhow!("Variable '{}' must be a number, got: {}", name, value))?;
            }
            VariableType::SingleLine => {
                if value.contains('\n') {
                    return Err(anyhow!("Variable '{}' must be a single line", name));
                }
            }
            _ => {} // Other types don't have strict validation
        }
        Ok(())
    }

    /// Get list of required variables
    pub fn required_variables(&self) -> Vec<&str> {
        self.variables
            .iter()
            .filter(|v| v.required && v.default.is_none())
            .map(|v| v.name.as_str())
            .collect()
    }

    /// Check if all required variables are provided
    pub fn validate_values(&self, values: &HashMap<String, String>) -> Vec<String> {
        let mut missing = Vec::new();
        for var in &self.variables {
            if var.required && var.default.is_none() && !values.contains_key(&var.name) {
                missing.push(var.name.clone());
            }
        }
        missing
    }
}

/// Builder for creating templates fluently
pub struct TemplateBuilder {
    template: PromptTemplate,
}

impl TemplateBuilder {
    /// Start building a new template
    pub fn new(name: &str) -> Self {
        Self {
            template: PromptTemplate {
                name: name.to_string(),
                description: String::new(),
                content: String::new(),
                variables: vec![],
                category: None,
                tags: vec![],
            },
        }
    }

    /// Set the template content
    pub fn content(mut self, content: &str) -> Self {
        self.template.content = content.to_string();
        self.template.variables = PromptTemplate::extract_variables(content);
        self
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.template.description = desc.to_string();
        self
    }

    /// Set category
    pub fn category(mut self, category: &str) -> Self {
        self.template.category = Some(category.to_string());
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: &str) -> Self {
        self.template.tags.push(tag.to_string());
        self
    }

    /// Define a variable with all properties
    pub fn variable(
        mut self,
        name: &str,
        description: &str,
        var_type: VariableType,
        required: bool,
        default: Option<&str>,
    ) -> Self {
        if let Some(var) = self.template.variables.iter_mut().find(|v| v.name == name) {
            var.description = description.to_string();
            var.var_type = var_type;
            var.required = required;
            var.default = default.map(|s| s.to_string());
        }
        self
    }

    /// Build the template
    pub fn build(self) -> PromptTemplate {
        self.template
    }
}

/// Collection of built-in templates
pub struct BuiltinTemplates;

impl BuiltinTemplates {
    /// Code review template
    pub fn code_review() -> PromptTemplate {
        TemplateBuilder::new("code_review")
            .content(r#"Please review the following {{language}} code:

```{{language}}
{{code}}
```

Focus on:
{{focus_areas}}

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement"#)
            .description("Review code for quality, bugs, and improvements")
            .category("development")
            .tag("code")
            .tag("review")
            .variable("language", "Programming language", VariableType::SingleLine, true, Some("rust"))
            .variable("code", "Code to review", VariableType::Code, true, None)
            .variable("focus_areas", "Areas to focus on", VariableType::Text, false, Some("- Correctness\n- Readability\n- Error handling"))
            .build()
    }

    /// Translation template
    pub fn translation() -> PromptTemplate {
        TemplateBuilder::new("translation")
            .content(r#"Translate the following text from {{source_language}} to {{target_language}}:

{{text}}

Requirements:
- Preserve the original meaning and tone
- Use natural expressions in the target language
- Maintain any formatting (markdown, etc.)
{{additional_instructions}}"#)
            .description("Translate text between languages")
            .category("translation")
            .tag("translate")
            .tag("language")
            .variable("source_language", "Source language", VariableType::Language, true, Some("English"))
            .variable("target_language", "Target language", VariableType::Language, true, None)
            .variable("text", "Text to translate", VariableType::Text, true, None)
            .variable("additional_instructions", "Additional instructions", VariableType::Text, false, Some(""))
            .build()
    }

    /// Explanation template
    pub fn explain() -> PromptTemplate {
        TemplateBuilder::new("explain")
            .content(r#"Explain {{topic}} in a way that a {{audience}} can understand.

Level of detail: {{detail_level}}

{{additional_context}}

Please include:
- A clear definition
- Key concepts
- Examples where appropriate"#)
            .description("Explain a topic for a specific audience")
            .category("education")
            .tag("explain")
            .tag("teach")
            .variable("topic", "Topic to explain", VariableType::Text, true, None)
            .variable("audience", "Target audience", VariableType::SingleLine, true, Some("beginner"))
            .variable("detail_level", "Level of detail", VariableType::SingleLine, false, Some("moderate"))
            .variable("additional_context", "Additional context", VariableType::Text, false, Some(""))
            .build()
    }

    /// Bug fix template
    pub fn bug_fix() -> PromptTemplate {
        TemplateBuilder::new("bug_fix")
            .content(r#"I have a bug in my {{language}} code.

**Error message:**
```
{{error_message}}
```

**Code:**
```{{language}}
{{code}}
```

**Expected behavior:**
{{expected_behavior}}

**Actual behavior:**
{{actual_behavior}}

Please help me:
1. Identify the root cause
2. Explain why it's happening
3. Provide a fix"#)
            .description("Help fix a bug in code")
            .category("development")
            .tag("bug")
            .tag("fix")
            .tag("debug")
            .variable("language", "Programming language", VariableType::SingleLine, true, None)
            .variable("error_message", "Error message", VariableType::Text, true, None)
            .variable("code", "Buggy code", VariableType::Code, true, None)
            .variable("expected_behavior", "What should happen", VariableType::Text, true, None)
            .variable("actual_behavior", "What actually happens", VariableType::Text, true, None)
            .build()
    }

    /// Summarization template
    pub fn summarize() -> PromptTemplate {
        TemplateBuilder::new("summarize")
            .content(r#"Summarize the following {{content_type}}:

{{content}}

Summary requirements:
- Length: {{length}}
- Format: {{format}}
- Include key points and main takeaways"#)
            .description("Summarize content")
            .category("writing")
            .tag("summary")
            .tag("condense")
            .variable("content_type", "Type of content", VariableType::SingleLine, false, Some("text"))
            .variable("content", "Content to summarize", VariableType::Text, true, None)
            .variable("length", "Desired length", VariableType::SingleLine, false, Some("2-3 paragraphs"))
            .variable("format", "Output format", VariableType::SingleLine, false, Some("prose"))
            .build()
    }

    /// API documentation template
    pub fn api_docs() -> PromptTemplate {
        TemplateBuilder::new("api_docs")
            .content(r#"Generate API documentation for the following {{language}} function/method:

```{{language}}
{{code}}
```

Documentation should include:
- Brief description
- Parameters with types and descriptions
- Return value with type and description
- Example usage
- Any important notes or caveats

Format: {{format}}"#)
            .description("Generate API documentation")
            .category("development")
            .tag("docs")
            .tag("api")
            .variable("language", "Programming language", VariableType::SingleLine, true, None)
            .variable("code", "Code to document", VariableType::Code, true, None)
            .variable("format", "Documentation format", VariableType::SingleLine, false, Some("markdown"))
            .build()
    }

    /// Refactoring template
    pub fn refactor() -> PromptTemplate {
        TemplateBuilder::new("refactor")
            .content(r#"Refactor the following {{language}} code:

```{{language}}
{{code}}
```

Refactoring goals:
{{goals}}

Constraints:
- Maintain the same functionality
- Keep the public API unchanged (unless specified)
- Follow {{language}} best practices"#)
            .description("Refactor code for better quality")
            .category("development")
            .tag("refactor")
            .tag("improve")
            .variable("language", "Programming language", VariableType::SingleLine, true, None)
            .variable("code", "Code to refactor", VariableType::Code, true, None)
            .variable("goals", "Refactoring goals", VariableType::Text, false, Some("- Improve readability\n- Reduce complexity\n- Follow SOLID principles"))
            .build()
    }

    /// Get all built-in templates
    pub fn all() -> Vec<PromptTemplate> {
        vec![
            Self::code_review(),
            Self::translation(),
            Self::explain(),
            Self::bug_fix(),
            Self::summarize(),
            Self::api_docs(),
            Self::refactor(),
        ]
    }
}

/// Template manager for storing and retrieving templates
#[derive(Debug, Default)]
pub struct TemplateManager {
    templates: HashMap<String, PromptTemplate>,
}

impl TemplateManager {
    /// Create a new template manager
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Create with built-in templates
    pub fn with_builtins() -> Self {
        let mut manager = Self::new();
        for template in BuiltinTemplates::all() {
            manager.add(template);
        }
        manager
    }

    /// Add a template
    pub fn add(&mut self, template: PromptTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Get a template by name
    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    /// Remove a template
    pub fn remove(&mut self, name: &str) -> Option<PromptTemplate> {
        self.templates.remove(name)
    }

    /// List all template names
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Find templates by category
    pub fn find_by_category(&self, category: &str) -> Vec<&PromptTemplate> {
        self.templates
            .values()
            .filter(|t| t.category.as_deref() == Some(category))
            .collect()
    }

    /// Find templates by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&PromptTemplate> {
        self.templates
            .values()
            .filter(|t| t.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Search templates by name or description
    pub fn search(&self, query: &str) -> Vec<&PromptTemplate> {
        let query_lower = query.to_lowercase();
        self.templates
            .values()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower)
                    || t.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Render a template by name
    pub fn render(&self, name: &str, values: &HashMap<String, String>) -> Result<String> {
        let template = self.get(name)
            .ok_or_else(|| anyhow!("Template not found: {}", name))?;
        template.render(values)
    }

    /// Export all templates to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.templates)
    }

    /// Import templates from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize, serde_json::Error> {
        let imported: HashMap<String, PromptTemplate> = serde_json::from_str(json)?;
        let count = imported.len();
        self.templates.extend(imported);
        Ok(count)
    }
}

/// Convenience macro for creating template values
#[macro_export]
macro_rules! template_values {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = std::collections::HashMap::new();
        $(
            map.insert($key.to_string(), $value.to_string());
        )*
        map
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_creation() {
        let template = PromptTemplate::new("test", "Hello {{name}}, welcome to {{place}}!");

        assert_eq!(template.name, "test");
        assert_eq!(template.variables.len(), 2);
        assert!(template.variables.iter().any(|v| v.name == "name"));
        assert!(template.variables.iter().any(|v| v.name == "place"));
    }

    #[test]
    fn test_template_render() {
        let template = PromptTemplate::new("test", "Hello {{name}}!");

        let mut values = HashMap::new();
        values.insert("name".to_string(), "World".to_string());

        let result = template.render(&values).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_template_with_default() {
        let template = PromptTemplate::new("test", "Language: {{lang}}")
            .configure_variable("lang", "Programming language", Some("Rust"), false);

        let result = template.render(&HashMap::new()).unwrap();
        assert_eq!(result, "Language: Rust");
    }

    #[test]
    fn test_template_missing_required() {
        let template = PromptTemplate::new("test", "Hello {{name}}!");

        let result = template.render(&HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_builtin_templates() {
        let templates = BuiltinTemplates::all();
        assert!(!templates.is_empty());

        let code_review = BuiltinTemplates::code_review();
        assert_eq!(code_review.name, "code_review");
        assert!(code_review.content.contains("{{code}}"));
    }

    #[test]
    fn test_template_manager() {
        let manager = TemplateManager::with_builtins();

        assert!(manager.get("code_review").is_some());
        assert!(manager.get("translation").is_some());

        let dev_templates = manager.find_by_category("development");
        assert!(!dev_templates.is_empty());

        let code_templates = manager.find_by_tag("code");
        assert!(!code_templates.is_empty());
    }

    #[test]
    fn test_template_builder() {
        let template = TemplateBuilder::new("custom")
            .content("{{greeting}} {{name}}!")
            .description("A greeting template")
            .category("greetings")
            .tag("hello")
            .variable("greeting", "The greeting word", VariableType::SingleLine, false, Some("Hello"))
            .variable("name", "Name to greet", VariableType::SingleLine, true, None)
            .build();

        assert_eq!(template.name, "custom");
        assert_eq!(template.category, Some("greetings".to_string()));

        let mut values = HashMap::new();
        values.insert("name".to_string(), "World".to_string());
        let result = template.render(&values).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_variable_validation() {
        let template = PromptTemplate::new("test", "Count: {{count}}")
            .set_variable_type("count", VariableType::Number);

        let mut valid = HashMap::new();
        valid.insert("count".to_string(), "42".to_string());
        assert!(template.render(&valid).is_ok());

        let mut invalid = HashMap::new();
        invalid.insert("count".to_string(), "not a number".to_string());
        assert!(template.render(&invalid).is_err());
    }
}
