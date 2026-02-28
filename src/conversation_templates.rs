//! Conversation templates
//!
//! Pre-defined conversation templates for common use cases.

use std::collections::HashMap;

/// Template category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateCategory {
    Coding,
    Writing,
    Analysis,
    Learning,
    Creative,
    Business,
    Research,
    Support,
}

impl TemplateCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Coding => "Coding",
            Self::Writing => "Writing",
            Self::Analysis => "Analysis",
            Self::Learning => "Learning",
            Self::Creative => "Creative",
            Self::Business => "Business",
            Self::Research => "Research",
            Self::Support => "Support",
        }
    }
}

/// Conversation template
#[derive(Debug, Clone)]
pub struct ConversationTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: TemplateCategory,
    pub system_prompt: String,
    pub starter_messages: Vec<String>,
    pub variables: Vec<TemplateVariable>,
    pub tags: Vec<String>,
}

/// Template variable
#[derive(Debug, Clone)]
pub struct TemplateVariable {
    pub name: String,
    pub description: String,
    pub default: Option<String>,
    pub required: bool,
}

impl ConversationTemplate {
    pub fn new(id: &str, name: &str, category: TemplateCategory) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            category,
            system_prompt: String::new(),
            starter_messages: Vec::new(),
            variables: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    pub fn with_starter(mut self, message: &str) -> Self {
        self.starter_messages.push(message.to_string());
        self
    }

    pub fn with_variable(mut self, name: &str, desc: &str, required: bool) -> Self {
        self.variables.push(TemplateVariable {
            name: name.to_string(),
            description: desc.to_string(),
            default: None,
            required,
        });
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Apply variables to template
    pub fn apply(&self, vars: &HashMap<String, String>) -> ConversationTemplate {
        let mut result = self.clone();

        for (key, value) in vars {
            result.system_prompt = result.system_prompt.replace(&format!("{{{}}}", key), value);
            result.starter_messages = result
                .starter_messages
                .iter()
                .map(|m| m.replace(&format!("{{{}}}", key), value))
                .collect();
        }

        result
    }
}

/// Template library
pub struct TemplateLibrary {
    templates: HashMap<String, ConversationTemplate>,
}

impl TemplateLibrary {
    pub fn new() -> Self {
        let mut lib = Self {
            templates: HashMap::new(),
        };
        lib.add_builtin_templates();
        lib
    }

    fn add_builtin_templates(&mut self) {
        // Code Review
        self.add(ConversationTemplate::new("code_review", "Code Review", TemplateCategory::Coding)
            .with_description("Get expert code review and suggestions")
            .with_system_prompt("You are an expert code reviewer. Analyze code for bugs, performance issues, security vulnerabilities, and best practices. Be specific and constructive.")
            .with_starter("Please review the following code and provide feedback on improvements:")
            .with_variable("language", "Programming language", false)
            .with_tag("code"));

        // Debug Helper
        self.add(ConversationTemplate::new("debug", "Debug Helper", TemplateCategory::Coding)
            .with_description("Help debug code issues")
            .with_system_prompt("You are a debugging expert. Help identify and fix bugs in code. Ask clarifying questions and provide step-by-step debugging guidance.")
            .with_starter("I'm having an issue with my code:")
            .with_tag("code")
            .with_tag("debug"));

        // Writing Assistant
        self.add(ConversationTemplate::new("writing", "Writing Assistant", TemplateCategory::Writing)
            .with_description("Help with writing and editing")
            .with_system_prompt("You are a professional writing assistant. Help improve clarity, grammar, style, and structure. Maintain the author's voice while enhancing the text.")
            .with_starter("Please help me improve this text:")
            .with_tag("writing"));

        // Learning Tutor
        self.add(ConversationTemplate::new("tutor", "Learning Tutor", TemplateCategory::Learning)
            .with_description("Patient teaching and explanations")
            .with_system_prompt("You are a patient and knowledgeable tutor. Explain concepts clearly, use examples, and adapt to the learner's level. Encourage questions.")
            .with_starter("I'd like to learn about {topic}. Can you explain it?")
            .with_variable("topic", "Topic to learn", true)
            .with_tag("learning"));

        // Brainstorm
        self.add(ConversationTemplate::new("brainstorm", "Brainstorm", TemplateCategory::Creative)
            .with_description("Creative brainstorming session")
            .with_system_prompt("You are a creative brainstorming partner. Generate diverse ideas, build on suggestions, and help explore possibilities without judgment.")
            .with_starter("Let's brainstorm ideas for:")
            .with_tag("creative"));

        // Business Analysis
        self.add(ConversationTemplate::new("business", "Business Analyst", TemplateCategory::Business)
            .with_description("Business analysis and strategy")
            .with_system_prompt("You are a business analyst. Provide data-driven insights, strategic recommendations, and help analyze business problems methodically.")
            .with_starter("I need help analyzing:")
            .with_tag("business"));

        // Research Assistant
        self.add(ConversationTemplate::new("research", "Research Assistant", TemplateCategory::Research)
            .with_description("Help with research and information gathering")
            .with_system_prompt("You are a research assistant. Help gather, organize, and synthesize information. Cite sources when possible and acknowledge limitations.")
            .with_starter("Help me research:")
            .with_tag("research"));

        // Technical Support
        self.add(ConversationTemplate::new("support", "Tech Support", TemplateCategory::Support)
            .with_description("Technical troubleshooting")
            .with_system_prompt("You are a technical support specialist. Help diagnose and solve technical problems step by step. Be patient and thorough.")
            .with_starter("I'm having a technical issue:")
            .with_tag("support")
            .with_tag("tech"));
    }

    pub fn add(&mut self, template: ConversationTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    pub fn get(&self, id: &str) -> Option<&ConversationTemplate> {
        self.templates.get(id)
    }

    pub fn get_by_category(&self, category: TemplateCategory) -> Vec<&ConversationTemplate> {
        self.templates
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    pub fn search(&self, query: &str) -> Vec<&ConversationTemplate> {
        let lower = query.to_lowercase();
        self.templates
            .values()
            .filter(|t| {
                t.name.to_lowercase().contains(&lower)
                    || t.description.to_lowercase().contains(&lower)
                    || t.tags.iter().any(|tag| tag.to_lowercase().contains(&lower))
            })
            .collect()
    }

    pub fn all(&self) -> Vec<&ConversationTemplate> {
        self.templates.values().collect()
    }

    pub fn categories(&self) -> Vec<TemplateCategory> {
        vec![
            TemplateCategory::Coding,
            TemplateCategory::Writing,
            TemplateCategory::Analysis,
            TemplateCategory::Learning,
            TemplateCategory::Creative,
            TemplateCategory::Business,
            TemplateCategory::Research,
            TemplateCategory::Support,
        ]
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_library() {
        let lib = TemplateLibrary::new();

        assert!(lib.get("code_review").is_some());
        assert!(lib.get("debug").is_some());
    }

    #[test]
    fn test_category_filter() {
        let lib = TemplateLibrary::new();
        let coding = lib.get_by_category(TemplateCategory::Coding);

        assert!(!coding.is_empty());
        assert!(coding
            .iter()
            .all(|t| t.category == TemplateCategory::Coding));
    }

    #[test]
    fn test_apply_variables() {
        let template = ConversationTemplate::new("test", "Test", TemplateCategory::Learning)
            .with_system_prompt("Help with {topic}")
            .with_starter("Learn about {topic}");

        let mut vars = HashMap::new();
        vars.insert("topic".to_string(), "Rust".to_string());

        let applied = template.apply(&vars);
        assert!(applied.system_prompt.contains("Rust"));
    }

    #[test]
    fn test_all_template_categories() {
        let lib = TemplateLibrary::new();
        let categories = lib.categories();

        // Every category returned by categories() should have at least one template
        for category in &categories {
            let templates = lib.get_by_category(*category);
            // Not all categories may have builtins (Analysis doesn't), but let's
            // verify the ones that do are non-empty and correct
            for t in &templates {
                assert_eq!(t.category, *category);
            }
        }

        // Verify specific categories we know have builtins
        assert!(!lib.get_by_category(TemplateCategory::Coding).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Writing).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Learning).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Creative).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Business).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Research).is_empty());
        assert!(!lib.get_by_category(TemplateCategory::Support).is_empty());

        // Verify the categories list has all 8 variants
        assert_eq!(categories.len(), 8);
    }

    #[test]
    fn test_template_builder() {
        let template = ConversationTemplate::new("my_template", "My Template", TemplateCategory::Creative)
            .with_description("A custom creative template")
            .with_system_prompt("You are a creative assistant helping with {project}")
            .with_starter("Let's get started on {project}")
            .with_starter("Tell me about your creative vision")
            .with_variable("project", "Project name", true)
            .with_variable("style", "Preferred style", false)
            .with_tag("creative")
            .with_tag("custom");

        assert_eq!(template.id, "my_template");
        assert_eq!(template.name, "My Template");
        assert_eq!(template.category, TemplateCategory::Creative);
        assert_eq!(template.description, "A custom creative template");
        assert!(template.system_prompt.contains("{project}"));
        assert_eq!(template.starter_messages.len(), 2);
        assert_eq!(template.variables.len(), 2);
        assert!(template.variables[0].required);
        assert!(!template.variables[1].required);
        assert_eq!(template.tags, vec!["creative", "custom"]);
    }

    #[test]
    fn test_custom_template_registration() {
        let mut lib = TemplateLibrary::new();
        let initial_count = lib.all().len();

        let custom = ConversationTemplate::new("custom_qa", "Q&A Bot", TemplateCategory::Support)
            .with_description("Custom Q&A template")
            .with_system_prompt("Answer questions concisely")
            .with_starter("Ask me anything")
            .with_tag("qa");

        lib.add(custom);

        // Library should have one more template
        assert_eq!(lib.all().len(), initial_count + 1);

        // Should be retrievable by id
        let retrieved = lib.get("custom_qa");
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Q&A Bot");
        assert_eq!(retrieved.category, TemplateCategory::Support);
        assert_eq!(retrieved.description, "Custom Q&A template");

        // Should appear in category filter
        let support_templates = lib.get_by_category(TemplateCategory::Support);
        assert!(support_templates.iter().any(|t| t.id == "custom_qa"));
    }

    #[test]
    fn test_search_templates() {
        let lib = TemplateLibrary::new();

        // Search by name
        let results = lib.search("code review");
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| t.id == "code_review"));

        // Search by tag
        let results = lib.search("debug");
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| t.id == "debug"));

        // Search by description keyword
        let results = lib.search("troubleshooting");
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| t.id == "support"));

        // Search with no matches
        let results = lib.search("xyznonexistent");
        assert!(results.is_empty());

        // Case-insensitive search
        let results = lib.search("WRITING");
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| t.id == "writing"));
    }

    #[test]
    fn test_template_with_variables() {
        let template = ConversationTemplate::new("multi_var", "Multi Variable", TemplateCategory::Learning)
            .with_system_prompt("Teach {topic} at {level} level using {language}")
            .with_starter("Explain {topic} for a {level} student")
            .with_starter("Give a {language} example of {topic}")
            .with_variable("topic", "Subject to learn", true)
            .with_variable("level", "Difficulty level", true)
            .with_variable("language", "Programming language", false);

        let mut vars = HashMap::new();
        vars.insert("topic".to_string(), "algorithms".to_string());
        vars.insert("level".to_string(), "intermediate".to_string());
        vars.insert("language".to_string(), "Python".to_string());

        let applied = template.apply(&vars);

        // Verify system_prompt substitution
        assert_eq!(applied.system_prompt, "Teach algorithms at intermediate level using Python");
        assert!(!applied.system_prompt.contains("{topic}"));
        assert!(!applied.system_prompt.contains("{level}"));
        assert!(!applied.system_prompt.contains("{language}"));

        // Verify starter messages substitution
        assert_eq!(applied.starter_messages[0], "Explain algorithms for a intermediate student");
        assert_eq!(applied.starter_messages[1], "Give a Python example of algorithms");
    }

    #[test]
    fn test_template_starters() {
        let lib = TemplateLibrary::new();

        // Every builtin template should have at least one starter message
        for template in lib.all() {
            assert!(
                !template.starter_messages.is_empty(),
                "Template '{}' should have at least one starter message",
                template.id
            );

            // Each starter message should be non-empty and meaningful (more than 5 chars)
            for starter in &template.starter_messages {
                assert!(
                    starter.len() > 5,
                    "Starter '{}' in template '{}' should be meaningful",
                    starter,
                    template.id
                );
            }
        }

        // Verify a specific known starter for the tutor template (has a variable)
        let tutor = lib.get("tutor").unwrap();
        assert!(tutor.starter_messages[0].contains("{topic}"));
    }

    #[test]
    fn test_category_name_roundtrip() {
        let categories = vec![
            TemplateCategory::Coding, TemplateCategory::Writing,
            TemplateCategory::Analysis, TemplateCategory::Learning,
            TemplateCategory::Creative, TemplateCategory::Business,
            TemplateCategory::Research, TemplateCategory::Support,
        ];
        for cat in &categories {
            assert!(!cat.name().is_empty());
        }
        assert_eq!(categories.len(), 8);
    }
}
