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
            result.starter_messages = result.starter_messages.iter()
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
        self.templates.values()
            .filter(|t| t.category == category)
            .collect()
    }

    pub fn search(&self, query: &str) -> Vec<&ConversationTemplate> {
        let lower = query.to_lowercase();
        self.templates.values()
            .filter(|t| {
                t.name.to_lowercase().contains(&lower) ||
                t.description.to_lowercase().contains(&lower) ||
                t.tags.iter().any(|tag| tag.to_lowercase().contains(&lower))
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
        assert!(coding.iter().all(|t| t.category == TemplateCategory::Coding));
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
}
