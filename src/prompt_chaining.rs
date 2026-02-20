//! Prompt chaining for multi-step AI workflows
//!
//! This module provides prompt chaining capabilities to automatically
//! chain multiple prompts together for complex tasks.
//!
//! # Features
//!
//! - **Sequential chains**: Execute prompts in sequence
//! - **Conditional branching**: Branch based on previous responses
//! - **Variable interpolation**: Pass data between chain steps
//! - **Error handling**: Continue or abort on errors

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for prompt chaining
#[derive(Debug, Clone)]
pub struct ChainConfig {
    /// Maximum steps in a chain
    pub max_steps: usize,
    /// Timeout per step
    pub step_timeout: Duration,
    /// Total chain timeout
    pub chain_timeout: Duration,
    /// Continue on step errors
    pub continue_on_error: bool,
    /// Enable caching of intermediate results
    pub cache_intermediates: bool,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            step_timeout: Duration::from_secs(60),
            chain_timeout: Duration::from_secs(300),
            continue_on_error: false,
            cache_intermediates: true,
        }
    }
}

/// A step in a prompt chain
#[derive(Debug, Clone)]
pub struct ChainStep {
    /// Step name/identifier
    pub name: String,
    /// The prompt template (can include {{variables}})
    pub prompt_template: String,
    /// Model to use (None = use chain default)
    pub model: Option<String>,
    /// Variables to extract from response
    pub extract_vars: Vec<VariableExtraction>,
    /// Condition for executing this step
    pub condition: Option<StepCondition>,
    /// Next step on success (None = next in sequence)
    pub next_on_success: Option<String>,
    /// Next step on failure (None = abort)
    pub next_on_failure: Option<String>,
}

/// How to extract a variable from a response
#[derive(Debug, Clone)]
pub struct VariableExtraction {
    /// Variable name to store as
    pub name: String,
    /// Extraction method
    pub method: ExtractionMethod,
}

/// Methods for extracting data from responses
#[derive(Debug, Clone)]
pub enum ExtractionMethod {
    /// Use the full response
    FullResponse,
    /// Extract using regex
    Regex(String),
    /// Extract first code block
    FirstCodeBlock,
    /// Extract JSON field
    JsonField(String),
    /// Extract first line
    FirstLine,
    /// Extract last line
    LastLine,
    /// Custom extraction function name
    Custom(String),
}

/// Condition for step execution
#[derive(Debug, Clone)]
pub enum StepCondition {
    /// Variable equals value
    VarEquals(String, String),
    /// Variable contains value
    VarContains(String, String),
    /// Variable matches regex
    VarMatches(String, String),
    /// Variable exists
    VarExists(String),
    /// Custom condition function name
    Custom(String),
    /// All conditions must pass
    All(Vec<StepCondition>),
    /// Any condition must pass
    Any(Vec<StepCondition>),
    /// Negate condition
    Not(Box<StepCondition>),
}

impl StepCondition {
    /// Evaluate the condition
    pub fn evaluate(&self, vars: &HashMap<String, String>) -> bool {
        match self {
            StepCondition::VarEquals(name, value) => {
                vars.get(name).map(|v| v == value).unwrap_or(false)
            }
            StepCondition::VarContains(name, value) => {
                vars.get(name).map(|v| v.contains(value)).unwrap_or(false)
            }
            StepCondition::VarMatches(name, pattern) => {
                if let Some(value) = vars.get(name) {
                    regex::Regex::new(pattern)
                        .map(|re| re.is_match(value))
                        .unwrap_or(false)
                } else {
                    false
                }
            }
            StepCondition::VarExists(name) => vars.contains_key(name),
            StepCondition::Custom(_) => true, // Custom must be implemented externally
            StepCondition::All(conditions) => conditions.iter().all(|c| c.evaluate(vars)),
            StepCondition::Any(conditions) => conditions.iter().any(|c| c.evaluate(vars)),
            StepCondition::Not(condition) => !condition.evaluate(vars),
        }
    }
}

/// A complete prompt chain
#[derive(Debug, Clone)]
pub struct PromptChain {
    /// Chain name
    pub name: String,
    /// Steps in the chain
    pub steps: Vec<ChainStep>,
    /// Default model for all steps
    pub default_model: String,
    /// Initial variables
    pub initial_vars: HashMap<String, String>,
    /// Configuration
    pub config: ChainConfig,
}

impl PromptChain {
    /// Create a new prompt chain
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            default_model: model.into(),
            initial_vars: HashMap::new(),
            config: ChainConfig::default(),
        }
    }

    /// Add a step to the chain
    pub fn add_step(&mut self, step: ChainStep) {
        self.steps.push(step);
    }

    /// Set initial variable
    pub fn set_var(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.initial_vars.insert(name.into(), value.into());
    }

    /// Set configuration
    pub fn with_config(mut self, config: ChainConfig) -> Self {
        self.config = config;
        self
    }
}

/// Result of a chain step execution
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step name
    pub step_name: String,
    /// Whether step succeeded
    pub success: bool,
    /// Response from the model
    pub response: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Extracted variables
    pub extracted_vars: HashMap<String, String>,
    /// Time taken
    pub duration: Duration,
    /// Whether step was skipped (condition not met)
    pub skipped: bool,
}

/// Result of a complete chain execution
#[derive(Debug, Clone)]
pub struct ChainResult {
    /// Chain name
    pub chain_name: String,
    /// Overall success
    pub success: bool,
    /// Results from each step
    pub step_results: Vec<StepResult>,
    /// Final variables (accumulated)
    pub final_vars: HashMap<String, String>,
    /// Total time
    pub total_duration: Duration,
    /// Steps executed
    pub steps_executed: usize,
    /// Steps skipped
    pub steps_skipped: usize,
}

/// Executor for prompt chains
pub struct ChainExecutor {
    config: ChainConfig,
    /// Custom extractors
    custom_extractors: HashMap<String, Box<dyn Fn(&str) -> Option<String> + Send + Sync>>,
}

impl ChainExecutor {
    /// Create a new chain executor
    pub fn new(config: ChainConfig) -> Self {
        Self {
            config,
            custom_extractors: HashMap::new(),
        }
    }

    /// Register a custom extractor
    pub fn register_extractor<F>(&mut self, name: impl Into<String>, f: F)
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        self.custom_extractors.insert(name.into(), Box::new(f));
    }

    /// Execute a prompt chain
    pub fn execute<F>(&self, chain: &PromptChain, generate: F) -> ChainResult
    where
        F: Fn(&str, &str) -> Result<String, String>,
    {
        let start = Instant::now();
        let mut vars = chain.initial_vars.clone();
        let mut step_results = Vec::new();
        let mut steps_executed = 0;
        let mut steps_skipped = 0;
        let mut success = true;

        let mut current_step: Option<&str> = chain.steps.first().map(|s| s.name.as_str());

        while let Some(step_name) = current_step {
            // Check chain timeout
            if start.elapsed() > self.config.chain_timeout {
                success = false;
                break;
            }

            // Check step limit
            if steps_executed >= self.config.max_steps {
                success = false;
                break;
            }

            // Find the step
            let step = match chain.steps.iter().find(|s| s.name == step_name) {
                Some(s) => s,
                None => break,
            };

            // Check condition
            if let Some(ref condition) = step.condition {
                if !condition.evaluate(&vars) {
                    steps_skipped += 1;
                    step_results.push(StepResult {
                        step_name: step.name.clone(),
                        success: true,
                        response: None,
                        error: None,
                        extracted_vars: HashMap::new(),
                        duration: Duration::ZERO,
                        skipped: true,
                    });

                    // Move to next step in sequence
                    current_step =
                        self.get_next_step(chain, step_name, true, &step.next_on_success);
                    continue;
                }
            }

            // Interpolate prompt
            let prompt = self.interpolate_prompt(&step.prompt_template, &vars);
            let model = step.model.as_ref().unwrap_or(&chain.default_model);

            // Execute
            let step_start = Instant::now();
            let result = generate(&prompt, model);
            let duration = step_start.elapsed();
            steps_executed += 1;

            match result {
                Ok(response) => {
                    // Extract variables
                    let extracted = self.extract_variables(&response, &step.extract_vars);
                    vars.extend(extracted.clone());

                    step_results.push(StepResult {
                        step_name: step.name.clone(),
                        success: true,
                        response: Some(response),
                        error: None,
                        extracted_vars: extracted,
                        duration,
                        skipped: false,
                    });

                    current_step =
                        self.get_next_step(chain, step_name, true, &step.next_on_success);
                }
                Err(e) => {
                    step_results.push(StepResult {
                        step_name: step.name.clone(),
                        success: false,
                        response: None,
                        error: Some(e.clone()),
                        extracted_vars: HashMap::new(),
                        duration,
                        skipped: false,
                    });

                    if self.config.continue_on_error {
                        current_step =
                            self.get_next_step(chain, step_name, false, &step.next_on_failure);
                    } else {
                        success = false;
                        break;
                    }
                }
            }
        }

        ChainResult {
            chain_name: chain.name.clone(),
            success,
            step_results,
            final_vars: vars,
            total_duration: start.elapsed(),
            steps_executed,
            steps_skipped,
        }
    }

    fn get_next_step<'a>(
        &self,
        chain: &'a PromptChain,
        current: &str,
        _success: bool,
        explicit_next: &Option<String>,
    ) -> Option<&'a str> {
        // Use explicit next if specified
        if let Some(ref next) = explicit_next {
            return chain
                .steps
                .iter()
                .find(|s| &s.name == next)
                .map(|s| s.name.as_str());
        }

        // Otherwise, go to next in sequence
        let current_idx = chain.steps.iter().position(|s| s.name == current)?;
        chain.steps.get(current_idx + 1).map(|s| s.name.as_str())
    }

    fn interpolate_prompt(&self, template: &str, vars: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }
        result
    }

    fn extract_variables(
        &self,
        response: &str,
        extractions: &[VariableExtraction],
    ) -> HashMap<String, String> {
        let mut vars = HashMap::new();

        for extraction in extractions {
            if let Some(value) = self.extract_one(response, &extraction.method) {
                vars.insert(extraction.name.clone(), value);
            }
        }

        vars
    }

    fn extract_one(&self, response: &str, method: &ExtractionMethod) -> Option<String> {
        match method {
            ExtractionMethod::FullResponse => Some(response.to_string()),
            ExtractionMethod::Regex(pattern) => {
                let re = regex::Regex::new(pattern).ok()?;
                re.captures(response)
                    .and_then(|c| c.get(1).or_else(|| c.get(0)))
                    .map(|m| m.as_str().to_string())
            }
            ExtractionMethod::FirstCodeBlock => {
                let re = regex::Regex::new(r"```(?:\w+)?\n([\s\S]*?)```").ok()?;
                re.captures(response)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str().trim().to_string())
            }
            ExtractionMethod::JsonField(field) => {
                let json: serde_json::Value = serde_json::from_str(response).ok()?;
                json.get(field).map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| v.to_string())
                })
            }
            ExtractionMethod::FirstLine => response.lines().next().map(|s| s.to_string()),
            ExtractionMethod::LastLine => response.lines().last().map(|s| s.to_string()),
            ExtractionMethod::Custom(name) => {
                self.custom_extractors.get(name).and_then(|f| f(response))
            }
        }
    }
}

impl Default for ChainExecutor {
    fn default() -> Self {
        Self::new(ChainConfig::default())
    }
}

/// Builder for creating prompt chains
pub struct ChainBuilder {
    chain: PromptChain,
}

impl ChainBuilder {
    /// Create a new chain builder
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            chain: PromptChain::new(name, model),
        }
    }

    /// Add a simple step
    pub fn step(mut self, name: impl Into<String>, prompt: impl Into<String>) -> Self {
        self.chain.steps.push(ChainStep {
            name: name.into(),
            prompt_template: prompt.into(),
            model: None,
            extract_vars: Vec::new(),
            condition: None,
            next_on_success: None,
            next_on_failure: None,
        });
        self
    }

    /// Add a step with variable extraction
    pub fn step_with_extraction(
        mut self,
        name: impl Into<String>,
        prompt: impl Into<String>,
        var_name: impl Into<String>,
        method: ExtractionMethod,
    ) -> Self {
        self.chain.steps.push(ChainStep {
            name: name.into(),
            prompt_template: prompt.into(),
            model: None,
            extract_vars: vec![VariableExtraction {
                name: var_name.into(),
                method,
            }],
            condition: None,
            next_on_success: None,
            next_on_failure: None,
        });
        self
    }

    /// Add a conditional step
    pub fn conditional_step(
        mut self,
        name: impl Into<String>,
        prompt: impl Into<String>,
        condition: StepCondition,
    ) -> Self {
        self.chain.steps.push(ChainStep {
            name: name.into(),
            prompt_template: prompt.into(),
            model: None,
            extract_vars: Vec::new(),
            condition: Some(condition),
            next_on_success: None,
            next_on_failure: None,
        });
        self
    }

    /// Set initial variable
    pub fn var(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.chain.initial_vars.insert(name.into(), value.into());
        self
    }

    /// Set configuration
    pub fn config(mut self, config: ChainConfig) -> Self {
        self.chain.config = config;
        self
    }

    /// Build the chain
    pub fn build(self) -> PromptChain {
        self.chain
    }
}

/// Pre-built chain templates
pub struct ChainTemplates;

impl ChainTemplates {
    /// Chain for summarization with key points extraction
    pub fn summarize_with_points(model: impl Into<String>) -> PromptChain {
        ChainBuilder::new("summarize_with_points", model)
            .var("text", "")
            .step_with_extraction(
                "summarize",
                "Summarize the following text in 2-3 sentences:\n\n{{text}}",
                "summary",
                ExtractionMethod::FullResponse,
            )
            .step(
                "extract_points",
                "Extract 3-5 key points from this summary:\n\n{{summary}}",
            )
            .build()
    }

    /// Chain for code review
    pub fn code_review(model: impl Into<String>) -> PromptChain {
        ChainBuilder::new("code_review", model)
            .var("code", "")
            .step_with_extraction(
                "identify_issues",
                "Review this code and list any issues:\n\n```\n{{code}}\n```",
                "issues",
                ExtractionMethod::FullResponse,
            )
            .step(
                "suggest_fixes",
                "For each of these issues, suggest a fix:\n\n{{issues}}",
            )
            .step(
                "generate_improved",
                "Generate an improved version of the code:\n\n```\n{{code}}\n```",
            )
            .build()
    }

    /// Chain for translation with verification
    pub fn translate_and_verify(model: impl Into<String>) -> PromptChain {
        ChainBuilder::new("translate_verify", model)
            .var("text", "")
            .var("source_lang", "")
            .var("target_lang", "")
            .step_with_extraction(
                "translate",
                "Translate from {{source_lang}} to {{target_lang}}:\n\n{{text}}",
                "translation",
                ExtractionMethod::FullResponse,
            )
            .step_with_extraction(
                "back_translate",
                "Translate from {{target_lang}} to {{source_lang}}:\n\n{{translation}}",
                "back_translation",
                ExtractionMethod::FullResponse,
            )
            .step(
                "compare",
                "Compare original and back-translated text. Rate accuracy 1-10:\n\nOriginal: {{text}}\n\nBack-translated: {{back_translation}}",
            )
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_creation() {
        let chain = ChainBuilder::new("test", "model")
            .step("step1", "Hello")
            .step("step2", "World")
            .build();

        assert_eq!(chain.steps.len(), 2);
        assert_eq!(chain.name, "test");
    }

    #[test]
    fn test_variable_interpolation() {
        let executor = ChainExecutor::default();
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());

        let template = "Hello {{name}}!";
        let result = executor.interpolate_prompt(template, &vars);
        assert_eq!(result, "Hello Alice!");
    }

    #[test]
    fn test_condition_evaluation() {
        let mut vars = HashMap::new();
        vars.insert("status".to_string(), "success".to_string());

        let cond = StepCondition::VarEquals("status".to_string(), "success".to_string());
        assert!(cond.evaluate(&vars));

        let cond2 = StepCondition::VarEquals("status".to_string(), "failure".to_string());
        assert!(!cond2.evaluate(&vars));
    }

    #[test]
    fn test_chain_execution() {
        let chain = ChainBuilder::new("test", "model")
            .var("input", "Hello")
            .step_with_extraction(
                "step1",
                "Process: {{input}}",
                "output",
                ExtractionMethod::FullResponse,
            )
            .build();

        let executor = ChainExecutor::default();
        let result = executor.execute(&chain, |prompt, _model| {
            Ok(format!("Processed: {}", prompt))
        });

        assert!(result.success);
        assert_eq!(result.steps_executed, 1);
    }

    #[test]
    fn test_extraction_methods() {
        let executor = ChainExecutor::default();

        // First line
        let result = executor.extract_one("Line 1\nLine 2", &ExtractionMethod::FirstLine);
        assert_eq!(result, Some("Line 1".to_string()));

        // Last line
        let result = executor.extract_one("Line 1\nLine 2", &ExtractionMethod::LastLine);
        assert_eq!(result, Some("Line 2".to_string()));

        // Code block
        let result = executor.extract_one(
            "Here's code:\n```python\nprint('hello')\n```",
            &ExtractionMethod::FirstCodeBlock,
        );
        assert_eq!(result, Some("print('hello')".to_string()));
    }

    #[test]
    fn test_chain_templates() {
        let chain = ChainTemplates::summarize_with_points("gpt-4");
        assert_eq!(chain.steps.len(), 2);

        let chain = ChainTemplates::code_review("gpt-4");
        assert_eq!(chain.steps.len(), 3);
    }
}
