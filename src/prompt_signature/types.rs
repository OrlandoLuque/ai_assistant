//! Core types for prompt signatures: fields, signatures, compiled prompts, metrics, and budgets.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::AiError;

// ============================================================================
// Field Types and Signature Fields
// ============================================================================

/// The data type of a signature field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum FieldType {
    /// Free-form text
    Text,
    /// Numeric value
    Number,
    /// Boolean true/false
    Boolean,
    /// List of items
    List,
    /// Structured JSON
    Json,
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldType::Text => write!(f, "text"),
            FieldType::Number => write!(f, "number"),
            FieldType::Boolean => write!(f, "boolean"),
            FieldType::List => write!(f, "list"),
            FieldType::Json => write!(f, "json"),
        }
    }
}

/// A single field in a signature (either input or output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureField {
    /// Field name (used as variable name in templates)
    pub name: String,
    /// Human-readable description of what this field represents
    pub description: String,
    /// The data type of this field
    pub field_type: FieldType,
    /// Whether this field is required
    pub required: bool,
    /// Optional prefix text to prepend when rendering this field in a prompt
    pub prefix: Option<String>,
}

impl SignatureField {
    /// Create a new required text field.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            field_type: FieldType::Text,
            required: true,
            prefix: None,
        }
    }

    /// Set the field type.
    pub fn with_type(mut self, field_type: FieldType) -> Self {
        self.field_type = field_type;
        self
    }

    /// Mark the field as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Set a prompt prefix for this field.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }
}

// ============================================================================
// Signature
// ============================================================================

/// A declarative input/output specification for a prompt.
///
/// Signatures define what a prompt should do (its inputs and outputs) without
/// specifying how. The `compile()` method generates concrete prompt text, and
/// optimizers can search for better formulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    /// Name of this signature (e.g., "question_answering")
    pub name: String,
    /// Description of the task this signature represents
    pub description: String,
    /// Input fields
    pub inputs: Vec<SignatureField>,
    /// Output fields
    pub outputs: Vec<SignatureField>,
    /// Optional custom instructions to include in the system prompt
    pub instructions: Option<String>,
}

impl Signature {
    /// Create a new signature with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            instructions: None,
        }
    }

    /// Add an input field to this signature (builder pattern).
    pub fn add_input(mut self, field: SignatureField) -> Self {
        self.inputs.push(field);
        self
    }

    /// Add an output field to this signature (builder pattern).
    pub fn add_output(mut self, field: SignatureField) -> Self {
        self.outputs.push(field);
        self
    }

    /// Set custom instructions for this signature (builder pattern).
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Compile this signature into a concrete prompt template.
    ///
    /// The compiled prompt contains:
    /// - A system prompt derived from the signature description, fields, and instructions
    /// - A user template with placeholders for input values
    pub fn compile(&self) -> CompiledPrompt {
        self.compile_with_examples(&[])
    }

    /// Compile this signature into a prompt template, including few-shot examples.
    pub fn compile_with_examples(&self, examples: &[PromptExample]) -> CompiledPrompt {
        let mut system_parts: Vec<String> = Vec::new();

        // Task description
        system_parts.push(format!("Task: {}", self.description));

        // Custom instructions
        if let Some(ref instructions) = self.instructions {
            system_parts.push(format!("\n{}", instructions));
        }

        // Input field descriptions
        if !self.inputs.is_empty() {
            system_parts.push("\nInput fields:".to_string());
            for field in &self.inputs {
                let req = if field.required { "required" } else { "optional" };
                system_parts.push(format!(
                    "- {} ({}): {} [{}]",
                    field.name, field.field_type, field.description, req
                ));
            }
        }

        // Output field descriptions
        if !self.outputs.is_empty() {
            system_parts.push("\nOutput fields:".to_string());
            for field in &self.outputs {
                let req = if field.required { "required" } else { "optional" };
                system_parts.push(format!(
                    "- {} ({}): {} [{}]",
                    field.name, field.field_type, field.description, req
                ));
            }
        }

        // Build user template with placeholders
        let mut user_parts: Vec<String> = Vec::new();
        for field in &self.inputs {
            let prefix = field
                .prefix
                .as_deref()
                .unwrap_or(&field.name);
            user_parts.push(format!("{}: {{{}}}", prefix, field.name));
        }

        // Add output field labels so the model knows what to produce
        for field in &self.outputs {
            let prefix = field
                .prefix
                .as_deref()
                .unwrap_or(&field.name);
            user_parts.push(format!("{}: ", prefix));
        }

        CompiledPrompt {
            system_prompt: system_parts.join("\n"),
            user_template: user_parts.join("\n"),
            examples: examples.to_vec(),
        }
    }

    /// Validate that the given inputs satisfy this signature's requirements.
    pub fn validate_inputs(&self, inputs: &HashMap<String, String>) -> Result<(), AiError> {
        for field in &self.inputs {
            if field.required && !inputs.contains_key(&field.name) {
                return Err(AiError::other(format!(
                    "Missing required input field '{}' for signature '{}'",
                    field.name, self.name
                )));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Compiled Prompt and Examples
// ============================================================================

/// A compiled prompt generated from a signature.
///
/// Contains the system prompt, user template with placeholders, and any
/// few-shot examples that should be included.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPrompt {
    /// The system prompt describing the task and field specifications
    pub system_prompt: String,
    /// The user message template with `{field_name}` placeholders
    pub user_template: String,
    /// Few-shot examples to include in the prompt
    pub examples: Vec<PromptExample>,
}

impl CompiledPrompt {
    /// Render the user template by substituting input values for placeholders.
    pub fn render(&self, inputs: &HashMap<String, String>) -> String {
        let mut result = self.user_template.clone();
        for (key, value) in inputs {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Build the full prompt text including examples and rendered inputs.
    pub fn build_full_prompt(&self, inputs: &HashMap<String, String>) -> String {
        let mut parts: Vec<String> = Vec::new();

        parts.push(self.system_prompt.clone());

        // Add examples if any
        if !self.examples.is_empty() {
            parts.push("\n--- Examples ---".to_string());
            for (i, ex) in self.examples.iter().enumerate() {
                parts.push(format!("\nExample {}:", i + 1));
                for (k, v) in &ex.inputs {
                    parts.push(format!("  {}: {}", k, v));
                }
                for (k, v) in &ex.outputs {
                    parts.push(format!("  {}: {}", k, v));
                }
            }
            parts.push("\n--- End Examples ---".to_string());
        }

        parts.push(format!("\n{}", self.render(inputs)));

        parts.join("\n")
    }
}

/// A single input/output example for few-shot prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptExample {
    /// Input field values for this example
    pub inputs: HashMap<String, String>,
    /// Expected output field values for this example
    pub outputs: HashMap<String, String>,
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

/// Trait for evaluating prompt output quality.
///
/// Metrics compare a predicted output against an expected output and return
/// a score between 0.0 (worst) and 1.0 (best).
pub trait EvalMetric: Send + Sync {
    /// The name of this metric.
    fn name(&self) -> &str;

    /// Score the predicted output against the expected output.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 is a perfect match.
    fn score(&self, predicted: &str, expected: &str) -> f64;
}

/// Exact string match metric.
///
/// Returns 1.0 if predicted equals expected (case-insensitive, trimmed), 0.0 otherwise.
pub struct ExactMatch;

impl EvalMetric for ExactMatch {
    fn name(&self) -> &str {
        "exact_match"
    }

    fn score(&self, predicted: &str, expected: &str) -> f64 {
        if predicted.trim().eq_ignore_ascii_case(expected.trim()) {
            1.0
        } else {
            0.0
        }
    }
}

/// Token-level F1 score metric.
///
/// Computes precision and recall over whitespace-separated tokens, then
/// returns their harmonic mean.
pub struct F1Score;

impl F1Score {
    /// Tokenize a string by splitting on whitespace and lowercasing.
    fn tokenize(s: &str) -> Vec<String> {
        s.split_whitespace()
            .map(|t| t.to_lowercase())
            .collect()
    }
}

impl EvalMetric for F1Score {
    fn name(&self) -> &str {
        "f1_score"
    }

    fn score(&self, predicted: &str, expected: &str) -> f64 {
        let pred_tokens = Self::tokenize(predicted);
        let exp_tokens = Self::tokenize(expected);

        if pred_tokens.is_empty() && exp_tokens.is_empty() {
            return 1.0;
        }
        if pred_tokens.is_empty() || exp_tokens.is_empty() {
            return 0.0;
        }

        // Count matches (each expected token can match at most once)
        let mut matched = vec![false; exp_tokens.len()];
        let mut true_positives = 0usize;

        for pt in &pred_tokens {
            for (i, et) in exp_tokens.iter().enumerate() {
                if !matched[i] && pt == et {
                    matched[i] = true;
                    true_positives += 1;
                    break;
                }
            }
        }

        let precision = true_positives as f64 / pred_tokens.len() as f64;
        let recall = true_positives as f64 / exp_tokens.len() as f64;

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

/// Contains-answer metric.
///
/// Returns 1.0 if the predicted output contains the expected answer
/// (case-insensitive), 0.0 otherwise.
pub struct ContainsAnswer;

impl EvalMetric for ContainsAnswer {
    fn name(&self) -> &str {
        "contains_answer"
    }

    fn score(&self, predicted: &str, expected: &str) -> f64 {
        if predicted.to_lowercase().contains(&expected.to_lowercase()) {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// Evaluation Budget
// ============================================================================

/// Controls how many trials and examples an optimizer is allowed to use.
#[derive(Debug, Clone)]
pub struct EvaluationBudget {
    /// Maximum number of optimization trials
    pub max_trials: usize,
    /// Maximum number of examples to evaluate per trial
    pub max_examples: usize,
    /// Optional timeout in seconds (not enforced here; for callers to check)
    pub timeout_seconds: Option<u64>,
    /// Number of trials already consumed
    trials_used: usize,
}

impl EvaluationBudget {
    /// Create a new budget with the given limits.
    pub fn new(max_trials: usize, max_examples: usize) -> Self {
        Self {
            max_trials,
            max_examples,
            timeout_seconds: None,
            trials_used: 0,
        }
    }

    /// Set an optional timeout.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Try to consume one trial from the budget.
    ///
    /// Returns `true` if there is budget remaining and the trial was consumed,
    /// `false` if the budget is exhausted.
    pub fn try_use(&mut self) -> bool {
        if self.trials_used < self.max_trials {
            self.trials_used += 1;
            true
        } else {
            false
        }
    }

    /// Return the number of remaining trials.
    pub fn remaining(&self) -> usize {
        self.max_trials.saturating_sub(self.trials_used)
    }

    /// Return how many trials have been used.
    pub fn used(&self) -> usize {
        self.trials_used
    }

    /// Reset the budget to its initial state.
    pub fn reset(&mut self) {
        self.trials_used = 0;
    }
}

// ============================================================================
// Training Example
// ============================================================================

/// A single training example for prompt optimization.
///
/// Contains input values and expected output values that optimizers use
/// to evaluate and improve prompt formulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input field values
    pub inputs: HashMap<String, String>,
    /// Expected output field values
    pub expected_outputs: HashMap<String, String>,
}

// ============================================================================
// Optimization Result
// ============================================================================

/// The result of a prompt optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// The best compiled prompt found during optimization
    pub best_prompt: CompiledPrompt,
    /// The score achieved by the best prompt
    pub best_score: f64,
    /// Number of trials that were run
    pub trials_run: usize,
    /// Score history across all trials (in order)
    pub scores_history: Vec<f64>,
}
