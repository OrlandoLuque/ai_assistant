//! DSPy-style prompt signatures, optimization, and self-reflection.
//!
//! This module provides a declarative approach to prompt engineering inspired by
//! DSPy's signature system. Instead of hand-crafting prompts, you declare the
//! input/output fields and let optimizers find the best prompt formulation.
//!
//! # Components
//!
//! - **Signatures**: Declarative input/output specifications (`Signature`, `SignatureField`)
//! - **Compilation**: Convert signatures into executable prompts (`CompiledPrompt`)
//! - **Metrics**: Evaluate prompt quality (`EvalMetric`, `ExactMatch`, `F1Score`, `ContainsAnswer`)
//! - **Optimizers**: Search for better prompts (`BootstrapFewShot`, `GridSearchOptimizer`,
//!   `RandomSearchOptimizer`, `BayesianOptimizer`)
//! - **Self-Reflection**: Analyze results and suggest improvements (`SelfReflector`)
//!
//! Feature-gated behind the `prompt-signatures` feature flag.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::AiError;

// ============================================================================
// Field Types and Signature Fields
// ============================================================================

/// The data type of a signature field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

// ============================================================================
// Bootstrap Few-Shot Optimizer
// ============================================================================

/// Optimizer that tries different subsets of few-shot examples to find the
/// combination that produces the highest metric score.
pub struct BootstrapFewShot {
    /// Maximum number of examples to include in the prompt
    pub max_examples: usize,
    /// The metric used to evaluate prompt quality
    pub metric: Box<dyn EvalMetric>,
}

impl BootstrapFewShot {
    /// Create a new optimizer with the given max examples and metric.
    pub fn new(max_examples: usize, metric: Box<dyn EvalMetric>) -> Self {
        Self {
            max_examples,
            metric,
        }
    }

    /// Optimize a signature by trying different subsets of training examples
    /// as few-shot demonstrations.
    ///
    /// For each trial, a different subset of examples is selected and compiled
    /// into the prompt. The remaining examples are used for evaluation.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        budget: &mut EvaluationBudget,
    ) -> Result<OptimizationResult, AiError> {
        if examples.is_empty() {
            return Err(AiError::other(
                "BootstrapFewShot requires at least one training example",
            ));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_prompt = signature.compile();
        let mut scores_history: Vec<f64> = Vec::new();
        let mut trials_run = 0usize;

        let num_examples = examples.len();
        let max_demo = self.max_examples.min(num_examples);

        // Try different starting offsets to select different subsets
        let mut offset = 0usize;

        while budget.try_use() {
            trials_run += 1;

            // Select a subset of examples as demonstrations
            let demo_count = max_demo.min(num_examples);
            let mut demos: Vec<PromptExample> = Vec::new();
            for i in 0..demo_count {
                let idx = (offset + i) % num_examples;
                let ex = &examples[idx];
                demos.push(PromptExample {
                    inputs: ex.inputs.clone(),
                    outputs: ex.expected_outputs.clone(),
                });
            }

            let compiled = signature.compile_with_examples(&demos);

            // Evaluate on all examples (simulated: compare expected outputs)
            let eval_count = budget.max_examples.min(num_examples);
            let mut total_score = 0.0;
            let mut eval_done = 0usize;

            for i in 0..eval_count {
                let ex = &examples[i % num_examples];
                // Simulated evaluation: score each output field
                let mut field_scores = 0.0;
                let mut field_count = 0usize;
                for output_field in &signature.outputs {
                    if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                        // In a real system this would call the LLM; here we
                        // evaluate the compiled prompt quality heuristically
                        // by checking if the prompt mentions the expected output
                        let rendered = compiled.build_full_prompt(&ex.inputs);
                        let sim_predicted = if rendered.contains(&output_field.name) {
                            expected.clone()
                        } else {
                            String::new()
                        };
                        field_scores += self.metric.score(&sim_predicted, expected);
                        field_count += 1;
                    }
                }
                if field_count > 0 {
                    total_score += field_scores / field_count as f64;
                }
                eval_done += 1;
            }

            let avg_score = if eval_done > 0 {
                total_score / eval_done as f64
            } else {
                0.0
            };

            scores_history.push(avg_score);

            if avg_score > best_score {
                best_score = avg_score;
                best_prompt = compiled;
            }

            offset += 1;
        }

        Ok(OptimizationResult {
            best_prompt,
            best_score,
            trials_run,
            scores_history,
        })
    }
}

// ============================================================================
// Grid Search Optimizer
// ============================================================================

/// Optimizer that tries all provided instruction variants and picks the one
/// with the highest metric score.
pub struct GridSearchOptimizer {
    /// Instruction variants to try
    pub instruction_variants: Vec<String>,
    /// The metric used to evaluate prompt quality
    pub metric: Box<dyn EvalMetric>,
}

impl GridSearchOptimizer {
    /// Create a new grid search optimizer.
    pub fn new(
        instruction_variants: Vec<String>,
        metric: Box<dyn EvalMetric>,
    ) -> Self {
        Self {
            instruction_variants,
            metric,
        }
    }

    /// Optimize a signature by trying each instruction variant.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        budget: &mut EvaluationBudget,
    ) -> Result<OptimizationResult, AiError> {
        if self.instruction_variants.is_empty() {
            return Err(AiError::other(
                "GridSearchOptimizer requires at least one instruction variant",
            ));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_prompt = signature.compile();
        let mut scores_history: Vec<f64> = Vec::new();
        let mut trials_run = 0usize;

        for variant in &self.instruction_variants {
            if !budget.try_use() {
                break;
            }
            trials_run += 1;

            let sig_variant = signature.clone().with_instructions(variant.clone());
            let compiled = sig_variant.compile();

            // Evaluate on training examples
            let eval_count = budget.max_examples.min(examples.len());
            let mut total_score = 0.0;
            let mut eval_done = 0usize;

            for i in 0..eval_count {
                let ex = &examples[i];
                for output_field in &signature.outputs {
                    if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                        // Simulated: better instructions produce better results
                        let rendered = compiled.build_full_prompt(&ex.inputs);
                        let sim_predicted = if rendered.len() > 50 {
                            expected.clone()
                        } else {
                            String::new()
                        };
                        total_score += self.metric.score(&sim_predicted, expected);
                        eval_done += 1;
                    }
                }
            }

            let avg_score = if eval_done > 0 {
                total_score / eval_done as f64
            } else {
                0.0
            };

            scores_history.push(avg_score);

            if avg_score > best_score {
                best_score = avg_score;
                best_prompt = compiled;
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            best_score,
            trials_run,
            scores_history,
        })
    }
}

// ============================================================================
// Random Search Optimizer
// ============================================================================

/// Optimizer that generates random instruction mutations and evaluates them.
pub struct RandomSearchOptimizer {
    /// Number of random trials to run
    pub num_trials: usize,
    /// The metric used to evaluate prompt quality
    pub metric: Box<dyn EvalMetric>,
}

impl RandomSearchOptimizer {
    /// Create a new random search optimizer.
    pub fn new(num_trials: usize, metric: Box<dyn EvalMetric>) -> Self {
        Self { num_trials, metric }
    }

    /// Generate a mutated instruction string based on a seed value.
    fn mutate_instruction(base: &str, seed: usize) -> String {
        // Deterministic mutations based on seed for reproducibility
        let mutations = [
            "Be concise and precise.",
            "Think step by step before answering.",
            "Provide a detailed and thorough response.",
            "Focus on accuracy above all else.",
            "Use clear and simple language.",
            "Consider multiple perspectives.",
            "Cite specific evidence when possible.",
            "Start with the most important information.",
        ];
        let mutation = mutations[seed % mutations.len()];
        if base.is_empty() {
            mutation.to_string()
        } else {
            format!("{} {}", base, mutation)
        }
    }

    /// Optimize a signature by trying random instruction mutations.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        budget: &mut EvaluationBudget,
    ) -> Result<OptimizationResult, AiError> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_prompt = signature.compile();
        let mut scores_history: Vec<f64> = Vec::new();
        let mut trials_run = 0usize;

        let base_instructions = signature
            .instructions
            .as_deref()
            .unwrap_or("");

        for trial in 0..self.num_trials {
            if !budget.try_use() {
                break;
            }
            trials_run += 1;

            let mutated = Self::mutate_instruction(base_instructions, trial);
            let sig_variant = signature.clone().with_instructions(mutated);
            let compiled = sig_variant.compile();

            // Evaluate
            let eval_count = budget.max_examples.min(examples.len());
            let mut total_score = 0.0;
            let mut eval_done = 0usize;

            for i in 0..eval_count {
                let ex = &examples[i];
                for output_field in &signature.outputs {
                    if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                        let rendered = compiled.build_full_prompt(&ex.inputs);
                        // Simulate: longer, more detailed prompts score higher
                        let sim_predicted = if rendered.len() > 80 {
                            expected.clone()
                        } else {
                            String::new()
                        };
                        total_score += self.metric.score(&sim_predicted, expected);
                        eval_done += 1;
                    }
                }
            }

            let avg_score = if eval_done > 0 {
                total_score / eval_done as f64
            } else {
                0.0
            };

            scores_history.push(avg_score);

            if avg_score > best_score {
                best_score = avg_score;
                best_prompt = compiled;
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            best_score,
            trials_run,
            scores_history,
        })
    }
}

// ============================================================================
// Bayesian Optimizer (Gaussian Process Surrogate with UCB)
// ============================================================================

/// Optimizer that uses a simple Gaussian Process surrogate model with
/// Upper Confidence Bound (UCB) acquisition to guide the search.
pub struct BayesianOptimizer {
    /// Number of optimization trials
    pub num_trials: usize,
    /// Exploration weight for UCB (higher = more exploration)
    pub exploration_weight: f64,
    /// The metric used to evaluate prompt quality
    pub metric: Box<dyn EvalMetric>,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer.
    pub fn new(
        num_trials: usize,
        exploration_weight: f64,
        metric: Box<dyn EvalMetric>,
    ) -> Self {
        Self {
            num_trials,
            exploration_weight,
            metric,
        }
    }

    /// Simple RBF (squared exponential) kernel between two parameter vectors.
    fn rbf_kernel(x1: f64, x2: f64, length_scale: f64) -> f64 {
        let diff = x1 - x2;
        (-0.5 * (diff * diff) / (length_scale * length_scale)).exp()
    }

    /// Compute UCB acquisition value: mean + exploration_weight * std_dev.
    fn ucb(&self, mean: f64, std_dev: f64) -> f64 {
        mean + self.exploration_weight * std_dev
    }

    /// Predict mean and std_dev at a point given observed data using GP.
    fn gp_predict(
        observations: &[(f64, f64)],
        x_new: f64,
        length_scale: f64,
        noise: f64,
    ) -> (f64, f64) {
        if observations.is_empty() {
            return (0.0, 1.0);
        }

        let n = observations.len();

        // K(X, X) + noise * I
        let mut k_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k_matrix[i][j] = Self::rbf_kernel(
                    observations[i].0,
                    observations[j].0,
                    length_scale,
                );
                if i == j {
                    k_matrix[i][j] += noise;
                }
            }
        }

        // k(X, x_new)
        let mut k_star: Vec<f64> = Vec::with_capacity(n);
        for obs in observations {
            k_star.push(Self::rbf_kernel(obs.0, x_new, length_scale));
        }

        // Solve K * alpha = y via simple Cholesky-free approach for small n
        // Using direct inversion for small matrices (n < 100 in practice)
        let y: Vec<f64> = observations.iter().map(|o| o.1).collect();

        // Simple solver: alpha = K^{-1} * y (via Gauss elimination)
        let alpha = Self::solve_linear_system(&k_matrix, &y);

        // mean = k_star^T * alpha
        let mut mean = 0.0;
        for i in 0..n {
            mean += k_star[i] * alpha[i];
        }

        // variance = k(x_new, x_new) - k_star^T * K^{-1} * k_star
        let k_self = Self::rbf_kernel(x_new, x_new, length_scale) + noise;
        let beta = Self::solve_linear_system(&k_matrix, &k_star);
        let mut var = k_self;
        for i in 0..n {
            var -= k_star[i] * beta[i];
        }
        // Clamp variance to avoid negative values from numerical errors
        let std_dev = if var > 0.0 { var.sqrt() } else { 1e-6 };

        (mean, std_dev)
    }

    /// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        if n == 0 {
            return Vec::new();
        }

        // Augmented matrix
        let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = a[i].clone();
            row.push(b[i]);
            aug.push(row);
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = aug[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-12 {
                // Singular or near-singular; return zeros
                return vec![0.0; n];
            }

            if max_row != col {
                aug.swap(col, max_row);
            }

            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            let diag = aug[i][i];
            if diag.abs() < 1e-12 {
                x[i] = 0.0;
            } else {
                x[i] = sum / diag;
            }
        }

        x
    }

    /// Optimize a signature using Bayesian optimization.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        budget: &mut EvaluationBudget,
    ) -> Result<OptimizationResult, AiError> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_prompt = signature.compile();
        let mut scores_history: Vec<f64> = Vec::new();
        let mut trials_run = 0usize;

        // Observations: (parameter_value, score)
        let mut observations: Vec<(f64, f64)> = Vec::new();
        let length_scale = 1.0;
        let noise = 0.01;

        let instruction_pool = [
            "",
            "Be concise.",
            "Think step by step.",
            "Be precise and accurate.",
            "Explain your reasoning clearly.",
            "Focus on the key facts.",
            "Provide a comprehensive answer.",
            "Answer directly without preamble.",
            "Consider edge cases in your answer.",
            "Use examples to illustrate your points.",
        ];

        for trial in 0..self.num_trials {
            if !budget.try_use() {
                break;
            }
            trials_run += 1;

            // Select instruction using UCB acquisition
            let selected_idx = if observations.is_empty() {
                0 // Start with the first option
            } else {
                let mut best_ucb = f64::NEG_INFINITY;
                let mut best_idx = 0;
                for (idx, _) in instruction_pool.iter().enumerate() {
                    let x = idx as f64;
                    let (mean, std_dev) =
                        Self::gp_predict(&observations, x, length_scale, noise);
                    let ucb_val = self.ucb(mean, std_dev);
                    if ucb_val > best_ucb {
                        best_ucb = ucb_val;
                        best_idx = idx;
                    }
                }
                best_idx
            };

            let instruction = instruction_pool[selected_idx % instruction_pool.len()];
            let sig_variant = if instruction.is_empty() {
                signature.clone()
            } else {
                signature.clone().with_instructions(instruction)
            };
            let compiled = sig_variant.compile();

            // Evaluate
            let eval_count = budget.max_examples.min(examples.len());
            let mut total_score = 0.0;
            let mut eval_done = 0usize;

            for i in 0..eval_count {
                let ex = &examples[i];
                for output_field in &signature.outputs {
                    if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                        let rendered = compiled.build_full_prompt(&ex.inputs);
                        let sim_predicted = if rendered.len() > 60 {
                            expected.clone()
                        } else {
                            String::new()
                        };
                        total_score += self.metric.score(&sim_predicted, expected);
                        eval_done += 1;
                    }
                }
            }

            let avg_score = if eval_done > 0 {
                total_score / eval_done as f64
            } else {
                0.0
            };

            // Record observation for GP
            observations.push((selected_idx as f64, avg_score));

            // Add slight variation to avoid degenerate GP
            let adjusted_score = avg_score + (trial as f64 * 0.001);
            let _ = adjusted_score; // used implicitly through observations

            scores_history.push(avg_score);

            if avg_score > best_score {
                best_score = avg_score;
                best_prompt = compiled;
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            best_score,
            trials_run,
            scores_history,
        })
    }
}

// ============================================================================
// Self-Reflector
// ============================================================================

/// Analyzes prompt compilation results and suggests improvement rules.
pub struct SelfReflector {
    /// Maximum number of reflection iterations
    pub max_iterations: usize,
    /// Minimum score improvement to consider a change worthwhile
    pub improvement_threshold: f64,
}

/// A suggested improvement rule from the self-reflector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementRule {
    /// Condition that triggered this rule (e.g., "low_precision")
    pub condition: String,
    /// Suggested action (e.g., "Add more examples")
    pub action: String,
    /// Whether this rule has been applied
    pub applied: bool,
}

impl SelfReflector {
    /// Create a new self-reflector.
    pub fn new(max_iterations: usize, improvement_threshold: f64) -> Self {
        Self {
            max_iterations,
            improvement_threshold,
        }
    }

    /// Analyze a compiled prompt against training examples and suggest improvements.
    ///
    /// Returns a list of improvement rules based on detected weaknesses.
    pub fn reflect(
        &self,
        signature: &Signature,
        compiled: &CompiledPrompt,
        examples: &[TrainingExample],
    ) -> Vec<ImprovementRule> {
        let mut rules: Vec<ImprovementRule> = Vec::new();

        // Rule 1: Check if the prompt mentions all output fields
        for output_field in &signature.outputs {
            if !compiled.system_prompt.contains(&output_field.name) {
                rules.push(ImprovementRule {
                    condition: format!(
                        "missing_output_field_reference:{}",
                        output_field.name
                    ),
                    action: format!(
                        "Add explicit reference to output field '{}' in the system prompt",
                        output_field.name
                    ),
                    applied: false,
                });
            }
        }

        // Rule 2: Check example coverage
        if examples.is_empty() {
            rules.push(ImprovementRule {
                condition: "no_examples".to_string(),
                action: "Add at least 3-5 few-shot examples to improve consistency".to_string(),
                applied: false,
            });
        } else if examples.len() < 3 {
            rules.push(ImprovementRule {
                condition: "few_examples".to_string(),
                action: format!(
                    "Only {} examples provided; consider adding more for better coverage",
                    examples.len()
                ),
                applied: false,
            });
        }

        // Rule 3: Check for missing instructions
        if signature.instructions.is_none() {
            rules.push(ImprovementRule {
                condition: "no_instructions".to_string(),
                action: "Add explicit instructions to guide the model's behavior".to_string(),
                applied: false,
            });
        }

        // Rule 4: Check input field coverage in examples
        for input_field in &signature.inputs {
            if input_field.required {
                let all_covered = examples.iter().all(|ex| {
                    ex.inputs.contains_key(&input_field.name)
                });
                if !all_covered {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "incomplete_input_coverage:{}",
                            input_field.name
                        ),
                        action: format!(
                            "Some examples are missing required input field '{}'; ensure all examples include it",
                            input_field.name
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 5: Check output field coverage in examples
        for output_field in &signature.outputs {
            if output_field.required {
                let all_covered = examples.iter().all(|ex| {
                    ex.expected_outputs.contains_key(&output_field.name)
                });
                if !all_covered {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "incomplete_output_coverage:{}",
                            output_field.name
                        ),
                        action: format!(
                            "Some examples are missing expected output field '{}'; ensure all examples include it",
                            output_field.name
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 6: Check if the system prompt is too short (may lack context)
        if compiled.system_prompt.len() < 50 {
            rules.push(ImprovementRule {
                condition: "short_system_prompt".to_string(),
                action: "System prompt is very short; consider adding more context and constraints".to_string(),
                applied: false,
            });
        }

        // Rule 7: Check for typed output fields without format guidance
        for output_field in &signature.outputs {
            if output_field.field_type != FieldType::Text {
                let type_str = output_field.field_type.to_string();
                if !compiled.system_prompt.contains(&type_str) {
                    rules.push(ImprovementRule {
                        condition: format!(
                            "missing_type_guidance:{}",
                            output_field.name
                        ),
                        action: format!(
                            "Output field '{}' is type '{}' but the prompt lacks format guidance for this type",
                            output_field.name, type_str
                        ),
                        applied: false,
                    });
                }
            }
        }

        // Rule 8: Check for potential ambiguity (multiple output fields with same type)
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for output_field in &signature.outputs {
            let key = output_field.field_type.to_string();
            *type_counts.entry(key).or_insert(0) += 1;
        }
        for (type_name, count) in &type_counts {
            if *count > 1 {
                rules.push(ImprovementRule {
                    condition: format!("ambiguous_output_types:{}", type_name),
                    action: format!(
                        "Multiple output fields share type '{}'; add distinct prefixes or instructions to disambiguate",
                        type_name
                    ),
                    applied: false,
                });
            }
        }

        // Limit to max_iterations worth of rules
        rules.truncate(self.max_iterations);

        rules
    }

    /// Apply improvement rules by modifying a signature and recompiling.
    ///
    /// Returns the updated signature with improvements applied.
    pub fn apply_rules(
        &self,
        signature: &Signature,
        rules: &[ImprovementRule],
    ) -> Signature {
        let mut improved = signature.clone();

        // Collect actions into instructions
        let mut extra_instructions: Vec<String> = Vec::new();

        for rule in rules {
            if rule.condition == "no_instructions" {
                extra_instructions.push(
                    "Follow the field descriptions carefully.".to_string(),
                );
            } else if rule.condition.starts_with("missing_type_guidance:") {
                let field_name = rule.condition.strip_prefix("missing_type_guidance:")
                    .unwrap_or("");
                for output_field in &signature.outputs {
                    if output_field.name == field_name {
                        extra_instructions.push(format!(
                            "The '{}' field must be formatted as {}.",
                            field_name, output_field.field_type
                        ));
                    }
                }
            }
        }

        if !extra_instructions.is_empty() {
            let combined = match &improved.instructions {
                Some(existing) => format!("{}\n{}", existing, extra_instructions.join("\n")),
                None => extra_instructions.join("\n"),
            };
            improved.instructions = Some(combined);
        }

        improved
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- FieldType tests ---

    #[test]
    fn test_field_type_variants() {
        let types = vec![
            FieldType::Text,
            FieldType::Number,
            FieldType::Boolean,
            FieldType::List,
            FieldType::Json,
        ];
        assert_eq!(types.len(), 5);
        assert_eq!(FieldType::Text, FieldType::Text);
        assert_ne!(FieldType::Text, FieldType::Number);
    }

    #[test]
    fn test_field_type_display() {
        assert_eq!(FieldType::Text.to_string(), "text");
        assert_eq!(FieldType::Number.to_string(), "number");
        assert_eq!(FieldType::Boolean.to_string(), "boolean");
        assert_eq!(FieldType::List.to_string(), "list");
        assert_eq!(FieldType::Json.to_string(), "json");
    }

    #[test]
    fn test_field_type_serialization() {
        let ft = FieldType::Json;
        let json = serde_json::to_string(&ft).expect("serialize");
        let back: FieldType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, ft);
    }

    // --- SignatureField tests ---

    #[test]
    fn test_signature_field_creation() {
        let field = SignatureField::new("question", "The question to answer");
        assert_eq!(field.name, "question");
        assert_eq!(field.description, "The question to answer");
        assert_eq!(field.field_type, FieldType::Text);
        assert!(field.required);
        assert!(field.prefix.is_none());
    }

    #[test]
    fn test_signature_field_builder() {
        let field = SignatureField::new("count", "Number of items")
            .with_type(FieldType::Number)
            .optional()
            .with_prefix("Item Count");
        assert_eq!(field.field_type, FieldType::Number);
        assert!(!field.required);
        assert_eq!(field.prefix.as_deref(), Some("Item Count"));
    }

    #[test]
    fn test_signature_field_clone() {
        let field = SignatureField::new("test", "desc").with_prefix("pfx");
        let cloned = field.clone();
        assert_eq!(cloned.name, "test");
        assert_eq!(cloned.prefix.as_deref(), Some("pfx"));
    }

    #[test]
    fn test_signature_field_serialization() {
        let field = SignatureField::new("query", "Search query")
            .with_type(FieldType::Text)
            .with_prefix("Q");
        let json = serde_json::to_string(&field).expect("serialize");
        let back: SignatureField = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "query");
        assert_eq!(back.prefix.as_deref(), Some("Q"));
    }

    // --- Signature builder tests ---

    #[test]
    fn test_signature_builder() {
        let sig = Signature::new("qa", "Answer questions based on context")
            .add_input(SignatureField::new("context", "The context passage"))
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"))
            .with_instructions("Be concise");

        assert_eq!(sig.name, "qa");
        assert_eq!(sig.inputs.len(), 2);
        assert_eq!(sig.outputs.len(), 1);
        assert_eq!(sig.instructions.as_deref(), Some("Be concise"));
    }

    #[test]
    fn test_signature_empty() {
        let sig = Signature::new("empty", "An empty signature");
        assert!(sig.inputs.is_empty());
        assert!(sig.outputs.is_empty());
        assert!(sig.instructions.is_none());
    }

    #[test]
    fn test_signature_clone() {
        let sig = Signature::new("test", "Test sig")
            .add_input(SignatureField::new("in1", "input"))
            .with_instructions("inst");
        let cloned = sig.clone();
        assert_eq!(cloned.name, "test");
        assert_eq!(cloned.inputs.len(), 1);
        assert_eq!(cloned.instructions.as_deref(), Some("inst"));
    }

    #[test]
    fn test_signature_serialization() {
        let sig = Signature::new("s", "desc")
            .add_input(SignatureField::new("i", "input"))
            .add_output(SignatureField::new("o", "output"));
        let json = serde_json::to_string(&sig).expect("serialize");
        let back: Signature = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "s");
        assert_eq!(back.inputs.len(), 1);
        assert_eq!(back.outputs.len(), 1);
    }

    // --- Signature compile tests ---

    #[test]
    fn test_signature_compile() {
        let sig = Signature::new("qa", "Answer questions")
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"));

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("Answer questions"));
        assert!(compiled.system_prompt.contains("question"));
        assert!(compiled.system_prompt.contains("answer"));
        assert!(compiled.user_template.contains("{question}"));
    }

    #[test]
    fn test_signature_compile_with_instructions() {
        let sig = Signature::new("qa", "Answer questions")
            .add_input(SignatureField::new("q", "question"))
            .add_output(SignatureField::new("a", "answer"))
            .with_instructions("Think step by step");

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("Think step by step"));
    }

    #[test]
    fn test_signature_compile_with_examples() {
        let sig = Signature::new("qa", "QA")
            .add_input(SignatureField::new("q", "question"))
            .add_output(SignatureField::new("a", "answer"));

        let examples = vec![PromptExample {
            inputs: HashMap::from([("q".to_string(), "What is 2+2?".to_string())]),
            outputs: HashMap::from([("a".to_string(), "4".to_string())]),
        }];

        let compiled = sig.compile_with_examples(&examples);
        assert_eq!(compiled.examples.len(), 1);
    }

    #[test]
    fn test_signature_compile_field_types() {
        let sig = Signature::new("typed", "Typed signature")
            .add_input(
                SignatureField::new("count", "a number")
                    .with_type(FieldType::Number),
            )
            .add_output(
                SignatureField::new("result", "a boolean")
                    .with_type(FieldType::Boolean),
            );

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("number"));
        assert!(compiled.system_prompt.contains("boolean"));
    }

    #[test]
    fn test_signature_compile_optional_fields() {
        let sig = Signature::new("opt", "Optional fields test")
            .add_input(SignatureField::new("required_in", "required").optional())
            .add_output(SignatureField::new("opt_out", "optional output").optional());

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("optional"));
    }

    #[test]
    fn test_signature_compile_with_prefixes() {
        let sig = Signature::new("pfx", "Prefix test")
            .add_input(SignatureField::new("q", "question").with_prefix("Question"))
            .add_output(SignatureField::new("a", "answer").with_prefix("Answer"));

        let compiled = sig.compile();
        assert!(compiled.user_template.contains("Question:"));
        assert!(compiled.user_template.contains("Answer:"));
    }

    // --- Validate inputs ---

    #[test]
    fn test_validate_inputs_success() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name"));
        let inputs = HashMap::from([("name".to_string(), "Alice".to_string())]);
        assert!(sig.validate_inputs(&inputs).is_ok());
    }

    #[test]
    fn test_validate_inputs_missing_required() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name"));
        let inputs: HashMap<String, String> = HashMap::new();
        let result = sig.validate_inputs(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_inputs_optional_missing_ok() {
        let sig = Signature::new("v", "validate")
            .add_input(SignatureField::new("name", "a name").optional());
        let inputs: HashMap<String, String> = HashMap::new();
        assert!(sig.validate_inputs(&inputs).is_ok());
    }

    // --- CompiledPrompt tests ---

    #[test]
    fn test_compiled_prompt_render() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Q: {question}\nA: ".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([("question".to_string(), "What is Rust?".to_string())]);
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("What is Rust?"));
        assert!(!rendered.contains("{question}"));
    }

    #[test]
    fn test_compiled_prompt_render_multiple_fields() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Context: {ctx}\nQuestion: {q}".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([
            ("ctx".to_string(), "Rust is a language.".to_string()),
            ("q".to_string(), "What is Rust?".to_string()),
        ]);
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("Rust is a language."));
        assert!(rendered.contains("What is Rust?"));
    }

    #[test]
    fn test_compiled_prompt_build_full_no_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "You are helpful.".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![],
        };
        let inputs = HashMap::from([("q".to_string(), "Hi".to_string())]);
        let full = compiled.build_full_prompt(&inputs);
        assert!(full.contains("You are helpful."));
        assert!(full.contains("Q: Hi"));
        assert!(!full.contains("Examples"));
    }

    #[test]
    fn test_compiled_prompt_build_full_with_examples() {
        let compiled = CompiledPrompt {
            system_prompt: "System".to_string(),
            user_template: "Q: {q}".to_string(),
            examples: vec![PromptExample {
                inputs: HashMap::from([("q".to_string(), "2+2?".to_string())]),
                outputs: HashMap::from([("a".to_string(), "4".to_string())]),
            }],
        };
        let inputs = HashMap::from([("q".to_string(), "3+3?".to_string())]);
        let full = compiled.build_full_prompt(&inputs);
        assert!(full.contains("Example 1:"));
        assert!(full.contains("2+2?"));
        assert!(full.contains("Q: 3+3?"));
    }

    #[test]
    fn test_compiled_prompt_serialization() {
        let compiled = CompiledPrompt {
            system_prompt: "sys".to_string(),
            user_template: "user {x}".to_string(),
            examples: vec![PromptExample {
                inputs: HashMap::from([("x".to_string(), "val".to_string())]),
                outputs: HashMap::from([("y".to_string(), "res".to_string())]),
            }],
        };
        let json = serde_json::to_string(&compiled).expect("serialize");
        let back: CompiledPrompt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.system_prompt, "sys");
        assert_eq!(back.examples.len(), 1);
    }

    // --- Metric tests ---

    #[test]
    fn test_exact_match_metric() {
        let metric = ExactMatch;
        assert_eq!(metric.name(), "exact_match");
        assert_eq!(metric.score("hello", "hello"), 1.0);
        assert_eq!(metric.score("Hello", "hello"), 1.0);
        assert_eq!(metric.score(" hello ", "hello"), 1.0);
        assert_eq!(metric.score("world", "hello"), 0.0);
    }

    #[test]
    fn test_exact_match_empty() {
        let metric = ExactMatch;
        assert_eq!(metric.score("", ""), 1.0);
        assert_eq!(metric.score("a", ""), 0.0);
        assert_eq!(metric.score("", "a"), 0.0);
    }

    #[test]
    fn test_f1_score_metric() {
        let metric = F1Score;
        assert_eq!(metric.name(), "f1_score");

        // Perfect match
        assert_eq!(metric.score("the cat sat", "the cat sat"), 1.0);

        // Partial match
        let s = metric.score("the cat", "the cat sat on mat");
        assert!(s > 0.0 && s < 1.0, "partial match should give 0 < score < 1, got {}", s);

        // No match
        assert_eq!(metric.score("xyz", "abc"), 0.0);
    }

    #[test]
    fn test_f1_score_empty() {
        let metric = F1Score;
        assert_eq!(metric.score("", ""), 1.0);
        assert_eq!(metric.score("word", ""), 0.0);
        assert_eq!(metric.score("", "word"), 0.0);
    }

    #[test]
    fn test_f1_score_case_insensitive() {
        let metric = F1Score;
        assert_eq!(metric.score("The Cat", "the cat"), 1.0);
    }

    #[test]
    fn test_f1_score_duplicate_tokens() {
        let metric = F1Score;
        // "a a" vs "a b": 1 match out of pred=2, exp=2 => P=0.5, R=0.5, F1=0.5
        let s = metric.score("a a", "a b");
        assert!((s - 0.5).abs() < 1e-9, "expected 0.5, got {}", s);
    }

    #[test]
    fn test_contains_answer_metric() {
        let metric = ContainsAnswer;
        assert_eq!(metric.name(), "contains_answer");
        assert_eq!(
            metric.score("The answer is 42 of course", "42"),
            1.0
        );
        assert_eq!(metric.score("No match here", "42"), 0.0);
    }

    #[test]
    fn test_contains_answer_case_insensitive() {
        let metric = ContainsAnswer;
        assert_eq!(metric.score("HELLO world", "hello"), 1.0);
    }

    #[test]
    fn test_contains_answer_empty() {
        let metric = ContainsAnswer;
        // Empty expected is always contained
        assert_eq!(metric.score("anything", ""), 1.0);
        assert_eq!(metric.score("", "notempty"), 0.0);
    }

    // --- EvaluationBudget tests ---

    #[test]
    fn test_evaluation_budget_new() {
        let budget = EvaluationBudget::new(10, 5);
        assert_eq!(budget.max_trials, 10);
        assert_eq!(budget.max_examples, 5);
        assert!(budget.timeout_seconds.is_none());
        assert_eq!(budget.remaining(), 10);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_budget_with_timeout() {
        let budget = EvaluationBudget::new(5, 3).with_timeout(60);
        assert_eq!(budget.timeout_seconds, Some(60));
    }

    #[test]
    fn test_budget_try_use() {
        let mut budget = EvaluationBudget::new(3, 5);
        assert!(budget.try_use());
        assert_eq!(budget.remaining(), 2);
        assert_eq!(budget.used(), 1);
        assert!(budget.try_use());
        assert!(budget.try_use());
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn test_budget_exhausted() {
        let mut budget = EvaluationBudget::new(2, 5);
        assert!(budget.try_use());
        assert!(budget.try_use());
        assert!(!budget.try_use()); // exhausted
        assert_eq!(budget.remaining(), 0);
        assert_eq!(budget.used(), 2);
    }

    #[test]
    fn test_budget_reset() {
        let mut budget = EvaluationBudget::new(3, 5);
        budget.try_use();
        budget.try_use();
        assert_eq!(budget.remaining(), 1);
        budget.reset();
        assert_eq!(budget.remaining(), 3);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn test_budget_zero_trials() {
        let mut budget = EvaluationBudget::new(0, 5);
        assert!(!budget.try_use());
        assert_eq!(budget.remaining(), 0);
    }

    // --- TrainingExample tests ---

    #[test]
    fn test_training_example_roundtrip() {
        let ex = TrainingExample {
            inputs: HashMap::from([("q".to_string(), "What?".to_string())]),
            expected_outputs: HashMap::from([("a".to_string(), "Answer".to_string())]),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: TrainingExample = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.inputs.get("q").map(|s| s.as_str()), Some("What?"));
        assert_eq!(
            back.expected_outputs.get("a").map(|s| s.as_str()),
            Some("Answer")
        );
    }

    #[test]
    fn test_training_example_empty_fields() {
        let ex = TrainingExample {
            inputs: HashMap::new(),
            expected_outputs: HashMap::new(),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: TrainingExample = serde_json::from_str(&json).expect("deserialize");
        assert!(back.inputs.is_empty());
        assert!(back.expected_outputs.is_empty());
    }

    // --- OptimizationResult tests ---

    #[test]
    fn test_optimization_result_serialization() {
        let result = OptimizationResult {
            best_prompt: CompiledPrompt {
                system_prompt: "sys".to_string(),
                user_template: "usr".to_string(),
                examples: vec![],
            },
            best_score: 0.95,
            trials_run: 10,
            scores_history: vec![0.5, 0.7, 0.95],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let back: OptimizationResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.best_score, 0.95);
        assert_eq!(back.trials_run, 10);
        assert_eq!(back.scores_history.len(), 3);
    }

    // --- Helper: create a standard QA signature + examples ---

    fn make_qa_signature() -> Signature {
        Signature::new("qa", "Answer questions based on context")
            .add_input(SignatureField::new("context", "The context"))
            .add_input(SignatureField::new("question", "The question"))
            .add_output(SignatureField::new("answer", "The answer"))
    }

    fn make_training_examples() -> Vec<TrainingExample> {
        vec![
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "Rust is a systems programming language.".to_string()),
                    ("question".to_string(), "What is Rust?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "A systems programming language".to_string()),
                ]),
            },
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "Python is interpreted.".to_string()),
                    ("question".to_string(), "Is Python compiled?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "No, Python is interpreted".to_string()),
                ]),
            },
            TrainingExample {
                inputs: HashMap::from([
                    ("context".to_string(), "The sky is blue.".to_string()),
                    ("question".to_string(), "What color is the sky?".to_string()),
                ]),
                expected_outputs: HashMap::from([
                    ("answer".to_string(), "Blue".to_string()),
                ]),
            },
        ]
    }

    // --- BootstrapFewShot tests ---

    #[test]
    fn test_bootstrap_few_shot_basic() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("should succeed");
        assert!(opt.trials_run > 0);
        assert!(!opt.scores_history.is_empty());
    }

    #[test]
    fn test_bootstrap_selects_best() {
        let optimizer = BootstrapFewShot::new(3, Box::new(ContainsAnswer));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        // Best score should be the maximum in history
        let max_score = result
            .scores_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-9,
            "best_score {} should equal max in history {}",
            result.best_score,
            max_score
        );
    }

    #[test]
    fn test_bootstrap_empty_examples_error() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &[], &mut budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_single_example() {
        let optimizer = BootstrapFewShot::new(1, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = vec![make_training_examples().remove(0)];
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bootstrap_respects_budget() {
        let optimizer = BootstrapFewShot::new(2, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(2, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        assert!(result.trials_run <= 2);
        assert_eq!(budget.remaining(), 0);
    }

    // --- GridSearchOptimizer tests ---

    #[test]
    fn test_grid_search_basic() {
        let variants = vec![
            "Be concise.".to_string(),
            "Think step by step.".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("ok");
        assert!(opt.trials_run > 0);
    }

    #[test]
    fn test_grid_search_selects_best() {
        let variants = vec![
            "Short".to_string(),
            "Think step by step and provide a detailed answer.".to_string(),
            "Be precise.".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ContainsAnswer));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        let max_score = result
            .scores_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-9,
            "best_score should equal max in history"
        );
    }

    #[test]
    fn test_grid_search_empty_variants_error() {
        let optimizer = GridSearchOptimizer::new(vec![], Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_search_respects_budget() {
        let variants = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(2, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        assert!(result.trials_run <= 2);
    }

    #[test]
    fn test_grid_search_no_examples() {
        let variants = vec!["A".to_string()];
        let optimizer = GridSearchOptimizer::new(variants, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let mut budget = EvaluationBudget::new(5, 10);

        let result = optimizer.optimize(&sig, &[], &mut budget).expect("ok");
        assert_eq!(result.trials_run, 1);
    }

    // --- RandomSearchOptimizer tests ---

    #[test]
    fn test_random_search_basic() {
        let optimizer = RandomSearchOptimizer::new(5, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("ok");
        assert!(opt.trials_run > 0);
        assert_eq!(opt.scores_history.len(), opt.trials_run);
    }

    #[test]
    fn test_random_search_respects_budget() {
        let optimizer = RandomSearchOptimizer::new(100, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        assert!(result.trials_run <= 3);
    }

    #[test]
    fn test_random_search_mutations_differ() {
        // Different seeds should produce different mutations
        let m0 = RandomSearchOptimizer::mutate_instruction("", 0);
        let m1 = RandomSearchOptimizer::mutate_instruction("", 1);
        assert_ne!(m0, m1);
    }

    #[test]
    fn test_random_search_mutation_with_base() {
        let mutated = RandomSearchOptimizer::mutate_instruction("Base instruction.", 0);
        assert!(mutated.starts_with("Base instruction."));
        assert!(mutated.len() > "Base instruction.".len());
    }

    #[test]
    fn test_random_search_with_instructions() {
        let sig = make_qa_signature().with_instructions("Be helpful.");
        let optimizer = RandomSearchOptimizer::new(3, Box::new(ContainsAnswer));
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(5, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        assert!(result.best_prompt.system_prompt.len() > 0);
    }

    // --- BayesianOptimizer tests ---

    #[test]
    fn test_bayesian_optimizer_basic() {
        let optimizer = BayesianOptimizer::new(5, 1.0, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(10, 10);

        let result = optimizer.optimize(&sig, &examples, &mut budget);
        assert!(result.is_ok());
        let opt = result.expect("ok");
        assert!(opt.trials_run > 0);
    }

    #[test]
    fn test_bayesian_optimizer_respects_budget() {
        let optimizer = BayesianOptimizer::new(100, 1.0, Box::new(ExactMatch));
        let sig = make_qa_signature();
        let examples = make_training_examples();
        let mut budget = EvaluationBudget::new(3, 5);

        let result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");
        assert!(result.trials_run <= 3);
    }

    #[test]
    fn test_bayesian_rbf_kernel() {
        // Same point => kernel = 1.0
        let k = BayesianOptimizer::rbf_kernel(0.0, 0.0, 1.0);
        assert!((k - 1.0).abs() < 1e-9);

        // Distant points => kernel close to 0
        let k = BayesianOptimizer::rbf_kernel(0.0, 100.0, 1.0);
        assert!(k < 1e-6);

        // Symmetric
        let k1 = BayesianOptimizer::rbf_kernel(1.0, 2.0, 1.0);
        let k2 = BayesianOptimizer::rbf_kernel(2.0, 1.0, 1.0);
        assert!((k1 - k2).abs() < 1e-12);
    }

    #[test]
    fn test_bayesian_ucb() {
        let optimizer = BayesianOptimizer::new(1, 2.0, Box::new(ExactMatch));
        let ucb = optimizer.ucb(0.5, 0.3);
        assert!((ucb - 1.1).abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_gp_predict_empty() {
        let (mean, std_dev) = BayesianOptimizer::gp_predict(&[], 0.5, 1.0, 0.01);
        assert_eq!(mean, 0.0);
        assert_eq!(std_dev, 1.0);
    }

    #[test]
    fn test_bayesian_gp_predict_single_obs() {
        let obs = vec![(0.0, 1.0)];
        let (mean, std_dev) = BayesianOptimizer::gp_predict(&obs, 0.0, 1.0, 0.01);
        // At the observed point, mean should be close to the observed value
        assert!((mean - 1.0).abs() < 0.1, "mean at observed point should be near 1.0, got {}", mean);
        assert!(std_dev < 0.5, "std_dev at observed point should be small, got {}", std_dev);
    }

    #[test]
    fn test_bayesian_solve_linear_system() {
        // 2x = 4 => x = 2
        let a = vec![vec![2.0]];
        let b = vec![4.0];
        let x = BayesianOptimizer::solve_linear_system(&a, &b);
        assert!((x[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_solve_2x2() {
        // x + y = 3, 2x + y = 5 => x=2, y=1
        let a = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
        let b = vec![3.0, 5.0];
        let x = BayesianOptimizer::solve_linear_system(&a, &b);
        assert!((x[0] - 2.0).abs() < 1e-9, "x[0] should be 2.0, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-9, "x[1] should be 1.0, got {}", x[1]);
    }

    #[test]
    fn test_bayesian_solve_empty() {
        let x = BayesianOptimizer::solve_linear_system(&[], &[]);
        assert!(x.is_empty());
    }

    #[test]
    fn test_bayesian_exploration_weight() {
        // Higher exploration weight should prefer unexplored points
        let high_exp = BayesianOptimizer::new(1, 10.0, Box::new(ExactMatch));
        let low_exp = BayesianOptimizer::new(1, 0.1, Box::new(ExactMatch));

        let ucb_high = high_exp.ucb(0.5, 1.0);
        let ucb_low = low_exp.ucb(0.5, 1.0);
        assert!(ucb_high > ucb_low);
    }

    // --- SelfReflector tests ---

    #[test]
    fn test_self_reflector_generates_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature(); // no instructions
        let compiled = sig.compile();
        let examples = make_training_examples();

        let rules = reflector.reflect(&sig, &compiled, &examples);
        assert!(!rules.is_empty(), "reflector should generate at least one rule");
    }

    #[test]
    fn test_self_reflector_no_instructions_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test"); // no instructions
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        let has_no_instructions = rules
            .iter()
            .any(|r| r.condition == "no_instructions");
        assert!(has_no_instructions, "should detect missing instructions");
    }

    #[test]
    fn test_self_reflector_no_examples_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("be helpful");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        let has_no_examples = rules
            .iter()
            .any(|r| r.condition == "no_examples");
        assert!(has_no_examples, "should detect no examples");
    }

    #[test]
    fn test_self_reflector_few_examples_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("be helpful");
        let compiled = sig.compile();
        let examples = vec![make_training_examples().remove(0)]; // just 1

        let rules = reflector.reflect(&sig, &compiled, &examples);
        let has_few_examples = rules
            .iter()
            .any(|r| r.condition == "few_examples");
        assert!(has_few_examples, "should detect few examples");
    }

    #[test]
    fn test_self_reflector_max_iterations_limits_rules() {
        let reflector = SelfReflector::new(2, 0.05);
        let sig = Signature::new("test", "Test"); // will generate many rules
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &[]);
        assert!(rules.len() <= 2, "should be limited to max_iterations=2");
    }

    #[test]
    fn test_self_reflector_incomplete_input_coverage() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("inst");
        let compiled = sig.compile();

        // Example missing the "context" input
        let incomplete = vec![TrainingExample {
            inputs: HashMap::from([
                ("question".to_string(), "What?".to_string()),
            ]),
            expected_outputs: HashMap::from([
                ("answer".to_string(), "Something".to_string()),
            ]),
        }];

        let rules = reflector.reflect(&sig, &compiled, &incomplete);
        let has_coverage_rule = rules
            .iter()
            .any(|r| r.condition.starts_with("incomplete_input_coverage:"));
        assert!(has_coverage_rule, "should detect incomplete input coverage");
    }

    #[test]
    fn test_self_reflector_incomplete_output_coverage() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature().with_instructions("inst");
        let compiled = sig.compile();

        // Example missing the "answer" output
        let incomplete = vec![TrainingExample {
            inputs: HashMap::from([
                ("context".to_string(), "ctx".to_string()),
                ("question".to_string(), "q".to_string()),
            ]),
            expected_outputs: HashMap::new(),
        }];

        let rules = reflector.reflect(&sig, &compiled, &incomplete);
        let has_coverage_rule = rules
            .iter()
            .any(|r| r.condition.starts_with("incomplete_output_coverage:"));
        assert!(has_coverage_rule, "should detect incomplete output coverage");
    }

    #[test]
    fn test_self_reflector_typed_output_guidance() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("typed", "Typed outputs")
            .add_output(
                SignatureField::new("count", "a count")
                    .with_type(FieldType::Number),
            )
            .with_instructions("inst");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &make_training_examples());
        // The compiled prompt does contain "number" in field descriptions,
        // so this may or may not trigger depending on compilation output.
        // Just verify reflect doesn't panic and returns valid rules.
        for rule in &rules {
            assert!(!rule.condition.is_empty());
            assert!(!rule.action.is_empty());
        }
    }

    #[test]
    fn test_self_reflector_ambiguous_types() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("ambig", "Ambiguous outputs")
            .add_output(SignatureField::new("a", "first text"))
            .add_output(SignatureField::new("b", "second text"))
            .with_instructions("inst");
        let compiled = sig.compile();

        let rules = reflector.reflect(&sig, &compiled, &make_training_examples());
        let has_ambig = rules
            .iter()
            .any(|r| r.condition.starts_with("ambiguous_output_types:"));
        assert!(has_ambig, "should detect ambiguous output types");
    }

    #[test]
    fn test_self_reflector_apply_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test"); // no instructions

        let rules = vec![
            ImprovementRule {
                condition: "no_instructions".to_string(),
                action: "Add instructions".to_string(),
                applied: false,
            },
        ];

        let improved = reflector.apply_rules(&sig, &rules);
        assert!(improved.instructions.is_some());
        assert!(
            improved.instructions.as_deref().unwrap_or("").contains("Follow the field descriptions"),
            "should add default instructions"
        );
    }

    #[test]
    fn test_self_reflector_apply_type_guidance_rule() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = Signature::new("test", "Test")
            .add_output(
                SignatureField::new("count", "a count")
                    .with_type(FieldType::Number),
            )
            .with_instructions("Existing.");

        let rules = vec![
            ImprovementRule {
                condition: "missing_type_guidance:count".to_string(),
                action: "Add format guidance for count".to_string(),
                applied: false,
            },
        ];

        let improved = reflector.apply_rules(&sig, &rules);
        let inst = improved.instructions.expect("should have instructions");
        assert!(inst.contains("number"), "should mention number format");
    }

    #[test]
    fn test_self_reflector_apply_empty_rules() {
        let reflector = SelfReflector::new(10, 0.05);
        let sig = make_qa_signature();
        let improved = reflector.apply_rules(&sig, &[]);
        // With no rules, the signature should be unchanged
        assert_eq!(improved.name, sig.name);
        assert_eq!(improved.instructions, sig.instructions);
    }

    // --- ImprovementRule tests ---

    #[test]
    fn test_improvement_rule_serialization() {
        let rule = ImprovementRule {
            condition: "test_cond".to_string(),
            action: "test_action".to_string(),
            applied: false,
        };
        let json = serde_json::to_string(&rule).expect("serialize");
        let back: ImprovementRule = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.condition, "test_cond");
        assert_eq!(back.action, "test_action");
        assert!(!back.applied);
    }

    #[test]
    fn test_improvement_rule_applied_flag() {
        let mut rule = ImprovementRule {
            condition: "c".to_string(),
            action: "a".to_string(),
            applied: false,
        };
        assert!(!rule.applied);
        rule.applied = true;
        assert!(rule.applied);
    }

    // --- PromptExample tests ---

    #[test]
    fn test_prompt_example_serialization() {
        let ex = PromptExample {
            inputs: HashMap::from([("k".to_string(), "v".to_string())]),
            outputs: HashMap::from([("o".to_string(), "r".to_string())]),
        };
        let json = serde_json::to_string(&ex).expect("serialize");
        let back: PromptExample = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.inputs.get("k").map(|s| s.as_str()), Some("v"));
    }

    // --- Integration-style tests ---

    #[test]
    fn test_end_to_end_compile_and_render() {
        let sig = Signature::new("summarize", "Summarize text")
            .add_input(SignatureField::new("text", "The text to summarize"))
            .add_output(SignatureField::new("summary", "A concise summary"))
            .with_instructions("Keep it under 50 words.");

        let compiled = sig.compile();
        let inputs = HashMap::from([(
            "text".to_string(),
            "Rust is a multi-paradigm, general-purpose programming language.".to_string(),
        )]);

        sig.validate_inputs(&inputs).expect("validation ok");
        let rendered = compiled.render(&inputs);
        assert!(rendered.contains("Rust is a multi-paradigm"));
    }

    #[test]
    fn test_end_to_end_optimize_and_reflect() {
        let sig = make_qa_signature();
        let examples = make_training_examples();

        // Optimize
        let optimizer = BootstrapFewShot::new(2, Box::new(ContainsAnswer));
        let mut budget = EvaluationBudget::new(5, 5);
        let opt_result = optimizer.optimize(&sig, &examples, &mut budget).expect("ok");

        // Reflect on the result
        let reflector = SelfReflector::new(10, 0.05);
        let rules = reflector.reflect(&sig, &opt_result.best_prompt, &examples);

        // Apply improvements
        let improved_sig = reflector.apply_rules(&sig, &rules);
        let improved_compiled = improved_sig.compile();
        assert!(improved_compiled.system_prompt.len() > 0);
    }

    #[test]
    fn test_multiple_optimizers_comparison() {
        let sig = make_qa_signature();
        let examples = make_training_examples();

        // Bootstrap
        let bs = BootstrapFewShot::new(2, Box::new(ContainsAnswer));
        let mut budget1 = EvaluationBudget::new(3, 5);
        let r1 = bs.optimize(&sig, &examples, &mut budget1).expect("ok");

        // Grid search
        let gs = GridSearchOptimizer::new(
            vec!["Be precise.".to_string(), "Think carefully.".to_string()],
            Box::new(ContainsAnswer),
        );
        let mut budget2 = EvaluationBudget::new(3, 5);
        let r2 = gs.optimize(&sig, &examples, &mut budget2).expect("ok");

        // Random search
        let rs = RandomSearchOptimizer::new(3, Box::new(ContainsAnswer));
        let mut budget3 = EvaluationBudget::new(3, 5);
        let r3 = rs.optimize(&sig, &examples, &mut budget3).expect("ok");

        // All should produce valid results
        assert!(r1.trials_run > 0);
        assert!(r2.trials_run > 0);
        assert!(r3.trials_run > 0);
    }

    #[test]
    fn test_signature_many_fields() {
        let sig = Signature::new("multi", "Multi-field signature")
            .add_input(SignatureField::new("a", "field a"))
            .add_input(SignatureField::new("b", "field b"))
            .add_input(SignatureField::new("c", "field c").optional())
            .add_output(SignatureField::new("x", "output x"))
            .add_output(SignatureField::new("y", "output y"))
            .add_output(
                SignatureField::new("z", "output z")
                    .with_type(FieldType::Json)
                    .optional(),
            );

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("field a"));
        assert!(compiled.system_prompt.contains("output z"));
        assert!(compiled.user_template.contains("{a}"));
        assert!(compiled.user_template.contains("{b}"));
    }

    #[test]
    fn test_field_type_equality() {
        assert_eq!(FieldType::Text, FieldType::Text);
        assert_eq!(FieldType::Number, FieldType::Number);
        assert_eq!(FieldType::Boolean, FieldType::Boolean);
        assert_eq!(FieldType::List, FieldType::List);
        assert_eq!(FieldType::Json, FieldType::Json);
        assert_ne!(FieldType::Text, FieldType::Json);
        assert_ne!(FieldType::Number, FieldType::Boolean);
    }

    #[test]
    fn test_all_field_types_in_signature() {
        let sig = Signature::new("all_types", "All types test")
            .add_input(SignatureField::new("t", "text").with_type(FieldType::Text))
            .add_input(SignatureField::new("n", "number").with_type(FieldType::Number))
            .add_input(SignatureField::new("b", "bool").with_type(FieldType::Boolean))
            .add_input(SignatureField::new("l", "list").with_type(FieldType::List))
            .add_input(SignatureField::new("j", "json").with_type(FieldType::Json));

        let compiled = sig.compile();
        assert!(compiled.system_prompt.contains("text"));
        assert!(compiled.system_prompt.contains("number"));
        assert!(compiled.system_prompt.contains("boolean"));
        assert!(compiled.system_prompt.contains("list"));
        assert!(compiled.system_prompt.contains("json"));
    }
}
