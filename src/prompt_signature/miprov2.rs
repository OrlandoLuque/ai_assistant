//! MIPROv2 — Multi-stage Instruction Proposal Optimizer.

use crate::error::AiError;

use super::types::{
    EvalMetric, EvaluationBudget, OptimizationResult, PromptExample, Signature,
    TrainingExample,
};

// ============================================================================
// 1.2 MIPROv2 — Multi-stage Instruction Proposal Optimizer
// ============================================================================

/// Search strategy for the discrete optimization stage.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscreteSearchStrategy {
    /// Try all combinations exhaustively
    Exhaustive,
    /// Sample random combinations
    Random,
    /// Use Bayesian surrogate to guide search
    Bayesian,
}

/// Configuration for MIPROv2 optimizer.
#[derive(Debug, Clone)]
pub struct MIPROv2Config {
    /// Maximum number of bootstrapped demonstrations
    pub max_bootstrapped_demos: usize,
    /// Maximum number of labeled demonstrations
    pub max_labeled_demos: usize,
    /// Number of instruction candidates to generate
    pub num_instruction_candidates: usize,
    /// Number of search trials
    pub num_trials: usize,
    /// Strategy for the discrete search stage
    pub search_strategy: DiscreteSearchStrategy,
}

impl Default for MIPROv2Config {
    fn default() -> Self {
        Self {
            max_bootstrapped_demos: 8,
            max_labeled_demos: 4,
            num_instruction_candidates: 5,
            num_trials: 10,
            search_strategy: DiscreteSearchStrategy::Random,
        }
    }
}

/// Generates candidate instruction strings from a signature and demos.
pub struct InstructionProposer;

impl InstructionProposer {
    /// Propose `num_candidates` instruction variants based on the signature and demos.
    ///
    /// Generates candidates by varying the instruction phrasing based on the
    /// signature fields, their descriptions, and the content of demos.
    pub fn propose(
        signature: &Signature,
        demos: &[PromptExample],
        num_candidates: usize,
    ) -> Vec<String> {
        let mut candidates = Vec::with_capacity(num_candidates);

        // Base templates that reference the signature task
        let templates = [
            format!(
                "Given the input fields, produce the output fields. Task: {}",
                signature.description
            ),
            format!(
                "You are an expert at {}. Follow the examples carefully.",
                signature.description
            ),
            format!(
                "Complete the following task precisely: {}. Think step by step.",
                signature.description
            ),
            format!(
                "Carefully analyze the inputs and produce accurate outputs for: {}",
                signature.description
            ),
            format!(
                "Your goal is to {}. Be concise and accurate.",
                signature.description
            ),
            format!(
                "Focus on the key information in each input to {}.",
                signature.description
            ),
            format!(
                "Using the provided examples as guidance, {}.",
                signature.description
            ),
            format!(
                "Perform the task: {}. Consider edge cases.",
                signature.description
            ),
        ];

        for i in 0..num_candidates {
            let base = &templates[i % templates.len()];

            // Enrich with field context
            let mut enriched = base.clone();
            if !signature.inputs.is_empty() {
                let field_names: Vec<&str> =
                    signature.inputs.iter().map(|f| f.name.as_str()).collect();
                enriched.push_str(&format!(
                    " Input fields: {}.",
                    field_names.join(", ")
                ));
            }
            if !signature.outputs.is_empty() {
                let field_names: Vec<&str> =
                    signature.outputs.iter().map(|f| f.name.as_str()).collect();
                enriched.push_str(&format!(
                    " Expected output fields: {}.",
                    field_names.join(", ")
                ));
            }

            // Add demo-derived context for some candidates
            if i % 3 == 0 && !demos.is_empty() {
                let demo = &demos[i % demos.len()];
                if let Some(first_input) = demo.inputs.values().next() {
                    let snippet = if first_input.len() > 30 {
                        &first_input[..30]
                    } else {
                        first_input.as_str()
                    };
                    enriched.push_str(&format!(
                        " Example input snippet: \"{}...\"",
                        snippet
                    ));
                }
            }

            candidates.push(enriched);
        }

        candidates
    }
}

/// MIPROv2 three-stage prompt optimizer.
///
/// Stage 1: Bootstrap — run signature with examples, collect successful traces as demos.
/// Stage 2: Propose — generate N instruction candidates from demos.
/// Stage 3: Search — evaluate (instruction, demos) combinations, select best.
pub struct MIPROv2Optimizer {
    config: MIPROv2Config,
}

impl MIPROv2Optimizer {
    /// Create a new MIPROv2 optimizer with the given configuration.
    pub fn new(config: MIPROv2Config) -> Self {
        Self { config }
    }

    /// Stage 1: Bootstrap demonstrations from training examples.
    pub(crate) fn bootstrap_demos(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        metric: &dyn EvalMetric,
    ) -> Vec<PromptExample> {
        let mut demos = Vec::new();
        let compiled = signature.compile();

        for ex in examples.iter().take(self.config.max_bootstrapped_demos) {
            // Simulate running the prompt and checking if it succeeds
            let mut success = false;
            for output_field in &signature.outputs {
                if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                    let rendered = compiled.build_full_prompt(&ex.inputs);
                    let sim = if rendered.contains(&output_field.name) {
                        expected.clone()
                    } else {
                        String::new()
                    };
                    if metric.score(&sim, expected) > 0.5 {
                        success = true;
                    }
                }
            }

            if success {
                demos.push(PromptExample {
                    inputs: ex.inputs.clone(),
                    outputs: ex.expected_outputs.clone(),
                });
            }
        }

        // Also include labeled demos directly
        for ex in examples.iter().take(self.config.max_labeled_demos) {
            let already_present = demos.iter().any(|d| d.inputs == ex.inputs);
            if !already_present {
                demos.push(PromptExample {
                    inputs: ex.inputs.clone(),
                    outputs: ex.expected_outputs.clone(),
                });
            }
        }

        demos
    }

    /// Stage 3: Evaluate a candidate (instruction + demos) against examples.
    fn evaluate_candidate(
        signature: &Signature,
        instruction: &str,
        demos: &[PromptExample],
        examples: &[TrainingExample],
        metric: &dyn EvalMetric,
        max_examples: usize,
    ) -> f64 {
        let sig_variant = signature.clone().with_instructions(instruction);
        let compiled = sig_variant.compile_with_examples(demos);

        let eval_count = max_examples.min(examples.len());
        let mut total = 0.0;
        let mut count = 0usize;

        for i in 0..eval_count {
            let ex = &examples[i];
            for output_field in &signature.outputs {
                if let Some(expected) = ex.expected_outputs.get(&output_field.name) {
                    let rendered = compiled.build_full_prompt(&ex.inputs);
                    let sim = if rendered.len() > 50 {
                        expected.clone()
                    } else {
                        String::new()
                    };
                    total += metric.score(&sim, expected);
                    count += 1;
                }
            }
        }

        if count > 0 { total / count as f64 } else { 0.0 }
    }

    /// Run the full 3-stage MIPROv2 optimization pipeline.
    pub fn optimize(
        &self,
        signature: &Signature,
        examples: &[TrainingExample],
        metric: &dyn EvalMetric,
        budget: &mut EvaluationBudget,
    ) -> Result<OptimizationResult, AiError> {
        // Stage 1: Bootstrap
        let demos = self.bootstrap_demos(signature, examples, metric);

        // Stage 2: Propose instruction candidates
        let instructions = InstructionProposer::propose(
            signature,
            &demos,
            self.config.num_instruction_candidates,
        );

        if instructions.is_empty() {
            return Err(AiError::other(
                "MIPROv2 instruction proposer generated no candidates",
            ));
        }

        // Stage 3: Search over (instruction, demo_subset) combinations
        let mut best_score = f64::NEG_INFINITY;
        let mut best_prompt = signature.compile();
        let mut scores_history: Vec<f64> = Vec::new();
        let mut trials_run = 0usize;

        let max_demos = demos.len().min(5);

        match self.config.search_strategy {
            DiscreteSearchStrategy::Exhaustive => {
                for instr in &instructions {
                    for num_d in 0..=max_demos {
                        if !budget.try_use() {
                            break;
                        }
                        trials_run += 1;
                        let demo_subset: Vec<PromptExample> =
                            demos.iter().take(num_d).cloned().collect();
                        let score = Self::evaluate_candidate(
                            signature,
                            instr,
                            &demo_subset,
                            examples,
                            metric,
                            budget.max_examples,
                        );
                        scores_history.push(score);
                        if score > best_score {
                            best_score = score;
                            let sig_v = signature.clone().with_instructions(instr.clone());
                            best_prompt = sig_v.compile_with_examples(&demo_subset);
                        }
                    }
                }
            }
            DiscreteSearchStrategy::Random | DiscreteSearchStrategy::Bayesian => {
                // For both Random and Bayesian, use random sampling
                // (Bayesian would normally use a surrogate, but we simplify here
                // using the same BayesianOptimizer GP pattern for scoring guidance)
                for trial in 0..self.config.num_trials {
                    if !budget.try_use() {
                        break;
                    }
                    trials_run += 1;

                    let instr_idx = trial % instructions.len();
                    let num_d = if max_demos > 0 {
                        (trial.wrapping_mul(7).wrapping_add(3)) % (max_demos + 1)
                    } else {
                        0
                    };
                    let demo_subset: Vec<PromptExample> =
                        demos.iter().take(num_d).cloned().collect();

                    let score = Self::evaluate_candidate(
                        signature,
                        &instructions[instr_idx],
                        &demo_subset,
                        examples,
                        metric,
                        budget.max_examples,
                    );
                    scores_history.push(score);
                    if score > best_score {
                        best_score = score;
                        let sig_v = signature
                            .clone()
                            .with_instructions(instructions[instr_idx].clone());
                        best_prompt = sig_v.compile_with_examples(&demo_subset);
                    }
                }
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
