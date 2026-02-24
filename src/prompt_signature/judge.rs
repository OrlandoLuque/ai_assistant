//! LLM-as-Judge automated grading: rubrics, metrics, and calibration.

use serde::{Deserialize, Serialize};

use super::types::EvalMetric;

// ============================================================================
// 4.3 LLM-as-Judge Automated Grading
// ============================================================================

/// A single criterion in a judge rubric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCriterion {
    /// Name of the criterion (e.g., "relevance", "fluency").
    pub name: String,
    /// Description of what this criterion measures.
    pub description: String,
    /// Weight of this criterion in the overall score.
    pub weight: f64,
    /// Score scale as (min, max).
    pub scale: (f64, f64),
}

/// A rubric defining multiple criteria for judging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeRubric {
    /// The criteria that make up this rubric.
    pub criteria: Vec<JudgeCriterion>,
    /// Overall instruction for the judge.
    pub overall_instruction: String,
}

impl JudgeRubric {
    /// Create a new rubric with the given overall instruction.
    pub fn new(overall_instruction: String) -> Self {
        Self {
            criteria: Vec::new(),
            overall_instruction,
        }
    }

    /// Add a criterion to this rubric.
    pub fn add_criterion(&mut self, criterion: JudgeCriterion) {
        self.criteria.push(criterion);
    }

    /// Return the number of criteria.
    pub fn criterion_count(&self) -> usize {
        self.criteria.len()
    }

    /// Return the sum of all criterion weights.
    pub fn total_weight(&self) -> f64 {
        self.criteria.iter().map(|c| c.weight).sum()
    }

    /// Validate the rubric: checks that weights are positive and criteria are non-empty.
    pub fn validate(&self) -> Result<(), String> {
        if self.criteria.is_empty() {
            return Err("Rubric must have at least one criterion".to_string());
        }
        for criterion in &self.criteria {
            if criterion.weight <= 0.0 {
                return Err(format!(
                    "Criterion '{}' has non-positive weight: {}",
                    criterion.name, criterion.weight
                ));
            }
        }
        Ok(())
    }
}

/// The result of an LLM-as-judge evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptJudgeResult {
    /// The weighted overall score.
    pub overall_score: f64,
    /// Individual scores per criterion.
    pub per_criterion: Vec<CriterionScore>,
    /// Reasoning for the overall judgment.
    pub reasoning: String,
    /// Confidence in the judgment (0.0..1.0).
    pub confidence: f64,
}

/// Score for a single criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    /// Name of the criterion that was scored.
    pub criterion_name: String,
    /// The score assigned.
    pub score: f64,
    /// Reasoning for this particular score.
    pub reasoning: String,
}

/// Configuration for an LLM-as-judge evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeConfig {
    /// The rubric to use for evaluation.
    pub rubric: JudgeRubric,
    /// Few-shot examples of judge evaluations.
    pub few_shot_examples: Vec<JudgeExample>,
    /// Temperature for judge inference.
    pub temperature: f64,
}

/// A few-shot example demonstrating judge behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeExample {
    /// The input that was evaluated.
    pub input: String,
    /// The output that was evaluated.
    pub output: String,
    /// The expected score.
    pub expected_score: f64,
    /// The expected reasoning.
    pub reasoning: String,
}

/// An evaluation metric that uses an LLM-as-judge scoring function.
pub struct JudgeMetric {
    pub(crate) config: JudgeConfig,
    pub(crate) scorer: Box<dyn Fn(&str, &str, &JudgeRubric) -> f64 + Send + Sync>,
}

impl JudgeMetric {
    /// Create a new judge metric with the given configuration and scorer function.
    pub fn new(
        config: JudgeConfig,
        scorer: Box<dyn Fn(&str, &str, &JudgeRubric) -> f64 + Send + Sync>,
    ) -> Self {
        Self { config, scorer }
    }

    /// Evaluate an input/output pair using the judge.
    pub fn evaluate(&self, input: &str, output: &str) -> PromptJudgeResult {
        let overall_score = (self.scorer)(input, output, &self.config.rubric);

        // Generate per-criterion scores proportionally
        let mut per_criterion = Vec::new();
        let total_weight = self.config.rubric.total_weight();

        for criterion in &self.config.rubric.criteria {
            let criterion_score = if total_weight > 0.0 {
                // Scale score within the criterion's scale range
                let (min_s, max_s) = criterion.scale;
                min_s + (max_s - min_s) * overall_score.clamp(0.0, 1.0)
            } else {
                overall_score
            };

            per_criterion.push(CriterionScore {
                criterion_name: criterion.name.clone(),
                score: criterion_score,
                reasoning: format!(
                    "Score for '{}' based on overall evaluation",
                    criterion.name
                ),
            });
        }

        PromptJudgeResult {
            overall_score,
            per_criterion,
            reasoning: format!(
                "Evaluated output against rubric: {}",
                self.config.rubric.overall_instruction
            ),
            confidence: 0.8,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &JudgeConfig {
        &self.config
    }
}

impl EvalMetric for JudgeMetric {
    fn name(&self) -> &str {
        "judge_metric"
    }

    fn score(&self, predicted: &str, expected: &str) -> f64 {
        (self.scorer)(predicted, expected, &self.config.rubric)
    }
}

/// A set of calibration examples for measuring judge accuracy.
pub struct CalibrationSet {
    examples: Vec<CalibrationExample>,
}

/// A single calibration example with a known human score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExample {
    /// The input text.
    pub input: String,
    /// The output text to judge.
    pub output: String,
    /// The human-assigned ground truth score.
    pub human_score: f64,
}

impl CalibrationSet {
    /// Create a new empty calibration set.
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Add a calibration example.
    pub fn add(&mut self, example: CalibrationExample) {
        self.examples.push(example);
    }

    /// Compute the mean absolute calibration error between judge scores and human scores.
    pub fn calibration_error(&self, judge: &JudgeMetric) -> f64 {
        if self.examples.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        for ex in &self.examples {
            let judge_score = (judge.scorer)(&ex.input, &ex.output, &judge.config.rubric);
            total_error += (judge_score - ex.human_score).abs();
        }

        total_error / self.examples.len() as f64
    }

    /// Return a slice of all calibration examples.
    pub fn examples(&self) -> &[CalibrationExample] {
        &self.examples
    }

    /// Return the number of calibration examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return whether the calibration set is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}
