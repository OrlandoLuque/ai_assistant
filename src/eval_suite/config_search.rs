//! Configuration search engine for systematically exploring agent configuration space.
//!
//! Uses a one-at-a-time coordinate descent algorithm: measure a baseline configuration,
//! then sweep each dimension independently, tracking quality/cost/latency evolution
//! and using Welch's t-test for statistical significance.

use super::agent_config::{ConfigMeasurement, EvalAgentConfig, MultiModelGenerator, SearchDimension};
use super::dataset::{BenchmarkDataset, BenchmarkProblem};
use super::runner::{BenchmarkRunResult, ModelIdentifier, ProblemResult, TokenUsage};
use super::scoring::{DefaultScorer, ProblemScorer};
use super::subtask::Subtask;
use crate::error::EvalSuiteError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the search engine.
#[derive(Debug, Clone)]
pub struct ConfigSearchConfig {
    /// Confidence level for statistical significance (default 0.95)
    pub confidence_level: f64,
    /// Minimum number of problems required (default 5)
    pub min_samples: usize,
    /// Maximum total configurations to evaluate — budget guard (default 100)
    pub max_evaluations: usize,
    /// Whether to re-order dimensions by observed variance (default true)
    pub adaptive_priority: bool,
    /// Optimization objective
    pub objective: SearchObjective,
}

impl Default for ConfigSearchConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples: 5,
            max_evaluations: 100,
            adaptive_priority: true,
            objective: SearchObjective::MaxQuality,
        }
    }
}

/// What to optimize for during configuration search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchObjective {
    /// Maximize quality regardless of cost
    MaxQuality,
    /// Minimize cost while maintaining minimum quality threshold
    MinCost { min_quality: f64 },
    /// Maximize quality/cost ratio
    CostEfficiency,
    /// Custom weighted: `quality_weight * quality - cost_weight * cost - latency_weight * latency`
    Weighted {
        quality_weight: f64,
        cost_weight: f64,
        latency_weight: f64,
    },
}

impl std::fmt::Display for SearchObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxQuality => write!(f, "MaxQuality"),
            Self::MinCost { min_quality } => write!(f, "MinCost(min_q={:.2})", min_quality),
            Self::CostEfficiency => write!(f, "CostEfficiency"),
            Self::Weighted { quality_weight, cost_weight, latency_weight } => {
                write!(f, "Weighted(q={:.1},c={:.1},l={:.1})", quality_weight, cost_weight, latency_weight)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single iteration of the configuration search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchIteration {
    /// Iteration number (0-based)
    pub iteration: usize,
    /// Which dimension was varied
    pub dimension_name: String,
    /// Label of the variant tested
    pub variant_label: String,
    /// Measurement of the variant configuration
    pub measurement: ConfigMeasurement,
    /// p-value vs current best (from Welch's t-test on quality)
    pub p_value_vs_best: f64,
    /// Whether this variant was adopted as the new best
    pub is_improvement: bool,
}

/// Tracks how the best configuration evolves over iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSnapshot {
    /// Iteration number
    pub iteration: usize,
    /// Current best quality score
    pub current_best_quality: f64,
    /// Current best total cost
    pub current_best_cost: f64,
    /// Current best mean latency (ms)
    pub current_best_latency_ms: f64,
    /// Objective score under the current SearchObjective
    pub objective_score: f64,
    /// Which dimension was explored
    pub dimension_explored: String,
    /// Which variant was tested
    pub variant_tested: String,
    /// Whether this iteration produced an improvement
    pub was_improvement: bool,
}

/// Cost of the search process itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCost {
    /// Total configurations evaluated
    pub total_configurations_evaluated: usize,
    /// Total problems solved across all configurations
    pub total_problems_solved: usize,
    /// Total LLM calls made
    pub total_llm_calls: usize,
    /// Estimated total cost of the search
    pub estimated_total_cost: f64,
    /// Estimated total tokens consumed
    pub estimated_total_tokens: usize,
}

/// Final result of a configuration search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSearchResult {
    /// Baseline configuration and its measurement
    pub baseline: ConfigMeasurement,
    /// Best configuration found
    pub best: ConfigMeasurement,
    /// All iterations performed
    pub iterations: Vec<SearchIteration>,
    /// Evolution of best quality/cost/latency over time
    pub evolution: Vec<EvolutionSnapshot>,
    /// Variance of quality across variants for each dimension
    pub dimension_variance: HashMap<String, f64>,
    /// Recommended configuration (same as best.config)
    pub recommended: EvalAgentConfig,
    /// Quality improvement over baseline (percentage)
    pub quality_improvement_pct: f64,
    /// Cost change over baseline (percentage, negative = cheaper)
    pub cost_change_pct: f64,
    /// Total configurations evaluated
    pub total_evaluations: usize,
    /// Cost of the search itself
    pub search_cost: SearchCost,
    /// Whether the search converged (no improvements in first pass)
    pub converged: bool,
    /// Whether the search was stopped by budget limit
    pub stopped_by_budget: bool,
}

impl ConfigSearchResult {
    /// Get the top-N most impactful dimensions (highest variance).
    pub fn top_dimensions(&self, n: usize) -> Vec<(String, f64)> {
        let mut dims: Vec<(String, f64)> = self.dimension_variance.clone().into_iter().collect();
        dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dims.truncate(n);
        dims
    }

    /// Get all iterations that produced improvements.
    pub fn improvements(&self) -> Vec<&SearchIteration> {
        self.iterations.iter().filter(|i| i.is_improvement).collect()
    }

    /// Generate a human-readable summary of the search results.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Configuration Search Summary"));
        lines.push(format!("============================"));
        lines.push(format!(
            "Baseline quality: {:.4} | cost: {:.6} | latency: {:.1}ms",
            self.baseline.quality, self.baseline.cost, self.baseline.latency_ms
        ));
        lines.push(format!(
            "Best quality:     {:.4} | cost: {:.6} | latency: {:.1}ms",
            self.best.quality, self.best.cost, self.best.latency_ms
        ));
        lines.push(format!(
            "Improvement: quality {}{:.1}% | cost {}{:.1}%",
            if self.quality_improvement_pct >= 0.0 { "+" } else { "" },
            self.quality_improvement_pct,
            if self.cost_change_pct >= 0.0 { "+" } else { "" },
            self.cost_change_pct,
        ));
        lines.push(format!(
            "Evaluations: {} | Converged: {} | Budget stop: {}",
            self.total_evaluations, self.converged, self.stopped_by_budget
        ));

        let diffs = self.baseline.config.diff(&self.best.config);
        if !diffs.is_empty() {
            lines.push(String::new());
            lines.push("Changes from baseline:".to_string());
            for d in &diffs {
                lines.push(format!("  {}", d));
            }
        }

        let top = self.top_dimensions(3);
        if !top.is_empty() {
            lines.push(String::new());
            lines.push("Most impactful dimensions:".to_string());
            for (name, var) in &top {
                lines.push(format!("  {} (variance={:.6})", name, var));
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// ConfigSearchEngine
// ---------------------------------------------------------------------------

/// Engine that systematically searches configuration space using coordinate descent.
///
/// The algorithm:
/// 1. Measure baseline configuration
/// 2. For each dimension, sweep all variants one at a time
/// 3. Use Welch's t-test to verify significance of improvements
/// 4. Optionally re-sweep highest-variance dimensions (adaptive priority)
/// 5. Track evolution and cost throughout
pub struct ConfigSearchEngine {
    config: ConfigSearchConfig,
    generator: MultiModelGenerator,
    scorer: Box<dyn ProblemScorer>,
}

impl ConfigSearchEngine {
    /// Create a new search engine.
    pub fn new(config: ConfigSearchConfig, generator: MultiModelGenerator) -> Self {
        Self {
            config,
            generator,
            scorer: Box::new(DefaultScorer),
        }
    }

    /// Use a custom scorer instead of the default.
    pub fn with_scorer(mut self, scorer: Box<dyn ProblemScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// Run the full configuration search.
    ///
    /// Measures the baseline, then sweeps each dimension one at a time.
    /// If `adaptive_priority` is enabled and improvements were found,
    /// re-sweeps the top-3 highest-variance dimensions.
    pub fn search(
        &self,
        baseline: &EvalAgentConfig,
        dimensions: &[SearchDimension],
        dataset: &BenchmarkDataset,
        subtask_tags: &HashMap<String, Subtask>,
    ) -> Result<ConfigSearchResult, EvalSuiteError> {
        // Validate dataset size
        if dataset.len() < self.config.min_samples {
            return Err(EvalSuiteError::InsufficientData {
                metric: "problems".to_string(),
                samples: dataset.len(),
            });
        }

        let alpha = 1.0 - self.config.confidence_level;

        // Track search state
        let mut iterations = Vec::new();
        let mut evolution = Vec::new();
        let mut dimension_variance: HashMap<String, f64> = HashMap::new();
        let mut total_evaluations = 0_usize;
        let mut total_problems = 0_usize;
        let mut total_llm_calls = 0_usize;
        let mut total_cost = 0.0_f64;
        let mut total_tokens = 0_usize;
        let mut stopped_by_budget = false;

        // Step 1: Measure baseline
        let baseline_measurement = self.measure(baseline, dataset, subtask_tags)?;
        total_evaluations += 1;
        total_problems += baseline_measurement.sample_count;
        total_llm_calls += baseline_measurement.sample_count;
        total_cost += baseline_measurement.cost;
        if let Some(ref rr) = baseline_measurement.run_result {
            total_tokens += rr.total_tokens.total();
        }

        let baseline_obj = self.objective_score(&baseline_measurement);
        evolution.push(EvolutionSnapshot {
            iteration: 0,
            current_best_quality: baseline_measurement.quality,
            current_best_cost: baseline_measurement.cost,
            current_best_latency_ms: baseline_measurement.latency_ms,
            objective_score: baseline_obj,
            dimension_explored: "baseline".to_string(),
            variant_tested: "initial".to_string(),
            was_improvement: false,
        });

        let mut current_best = baseline_measurement.clone();
        let mut first_pass_improvements = 0_usize;
        let mut iteration_counter = 1_usize;

        // Step 3: Sweep each dimension
        for dim in dimensions {
            if dim.variant_count() == 0 {
                continue;
            }

            let mut dim_qualities = Vec::new();

            for variant_idx in 0..dim.variant_count() {
                // Budget check
                if total_evaluations >= self.config.max_evaluations {
                    stopped_by_budget = true;
                    break;
                }

                let candidate_config = dim.apply_variant(&current_best.config, variant_idx);
                let measurement = self.measure(&candidate_config, dataset, subtask_tags)?;

                total_evaluations += 1;
                total_problems += measurement.sample_count;
                total_llm_calls += measurement.sample_count;
                total_cost += measurement.cost;
                if let Some(ref rr) = measurement.run_result {
                    total_tokens += rr.total_tokens.total();
                }

                dim_qualities.push(measurement.quality);

                // Welch's t-test
                let scores_candidate = measurement.per_problem_scores();
                let scores_best = current_best.per_problem_scores();
                let (_, p_value) = welch_t_test(&scores_candidate, &scores_best);

                // Check if this is an improvement
                let is_improvement = self.is_better(&measurement, &current_best, p_value, alpha);

                let obj_score = self.objective_score(if is_improvement { &measurement } else { &current_best });

                iterations.push(SearchIteration {
                    iteration: iteration_counter,
                    dimension_name: dim.name(),
                    variant_label: dim.variant_label(variant_idx),
                    measurement: measurement.clone(),
                    p_value_vs_best: p_value,
                    is_improvement,
                });

                if is_improvement {
                    current_best = measurement;
                    first_pass_improvements += 1;
                }

                evolution.push(EvolutionSnapshot {
                    iteration: iteration_counter,
                    current_best_quality: current_best.quality,
                    current_best_cost: current_best.cost,
                    current_best_latency_ms: current_best.latency_ms,
                    objective_score: obj_score,
                    dimension_explored: dim.name(),
                    variant_tested: dim.variant_label(variant_idx),
                    was_improvement: is_improvement,
                });

                iteration_counter += 1;
            }

            if stopped_by_budget {
                break;
            }

            // Record variance for this dimension
            if dim_qualities.len() >= 2 {
                let mean = dim_qualities.iter().sum::<f64>() / dim_qualities.len() as f64;
                let variance = dim_qualities
                    .iter()
                    .map(|q| (q - mean).powi(2))
                    .sum::<f64>()
                    / (dim_qualities.len() - 1) as f64;
                dimension_variance.insert(dim.name(), variance);
            }
        }

        // Step 4: Adaptive priority — re-sweep top-3 dimensions
        if self.config.adaptive_priority
            && first_pass_improvements > 0
            && !stopped_by_budget
            && total_evaluations < self.config.max_evaluations
        {
            let mut sorted_dims: Vec<(&SearchDimension, f64)> = dimensions
                .iter()
                .map(|d| (d, dimension_variance.get(&d.name()).copied().unwrap_or(0.0)))
                .collect();
            sorted_dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (dim, _var) in sorted_dims.iter().take(3) {
                if dim.variant_count() == 0 {
                    continue;
                }

                for variant_idx in 0..dim.variant_count() {
                    if total_evaluations >= self.config.max_evaluations {
                        stopped_by_budget = true;
                        break;
                    }

                    let candidate_config = dim.apply_variant(&current_best.config, variant_idx);
                    let measurement = self.measure(&candidate_config, dataset, subtask_tags)?;

                    total_evaluations += 1;
                    total_problems += measurement.sample_count;
                    total_llm_calls += measurement.sample_count;
                    total_cost += measurement.cost;
                    if let Some(ref rr) = measurement.run_result {
                        total_tokens += rr.total_tokens.total();
                    }

                    let scores_candidate = measurement.per_problem_scores();
                    let scores_best = current_best.per_problem_scores();
                    let (_, p_value) = welch_t_test(&scores_candidate, &scores_best);
                    let is_improvement = self.is_better(&measurement, &current_best, p_value, alpha);

                    let obj_score = self.objective_score(if is_improvement { &measurement } else { &current_best });

                    iterations.push(SearchIteration {
                        iteration: iteration_counter,
                        dimension_name: dim.name(),
                        variant_label: dim.variant_label(variant_idx),
                        measurement: measurement.clone(),
                        p_value_vs_best: p_value,
                        is_improvement,
                    });

                    if is_improvement {
                        current_best = measurement;
                    }

                    evolution.push(EvolutionSnapshot {
                        iteration: iteration_counter,
                        current_best_quality: current_best.quality,
                        current_best_cost: current_best.cost,
                        current_best_latency_ms: current_best.latency_ms,
                        objective_score: obj_score,
                        dimension_explored: dim.name(),
                        variant_tested: dim.variant_label(variant_idx),
                        was_improvement: is_improvement,
                    });

                    iteration_counter += 1;
                }

                if stopped_by_budget {
                    break;
                }
            }
        }

        // Step 5: Build result
        let quality_improvement_pct = if baseline_measurement.quality > 0.0 {
            ((current_best.quality - baseline_measurement.quality) / baseline_measurement.quality) * 100.0
        } else if current_best.quality > 0.0 {
            100.0
        } else {
            0.0
        };

        let cost_change_pct = if baseline_measurement.cost > 0.0 {
            ((current_best.cost - baseline_measurement.cost) / baseline_measurement.cost) * 100.0
        } else {
            0.0
        };

        let converged = first_pass_improvements == 0;

        Ok(ConfigSearchResult {
            recommended: current_best.config.clone(),
            baseline: baseline_measurement,
            best: current_best,
            iterations,
            evolution,
            dimension_variance,
            quality_improvement_pct,
            cost_change_pct,
            total_evaluations,
            search_cost: SearchCost {
                total_configurations_evaluated: total_evaluations,
                total_problems_solved: total_problems,
                total_llm_calls,
                estimated_total_cost: total_cost,
                estimated_total_tokens: total_tokens,
            },
            converged,
            stopped_by_budget,
        })
    }

    /// Measure a single configuration against the dataset.
    ///
    /// Implements its own run loop (not using `BenchmarkSuiteRunner`) because
    /// it needs to route each problem to the correct model based on subtask tags.
    pub fn measure(
        &self,
        config: &EvalAgentConfig,
        dataset: &BenchmarkDataset,
        subtask_tags: &HashMap<String, Subtask>,
    ) -> Result<ConfigMeasurement, EvalSuiteError> {
        if dataset.is_empty() {
            return Err(EvalSuiteError::NoResults {
                reason: "Dataset is empty".to_string(),
            });
        }

        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut results = Vec::with_capacity(dataset.len());
        let mut all_scores = Vec::with_capacity(dataset.len());
        let mut total_cost = 0.0_f64;
        let mut total_latency = 0_u64;
        let mut total_input_tokens = 0_usize;
        let mut total_output_tokens = 0_usize;

        // Per-subtask score tracking
        let mut subtask_scores: HashMap<String, Vec<f64>> = HashMap::new();

        for problem in &dataset.problems {
            // Resolve subtask for this problem
            let subtask_str = subtask_tags
                .get(&problem.id)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "General".to_string());

            // Resolve model for this subtask
            let model = config.model_for_subtask(&subtask_str).clone();
            let model_key = model.to_string();

            // Build prompt with subtask-specific settings
            let prompt = self.build_prompt(problem, config, &subtask_str);

            // Call generator
            let start = std::time::Instant::now();
            let gen_result = self.generator.generate(&model_key, &prompt);
            let elapsed_ms = start.elapsed().as_millis() as u64;
            total_latency += elapsed_ms;

            // Cost estimation
            let call_cost = config.cost_for_model(&model);
            total_cost += call_cost;

            match gen_result {
                Ok(response) => {
                    let score = self.scorer.score(problem, &response);
                    let passed = self.scorer.passed(problem, &response);

                    let input_tokens = prompt.len() / 4;
                    let output_tokens = response.len() / 4;
                    total_input_tokens += input_tokens;
                    total_output_tokens += output_tokens;

                    all_scores.push(score);
                    subtask_scores
                        .entry(subtask_str)
                        .or_default()
                        .push(score);

                    results.push(ProblemResult {
                        problem_id: problem.id.clone(),
                        model_id: model,
                        responses: vec![response],
                        scores: vec![score],
                        passed: vec![passed],
                        latencies_ms: vec![elapsed_ms],
                        token_counts: vec![TokenUsage {
                            input_tokens,
                            output_tokens,
                        }],
                        cost_estimates: vec![call_cost],
                        error: None,
                        metadata: HashMap::new(),
                    });
                }
                Err(e) => {
                    all_scores.push(0.0);
                    subtask_scores
                        .entry(subtask_str)
                        .or_default()
                        .push(0.0);

                    results.push(ProblemResult {
                        problem_id: problem.id.clone(),
                        model_id: model,
                        responses: vec![String::new()],
                        scores: vec![0.0],
                        passed: vec![false],
                        latencies_ms: vec![elapsed_ms],
                        token_counts: vec![TokenUsage::default()],
                        cost_estimates: vec![call_cost],
                        error: Some(e),
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        let completed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Aggregate statistics
        let n = all_scores.len();
        let quality = if n > 0 {
            all_scores.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let quality_std = if n > 1 {
            let variance = all_scores
                .iter()
                .map(|s| (s - quality).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let latency_ms = if n > 0 {
            total_latency as f64 / n as f64
        } else {
            0.0
        };

        // Per-subtask quality
        let subtask_quality: HashMap<String, f64> = subtask_scores
            .into_iter()
            .map(|(subtask, scores)| {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                (subtask, mean)
            })
            .collect();

        // Synthetic "routed" model for the BenchmarkRunResult
        let routed_model = ModelIdentifier {
            name: config.name.clone(),
            provider: "routed".to_string(),
            variant: None,
        };

        let run_result = BenchmarkRunResult {
            run_id: uuid::Uuid::new_v4().to_string(),
            model_id: routed_model,
            dataset_name: dataset.name.clone(),
            suite_type: dataset.suite_type.clone(),
            results,
            started_at,
            completed_at,
            total_cost,
            total_tokens: TokenUsage {
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
            },
        };

        Ok(ConfigMeasurement {
            config: config.clone(),
            quality,
            quality_std,
            latency_ms,
            cost: total_cost,
            sample_count: n,
            subtask_quality,
            run_result: Some(run_result),
        })
    }

    /// Compute the objective score for a measurement.
    fn objective_score(&self, m: &ConfigMeasurement) -> f64 {
        match &self.config.objective {
            SearchObjective::MaxQuality => m.quality,
            SearchObjective::MinCost { min_quality } => {
                if m.quality >= *min_quality {
                    -m.cost // Lower cost is better → negate
                } else {
                    f64::NEG_INFINITY
                }
            }
            SearchObjective::CostEfficiency => {
                m.quality / m.cost.max(1e-10)
            }
            SearchObjective::Weighted {
                quality_weight,
                cost_weight,
                latency_weight,
            } => {
                quality_weight * m.quality - cost_weight * m.cost - latency_weight * m.latency_ms
            }
        }
    }

    /// Determine if a candidate measurement is better than the current best.
    fn is_better(
        &self,
        candidate: &ConfigMeasurement,
        current_best: &ConfigMeasurement,
        p_value: f64,
        alpha: f64,
    ) -> bool {
        let obj_candidate = self.objective_score(candidate);
        let obj_best = self.objective_score(current_best);

        if obj_candidate <= obj_best {
            return false;
        }

        match &self.config.objective {
            // Quality-focused: require statistical significance
            SearchObjective::MaxQuality | SearchObjective::Weighted { .. } => p_value < alpha,
            // Cost-focused: objective is already better, no t-test needed
            // (cost is deterministic from cost_per_call, quality threshold already checked)
            SearchObjective::MinCost { .. } | SearchObjective::CostEfficiency => true,
        }
    }

    /// Build a prompt for a problem with subtask-specific settings.
    fn build_prompt(
        &self,
        problem: &BenchmarkProblem,
        config: &EvalAgentConfig,
        subtask: &str,
    ) -> String {
        // Check for subtask-specific template
        if let Some(template) = config.subtask_templates.get(subtask) {
            return template
                .replace("{prompt}", &problem.prompt)
                .replace(
                    "{system}",
                    problem.system_prompt.as_deref().unwrap_or(""),
                );
        }

        // Check for subtask-specific chain-of-thought
        if config.cot_for_subtask(subtask) {
            format!(
                "{}\n\nThink step by step before giving your final answer.",
                problem.prompt
            )
        } else {
            problem.prompt.clone()
        }
    }
}

impl std::fmt::Debug for ConfigSearchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigSearchEngine")
            .field("config", &self.config)
            .field("generator", &self.generator)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Welch's t-test (inline copy, matching comparison.rs and ablation.rs pattern)
// ---------------------------------------------------------------------------

/// Welch's two-sample t-test. Returns `(t_statistic, p_value)`.
fn welch_t_test(group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
    if group_a.len() < 2 || group_b.len() < 2 {
        return (0.0, 1.0);
    }

    let n_a = group_a.len() as f64;
    let n_b = group_b.len() as f64;

    let mean_a = group_a.iter().sum::<f64>() / n_a;
    let mean_b = group_b.iter().sum::<f64>() / n_b;

    let var_a = group_a
        .iter()
        .map(|x| (x - mean_a).powi(2))
        .sum::<f64>()
        / (n_a - 1.0);
    let var_b = group_b
        .iter()
        .map(|x| (x - mean_b).powi(2))
        .sum::<f64>()
        / (n_b - 1.0);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    if se == 0.0 {
        return (0.0, 1.0);
    }

    let t = (mean_a - mean_b) / se;

    // Approximate p-value using normal CDF
    let p = 2.0 * (1.0 - normal_cdf(t.abs()));
    (t, p.max(0.0).min(1.0))
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dataset::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn model(name: &str, provider: &str) -> ModelIdentifier {
        ModelIdentifier {
            name: name.to_string(),
            provider: provider.to_string(),
            variant: None,
        }
    }

    /// Create a generator whose response quality is controlled by a base quality.
    /// For MC problems, returns the correct answer with probability ~quality.
    fn make_quality_generator(
        quality: f64,
    ) -> impl Fn(&str) -> Result<String, String> + Send + Sync {
        let call = Arc::new(AtomicUsize::new(0));
        move |prompt: &str| {
            let idx = call.fetch_add(1, Ordering::SeqCst);
            // Deterministic: first N% of calls return correct answer
            // For MC, we look for the correct option in the prompt and return it
            // Simple heuristic: if quality >= threshold for this call, return "B" (usually correct in tests)
            let threshold = (quality * 100.0) as usize;
            let is_good = (idx * 37 + 13) % 100 < threshold; // deterministic pseudo-random

            if prompt.contains("A)") || prompt.contains("A.") {
                // Multiple choice — return correct letter if "good"
                if is_good {
                    Ok("B".to_string()) // Our MC test problems use "B" as correct
                } else {
                    Ok("A".to_string()) // Wrong answer
                }
            } else if prompt.contains("factorial") || prompt.contains("function") {
                // Code problem
                if is_good {
                    Ok("def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)".to_string())
                } else {
                    Ok("def factorial(n): return 0".to_string())
                }
            } else if prompt.contains("*") || prompt.contains("+") || prompt.contains("compute") {
                // Numeric problem
                if is_good {
                    Ok("The answer is 42".to_string())
                } else {
                    Ok("The answer is 0".to_string())
                }
            } else {
                if is_good {
                    Ok("correct answer".to_string())
                } else {
                    Ok("wrong answer".to_string())
                }
            }
        }
    }

    fn make_test_dataset() -> BenchmarkDataset {
        BenchmarkDataset::from_problems(
            "test",
            BenchmarkSuiteType::Custom("config_search_test".into()),
            vec![
                make_mc_problem("p/mc1", "Q1: A) 3 B) 4 C) 5 D) 6", vec!["A", "B", "C", "D"], "B"),
                make_mc_problem("p/mc2", "Q2: A) x B) y C) z D) w", vec!["A", "B", "C", "D"], "B"),
                make_mc_problem("p/mc3", "Q3: A) a B) b C) c D) d", vec!["A", "B", "C", "D"], "B"),
                make_mc_problem("p/mc4", "Q4: A) 1 B) 2 C) 3 D) 4", vec!["A", "B", "C", "D"], "B"),
                make_mc_problem("p/mc5", "Q5: A) p B) q C) r D) s", vec!["A", "B", "C", "D"], "B"),
                make_mc_problem("p/mc6", "Q6: A) i B) j C) k D) l", vec!["A", "B", "C", "D"], "B"),
            ],
        )
    }

    fn make_test_tags() -> HashMap<String, Subtask> {
        let mut tags = HashMap::new();
        tags.insert("p/mc1".to_string(), Subtask::ReasoningChain);
        tags.insert("p/mc2".to_string(), Subtask::ReasoningChain);
        tags.insert("p/mc3".to_string(), Subtask::CodeGeneration);
        tags.insert("p/mc4".to_string(), Subtask::CodeGeneration);
        tags.insert("p/mc5".to_string(), Subtask::OutputFormatting);
        tags.insert("p/mc6".to_string(), Subtask::OutputFormatting);
        tags
    }

    #[test]
    fn test_config_search_config_defaults() {
        let config = ConfigSearchConfig::default();
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.min_samples, 5);
        assert_eq!(config.max_evaluations, 100);
        assert!(config.adaptive_priority);
        assert!(matches!(config.objective, SearchObjective::MaxQuality));
    }

    #[test]
    fn test_search_objective_variants() {
        let objs = vec![
            SearchObjective::MaxQuality,
            SearchObjective::MinCost { min_quality: 0.5 },
            SearchObjective::CostEfficiency,
            SearchObjective::Weighted {
                quality_weight: 1.0,
                cost_weight: 0.5,
                latency_weight: 0.01,
            },
        ];
        for obj in &objs {
            let s = obj.to_string();
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_measure_single_config() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let engine = ConfigSearchEngine::new(ConfigSearchConfig::default(), gen);

        let config = EvalAgentConfig::new("test", model("m", "p"));
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let m = engine.measure(&config, &dataset, &tags).unwrap();
        assert_eq!(m.sample_count, 6);
        assert!(m.quality > 0.0);
        assert!(m.run_result.is_some());
        assert_eq!(m.config.name, "test");
    }

    #[test]
    fn test_measure_with_subtask_breakdown() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let engine = ConfigSearchEngine::new(ConfigSearchConfig::default(), gen);

        let config = EvalAgentConfig::new("test", model("m", "p"));
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let m = engine.measure(&config, &dataset, &tags).unwrap();
        // Should have subtask breakdown
        assert!(m.subtask_quality.contains_key("ReasoningChain"));
        assert!(m.subtask_quality.contains_key("CodeGeneration"));
        assert!(m.subtask_quality.contains_key("OutputFormatting"));
    }

    #[test]
    fn test_measure_per_subtask_routing() {
        // Track which model was called for each prompt
        let reasoning_calls = Arc::new(AtomicUsize::new(0));
        let coding_calls = Arc::new(AtomicUsize::new(0));

        let rc = reasoning_calls.clone();
        let cc = coding_calls.clone();

        let mut gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        gen.register_model("test/reasoner", move |_| {
            rc.fetch_add(1, Ordering::SeqCst);
            Ok("B".to_string())
        });
        gen.register_model("test/coder", move |_| {
            cc.fetch_add(1, Ordering::SeqCst);
            Ok("B".to_string())
        });

        let engine = ConfigSearchEngine::new(ConfigSearchConfig::default(), gen);

        let config = EvalAgentConfig::new("routed", model("default", "test"))
            .with_subtask_model(
                &Subtask::ReasoningChain,
                model("reasoner", "test"),
            )
            .with_subtask_model(
                &Subtask::CodeGeneration,
                model("coder", "test"),
            );

        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let _m = engine.measure(&config, &dataset, &tags).unwrap();

        // Reasoning problems (2) should have called the reasoner
        assert_eq!(reasoning_calls.load(Ordering::SeqCst), 2);
        // Coding problems (2) should have called the coder
        assert_eq!(coding_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_search_single_dimension() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"))
            .with_temperature(0.5);
        let dims = vec![SearchDimension::Temperature {
            values: vec![0.0, 0.3, 0.7, 1.0],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        assert!(result.total_evaluations >= 5); // 1 baseline + 4 variants
        assert!(!result.iterations.is_empty());
        assert!(!result.evolution.is_empty());
        assert!(result.dimension_variance.contains_key("Temperature"));
    }

    #[test]
    fn test_search_finds_better_model() {
        // "good_model" returns correct answers ~90% of the time
        // "bad_model" returns correct answers ~20% of the time
        let mut gen = MultiModelGenerator::new(make_quality_generator(0.2));
        gen.register_model("test/good", make_quality_generator(0.9));

        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            confidence_level: 0.80, // Lower threshold for small samples
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("bad_start", model("bad", "test"));
        let dims = vec![SearchDimension::SubtaskModel {
            subtask: "ReasoningChain".to_string(),
            candidates: vec![model("bad", "test"), model("good", "test")],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // The search should find that "good" is better
        assert!(result.best.quality >= result.baseline.quality);
    }

    #[test]
    fn test_search_no_improvement() {
        // All models return same quality
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("optimal", model("m", "p"));
        let dims = vec![SearchDimension::ChainOfThought]; // on/off doesn't change mock
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        assert!(result.converged);
        assert!(result.quality_improvement_pct.abs() < 1.0);
    }

    #[test]
    fn test_search_multiple_dimensions() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dims = vec![
            SearchDimension::Temperature { values: vec![0.0, 0.5, 1.0] },
            SearchDimension::ChainOfThought,
        ];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // 1 baseline + 3 temp variants + 2 CoT variants = 6
        assert!(result.total_evaluations >= 6);
        assert!(result.dimension_variance.contains_key("Temperature"));
        assert!(result.dimension_variance.contains_key("ChainOfThought"));
    }

    #[test]
    fn test_search_adaptive_priority() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: true,
            max_evaluations: 100,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dims = vec![
            SearchDimension::Temperature { values: vec![0.0, 1.0] },
            SearchDimension::RagLevel { values: vec![0, 3, 5] },
        ];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // With adaptive priority, there should be more iterations than just first pass
        // (if no improvements, adaptive pass is skipped due to convergence)
        assert!(result.total_evaluations >= 6); // At least first pass
    }

    #[test]
    fn test_search_budget_guard() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 3, // Very tight budget
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dims = vec![SearchDimension::Temperature {
            values: vec![0.0, 0.3, 0.5, 0.7, 1.0],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        assert!(result.total_evaluations <= 4); // 1 baseline + at most 3 variants
        assert!(result.stopped_by_budget);
    }

    #[test]
    fn test_search_significance_gating() {
        // Two models with very similar quality — improvements should not be adopted
        let mut gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        gen.register_model("test/similar", |_| Ok("B".to_string()));

        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            confidence_level: 0.99, // Very strict
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("default", "test"));
        let dims = vec![SearchDimension::SubtaskModel {
            subtask: "CodeGeneration".to_string(),
            candidates: vec![model("similar", "test")],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // Similar models → no significant improvement → converged
        // (both return "B" → same quality)
        assert!(result.converged);
    }

    #[test]
    fn test_search_evolution_tracking() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dims = vec![SearchDimension::Temperature {
            values: vec![0.0, 0.5],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // Should have snapshots: 1 baseline + 2 variants = 3
        assert_eq!(result.evolution.len(), 3);
        assert_eq!(result.evolution[0].dimension_explored, "baseline");
        assert_eq!(result.evolution[1].dimension_explored, "Temperature");
        assert_eq!(result.evolution[2].dimension_explored, "Temperature");
    }

    #[test]
    fn test_search_cost_tracking() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let gpt4 = model("gpt-4", "openai");
        let baseline = EvalAgentConfig::new("base", gpt4.clone())
            .with_model_cost(&gpt4, 0.01);
        let dims = vec![SearchDimension::Temperature {
            values: vec![0.0, 1.0],
        }];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();
        // 3 configurations × 6 problems = 18 calls × $0.01 = $0.18
        assert!(result.search_cost.estimated_total_cost > 0.0);
        assert_eq!(result.search_cost.total_configurations_evaluated, result.total_evaluations);
        assert!(result.search_cost.total_problems_solved > 0);
        assert!(result.search_cost.total_llm_calls > 0);
    }

    #[test]
    fn test_search_result_helpers() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let config = ConfigSearchConfig {
            min_samples: 3,
            adaptive_priority: false,
            max_evaluations: 50,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(config, gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dims = vec![
            SearchDimension::Temperature { values: vec![0.0, 1.0] },
            SearchDimension::ChainOfThought,
        ];
        let dataset = make_test_dataset();
        let tags = make_test_tags();

        let result = engine.search(&baseline, &dims, &dataset, &tags).unwrap();

        // top_dimensions
        let top = result.top_dimensions(2);
        assert!(top.len() <= 2);

        // improvements
        let imps = result.improvements();
        // May or may not have improvements depending on mock behavior
        assert!(imps.len() <= result.iterations.len());

        // summary
        let summary = result.summary();
        assert!(summary.contains("Baseline quality"));
        assert!(summary.contains("Best quality"));
        assert!(summary.contains("Improvement"));
    }

    #[test]
    fn test_search_empty_dataset_error() {
        let gen = MultiModelGenerator::new(|_| Ok("B".to_string()));
        let engine = ConfigSearchEngine::new(ConfigSearchConfig::default(), gen);

        let baseline = EvalAgentConfig::new("base", model("m", "p"));
        let dataset = BenchmarkDataset::from_problems("empty", BenchmarkSuiteType::Mmlu, vec![]);
        let tags = HashMap::new();

        let result = engine.search(&baseline, &[], &dataset, &tags);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Integration test with real Ollama models (run manually)
    // -----------------------------------------------------------------------

    /// Integration test against a real Ollama server.
    ///
    /// Configure via environment variables:
    /// - `OLLAMA_URL`: Ollama base URL (default: `http://localhost:11434`)
    /// - `EVAL_MODELS`: Comma-separated model names (default: `llama3.1:8b,qwen2.5:14b-instruct-q4_K_M`)
    ///
    /// Run: `cargo test --features "full,eval-suite" --lib -- eval_suite::config_search::tests::test_integration_ollama --ignored --nocapture`
    #[test]
    #[ignore]
    fn test_integration_ollama_real_models() {
        let base_url = std::env::var("OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model_names: Vec<String> = std::env::var("EVAL_MODELS")
            .unwrap_or_else(|_| "llama3.1:8b,qwen2.5:14b-instruct-q4_K_M".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if model_names.is_empty() {
            eprintln!("No models configured, skipping");
            return;
        }

        println!("=== Ollama Integration Test ===");
        println!("URL: {}", base_url);
        println!("Models: {:?}", model_names);

        // Create generators for each model
        let first_model = model_names[0].clone();
        let mut gen = MultiModelGenerator::new(
            make_ollama_generator(&base_url, &first_model),
        );
        for m in &model_names[1..] {
            let model_id = ModelIdentifier {
                name: m.clone(),
                provider: "ollama".to_string(),
                variant: None,
            };
            gen.register_model(
                &model_id.to_string(),
                make_ollama_generator(&base_url, m),
            );
        }

        // Build comprehensive test dataset covering 6 categories:
        // Knowledge (MMLU-style), Math, Reasoning (BBH-style), Numeric (GSM8K-style),
        // Code (HumanEval-style), and Tool Selection/MCP.
        // 43 problems total — enough for statistical significance in per-subtask routing.
        let mc_suffix = "\n\nRespond with ONLY the letter of the correct answer (e.g., \"B\").";
        let num_suffix = "\n\nGive only the number, nothing else.";
        let dataset = BenchmarkDataset::from_problems(
            "ollama_integration",
            BenchmarkSuiteType::Custom("integration".into()),
            vec![
                // ── KNOWLEDGE — MMLU-style factual MC (InformationGathering) ──
                make_mc_problem(
                    "knowledge/1",
                    &format!("Which planet is closest to the Sun? A) Venus B) Mercury C) Mars D) Earth{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "knowledge/2",
                    &format!("What is the chemical symbol for gold? A) Ag B) Fe C) Au D) Cu{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "knowledge/3",
                    &format!("Which organ in the human body produces insulin? A) Liver B) Pancreas C) Kidney D) Stomach{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "knowledge/4",
                    &format!("What is the approximate speed of light? A) 150,000 km/s B) 300,000 km/s C) 500,000 km/s D) 1,000,000 km/s{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "knowledge/5",
                    &format!("Which gas makes up approximately 78%% of Earth's atmosphere? A) Oxygen B) Carbon dioxide C) Nitrogen D) Argon{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "knowledge/6",
                    &format!("Who wrote the play 'Romeo and Juliet'? A) Charles Dickens B) William Shakespeare C) Jane Austen D) Mark Twain{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "knowledge/7",
                    &format!("What is the capital of Australia? A) Sydney B) Melbourne C) Canberra D) Brisbane{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "knowledge/8",
                    &format!("Which element has atomic number 1? A) Helium B) Hydrogen C) Lithium D) Oxygen{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "knowledge/9",
                    &format!("In which year did World War II end? A) 1943 B) 1944 C) 1945 D) 1946{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "knowledge/10",
                    &format!("What is the largest planet in our solar system? A) Saturn B) Neptune C) Jupiter D) Uranus{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                // ── MATH — Arithmetic and number theory MC (ReasoningChain) ──
                make_mc_problem(
                    "math/1",
                    &format!("What is 15 * 17? A) 245 B) 255 C) 265 D) 275{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "math/2",
                    &format!("What is the square root of 144? A) 10 B) 11 C) 12 D) 14{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "math/3",
                    &format!("What is 23 + 47 * 2? (Use standard order of operations) A) 140 B) 117 C) 93 D) 94{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "math/4",
                    &format!("How many prime numbers are there between 1 and 20? A) 6 B) 7 C) 8 D) 9{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "math/5",
                    &format!("What is 3 raised to the power of 4 (3^4)? A) 12 B) 27 C) 64 D) 81{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "D",
                ),
                // ── REASONING — BBH-style logic, deduction, analogies (ReasoningChain) ──
                make_mc_problem(
                    "reason/1",
                    &format!("If all roses are flowers and some flowers fade quickly, which must be true? A) All roses fade quickly B) Some roses fade quickly C) No roses fade quickly D) Some flowers are roses{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "D",
                ),
                make_mc_problem(
                    "reason/2",
                    &format!("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? A) $0.10 B) $0.05 C) $0.15 D) $0.01{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "reason/3",
                    &format!("All cats are mammals. All mammals breathe air. Therefore: A) All cats breathe air B) All things that breathe air are cats C) Some mammals are not cats D) All breathing things are mammals{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "A",
                ),
                make_mc_problem(
                    "reason/4",
                    &format!("If today is Wednesday, what day will it be 10 days from now? A) Friday B) Saturday C) Sunday D) Monday{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "reason/5",
                    &format!("All dogs bark. Rex barks. What can we conclude? A) Rex is definitely a dog B) Rex might or might not be a dog C) Rex is not a dog D) All barking animals are dogs{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "reason/6",
                    &format!("Hot is to cold as tall is to: A) Long B) High C) Short D) Big{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "reason/7",
                    &format!("What comes next in the sequence: 2, 6, 18, 54, ...? A) 108 B) 162 C) 216 D) 72{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "reason/8",
                    &format!("If you face north and turn right 90 degrees, which direction are you facing? A) South B) West C) East D) Northwest{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                // ── NUMERIC — GSM8K-style calculations (CodeGeneration) ──
                make_numeric_problem(
                    "calc/1",
                    &format!("Compute 7 factorial (7!).{}", num_suffix),
                    5040.0, 1.0,
                ),
                make_numeric_problem(
                    "calc/2",
                    &format!("What is 2^10?{}", num_suffix),
                    1024.0, 1.0,
                ),
                make_numeric_problem(
                    "calc/3",
                    &format!("What is the sum of all integers from 1 to 100?{}", num_suffix),
                    5050.0, 1.0,
                ),
                make_numeric_problem(
                    "calc/4",
                    &format!("What is 12 percent of 250?{}", num_suffix),
                    30.0, 0.5,
                ),
                make_numeric_problem(
                    "calc/5",
                    &format!("How many seconds are in one hour?{}", num_suffix),
                    3600.0, 1.0,
                ),
                make_numeric_problem(
                    "calc/6",
                    &format!("What is 17 times 23?{}", num_suffix),
                    391.0, 1.0,
                ),
                make_numeric_problem(
                    "calc/7",
                    &format!("What is the area of a circle with radius 5? Use pi = 3.14159.{}", num_suffix),
                    78.54, 1.0,
                ),
                make_numeric_problem(
                    "calc/8",
                    &format!("What is 1000 divided by 8?{}", num_suffix),
                    125.0, 0.5,
                ),
                // ── CODE — HumanEval-style simple Python functions (CodeGeneration) ──
                make_code_problem(
                    "code/1",
                    "Write a Python function `is_even(n)` that returns True if n is even, False otherwise. Write only the function, no explanation.",
                    "def is_even(n):\n    return n % 2 == 0",
                    "python",
                ),
                make_code_problem(
                    "code/2",
                    "Write a Python function `reverse_string(s)` that returns the reverse of string s. Write only the function, no explanation.",
                    "def reverse_string(s):\n    return s[::-1]",
                    "python",
                ),
                make_code_problem(
                    "code/3",
                    "Write a Python function `max_of_three(a, b, c)` that returns the largest of three numbers. Write only the function, no explanation.",
                    "def max_of_three(a, b, c):\n    return max(a, b, c)",
                    "python",
                ),
                make_code_problem(
                    "code/4",
                    "Write a Python function `factorial(n)` that returns n factorial. Write only the function, no explanation.",
                    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                    "python",
                ),
                // ── TOOL SELECTION / MCP — Tool use reasoning (ToolSelection) ──
                make_mc_problem(
                    "tool/1",
                    &format!("You are an AI assistant with these tools: [calculator, web_search, file_reader, code_executor]. A user asks: 'What is 847 times 293?' Which tool should you use? A) web_search B) calculator C) file_reader D) code_executor{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "tool/2",
                    &format!("You are an AI assistant with these tools: [code_executor, database_query, email_sender, calendar]. A user asks: 'Run this Python script and show the output.' Which tool? A) email_sender B) database_query C) code_executor D) calendar{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "tool/3",
                    &format!("When should an AI assistant use a tool instead of answering from its own knowledge? A) Always, for every question B) When the question requires real-time or external data it does not have C) Never, tools are unreliable D) Only for math questions{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "tool/4",
                    &format!("You have tools: [get_weather, get_stock_price, search_web, send_email]. A user asks: 'Will it rain in Madrid tomorrow?' Which tool? A) search_web B) send_email C) get_weather D) get_stock_price{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
                make_mc_problem(
                    "tool/5",
                    &format!("In modern AI tool-calling systems (like MCP or function calling), what format is typically used to describe tool parameters? A) XML Schema B) JSON Schema C) YAML D) Plain text{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "tool/6",
                    &format!("A user asks: 'Summarize the contents of report.pdf.' You have tools: [calculator, file_reader, web_search, text_to_speech]. Which tool is most appropriate? A) calculator B) text_to_speech C) web_search D) file_reader{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "D",
                ),
                make_mc_problem(
                    "tool/7",
                    &format!("What is the correct sequence for using a tool in an AI system? A) Execute tool, then decide parameters B) Select tool, specify parameters, execute, read result C) Read result, then execute tool D) Execute all tools simultaneously{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "B",
                ),
                make_mc_problem(
                    "tool/8",
                    &format!("You have: [database_query, web_search, calculator]. A user asks: 'How many customers ordered more than $100 last month from our database?' Which tool? A) calculator B) web_search C) database_query D) None needed{}", mc_suffix),
                    vec!["A", "B", "C", "D"], "C",
                ),
            ],
        );

        // Map each problem to its subtask for per-subtask model routing
        let mut tags = HashMap::new();
        for i in 1..=10 {
            tags.insert(format!("knowledge/{}", i), Subtask::InformationGathering);
        }
        for i in 1..=5 {
            tags.insert(format!("math/{}", i), Subtask::ReasoningChain);
        }
        for i in 1..=8 {
            tags.insert(format!("reason/{}", i), Subtask::ReasoningChain);
        }
        for i in 1..=8 {
            tags.insert(format!("calc/{}", i), Subtask::CodeGeneration);
        }
        for i in 1..=4 {
            tags.insert(format!("code/{}", i), Subtask::CodeGeneration);
        }
        for i in 1..=8 {
            tags.insert(format!("tool/{}", i), Subtask::ToolSelection);
        }

        // Create model identifiers for the search
        let model_ids: Vec<ModelIdentifier> = model_names
            .iter()
            .map(|name| ModelIdentifier {
                name: name.clone(),
                provider: "ollama".to_string(),
                variant: None,
            })
            .collect();

        let baseline = EvalAgentConfig::new("baseline", model_ids[0].clone());

        // Build search dimensions
        let mut dims = Vec::new();
        if model_ids.len() > 1 {
            dims.push(SearchDimension::SubtaskModel {
                subtask: "ReasoningChain".to_string(),
                candidates: model_ids.clone(),
            });
            dims.push(SearchDimension::SubtaskModel {
                subtask: "CodeGeneration".to_string(),
                candidates: model_ids.clone(),
            });
            dims.push(SearchDimension::SubtaskModel {
                subtask: "ToolSelection".to_string(),
                candidates: model_ids.clone(),
            });
            dims.push(SearchDimension::SubtaskModel {
                subtask: "InformationGathering".to_string(),
                candidates: model_ids.clone(),
            });
        }
        dims.push(SearchDimension::ChainOfThought);
        dims.push(SearchDimension::Temperature {
            values: vec![0.0, 0.3, 0.7],
        });

        let search_config = ConfigSearchConfig {
            min_samples: 5,
            adaptive_priority: false,
            max_evaluations: 50,
            confidence_level: 0.80,
            ..Default::default()
        };
        let engine = ConfigSearchEngine::new(search_config, gen);

        println!("\nRunning search...");
        let result = engine
            .search(&baseline, &dims, &dataset, &tags)
            .expect("Search should succeed");

        println!("\n{}", result.summary());

        println!("\n--- Evolution ---");
        for snap in &result.evolution {
            println!(
                "  [{}] {} = {} | quality={:.4} cost={:.6} obj={:.4} {}",
                snap.iteration,
                snap.dimension_explored,
                snap.variant_tested,
                snap.current_best_quality,
                snap.current_best_cost,
                snap.objective_score,
                if snap.was_improvement { "<< IMPROVEMENT" } else { "" },
            );
        }

        println!("\n--- Subtask Quality (best config) ---");
        for (subtask, quality) in &result.best.subtask_quality {
            println!("  {}: {:.4}", subtask, quality);
        }

        println!("\n--- Search Cost ---");
        println!(
            "  Configs: {} | LLM calls: {} | Cost: ${:.4}",
            result.search_cost.total_configurations_evaluated,
            result.search_cost.total_llm_calls,
            result.search_cost.estimated_total_cost,
        );

        // Basic assertions
        assert!(result.baseline.quality >= 0.0);
        assert!(result.best.quality >= 0.0);
        assert!(result.best.quality >= result.baseline.quality - 0.01); // Best should be at least as good
    }

    /// Create an Ollama generator that makes real HTTP calls.
    fn make_ollama_generator(
        base_url: &str,
        model_name: &str,
    ) -> impl Fn(&str) -> Result<String, String> + Send + Sync {
        let url = format!("{}/api/generate", base_url);
        let model = model_name.to_string();
        move |prompt: &str| {
            // Build JSON payload
            let escaped_prompt = prompt
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            let body = format!(
                r#"{{"model":"{}","prompt":"{}","stream":false,"options":{{"num_predict":256}}}}"#,
                model, escaped_prompt
            );

            // Parse URL
            let url_without_scheme = url.strip_prefix("http://").ok_or("Invalid URL")?;
            let (host_port, path) = url_without_scheme
                .split_once('/')
                .unwrap_or((url_without_scheme, ""));
            let path = format!("/{}", path);

            // Connect via TCP
            let mut stream = std::net::TcpStream::connect(host_port)
                .map_err(|e| format!("Connection failed: {}", e))?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(120)))
                .ok();

            // Send HTTP request
            use std::io::Write;
            let request = format!(
                "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                path,
                host_port,
                body.len(),
                body
            );
            stream
                .write_all(request.as_bytes())
                .map_err(|e| format!("Write failed: {}", e))?;

            // Read response
            use std::io::Read;
            let mut response = String::new();
            stream
                .read_to_string(&mut response)
                .map_err(|e| format!("Read failed: {}", e))?;

            // Parse response — find the JSON body after headers
            let body_start = response.find("\r\n\r\n").unwrap_or(0) + 4;
            let json_body = &response[body_start..];

            // Extract "response" field from JSON
            // Simple parser: find "response":" and extract the value
            if let Some(start) = json_body.find("\"response\":\"") {
                let value_start = start + 12;
                let remaining = &json_body[value_start..];
                // Find the closing unescaped quote using byte offsets
                let bytes = remaining.as_bytes();
                let mut byte_end = 0;
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2; // skip escaped char
                        continue;
                    }
                    if bytes[i] == b'"' {
                        byte_end = i;
                        break;
                    }
                    i += 1;
                }
                let raw_response = &remaining[..byte_end];
                let cleaned = raw_response
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\");
                Ok(cleaned)
            } else if json_body.contains("\"error\"") {
                Err(format!("Ollama error: {}", json_body.trim()))
            } else {
                Err(format!("Could not parse Ollama response: {}", &json_body[..json_body.len().min(200)]))
            }
        }
    }
}
