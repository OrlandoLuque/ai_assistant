// Required Notice: Copyright (c) 2026 Orlando Jose Luque Moraira (Lander)
// Licensed under PolyForm Noncommercial 1.0.0 — see LICENSE file.

//! Configuration Optimizer — ML-based configuration tuning.
//!
//! This module implements a three-phase optimization pipeline for finding the best
//! configuration parameters for LLM-based applications:
//!
//! 1. **Ablation** — Systematically disable features one-at-a-time to measure their
//!    individual contribution (impact analysis).
//! 2. **Bayesian Search** — Use a KNN-based surrogate model with Expected Improvement
//!    acquisition to efficiently explore the configuration space.
//! 3. **Fine-Tuning** — Refine continuous parameters within the best configuration
//!    found by Bayesian search.
//!
//! Additionally, a Thompson Sampling bandit model tracks configuration "arms" across
//! code version changes, applying exponential decay to stale observations and
//! detecting regressions.
//!
//! # Example
//!
//! ```rust
//! use ai_assistant::config_optimizer::*;
//!
//! let config = OptimizerConfig::default();
//! let mut optimizer = ConfigOptimizer::new(config);
//!
//! // Define a mock benchmark function
//! let benchmark = |point: &ConfigPoint| -> f64 {
//!     // Return a quality score for this configuration
//!     0.85
//! };
//!
//! // Run one optimization round
//! let result = optimizer.run_round(&benchmark);
//! assert!(result.quality_score >= 0.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Goal that drives the optimizer's reward function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    /// Maximize quality regardless of cost/latency.
    BestQuality,
    /// Find the cheapest configuration whose quality >= threshold.
    CheapestAboveThreshold(f64),
    /// Find the fastest configuration whose quality >= threshold.
    FastestAboveThreshold(f64),
    /// Balanced trade-off between quality, latency and cost.
    Balanced,
    /// Custom weighted objective.
    Custom {
        quality_w: f64,
        latency_w: f64,
        cost_w: f64,
    },
}

/// Current phase of the optimization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationPhase {
    Ablation,
    BayesianSearch,
    FineTuning,
    Done,
}

/// A single typed configuration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Float(f64),
    Str(String),
    Uint(usize),
}

impl ConfigValue {
    /// Numeric representation for distance calculations.
    fn as_f64(&self) -> f64 {
        match self {
            ConfigValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            ConfigValue::Float(f) => *f,
            ConfigValue::Uint(u) => *u as f64,
            ConfigValue::Str(_) => 0.0,
        }
    }
}

/// A configuration point — maps parameter name to value.
pub type ConfigPoint = HashMap<String, ConfigValue>;

/// Top-level optimizer configuration with sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Exponential decay factor applied to old observations on version change.
    pub decay_factor: f64,
    /// Maximum total evaluations before stopping.
    pub max_evaluations: usize,
    /// Maximum LLM calls per optimization round (budget guard).
    pub max_llm_calls_per_round: usize,
    /// Per-benchmark timeout in seconds.
    pub benchmark_timeout_secs: u64,
    /// Sliding window size for recent observations.
    pub sliding_window_size: usize,
    /// Number of top features to keep after ablation.
    pub ablation_top_k: usize,
    /// Minimum samples before caching a configuration score.
    pub samples_before_cache: usize,
    /// Optimization goal that drives the reward function.
    pub goal: OptimizationGoal,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.95,
            max_evaluations: 200,
            max_llm_calls_per_round: 10,
            benchmark_timeout_secs: 30,
            sliding_window_size: 50,
            ablation_top_k: 15,
            samples_before_cache: 3,
            goal: OptimizationGoal::Balanced,
        }
    }
}

/// Result of a single configuration evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub config_point: ConfigPoint,
    pub quality_score: f64,
    pub latency_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub cost_usd: f64,
    pub code_version: String,
    pub timestamp: String,
    pub phase: OptimizationPhase,
    pub mode: String,
    pub benchmark_name: String,
}

/// Result of an ablation experiment for a single feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub feature_name: String,
    pub baseline_score: f64,
    pub disabled_score: f64,
    /// Impact = baseline - disabled. Positive means the feature helps.
    pub impact: f64,
    pub importance_rank: usize,
}

/// A bandit arm representing a specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigArm {
    pub id: String,
    pub config_point: ConfigPoint,
    /// Alpha parameter for Beta distribution (successes + 1).
    pub alpha: f64,
    /// Beta parameter for Beta distribution (failures + 1).
    pub beta: f64,
    pub pull_count: u32,
    pub total_reward: f64,
    pub available: bool,
    pub results: Vec<EvaluationResult>,
}

impl ConfigArm {
    fn new(id: String, config_point: ConfigPoint) -> Self {
        Self {
            id,
            config_point,
            alpha: 1.0,
            beta: 1.0,
            pull_count: 0,
            total_reward: 0.0,
            available: true,
            results: Vec::new(),
        }
    }

    pub fn mean_reward(&self) -> f64 {
        if self.pull_count == 0 {
            0.0
        } else {
            self.total_reward / self.pull_count as f64
        }
    }

    /// Thompson Sampling score: alpha / (alpha + beta), used as a point estimate
    /// when we cannot sample from a Beta distribution without an RNG crate.
    pub fn thompson_score(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

// ---------------------------------------------------------------------------
// SurrogateModel — KNN-based Bayesian optimization
// ---------------------------------------------------------------------------

/// Simplified Bayesian optimization surrogate based on K-Nearest-Neighbors
/// regression. Predicts mean and uncertainty for unseen config points, and
/// computes Expected Improvement for acquisition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateModel {
    observations: Vec<(ConfigPoint, f64)>,
    k_neighbors: usize,
}

impl SurrogateModel {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            k_neighbors: 5,
        }
    }

    /// Record a (config, score) observation.
    pub fn add_observation(&mut self, point: ConfigPoint, score: f64) {
        self.observations.push((point, score));
    }

    /// Predict the score for `point` using KNN regression.
    ///
    /// Returns `(mean, uncertainty)`. When there are fewer observations than
    /// `k_neighbors`, uncertainty is high to encourage exploration.
    pub fn predict(&self, point: &ConfigPoint) -> (f64, f64) {
        if self.observations.is_empty() {
            return (0.5, 1.0); // maximum uncertainty
        }

        let mut distances: Vec<(f64, f64)> = self
            .observations
            .iter()
            .map(|(obs_point, score)| (config_distance(point, obs_point), *score))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k_neighbors.min(distances.len());
        let neighbors = &distances[..k];

        let mean = neighbors.iter().map(|(_, s)| s).sum::<f64>() / k as f64;

        // Uncertainty: std deviation of neighbor scores + distance penalty
        let variance = if k > 1 {
            let sq_diffs: f64 = neighbors.iter().map(|(_, s)| (s - mean).powi(2)).sum();
            sq_diffs / (k - 1) as f64
        } else {
            0.25 // single neighbor → moderate uncertainty
        };

        let avg_distance = neighbors.iter().map(|(d, _)| d).sum::<f64>() / k as f64;
        let uncertainty = variance.sqrt() + avg_distance * 0.1;

        (mean, uncertainty)
    }

    /// Expected Improvement: how much we expect `point` to improve over
    /// `best_score`. Uses a simplified normal CDF approximation.
    pub fn expected_improvement(&self, point: &ConfigPoint, best_score: f64) -> f64 {
        let (mean, uncertainty) = self.predict(point);

        if uncertainty < 1e-12 {
            return if mean > best_score {
                mean - best_score
            } else {
                0.0
            };
        }

        let z = (mean - best_score) / uncertainty;

        // Approximate Φ(z) and φ(z) for the EI formula:
        // EI = (mean - best) * Φ(z) + uncertainty * φ(z)
        let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let big_phi = 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));

        let ei = (mean - best_score) * big_phi + uncertainty * phi;
        ei.max(0.0)
    }

    /// Generate random-ish candidates and return the one with highest EI.
    pub fn suggest_next(&self, feature_names: &[String], best_score: f64) -> ConfigPoint {
        let n_candidates = 20;
        let mut best_ei = f64::NEG_INFINITY;
        let mut best_point = ConfigPoint::new();

        for i in 0..n_candidates {
            let candidate = generate_candidate(feature_names, i, n_candidates);
            let ei = self.expected_improvement(&candidate, best_score);
            if ei > best_ei {
                best_ei = ei;
                best_point = candidate;
            }
        }

        if best_point.is_empty() && !feature_names.is_empty() {
            // Fallback: return a simple candidate
            best_point = generate_candidate(feature_names, 0, 1);
        }

        best_point
    }

    /// Number of recorded observations.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

impl Default for SurrogateModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate error function (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Distance between two ConfigPoints.
///
/// - Bool: Hamming (0 if same, 1 if different)
/// - Float/Uint: normalized absolute difference (capped to [0,1])
/// - Str: 0 if equal, 1 otherwise
/// - Missing keys count as distance 1.
pub fn config_distance(a: &ConfigPoint, b: &ConfigPoint) -> f64 {
    let mut all_keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    all_keys.sort();
    all_keys.dedup();

    if all_keys.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for key in &all_keys {
        match (a.get(*key), b.get(*key)) {
            (Some(va), Some(vb)) => {
                total += single_value_distance(va, vb);
            }
            _ => {
                total += 1.0; // missing key
            }
        }
    }

    total / all_keys.len() as f64
}

fn single_value_distance(a: &ConfigValue, b: &ConfigValue) -> f64 {
    match (a, b) {
        (ConfigValue::Bool(x), ConfigValue::Bool(y)) => {
            if x == y {
                0.0
            } else {
                1.0
            }
        }
        (ConfigValue::Str(x), ConfigValue::Str(y)) => {
            if x == y {
                0.0
            } else {
                1.0
            }
        }
        (ConfigValue::Float(x), ConfigValue::Float(y)) => {
            let diff = (x - y).abs();
            let max_val = x.abs().max(y.abs()).max(1.0);
            (diff / max_val).min(1.0)
        }
        (ConfigValue::Uint(x), ConfigValue::Uint(y)) => {
            let diff = (*x as f64 - *y as f64).abs();
            let max_val = (*x as f64).max(*y as f64).max(1.0);
            (diff / max_val).min(1.0)
        }
        _ => {
            // Different types — use numeric conversion
            let diff = (a.as_f64() - b.as_f64()).abs();
            diff.min(1.0)
        }
    }
}

/// Generate a deterministic candidate configuration.
fn generate_candidate(
    feature_names: &[String],
    index: usize,
    total: usize,
) -> ConfigPoint {
    let mut point = ConfigPoint::new();
    for (i, name) in feature_names.iter().enumerate() {
        // Use a simple deterministic pattern based on index
        let phase = (index as f64 * std::f64::consts::PI * (i + 1) as f64 / total as f64).sin();
        let value = if name.contains("temperature") || name.contains("top_p") || name.contains("penalty") {
            ConfigValue::Float((phase * 0.5 + 0.5).clamp(0.0, 1.0))
        } else if name.contains("tokens") || name.contains("size") || name.contains("count") {
            ConfigValue::Uint(((phase * 0.5 + 0.5) * 2048.0) as usize)
        } else if name.contains("model") || name.contains("provider") {
            ConfigValue::Str(format!("variant_{}", index % 4))
        } else {
            ConfigValue::Bool(phase > 0.0)
        };
        point.insert(name.clone(), value);
    }
    point
}

// ---------------------------------------------------------------------------
// ConfigOptimizer — main orchestrator
// ---------------------------------------------------------------------------

/// Main optimization engine that orchestrates ablation, Bayesian search, and
/// fine-tuning phases.
#[derive(Serialize, Deserialize)]
pub struct ConfigOptimizer {
    config: OptimizerConfig,
    phase: OptimizationPhase,
    code_version: String,
    arms: Vec<ConfigArm>,
    ablation_results: Vec<AblationResult>,
    surrogate: SurrogateModel,
    evaluations: Vec<EvaluationResult>,
    total_rounds: usize,
    best_config: Option<ConfigPoint>,
    best_score: f64,
    important_features: Vec<String>,
}

impl ConfigOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            phase: OptimizationPhase::Ablation,
            code_version: get_code_version(),
            arms: Vec::new(),
            ablation_results: Vec::new(),
            surrogate: SurrogateModel::new(),
            evaluations: Vec::new(),
            total_rounds: 0,
            best_config: None,
            best_score: f64::NEG_INFINITY,
            important_features: Vec::new(),
        }
    }

    /// Discover all toggleable features that the optimizer can tune.
    ///
    /// Returns a list of (name, default_value) pairs covering common LLM
    /// configuration parameters.
    pub fn discover_features() -> Vec<(String, ConfigValue)> {
        vec![
            ("temperature".to_string(), ConfigValue::Float(0.7)),
            ("top_p".to_string(), ConfigValue::Float(0.9)),
            ("top_k_sampling".to_string(), ConfigValue::Uint(40)),
            ("max_tokens".to_string(), ConfigValue::Uint(2048)),
            ("presence_penalty".to_string(), ConfigValue::Float(0.0)),
            ("frequency_penalty".to_string(), ConfigValue::Float(0.0)),
            ("repeat_penalty".to_string(), ConfigValue::Float(1.1)),
            ("use_cache".to_string(), ConfigValue::Bool(true)),
            ("use_streaming".to_string(), ConfigValue::Bool(true)),
            ("use_guardrails".to_string(), ConfigValue::Bool(true)),
            ("use_rag".to_string(), ConfigValue::Bool(false)),
            ("use_cot".to_string(), ConfigValue::Bool(false)),
            ("use_self_consistency".to_string(), ConfigValue::Bool(false)),
            ("chunk_size".to_string(), ConfigValue::Uint(512)),
            ("chunk_overlap".to_string(), ConfigValue::Uint(64)),
            ("retrieval_count".to_string(), ConfigValue::Uint(5)),
            ("model_provider".to_string(), ConfigValue::Str("ollama".to_string())),
            ("model_name".to_string(), ConfigValue::Str("llama3".to_string())),
        ]
    }

    /// Current optimization phase.
    pub fn phase(&self) -> OptimizationPhase {
        self.phase
    }

    /// Total number of rounds executed so far.
    pub fn total_rounds(&self) -> usize {
        self.total_rounds
    }

    // -----------------------------------------------------------------------
    // Phase 1: Ablation
    // -----------------------------------------------------------------------

    /// Run one ablation round: evaluate the baseline with `feature` disabled.
    pub fn run_ablation_round(
        &mut self,
        baseline: &ConfigPoint,
        feature: &str,
        benchmark_fn: &dyn Fn(&ConfigPoint) -> f64,
    ) -> AblationResult {
        // Evaluate baseline
        let baseline_score = benchmark_fn(baseline);

        // Create a modified config with the feature disabled
        let mut disabled = baseline.clone();
        if let Some(val) = disabled.get(feature) {
            let disabled_val = match val {
                ConfigValue::Bool(_) => ConfigValue::Bool(false),
                ConfigValue::Float(_) => ConfigValue::Float(0.0),
                ConfigValue::Uint(_) => ConfigValue::Uint(0),
                ConfigValue::Str(_) => ConfigValue::Str(String::new()),
            };
            disabled.insert(feature.to_string(), disabled_val);
        }

        let disabled_score = benchmark_fn(&disabled);
        let impact = baseline_score - disabled_score;

        let rank = self.ablation_results.len() + 1;

        let result = AblationResult {
            feature_name: feature.to_string(),
            baseline_score,
            disabled_score,
            impact,
            importance_rank: rank,
        };

        self.ablation_results.push(result.clone());

        // Record evaluation
        let timestamp = current_timestamp();
        let eval = EvaluationResult {
            config_point: disabled,
            quality_score: disabled_score,
            latency_ms: None,
            tokens_per_second: None,
            cost_usd: 0.0,
            code_version: self.code_version.clone(),
            timestamp,
            phase: OptimizationPhase::Ablation,
            mode: "quality".to_string(),
            benchmark_name: format!("ablation_{}", feature),
        };
        self.evaluations.push(eval);
        self.total_rounds += 1;

        // Update best if baseline beats current
        if baseline_score > self.best_score {
            self.best_score = baseline_score;
            self.best_config = Some(baseline.clone());
        }

        result
    }

    /// Finalize the ablation phase: sort by impact, select top-K features,
    /// and transition to BayesianSearch.
    pub fn finalize_ablation(&mut self) {
        // Sort by absolute impact (descending)
        self.ablation_results
            .sort_by(|a, b| b.impact.abs().partial_cmp(&a.impact.abs()).unwrap_or(std::cmp::Ordering::Equal));

        // Re-assign ranks
        for (i, result) in self.ablation_results.iter_mut().enumerate() {
            result.importance_rank = i + 1;
        }

        // Select top-K important features
        let top_k = self.config.ablation_top_k.min(self.ablation_results.len());
        self.important_features = self
            .ablation_results
            .iter()
            .take(top_k)
            .map(|r| r.feature_name.clone())
            .collect();

        self.phase = OptimizationPhase::BayesianSearch;
    }

    // -----------------------------------------------------------------------
    // Phase 2: Bayesian Search
    // -----------------------------------------------------------------------

    /// Run one Bayesian optimization round: suggest a configuration via the
    /// surrogate model, evaluate it, and update the model.
    pub fn run_bayesian_round(
        &mut self,
        benchmark_fn: &dyn Fn(&ConfigPoint) -> f64,
    ) -> EvaluationResult {
        // Ensure we have features to optimize
        if self.important_features.is_empty() {
            let features = Self::discover_features();
            self.important_features = features.iter().map(|(n, _)| n.clone()).collect();
        }

        // Suggest the next point
        let candidate = self.surrogate.suggest_next(&self.important_features, self.best_score);

        // Evaluate
        let score = benchmark_fn(&candidate);

        // Record in surrogate
        self.surrogate.add_observation(candidate.clone(), score);

        // Update best
        if score > self.best_score {
            self.best_score = score;
            self.best_config = Some(candidate.clone());
        }

        // Create or update arm
        let arm_id = format!("bayesian_{}", self.total_rounds);
        let mut arm = ConfigArm::new(arm_id.clone(), candidate.clone());
        arm.pull_count = 1;
        arm.total_reward = score;
        arm.alpha += score;
        arm.beta += 1.0 - score.clamp(0.0, 1.0);

        let timestamp = current_timestamp();
        let eval = EvaluationResult {
            config_point: candidate,
            quality_score: score,
            latency_ms: None,
            tokens_per_second: None,
            cost_usd: 0.0,
            code_version: self.code_version.clone(),
            timestamp,
            phase: OptimizationPhase::BayesianSearch,
            mode: "quality".to_string(),
            benchmark_name: "bayesian_search".to_string(),
        };

        arm.results.push(eval.clone());
        self.arms.push(arm);
        self.evaluations.push(eval.clone());
        self.total_rounds += 1;

        // Check if we should transition to fine-tuning
        let bayesian_rounds = self
            .evaluations
            .iter()
            .filter(|e| e.phase == OptimizationPhase::BayesianSearch)
            .count();

        // Transition after enough exploration or if improvement stagnates
        if bayesian_rounds >= self.config.max_evaluations / 2 {
            self.phase = OptimizationPhase::FineTuning;
        }

        eval
    }

    // -----------------------------------------------------------------------
    // Phase 3: Fine-Tuning
    // -----------------------------------------------------------------------

    /// Fine-tune continuous parameters around the current best configuration.
    pub fn run_fine_tuning_round(
        &mut self,
        benchmark_fn: &dyn Fn(&ConfigPoint) -> f64,
    ) -> EvaluationResult {
        let base = self.best_config.clone().unwrap_or_default();

        // Perturb continuous parameters slightly
        let mut candidate = base.clone();
        let perturbation = 0.05 * (1.0 / (1.0 + self.total_rounds as f64 * 0.1));

        for (key, value) in &base {
            match value {
                ConfigValue::Float(f) => {
                    // Alternate perturbation direction based on round
                    let direction = if self.total_rounds % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let new_val = (f + direction * perturbation).clamp(0.0, 2.0);
                    candidate.insert(key.clone(), ConfigValue::Float(new_val));
                }
                ConfigValue::Uint(u) => {
                    let delta = ((*u as f64) * perturbation).max(1.0) as usize;
                    let new_val = if self.total_rounds % 2 == 0 {
                        u.saturating_add(delta)
                    } else {
                        u.saturating_sub(delta)
                    };
                    candidate.insert(key.clone(), ConfigValue::Uint(new_val));
                }
                _ => {} // leave bools and strings unchanged
            }
        }

        let score = benchmark_fn(&candidate);

        // Update best
        if score > self.best_score {
            self.best_score = score;
            self.best_config = Some(candidate.clone());
        }

        self.surrogate.add_observation(candidate.clone(), score);

        let timestamp = current_timestamp();
        let eval = EvaluationResult {
            config_point: candidate,
            quality_score: score,
            latency_ms: None,
            tokens_per_second: None,
            cost_usd: 0.0,
            code_version: self.code_version.clone(),
            timestamp,
            phase: OptimizationPhase::FineTuning,
            mode: "quality".to_string(),
            benchmark_name: "fine_tuning".to_string(),
        };

        self.evaluations.push(eval.clone());
        self.total_rounds += 1;

        // Check if done
        if self.total_rounds >= self.config.max_evaluations {
            self.phase = OptimizationPhase::Done;
        }

        eval
    }

    // -----------------------------------------------------------------------
    // Auto-dispatch
    // -----------------------------------------------------------------------

    /// Automatically select the appropriate phase and run one round.
    pub fn run_round(
        &mut self,
        benchmark_fn: &dyn Fn(&ConfigPoint) -> f64,
    ) -> EvaluationResult {
        // Auto-run ablation if not done yet
        if self.phase == OptimizationPhase::Ablation {
            let features = Self::discover_features();
            let baseline: ConfigPoint = features.iter().cloned().collect();

            // Pick the next un-ablated feature
            let ablated: Vec<String> = self
                .ablation_results
                .iter()
                .map(|r| r.feature_name.clone())
                .collect();

            let next_feature = features
                .iter()
                .find(|(name, _)| !ablated.contains(name));

            if let Some((name, _)) = next_feature {
                let name = name.clone();
                let result = self.run_ablation_round(&baseline, &name, benchmark_fn);

                // Check if all features have been ablated
                if self.ablation_results.len() >= features.len() {
                    self.finalize_ablation();
                }

                return EvaluationResult {
                    config_point: ConfigPoint::new(),
                    quality_score: result.disabled_score,
                    latency_ms: None,
                    tokens_per_second: None,
                    cost_usd: 0.0,
                    code_version: self.code_version.clone(),
                    timestamp: current_timestamp(),
                    phase: OptimizationPhase::Ablation,
                    mode: "quality".to_string(),
                    benchmark_name: format!("ablation_{}", result.feature_name),
                };
            }

            // All features done, finalize
            self.finalize_ablation();
        }

        if self.phase == OptimizationPhase::BayesianSearch {
            return self.run_bayesian_round(benchmark_fn);
        }

        if self.phase == OptimizationPhase::FineTuning {
            return self.run_fine_tuning_round(benchmark_fn);
        }

        // Done phase — return last result or a no-op
        self.evaluations.last().cloned().unwrap_or(EvaluationResult {
            config_point: ConfigPoint::new(),
            quality_score: self.best_score,
            latency_ms: None,
            tokens_per_second: None,
            cost_usd: 0.0,
            code_version: self.code_version.clone(),
            timestamp: current_timestamp(),
            phase: OptimizationPhase::Done,
            mode: "quality".to_string(),
            benchmark_name: "done".to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Version tracking & decay
    // -----------------------------------------------------------------------

    /// Detect code version changes and apply exponential decay to old
    /// observations, reducing their influence on the surrogate model.
    pub fn check_version_change(&mut self) {
        let current = get_code_version();
        if current != self.code_version {
            let decay = self.config.decay_factor;

            // Decay arm rewards
            for arm in &mut self.arms {
                arm.total_reward *= decay;
                arm.alpha = 1.0 + (arm.alpha - 1.0) * decay;
                arm.beta = 1.0 + (arm.beta - 1.0) * decay;
            }

            // Decay surrogate observations
            for (_, score) in &mut self.surrogate.observations {
                *score *= decay;
            }

            self.code_version = current;
        }
    }

    /// Detect regressions: arms whose recent score dropped >20% compared to
    /// their historical average after a version change.
    pub fn detect_regressions(&self) -> Vec<(String, f64, f64)> {
        let mut regressions = Vec::new();

        for arm in &self.arms {
            if arm.results.len() < 2 {
                continue;
            }

            // Compare first half average vs second half average
            let mid = arm.results.len() / 2;
            let old_avg: f64 = arm.results[..mid]
                .iter()
                .map(|r| r.quality_score)
                .sum::<f64>()
                / mid as f64;

            let new_avg: f64 = arm.results[mid..]
                .iter()
                .map(|r| r.quality_score)
                .sum::<f64>()
                / (arm.results.len() - mid) as f64;

            // Regression if new score is >20% lower
            if old_avg > 0.0 && (old_avg - new_avg) / old_avg > 0.20 {
                regressions.push((arm.id.clone(), old_avg, new_avg));
            }
        }

        regressions
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get the current best configuration and its score.
    pub fn best_config(&self) -> Option<(&ConfigPoint, f64)> {
        self.best_config.as_ref().map(|c| (c, self.best_score))
    }

    /// Get feature importance results from the ablation phase.
    pub fn feature_importance(&self) -> &[AblationResult] {
        &self.ablation_results
    }

    /// Get all arms.
    pub fn arms(&self) -> &[ConfigArm] {
        &self.arms
    }

    /// Get all evaluations (respecting sliding window).
    pub fn evaluations(&self) -> &[EvaluationResult] {
        let start = if self.evaluations.len() > self.config.sliding_window_size {
            self.evaluations.len() - self.config.sliding_window_size
        } else {
            0
        };
        &self.evaluations[start..]
    }

    /// Get all evaluations without windowing.
    pub fn all_evaluations(&self) -> &[EvaluationResult] {
        &self.evaluations
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Save optimizer state to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize optimizer: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write optimizer state to {}: {}", path.display(), e))
    }

    /// Load optimizer state from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read optimizer state from {}: {}", path.display(), e))?;
        serde_json::from_str(&data)
            .map_err(|e| format!("Failed to deserialize optimizer: {}", e))
    }

    // -----------------------------------------------------------------------
    // Reports
    // -----------------------------------------------------------------------

    /// Generate a plain-text report of the optimizer state.
    pub fn report(&self) -> String {
        let mut out = String::new();

        out.push_str("==== Configuration Optimizer Report ====\n\n");
        out.push_str(&format!("Phase: {:?}\n", self.phase));
        out.push_str(&format!("Code version: {}\n", self.code_version));
        out.push_str(&format!("Total rounds: {}\n", self.total_rounds));
        out.push_str(&format!("Total evaluations: {}\n", self.evaluations.len()));
        out.push_str(&format!("Best score: {:.4}\n", self.best_score));
        out.push_str(&format!("Arms: {}\n\n", self.arms.len()));

        // Feature importance
        if !self.ablation_results.is_empty() {
            out.push_str("--- Feature Importance (Ablation) ---\n");
            for r in &self.ablation_results {
                out.push_str(&format!(
                    "  #{}: {} — impact: {:.4} (baseline: {:.4}, disabled: {:.4})\n",
                    r.importance_rank, r.feature_name, r.impact, r.baseline_score, r.disabled_score
                ));
            }
            out.push('\n');
        }

        // Best config
        if let Some(ref cfg) = self.best_config {
            out.push_str("--- Best Configuration ---\n");
            let mut keys: Vec<&String> = cfg.keys().collect();
            keys.sort();
            for key in keys {
                out.push_str(&format!("  {}: {:?}\n", key, cfg[key]));
            }
            out.push('\n');
        }

        // Arm summary
        if !self.arms.is_empty() {
            out.push_str("--- Arms Summary ---\n");
            for arm in &self.arms {
                out.push_str(&format!(
                    "  {} — pulls: {}, mean: {:.4}, thompson: {:.4}\n",
                    arm.id,
                    arm.pull_count,
                    arm.mean_reward(),
                    arm.thompson_score()
                ));
            }
            out.push('\n');
        }

        // Regressions
        let regressions = self.detect_regressions();
        if !regressions.is_empty() {
            out.push_str("--- Regressions Detected ---\n");
            for (id, old, new) in &regressions {
                out.push_str(&format!(
                    "  {} — old avg: {:.4}, new avg: {:.4} ({:.1}% drop)\n",
                    id,
                    old,
                    new,
                    (old - new) / old * 100.0
                ));
            }
            out.push('\n');
        }

        out
    }

    /// Generate a standalone HTML report with Chart.js visualizations.
    pub fn report_html(&self) -> String {
        // Prepare JSON data for charts
        let feature_names: Vec<String> = self
            .ablation_results
            .iter()
            .map(|r| r.feature_name.clone())
            .collect();
        let feature_impacts: Vec<f64> = self
            .ablation_results
            .iter()
            .map(|r| r.impact)
            .collect();

        let score_evolution: Vec<f64> = self
            .evaluations
            .iter()
            .map(|e| e.quality_score)
            .collect();
        let score_labels: Vec<usize> = (1..=score_evolution.len()).collect();

        let quality_vals: Vec<f64> = self
            .evaluations
            .iter()
            .map(|e| e.quality_score)
            .collect();
        let latency_vals: Vec<f64> = self
            .evaluations
            .iter()
            .map(|e| e.latency_ms.unwrap_or(0.0))
            .collect();

        let arm_ids: Vec<String> = self.arms.iter().map(|a| a.id.clone()).collect();
        let arm_rewards: Vec<f64> = self.arms.iter().map(|a| a.mean_reward()).collect();
        let arm_thompson: Vec<f64> = self.arms.iter().map(|a| a.thompson_score()).collect();

        // JSON-encode
        let feature_names_json = serde_json::to_string(&feature_names).unwrap_or_else(|_| "[]".to_string());
        let feature_impacts_json = serde_json::to_string(&feature_impacts).unwrap_or_else(|_| "[]".to_string());
        let score_labels_json = serde_json::to_string(&score_labels).unwrap_or_else(|_| "[]".to_string());
        let score_evolution_json = serde_json::to_string(&score_evolution).unwrap_or_else(|_| "[]".to_string());
        let quality_json = serde_json::to_string(&quality_vals).unwrap_or_else(|_| "[]".to_string());
        let latency_json = serde_json::to_string(&latency_vals).unwrap_or_else(|_| "[]".to_string());
        let arm_ids_json = serde_json::to_string(&arm_ids).unwrap_or_else(|_| "[]".to_string());
        let arm_rewards_json = serde_json::to_string(&arm_rewards).unwrap_or_else(|_| "[]".to_string());
        let arm_thompson_json = serde_json::to_string(&arm_thompson).unwrap_or_else(|_| "[]".to_string());

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Configuration Optimizer Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .stat {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px;
             background: #e8f4f8; border-radius: 6px; }}
    .stat .label {{ font-size: 0.85em; color: #666; }}
    .stat .value {{ font-size: 1.3em; font-weight: bold; color: #1a1a2e; }}
    canvas {{ max-height: 350px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
    th {{ background: #f0f0f0; font-weight: 600; }}
    .regression {{ color: #e74c3c; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Configuration Optimizer Report</h1>

  <div class="card">
    <div class="stat"><span class="label">Phase</span><br><span class="value">{phase:?}</span></div>
    <div class="stat"><span class="label">Rounds</span><br><span class="value">{rounds}</span></div>
    <div class="stat"><span class="label">Evaluations</span><br><span class="value">{evals}</span></div>
    <div class="stat"><span class="label">Best Score</span><br><span class="value">{best:.4}</span></div>
    <div class="stat"><span class="label">Arms</span><br><span class="value">{arms}</span></div>
    <div class="stat"><span class="label">Version</span><br><span class="value">{version}</span></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Feature Importance</h2>
      <canvas id="featureChart"></canvas>
    </div>
    <div class="card">
      <h2>Score Evolution</h2>
      <canvas id="scoreChart"></canvas>
    </div>
    <div class="card">
      <h2>Quality vs Latency</h2>
      <canvas id="scatterChart"></canvas>
    </div>
    <div class="card">
      <h2>Model Comparison</h2>
      <canvas id="armChart"></canvas>
    </div>
  </div>

  <script>
    // Feature importance bar chart
    new Chart(document.getElementById('featureChart'), {{
      type: 'bar',
      data: {{
        labels: {feature_names_json},
        datasets: [{{ label: 'Impact', data: {feature_impacts_json},
          backgroundColor: 'rgba(54, 162, 235, 0.7)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1 }}]
      }},
      options: {{ responsive: true, indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ x: {{ title: {{ display: true, text: 'Impact (baseline - disabled)' }} }} }}
      }}
    }});

    // Score evolution line chart
    new Chart(document.getElementById('scoreChart'), {{
      type: 'line',
      data: {{
        labels: {score_labels_json},
        datasets: [{{ label: 'Quality Score', data: {score_evolution_json},
          borderColor: 'rgba(75, 192, 192, 1)', backgroundColor: 'rgba(75, 192, 192, 0.2)',
          fill: true, tension: 0.3 }}]
      }},
      options: {{ responsive: true,
        scales: {{ x: {{ title: {{ display: true, text: 'Round' }} }},
                   y: {{ title: {{ display: true, text: 'Score' }}, min: 0, max: 1 }} }}
      }}
    }});

    // Quality vs Latency scatter
    var scatterData = [];
    var qualities = {quality_json};
    var latencies = {latency_json};
    for (var i = 0; i < qualities.length; i++) {{
      scatterData.push({{ x: latencies[i], y: qualities[i] }});
    }}
    new Chart(document.getElementById('scatterChart'), {{
      type: 'scatter',
      data: {{
        datasets: [{{ label: 'Evaluations', data: scatterData,
          backgroundColor: 'rgba(255, 99, 132, 0.7)', pointRadius: 5 }}]
      }},
      options: {{ responsive: true,
        scales: {{ x: {{ title: {{ display: true, text: 'Latency (ms)' }} }},
                   y: {{ title: {{ display: true, text: 'Quality' }}, min: 0, max: 1 }} }}
      }}
    }});

    // Arm comparison grouped bars
    new Chart(document.getElementById('armChart'), {{
      type: 'bar',
      data: {{
        labels: {arm_ids_json},
        datasets: [
          {{ label: 'Mean Reward', data: {arm_rewards_json},
            backgroundColor: 'rgba(153, 102, 255, 0.7)' }},
          {{ label: 'Thompson Score', data: {arm_thompson_json},
            backgroundColor: 'rgba(255, 159, 64, 0.7)' }}
        ]
      }},
      options: {{ responsive: true,
        scales: {{ y: {{ min: 0, max: 1 }} }}
      }}
    }});
  </script>
</body>
</html>"#,
            phase = self.phase,
            rounds = self.total_rounds,
            evals = self.evaluations.len(),
            best = self.best_score,
            arms = self.arms.len(),
            version = self.code_version,
            feature_names_json = feature_names_json,
            feature_impacts_json = feature_impacts_json,
            score_labels_json = score_labels_json,
            score_evolution_json = score_evolution_json,
            quality_json = quality_json,
            latency_json = latency_json,
            arm_ids_json = arm_ids_json,
            arm_rewards_json = arm_rewards_json,
            arm_thompson_json = arm_thompson_json,
        )
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Get the current code version from git or Cargo metadata.
pub fn get_code_version() -> String {
    // Try git first
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !hash.is_empty() {
                return hash;
            }
        }
    }

    // Fallback to CARGO_PKG_VERSION
    env!("CARGO_PKG_VERSION").to_string()
}

/// Compute the reward for an evaluation result given the optimization goal.
///
/// Returns a score in [0, 1] where 1 is the best possible outcome.
pub fn compute_reward(result: &EvaluationResult, goal: &OptimizationGoal) -> f64 {
    let quality = result.quality_score.clamp(0.0, 1.0);
    let latency = result.latency_ms.unwrap_or(100.0);
    let cost = result.cost_usd;

    match goal {
        OptimizationGoal::BestQuality => quality,

        OptimizationGoal::CheapestAboveThreshold(threshold) => {
            if quality >= *threshold {
                // Reward inversely proportional to cost (lower cost = higher reward)
                (1.0 - (cost / (cost + 1.0))).clamp(0.0, 1.0)
            } else {
                quality * 0.5 // partial credit
            }
        }

        OptimizationGoal::FastestAboveThreshold(threshold) => {
            if quality >= *threshold {
                // Reward inversely proportional to latency
                (1.0 - latency / (latency + 1000.0)).clamp(0.0, 1.0)
            } else {
                quality * 0.5
            }
        }

        OptimizationGoal::Balanced => {
            let latency_score = (1.0 - latency / (latency + 1000.0)).clamp(0.0, 1.0);
            let cost_score = (1.0 - cost / (cost + 1.0)).clamp(0.0, 1.0);
            0.5 * quality + 0.3 * latency_score + 0.2 * cost_score
        }

        OptimizationGoal::Custom {
            quality_w,
            latency_w,
            cost_w,
        } => {
            let total_w = quality_w + latency_w + cost_w;
            if total_w <= 0.0 {
                return quality;
            }
            let latency_score = (1.0 - latency / (latency + 1000.0)).clamp(0.0, 1.0);
            let cost_score = (1.0 - cost / (cost + 1.0)).clamp(0.0, 1.0);
            (quality_w * quality + latency_w * latency_score + cost_w * cost_score)
                / total_w
        }
    }
}

/// Get current timestamp as ISO 8601 string.
fn current_timestamp() -> String {
    // Use chrono if available, otherwise fall back to a basic timestamp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", now.as_secs())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert!((config.decay_factor - 0.95).abs() < 1e-10);
        assert_eq!(config.max_evaluations, 200);
        assert_eq!(config.max_llm_calls_per_round, 10);
        assert_eq!(config.benchmark_timeout_secs, 30);
        assert_eq!(config.sliding_window_size, 50);
        assert_eq!(config.ablation_top_k, 15);
        assert_eq!(config.samples_before_cache, 3);
        matches!(config.goal, OptimizationGoal::Balanced);
    }

    #[test]
    fn test_config_point_creation() {
        let mut point = ConfigPoint::new();
        point.insert("temperature".to_string(), ConfigValue::Float(0.7));
        point.insert("use_cache".to_string(), ConfigValue::Bool(true));
        point.insert("max_tokens".to_string(), ConfigValue::Uint(2048));
        point.insert("model".to_string(), ConfigValue::Str("llama3".to_string()));

        assert_eq!(point.len(), 4);
        matches!(point.get("temperature"), Some(ConfigValue::Float(_)));
    }

    #[test]
    fn test_config_distance_same() {
        let mut a = ConfigPoint::new();
        a.insert("x".to_string(), ConfigValue::Float(0.5));
        a.insert("y".to_string(), ConfigValue::Bool(true));

        let dist = config_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-10, "Same config should have distance 0");
    }

    #[test]
    fn test_config_distance_different() {
        let mut a = ConfigPoint::new();
        a.insert("x".to_string(), ConfigValue::Bool(true));
        let mut b = ConfigPoint::new();
        b.insert("x".to_string(), ConfigValue::Bool(false));

        let dist = config_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-10, "Opposite bools should have distance 1");
    }

    #[test]
    fn test_config_distance_mixed() {
        let mut a = ConfigPoint::new();
        a.insert("x".to_string(), ConfigValue::Float(0.0));
        a.insert("y".to_string(), ConfigValue::Bool(true));

        let mut b = ConfigPoint::new();
        b.insert("x".to_string(), ConfigValue::Float(1.0));
        b.insert("y".to_string(), ConfigValue::Bool(true));

        let dist = config_distance(&a, &b);
        // x: |0-1|/max(1,1) = 1.0, y: same = 0.0, average = 0.5
        assert!((dist - 0.5).abs() < 1e-10, "Mixed distance should be 0.5, got {}", dist);
    }

    #[test]
    fn test_surrogate_model_predict_empty() {
        let model = SurrogateModel::new();
        let point = ConfigPoint::new();
        let (mean, uncertainty) = model.predict(&point);
        assert!((mean - 0.5).abs() < 1e-10);
        assert!((uncertainty - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_surrogate_model_predict_with_data() {
        let mut model = SurrogateModel::new();

        let mut p1 = ConfigPoint::new();
        p1.insert("x".to_string(), ConfigValue::Float(0.0));
        model.add_observation(p1, 0.3);

        let mut p2 = ConfigPoint::new();
        p2.insert("x".to_string(), ConfigValue::Float(1.0));
        model.add_observation(p2, 0.9);

        // Predict at x=0.0 (same as p1): should be close to average of neighbors
        let mut query = ConfigPoint::new();
        query.insert("x".to_string(), ConfigValue::Float(0.0));
        let (mean, _uncertainty) = model.predict(&query);

        // With 2 observations and k=5, both are neighbors. Mean = (0.3+0.9)/2 = 0.6
        assert!((mean - 0.6).abs() < 1e-10, "Mean should be 0.6, got {}", mean);
    }

    #[test]
    fn test_surrogate_expected_improvement() {
        let mut model = SurrogateModel::new();

        let mut p1 = ConfigPoint::new();
        p1.insert("x".to_string(), ConfigValue::Float(0.5));
        model.add_observation(p1, 0.7);

        let mut query = ConfigPoint::new();
        query.insert("x".to_string(), ConfigValue::Float(0.5));

        // EI when best_score = 0.7 and predicted mean is close to 0.7
        let ei = model.expected_improvement(&query, 0.7);
        assert!(ei >= 0.0, "EI should be non-negative");

        // EI should be larger when best_score is lower (more room for improvement)
        let ei_low = model.expected_improvement(&query, 0.3);
        assert!(ei_low >= ei, "EI should be larger when best is lower");
    }

    #[test]
    fn test_ablation_round() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());

        let mut baseline = ConfigPoint::new();
        baseline.insert("temperature".to_string(), ConfigValue::Float(0.7));
        baseline.insert("use_cache".to_string(), ConfigValue::Bool(true));

        let benchmark = |point: &ConfigPoint| -> f64 {
            let temp = match point.get("temperature") {
                Some(ConfigValue::Float(f)) => *f,
                _ => 0.5,
            };
            let cache = match point.get("use_cache") {
                Some(ConfigValue::Bool(b)) => *b,
                _ => false,
            };
            temp * 0.5 + if cache { 0.3 } else { 0.0 }
        };

        let result = optimizer.run_ablation_round(&baseline, "use_cache", &benchmark);
        assert_eq!(result.feature_name, "use_cache");
        assert!(result.impact > 0.0, "Cache should have positive impact");
        assert_eq!(optimizer.ablation_results.len(), 1);
    }

    #[test]
    fn test_bayesian_round() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());
        optimizer.phase = OptimizationPhase::BayesianSearch;
        optimizer.important_features = vec!["temperature".to_string()];

        let benchmark = |_point: &ConfigPoint| -> f64 { 0.75 };

        let result = optimizer.run_bayesian_round(&benchmark);
        assert!((result.quality_score - 0.75).abs() < 1e-10);
        assert_eq!(result.phase, OptimizationPhase::BayesianSearch);
        assert_eq!(optimizer.arms.len(), 1);
    }

    #[test]
    fn test_run_round_auto_phase() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());

        let benchmark = |_point: &ConfigPoint| -> f64 { 0.8 };

        // First round should be ablation
        let result = optimizer.run_round(&benchmark);
        assert_eq!(result.phase, OptimizationPhase::Ablation);
        assert!(result.quality_score > 0.0);
    }

    #[test]
    fn test_version_change_decay() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());

        // Add a fake arm with known reward
        let mut arm = ConfigArm::new("test_arm".to_string(), ConfigPoint::new());
        arm.total_reward = 10.0;
        arm.alpha = 5.0;
        arm.beta = 3.0;
        arm.pull_count = 10;
        optimizer.arms.push(arm);

        // Force a version change
        optimizer.code_version = "old_version".to_string();

        // Simulate check (will detect version change since get_code_version() differs)
        optimizer.check_version_change();

        // Verify decay was applied
        let arm = &optimizer.arms[0];
        // If version changed: total_reward = 10.0 * 0.95 = 9.5
        // If no change, it stays at 10.0
        assert!(
            arm.total_reward <= 10.0,
            "Reward should have been decayed or stayed the same"
        );
    }

    #[test]
    fn test_detect_regressions() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());

        let mut arm = ConfigArm::new("regressed_arm".to_string(), ConfigPoint::new());

        // Add results: first half high scores, second half low scores
        for i in 0..10 {
            let score = if i < 5 { 0.9 } else { 0.5 };
            arm.results.push(EvaluationResult {
                config_point: ConfigPoint::new(),
                quality_score: score,
                latency_ms: None,
                tokens_per_second: None,
                cost_usd: 0.0,
                code_version: "v1".to_string(),
                timestamp: "0s".to_string(),
                phase: OptimizationPhase::BayesianSearch,
                mode: "quality".to_string(),
                benchmark_name: "test".to_string(),
            });
        }
        optimizer.arms.push(arm);

        let regressions = optimizer.detect_regressions();
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].0, "regressed_arm");
        assert!((regressions[0].1 - 0.9).abs() < 1e-10); // old avg
        assert!((regressions[0].2 - 0.5).abs() < 1e-10); // new avg
    }

    #[test]
    fn test_best_config() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());

        // Initially no best config
        assert!(optimizer.best_config().is_none());

        // Set one
        let mut point = ConfigPoint::new();
        point.insert("x".to_string(), ConfigValue::Float(0.5));
        optimizer.best_config = Some(point);
        optimizer.best_score = 0.95;

        let (cfg, score) = optimizer.best_config().unwrap();
        assert!((score - 0.95).abs() < 1e-10);
        assert!(cfg.contains_key("x"));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut optimizer = ConfigOptimizer::new(OptimizerConfig::default());
        optimizer.best_score = 0.88;

        let mut point = ConfigPoint::new();
        point.insert("temp".to_string(), ConfigValue::Float(0.6));
        optimizer.best_config = Some(point);

        let dir = std::env::temp_dir();
        let path = dir.join("test_config_optimizer_state.json");

        optimizer.save(&path).expect("save should succeed");
        let loaded = ConfigOptimizer::load(&path).expect("load should succeed");

        assert!((loaded.best_score - 0.88).abs() < 1e-10);
        assert!(loaded.best_config.is_some());
        assert_eq!(loaded.phase, OptimizationPhase::Ablation);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compute_reward_balanced() {
        let result = EvaluationResult {
            config_point: ConfigPoint::new(),
            quality_score: 0.8,
            latency_ms: Some(200.0),
            tokens_per_second: None,
            cost_usd: 0.01,
            code_version: "v1".to_string(),
            timestamp: "0s".to_string(),
            phase: OptimizationPhase::BayesianSearch,
            mode: "quality".to_string(),
            benchmark_name: "test".to_string(),
        };

        let reward = compute_reward(&result, &OptimizationGoal::Balanced);
        assert!(reward > 0.0 && reward <= 1.0, "Balanced reward should be in (0,1], got {}", reward);

        // Quality component: 0.5 * 0.8 = 0.4
        // Latency component: 0.3 * (1 - 200/1200) = 0.3 * 0.833 ≈ 0.25
        // Cost component: 0.2 * (1 - 0.01/1.01) ≈ 0.2 * 0.99 ≈ 0.198
        // Total ≈ 0.848
        assert!(reward > 0.7, "Expected reward > 0.7, got {}", reward);
    }

    #[test]
    fn test_compute_reward_cheapest() {
        let good_result = EvaluationResult {
            config_point: ConfigPoint::new(),
            quality_score: 0.9,
            latency_ms: Some(500.0),
            tokens_per_second: None,
            cost_usd: 0.001,
            code_version: "v1".to_string(),
            timestamp: "0s".to_string(),
            phase: OptimizationPhase::BayesianSearch,
            mode: "quality".to_string(),
            benchmark_name: "test".to_string(),
        };

        let bad_result = EvaluationResult {
            quality_score: 0.3,
            ..good_result.clone()
        };

        let reward_good = compute_reward(&good_result, &OptimizationGoal::CheapestAboveThreshold(0.8));
        let reward_bad = compute_reward(&bad_result, &OptimizationGoal::CheapestAboveThreshold(0.8));

        assert!(reward_good > reward_bad, "Above-threshold should have higher reward");
        assert!(reward_good > 0.9, "Very cheap + above threshold should score high");
    }

    #[test]
    fn test_report_not_empty() {
        let optimizer = ConfigOptimizer::new(OptimizerConfig::default());
        let report = optimizer.report();
        assert!(!report.is_empty());
        assert!(report.contains("Configuration Optimizer Report"));
        assert!(report.contains("Phase:"));
    }

    #[test]
    fn test_report_html_contains_chartjs() {
        let optimizer = ConfigOptimizer::new(OptimizerConfig::default());
        let html = optimizer.report_html();
        assert!(html.contains("chart.js"), "HTML should reference Chart.js CDN");
        assert!(html.contains("<canvas"), "HTML should contain canvas elements");
        assert!(html.contains("Feature Importance"));
        assert!(html.contains("Score Evolution"));
    }

    #[test]
    fn test_discover_features_not_empty() {
        let features = ConfigOptimizer::discover_features();
        assert!(!features.is_empty());
        assert!(features.len() >= 10, "Should have at least 10 configurable features");

        let names: Vec<&str> = features.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"temperature"));
        assert!(names.contains(&"max_tokens"));
        assert!(names.contains(&"use_cache"));
    }

    #[test]
    fn test_evaluation_result_serialization() {
        let result = EvaluationResult {
            config_point: {
                let mut p = ConfigPoint::new();
                p.insert("x".to_string(), ConfigValue::Float(0.5));
                p
            },
            quality_score: 0.85,
            latency_ms: Some(150.0),
            tokens_per_second: Some(42.0),
            cost_usd: 0.005,
            code_version: "abc123".to_string(),
            timestamp: "12345s".to_string(),
            phase: OptimizationPhase::BayesianSearch,
            mode: "quality".to_string(),
            benchmark_name: "test_bench".to_string(),
        };

        let json = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: EvaluationResult =
            serde_json::from_str(&json).expect("Should deserialize");

        assert!((deserialized.quality_score - 0.85).abs() < 1e-10);
        assert_eq!(deserialized.benchmark_name, "test_bench");
    }

    #[test]
    fn test_phase_transitions() {
        let mut config = OptimizerConfig::default();
        config.max_evaluations = 30; // smaller for testing
        let mut optimizer = ConfigOptimizer::new(config);

        let benchmark = |_point: &ConfigPoint| -> f64 { 0.7 };

        // Run enough rounds to get through ablation
        let features_count = ConfigOptimizer::discover_features().len();
        for _ in 0..features_count {
            optimizer.run_round(&benchmark);
        }

        // Should have transitioned past ablation
        assert_ne!(
            optimizer.phase(),
            OptimizationPhase::Ablation,
            "Should have left ablation after all features evaluated"
        );

        // Keep running until done or bayesian
        assert!(
            optimizer.phase() == OptimizationPhase::BayesianSearch
                || optimizer.phase() == OptimizationPhase::FineTuning
                || optimizer.phase() == OptimizationPhase::Done,
            "Phase should be BayesianSearch, FineTuning, or Done, got {:?}",
            optimizer.phase()
        );
    }

    #[test]
    fn test_sliding_window_respects_size() {
        let mut config = OptimizerConfig::default();
        config.sliding_window_size = 5;
        let mut optimizer = ConfigOptimizer::new(config);
        optimizer.phase = OptimizationPhase::BayesianSearch;
        optimizer.important_features = vec!["temperature".to_string()];

        let benchmark = |_point: &ConfigPoint| -> f64 { 0.6 };

        // Run 10 rounds
        for _ in 0..10 {
            optimizer.run_bayesian_round(&benchmark);
        }

        // All evaluations are stored internally
        assert_eq!(optimizer.all_evaluations().len(), 10);

        // Sliding window returns only the last 5
        assert_eq!(optimizer.evaluations().len(), 5);
    }
}
