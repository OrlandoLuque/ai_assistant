//! Multi-model comparison matrix with statistical significance testing.
//!
//! Compares multiple model benchmark runs side-by-side, computing accuracy, cost,
//! ELO ratings, and pairwise statistical significance via Welch's t-test.

use super::runner::{BenchmarkRunResult, ModelIdentifier};
use super::scoring::EloCalculator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for multi-model comparison.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComparisonConfig {
    /// Confidence level for significance testing (default 0.95)
    pub confidence_level: f64,
    /// K-factor for ELO calculation (default 32.0)
    pub elo_k_factor: f64,
    /// Optional metric weights for overall ranking
    pub metric_weights: HashMap<String, f64>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            elo_k_factor: 32.0,
            metric_weights: HashMap::new(),
        }
    }
}

/// A comparison matrix: model × metric with significance and rankings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMatrix {
    /// Models being compared
    pub models: Vec<ModelIdentifier>,
    /// Metric names (e.g., "accuracy", "mean_score", "mean_latency_ms", "total_cost")
    pub metrics: Vec<String>,
    /// Scores indexed as `scores[model_idx][metric_idx]`
    pub scores: Vec<Vec<f64>>,
    /// Pairwise significance p-values: `significance[i][j]` = p-value of model i vs model j
    pub significance: Vec<Vec<f64>>,
    /// ELO ratings per model (key = model display string)
    pub elo_ratings: HashMap<String, f64>,
    /// Total cost per model
    pub costs: Vec<f64>,
    /// Cost-effectiveness per model (score / cost, higher = better)
    pub cost_effectiveness: Vec<f64>,
}

impl ComparisonMatrix {
    /// Build a comparison matrix from multiple benchmark run results.
    ///
    /// Each `BenchmarkRunResult` should be from a different model on the same dataset.
    pub fn from_runs(runs: &[BenchmarkRunResult], config: &ComparisonConfig) -> Self {
        let models: Vec<ModelIdentifier> = runs.iter().map(|r| r.model_id.clone()).collect();
        let metrics = vec![
            "accuracy".to_string(),
            "mean_score".to_string(),
            "mean_latency_ms".to_string(),
            "total_cost".to_string(),
        ];

        let mut scores = Vec::new();
        let mut costs = Vec::new();

        for run in runs {
            let acc = run.accuracy();
            let ms = run.mean_score();
            let lat = run.mean_latency_ms();
            let cost = run.total_cost;
            scores.push(vec![acc, ms, lat, cost]);
            costs.push(cost);
        }

        // Cost-effectiveness: accuracy / cost (avoid division by zero)
        let cost_effectiveness: Vec<f64> = scores
            .iter()
            .zip(costs.iter())
            .map(|(s, &c)| {
                if c > 0.0 {
                    s[0] / c // accuracy / cost
                } else {
                    s[0] * 1e6 // Free model → very cost-effective
                }
            })
            .collect();

        // Pairwise significance (accuracy scores per problem)
        let mut significance = vec![vec![1.0; models.len()]; models.len()];
        for i in 0..runs.len() {
            for j in (i + 1)..runs.len() {
                let scores_i: Vec<f64> = runs[i].results.iter().map(|r| {
                    if r.scores.is_empty() { 0.0 } else { r.scores.iter().sum::<f64>() / r.scores.len() as f64 }
                }).collect();
                let scores_j: Vec<f64> = runs[j].results.iter().map(|r| {
                    if r.scores.is_empty() { 0.0 } else { r.scores.iter().sum::<f64>() / r.scores.len() as f64 }
                }).collect();

                let (_, p_value) = welch_t_test(&scores_i, &scores_j);
                significance[i][j] = p_value;
                significance[j][i] = p_value;
            }
        }

        // ELO ratings
        let mut elo = EloCalculator::new(config.elo_k_factor);
        for i in 0..runs.len() {
            for j in (i + 1)..runs.len() {
                elo.update_from_pairwise(
                    &models[i],
                    &models[j],
                    scores[i][0], // accuracy
                    scores[j][0],
                );
            }
        }
        let elo_ratings: HashMap<String, f64> = elo
            .ratings()
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        ComparisonMatrix {
            models,
            metrics,
            scores,
            significance,
            elo_ratings,
            costs,
            cost_effectiveness,
        }
    }

    /// Get the best model for each metric.
    pub fn best_per_metric(&self) -> HashMap<String, ModelIdentifier> {
        let mut result = HashMap::new();
        for (mi, metric) in self.metrics.iter().enumerate() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (model_idx, model_scores) in self.scores.iter().enumerate() {
                let score = model_scores[mi];
                // For latency and cost, lower is better
                let effective = if metric == "mean_latency_ms" || metric == "total_cost" {
                    -score
                } else {
                    score
                };
                if effective > best_score {
                    best_score = effective;
                    best_idx = model_idx;
                }
            }

            result.insert(metric.clone(), self.models[best_idx].clone());
        }
        result
    }

    /// Get overall model ranking with optional metric weights.
    ///
    /// Default weighting: accuracy=1.0, mean_score=0.5, latency=-0.1 (lower is better),
    /// cost=-0.2 (lower is better).
    pub fn overall_ranking(&self, weights: &HashMap<String, f64>) -> Vec<(ModelIdentifier, f64)> {
        let default_weights: HashMap<String, f64> = [
            ("accuracy".to_string(), 1.0),
            ("mean_score".to_string(), 0.5),
            ("mean_latency_ms".to_string(), -0.1),
            ("total_cost".to_string(), -0.2),
        ]
        .into_iter()
        .collect();

        let w = if weights.is_empty() { &default_weights } else { weights };

        let mut ranking: Vec<(ModelIdentifier, f64)> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let mut weighted_sum = 0.0;
                for (mi, metric) in self.metrics.iter().enumerate() {
                    let weight = w.get(metric).unwrap_or(&0.0);
                    // Normalize: for each metric, scale to 0-1 range
                    let values: Vec<f64> = self.scores.iter().map(|s| s[mi]).collect();
                    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max_val - min_val;
                    let normalized = if range > 0.0 {
                        (self.scores[i][mi] - min_val) / range
                    } else {
                        0.5
                    };
                    weighted_sum += weight * normalized;
                }
                (model.clone(), weighted_sum)
            })
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranking
    }

    /// Export to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Welch's t-test (simplified inline, matching ab_testing::SignificanceCalculator)
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

    let var_a = group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    if se == 0.0 {
        return (0.0, 1.0);
    }

    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom > 0.0 { num / denom } else { 1.0 };

    // Approximate p-value using normal distribution for large df
    let p = if df > 30.0 {
        2.0 * (1.0 - normal_cdf(t.abs()))
    } else {
        2.0 * (1.0 - students_t_cdf(t.abs(), df))
    };

    (t, p.max(0.0).min(1.0))
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
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

/// Student's t CDF approximation using the incomplete beta function.
fn students_t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let ib = incomplete_beta(df / 2.0, 0.5, x);
    1.0 - 0.5 * ib
}

/// Regularized incomplete beta function approximation (continued fraction).
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;

    // Lentz's continued fraction
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..200 {
        let m_f = m as f64;

        // Even step
        let num = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let num = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    front * f
}

/// Log-gamma function (Lanczos approximation).
fn ln_gamma(x: f64) -> f64 {
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];

    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x - 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for (i, &c) in coefficients.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::runner::{ProblemResult, TokenUsage};
    use super::super::dataset::BenchmarkSuiteType;

    fn make_run(model_name: &str, accuracies: &[f64], cost: f64) -> BenchmarkRunResult {
        let model = ModelIdentifier { name: model_name.into(), provider: "test".into(), variant: None };
        let results: Vec<ProblemResult> = accuracies.iter().enumerate().map(|(i, &acc)| {
            ProblemResult {
                problem_id: format!("p/{}", i),
                model_id: model.clone(),
                responses: vec!["resp".into()],
                scores: vec![acc],
                passed: vec![acc >= 0.99],
                latencies_ms: vec![100],
                token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 20 }],
                cost_estimates: vec![cost / accuracies.len() as f64],
                error: None,
                metadata: HashMap::new(),
            }
        }).collect();

        BenchmarkRunResult {
            run_id: format!("run_{}", model_name),
            model_id: model,
            dataset_name: "test".into(),
            suite_type: BenchmarkSuiteType::Mmlu,
            results,
            started_at: 1000,
            completed_at: 1010,
            total_cost: cost,
            total_tokens: TokenUsage { input_tokens: 500, output_tokens: 200 },
        }
    }

    #[test]
    fn test_comparison_matrix_from_runs() {
        let runs = vec![
            make_run("gpt-4", &[1.0, 1.0, 1.0, 0.0, 1.0], 0.05),
            make_run("llama3", &[1.0, 0.0, 1.0, 0.0, 0.0], 0.001),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        assert_eq!(matrix.models.len(), 2);
        assert_eq!(matrix.metrics.len(), 4);
        assert!(matrix.scores[0][0] > matrix.scores[1][0]); // gpt-4 higher accuracy
    }

    #[test]
    fn test_significance_computation() {
        // Groups need variance for t-test to work
        let runs = vec![
            make_run("good", &[0.9, 1.0, 0.8, 1.0, 0.9, 1.0, 0.8, 1.0, 0.9, 1.0], 0.1),
            make_run("bad", &[0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0], 0.1),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        // p-value should be very small (significant difference)
        assert!(matrix.significance[0][1] < 0.05);
    }

    #[test]
    fn test_elo_ratings() {
        let runs = vec![
            make_run("best", &[1.0, 1.0, 1.0, 1.0, 1.0], 0.1),
            make_run("worst", &[0.0, 0.0, 0.0, 0.0, 0.0], 0.1),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        let best_elo = matrix.elo_ratings.get("test/best").unwrap_or(&0.0);
        let worst_elo = matrix.elo_ratings.get("test/worst").unwrap_or(&0.0);
        assert!(best_elo > worst_elo);
    }

    #[test]
    fn test_best_per_metric() {
        let runs = vec![
            make_run("accurate", &[1.0, 1.0, 1.0, 1.0, 1.0], 0.1),
            make_run("cheap", &[0.5, 0.5, 0.5, 0.5, 0.5], 0.001),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());
        let best = matrix.best_per_metric();

        assert_eq!(best["accuracy"].name, "accurate");
        assert_eq!(best["total_cost"].name, "cheap"); // Lower cost is better
    }

    #[test]
    fn test_overall_ranking() {
        let runs = vec![
            make_run("model_a", &[0.8, 0.8, 0.8, 0.8, 0.8], 0.05),
            make_run("model_b", &[0.6, 0.6, 0.6, 0.6, 0.6], 0.01),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());
        let ranking = matrix.overall_ranking(&HashMap::new());

        assert!(!ranking.is_empty());
        // Model A should rank higher (accuracy weight dominates)
        assert_eq!(ranking[0].0.name, "model_a");
    }

    #[test]
    fn test_cost_effectiveness() {
        let runs = vec![
            make_run("expensive", &[1.0, 1.0, 1.0, 1.0, 1.0], 10.0),
            make_run("cheap", &[1.0, 1.0, 1.0, 0.0, 0.0], 0.001),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        // Cheap: accuracy=0.6, cost=0.001, effectiveness=600
        // Expensive: accuracy=1.0, cost=10.0, effectiveness=0.1
        assert!(matrix.cost_effectiveness[1] > matrix.cost_effectiveness[0]);
    }

    #[test]
    fn test_comparison_with_single_model() {
        let runs = vec![make_run("only", &[0.8, 0.7, 0.9], 0.02)];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        assert_eq!(matrix.models.len(), 1);
        assert_eq!(matrix.significance.len(), 1);
    }

    #[test]
    fn test_comparison_json_export() {
        let runs = vec![
            make_run("a", &[1.0, 0.5], 0.01),
            make_run("b", &[0.5, 1.0], 0.02),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());
        let json = matrix.to_json();
        assert!(json.contains("accuracy"));
        assert!(json.contains("models"));
    }

    #[test]
    fn test_comparison_with_weights() {
        let runs = vec![
            make_run("fast", &[0.5, 0.5, 0.5], 0.1),
            make_run("accurate", &[1.0, 1.0, 1.0], 0.5),
        ];
        let matrix = ComparisonMatrix::from_runs(&runs, &ComparisonConfig::default());

        // With only accuracy weight
        let mut weights = HashMap::new();
        weights.insert("accuracy".to_string(), 1.0);
        let ranking = matrix.overall_ranking(&weights);
        assert_eq!(ranking[0].0.name, "accurate");
    }

    #[test]
    fn test_welch_t_test_basic() {
        // Need variance within groups for t-test to produce meaningful results
        let a = vec![9.0, 10.0, 11.0, 10.0, 10.0];
        let b = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(t > 0.0);
        assert!(p < 0.01); // Should be highly significant
    }

    #[test]
    fn test_welch_t_test_equal_groups() {
        let a = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let b = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (t, p) = welch_t_test(&a, &b);
        assert_eq!(t, 0.0);
        assert!((p - 1.0).abs() < 0.01); // Not significant at all
    }
}
