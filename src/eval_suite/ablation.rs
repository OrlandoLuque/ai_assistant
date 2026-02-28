//! Ablation studies for measuring technique/concept impact.
//!
//! Compares a control run (technique disabled) against a treatment run (technique enabled)
//! to quantify quality delta, latency delta, cost delta, and statistical significance.

use super::runner::BenchmarkRunResult;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for an ablation study: control vs treatment.
#[derive(Debug, Clone)]
pub struct AblationStudy {
    /// Name of the study (e.g., "RAG multi-level graph impact")
    pub name: String,
    /// Description of what is being tested
    pub description: String,
    /// The technique being tested (e.g., "multi_level_graph_rag")
    pub technique: String,
    /// Control run (technique disabled)
    pub control: BenchmarkRunResult,
    /// Treatment run (technique enabled)
    pub treatment: BenchmarkRunResult,
}

/// Summary statistics for one arm of an ablation study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    /// Mean score across all problems
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_dev: f64,
    /// Mean latency in milliseconds
    pub mean_latency_ms: f64,
    /// Total cost in USD
    pub total_cost: f64,
    /// Number of samples
    pub sample_count: usize,
}

/// Recommendation from an ablation study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AblationRecommendation {
    /// Technique significantly improves quality
    Enable {
        /// Quality improvement percentage
        quality_gain_pct: f64,
        /// Cost increase percentage
        cost_increase_pct: f64,
    },
    /// Technique has no significant effect
    Neutral,
    /// Technique hurts quality
    Disable {
        /// Quality degradation percentage
        quality_loss_pct: f64,
    },
    /// Not enough data to conclude
    InsufficientData,
}

/// Result of an ablation study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    /// Study name
    pub study_name: String,
    /// Technique tested
    pub technique: String,
    /// Quality delta (treatment - control, positive = improvement)
    pub quality_delta: f64,
    /// Latency delta in ms (treatment - control, positive = slower)
    pub latency_delta_ms: f64,
    /// Cost delta in USD (treatment - control, positive = more expensive)
    pub cost_delta: f64,
    /// p-value from Welch's t-test
    pub p_value: f64,
    /// Whether the result is statistically significant
    pub is_significant: bool,
    /// Cohen's d effect size
    pub effect_size: f64,
    /// Recommendation based on results
    pub recommendation: AblationRecommendation,
    /// Control arm summary
    pub control_summary: RunSummary,
    /// Treatment arm summary
    pub treatment_summary: RunSummary,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Engine for running ablation studies.
pub struct AblationEngine;

impl AblationEngine {
    /// Analyze a single ablation study.
    pub fn analyze(study: &AblationStudy, confidence: f64) -> AblationResult {
        let control_scores = Self::extract_scores(&study.control);
        let treatment_scores = Self::extract_scores(&study.treatment);

        let control_summary = Self::summarize(&study.control, &control_scores);
        let treatment_summary = Self::summarize(&study.treatment, &treatment_scores);

        let quality_delta = treatment_summary.mean_score - control_summary.mean_score;
        let latency_delta = treatment_summary.mean_latency_ms - control_summary.mean_latency_ms;
        let cost_delta = treatment_summary.total_cost - control_summary.total_cost;

        // Welch's t-test
        let (_, p_value) = welch_t_test(&control_scores, &treatment_scores);
        let alpha = 1.0 - confidence;
        let is_significant = p_value < alpha;

        // Cohen's d
        let effect_size = cohens_d(&control_scores, &treatment_scores);

        // Generate recommendation
        let recommendation = if control_scores.len() < 5 || treatment_scores.len() < 5 {
            AblationRecommendation::InsufficientData
        } else if !is_significant {
            AblationRecommendation::Neutral
        } else if quality_delta > 0.0 {
            let quality_gain_pct = if control_summary.mean_score > 0.0 {
                (quality_delta / control_summary.mean_score) * 100.0
            } else {
                100.0
            };
            let cost_increase_pct = if control_summary.total_cost > 0.0 {
                (cost_delta / control_summary.total_cost) * 100.0
            } else {
                0.0
            };
            AblationRecommendation::Enable {
                quality_gain_pct,
                cost_increase_pct,
            }
        } else {
            let quality_loss_pct = if control_summary.mean_score > 0.0 {
                (quality_delta.abs() / control_summary.mean_score) * 100.0
            } else {
                0.0
            };
            AblationRecommendation::Disable { quality_loss_pct }
        };

        AblationResult {
            study_name: study.name.clone(),
            technique: study.technique.clone(),
            quality_delta,
            latency_delta_ms: latency_delta,
            cost_delta,
            p_value,
            is_significant,
            effect_size,
            recommendation,
            control_summary,
            treatment_summary,
        }
    }

    /// Run multiple ablation studies in batch.
    pub fn analyze_batch(studies: &[AblationStudy], confidence: f64) -> Vec<AblationResult> {
        studies.iter().map(|s| Self::analyze(s, confidence)).collect()
    }

    /// Extract per-problem mean scores from a run.
    fn extract_scores(run: &BenchmarkRunResult) -> Vec<f64> {
        run.results
            .iter()
            .map(|r| {
                if r.scores.is_empty() {
                    0.0
                } else {
                    r.scores.iter().sum::<f64>() / r.scores.len() as f64
                }
            })
            .collect()
    }

    /// Summarize a benchmark run.
    fn summarize(run: &BenchmarkRunResult, scores: &[f64]) -> RunSummary {
        let n = scores.len();
        let mean = if n > 0 {
            scores.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let std_dev = if n > 1 {
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        RunSummary {
            mean_score: mean,
            std_dev,
            mean_latency_ms: run.mean_latency_ms(),
            total_cost: run.total_cost,
            sample_count: n,
        }
    }
}

/// Cohen's d effect size.
fn cohens_d(group_a: &[f64], group_b: &[f64]) -> f64 {
    if group_a.len() < 2 || group_b.len() < 2 {
        return 0.0;
    }

    let n_a = group_a.len() as f64;
    let n_b = group_b.len() as f64;
    let mean_a = group_a.iter().sum::<f64>() / n_a;
    let mean_b = group_b.iter().sum::<f64>() / n_b;

    let var_a = group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    // Pooled standard deviation
    let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd == 0.0 {
        return 0.0;
    }

    (mean_a - mean_b).abs() / pooled_sd
}

/// Welch's t-test (inline copy to avoid cross-module dependency).
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

    // Approximate p-value using normal distribution for simplicity
    let p = 2.0 * (1.0 - normal_cdf(t.abs()));
    (t, p.max(0.0).min(1.0))
}

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
    use super::super::runner::{ModelIdentifier, ProblemResult, TokenUsage};
    use super::super::dataset::BenchmarkSuiteType;
    use std::collections::HashMap;

    fn make_run(scores: &[f64], cost: f64) -> BenchmarkRunResult {
        let model = ModelIdentifier { name: "test".into(), provider: "test".into(), variant: None };
        let results: Vec<ProblemResult> = scores.iter().enumerate().map(|(i, &s)| {
            ProblemResult {
                problem_id: format!("p/{}", i),
                model_id: model.clone(),
                responses: vec!["resp".into()],
                scores: vec![s],
                passed: vec![s >= 0.99],
                latencies_ms: vec![100],
                token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 20 }],
                cost_estimates: vec![cost / scores.len() as f64],
                error: None,
                metadata: HashMap::new(),
            }
        }).collect();

        BenchmarkRunResult {
            run_id: "run".into(),
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
    fn test_ablation_significant_improvement() {
        let study = AblationStudy {
            name: "RAG test".into(),
            description: "Test RAG impact".into(),
            technique: "multi_level_rag".into(),
            control: make_run(&[0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4], 0.01),
            treatment: make_run(&[0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8], 0.02),
        };
        let result = AblationEngine::analyze(&study, 0.95);

        assert!(result.quality_delta > 0.0);
        assert!(result.is_significant);
        assert!(result.effect_size > 0.5); // Large effect
        assert!(matches!(result.recommendation, AblationRecommendation::Enable { .. }));
    }

    #[test]
    fn test_ablation_no_effect() {
        let study = AblationStudy {
            name: "NoOp test".into(),
            description: "Test no-op".into(),
            technique: "noop".into(),
            control: make_run(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.01),
            treatment: make_run(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.01),
        };
        let result = AblationEngine::analyze(&study, 0.95);

        assert!((result.quality_delta).abs() < 0.01);
        assert!(!result.is_significant);
        assert!(matches!(result.recommendation, AblationRecommendation::Neutral));
    }

    #[test]
    fn test_ablation_degradation() {
        let study = AblationStudy {
            name: "Bad technique".into(),
            description: "Technique that hurts".into(),
            technique: "bad_technique".into(),
            control: make_run(&[0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8], 0.01),
            treatment: make_run(&[0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4], 0.01),
        };
        let result = AblationEngine::analyze(&study, 0.95);

        assert!(result.quality_delta < 0.0);
        assert!(result.is_significant);
        assert!(matches!(result.recommendation, AblationRecommendation::Disable { .. }));
    }

    #[test]
    fn test_ablation_insufficient_data() {
        let study = AblationStudy {
            name: "Small test".into(),
            description: "Too few samples".into(),
            technique: "small".into(),
            control: make_run(&[0.5, 0.5], 0.01),
            treatment: make_run(&[0.8, 0.8], 0.01),
        };
        let result = AblationEngine::analyze(&study, 0.95);

        assert!(matches!(result.recommendation, AblationRecommendation::InsufficientData));
    }

    #[test]
    fn test_cohens_d_calculation() {
        // Groups with variance
        let a = vec![9.0, 10.0, 11.0, 10.0, 10.0];
        let b = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let d = cohens_d(&a, &b);
        assert!(d > 2.0); // Very large effect
    }

    #[test]
    fn test_cohens_d_zero_difference() {
        let a = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let b = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let d = cohens_d(&a, &b);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_ablation_cost_delta() {
        let study = AblationStudy {
            name: "Cost test".into(),
            description: "Cost comparison".into(),
            technique: "expensive_technique".into(),
            control: make_run(&[0.5; 10], 0.01),
            treatment: make_run(&[0.5; 10], 0.10),
        };
        let result = AblationEngine::analyze(&study, 0.95);
        assert!(result.cost_delta > 0.0);
    }

    #[test]
    fn test_ablation_latency_delta() {
        let study = AblationStudy {
            name: "Latency test".into(),
            description: "Latency comparison".into(),
            technique: "slow_technique".into(),
            control: make_run(&[0.5; 10], 0.01),
            treatment: make_run(&[0.5; 10], 0.01),
        };
        let result = AblationEngine::analyze(&study, 0.95);
        // Both use same mock latency of 100ms each
        assert!((result.latency_delta_ms).abs() < 1.0);
    }

    #[test]
    fn test_ablation_batch() {
        let studies = vec![
            AblationStudy {
                name: "A".into(),
                description: "desc".into(),
                technique: "t1".into(),
                control: make_run(&[0.5; 10], 0.01),
                treatment: make_run(&[0.8; 10], 0.02),
            },
            AblationStudy {
                name: "B".into(),
                description: "desc".into(),
                technique: "t2".into(),
                control: make_run(&[0.5; 10], 0.01),
                treatment: make_run(&[0.5; 10], 0.01),
            },
        ];
        let results = AblationEngine::analyze_batch(&studies, 0.95);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_run_summary_fields() {
        let run = make_run(&[0.8, 0.6, 0.7, 0.9, 0.5], 0.05);
        let scores = AblationEngine::extract_scores(&run);
        let summary = AblationEngine::summarize(&run, &scores);

        assert!((summary.mean_score - 0.7).abs() < 0.01);
        assert!(summary.std_dev > 0.0);
        assert_eq!(summary.total_cost, 0.05);
        assert_eq!(summary.sample_count, 5);
    }
}
