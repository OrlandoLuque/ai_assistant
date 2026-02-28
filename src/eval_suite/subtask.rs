//! Per-subtask performance analysis and optimal model routing.
//!
//! Breaks down benchmark results by subtask type (task decomposition, code generation, etc.)
//! and identifies which model excels at each subtask, enabling optimal routing strategies.

use super::runner::{BenchmarkRunResult, ModelIdentifier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Subtask categories within agent workflows.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Subtask {
    /// Breaking a complex task into steps
    TaskDecomposition,
    /// Gathering information from sources
    InformationGathering,
    /// Writing code
    CodeGeneration,
    /// Reviewing and improving code
    CodeReview,
    /// Multi-step reasoning chains
    ReasoningChain,
    /// Selecting the right tool for a task
    ToolSelection,
    /// Recovering from errors or unexpected situations
    ErrorRecovery,
    /// Formatting output correctly
    OutputFormatting,
    /// Custom user-defined subtask
    Custom(String),
}

impl std::fmt::Display for Subtask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TaskDecomposition => write!(f, "TaskDecomposition"),
            Self::InformationGathering => write!(f, "InformationGathering"),
            Self::CodeGeneration => write!(f, "CodeGeneration"),
            Self::CodeReview => write!(f, "CodeReview"),
            Self::ReasoningChain => write!(f, "ReasoningChain"),
            Self::ToolSelection => write!(f, "ToolSelection"),
            Self::ErrorRecovery => write!(f, "ErrorRecovery"),
            Self::OutputFormatting => write!(f, "OutputFormatting"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Per-subtask performance for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPerformance {
    /// Which subtask
    pub subtask: Subtask,
    /// Which model
    pub model_id: ModelIdentifier,
    /// Mean score on this subtask
    pub score: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Mean latency in ms
    pub latency_mean_ms: f64,
    /// Mean cost per problem
    pub cost_mean: f64,
}

/// Result of subtask analysis across multiple models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskAnalysis {
    /// All per-subtask, per-model performance entries
    pub performances: Vec<SubtaskPerformance>,
    /// Best model for each subtask
    pub optimal_routing: HashMap<String, ModelIdentifier>,
    /// If routing to best-per-subtask, estimated composite score
    pub routed_composite_score: f64,
    /// Best single model's composite score
    pub best_single_model_score: f64,
    /// Improvement from routing vs best single model (percentage)
    pub routing_improvement_pct: f64,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Analyzes per-subtask model performance.
pub struct SubtaskAnalyzer;

impl SubtaskAnalyzer {
    /// Analyze per-subtask performance from tagged benchmark results.
    ///
    /// `subtask_tags` maps problem IDs to their subtask type. Problems not in
    /// the map are ignored.
    pub fn analyze(
        results: &[BenchmarkRunResult],
        subtask_tags: &HashMap<String, Subtask>,
    ) -> SubtaskAnalysis {
        // Collect scores per (model, subtask)
        let mut data: HashMap<(ModelIdentifier, Subtask), Vec<(f64, f64, f64)>> = HashMap::new();

        for run in results {
            for pr in &run.results {
                if let Some(subtask) = subtask_tags.get(&pr.problem_id) {
                    let key = (run.model_id.clone(), subtask.clone());
                    let mean_score = if pr.scores.is_empty() {
                        0.0
                    } else {
                        pr.scores.iter().sum::<f64>() / pr.scores.len() as f64
                    };
                    let mean_latency = if pr.latencies_ms.is_empty() {
                        0.0
                    } else {
                        pr.latencies_ms.iter().sum::<u64>() as f64 / pr.latencies_ms.len() as f64
                    };
                    let mean_cost = if pr.cost_estimates.is_empty() {
                        0.0
                    } else {
                        pr.cost_estimates.iter().sum::<f64>() / pr.cost_estimates.len() as f64
                    };
                    data.entry(key).or_default().push((mean_score, mean_latency, mean_cost));
                }
            }
        }

        // Build performance entries
        let mut performances = Vec::new();
        for ((model, subtask), samples) in &data {
            let n = samples.len();
            let score = samples.iter().map(|(s, _, _)| s).sum::<f64>() / n as f64;
            let latency = samples.iter().map(|(_, l, _)| l).sum::<f64>() / n as f64;
            let cost = samples.iter().map(|(_, _, c)| c).sum::<f64>() / n as f64;

            performances.push(SubtaskPerformance {
                subtask: subtask.clone(),
                model_id: model.clone(),
                score,
                sample_count: n,
                latency_mean_ms: latency,
                cost_mean: cost,
            });
        }

        // Find optimal routing (best model per subtask)
        let mut optimal_routing: HashMap<String, ModelIdentifier> = HashMap::new();
        let all_subtasks: Vec<Subtask> = subtask_tags.values().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();

        for subtask in &all_subtasks {
            let best = performances
                .iter()
                .filter(|p| &p.subtask == subtask)
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(best_perf) = best {
                optimal_routing.insert(subtask.to_string(), best_perf.model_id.clone());
            }
        }

        // Compute routed composite score (weighted average of best-per-subtask)
        let routed_composite_score = Self::compute_routed_score(&performances, &optimal_routing, &all_subtasks);

        // Compute best single model score
        let all_models: Vec<ModelIdentifier> = results.iter().map(|r| r.model_id.clone()).collect::<std::collections::HashSet<_>>().into_iter().collect();
        let best_single_model_score = all_models
            .iter()
            .map(|model| {
                Self::compute_model_subtask_score(&performances, model, &all_subtasks)
            })
            .fold(0.0_f64, f64::max);

        let routing_improvement_pct = if best_single_model_score > 0.0 {
            ((routed_composite_score - best_single_model_score) / best_single_model_score) * 100.0
        } else {
            0.0
        };

        SubtaskAnalysis {
            performances,
            optimal_routing,
            routed_composite_score,
            best_single_model_score,
            routing_improvement_pct,
        }
    }

    fn compute_routed_score(
        performances: &[SubtaskPerformance],
        routing: &HashMap<String, ModelIdentifier>,
        subtasks: &[Subtask],
    ) -> f64 {
        if subtasks.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for subtask in subtasks {
            if let Some(model) = routing.get(&subtask.to_string()) {
                if let Some(perf) = performances.iter().find(|p| &p.subtask == subtask && &p.model_id == model) {
                    total += perf.score;
                    count += 1;
                }
            }
        }
        if count > 0 { total / count as f64 } else { 0.0 }
    }

    fn compute_model_subtask_score(
        performances: &[SubtaskPerformance],
        model: &ModelIdentifier,
        subtasks: &[Subtask],
    ) -> f64 {
        if subtasks.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for subtask in subtasks {
            if let Some(perf) = performances.iter().find(|p| &p.subtask == subtask && &p.model_id == model) {
                total += perf.score;
                count += 1;
            }
        }
        if count > 0 { total / count as f64 } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::runner::{ProblemResult, TokenUsage};
    use super::super::dataset::BenchmarkSuiteType;

    fn make_run(model_name: &str, problem_scores: &[(&str, f64)]) -> BenchmarkRunResult {
        let model = ModelIdentifier { name: model_name.into(), provider: "test".into(), variant: None };
        let results: Vec<ProblemResult> = problem_scores.iter().map(|(id, score)| {
            ProblemResult {
                problem_id: id.to_string(),
                model_id: model.clone(),
                responses: vec!["resp".into()],
                scores: vec![*score],
                passed: vec![*score >= 0.99],
                latencies_ms: vec![100],
                token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 20 }],
                cost_estimates: vec![0.001],
                error: None,
                metadata: HashMap::new(),
            }
        }).collect();

        BenchmarkRunResult {
            run_id: format!("run_{}", model_name),
            model_id: model,
            dataset_name: "test".into(),
            suite_type: BenchmarkSuiteType::Custom("subtask_test".into()),
            results,
            started_at: 1000,
            completed_at: 1010,
            total_cost: 0.01,
            total_tokens: TokenUsage { input_tokens: 500, output_tokens: 200 },
        }
    }

    fn make_tags() -> HashMap<String, Subtask> {
        let mut tags = HashMap::new();
        tags.insert("p/decompose".to_string(), Subtask::TaskDecomposition);
        tags.insert("p/codegen".to_string(), Subtask::CodeGeneration);
        tags.insert("p/review".to_string(), Subtask::CodeReview);
        tags
    }

    #[test]
    fn test_subtask_analysis_basic() {
        let runs = vec![
            make_run("gpt-4", &[("p/decompose", 0.9), ("p/codegen", 0.8), ("p/review", 0.7)]),
            make_run("llama3", &[("p/decompose", 0.6), ("p/codegen", 0.9), ("p/review", 0.5)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        assert!(!analysis.performances.is_empty());
        assert!(!analysis.optimal_routing.is_empty());
    }

    #[test]
    fn test_optimal_routing() {
        let runs = vec![
            make_run("model_a", &[("p/decompose", 0.9), ("p/codegen", 0.5)]),
            make_run("model_b", &[("p/decompose", 0.5), ("p/codegen", 0.9)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        assert_eq!(analysis.optimal_routing["TaskDecomposition"].name, "model_a");
        assert_eq!(analysis.optimal_routing["CodeGeneration"].name, "model_b");
    }

    #[test]
    fn test_routing_improvement() {
        let runs = vec![
            make_run("model_a", &[("p/decompose", 0.9), ("p/codegen", 0.3)]),
            make_run("model_b", &[("p/decompose", 0.3), ("p/codegen", 0.9)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        // Routed should be better than best single model
        assert!(analysis.routed_composite_score >= analysis.best_single_model_score);
        assert!(analysis.routing_improvement_pct >= 0.0);
    }

    #[test]
    fn test_subtask_enum_variants() {
        assert_eq!(Subtask::TaskDecomposition.to_string(), "TaskDecomposition");
        assert_eq!(Subtask::CodeGeneration.to_string(), "CodeGeneration");
        assert_eq!(Subtask::Custom("MyTask".into()).to_string(), "MyTask");
    }

    #[test]
    fn test_subtask_performance_ordering() {
        let runs = vec![
            make_run("good", &[("p/codegen", 0.9)]),
            make_run("bad", &[("p/codegen", 0.2)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        let codegen_perfs: Vec<&SubtaskPerformance> = analysis
            .performances
            .iter()
            .filter(|p| p.subtask == Subtask::CodeGeneration)
            .collect();
        assert_eq!(codegen_perfs.len(), 2);
    }

    #[test]
    fn test_subtask_analysis_single_model() {
        let runs = vec![
            make_run("only", &[("p/decompose", 0.8), ("p/codegen", 0.7), ("p/review", 0.6)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        // Single model → routing and single model scores should be equal
        assert!((analysis.routed_composite_score - analysis.best_single_model_score).abs() < 0.01);
    }

    #[test]
    fn test_subtask_analysis_empty() {
        let runs: Vec<BenchmarkRunResult> = vec![];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        assert!(analysis.performances.is_empty());
        assert!(analysis.optimal_routing.is_empty());
    }

    #[test]
    fn test_subtask_custom_tags() {
        let mut tags = HashMap::new();
        tags.insert("p/1".to_string(), Subtask::Custom("Planning".into()));
        tags.insert("p/2".to_string(), Subtask::Custom("Execution".into()));

        let runs = vec![make_run("model", &[("p/1", 0.8), ("p/2", 0.6)])];
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        assert!(analysis.optimal_routing.contains_key("Planning"));
        assert!(analysis.optimal_routing.contains_key("Execution"));
    }

    #[test]
    fn test_best_single_vs_routed() {
        // When one model is universally best, routing shouldn't help
        let runs = vec![
            make_run("best", &[("p/decompose", 0.9), ("p/codegen", 0.9), ("p/review", 0.9)]),
            make_run("worse", &[("p/decompose", 0.5), ("p/codegen", 0.5), ("p/review", 0.5)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        // Best single model should equal routed score
        assert!((analysis.routed_composite_score - analysis.best_single_model_score).abs() < 0.01);
        assert!(analysis.routing_improvement_pct.abs() < 1.0);
    }

    #[test]
    fn test_subtask_cost_comparison() {
        let runs = vec![
            make_run("model_a", &[("p/codegen", 0.8)]),
        ];
        let tags = make_tags();
        let analysis = SubtaskAnalyzer::analyze(&runs, &tags);

        let codegen: Vec<&SubtaskPerformance> = analysis
            .performances
            .iter()
            .filter(|p| p.subtask == Subtask::CodeGeneration)
            .collect();
        assert_eq!(codegen.len(), 1);
        assert!(codegen[0].cost_mean > 0.0);
    }
}
