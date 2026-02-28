//! Report generation and JSON export for evaluation results.
//!
//! Provides a builder pattern for constructing comprehensive evaluation reports
//! from benchmark runs, comparisons, ablation studies, and subtask analyses.

use super::ablation::AblationResult;
use super::comparison::ComparisonMatrix;
use super::runner::{BenchmarkRunResult, ModelIdentifier, TokenUsage};
use super::subtask::SubtaskAnalysis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Cost breakdown across all evaluation runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Total cost in USD
    pub total_cost: f64,
    /// Cost per model (key = model display string)
    pub cost_by_model: HashMap<String, f64>,
    /// Cost per suite (key = suite name)
    pub cost_by_suite: HashMap<String, f64>,
    /// Cost per technique (from ablation studies, key = technique name)
    pub cost_by_technique: HashMap<String, f64>,
    /// Total token usage
    pub total_tokens: TokenUsage,
    /// Total problems evaluated
    pub total_problems_evaluated: usize,
    /// Total LLM calls made
    pub total_llm_calls: usize,
}

/// Summary and recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Best overall model (by accuracy)
    pub best_overall_model: Option<ModelIdentifier>,
    /// Most cost-effective model
    pub best_cost_effective_model: Option<ModelIdentifier>,
    /// Best model per category (key = category name)
    pub best_per_category: HashMap<String, ModelIdentifier>,
    /// Technique recommendations from ablation studies
    pub technique_recommendations: Vec<String>,
    /// Key findings (auto-generated insights)
    pub key_findings: Vec<String>,
}

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuiteReport {
    /// Unique report identifier
    pub report_id: String,
    /// Report title
    pub title: String,
    /// Unix timestamp when the report was generated
    pub generated_at: u64,
    /// Per-model benchmark results
    pub runs: Vec<BenchmarkRunResult>,
    /// Multi-model comparison (if applicable)
    pub comparison: Option<ComparisonMatrix>,
    /// Per-subtask analysis (if applicable)
    pub subtask_analysis: Option<SubtaskAnalysis>,
    /// Ablation study results
    pub ablations: Vec<AblationResult>,
    /// Cost breakdown
    pub cost_breakdown: CostBreakdown,
    /// Summary and recommendations
    pub summary: ReportSummary,
}

impl EvalSuiteReport {
    /// Export the report to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Export the report to a compact JSON string.
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing evaluation reports.
pub struct ReportBuilder {
    title: String,
    runs: Vec<BenchmarkRunResult>,
    comparison: Option<ComparisonMatrix>,
    subtask_analysis: Option<SubtaskAnalysis>,
    ablations: Vec<AblationResult>,
}

impl ReportBuilder {
    /// Create a new report builder with the given title.
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            runs: Vec::new(),
            comparison: None,
            subtask_analysis: None,
            ablations: Vec::new(),
        }
    }

    /// Add a benchmark run result.
    pub fn add_run(mut self, run: BenchmarkRunResult) -> Self {
        self.runs.push(run);
        self
    }

    /// Set the multi-model comparison matrix.
    pub fn with_comparison(mut self, matrix: ComparisonMatrix) -> Self {
        self.comparison = Some(matrix);
        self
    }

    /// Set the subtask analysis.
    pub fn with_subtask_analysis(mut self, analysis: SubtaskAnalysis) -> Self {
        self.subtask_analysis = Some(analysis);
        self
    }

    /// Add an ablation study result.
    pub fn add_ablation(mut self, ablation: AblationResult) -> Self {
        self.ablations.push(ablation);
        self
    }

    /// Build the final report.
    pub fn build(self) -> EvalSuiteReport {
        let cost_breakdown = self.compute_cost_breakdown();
        let summary = self.generate_summary();

        EvalSuiteReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            title: self.title,
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            runs: self.runs,
            comparison: self.comparison,
            subtask_analysis: self.subtask_analysis,
            ablations: self.ablations,
            cost_breakdown,
            summary,
        }
    }

    /// Compute cost breakdown from all runs and ablations.
    fn compute_cost_breakdown(&self) -> CostBreakdown {
        let mut cost_by_model: HashMap<String, f64> = HashMap::new();
        let mut cost_by_suite: HashMap<String, f64> = HashMap::new();
        let mut total_cost = 0.0;
        let mut total_input = 0_usize;
        let mut total_output = 0_usize;
        let mut total_problems = 0_usize;
        let mut total_calls = 0_usize;

        for run in &self.runs {
            let model_key = run.model_id.to_string();
            *cost_by_model.entry(model_key).or_insert(0.0) += run.total_cost;

            let suite_key = run.suite_type.to_string();
            *cost_by_suite.entry(suite_key).or_insert(0.0) += run.total_cost;

            total_cost += run.total_cost;
            total_input += run.total_tokens.input_tokens;
            total_output += run.total_tokens.output_tokens;
            total_problems += run.results.len();
            total_calls += run.results.iter().map(|r| r.responses.len()).sum::<usize>();
        }

        let mut cost_by_technique: HashMap<String, f64> = HashMap::new();
        for ablation in &self.ablations {
            let technique_cost = ablation.control_summary.total_cost + ablation.treatment_summary.total_cost;
            *cost_by_technique.entry(ablation.technique.clone()).or_insert(0.0) += technique_cost;
        }

        CostBreakdown {
            total_cost,
            cost_by_model,
            cost_by_suite,
            cost_by_technique,
            total_tokens: TokenUsage {
                input_tokens: total_input,
                output_tokens: total_output,
            },
            total_problems_evaluated: total_problems,
            total_llm_calls: total_calls,
        }
    }

    /// Generate summary with recommendations.
    fn generate_summary(&self) -> ReportSummary {
        let mut best_overall_model = None;
        let mut best_overall_accuracy = -1.0;
        let mut best_cost_effective_model = None;
        let mut best_cost_effectiveness = -1.0;

        for run in &self.runs {
            let acc = run.accuracy();
            if acc > best_overall_accuracy {
                best_overall_accuracy = acc;
                best_overall_model = Some(run.model_id.clone());
            }

            let cost_eff = if run.total_cost > 0.0 {
                acc / run.total_cost
            } else {
                acc * 1e6
            };
            if cost_eff > best_cost_effectiveness {
                best_cost_effectiveness = cost_eff;
                best_cost_effective_model = Some(run.model_id.clone());
            }
        }

        // Best per category (from comparison if available)
        let best_per_category = if let Some(ref comparison) = self.comparison {
            comparison.best_per_metric()
                .into_iter()
                .collect()
        } else {
            HashMap::new()
        };

        // Technique recommendations from ablation results
        let technique_recommendations: Vec<String> = self
            .ablations
            .iter()
            .map(|a| {
                match &a.recommendation {
                    super::ablation::AblationRecommendation::Enable { quality_gain_pct, cost_increase_pct } => {
                        format!(
                            "ENABLE '{}': +{:.1}% quality, +{:.1}% cost",
                            a.technique, quality_gain_pct, cost_increase_pct
                        )
                    }
                    super::ablation::AblationRecommendation::Neutral => {
                        format!("NEUTRAL '{}': no significant effect", a.technique)
                    }
                    super::ablation::AblationRecommendation::Disable { quality_loss_pct } => {
                        format!("DISABLE '{}': -{:.1}% quality", a.technique, quality_loss_pct)
                    }
                    super::ablation::AblationRecommendation::InsufficientData => {
                        format!("INSUFFICIENT DATA for '{}': collect more samples", a.technique)
                    }
                }
            })
            .collect();

        // Key findings
        let mut key_findings = Vec::new();

        if let Some(ref model) = best_overall_model {
            key_findings.push(format!(
                "Best overall model: {} ({:.1}% accuracy)",
                model, best_overall_accuracy * 100.0
            ));
        }

        if let Some(ref model) = best_cost_effective_model {
            if best_cost_effective_model != best_overall_model {
                key_findings.push(format!(
                    "Most cost-effective: {} (best accuracy/cost ratio)",
                    model
                ));
            }
        }

        if let Some(ref analysis) = self.subtask_analysis {
            if analysis.routing_improvement_pct > 5.0 {
                key_findings.push(format!(
                    "Multi-model routing could improve scores by {:.1}%",
                    analysis.routing_improvement_pct
                ));
            }
        }

        let significant_ablations: Vec<&AblationResult> = self
            .ablations
            .iter()
            .filter(|a| a.is_significant)
            .collect();
        if !significant_ablations.is_empty() {
            key_findings.push(format!(
                "{} out of {} techniques showed significant impact",
                significant_ablations.len(),
                self.ablations.len()
            ));
        }

        ReportSummary {
            best_overall_model,
            best_cost_effective_model,
            best_per_category,
            technique_recommendations,
            key_findings,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::runner::ProblemResult;
    use super::super::dataset::BenchmarkSuiteType;
    use super::super::ablation::{AblationRecommendation, RunSummary};

    fn make_run(model_name: &str, accuracy: f64, cost: f64) -> BenchmarkRunResult {
        let model = ModelIdentifier { name: model_name.into(), provider: "test".into(), variant: None };
        let n_problems = 10;
        let results: Vec<ProblemResult> = (0..n_problems).map(|i| {
            let score = if (i as f64 / n_problems as f64) < accuracy { 1.0 } else { 0.0 };
            ProblemResult {
                problem_id: format!("p/{}", i),
                model_id: model.clone(),
                responses: vec!["resp".into()],
                scores: vec![score],
                passed: vec![score >= 0.99],
                latencies_ms: vec![100],
                token_counts: vec![TokenUsage { input_tokens: 50, output_tokens: 20 }],
                cost_estimates: vec![cost / n_problems as f64],
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

    fn make_ablation_result(technique: &str, significant: bool, quality_delta: f64) -> AblationResult {
        AblationResult {
            study_name: format!("{} study", technique),
            technique: technique.into(),
            quality_delta,
            latency_delta_ms: 10.0,
            cost_delta: 0.01,
            p_value: if significant { 0.01 } else { 0.5 },
            is_significant: significant,
            effect_size: quality_delta.abs() * 2.0,
            recommendation: if quality_delta > 0.0 && significant {
                AblationRecommendation::Enable { quality_gain_pct: quality_delta * 100.0, cost_increase_pct: 10.0 }
            } else if quality_delta < 0.0 && significant {
                AblationRecommendation::Disable { quality_loss_pct: quality_delta.abs() * 100.0 }
            } else {
                AblationRecommendation::Neutral
            },
            control_summary: RunSummary { mean_score: 0.5, std_dev: 0.1, mean_latency_ms: 100.0, total_cost: 0.01, sample_count: 10 },
            treatment_summary: RunSummary { mean_score: 0.5 + quality_delta, std_dev: 0.1, mean_latency_ms: 110.0, total_cost: 0.02, sample_count: 10 },
        }
    }

    #[test]
    fn test_report_builder_single_run() {
        let report = ReportBuilder::new("Single Run Test")
            .add_run(make_run("gpt-4", 0.8, 0.05))
            .build();

        assert_eq!(report.title, "Single Run Test");
        assert_eq!(report.runs.len(), 1);
        assert!(report.comparison.is_none());
        assert!(report.subtask_analysis.is_none());
        assert!(report.ablations.is_empty());
    }

    #[test]
    fn test_report_builder_multi_run() {
        let report = ReportBuilder::new("Multi Run Test")
            .add_run(make_run("gpt-4", 0.9, 0.05))
            .add_run(make_run("llama3", 0.6, 0.001))
            .build();

        assert_eq!(report.runs.len(), 2);
    }

    #[test]
    fn test_report_cost_breakdown() {
        let report = ReportBuilder::new("Cost Test")
            .add_run(make_run("gpt-4", 0.8, 0.05))
            .add_run(make_run("llama3", 0.6, 0.001))
            .build();

        assert!((report.cost_breakdown.total_cost - 0.051).abs() < 0.001);
        assert_eq!(report.cost_breakdown.cost_by_model.len(), 2);
        assert!(report.cost_breakdown.total_problems_evaluated > 0);
        assert!(report.cost_breakdown.total_llm_calls > 0);
    }

    #[test]
    fn test_report_summary_generation() {
        let report = ReportBuilder::new("Summary Test")
            .add_run(make_run("gpt-4", 0.9, 0.05))
            .add_run(make_run("llama3", 0.6, 0.001))
            .build();

        assert!(report.summary.best_overall_model.is_some());
        assert_eq!(report.summary.best_overall_model.as_ref().unwrap().name, "gpt-4");
    }

    #[test]
    fn test_report_with_ablation() {
        let report = ReportBuilder::new("Ablation Test")
            .add_run(make_run("model", 0.7, 0.02))
            .add_ablation(make_ablation_result("rag_graph", true, 0.2))
            .add_ablation(make_ablation_result("cot", false, 0.01))
            .build();

        assert_eq!(report.ablations.len(), 2);
        assert_eq!(report.summary.technique_recommendations.len(), 2);
    }

    #[test]
    fn test_report_json_export() {
        let report = ReportBuilder::new("JSON Test")
            .add_run(make_run("model", 0.7, 0.02))
            .build();

        let json = report.to_json();
        assert!(json.contains("JSON Test"));
        assert!(json.contains("report_id"));
        assert!(json.contains("runs"));
    }

    #[test]
    fn test_report_compact_json() {
        let report = ReportBuilder::new("Compact Test")
            .add_run(make_run("model", 0.7, 0.02))
            .build();

        let json = report.to_json_compact();
        assert!(!json.contains('\n'));
        assert!(json.contains("Compact Test"));
    }

    #[test]
    fn test_report_empty() {
        let report = ReportBuilder::new("Empty Test").build();
        assert!(report.runs.is_empty());
        assert!(report.summary.best_overall_model.is_none());
        assert_eq!(report.cost_breakdown.total_cost, 0.0);
    }

    #[test]
    fn test_report_best_model_selection() {
        let report = ReportBuilder::new("Best Model")
            .add_run(make_run("cheap_bad", 0.3, 0.001))
            .add_run(make_run("expensive_good", 0.9, 1.0))
            .build();

        // Best overall = expensive_good (highest accuracy)
        assert_eq!(report.summary.best_overall_model.as_ref().unwrap().name, "expensive_good");
        // Best cost-effective = cheap_bad (best acc/cost ratio)
        assert_eq!(report.summary.best_cost_effective_model.as_ref().unwrap().name, "cheap_bad");
    }

    #[test]
    fn test_report_key_findings() {
        let report = ReportBuilder::new("Findings Test")
            .add_run(make_run("model", 0.8, 0.05))
            .add_ablation(make_ablation_result("technique_a", true, 0.3))
            .build();

        assert!(!report.summary.key_findings.is_empty());
    }

    #[test]
    fn test_report_cost_by_technique() {
        let report = ReportBuilder::new("Technique Cost")
            .add_ablation(make_ablation_result("rag", true, 0.1))
            .add_ablation(make_ablation_result("cot", false, 0.0))
            .build();

        assert!(report.cost_breakdown.cost_by_technique.contains_key("rag"));
        assert!(report.cost_breakdown.cost_by_technique.contains_key("cot"));
    }
}
