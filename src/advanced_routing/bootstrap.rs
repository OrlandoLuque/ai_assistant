//! Eval-suite -> bandit warm-start bootstrapping (feature-gated).

#[cfg(feature = "eval-suite")]
use super::*;
#[cfg(feature = "eval-suite")]
use std::collections::HashMap;

// =============================================================================
// EVAL-TO-RUNTIME FEEDBACK LOOP
// =============================================================================

#[cfg(feature = "eval-suite")]
use crate::eval_suite::ConfigSearchResult;
#[cfg(feature = "eval-suite")]
use crate::eval_suite::{ComparisonMatrix, SubtaskAnalysis};

/// Maps eval suite results to bandit priors for warm-starting production routing.
#[cfg(feature = "eval-suite")]
pub struct EvalFeedbackMapper;

#[cfg(feature = "eval-suite")]
impl EvalFeedbackMapper {
    /// Convert ConfigSearchResult into per-subtask bandit priors.
    ///
    /// For each subtask with a measured quality score, computes:
    /// alpha = quality * scale, beta = (1 - quality) * scale.
    pub fn map_to_priors(
        result: &ConfigSearchResult,
        scale: f64,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();

        for (subtask_name, &quality) in &result.best.subtask_quality {
            let arm_id = if let Some(model) = result.best.config.subtask_models.get(subtask_name) {
                model.to_string()
            } else {
                result.best.config.default_model.to_string()
            };

            let alpha = quality * scale;
            let beta = (1.0 - quality) * scale;

            priors.entry(subtask_name.clone()).or_default().insert(
                arm_id,
                BetaParams {
                    alpha: alpha.max(0.01),
                    beta: beta.max(0.01),
                },
            );
        }

        priors
    }

    /// Apply eval-derived priors to an existing BanditRouter.
    pub fn apply_to_bandit(
        bandit: &mut BanditRouter,
        priors: &HashMap<String, HashMap<ArmId, BetaParams>>,
    ) {
        for (task_type, arm_priors) in priors {
            for (arm_id, params) in arm_priors {
                bandit.warm_start_for_task(task_type, arm_id, params.alpha, params.beta);
            }
        }
    }

    /// Create a warm-started BanditRouter from eval results.
    pub fn create_warm_started_bandit(
        result: &ConfigSearchResult,
        config: BanditConfig,
        scale: f64,
    ) -> BanditRouter {
        let mut bandit = BanditRouter::new(config);
        let priors = Self::map_to_priors(result, scale);
        Self::apply_to_bandit(&mut bandit, &priors);
        bandit
    }
}

/// Bootstraps bandit priors from eval-suite benchmark results.
///
/// Converts ComparisonMatrix (multi-model benchmark) or SubtaskAnalysis
/// (per-subtask routing) into warm-start priors for the bandit router.
#[cfg(feature = "eval-suite")]
pub struct BanditBootstrapper;

#[cfg(feature = "eval-suite")]
impl BanditBootstrapper {
    /// Build per-task priors from a ComparisonMatrix.
    ///
    /// Uses mean_score (metric index 1) and cost_effectiveness, weighted by
    /// the given RewardPolicy. Returns `task_type -> arm_id -> BetaParams`.
    /// The task_type key is `"global"` since ComparisonMatrix has no per-task breakdown.
    ///
    /// Returns empty HashMap if matrix has no models.
    pub fn from_comparison_matrix(
        matrix: &ComparisonMatrix,
        reward_policy: &RewardPolicy,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        if matrix.models.is_empty() {
            return HashMap::new();
        }

        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();
        let global = priors.entry("global".to_string()).or_default();

        let max_ce = matrix
            .cost_effectiveness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.001);

        let (qw, lw, cw) = reward_policy.normalize_weights();
        // Latency not directly available in ComparisonMatrix -> redistribute to quality
        let adjusted_qw = qw + lw;
        let total_w = adjusted_qw + cw;

        for (i, model) in matrix.models.iter().enumerate() {
            let mean_score = matrix
                .scores
                .get(i)
                .and_then(|s| s.get(1))
                .copied()
                .unwrap_or(0.5);

            let ce = matrix.cost_effectiveness.get(i).copied().unwrap_or(1.0);
            let ce_norm = (ce / max_ce).clamp(0.0, 1.0);

            let composite = if total_w > 1e-12 {
                (adjusted_qw * mean_score + cw * ce_norm) / total_w
            } else {
                mean_score
            };

            let scale = 10.0;
            let alpha = (composite * scale).max(0.01);
            let beta = ((1.0 - composite) * scale).max(0.01);

            let arm_id = model.to_string();
            global.insert(arm_id, BetaParams { alpha, beta });
        }

        priors
    }

    /// Build per-subtask priors from SubtaskAnalysis.
    ///
    /// Uses each `SubtaskPerformance.score` scaled by `scale` to compute priors.
    /// Returns empty HashMap if analysis has no performances.
    pub fn from_subtask_analysis(
        analysis: &SubtaskAnalysis,
        scale: f64,
    ) -> HashMap<String, HashMap<ArmId, BetaParams>> {
        if analysis.performances.is_empty() {
            return HashMap::new();
        }

        let mut priors: HashMap<String, HashMap<ArmId, BetaParams>> = HashMap::new();

        for perf in &analysis.performances {
            let subtask_name = perf.subtask.to_string();
            let arm_id = perf.model_id.to_string();
            let alpha = (perf.score * scale).max(0.01);
            let beta = ((1.0 - perf.score) * scale).max(0.01);

            priors
                .entry(subtask_name)
                .or_default()
                .insert(arm_id, BetaParams { alpha, beta });
        }

        priors
    }

    /// Create a warm-started RoutingPipeline from priors.
    ///
    /// Applies the given priors to a new pipeline's bandit router.
    /// Task types keyed as `"global"` are applied to the global bandit;
    /// all others are applied as task-specific arms.
    pub fn bootstrap_pipeline(
        priors: &HashMap<String, HashMap<ArmId, BetaParams>>,
        bandit_config: BanditConfig,
        pipeline_config: PipelineConfig,
    ) -> RoutingPipeline {
        let mut pipeline = RoutingPipeline::new(bandit_config, pipeline_config);
        for (task_type, arm_priors) in priors {
            for (arm_id, params) in arm_priors {
                if task_type == "global" {
                    pipeline
                        .bandit_mut()
                        .warm_start(arm_id, params.alpha, params.beta);
                } else {
                    pipeline.bandit_mut().warm_start_for_task(
                        task_type,
                        arm_id,
                        params.alpha,
                        params.beta,
                    );
                }
            }
        }
        pipeline
    }
}

#[cfg(test)]
#[cfg(feature = "eval-suite")]
mod eval_feedback_tests {
    use super::*;
    use crate::eval_suite::ModelIdentifier;
    use crate::eval_suite::{ConfigMeasurement, ConfigSearchResult, EvalAgentConfig};

    fn mock_search_result(quality: f64) -> ConfigSearchResult {
        let mut subtask_quality = HashMap::new();
        subtask_quality.insert("CodeGeneration".to_string(), quality);
        subtask_quality.insert("Reasoning".to_string(), quality * 0.9);

        let mut subtask_models = HashMap::new();
        subtask_models.insert(
            "CodeGeneration".to_string(),
            ModelIdentifier {
                name: "gpt-4".to_string(),
                provider: "openai".to_string(),
                variant: None,
            },
        );

        let config = EvalAgentConfig {
            subtask_models,
            ..Default::default()
        };

        let measurement = ConfigMeasurement {
            config: config.clone(),
            quality,
            quality_std: 0.1,
            latency_ms: 100.0,
            cost: 0.5,
            sample_count: 10,
            subtask_quality,
            run_result: None,
        };

        ConfigSearchResult {
            baseline: measurement.clone(),
            best: measurement,
            iterations: Vec::new(),
            evolution: Vec::new(),
            dimension_variance: HashMap::new(),
            recommended: config,
            quality_improvement_pct: 0.0,
            cost_change_pct: 0.0,
            total_evaluations: 1,
            search_cost: crate::eval_suite::SearchCost {
                total_configurations_evaluated: 5,
                total_problems_solved: 50,
                total_llm_calls: 10,
                estimated_total_cost: 5.0,
                estimated_total_tokens: 10000,
            },
            converged: true,
            stopped_by_budget: false,
        }
    }

    #[test]
    fn test_map_to_priors_basic() {
        let result = mock_search_result(0.8);
        let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
        assert!(priors.contains_key("CodeGeneration"));
        let code_priors = &priors["CodeGeneration"];
        assert!(!code_priors.is_empty());
        // quality=0.8, scale=10 -> alpha=8, beta=2
        for (_, params) in code_priors {
            assert!((params.alpha - 8.0).abs() < 0.1);
            assert!((params.beta - 2.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_apply_to_bandit() {
        let result = mock_search_result(0.8);
        let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
        let mut bandit = BanditRouter::new(BanditConfig::default());
        EvalFeedbackMapper::apply_to_bandit(&mut bandit, &priors);
        // Should have arms for CodeGeneration and Reasoning tasks
        assert!(!bandit.all_arms(Some("CodeGeneration")).is_empty());
    }

    #[test]
    fn test_create_warm_started_bandit() {
        let result = mock_search_result(0.9);
        let bandit =
            EvalFeedbackMapper::create_warm_started_bandit(&result, BanditConfig::default(), 10.0);
        assert!(!bandit.all_arms(Some("CodeGeneration")).is_empty());
    }

    #[test]
    fn test_map_zero_quality() {
        let result = mock_search_result(0.0);
        let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
        for (_, arm_priors) in &priors {
            for (_, params) in arm_priors {
                assert!(params.alpha >= 0.01);
                assert!(params.beta >= 0.01);
            }
        }
    }

    #[test]
    fn test_map_perfect_quality() {
        let result = mock_search_result(1.0);
        let priors = EvalFeedbackMapper::map_to_priors(&result, 10.0);
        let code_priors = &priors["CodeGeneration"];
        for (_, params) in code_priors {
            assert!((params.alpha - 10.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_round_trip_eval_to_bandit() {
        let result = mock_search_result(0.75);
        let mut bandit =
            EvalFeedbackMapper::create_warm_started_bandit(&result, BanditConfig::default(), 10.0);
        // Should be able to select from the warm-started bandit
        let outcome = bandit.select(Some("CodeGeneration"));
        assert!(outcome.is_ok());
    }

    // =====================================================================
    // BANDIT BOOTSTRAPPER TESTS
    // =====================================================================

    use crate::eval_suite::{ComparisonMatrix, Subtask, SubtaskAnalysis, SubtaskPerformance};

    fn mock_comparison_matrix(n_models: usize) -> ComparisonMatrix {
        let models: Vec<ModelIdentifier> = (0..n_models)
            .map(|i| ModelIdentifier {
                name: format!("model-{}", i),
                provider: "test".to_string(),
                variant: None,
            })
            .collect();
        let metrics = vec![
            "accuracy".to_string(),
            "mean_score".to_string(),
            "mean_latency_ms".to_string(),
            "total_cost".to_string(),
        ];
        let scores: Vec<Vec<f64>> = (0..n_models)
            .map(|i| {
                let q = 0.5 + 0.1 * i as f64;
                vec![q, q, 200.0, 0.01 * (i + 1) as f64]
            })
            .collect();
        let costs: Vec<f64> = (0..n_models).map(|i| 0.01 * (i + 1) as f64).collect();
        let cost_effectiveness: Vec<f64> = scores
            .iter()
            .zip(costs.iter())
            .map(|(s, c)| if *c > 0.0 { s[1] / c } else { 0.0 })
            .collect();
        ComparisonMatrix {
            models,
            metrics,
            scores,
            significance: vec![vec![1.0; n_models]; n_models],
            elo_ratings: HashMap::new(),
            costs,
            cost_effectiveness,
        }
    }

    #[test]
    fn test_bootstrapper_from_empty_matrix() {
        let matrix = ComparisonMatrix {
            models: vec![],
            metrics: vec![],
            scores: vec![],
            significance: vec![],
            elo_ratings: HashMap::new(),
            costs: vec![],
            cost_effectiveness: vec![],
        };
        let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
        assert!(priors.is_empty());
    }

    #[test]
    fn test_bootstrapper_from_single_model() {
        let matrix = mock_comparison_matrix(1);
        let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
        assert!(priors.contains_key("global"));
        assert_eq!(priors["global"].len(), 1);
        let arm_id = matrix.models[0].to_string();
        assert!(priors["global"].contains_key(&arm_id));
    }

    #[test]
    fn test_bootstrapper_from_multiple_models() {
        let matrix = mock_comparison_matrix(3);
        let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
        assert_eq!(priors["global"].len(), 3);
    }

    #[test]
    fn test_bootstrapper_uses_reward_policy_weights() {
        let matrix = mock_comparison_matrix(2);
        let cost_policy = RewardPolicy {
            quality_weight: 0.1,
            latency_weight: 0.0,
            cost_weight: 0.9,
            latency_ref_ms: 5000.0,
            cost_ref: 0.1,
        };
        let priors_cost = BanditBootstrapper::from_comparison_matrix(&matrix, &cost_policy);
        let qual_policy = RewardPolicy {
            quality_weight: 0.9,
            latency_weight: 0.0,
            cost_weight: 0.1,
            latency_ref_ms: 5000.0,
            cost_ref: 0.1,
        };
        let priors_qual = BanditBootstrapper::from_comparison_matrix(&matrix, &qual_policy);
        let arm_id = matrix.models[0].to_string();
        let alpha_cost = priors_cost["global"][&arm_id].alpha;
        let alpha_qual = priors_qual["global"][&arm_id].alpha;
        assert!(alpha_cost > 0.0);
        assert!(alpha_qual > 0.0);
    }

    fn mock_subtask_analysis() -> SubtaskAnalysis {
        SubtaskAnalysis {
            performances: vec![
                SubtaskPerformance {
                    subtask: Subtask::CodeGeneration,
                    model_id: ModelIdentifier {
                        name: "gpt-4o".to_string(),
                        provider: "openai".to_string(),
                        variant: None,
                    },
                    score: 0.85,
                    sample_count: 50,
                    latency_mean_ms: 300.0,
                    cost_mean: 0.02,
                },
                SubtaskPerformance {
                    subtask: Subtask::ReasoningChain,
                    model_id: ModelIdentifier {
                        name: "claude-3.5-sonnet".to_string(),
                        provider: "anthropic".to_string(),
                        variant: None,
                    },
                    score: 0.92,
                    sample_count: 50,
                    latency_mean_ms: 400.0,
                    cost_mean: 0.03,
                },
            ],
            optimal_routing: HashMap::new(),
            routed_composite_score: 0.88,
            best_single_model_score: 0.85,
            routing_improvement_pct: 3.5,
        }
    }

    #[test]
    fn test_bootstrapper_from_subtask_analysis_empty() {
        let analysis = SubtaskAnalysis {
            performances: vec![],
            optimal_routing: HashMap::new(),
            routed_composite_score: 0.0,
            best_single_model_score: 0.0,
            routing_improvement_pct: 0.0,
        };
        let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
        assert!(priors.is_empty());
    }

    #[test]
    fn test_bootstrapper_from_subtask_analysis_basic() {
        let analysis = mock_subtask_analysis();
        let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
        assert!(priors.contains_key("CodeGeneration"));
        assert!(priors.contains_key("ReasoningChain"));
        let arm_id = "openai/gpt-4o".to_string();
        let code_priors = &priors["CodeGeneration"];
        assert!(code_priors.contains_key(&arm_id));
        assert!((code_priors[&arm_id].alpha - 8.5).abs() < 0.1);
        assert!((code_priors[&arm_id].beta - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_bootstrapper_from_subtask_analysis_multiple() {
        let analysis = mock_subtask_analysis();
        let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
        assert_eq!(priors.len(), 2);
    }

    #[test]
    fn test_bootstrapper_bootstrap_pipeline() {
        let analysis = mock_subtask_analysis();
        let priors = BanditBootstrapper::from_subtask_analysis(&analysis, 10.0);
        let pipeline = BanditBootstrapper::bootstrap_pipeline(
            &priors,
            BanditConfig::default(),
            PipelineConfig::default(),
        );
        assert!(!pipeline
            .bandit()
            .all_arms(Some("CodeGeneration"))
            .is_empty());
    }

    #[test]
    fn test_bootstrapper_round_trip_select() {
        // Use from_comparison_matrix which creates "global" priors (accessible to all domains)
        let matrix = mock_comparison_matrix(2);
        let priors = BanditBootstrapper::from_comparison_matrix(&matrix, &RewardPolicy::default());
        let mut pipeline = BanditBootstrapper::bootstrap_pipeline(
            &priors,
            BanditConfig::default(),
            PipelineConfig::default(),
        );
        let features = QueryFeatureExtractor::extract("implement a function");
        let outcome = pipeline.route(&features);
        assert!(outcome.is_ok());
    }

    #[test]
    fn test_bootstrapper_scale_effect() {
        let analysis = mock_subtask_analysis();
        let priors_low = BanditBootstrapper::from_subtask_analysis(&analysis, 1.0);
        let priors_high = BanditBootstrapper::from_subtask_analysis(&analysis, 100.0);
        let arm_id = "openai/gpt-4o".to_string();
        let alpha_low = priors_low["CodeGeneration"][&arm_id].alpha;
        let alpha_high = priors_high["CodeGeneration"][&arm_id].alpha;
        assert!(alpha_high > alpha_low * 10.0);
    }
}
