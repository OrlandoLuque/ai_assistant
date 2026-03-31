//! Self-evolving procedures (MemRL-style): feedback-driven procedure evolution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::procedural::ProceduralStore;

/// Feedback for a procedure execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureFeedback {
    pub procedure_id: String,
    pub outcome: FeedbackOutcome,
    pub context: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The outcome of a procedure execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FeedbackOutcome {
    /// The procedure succeeded.
    Success,
    /// The procedure failed.
    Failure,
    /// The procedure partially succeeded with a score in [0.0, 1.0].
    Partial { score: f64 },
}

/// Configuration for procedure evolution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvolutionConfig {
    /// How much confidence increases on success (default: 0.1).
    pub success_boost: f64,
    /// How much confidence decreases on failure (default: 0.15).
    pub failure_penalty: f64,
    /// Minimum confidence to keep a procedure (default: 0.2).
    pub min_confidence_to_keep: f64,
    /// Create a new procedure after this many similar episodes without a
    /// matching procedure (default: 3).
    pub auto_create_threshold: usize,
    /// Maximum number of procedures to track (default: 500).
    pub max_procedures: usize,
    /// Use LLM to analyze procedure failures and suggest improvements.
    /// When false (default), uses heuristic confidence adjustments only.
    pub llm_enhanced: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            success_boost: 0.1,
            failure_penalty: 0.15,
            min_confidence_to_keep: 0.2,
            auto_create_threshold: 3,
            max_procedures: 500,
            llm_enhanced: false,
        }
    }
}

impl EvolutionConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Report from a procedure evolution pass.
#[derive(Debug, Clone)]
pub struct EvolutionReport {
    /// Number of procedures whose confidence was updated.
    pub procedures_updated: usize,
    /// Number of new procedures created from patterns.
    pub procedures_created: usize,
    /// Number of procedures retired (below min confidence).
    pub procedures_retired: usize,
    /// Total feedback entries processed.
    pub feedback_processed: usize,
}

/// Aggregate statistics about the evolution process.
#[derive(Debug, Clone)]
pub struct EvolutionStatistics {
    /// Total feedback entries recorded.
    pub total_feedback: usize,
    /// Fraction of feedback that was Success (0.0 - 1.0).
    pub success_rate: f64,
    /// Average confidence across all tracked procedures.
    pub avg_confidence: f64,
    /// Number of distinct procedures that have received feedback.
    pub procedures_tracked: usize,
}

/// Analyzes feedback and evolves procedures over time.
pub struct ProcedureEvolver {
    config: EvolutionConfig,
    feedback_log: Vec<ProcedureFeedback>,
}

impl ProcedureEvolver {
    /// Create a new evolver with the given configuration.
    pub fn new(config: EvolutionConfig) -> Self {
        Self {
            config,
            feedback_log: Vec::new(),
        }
    }

    /// Record a piece of feedback.
    pub fn record_feedback(&mut self, feedback: ProcedureFeedback) {
        self.feedback_log.push(feedback);
    }

    /// Evolve procedures based on accumulated feedback.
    ///
    /// For each procedure that has feedback:
    /// - Success: boost confidence by `success_boost`
    /// - Failure: reduce confidence by `failure_penalty`
    /// - Partial: adjust by `score * success_boost - (1 - score) * failure_penalty`
    ///
    /// Procedures whose confidence drops below `min_confidence_to_keep` are removed.
    pub fn evolve(&mut self, store: &mut ProceduralStore) -> EvolutionReport {
        let mut updated = 0usize;
        let mut retired_ids = Vec::new();
        let feedback_processed = self.feedback_log.len();

        // Group feedback by procedure_id
        let mut feedback_by_proc: HashMap<String, Vec<&ProcedureFeedback>> = HashMap::new();
        for fb in &self.feedback_log {
            feedback_by_proc
                .entry(fb.procedure_id.clone())
                .or_default()
                .push(fb);
        }

        // Apply feedback to each procedure
        for (proc_id, feedbacks) in &feedback_by_proc {
            if let Some(proc) = store.procedures.iter_mut().find(|p| p.id == *proc_id) {
                for fb in feedbacks {
                    match &fb.outcome {
                        FeedbackOutcome::Success => {
                            proc.confidence =
                                (proc.confidence + self.config.success_boost).min(1.0);
                            proc.success_count += 1;
                        }
                        FeedbackOutcome::Failure => {
                            proc.confidence =
                                (proc.confidence - self.config.failure_penalty).max(0.0);
                            proc.failure_count += 1;
                        }
                        FeedbackOutcome::Partial { score } => {
                            let adjustment = score * self.config.success_boost
                                - (1.0 - score) * self.config.failure_penalty;
                            proc.confidence = (proc.confidence + adjustment).clamp(0.0, 1.0);
                            if *score >= 0.5 {
                                proc.success_count += 1;
                            } else {
                                proc.failure_count += 1;
                            }
                        }
                    }
                }
                updated += 1;

                // Check for retirement
                if proc.confidence < self.config.min_confidence_to_keep {
                    retired_ids.push(proc_id.clone());
                }
            }
        }

        // Retire procedures below threshold
        for id in &retired_ids {
            store.procedures.retain(|p| p.id != *id);
        }

        // Clear processed feedback
        self.feedback_log.clear();

        EvolutionReport {
            procedures_updated: updated,
            procedures_created: 0,
            procedures_retired: retired_ids.len(),
            feedback_processed,
        }
    }

    /// Get all feedback entries for a specific procedure.
    pub fn get_feedback_for(&self, procedure_id: &str) -> Vec<&ProcedureFeedback> {
        self.feedback_log
            .iter()
            .filter(|fb| fb.procedure_id == procedure_id)
            .collect()
    }

    /// Compute aggregate statistics from the feedback log.
    pub fn get_statistics(&self) -> EvolutionStatistics {
        let total = self.feedback_log.len();
        let success_count = self
            .feedback_log
            .iter()
            .filter(|fb| matches!(fb.outcome, FeedbackOutcome::Success))
            .count();

        let success_rate = if total == 0 {
            0.0
        } else {
            success_count as f64 / total as f64
        };

        let tracked: std::collections::HashSet<&str> = self
            .feedback_log
            .iter()
            .map(|fb| fb.procedure_id.as_str())
            .collect();

        EvolutionStatistics {
            total_feedback: total,
            success_rate,
            avg_confidence: 0.0, // Confidence lives in ProceduralStore, not here
            procedures_tracked: tracked.len(),
        }
    }
}

// ============================================================================
// LLM Enhancement: Failure Analysis (V68)
// ============================================================================

/// Result of LLM-based failure analysis for a procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureAnalysis {
    /// Root cause identified by the LLM.
    pub cause: String,
    /// Suggested improvement.
    pub suggestion: String,
    /// Recommended confidence adjustment (e.g., -0.1).
    pub confidence_adjustment: f64,
}

impl ProcedureEvolver {
    /// Build a prompt for LLM-based failure analysis.
    ///
    /// Returns None if LLM enhancement is disabled or there are no failure feedbacks.
    pub fn build_failure_analysis_prompt(
        &self,
        procedure_id: &str,
        context: &str,
    ) -> Option<String> {
        if !self.config.llm_enhanced {
            return None;
        }

        let failures: Vec<_> = self
            .feedback_log
            .iter()
            .filter(|fb| {
                fb.procedure_id == procedure_id
                    && matches!(fb.outcome, FeedbackOutcome::Failure)
            })
            .collect();

        if failures.is_empty() {
            return None;
        }

        let mut prompt = String::from(
            "A procedure failed. Analyze why and suggest improvement. \
             Return JSON: {\"cause\":\"...\",\"suggestion\":\"...\",\"confidence_adjustment\":-0.1}\n\n",
        );

        prompt.push_str(&format!("Procedure ID: {}\n", procedure_id));
        prompt.push_str(&format!(
            "Context: {}\n",
            crate::llm_enhance::prompt_wrap(context)
        ));
        prompt.push_str(&format!("Failure count: {}\n", failures.len()));

        for (i, fb) in failures.iter().enumerate().take(10) {
            prompt.push_str(&format!(
                "{}. [{}] {}\n",
                i + 1,
                fb.timestamp.format("%Y-%m-%d %H:%M"),
                crate::llm_enhance::prompt_wrap(&fb.context)
            ));
        }

        Some(prompt)
    }

    /// Parse LLM response for failure analysis.
    pub fn parse_failure_analysis_response(response: &str) -> Option<FailureAnalysis> {
        if let Some(json_str) = crate::llm_enhance::extract_json(response) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let cause = val.get("cause").and_then(|s| s.as_str()).unwrap_or("unknown");
                let suggestion = val
                    .get("suggestion")
                    .and_then(|s| s.as_str())
                    .unwrap_or("no suggestion");
                let confidence_adjustment = val
                    .get("confidence_adjustment")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(-0.1);
                return Some(FailureAnalysis {
                    cause: cause.to_string(),
                    suggestion: suggestion.to_string(),
                    confidence_adjustment: confidence_adjustment.clamp(-0.5, 0.0),
                });
            }
        }
        None
    }

    /// Analyze a procedure failure with optional LLM enhancement.
    ///
    /// If `llm` is Some and config.llm_enhanced is true, uses LLM for root-cause
    /// analysis. Otherwise returns a heuristic-based FailureAnalysis.
    pub fn analyze_failure_with_llm(
        &self,
        procedure_id: &str,
        context: &str,
        llm: Option<&dyn crate::llm_enhance::LlmEnhancer>,
    ) -> FailureAnalysis {
        // Heuristic baseline
        let failure_count = self
            .feedback_log
            .iter()
            .filter(|fb| {
                fb.procedure_id == procedure_id
                    && matches!(fb.outcome, FeedbackOutcome::Failure)
            })
            .count();

        let heuristic = FailureAnalysis {
            cause: format!("Procedure failed {} time(s)", failure_count),
            suggestion: "Review procedure steps and adjust parameters".to_string(),
            confidence_adjustment: -(failure_count as f64 * 0.05).min(0.3),
        };

        // Try LLM enhancement
        if let Some(enhancer) = llm {
            if self.config.llm_enhanced && enhancer.is_available() {
                if let Some(prompt) = self.build_failure_analysis_prompt(procedure_id, context) {
                    if let Ok(response) = enhancer.generate(&prompt, 300) {
                        if let Some(analysis) = Self::parse_failure_analysis_response(&response) {
                            return analysis;
                        }
                    }
                }
            }
        }

        heuristic
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feedback(proc_id: &str, outcome: FeedbackOutcome) -> ProcedureFeedback {
        ProcedureFeedback {
            procedure_id: proc_id.to_string(),
            outcome,
            context: "test context".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_analyze_failure_heuristic_without_llm() {
        let config = EvolutionConfig {
            llm_enhanced: false,
            ..Default::default()
        };
        let mut evolver = ProcedureEvolver::new(config);
        evolver.record_feedback(make_feedback("proc-1", FeedbackOutcome::Failure));
        evolver.record_feedback(make_feedback("proc-1", FeedbackOutcome::Failure));

        let analysis = evolver.analyze_failure_with_llm("proc-1", "test", None);
        assert!(analysis.cause.contains("2 time(s)"));
        assert!(analysis.confidence_adjustment < 0.0);
    }

    #[test]
    fn test_analyze_failure_with_mock_llm() {
        let config = EvolutionConfig {
            llm_enhanced: true,
            ..Default::default()
        };
        let mut evolver = ProcedureEvolver::new(config);
        evolver.record_feedback(make_feedback("proc-1", FeedbackOutcome::Failure));

        let mock = crate::llm_enhance::MockLlm::new(
            "{\"cause\":\"timeout in step 3\",\"suggestion\":\"increase timeout\",\"confidence_adjustment\":-0.15}",
        );
        let analysis = evolver.analyze_failure_with_llm("proc-1", "step 3 timed out", Some(&mock));
        assert!(analysis.cause.contains("timeout"), "Expected LLM cause, got: {}", analysis.cause);
        assert!(analysis.suggestion.contains("increase timeout"));
        assert!((analysis.confidence_adjustment - (-0.15)).abs() < 0.01);
    }

    #[test]
    fn test_analyze_failure_llm_fallback_on_failure() {
        let config = EvolutionConfig {
            llm_enhanced: true,
            ..Default::default()
        };
        let mut evolver = ProcedureEvolver::new(config);
        evolver.record_feedback(make_feedback("proc-1", FeedbackOutcome::Failure));

        let failing = crate::llm_enhance::FailingMockLlm;
        let analysis = evolver.analyze_failure_with_llm("proc-1", "context", Some(&failing));
        // Should fall back to heuristic (not crash)
        assert!(analysis.cause.contains("1 time(s)"));
        assert!(analysis.confidence_adjustment < 0.0);
    }
}
