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
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            success_boost: 0.1,
            failure_penalty: 0.15,
            min_confidence_to_keep: 0.2,
            auto_create_threshold: 3,
            max_procedures: 500,
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
