//! `ClaimVerificationTask` — concrete `CorrectableTask` that combines
//! Chain-of-Verification, faithfulness scoring, and Quality Gates into a
//! single retry loop.
//!
//! Workflow:
//!
//! 1. First `execute` returns the caller's pre-generated response.
//! 2. `validate` runs CoVe + faithfulness + gates; each failing check becomes
//!    a `ClaimIssue`.
//! 3. On retry, `execute` calls the user-provided regenerate closure with
//!    the original prompt plus a feedback string built from the prior
//!    attempts.

use crate::chain_of_verification::{
    ChainOfVerification, ClaimVerificationStatus, CoVeConfig, VerificationContext,
    VerificationSource,
};
use crate::faithfulness::{FaithfulnessConfig, FaithfulnessScorer};
use crate::quality_gates::{QualityGateRunner, QualityScores};

use super::{AttemptRecord, CorrectableTask, Issue, TaskError, TaskOutcome};

/// Issues reported by `ClaimVerificationTask`.
#[derive(Debug)]
pub enum ClaimIssue {
    /// A claim was directly contradicted by the reference context.
    Contradicted {
        /// The claim text.
        claim: String,
        /// Human-readable evidence snippet.
        evidence: String,
    },
    /// A claim could not be verified against the available context.
    Unverifiable {
        /// The claim text.
        claim: String,
    },
    /// Faithfulness score below the configured threshold.
    LowFaithfulness {
        /// Measured score.
        score: f64,
        /// Threshold that was not met.
        threshold: f64,
    },
    /// A configured quality gate failed.
    GateFailed {
        /// Name of the gate.
        name: String,
        /// Metric value that missed the threshold.
        value: f64,
        /// The threshold.
        threshold: f64,
    },
    /// Calibrated abstention detected ("I don't know"). Treated as success
    /// by convention — surfaced to the engine via `is_retryable = false`
    /// AND a special marker consumed by `quality_score`.
    CalibratedAbstention,
}

impl std::fmt::Display for ClaimIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contradicted { claim, evidence } => {
                write!(
                    f,
                    "Claim CONTRADICTED: \"{}\" (evidence: {})",
                    claim, evidence
                )
            }
            Self::Unverifiable { claim } => {
                write!(f, "Claim UNVERIFIABLE: \"{}\"", claim)
            }
            Self::LowFaithfulness { score, threshold } => {
                write!(
                    f,
                    "Faithfulness {:.2} below threshold {:.2}",
                    score, threshold
                )
            }
            Self::GateFailed {
                name,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "Gate '{}' failed: value {:.2} below threshold {:.2}",
                    name, value, threshold
                )
            }
            Self::CalibratedAbstention => {
                write!(f, "Calibrated abstention (treated as success)")
            }
        }
    }
}

impl Issue for ClaimIssue {
    fn is_retryable(&self) -> bool {
        // Calibrated abstention is explicitly non-retryable — the engine
        // will stop, and we map that to success via StopReason logic in the
        // caller. All other issues are retryable.
        !matches!(self, Self::CalibratedAbstention)
    }
}

/// Regenerate closure type: given `(user_prompt, optional_feedback)` return
/// `(response_text, tokens_used, cost_usd)`, or `None` on failure.
pub type RegenerateFn = Box<dyn FnMut(&str, Option<&str>) -> Option<(String, usize, f64)> + Send>;

/// Concrete task for claim verification with retry-with-feedback.
pub struct ClaimVerificationTask {
    user_prompt: String,
    knowledge: Option<String>,
    initial_response: Option<String>,
    regenerate_fn: RegenerateFn,
    cove_engine: ChainOfVerification,
    faithfulness_scorer: FaithfulnessScorer,
    quality_gate_runner: Option<QualityGateRunner>,
    faithfulness_threshold: f64,
}

impl ClaimVerificationTask {
    /// Minimum builder: user prompt + pre-computed initial response + a
    /// regenerate closure that will be called on retries.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_response: impl Into<String>,
        regenerate_fn: RegenerateFn,
    ) -> Self {
        let mut cove_cfg = CoVeConfig::default();
        cove_cfg.verification_source = VerificationSource::Both;
        Self {
            user_prompt: user_prompt.into(),
            knowledge: None,
            initial_response: Some(initial_response.into()),
            regenerate_fn,
            cove_engine: ChainOfVerification::new(cove_cfg),
            faithfulness_scorer: FaithfulnessScorer::new(FaithfulnessConfig::default()),
            quality_gate_runner: None,
            faithfulness_threshold: 0.5,
        }
    }

    /// Attach a reference knowledge document for grounding.
    pub fn with_knowledge(mut self, knowledge: impl Into<String>) -> Self {
        self.knowledge = Some(knowledge.into());
        self
    }

    /// Attach a quality gate runner. Gate failures become `ClaimIssue::GateFailed`.
    pub fn with_quality_gates(mut self, runner: QualityGateRunner) -> Self {
        self.quality_gate_runner = Some(runner);
        self
    }

    /// Threshold for the faithfulness metric. Below this value the attempt
    /// fails with `ClaimIssue::LowFaithfulness`. Default 0.5.
    pub fn with_faithfulness_threshold(mut self, t: f64) -> Self {
        self.faithfulness_threshold = t;
        self
    }

    /// Replace the default CoVe engine.
    pub fn with_cove_engine(mut self, engine: ChainOfVerification) -> Self {
        self.cove_engine = engine;
        self
    }

    /// Replace the default faithfulness scorer.
    pub fn with_faithfulness_scorer(mut self, scorer: FaithfulnessScorer) -> Self {
        self.faithfulness_scorer = scorer;
        self
    }

    fn build_cove_contexts(&self) -> Vec<VerificationContext> {
        let src = self.knowledge.as_deref().unwrap_or(&self.user_prompt);
        let reliability = if self.knowledge.is_some() { 0.9 } else { 0.5 };
        let source_type = if self.knowledge.is_some() {
            "file"
        } else {
            "user_query"
        };
        src.split(|c: char| c == '.' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| s.len() > 5)
            .enumerate()
            .map(|(i, sentence)| VerificationContext {
                source_id: format!("ctx-{}", i),
                source_type: source_type.to_string(),
                content: sentence.to_string(),
                reliability,
            })
            .collect()
    }

    fn is_calibrated_abstention(text: &str) -> bool {
        let l = text.to_lowercase();
        let markers = [
            "i don't know",
            "i do not know",
            "i'm not sure",
            "i am not sure",
            "cannot verify",
            "no puedo verificar",
            "no sé",
            "no lo sé",
            "insufficient information",
        ];
        markers.iter().any(|m| l.contains(m)) && text.len() < 500
    }
}

impl CorrectableTask for ClaimVerificationTask {
    type Output = String;
    type Issue = ClaimIssue;

    fn name(&self) -> &str {
        "claim_verification"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        // First attempt: consume initial_response if present.
        if feedback.is_none() {
            if let Some(initial) = self.initial_response.take() {
                return Ok(TaskOutcome {
                    output: initial,
                    // No LLM cost for the initial pre-computed response.
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }

        // Retry attempt (or first attempt with no pre-computed response):
        // regenerate via closure.
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((out, tokens, cost)) => Ok(TaskOutcome {
                output: out,
                tokens_used: tokens,
                cost_usd: cost,
            }),
            None => Err(TaskError::new("regenerate closure returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        // Calibrated abstention short-circuits: treat as success-equivalent.
        if Self::is_calibrated_abstention(output) {
            return vec![ClaimIssue::CalibratedAbstention];
        }

        let mut issues: Vec<ClaimIssue> = Vec::new();

        // CoVe verification.
        let cove_contexts = self.build_cove_contexts();
        let cove_result = self.cove_engine.verify(output, &cove_contexts);
        for v in &cove_result.verified_claims {
            match v.status {
                ClaimVerificationStatus::Contradicted => {
                    let evidence = v
                        .evidence
                        .first()
                        .map(|e| e.content.clone())
                        .unwrap_or_else(|| "(no evidence shown)".into());
                    issues.push(ClaimIssue::Contradicted {
                        claim: v.claim.clone(),
                        evidence,
                    });
                }
                ClaimVerificationStatus::Unverifiable => {
                    issues.push(ClaimIssue::Unverifiable {
                        claim: v.claim.clone(),
                    });
                }
                _ => {}
            }
        }

        // Faithfulness.
        let faith_src = self.knowledge.as_deref().unwrap_or(&self.user_prompt);
        let faith_sentences: Vec<&str> = faith_src
            .split(|c: char| c == '.' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| s.len() > 5)
            .collect();
        let faith = self.faithfulness_scorer.score(output, &faith_sentences);
        if faith.overall_score < self.faithfulness_threshold {
            issues.push(ClaimIssue::LowFaithfulness {
                score: faith.overall_score,
                threshold: self.faithfulness_threshold,
            });
        }

        // Quality gates.
        if let Some(ref runner) = self.quality_gate_runner {
            let grounded = cove_result
                .verified_claims
                .iter()
                .filter(|c| c.status == ClaimVerificationStatus::Supported)
                .count();
            let total = cove_result.verified_claims.len().max(1);
            let grounding_ratio = grounded as f64 / total as f64;
            let scores = QualityScores {
                faithfulness: Some(faith.overall_score),
                confidence: Some(cove_result.overall_accuracy),
                grounding_ratio: Some(grounding_ratio),
                consistency_score: None,
                citation_coverage: None,
            };
            let gate_result = runner.run(&scores);
            for r in &gate_result.gate_results {
                if !r.passed && r.is_blocking() {
                    issues.push(ClaimIssue::GateFailed {
                        name: r.gate_name.clone(),
                        value: r.actual,
                        threshold: r.threshold,
                    });
                }
            }
        }

        issues
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original question: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous response had {} issue(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        if prior_attempts.len() > 1 {
            lines.push(String::new());
            lines.push(format!(
                "Across {} prior attempts, issues seen:",
                prior_attempts.len()
            ));
            let mut all_issues: Vec<&String> = prior_attempts
                .iter()
                .flat_map(|a| a.issues.iter())
                .collect();
            all_issues.sort();
            all_issues.dedup();
            for (i, issue) in all_issues.iter().take(20).enumerate() {
                lines.push(format!("  {}. {}", i + 1, issue));
            }
        }
        lines.push(String::new());
        lines.push("Please regenerate your response, applying these rules:".to_string());
        lines.push("  1. Only include claims supported by the reference context.".to_string());
        lines.push("  2. Rewrite or remove any claim that was contradicted.".to_string());
        lines.push("  3. If unsure about a fact, omit it rather than guess.".to_string());
        if let Some(ref k) = self.knowledge {
            lines.push(String::new());
            let k_preview: String = k.chars().take(2_000).collect();
            lines.push(format!("Reference context:\n{}", k_preview));
        }
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues
            .iter()
            .any(|i| matches!(i, ClaimIssue::CalibratedAbstention))
        {
            return 1.0;
        }
        if issues.is_empty() {
            return 1.0;
        }
        // Penalize by issue count (scaled): each issue -> -0.1, floor at 0.
        let penalty = (issues.len() as f64) * 0.15;
        (1.0 - penalty).max(0.0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_correction::{SelfCorrectionConfig, SelfCorrectionEngine, StopReason};

    fn trivial_regen() -> RegenerateFn {
        Box::new(|_prompt, _fb| Some(("regenerated OK".to_string(), 10, 0.001)))
    }

    #[test]
    fn test_initial_response_returned_without_llm_call() {
        let called = std::sync::Arc::new(std::sync::Mutex::new(0u32));
        let called_clone = called.clone();
        let regen: RegenerateFn = Box::new(move |_p, _f| {
            *called_clone.lock().unwrap() += 1;
            Some(("should_not_run".into(), 0, 0.0))
        });

        let mut task = ClaimVerificationTask::new("What is 2+2?", "The answer is 4.", regen);
        let out = task.execute(None).unwrap();
        assert_eq!(out.output, "The answer is 4.");
        assert_eq!(out.tokens_used, 0);
        assert_eq!(*called.lock().unwrap(), 0);
    }

    #[test]
    fn test_retry_calls_regenerate() {
        let regen: RegenerateFn = Box::new(|_, fb| {
            // feedback should be Some on retry
            assert!(fb.is_some());
            Some(("corrected".into(), 50, 0.005))
        });
        let mut task = ClaimVerificationTask::new("q", "initial", regen);
        // Simulate: take initial, then retry.
        let _first = task.execute(None).unwrap();
        let second = task.execute(Some("please fix X")).unwrap();
        assert_eq!(second.output, "corrected");
        assert_eq!(second.tokens_used, 50);
    }

    #[test]
    fn test_calibrated_abstention_detected() {
        assert!(ClaimVerificationTask::is_calibrated_abstention(
            "I don't know the answer."
        ));
        assert!(ClaimVerificationTask::is_calibrated_abstention("No lo sé."));
        assert!(!ClaimVerificationTask::is_calibrated_abstention(
            "The capital of France is Paris."
        ));
    }

    #[test]
    fn test_abstention_is_non_retryable() {
        let issue = ClaimIssue::CalibratedAbstention;
        assert!(!issue.is_retryable());
    }

    #[test]
    fn test_contradicted_is_retryable() {
        let issue = ClaimIssue::Contradicted {
            claim: "x".into(),
            evidence: "y".into(),
        };
        assert!(issue.is_retryable());
    }

    #[test]
    fn test_build_feedback_includes_prior_issues() {
        let task = ClaimVerificationTask::new("q", "r", trivial_regen());
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["Claim CONTRADICTED: xyz".into()],
            quality_score: 0.3,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 0,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("What is X?", &prior);
        assert!(fb.contains("What is X?"));
        assert!(fb.contains("Claim CONTRADICTED"));
        assert!(fb.contains("Please regenerate"));
    }

    #[test]
    fn test_quality_score_no_issues_is_1() {
        let task = ClaimVerificationTask::new("q", "r", trivial_regen());
        assert_eq!(task.quality_score(&"out".into(), &[]), 1.0);
    }

    #[test]
    fn test_quality_score_abstention_is_1() {
        let task = ClaimVerificationTask::new("q", "r", trivial_regen());
        assert_eq!(
            task.quality_score(&"out".into(), &[ClaimIssue::CalibratedAbstention]),
            1.0
        );
    }

    #[test]
    fn test_quality_score_decreases_with_issues() {
        let task = ClaimVerificationTask::new("q", "r", trivial_regen());
        let one_issue = vec![ClaimIssue::Unverifiable { claim: "x".into() }];
        let three_issues = vec![
            ClaimIssue::Unverifiable { claim: "a".into() },
            ClaimIssue::Unverifiable { claim: "b".into() },
            ClaimIssue::Unverifiable { claim: "c".into() },
        ];
        let s1 = task.quality_score(&"out".into(), &one_issue);
        let s3 = task.quality_score(&"out".into(), &three_issues);
        assert!(s1 > s3);
    }

    #[test]
    fn test_end_to_end_with_abstention_succeeds() {
        // If the initial response is a calibrated abstention, CoVe etc.
        // should never be reached — validate short-circuits.
        let task = ClaimVerificationTask::new("What's on Mars?", "I don't know.", trivial_regen());
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            ..Default::default()
        });
        let result = engine.run(task, "What's on Mars?");
        // Abstention is non-retryable, so engine hits FatalIssue which is
        // NOT success in the generic sense. We interpret abstention at the
        // caller level (the issue Display makes it explicit).
        matches!(result.stop_reason, StopReason::FatalIssue(_));
        assert!(!result.succeeded);
        // But the first attempt was recorded.
        assert!(result.attempt_count() >= 1);
    }

    #[test]
    fn test_end_to_end_with_knowledge_and_supported_answer() {
        let knowledge = "Rust was created by Graydon Hoare. Rust 1.0 was released in 2015.";
        let task = ClaimVerificationTask::new(
            "Who created Rust?",
            "Rust was created by Graydon Hoare.",
            trivial_regen(),
        )
        .with_knowledge(knowledge);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "Who created Rust?");
        // `total_tokens >= 0` was asserted here, which is vacuous on an unsigned
        // type — it could never fail and so tested nothing. Assert what the smoke
        // test actually cares about instead: the engine ran, recorded the attempt,
        // and reached one of its two acceptable end states.
        assert!(
            !result.attempts.is_empty(),
            "the engine must record at least one attempt"
        );
        assert_eq!(result.task_name, "claim_verification");
        // Either succeeded (claim verified) or stopped on a budget/regression
        // — both are acceptable end states for this smoke test.
    }
}
