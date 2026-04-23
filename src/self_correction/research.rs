//! V100 — Research-citation self-correction.
//!
//! `ResearchCitationTask` retries a research-style response until the
//! citations it contains are valid (resolvable + cover every non-trivial
//! claim). Typical use: a pipeline that generates literature-review text
//! with `[1]`, `[2]`, … style references.

use std::cell::RefCell;

use super::{AttemptRecord, CorrectableTask, Issue, TaskError, TaskOutcome};

/// Issues surfaced by the citation validator.
#[derive(Debug)]
pub enum CitationIssue {
    /// A `[N]` reference was used but no bibliography entry matches.
    DanglingReference { marker: String },
    /// A bibliography entry exists but no text reference uses it.
    UnusedReference { marker: String },
    /// A claim that requires a citation has none attached.
    UnsupportedClaim { claim_excerpt: String },
    /// A citation target could not be resolved (DOI / URL / arXiv not found).
    UnresolvableTarget { marker: String, detail: String },
    /// Citation-coverage ratio below configured threshold.
    LowCoverage { ratio: f64, threshold: f64 },
}

impl std::fmt::Display for CitationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DanglingReference { marker } => {
                write!(f, "dangling reference [{}] — no bibliography entry", marker)
            }
            Self::UnusedReference { marker } => {
                write!(f, "unused reference [{}] — no in-text use", marker)
            }
            Self::UnsupportedClaim { claim_excerpt } => {
                write!(f, "unsupported claim: \"{}\"", claim_excerpt)
            }
            Self::UnresolvableTarget { marker, detail } => {
                write!(f, "unresolvable target [{}]: {}", marker, detail)
            }
            Self::LowCoverage { ratio, threshold } => write!(
                f,
                "citation coverage {:.2} below threshold {:.2}",
                ratio, threshold
            ),
        }
    }
}

impl Issue for CitationIssue {}

/// Validation outcome with raw per-issue kind + message.
#[derive(Debug, Clone, Default)]
pub struct CitationValidationResult {
    /// One entry per issue.
    pub issues: Vec<(&'static str, String)>,
    /// Observed coverage ratio, if computed. Used for `LowCoverage`.
    pub coverage_ratio: Option<f64>,
}

/// Validator closure. Takes the full response and optional bibliography.
pub type CitationValidateFn = Box<dyn FnMut(&str) -> CitationValidationResult + Send>;

/// Regeneration closure.
pub type CitationRegenerateFn =
    Box<dyn FnMut(&str, Option<&str>) -> Option<(String, usize, f64)> + Send>;

/// Research-citation self-correction task.
pub struct ResearchCitationTask {
    user_prompt: String,
    initial_response: Option<String>,
    regenerate_fn: CitationRegenerateFn,
    validate_fn: RefCell<CitationValidateFn>,
    coverage_threshold: f64,
}

impl ResearchCitationTask {
    /// Build.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_response: impl Into<String>,
        regenerate_fn: CitationRegenerateFn,
        validate_fn: CitationValidateFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_response: Some(initial_response.into()),
            regenerate_fn,
            validate_fn: RefCell::new(validate_fn),
            coverage_threshold: 0.7,
        }
    }

    /// Minimum coverage ratio (claims-with-citations / total-claims). Below
    /// this, `LowCoverage` is emitted.
    pub fn with_coverage_threshold(mut self, t: f64) -> Self {
        self.coverage_threshold = t.clamp(0.0, 1.0);
        self
    }
}

impl CorrectableTask for ResearchCitationTask {
    type Output = String;
    type Issue = CitationIssue;

    fn name(&self) -> &str {
        "research_citation"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(r) = self.initial_response.take() {
                return Ok(TaskOutcome {
                    output: r,
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((out, t, c)) => Ok(TaskOutcome {
                output: out,
                tokens_used: t,
                cost_usd: c,
            }),
            None => Err(TaskError::new("regenerate_fn returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        let result = (self.validate_fn.borrow_mut())(output);
        let mut issues: Vec<CitationIssue> = result
            .issues
            .into_iter()
            .map(|(kind, msg)| match kind {
                "dangling" => CitationIssue::DanglingReference { marker: msg },
                "unused" => CitationIssue::UnusedReference { marker: msg },
                "unsupported" => CitationIssue::UnsupportedClaim { claim_excerpt: msg },
                "unresolvable" => CitationIssue::UnresolvableTarget {
                    marker: String::new(),
                    detail: msg,
                },
                _ => CitationIssue::UnsupportedClaim { claim_excerpt: msg },
            })
            .collect();
        if let Some(cov) = result.coverage_ratio {
            if cov < self.coverage_threshold {
                issues.push(CitationIssue::LowCoverage {
                    ratio: cov,
                    threshold: self.coverage_threshold,
                });
            }
        }
        issues
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original request: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous response had {} citation issue(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        lines.push(String::new());
        lines.push("Regenerate the response, applying these rules:".to_string());
        lines.push("  1. Every non-trivial claim must carry an inline [N] citation.".to_string());
        lines.push("  2. Every [N] marker must resolve to a bibliography entry.".to_string());
        lines.push("  3. Remove bibliography entries you no longer cite.".to_string());
        lines.push(
            "  4. Prefer citations you can actually justify; omit rather than fabricate."
                .to_string(),
        );
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            1.0
        } else {
            (1.0 - 0.1 * issues.len() as f64).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SelfCorrectionConfig, SelfCorrectionEngine, StopReason};
    use super::*;

    #[test]
    fn test_citations_clean_first_try() {
        let validate: CitationValidateFn = Box::new(|_| CitationValidationResult::default());
        let regen: CitationRegenerateFn = Box::new(|_p, _f| Some(("ok".into(), 0, 0.0)));
        let task = ResearchCitationTask::new("review X", "Foo et al [1].", regen, validate);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "review X");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
    }

    #[test]
    fn test_dangling_reference_triggers_retry() {
        let validate: CitationValidateFn = Box::new(|out| {
            if out.contains("BIB") {
                CitationValidationResult::default()
            } else {
                CitationValidationResult {
                    issues: vec![("dangling", "3".into())],
                    coverage_ratio: None,
                }
            }
        });
        let regen: CitationRegenerateFn =
            Box::new(|_p, _f| Some(("with [3] and BIB entry [3]".into(), 50, 0.005)));
        let task = ResearchCitationTask::new("x", "bare [3]", regen, validate);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert!(result.succeeded);
        assert!(result.attempt_count() >= 2);
    }

    #[test]
    fn test_low_coverage_issue() {
        let validate: CitationValidateFn = Box::new(|_| CitationValidationResult {
            issues: Vec::new(),
            coverage_ratio: Some(0.4),
        });
        let regen: CitationRegenerateFn = Box::new(|_p, _f| Some(("still low".into(), 50, 0.005)));
        let task =
            ResearchCitationTask::new("x", "initial", regen, validate).with_coverage_threshold(0.7);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 2,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert!(!result.succeeded);
        let has_coverage_issue = result
            .attempts
            .iter()
            .any(|a| a.issues.iter().any(|s| s.contains("coverage")));
        assert!(has_coverage_issue);
    }

    #[test]
    fn test_feedback_mentions_rules() {
        let task = ResearchCitationTask::new(
            "x",
            "y",
            Box::new(|_, _| None),
            Box::new(|_| CitationValidationResult::default()),
        );
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["dangling reference [5]".into()],
            quality_score: 0.5,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 0,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("review Rust macros", &prior);
        assert!(fb.contains("review Rust macros"));
        assert!(fb.contains("[N] marker must resolve"));
        assert!(fb.contains("omit rather than fabricate"));
    }

    #[test]
    fn test_issue_displays() {
        assert!(format!(
            "{}",
            CitationIssue::DanglingReference { marker: "5".into() }
        )
        .contains("[5]"));
        assert!(format!(
            "{}",
            CitationIssue::LowCoverage {
                ratio: 0.3,
                threshold: 0.7
            }
        )
        .contains("0.30"));
    }
}
