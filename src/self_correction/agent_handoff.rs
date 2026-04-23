//! V100 — Agent-handoff self-correction.
//!
//! `AgentHandoffTask` retries an agent's handoff payload until it contains
//! every field the downstream role expects. Typical use: a planner agent
//! passes work to an executor agent; without correction, missing fields
//! cascade into hard-to-debug failures.

use std::cell::RefCell;
use std::collections::HashSet;

use super::{AttemptRecord, CorrectableTask, Issue, TaskError, TaskOutcome};

/// Issues for the handoff validator.
#[derive(Debug)]
pub enum HandoffIssue {
    /// A required field is missing.
    MissingField { field: String },
    /// A field had the wrong type or value.
    InvalidField { field: String, detail: String },
    /// Handoff target role is not recognized.
    UnknownTarget { target: String },
    /// A dependency (prior step) is not satisfied.
    DependencyNotMet { detail: String },
}

impl std::fmt::Display for HandoffIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing required field: '{}'", field),
            Self::InvalidField { field, detail } => {
                write!(f, "invalid field '{}': {}", field, detail)
            }
            Self::UnknownTarget { target } => write!(f, "unknown handoff target: '{}'", target),
            Self::DependencyNotMet { detail } => write!(f, "dependency not met: {}", detail),
        }
    }
}

impl Issue for HandoffIssue {}

/// Handoff validation outcome.
#[derive(Debug, Clone, Default)]
pub struct HandoffValidationResult {
    /// Missing field names.
    pub missing_fields: Vec<String>,
    /// `(field_name, reason)` pairs.
    pub invalid_fields: Vec<(String, String)>,
    /// Unknown target role if detected.
    pub unknown_target: Option<String>,
    /// Free-form dependency errors.
    pub dependency_errors: Vec<String>,
}

impl HandoffValidationResult {
    /// No issues.
    pub fn ok() -> Self {
        Self::default()
    }

    /// Whether this represents a successful validation.
    pub fn is_ok(&self) -> bool {
        self.missing_fields.is_empty()
            && self.invalid_fields.is_empty()
            && self.unknown_target.is_none()
            && self.dependency_errors.is_empty()
    }
}

/// Validator closure.
pub type HandoffValidateFn = Box<dyn FnMut(&str) -> HandoffValidationResult + Send>;

/// Regenerator closure.
pub type HandoffRegenerateFn =
    Box<dyn FnMut(&str, Option<&str>) -> Option<(String, usize, f64)> + Send>;

/// Agent-handoff self-correction task.
pub struct AgentHandoffTask {
    user_prompt: String,
    initial_payload: Option<String>,
    regenerate_fn: HandoffRegenerateFn,
    validate_fn: RefCell<HandoffValidateFn>,
    required_fields: Vec<String>,
    valid_targets: HashSet<String>,
}

impl AgentHandoffTask {
    /// Build.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_payload: impl Into<String>,
        regenerate_fn: HandoffRegenerateFn,
        validate_fn: HandoffValidateFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_payload: Some(initial_payload.into()),
            regenerate_fn,
            validate_fn: RefCell::new(validate_fn),
            required_fields: Vec::new(),
            valid_targets: HashSet::new(),
        }
    }

    /// List of required field names used only for the feedback prompt
    /// (the validator is still the source of truth).
    pub fn with_required_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.required_fields = fields.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set of known target roles — included in feedback to help the LLM.
    pub fn with_valid_targets<I, S>(mut self, targets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.valid_targets = targets.into_iter().map(|s| s.into()).collect();
        self
    }
}

impl CorrectableTask for AgentHandoffTask {
    type Output = String;
    type Issue = HandoffIssue;

    fn name(&self) -> &str {
        "agent_handoff"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(p) = self.initial_payload.take() {
                return Ok(TaskOutcome {
                    output: p,
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((p, t, c)) => Ok(TaskOutcome {
                output: p,
                tokens_used: t,
                cost_usd: c,
            }),
            None => Err(TaskError::new("regenerate_fn returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        let r = (self.validate_fn.borrow_mut())(output);
        let mut issues = Vec::new();
        for f in r.missing_fields {
            issues.push(HandoffIssue::MissingField { field: f });
        }
        for (f, d) in r.invalid_fields {
            issues.push(HandoffIssue::InvalidField {
                field: f,
                detail: d,
            });
        }
        if let Some(t) = r.unknown_target {
            issues.push(HandoffIssue::UnknownTarget { target: t });
        }
        for d in r.dependency_errors {
            issues.push(HandoffIssue::DependencyNotMet { detail: d });
        }
        issues
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original request: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous handoff had {} issue(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        lines.push(String::new());
        lines.push("Regenerate the handoff payload, applying these rules:".to_string());
        lines.push("  1. Include every required field with the correct type.".to_string());
        lines.push("  2. Pick a valid target role from the list below.".to_string());
        lines
            .push("  3. Ensure all dependencies of the downstream task are satisfied.".to_string());
        if !self.required_fields.is_empty() {
            lines.push(String::new());
            lines.push(format!(
                "Required fields: {}",
                self.required_fields.join(", ")
            ));
        }
        if !self.valid_targets.is_empty() {
            let mut ts: Vec<&String> = self.valid_targets.iter().collect();
            ts.sort();
            lines.push(format!(
                "Valid targets: {}",
                ts.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            1.0
        } else {
            (1.0 - 0.15 * issues.len() as f64).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SelfCorrectionConfig, SelfCorrectionEngine, StopReason};
    use super::*;

    #[test]
    fn test_handoff_clean() {
        let v: HandoffValidateFn = Box::new(|_| HandoffValidationResult::ok());
        let r: HandoffRegenerateFn = Box::new(|_p, _f| Some(("ok".into(), 0, 0.0)));
        let task = AgentHandoffTask::new("hand off", "{}", r, v);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "hand off");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
    }

    #[test]
    fn test_handoff_missing_field_retry() {
        let v: HandoffValidateFn = Box::new(|out| {
            if out.contains("\"goal\"") {
                HandoffValidationResult::ok()
            } else {
                HandoffValidationResult {
                    missing_fields: vec!["goal".into()],
                    ..Default::default()
                }
            }
        });
        let r: HandoffRegenerateFn = Box::new(|_p, _f| {
            Some((
                "{\"goal\": \"build it\", \"target\": \"executor\"}".into(),
                50,
                0.005,
            ))
        });
        let task = AgentHandoffTask::new("build", "{}", r, v)
            .with_required_fields(["goal", "target"])
            .with_valid_targets(["planner", "executor"]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "build");
        assert!(result.succeeded);
        assert!(result.attempt_count() >= 2);
    }

    #[test]
    fn test_handoff_unknown_target() {
        let v: HandoffValidateFn = Box::new(|_| HandoffValidationResult {
            unknown_target: Some("wanderer".into()),
            ..Default::default()
        });
        let r: HandoffRegenerateFn = Box::new(|_p, _f| Some(("still wrong".into(), 50, 0.005)));
        let task =
            AgentHandoffTask::new("x", "initial", r, v).with_valid_targets(["planner", "executor"]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 2,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert!(!result.succeeded);
        assert!(result.attempts[0]
            .issues
            .iter()
            .any(|s| s.contains("unknown handoff target")));
    }

    #[test]
    fn test_feedback_lists_required_and_valid() {
        let task = AgentHandoffTask::new(
            "x",
            "y",
            Box::new(|_, _| None),
            Box::new(|_| HandoffValidationResult::ok()),
        )
        .with_required_fields(["goal", "ttl"])
        .with_valid_targets(["planner", "executor"]);
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["missing required field: 'goal'".into()],
            quality_score: 0.5,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 0,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("coordinate", &prior);
        assert!(fb.contains("coordinate"));
        assert!(fb.contains("Required fields"));
        assert!(fb.contains("goal"));
        assert!(fb.contains("Valid targets"));
        assert!(fb.contains("planner"));
    }

    #[test]
    fn test_issue_displays() {
        let i = HandoffIssue::MissingField {
            field: "goal".into(),
        };
        assert!(format!("{}", i).contains("goal"));
        let i2 = HandoffIssue::DependencyNotMet {
            detail: "step 1 unfinished".into(),
        };
        assert!(format!("{}", i2).contains("step 1"));
    }
}
