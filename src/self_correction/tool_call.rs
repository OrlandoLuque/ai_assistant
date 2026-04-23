//! V100 — Tool-call self-correction.
//!
//! `ToolCallTask` retries a tool invocation whose output fails schema
//! validation or domain constraints. The LLM produces JSON arguments; a
//! validator inspects them before they're dispatched.

use std::cell::RefCell;

use super::{AttemptRecord, CorrectableTask, Issue, TaskError, TaskOutcome};

/// Issues detected by the tool-call validator.
#[derive(Debug)]
pub enum ToolCallIssue {
    /// JSON parse failed.
    InvalidJson { detail: String },
    /// A required field was missing or had the wrong type.
    SchemaViolation { detail: String },
    /// A domain constraint was violated (e.g. "temperature must be 0..=2").
    ConstraintViolation { detail: String },
    /// Unknown tool or method.
    UnknownTool { tool_name: String },
}

impl std::fmt::Display for ToolCallIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson { detail } => write!(f, "invalid JSON: {}", detail),
            Self::SchemaViolation { detail } => write!(f, "schema violation: {}", detail),
            Self::ConstraintViolation { detail } => write!(f, "constraint violation: {}", detail),
            Self::UnknownTool { tool_name } => write!(f, "unknown tool: '{}'", tool_name),
        }
    }
}

impl Issue for ToolCallIssue {}

/// Result of a single tool-call validation pass.
#[derive(Debug, Clone, Default)]
pub struct ToolValidationResult {
    /// Issues found. Empty = valid call.
    pub issues: Vec<String>,
    /// Classification: which category each issue belongs to.
    pub kinds: Vec<&'static str>,
}

impl ToolValidationResult {
    /// No issues.
    pub fn ok() -> Self {
        Self::default()
    }

    /// Helper to build a single-issue failure.
    pub fn fail(kind: &'static str, msg: impl Into<String>) -> Self {
        Self {
            issues: vec![msg.into()],
            kinds: vec![kind],
        }
    }
}

/// Closure that validates raw LLM output (tool-call arguments, usually JSON).
pub type ToolValidateFn = Box<dyn FnMut(&str) -> ToolValidationResult + Send>;

/// Closure that (re)generates tool-call arguments given the prompt and
/// optional feedback.
pub type ToolRegenerateFn =
    Box<dyn FnMut(&str, Option<&str>) -> Option<(String, usize, f64)> + Send>;

/// Tool-call self-correction task.
pub struct ToolCallTask {
    user_prompt: String,
    initial_output: Option<String>,
    regenerate_fn: ToolRegenerateFn,
    validate_fn: RefCell<ToolValidateFn>,
    /// Optional schema/description to include verbatim in the feedback prompt.
    schema_hint: Option<String>,
}

impl ToolCallTask {
    /// Build.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_output: impl Into<String>,
        regenerate_fn: ToolRegenerateFn,
        validate_fn: ToolValidateFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_output: Some(initial_output.into()),
            regenerate_fn,
            validate_fn: RefCell::new(validate_fn),
            schema_hint: None,
        }
    }

    /// Attach a schema / tool description that will be appended to the
    /// feedback prompt so the LLM can see what it should be producing.
    pub fn with_schema_hint(mut self, hint: impl Into<String>) -> Self {
        self.schema_hint = Some(hint.into());
        self
    }
}

impl CorrectableTask for ToolCallTask {
    type Output = String;
    type Issue = ToolCallIssue;

    fn name(&self) -> &str {
        "tool_call"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(o) = self.initial_output.take() {
                return Ok(TaskOutcome {
                    output: o,
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
        result
            .issues
            .into_iter()
            .zip(result.kinds)
            .map(|(msg, kind)| match kind {
                "invalid_json" => ToolCallIssue::InvalidJson { detail: msg },
                "schema" => ToolCallIssue::SchemaViolation { detail: msg },
                "constraint" => ToolCallIssue::ConstraintViolation { detail: msg },
                "unknown_tool" => ToolCallIssue::UnknownTool { tool_name: msg },
                _ => ToolCallIssue::ConstraintViolation { detail: msg },
            })
            .collect()
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original request: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous tool call had {} issue(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        lines.push(String::new());
        lines.push("Regenerate the tool call, applying these rules:".to_string());
        lines.push("  1. Produce valid JSON only — no prose, no code fences.".to_string());
        lines.push("  2. Match the schema exactly: required fields, types, enums.".to_string());
        lines.push("  3. Respect domain constraints mentioned above.".to_string());
        if let Some(ref hint) = self.schema_hint {
            lines.push(String::new());
            lines.push(format!("Tool schema:\n{}", hint));
        }
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            1.0
        } else {
            (1.0 - 0.2 * issues.len() as f64).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{SelfCorrectionConfig, SelfCorrectionEngine, StopReason};
    use super::*;

    #[test]
    fn test_tool_call_valid_first_try() {
        let validate: ToolValidateFn = Box::new(|_out| ToolValidationResult::ok());
        let regen: ToolRegenerateFn = Box::new(|_p, _f| Some(("{}".into(), 0, 0.0)));
        let task = ToolCallTask::new("do X", "{}", regen, validate);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "do X");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
    }

    #[test]
    fn test_tool_call_schema_violation_retries() {
        let count = std::sync::Arc::new(std::sync::Mutex::new(0u32));
        let count_clone = count.clone();
        let validate: ToolValidateFn = Box::new(move |out| {
            *count_clone.lock().unwrap() += 1;
            if out.contains("\"ok\"") {
                ToolValidationResult::ok()
            } else {
                ToolValidationResult::fail("schema", "missing required field 'ok'")
            }
        });
        let regen: ToolRegenerateFn = Box::new(|_p, _f| Some(("{\"ok\": true}".into(), 50, 0.005)));
        let task = ToolCallTask::new("x", "{}", regen, validate);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert!(result.succeeded);
        assert!(result.attempt_count() >= 2);
        assert!(*count.lock().unwrap() >= 2);
    }

    #[test]
    fn test_tool_call_persistent_invalid_json() {
        let validate: ToolValidateFn =
            Box::new(|_| ToolValidationResult::fail("invalid_json", "unexpected token"));
        let regen: ToolRegenerateFn = Box::new(|_p, _f| Some(("not json".into(), 50, 0.005)));
        let task = ToolCallTask::new("x", "still not json", regen, validate);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 2,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert!(!result.succeeded);
    }

    #[test]
    fn test_tool_call_feedback_includes_schema_hint() {
        let task = ToolCallTask::new(
            "x",
            "{}",
            Box::new(|_, _| None),
            Box::new(|_| ToolValidationResult::ok()),
        )
        .with_schema_hint("{\"name\": string, \"count\": int}");
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["schema violation: missing 'name'".into()],
            quality_score: 0.5,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 0,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("do X", &prior);
        assert!(fb.contains("Tool schema"));
        assert!(fb.contains("\"name\": string"));
        assert!(fb.contains("Produce valid JSON only"));
    }

    #[test]
    fn test_issue_display() {
        let issue = ToolCallIssue::UnknownTool {
            tool_name: "weirdo".into(),
        };
        assert!(format!("{}", issue).contains("weirdo"));
        let issue2 = ToolCallIssue::ConstraintViolation {
            detail: "temperature > 2".into(),
        };
        assert!(format!("{}", issue2).contains("temperature"));
    }
}
