//! Self-correction engine — orchestrator with 4-dim budget.

use std::time::Instant;

use super::{
    sanitize_for_feedback, AttemptRecord, CorrectableTask, Issue, SelfCorrectionConfig,
    SelfCorrectionResult, StopReason,
};

/// The orchestrator. Holds config; `run` is parametric over the task.
#[derive(Debug, Clone)]
pub struct SelfCorrectionEngine {
    config: SelfCorrectionConfig,
}

impl SelfCorrectionEngine {
    /// Construct an engine with the given config.
    pub fn new(config: SelfCorrectionConfig) -> Self {
        Self { config }
    }

    /// Construct with default config.
    pub fn with_defaults() -> Self {
        Self::new(SelfCorrectionConfig::default())
    }

    /// Access the config.
    pub fn config(&self) -> &SelfCorrectionConfig {
        &self.config
    }

    /// Run the correction loop on a task.
    ///
    /// `user_intent` is the original user prompt / task description; it is
    /// passed to `task.build_feedback` so the feedback template can remind
    /// the LLM what the user asked for.
    pub fn run<T: CorrectableTask>(
        &self,
        mut task: T,
        user_intent: &str,
    ) -> SelfCorrectionResult<T::Output> {
        let overall_start = Instant::now();
        let mut attempts: Vec<AttemptRecord> = Vec::new();
        let mut best_output: Option<T::Output> = None;
        let mut total_tokens: usize = 0;
        let mut total_cost: f64 = 0.0;
        let mut last_quality: Option<f64> = None;
        let mut feedback: Option<String> = None;

        let task_name = task.name().to_string();

        for attempt_idx in 0..self.config.max_attempts {
            let attempt_start = Instant::now();

            // ── Execute ────────────────────────────────────────────────
            let exec_result = task.execute(feedback.as_deref());
            let elapsed_ms = attempt_start.elapsed().as_millis() as u64;

            let (output, attempt_tokens, attempt_cost) = match exec_result {
                Ok(super::TaskOutcome {
                    output,
                    tokens_used,
                    cost_usd,
                }) => {
                    total_tokens += tokens_used;
                    total_cost += cost_usd;
                    (output, tokens_used, cost_usd)
                }
                Err(e) => {
                    total_tokens += e.tokens_used;
                    total_cost += e.cost_usd;
                    attempts.push(AttemptRecord {
                        attempt_num: attempt_idx + 1,
                        issues: vec![format!("Task execution failed: {}", e.reason)],
                        quality_score: 0.0,
                        tokens_used: e.tokens_used,
                        cost_usd: e.cost_usd,
                        elapsed_ms,
                        feedback_given: feedback.clone(),
                        succeeded: false,
                    });
                    return self.finalize(
                        best_output,
                        attempts,
                        StopReason::RegenerationFailed,
                        overall_start,
                        total_tokens,
                        total_cost,
                        task_name,
                    );
                }
            };

            // ── Validate ───────────────────────────────────────────────
            let issues = task.validate(&output);
            let quality = task.quality_score(&output, &issues);
            let issues_str: Vec<String> = issues.iter().map(|i| i.to_string()).collect();
            let has_issues = !issues.is_empty();

            // Check for non-retryable issue.
            let fatal = issues.iter().find(|i| !i.is_retryable());
            if let Some(f) = fatal {
                let reason_text = f.to_string();
                attempts.push(AttemptRecord {
                    attempt_num: attempt_idx + 1,
                    issues: issues_str,
                    quality_score: quality,
                    tokens_used: attempt_tokens,
                    cost_usd: attempt_cost,
                    elapsed_ms,
                    feedback_given: feedback.clone(),
                    succeeded: false,
                });
                best_output = Some(output);
                return self.finalize(
                    best_output,
                    attempts,
                    StopReason::FatalIssue(reason_text),
                    overall_start,
                    total_tokens,
                    total_cost,
                    task_name,
                );
            }

            // Success?
            if !has_issues {
                attempts.push(AttemptRecord {
                    attempt_num: attempt_idx + 1,
                    issues: Vec::new(),
                    quality_score: quality,
                    tokens_used: attempt_tokens,
                    cost_usd: attempt_cost,
                    elapsed_ms,
                    feedback_given: feedback.clone(),
                    succeeded: true,
                });
                best_output = Some(output);
                return self.finalize(
                    best_output,
                    attempts,
                    StopReason::AllPassed,
                    overall_start,
                    total_tokens,
                    total_cost,
                    task_name,
                );
            }

            // Regression / no-improvement detection (only from the 2nd attempt).
            if let Some(prev_q) = last_quality {
                let delta = quality - prev_q;
                if delta < -self.config.min_improvement {
                    attempts.push(AttemptRecord {
                        attempt_num: attempt_idx + 1,
                        issues: issues_str,
                        quality_score: quality,
                        tokens_used: attempt_tokens,
                        cost_usd: attempt_cost,
                        elapsed_ms,
                        feedback_given: feedback.clone(),
                        succeeded: false,
                    });
                    best_output = Some(output);
                    return self.finalize(
                        best_output,
                        attempts,
                        StopReason::QualityRegression,
                        overall_start,
                        total_tokens,
                        total_cost,
                        task_name,
                    );
                }
                if delta.abs() < self.config.min_improvement {
                    attempts.push(AttemptRecord {
                        attempt_num: attempt_idx + 1,
                        issues: issues_str,
                        quality_score: quality,
                        tokens_used: attempt_tokens,
                        cost_usd: attempt_cost,
                        elapsed_ms,
                        feedback_given: feedback.clone(),
                        succeeded: false,
                    });
                    best_output = Some(output);
                    return self.finalize(
                        best_output,
                        attempts,
                        StopReason::NoImprovement,
                        overall_start,
                        total_tokens,
                        total_cost,
                        task_name,
                    );
                }
            }

            // Record this attempt before deciding about the next one.
            attempts.push(AttemptRecord {
                attempt_num: attempt_idx + 1,
                issues: issues_str,
                quality_score: quality,
                tokens_used: attempt_tokens,
                cost_usd: attempt_cost,
                elapsed_ms,
                feedback_given: feedback.clone(),
                succeeded: false,
            });
            last_quality = Some(quality);
            best_output = Some(output);

            // ── Budget checks (before building feedback / next attempt) ──
            if total_tokens >= self.config.max_total_tokens {
                return self.finalize(
                    best_output,
                    attempts,
                    StopReason::TokenBudgetExhausted,
                    overall_start,
                    total_tokens,
                    total_cost,
                    task_name,
                );
            }
            if total_cost >= self.config.max_total_cost_usd {
                return self.finalize(
                    best_output,
                    attempts,
                    StopReason::CostBudgetExhausted,
                    overall_start,
                    total_tokens,
                    total_cost,
                    task_name,
                );
            }
            let total_elapsed = overall_start.elapsed().as_millis() as u64;
            if total_elapsed >= self.config.max_total_time_ms {
                return self.finalize(
                    best_output,
                    attempts,
                    StopReason::TimeBudgetExhausted,
                    overall_start,
                    total_tokens,
                    total_cost,
                    task_name,
                );
            }

            // If this was the last attempt, don't build feedback.
            if attempt_idx + 1 >= self.config.max_attempts {
                break;
            }

            // ── Build feedback for next attempt ────────────────────────
            let fb = task.build_feedback(user_intent, &attempts);
            feedback = Some(if self.config.sanitize_feedback {
                // Sanitize any embedded prior-response segments inside the
                // task-produced feedback by wrapping the whole thing as a
                // boundary. The task is expected to already include markers
                // like "Prior response:" — but we defensively add delimiters
                // so downstream LLMs treat the whole block as data, not
                // instructions.
                sanitize_for_feedback(&fb, self.config.sanitize_max_chars)
            } else {
                fb
            });
        }

        self.finalize(
            best_output,
            attempts,
            StopReason::MaxAttempts,
            overall_start,
            total_tokens,
            total_cost,
            task_name,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize<O>(
        &self,
        final_output: Option<O>,
        attempts: Vec<AttemptRecord>,
        stop_reason: StopReason,
        overall_start: Instant,
        total_tokens: usize,
        total_cost: f64,
        task_name: String,
    ) -> SelfCorrectionResult<O> {
        let total_elapsed_ms = overall_start.elapsed().as_millis() as u64;
        let succeeded = stop_reason.is_success();
        let result = SelfCorrectionResult {
            final_output,
            attempts,
            succeeded,
            stop_reason,
            total_tokens,
            total_cost_usd: total_cost,
            total_elapsed_ms,
            task_name,
        };

        // Optional ledger append. Failures here don't propagate — the engine
        // does not want to hide a successful correction because of a disk
        // error, and the stop reason is already in the in-memory result.
        if let Some(ref path) = self.config.ledger_path {
            if let Ok(ledger) = super::CorrectionLedger::open(path) {
                let _ = ledger.append(&result);
            }
        }

        result
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::SelfCorrectionEngine;
    use std::cell::RefCell;

    /// Issue type for mock tests.
    #[derive(Debug)]
    struct MockIssue {
        msg: String,
        retryable: bool,
    }

    impl std::fmt::Display for MockIssue {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl Issue for MockIssue {
        fn is_retryable(&self) -> bool {
            self.retryable
        }
    }

    /// A mock task that returns a pre-recorded sequence of (output, issues,
    /// quality) triples, one per attempt.
    struct MockTask {
        outputs: RefCell<Vec<(String, Vec<MockIssue>, f64, usize, f64)>>,
    }

    impl MockTask {
        fn new(outputs: Vec<(String, Vec<MockIssue>, f64, usize, f64)>) -> Self {
            Self {
                outputs: RefCell::new(outputs),
            }
        }
    }

    impl CorrectableTask for MockTask {
        type Output = String;
        type Issue = MockIssue;

        fn name(&self) -> &str {
            "mock"
        }

        fn execute(
            &mut self,
            _feedback: Option<&str>,
        ) -> Result<TaskOutcome<Self::Output>, TaskError> {
            let (out, _issues, _q, tokens, cost) = self.outputs.borrow_mut().remove(0);
            Ok(TaskOutcome {
                output: out,
                tokens_used: tokens,
                cost_usd: cost,
            })
        }

        fn validate(&self, _output: &Self::Output) -> Vec<Self::Issue> {
            // Peek next issues by cloning from the same position as last
            // remove — but since remove already popped, use a trick: store
            // issues alongside output and re-use. Simpler: issues are keyed
            // by the current state of outputs before removal.
            // This method is called AFTER execute(), so the issues we want
            // are for the most recent attempt. We stored them in `outputs[0]`
            // before `execute` popped — so they're gone. Use a side-channel:
            // the `MockTask` is designed so issues/quality are pre-computed
            // and we return them from `validate` by re-inspecting state.
            //
            // For simplicity, the mock task in tests uses a separate wrapper
            // below (`PlannedMockTask`) that decouples execute from validate.
            vec![]
        }

        fn build_feedback(&self, _user_intent: &str, _prior_attempts: &[AttemptRecord]) -> String {
            String::from("please fix")
        }

        fn quality_score(&self, _output: &Self::Output, _issues: &[Self::Issue]) -> f64 {
            1.0
        }
    }

    /// A mock task with explicit per-attempt script. Each entry:
    /// (output, issues, quality, tokens, cost).
    struct PlannedMockTask {
        script: Vec<(String, Vec<MockIssue>, f64, usize, f64)>,
        last_issues: Vec<MockIssue>,
        last_quality: f64,
    }

    impl PlannedMockTask {
        fn new(script: Vec<(String, Vec<MockIssue>, f64, usize, f64)>) -> Self {
            Self {
                script,
                last_issues: Vec::new(),
                last_quality: 0.0,
            }
        }
    }

    impl CorrectableTask for PlannedMockTask {
        type Output = String;
        type Issue = MockIssue;

        fn name(&self) -> &str {
            "planned_mock"
        }

        fn execute(
            &mut self,
            _feedback: Option<&str>,
        ) -> Result<TaskOutcome<Self::Output>, TaskError> {
            if self.script.is_empty() {
                return Err(TaskError::new("script exhausted"));
            }
            let (out, issues, quality, tokens, cost) = self.script.remove(0);
            self.last_issues = issues;
            self.last_quality = quality;
            Ok(TaskOutcome {
                output: out,
                tokens_used: tokens,
                cost_usd: cost,
            })
        }

        fn validate(&self, _output: &Self::Output) -> Vec<Self::Issue> {
            self.last_issues
                .iter()
                .map(|i| MockIssue {
                    msg: i.msg.clone(),
                    retryable: i.retryable,
                })
                .collect()
        }

        fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
            format!(
                "intent={} attempts={} last_issues={:?}",
                user_intent,
                prior_attempts.len(),
                self.last_issues.iter().map(|i| &i.msg).collect::<Vec<_>>()
            )
        }

        fn quality_score(&self, _output: &Self::Output, _issues: &[Self::Issue]) -> f64 {
            self.last_quality
        }
    }

    fn issue_retryable(msg: &str) -> MockIssue {
        MockIssue {
            msg: msg.into(),
            retryable: true,
        }
    }

    fn issue_fatal(msg: &str) -> MockIssue {
        MockIssue {
            msg: msg.into(),
            retryable: false,
        }
    }

    #[test]
    fn test_happy_path_passes_first_try() {
        let task = PlannedMockTask::new(vec![("ok".into(), vec![], 1.0, 100, 0.01)]);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "say something");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
        assert_eq!(result.attempt_count(), 1);
        assert_eq!(result.final_output.as_deref(), Some("ok"));
    }

    #[test]
    fn test_fails_once_then_succeeds() {
        let task = PlannedMockTask::new(vec![
            ("bad".into(), vec![issue_retryable("wrong")], 0.3, 100, 0.01),
            ("good".into(), vec![], 0.9, 100, 0.01),
        ]);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "say something");
        assert!(result.succeeded);
        assert_eq!(result.attempt_count(), 2);
        assert_eq!(result.final_output.as_deref(), Some("good"));
    }

    #[test]
    fn test_budget_exhausted_max_attempts() {
        let task = PlannedMockTask::new(vec![
            ("x".into(), vec![issue_retryable("a")], 0.3, 100, 0.01),
            ("y".into(), vec![issue_retryable("b")], 0.5, 100, 0.01),
            ("z".into(), vec![issue_retryable("c")], 0.7, 100, 0.01),
        ]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "say something");
        assert!(!result.succeeded);
        assert_eq!(result.stop_reason, StopReason::MaxAttempts);
        assert_eq!(result.attempt_count(), 3);
    }

    #[test]
    fn test_fatal_issue_stops_immediately() {
        let task = PlannedMockTask::new(vec![(
            "contains_pii".into(),
            vec![issue_fatal("PII leaked")],
            0.0,
            100,
            0.01,
        )]);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "any");
        assert!(!result.succeeded);
        matches!(result.stop_reason, StopReason::FatalIssue(_));
        assert_eq!(result.attempt_count(), 1);
    }

    #[test]
    fn test_no_improvement_early_stop() {
        let task = PlannedMockTask::new(vec![
            ("a".into(), vec![issue_retryable("bad")], 0.5, 100, 0.01),
            (
                "b".into(),
                vec![issue_retryable("still bad")],
                0.5,
                100,
                0.01,
            ),
            (
                "c".into(),
                vec![issue_retryable("still bad")],
                0.5,
                100,
                0.01,
            ),
        ]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            min_improvement: 0.1,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert_eq!(result.stop_reason, StopReason::NoImprovement);
        assert_eq!(result.attempt_count(), 2);
    }

    #[test]
    fn test_quality_regression() {
        let task = PlannedMockTask::new(vec![
            ("a".into(), vec![issue_retryable("bad")], 0.7, 100, 0.01),
            ("b".into(), vec![issue_retryable("worse")], 0.3, 100, 0.01),
            ("c".into(), vec![issue_retryable("worse")], 0.3, 100, 0.01),
        ]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            min_improvement: 0.1,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert_eq!(result.stop_reason, StopReason::QualityRegression);
    }

    #[test]
    fn test_token_budget_exhausted() {
        let task = PlannedMockTask::new(vec![
            ("a".into(), vec![issue_retryable("bad")], 0.3, 10_000, 0.01),
            ("b".into(), vec![issue_retryable("bad")], 0.6, 10_000, 0.01),
            ("c".into(), vec![issue_retryable("bad")], 0.9, 10_000, 0.01),
        ]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_total_tokens: 15_000,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert_eq!(result.stop_reason, StopReason::TokenBudgetExhausted);
        assert_eq!(result.attempt_count(), 2);
    }

    #[test]
    fn test_cost_budget_exhausted() {
        let task = PlannedMockTask::new(vec![
            ("a".into(), vec![issue_retryable("bad")], 0.3, 100, 0.60),
            ("b".into(), vec![issue_retryable("bad")], 0.6, 100, 0.60),
            ("c".into(), vec![issue_retryable("bad")], 0.9, 100, 0.60),
        ]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_total_cost_usd: 1.0,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert_eq!(result.stop_reason, StopReason::CostBudgetExhausted);
    }

    #[test]
    fn test_regeneration_failure_returns_best_so_far() {
        struct FailingTask {
            attempt: usize,
        }
        impl CorrectableTask for FailingTask {
            type Output = String;
            type Issue = MockIssue;
            fn name(&self) -> &str {
                "failing"
            }
            fn execute(
                &mut self,
                _f: Option<&str>,
            ) -> Result<TaskOutcome<Self::Output>, TaskError> {
                self.attempt += 1;
                if self.attempt == 1 {
                    Ok(TaskOutcome {
                        output: "partial".into(),
                        tokens_used: 50,
                        cost_usd: 0.005,
                    })
                } else {
                    Err(TaskError {
                        reason: "llm timed out".into(),
                        tokens_used: 10,
                        cost_usd: 0.001,
                    })
                }
            }
            fn validate(&self, _o: &Self::Output) -> Vec<Self::Issue> {
                vec![MockIssue {
                    msg: "needs work".into(),
                    retryable: true,
                }]
            }
            fn build_feedback(&self, _intent: &str, _prior_attempts: &[AttemptRecord]) -> String {
                "fix it".into()
            }
            fn quality_score(&self, _o: &Self::Output, _i: &[Self::Issue]) -> f64 {
                0.5
            }
        }
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(FailingTask { attempt: 0 }, "x");
        assert_eq!(result.stop_reason, StopReason::RegenerationFailed);
        assert_eq!(result.final_output.as_deref(), Some("partial"));
    }

    #[test]
    fn test_feedback_is_passed_to_next_attempt() {
        use std::sync::{Arc, Mutex};
        let log: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));

        struct LoggingTask {
            log: Arc<Mutex<Vec<Option<String>>>>,
            attempts: Vec<(String, Vec<MockIssue>, f64)>,
            last_issues: Vec<MockIssue>,
            last_quality: f64,
        }
        impl CorrectableTask for LoggingTask {
            type Output = String;
            type Issue = MockIssue;
            fn name(&self) -> &str {
                "log"
            }
            fn execute(
                &mut self,
                feedback: Option<&str>,
            ) -> Result<TaskOutcome<Self::Output>, TaskError> {
                self.log
                    .lock()
                    .unwrap()
                    .push(feedback.map(|s| s.to_string()));
                let (out, issues, quality) = self.attempts.remove(0);
                self.last_issues = issues;
                self.last_quality = quality;
                Ok(TaskOutcome {
                    output: out,
                    tokens_used: 10,
                    cost_usd: 0.001,
                })
            }
            fn validate(&self, _o: &Self::Output) -> Vec<Self::Issue> {
                self.last_issues
                    .iter()
                    .map(|i| MockIssue {
                        msg: i.msg.clone(),
                        retryable: i.retryable,
                    })
                    .collect()
            }
            fn build_feedback(&self, _intent: &str, _prior_attempts: &[AttemptRecord]) -> String {
                "CORRECT_THIS".into()
            }
            fn quality_score(&self, _o: &Self::Output, _i: &[Self::Issue]) -> f64 {
                self.last_quality
            }
        }

        let task = LoggingTask {
            log: log.clone(),
            attempts: vec![
                ("bad".into(), vec![issue_retryable("wrong")], 0.3),
                ("good".into(), vec![], 0.9),
            ],
            last_issues: Vec::new(),
            last_quality: 0.0,
        };
        let engine = SelfCorrectionEngine::with_defaults();
        let _result = engine.run(task, "say something");
        let log_data = log.lock().unwrap();
        assert_eq!(log_data.len(), 2);
        assert!(log_data[0].is_none());
        assert!(log_data[1].is_some());
        // The feedback should contain the sanitizer delimiters.
        assert!(log_data[1].as_ref().unwrap().contains("<<<PRIOR_RESPONSE"));
        assert!(log_data[1].as_ref().unwrap().contains("CORRECT_THIS"));
    }

    #[test]
    fn test_zero_attempts_returns_immediately() {
        let task = PlannedMockTask::new(vec![]);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 0,
            ..Default::default()
        });
        let result = engine.run(task, "x");
        assert_eq!(result.stop_reason, StopReason::MaxAttempts);
        assert_eq!(result.attempt_count(), 0);
    }

    #[test]
    fn test_total_resources_aggregate() {
        let task = PlannedMockTask::new(vec![
            ("x".into(), vec![issue_retryable("a")], 0.3, 100, 0.01),
            ("y".into(), vec![], 0.9, 200, 0.02),
        ]);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "x");
        assert_eq!(result.total_tokens, 300);
        assert!((result.total_cost_usd - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_task_name_recorded() {
        let task = PlannedMockTask::new(vec![("ok".into(), vec![], 1.0, 10, 0.001)]);
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "x");
        assert_eq!(result.task_name, "planned_mock");
    }
}
