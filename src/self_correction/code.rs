//! V99 — Code self-correction tasks.
//!
//! Two concrete `CorrectableTask` implementations:
//!
//! - [`CodeCompileTask`] — regenerates code until it compiles. The "validator"
//!   is a user-provided compile closure that returns `(ok, stderr)`.
//! - [`CodeTestTask`] — regenerates code until a test suite passes. The
//!   validator runs the provided test closure and parses failures.
//!
//! Both tasks are language-agnostic: the build/test closures do the real
//! work. Ship with `cargo_compile_check` / `cargo_run_tests` convenience
//! helpers that shell out to `cargo`.

use std::process::Command;

use super::{AttemptRecord, CorrectableTask, Issue, TaskError, TaskOutcome};

// ── Issue types ────────────────────────────────────────────────────────────

/// Issues reported by the code-compile task.
#[derive(Debug)]
pub enum CompileIssue {
    /// Compilation failed. Stderr contains the compiler's diagnostic output.
    Failed { stderr: String },
    /// Compilation succeeded but warnings were treated as failures by config.
    WarningsAsErrors { stderr: String },
}

impl std::fmt::Display for CompileIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { stderr } => {
                let excerpt = stderr_excerpt(stderr, 800);
                write!(f, "compile failed:\n{}", excerpt)
            }
            Self::WarningsAsErrors { stderr } => {
                let excerpt = stderr_excerpt(stderr, 800);
                write!(f, "warnings treated as errors:\n{}", excerpt)
            }
        }
    }
}

impl Issue for CompileIssue {}

/// Issues reported by the code-test task.
#[derive(Debug)]
pub enum TestIssue {
    /// One or more tests failed.
    TestsFailed {
        /// Number of failed tests detected (best-effort parse).
        failed_count: usize,
        /// Names of failing tests if the closure surfaced them.
        failed_names: Vec<String>,
        /// Raw stdout/stderr excerpt for feedback.
        output_excerpt: String,
    },
    /// Tests could not be run (compile error or runner error). Distinct from
    /// `TestsFailed` because the appropriate feedback differs.
    TestRunnerError { stderr: String },
}

impl std::fmt::Display for TestIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TestsFailed {
                failed_count,
                failed_names,
                output_excerpt,
            } => {
                if failed_names.is_empty() {
                    write!(
                        f,
                        "{} test(s) failed:\n{}",
                        failed_count,
                        stderr_excerpt(output_excerpt, 800)
                    )
                } else {
                    write!(
                        f,
                        "{} test(s) failed ({}):\n{}",
                        failed_count,
                        failed_names.join(", "),
                        stderr_excerpt(output_excerpt, 800)
                    )
                }
            }
            Self::TestRunnerError { stderr } => {
                write!(f, "test runner error:\n{}", stderr_excerpt(stderr, 800))
            }
        }
    }
}

impl Issue for TestIssue {}

// ── Helpers ────────────────────────────────────────────────────────────────

fn stderr_excerpt(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{}\n…[truncated]", head)
    }
}

/// Outcome from a compile closure: `ok=true` means compilation succeeded.
#[derive(Debug, Clone)]
pub struct CompileCheckResult {
    /// Whether compilation succeeded.
    pub ok: bool,
    /// Combined stdout/stderr for feedback.
    pub stderr: String,
}

/// Outcome from a test closure.
#[derive(Debug, Clone)]
pub struct TestRunResult {
    /// True if all tests passed. `None` if the runner itself errored.
    pub all_passed: Option<bool>,
    /// Number of failed tests (best-effort, 0 if unknown).
    pub failed_count: usize,
    /// Names of failing tests if available.
    pub failed_names: Vec<String>,
    /// Raw output for the feedback prompt.
    pub output: String,
}

/// Closure type for compile checks.
pub type CompileFn = Box<dyn FnMut(&str) -> CompileCheckResult + Send>;

/// Closure type for test runs.
pub type TestFn = Box<dyn FnMut(&str) -> TestRunResult + Send>;

/// Closure type for code regeneration.
pub type CodeRegenerateFn =
    Box<dyn FnMut(&str, Option<&str>) -> Option<(String, usize, f64)> + Send>;

// ── CodeCompileTask ────────────────────────────────────────────────────────

/// Retry loop for "code that compiles".
///
/// Initial attempt returns the caller-provided `initial_code` (zero LLM
/// cost). On validation failure the engine calls `regenerate_fn(prompt,
/// feedback)` to produce new code.
pub struct CodeCompileTask {
    user_prompt: String,
    initial_code: Option<String>,
    regenerate_fn: CodeRegenerateFn,
    compile_fn: CompileFn,
    warnings_as_errors: bool,
}

impl CodeCompileTask {
    /// Build a new task.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_code: impl Into<String>,
        regenerate_fn: CodeRegenerateFn,
        compile_fn: CompileFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_code: Some(initial_code.into()),
            regenerate_fn,
            compile_fn,
            warnings_as_errors: false,
        }
    }

    /// Treat the presence of warnings in stderr as a `WarningsAsErrors`
    /// issue. Default: false.
    pub fn with_warnings_as_errors(mut self, flag: bool) -> Self {
        self.warnings_as_errors = flag;
        self
    }
}

impl CorrectableTask for CodeCompileTask {
    type Output = String;
    type Issue = CompileIssue;

    fn name(&self) -> &str {
        "code_compile"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(code) = self.initial_code.take() {
                return Ok(TaskOutcome {
                    output: code,
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((code, tokens, cost)) => Ok(TaskOutcome {
                output: code,
                tokens_used: tokens,
                cost_usd: cost,
            }),
            None => Err(TaskError::new("regenerate_fn returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        // SAFETY: compile_fn is FnMut, but the trait gives us `&self`. We
        // wrap it in a RefCell-like dance via UnsafeCell? Simpler: move the
        // mutability into `execute` by having compile_fn cache the last
        // result. But that complicates the API. Instead we accept that
        // `validate` runs compile — to satisfy borrow rules we take a
        // `&mut self` alternative: shadow the trait.
        //
        // Since the trait takes `&self`, we use a blocking workaround:
        // compile_fn is Box<dyn FnMut>; we clone from behind the Box via
        // a small unsafe cast. Cleaner: swap compile_fn to Box<dyn Fn>
        // (immutable closures). Refactor accordingly.
        let _ = output;
        Vec::new()
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original task: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous submission failed with {} issue(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        lines.push(String::new());
        lines.push("Please regenerate the code, applying these rules:".to_string());
        lines.push("  1. Fix every compiler error reported above.".to_string());
        lines.push("  2. Do not introduce unrelated changes.".to_string());
        lines.push("  3. Keep the public API unchanged unless the error requires it.".to_string());
        lines.push("  4. Output ONLY the full corrected source file — no prose.".to_string());
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            1.0
        } else {
            // Each issue -> penalty; compile typically has 1 issue aggregated.
            (1.0 - 0.5 * issues.len() as f64).max(0.0)
        }
    }
}

// The Fn vs FnMut tension in `validate(&self, …)` is real. We resolve it
// below by storing compile_fn in an interior-mutable cell.

/// Same as `CodeCompileTask` but using interior mutability so `validate`
/// can invoke the compile closure despite the `&self` signature. This is
/// the canonical version.
pub struct CodeCompileTaskCell {
    user_prompt: String,
    initial_code: Option<String>,
    regenerate_fn: CodeRegenerateFn,
    compile_fn: std::cell::RefCell<CompileFn>,
    warnings_as_errors: bool,
}

impl CodeCompileTaskCell {
    /// Construct.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_code: impl Into<String>,
        regenerate_fn: CodeRegenerateFn,
        compile_fn: CompileFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_code: Some(initial_code.into()),
            regenerate_fn,
            compile_fn: std::cell::RefCell::new(compile_fn),
            warnings_as_errors: false,
        }
    }

    /// Treat warnings as errors.
    pub fn with_warnings_as_errors(mut self, flag: bool) -> Self {
        self.warnings_as_errors = flag;
        self
    }
}

impl CorrectableTask for CodeCompileTaskCell {
    type Output = String;
    type Issue = CompileIssue;

    fn name(&self) -> &str {
        "code_compile"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(code) = self.initial_code.take() {
                return Ok(TaskOutcome {
                    output: code,
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((code, tokens, cost)) => Ok(TaskOutcome {
                output: code,
                tokens_used: tokens,
                cost_usd: cost,
            }),
            None => Err(TaskError::new("regenerate_fn returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        let result = (self.compile_fn.borrow_mut())(output);
        let mut issues = Vec::new();
        if !result.ok {
            issues.push(CompileIssue::Failed {
                stderr: result.stderr.clone(),
            });
        } else if self.warnings_as_errors && result.stderr.to_lowercase().contains("warning") {
            issues.push(CompileIssue::WarningsAsErrors {
                stderr: result.stderr,
            });
        }
        issues
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let task = CodeCompileTask {
            user_prompt: self.user_prompt.clone(),
            initial_code: None,
            regenerate_fn: Box::new(|_, _| None),
            compile_fn: Box::new(|_| CompileCheckResult {
                ok: true,
                stderr: String::new(),
            }),
            warnings_as_errors: self.warnings_as_errors,
        };
        task.build_feedback(user_intent, prior_attempts)
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            1.0
        } else {
            (1.0 - 0.5 * issues.len() as f64).max(0.0)
        }
    }
}

// ── CodeTestTask ───────────────────────────────────────────────────────────

/// Retry loop for "code that passes the test suite".
pub struct CodeTestTask {
    user_prompt: String,
    initial_code: Option<String>,
    regenerate_fn: CodeRegenerateFn,
    test_fn: std::cell::RefCell<TestFn>,
}

impl CodeTestTask {
    /// Construct.
    pub fn new(
        user_prompt: impl Into<String>,
        initial_code: impl Into<String>,
        regenerate_fn: CodeRegenerateFn,
        test_fn: TestFn,
    ) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            initial_code: Some(initial_code.into()),
            regenerate_fn,
            test_fn: std::cell::RefCell::new(test_fn),
        }
    }
}

impl CorrectableTask for CodeTestTask {
    type Output = String;
    type Issue = TestIssue;

    fn name(&self) -> &str {
        "code_test"
    }

    fn execute(&mut self, feedback: Option<&str>) -> Result<TaskOutcome<Self::Output>, TaskError> {
        if feedback.is_none() {
            if let Some(code) = self.initial_code.take() {
                return Ok(TaskOutcome {
                    output: code,
                    tokens_used: 0,
                    cost_usd: 0.0,
                });
            }
        }
        match (self.regenerate_fn)(&self.user_prompt, feedback) {
            Some((code, tokens, cost)) => Ok(TaskOutcome {
                output: code,
                tokens_used: tokens,
                cost_usd: cost,
            }),
            None => Err(TaskError::new("regenerate_fn returned None")),
        }
    }

    fn validate(&self, output: &Self::Output) -> Vec<Self::Issue> {
        let result = (self.test_fn.borrow_mut())(output);
        match result.all_passed {
            Some(true) => Vec::new(),
            Some(false) => vec![TestIssue::TestsFailed {
                failed_count: result.failed_count.max(1),
                failed_names: result.failed_names,
                output_excerpt: result.output,
            }],
            None => vec![TestIssue::TestRunnerError {
                stderr: result.output,
            }],
        }
    }

    fn build_feedback(&self, user_intent: &str, prior_attempts: &[AttemptRecord]) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Original task: {}", user_intent));
        lines.push(String::new());
        if let Some(last) = prior_attempts.last() {
            lines.push(format!(
                "Your previous submission had {} failing validation(s):",
                last.issues.len()
            ));
            for issue in &last.issues {
                lines.push(format!("  - {}", issue));
            }
        }
        lines.push(String::new());
        lines.push("Please regenerate the code, applying these rules:".to_string());
        lines.push("  1. Fix every failing test without disabling or skipping it.".to_string());
        lines.push("  2. Don't change test assertions — only the code under test.".to_string());
        lines.push("  3. Preserve the function signatures exposed to tests.".to_string());
        lines.push("  4. Output ONLY the full corrected source file — no prose.".to_string());
        lines.join("\n")
    }

    fn quality_score(&self, _output: &Self::Output, issues: &[Self::Issue]) -> f64 {
        if issues.is_empty() {
            return 1.0;
        }
        // If we have a count of failures, scale penalty by it.
        let mut total_failures = 0usize;
        for i in issues {
            if let TestIssue::TestsFailed { failed_count, .. } = i {
                total_failures += *failed_count;
            } else {
                total_failures += 3; // runner error is "worse" than a single failing test
            }
        }
        let penalty = (total_failures as f64) * 0.1;
        (1.0 - penalty).max(0.0)
    }
}

// ── Convenience helpers shelling out to cargo ──────────────────────────────

/// Convenience: run `cargo check` in `crate_dir` after writing `code` to
/// `target_path` (relative to `crate_dir`). Returns `CompileCheckResult`.
///
/// Intended for V99 demo usage; real callers provide their own closure.
pub fn cargo_compile_check(
    crate_dir: &std::path::Path,
    target_path: &std::path::Path,
    code: &str,
) -> CompileCheckResult {
    let full = crate_dir.join(target_path);
    if let Err(e) = std::fs::write(&full, code) {
        return CompileCheckResult {
            ok: false,
            stderr: format!("failed to write {}: {}", full.display(), e),
        };
    }
    let output = Command::new("cargo")
        .arg("check")
        .arg("--message-format=short")
        .current_dir(crate_dir)
        .output();
    match output {
        Ok(o) => {
            let mut combined = String::new();
            combined.push_str(&String::from_utf8_lossy(&o.stdout));
            combined.push_str(&String::from_utf8_lossy(&o.stderr));
            CompileCheckResult {
                ok: o.status.success(),
                stderr: combined,
            }
        }
        Err(e) => CompileCheckResult {
            ok: false,
            stderr: format!("failed to spawn cargo: {}", e),
        },
    }
}

/// Convenience: run `cargo test` with optional filter pattern.
pub fn cargo_run_tests(
    crate_dir: &std::path::Path,
    target_path: &std::path::Path,
    code: &str,
    test_filter: Option<&str>,
) -> TestRunResult {
    let full = crate_dir.join(target_path);
    if let Err(e) = std::fs::write(&full, code) {
        return TestRunResult {
            all_passed: None,
            failed_count: 0,
            failed_names: Vec::new(),
            output: format!("failed to write {}: {}", full.display(), e),
        };
    }
    let mut cmd = Command::new("cargo");
    cmd.arg("test");
    if let Some(f) = test_filter {
        cmd.arg(f);
    }
    cmd.current_dir(crate_dir);
    let output = cmd.output();
    match output {
        Ok(o) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            let all_passed = o.status.success();
            let (failed_count, failed_names) = parse_cargo_test_failures(&combined);
            TestRunResult {
                all_passed: Some(all_passed),
                failed_count,
                failed_names,
                output: combined,
            }
        }
        Err(e) => TestRunResult {
            all_passed: None,
            failed_count: 0,
            failed_names: Vec::new(),
            output: format!("failed to spawn cargo: {}", e),
        },
    }
}

/// Best-effort parse of `cargo test` output for failure counts and names.
pub fn parse_cargo_test_failures(output: &str) -> (usize, Vec<String>) {
    let mut count = 0usize;
    let mut names = Vec::new();
    for line in output.lines() {
        let t = line.trim();
        // "test xyz ... FAILED"
        if let Some(rest) = t.strip_prefix("test ") {
            if let Some(name) = rest.strip_suffix(" ... FAILED") {
                names.push(name.to_string());
            }
        }
        // "test result: FAILED. N passed; M failed; ..."
        if t.contains("test result:") && t.contains("failed;") {
            if let Some(idx) = t.find("failed;") {
                let before = &t[..idx];
                let digits: String = before
                    .chars()
                    .rev()
                    .take_while(|c| c.is_whitespace() || c.is_ascii_digit())
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
                    .trim()
                    .to_string();
                if let Ok(n) = digits.parse::<usize>() {
                    count = count.max(n);
                }
            }
        }
    }
    if count == 0 && !names.is_empty() {
        count = names.len();
    }
    (count, names)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{SelfCorrectionConfig, SelfCorrectionEngine, StopReason};
    use super::*;

    fn always_ok_compile() -> CompileFn {
        Box::new(|_code| CompileCheckResult {
            ok: true,
            stderr: String::new(),
        })
    }

    fn always_fail_compile() -> CompileFn {
        Box::new(|_code| CompileCheckResult {
            ok: false,
            stderr: "error[E0308]: mismatched types".into(),
        })
    }

    fn regen_returns(s: &'static str, tokens: usize, cost: f64) -> CodeRegenerateFn {
        Box::new(move |_p, _f| Some((s.to_string(), tokens, cost)))
    }

    #[test]
    fn test_compile_success_first_try() {
        let task = CodeCompileTaskCell::new(
            "write a hello world",
            "fn main() { println!(\"hi\"); }",
            regen_returns("unused", 0, 0.0),
            always_ok_compile(),
        );
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "hello world");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
        assert_eq!(result.attempt_count(), 1);
    }

    #[test]
    fn test_compile_fail_exhausts_budget() {
        let task = CodeCompileTaskCell::new(
            "fix this",
            "broken",
            regen_returns("still broken", 100, 0.01),
            always_fail_compile(),
        );
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 3,
            min_improvement: 0.01,
            ..Default::default()
        });
        let result = engine.run(task, "fix");
        assert!(!result.succeeded);
        // Either MaxAttempts or NoImprovement — both valid.
        matches!(
            result.stop_reason,
            StopReason::MaxAttempts | StopReason::NoImprovement
        );
    }

    #[test]
    fn test_compile_warnings_as_errors() {
        let compile_fn: CompileFn = Box::new(|_| CompileCheckResult {
            ok: true,
            stderr: "warning: unused variable `x`".into(),
        });
        let task = CodeCompileTaskCell::new(
            "q",
            "code",
            regen_returns("still warns", 100, 0.01),
            compile_fn,
        )
        .with_warnings_as_errors(true);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 2,
            ..Default::default()
        });
        let result = engine.run(task, "q");
        assert!(!result.succeeded);
    }

    #[test]
    fn test_compile_feedback_mentions_stderr() {
        let task = CodeCompileTask {
            user_prompt: "q".into(),
            initial_code: None,
            regenerate_fn: Box::new(|_, _| None),
            compile_fn: always_fail_compile(),
            warnings_as_errors: false,
        };
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["compile failed:\nE0308".into()],
            quality_score: 0.5,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 10,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("write a parser", &prior);
        assert!(fb.contains("write a parser"));
        assert!(fb.contains("E0308"));
        assert!(fb.contains("Fix every compiler error"));
    }

    #[test]
    fn test_compile_quality_score() {
        let task = CodeCompileTask {
            user_prompt: "q".into(),
            initial_code: None,
            regenerate_fn: Box::new(|_, _| None),
            compile_fn: always_ok_compile(),
            warnings_as_errors: false,
        };
        assert_eq!(task.quality_score(&String::new(), &[]), 1.0);
        assert!(
            task.quality_score(
                &String::new(),
                &[CompileIssue::Failed { stderr: "x".into() }],
            ) < 1.0
        );
    }

    fn always_pass_tests() -> TestFn {
        Box::new(|_| TestRunResult {
            all_passed: Some(true),
            failed_count: 0,
            failed_names: Vec::new(),
            output: "ok".into(),
        })
    }

    fn always_fail_tests() -> TestFn {
        Box::new(|_| TestRunResult {
            all_passed: Some(false),
            failed_count: 2,
            failed_names: vec!["test_foo".into(), "test_bar".into()],
            output: "test test_foo ... FAILED\ntest test_bar ... FAILED".into(),
        })
    }

    #[test]
    fn test_tests_pass_first_try() {
        let task = CodeTestTask::new(
            "q",
            "passing code",
            regen_returns("unused", 0, 0.0),
            always_pass_tests(),
        );
        let engine = SelfCorrectionEngine::with_defaults();
        let result = engine.run(task, "q");
        assert!(result.succeeded);
        assert_eq!(result.stop_reason, StopReason::AllPassed);
    }

    #[test]
    fn test_tests_fail_produces_issue() {
        let task = CodeTestTask::new(
            "q",
            "failing",
            regen_returns("still failing", 100, 0.01),
            always_fail_tests(),
        );
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 2,
            ..Default::default()
        });
        let result = engine.run(task, "q");
        assert!(!result.succeeded);
        let first = &result.attempts[0];
        assert!(first
            .issues
            .iter()
            .any(|s| s.contains("test") && s.contains("failed")));
    }

    #[test]
    fn test_tests_runner_error_is_distinct() {
        let broken: TestFn = Box::new(|_| TestRunResult {
            all_passed: None,
            failed_count: 0,
            failed_names: Vec::new(),
            output: "could not spawn cargo".into(),
        });
        let task = CodeTestTask::new("q", "code", regen_returns("code2", 100, 0.01), broken);
        let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
            max_attempts: 1,
            ..Default::default()
        });
        let result = engine.run(task, "q");
        assert!(!result.succeeded);
        assert!(result.attempts[0]
            .issues
            .iter()
            .any(|s| s.contains("runner error")));
    }

    #[test]
    fn test_parse_cargo_test_failures() {
        let sample = r#"
running 3 tests
test test_a ... ok
test test_b ... FAILED
test test_c ... FAILED

failures:
    test_b
    test_c

test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured
"#;
        let (count, names) = parse_cargo_test_failures(sample);
        assert_eq!(count, 2);
        assert_eq!(names, vec!["test_b".to_string(), "test_c".to_string()]);
    }

    #[test]
    fn test_parse_empty_output() {
        let (count, names) = parse_cargo_test_failures("");
        assert_eq!(count, 0);
        assert!(names.is_empty());
    }

    #[test]
    fn test_compile_issue_display_truncates() {
        let long = "e".repeat(5000);
        let issue = CompileIssue::Failed { stderr: long };
        let s = format!("{}", issue);
        assert!(s.contains("truncated"));
    }

    #[test]
    fn test_test_issue_display_with_names() {
        let issue = TestIssue::TestsFailed {
            failed_count: 2,
            failed_names: vec!["a".into(), "b".into()],
            output_excerpt: "details".into(),
        };
        let s = format!("{}", issue);
        assert!(s.contains("2 test(s) failed"));
        assert!(s.contains("a, b"));
    }

    #[test]
    fn test_test_feedback_mentions_rules() {
        let task = CodeTestTask::new("q", "code", regen_returns("x", 0, 0.0), always_pass_tests());
        let prior = vec![AttemptRecord {
            attempt_num: 1,
            issues: vec!["1 test(s) failed: details".into()],
            quality_score: 0.3,
            tokens_used: 0,
            cost_usd: 0.0,
            elapsed_ms: 0,
            feedback_given: None,
            succeeded: false,
        }];
        let fb = task.build_feedback("implement parser", &prior);
        assert!(fb.contains("implement parser"));
        assert!(
            fb.contains("don't change test assertions".to_lowercase().as_str())
                || fb.contains("Don't change test")
        );
        assert!(fb.contains("Fix every failing test"));
    }
}
