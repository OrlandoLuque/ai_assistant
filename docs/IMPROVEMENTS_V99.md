# V99 — Self-Correction for Code Tasks

**Version**: 0.2.30 → 0.2.31
**Feature flag**: `self-correction` (same as V98, no new feature)

## Scope

V98 landed the generic `CorrectableTask` trait and one concrete task for
claim verification. V99 extends the harness to the code domain:

- **`CodeCompileTask`** + **`CodeCompileTaskCell`** — regenerate until a
  user-provided compile closure returns `ok = true`. The `Cell` variant uses
  `RefCell<CompileFn>` so the closure can be invoked from `validate(&self)`
  despite `FnMut` semantics.
- **`CodeTestTask`** — regenerate until the test closure reports
  `all_passed = Some(true)`. Distinguishes `TestsFailed` (test assertion
  failed) from `TestRunnerError` (runner itself errored — compile failure
  inside test binary, subprocess spawn failure, etc.).
- **Convenience helpers** for the common Rust case:
  - `cargo_compile_check(crate_dir, target_path, code)` — writes the code,
    shells out to `cargo check --message-format=short`, returns
    `CompileCheckResult`.
  - `cargo_run_tests(crate_dir, target_path, code, test_filter)` — writes,
    shells out to `cargo test`, returns `TestRunResult`.
  - `parse_cargo_test_failures(output)` — best-effort parser for
    `test X ... FAILED` lines and `test result:` summaries.

## Issue types

```rust
pub enum CompileIssue {
    Failed { stderr: String },
    WarningsAsErrors { stderr: String },
}

pub enum TestIssue {
    TestsFailed {
        failed_count: usize,
        failed_names: Vec<String>,
        output_excerpt: String,
    },
    TestRunnerError { stderr: String },
}
```

Both default to retryable. The engine's quality-score delta and
no-improvement detection decides whether to keep retrying after the LLM
produces a fix that's close but not quite right.

## Why `CodeCompileTaskCell`

The `CorrectableTask` trait has `validate(&self, …)`, but compiling
involves writing a file and spawning `cargo`, which is conceptually
mutating. Two fixes were considered:

1. Change the trait to `validate(&mut self, …)` — pollutes everyone else's
   implementation.
2. Use interior mutability (`RefCell<CompileFn>`) locally.

V99 picks option (2). `CodeCompileTask` (non-cell variant) is kept for the
rare case where the compile closure is truly `Fn` (no state). Both expose
the same public shape; callers pick based on their closure.

## Feedback template

For compile failures:

```
Original task: <user intent>

Your previous submission failed with N issue(s):
  - compile failed:
<first ~800 chars of stderr>
…[truncated]

Please regenerate the code, applying these rules:
  1. Fix every compiler error reported above.
  2. Do not introduce unrelated changes.
  3. Keep the public API unchanged unless the error requires it.
  4. Output ONLY the full corrected source file — no prose.
```

Test feedback is analogous with different rules (don't change test
assertions; preserve function signatures).

## Tests

13 unit tests across `src/self_correction/code.rs`:

- `test_compile_success_first_try` — fast path
- `test_compile_fail_exhausts_budget` — budget exhaustion
- `test_compile_warnings_as_errors` — `with_warnings_as_errors` flag
- `test_compile_feedback_mentions_stderr` — feedback contains user intent +
  stderr excerpt + "Fix every compiler error"
- `test_compile_quality_score` — empty issues → 1.0, non-empty → < 1.0
- `test_tests_pass_first_try`
- `test_tests_fail_produces_issue`
- `test_tests_runner_error_is_distinct` — `TestRunnerError` vs
  `TestsFailed` surfaced correctly
- `test_parse_cargo_test_failures` — parser extracts names + count from
  realistic `cargo test` output
- `test_parse_empty_output`
- `test_compile_issue_display_truncates` — long stderr truncated
- `test_test_issue_display_with_names` — "2 test(s) failed (a, b)"
- `test_test_feedback_mentions_rules`

Total self-correction tests: **49** (36 from V98 + 13 from V99).

## Public API additions (in `lib.rs`)

```rust
#[cfg(feature = "self-correction")]
pub use self_correction::{
    // V98:
    ClaimVerificationTask, ClaimIssue, CorrectableTask, SelfCorrectionEngine, /* … */
    // V99:
    CodeCompileTask, CodeCompileTaskCell, CodeTestTask,
    CompileFn, CompileCheckResult, CompileIssue,
    TestFn, TestRunResult, TestIssue,
    CodeRegenerateFn,
    correction_cargo_compile_check, correction_cargo_run_tests,
    correction_parse_cargo_test_failures,
};
```

## Usage example

```rust
use std::path::Path;
use ai_assistant::{
    CodeCompileTaskCell, SelfCorrectionEngine, SelfCorrectionConfig,
    correction_cargo_compile_check,
};

let crate_dir = Path::new("/tmp/playground");
let target = Path::new("src/main.rs");

let compile_fn = Box::new(move |code: &str| {
    correction_cargo_compile_check(crate_dir, target, code)
});

let regen = Box::new(|prompt: &str, feedback: Option<&str>| {
    // your LLM call
    Some(("fn main() { println!(\"hello\"); }".to_string(), 100, 0.01))
});

let task = CodeCompileTaskCell::new(
    "write a hello-world binary",
    "fn main( {}",  // broken initial
    regen,
    compile_fn,
);

let engine = SelfCorrectionEngine::new(SelfCorrectionConfig {
    max_attempts: 5,
    max_total_cost_usd: 0.50,
    ..Default::default()
});

let result = engine.run(task, "write a hello-world binary");
```

## What's NOT in V99

- Auditor binaries (`ai_corrections`, `ai_corrections_gui`) — still scheduled.
- `ai_cli code --auto-fix` flag.
- HTTP / MCP endpoints.
- Language-specific helpers beyond cargo (gcc/clang/python/node/etc).

V100 will finish the task-type matrix (tool-call, research-citation,
agent-handoff, safety-guardrail). Auditor + CLI + server wiring is
grouped separately because it spans all three V-versions.
