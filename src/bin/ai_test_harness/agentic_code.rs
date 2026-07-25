use super::*;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ai_assistant::unified_tools::{ToolBuilder, ToolCall, ToolError, ToolOutput, ToolRegistry};
use ai_assistant::{
    AgentPolicyBuilder, AutonomousAgent, AutonomyLevel, LoopMessage, LoopRole, OperationMode,
};

// ─── Agentic coding loop (live model drives tools, execution-verified) ─────────
//
// The library's OWN autonomous agent (AutonomousAgent) is pointed at a real local
// model and given three workspace-scoped tools: write_file, read_file, run_python.
// The model must DRIVE those tools (emit JSON tool calls the agent parses) to
// build or fix a small program. We then verify the ARTIFACT it produced by
// executing it against a checker — success means the model sustained the
// write → run → observe → fix loop, not just that it can emit code.
//
// This is where models actually separate: single-function code-gen saturates at
// 3B (see code_gen_bench), but reliably emitting the tool-call protocol AND
// iterating to a green test is much harder for small models.
//
// Skips unless BOTH a `python` interpreter and the configured backend
// (AI_BENCH_* — see bench_util) are present.

const MAX_ITERS: usize = 6;
const RUN_TIMEOUT: Duration = Duration::from_secs(15);

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Teaches the exact tool-call protocol `parse_tool_calls` accepts: a bare JSON
/// array of `{"name", "arguments"}` objects, no prose, no markdown fences.
const AGENT_SYSTEM_PROMPT: &str = r#"You are an autonomous coding agent working inside a fixed workspace directory. You get things done by calling tools.

To call tools, reply with ONLY a JSON array (no prose, no ``` fences), each element being {"name": <tool>, "arguments": {<args>}}. Example:
[{"name":"write_file","arguments":{"path":"solution.py","content":"def add(a, b):\n    return a + b\n"}}]

Available tools:
- write_file(path, content): create or overwrite a file. `path` is RELATIVE to the workspace, e.g. "solution.py". `content` is the full file text.
- read_file(path): return the current contents of a file.
- run_python(path): execute a python file; returns its stdout, stderr and exit code.

Rules:
- When you act, output ONLY the JSON array — nothing before or after it, no code fences.
- After writing code, run_python it to confirm it works before you finish.
- Use RELATIVE paths only (no absolute paths, no ".." segments).
- When the task is fully complete and verified, reply with a short plain-text confirmation and NO JSON array."#;

struct AgenticTask {
    name: &'static str,
    /// Files pre-created in the workspace before the agent runs (path, content).
    seed: &'static [(&'static str, &'static str)],
    /// Instruction handed to the agent.
    prompt: &'static str,
    /// File the agent is expected to end up producing/fixing.
    target: &'static str,
    /// Python appended to the target to verify it (asserts raise → non-zero
    /// exit, so exit 0 == all checks passed).
    checker: &'static str,
}

const TASKS: &[AgenticTask] = &[
    AgenticTask {
        name: "create fizzbuzz",
        seed: &[],
        prompt: "Create a file `solution.py` containing a function `fizzbuzz(n)` that returns \
                 the string 'Fizz' if n is divisible by 3, 'Buzz' if divisible by 5, 'FizzBuzz' \
                 if divisible by both 3 and 5, otherwise str(n). Write the file, then run it to \
                 make sure it imports cleanly.",
        target: "solution.py",
        checker: "assert fizzbuzz(3) == 'Fizz'\n\
                  assert fizzbuzz(5) == 'Buzz'\n\
                  assert fizzbuzz(15) == 'FizzBuzz'\n\
                  assert fizzbuzz(7) == '7'\n",
    },
    AgenticTask {
        name: "fix is_even bug",
        seed: &[(
            "buggy.py",
            "def is_even(n):\n    # BUG: wrong operator\n    return n % 2 == 1\n",
        )],
        prompt: "The file `buggy.py` defines `is_even(n)`, but it is WRONG — it returns True for \
                 ODD numbers. Read the file, fix the bug so `is_even(n)` returns True exactly when \
                 n is even, and save the corrected file at the same path.",
        target: "buggy.py",
        checker: "assert is_even(2) == True\n\
                  assert is_even(4) == True\n\
                  assert is_even(3) == False\n\
                  assert is_even(0) == True\n\
                  assert is_even(7) == False\n",
    },
    AgenticTask {
        name: "bank account class",
        seed: &[],
        prompt: "Create `bank.py` defining a class `BankAccount` whose constructor takes an \
                 optional starting balance (default 0). It has a `balance` attribute, a method \
                 `deposit(self, amount)` that adds to the balance, and `withdraw(self, amount)` \
                 that subtracts from the balance but raises ValueError('insufficient funds') if \
                 amount is greater than the current balance. Write it and run it.",
        target: "bank.py",
        checker: "a = BankAccount()\n\
                  assert a.balance == 0\n\
                  a.deposit(100)\n\
                  assert a.balance == 100\n\
                  a.withdraw(30)\n\
                  assert a.balance == 70\n\
                  raised = False\n\
                  try:\n    a.withdraw(1000)\nexcept ValueError:\n    raised = True\n\
                  assert raised, 'overdraw must raise ValueError'\n",
    },
    AgenticTask {
        name: "fix binary_search off-by-one",
        seed: &[(
            "bsearch.py",
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1\n",
        )],
        prompt: "The file `bsearch.py` has a `binary_search(arr, target)` that should return the \
                 index of target in the sorted list `arr`, or -1 if absent. It has a bug that makes \
                 it loop forever on some inputs. Read it, fix the bug, and save the file.",
        target: "bsearch.py",
        checker: "assert binary_search([1, 3, 5, 7, 9], 5) == 2\n\
                  assert binary_search([1, 3, 5, 7, 9], 1) == 0\n\
                  assert binary_search([1, 3, 5, 7, 9], 9) == 4\n\
                  assert binary_search([1, 3, 5, 7, 9], 4) == -1\n\
                  assert binary_search([], 1) == -1\n",
    },
    AgenticTask {
        name: "run-length encode",
        seed: &[],
        prompt: "Create `rle.py` with a function `encode(s)` that run-length encodes the string `s`, \
                 returning a list of [character, count] pairs for each run of consecutive identical \
                 characters (e.g. the string aaabbc encodes to a list with three pairs: a with 3, \
                 b with 2, c with 1). An empty string returns an empty list. Write the file, then \
                 run it to check it works.",
        target: "rle.py",
        checker: "assert encode('aaabbc') == [['a', 3], ['b', 2], ['c', 1]]\n\
                  assert encode('') == []\n\
                  assert encode('x') == [['x', 1]]\n\
                  assert encode('aabbaa') == [['a', 2], ['b', 2], ['a', 2]]\n",
    },
];

/// Resolve a model-supplied relative path inside `workspace`, rejecting escapes.
fn safe_join(workspace: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim().replace('\\', "/");
    let rel = rel.trim_start_matches('/');
    if rel.split('/').any(|c| c == "..") || rel.is_empty() {
        return Err(format!("illegal workspace path: {rel:?}"));
    }
    Ok(workspace.join(rel))
}

/// First available Python interpreter, if any.
fn python_cmd() -> Option<&'static str> {
    for c in ["python", "python3"] {
        let ok = std::process::Command::new(c)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(c);
        }
    }
    None
}

/// Run a python file with a timeout, capturing stdout+stderr and exit code as
/// text — this is what the agent's `run_python` tool hands back to the model so
/// it can see errors and iterate.
fn run_python_capture(py: &str, file: &Path, timeout: Duration) -> String {
    use std::io::Read;
    let spawn = std::process::Command::new(py)
        .arg(file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut child = match spawn {
        Ok(c) => c,
        Err(e) => return format!("failed to launch python: {e}"),
    };
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };
    let mut out = String::new();
    if let Some(mut so) = child.stdout.take() {
        let _ = so.read_to_string(&mut out);
    }
    let mut err = String::new();
    if let Some(mut se) = child.stderr.take() {
        let _ = se.read_to_string(&mut err);
    }
    match status {
        Some(st) => format!(
            "exit_code={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            st.code().unwrap_or(-1),
            out.trim(),
            err.trim()
        ),
        None => format!(
            "TIMEOUT after {}s (possible infinite loop)\n--- stdout ---\n{}",
            timeout.as_secs(),
            out.trim()
        ),
    }
}

/// Run a python file with a timeout; true iff it exits 0. Used for verification.
fn run_python_exit_ok(py: &str, file: &Path, timeout: Duration) -> bool {
    let spawn = std::process::Command::new(py)
        .arg(file)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let mut child = match spawn {
        Ok(c) => c,
        Err(_) => return false,
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => return st.success(),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return false,
        }
    }
}

/// Verify the agent's artifact: concatenate the file the agent produced with the
/// checker asserts and execute — exit 0 == the code actually works.
fn verify_file(py: &str, target: &Path, checker: &str) -> bool {
    let src = match std::fs::read_to_string(target) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let full = format!("{src}\n\n# --- checker ---\n{checker}\nprint('OK')\n");
    let vpath = target.with_file_name(format!(
        "__verify_{}.py",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::write(&vpath, full).is_err() {
        return false;
    }
    let ok = run_python_exit_ok(py, &vpath, RUN_TIMEOUT);
    let _ = std::fs::remove_file(&vpath);
    ok
}

/// Extract the first balanced JSON array from `s`, respecting string quoting so
/// brackets inside code content don't throw off the depth count. Returns the
/// `[..]` substring, or None if there's no balanced array.
fn first_json_array(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s.char_indices().skip_while(|&(i, _)| i < start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(s[start..i + c.len_utf8()].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Flatten the agent's loop conversation into a single stateless prompt.
fn render_conversation(conv: &[LoopMessage]) -> String {
    let mut s = String::new();
    for m in conv {
        let tag = match m.role {
            LoopRole::System => "SYSTEM",
            LoopRole::User => "USER",
            LoopRole::Assistant => "ASSISTANT",
            LoopRole::Tool => "TOOL_RESULT",
            _ => "NOTE",
        };
        s.push_str(tag);
        s.push_str(":\n");
        s.push_str(&m.content);
        s.push_str("\n\n");
    }
    s.push_str("Respond now as ASSISTANT:\n");
    s
}

/// Build the response generator: each call renders the full loop conversation
/// and asks the live model statelessly (the assistant's own history is cleared
/// so only the rendered transcript drives generation).
type Generator = Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>;

fn make_generator(assistant: Arc<Mutex<ai_assistant::AiAssistant>>) -> Generator {
    Arc::new(move |conv: &[LoopMessage]| {
        let prompt = render_conversation(conv);
        let mut a = match assistant.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        a.clear_conversation();
        let reply = match a.generate_sync(prompt, "") {
            Ok(reply) => reply,
            Err(e) => return format!("(generation error: {e})"),
        };
        // Local models (especially mid-conversation) tend to emit the tool-call
        // array and then HALLUCINATE the rest of the transcript ("TOOL_RESULT: …",
        // more ASSISTANT turns). Keep only the first tool-call array so the agent
        // parses a clean call and the fabricated continuation is discarded; if the
        // reply is a plain-text final answer, pass it through untouched.
        match first_json_array(&reply) {
            Some(arr) if arr.contains("\"name\"") => arr,
            _ => reply,
        }
    })
}

/// Register the three workspace-scoped coding tools.
fn build_tools(workspace: PathBuf, py: &'static str) -> ToolRegistry {
    let mut reg = ToolRegistry::new();

    // write_file
    {
        let ws = workspace.clone();
        let def = ToolBuilder::new(
            "write_file",
            "Create or overwrite a text file in the workspace.",
        )
        .required_string(
            "path",
            "File path relative to the workspace, e.g. \"solution.py\"",
        )
        .required_string("content", "The full file content to write")
        .category("filesystem")
        .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |call: &ToolCall| {
                let path = call
                    .get_string("path")
                    .ok_or_else(|| ToolError::MissingParameter("path".into()))?;
                let content = call
                    .get_string("content")
                    .ok_or_else(|| ToolError::MissingParameter("content".into()))?;
                let full = safe_join(&ws, path).map_err(ToolError::ExecutionFailed)?;
                if let Some(parent) = full.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::write(&full, content)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(ToolOutput::text(format!(
                    "wrote {} bytes to {path}",
                    content.len()
                )))
            });
        reg.register(def, handler);
    }

    // read_file
    {
        let ws = workspace.clone();
        let def = ToolBuilder::new("read_file", "Read a text file from the workspace.")
            .required_string("path", "File path relative to the workspace")
            .category("filesystem")
            .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |call: &ToolCall| {
                let path = call
                    .get_string("path")
                    .ok_or_else(|| ToolError::MissingParameter("path".into()))?;
                let full = safe_join(&ws, path).map_err(ToolError::ExecutionFailed)?;
                match std::fs::read_to_string(&full) {
                    Ok(c) => Ok(ToolOutput::text(c)),
                    Err(e) => Ok(ToolOutput::text(format!("(could not read {path}: {e})"))),
                }
            });
        reg.register(def, handler);
    }

    // run_python
    {
        let ws = workspace.clone();
        let def = ToolBuilder::new(
            "run_python",
            "Execute a Python file in the workspace and return stdout, stderr and exit code.",
        )
        .required_string("path", "Python file path relative to the workspace")
        .category("shell")
        .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |call: &ToolCall| {
                let path = call
                    .get_string("path")
                    .ok_or_else(|| ToolError::MissingParameter("path".into()))?;
                let full = safe_join(&ws, path).map_err(ToolError::ExecutionFailed)?;
                if !full.exists() {
                    return Ok(ToolOutput::text(format!("(file does not exist: {path})")));
                }
                Ok(ToolOutput::text(run_python_capture(py, &full, RUN_TIMEOUT)))
            });
        reg.register(def, handler);
    }

    reg
}

fn run_one_task(py: &'static str, task: &AgenticTask) -> Result<(), String> {
    // Fresh isolated workspace.
    let workspace = std::env::temp_dir().join(format!(
        "agentic_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&workspace).map_err(|e| e.to_string())?;
    for (p, c) in task.seed {
        if let Ok(full) = safe_join(&workspace, p) {
            let _ = std::fs::write(full, c);
        }
    }

    // Wire the agent: live model as response generator + workspace tools.
    let assistant = Arc::new(Mutex::new(crate::bench_util::bench_assistant()));
    let generator = make_generator(assistant);
    let policy = AgentPolicyBuilder::new()
        .autonomy(AutonomyLevel::Autonomous)
        .working_directory(workspace.clone())
        .allow_path(workspace.clone())
        .allow_command("python")
        .allow_command("python3")
        .allow_tool("write_file")
        .allow_tool("read_file")
        .allow_tool("run_python")
        .build();
    let registry = build_tools(workspace.clone(), py);
    let mut agent = AutonomousAgent::builder("coder", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(AGENT_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    // Run to completion (or max-iters); we care about the ARTIFACT regardless of
    // how the agent decided to terminate.
    let outcome = agent.run(task.prompt);
    let iters = match &outcome {
        Ok(r) => r.iterations,
        Err(_) => MAX_ITERS,
    };

    let target = safe_join(&workspace, task.target)?;
    let passed = verify_file(py, &target, task.checker);
    let _ = std::fs::remove_dir_all(&workspace);

    if passed {
        Ok(())
    } else {
        Err(format!(
            "artifact failed checker after {iters} agent iteration(s) — model did not sustain the tool loop"
        ))
    }
}

pub(crate) fn tests_agentic_code() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Agentic coding loop (live model drives tools)"))
    );
    let mut results = Vec::new();

    let py = python_cmd();
    if !crate::bench_util::backend_reachable() || py.is_none() {
        let why = if py.is_none() {
            "no python interpreter"
        } else {
            "backend not reachable"
        };
        println!("  {} skipping agentic-code ({why})", yellow("SKIP"));
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some(format!("Skipped — {why}")),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "agentic_code".to_string(),
            results,
        };
    }
    let py = py.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());

    for task in TASKS {
        results.push(run_test(&format!("agentic: {}", task.name), || {
            run_one_task(py, task)
        }));
    }

    let solved = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} agentic_code solved: {}/{} (tool-driven, execution-verified, backend={})",
        bold(&cyan("∑")),
        solved,
        total,
        crate::bench_util::bench_label()
    );

    CategoryResult {
        name: "agentic_code".to_string(),
        results,
    }
}

// ─── Multi-step agentic coding (build → extend → fix, persistent workspace) ────
//
// The SAME agent runs a SEQUENCE of instructions against ONE workspace. Because
// `AutonomousAgent::run()` appends to (never clears) its conversation, each step
// sees everything the earlier steps built — this is iterative development, the
// "use it like Claude Code" flow. The final accumulated artifact is verified
// against a checker that exercises EVERY feature, so the model only passes if it
// preserved earlier work while adding to it (write_file overwrites the whole
// file, so it must re-emit prior code correctly each edit).

struct MultiStepTask {
    name: &'static str,
    /// File that accumulates across the steps.
    target: &'static str,
    seed: &'static [(&'static str, &'static str)],
    /// Sequential instructions handed to the same agent, in order.
    steps: &'static [&'static str],
    /// Verifies the FINAL artifact against every feature that should exist.
    checker: &'static str,
}

const MULTI_TASKS: &[MultiStepTask] = &[
    MultiStepTask {
        name: "calculator (build up)",
        target: "calc.py",
        seed: &[],
        steps: &[
            "Create `calc.py` with a function `add(a, b)` that returns a + b. Then run it to check it works.",
            "Add a function `subtract(a, b)` to `calc.py` returning a - b. KEEP the existing add function. Save the whole file and run it.",
            "Add two more functions to `calc.py`, keeping everything already there: `multiply(a, b)` returning a * b, and `divide(a, b)` returning a / b but raising ValueError('division by zero') when b == 0. Save and run.",
        ],
        checker: "assert add(2, 3) == 5\n\
                  assert subtract(10, 4) == 6\n\
                  assert multiply(3, 4) == 12\n\
                  assert divide(9, 3) == 3\n\
                  raised = False\n\
                  try:\n    divide(1, 0)\nexcept ValueError:\n    raised = True\n\
                  assert raised, 'divide by zero must raise ValueError'\n",
    },
    MultiStepTask {
        name: "stack class (build + fix)",
        target: "stack.py",
        seed: &[],
        steps: &[
            "Create `stack.py` defining a class `Stack` with methods `push(self, x)`, `pop(self)` and `is_empty(self)`, backed by a Python list. Run it to make sure it imports.",
            "Add a `peek(self)` method to the `Stack` class returning the top item without removing it, and a `size(self)` method returning the number of items. Keep all existing methods. Save and run.",
            "Make `pop(self)` raise IndexError('pop from empty stack') when the stack is empty. Keep everything else intact. Save and run.",
        ],
        checker: "s = Stack()\n\
                  assert s.is_empty() == True\n\
                  s.push(1)\ns.push(2)\ns.push(3)\n\
                  assert s.size() == 3\n\
                  assert s.peek() == 3\n\
                  assert s.pop() == 3\n\
                  assert s.pop() == 2\n\
                  assert s.pop() == 1\n\
                  raised = False\n\
                  try:\n    s.pop()\nexcept IndexError:\n    raised = True\n\
                  assert raised, 'pop on empty must raise IndexError'\n",
    },
];

fn run_multi_task(py: &'static str, task: &MultiStepTask) -> Result<(), String> {
    let workspace = std::env::temp_dir().join(format!(
        "agentic_multi_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&workspace).map_err(|e| e.to_string())?;
    for (p, c) in task.seed {
        if let Ok(full) = safe_join(&workspace, p) {
            let _ = std::fs::write(full, c);
        }
    }

    // ONE agent for the whole sequence — the conversation carries prior steps.
    let assistant = Arc::new(Mutex::new(crate::bench_util::bench_assistant()));
    let generator = make_generator(assistant);
    let policy = AgentPolicyBuilder::new()
        .autonomy(AutonomyLevel::Autonomous)
        .working_directory(workspace.clone())
        .allow_path(workspace.clone())
        .allow_command("python")
        .allow_command("python3")
        .allow_tool("write_file")
        .allow_tool("read_file")
        .allow_tool("run_python")
        .build();
    let registry = build_tools(workspace.clone(), py);
    let mut agent = AutonomousAgent::builder("coder", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(AGENT_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    let debug = std::env::var("AGENTIC_DEBUG").is_ok();
    let mut total_iters = 0usize;
    for (i, step) in task.steps.iter().enumerate() {
        let prompt = format!("Step {}/{}: {}", i + 1, task.steps.len(), step);
        let outcome = agent.run(&prompt);
        let (it, tools) = match &outcome {
            Ok(r) => (r.iterations, r.tools_called.join(",")),
            Err(e) => (MAX_ITERS, format!("<err: {e}>")),
        };
        total_iters += it;
        if debug {
            let snap =
                std::fs::read_to_string(safe_join(&workspace, task.target).unwrap_or_default())
                    .unwrap_or_else(|_| "<target missing>".to_string());
            let last_reply = agent
                .conversation()
                .iter()
                .rev()
                .find(|m| matches!(m.role, LoopRole::Assistant))
                .map(|m| m.content.clone())
                .unwrap_or_else(|| "<none>".to_string());
            println!(
                "    [dbg] step {} iters={it} tools=[{tools}] target {} bytes\n      model_reply: {:?}\n      target:\n----\n{}\n----",
                i + 1,
                snap.len(),
                last_reply.chars().take(400).collect::<String>(),
                snap
            );
        }
    }

    let target = safe_join(&workspace, task.target)?;
    let passed = verify_file(py, &target, task.checker);
    if !debug {
        let _ = std::fs::remove_dir_all(&workspace);
    }

    if passed {
        Ok(())
    } else {
        Err(format!(
            "final artifact failed checker after {} steps ({total_iters} total iters) — model lost state across edits",
            task.steps.len()
        ))
    }
}

pub(crate) fn tests_agentic_multi() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Multi-step agentic coding (build → extend → fix, persistent workspace)"
        ))
    );
    let mut results = Vec::new();

    let py = python_cmd();
    if !crate::bench_util::backend_reachable() || py.is_none() {
        let why = if py.is_none() {
            "no python interpreter"
        } else {
            "backend not reachable"
        };
        println!("  {} skipping agentic-multi ({why})", yellow("SKIP"));
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some(format!("Skipped — {why}")),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "agentic_multi".to_string(),
            results,
        };
    }
    let py = py.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());

    for task in MULTI_TASKS {
        results.push(run_test(&format!("multi: {}", task.name), || {
            run_multi_task(py, task)
        }));
    }

    let solved = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} agentic_multi solved: {}/{} (multi-step, execution-verified, backend={})",
        bold(&cyan("∑")),
        solved,
        total,
        crate::bench_util::bench_label()
    );

    CategoryResult {
        name: "agentic_multi".to_string(),
        results,
    }
}
