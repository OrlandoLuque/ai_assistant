use super::*;

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ai_assistant::unified_tools::{ToolBuilder, ToolCall, ToolError, ToolOutput, ToolRegistry};
use ai_assistant::{AgentPolicyBuilder, AutonomousAgent, AutonomyLevel, OperationMode};

use crate::agentic_code::{make_generator, run_capture, safe_join, COUNTER, MAX_ITERS};

// ─── Agentic RUST coding (compiler + cargo test as the verifier) ──────────────
//
// The BACKLOG asks for "un set propio de tareas reales sobre este mismo repo" and
// names execution/compilation as the key lever. Everything else in this benchmark
// is Python, where the interpreter only complains at runtime. Rust is the harder
// and more relevant target for THIS project: the model must satisfy the type
// checker and the borrow checker before a single test runs, and `cargo test` is a
// far stricter verifier than `python file.py`.
//
// Each task gets a throwaway zero-dependency cargo crate; the model edits
// `src/lib.rs` through the tools and can run `cargo test` / `cargo build` itself.
// Verification appends a checker test module to whatever the model produced and
// runs `cargo test` — exit 0 means it compiles AND the assertions hold.
//
// NOTE on task design: Rust string literals require double quotes, which the model
// has to escape inside the JSON tool-call payload — a known confound that measures
// escaping rather than coding (see docs/MODEL_BENCHMARKS.md). Most tasks here are
// therefore quote-free (numbers, Option/Result, structs); `fizzbuzz` is kept
// deliberately as the one task that does stress escaping.

/// cargo is much slower than python: a cold crate build can take a few seconds,
/// and a model can write code that takes a while to compile.
const CARGO_TIMEOUT: Duration = Duration::from_secs(90);

const CARGO_TOML: &str =
    "[package]\nname = \"task\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n";

const RUST_SYSTEM_PROMPT: &str = r#"You are an autonomous Rust coding agent working inside a cargo project. You get things done by calling tools.

To call tools, reply with ONLY a JSON array (no prose, no ``` fences), each element being {"name": <tool>, "arguments": {<args>}}. Example:
[{"name":"write_file","arguments":{"path":"src/lib.rs","content":"pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"}}]

The crate is a normal cargo project: the library source is `src/lib.rs`. Items you must expose have to be `pub`.

Available tools:
- write_file(path, content): create or overwrite a file, e.g. "src/lib.rs". `content` is the full file text.
- read_file(path): return the current contents of a file.
- list_dir(): list the files in the project.
- run_command(command): run a whitelisted command — `cargo build`, `cargo test`, `cargo check`.

Rules:
- When you act, output ONLY the JSON array — nothing before or after it, no code fences.
- After writing code, run `cargo test` (or `cargo build`) to confirm it COMPILES before you finish; fix any compiler errors it reports.
- Use RELATIVE paths only (no absolute paths, no ".." segments).
- When the task is fully complete and it compiles, reply with a short plain-text confirmation and NO JSON array."#;

struct RustTask {
    name: &'static str,
    /// Initial contents of `src/lib.rs` (empty string = start from scratch).
    seed_lib: &'static str,
    prompt: &'static str,
    /// Test body appended inside a `mod harness_checker` to verify the artifact.
    checker: &'static str,
}

const RUST_TASKS: &[RustTask] = &[
    RustTask {
        name: "sum_even (from scratch)",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function `sum_even(nums: &[i32]) -> i32` that \
                 returns the sum of the even numbers in the slice. Then run cargo test to make \
                 sure it compiles.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(sum_even(&[1, 2, 3, 4, 5, 6]), 12);\n        \
                  assert_eq!(sum_even(&[1, 3, 5]), 0);\n        \
                  assert_eq!(sum_even(&[]), 0);\n    }\n",
    },
    RustTask {
        name: "fix is_even bug",
        seed_lib: "// BUG: this returns true for ODD numbers.\npub fn is_even(n: i32) -> bool {\n    n % 2 == 1\n}\n",
        prompt: "The file `src/lib.rs` defines `is_even(n: i32) -> bool`, but it is WRONG — it \
                 returns true for ODD numbers. Read the file, fix the bug so it returns true \
                 exactly when n is even, and save it. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert!(is_even(2));\n        \
                  assert!(is_even(0));\n        \
                  assert!(is_even(-4));\n        \
                  assert!(!is_even(3));\n        \
                  assert!(!is_even(7));\n    }\n",
    },
    RustTask {
        name: "divide returning Option",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function `divide(a: f64, b: f64) -> Option<f64>` \
                 that returns Some(a / b), or None when b is 0.0. Then run cargo test to make sure \
                 it compiles.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(divide(6.0, 3.0), Some(2.0));\n        \
                  assert_eq!(divide(1.0, 0.0), None);\n        \
                  assert!(divide(-9.0, 3.0).is_some());\n    }\n",
    },
    RustTask {
        name: "Stack struct with impl",
        seed_lib: "",
        prompt: "In `src/lib.rs`, define a public struct `Stack` holding a vector of i32, with a \
                 public `new()` constructor, and public methods `push(&mut self, x: i32)`, \
                 `pop(&mut self) -> Option<i32>` and `is_empty(&self) -> bool`. Then run cargo \
                 test to make sure it compiles.",
        checker: "    #[test]\n    fn check() {\n        \
                  let mut s = Stack::new();\n        \
                  assert!(s.is_empty());\n        \
                  s.push(1);\n        s.push(2);\n        \
                  assert!(!s.is_empty());\n        \
                  assert_eq!(s.pop(), Some(2));\n        \
                  assert_eq!(s.pop(), Some(1));\n        \
                  assert_eq!(s.pop(), None);\n    }\n",
    },
    RustTask {
        name: "count occurrences (HashMap)",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function \
                 `count_occurrences(nums: &[i32]) -> std::collections::HashMap<i32, usize>` that \
                 maps each distinct number to how many times it appears in the slice. Then run \
                 cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let m = count_occurrences(&[1, 2, 2, 3, 3, 3]);\n        \
                  assert_eq!(m.get(&1), Some(&1));\n        \
                  assert_eq!(m.get(&2), Some(&2));\n        \
                  assert_eq!(m.get(&3), Some(&3));\n        \
                  assert_eq!(m.get(&9), None);\n        \
                  assert!(count_occurrences(&[]).is_empty());\n    }\n",
    },
    // ── Harder: the things that make Rust Rust (generics, traits, lifetimes,
    //    the borrow checker). Single functions saturate; these should not.
    RustTask {
        name: "generic largest<T> with trait bounds",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public GENERIC function \
                 `largest<T: PartialOrd + Copy>(list: &[T]) -> T` returning the largest element of \
                 the slice. It must compile for both integers and floats. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(largest(&[1, 5, 3]), 5);\n        \
                  assert_eq!(largest(&[-4, -9]), -4);\n        \
                  assert!((largest(&[1.5f64, 2.5, 0.5]) - 2.5).abs() < 1e-9);\n    }\n",
    },
    RustTask {
        name: "trait with two impls",
        seed_lib: "",
        prompt: "In `src/lib.rs`, define a public trait `Shape` with a method \
                 `area(&self) -> f64`. Then define public structs `Circle` (field `radius: f64`) \
                 and `Rect` (fields `w: f64` and `h: f64`), each with a public `new` constructor, \
                 and implement `Shape` for both. Circle area is PI * r * r. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let c = Circle::new(1.0);\n        \
                  let r = Rect::new(2.0, 3.0);\n        \
                  assert!((c.area() - std::f64::consts::PI).abs() < 1e-9);\n        \
                  assert!((r.area() - 6.0).abs() < 1e-9);\n        \
                  let shapes: Vec<Box<dyn Shape>> = vec![Box::new(Circle::new(1.0)), Box::new(Rect::new(2.0, 3.0))];\n        \
                  assert_eq!(shapes.len(), 2);\n    }\n",
    },
    RustTask {
        name: "explicit lifetimes",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function `longest` that takes two string slices \
                 with the SAME lifetime and returns the longer one (the first when equal length), \
                 using an explicit lifetime annotation. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let a = String::from(\"hello\");\n        \
                  let b = String::from(\"hi\");\n        \
                  assert_eq!(longest(a.as_str(), b.as_str()).len(), 5);\n        \
                  assert_eq!(longest(\"ab\", \"cd\").len(), 2);\n    }\n",
    },
    RustTask {
        name: "implement the Iterator trait",
        seed_lib: "",
        prompt: "In `src/lib.rs`, define a public struct `Countdown` holding a u32, with a public \
                 `new(start: u32)` constructor, and implement the standard `Iterator` trait for it \
                 so it yields the numbers from start down to 1 (Item = u32), then None. Then run \
                 cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let v: Vec<u32> = Countdown::new(3).collect();\n        \
                  assert_eq!(v, vec![3, 2, 1]);\n        \
                  assert_eq!(Countdown::new(0).count(), 0);\n        \
                  assert_eq!(Countdown::new(5).sum::<u32>(), 15);\n    }\n",
    },
    RustTask {
        name: "borrow checker: dedup in place",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function `dedup_in_place(v: &mut Vec<i32>)` that \
                 removes duplicate values IN PLACE while preserving the order of first appearance \
                 (do not sort). Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let mut v = vec![3, 1, 3, 2, 1, 4];\n        \
                  dedup_in_place(&mut v);\n        \
                  assert_eq!(v, vec![3, 1, 2, 4]);\n        \
                  let mut e: Vec<i32> = vec![];\n        \
                  dedup_in_place(&mut e);\n        \
                  assert!(e.is_empty());\n    }\n",
    },
    RustTask {
        name: "enum + match evaluator",
        seed_lib: "",
        prompt: "In `src/lib.rs`, define a public enum `Op` with variants `Add`, `Sub`, `Mul` and \
                 `Div`, and a public function `apply(op: Op, a: f64, b: f64) -> Option<f64>` that \
                 applies the operation, returning None for a division by zero. Derive Clone and \
                 Copy on the enum. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(apply(Op::Add, 2.0, 3.0), Some(5.0));\n        \
                  assert_eq!(apply(Op::Sub, 2.0, 3.0), Some(-1.0));\n        \
                  assert_eq!(apply(Op::Mul, 2.0, 3.0), Some(6.0));\n        \
                  assert_eq!(apply(Op::Div, 6.0, 3.0), Some(2.0));\n        \
                  assert_eq!(apply(Op::Div, 1.0, 0.0), None);\n    }\n",
    },
    RustTask {
        name: "fizzbuzz (stresses quote escaping)",
        seed_lib: "",
        prompt: "In `src/lib.rs`, write a public function `fizzbuzz(n: u32) -> String` returning \
                 Fizz when n is divisible by 3, Buzz when divisible by 5, FizzBuzz when divisible \
                 by both, and otherwise the number as a string. Then run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(fizzbuzz(3), \"Fizz\");\n        \
                  assert_eq!(fizzbuzz(5), \"Buzz\");\n        \
                  assert_eq!(fizzbuzz(15), \"FizzBuzz\");\n        \
                  assert_eq!(fizzbuzz(7), \"7\");\n    }\n",
    },
];

/// Is a cargo binary available?
fn cargo_cmd() -> Option<&'static str> {
    let ok = std::process::Command::new("cargo")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        Some("cargo")
    } else {
        None
    }
}

/// Create a throwaway zero-dependency cargo crate seeded with `seed_lib`.
/// All task crates share one target dir so only the first build pays full cost.
fn scaffold_crate(seed_lib: &str) -> Result<std::path::PathBuf, String> {
    let root = std::env::temp_dir().join("agentic_rust_bench");
    let shared_target = root.join("_target");
    let ws = root.join(format!(
        "task_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(ws.join("src")).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(ws.join(".cargo")).map_err(|e| e.to_string())?;
    std::fs::write(ws.join("Cargo.toml"), CARGO_TOML).map_err(|e| e.to_string())?;
    std::fs::write(ws.join("src").join("lib.rs"), seed_lib).map_err(|e| e.to_string())?;
    // Share one target dir across tasks (path escaped for TOML on Windows).
    let cfg = format!(
        "[build]\ntarget-dir = \"{}\"\n",
        shared_target.to_string_lossy().replace('\\', "\\\\")
    );
    std::fs::write(ws.join(".cargo").join("config.toml"), cfg).map_err(|e| e.to_string())?;
    Ok(ws)
}

/// Append the checker as a test module to whatever the model wrote and run
/// `cargo test`: exit 0 means it compiled AND the assertions passed.
fn verify_crate(cargo: &str, ws: &Path, checker: &str) -> bool {
    verify_crate_verbose(cargo, ws, checker).0
}

/// Same as [`verify_crate`] but also returns the cargo output, so a scaffolding
/// loop can hand the compiler's own errors back to the model.
fn verify_crate_verbose(cargo: &str, ws: &Path, checker: &str) -> (bool, String) {
    let lib = ws.join("src").join("lib.rs");
    let Ok(src) = std::fs::read_to_string(&lib) else {
        return (false, "src/lib.rs is missing".to_string());
    };
    let full =
        format!("{src}\n\n#[cfg(test)]\nmod harness_checker {{\n    use super::*;\n{checker}}}\n");
    if std::fs::write(&lib, &full).is_err() {
        return (false, "could not write src/lib.rs".to_string());
    }
    let out = run_capture(
        cargo,
        &["test".to_string(), "--quiet".to_string()],
        ws,
        CARGO_TIMEOUT,
    );
    let _ = std::fs::write(&lib, src); // restore the model's own file
    (out.starts_with("exit_code=0"), out)
}

/// How many verify→retry rounds the agent gets (`AI_BENCH_SCAFFOLD`, default 1 =
/// no scaffolding, a single shot). This is the BACKLOG's central hypothesis —
/// "agentic scaffolding compensates for a weaker local model, at the cost of
/// time" — expressed as an A/B knob: same model, same tasks, only the number of
/// verify→feedback→retry rounds changes.
fn scaffold_rounds() -> usize {
    std::env::var("AI_BENCH_SCAFFOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1)
}

/// Turn a failing cargo run into a retry instruction for the model. The hidden
/// checker itself is NOT revealed — only the compiler/test output, which is what
/// a developer would see.
fn retry_prompt(cargo_output: &str) -> String {
    let tail: String = cargo_output.chars().take(2500).collect();
    format!(
        "Your code was compiled and tested against a hidden test suite, and it FAILED. \
         Here is the exact cargo output:\n\n{tail}\n\n\
         Fix `src/lib.rs` so it compiles and satisfies the requirements of the original task. \
         Re-read the file first if you need to, write the corrected version, and run cargo test."
    )
}

/// Tools for the Rust workspace: file I/O plus a cargo-only run_command.
fn build_rust_tools(ws: std::path::PathBuf, cargo: &'static str) -> ToolRegistry {
    let mut reg = ToolRegistry::new();

    {
        let w = ws.clone();
        let def = ToolBuilder::new("write_file", "Create or overwrite a file in the crate.")
            .required_string(
                "path",
                "Path relative to the crate root, e.g. \"src/lib.rs\"",
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
                let full = safe_join(&w, path).map_err(ToolError::ExecutionFailed)?;
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

    {
        let w = ws.clone();
        let def = ToolBuilder::new("read_file", "Read a file from the crate.")
            .required_string("path", "Path relative to the crate root")
            .category("filesystem")
            .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |call: &ToolCall| {
                let path = call
                    .get_string("path")
                    .ok_or_else(|| ToolError::MissingParameter("path".into()))?;
                let full = safe_join(&w, path).map_err(ToolError::ExecutionFailed)?;
                match std::fs::read_to_string(&full) {
                    Ok(c) => Ok(ToolOutput::text(c)),
                    Err(e) => Ok(ToolOutput::text(format!("(could not read {path}: {e})"))),
                }
            });
        reg.register(def, handler);
    }

    {
        let w = ws.clone();
        let def = ToolBuilder::new("list_dir", "List the files in the crate.")
            .category("filesystem")
            .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |_call: &ToolCall| {
                let mut names: Vec<String> = Vec::new();
                for sub in ["", "src"] {
                    let dir = if sub.is_empty() {
                        w.clone()
                    } else {
                        w.join(sub)
                    };
                    if let Ok(rd) = std::fs::read_dir(&dir) {
                        for e in rd.flatten() {
                            let n = e.file_name().to_string_lossy().into_owned();
                            names.push(if sub.is_empty() {
                                n
                            } else {
                                format!("{sub}/{n}")
                            });
                        }
                    }
                }
                names.sort();
                Ok(ToolOutput::text(names.join("\n")))
            });
        reg.register(def, handler);
    }

    {
        let w = ws.clone();
        let def = ToolBuilder::new(
            "run_command",
            "Run a cargo command in the crate (cargo build / cargo test / cargo check).",
        )
        .required_string("command", "The command line, e.g. \"cargo test\"")
        .category("shell")
        .build();
        let handler: Arc<dyn Fn(&ToolCall) -> Result<ToolOutput, ToolError> + Send + Sync> =
            Arc::new(move |call: &ToolCall| {
                let cmd = call
                    .get_string("command")
                    .ok_or_else(|| ToolError::MissingParameter("command".into()))?;
                let parts: Vec<String> = cmd.split_whitespace().map(|s| s.to_string()).collect();
                if parts.first().map(|s| s.as_str()) != Some("cargo") {
                    return Ok(ToolOutput::text(
                        "(only cargo commands are allowed here, e.g. \"cargo test\")".to_string(),
                    ));
                }
                const ALLOWED_SUB: &[&str] = &["build", "test", "check"];
                let sub = parts.get(1).map(|s| s.as_str()).unwrap_or("");
                if !ALLOWED_SUB.contains(&sub) {
                    return Ok(ToolOutput::text(format!(
                        "(cargo subcommand not allowed: '{sub}'. Allowed: build, test, check)"
                    )));
                }
                // Cap output so a wall of compiler errors can't blow up the context.
                let out = run_capture(cargo, &parts[1..], &w, CARGO_TIMEOUT);
                let trimmed: String = out.chars().take(4000).collect();
                Ok(ToolOutput::text(trimmed))
            });
        reg.register(def, handler);
    }

    reg
}

/// How many INDEPENDENT attempts a task gets (`AI_BENCH_SAMPLES`, default 1).
/// This is best-of-N: unlike the retry loop, each sample starts from a clean crate
/// with a fresh agent and no memory of the failure — it tests whether *sampling*
/// rescues a weak model where *feedback* did not. Pair it with `AI_BENCH_TEMP>0`,
/// or every sample is the same answer.
fn sample_count() -> usize {
    std::env::var("AI_BENCH_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1)
}

fn run_rust_task(cargo: &'static str, task: &RustTask) -> Result<(), String> {
    let samples = sample_count();
    let mut last_err = String::new();
    for _ in 0..samples {
        match run_rust_task_once(cargo, task) {
            Ok(()) => return Ok(()),
            Err(e) => last_err = e,
        }
    }
    if samples > 1 {
        Err(format!(
            "all {samples} independent samples failed; last: {last_err}"
        ))
    } else {
        Err(last_err)
    }
}

fn run_rust_task_once(cargo: &'static str, task: &RustTask) -> Result<(), String> {
    let ws = scaffold_crate(task.seed_lib)?;

    let assistant = Arc::new(Mutex::new(crate::bench_util::bench_assistant()));
    let generator = make_generator(assistant);
    let policy = AgentPolicyBuilder::new()
        .autonomy(AutonomyLevel::Autonomous)
        .working_directory(ws.clone())
        .allow_path(ws.clone())
        .allow_command("cargo")
        .allow_tool("write_file")
        .allow_tool("read_file")
        .allow_tool("list_dir")
        .allow_tool("run_command")
        .build();
    let registry = build_rust_tools(ws.clone(), cargo);
    let mut agent = AutonomousAgent::builder("rust-coder", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(RUST_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    // Scaffolding loop: run, verify by compiling+testing, and on failure hand the
    // model the compiler's own output and let it try again (AI_BENCH_SCAFFOLD).
    let rounds = scaffold_rounds();
    let mut iters = 0usize;
    let mut last_out = String::new();
    let mut passed = false;
    for round in 0..rounds {
        let prompt = if round == 0 {
            task.prompt.to_string()
        } else {
            retry_prompt(&last_out)
        };
        match agent.run(&prompt) {
            Ok(r) => iters += r.iterations,
            Err(_) => iters += MAX_ITERS,
        }
        let (ok, out) = verify_crate_verbose(cargo, &ws, task.checker);
        passed = ok;
        last_out = out;
        if passed {
            break;
        }
    }
    let _ = std::fs::remove_dir_all(&ws);

    if passed {
        Ok(())
    } else {
        Err(format!(
            "cargo test failed after {rounds} round(s), {iters} agent iteration(s) — code did not compile or assertions failed"
        ))
    }
}

// ─── Multi-step Rust (build -> extend -> refactor, one persistent crate) ──────
//
// Single-function Rust saturates exactly like single-function Python (7B and 14B
// both score 12/12 on the set above). What discriminates is sustaining a SEQUENCE
// of edits on one crate: the model must keep prior code compiling while adding to
// it, and — the hardest variant — REWRITE earlier code when a later step changes
// its shape (e.g. making a concrete type generic).

struct RustMultiTask {
    name: &'static str,
    steps: &'static [&'static str],
    checker: &'static str,
}

const RUST_MULTI_TASKS: &[RustMultiTask] = &[
    RustMultiTask {
        name: "shapes: trait then impls then aggregate",
        steps: &[
            "In `src/lib.rs`, define a public trait `Shape` with `area(&self) -> f64`, and a \
             public struct `Rect` (fields `w: f64`, `h: f64`) with a public `new(w, h)` \
             constructor implementing it. Run cargo test.",
            "Add a public struct `Circle` (field `radius: f64`) with a public `new(radius)` \
             constructor, also implementing `Shape` (area = PI * r * r). Keep `Rect` working. \
             Run cargo test.",
            "Add a public free function `total_area(shapes: &[Box<dyn Shape>]) -> f64` returning \
             the sum of the areas. Keep everything already there. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  let shapes: Vec<Box<dyn Shape>> = vec![Box::new(Rect::new(2.0, 3.0)), Box::new(Circle::new(1.0))];\n        \
                  assert!((total_area(&shapes) - (6.0 + std::f64::consts::PI)).abs() < 1e-9);\n        \
                  assert!((Rect::new(2.0, 2.0).area() - 4.0).abs() < 1e-9);\n    }\n",
    },
    RustMultiTask {
        name: "stack: concrete then generic (must rewrite)",
        steps: &[
            "In `src/lib.rs`, define a public struct `Stack` holding a Vec<i32>, with public \
             `new()`, `push(&mut self, x: i32)` and `pop(&mut self) -> Option<i32>`. Run cargo test.",
            "Add public methods `len(&self) -> usize` and `peek(&self) -> Option<&i32>`. Keep the \
             existing ones. Run cargo test.",
            "Now make the whole thing GENERIC: `Stack<T>` must work for any element type, so \
             rewrite the struct and all its methods accordingly (push takes T, pop returns \
             Option<T>, peek returns Option<&T>). Everything must still compile. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  let mut s: Stack<i32> = Stack::new();\n        \
                  s.push(1);\n        s.push(2);\n        \
                  assert_eq!(s.len(), 2);\n        \
                  assert_eq!(s.peek(), Some(&2));\n        \
                  assert_eq!(s.pop(), Some(2));\n        \
                  let mut t: Stack<bool> = Stack::new();\n        \
                  t.push(true);\n        \
                  assert_eq!(t.pop(), Some(true));\n        \
                  assert_eq!(t.len(), 0);\n    }\n",
    },
    RustMultiTask {
        name: "errors: Option then custom error type",
        steps: &[
            "In `src/lib.rs`, write a public function `parse_pair(s: &str) -> Option<(i32, i32)>` \
             that parses a string like 3,4 (two integers separated by a comma) into a tuple, \
             returning None when it cannot. Run cargo test.",
            "Define a public enum `ParseErr` with variants `Missing` and `NotANumber`, and change \
             `parse_pair` to return `Result<(i32, i32), ParseErr>` instead: `Missing` when there \
             is no comma, `NotANumber` when a side is not an integer. Derive Debug and PartialEq \
             on the enum. Run cargo test.",
            "Add a public function `sum_pair(s: &str) -> Result<i32, ParseErr>` that uses \
             `parse_pair` and returns the sum of the two numbers. Keep everything. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(parse_pair(\"3,4\"), Ok((3, 4)));\n        \
                  assert_eq!(parse_pair(\"7\"), Err(ParseErr::Missing));\n        \
                  assert_eq!(parse_pair(\"a,2\"), Err(ParseErr::NotANumber));\n        \
                  assert_eq!(sum_pair(\"10,5\"), Ok(15));\n        \
                  assert!(sum_pair(\"x\").is_err());\n    }\n",
    },
    RustMultiTask {
        name: "builder pattern with validation",
        steps: &[
            "In `src/lib.rs`, define a public struct `Server` with public fields `port: u16` and \
             `retries: u32`, plus a public `new(port: u16, retries: u32)` constructor. Run cargo test.",
            "Add a public struct `ServerBuilder` with a public `new()` returning a builder whose \
             defaults are port 80 and retries 0, chainable methods `port(self, p: u16) -> Self` and \
             `retries(self, r: u32) -> Self`, and `build(self) -> Server`. Keep `Server`. Run cargo test.",
            "Change `build` to `build(self) -> Result<Server, String>`, returning Err when the port \
             is 0. Everything else must keep working. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  let s = ServerBuilder::new().port(8080).retries(3).build().unwrap();\n        \
                  assert_eq!(s.port, 8080);\n        \
                  assert_eq!(s.retries, 3);\n        \
                  let d = ServerBuilder::new().build().unwrap();\n        \
                  assert_eq!(d.port, 80);\n        \
                  assert_eq!(d.retries, 0);\n        \
                  assert!(ServerBuilder::new().port(0).build().is_err());\n    }\n",
    },
    RustMultiTask {
        name: "matrix ops accumulate",
        steps: &[
            "In `src/lib.rs`, write a public function `transpose(m: &Vec<Vec<i32>>) -> Vec<Vec<i32>>` \
             returning the transpose of the matrix. Run cargo test.",
            "Add a public function `identity(n: usize) -> Vec<Vec<i32>>` returning the n by n \
             identity matrix. Keep `transpose`. Run cargo test.",
            "Add a public function `multiply(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>>` \
             returning the matrix product. Keep everything already there. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(transpose(&vec![vec![1, 2], vec![3, 4]]), vec![vec![1, 3], vec![2, 4]]);\n        \
                  assert_eq!(identity(2), vec![vec![1, 0], vec![0, 1]]);\n        \
                  assert_eq!(multiply(&vec![vec![1, 2]], &vec![vec![3], vec![4]]), vec![vec![11]]);\n        \
                  let m = vec![vec![1, 2], vec![3, 4]];\n        \
                  assert_eq!(multiply(&m, &identity(2)), m);\n    }\n",
    },
    RustMultiTask {
        name: "trait then generic over it",
        steps: &[
            "In `src/lib.rs`, define a public trait `Score` with a method `score(&self) -> i32`, \
             and a public struct `Player` (public field `points: i32`) with a public `new(points)` \
             constructor implementing it (score returns points). Run cargo test.",
            "Add a public struct `Team` (public field `members: Vec<Player>`) with a public \
             `new(members: Vec<Player>)` constructor, also implementing `Score` — a team's score is \
             the sum of its members' scores. Keep `Player`. Run cargo test.",
            "Add a public GENERIC function `best<T: Score>(items: &[T]) -> Option<i32>` returning \
             the highest score among the items, or None when empty. Keep everything. Run cargo test.",
        ],
        checker: "    #[test]\n    fn check() {\n        \
                  let p = Player::new(7);\n        \
                  assert_eq!(p.score(), 7);\n        \
                  let t = Team::new(vec![Player::new(2), Player::new(5)]);\n        \
                  assert_eq!(t.score(), 7);\n        \
                  assert_eq!(best(&[Player::new(1), Player::new(9)]), Some(9));\n        \
                  let empty: Vec<Player> = vec![];\n        \
                  assert_eq!(best(&empty), None);\n    }\n",
    },
];

fn run_rust_multi_task(cargo: &'static str, task: &RustMultiTask) -> Result<(), String> {
    let ws = scaffold_crate("")?;

    let assistant = Arc::new(Mutex::new(crate::bench_util::bench_assistant()));
    let generator = make_generator(assistant);
    let policy = AgentPolicyBuilder::new()
        .autonomy(AutonomyLevel::Autonomous)
        .working_directory(ws.clone())
        .allow_path(ws.clone())
        .allow_command("cargo")
        .allow_tool("write_file")
        .allow_tool("read_file")
        .allow_tool("list_dir")
        .allow_tool("run_command")
        .build();
    let registry = build_rust_tools(ws.clone(), cargo);
    let mut agent = AutonomousAgent::builder("rust-coder", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(RUST_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    let mut total_iters = 0usize;
    for (i, step) in task.steps.iter().enumerate() {
        let prompt = format!("Step {}/{}: {}", i + 1, task.steps.len(), step);
        match agent.run(&prompt) {
            Ok(r) => total_iters += r.iterations,
            Err(_) => total_iters += MAX_ITERS,
        }
    }

    let passed = verify_crate(cargo, &ws, task.checker);
    let _ = std::fs::remove_dir_all(&ws);

    if passed {
        Ok(())
    } else {
        Err(format!(
            "cargo test failed after {} steps ({total_iters} iters) — lost state or broke compilation across edits",
            task.steps.len()
        ))
    }
}

pub(crate) fn tests_agentic_rust_multi() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Multi-step RUST coding (build → extend → refactor)"
        ))
    );
    let mut results = Vec::new();

    let cargo = cargo_cmd();
    if !crate::bench_util::backend_reachable() || cargo.is_none() {
        let why = if cargo.is_none() {
            "cargo not available"
        } else {
            "backend not reachable"
        };
        println!("  {} skipping agentic-rust-multi ({why})", yellow("SKIP"));
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
            name: "agentic_rust_multi".to_string(),
            results,
        };
    }
    let cargo = cargo.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());

    for task in RUST_MULTI_TASKS {
        results.push(run_test(&format!("rust multi: {}", task.name), || {
            run_rust_multi_task(cargo, task)
        }));
    }

    let solved = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} agentic_rust_multi solved: {}/{} (cargo-verified, backend={})",
        bold(&cyan("∑")),
        solved,
        total,
        crate::bench_util::bench_label()
    );

    CategoryResult {
        name: "agentic_rust_multi".to_string(),
        results,
    }
}

pub(crate) fn tests_agentic_rust() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Agentic RUST coding (cargo test as the verifier)"))
    );
    let mut results = Vec::new();

    let cargo = cargo_cmd();
    if !crate::bench_util::backend_reachable() || cargo.is_none() {
        let why = if cargo.is_none() {
            "cargo not available"
        } else {
            "backend not reachable"
        };
        println!("  {} skipping agentic-rust ({why})", yellow("SKIP"));
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
            name: "agentic_rust".to_string(),
            results,
        };
    }
    let cargo = cargo.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());

    for task in RUST_TASKS {
        results.push(run_test(&format!("rust: {}", task.name), || {
            run_rust_task(cargo, task)
        }));
    }

    let solved = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} agentic_rust solved: {}/{} (cargo-verified, backend={})",
        bold(&cyan("∑")),
        solved,
        total,
        crate::bench_util::bench_label()
    );

    CategoryResult {
        name: "agentic_rust".to_string(),
        results,
    }
}
