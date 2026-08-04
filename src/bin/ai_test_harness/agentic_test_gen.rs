use super::*;

use std::sync::{Arc, Mutex};

use ai_assistant::unified_tools::ToolRegistry;
use ai_assistant::{AgentPolicyBuilder, AutonomousAgent, AutonomyLevel, OperationMode};

use crate::agentic_code::{make_generator, MAX_ITERS};
use crate::checker_adequacy::ADEQUACY;

// ─── Can the model write COMPETENT TESTS? ─────────────────────────────────────
//
// Every other category asks the model for an implementation and judges it with OUR
// tests. This one inverts that: the model is handed a CORRECT implementation and
// asked to write the test suite, and its suite is judged the same way we judge our
// own oracles — it must accept the correct code AND catch known-wrong variants.
//
// This matters far more than "can it produce something that looks like tests". An
// autonomous agent that writes weak tests will rubber-stamp its own broken work: it
// runs them, sees green, and reports success. Test-writing is the capability that
// decides whether an agent's self-verification means anything at all.
//
// Scoring is a real mutation score: tests that pass on the reference and kill every
// mutant earn the task; tests that pass everything (the classic `assert!(true)`
// suite) earn nothing, which is exactly the outcome a naive "did it write tests?"
// check would miss.

// Repeats, rate scoring and the reporting block live in `bench_stats`, shared with
// the other live-model categories so they cannot drift apart in what "flaky" means.
use crate::bench_stats::BACKEND_CRASH_PREFIX;

const TEST_GEN_SYSTEM_PROMPT: &str = r#"You are an autonomous Rust engineer writing a TEST SUITE. You act only by calling tools.

To call tools, reply with ONLY a JSON array (no prose, no ``` fences), each element being {"name": <tool>, "arguments": {<args>}}. Example:
[{"name":"write_file","arguments":{"path":"tests/model_tests.rs","content":"use task::*;\n\n#[test]\nfn works() {\n    assert_eq!(add(1, 2), 3);\n}\n"}}]

The crate is called `task` and its implementation is already written in `src/lib.rs`. Your job is ONLY to write integration tests at `tests/model_tests.rs`, which must start with `use task::*;` and use the crate's PUBLIC API.

Available tools:
- read_file(path): read a file, e.g. "src/lib.rs" — read the implementation first.
- write_file(path, content): create or overwrite a file.
- run_command(command): run `cargo test`, `cargo build` or `cargo check`.

Rules:
- Write THOROUGH tests: cover edge cases, boundaries and error paths, not just one happy path. A test suite that only checks the obvious case is worthless, because it would pass even on a broken implementation.
- Test ONLY the behaviour the code actually has. Do NOT invent requirements it was never given: if `src/lib.rs` does not validate its inputs, do not assert that it rejects them. Asserting behaviour that was never specified produces a suite which fails on correct code, which is worse than no suite at all.
- Every test you write MUST pass against the implementation in `src/lib.rs` as it stands. Run `cargo test` and fix any test of yours that fails — the implementation is the specification here.
- Do NOT modify `src/lib.rs`. Only write `tests/model_tests.rs`.
- Output ONLY the JSON array — nothing before or after it, no code fences.
- When the tests are written and `cargo test` passes, reply with a short plain-text confirmation and NO JSON array."#;

/// Scaffold a crate whose `src/lib.rs` is the reference implementation and whose
/// `tests/` directory is empty, ready for the model to fill.
fn scaffold_for_tests(reference: &str) -> Result<std::path::PathBuf, String> {
    let ws = crate::agentic_rust::scaffold_crate_pub(reference)?;
    std::fs::create_dir_all(ws.join("tests")).map_err(|e| e.to_string())?;
    Ok(ws)
}

/// Run the model's test file against a given implementation.
/// Returns true when `cargo test` passes.
fn run_model_tests_against(cargo: &str, ws: &std::path::Path, implementation: &str) -> bool {
    let lib = ws.join("src").join("lib.rs");
    if std::fs::write(&lib, implementation).is_err() {
        return false;
    }
    let out = crate::agentic_code::run_capture(
        cargo,
        &["test".to_string(), "--quiet".to_string()],
        ws,
        std::time::Duration::from_secs(90),
    );
    out.starts_with("exit_code=0")
}

/// The model's suite is scored exactly like our own oracles: accept the reference,
/// reject every mutant.
struct SuiteVerdict {
    wrote_tests: bool,
    accepts_reference: bool,
    mutants_killed: usize,
    mutants_total: usize,
}

fn judge_model_suite(
    cargo: &'static str,
    ws: &std::path::Path,
    entry: &crate::checker_adequacy::Adequacy,
) -> SuiteVerdict {
    let test_file = ws.join("tests").join("model_tests.rs");
    let wrote = std::fs::read_to_string(&test_file)
        .map(|c| c.contains("#[test]"))
        .unwrap_or(false);
    if !wrote {
        return SuiteVerdict {
            wrote_tests: false,
            accepts_reference: false,
            mutants_killed: 0,
            mutants_total: entry.mutants.len(),
        };
    }

    // 1. The suite must pass on correct code, or it is simply broken.
    let accepts = run_model_tests_against(cargo, ws, entry.reference);

    // 2. And it must FAIL on each known-wrong variant. A suite that passes a mutant
    //    would have let that bug through in real use.
    let mut killed = 0;
    for (_label, mutant) in entry.mutants {
        if !run_model_tests_against(cargo, ws, mutant) {
            killed += 1;
        }
    }
    // Leave the crate holding the reference again.
    let _ = std::fs::write(ws.join("src").join("lib.rs"), entry.reference);

    SuiteVerdict {
        wrote_tests: true,
        accepts_reference: accepts,
        mutants_killed: killed,
        mutants_total: entry.mutants.len(),
    }
}

fn run_one_test_gen(
    cargo: &'static str,
    entry: &crate::checker_adequacy::Adequacy,
) -> Result<(), String> {
    let ws = scaffold_for_tests(entry.reference)?;

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
    let registry: ToolRegistry = crate::agentic_rust::build_rust_tools_pub(ws.clone(), cargo);
    let mut agent = AutonomousAgent::builder("test-writer", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(TEST_GEN_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    let prompt = "Read `src/lib.rs`, then write a thorough test suite at \
                  `tests/model_tests.rs` for it. The implementation is believed correct; \
                  your tests must prove it, and must be strong enough to FAIL if someone \
                  later broke the behaviour. Cover the edge cases, not just one obvious \
                  case.";
    let outcome = agent.run(prompt);

    if std::env::var("AGENTIC_DEBUG").is_ok() {
        let (iters, tools) = match &outcome {
            Ok(r) => (r.iterations, r.tools_called.join(",")),
            Err(e) => (0, format!("<err: {e}>")),
        };
        println!("    [dbg] iters={iters} tools=[{tools}]");
        // The last thing the model said, which is where a malformed tool call shows.
        // Printed in FULL: truncating it here once cost a wrong diagnosis — a reply
        // cut at the 500-char cap looked like the backend truncating the generation,
        // when the reply was complete and the tool call was what failed.
        if let Some(last) = agent
            .conversation()
            .iter()
            .rev()
            .find(|m| matches!(m.role, ai_assistant::LoopRole::Assistant))
        {
            println!(
                "    [dbg] last model reply ({} chars, library parses {} tool call(s)): {:?}",
                last.content.len(),
                // NOT `ai_assistant::parse_tool_calls` — that name is re-exported
                // from `unified_tools`, a different parser from the one the agent
                // loop actually uses. Measuring the wrong one sent this diagnosis
                // down a blind alley once already.
                ai_assistant::autonomous_loop::parse_tool_calls(&last.content).len(),
                last.content
            );
            // The exact bytes, for offline analysis: escaping games in the terminal
            // are how this diagnosis went wrong twice.
            if let Ok(dir) = std::env::var("AGENTIC_DUMP_DIR") {
                let path = std::path::Path::new(&dir).join("last_reply.txt");
                match std::fs::write(&path, last.content.as_bytes()) {
                    Ok(()) => println!("    [dbg] raw reply written to {}", path.display()),
                    Err(e) => println!("    [dbg] could not write dump: {e}"),
                }
            }
            // Why serde rejected it, which the parse helpers swallow with `.ok()?`.
            if let Some(start) = last.content.find('[') {
                if let Some(end) = last.content.rfind(']') {
                    if end > start {
                        let candidate = &last.content[start..=end];
                        match serde_json::from_str::<serde_json::Value>(candidate) {
                            Ok(_) => println!("    [dbg] the array IS valid JSON"),
                            Err(e) => {
                                println!("    [dbg] serde rejects it: {e}");
                                // Show the offending neighbourhood, since the
                                // line/column alone says nothing about which
                                // sequence the model actually got wrong.
                                if let Some(line) =
                                    candidate.lines().nth(e.line().saturating_sub(1))
                                {
                                    let col = e.column();
                                    let from = col.saturating_sub(30);
                                    let window: String = line.chars().skip(from).take(60).collect();
                                    println!("    [dbg] around col {col}: {window:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
        // `tools_called` lists only the SUCCESSES, so a tool that was invoked and
        // rejected leaves no trace there — which reads as "the model never tried".
        for m in agent.conversation() {
            if m.content.contains("Error]") || m.content.contains("error:") {
                println!(
                    "    [dbg] tool error: {:?}",
                    m.content.chars().take(300).collect::<String>()
                );
            }
        }
    }

    // A generation request that never came back is not evidence about the model, so
    // it is reported separately rather than scored as incompetence.
    //
    // Diagnosed 2026-07-31 on Ollama 0.21.2: the llama.cpp runner *aborts* on some
    // inputs when sampling is near-greedy — `Assertion failed: found,
    // llama-sampling.cpp, line 660` in the server log. The runner dies mid-request,
    // so the client just waits out its hard-coded 120 s ceiling and reports a send
    // failure. It is neither a slow model nor a repetition loop: the GPU sits idle.
    //
    // It is input-dependent: the request that first exposed it crashes at temperature
    // 0.0–0.3 and answers in seconds at 0.5, which is why the benchmark defaults sit
    // there with a pinned `seed` for reproducibility (V258). But 0.5 is not immunity —
    // other inputs still crash the runner at that temperature — so the case has to be
    // handled rather than merely avoided.
    if crate::agentic_code::hit_backend_failure(&agent) {
        let _ = std::fs::remove_dir_all(&ws);
        return Err(format!(
            "{BACKEND_CRASH_PREFIX}, not a model failure (this run leaves the denominator) \
             — the generation never returned. On Ollama this is the llama.cpp runner \
             aborting on the near-greedy sampling path; check the server log for \
             `Assertion failed: found ... llama-sampling.cpp`."
        ));
    }

    let verdict = judge_model_suite(cargo, &ws, entry);

    if std::env::var("AGENTIC_DEBUG").is_ok() {
        let tf = ws.join("tests").join("model_tests.rs");
        match std::fs::read_to_string(&tf) {
            Ok(c) => println!(
                "    [dbg] model_tests.rs ({} bytes):\n----\n{}\n----",
                c.len(),
                c.chars().take(1200).collect::<String>()
            ),
            Err(e) => {
                println!("    [dbg] no tests/model_tests.rs ({e})");
                if let Ok(rd) = std::fs::read_dir(&ws) {
                    let names: Vec<String> = rd
                        .flatten()
                        .map(|d| d.file_name().to_string_lossy().into_owned())
                        .collect();
                    println!("    [dbg] crate root holds: {}", names.join(", "));
                }
                if let Ok(rd) = std::fs::read_dir(ws.join("tests")) {
                    let names: Vec<String> = rd
                        .flatten()
                        .map(|d| d.file_name().to_string_lossy().into_owned())
                        .collect();
                    println!("    [dbg] tests/ holds: {}", names.join(", "));
                }
            }
        }
        // Why the suite rejects correct code, when it does.
        if verdict.wrote_tests && !verdict.accepts_reference {
            let _ = std::fs::write(ws.join("src").join("lib.rs"), entry.reference);
            let out = crate::agentic_code::run_capture(
                cargo,
                &["test".to_string()],
                &ws,
                std::time::Duration::from_secs(90),
            );
            println!(
                "    [dbg] cargo test on CORRECT code said:\n----\n{}\n----",
                out.chars().take(1200).collect::<String>()
            );
        }
    }

    let _ = std::fs::remove_dir_all(&ws);

    if !verdict.wrote_tests {
        return Err("the model never produced a test file with any #[test]".to_string());
    }
    if !verdict.accepts_reference {
        return Err(format!(
            "the model's tests FAIL on correct code — the suite itself is broken \
             (it did kill {}/{} mutants, which is meaningless if it rejects valid \
             implementations)",
            verdict.mutants_killed, verdict.mutants_total
        ));
    }
    if verdict.mutants_killed < verdict.mutants_total {
        return Err(format!(
            "the model's tests are too weak: they caught only {}/{} known bugs — code \
             an agent would have approved as correct",
            verdict.mutants_killed, verdict.mutants_total
        ));
    }
    Ok(())
}

pub(crate) fn tests_agentic_test_gen() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Test generation (can the model write tests that catch real bugs?)"
        ))
    );
    let mut results = Vec::new();

    let cargo = crate::agentic_rust::cargo_cmd_pub();
    if !crate::bench_util::backend_reachable() || cargo.is_none() {
        let why = if cargo.is_none() {
            "cargo not available"
        } else {
            "backend not reachable"
        };
        println!("  {} skipping test-generation ({why})", yellow("SKIP"));
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
            name: "agentic_test_gen".to_string(),
            results,
        };
    }
    let cargo = cargo.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());
    crate::bench_util::warn_if_cpu_offloaded();

    // The corpus is the same reference/mutant set that validates our own oracles.
    //
    // Each task is repeated and scored as a pass RATE — see `bench_stats` for why
    // that is not optional and why the repeats are interleaved.
    let repeats = crate::bench_util::bench_repeats();
    let tasks: Vec<_> = ADEQUACY
        .iter()
        .filter(|e| crate::should_run(&format!("test-gen: {}", e.task)))
        .collect();
    let outcomes = crate::bench_stats::run_interleaved(
        &tasks,
        |e| format!("test-gen: {}", e.task),
        repeats,
        |entry| run_one_test_gen(cargo, entry),
    );

    results.extend(crate::bench_stats::to_results(&outcomes, repeats));
    crate::bench_stats::print_summary(
        "agentic_test_gen",
        "suites that both accept correct code and catch every planted bug",
        &[
            ("FAIL on correct code", "rejected valid code"),
            ("too weak", "too weak to catch the bug"),
            ("never produced a test file", "produced no tests"),
        ],
        &outcomes,
        repeats,
    );

    CategoryResult {
        name: "agentic_test_gen".to_string(),
        results,
    }
}
