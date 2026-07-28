use super::*;

// ─── Code-generation benchmark (execution-verified, live model) ───────────────
//
// First increment of the code-gen benchmark (see ai_assistant_plans/BACKLOG.md):
// ask a REAL local model (Ollama / llama3.2:3b) to write small Python functions,
// then SCORE by actually EXECUTING the generated code against assert-based
// checkers in a subprocess with a timeout — the "execution as verifier" lever.
// pass@1 = fraction whose generated code passes its checker.
//
// Later increments scale this to SWE-bench / Aider-polyglot and compare against
// top open-source coding agents and Claude Code (the quality ceiling).
//
// Security note: this runs model-generated code. Tasks are tiny and pure, run in
// a temp file as a separate process with a hard timeout. It only runs when BOTH
// Ollama and a `python` interpreter are present (so CI, which has neither, skips
// the whole category).

const RUN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

struct CodeTask {
    name: &'static str,
    /// Natural-language spec handed to the model.
    spec: &'static str,
    /// Python appended after the generated code — asserts that raise (non-zero
    /// exit) on any failure, so process exit 0 == all checks passed.
    checker: &'static str,
}

const TASKS: &[CodeTask] = &[
    CodeTask {
        name: "has_close_elements",
        spec: "Define a function `has_close_elements(numbers, threshold)` that returns True if \
                any two numbers in the list `numbers` are closer to each other than `threshold` \
                (strictly less than), else False.",
        checker: "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n\
                  assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0], 0.3) == True\n\
                  assert has_close_elements([], 1.0) == False\n",
    },
    CodeTask {
        name: "sum_even_numbers",
        spec: "Define a function `sum_even_numbers(nums)` that returns the sum of the even \
                integers in the list `nums`.",
        checker: "assert sum_even_numbers([1, 2, 3, 4, 5, 6]) == 12\n\
                  assert sum_even_numbers([1, 3, 5]) == 0\n\
                  assert sum_even_numbers([]) == 0\n",
    },
    CodeTask {
        name: "reverse_words",
        spec: "Define a function `reverse_words(s)` that returns the string `s` with the order \
                of its whitespace-separated words reversed (single spaces between words in the \
                result).",
        checker: "assert reverse_words('hello world foo') == 'foo world hello'\n\
                  assert reverse_words('single') == 'single'\n\
                  assert reverse_words('') == ''\n",
    },
    CodeTask {
        name: "is_prime",
        spec: "Define a function `is_prime(n)` that returns True if the integer `n` is a prime \
                number, else False. Numbers below 2 are not prime.",
        checker: "assert is_prime(2) == True\n\
                  assert is_prime(17) == True\n\
                  assert is_prime(1) == False\n\
                  assert is_prime(15) == False\n\
                  assert is_prime(0) == False\n",
    },
    CodeTask {
        name: "roman_to_int",
        spec: "Define a function `roman_to_int(s)` that converts a valid Roman numeral string \
                `s` (uppercase, e.g. 'MCMXciv' will not appear — only uppercase) to its integer \
                value.",
        checker: "assert roman_to_int('III') == 3\n\
                  assert roman_to_int('IV') == 4\n\
                  assert roman_to_int('IX') == 9\n\
                  assert roman_to_int('LVIII') == 58\n\
                  assert roman_to_int('MCMXCIV') == 1994\n",
    },
    // ── Harder tasks (DP / parsing / algorithms) — these discriminate ──────────
    CodeTask {
        name: "is_balanced_brackets",
        spec: "Define a function `is_balanced(s)` that returns True if the brackets in `s` \
                (only the characters (), [], {}) are correctly balanced and nested, else False. \
                Other characters may appear and are ignored.",
        checker: "assert is_balanced('(a[b]{c})') == True\n\
                  assert is_balanced('([)]') == False\n\
                  assert is_balanced('(((') == False\n\
                  assert is_balanced('') == True\n\
                  assert is_balanced('a)') == False\n",
    },
    CodeTask {
        name: "two_sum",
        spec: "Define a function `two_sum(nums, target)` that returns a list of the two DISTINCT \
                indices `[i, j]` (i < j) such that nums[i] + nums[j] == target. Exactly one \
                solution exists. Return the indices in increasing order.",
        checker: "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n\
                  assert two_sum([3, 2, 4], 6) == [1, 2]\n\
                  assert two_sum([3, 3], 6) == [0, 1]\n",
    },
    CodeTask {
        name: "merge_intervals",
        spec: "Define a function `merge_intervals(intervals)` that merges all overlapping \
                intervals. `intervals` is a list of [start, end] pairs. Return the merged list \
                sorted by start. Intervals that touch at an endpoint (e.g. [1,4] and [4,5]) merge.",
        checker: "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n\
                  assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n\
                  assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n\
                  assert merge_intervals([]) == []\n",
    },
    CodeTask {
        name: "longest_common_subsequence",
        spec: "Define a function `lcs(a, b)` that returns the LENGTH of the longest common \
                subsequence of strings `a` and `b` (characters in order but not necessarily \
                contiguous).",
        checker: "assert lcs('abcde', 'ace') == 3\n\
                  assert lcs('abc', 'abc') == 3\n\
                  assert lcs('abc', 'def') == 0\n\
                  assert lcs('AGGTAB', 'GXTXAYB') == 4\n",
    },
    CodeTask {
        name: "int_to_roman",
        spec: "Define a function `int_to_roman(n)` that converts an integer `n` (1..=3999) to its \
                Roman numeral string (uppercase). Uses subtractive forms (IV, IX, XL, XC, CD, CM).",
        checker: "assert int_to_roman(3) == 'III'\n\
                  assert int_to_roman(4) == 'IV'\n\
                  assert int_to_roman(9) == 'IX'\n\
                  assert int_to_roman(58) == 'LVIII'\n\
                  assert int_to_roman(1994) == 'MCMXCIV'\n",
    },
    CodeTask {
        name: "flatten_nested_list",
        spec: "Define a function `flatten(xs)` that returns a flat list of all the non-list \
                elements of the arbitrarily-nested list `xs`, preserving left-to-right order.",
        checker: "assert flatten([1, [2, [3, 4], 5], [6]]) == [1, 2, 3, 4, 5, 6]\n\
                  assert flatten([]) == []\n\
                  assert flatten([[[[1]]]]) == [1]\n\
                  assert flatten([1, 2, 3]) == [1, 2, 3]\n",
    },
];

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

/// Pull the code out of the first fenced block (```lang … ```), else the whole
/// response.
fn extract_code(response: &str) -> String {
    if let Some(fence) = response.find("```") {
        let after = &response[fence + 3..];
        // Skip the optional language tag on the fence line.
        let body = match after.find('\n') {
            Some(nl) => &after[nl + 1..],
            None => after,
        };
        if let Some(end) = body.find("```") {
            return body[..end].to_string();
        }
        return body.to_string();
    }
    response.to_string()
}

/// Write `code` to a temp .py, run it with a timeout, return whether it exited 0.
fn run_python(py: &str, code: &str) -> Result<bool, String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);

    let path = std::env::temp_dir().join(format!(
        "cgbench_{}_{}.py",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, code).map_err(|e| e.to_string())?;

    let spawn = std::process::Command::new(py)
        .arg(&path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let mut child = match spawn {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            return Err(e.to_string());
        }
    };

    let start = std::time::Instant::now();
    let exit_ok = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status.success(),
            None => {
                if start.elapsed() > RUN_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    break false; // timed out (e.g. infinite loop)
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    };
    let _ = std::fs::remove_file(&path);
    Ok(exit_ok)
}

pub(crate) fn tests_code_gen_bench() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Code-gen benchmark (live model, execution-verified)"
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
        println!("  {} skipping code-gen benchmark ({})", yellow("SKIP"), why);
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
            name: "code_gen_bench".to_string(),
            results,
        };
    }
    let py = py.unwrap();
    crate::bench_util::warn_if_cpu_offloaded();

    for task in TASKS {
        results.push(run_test(
            &format!("code-gen: {} (pass@1)", task.name),
            || {
                let mut a = crate::bench_util::bench_assistant();
                let prompt = format!(
                    "Write Python 3 code for the following. Output ONLY the code inside a single \
                     ```python code block, with no explanation and no example usage.\n\n{}",
                    task.spec
                );
                let resp = a.generate_sync(prompt, "").map_err(|e| e.to_string())?;
                let code = extract_code(&resp);
                let program = format!(
                    "{code}\n\n# --- checker ---\n{}\nprint('OK')\n",
                    task.checker
                );
                match run_python(py, &program) {
                    Ok(true) => Ok(()),
                    Ok(false) => Err(format!(
                        "generated code failed its checker. code:\n{:.500}",
                        code
                    )),
                    Err(e) => Err(format!("could not run python: {e}")),
                }
            },
        ));
    }

    // Summary line: pass@1 across the suite.
    let solved = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} code_gen_bench pass@1: {}/{} (execution-verified, backend={})",
        bold(&cyan("∑")),
        solved,
        total,
        crate::bench_util::bench_label()
    );

    CategoryResult {
        name: "code_gen_bench".to_string(),
        results,
    }
}
