use super::*;

use std::sync::{Arc, Mutex};

use ai_assistant::unified_tools::ToolRegistry;
use ai_assistant::{AgentPolicyBuilder, AutonomousAgent, AutonomyLevel, OperationMode};

use crate::agentic_code::{make_generator, MAX_ITERS};
use crate::bench_stats::BACKEND_CRASH_PREFIX;

// ─── Can the model EDIT code, rather than write it? ───────────────────────────
//
// Every other agentic category starts the model from an empty `src/lib.rs`. That
// measures writing, and by V276 the single-step versions were saturated: 12/12 for
// anything from 7B up. Editing an existing codebase is a different skill, and two
// parts of it are invisible to a one-file scaffold:
//
//   * LOCALISATION — the task says what is wrong, not where. With four modules and a
//     test file, the model has to read before it writes. A one-file crate hands the
//     answer over for free.
//   * NOT BREAKING THINGS — the crate arrives with a passing test suite covering code
//     the task never mentions. Rewriting a file wholesale, the classic failure of a
//     model that would rather regenerate than edit, is caught by construction.
//
// So each task is scored through TWO gates, run separately on purpose:
//
//   1. the seeded tests still pass  (did it break the codebase?)
//   2. our checker passes           (did it do the job?)
//
// One `cargo test` could answer both at once, and that is exactly what would make the
// result useless: "broke something else" and "never made the change" would arrive as
// the same failure. They call for opposite responses, so they are measured apart and
// reported apart — see the failure-mode table at the bottom.

/// A pre-existing crate plus a change request.
struct EditTask {
    name: &'static str,
    /// The crate as it stands BEFORE the agent touches it: (relative path, contents).
    /// Must include a `tests/` file that passes as seeded.
    files: &'static [(&'static str, &'static str)],
    /// What the agent is asked to do. Deliberately says WHAT is wrong, not WHERE.
    prompt: &'static str,
    /// Appended to `src/lib.rs` to prove the requested change happened.
    checker: &'static str,
}

// ── Seed crate: four small modules and a test suite that passes ──────────────
//
// Kept small enough to read in a couple of tool calls, but with more than one
// plausible place for each bug, so "find the file" is a real step.

const LIB_RS: &str = "pub mod format;\npub mod parser;\npub mod tally;\npub mod version;\n";

const FORMAT_RS: &str = r#"/// Shorten `s` so the result is never longer than `max` characters,
/// appending an ellipsis when it had to cut.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max - 3).collect();
    format!("{kept}...")
}

/// Pad `s` on the right with spaces up to `width`.
pub fn pad_right(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        return s.to_string();
    }
    let mut out = s.to_string();
    out.extend(std::iter::repeat_n(' ', width - len));
    out
}
"#;

const PARSER_RS: &str = r#"/// Parse a size such as 10kb or 3mb into bytes. Case-insensitive.
/// Returns None when the input is not a number followed by a known unit.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    let (digits, unit) = s.split_at(s.find(|c: char| c.is_alphabetic())?);
    let n: u64 = digits.trim().parse().ok()?;
    match unit {
        "b" => Some(n),
        "kb" => Some(n * 1024),
        "mb" => Some(n * 1024 * 1024),
        _ => None,
    }
}
"#;

const TALLY_RS: &str = r#"use std::collections::HashMap;

/// Count how many times each item appears.
pub fn counts(items: &[String]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for i in items {
        *m.entry(i.clone()).or_insert(0) += 1;
    }
    m
}

/// The item that appears most often, or None when there is nothing to count.
pub fn most_common(items: &[String]) -> Option<String> {
    let m = counts(items);
    m.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)
}
"#;

const VERSION_RS: &str = r#"/// Compare two dotted versions. Returns Ordering::Less when `a` is older.
pub fn compare(a: &str, b: &str) -> std::cmp::Ordering {
    let parse = |v: &str| -> Vec<u64> { v.split('.').map(|p| p.parse().unwrap_or(0)).collect() };
    parse(a).cmp(&parse(b))
}
"#;

const TESTS_RS: &str = r#"use task::*;

#[test]
fn pad_right_pads_and_leaves_long_strings_alone() {
    assert_eq!(format::pad_right("ab", 4), "ab  ");
    assert_eq!(format::pad_right("abcde", 3), "abcde");
}

#[test]
fn truncate_keeps_short_strings() {
    assert_eq!(format::truncate("hello", 10), "hello");
}

#[test]
fn truncate_marks_what_it_cut() {
    // Passes as seeded (the bug only bites below 3), and it is what stops a "fix"
    // that simply deletes the ellipsis: dropping documented behaviour to make a
    // length assertion pass is breaking the crate, not repairing it.
    assert_eq!(format::truncate("hello world", 8), "hello...");
}

#[test]
fn parse_size_handles_the_known_units() {
    assert_eq!(parser::parse_size("10b"), Some(10));
    assert_eq!(parser::parse_size("2kb"), Some(2048));
    assert_eq!(parser::parse_size("1mb"), Some(1024 * 1024));
    assert_eq!(parser::parse_size("nonsense"), None);
}

#[test]
fn tally_counts_and_finds_the_most_common() {
    let items: Vec<String> = ["a", "b", "a"].iter().map(|s| s.to_string()).collect();
    assert_eq!(tally::counts(&items).get("a"), Some(&2));
    assert_eq!(tally::most_common(&items), Some("a".to_string()));
    assert_eq!(tally::most_common(&[]), None);
}

#[test]
fn version_compare_orders_releases() {
    assert_eq!(version::compare("1.2.0", "1.10.0"), std::cmp::Ordering::Less);
    assert_eq!(version::compare("2.0.0", "2.0.0"), std::cmp::Ordering::Equal);
}
"#;

// A caller the task never mentions. `most_common` looks like a one-line change until
// you notice something already depends on its return TYPE.
//
// The first version of this file interpolated the value — `format!("{top}")` — and the
// oracle audit caught that immediately: `Display` is implemented for both `String` and
// `&String`, so the caller kept compiling and the "re-point the callers" task never
// asked anyone to re-point anything. Storing it in an owned field is what makes the
// dependency real: with `Option<&String>`, `Summary { top }` stops compiling until the
// caller clones.
const REPORT_RS: &str = r#"use crate::tally;

/// The headline figure for a run, kept for later.
pub struct Summary {
    pub top: String,
    pub total: usize,
}

pub fn summarise(items: &[String]) -> Summary {
    match tally::most_common(items) {
        Some(top) => Summary {
            top,
            total: items.len(),
        },
        None => Summary {
            top: String::new(),
            total: 0,
        },
    }
}

/// The most common item, padded for a fixed-width column.
pub fn padded_top(items: &[String], width: usize) -> String {
    // Second copy of `format::pad_right`. Deliberate: the deduplication task needs two
    // implementations that are the same thing under different names, so that noticing
    // they ARE the same is the work.
    let s = summarise(items).top;
    let len = s.chars().count();
    if len >= width {
        return s;
    }
    let mut out = s;
    out.extend(std::iter::repeat_n(' ', width - len));
    out
}
"#;

const LIB_WITH_REPORT_RS: &str =
    "pub mod format;\npub mod parser;\npub mod report;\npub mod tally;\npub mod version;\n";

const TESTS_WITH_REPORT_RS: &str = r#"use task::*;

#[test]
fn pad_right_pads_and_leaves_long_strings_alone() {
    assert_eq!(format::pad_right("ab", 4), "ab  ");
    assert_eq!(format::pad_right("abcde", 3), "abcde");
}

#[test]
fn parse_size_handles_the_known_units() {
    assert_eq!(parser::parse_size("2kb"), Some(2048));
    assert_eq!(parser::parse_size("nonsense"), None);
}

#[test]
fn report_summarises_the_most_common_item() {
    let items: Vec<String> = ["a", "b", "a"].iter().map(|s| s.to_string()).collect();
    let s = report::summarise(&items);
    assert_eq!(s.top, "a");
    assert_eq!(s.total, 3);
    assert_eq!(report::summarise(&[]).top, "");
    assert_eq!(report::padded_top(&items, 4), "a   ");
}

#[test]
fn version_compare_orders_releases() {
    assert_eq!(version::compare("1.2.0", "1.10.0"), std::cmp::Ordering::Less);
}
"#;

// Same crate with one function nobody calls, sitting next to a near-identical name
// that everything does. Deleting by resemblance rather than by usage breaks it.
const TALLY_DEAD_RS: &str = r#"use std::collections::HashMap;

/// Count how many times each item appears.
pub fn counts(items: &[String]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for i in items {
        *m.entry(i.clone()).or_insert(0) += 1;
    }
    m
}

/// The item that appears most often, or None when there is nothing to count.
pub fn most_common(items: &[String]) -> Option<String> {
    let m = counts(items);
    m.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)
}

/// Quadratic version kept around from an earlier iteration. Nothing calls it.
pub fn most_common_slow(items: &[String]) -> Option<String> {
    let mut best: Option<(String, usize)> = None;
    for a in items {
        let n = items.iter().filter(|b| *b == a).count();
        if best.as_ref().is_none_or(|(_, bn)| n > *bn) {
            best = Some((a.clone(), n));
        }
    }
    best.map(|(k, _)| k)
}
"#;

const EDIT_TASKS: &[EditTask] = &[
    EditTask {
        name: "find and fix the reported panic",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        // The symptom, not the location: `max - 3` underflows for max < 3.
        prompt: "A user reports that this crate panics with an arithmetic overflow when \
                 shortening a string to a very small maximum length, for example a maximum \
                 of 2. Find the cause and fix it so that no input panics and the result is \
                 never longer than the maximum asked for. Do not change any public function \
                 signature, and do not break the existing tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(crate::format::truncate(\"hello world\", 2).chars().count() <= 2, true);\n        \
                  assert_eq!(crate::format::truncate(\"hello world\", 0).chars().count(), 0);\n        \
                  assert_eq!(crate::format::truncate(\"hello\", 10), \"hello\");\n        \
                  assert!(crate::format::truncate(\"hello world\", 8).chars().count() <= 8);\n    }\n",
    },
    EditTask {
        name: "extend a function without changing its shape",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        prompt: "This crate parses sizes like 10kb and 3mb. Add support for gigabytes, so \
                 that a value such as 2gb parses as well, using the same 1024-based scale as \
                 the existing units. Keep the existing units working, keep the function \
                 signature exactly as it is, and do not break the existing tests. \
                 Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(crate::parser::parse_size(\"2gb\"), Some(2 * 1024 * 1024 * 1024));\n        \
                  assert_eq!(crate::parser::parse_size(\"1gb\"), Some(1024 * 1024 * 1024));\n        \
                  assert_eq!(crate::parser::parse_size(\"2kb\"), Some(2048));\n        \
                  assert_eq!(crate::parser::parse_size(\"nonsense\"), None);\n    }\n",
    },
    EditTask {
        name: "change a signature and re-point its callers",
        files: &[
            ("src/lib.rs", LIB_WITH_REPORT_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("src/report.rs", REPORT_RS),
            ("tests/existing.rs", TESTS_WITH_REPORT_RS),
        ],
        // The invalidation flavour, in an editing setting: the change is trivial in
        // isolation and only hard because something ELSE already calls it. `report.rs`
        // is the caller, and the model is not told it exists.
        prompt: "The function `tally::most_common` returns an owned String, which forces a \
                 clone on every call. Change it to return a borrowed value instead, so that \
                 it hands back a reference into the input rather than a copy, and update \
                 everything in this crate that calls it so the crate still compiles. Do not \
                 break the existing tests. Run cargo test.",
        // The type annotation IS the assertion: it only compiles if the signature
        // actually changed. Without it, `.map(|s| s.to_string())` accepts both the old
        // `Option<String>` and the new `Option<&String>`, so a model that changed
        // nothing at all would pass.
        checker: "    #[test]\n    fn check() {\n        \
                  let items: Vec<String> = [\"x\", \"y\", \"x\"].iter().map(|s| s.to_string()).collect();\n        \
                  let got: Option<&String> = crate::tally::most_common(&items);\n        \
                  assert_eq!(got.map(|s| s.as_str()), Some(\"x\"));\n        \
                  let none: Vec<String> = vec![];\n        \
                  let empty: Option<&String> = crate::tally::most_common(&none);\n        \
                  assert!(empty.is_none());\n    }\n",
    },
    EditTask {
        name: "fix an edge case the tests never covered",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        // `parse(a).cmp(&parse(b))` on vectors of different length: [1,2] sorts before
        // [1,2,0] because a prefix is Less. The seeded tests all compare versions with
        // the same number of parts, so nothing catches it — which is the point. The
        // model cannot lean on a failing test to find this one.
        prompt: "Comparing versions with a different number of parts is wrong in this \
                 crate: a version like 1.2 is treated as OLDER than 1.2.0, when the two \
                 mean the same release. Find it and fix it so that missing trailing parts \
                 count as zero. Keep every existing comparison behaving as it does now, \
                 and do not break the existing tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  use std::cmp::Ordering;\n        \
                  assert_eq!(crate::version::compare(\"1.2\", \"1.2.0\"), Ordering::Equal);\n        \
                  assert_eq!(crate::version::compare(\"1.2\", \"1.2.1\"), Ordering::Less);\n        \
                  assert_eq!(crate::version::compare(\"1.3\", \"1.2.9\"), Ordering::Greater);\n        \
                  assert_eq!(crate::version::compare(\"1.2.0\", \"1.10.0\"), Ordering::Less);\n    }\n",
    },
    EditTask {
        name: "add a module and wire it in",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        // Writing the function is trivial; the step models skip is `pub mod stats;` in
        // lib.rs. A new file nobody declares is invisible to the compiler, and the
        // failure is silent in the sense that matters: the crate still builds, and the
        // module simply is not there.
        prompt: "Add a new module `stats` to this crate, in its own file, with one public \
                 function `mean(values: &[f64]) -> f64` returning the arithmetic mean, or \
                 0.0 for an empty slice. Make sure it is reachable from outside the crate \
                 as `stats::mean`, keep everything else as it is, and do not break the \
                 existing tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert!((crate::stats::mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-9);\n        \
                  assert!((crate::stats::mean(&[5.0]) - 5.0).abs() < 1e-9);\n        \
                  assert!((crate::stats::mean(&[]) - 0.0).abs() < 1e-9);\n    }\n",
    },
    EditTask {
        name: "delete the dead one, keep the live one",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_DEAD_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        // Two near-identical names, one used and one not. Deleting by name-similarity
        // instead of by usage takes the crate down, and gate 1 says so.
        prompt: "This crate has accumulated an unused public function in one of its modules \
                 — nothing in the crate or its tests calls it. Find it and delete it. Leave \
                 everything that IS used exactly as it is, and do not break the existing \
                 tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  let items: Vec<String> = [\"a\", \"b\", \"a\"].iter().map(|s| s.to_string()).collect();\n        \
                  assert_eq!(crate::tally::most_common(&items), Some(\"a\".to_string()));\n        \
                  let src = include_str!(\"tally.rs\");\n        \
                  assert!(!src.contains(\"fn most_common_slow\"), \"the dead function is still there\");\n    }\n",
    },
    EditTask {
        name: "make a panicking function fallible",
        files: &[
            ("src/lib.rs", LIB_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("tests/existing.rs", TESTS_RS),
        ],
        // Changing a return type ripples: `parse_size` is called from the seeded tests,
        // which the model may not touch — so the only way through is a signature the
        // existing assertions still satisfy. `Option<u64>` -> `Result<u64, String>`
        // would break them; the task asks for the one change that does not.
        prompt: "`parser::parse_size` currently multiplies without checking, so a huge \
                 value such as 999999999999999999mb overflows and panics in debug builds. \
                 Make it return None for values that would overflow, instead of panicking. \
                 Keep the signature and every existing behaviour exactly as they are, and \
                 do not break the existing tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(crate::parser::parse_size(\"999999999999999999mb\"), None);\n        \
                  assert_eq!(crate::parser::parse_size(\"18446744073709551615kb\"), None);\n        \
                  assert_eq!(crate::parser::parse_size(\"2kb\"), Some(2048));\n        \
                  assert_eq!(crate::parser::parse_size(\"1mb\"), Some(1024 * 1024));\n        \
                  assert_eq!(crate::parser::parse_size(\"10b\"), Some(10));\n    }\n",
    },
    EditTask {
        name: "two modules, one shared helper",
        files: &[
            ("src/lib.rs", LIB_WITH_REPORT_RS),
            ("src/format.rs", FORMAT_RS),
            ("src/parser.rs", PARSER_RS),
            ("src/tally.rs", TALLY_RS),
            ("src/version.rs", VERSION_RS),
            ("src/report.rs", REPORT_RS),
            ("tests/existing.rs", TESTS_WITH_REPORT_RS),
        ],
        // Deduplication across files: the same padding logic exists in two modules, and
        // the model must notice they are the same thing before it can remove one. Doing
        // it by deleting the duplicate without re-pointing its user breaks the crate.
        prompt: "Two modules in this crate each contain their own copy of the same \
                 right-padding logic. Remove the duplication: keep one implementation, \
                 have the other module use it, and leave the public behaviour of both \
                 modules exactly as it is. Do not break the existing tests. Run cargo test.",
        checker: "    #[test]\n    fn check() {\n        \
                  assert_eq!(crate::format::pad_right(\"ab\", 4), \"ab  \");\n        \
                  let items: Vec<String> = [\"a\", \"b\", \"a\"].iter().map(|s| s.to_string()).collect();\n        \
                  assert_eq!(crate::report::padded_top(&items, 4), \"a   \");\n        \
                  let src = include_str!(\"report.rs\");\n        \
                  assert!(!src.contains(\"repeat_n\"), \"report.rs still pads on its own\");\n    }\n",
    },
];

/// What went wrong, kept apart because the two failures mean opposite things.
enum EditOutcome {
    Solved,
    /// The seeded suite no longer passes: the model damaged code it was not asked to touch.
    BrokeExisting(String),
    /// The seeded suite still passes, but the requested change is not there.
    NotDone(String),
}

fn run_one_edit(cargo: &'static str, task: &EditTask) -> Result<(), String> {
    let ws = crate::agentic_rust::scaffold_crate_files(task.files)?;

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
    let mut agent = AutonomousAgent::builder("rust-editor", generator)
        .max_iterations(MAX_ITERS)
        .system_prompt(EDIT_SYSTEM_PROMPT)
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    let _ = agent.run(task.prompt);

    if crate::agentic_code::hit_backend_failure(&agent) {
        let _ = std::fs::remove_dir_all(&ws);
        return Err(format!(
            "{BACKEND_CRASH_PREFIX}, not a model failure (this run leaves the denominator) \
             — the generation never returned; check the backend log."
        ));
    }

    let outcome = judge(cargo, &ws, task);
    let _ = std::fs::remove_dir_all(&ws);

    match outcome {
        EditOutcome::Solved => Ok(()),
        EditOutcome::BrokeExisting(out) => Err(format!(
            "broke the existing tests — it damaged code the task never mentioned: {}",
            first_failure_line(&out)
        )),
        EditOutcome::NotDone(out) => Err(format!(
            "existing tests still pass but the change is not there: {}",
            first_failure_line(&out)
        )),
    }
}

/// Gate 1 then gate 2, in that order and separately — see the module comment.
fn judge(cargo: &'static str, ws: &std::path::Path, task: &EditTask) -> EditOutcome {
    // Gate 1: the crate as the model left it, with only the seeded tests.
    let out = crate::agentic_code::run_capture(
        cargo,
        &["test".to_string(), "--quiet".to_string()],
        ws,
        std::time::Duration::from_secs(120),
    );
    if !out.starts_with("exit_code=0") {
        return EditOutcome::BrokeExisting(out);
    }

    // Gate 2: same crate, with our checker appended to lib.rs.
    let lib = ws.join("src").join("lib.rs");
    let Ok(src) = std::fs::read_to_string(&lib) else {
        return EditOutcome::NotDone("src/lib.rs is missing".to_string());
    };
    let full = format!(
        "{src}\n\n#[cfg(test)]\nmod harness_checker {{\n    use super::*;\n{}}}\n",
        task.checker
    );
    if std::fs::write(&lib, &full).is_err() {
        return EditOutcome::NotDone("could not write the checker".to_string());
    }
    let out = crate::agentic_code::run_capture(
        cargo,
        &["test".to_string(), "--quiet".to_string()],
        ws,
        std::time::Duration::from_secs(120),
    );
    let _ = std::fs::write(&lib, src); // put the model's file back for the debug dump
    if out.starts_with("exit_code=0") {
        EditOutcome::Solved
    } else {
        EditOutcome::NotDone(out)
    }
}

/// The first line of cargo output that names a failure, so the per-task line says
/// something instead of carrying 2 kB of build log.
fn first_failure_line(out: &str) -> String {
    out.lines()
        .find(|l| {
            let l = l.trim_start();
            l.starts_with("error") || l.starts_with("assertion") || l.contains("panicked at")
        })
        .unwrap_or("(no error line in the cargo output)")
        .trim()
        .chars()
        .take(160)
        .collect()
}

const EDIT_SYSTEM_PROMPT: &str = r#"You are an autonomous Rust engineer EDITING an existing crate. You act only by calling tools.

To call tools, reply with ONLY a JSON array (no prose, no ``` fences), each element being {"name": <tool>, "arguments": {<args>}}. Example:
[{"name":"read_file","arguments":{"path":"src/lib.rs"}}]

The crate is called `task` and it already exists, with several modules under `src/` and a test suite under `tests/`. It compiles and its tests pass right now.

Available tools:
- list_dir(): list the files in the crate.
- read_file(path): read a file, e.g. "src/lib.rs".
- write_file(path, content): overwrite a file with the FULL new content.
- run_command(command): run `cargo test`, `cargo build` or `cargo check`.

Rules:
- READ before you write. You are not told which file to change; find it.
- Change as little as possible. Every existing test must still pass when you are done — they cover code your task does not mention, and breaking them is a failure even if your own change works.
- `write_file` replaces the whole file, so include everything you want to keep.
- Output ONLY the JSON array — nothing before or after it, no code fences.
- When the change is made and `cargo test` passes, reply with a short plain-text confirmation and NO JSON array."#;

pub(crate) fn tests_agentic_edit() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Editing an existing crate (find the file, fix it, break nothing)"
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
        println!("  {} skipping agentic-edit ({why})", yellow("SKIP"));
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
            name: "agentic_edit".to_string(),
            results,
        };
    }
    let cargo = cargo.unwrap();
    println!("  backend: {}", crate::bench_util::bench_label());
    crate::bench_util::warn_if_cpu_offloaded();

    let repeats = crate::bench_util::bench_repeats();
    let tasks: Vec<&EditTask> = EDIT_TASKS
        .iter()
        .filter(|t| crate::should_run(&format!("edit: {}", t.name)))
        .collect();
    let outcomes = crate::bench_stats::run_interleaved(
        &tasks,
        |t| format!("edit: {}", t.name),
        repeats,
        |task| run_one_edit(cargo, task),
    );

    results.extend(crate::bench_stats::to_results(&outcomes, repeats));
    crate::bench_stats::print_summary(
        "agentic_edit",
        "changes made without breaking the crate",
        &[
            ("broke the existing tests", "broke untouched code"),
            ("the change is not there", "did not make the change"),
        ],
        &outcomes,
        repeats,
    );

    CategoryResult {
        name: "agentic_edit".to_string(),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests below. They each build a crate called `task`, and
    /// `scaffold_crate` deliberately points every one of them at ONE shared cargo
    /// target directory — which is a large speed win when the category runs its tasks
    /// in sequence, and a race when `cargo test` runs these in parallel: same package
    /// name, different sources, one set of build artifacts. It surfaced as a seed crate
    /// "failing its own tests" with an error from a DIFFERENT task's test file, which
    /// reads like a wiring bug and is not one.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The seeded crate must arrive GREEN, or gate 1 accuses the model of breaking
    /// something that was already broken. Cheap to assert, and impossible to notice
    /// otherwise until a whole sweep reports "broke untouched code" for every task.
    #[test]
    fn every_task_seeds_a_crate_whose_tests_already_pass() {
        let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Some(cargo) = crate::agentic_rust::cargo_cmd_pub() else {
            return; // no cargo here; the category skips too
        };
        for task in EDIT_TASKS {
            let ws = crate::agentic_rust::scaffold_crate_files(task.files)
                .expect("scaffolding the seed crate");
            let out = crate::agentic_code::run_capture(
                cargo,
                &["test".to_string(), "--quiet".to_string()],
                &ws,
                std::time::Duration::from_secs(180),
            );
            let ok = out.starts_with("exit_code=0");
            let _ = std::fs::remove_dir_all(&ws);
            assert!(
                ok,
                "seed crate for {:?} does not pass its own tests:\n{}",
                task.name,
                out.chars().take(600).collect::<String>()
            );
        }
    }

    /// Apply `replacement` in place of one seeded file and report which gate fires.
    /// This is the same crate, the same two gates and the same cargo the category
    /// uses — auditing a copy would audit the copy.
    fn judge_with(task: &EditTask, path: &str, replacement: &str) -> &'static str {
        let cargo = crate::agentic_rust::cargo_cmd_pub().expect("cargo");
        let files: Vec<(&str, &str)> = task
            .files
            .iter()
            .map(|(p, c)| {
                if *p == path {
                    (*p, replacement)
                } else {
                    (*p, *c)
                }
            })
            .collect();
        let ws = crate::agentic_rust::scaffold_crate_files(&files).expect("scaffold");
        // Print WHY, not just the verdict: an unexpected verdict here is usually the
        // mutant failing to compile rather than the oracle misjudging it, and those look
        // identical from the outside. Visible with `cargo test -- --nocapture`.
        let verdict = match judge(cargo, &ws, task) {
            EditOutcome::Solved => "solved",
            EditOutcome::BrokeExisting(out) => {
                println!(
                    "    [audit] {} / {path}: BROKE — {}",
                    task.name,
                    first_failure_line(&out)
                );
                "broke"
            }
            EditOutcome::NotDone(out) => {
                println!(
                    "    [audit] {} / {path}: NOT DONE — {}",
                    task.name,
                    first_failure_line(&out)
                );
                "not-done"
            }
        };
        let _ = std::fs::remove_dir_all(&ws);
        verdict
    }

    /// Mutation-test the oracle before any model is scored against it: a correct fix
    /// must pass BOTH gates, and each plausible wrong fix must fail the RIGHT one.
    /// A checker that cannot tell "broke the crate" from "did not do the job" would
    /// make this whole category report one number for two opposite problems.
    #[test]
    fn the_two_gates_accept_a_real_fix_and_reject_the_plausible_wrong_ones() {
        let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        if crate::agentic_rust::cargo_cmd_pub().is_none() {
            return;
        }
        let panic_task = &EDIT_TASKS[0];

        // Correct: clamp so the ellipsis only appears when there is room for it.
        let fixed = FORMAT_RS.replace(
            "    let kept: String = s.chars().take(max - 3).collect();\n    format!(\"{kept}...\")",
            "    if max <= 3 {\n        return s.chars().take(max).collect();\n    }\n    \
             let kept: String = s.chars().take(max - 3).collect();\n    format!(\"{kept}...\")",
        );
        assert_ne!(fixed, FORMAT_RS, "the reference edit did not apply");
        assert_eq!(judge_with(panic_task, "src/format.rs", &fixed), "solved");

        // Wrong 1: saturating_sub stops the panic but still returns max+1 characters
        // for max = 2 ("..." is three). The classic "made the crash go away" fix.
        let saturating = FORMAT_RS.replace("take(max - 3)", "take(max.saturating_sub(3))");
        assert_ne!(saturating, FORMAT_RS);
        assert_eq!(
            judge_with(panic_task, "src/format.rs", &saturating),
            "not-done",
            "a fix that stops the panic but still overruns `max` must not count as solved"
        );

        // Wrong 2: satisfies the length rule by dropping the ellipsis entirely —
        // caught by the SEEDED suite, because that behaviour was already documented
        // and tested. This is the case that separates "fixed it" from "made the
        // assertion pass".
        let no_ellipsis = FORMAT_RS.replace(
            "    let kept: String = s.chars().take(max - 3).collect();\n    format!(\"{kept}...\")",
            "    s.chars().take(max).collect()",
        );
        assert_ne!(no_ellipsis, FORMAT_RS);
        assert_eq!(
            judge_with(panic_task, "src/format.rs", &no_ellipsis),
            "broke",
            "dropping documented behaviour must register as breaking the crate"
        );

        // And the same for the second task: a 1000-based gigabyte is the likeliest
        // wrong answer, and the units already in the crate are 1024-based.
        let extend_task = &EDIT_TASKS[1];
        let decimal_gb = PARSER_RS.replace(
            "        _ => None,",
            "        \"gb\" => Some(n * 1000 * 1000 * 1000),\n        _ => None,",
        );
        assert_ne!(decimal_gb, PARSER_RS);
        assert_eq!(
            judge_with(extend_task, "src/parser.rs", &decimal_gb),
            "not-done"
        );

        // Task 3 (borrow instead of clone). The case that matters is DOING NOTHING:
        // without the type annotation in the checker, `.map(|s| s.to_string())` accepts
        // the old signature and the new one alike, so an untouched crate scored as
        // solved. It must read as "not done".
        let borrow_task = &EDIT_TASKS[2];
        assert_eq!(
            judge_with(borrow_task, "src/tally.rs", TALLY_RS),
            "not-done",
            "leaving most_common untouched must not count as changing its signature"
        );
        // And changing it without updating the caller in report.rs takes the crate
        // down — which is a break, not a missing change.
        let borrowed_only = TALLY_RS.replace(
            "pub fn most_common(items: &[String]) -> Option<String> {\n    let m = counts(items);\n    m.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)\n}",
            "pub fn most_common(items: &[String]) -> Option<&String> {\n    let m = counts(items);\n    let top = m.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)?;\n    items.iter().find(|i| **i == top)\n}",
        );
        assert_ne!(
            borrowed_only, TALLY_RS,
            "the borrowed variant did not apply"
        );
        assert_eq!(
            judge_with(borrow_task, "src/tally.rs", &borrowed_only),
            "broke",
            "changing the signature and leaving the caller behind is a break"
        );

        // Task 4 (version edge case). The likeliest wrong fix is to TRUNCATE both sides
        // to the shorter length instead of padding the shorter one with zeros: it makes
        // 1.2 == 1.2.0 (the reported symptom) while quietly making 1.2 == 1.2.1 as well.
        // A checker that only tested the reported case would wave it through, which is
        // why the separating case is in there.
        let version_task = &EDIT_TASKS[3];
        let truncating = VERSION_RS.replace(
            "    parse(a).cmp(&parse(b))",
            "    let (mut x, mut y) = (parse(a), parse(b));\n    let n = x.len().min(y.len());\n    x.truncate(n);\n    y.truncate(n);\n    x.cmp(&y)",
        );
        assert_ne!(truncating, VERSION_RS);
        assert_eq!(
            judge_with(version_task, "src/version.rs", &truncating),
            "not-done",
            "truncating to the shorter version makes 1.2 == 1.2.1, which is not a fix"
        );

        // Task 5 (add a module and wire it in). The failure worth catching is writing
        // src/stats.rs and never declaring it: the crate still compiles, the seeded
        // tests still pass, and the module simply does not exist. Gate 1 stays green
        // and gate 2 must be the one that fires.
        let wire_task = &EDIT_TASKS[4];
        {
            let cargo = crate::agentic_rust::cargo_cmd_pub().expect("cargo");
            let mut files: Vec<(&str, &str)> = wire_task.files.to_vec();
            files.push((
                "src/stats.rs",
                "pub fn mean(values: &[f64]) -> f64 {\n    if values.is_empty() {\n        return 0.0;\n    }\n    values.iter().sum::<f64>() / values.len() as f64\n}\n",
            ));
            let ws = crate::agentic_rust::scaffold_crate_files(&files).expect("scaffold");
            let verdict = match judge(cargo, &ws, wire_task) {
                EditOutcome::Solved => "solved",
                EditOutcome::BrokeExisting(_) => "broke",
                EditOutcome::NotDone(_) => "not-done",
            };
            let _ = std::fs::remove_dir_all(&ws);
            assert_eq!(
                verdict, "not-done",
                "a module file that lib.rs never declares is not a wired-in module"
            );
        }

        // Task 6 (delete the dead one). Deleting the LIVE function instead — the
        // failure of picking by name resemblance rather than by usage.
        let dead_task = &EDIT_TASKS[5];
        let killed_the_wrong_one = TALLY_DEAD_RS.replace(
            "/// The item that appears most often, or None when there is nothing to count.\npub fn most_common(items: &[String]) -> Option<String> {\n    let m = counts(items);\n    m.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)\n}\n\n",
            "",
        );
        assert_ne!(killed_the_wrong_one, TALLY_DEAD_RS);
        assert_eq!(
            judge_with(dead_task, "src/tally.rs", &killed_the_wrong_one),
            "broke"
        );
        // And deleting nothing is not solving it either.
        assert_eq!(
            judge_with(dead_task, "src/tally.rs", TALLY_DEAD_RS),
            "not-done"
        );

        // Task 7 (overflow). `saturating_mul` is the tempting one-word fix: it stops the
        // panic and returns u64::MAX, which is a WRONG size rather than a refusal. The
        // task asks for None, so this must not pass.
        let overflow_task = &EDIT_TASKS[6];
        let saturating = PARSER_RS
            .replace("Some(n * 1024)", "Some(n.saturating_mul(1024))")
            .replace(
                "Some(n * 1024 * 1024)",
                "Some(n.saturating_mul(1024).saturating_mul(1024))",
            );
        assert_ne!(saturating, PARSER_RS);
        assert_eq!(
            judge_with(overflow_task, "src/parser.rs", &saturating),
            "not-done",
            "saturating to u64::MAX answers with a wrong size instead of refusing"
        );

        // Task 8 (deduplicate). Deleting the duplicate without giving its caller a
        // replacement takes the crate down — a break, not a missing change.
        let dedup_task = &EDIT_TASKS[7];
        let deleted_not_moved = REPORT_RS
            .split("/// The most common item, padded for a fixed-width column.")
            .next()
            .unwrap()
            .to_string();
        assert!(deleted_not_moved.len() < REPORT_RS.len());
        assert_eq!(
            judge_with(dedup_task, "src/report.rs", &deleted_not_moved),
            "broke",
            "removing the duplicate without replacing it is not deduplication"
        );
        // And leaving both copies in place is simply not done.
        assert_eq!(
            judge_with(dedup_task, "src/report.rs", REPORT_RS),
            "not-done"
        );
    }

    /// And the bug the first task describes must actually BE there: a seed that already
    /// behaves correctly would score every model as a pass without it doing anything.
    #[test]
    fn the_reported_panic_is_really_in_the_seed() {
        let hit = std::panic::catch_unwind(|| {
            // Same body as `format::truncate` in the seed.
            let s = "hello world";
            let max = 2usize;
            if s.chars().count() <= max {
                return s.to_string();
            }
            let kept: String = s.chars().take(max - 3).collect();
            format!("{kept}...")
        });
        assert!(
            hit.is_err(),
            "the seeded truncate() no longer panics for max < 3, so the first task \
             describes a bug that is not there"
        );
    }
}
