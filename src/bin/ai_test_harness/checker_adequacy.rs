use super::*;

use crate::agentic_rust::{rust_task_checker, verify_snippet_with_checker, RUST_TASK_NAMES};

// ─── Are the benchmark's own checkers competent oracles? ──────────────────────
//
// Every Rust task is scored by appending a small assert suite to whatever the model
// wrote and running `cargo test`. That makes the checker an ORACLE, and until now it
// had only ever been validated in one direction: a correct implementation passes.
//
// The other direction was never checked — that a plausible-but-WRONG implementation
// FAILS. It matters far more than it sounds:
//
//   * a checker that accepts wrong code silently inflates every model's score;
//   * and it becomes actively dangerous once automated repair (N9) is allowed to
//     accept an edit because "the tests pass" — a weak oracle would rubber-stamp
//     broken code.
//
// So this category is mutation testing applied to our own benchmark: for each task,
// a reference implementation must PASS and each mutant must FAIL. Unlike the agentic
// categories this measures OUR CODE, not a model, so it belongs in the regression
// battery.

/// A task's oracle under test: one correct implementation and several wrong ones.
struct Adequacy {
    /// Must match a `RustTask::name` in `agentic_rust`.
    task: &'static str,
    /// Correct implementation — the checker must accept it.
    reference: &'static str,
    /// (label, implementation) pairs that are plausible but WRONG — the checker
    /// must reject every one of them.
    mutants: &'static [(&'static str, &'static str)],
}

const ADEQUACY: &[Adequacy] = &[
    Adequacy {
        task: "sum_even (from scratch)",
        reference: "pub fn sum_even(nums: &[i32]) -> i32 { nums.iter().filter(|n| *n % 2 == 0).sum() }",
        mutants: &[
            ("sums everything", "pub fn sum_even(nums: &[i32]) -> i32 { nums.iter().sum() }"),
            ("sums the odd ones", "pub fn sum_even(nums: &[i32]) -> i32 { nums.iter().filter(|n| *n % 2 != 0).sum() }"),
        ],
    },
    Adequacy {
        task: "fix is_even bug",
        reference: "pub fn is_even(n: i32) -> bool { n % 2 == 0 }",
        mutants: &[
            ("left unfixed", "pub fn is_even(n: i32) -> bool { n % 2 == 1 }"),
            ("always true", "pub fn is_even(_n: i32) -> bool { true }"),
        ],
    },
    Adequacy {
        task: "divide returning Option",
        reference: "pub fn divide(a: f64, b: f64) -> Option<f64> { if b == 0.0 { None } else { Some(a / b) } }",
        mutants: &[
            ("no zero guard (returns inf)", "pub fn divide(a: f64, b: f64) -> Option<f64> { Some(a / b) }"),
            ("always None", "pub fn divide(_a: f64, _b: f64) -> Option<f64> { None }"),
        ],
    },
    Adequacy {
        task: "Stack struct with impl",
        reference: "pub struct Stack { items: Vec<i32> }\nimpl Stack {\n    pub fn new() -> Self { Stack { items: Vec::new() } }\n    pub fn push(&mut self, x: i32) { self.items.push(x) }\n    pub fn pop(&mut self) -> Option<i32> { self.items.pop() }\n    pub fn is_empty(&self) -> bool { self.items.is_empty() }\n}",
        mutants: &[
            // FIFO instead of LIFO — the classic stack/queue confusion.
            ("pops from the front (FIFO)", "pub struct Stack { items: Vec<i32> }\nimpl Stack {\n    pub fn new() -> Self { Stack { items: Vec::new() } }\n    pub fn push(&mut self, x: i32) { self.items.push(x) }\n    pub fn pop(&mut self) -> Option<i32> { if self.items.is_empty() { None } else { Some(self.items.remove(0)) } }\n    pub fn is_empty(&self) -> bool { self.items.is_empty() }\n}"),
        ],
    },
    Adequacy {
        task: "count occurrences (HashMap)",
        reference: "pub fn count_occurrences(nums: &[i32]) -> std::collections::HashMap<i32, usize> {\n    let mut m = std::collections::HashMap::new();\n    for n in nums { *m.entry(*n).or_insert(0) += 1; }\n    m\n}",
        mutants: &[
            // Records presence rather than frequency.
            ("counts 1 per distinct value", "pub fn count_occurrences(nums: &[i32]) -> std::collections::HashMap<i32, usize> {\n    let mut m = std::collections::HashMap::new();\n    for n in nums { m.insert(*n, 1); }\n    m\n}"),
        ],
    },
    Adequacy {
        task: "generic largest<T> with trait bounds",
        reference: "pub fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {\n    let mut m = list[0];\n    for &x in list { if x > m { m = x } }\n    m\n}",
        mutants: &[
            ("returns the first element", "pub fn largest<T: PartialOrd + Copy>(list: &[T]) -> T { list[0] }"),
            ("returns the smallest", "pub fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {\n    let mut m = list[0];\n    for &x in list { if x < m { m = x } }\n    m\n}"),
        ],
    },
    Adequacy {
        task: "trait with two impls",
        reference: "pub trait Shape { fn area(&self) -> f64; }\npub struct Circle { pub radius: f64 }\nimpl Circle { pub fn new(radius: f64) -> Self { Circle { radius } } }\nimpl Shape for Circle { fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius } }\npub struct Rect { pub w: f64, pub h: f64 }\nimpl Rect { pub fn new(w: f64, h: f64) -> Self { Rect { w, h } } }\nimpl Shape for Rect { fn area(&self) -> f64 { self.w * self.h } }",
        mutants: &[
            // Circumference instead of area — passes for r=1 only if the checker is sloppy.
            ("circle uses PI*r not PI*r^2", "pub trait Shape { fn area(&self) -> f64; }\npub struct Circle { pub radius: f64 }\nimpl Circle { pub fn new(radius: f64) -> Self { Circle { radius } } }\nimpl Shape for Circle { fn area(&self) -> f64 { std::f64::consts::PI * self.radius } }\npub struct Rect { pub w: f64, pub h: f64 }\nimpl Rect { pub fn new(w: f64, h: f64) -> Self { Rect { w, h } } }\nimpl Shape for Rect { fn area(&self) -> f64 { self.w * self.h } }"),
            ("rect adds instead of multiplying", "pub trait Shape { fn area(&self) -> f64; }\npub struct Circle { pub radius: f64 }\nimpl Circle { pub fn new(radius: f64) -> Self { Circle { radius } } }\nimpl Shape for Circle { fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius } }\npub struct Rect { pub w: f64, pub h: f64 }\nimpl Rect { pub fn new(w: f64, h: f64) -> Self { Rect { w, h } } }\nimpl Shape for Rect { fn area(&self) -> f64 { self.w + self.h } }"),
        ],
    },
    Adequacy {
        task: "explicit lifetimes",
        reference: "pub fn longest<'a>(a: &'a str, b: &'a str) -> &'a str { if a.len() >= b.len() { a } else { b } }",
        mutants: &[
            ("always returns the first", "pub fn longest<'a>(a: &'a str, _b: &'a str) -> &'a str { a }"),
            ("returns the shorter one", "pub fn longest<'a>(a: &'a str, b: &'a str) -> &'a str { if a.len() <= b.len() { a } else { b } }"),
        ],
    },
    Adequacy {
        task: "implement the Iterator trait",
        reference: "pub struct Countdown { n: u32 }\nimpl Countdown { pub fn new(start: u32) -> Self { Countdown { n: start } } }\nimpl Iterator for Countdown {\n    type Item = u32;\n    fn next(&mut self) -> Option<u32> { if self.n == 0 { None } else { let v = self.n; self.n -= 1; Some(v) } }\n}",
        mutants: &[
            ("counts up instead of down", "pub struct Countdown { n: u32, max: u32 }\nimpl Countdown { pub fn new(start: u32) -> Self { Countdown { n: 0, max: start } } }\nimpl Iterator for Countdown {\n    type Item = u32;\n    fn next(&mut self) -> Option<u32> { if self.n >= self.max { None } else { self.n += 1; Some(self.n) } }\n}"),
        ],
    },
    Adequacy {
        task: "borrow checker: dedup in place",
        reference: "pub fn dedup_in_place(v: &mut Vec<i32>) {\n    let mut seen = std::collections::HashSet::new();\n    v.retain(|x| seen.insert(*x));\n}",
        mutants: &[
            // std `dedup` only removes CONSECUTIVE duplicates.
            ("uses Vec::dedup (consecutive only)", "pub fn dedup_in_place(v: &mut Vec<i32>) { v.dedup(); }"),
            // Sorting loses first-appearance order, which the task requires.
            ("sorts first, losing order", "pub fn dedup_in_place(v: &mut Vec<i32>) { v.sort(); v.dedup(); }"),
        ],
    },
    Adequacy {
        task: "enum + match evaluator",
        reference: "#[derive(Clone, Copy)]\npub enum Op { Add, Sub, Mul, Div }\npub fn apply(op: Op, a: f64, b: f64) -> Option<f64> {\n    match op { Op::Add => Some(a + b), Op::Sub => Some(a - b), Op::Mul => Some(a * b), Op::Div => if b == 0.0 { None } else { Some(a / b) } }\n}",
        mutants: &[
            ("no division-by-zero guard", "#[derive(Clone, Copy)]\npub enum Op { Add, Sub, Mul, Div }\npub fn apply(op: Op, a: f64, b: f64) -> Option<f64> {\n    match op { Op::Add => Some(a + b), Op::Sub => Some(a - b), Op::Mul => Some(a * b), Op::Div => Some(a / b) }\n}"),
            ("Sub has its operands swapped", "#[derive(Clone, Copy)]\npub enum Op { Add, Sub, Mul, Div }\npub fn apply(op: Op, a: f64, b: f64) -> Option<f64> {\n    match op { Op::Add => Some(a + b), Op::Sub => Some(b - a), Op::Mul => Some(a * b), Op::Div => if b == 0.0 { None } else { Some(a / b) } }\n}"),
        ],
    },
    Adequacy {
        task: "fizzbuzz (stresses quote escaping)",
        reference: "pub fn fizzbuzz(n: u32) -> String {\n    match (n % 3, n % 5) { (0, 0) => \"FizzBuzz\".to_string(), (0, _) => \"Fizz\".to_string(), (_, 0) => \"Buzz\".to_string(), _ => n.to_string() }\n}",
        mutants: &[
            // The classic: checking 3 before the combined case, so 15 yields "Fizz".
            ("checks 3 before 15", "pub fn fizzbuzz(n: u32) -> String {\n    if n % 3 == 0 { \"Fizz\".to_string() } else if n % 5 == 0 { \"Buzz\".to_string() } else { n.to_string() }\n}"),
        ],
    },
];

pub(crate) fn tests_checker_adequacy() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan("▶ Checker adequacy (are our own oracles competent?)"))
    );
    let mut results = Vec::new();

    let cargo = crate::agentic_rust::cargo_available();
    if !cargo {
        println!("  {} cargo not available — skipping", yellow("SKIP"));
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some("Skipped — cargo not available".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "checker_adequacy".to_string(),
            results,
        };
    }

    // Guard against silent drift: an entry naming a task that no longer exists
    // would quietly test nothing.
    results.push(run_test("adequacy: every entry names a real task", || {
        for a in ADEQUACY {
            if !RUST_TASK_NAMES.contains(&a.task) {
                return Err(format!("no Rust task named {:?}", a.task));
            }
        }
        Ok(())
    }));

    for a in ADEQUACY {
        let Some(checker) = rust_task_checker(a.task) else {
            continue;
        };

        results.push(run_test(
            &format!("adequacy: {} accepts a correct impl", a.task),
            || {
                if verify_snippet_with_checker(a.reference, checker) {
                    Ok(())
                } else {
                    Err(
                        "the checker REJECTED a correct implementation — it is too strict \
                         or simply wrong"
                            .to_string(),
                    )
                }
            },
        ));

        for (label, mutant) in a.mutants {
            results.push(run_test(
                &format!("adequacy: {} rejects mutant ({})", a.task, label),
                || {
                    if verify_snippet_with_checker(mutant, checker) {
                        Err(format!(
                            "the checker ACCEPTED a knowingly wrong implementation ({label}) — \
                             as an oracle it is too weak to score models or to authorise \
                             automated repair"
                        ))
                    } else {
                        Ok(())
                    }
                },
            ));
        }
    }

    let passed = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} checker_adequacy: {}/{} oracle checks",
        bold(&cyan("∑")),
        passed,
        total
    );

    CategoryResult {
        name: "checker_adequacy".to_string(),
        results,
    }
}
