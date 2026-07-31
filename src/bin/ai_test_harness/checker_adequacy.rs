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
///
/// Also reused as the corpus for `agentic_test_gen`: the same reference/mutant pair
/// that proves OUR checker competent is what a MODEL-written test suite is scored
/// against.
pub(crate) struct Adequacy {
    /// Must match a `RustTask::name` in `agentic_rust`.
    pub(crate) task: &'static str,
    /// Correct implementation — the checker must accept it.
    pub(crate) reference: &'static str,
    /// (label, implementation) pairs that are plausible but WRONG — the checker
    /// must reject every one of them.
    pub(crate) mutants: &'static [(&'static str, &'static str)],
}

pub(crate) const ADEQUACY: &[Adequacy] = &[
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
        task: "dedup preserving first-appearance order",
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

    results.extend(multi_adequacy_results());

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

// ─── Multi-step tasks: the same oracles, verified the same way ────────────────
//
// The multi-step checkers only ever see the FINAL accumulated crate, so adequacy
// works identically: a complete correct implementation must pass, and a plausible
// wrong one must fail. These oracles are arguably likelier to be weak, since one
// assert suite has to cover everything several steps built up.

struct MultiAdequacy {
    task: &'static str,
    reference: &'static str,
    mutants: &'static [(&'static str, &'static str)],
}

const SHAPES_OK: &str = "pub trait Shape { fn area(&self) -> f64; }\npub struct Rect { pub w: f64, pub h: f64 }\nimpl Rect { pub fn new(w: f64, h: f64) -> Self { Rect { w, h } } }\nimpl Shape for Rect { fn area(&self) -> f64 { self.w * self.h } }\npub struct Circle { pub radius: f64 }\nimpl Circle { pub fn new(radius: f64) -> Self { Circle { radius } } }\nimpl Shape for Circle { fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius } }\npub fn total_area(shapes: &[Box<dyn Shape>]) -> f64 { shapes.iter().map(|s| s.area()).sum() }";

const SHAPES_FIRST_ONLY: &str = "pub trait Shape { fn area(&self) -> f64; }\npub struct Rect { pub w: f64, pub h: f64 }\nimpl Rect { pub fn new(w: f64, h: f64) -> Self { Rect { w, h } } }\nimpl Shape for Rect { fn area(&self) -> f64 { self.w * self.h } }\npub struct Circle { pub radius: f64 }\nimpl Circle { pub fn new(radius: f64) -> Self { Circle { radius } } }\nimpl Shape for Circle { fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius } }\npub fn total_area(shapes: &[Box<dyn Shape>]) -> f64 { shapes.first().map(|s| s.area()).unwrap_or(0.0) }";

const STACK_OK: &str = "pub struct Stack<T> { items: Vec<T> }\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Stack { items: Vec::new() } }\n    pub fn push(&mut self, x: T) { self.items.push(x) }\n    pub fn pop(&mut self) -> Option<T> { self.items.pop() }\n    pub fn len(&self) -> usize { self.items.len() }\n    pub fn peek(&self) -> Option<&T> { self.items.last() }\n}";

const STACK_PEEK_BOTTOM: &str = "pub struct Stack<T> { items: Vec<T> }\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Stack { items: Vec::new() } }\n    pub fn push(&mut self, x: T) { self.items.push(x) }\n    pub fn pop(&mut self) -> Option<T> { self.items.pop() }\n    pub fn len(&self) -> usize { self.items.len() }\n    pub fn peek(&self) -> Option<&T> { self.items.first() }\n}";

const STACK_FIFO: &str = "pub struct Stack<T> { items: Vec<T> }\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Stack { items: Vec::new() } }\n    pub fn push(&mut self, x: T) { self.items.push(x) }\n    pub fn pop(&mut self) -> Option<T> { if self.items.is_empty() { None } else { Some(self.items.remove(0)) } }\n    pub fn len(&self) -> usize { self.items.len() }\n    pub fn peek(&self) -> Option<&T> { self.items.last() }\n}";

const ERRORS_OK: &str = "#[derive(Debug, PartialEq)]\npub enum ParseErr { Missing, NotANumber }\npub fn parse_pair(s: &str) -> Result<(i32, i32), ParseErr> {\n    let (a, b) = s.split_once(',').ok_or(ParseErr::Missing)?;\n    let a = a.trim().parse::<i32>().map_err(|_| ParseErr::NotANumber)?;\n    let b = b.trim().parse::<i32>().map_err(|_| ParseErr::NotANumber)?;\n    Ok((a, b))\n}\npub fn sum_pair(s: &str) -> Result<i32, ParseErr> { parse_pair(s).map(|(a, b)| a + b) }";

const ERRORS_SWAPPED: &str = "#[derive(Debug, PartialEq)]\npub enum ParseErr { Missing, NotANumber }\npub fn parse_pair(s: &str) -> Result<(i32, i32), ParseErr> {\n    let (a, b) = s.split_once(',').ok_or(ParseErr::NotANumber)?;\n    let a = a.trim().parse::<i32>().map_err(|_| ParseErr::Missing)?;\n    let b = b.trim().parse::<i32>().map_err(|_| ParseErr::Missing)?;\n    Ok((a, b))\n}\npub fn sum_pair(s: &str) -> Result<i32, ParseErr> { parse_pair(s).map(|(a, b)| a + b) }";

const BUILDER_OK: &str = "pub struct Server { pub port: u16, pub retries: u32 }\nimpl Server { pub fn new(port: u16, retries: u32) -> Self { Server { port, retries } } }\npub struct ServerBuilder { port: u16, retries: u32 }\nimpl ServerBuilder {\n    pub fn new() -> Self { ServerBuilder { port: 80, retries: 0 } }\n    pub fn port(mut self, p: u16) -> Self { self.port = p; self }\n    pub fn retries(mut self, r: u32) -> Self { self.retries = r; self }\n    pub fn build(self) -> Result<Server, String> { if self.port == 0 { return Err(String::new()); } Ok(Server::new(self.port, self.retries)) }\n}";

const BUILDER_NO_VALIDATION: &str = "pub struct Server { pub port: u16, pub retries: u32 }\nimpl Server { pub fn new(port: u16, retries: u32) -> Self { Server { port, retries } } }\npub struct ServerBuilder { port: u16, retries: u32 }\nimpl ServerBuilder {\n    pub fn new() -> Self { ServerBuilder { port: 80, retries: 0 } }\n    pub fn port(mut self, p: u16) -> Self { self.port = p; self }\n    pub fn retries(mut self, r: u32) -> Self { self.retries = r; self }\n    pub fn build(self) -> Result<Server, String> { Ok(Server::new(self.port, self.retries)) }\n}";

const BUILDER_BAD_DEFAULTS: &str = "pub struct Server { pub port: u16, pub retries: u32 }\nimpl Server { pub fn new(port: u16, retries: u32) -> Self { Server { port, retries } } }\npub struct ServerBuilder { port: u16, retries: u32 }\nimpl ServerBuilder {\n    pub fn new() -> Self { ServerBuilder { port: 8080, retries: 3 } }\n    pub fn port(mut self, p: u16) -> Self { self.port = p; self }\n    pub fn retries(mut self, r: u32) -> Self { self.retries = r; self }\n    pub fn build(self) -> Result<Server, String> { if self.port == 0 { return Err(String::new()); } Ok(Server::new(self.port, self.retries)) }\n}";

const MATRIX_OK: &str = "pub fn transpose(m: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {\n    if m.is_empty() { return vec![]; }\n    (0..m[0].len()).map(|j| m.iter().map(|row| row[j]).collect()).collect()\n}\npub fn identity(n: usize) -> Vec<Vec<i32>> {\n    (0..n).map(|i| (0..n).map(|j| if i == j { 1 } else { 0 }).collect()).collect()\n}\npub fn multiply(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {\n    let n = b[0].len();\n    a.iter().map(|row| (0..n).map(|j| row.iter().enumerate().map(|(k, v)| v * b[k][j]).sum()).collect()).collect()\n}";

const MATRIX_IDENTITY_ONES: &str = "pub fn transpose(m: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {\n    if m.is_empty() { return vec![]; }\n    (0..m[0].len()).map(|j| m.iter().map(|row| row[j]).collect()).collect()\n}\npub fn identity(n: usize) -> Vec<Vec<i32>> {\n    (0..n).map(|_| (0..n).map(|_| 1).collect()).collect()\n}\npub fn multiply(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {\n    let n = b[0].len();\n    a.iter().map(|row| (0..n).map(|j| row.iter().enumerate().map(|(k, v)| v * b[k][j]).sum()).collect()).collect()\n}";

const MATRIX_TRANSPOSE_NOOP: &str = "pub fn transpose(m: &Vec<Vec<i32>>) -> Vec<Vec<i32>> { m.clone() }\npub fn identity(n: usize) -> Vec<Vec<i32>> {\n    (0..n).map(|i| (0..n).map(|j| if i == j { 1 } else { 0 }).collect()).collect()\n}\npub fn multiply(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {\n    let n = b[0].len();\n    a.iter().map(|row| (0..n).map(|j| row.iter().enumerate().map(|(k, v)| v * b[k][j]).sum()).collect()).collect()\n}";

const SCORE_OK: &str = "pub trait Score { fn score(&self) -> i32; }\npub struct Player { pub points: i32 }\nimpl Player { pub fn new(points: i32) -> Self { Player { points } } }\nimpl Score for Player { fn score(&self) -> i32 { self.points } }\npub struct Team { pub members: Vec<Player> }\nimpl Team { pub fn new(members: Vec<Player>) -> Self { Team { members } } }\nimpl Score for Team { fn score(&self) -> i32 { self.members.iter().map(|m| m.score()).sum() } }\npub fn best<T: Score>(items: &[T]) -> Option<i32> { items.iter().map(|i| i.score()).max() }";

const SCORE_BEST_FIRST: &str = "pub trait Score { fn score(&self) -> i32; }\npub struct Player { pub points: i32 }\nimpl Player { pub fn new(points: i32) -> Self { Player { points } } }\nimpl Score for Player { fn score(&self) -> i32 { self.points } }\npub struct Team { pub members: Vec<Player> }\nimpl Team { pub fn new(members: Vec<Player>) -> Self { Team { members } } }\nimpl Score for Team { fn score(&self) -> i32 { self.members.iter().map(|m| m.score()).sum() } }\npub fn best<T: Score>(items: &[T]) -> Option<i32> { items.first().map(|i| i.score()) }";

const SCORE_TEAM_FIRST: &str = "pub trait Score { fn score(&self) -> i32; }\npub struct Player { pub points: i32 }\nimpl Player { pub fn new(points: i32) -> Self { Player { points } } }\nimpl Score for Player { fn score(&self) -> i32 { self.points } }\npub struct Team { pub members: Vec<Player> }\nimpl Team { pub fn new(members: Vec<Player>) -> Self { Team { members } } }\nimpl Score for Team { fn score(&self) -> i32 { self.members.first().map(|m| m.score()).unwrap_or(0) } }\npub fn best<T: Score>(items: &[T]) -> Option<i32> { items.iter().map(|i| i.score()).max() }";

const MULTI_ADEQUACY: &[MultiAdequacy] = &[
    MultiAdequacy {
        task: "shapes: trait then impls then aggregate",
        reference: SHAPES_OK,
        mutants: &[("total_area only counts the first shape", SHAPES_FIRST_ONLY)],
    },
    MultiAdequacy {
        task: "stack: concrete then generic (must rewrite)",
        reference: STACK_OK,
        mutants: &[
            ("peek returns the bottom item", STACK_PEEK_BOTTOM),
            ("pop removes from the front (FIFO)", STACK_FIFO),
        ],
    },
    MultiAdequacy {
        task: "errors: Option then custom error type",
        reference: ERRORS_OK,
        mutants: &[("the two error variants are swapped", ERRORS_SWAPPED)],
    },
    MultiAdequacy {
        task: "builder pattern with validation",
        reference: BUILDER_OK,
        mutants: &[
            // The validation introduced by the LAST step is simply absent.
            ("build never rejects port 0", BUILDER_NO_VALIDATION),
            ("the defaults are wrong", BUILDER_BAD_DEFAULTS),
        ],
    },
    MultiAdequacy {
        task: "matrix ops accumulate",
        reference: MATRIX_OK,
        mutants: &[
            ("identity is all ones", MATRIX_IDENTITY_ONES),
            ("transpose does nothing", MATRIX_TRANSPOSE_NOOP),
        ],
    },
    MultiAdequacy {
        task: "trait then generic over it",
        reference: SCORE_OK,
        mutants: &[
            ("best returns the first score", SCORE_BEST_FIRST),
            ("a team scores like its first member", SCORE_TEAM_FIRST),
        ],
    },
];

/// Adequacy for the multi-step oracles, appended to the same category so one run
/// answers "are ALL our Rust oracles competent?".
fn multi_adequacy_results() -> Vec<TestResult> {
    use crate::agentic_rust::{rust_multi_task_checker, RUST_MULTI_TASK_NAMES};
    let mut results = Vec::new();

    results.push(run_test(
        "adequacy: every multi-step entry names a real task",
        || {
            for a in MULTI_ADEQUACY {
                if !RUST_MULTI_TASK_NAMES.contains(&a.task) {
                    return Err(format!("no multi-step task named {:?}", a.task));
                }
            }
            Ok(())
        },
    ));

    for a in MULTI_ADEQUACY {
        let Some(checker) = rust_multi_task_checker(a.task) else {
            continue;
        };
        results.push(run_test(
            &format!("adequacy[multi]: {} accepts a correct impl", a.task),
            || {
                if verify_snippet_with_checker(a.reference, checker) {
                    Ok(())
                } else {
                    Err("the checker REJECTED a correct implementation".to_string())
                }
            },
        ));
        for (label, mutant) in a.mutants {
            results.push(run_test(
                &format!("adequacy[multi]: {} rejects mutant ({})", a.task, label),
                || {
                    if verify_snippet_with_checker(mutant, checker) {
                        Err(format!(
                            "the checker ACCEPTED a knowingly wrong implementation ({label})"
                        ))
                    } else {
                        Ok(())
                    }
                },
            ));
        }
    }
    results
}
