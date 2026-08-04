use super::*;

// ─── Scoring a live model as a RATE, shared by the benchmark categories ───────
//
// A single live-model run is one sample of a stochastic process, and its
// knife-edge verdict flips between runs even with a pinned seed (~±1 task,
// measured). So a category that scores pass/fail once is reporting a coin toss
// with three significant figures.
//
// `agentic_test_gen` grew this machinery in V263; this module is that code
// lifted out so every category can use it rather than each growing its own
// slightly-different copy — which is how two categories end up disagreeing about
// what "flaky" means.
//
// The two design decisions that matter are documented at their point of use:
// repeats are INTERLEAVED (see `run_interleaved`), and runs the backend never
// completed leave the DENOMINATOR (see `RepeatedOutcome::rate`).

/// Marks a run the backend never completed. Such a run carries no evidence about
/// the model, so it is dropped from the denominator rather than counted as a
/// failed attempt — otherwise a crashing runner quietly deflates the score of
/// whichever model happened to be loaded.
pub(crate) const BACKEND_CRASH_PREFIX: &str = "BACKEND CRASH";

/// What repeating one task produced.
pub(crate) struct RepeatedOutcome {
    pub(crate) name: String,
    pub(crate) passes: usize,
    /// Runs that actually produced evidence — excludes backend crashes.
    pub(crate) attempts: usize,
    pub(crate) crashes: usize,
    pub(crate) elapsed_ms: f64,
    /// The last failure seen, for reporting WHY when the task never passed.
    pub(crate) last_failure: Option<String>,
}

impl RepeatedOutcome {
    /// Pass rate over the runs that carried evidence.
    ///
    /// With every run lost to the backend there is nothing to score, so the task
    /// is neither credited nor blamed.
    pub(crate) fn rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.passes as f64 / self.attempts as f64
        }
    }

    pub(crate) fn passed(&self) -> bool {
        self.attempts > 0 && self.rate() >= 0.5
    }

    /// Strictly between "never" and "always" — the model being inconsistent,
    /// which is a finding in itself rather than something to average away.
    pub(crate) fn is_flaky(&self) -> bool {
        self.passes > 0 && self.passes < self.attempts
    }
}

/// Run every task `repeats` times, **interleaved**: pass 1 of every task, then
/// pass 2, and so on.
///
/// Consecutive repeats of one task are correlated samples — they hit the backend
/// with near-identical KV-cache state and so nearly always agree, hiding exactly
/// the variance being measured. Laid out back to back, a 12-task category
/// reported ZERO flaky tasks while two separate invocations disagreed on several;
/// interleaved, the same 36 runs surfaced four to six. Same cost, real answer.
pub(crate) fn run_interleaved<T>(
    items: &[T],
    name_of: impl Fn(&T) -> String,
    repeats: usize,
    mut run_one: impl FnMut(&T) -> Result<(), String>,
) -> Vec<RepeatedOutcome> {
    let mut out: Vec<RepeatedOutcome> = items
        .iter()
        .map(|it| RepeatedOutcome {
            name: name_of(it),
            passes: 0,
            attempts: 0,
            crashes: 0,
            elapsed_ms: 0.0,
            last_failure: None,
        })
        .collect();

    for pass in 0..repeats {
        for (i, item) in items.iter().enumerate() {
            let t0 = std::time::Instant::now();
            // Caught for the same reason `run_test` catches it: one panicking task must
            // not take down a sweep that has already spent an hour of GPU time. These
            // categories used to go through `run_test` and inherited this; the rate
            // loop has to provide it itself. (Needs a panic=unwind profile.)
            let outcome =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(item))) {
                    Ok(r) => r,
                    Err(panic) => {
                        let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        Err(format!("PANIC: {msg}"))
                    }
                };
            out[i].elapsed_ms += t0.elapsed().as_secs_f64() * 1000.0;
            match outcome {
                Ok(()) => {
                    out[i].attempts += 1;
                    out[i].passes += 1;
                }
                Err(e) if e.starts_with(BACKEND_CRASH_PREFIX) => {
                    out[i].crashes += 1;
                    out[i].last_failure = Some(e);
                }
                Err(e) => {
                    out[i].attempts += 1;
                    out[i].last_failure = Some(e);
                }
            }
        }
        if repeats > 1 {
            println!("  {} pass {}/{} done", cyan("·"), pass + 1, repeats);
        }
    }
    out
}

/// Turn outcomes into `TestResult`s and print the per-task lines.
///
/// Reported here rather than through `run_test`/`run_test_scored`, because the
/// work happened in the interleaved loop: a closure handed to those helpers would
/// time an instant lookup and print a meaningless 0.0 ms.
pub(crate) fn to_results(outcomes: &[RepeatedOutcome], repeats: usize) -> Vec<TestResult> {
    let mut results = Vec::with_capacity(outcomes.len());
    for o in outcomes {
        let message = match &o.last_failure {
            // Never solved: report WHY, not just the zero.
            Some(e) if o.passes == 0 => Some(e.clone()),
            _ => None,
        };
        let slow = o.elapsed_ms > crate::get_timeout_ms() * repeats as f64;
        if !crate::json_mode() {
            let status = if o.passed() {
                green("PASS")
            } else {
                red("FAIL")
            };
            let slow_tag = if slow { yellow(" SLOW") } else { String::new() };
            let lost = if o.crashes > 0 {
                format!(" [{} run(s) lost to backend crashes]", o.crashes)
            } else {
                String::new()
            };
            match &message {
                Some(m) => println!(
                    "  {} {} {}/{} - {} ({:.1}ms){}{}",
                    status, o.name, o.passes, o.attempts, m, o.elapsed_ms, lost, slow_tag
                ),
                None => println!(
                    "  {} {} {}/{} runs (score={:.2}) ({:.1}ms){}{}",
                    status,
                    o.name,
                    o.passes,
                    o.attempts,
                    o.rate(),
                    o.elapsed_ms,
                    lost,
                    slow_tag
                ),
            }
        }
        results.push(TestResult {
            name: o.name.clone(),
            passed: o.passed(),
            message,
            duration_ms: o.elapsed_ms,
            score: Some(o.rate()),
            details: Vec::new(),
            skipped: false,
            slow,
        });
    }
    results
}

/// The summary block: total, flaky list, distribution, retry projection, failure
/// modes. `unit` names what is being counted, e.g. "suites" or "tasks"; `modes`
/// maps a substring of the failure message to the label to report it under.
pub(crate) fn print_summary(
    category: &str,
    unit: &str,
    modes: &[(&str, &str)],
    outcomes: &[RepeatedOutcome],
    repeats: usize,
) {
    let rates: Vec<f64> = outcomes.iter().map(|o| o.rate()).collect();
    if rates.is_empty() {
        return;
    }
    let earned: f64 = rates.iter().sum();
    println!(
        "  {} {category}: {:.2}/{} {unit} — pass rate over {repeats} run{} each (backend={})",
        bold(&cyan("∑")),
        earned,
        rates.len(),
        if repeats == 1 { "" } else { "s" },
        crate::bench_util::bench_label()
    );

    let flaky: Vec<&RepeatedOutcome> = outcomes.iter().filter(|o| o.is_flaky()).collect();
    if !flaky.is_empty() {
        let list = flaky
            .iter()
            .map(|o| format!("{} ({}/{})", o.name, o.passes, o.attempts))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {} inconsistent across runs: {}", yellow("FLAKY"), list);
    }

    // How much of the sweep the backend ate, stated once and in RUNS. Without this
    // the header reads "pass rate over 3 runs each" while some tasks were scored on
    // one — a 30B measured at 33 % CPU offload lost 4 of 18 runs to runner aborts,
    // and the only trace was a tag at the end of three per-task lines.
    let lost: usize = outcomes.iter().map(|o| o.crashes).sum();
    if lost > 0 {
        let affected = outcomes.iter().filter(|o| o.crashes > 0).count();
        println!(
            "  {} {} of {} runs never completed and left the denominator, across {} of {} {unit}",
            yellow("LOST"),
            lost,
            outcomes.len() * repeats,
            affected,
            outcomes.len()
        );
    }

    print_distribution(&rates);
    print_blind_retry_projection(&rates);
    print_failure_modes(modes, outcomes);
}

/// Where the model sits, not just what it totalled.
///
/// A single number hides the shape: 6/12 can mean six tasks solved reliably and
/// six never, or twelve solved half the time. Those are different models to work
/// with, and the middle band is where retrying might help.
fn print_distribution(rates: &[f64]) {
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let variance = rates.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let never = rates.iter().filter(|s| **s <= f64::EPSILON).count();
    let always = rates.iter().filter(|s| **s >= 1.0 - f64::EPSILON).count();
    println!(
        "  {} mean rate {:.2} (sd {:.2}) — always {}, sometimes {}, never {}",
        bold(&cyan("∑")),
        mean,
        variance.sqrt(),
        always,
        rates.len() - never - always,
        never
    );
}

/// What plain retrying would buy, before anyone builds a repair loop.
///
/// Attempts are independent, so a task solved with probability `p` succeeds at
/// least once in `k` tries with probability `1-(1-p)^k`. That is the score a
/// strategy of simply *buying more lottery tickets* would reach — and the bar any
/// feedback-driven repair has to clear to have earned its complexity. Matching
/// this number means the feedback added nothing.
///
/// Note what it cannot move: a task at `p = 0` stays at 0 for any `k`. Measured
/// on qwen3-coder:30b, where every task is 1.0 or 0.0, the projection EQUALS the
/// score at every k — i.e. retrying that model buys literally nothing.
fn print_blind_retry_projection(rates: &[f64]) {
    let project = |k: i32| -> f64 { rates.iter().map(|p| 1.0 - (1.0 - p).powi(k)).sum() };
    println!(
        "  {} blind-retry projection: {:.2} at k=2, {:.2} at k=3 (of {}) — the bar a \
         feedback-driven repair must beat",
        bold(&cyan("∑")),
        project(2),
        project(3),
        rates.len()
    );
}

/// Which way the tasks failed, which is more actionable than how many.
///
/// Counts every task that had **at least one** failing run, by its last failure —
/// not only the tasks that failed outright. A task solved 2 times in 3 still
/// failed once, and how it failed is the same evidence.
///
/// The modes are the CALLER's, given as (substring of the failure message, label),
/// because what "wrong" means differs per category — a test-generation suite that
/// rejects valid code and a multi-step task that broke compilation are not the same
/// diagnosis. Two buckets are always present: backend crash (infrastructure, not the
/// model) and other (a message no mode matched — a standing hint that the table has
/// gone stale).
fn print_failure_modes(modes: &[(&str, &str)], outcomes: &[RepeatedOutcome]) {
    let mut counts = vec![0usize; modes.len()];
    let (mut crashed, mut other) = (0usize, 0usize);
    for msg in outcomes.iter().filter_map(|o| o.last_failure.as_deref()) {
        if msg.starts_with(BACKEND_CRASH_PREFIX) {
            crashed += 1;
        } else if let Some(i) = modes.iter().position(|(needle, _)| msg.contains(needle)) {
            counts[i] += 1;
        } else {
            other += 1;
        }
    }
    if counts.iter().sum::<usize>() + crashed + other == 0 {
        return;
    }
    let mut parts: Vec<String> = modes
        .iter()
        .zip(&counts)
        .map(|((_, label), n)| format!("{n} {label}"))
        .collect();
    parts.push(format!("{crashed} backend crash"));
    parts.push(format!("{other} other"));
    // "tasks", said out loud: these count tasks by their LAST failure, so a task that
    // failed three different ways appears once. The run-level count is the LOST line.
    println!(
        "  {} failure modes (tasks, by last failure): {}",
        bold(&cyan("∑")),
        parts.join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(name: &str, passes: usize, attempts: usize, crashes: usize) -> RepeatedOutcome {
        RepeatedOutcome {
            name: name.to_string(),
            passes,
            attempts,
            crashes,
            elapsed_ms: 1.0,
            last_failure: None,
        }
    }

    #[test]
    fn a_crashed_run_leaves_the_denominator() {
        // The whole point: infrastructure failure must not be scored as model
        // incompetence. 2 of 3 runs crashed, the one that ran passed -> 1.0.
        let o = outcome("t", 1, 1, 2);
        assert_eq!(o.rate(), 1.0);
        assert!(o.passed());
        assert!(!o.is_flaky(), "1/1 is consistent, not flaky");
    }

    #[test]
    fn every_run_lost_scores_zero_without_blaming_the_model() {
        let o = outcome("t", 0, 0, 3);
        assert_eq!(o.rate(), 0.0);
        assert!(!o.passed());
        assert!(!o.is_flaky());
    }

    #[test]
    fn flaky_is_strictly_between_never_and_always() {
        assert!(!outcome("t", 0, 3, 0).is_flaky());
        assert!(outcome("t", 1, 3, 0).is_flaky());
        assert!(outcome("t", 2, 3, 0).is_flaky());
        assert!(!outcome("t", 3, 3, 0).is_flaky());
    }

    #[test]
    fn interleaving_runs_every_task_once_before_repeating_any() {
        // The ordering IS the feature: back-to-back repeats are correlated
        // samples and hide the variance being measured.
        let items = vec!["a", "b", "c"];
        let mut order = Vec::new();
        run_interleaved(
            &items,
            |i| i.to_string(),
            3,
            |i| {
                order.push(*i);
                Ok(())
            },
        );
        assert_eq!(
            order,
            vec!["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "each pass must sweep all tasks before the next pass starts"
        );
    }

    #[test]
    fn a_task_scored_on_fewer_runs_than_asked_is_still_reported_as_full_rate() {
        // The trap this pairs with the LOST line to close: 1/1 and 3/3 both print
        // score=1.00, so without the run-level count a sweep in which the backend
        // ate a third of the runs looks exactly like a clean one.
        let one_run = outcome("t", 1, 1, 2);
        let three_runs = outcome("u", 3, 3, 0);
        assert_eq!(one_run.rate(), three_runs.rate());
        assert_eq!(one_run.crashes + three_runs.crashes, 2);
    }

    #[test]
    fn a_panicking_task_fails_that_run_and_the_sweep_continues() {
        // Without this the first panic would abort a sweep that may already have spent
        // an hour of GPU time, losing every result gathered so far.
        let items = vec![0usize, 1];
        let out = run_interleaved(
            &items,
            |i| format!("t{i}"),
            2,
            |i| {
                if *i == 0 {
                    panic!("boom");
                }
                Ok(())
            },
        );
        assert_eq!(out[0].passes, 0);
        assert_eq!(out[0].attempts, 2, "a panic is evidence, not a lost run");
        assert!(out[0].last_failure.as_deref().unwrap().contains("boom"));
        assert_eq!(out[1].passes, 2, "the other task still ran, both passes");
    }

    #[test]
    fn counts_passes_attempts_and_crashes_separately() {
        let items = vec![0usize];
        let mut n = 0;
        let out = run_interleaved(
            &items,
            |_| "t".into(),
            3,
            |_| {
                n += 1;
                match n {
                    1 => Ok(()),
                    2 => Err(format!("{BACKEND_CRASH_PREFIX}: runner died")),
                    _ => Err("wrong answer".to_string()),
                }
            },
        );
        assert_eq!(out[0].passes, 1);
        assert_eq!(out[0].attempts, 2, "the crash must not count as an attempt");
        assert_eq!(out[0].crashes, 1);
        assert_eq!(out[0].rate(), 0.5);
    }
}
