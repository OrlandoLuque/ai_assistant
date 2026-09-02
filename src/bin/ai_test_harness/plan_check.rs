// ─── Judging a development PLAN without asking a model to judge it ────────────
//
// The category this feeds (N33) measures planning by EXECUTING the plan: a second agent
// follows the steps against a real crate, and the score is whether following it reaches
// the goal. That is the only judgement that cannot be argued with, and it is expensive —
// minutes of model time per plan.
//
// This module is the cheap half, and it runs first. Every check below is mechanical, and
// each one rejects a plan that would waste that execution:
//
//   * it names files the crate does not have          → the plan is about another repo
//   * a step has no way of telling whether it worked  → "done" would be an opinion
//   * a step edits something a later step creates     → the order is impossible
//
// None of these say the plan is GOOD. They say it is executable, which is a different
// property and the one worth establishing before spending the model time. Keeping the two
// apart matters: a vague plan is perfectly executable and reaches nothing, so a category
// that merged them would report one number for "could be followed" and "was worth
// following".

/// One step of a plan the model produced.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlanStep {
    /// What the step does, in the model's words. Not scored here — prose is what this
    /// module deliberately refuses to judge.
    pub(crate) action: String,
    /// Files this step CREATES. Distinguishing these from `edits` is what makes the
    /// checks below possible at all: without it, "names a file the crate does not have"
    /// and "makes a new file" are the same event, and the first can never be reported.
    pub(crate) creates: Vec<String>,
    /// Files this step EDITS. Each must already exist in the crate, or be created by an
    /// EARLIER step.
    pub(crate) edits: Vec<String>,
    /// How the step is checked. Empty means the step cannot be verified, which is the
    /// point of `MissingVerification`.
    pub(crate) verify: String,
}

/// Why a plan is not executable. Separate variants because they call for different
/// feedback: a wrong path is a misunderstanding of the repo, a missing verification is a
/// missing habit, and a bad order is a reasoning error.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PlanFault {
    /// Names a file that is neither in the crate nor created by an earlier step.
    UnknownFile { step: usize, path: String },
    /// No way to tell whether the step worked.
    MissingVerification { step: usize },
    /// Touches a file that only a LATER step creates.
    OutOfOrder {
        step: usize,
        path: String,
        created_by: usize,
    },
    /// No steps at all.
    Empty,
}

impl std::fmt::Display for PlanFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanFault::Empty => write!(f, "the plan has no steps"),
            PlanFault::UnknownFile { step, path } => write!(
                f,
                "step {} names {path:?}, which the crate does not have and no earlier step creates",
                step + 1
            ),
            PlanFault::MissingVerification { step } => write!(
                f,
                "step {} says how to change things but not how to tell whether it worked",
                step + 1
            ),
            PlanFault::OutOfOrder {
                step,
                path,
                created_by,
            } => write!(
                f,
                "step {} edits {path:?}, which step {} is the one that creates",
                step + 1,
                created_by + 1
            ),
        }
    }
}

/// Check a plan against the files a crate actually has.
///
/// `existing` is the crate's real file list. A step may name a file that does not exist
/// yet **if an earlier step creates it** — that is an ordinary plan, not a fault, which is
/// why this walks the steps in order rather than checking each against `existing` alone.
///
/// Returns every fault found, not just the first: a caller reporting "your plan is wrong"
/// once per run teaches less than one listing all three problems at once.
pub(crate) fn check_plan(steps: &[PlanStep], existing: &[String]) -> Vec<PlanFault> {
    if steps.is_empty() {
        return vec![PlanFault::Empty];
    }

    // Which step creates each path, so "created later" is distinguishable from "never".
    let mut created_at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, step) in steps.iter().enumerate() {
        for path in &step.creates {
            created_at.entry(normalise(path)).or_insert(i);
        }
    }

    let known: std::collections::HashSet<String> = existing.iter().map(|s| normalise(s)).collect();
    let mut faults = Vec::new();

    for (i, step) in steps.iter().enumerate() {
        if step.verify.trim().is_empty() {
            faults.push(PlanFault::MissingVerification { step: i });
        }
        for path in &step.edits {
            let norm = normalise(path);
            if known.contains(&norm) {
                continue;
            }
            match created_at.get(&norm) {
                // Created by an earlier step: an ordinary plan.
                Some(&creator) if creator < i => {}
                // Created later than the step that edits it: an impossible order.
                Some(&creator) => faults.push(PlanFault::OutOfOrder {
                    step: i,
                    path: path.clone(),
                    created_by: creator,
                }),
                // Nobody creates it and the crate does not have it.
                None => faults.push(PlanFault::UnknownFile {
                    step: i,
                    path: path.clone(),
                }),
            }
        }
    }
    faults
}

/// Pull a plan out of whatever the model replied with.
///
/// Two rules, both deliberate:
///
/// * **Malformed JSON is not repaired.** Same rule the tool-call parser follows: a model
///   that cannot emit the requested format has failed at the format, and patching its
///   output would inflate the score for a skill it does not have.
/// * **Missing fields become empty, never invented.** A step with no `verify` arrives with
///   an empty one and is caught by [`check_plan`] — if the parser supplied a plausible
///   default here, the check for unverifiable steps could never fire. The parser's job is
///   to read, not to improve.
///
/// Prose around the array is fine: models preface plans with an explanation, and refusing
/// that would measure obedience to formatting rather than planning.
pub(crate) fn parse_plan(reply: &str) -> Result<Vec<PlanStep>, String> {
    let value = first_plan_array(reply)
        .ok_or_else(|| "no JSON array of plan steps in the reply".to_string())?;
    let arr = value.as_array().ok_or("not an array")?;

    let mut steps = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let obj = item
            .as_object()
            .ok_or_else(|| format!("step {} is not an object", i + 1))?;
        let text = |key: &str| -> String {
            obj.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let list = |key: &str| -> Vec<String> {
            obj.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|e| e.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };
        steps.push(PlanStep {
            action: text("action"),
            creates: list("creates"),
            edits: list("edits"),
            verify: text("verify"),
        });
    }
    Ok(steps)
}

/// The first balanced `[..]` that parses AND looks like plan steps. Scanning every `[`
/// rather than the first one matters because an explanation can easily contain brackets
/// before the plan begins.
fn first_plan_array(s: &str) -> Option<serde_json::Value> {
    for (start, c) in s.char_indices() {
        if c != '[' {
            continue;
        }
        let Some(cand) = balanced_array_at(s, start) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&cand) else {
            continue;
        };
        let looks_like_plan = v
            .as_array()
            .map(|a| {
                !a.is_empty()
                    && a.iter()
                        .any(|e| e.get("action").is_some() || e.get("verify").is_some())
            })
            .unwrap_or(false);
        if looks_like_plan {
            return Some(v);
        }
    }
    None
}

/// Balanced `[..]` from `start`, respecting string quoting so a bracket inside a path or
/// a command does not throw off the depth count.
fn balanced_array_at(s: &str, start: usize) -> Option<String> {
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
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Paths compare on `/` regardless of what the model wrote: a plan that says
/// `src\parser.rs` on Windows means the same file as `src/parser.rs`, and rejecting it
/// would be scoring the separator rather than the plan.
fn normalise(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edits(action: &str, files: &[&str], verify: &str) -> PlanStep {
        PlanStep {
            action: action.to_string(),
            creates: Vec::new(),
            edits: files.iter().map(|s| s.to_string()).collect(),
            verify: verify.to_string(),
        }
    }

    fn creates(action: &str, made: &[&str], edited: &[&str], verify: &str) -> PlanStep {
        PlanStep {
            action: action.to_string(),
            creates: made.iter().map(|s| s.to_string()).collect(),
            edits: edited.iter().map(|s| s.to_string()).collect(),
            verify: verify.to_string(),
        }
    }

    fn crate_files() -> Vec<String> {
        ["src/lib.rs", "src/parser.rs", "tests/existing.rs"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn a_plan_that_touches_real_files_and_verifies_each_step_is_clean() {
        let plan = vec![
            edits("add the gb unit", &["src/parser.rs"], "cargo test"),
            edits("cover it", &["tests/existing.rs"], "cargo test"),
        ];
        assert!(check_plan(&plan, &crate_files()).is_empty());
    }

    #[test]
    fn creating_a_file_and_using_it_later_is_an_ordinary_plan() {
        // The false positive worth avoiding: `src/stats.rs` is not in the crate, and
        // that is fine, because step 1 is the one making it.
        let plan = vec![
            creates("write the module", &["src/stats.rs"], &[], "cargo build"),
            edits("declare it", &["src/lib.rs", "src/stats.rs"], "cargo test"),
        ];
        assert!(
            check_plan(&plan, &crate_files()).is_empty(),
            "{:?}",
            check_plan(&plan, &crate_files())
        );
    }

    #[test]
    fn a_file_from_some_other_repo_is_caught() {
        let plan = vec![edits("edit the config", &["src/config.rs"], "cargo test")];
        assert_eq!(
            check_plan(&plan, &crate_files()),
            vec![PlanFault::UnknownFile {
                step: 0,
                path: "src/config.rs".to_string()
            }]
        );
    }

    #[test]
    fn a_step_with_no_verification_is_caught() {
        let plan = vec![edits("tidy up the parser", &["src/parser.rs"], "  ")];
        assert_eq!(
            check_plan(&plan, &crate_files()),
            vec![PlanFault::MissingVerification { step: 0 }]
        );
    }

    #[test]
    fn editing_before_creating_is_caught_as_an_ordering_error() {
        let plan = vec![
            edits("declare the module", &["src/lib.rs"], "cargo build"),
            edits("edit the new module", &["src/stats.rs"], "cargo test"),
            creates("create it", &["src/stats.rs"], &[], "cargo build"),
        ];
        let faults = check_plan(&plan, &crate_files());
        assert_eq!(faults.len(), 1, "got {faults:?}");
        assert!(
            matches!(
                faults[0],
                PlanFault::OutOfOrder {
                    step: 1,
                    created_by: 2,
                    ..
                }
            ),
            "got {faults:?}"
        );
    }

    #[test]
    fn every_fault_is_reported_not_just_the_first() {
        // Reporting one problem per run teaches less than reporting all of them, and if
        // the model is asked to repair the plan it is the difference between one round
        // trip and three.
        let plan = vec![
            edits("edit a stranger", &["src/config.rs"], ""),
            edits("and another", &["src/nope.rs"], "cargo test"),
        ];
        let faults = check_plan(&plan, &crate_files());
        assert_eq!(faults.len(), 3, "got {faults:?}");
    }

    #[test]
    fn backslashes_are_the_same_file_as_forward_slashes() {
        let plan = vec![edits("windows path", &[r"src\parser.rs"], "cargo test")];
        assert!(
            check_plan(&plan, &crate_files()).is_empty(),
            "rejecting a Windows separator would score the separator, not the plan"
        );
    }

    #[test]
    fn a_plan_is_read_out_of_a_reply_that_also_contains_prose() {
        // Models explain before they answer. Refusing that would measure obedience to
        // formatting rather than planning — and the explanation here even contains a
        // bracket before the plan starts, which is why the scan tries every `[`.
        let reply = r#"Sure. I looked at src/parser.rs [the unit table] and here is the plan:
[
  {"action": "add the gb unit", "edits": ["src/parser.rs"], "verify": "cargo test"},
  {"action": "cover it", "creates": ["tests/gb.rs"], "verify": "cargo test"}
]
That should do it."#;
        let plan = parse_plan(reply).expect("a plan");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].edits, vec!["src/parser.rs".to_string()]);
        assert_eq!(plan[1].creates, vec!["tests/gb.rs".to_string()]);
        assert!(plan[0].creates.is_empty());
    }

    #[test]
    fn a_missing_verify_survives_parsing_so_the_check_can_catch_it() {
        // The parser must not improve the plan. If a sensible default were supplied
        // here, `MissingVerification` could never fire and the check would be decorative.
        let reply = r#"[{"action": "tidy the parser", "edits": ["src/parser.rs"]}]"#;
        let plan = parse_plan(reply).expect("a plan");
        assert_eq!(plan[0].verify, "");
        assert_eq!(
            check_plan(&plan, &crate_files()),
            vec![PlanFault::MissingVerification { step: 0 }]
        );
    }

    #[test]
    fn malformed_json_is_a_failure_rather_than_something_to_repair() {
        // Same rule as the tool-call parser: a model that cannot emit the format has
        // failed at the format, and patching it would inflate the score.
        let reply = r#"[{"action": "do it", "edits": ["src/parser.rs",}]"#;
        assert!(parse_plan(reply).is_err());
    }

    #[test]
    fn a_reply_with_no_plan_at_all_is_an_error_not_an_empty_plan() {
        // "No steps" and "no answer" are different failures; collapsing them would hide
        // a model that ignored the request behind one that planned nothing.
        assert!(parse_plan("I would start by reading the parser module.").is_err());
    }

    #[test]
    fn an_empty_plan_is_a_fault_rather_than_a_clean_sheet() {
        // Vacuous success is what a "no faults found" check invites: no steps means
        // nothing to object to.
        assert_eq!(check_plan(&[], &crate_files()), vec![PlanFault::Empty]);
    }
}
