use super::*;

use std::collections::BTreeSet;
use std::process::Command;

// ─── Does every feature combination actually build? ───────────────────────────
//
// The crate has ninety-odd feature flags. CI has had a "feature matrix" job for a
// long time, and it was measuring the wrong thing: it ran
//
//     cargo check --features "<flag>"
//
// without `--no-default-features`. Since `default = ["full"]`, every entry compiled
// **full + flag** — thirty jobs that were near-duplicates of the full build. Nothing
// was ever compiled BELOW `full`, which is how the reduced build rotted until it did
// not compile at all (V267).
//
// This category compiles `MIN + flag` for each flag, where MIN is the smallest set
// that builds. When a combination fails it reports the first real error, because
// "some combination is broken" is not actionable and "`rag` alone is missing
// rusqlite at src/x.rs:12" is.
//
// The flag list is READ FROM Cargo.toml rather than hardcoded, so a flag added
// tomorrow is covered without anyone remembering to add it here.
//
// Slow by nature — one cargo invocation per flag. Excluded from `--all`; run it
// deliberately with `--category=feature_matrix`, optionally narrowed with
// `--filter=<substring>`.
//
// IT REPORTS WHAT *THIS* PLATFORM BUILDS. Development here is Windows and CI is
// ubuntu, so the two can legitimately disagree — a flag whose C dependency
// generates different bindings per target will differ, and a `#[cfg(unix)]` path
// is never compiled here at all (which is exactly how CI stayed red for days in
// V268). A failure seen locally is a lead, not a verdict, until CI agrees.

/// The smallest set that builds. Must stay in step with `FEATURES_MIN` in
/// `.github/workflows/ci.yml`; `min_set_matches_ci` below fails if they drift.
const MIN: &str = "tools,security,advanced-streaming,rag,adapters,analytics,embeddings,documents";

/// Flags that cannot be checked this way, with the reason. Kept explicit — a
/// silent skip is how coverage quietly shrinks.
const SKIP: &[(&str, &str)] = &[
    ("default", "alias for `full`, not a real combination"),
    (
        "full",
        "the always-tested baseline, covered by every other CI job",
    ),
];

/// Features deliberately absent from CI's matrix, each with the reason.
///
/// The list exists so that "not in CI" is a decision someone wrote down rather
/// than an accident nobody noticed. `whisper-local` is the case that prompted
/// it: a declared feature that **no CI job has ever compiled**, so neither its
/// breakage nor its repair would have shown up anywhere.
#[cfg(test)]
const NOT_IN_CI: &[(&str, &str)] = &[
    ("default", "alias for `full`"),
    ("full", "the baseline every other job builds on"),
    (
        "whisper-local",
        "needs libclang on the runner, and currently fails on Windows/MSVC: \
         whisper-rs-sys' bindgen emits glibc types (_IO_FILE, _G_fpos_t) that \
         overflow a layout assert. See N27; workaround is WHISPER_DONT_GENERATE_BINDINGS",
    ),
    (
        "local-inference-llama-cpp",
        "needs libclang on the runner, same bindgen prerequisite",
    ),
    ("gui", "brings windowing deps; not worth runner time"),
    ("gui-pro", "as `gui`"),
    ("egui-widgets", "as `gui`"),
    ("audio-io", "needs host audio devices"),
    ("video-io", "needs host video devices"),
    (
        "webrtc",
        "heavy native stack, covered by `voice-agent` in practice",
    ),
    (
        "vector-lancedb",
        "heavy native dep; the other vector backends cover the trait",
    ),
    ("aws-bedrock", "needs AWS credentials"),
    ("server-tls", "needs PEM material at build time"),
];

/// The members of `full`, which CI builds in its check / clippy / test jobs via
/// `FEATURES_STD`. A feature inside `full` is compiled by CI even when it has no
/// matrix entry of its own, and counting it as uncovered would be false.
#[cfg(test)]
fn full_members(manifest: &str) -> Vec<String> {
    let Some(start) = manifest.find(
        "
full = [",
    ) else {
        return Vec::new();
    };
    let Some(end) = manifest[start..].find(
        "
]",
    ) else {
        return Vec::new();
    };
    manifest[start..start + end]
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with('#') {
                return None;
            }
            let t = t.trim_start_matches("full = [").trim();
            let t = t.trim_end_matches(',').trim();
            let t = t.trim_matches('"');
            (!t.is_empty()
                && t.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
            .then(|| t.to_string())
        })
        .collect()
}

/// Declared features that **no CI job compiles today**, recorded so the number
/// cannot grow unnoticed.
///
/// This is a ratchet, not an excuse list: every name here is a real gap awaiting
/// a decision (put it in CI, or drop the feature). What the test enforces is
/// that a *new* feature cannot join them silently — which is exactly how
/// `whisper-local` came to be declared, shipped and never built by anything.
///
/// Shrinking this list is the goal; growing it must be deliberate.
///
/// **Measured, not assumed (2026-09-11, full battery, 84/87):** every feature
/// below *compiles* on top of the minimum set. None of them is broken — they are
/// simply absent from CI, so nothing would notice if one broke tomorrow. The only
/// three combinations that fail are the native-toolchain ones already excused in
/// [`NOT_IN_CI`] (`vector-lancedb`, `whisper-local`, `local-inference-llama-cpp`),
/// and they fail here for a *local* reason — no libclang / C++ toolchain on the
/// dev laptop — which is a lead about this machine, not a verdict about the code.
///
/// So the cost of closing this list is CI minutes, not repair work. That is a
/// budget decision for the maintainer, which is why the list records the fact
/// rather than acting on it.
#[cfg(test)]
const UNCOVERED_BACKLOG: &[&str] = &[
    // Platform-specific: the Linux runner cannot build these at all.
    "hardware-rocm",
    "hardware-metal",
    // Implicit dependency features, enabled transitively by `documents` / `rag`,
    // which CI does build — listed because nothing exercises them on their own.
    "aes-gcm",
    "zip",
    "pdf-extract",
    // Genuinely uncovered. Each needs triage.
    "wasm",
    "vector-pgvector",
    "code-sandbox",
    "cloud-connectors",
    "prompt-fragments",
    "stall-detection",
    "stall-detection-llm",
    "sub-agents",
    "acp",
    // V330 removed `local-inference` from here by putting it in CI: it declares
    // no dependencies of its own (the umbrella is the `Backend` trait plus the
    // stub), so the CI minutes this list was weighing were almost none, and
    // leaving it out meant the V330 `LlmProvider` bridge compiled on no job.
    "local-inference-candle",
    "embeddings-local",
    "redis-backend",
    "skill-forge",
    "prompt-synthesis",
    "feedback-loop",
    "prompt-breeder",
    // GUI variants: windowing deps, same reason as `gui` itself.
    "gui-logs",
    "gui-recipes",
    "gui-acp",
    "gui-corrections",
    "gui-local-inference",
];

/// Every feature name CI compiles: the matrix entries plus the `FEATURES_*`
/// environment sets used by the check, clippy and test jobs.
#[cfg(test)]
fn ci_matrix_features(ci_yml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_matrix = false;
    for line in ci_yml.lines() {
        let t = line.trim();
        // The env blocks are feature sets CI really builds too: the check,
        // clippy and test jobs run against FEATURES_STD and FEATURES_NETWORK,
        // so a feature named there is compiled even with no matrix entry.
        for key in ["FEATURES_STD:", "FEATURES_NETWORK:", "FEATURES_MIN:"] {
            if let Some(rest) = t.strip_prefix(key) {
                for name in rest.trim().trim_matches('"').split(',') {
                    let name = name.trim();
                    if !name.is_empty() {
                        out.push(name.to_string());
                    }
                }
            }
        }
        if t.starts_with("features:") {
            in_matrix = true;
            continue;
        }
        if !in_matrix {
            continue;
        }
        // The list is `- "a"` / `- "a,b"`; anything else ends it.
        let Some(rest) = t.strip_prefix("- ") else {
            if !t.is_empty() && !t.starts_with('#') {
                in_matrix = false;
            }
            continue;
        };
        for name in rest.trim().trim_matches('"').split(',') {
            let name = name.trim();
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Every feature name declared in `[features]`.
///
/// Parsed from the manifest so the matrix cannot silently fall behind the crate.
fn declared_features(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_features = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if !in_features || t.starts_with('#') || t.is_empty() {
            continue;
        }
        // `name = [...]` — take the left of the first `=`.
        let Some((name, _)) = t.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            out.push(name.to_string());
        }
    }
    out
}

/// First line that looks like a compiler error, plus where it happened.
///
/// The whole point of the category: report WHY, not just that it broke.
fn first_error(stderr: &str) -> String {
    let mut msg = None;
    for (i, line) in stderr.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("error") && msg.is_none() {
            msg = Some((i, t.to_string()));
        }
        if let Some((mi, ref m)) = msg {
            // The `--> file:line` note usually follows within a couple of lines.
            if i > mi && t.starts_with("-->") {
                return format!("{m}  at {}", t.trim_start_matches("-->").trim());
            }
            if i > mi + 6 {
                return m.clone();
            }
        }
    }
    msg.map(|(_, m)| m)
        .unwrap_or_else(|| "failed with no recognisable error line".to_string())
}

fn cargo_cmd() -> Option<&'static str> {
    let ok = Command::new("cargo")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some("cargo")
}

/// `cargo check --no-default-features --features "<features>"`.
fn check(cargo: &str, features: &str) -> Result<(), String> {
    let out = Command::new(cargo)
        .args([
            "check",
            "--quiet",
            "--no-default-features",
            "--features",
            features,
        ])
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    Err(first_error(&String::from_utf8_lossy(out.stderr.as_slice())))
}

pub(crate) fn tests_feature_matrix() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Feature matrix (does each flag build on top of the minimum set?)"
        ))
    );
    let mut results = Vec::new();

    let Some(cargo) = cargo_cmd() else {
        println!("  {} skipping — cargo not available", yellow("SKIP"));
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some("Skipped — cargo not on PATH".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "feature_matrix".to_string(),
            results,
        };
    };

    let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    let declared = declared_features(&manifest);
    let skip: BTreeSet<&str> = SKIP.iter().map(|(n, _)| *n).collect();
    let min_members: BTreeSet<&str> = MIN.split(',').collect();

    // Guard against the manifest parser silently returning nothing — an empty
    // matrix would pass and prove nothing.
    results.push(run_test("feature-matrix: manifest is readable", || {
        if declared.len() < 20 {
            return Err(format!(
                "only {} features parsed from Cargo.toml — the parser or the manifest \
                 layout changed, and an empty matrix would pass while testing nothing",
                declared.len()
            ));
        }
        Ok(())
    }));

    // The baseline itself.
    results.push(run_test("feature-matrix: the minimum set", || {
        check(cargo, MIN)
    }));

    println!(
        "  {} {} flags declared; checking each on top of the minimum set",
        cyan("·"),
        declared.len()
    );

    for feat in &declared {
        if skip.contains(feat.as_str()) || min_members.contains(feat.as_str()) {
            continue;
        }
        let name = format!("feature-matrix: min + {feat}");
        if !crate::should_run(&name) {
            continue;
        }
        let combo = format!("{MIN},{feat}");
        results.push(run_test(&name, || check(cargo, &combo)));
    }

    let failed = results.iter().filter(|r| !r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} feature_matrix: {}/{} combinations build",
        bold(&cyan("∑")),
        total - failed,
        total
    );
    for (name, why) in SKIP {
        println!("  {} skipped `{}` — {}", cyan("·"), name, why);
    }

    CategoryResult {
        name: "feature_matrix".to_string(),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feature_names_from_a_manifest() {
        let manifest = "\
[package]
name = \"x\"

[features]
# a comment
default = [\"full\"]
security = [\"dep:sha2\"]
rag = [\"rusqlite\", \"aes-gcm\"]
multi-agent = []

[dependencies]
serde = \"1\"
";
        let f = declared_features(manifest);
        assert_eq!(f, vec!["default", "security", "rag", "multi-agent"]);
        // Crucially, it must stop at the next section: `serde` is a dependency,
        // not a feature, and checking `--features serde` would fail confusingly.
        assert!(!f.contains(&"serde".to_string()));
    }

    #[test]
    fn min_set_matches_ci() {
        // Drift here would mean the local battery and CI test different things,
        // and the one that matters would be whichever nobody reads.
        let ci = std::fs::read_to_string(".github/workflows/ci.yml").unwrap_or_default();
        if ci.is_empty() {
            return; // not running from the repo root; nothing to compare
        }
        let line = ci
            .lines()
            .find(|l| l.trim_start().starts_with("FEATURES_MIN:"))
            .unwrap_or("");
        assert!(
            line.contains(MIN),
            "MIN here and FEATURES_MIN in ci.yml have drifted apart.\n  here: {MIN}\n  ci:   {line}"
        );
    }

    /// Every declared feature is either built by CI or excluded on purpose.
    ///
    /// The gap this closes: `whisper-local` is declared in `Cargo.toml`, is not
    /// in CI's matrix, and is excluded from release builds — so **no job has
    /// ever compiled it**, and nothing would have said so. The harness's own
    /// matrix derives from the manifest and does cover it, but that category is
    /// excluded from `--all` because it shells out to cargo per feature.
    ///
    /// This test does not force anything into CI. It forces the *decision* to be
    /// written down, so a feature added tomorrow cannot fall out of CI silently.
    #[test]
    fn every_declared_feature_is_built_by_ci_or_excluded_on_purpose() {
        let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
        let ci = std::fs::read_to_string(".github/workflows/ci.yml").unwrap_or_default();
        if manifest.is_empty() || ci.is_empty() {
            return; // not running from the repo root
        }

        let built: BTreeSet<String> = ci_matrix_features(&ci).into_iter().collect();
        assert!(
            built.len() > 20,
            "only {} features parsed out of ci.yml — the workflow layout changed and this \
             test would pass while checking nothing",
            built.len()
        );

        let min_members: BTreeSet<&str> = MIN.split(',').collect();
        let excused: BTreeSet<&str> = NOT_IN_CI.iter().map(|(n, _)| *n).collect();
        // Membership of `full` counts as coverage: CI's check, clippy and test
        // jobs all build FEATURES_STD, which starts from `full`.
        let in_full: BTreeSet<String> = full_members(&manifest).into_iter().collect();
        assert!(
            in_full.len() > 10,
            "only {} members parsed out of `full` — the manifest layout changed and this              test would report covered features as orphans",
            in_full.len()
        );

        let orphans: Vec<String> = declared_features(&manifest)
            .into_iter()
            .filter(|f| {
                !built.contains(f)
                    && !in_full.contains(f)
                    && !min_members.contains(f.as_str())
                    && !excused.contains(f.as_str())
                    && !UNCOVERED_BACKLOG.contains(&f.as_str())
            })
            .collect();

        assert!(
            orphans.is_empty(),
            "a NEW feature is declared that nothing in CI compiles: {orphans:?}\n\
             Add them to the matrix in ci.yml, or to NOT_IN_CI with the reason. \
             A feature no job builds is where declared debt accumulates unseen."
        );

        // The same check in the other direction, added V330. The assertion above
        // only catches features nothing covers; it says nothing about a backlog
        // entry that quietly became covered. So the list could keep calling a
        // feature "genuinely uncovered" forever after someone fixed it, and the
        // next reader would budget CI minutes for work already done. A list that
        // is only checked in the direction that grows it is how a ratchet rusts.
        let now_covered: Vec<&str> = UNCOVERED_BACKLOG
            .iter()
            .copied()
            .filter(|f| built.contains(*f) || in_full.contains(*f) || min_members.contains(f))
            .collect();

        assert!(
            now_covered.is_empty(),
            "UNCOVERED_BACKLOG still lists features CI now compiles: {now_covered:?}\n\
             Delete them from the list. Leaving them makes it overstate the gap, \
             which is the same defect as understating it."
        );
    }

    #[test]
    fn the_ci_exclusion_list_does_not_name_features_that_stopped_existing() {
        // The other direction: an excuse for a feature that was renamed or
        // deleted is a stale note that hides the fact nothing is excluded.
        let manifest = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
        if manifest.is_empty() {
            return;
        }
        let declared: BTreeSet<String> = declared_features(&manifest).into_iter().collect();
        for (name, reason) in NOT_IN_CI {
            assert!(
                declared.contains(*name),
                "NOT_IN_CI excuses {name:?} ({reason}), but no such feature is declared"
            );
        }
    }

    #[test]
    fn the_ci_matrix_parser_reads_the_real_list() {
        let yml = "\
jobs:\n  matrix:\n    strategy:\n      matrix:\n        features:\n          - \"security\"\n          - \"autonomous,scheduler\"\n          - \"rag\"\n    steps:\n      - uses: actions/checkout@v5\n";
        let got = ci_matrix_features(yml);
        assert_eq!(got, vec!["security", "autonomous", "scheduler", "rag"]);
    }

    #[test]
    fn extracts_the_error_and_its_location() {
        let stderr = "\
   Compiling ai_assistant v0.2.0
error[E0432]: unresolved import `crate::guardrail_pipeline`
  --> src/prelude.rs:33:16
   |
33 | pub use crate::guardrail_pipeline::{Guard};
";
        let e = first_error(stderr);
        assert!(e.contains("E0432"), "{e}");
        assert!(e.contains("src/prelude.rs:33"), "{e}");
    }

    #[test]
    fn reports_something_useful_when_there_is_no_error_line() {
        assert!(first_error("linker died\n").contains("no recognisable error"));
    }
}
