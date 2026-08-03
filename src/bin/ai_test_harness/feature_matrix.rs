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
