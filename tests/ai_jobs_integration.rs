//! Integration tests for the `ai_jobs` binary.
//!
//! These tests spawn the compiled binary (via `CARGO_BIN_EXE_ai_jobs`) and
//! exercise its subcommands end-to-end against `examples/jobs.json`.

#![cfg(feature = "scheduler")]

use std::process::Command;

fn ai_jobs_bin() -> &'static str {
    env!("CARGO_BIN_EXE_ai_jobs")
}

fn manifest_path() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{}/examples/jobs.json", root)
}

#[test]
fn test_validate_example_jobs_json() {
    let output = Command::new(ai_jobs_bin())
        .args(["validate", &manifest_path()])
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(
        output.status.success(),
        "validate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ok"));
    assert!(stdout.contains("4 job(s)"));
}

#[test]
fn test_list_subcommand_exits_zero() {
    let output = Command::new(ai_jobs_bin())
        .args(["list", &manifest_path()])
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("backup_nightly"));
    assert!(stdout.contains("rag_daily_brief"));
    assert!(stdout.contains("shell"));
    assert!(stdout.contains("agent"));
}

#[test]
fn test_dry_run_subcommand_shows_firings() {
    let output = Command::new(ai_jobs_bin())
        .args(["dry-run", &manifest_path(), "--minutes", "180"])
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // health_check fires every 5 minutes → should appear multiple times
    assert!(stdout.contains("health_check") || stdout.contains("Quick health ping"));
    assert!(stdout.contains("firing(s) in window"));
}

#[test]
fn test_help_subcommand() {
    let output = Command::new(ai_jobs_bin())
        .arg("help")
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("SUBCOMMANDS"));
    assert!(stdout.contains("validate"));
    assert!(stdout.contains("dry-run"));
    assert!(stdout.contains("run"));
}

#[test]
fn test_unknown_subcommand_errors() {
    let output = Command::new(ai_jobs_bin())
        .arg("not-a-subcommand")
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown subcommand"));
}

#[test]
fn test_missing_manifest_argument() {
    let output = Command::new(ai_jobs_bin())
        .arg("validate")
        .output()
        .expect("failed to spawn ai_jobs");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing"));
}
