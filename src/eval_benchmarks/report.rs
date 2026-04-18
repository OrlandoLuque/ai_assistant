//! Rendering helpers for [`BenchmarkReport`](super::runner::BenchmarkReport)
//! and [`CalibrationReport`](super::calibration::CalibrationReport).
//!
//! Two output modes:
//!  * `to_text` — single-line human-readable summary + optional per-category
//!    breakdown. Used by `ai_cli benchmark run`.
//!  * `to_json` — stable machine-readable form. Used by `--json` flag, by
//!    the server endpoint, and by the MCP tool.

use super::calibration::CalibrationReport;
use super::runner::BenchmarkReport;

/// Render a benchmark report as a multi-line string suitable for terminal output.
pub fn to_text(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Benchmark: {}\n", report.benchmark));
    out.push_str(&format!(
        "  Samples:      {:>6}\n  Correct:      {:>6}\n  Accuracy:     {:>6.3}\n  Mean score:   {:>6.3}\n  Duration:     {:?}\n",
        report.total, report.correct, report.accuracy, report.mean_score, report.total_duration,
    ));
    if !report.per_category.is_empty() {
        out.push_str("  Per category:\n");
        let mut cats: Vec<_> = report.per_category.iter().collect();
        cats.sort_by(|a, b| a.0.cmp(b.0));
        for (cat, (n, k)) in cats {
            let acc = if *n == 0 { 0.0 } else { *k as f64 / *n as f64 };
            out.push_str(&format!("    {cat:<30} {k:>4}/{n:<4}  {acc:.3}\n"));
        }
    }
    out
}

/// Render a benchmark report as pretty-printed JSON.
pub fn to_json(report: &BenchmarkReport) -> String {
    let per_cat: serde_json::Value = report
        .per_category
        .iter()
        .map(|(k, (n, c))| (k.clone(), serde_json::json!({ "total": n, "correct": c })))
        .collect::<serde_json::Map<_, _>>()
        .into();
    let samples: Vec<serde_json::Value> = report
        .samples
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "score": s.score,
                "correct": s.correct,
                "duration_ms": s.duration.as_millis() as u64,
                "details": s.details,
            })
        })
        .collect();
    let body = serde_json::json!({
        "benchmark": report.benchmark,
        "total": report.total,
        "correct": report.correct,
        "accuracy": report.accuracy,
        "mean_score": report.mean_score,
        "total_duration_ms": report.total_duration.as_millis() as u64,
        "per_category": per_cat,
        "samples": samples,
    });
    serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".into())
}

/// Render a calibration report as a human-readable table.
pub fn calibration_to_text(cal: &CalibrationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Calibration objective: {:?}\n", cal.objective));
    out.push_str("  threshold | accuracy | precision | recall |    f1 | correct/total\n");
    for p in &cal.points {
        out.push_str(&format!(
            "    {:>5.2}   |  {:>5.3}  |   {:>5.3}   | {:>5.3}  | {:>5.3} | {:>4}/{:<4}\n",
            p.threshold, p.accuracy, p.precision, p.recall, p.f1, p.correct, p.total,
        ));
    }
    out.push_str(&format!(
        "  BEST: threshold={:.2}, accuracy={:.3}, f1={:.3}\n",
        cal.best.threshold, cal.best.accuracy, cal.best.f1
    ));
    out
}

/// Render a calibration report as pretty-printed JSON.
pub fn calibration_to_json(cal: &CalibrationReport) -> String {
    let points: Vec<serde_json::Value> = cal
        .points
        .iter()
        .map(|p| {
            serde_json::json!({
                "threshold": p.threshold,
                "accuracy": p.accuracy,
                "precision": p.precision,
                "recall": p.recall,
                "f1": p.f1,
                "correct": p.correct,
                "total": p.total,
            })
        })
        .collect();
    let body = serde_json::json!({
        "objective": format!("{:?}", cal.objective),
        "best": {
            "threshold": cal.best.threshold,
            "accuracy": cal.best.accuracy,
            "precision": cal.best.precision,
            "recall": cal.best.recall,
            "f1": cal.best.f1,
        },
        "points": points,
    });
    serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_benchmarks::calibration::{default_grid, sweep, Objective};
    use crate::eval_benchmarks::runner::{BenchmarkReport, SampleResult};
    use std::collections::HashMap;
    use std::time::Duration;

    fn fake_report() -> BenchmarkReport {
        let samples = vec![SampleResult {
            id: "s1".into(),
            prompt: "p".into(),
            response: "r".into(),
            duration: Duration::from_millis(1),
            score: 0.8,
            correct: true,
            details: HashMap::new(),
        }];
        let mut per_cat = HashMap::new();
        per_cat.insert("catA".into(), (1usize, 1usize));
        BenchmarkReport {
            benchmark: "test".into(),
            total: 1,
            correct: 1,
            accuracy: 1.0,
            mean_score: 0.8,
            total_duration: Duration::from_millis(5),
            per_category: per_cat,
            samples,
        }
    }

    #[test]
    fn text_rendering_contains_fields() {
        let r = fake_report();
        let text = to_text(&r);
        assert!(text.contains("test"));
        assert!(text.contains("Accuracy"));
        assert!(text.contains("catA"));
    }

    #[test]
    fn json_rendering_parses_back() {
        let r = fake_report();
        let json = to_json(&r);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["benchmark"], "test");
        assert_eq!(v["correct"], 1);
    }

    #[test]
    fn calibration_text_contains_best() {
        let r = fake_report();
        let cal = sweep(&r, &default_grid(), Objective::Accuracy);
        let text = calibration_to_text(&cal);
        assert!(text.contains("BEST"));
    }

    #[test]
    fn calibration_json_parses_back() {
        let r = fake_report();
        let cal = sweep(&r, &default_grid(), Objective::F1);
        let json = calibration_to_json(&cal);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("points").is_some());
        assert!(v.get("best").is_some());
    }
}
