//! Threshold calibration over an existing [`BenchmarkReport`].
//!
//! Given a report whose samples carry numeric scores in `[0.0, 1.0]`, sweep
//! the `correctness_threshold` across a grid and return the point that
//! maximizes the chosen objective (accuracy or F1).
//!
//! The calibration is *post-hoc*: it does not re-run the model, it just
//! re-partitions the existing scores. That keeps calibration cheap — one
//! model run, many threshold experiments.

use super::runner::{BenchmarkReport, SampleResult};

/// Objective to maximize while sweeping.
#[derive(Debug, Clone, Copy)]
pub enum Objective {
    /// Fraction of samples classified correctly.
    Accuracy,
    /// Harmonic mean of precision and recall over the "correct" class.
    F1,
}

/// A single (threshold, metric) point on the sweep curve.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// Threshold applied to `SampleResult::score`.
    pub threshold: f64,
    /// Accuracy at this threshold.
    pub accuracy: f64,
    /// Precision at this threshold (for F1 objective).
    pub precision: f64,
    /// Recall at this threshold (for F1 objective).
    pub recall: f64,
    /// F1 at this threshold.
    pub f1: f64,
    /// Samples counted as correct.
    pub correct: usize,
    /// Total samples evaluated.
    pub total: usize,
}

/// Result of a calibration sweep.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Objective that was maximized.
    pub objective: Objective,
    /// All points evaluated, in ascending threshold order.
    pub points: Vec<CalibrationPoint>,
    /// Point with the best objective value.
    pub best: CalibrationPoint,
}

/// Generate a default grid: 0.0..=1.0 in 0.05 steps (21 points).
pub fn default_grid() -> Vec<f64> {
    (0..=20).map(|i| i as f64 / 20.0).collect()
}

/// Sweep `thresholds` against the report's samples, picking the one that
/// maximizes `objective`. Requires that each `SampleResult` carries a
/// meaningful `score`; error samples (score=0.0) count as false.
///
/// For F1, we treat each sample's *ground-truth correctness proxy* as
/// "score > median threshold on a noise-free run"; since we don't have that,
/// we simply define positive = reference correctness of the *best observed*
/// threshold. A more rigorous ground-truth would require per-sample labels,
/// which the higher-level loader ground_truths provide — but projecting
/// those back into a scalar binary is left to the runner; here we use the
/// `correct` flag produced at the report's *current* threshold as the gold
/// when `objective` is F1.
pub fn sweep(
    report: &BenchmarkReport,
    thresholds: &[f64],
    objective: Objective,
) -> CalibrationReport {
    assert!(
        !thresholds.is_empty(),
        "calibration::sweep requires at least one threshold"
    );

    let mut points = Vec::with_capacity(thresholds.len());
    for &t in thresholds {
        points.push(point_at(&report.samples, t));
    }
    let best = points
        .iter()
        .max_by(|a, b| {
            let va = match objective {
                Objective::Accuracy => a.accuracy,
                Objective::F1 => a.f1,
            };
            let vb = match objective {
                Objective::Accuracy => b.accuracy,
                Objective::F1 => b.f1,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .expect("non-empty point list");

    CalibrationReport {
        objective,
        points,
        best,
    }
}

fn point_at(samples: &[SampleResult], threshold: f64) -> CalibrationPoint {
    // Gold label = was the sample marked correct at the run's original
    // threshold. This is a pragmatic proxy that keeps the calibration math
    // well-defined without needing per-sample ground truth at this layer.
    let total = samples.len();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    for s in samples {
        let pred = s.score >= threshold;
        let gold = s.correct;
        match (pred, gold) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let correct = tp + tn;
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    CalibrationPoint {
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        correct,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_benchmarks::runner::{BenchmarkReport, SampleResult};
    use std::collections::HashMap;
    use std::time::Duration;

    fn mk_sample(score: f64, correct: bool) -> SampleResult {
        SampleResult {
            id: "x".into(),
            prompt: "p".into(),
            response: "r".into(),
            duration: Duration::from_millis(1),
            score,
            correct,
            details: HashMap::new(),
        }
    }

    fn mk_report(samples: Vec<SampleResult>) -> BenchmarkReport {
        let total = samples.len();
        let correct = samples.iter().filter(|s| s.correct).count();
        let accuracy = if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        };
        BenchmarkReport {
            benchmark: "x".into(),
            total,
            correct,
            accuracy,
            mean_score: 0.0,
            total_duration: Duration::from_millis(1),
            per_category: HashMap::new(),
            samples,
        }
    }

    #[test]
    fn default_grid_spans_0_to_1() {
        let g = default_grid();
        assert_eq!(g.len(), 21);
        assert!((g[0]).abs() < 1e-9);
        assert!((g[20] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sweep_picks_optimal_threshold() {
        let samples = vec![
            mk_sample(0.9, true),
            mk_sample(0.8, true),
            mk_sample(0.4, false),
            mk_sample(0.3, false),
        ];
        let report = mk_report(samples);
        let cal = sweep(&report, &default_grid(), Objective::Accuracy);
        // Optimal threshold should lie between 0.4 and 0.8 and yield 1.0.
        assert!((cal.best.accuracy - 1.0).abs() < 1e-9);
        assert!(cal.best.threshold >= 0.5 && cal.best.threshold <= 0.8);
    }

    #[test]
    fn f1_sweep_returns_nontrivial_f1() {
        let samples = vec![
            mk_sample(0.9, true),
            mk_sample(0.6, true),
            mk_sample(0.2, false),
        ];
        let report = mk_report(samples);
        let cal = sweep(&report, &default_grid(), Objective::F1);
        assert!(cal.best.f1 > 0.0 && cal.best.f1 <= 1.0);
        assert!(cal.best.precision > 0.0);
        assert!(cal.best.recall > 0.0);
    }

    #[test]
    fn empty_samples_handled() {
        let report = mk_report(vec![]);
        let cal = sweep(&report, &default_grid(), Objective::Accuracy);
        assert_eq!(cal.best.total, 0);
        assert_eq!(cal.best.accuracy, 0.0);
    }
}
