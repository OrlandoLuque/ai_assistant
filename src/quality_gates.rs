//! Quality gates for LLM output validation
//!
//! Provides configurable quality gates that check LLM outputs against
//! minimum thresholds for faithfulness, confidence, grounding ratio,
//! consistency score, and citation coverage.
//!
//! Quality gates can be configured to fail, warn, or log when thresholds
//! are not met.

// =============================================================================
// Core Types
// =============================================================================

/// Quality metric to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QualityMetric {
    /// Faithfulness score (0.0 - 1.0)
    Faithfulness,
    /// Overall confidence score (0.0 - 1.0)
    Confidence,
    /// Ratio of grounded claims to total claims (0.0 - 1.0)
    GroundingRatio,
    /// Self-consistency score (0.0 - 1.0)
    ConsistencyScore,
    /// Ratio of claims with citations (0.0 - 1.0)
    CitationCoverage,
}

impl QualityMetric {
    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Faithfulness => "Faithfulness",
            Self::Confidence => "Confidence",
            Self::GroundingRatio => "Grounding Ratio",
            Self::ConsistencyScore => "Consistency Score",
            Self::CitationCoverage => "Citation Coverage",
        }
    }

    /// All metric variants.
    pub fn all() -> &'static [QualityMetric] {
        &[
            Self::Faithfulness,
            Self::Confidence,
            Self::GroundingRatio,
            Self::ConsistencyScore,
            Self::CitationCoverage,
        ]
    }
}

impl std::fmt::Display for QualityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Action to take when a gate fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GateAction {
    /// Block the output — return an error
    Fail,
    /// Add a warning but allow the output
    Warn,
    /// Log the failure but pass silently
    Log,
}

impl GateAction {
    /// Whether this action blocks the output.
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Fail)
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Fail => "Fail",
            Self::Warn => "Warn",
            Self::Log => "Log",
        }
    }
}

impl std::fmt::Display for GateAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// A single quality gate: metric + threshold + action.
#[derive(Debug, Clone)]
pub struct QualityGate {
    /// Gate name
    pub name: String,
    /// Which metric to check
    pub metric: QualityMetric,
    /// Minimum acceptable value (0.0 - 1.0)
    pub threshold: f64,
    /// Action when the metric falls below threshold
    pub action: GateAction,
}

impl QualityGate {
    /// Create a new quality gate.
    pub fn new(name: &str, metric: QualityMetric, threshold: f64, action: GateAction) -> Self {
        Self {
            name: name.to_string(),
            metric,
            threshold: threshold.clamp(0.0, 1.0),
            action,
        }
    }

    /// Create a failing gate.
    pub fn fail_below(name: &str, metric: QualityMetric, threshold: f64) -> Self {
        Self::new(name, metric, threshold, GateAction::Fail)
    }

    /// Create a warning gate.
    pub fn warn_below(name: &str, metric: QualityMetric, threshold: f64) -> Self {
        Self::new(name, metric, threshold, GateAction::Warn)
    }

    /// Check if a value passes this gate.
    pub fn check(&self, value: f64) -> GateCheckResult {
        let passed = value >= self.threshold;
        GateCheckResult {
            gate_name: self.name.clone(),
            metric: self.metric,
            threshold: self.threshold,
            actual: value,
            passed,
            action: if passed { None } else { Some(self.action) },
        }
    }
}

/// Result of checking a single gate.
#[derive(Debug, Clone)]
pub struct GateCheckResult {
    /// Gate name
    pub gate_name: String,
    /// Which metric was checked
    pub metric: QualityMetric,
    /// Required threshold
    pub threshold: f64,
    /// Actual value
    pub actual: f64,
    /// Whether the gate passed
    pub passed: bool,
    /// Action to take (None if passed)
    pub action: Option<GateAction>,
}

impl GateCheckResult {
    /// Whether this result requires blocking the output.
    pub fn is_blocking(&self) -> bool {
        self.action.map(|a| a.is_blocking()).unwrap_or(false)
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "{}: PASS ({:.2} >= {:.2})",
                self.gate_name, self.actual, self.threshold
            )
        } else {
            format!(
                "{}: FAIL ({:.2} < {:.2}) [{}]",
                self.gate_name,
                self.actual,
                self.threshold,
                self.action.map(|a| a.display_name()).unwrap_or("N/A")
            )
        }
    }
}

// =============================================================================
// Quality Scores
// =============================================================================

/// A set of quality scores for an LLM output.
#[derive(Debug, Clone, Default)]
pub struct QualityScores {
    /// Faithfulness score (0.0 - 1.0)
    pub faithfulness: Option<f64>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: Option<f64>,
    /// Grounding ratio (0.0 - 1.0)
    pub grounding_ratio: Option<f64>,
    /// Consistency score (0.0 - 1.0)
    pub consistency_score: Option<f64>,
    /// Citation coverage (0.0 - 1.0)
    pub citation_coverage: Option<f64>,
}

impl QualityScores {
    /// Get the value for a specific metric.
    pub fn get(&self, metric: QualityMetric) -> Option<f64> {
        match metric {
            QualityMetric::Faithfulness => self.faithfulness,
            QualityMetric::Confidence => self.confidence,
            QualityMetric::GroundingRatio => self.grounding_ratio,
            QualityMetric::ConsistencyScore => self.consistency_score,
            QualityMetric::CitationCoverage => self.citation_coverage,
        }
    }

    /// Overall quality score (average of available metrics).
    pub fn overall(&self) -> f64 {
        let values: Vec<f64> = [
            self.faithfulness,
            self.confidence,
            self.grounding_ratio,
            self.consistency_score,
            self.citation_coverage,
        ]
        .iter()
        .copied()
        .flatten()
        .collect();

        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Quality badge color: "green" (>= 0.8), "yellow" (>= 0.5), "red" (< 0.5).
    pub fn badge_color(&self) -> &'static str {
        let overall = self.overall();
        if overall >= 0.8 {
            "green"
        } else if overall >= 0.5 {
            "yellow"
        } else {
            "red"
        }
    }
}

// =============================================================================
// Gate Runner
// =============================================================================

/// Runs a set of quality gates against quality scores.
pub struct QualityGateRunner {
    gates: Vec<QualityGate>,
}

impl QualityGateRunner {
    /// Create a new runner with the given gates.
    pub fn new(gates: Vec<QualityGate>) -> Self {
        Self { gates }
    }

    /// Create a runner with default gates for production use.
    pub fn production_defaults() -> Self {
        Self::new(vec![
            QualityGate::fail_below("min-confidence", QualityMetric::Confidence, 0.3),
            QualityGate::warn_below("faithfulness-check", QualityMetric::Faithfulness, 0.7),
            QualityGate::warn_below("grounding-check", QualityMetric::GroundingRatio, 0.5),
        ])
    }

    /// Create a runner with strict gates.
    pub fn strict() -> Self {
        Self::new(vec![
            QualityGate::fail_below("min-confidence", QualityMetric::Confidence, 0.5),
            QualityGate::fail_below("min-faithfulness", QualityMetric::Faithfulness, 0.7),
            QualityGate::fail_below("min-grounding", QualityMetric::GroundingRatio, 0.6),
            QualityGate::warn_below("consistency", QualityMetric::ConsistencyScore, 0.5),
            QualityGate::warn_below("citations", QualityMetric::CitationCoverage, 0.3),
        ])
    }

    /// Run all gates against the given scores.
    pub fn run(&self, scores: &QualityScores) -> QualityGateResult {
        let mut gate_results = Vec::new();
        let mut failing_gates = Vec::new();
        let mut warnings = Vec::new();

        for gate in &self.gates {
            if let Some(value) = scores.get(gate.metric) {
                let result = gate.check(value);
                if !result.passed {
                    if result.is_blocking() {
                        failing_gates.push(result.gate_name.clone());
                    } else {
                        warnings.push(result.gate_name.clone());
                    }
                }
                gate_results.push(result);
            }
            // Skip gates where the metric is not available
        }

        let passed = failing_gates.is_empty();

        QualityGateResult {
            passed,
            gate_results,
            failing_gates,
            warnings,
            overall_score: scores.overall(),
        }
    }

    /// Get the list of configured gates.
    pub fn gates(&self) -> &[QualityGate] {
        &self.gates
    }

    /// Add a gate.
    pub fn add_gate(&mut self, gate: QualityGate) {
        self.gates.push(gate);
    }

    /// Number of gates.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
}

/// Result of running all quality gates.
#[derive(Debug, Clone)]
pub struct QualityGateResult {
    /// Whether all blocking gates passed
    pub passed: bool,
    /// Individual gate results
    pub gate_results: Vec<GateCheckResult>,
    /// Names of gates that failed with Fail action
    pub failing_gates: Vec<String>,
    /// Names of gates that failed with Warn action
    pub warnings: Vec<String>,
    /// Overall quality score
    pub overall_score: f64,
}

impl QualityGateResult {
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let status = if self.passed { "PASSED" } else { "FAILED" };
        let mut lines = vec![format!(
            "Quality Gates: {} (overall: {:.2})",
            status, self.overall_score
        )];

        for result in &self.gate_results {
            lines.push(format!("  {}", result.summary()));
        }

        lines.join("\n")
    }

    /// Number of gates that passed.
    pub fn passed_count(&self) -> usize {
        self.gate_results.iter().filter(|r| r.passed).count()
    }

    /// Total gates checked.
    pub fn total_checked(&self) -> usize {
        self.gate_results.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metric_display() {
        assert_eq!(QualityMetric::Faithfulness.display_name(), "Faithfulness");
        assert_eq!(QualityMetric::Confidence.display_name(), "Confidence");
        assert_eq!(
            QualityMetric::GroundingRatio.display_name(),
            "Grounding Ratio"
        );
    }

    #[test]
    fn test_quality_metric_all() {
        assert_eq!(QualityMetric::all().len(), 5);
    }

    #[test]
    fn test_gate_action_blocking() {
        assert!(GateAction::Fail.is_blocking());
        assert!(!GateAction::Warn.is_blocking());
        assert!(!GateAction::Log.is_blocking());
    }

    #[test]
    fn test_gate_check_pass() {
        let gate = QualityGate::new("test", QualityMetric::Confidence, 0.5, GateAction::Fail);
        let result = gate.check(0.8);
        assert!(result.passed);
        assert!(result.action.is_none());
    }

    #[test]
    fn test_gate_check_fail() {
        let gate = QualityGate::fail_below("test", QualityMetric::Confidence, 0.5);
        let result = gate.check(0.3);
        assert!(!result.passed);
        assert_eq!(result.action, Some(GateAction::Fail));
        assert!(result.is_blocking());
    }

    #[test]
    fn test_gate_check_warn() {
        let gate = QualityGate::warn_below("test", QualityMetric::Faithfulness, 0.7);
        let result = gate.check(0.5);
        assert!(!result.passed);
        assert_eq!(result.action, Some(GateAction::Warn));
        assert!(!result.is_blocking());
    }

    #[test]
    fn test_gate_threshold_clamp() {
        let gate = QualityGate::new("test", QualityMetric::Confidence, 1.5, GateAction::Fail);
        assert_eq!(gate.threshold, 1.0);

        let gate2 = QualityGate::new("test", QualityMetric::Confidence, -0.5, GateAction::Fail);
        assert_eq!(gate2.threshold, 0.0);
    }

    #[test]
    fn test_quality_scores_overall() {
        let scores = QualityScores {
            faithfulness: Some(0.8),
            confidence: Some(0.6),
            grounding_ratio: None,
            consistency_score: None,
            citation_coverage: None,
        };
        let overall = scores.overall();
        assert!((overall - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_quality_scores_empty() {
        let scores = QualityScores::default();
        assert_eq!(scores.overall(), 0.0);
    }

    #[test]
    fn test_quality_scores_badge() {
        let mut scores = QualityScores::default();
        scores.confidence = Some(0.9);
        assert_eq!(scores.badge_color(), "green");

        scores.confidence = Some(0.6);
        assert_eq!(scores.badge_color(), "yellow");

        scores.confidence = Some(0.2);
        assert_eq!(scores.badge_color(), "red");
    }

    #[test]
    fn test_quality_scores_get() {
        let scores = QualityScores {
            faithfulness: Some(0.9),
            confidence: Some(0.8),
            ..Default::default()
        };
        assert_eq!(scores.get(QualityMetric::Faithfulness), Some(0.9));
        assert_eq!(scores.get(QualityMetric::Confidence), Some(0.8));
        assert_eq!(scores.get(QualityMetric::GroundingRatio), None);
    }

    #[test]
    fn test_runner_all_pass() {
        let runner = QualityGateRunner::new(vec![
            QualityGate::fail_below("conf", QualityMetric::Confidence, 0.3),
            QualityGate::warn_below("faith", QualityMetric::Faithfulness, 0.5),
        ]);

        let scores = QualityScores {
            confidence: Some(0.8),
            faithfulness: Some(0.9),
            ..Default::default()
        };

        let result = runner.run(&scores);
        assert!(result.passed);
        assert!(result.failing_gates.is_empty());
        assert!(result.warnings.is_empty());
        assert_eq!(result.passed_count(), 2);
    }

    #[test]
    fn test_runner_fail_blocks() {
        let runner = QualityGateRunner::new(vec![QualityGate::fail_below(
            "min-conf",
            QualityMetric::Confidence,
            0.5,
        )]);

        let scores = QualityScores {
            confidence: Some(0.2),
            ..Default::default()
        };

        let result = runner.run(&scores);
        assert!(!result.passed);
        assert_eq!(result.failing_gates, vec!["min-conf"]);
    }

    #[test]
    fn test_runner_warn_passes() {
        let runner = QualityGateRunner::new(vec![QualityGate::warn_below(
            "faith",
            QualityMetric::Faithfulness,
            0.7,
        )]);

        let scores = QualityScores {
            faithfulness: Some(0.5),
            ..Default::default()
        };

        let result = runner.run(&scores);
        assert!(result.passed); // Warnings don't block
        assert_eq!(result.warnings, vec!["faith"]);
    }

    #[test]
    fn test_runner_skips_missing_metrics() {
        let runner = QualityGateRunner::new(vec![
            QualityGate::fail_below("conf", QualityMetric::Confidence, 0.5),
            QualityGate::fail_below("faith", QualityMetric::Faithfulness, 0.5),
        ]);

        let scores = QualityScores {
            confidence: Some(0.8),
            // faithfulness not set — gate should be skipped
            ..Default::default()
        };

        let result = runner.run(&scores);
        assert!(result.passed);
        assert_eq!(result.total_checked(), 1); // Only confidence was checked
    }

    #[test]
    fn test_runner_production_defaults() {
        let runner = QualityGateRunner::production_defaults();
        assert_eq!(runner.gate_count(), 3);
    }

    #[test]
    fn test_runner_strict() {
        let runner = QualityGateRunner::strict();
        assert_eq!(runner.gate_count(), 5);
    }

    #[test]
    fn test_runner_add_gate() {
        let mut runner = QualityGateRunner::new(Vec::new());
        runner.add_gate(QualityGate::warn_below(
            "test",
            QualityMetric::Confidence,
            0.5,
        ));
        assert_eq!(runner.gate_count(), 1);
    }

    #[test]
    fn test_result_summary() {
        let runner = QualityGateRunner::new(vec![QualityGate::fail_below(
            "conf",
            QualityMetric::Confidence,
            0.5,
        )]);
        let scores = QualityScores {
            confidence: Some(0.8),
            ..Default::default()
        };
        let result = runner.run(&scores);
        let summary = result.summary();
        assert!(summary.contains("PASSED"));
        assert!(summary.contains("conf"));
    }

    #[test]
    fn test_gate_check_result_summary_fail() {
        let gate = QualityGate::fail_below("test-gate", QualityMetric::Confidence, 0.7);
        let result = gate.check(0.3);
        let summary = result.summary();
        assert!(summary.contains("FAIL"));
        assert!(summary.contains("0.30"));
        assert!(summary.contains("0.70"));
    }

    #[test]
    fn test_gate_check_exact_threshold() {
        let gate = QualityGate::fail_below("test", QualityMetric::Confidence, 0.5);
        let result = gate.check(0.5); // Exactly at threshold = pass
        assert!(result.passed);
    }
}
