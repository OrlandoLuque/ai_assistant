//! Agent Trajectory Evaluation
//!
//! Records, analyzes, and scores agent execution trajectories. Provides
//! tool-call accuracy metrics (precision, recall, F1) and composite scoring
//! with configurable weights for cost, time, and efficiency.

use crate::error::AgentEvalError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// StepActionType
// ---------------------------------------------------------------------------

/// The kind of action an agent performed in a single step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepActionType {
    ToolCall,
    LlmQuery,
    Planning,
    MemoryAccess,
    UserInteraction,
    InternalDecision,
}

// ---------------------------------------------------------------------------
// EvalTrajectoryStep
// ---------------------------------------------------------------------------

/// A single recorded step in an agent's execution trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTrajectoryStep {
    pub step_index: usize,
    pub action_type: StepActionType,
    pub action_name: String,
    pub input: String,
    pub output: String,
    pub duration_ms: u64,
    pub token_count: usize,
    pub cost: f64,
    pub success: bool,
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// TrajectoryRecorder
// ---------------------------------------------------------------------------

/// Records agent steps during execution.
#[derive(Debug)]
pub struct TrajectoryRecorder {
    agent_id: String,
    steps: Vec<EvalTrajectoryStep>,
    start_time: u64,
    recording: bool,
    metadata: HashMap<String, String>,
}

impl TrajectoryRecorder {
    /// Create a new recorder for the given agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            steps: Vec::new(),
            start_time: 0,
            recording: false,
            metadata: HashMap::new(),
        }
    }

    /// Begin recording.
    pub fn start(&mut self) {
        self.start_time = now_unix_ms();
        self.recording = true;
    }

    /// Stop recording.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Whether the recorder is currently active.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record a step. Returns an error if not currently recording.
    pub fn record_step(&mut self, step: EvalTrajectoryStep) -> Result<(), AgentEvalError> {
        if !self.recording {
            return Err(AgentEvalError::InvalidConfig {
                field: "recording".into(),
                reason: "TrajectoryRecorder is not recording — call start() first".into(),
            });
        }
        self.steps.push(step);
        Ok(())
    }

    /// All recorded steps.
    pub fn steps(&self) -> &[EvalTrajectoryStep] {
        &self.steps
    }

    /// Number of recorded steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Sum of all step durations in milliseconds.
    pub fn total_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.duration_ms).sum()
    }

    /// Sum of all step token counts.
    pub fn total_tokens(&self) -> usize {
        self.steps.iter().map(|s| s.token_count).sum()
    }

    /// Sum of all step costs.
    pub fn total_cost(&self) -> f64 {
        self.steps.iter().map(|s| s.cost).sum()
    }

    /// Agent identifier.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Attach metadata to the recording session.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Clear all recorded steps and metadata, resetting the recorder.
    pub fn clear(&mut self) {
        self.steps.clear();
        self.metadata.clear();
        self.start_time = 0;
        self.recording = false;
    }

    /// Number of steps whose action type is `ToolCall`.
    pub fn tool_call_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.action_type == StepActionType::ToolCall)
            .count()
    }

    /// Ratio of successful steps to total steps (0.0 if no steps).
    pub fn success_rate(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let successes = self.steps.iter().filter(|s| s.success).count();
        successes as f64 / self.steps.len() as f64
    }
}

// ---------------------------------------------------------------------------
// AgentMetrics
// ---------------------------------------------------------------------------

/// Computed metrics from a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub failed_steps: usize,
    pub total_duration_ms: u64,
    pub avg_step_duration_ms: f64,
    pub total_tokens: usize,
    pub total_cost: f64,
    pub tool_call_count: usize,
    pub llm_query_count: usize,
    pub tool_accuracy: f64,
    pub step_efficiency: f64,
    pub cost_per_successful_step: f64,
}

// ---------------------------------------------------------------------------
// AnalyzerConfig
// ---------------------------------------------------------------------------

/// Configuration for the trajectory analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AnalyzerConfig {
    pub include_failed_steps: bool,
    pub cost_weight: f64,
    pub time_weight: f64,
    pub efficiency_weight: f64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            include_failed_steps: true,
            cost_weight: 1.0,
            time_weight: 1.0,
            efficiency_weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsComparison
// ---------------------------------------------------------------------------

/// Result of comparing two sets of agent metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsComparison {
    pub duration_change_pct: f64,
    pub cost_change_pct: f64,
    pub efficiency_change_pct: f64,
    pub accuracy_change_pct: f64,
    pub improved: bool,
}

// ---------------------------------------------------------------------------
// TrajectoryAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes trajectories and computes metrics.
#[derive(Debug)]
pub struct TrajectoryAnalyzer {
    config: AnalyzerConfig,
}

impl TrajectoryAnalyzer {
    /// Create an analyzer with the given configuration.
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Create an analyzer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AnalyzerConfig::default())
    }

    /// Analyze a slice of trajectory steps, producing aggregate metrics.
    ///
    /// Returns an error if the slice is empty.
    pub fn analyze(&self, steps: &[EvalTrajectoryStep]) -> Result<AgentMetrics, AgentEvalError> {
        if steps.is_empty() {
            return Err(AgentEvalError::TrajectoryEmpty {
                agent_id: "unknown".into(),
            });
        }

        let considered: Vec<&EvalTrajectoryStep> = if self.config.include_failed_steps {
            steps.iter().collect()
        } else {
            steps.iter().filter(|s| s.success).collect()
        };

        let total_steps = considered.len();
        let successful_steps = considered.iter().filter(|s| s.success).count();
        let failed_steps = total_steps - successful_steps;
        let total_duration_ms: u64 = considered.iter().map(|s| s.duration_ms).sum();
        let avg_step_duration_ms = if total_steps > 0 {
            total_duration_ms as f64 / total_steps as f64
        } else {
            0.0
        };
        let total_tokens: usize = considered.iter().map(|s| s.token_count).sum();
        let total_cost: f64 = considered.iter().map(|s| s.cost).sum();

        let tool_call_count = considered
            .iter()
            .filter(|s| s.action_type == StepActionType::ToolCall)
            .count();
        let llm_query_count = considered
            .iter()
            .filter(|s| s.action_type == StepActionType::LlmQuery)
            .count();

        // Tool accuracy: ratio of successful tool calls to total tool calls
        let successful_tool_calls = considered
            .iter()
            .filter(|s| s.action_type == StepActionType::ToolCall && s.success)
            .count();
        let tool_accuracy = if tool_call_count > 0 {
            successful_tool_calls as f64 / tool_call_count as f64
        } else {
            1.0 // no tool calls means no failures
        };

        // Step efficiency: ratio of successful steps to total steps
        let step_efficiency = if total_steps > 0 {
            successful_steps as f64 / total_steps as f64
        } else {
            0.0
        };

        let cost_per_successful_step = if successful_steps > 0 {
            total_cost / successful_steps as f64
        } else {
            0.0
        };

        Ok(AgentMetrics {
            total_steps,
            successful_steps,
            failed_steps,
            total_duration_ms,
            avg_step_duration_ms,
            total_tokens,
            total_cost,
            tool_call_count,
            llm_query_count,
            tool_accuracy,
            step_efficiency,
            cost_per_successful_step,
        })
    }

    /// Compare baseline metrics against current metrics.
    pub fn compare(&self, baseline: &AgentMetrics, current: &AgentMetrics) -> MetricsComparison {
        let duration_change_pct = pct_change(
            baseline.total_duration_ms as f64,
            current.total_duration_ms as f64,
        );
        let cost_change_pct = pct_change(baseline.total_cost, current.total_cost);
        let efficiency_change_pct =
            pct_change(baseline.step_efficiency, current.step_efficiency);
        let accuracy_change_pct =
            pct_change(baseline.tool_accuracy, current.tool_accuracy);

        // "improved" = duration went down OR stayed same, AND efficiency went up or stayed same
        let improved = current.total_duration_ms <= baseline.total_duration_ms
            && current.step_efficiency >= baseline.step_efficiency;

        MetricsComparison {
            duration_change_pct,
            cost_change_pct,
            efficiency_change_pct,
            accuracy_change_pct,
            improved,
        }
    }

    /// Composite score in 0.0..=1.0 based on config weights.
    ///
    /// The score blends three normalised components:
    /// - **cost component**: lower cost → higher score
    /// - **time component**: lower average step time → higher score
    /// - **efficiency component**: higher step efficiency → higher score
    pub fn score(&self, metrics: &AgentMetrics) -> f64 {
        // Normalise cost component: use 1/(1+cost) so it maps (0..inf) → (0..1]
        let cost_norm = 1.0 / (1.0 + metrics.total_cost);

        // Normalise time component: use 1/(1+avg_ms/1000)
        let time_norm = 1.0 / (1.0 + metrics.avg_step_duration_ms / 1000.0);

        // Efficiency is already 0..1
        let eff_norm = metrics.step_efficiency;

        let total_weight =
            self.config.cost_weight + self.config.time_weight + self.config.efficiency_weight;

        if total_weight <= 0.0 {
            return 0.0;
        }

        let raw = (self.config.cost_weight * cost_norm
            + self.config.time_weight * time_norm
            + self.config.efficiency_weight * eff_norm)
            / total_weight;

        raw.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// StepBreakdown
// ---------------------------------------------------------------------------

/// Per-action-type breakdown inside an evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepBreakdown {
    pub action_type: StepActionType,
    pub count: usize,
    pub avg_duration_ms: f64,
    pub success_rate: f64,
    pub total_cost: f64,
}

// ---------------------------------------------------------------------------
// EvalReport
// ---------------------------------------------------------------------------

/// Full evaluation report for an agent trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub agent_id: String,
    pub metrics: AgentMetrics,
    pub score: f64,
    pub comparison: Option<MetricsComparison>,
    pub step_breakdown: Vec<StepBreakdown>,
    pub recommendations: Vec<String>,
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// ReportBuilder
// ---------------------------------------------------------------------------

/// Builds a complete `EvalReport` from a recorder and analyzer.
#[derive(Debug)]
pub struct ReportBuilder {
    analyzer: TrajectoryAnalyzer,
}

impl ReportBuilder {
    /// Create a builder with a custom analyzer.
    pub fn new(analyzer: TrajectoryAnalyzer) -> Self {
        Self { analyzer }
    }

    /// Create a builder with default analyzer settings.
    pub fn with_defaults() -> Self {
        Self::new(TrajectoryAnalyzer::with_defaults())
    }

    /// Build a full evaluation report.
    ///
    /// If `baseline` is provided the report will include a metrics comparison.
    pub fn build(
        &self,
        recorder: &TrajectoryRecorder,
        baseline: Option<&AgentMetrics>,
    ) -> Result<EvalReport, AgentEvalError> {
        let steps = recorder.steps();
        if steps.is_empty() {
            return Err(AgentEvalError::TrajectoryEmpty {
                agent_id: recorder.agent_id().into(),
            });
        }

        // 1. Analyze
        let metrics = self.analyzer.analyze(steps)?;

        // 2. Comparison
        let comparison = baseline.map(|b| self.analyzer.compare(b, &metrics));

        // 3. Score
        let score = self.analyzer.score(&metrics);

        // 4. Step breakdown by action type
        let step_breakdown = compute_breakdowns(steps);

        // 5. Recommendations
        let recommendations = generate_recommendations(&metrics, &step_breakdown);

        Ok(EvalReport {
            agent_id: recorder.agent_id().into(),
            metrics,
            score,
            comparison,
            step_breakdown,
            recommendations,
            timestamp: now_unix_ms(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tool-Call Accuracy (Item 7.2)
// ---------------------------------------------------------------------------

/// An expected tool call for accuracy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedToolCall {
    pub tool_name: String,
    pub expected_args: Option<HashMap<String, serde_json::Value>>,
    pub order_sensitive: bool,
}

/// The result of matching one expected call to one actual step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallMatch {
    pub expected: ExpectedToolCall,
    pub actual: Option<EvalTrajectoryStep>,
    pub name_match: bool,
    pub args_match: bool,
}

/// Aggregate tool-call accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAccuracyMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub arg_accuracy: f64,
    pub order_accuracy: f64,
    pub matches: Vec<ToolCallMatch>,
}

/// Evaluates tool-call accuracy comparing expected calls against actual steps.
#[derive(Debug)]
pub struct ToolCallEvaluator {
    order_sensitive: bool,
}

impl ToolCallEvaluator {
    /// Create an evaluator. If `order_sensitive` is true the order of tool
    /// calls is factored into the `order_accuracy` metric.
    pub fn new(order_sensitive: bool) -> Self {
        Self { order_sensitive }
    }

    /// Evaluate expected tool calls against actual trajectory steps.
    ///
    /// Only steps with `action_type == ToolCall` are considered.
    pub fn evaluate(
        &self,
        expected: &[ExpectedToolCall],
        actual: &[EvalTrajectoryStep],
    ) -> ToolAccuracyMetrics {
        let actual_tool_calls: Vec<&EvalTrajectoryStep> = actual
            .iter()
            .filter(|s| s.action_type == StepActionType::ToolCall)
            .collect();

        if expected.is_empty() && actual_tool_calls.is_empty() {
            return ToolAccuracyMetrics {
                precision: 1.0,
                recall: 1.0,
                f1_score: 1.0,
                arg_accuracy: 1.0,
                order_accuracy: 1.0,
                matches: Vec::new(),
            };
        }

        // Track which actual calls have been claimed
        let mut used: Vec<bool> = vec![false; actual_tool_calls.len()];
        let mut matches: Vec<ToolCallMatch> = Vec::new();
        let mut matched_expected_count = 0usize;
        let mut arg_match_count = 0usize;
        let mut arg_eval_count = 0usize;

        // For order accuracy: record the index of the matched actual call
        let mut matched_actual_indices: Vec<Option<usize>> = Vec::new();

        for exp in expected {
            // Find the first unused actual call with the same name
            let mut found = None;
            for (idx, act) in actual_tool_calls.iter().enumerate() {
                if !used[idx] && act.action_name == exp.tool_name {
                    found = Some(idx);
                    break;
                }
            }

            if let Some(idx) = found {
                used[idx] = true;
                matched_expected_count += 1;
                let act = actual_tool_calls[idx];

                // Argument accuracy: parse actual output as JSON args map if
                // expected_args is Some; otherwise count as matching.
                let args_ok = match &exp.expected_args {
                    Some(exp_args) => {
                        arg_eval_count += 1;
                        let actual_args: HashMap<String, serde_json::Value> =
                            serde_json::from_str(&act.input).unwrap_or_default();
                        let ok = args_subset_match(exp_args, &actual_args);
                        if ok {
                            arg_match_count += 1;
                        }
                        ok
                    }
                    None => {
                        // No expected args — consider args matching by default
                        true
                    }
                };

                matches.push(ToolCallMatch {
                    expected: exp.clone(),
                    actual: Some(act.clone()),
                    name_match: true,
                    args_match: args_ok,
                });
                matched_actual_indices.push(Some(idx));
            } else {
                matches.push(ToolCallMatch {
                    expected: exp.clone(),
                    actual: None,
                    name_match: false,
                    args_match: false,
                });
                matched_actual_indices.push(None);
            }
        }

        let matched_actual_count = used.iter().filter(|&&u| u).count();

        // Precision = matched / actual
        let precision = if actual_tool_calls.is_empty() {
            0.0
        } else {
            matched_actual_count as f64 / actual_tool_calls.len() as f64
        };

        // Recall = matched / expected
        let recall = if expected.is_empty() {
            0.0
        } else {
            matched_expected_count as f64 / expected.len() as f64
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let arg_accuracy = if arg_eval_count > 0 {
            arg_match_count as f64 / arg_eval_count as f64
        } else {
            1.0
        };

        // Order accuracy: percentage of matched indices that are in
        // non-decreasing order (sequential match).
        let order_accuracy = if self.order_sensitive {
            compute_order_accuracy(&matched_actual_indices)
        } else {
            1.0
        };

        ToolAccuracyMetrics {
            precision,
            recall,
            f1_score,
            arg_accuracy,
            order_accuracy,
            matches,
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Percentage change from `old` to `new`. Returns 0.0 when `old` is zero.
fn pct_change(old: f64, new: f64) -> f64 {
    if old.abs() < f64::EPSILON {
        if new.abs() < f64::EPSILON {
            0.0
        } else {
            100.0 // from zero to something
        }
    } else {
        ((new - old) / old) * 100.0
    }
}

/// Current wall-clock time as milliseconds since the UNIX epoch.
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Build per-action-type breakdowns.
fn compute_breakdowns(steps: &[EvalTrajectoryStep]) -> Vec<StepBreakdown> {
    let action_types = [
        StepActionType::ToolCall,
        StepActionType::LlmQuery,
        StepActionType::Planning,
        StepActionType::MemoryAccess,
        StepActionType::UserInteraction,
        StepActionType::InternalDecision,
    ];

    let mut breakdowns = Vec::new();

    for at in &action_types {
        let matching: Vec<&EvalTrajectoryStep> =
            steps.iter().filter(|s| &s.action_type == at).collect();
        if matching.is_empty() {
            continue;
        }
        let count = matching.len();
        let total_dur: u64 = matching.iter().map(|s| s.duration_ms).sum();
        let avg_duration_ms = total_dur as f64 / count as f64;
        let successes = matching.iter().filter(|s| s.success).count();
        let success_rate = successes as f64 / count as f64;
        let total_cost: f64 = matching.iter().map(|s| s.cost).sum();

        breakdowns.push(StepBreakdown {
            action_type: at.clone(),
            count,
            avg_duration_ms,
            success_rate,
            total_cost,
        });
    }

    breakdowns
}

/// Generate human-readable recommendations from metrics and breakdowns.
fn generate_recommendations(
    metrics: &AgentMetrics,
    breakdowns: &[StepBreakdown],
) -> Vec<String> {
    let mut recs = Vec::new();

    // High failure rate overall
    if metrics.step_efficiency < 0.8 && metrics.total_steps > 0 {
        recs.push(format!(
            "Overall step success rate is {:.0}% — investigate failing steps",
            metrics.step_efficiency * 100.0
        ));
    }

    // High failure rate per action type
    for bd in breakdowns {
        if bd.success_rate < 0.7 && bd.count >= 2 {
            recs.push(format!(
                "High failure rate on {:?} steps ({:.0}% success across {} calls)",
                bd.action_type,
                bd.success_rate * 100.0,
                bd.count
            ));
        }
    }

    // Cost per step above average (heuristic: > $0.05)
    if metrics.cost_per_successful_step > 0.05 && metrics.successful_steps > 0 {
        recs.push(format!(
            "Cost per successful step is ${:.4} — consider cheaper models or fewer retries",
            metrics.cost_per_successful_step
        ));
    }

    // Too many tool calls relative to total steps
    if metrics.total_steps > 3 && metrics.tool_call_count as f64 / metrics.total_steps as f64 > 0.8
    {
        recs.push(
            "Agent is heavily tool-dependent — consider caching or planning steps".into(),
        );
    }

    // High average latency
    if metrics.avg_step_duration_ms > 5000.0 {
        recs.push(format!(
            "Average step duration is {:.0}ms — look for slow tool calls or large prompts",
            metrics.avg_step_duration_ms
        ));
    }

    recs
}

/// Check whether every key-value pair in `expected` appears in `actual`.
fn args_subset_match(
    expected: &HashMap<String, serde_json::Value>,
    actual: &HashMap<String, serde_json::Value>,
) -> bool {
    for (k, v) in expected {
        match actual.get(k) {
            Some(av) if av == v => {}
            _ => return false,
        }
    }
    true
}

/// Compute order accuracy as the fraction of consecutive matched-index pairs
/// that are in non-decreasing order.
fn compute_order_accuracy(indices: &[Option<usize>]) -> f64 {
    let valid: Vec<usize> = indices.iter().filter_map(|i| *i).collect();
    if valid.len() <= 1 {
        return 1.0;
    }
    let pairs = valid.len() - 1;
    let ordered = valid.windows(2).filter(|w| w[0] <= w[1]).count();
    ordered as f64 / pairs as f64
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn make_step(name: &str, action: StepActionType, success: bool) -> EvalTrajectoryStep {
        EvalTrajectoryStep {
            step_index: 0,
            action_type: action,
            action_name: name.to_string(),
            input: String::new(),
            output: String::new(),
            duration_ms: 100,
            token_count: 50,
            cost: 0.01,
            success,
            metadata: HashMap::new(),
        }
    }

    fn make_step_full(
        name: &str,
        action: StepActionType,
        success: bool,
        duration_ms: u64,
        token_count: usize,
        cost: f64,
    ) -> EvalTrajectoryStep {
        EvalTrajectoryStep {
            step_index: 0,
            action_type: action,
            action_name: name.to_string(),
            input: String::new(),
            output: String::new(),
            duration_ms,
            token_count,
            cost,
            success,
            metadata: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // TrajectoryRecorder
    // -----------------------------------------------------------------------

    #[test]
    fn test_recorder_new() {
        let rec = TrajectoryRecorder::new("agent-1");
        assert_eq!(rec.agent_id(), "agent-1");
        assert!(!rec.is_recording());
        assert_eq!(rec.step_count(), 0);
    }

    #[test]
    fn test_recorder_start_stop() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        assert!(rec.is_recording());
        rec.stop();
        assert!(!rec.is_recording());
    }

    #[test]
    fn test_recorder_is_recording() {
        let mut rec = TrajectoryRecorder::new("a");
        assert!(!rec.is_recording());
        rec.start();
        assert!(rec.is_recording());
    }

    #[test]
    fn test_record_step_not_recording_error() {
        let mut rec = TrajectoryRecorder::new("a");
        let step = make_step("s1", StepActionType::ToolCall, true);
        let res = rec.record_step(step);
        assert!(res.is_err());
    }

    #[test]
    fn test_record_step_recording_success() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        let step = make_step("s1", StepActionType::ToolCall, true);
        assert!(rec.record_step(step).is_ok());
        assert_eq!(rec.step_count(), 1);
    }

    #[test]
    fn test_steps_returns_all() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("s1", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("s2", StepActionType::LlmQuery, false))
            .unwrap();
        assert_eq!(rec.steps().len(), 2);
        assert_eq!(rec.steps()[0].action_name, "s1");
        assert_eq!(rec.steps()[1].action_name, "s2");
    }

    #[test]
    fn test_step_count() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("s1", StepActionType::Planning, true))
            .unwrap();
        rec.record_step(make_step("s2", StepActionType::Planning, true))
            .unwrap();
        rec.record_step(make_step("s3", StepActionType::Planning, true))
            .unwrap();
        assert_eq!(rec.step_count(), 3);
    }

    #[test]
    fn test_total_duration_ms() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step_full("a", StepActionType::ToolCall, true, 200, 10, 0.0))
            .unwrap();
        rec.record_step(make_step_full("b", StepActionType::ToolCall, true, 300, 10, 0.0))
            .unwrap();
        assert_eq!(rec.total_duration_ms(), 500);
    }

    #[test]
    fn test_total_tokens() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step_full("a", StepActionType::LlmQuery, true, 10, 100, 0.0))
            .unwrap();
        rec.record_step(make_step_full("b", StepActionType::LlmQuery, true, 10, 250, 0.0))
            .unwrap();
        assert_eq!(rec.total_tokens(), 350);
    }

    #[test]
    fn test_total_cost() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step_full("a", StepActionType::LlmQuery, true, 10, 10, 0.05))
            .unwrap();
        rec.record_step(make_step_full("b", StepActionType::LlmQuery, true, 10, 10, 0.03))
            .unwrap();
        let total = rec.total_cost();
        assert!((total - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_tool_call_count_filters() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("t1", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("q1", StepActionType::LlmQuery, true))
            .unwrap();
        rec.record_step(make_step("t2", StepActionType::ToolCall, false))
            .unwrap();
        assert_eq!(rec.tool_call_count(), 2);
    }

    #[test]
    fn test_success_rate_all_success() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("a", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("b", StepActionType::ToolCall, true))
            .unwrap();
        assert!((rec.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_all_fail() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("a", StepActionType::ToolCall, false))
            .unwrap();
        rec.record_step(make_step("b", StepActionType::ToolCall, false))
            .unwrap();
        assert!((rec.success_rate()).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("a", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("b", StepActionType::ToolCall, false))
            .unwrap();
        rec.record_step(make_step("c", StepActionType::LlmQuery, true))
            .unwrap();
        // 2/3
        assert!((rec.success_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("a", StepActionType::ToolCall, true))
            .unwrap();
        rec.add_metadata("k".into(), "v".into());
        rec.clear();
        assert_eq!(rec.step_count(), 0);
        assert!(!rec.is_recording());
    }

    // -----------------------------------------------------------------------
    // TrajectoryAnalyzer
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_empty_error() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let res = analyzer.analyze(&[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_analyze_valid_steps() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let steps = vec![
            make_step_full("t1", StepActionType::ToolCall, true, 100, 50, 0.01),
            make_step_full("q1", StepActionType::LlmQuery, true, 200, 100, 0.02),
            make_step_full("t2", StepActionType::ToolCall, false, 150, 30, 0.005),
        ];
        let m = analyzer.analyze(&steps).unwrap();
        assert_eq!(m.total_steps, 3);
        assert_eq!(m.successful_steps, 2);
        assert_eq!(m.failed_steps, 1);
        assert_eq!(m.total_duration_ms, 450);
        assert!((m.avg_step_duration_ms - 150.0).abs() < 1e-9);
        assert_eq!(m.total_tokens, 180);
        assert!((m.total_cost - 0.035).abs() < 1e-9);
        assert_eq!(m.tool_call_count, 2);
        assert_eq!(m.llm_query_count, 1);
        // tool accuracy: 1 success / 2 calls = 0.5
        assert!((m.tool_accuracy - 0.5).abs() < 1e-9);
        // efficiency: 2/3
        assert!((m.step_efficiency - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compare_improvement() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let baseline = AgentMetrics {
            total_steps: 5,
            successful_steps: 3,
            failed_steps: 2,
            total_duration_ms: 1000,
            avg_step_duration_ms: 200.0,
            total_tokens: 500,
            total_cost: 0.10,
            tool_call_count: 3,
            llm_query_count: 2,
            tool_accuracy: 0.6,
            step_efficiency: 0.6,
            cost_per_successful_step: 0.033,
        };
        let current = AgentMetrics {
            total_steps: 4,
            successful_steps: 4,
            failed_steps: 0,
            total_duration_ms: 800,
            avg_step_duration_ms: 200.0,
            total_tokens: 400,
            total_cost: 0.08,
            tool_call_count: 2,
            llm_query_count: 2,
            tool_accuracy: 1.0,
            step_efficiency: 1.0,
            cost_per_successful_step: 0.02,
        };
        let cmp = analyzer.compare(&baseline, &current);
        assert!(cmp.improved);
        assert!(cmp.duration_change_pct < 0.0); // duration went down
        assert!(cmp.efficiency_change_pct > 0.0); // efficiency went up
    }

    #[test]
    fn test_compare_regression() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let baseline = AgentMetrics {
            total_steps: 3,
            successful_steps: 3,
            failed_steps: 0,
            total_duration_ms: 300,
            avg_step_duration_ms: 100.0,
            total_tokens: 150,
            total_cost: 0.03,
            tool_call_count: 2,
            llm_query_count: 1,
            tool_accuracy: 1.0,
            step_efficiency: 1.0,
            cost_per_successful_step: 0.01,
        };
        let current = AgentMetrics {
            total_steps: 5,
            successful_steps: 2,
            failed_steps: 3,
            total_duration_ms: 500,
            avg_step_duration_ms: 100.0,
            total_tokens: 250,
            total_cost: 0.05,
            tool_call_count: 3,
            llm_query_count: 2,
            tool_accuracy: 0.5,
            step_efficiency: 0.4,
            cost_per_successful_step: 0.025,
        };
        let cmp = analyzer.compare(&baseline, &current);
        assert!(!cmp.improved);
    }

    #[test]
    fn test_score_range_0_to_1() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let m = AgentMetrics {
            total_steps: 10,
            successful_steps: 8,
            failed_steps: 2,
            total_duration_ms: 5000,
            avg_step_duration_ms: 500.0,
            total_tokens: 1000,
            total_cost: 0.50,
            tool_call_count: 5,
            llm_query_count: 5,
            tool_accuracy: 0.8,
            step_efficiency: 0.8,
            cost_per_successful_step: 0.0625,
        };
        let s = analyzer.score(&m);
        assert!(s >= 0.0 && s <= 1.0, "score {} out of range", s);
    }

    #[test]
    fn test_cost_per_successful_step_calculation() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let steps = vec![
            make_step_full("a", StepActionType::ToolCall, true, 100, 50, 0.10),
            make_step_full("b", StepActionType::ToolCall, false, 100, 50, 0.10),
        ];
        let m = analyzer.analyze(&steps).unwrap();
        // cost_per_successful_step = 0.20 / 1 = 0.20
        assert!((m.cost_per_successful_step - 0.20).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_comparison_improved_flag() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        // Same duration, same efficiency → improved = true
        let baseline = AgentMetrics {
            total_steps: 2,
            successful_steps: 2,
            failed_steps: 0,
            total_duration_ms: 200,
            avg_step_duration_ms: 100.0,
            total_tokens: 100,
            total_cost: 0.02,
            tool_call_count: 1,
            llm_query_count: 1,
            tool_accuracy: 1.0,
            step_efficiency: 1.0,
            cost_per_successful_step: 0.01,
        };
        let cmp = analyzer.compare(&baseline, &baseline);
        assert!(cmp.improved);
    }

    // -----------------------------------------------------------------------
    // ReportBuilder
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_builder_no_baseline() {
        let mut rec = TrajectoryRecorder::new("agent-42");
        rec.start();
        rec.record_step(make_step("t1", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("q1", StepActionType::LlmQuery, true))
            .unwrap();
        rec.stop();

        let builder = ReportBuilder::with_defaults();
        let report = builder.build(&rec, None).unwrap();
        assert_eq!(report.agent_id, "agent-42");
        assert!(report.comparison.is_none());
        assert!(report.score >= 0.0 && report.score <= 1.0);
        assert!(!report.step_breakdown.is_empty());
    }

    #[test]
    fn test_report_builder_with_baseline() {
        let mut rec = TrajectoryRecorder::new("agent-42");
        rec.start();
        rec.record_step(make_step("t1", StepActionType::ToolCall, true))
            .unwrap();
        rec.stop();

        let baseline = AgentMetrics {
            total_steps: 5,
            successful_steps: 3,
            failed_steps: 2,
            total_duration_ms: 1000,
            avg_step_duration_ms: 200.0,
            total_tokens: 500,
            total_cost: 0.10,
            tool_call_count: 3,
            llm_query_count: 2,
            tool_accuracy: 0.6,
            step_efficiency: 0.6,
            cost_per_successful_step: 0.033,
        };

        let builder = ReportBuilder::with_defaults();
        let report = builder.build(&rec, Some(&baseline)).unwrap();
        assert!(report.comparison.is_some());
    }

    #[test]
    fn test_report_step_breakdown_by_action_type() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        rec.record_step(make_step("t1", StepActionType::ToolCall, true))
            .unwrap();
        rec.record_step(make_step("t2", StepActionType::ToolCall, false))
            .unwrap();
        rec.record_step(make_step("q1", StepActionType::LlmQuery, true))
            .unwrap();
        rec.stop();

        let builder = ReportBuilder::with_defaults();
        let report = builder.build(&rec, None).unwrap();

        // Should have breakdowns for ToolCall and LlmQuery
        let tool_bd = report
            .step_breakdown
            .iter()
            .find(|b| b.action_type == StepActionType::ToolCall);
        assert!(tool_bd.is_some());
        let tool_bd = tool_bd.unwrap();
        assert_eq!(tool_bd.count, 2);
        assert!((tool_bd.success_rate - 0.5).abs() < 1e-9);

        let llm_bd = report
            .step_breakdown
            .iter()
            .find(|b| b.action_type == StepActionType::LlmQuery);
        assert!(llm_bd.is_some());
        assert_eq!(llm_bd.unwrap().count, 1);
    }

    #[test]
    fn test_report_recommendations_for_failures() {
        let mut rec = TrajectoryRecorder::new("a");
        rec.start();
        // Create many failing tool calls to trigger recommendations
        for i in 0..5 {
            rec.record_step(make_step(
                &format!("t{}", i),
                StepActionType::ToolCall,
                false,
            ))
            .unwrap();
        }
        rec.stop();

        let builder = ReportBuilder::with_defaults();
        let report = builder.build(&rec, None).unwrap();
        // Should have recommendation about low success rate
        assert!(
            !report.recommendations.is_empty(),
            "Expected recommendations for high failure rate"
        );
    }

    // -----------------------------------------------------------------------
    // ToolCallEvaluator
    // -----------------------------------------------------------------------

    #[test]
    fn test_expected_tool_call_construction() {
        let etc = ExpectedToolCall {
            tool_name: "search".into(),
            expected_args: None,
            order_sensitive: false,
        };
        assert_eq!(etc.tool_name, "search");
        assert!(etc.expected_args.is_none());
    }

    #[test]
    fn test_tool_evaluator_no_expected() {
        let eval = ToolCallEvaluator::new(false);
        let result = eval.evaluate(&[], &[]);
        assert!((result.precision - 1.0).abs() < 1e-9);
        assert!((result.recall - 1.0).abs() < 1e-9);
        assert!((result.f1_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_perfect_match() {
        let eval = ToolCallEvaluator::new(false);
        let expected = vec![
            ExpectedToolCall {
                tool_name: "search".into(),
                expected_args: None,
                order_sensitive: false,
            },
            ExpectedToolCall {
                tool_name: "fetch".into(),
                expected_args: None,
                order_sensitive: false,
            },
        ];
        let actual = vec![
            make_step("search", StepActionType::ToolCall, true),
            make_step("fetch", StepActionType::ToolCall, true),
        ];
        let result = eval.evaluate(&expected, &actual);
        assert!((result.precision - 1.0).abs() < 1e-9);
        assert!((result.recall - 1.0).abs() < 1e-9);
        assert!((result.f1_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_partial_match_missing() {
        let eval = ToolCallEvaluator::new(false);
        let expected = vec![
            ExpectedToolCall {
                tool_name: "search".into(),
                expected_args: None,
                order_sensitive: false,
            },
            ExpectedToolCall {
                tool_name: "fetch".into(),
                expected_args: None,
                order_sensitive: false,
            },
        ];
        let actual = vec![make_step("search", StepActionType::ToolCall, true)];
        let result = eval.evaluate(&expected, &actual);
        // precision: 1/1 = 1.0, recall: 1/2 = 0.5
        assert!((result.precision - 1.0).abs() < 1e-9);
        assert!((result.recall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_extra_calls() {
        let eval = ToolCallEvaluator::new(false);
        let expected = vec![ExpectedToolCall {
            tool_name: "search".into(),
            expected_args: None,
            order_sensitive: false,
        }];
        let actual = vec![
            make_step("search", StepActionType::ToolCall, true),
            make_step("extra", StepActionType::ToolCall, true),
        ];
        let result = eval.evaluate(&expected, &actual);
        // precision: 1/2 = 0.5, recall: 1/1 = 1.0
        assert!((result.precision - 0.5).abs() < 1e-9);
        assert!((result.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_arg_accuracy_match() {
        let eval = ToolCallEvaluator::new(false);
        let mut args = HashMap::new();
        args.insert("query".into(), serde_json::Value::String("rust".into()));

        let expected = vec![ExpectedToolCall {
            tool_name: "search".into(),
            expected_args: Some(args),
            order_sensitive: false,
        }];

        let mut step = make_step("search", StepActionType::ToolCall, true);
        step.input = r#"{"query":"rust"}"#.to_string();

        let result = eval.evaluate(&expected, &[step]);
        assert!((result.arg_accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_arg_accuracy_differ() {
        let eval = ToolCallEvaluator::new(false);
        let mut args = HashMap::new();
        args.insert("query".into(), serde_json::Value::String("rust".into()));

        let expected = vec![ExpectedToolCall {
            tool_name: "search".into(),
            expected_args: Some(args),
            order_sensitive: false,
        }];

        let mut step = make_step("search", StepActionType::ToolCall, true);
        step.input = r#"{"query":"python"}"#.to_string();

        let result = eval.evaluate(&expected, &[step]);
        assert!((result.arg_accuracy - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tool_evaluator_order_sensitive() {
        let eval = ToolCallEvaluator::new(true);
        let expected = vec![
            ExpectedToolCall {
                tool_name: "a".into(),
                expected_args: None,
                order_sensitive: true,
            },
            ExpectedToolCall {
                tool_name: "b".into(),
                expected_args: None,
                order_sensitive: true,
            },
        ];
        // Actual order matches expected
        let actual_ordered = vec![
            make_step("a", StepActionType::ToolCall, true),
            make_step("b", StepActionType::ToolCall, true),
        ];
        let result = eval.evaluate(&expected, &actual_ordered);
        assert!((result.order_accuracy - 1.0).abs() < 1e-9);

        // Actual order is reversed
        let actual_reversed = vec![
            make_step("b", StepActionType::ToolCall, true),
            make_step("a", StepActionType::ToolCall, true),
        ];
        let result2 = eval.evaluate(&expected, &actual_reversed);
        assert!((result2.order_accuracy - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_f1_score_computation() {
        // precision = 0.5, recall = 1.0 → F1 = 2*0.5*1.0/(0.5+1.0) = 2/3
        let eval = ToolCallEvaluator::new(false);
        let expected = vec![ExpectedToolCall {
            tool_name: "search".into(),
            expected_args: None,
            order_sensitive: false,
        }];
        let actual = vec![
            make_step("search", StepActionType::ToolCall, true),
            make_step("extra", StepActionType::ToolCall, true),
        ];
        let result = eval.evaluate(&expected, &actual);
        let expected_f1 = 2.0 * 0.5 * 1.0 / (0.5 + 1.0);
        assert!((result.f1_score - expected_f1).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // StepActionType
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_action_type_all_variants() {
        let variants = vec![
            StepActionType::ToolCall,
            StepActionType::LlmQuery,
            StepActionType::Planning,
            StepActionType::MemoryAccess,
            StepActionType::UserInteraction,
            StepActionType::InternalDecision,
        ];
        assert_eq!(variants.len(), 6);
        // Each variant is distinct
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // AnalyzerConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyzer_config_defaults() {
        let cfg = AnalyzerConfig::default();
        assert!(cfg.include_failed_steps);
        assert!((cfg.cost_weight - 1.0).abs() < 1e-9);
        assert!((cfg.time_weight - 1.0).abs() < 1e-9);
        assert!((cfg.efficiency_weight - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_success_rate_empty_recorder() {
        let rec = TrajectoryRecorder::new("a");
        assert!((rec.success_rate()).abs() < 1e-9);
    }

    #[test]
    fn test_report_builder_empty_recorder_error() {
        let rec = TrajectoryRecorder::new("agent-empty");
        let builder = ReportBuilder::with_defaults();
        let res = builder.build(&rec, None);
        assert!(res.is_err());
    }

    #[test]
    fn test_score_perfect_agent() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let m = AgentMetrics {
            total_steps: 3,
            successful_steps: 3,
            failed_steps: 0,
            total_duration_ms: 30,
            avg_step_duration_ms: 10.0,
            total_tokens: 60,
            total_cost: 0.0,
            tool_call_count: 2,
            llm_query_count: 1,
            tool_accuracy: 1.0,
            step_efficiency: 1.0,
            cost_per_successful_step: 0.0,
        };
        let s = analyzer.score(&m);
        // cost=0 → cost_norm=1.0, time=10ms → time_norm≈0.99, eff=1.0
        assert!(s > 0.95, "perfect agent score should be high, got {}", s);
    }

    #[test]
    fn test_score_terrible_agent() {
        let analyzer = TrajectoryAnalyzer::with_defaults();
        let m = AgentMetrics {
            total_steps: 10,
            successful_steps: 0,
            failed_steps: 10,
            total_duration_ms: 100_000,
            avg_step_duration_ms: 10_000.0,
            total_tokens: 50_000,
            total_cost: 100.0,
            tool_call_count: 10,
            llm_query_count: 0,
            tool_accuracy: 0.0,
            step_efficiency: 0.0,
            cost_per_successful_step: 0.0,
        };
        let s = analyzer.score(&m);
        assert!(s < 0.2, "terrible agent score should be low, got {}", s);
    }
}
