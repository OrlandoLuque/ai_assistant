//! Human-in-the-Loop (HITL) module for AI assistant operations.
//!
//! Provides four complementary capabilities:
//!
//! - **3.1 Tool Approval Gates** — Intercept tool calls and require human approval
//!   before execution, with pluggable gate implementations (callback, auto-approve,
//!   auto-deny) and full audit logging.
//!
//! - **3.2 Confidence-Based Escalation** — Estimate confidence from weighted signals
//!   and evaluate escalation policies that trigger actions (pause, abort, request
//!   approval) when thresholds are breached.
//!
//! - **3.3 Interactive Corrections** — Allow humans to inject corrections into an
//!   agent's execution (replace output, modify plan, add context, skip/retry steps)
//!   with a tracked correction history.
//!
//! - **3.4 Declarative Approval Policies** — Define approval rules in code or JSON
//!   with conditions (tool name match, impact level, cost, agent id) and actions
//!   (auto-approve, auto-deny, require human approval), evaluated by priority.
//!
//! Feature-gated behind the `hitl` feature flag (handled by lib.rs).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{AiError, HitlError};

// ============================================================================
// Utility
// ============================================================================

/// Returns the current UNIX timestamp in seconds.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// 3.1 — Tool Approval Gates
// ============================================================================

/// Impact level of a tool invocation, from benign to potentially destructive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl ImpactLevel {
    /// Returns a numeric rank (0 = Low, 3 = Critical) for ordering comparisons.
    pub fn rank(&self) -> u8 {
        match self {
            ImpactLevel::Low => 0,
            ImpactLevel::Medium => 1,
            ImpactLevel::High => 2,
            ImpactLevel::Critical => 3,
        }
    }
}

impl PartialOrd for ImpactLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ImpactLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl Eq for ImpactLevel {}

/// A request for human approval before executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Unique identifier for this approval request.
    pub request_id: String,
    /// Name of the tool to be executed.
    pub tool_name: String,
    /// Arguments that will be passed to the tool.
    pub arguments: HashMap<String, serde_json::Value>,
    /// Identifier of the agent requesting execution.
    pub agent_id: String,
    /// Human-readable context describing why the tool is being called.
    pub context: String,
    /// Estimated impact of this tool invocation.
    pub estimated_impact: ImpactLevel,
    /// UNIX timestamp (seconds) when the request was created.
    pub timestamp: u64,
}

impl ApprovalRequest {
    /// Creates a new `ApprovalRequest` with the current timestamp.
    pub fn new(
        request_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: HashMap<String, serde_json::Value>,
        agent_id: impl Into<String>,
        context: impl Into<String>,
        estimated_impact: ImpactLevel,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            tool_name: tool_name.into(),
            arguments,
            agent_id: agent_id.into(),
            context: context.into(),
            estimated_impact,
            timestamp: now_secs(),
        }
    }
}

/// The decision made by a human (or automated gate) for an approval request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// The request is approved as-is.
    Approve,
    /// The request is denied with a reason.
    Deny { reason: String },
    /// The request is approved with modified arguments.
    Modify {
        modified_args: HashMap<String, serde_json::Value>,
    },
    /// The approval request timed out without a decision.
    Timeout,
}

/// Trait for approval gates that decide whether a tool call may proceed.
///
/// Named `HitlApprovalGate` to avoid collision with `ApprovalGate` in `tools.rs`.
pub trait HitlApprovalGate: Send + Sync {
    /// Evaluate the approval request and return a decision.
    fn request_approval(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, HitlError>;
    /// The human-readable name of this gate.
    fn name(&self) -> &str;
}

/// An approval gate backed by a user-supplied callback function.
pub struct CallbackApprovalGate {
    name: String,
    callback: Box<dyn Fn(&ApprovalRequest) -> ApprovalDecision + Send + Sync>,
}

impl CallbackApprovalGate {
    /// Creates a new `CallbackApprovalGate` with the given name and decision callback.
    pub fn new(
        name: impl Into<String>,
        callback: impl Fn(&ApprovalRequest) -> ApprovalDecision + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            callback: Box::new(callback),
        }
    }
}

impl HitlApprovalGate for CallbackApprovalGate {
    fn request_approval(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, HitlError> {
        Ok((self.callback)(request))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for CallbackApprovalGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackApprovalGate")
            .field("name", &self.name)
            .finish()
    }
}

/// An approval gate that always approves every request.
#[derive(Debug, Clone)]
pub struct AutoApproveGate;

impl HitlApprovalGate for AutoApproveGate {
    fn request_approval(
        &self,
        _request: &ApprovalRequest,
    ) -> Result<ApprovalDecision, HitlError> {
        Ok(ApprovalDecision::Approve)
    }

    fn name(&self) -> &str {
        "auto-approve"
    }
}

/// An approval gate that always denies every request with a fixed reason.
#[derive(Debug, Clone)]
pub struct AutoDenyGate {
    /// Reason provided for every denial.
    pub reason: String,
}

impl AutoDenyGate {
    /// Creates a new `AutoDenyGate` with the given denial reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl HitlApprovalGate for AutoDenyGate {
    fn request_approval(
        &self,
        _request: &ApprovalRequest,
    ) -> Result<ApprovalDecision, HitlError> {
        Ok(ApprovalDecision::Deny {
            reason: self.reason.clone(),
        })
    }

    fn name(&self) -> &str {
        "auto-deny"
    }
}

/// A single entry in the approval audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalLogEntry {
    /// The original approval request.
    pub request: ApprovalRequest,
    /// The decision that was made.
    pub decision: ApprovalDecision,
    /// Name of the gate that produced this decision.
    pub gate_name: String,
    /// UNIX timestamp (seconds) when the decision was recorded.
    pub timestamp: u64,
}

/// Audit log that records all approval decisions for traceability.
#[derive(Debug, Clone)]
pub struct ApprovalLog {
    entries: Vec<ApprovalLogEntry>,
    max_entries: usize,
}

impl ApprovalLog {
    /// Creates a new `ApprovalLog` that retains at most `max_entries` entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Records an entry into the log. If the log is at capacity the oldest entry
    /// is evicted (FIFO).
    pub fn record(&mut self, entry: ApprovalLogEntry) {
        if self.entries.len() >= self.max_entries && self.max_entries > 0 {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Returns all recorded entries.
    pub fn entries(&self) -> &[ApprovalLogEntry] {
        &self.entries
    }

    /// Returns entries where the tool name matches `tool_name`.
    pub fn filter_by_tool(&self, tool_name: &str) -> Vec<&ApprovalLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.request.tool_name == tool_name)
            .collect()
    }

    /// Returns entries filtered by whether they were approved or not.
    ///
    /// When `approved` is `true`, returns entries with `ApprovalDecision::Approve`.
    /// When `approved` is `false`, returns all other entries.
    pub fn filter_by_decision(&self, approved: bool) -> Vec<&ApprovalLogEntry> {
        self.entries
            .iter()
            .filter(|e| {
                let is_approve = matches!(e.decision, ApprovalDecision::Approve);
                is_approve == approved
            })
            .collect()
    }

    /// Returns the ratio of `Approve` decisions to total decisions.
    /// Returns `0.0` if the log is empty.
    pub fn approval_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let approved = self
            .entries
            .iter()
            .filter(|e| matches!(e.decision, ApprovalDecision::Approve))
            .count();
        approved as f64 / self.entries.len() as f64
    }

    /// Returns the number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all entries from the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ============================================================================
// 3.2 — Confidence-Based Escalation
// ============================================================================

/// A single confidence signal from a named source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceSignal {
    /// Human-readable identifier of the signal source.
    pub source: String,
    /// Confidence value in the range `[0.0, 1.0]`.
    pub value: f64,
    /// Importance weight for aggregation.
    pub weight: f64,
}

/// Trait for estimating overall confidence from a set of signals.
pub trait ConfidenceEstimator: Send + Sync {
    /// Estimates a single confidence score in `[0.0, 1.0]` from the given signals.
    fn estimate(&self, signals: &[ConfidenceSignal]) -> f64;
    /// Human-readable name of the estimator.
    fn name(&self) -> &str;
}

/// Estimator that computes the weighted average of all signals.
///
/// If no signals are provided or the total weight is zero, returns `0.0`.
#[derive(Debug, Clone)]
pub struct WeightedAverageEstimator;

impl ConfidenceEstimator for WeightedAverageEstimator {
    fn estimate(&self, signals: &[ConfidenceSignal]) -> f64 {
        if signals.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = signals.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let weighted_sum: f64 = signals.iter().map(|s| s.value * s.weight).sum();
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    }

    fn name(&self) -> &str {
        "weighted-average"
    }
}

/// Conservative estimator that returns the minimum signal value.
///
/// If no signals are provided, returns `0.0`.
#[derive(Debug, Clone)]
pub struct MinimumEstimator;

impl ConfidenceEstimator for MinimumEstimator {
    fn estimate(&self, signals: &[ConfidenceSignal]) -> f64 {
        if signals.is_empty() {
            return 0.0;
        }
        signals
            .iter()
            .map(|s| s.value)
            .fold(f64::INFINITY, f64::min)
            .clamp(0.0, 1.0)
    }

    fn name(&self) -> &str {
        "minimum"
    }
}

/// A condition that triggers an escalation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EscalationTrigger {
    /// Fires when the confidence score drops below the given threshold.
    ConfidenceBelow(f64),
    /// Fires when the number of consecutive errors reaches the given count.
    ConsecutiveErrors(usize),
    /// Fires when accumulated cost exceeds the given amount.
    CostAbove(f64),
    /// Fires when accumulated token usage exceeds the given count.
    TokensAbove(usize),
    /// Fires when a custom named signal crosses its threshold.
    CustomSignal { name: String, threshold: f64 },
}

/// Action to take when an escalation trigger fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EscalationAction {
    /// Continue execution without intervention.
    Continue,
    /// Pause execution and request human approval before proceeding.
    RequestApproval,
    /// Pause execution and notify a human operator.
    PauseAndNotify,
    /// Abort execution immediately with a reason.
    Abort { reason: String },
}

/// A single threshold rule in an escalation policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationThreshold {
    /// The condition that triggers this threshold.
    pub trigger: EscalationTrigger,
    /// The action to take when the trigger fires.
    pub action: EscalationAction,
}

/// A collection of escalation thresholds with a fallback default action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Ordered list of thresholds to evaluate.
    pub thresholds: Vec<EscalationThreshold>,
    /// Action to take when no threshold fires.
    pub default_action: EscalationAction,
}

/// Evaluates confidence signals and accumulated metrics against an escalation policy.
#[derive(Debug, Clone)]
pub struct EscalationEvaluator {
    policy: EscalationPolicy,
    error_count: usize,
    total_cost: f64,
    total_tokens: usize,
}

impl EscalationEvaluator {
    /// Creates a new evaluator from the given policy, with all counters at zero.
    pub fn new(policy: EscalationPolicy) -> Self {
        Self {
            policy,
            error_count: 0,
            total_cost: 0.0,
            total_tokens: 0,
        }
    }

    /// Evaluates only `ConfidenceBelow` thresholds against the given confidence score.
    /// Returns the action of the first matching threshold, or the default action.
    pub fn evaluate(&self, confidence: f64) -> EscalationAction {
        for threshold in &self.policy.thresholds {
            if let EscalationTrigger::ConfidenceBelow(min) = &threshold.trigger {
                if confidence < *min {
                    return threshold.action.clone();
                }
            }
        }
        self.policy.default_action.clone()
    }

    /// Increments the consecutive error counter.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Resets the consecutive error counter to zero.
    pub fn reset_errors(&mut self) {
        self.error_count = 0;
    }

    /// Adds to the accumulated cost.
    pub fn add_cost(&mut self, cost: f64) {
        self.total_cost += cost;
    }

    /// Adds to the accumulated token count.
    pub fn add_tokens(&mut self, tokens: usize) {
        self.total_tokens += tokens;
    }

    /// Evaluates ALL trigger types against the current state and the given
    /// confidence score. Returns the action of the first matching trigger,
    /// or the default action if none match.
    pub fn check_all_triggers(&self, confidence: f64) -> EscalationAction {
        for threshold in &self.policy.thresholds {
            let fired = match &threshold.trigger {
                EscalationTrigger::ConfidenceBelow(min) => confidence < *min,
                EscalationTrigger::ConsecutiveErrors(max) => self.error_count >= *max,
                EscalationTrigger::CostAbove(max) => self.total_cost > *max,
                EscalationTrigger::TokensAbove(max) => self.total_tokens > *max,
                EscalationTrigger::CustomSignal { .. } => {
                    // Custom signals are evaluated externally; the evaluator does
                    // not track custom signal values, so this trigger never fires
                    // from check_all_triggers alone.
                    false
                }
            };
            if fired {
                return threshold.action.clone();
            }
        }
        self.policy.default_action.clone()
    }

    /// Returns the current consecutive error count.
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Returns the accumulated total cost.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }
}

// ============================================================================
// 3.3 — Interactive Corrections
// ============================================================================

/// The kind of correction a human can apply to an agent's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CorrectionType {
    /// Replace the agent's output entirely.
    ReplaceOutput { new_output: String },
    /// Modify a specific step in the plan.
    ModifyPlan {
        step_index: usize,
        new_action: String,
    },
    /// Inject additional context for the agent to consider.
    AddContext { context: String },
    /// Skip a specific step in the plan.
    SkipStep { step_index: usize },
    /// Retry a specific step in the plan.
    RetryStep { step_index: usize },
}

/// A human correction applied to an agent's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// Unique identifier for this correction.
    pub correction_id: String,
    /// The kind of correction.
    pub correction_type: CorrectionType,
    /// Human-readable reason for the correction.
    pub reason: String,
    /// UNIX timestamp (seconds) when the correction was created.
    pub timestamp: u64,
    /// Whether this correction has been applied.
    pub applied: bool,
}

impl Correction {
    /// Creates a new unapplied `Correction` with the current timestamp.
    pub fn new(
        correction_id: impl Into<String>,
        correction_type: CorrectionType,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            correction_id: correction_id.into(),
            correction_type,
            reason: reason.into(),
            timestamp: now_secs(),
            applied: false,
        }
    }
}

/// Tracks corrections applied during an agent's execution.
#[derive(Debug, Clone)]
pub struct CorrectionHistory {
    corrections: Vec<Correction>,
    max_entries: usize,
}

impl CorrectionHistory {
    /// Creates a new `CorrectionHistory` that retains at most `max_entries` corrections.
    pub fn new(max_entries: usize) -> Self {
        Self {
            corrections: Vec::new(),
            max_entries,
        }
    }

    /// Adds a correction. If at capacity the oldest correction is evicted.
    pub fn add(&mut self, correction: Correction) {
        if self.corrections.len() >= self.max_entries && self.max_entries > 0 {
            self.corrections.remove(0);
        }
        self.corrections.push(correction);
    }

    /// Looks up a correction by its ID.
    pub fn get(&self, correction_id: &str) -> Option<&Correction> {
        self.corrections
            .iter()
            .find(|c| c.correction_id == correction_id)
    }

    /// Marks a correction as applied. Returns `true` if the correction was found
    /// and was not already applied; `false` otherwise.
    pub fn mark_applied(&mut self, correction_id: &str) -> bool {
        if let Some(c) = self
            .corrections
            .iter_mut()
            .find(|c| c.correction_id == correction_id)
        {
            if !c.applied {
                c.applied = true;
                return true;
            }
        }
        false
    }

    /// Returns all corrections.
    pub fn all(&self) -> &[Correction] {
        &self.corrections
    }

    /// Returns corrections that have NOT been applied yet.
    pub fn pending(&self) -> Vec<&Correction> {
        self.corrections.iter().filter(|c| !c.applied).collect()
    }

    /// Returns corrections that HAVE been applied.
    pub fn applied(&self) -> Vec<&Correction> {
        self.corrections.iter().filter(|c| c.applied).collect()
    }

    /// Returns the total number of corrections.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Returns `true` if there are no corrections.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }
}

// ============================================================================
// 3.4 — Declarative Approval Policies
// ============================================================================

/// A condition that determines whether a policy rule applies to a given request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PolicyCondition {
    /// Matches when the tool name equals the given string exactly.
    ToolNameMatch(String),
    /// Matches when the tool name contains the given substring.
    ToolNamePattern(String),
    /// Matches when the request's impact level is at least the given level.
    ImpactAtLeast(ImpactLevel),
    /// Matches when the estimated cost exceeds the given amount.
    CostAbove(f64),
    /// Matches when the agent ID equals the given string exactly.
    AgentIdMatch(String),
    /// Always matches.
    Always,
}

impl PolicyCondition {
    /// Evaluates whether this condition matches the given approval request.
    ///
    /// For `CostAbove`, the request's arguments are checked for a `"cost"` key
    /// with a numeric JSON value.
    pub fn matches(&self, request: &ApprovalRequest) -> bool {
        match self {
            PolicyCondition::ToolNameMatch(name) => request.tool_name == *name,
            PolicyCondition::ToolNamePattern(pattern) => request.tool_name.contains(pattern),
            PolicyCondition::ImpactAtLeast(level) => request.estimated_impact >= *level,
            PolicyCondition::CostAbove(max_cost) => {
                if let Some(cost_val) = request.arguments.get("cost") {
                    if let Some(cost) = cost_val.as_f64() {
                        return cost > *max_cost;
                    }
                }
                false
            }
            PolicyCondition::AgentIdMatch(agent_id) => request.agent_id == *agent_id,
            PolicyCondition::Always => true,
        }
    }
}

/// The action to take when a policy rule matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PolicyAction {
    /// Automatically approve the request.
    AutoApprove,
    /// Automatically deny the request with a reason.
    AutoDeny { reason: String },
    /// Require a human to approve the request.
    RequireHumanApproval,
}

/// A single rule in an approval policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Condition that must be met for this rule to apply.
    pub condition: PolicyCondition,
    /// Action to take when the condition is met.
    pub action: PolicyAction,
    /// Priority for ordering — higher priority rules are evaluated first.
    pub priority: u32,
}

/// A named collection of policy rules with a default action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalPolicy {
    /// Human-readable name for this policy.
    pub name: String,
    /// Rules evaluated in priority order (highest first).
    pub rules: Vec<PolicyRule>,
    /// Action to take when no rules match.
    pub default_action: PolicyAction,
}

/// Engine that evaluates approval requests against a set of policies.
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    policies: Vec<ApprovalPolicy>,
}

impl PolicyEngine {
    /// Creates a new empty `PolicyEngine`.
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Adds a policy to the engine.
    pub fn add_policy(&mut self, policy: ApprovalPolicy) {
        self.policies.push(policy);
    }

    /// Evaluates the request against all policies. The highest-priority matching
    /// rule across all policies wins. If no rules match, the default action of
    /// the first policy is returned. If there are no policies at all, returns
    /// `PolicyAction::RequireHumanApproval`.
    pub fn evaluate(&self, request: &ApprovalRequest) -> PolicyAction {
        let mut best_match: Option<(u32, &PolicyAction)> = None;

        for policy in &self.policies {
            for rule in &policy.rules {
                if rule.condition.matches(request) {
                    match &best_match {
                        Some((best_priority, _)) if rule.priority <= *best_priority => {}
                        _ => {
                            best_match = Some((rule.priority, &rule.action));
                        }
                    }
                }
            }
        }

        if let Some((_, action)) = best_match {
            return action.clone();
        }

        // Return the default action of the first policy, or RequireHumanApproval
        self.policies
            .first()
            .map(|p| p.default_action.clone())
            .unwrap_or(PolicyAction::RequireHumanApproval)
    }

    /// Returns all registered policies.
    pub fn policies(&self) -> &[ApprovalPolicy] {
        &self.policies
    }

    /// Removes a policy by name. Returns `true` if a policy was found and removed.
    pub fn remove_policy(&mut self, name: &str) -> bool {
        let before = self.policies.len();
        self.policies.retain(|p| p.name != name);
        self.policies.len() < before
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility for loading and serializing approval policies.
///
/// Currently supports JSON. TOML support would require the `toml` crate.
pub struct PolicyLoader;

impl PolicyLoader {
    /// Parses an `ApprovalPolicy` from a JSON string.
    pub fn from_json(json: &str) -> Result<ApprovalPolicy, AiError> {
        serde_json::from_str(json).map_err(|e| {
            AiError::Hitl(HitlError::PolicyViolation {
                policy_name: "json-parse".into(),
                reason: format!("Failed to parse policy JSON: {}", e),
            })
        })
    }

    /// Serializes an `ApprovalPolicy` to a pretty-printed JSON string.
    pub fn to_json(policy: &ApprovalPolicy) -> Result<String, AiError> {
        serde_json::to_string_pretty(policy).map_err(|e| {
            AiError::Hitl(HitlError::PolicyViolation {
                policy_name: policy.name.clone(),
                reason: format!("Failed to serialize policy to JSON: {}", e),
            })
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper ---------------------------------------------------------------

    fn make_request(tool_name: &str) -> ApprovalRequest {
        ApprovalRequest {
            request_id: "req-001".into(),
            tool_name: tool_name.into(),
            arguments: HashMap::new(),
            agent_id: "agent-1".into(),
            context: "test context".into(),
            estimated_impact: ImpactLevel::Medium,
            timestamp: 1_700_000_000,
        }
    }

    fn make_request_with_impact(tool_name: &str, impact: ImpactLevel) -> ApprovalRequest {
        ApprovalRequest {
            request_id: "req-002".into(),
            tool_name: tool_name.into(),
            arguments: HashMap::new(),
            agent_id: "agent-1".into(),
            context: "test context".into(),
            estimated_impact: impact,
            timestamp: 1_700_000_000,
        }
    }

    fn make_log_entry(
        tool_name: &str,
        decision: ApprovalDecision,
        gate_name: &str,
    ) -> ApprovalLogEntry {
        ApprovalLogEntry {
            request: make_request(tool_name),
            decision,
            gate_name: gate_name.into(),
            timestamp: now_secs(),
        }
    }

    // =========================================================================
    // 3.1 — Tool Approval Gates
    // =========================================================================

    #[test]
    fn test_auto_approve_gate_always_approves() {
        let gate = AutoApproveGate;
        let req = make_request("read_file");
        let decision = gate.request_approval(&req).expect("should not error");
        assert_eq!(decision, ApprovalDecision::Approve);
    }

    #[test]
    fn test_auto_approve_gate_name() {
        let gate = AutoApproveGate;
        assert_eq!(gate.name(), "auto-approve");
    }

    #[test]
    fn test_auto_deny_gate_always_denies() {
        let gate = AutoDenyGate::new("forbidden");
        let req = make_request("delete_db");
        let decision = gate.request_approval(&req).expect("should not error");
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                reason: "forbidden".into()
            }
        );
    }

    #[test]
    fn test_auto_deny_gate_name() {
        let gate = AutoDenyGate::new("reason");
        assert_eq!(gate.name(), "auto-deny");
    }

    #[test]
    fn test_callback_gate_delegates_to_callback() {
        let gate = CallbackApprovalGate::new("my-gate", |req: &ApprovalRequest| {
            if req.tool_name == "safe_tool" {
                ApprovalDecision::Approve
            } else {
                ApprovalDecision::Deny {
                    reason: "not safe".into(),
                }
            }
        });

        let safe_req = make_request("safe_tool");
        assert_eq!(
            gate.request_approval(&safe_req).unwrap(),
            ApprovalDecision::Approve
        );

        let unsafe_req = make_request("dangerous_tool");
        assert_eq!(
            gate.request_approval(&unsafe_req).unwrap(),
            ApprovalDecision::Deny {
                reason: "not safe".into()
            }
        );
    }

    #[test]
    fn test_callback_gate_name() {
        let gate = CallbackApprovalGate::new("custom", |_| ApprovalDecision::Approve);
        assert_eq!(gate.name(), "custom");
    }

    #[test]
    fn test_callback_gate_modify_decision() {
        let gate = CallbackApprovalGate::new("modifier", |_req: &ApprovalRequest| {
            let mut modified = HashMap::new();
            modified.insert("limit".into(), serde_json::json!(100));
            ApprovalDecision::Modify {
                modified_args: modified,
            }
        });
        let req = make_request("query");
        let decision = gate.request_approval(&req).unwrap();
        match decision {
            ApprovalDecision::Modify { modified_args } => {
                assert_eq!(modified_args.get("limit"), Some(&serde_json::json!(100)));
            }
            other => panic!("Expected Modify, got {:?}", other),
        }
    }

    #[test]
    fn test_approval_request_new() {
        let mut args = HashMap::new();
        args.insert("path".into(), serde_json::json!("/tmp/file"));
        let req = ApprovalRequest::new("r-1", "write_file", args, "agent-x", "writing", ImpactLevel::High);
        assert_eq!(req.request_id, "r-1");
        assert_eq!(req.tool_name, "write_file");
        assert_eq!(req.agent_id, "agent-x");
        assert_eq!(req.context, "writing");
        assert_eq!(req.estimated_impact, ImpactLevel::High);
        assert!(req.timestamp > 0);
        assert_eq!(
            req.arguments.get("path"),
            Some(&serde_json::json!("/tmp/file"))
        );
    }

    #[test]
    fn test_approval_decision_all_variants() {
        let approve = ApprovalDecision::Approve;
        assert_eq!(approve, ApprovalDecision::Approve);

        let deny = ApprovalDecision::Deny {
            reason: "nope".into(),
        };
        assert_eq!(
            deny,
            ApprovalDecision::Deny {
                reason: "nope".into()
            }
        );

        let mut args = HashMap::new();
        args.insert("k".into(), serde_json::json!(42));
        let modify = ApprovalDecision::Modify {
            modified_args: args.clone(),
        };
        assert_eq!(
            modify,
            ApprovalDecision::Modify {
                modified_args: args
            }
        );

        let timeout = ApprovalDecision::Timeout;
        assert_eq!(timeout, ApprovalDecision::Timeout);
    }

    #[test]
    fn test_impact_level_ordering() {
        assert!(ImpactLevel::Low < ImpactLevel::Medium);
        assert!(ImpactLevel::Medium < ImpactLevel::High);
        assert!(ImpactLevel::High < ImpactLevel::Critical);
        assert!(ImpactLevel::Low < ImpactLevel::Critical);
    }

    #[test]
    fn test_impact_level_equality() {
        assert_eq!(ImpactLevel::Low, ImpactLevel::Low);
        assert_eq!(ImpactLevel::Critical, ImpactLevel::Critical);
        assert_ne!(ImpactLevel::Low, ImpactLevel::High);
    }

    #[test]
    fn test_impact_level_rank() {
        assert_eq!(ImpactLevel::Low.rank(), 0);
        assert_eq!(ImpactLevel::Medium.rank(), 1);
        assert_eq!(ImpactLevel::High.rank(), 2);
        assert_eq!(ImpactLevel::Critical.rank(), 3);
    }

    // -- ApprovalLog ----------------------------------------------------------

    #[test]
    fn test_approval_log_new_is_empty() {
        let log = ApprovalLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_approval_log_record_and_entries() {
        let mut log = ApprovalLog::new(100);
        log.record(make_log_entry("tool_a", ApprovalDecision::Approve, "gate1"));
        log.record(make_log_entry(
            "tool_b",
            ApprovalDecision::Deny {
                reason: "no".into(),
            },
            "gate1",
        ));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
        assert_eq!(log.entries().len(), 2);
    }

    #[test]
    fn test_approval_log_eviction() {
        let mut log = ApprovalLog::new(2);
        log.record(make_log_entry("first", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("second", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("third", ApprovalDecision::Approve, "g"));
        assert_eq!(log.len(), 2);
        // The first entry should have been evicted
        assert_eq!(log.entries()[0].request.tool_name, "second");
        assert_eq!(log.entries()[1].request.tool_name, "third");
    }

    #[test]
    fn test_approval_log_filter_by_tool() {
        let mut log = ApprovalLog::new(100);
        log.record(make_log_entry("read", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("write", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("read", ApprovalDecision::Timeout, "g"));
        let reads = log.filter_by_tool("read");
        assert_eq!(reads.len(), 2);
        let writes = log.filter_by_tool("write");
        assert_eq!(writes.len(), 1);
        let deletes = log.filter_by_tool("delete");
        assert_eq!(deletes.len(), 0);
    }

    #[test]
    fn test_approval_log_filter_by_decision() {
        let mut log = ApprovalLog::new(100);
        log.record(make_log_entry("a", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry(
            "b",
            ApprovalDecision::Deny {
                reason: "x".into(),
            },
            "g",
        ));
        log.record(make_log_entry("c", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("d", ApprovalDecision::Timeout, "g"));

        let approved = log.filter_by_decision(true);
        assert_eq!(approved.len(), 2);

        let not_approved = log.filter_by_decision(false);
        assert_eq!(not_approved.len(), 2);
    }

    #[test]
    fn test_approval_log_approval_rate() {
        let mut log = ApprovalLog::new(100);
        assert_eq!(log.approval_rate(), 0.0);

        log.record(make_log_entry("a", ApprovalDecision::Approve, "g"));
        assert!((log.approval_rate() - 1.0).abs() < f64::EPSILON);

        log.record(make_log_entry(
            "b",
            ApprovalDecision::Deny {
                reason: "x".into(),
            },
            "g",
        ));
        assert!((log.approval_rate() - 0.5).abs() < f64::EPSILON);

        log.record(make_log_entry("c", ApprovalDecision::Approve, "g"));
        let rate = log.approval_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_approval_log_clear() {
        let mut log = ApprovalLog::new(100);
        log.record(make_log_entry("a", ApprovalDecision::Approve, "g"));
        log.record(make_log_entry("b", ApprovalDecision::Approve, "g"));
        assert_eq!(log.len(), 2);
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    // =========================================================================
    // 3.2 — Confidence-Based Escalation
    // =========================================================================

    #[test]
    fn test_weighted_average_estimator_single_signal() {
        let est = WeightedAverageEstimator;
        let signals = vec![ConfidenceSignal {
            source: "llm".into(),
            value: 0.8,
            weight: 1.0,
        }];
        let c = est.estimate(&signals);
        assert!((c - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_average_estimator_multiple_signals() {
        let est = WeightedAverageEstimator;
        let signals = vec![
            ConfidenceSignal {
                source: "llm".into(),
                value: 0.9,
                weight: 2.0,
            },
            ConfidenceSignal {
                source: "validator".into(),
                value: 0.3,
                weight: 1.0,
            },
        ];
        // Expected: (0.9*2 + 0.3*1) / (2+1) = 2.1/3 = 0.7
        let c = est.estimate(&signals);
        assert!((c - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_average_estimator_empty_signals() {
        let est = WeightedAverageEstimator;
        assert_eq!(est.estimate(&[]), 0.0);
    }

    #[test]
    fn test_weighted_average_estimator_zero_weight() {
        let est = WeightedAverageEstimator;
        let signals = vec![ConfidenceSignal {
            source: "x".into(),
            value: 0.5,
            weight: 0.0,
        }];
        assert_eq!(est.estimate(&signals), 0.0);
    }

    #[test]
    fn test_weighted_average_estimator_name() {
        let est = WeightedAverageEstimator;
        assert_eq!(est.name(), "weighted-average");
    }

    #[test]
    fn test_minimum_estimator_returns_lowest() {
        let est = MinimumEstimator;
        let signals = vec![
            ConfidenceSignal {
                source: "a".into(),
                value: 0.9,
                weight: 1.0,
            },
            ConfidenceSignal {
                source: "b".into(),
                value: 0.2,
                weight: 1.0,
            },
            ConfidenceSignal {
                source: "c".into(),
                value: 0.5,
                weight: 1.0,
            },
        ];
        let c = est.estimate(&signals);
        assert!((c - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_minimum_estimator_empty_signals() {
        let est = MinimumEstimator;
        assert_eq!(est.estimate(&[]), 0.0);
    }

    #[test]
    fn test_minimum_estimator_single_signal() {
        let est = MinimumEstimator;
        let signals = vec![ConfidenceSignal {
            source: "x".into(),
            value: 0.7,
            weight: 1.0,
        }];
        assert!((est.estimate(&signals) - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_minimum_estimator_name() {
        let est = MinimumEstimator;
        assert_eq!(est.name(), "minimum");
    }

    #[test]
    fn test_escalation_evaluate_confidence_below() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::ConfidenceBelow(0.5),
                action: EscalationAction::RequestApproval,
            }],
            default_action: EscalationAction::Continue,
        };
        let evaluator = EscalationEvaluator::new(policy);
        // Confidence 0.3 < 0.5 => RequestApproval
        assert_eq!(evaluator.evaluate(0.3), EscalationAction::RequestApproval);
        // Confidence 0.7 >= 0.5 => Continue (default)
        assert_eq!(evaluator.evaluate(0.7), EscalationAction::Continue);
    }

    #[test]
    fn test_escalation_evaluate_returns_default_when_no_match() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::ConfidenceBelow(0.1),
                action: EscalationAction::Abort {
                    reason: "too low".into(),
                },
            }],
            default_action: EscalationAction::Continue,
        };
        let evaluator = EscalationEvaluator::new(policy);
        assert_eq!(evaluator.evaluate(0.5), EscalationAction::Continue);
    }

    #[test]
    fn test_escalation_record_error_and_consecutive_errors() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::ConsecutiveErrors(3),
                action: EscalationAction::PauseAndNotify,
            }],
            default_action: EscalationAction::Continue,
        };
        let mut evaluator = EscalationEvaluator::new(policy);
        assert_eq!(evaluator.error_count(), 0);

        evaluator.record_error();
        evaluator.record_error();
        assert_eq!(evaluator.error_count(), 2);
        // 2 errors, threshold is 3 => Continue
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::Continue
        );

        evaluator.record_error();
        assert_eq!(evaluator.error_count(), 3);
        // 3 errors >= 3 => PauseAndNotify
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::PauseAndNotify
        );
    }

    #[test]
    fn test_escalation_reset_errors() {
        let policy = EscalationPolicy {
            thresholds: vec![],
            default_action: EscalationAction::Continue,
        };
        let mut evaluator = EscalationEvaluator::new(policy);
        evaluator.record_error();
        evaluator.record_error();
        assert_eq!(evaluator.error_count(), 2);
        evaluator.reset_errors();
        assert_eq!(evaluator.error_count(), 0);
    }

    #[test]
    fn test_escalation_add_cost_and_cost_above() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::CostAbove(10.0),
                action: EscalationAction::Abort {
                    reason: "budget exceeded".into(),
                },
            }],
            default_action: EscalationAction::Continue,
        };
        let mut evaluator = EscalationEvaluator::new(policy);
        assert!((evaluator.total_cost() - 0.0).abs() < f64::EPSILON);

        evaluator.add_cost(5.0);
        assert!((evaluator.total_cost() - 5.0).abs() < f64::EPSILON);
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::Continue
        );

        evaluator.add_cost(6.0);
        assert!((evaluator.total_cost() - 11.0).abs() < f64::EPSILON);
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::Abort {
                reason: "budget exceeded".into()
            }
        );
    }

    #[test]
    fn test_escalation_add_tokens_and_tokens_above() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::TokensAbove(1000),
                action: EscalationAction::RequestApproval,
            }],
            default_action: EscalationAction::Continue,
        };
        let mut evaluator = EscalationEvaluator::new(policy);
        evaluator.add_tokens(500);
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::Continue
        );
        evaluator.add_tokens(600);
        assert_eq!(
            evaluator.check_all_triggers(1.0),
            EscalationAction::RequestApproval
        );
    }

    #[test]
    fn test_escalation_check_all_triggers_multiple() {
        let policy = EscalationPolicy {
            thresholds: vec![
                EscalationThreshold {
                    trigger: EscalationTrigger::ConfidenceBelow(0.3),
                    action: EscalationAction::Abort {
                        reason: "low confidence".into(),
                    },
                },
                EscalationThreshold {
                    trigger: EscalationTrigger::ConsecutiveErrors(2),
                    action: EscalationAction::PauseAndNotify,
                },
                EscalationThreshold {
                    trigger: EscalationTrigger::CostAbove(50.0),
                    action: EscalationAction::RequestApproval,
                },
            ],
            default_action: EscalationAction::Continue,
        };
        let mut evaluator = EscalationEvaluator::new(policy);

        // Nothing triggered
        assert_eq!(
            evaluator.check_all_triggers(0.8),
            EscalationAction::Continue
        );

        // Confidence low => first trigger fires
        assert_eq!(
            evaluator.check_all_triggers(0.1),
            EscalationAction::Abort {
                reason: "low confidence".into()
            }
        );

        // Errors triggered, but confidence OK
        evaluator.record_error();
        evaluator.record_error();
        assert_eq!(
            evaluator.check_all_triggers(0.8),
            EscalationAction::PauseAndNotify
        );
    }

    #[test]
    fn test_escalation_custom_signal_does_not_fire() {
        let policy = EscalationPolicy {
            thresholds: vec![EscalationThreshold {
                trigger: EscalationTrigger::CustomSignal {
                    name: "latency".into(),
                    threshold: 1000.0,
                },
                action: EscalationAction::PauseAndNotify,
            }],
            default_action: EscalationAction::Continue,
        };
        let evaluator = EscalationEvaluator::new(policy);
        // Custom signals don't fire from check_all_triggers
        assert_eq!(
            evaluator.check_all_triggers(0.5),
            EscalationAction::Continue
        );
    }

    // =========================================================================
    // 3.3 — Interactive Corrections
    // =========================================================================

    #[test]
    fn test_correction_new() {
        let c = Correction::new(
            "c-1",
            CorrectionType::ReplaceOutput {
                new_output: "fixed".into(),
            },
            "wrong output",
        );
        assert_eq!(c.correction_id, "c-1");
        assert_eq!(c.reason, "wrong output");
        assert!(!c.applied);
        assert!(c.timestamp > 0);
    }

    #[test]
    fn test_correction_type_replace_output() {
        let ct = CorrectionType::ReplaceOutput {
            new_output: "new".into(),
        };
        match ct {
            CorrectionType::ReplaceOutput { new_output } => assert_eq!(new_output, "new"),
            _ => panic!("Expected ReplaceOutput"),
        }
    }

    #[test]
    fn test_correction_type_modify_plan() {
        let ct = CorrectionType::ModifyPlan {
            step_index: 2,
            new_action: "search instead".into(),
        };
        match ct {
            CorrectionType::ModifyPlan {
                step_index,
                new_action,
            } => {
                assert_eq!(step_index, 2);
                assert_eq!(new_action, "search instead");
            }
            _ => panic!("Expected ModifyPlan"),
        }
    }

    #[test]
    fn test_correction_type_add_context() {
        let ct = CorrectionType::AddContext {
            context: "extra info".into(),
        };
        match ct {
            CorrectionType::AddContext { context } => assert_eq!(context, "extra info"),
            _ => panic!("Expected AddContext"),
        }
    }

    #[test]
    fn test_correction_type_skip_step() {
        let ct = CorrectionType::SkipStep { step_index: 5 };
        match ct {
            CorrectionType::SkipStep { step_index } => assert_eq!(step_index, 5),
            _ => panic!("Expected SkipStep"),
        }
    }

    #[test]
    fn test_correction_type_retry_step() {
        let ct = CorrectionType::RetryStep { step_index: 3 };
        match ct {
            CorrectionType::RetryStep { step_index } => assert_eq!(step_index, 3),
            _ => panic!("Expected RetryStep"),
        }
    }

    #[test]
    fn test_correction_history_new_is_empty() {
        let history = CorrectionHistory::new(50);
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_correction_history_add_and_get() {
        let mut history = CorrectionHistory::new(50);
        let c = Correction::new(
            "c-1",
            CorrectionType::AddContext {
                context: "info".into(),
            },
            "need context",
        );
        history.add(c);
        assert_eq!(history.len(), 1);
        let found = history.get("c-1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().correction_id, "c-1");
    }

    #[test]
    fn test_correction_history_get_not_found() {
        let history = CorrectionHistory::new(50);
        assert!(history.get("nonexistent").is_none());
    }

    #[test]
    fn test_correction_history_mark_applied() {
        let mut history = CorrectionHistory::new(50);
        history.add(Correction::new(
            "c-1",
            CorrectionType::SkipStep { step_index: 0 },
            "skip it",
        ));
        assert!(history.mark_applied("c-1"));
        assert!(history.get("c-1").unwrap().applied);
        // marking again returns false (already applied)
        assert!(!history.mark_applied("c-1"));
    }

    #[test]
    fn test_correction_history_mark_applied_not_found() {
        let mut history = CorrectionHistory::new(50);
        assert!(!history.mark_applied("nonexistent"));
    }

    #[test]
    fn test_correction_history_pending_and_applied() {
        let mut history = CorrectionHistory::new(50);
        history.add(Correction::new(
            "c-1",
            CorrectionType::RetryStep { step_index: 0 },
            "retry",
        ));
        history.add(Correction::new(
            "c-2",
            CorrectionType::SkipStep { step_index: 1 },
            "skip",
        ));
        history.add(Correction::new(
            "c-3",
            CorrectionType::AddContext {
                context: "more".into(),
            },
            "context",
        ));

        assert_eq!(history.pending().len(), 3);
        assert_eq!(history.applied().len(), 0);

        history.mark_applied("c-2");
        assert_eq!(history.pending().len(), 2);
        assert_eq!(history.applied().len(), 1);
        assert_eq!(history.applied()[0].correction_id, "c-2");
    }

    #[test]
    fn test_correction_history_all() {
        let mut history = CorrectionHistory::new(50);
        history.add(Correction::new(
            "c-1",
            CorrectionType::ReplaceOutput {
                new_output: "x".into(),
            },
            "r",
        ));
        history.add(Correction::new(
            "c-2",
            CorrectionType::ReplaceOutput {
                new_output: "y".into(),
            },
            "r",
        ));
        assert_eq!(history.all().len(), 2);
    }

    #[test]
    fn test_correction_history_eviction() {
        let mut history = CorrectionHistory::new(2);
        history.add(Correction::new(
            "c-1",
            CorrectionType::SkipStep { step_index: 0 },
            "r",
        ));
        history.add(Correction::new(
            "c-2",
            CorrectionType::SkipStep { step_index: 1 },
            "r",
        ));
        history.add(Correction::new(
            "c-3",
            CorrectionType::SkipStep { step_index: 2 },
            "r",
        ));
        assert_eq!(history.len(), 2);
        assert!(history.get("c-1").is_none());
        assert!(history.get("c-2").is_some());
        assert!(history.get("c-3").is_some());
    }

    // =========================================================================
    // 3.4 — Declarative Approval Policies
    // =========================================================================

    #[test]
    fn test_policy_condition_tool_name_match() {
        let cond = PolicyCondition::ToolNameMatch("delete_file".into());
        let req = make_request("delete_file");
        assert!(cond.matches(&req));

        let req2 = make_request("read_file");
        assert!(!cond.matches(&req2));
    }

    #[test]
    fn test_policy_condition_tool_name_pattern() {
        let cond = PolicyCondition::ToolNamePattern("delete".into());
        assert!(cond.matches(&make_request("delete_file")));
        assert!(cond.matches(&make_request("bulk_delete")));
        assert!(!cond.matches(&make_request("read_file")));
    }

    #[test]
    fn test_policy_condition_impact_at_least() {
        let cond = PolicyCondition::ImpactAtLeast(ImpactLevel::High);
        assert!(cond.matches(&make_request_with_impact("t", ImpactLevel::High)));
        assert!(cond.matches(&make_request_with_impact("t", ImpactLevel::Critical)));
        assert!(!cond.matches(&make_request_with_impact("t", ImpactLevel::Medium)));
        assert!(!cond.matches(&make_request_with_impact("t", ImpactLevel::Low)));
    }

    #[test]
    fn test_policy_condition_cost_above() {
        let cond = PolicyCondition::CostAbove(100.0);
        let mut req = make_request("tool");
        req.arguments
            .insert("cost".into(), serde_json::json!(150.0));
        assert!(cond.matches(&req));

        let mut req2 = make_request("tool");
        req2.arguments
            .insert("cost".into(), serde_json::json!(50.0));
        assert!(!cond.matches(&req2));

        // No cost argument => does not match
        let req3 = make_request("tool");
        assert!(!cond.matches(&req3));
    }

    #[test]
    fn test_policy_condition_agent_id_match() {
        let cond = PolicyCondition::AgentIdMatch("agent-1".into());
        assert!(cond.matches(&make_request("t")));

        let mut req = make_request("t");
        req.agent_id = "agent-2".into();
        assert!(!cond.matches(&req));
    }

    #[test]
    fn test_policy_condition_always() {
        let cond = PolicyCondition::Always;
        assert!(cond.matches(&make_request("anything")));
    }

    #[test]
    fn test_policy_engine_new_is_empty() {
        let engine = PolicyEngine::new();
        assert!(engine.policies().is_empty());
    }

    #[test]
    fn test_policy_engine_add_and_remove_policy() {
        let mut engine = PolicyEngine::new();
        let policy = ApprovalPolicy {
            name: "test-policy".into(),
            rules: vec![],
            default_action: PolicyAction::AutoApprove,
        };
        engine.add_policy(policy);
        assert_eq!(engine.policies().len(), 1);

        assert!(engine.remove_policy("test-policy"));
        assert!(engine.policies().is_empty());
    }

    #[test]
    fn test_policy_engine_remove_nonexistent() {
        let mut engine = PolicyEngine::new();
        assert!(!engine.remove_policy("nope"));
    }

    #[test]
    fn test_policy_engine_evaluate_highest_priority_wins() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(ApprovalPolicy {
            name: "p1".into(),
            rules: vec![
                PolicyRule {
                    name: "low-priority".into(),
                    condition: PolicyCondition::Always,
                    action: PolicyAction::AutoApprove,
                    priority: 1,
                },
                PolicyRule {
                    name: "high-priority".into(),
                    condition: PolicyCondition::Always,
                    action: PolicyAction::RequireHumanApproval,
                    priority: 10,
                },
            ],
            default_action: PolicyAction::AutoApprove,
        });

        let result = engine.evaluate(&make_request("tool"));
        assert_eq!(result, PolicyAction::RequireHumanApproval);
    }

    #[test]
    fn test_policy_engine_evaluate_default_when_no_match() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(ApprovalPolicy {
            name: "p1".into(),
            rules: vec![PolicyRule {
                name: "specific".into(),
                condition: PolicyCondition::ToolNameMatch("deploy".into()),
                action: PolicyAction::RequireHumanApproval,
                priority: 10,
            }],
            default_action: PolicyAction::AutoApprove,
        });

        // "read" does not match "deploy" => default
        let result = engine.evaluate(&make_request("read"));
        assert_eq!(result, PolicyAction::AutoApprove);
    }

    #[test]
    fn test_policy_engine_evaluate_no_policies() {
        let engine = PolicyEngine::new();
        let result = engine.evaluate(&make_request("tool"));
        assert_eq!(result, PolicyAction::RequireHumanApproval);
    }

    #[test]
    fn test_policy_engine_evaluate_across_multiple_policies() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(ApprovalPolicy {
            name: "p1".into(),
            rules: vec![PolicyRule {
                name: "r1".into(),
                condition: PolicyCondition::ToolNameMatch("safe".into()),
                action: PolicyAction::AutoApprove,
                priority: 5,
            }],
            default_action: PolicyAction::RequireHumanApproval,
        });
        engine.add_policy(ApprovalPolicy {
            name: "p2".into(),
            rules: vec![PolicyRule {
                name: "r2".into(),
                condition: PolicyCondition::Always,
                action: PolicyAction::AutoDeny {
                    reason: "denied by p2".into(),
                },
                priority: 20,
            }],
            default_action: PolicyAction::AutoApprove,
        });

        // r2 has priority 20 > r1 priority 5, and Always matches => AutoDeny wins
        let result = engine.evaluate(&make_request("safe"));
        assert_eq!(
            result,
            PolicyAction::AutoDeny {
                reason: "denied by p2".into()
            }
        );
    }

    // -- PolicyLoader ---------------------------------------------------------

    #[test]
    fn test_policy_loader_from_json() {
        let json = r#"{
            "name": "test-policy",
            "rules": [
                {
                    "name": "deny-delete",
                    "condition": {"ToolNameMatch": "delete"},
                    "action": {"AutoDeny": {"reason": "no deletions"}},
                    "priority": 10
                }
            ],
            "default_action": "AutoApprove"
        }"#;
        let policy = PolicyLoader::from_json(json).expect("should parse");
        assert_eq!(policy.name, "test-policy");
        assert_eq!(policy.rules.len(), 1);
        assert_eq!(policy.rules[0].name, "deny-delete");
        assert_eq!(policy.rules[0].priority, 10);
        assert_eq!(policy.default_action, PolicyAction::AutoApprove);
    }

    #[test]
    fn test_policy_loader_from_json_invalid() {
        let result = PolicyLoader::from_json("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_loader_to_json() {
        let policy = ApprovalPolicy {
            name: "roundtrip".into(),
            rules: vec![PolicyRule {
                name: "always-approve".into(),
                condition: PolicyCondition::Always,
                action: PolicyAction::AutoApprove,
                priority: 1,
            }],
            default_action: PolicyAction::RequireHumanApproval,
        };
        let json = PolicyLoader::to_json(&policy).expect("should serialize");
        assert!(json.contains("roundtrip"));
        assert!(json.contains("always-approve"));

        // Roundtrip: parse back and compare
        let restored = PolicyLoader::from_json(&json).expect("should parse back");
        assert_eq!(restored.name, policy.name);
        assert_eq!(restored.rules.len(), policy.rules.len());
        assert_eq!(restored.default_action, policy.default_action);
    }

    #[test]
    fn test_policy_loader_roundtrip_complex() {
        let policy = ApprovalPolicy {
            name: "complex".into(),
            rules: vec![
                PolicyRule {
                    name: "pattern-rule".into(),
                    condition: PolicyCondition::ToolNamePattern("delete".into()),
                    action: PolicyAction::RequireHumanApproval,
                    priority: 100,
                },
                PolicyRule {
                    name: "impact-rule".into(),
                    condition: PolicyCondition::ImpactAtLeast(ImpactLevel::Critical),
                    action: PolicyAction::AutoDeny {
                        reason: "critical impact".into(),
                    },
                    priority: 200,
                },
                PolicyRule {
                    name: "cost-rule".into(),
                    condition: PolicyCondition::CostAbove(1000.0),
                    action: PolicyAction::RequireHumanApproval,
                    priority: 50,
                },
            ],
            default_action: PolicyAction::AutoApprove,
        };
        let json = PolicyLoader::to_json(&policy).expect("serialize");
        let restored = PolicyLoader::from_json(&json).expect("parse");
        assert_eq!(restored.rules.len(), 3);
        assert_eq!(restored.name, "complex");
    }

    // =========================================================================
    // Serialization round-trips for data types
    // =========================================================================

    #[test]
    fn test_approval_decision_serde_roundtrip() {
        let decisions = vec![
            ApprovalDecision::Approve,
            ApprovalDecision::Deny {
                reason: "bad".into(),
            },
            ApprovalDecision::Modify {
                modified_args: {
                    let mut m = HashMap::new();
                    m.insert("k".into(), serde_json::json!("v"));
                    m
                },
            },
            ApprovalDecision::Timeout,
        ];
        for d in &decisions {
            let json = serde_json::to_string(d).expect("serialize");
            let restored: ApprovalDecision = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&restored, d);
        }
    }

    #[test]
    fn test_impact_level_serde_roundtrip() {
        let levels = vec![
            ImpactLevel::Low,
            ImpactLevel::Medium,
            ImpactLevel::High,
            ImpactLevel::Critical,
        ];
        for level in &levels {
            let json = serde_json::to_string(level).expect("serialize");
            let restored: ImpactLevel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&restored, level);
        }
    }

    #[test]
    fn test_correction_serde_roundtrip() {
        let c = Correction::new(
            "c-serde",
            CorrectionType::ModifyPlan {
                step_index: 1,
                new_action: "revised".into(),
            },
            "plan was wrong",
        );
        let json = serde_json::to_string(&c).expect("serialize");
        let restored: Correction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.correction_id, "c-serde");
        assert_eq!(restored.reason, "plan was wrong");
    }

    #[test]
    fn test_escalation_trigger_serde_roundtrip() {
        let triggers = vec![
            EscalationTrigger::ConfidenceBelow(0.5),
            EscalationTrigger::ConsecutiveErrors(3),
            EscalationTrigger::CostAbove(100.0),
            EscalationTrigger::TokensAbove(5000),
            EscalationTrigger::CustomSignal {
                name: "latency".into(),
                threshold: 500.0,
            },
        ];
        for trigger in &triggers {
            let json = serde_json::to_string(trigger).expect("serialize");
            let _restored: EscalationTrigger = serde_json::from_str(&json).expect("deserialize");
        }
    }

    #[test]
    fn test_escalation_action_serde_roundtrip() {
        let actions = vec![
            EscalationAction::Continue,
            EscalationAction::RequestApproval,
            EscalationAction::PauseAndNotify,
            EscalationAction::Abort {
                reason: "done".into(),
            },
        ];
        for action in &actions {
            let json = serde_json::to_string(action).expect("serialize");
            let restored: EscalationAction = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&restored, action);
        }
    }
}
