//! Agent sandbox — action validation and audit trail
//!
//! Validates every agent action against an [`AgentPolicy`] before execution.
//! Maintains a full audit log of all decisions.

use crate::agent_policy::{ActionDescriptor, ActionType, AgentPolicy, ApprovalHandler, RiskLevel};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Types
// ============================================================================

/// Decision made for an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditDecision {
    /// Allowed by policy without needing approval.
    Approved,
    /// Denied by policy.
    Denied,
    /// Required approval and user approved.
    ApprovedByUser,
    /// Required approval and user denied.
    DeniedByUser,
}

/// A single entry in the audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub action: ActionDescriptor,
    pub decision: AuditDecision,
    pub risk: RiskLevel,
    pub reason: Option<String>,
}

/// Error type for sandbox validation.
#[derive(Debug, Clone)]
pub struct SandboxError {
    pub message: String,
    pub action: ActionDescriptor,
    pub risk: RiskLevel,
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sandbox denied {:?} ({}): {}",
            self.action.action_type, self.action.target, self.message
        )
    }
}

// ============================================================================
// SandboxValidator
// ============================================================================

/// Validates agent actions against a policy and maintains an audit trail.
pub struct SandboxValidator {
    policy: AgentPolicy,
    audit_log: Vec<AuditEntry>,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
    total_cost: f64,
    start_time: u64,
}

impl SandboxValidator {
    /// Create a new sandbox with the given policy.
    pub fn new(policy: AgentPolicy) -> Self {
        Self {
            policy,
            audit_log: Vec::new(),
            approval_handler: None,
            total_cost: 0.0,
            start_time: now_millis(),
        }
    }

    /// Create a sandbox with a policy and approval handler.
    pub fn with_approval(policy: AgentPolicy, handler: Arc<dyn ApprovalHandler>) -> Self {
        Self {
            policy,
            audit_log: Vec::new(),
            approval_handler: Some(handler),
            total_cost: 0.0,
            start_time: now_millis(),
        }
    }

    /// Set the approval handler.
    pub fn set_approval_handler(&mut self, handler: Arc<dyn ApprovalHandler>) {
        self.approval_handler = Some(handler);
    }

    /// Get the current policy.
    pub fn policy(&self) -> &AgentPolicy {
        &self.policy
    }

    /// Update the policy.
    pub fn set_policy(&mut self, policy: AgentPolicy) {
        self.policy = policy;
    }

    /// Validate an action against the policy.
    /// Returns Ok(()) if allowed, Err(SandboxError) if denied.
    pub fn validate(&mut self, action: &ActionDescriptor) -> Result<(), SandboxError> {
        let risk = self.policy.assess_risk(action);

        // Check runtime limits
        if let Err(msg) = self.check_limits() {
            self.log_entry(action, AuditDecision::Denied, risk, Some(msg.clone()));
            return Err(SandboxError {
                message: msg,
                action: action.clone(),
                risk,
            });
        }

        // Validate against policy
        match self
            .policy
            .validate_action(action, self.approval_handler.as_ref())
        {
            Ok(()) => {
                // Determine if approval was needed
                let decision = if self.policy.needs_approval(action) {
                    AuditDecision::ApprovedByUser
                } else {
                    AuditDecision::Approved
                };
                self.log_entry(action, decision, risk, None);
                Ok(())
            }
            Err(reason) => {
                let decision = if reason.contains("denied by user") {
                    AuditDecision::DeniedByUser
                } else {
                    AuditDecision::Denied
                };
                self.log_entry(action, decision, risk, Some(reason.clone()));
                Err(SandboxError {
                    message: reason,
                    action: action.clone(),
                    risk,
                })
            }
        }
    }

    /// Validate a file read.
    pub fn validate_file_read(&mut self, path: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::FileRead, path))
    }

    /// Validate a file write.
    pub fn validate_file_write(&mut self, path: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::FileWrite, path))
    }

    /// Validate a shell command.
    pub fn validate_command(&mut self, cmd: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::ShellExec, cmd))
    }

    /// Validate an HTTP request.
    pub fn validate_url(&mut self, url: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::HttpRequest, url))
    }

    /// Validate an MCP call.
    pub fn validate_mcp(&mut self, server: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::McpCall, server))
    }

    /// Validate a tool call.
    pub fn validate_tool(&mut self, tool_name: &str) -> Result<(), SandboxError> {
        self.validate(&ActionDescriptor::new(ActionType::ToolCall, tool_name))
    }

    /// Record cost for budget tracking.
    pub fn record_cost(&mut self, cost: f64) {
        self.total_cost += cost;
    }

    /// Get total cost so far.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Check if within budget and time limits.
    fn check_limits(&self) -> Result<(), String> {
        if self.total_cost > self.policy.max_cost_usd {
            return Err(format!(
                "Cost limit exceeded: ${:.2} > ${:.2}",
                self.total_cost, self.policy.max_cost_usd
            ));
        }
        let elapsed_secs = (now_millis() - self.start_time) / 1000;
        if elapsed_secs > self.policy.max_runtime_secs {
            return Err(format!(
                "Runtime limit exceeded: {}s > {}s",
                elapsed_secs, self.policy.max_runtime_secs
            ));
        }
        Ok(())
    }

    /// Add an entry to the audit log.
    fn log_entry(
        &mut self,
        action: &ActionDescriptor,
        decision: AuditDecision,
        risk: RiskLevel,
        reason: Option<String>,
    ) {
        self.audit_log.push(AuditEntry {
            timestamp: now_millis(),
            action: action.clone(),
            decision,
            risk,
            reason,
        });
    }

    /// Get the full audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Get the number of audit entries.
    pub fn audit_count(&self) -> usize {
        self.audit_log.len()
    }

    /// Count entries by decision type.
    pub fn count_by_decision(&self, decision: &AuditDecision) -> usize {
        self.audit_log
            .iter()
            .filter(|e| &e.decision == decision)
            .count()
    }

    /// Export audit log as JSON.
    pub fn export_audit(&self) -> String {
        let entries: Vec<String> = self.audit_log.iter().map(|e| {
            format!(
                r#"  {{"timestamp":{},"action_type":"{:?}","target":"{}","decision":"{:?}","risk":"{:?}"{}}}"#,
                e.timestamp,
                e.action.action_type,
                e.action.target.replace('\\', "\\\\").replace('"', "\\\""),
                e.decision,
                e.risk,
                if let Some(ref r) = e.reason {
                    format!(r#","reason":"{}""#, r.replace('"', "\\\""))
                } else {
                    String::new()
                }
            )
        }).collect();
        format!("[\n{}\n]", entries.join(",\n"))
    }

    /// Clear the audit log.
    pub fn clear_audit(&mut self) {
        self.audit_log.clear();
    }

    /// Reset cost and start time.
    pub fn reset_limits(&mut self) {
        self.total_cost = 0.0;
        self.start_time = now_millis();
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_policy::{AgentPolicyBuilder, AutoApproveAll, AutoDenyAll, InternetMode};

    #[test]
    fn test_validate_file_read_allowed() {
        let policy = AgentPolicyBuilder::new()
            .allow_path("/home/user/project")
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        assert!(sandbox
            .validate_file_read("/home/user/project/src/main.rs")
            .is_ok());
        assert_eq!(sandbox.audit_count(), 1);
        assert_eq!(sandbox.audit_log()[0].decision, AuditDecision::Approved);
    }

    #[test]
    fn test_validate_file_read_denied() {
        let policy = AgentPolicyBuilder::new()
            .allow_path("/home/user/project")
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        assert!(sandbox.validate_file_read("/etc/passwd").is_err());
        assert_eq!(sandbox.audit_count(), 1);
        assert_eq!(sandbox.audit_log()[0].decision, AuditDecision::Denied);
    }

    #[test]
    fn test_validate_command_with_approval() {
        let policy = AgentPolicyBuilder::new().allow_command("cargo").build();
        let handler: Arc<dyn ApprovalHandler> = Arc::new(AutoApproveAll);
        let mut sandbox = SandboxValidator::with_approval(policy, handler);

        // cargo is allowed, but ShellExec is Medium risk → needs approval
        assert!(sandbox.validate_command("cargo build").is_ok());
        assert_eq!(
            sandbox.audit_log()[0].decision,
            AuditDecision::ApprovedByUser
        );
    }

    #[test]
    fn test_validate_command_denied_by_user() {
        let policy = AgentPolicyBuilder::new().allow_command("cargo").build();
        let handler: Arc<dyn ApprovalHandler> = Arc::new(AutoDenyAll);
        let mut sandbox = SandboxValidator::with_approval(policy, handler);

        assert!(sandbox.validate_command("cargo build").is_err());
        assert_eq!(sandbox.audit_log()[0].decision, AuditDecision::DeniedByUser);
    }

    #[test]
    fn test_validate_url() {
        let policy = AgentPolicyBuilder::new()
            .internet(InternetMode::AllowList(vec!["github.com".to_string()]))
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        assert!(sandbox.validate_url("https://github.com/repo").is_ok());
        assert!(sandbox.validate_url("https://evil.com/data").is_err());
    }

    #[test]
    fn test_validate_mcp() {
        let policy = AgentPolicyBuilder::new().allow_mcp("filesystem").build();
        let mut sandbox = SandboxValidator::new(policy);

        assert!(sandbox.validate_mcp("filesystem").is_ok());
        assert!(sandbox.validate_mcp("slack").is_err());
    }

    #[test]
    fn test_cost_limit() {
        let policy = AgentPolicyBuilder::new()
            .max_cost(0.10)
            .allow_path("/tmp")
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        sandbox.record_cost(0.05);
        assert!(sandbox.validate_file_read("/tmp/a.txt").is_ok());

        sandbox.record_cost(0.06);
        assert!(sandbox.validate_file_read("/tmp/b.txt").is_err());
    }

    #[test]
    fn test_export_audit() {
        let policy = AgentPolicyBuilder::new().allow_path("/tmp").build();
        let mut sandbox = SandboxValidator::new(policy);

        sandbox.validate_file_read("/tmp/file.txt").unwrap();
        let json = sandbox.export_audit();
        assert!(json.contains("FileRead"));
        assert!(json.contains("/tmp/file.txt"));
        assert!(json.contains("Approved"));
    }

    #[test]
    fn test_count_by_decision() {
        let policy = AgentPolicyBuilder::new().allow_path("/tmp").build();
        let mut sandbox = SandboxValidator::new(policy);

        sandbox.validate_file_read("/tmp/a.txt").unwrap();
        sandbox.validate_file_read("/tmp/b.txt").unwrap();
        let _ = sandbox.validate_file_read("/etc/passwd"); // denied

        assert_eq!(sandbox.count_by_decision(&AuditDecision::Approved), 2);
        assert_eq!(sandbox.count_by_decision(&AuditDecision::Denied), 1);
    }

    #[test]
    fn test_reset_limits() {
        let policy = AgentPolicyBuilder::new()
            .max_cost(0.10)
            .allow_path("/tmp")
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        sandbox.record_cost(0.20);
        assert!(sandbox.validate_file_read("/tmp/a.txt").is_err());

        sandbox.reset_limits();
        assert!(sandbox.validate_file_read("/tmp/a.txt").is_ok());
    }

    #[test]
    fn test_set_policy() {
        // Start with a policy that only allows /tmp
        let policy1 = AgentPolicyBuilder::new().allow_path("/tmp").build();
        let mut sandbox = SandboxValidator::new(policy1);

        // /home should be denied under the first policy
        assert!(sandbox.validate_file_read("/home/user/file.txt").is_err());

        // Switch to a policy that allows /home
        let policy2 = AgentPolicyBuilder::new().allow_path("/home").build();
        sandbox.set_policy(policy2);

        // Now /home should be allowed
        assert!(sandbox.validate_file_read("/home/user/file.txt").is_ok());
        // And /tmp should be denied under the new policy
        assert!(sandbox.validate_file_read("/tmp/a.txt").is_err());
    }

    #[test]
    fn test_validate_tool() {
        let policy = AgentPolicyBuilder::new()
            .allow_tool("read_file")
            .deny_tool("delete_file")
            .build();
        let mut sandbox = SandboxValidator::new(policy);

        // Allowed tool should pass
        assert!(sandbox.validate_tool("read_file").is_ok());
        assert_eq!(
            sandbox.audit_log().last().unwrap().decision,
            AuditDecision::Approved
        );

        // Denied tool should fail
        assert!(sandbox.validate_tool("delete_file").is_err());
        assert_eq!(
            sandbox.audit_log().last().unwrap().decision,
            AuditDecision::Denied
        );
    }

    #[test]
    fn test_clear_audit() {
        let policy = AgentPolicyBuilder::new().allow_path("/tmp").build();
        let mut sandbox = SandboxValidator::new(policy);

        // Perform some actions to generate audit entries
        sandbox.validate_file_read("/tmp/a.txt").unwrap();
        sandbox.validate_file_read("/tmp/b.txt").unwrap();
        let _ = sandbox.validate_file_read("/etc/passwd"); // denied
        assert_eq!(sandbox.audit_count(), 3);

        // Clear and verify empty
        sandbox.clear_audit();
        assert_eq!(sandbox.audit_count(), 0);
        assert!(sandbox.audit_log().is_empty());
    }
}
