//! Agent policy — permission model and sandbox configuration
//!
//! Defines what an autonomous agent can and cannot do: paths, commands,
//! internet access, MCP servers, cost limits, and risk-based approval.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Enums
// ============================================================================

/// How much autonomy the agent has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AutonomyLevel {
    /// Every action requires user approval.
    Paranoid,
    /// Normal operation — approve risky actions only.
    Normal,
    /// Full autonomy — only approve Critical-risk actions.
    Autonomous,
}

/// How the agent can access the internet.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InternetMode {
    /// No internet access at all.
    Disabled,
    /// Search-only (web search tool allowed, direct HTTP not).
    SearchOnly,
    /// Full access to any URL.
    FullAccess,
    /// Only these domains are allowed.
    AllowList(Vec<String>),
}

/// Risk level of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Type of action an agent wants to perform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ActionType {
    FileRead,
    FileWrite,
    FileDelete,
    ShellExec,
    HttpRequest,
    McpCall,
    ToolCall,
    BrowserAction,
}

// ============================================================================
// ActionDescriptor
// ============================================================================

/// Describes a specific action an agent wants to perform.
#[derive(Debug, Clone)]
pub struct ActionDescriptor {
    pub action_type: ActionType,
    /// Path, URL, command, or tool name depending on action_type.
    pub target: String,
    /// Extra parameters.
    pub parameters: HashMap<String, String>,
}

impl ActionDescriptor {
    pub fn new(action_type: ActionType, target: impl Into<String>) -> Self {
        Self {
            action_type,
            target: target.into(),
            parameters: HashMap::new(),
        }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

// ============================================================================
// ApprovalHandler trait
// ============================================================================

/// Trait for handling approval requests when an action exceeds the agent's
/// autonomy level.
pub trait ApprovalHandler: Send + Sync {
    /// Ask the user/system to approve an action. Returns true if approved.
    fn request_approval(&self, action: &str, risk: RiskLevel) -> bool;
}

/// An approval handler that always approves.
///
/// # Security Warning
///
/// This handler bypasses ALL approval checks, including HITL gates.
/// It should ONLY be used in test code. Using it in production disables
/// the entire human-in-the-loop safety system.
#[deprecated(note = "Use an explicit ApprovalHandler in production — AutoApproveAll bypasses all safety checks")]
pub struct AutoApproveAll;

impl ApprovalHandler for AutoApproveAll {
    fn request_approval(&self, action: &str, risk: RiskLevel) -> bool {
        if risk >= RiskLevel::High {
            log::warn!(
                "AutoApproveAll: auto-approving {:?}-risk action '{}' — \
                 this is unsafe in production",
                risk,
                action
            );
        }
        true
    }
}

/// An approval handler that always denies.
pub struct AutoDenyAll;

impl ApprovalHandler for AutoDenyAll {
    fn request_approval(&self, _action: &str, _risk: RiskLevel) -> bool {
        false
    }
}

/// An approval handler backed by a closure.
pub struct ClosureApprovalHandler {
    handler: Box<dyn Fn(&str, RiskLevel) -> bool + Send + Sync>,
}

impl ClosureApprovalHandler {
    pub fn new(f: impl Fn(&str, RiskLevel) -> bool + Send + Sync + 'static) -> Self {
        Self {
            handler: Box::new(f),
        }
    }
}

impl ApprovalHandler for ClosureApprovalHandler {
    fn request_approval(&self, action: &str, risk: RiskLevel) -> bool {
        (self.handler)(action, risk)
    }
}

// ============================================================================
// AgentPolicy
// ============================================================================

/// Complete policy configuration for an autonomous agent.
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    pub autonomy: AutonomyLevel,
    pub internet: InternetMode,
    /// Paths the agent is allowed to access. Empty = cwd only.
    pub allowed_paths: Vec<PathBuf>,
    /// Paths explicitly denied (takes priority over allowed).
    pub denied_paths: Vec<PathBuf>,
    /// Shell commands whitelisted. Empty = all denied.
    pub allowed_commands: Vec<String>,
    /// Shell commands blacklisted (takes priority over allowed).
    pub denied_commands: Vec<String>,
    /// MCP servers the agent may use. Empty = none.
    pub mcp_servers: Vec<String>,
    /// Maximum iterations for the agent loop.
    pub max_iterations: usize,
    /// Maximum cost in USD before stopping.
    pub max_cost_usd: f64,
    /// Maximum runtime in seconds.
    pub max_runtime_secs: u64,
    /// Actions at this risk level or above require approval.
    pub require_approval_above: RiskLevel,
    /// Per-tool allow/deny overrides.
    pub tool_permissions: HashMap<String, bool>,
    /// Environment variables available to the agent.
    pub env_vars: HashMap<String, String>,
    /// Working directory for the agent.
    pub working_directory: Option<PathBuf>,
}

impl Default for AgentPolicy {
    /// Normal autonomy, search-only internet, cwd, 50 iterations, $1 limit.
    fn default() -> Self {
        Self {
            autonomy: AutonomyLevel::Normal,
            internet: InternetMode::SearchOnly,
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            allowed_commands: Vec::new(),
            denied_commands: Vec::new(),
            mcp_servers: Vec::new(),
            max_iterations: 50,
            max_cost_usd: 1.0,
            max_runtime_secs: 600,
            require_approval_above: RiskLevel::Medium,
            tool_permissions: HashMap::new(),
            env_vars: HashMap::new(),
            working_directory: None,
        }
    }
}

impl AgentPolicy {
    /// Paranoid policy: every action needs approval, no internet, no shell.
    pub fn paranoid() -> Self {
        Self {
            autonomy: AutonomyLevel::Paranoid,
            internet: InternetMode::Disabled,
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            allowed_commands: Vec::new(),
            denied_commands: Vec::new(),
            mcp_servers: Vec::new(),
            max_iterations: 10,
            max_cost_usd: 0.10,
            max_runtime_secs: 120,
            require_approval_above: RiskLevel::Safe,
            tool_permissions: HashMap::new(),
            env_vars: HashMap::new(),
            working_directory: None,
        }
    }

    /// Autonomous policy: full access, only approve Critical.
    pub fn autonomous() -> Self {
        Self {
            autonomy: AutonomyLevel::Autonomous,
            internet: InternetMode::FullAccess,
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            allowed_commands: vec!["*".to_string()],
            denied_commands: Vec::new(),
            mcp_servers: vec!["*".to_string()],
            max_iterations: 200,
            max_cost_usd: 10.0,
            max_runtime_secs: 3600,
            require_approval_above: RiskLevel::Critical,
            tool_permissions: HashMap::new(),
            env_vars: HashMap::new(),
            working_directory: None,
        }
    }

    /// Check if the agent can access a path.
    ///
    /// Rejects `..` path components to prevent traversal attacks (H6).
    /// Uses canonicalization when both paths exist on disk; otherwise
    /// falls back to raw `starts_with` comparison.
    pub fn can_access_path(&self, path: &Path) -> bool {
        // Reject any path containing ".." components (traversal attack)
        for component in path.components() {
            if component == std::path::Component::ParentDir {
                return false;
            }
        }

        // Helper: compare two paths using canonicalization when possible.
        // If both canonicalize, compare canonicalized forms.
        // If neither canonicalizes, compare raw forms.
        // If only one canonicalizes, compare both raw AND canonicalized forms
        // (to avoid false negatives when one path exists and the other doesn't).
        fn path_starts_with(child: &Path, parent: &Path) -> bool {
            let child_canon = std::fs::canonicalize(child).ok();
            let parent_canon = std::fs::canonicalize(parent).ok();

            match (&child_canon, &parent_canon) {
                (Some(cc), Some(pc)) => cc.starts_with(pc),
                _ => {
                    // Fallback: raw path comparison (safe because .. is already rejected)
                    child.starts_with(parent)
                }
            }
        }

        // Denied paths take priority
        for denied in &self.denied_paths {
            if path_starts_with(path, denied) {
                return false;
            }
        }
        // If allowed_paths is empty, allow cwd only
        if self.allowed_paths.is_empty() {
            if let Some(ref wd) = self.working_directory {
                return path_starts_with(path, wd);
            }
            // No working directory set and no allowed paths = deny all (safe default)
            return false;
        }
        // Check if path is under any allowed path
        for allowed in &self.allowed_paths {
            if path_starts_with(path, allowed) {
                return true;
            }
        }
        false
    }

    /// Check if the agent can run a shell command.
    pub fn can_run_command(&self, cmd: &str) -> bool {
        // Extract the base command (first word)
        let base = cmd.split_whitespace().next().unwrap_or("");

        // Denied commands take priority
        for denied in &self.denied_commands {
            if denied == "*" {
                return false;
            }
            if base == denied || cmd.contains(denied) {
                return false;
            }
        }
        // Empty allowed = none allowed (unless "*")
        if self.allowed_commands.is_empty() {
            return false;
        }
        for allowed in &self.allowed_commands {
            if allowed == "*" || base == allowed {
                return true;
            }
        }
        false
    }

    /// Check if the agent can use an MCP server.
    pub fn can_use_mcp(&self, server: &str) -> bool {
        if self.mcp_servers.is_empty() {
            return false;
        }
        self.mcp_servers.iter().any(|s| s == "*" || s == server)
    }

    /// Check if the agent can access a URL.
    pub fn can_access_internet(&self, url: &str) -> bool {
        match &self.internet {
            InternetMode::Disabled => false,
            InternetMode::SearchOnly => false, // only search tool, not direct HTTP
            InternetMode::FullAccess => true,
            InternetMode::AllowList(domains) => {
                // Extract domain from URL and match on subdomain boundaries
                let domain = extract_domain(url);
                domains.iter().any(|d| domain == d.as_str() || domain.ends_with(&format!(".{}", d)))
            }
        }
    }

    /// Check if a tool is allowed by per-tool overrides.
    pub fn can_use_tool(&self, tool_name: &str) -> bool {
        match self.tool_permissions.get(tool_name) {
            Some(&allowed) => allowed,
            None => true, // default: allowed unless explicitly denied
        }
    }

    /// Assess the risk level of an action.
    pub fn assess_risk(&self, action: &ActionDescriptor) -> RiskLevel {
        match action.action_type {
            ActionType::FileRead => RiskLevel::Safe,
            ActionType::ToolCall => RiskLevel::Low,
            ActionType::HttpRequest => RiskLevel::Low,
            ActionType::McpCall => RiskLevel::Low,
            ActionType::BrowserAction => RiskLevel::Medium,
            ActionType::FileWrite => RiskLevel::Medium,
            ActionType::ShellExec => {
                let cmd = &action.target;
                if is_dangerous_command(cmd) {
                    RiskLevel::Critical
                } else if is_risky_command(cmd) {
                    RiskLevel::High
                } else {
                    RiskLevel::Medium
                }
            }
            ActionType::FileDelete => RiskLevel::High,
        }
    }

    /// Check if an action needs user approval based on risk vs policy.
    pub fn needs_approval(&self, action: &ActionDescriptor) -> bool {
        if self.autonomy == AutonomyLevel::Paranoid {
            return true;
        }
        let risk = self.assess_risk(action);
        risk >= self.require_approval_above
    }

    /// Validate an action against the full policy. Returns Ok(()) if allowed,
    /// Err(reason) if denied.
    pub fn validate_action(
        &self,
        action: &ActionDescriptor,
        approval_handler: Option<&Arc<dyn ApprovalHandler>>,
    ) -> Result<(), String> {
        // Check per-type restrictions
        match action.action_type {
            ActionType::FileRead | ActionType::FileWrite | ActionType::FileDelete => {
                let path = Path::new(&action.target);
                if !self.can_access_path(path) {
                    return Err(format!("Path not allowed: {}", action.target));
                }
            }
            ActionType::ShellExec => {
                if !self.can_run_command(&action.target) {
                    return Err(format!("Command not allowed: {}", action.target));
                }
            }
            ActionType::HttpRequest => {
                if !self.can_access_internet(&action.target) {
                    return Err(format!("URL not allowed: {}", action.target));
                }
            }
            ActionType::McpCall => {
                if !self.can_use_mcp(&action.target) {
                    return Err(format!("MCP server not allowed: {}", action.target));
                }
            }
            ActionType::ToolCall => {
                if !self.can_use_tool(&action.target) {
                    return Err(format!("Tool not allowed: {}", action.target));
                }
            }
            ActionType::BrowserAction => {
                // Browser actions need the browser feature enabled
                // but policy-wise, check internet access for navigation
                if let Some(url) = action.parameters.get("url") {
                    if !self.can_access_internet(url) {
                        return Err(format!("Browser URL not allowed: {}", url));
                    }
                }
            }
        }

        // Check if approval is needed
        if self.needs_approval(action) {
            if let Some(handler) = approval_handler {
                let desc = format!("{:?}: {}", action.action_type, action.target);
                let risk = self.assess_risk(action);
                if !handler.request_approval(&desc, risk) {
                    return Err("Action denied by user".to_string());
                }
            } else if self.autonomy == AutonomyLevel::Paranoid {
                return Err("No approval handler and policy is Paranoid".to_string());
            }
        }

        Ok(())
    }
}

// ============================================================================
// AgentPolicyBuilder
// ============================================================================

/// Builder for constructing AgentPolicy with fluent API.
pub struct AgentPolicyBuilder {
    policy: AgentPolicy,
}

impl AgentPolicyBuilder {
    pub fn new() -> Self {
        Self {
            policy: AgentPolicy::default(),
        }
    }

    pub fn autonomy(mut self, level: AutonomyLevel) -> Self {
        self.policy.autonomy = level;
        self
    }

    pub fn internet(mut self, mode: InternetMode) -> Self {
        self.policy.internet = mode;
        self
    }

    pub fn allow_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.policy.allowed_paths.push(path.into());
        self
    }

    pub fn deny_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.policy.denied_paths.push(path.into());
        self
    }

    pub fn allow_command(mut self, cmd: impl Into<String>) -> Self {
        self.policy.allowed_commands.push(cmd.into());
        self
    }

    pub fn deny_command(mut self, cmd: impl Into<String>) -> Self {
        self.policy.denied_commands.push(cmd.into());
        self
    }

    pub fn allow_mcp(mut self, server: impl Into<String>) -> Self {
        self.policy.mcp_servers.push(server.into());
        self
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.policy.max_iterations = n;
        self
    }

    pub fn max_cost(mut self, usd: f64) -> Self {
        self.policy.max_cost_usd = usd;
        self
    }

    pub fn max_runtime(mut self, secs: u64) -> Self {
        self.policy.max_runtime_secs = secs;
        self
    }

    pub fn require_approval_above(mut self, risk: RiskLevel) -> Self {
        self.policy.require_approval_above = risk;
        self
    }

    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.policy.tool_permissions.insert(name.into(), true);
        self
    }

    pub fn deny_tool(mut self, name: impl Into<String>) -> Self {
        self.policy.tool_permissions.insert(name.into(), false);
        self
    }

    pub fn env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.policy.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn working_directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.policy.working_directory = Some(path.into());
        self
    }

    pub fn build(self) -> AgentPolicy {
        self.policy
    }
}

impl Default for AgentPolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_domain(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    without_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_lowercase()
}

fn is_dangerous_command(cmd: &str) -> bool {
    let dangerous = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "chmod -R 777 /",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "format",
    ];
    let lower = cmd.to_lowercase();
    dangerous.iter().any(|d| lower.contains(d))
}

fn is_risky_command(cmd: &str) -> bool {
    let risky = [
        "rm -rf",
        "rm -r",
        "chmod",
        "chown",
        "sudo",
        "su ",
        "kill",
        "pkill",
        "docker rm",
        "docker rmi",
        "git push --force",
        "git reset --hard",
        "drop table",
        "drop database",
        "truncate",
    ];
    let lower = cmd.to_lowercase();
    risky.iter().any(|r| lower.contains(r))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = AgentPolicy::default();
        assert_eq!(policy.autonomy, AutonomyLevel::Normal);
        assert_eq!(policy.internet, InternetMode::SearchOnly);
        assert_eq!(policy.max_iterations, 50);
        assert_eq!(policy.max_cost_usd, 1.0);
    }

    #[test]
    fn test_paranoid_policy() {
        let policy = AgentPolicy::paranoid();
        assert_eq!(policy.autonomy, AutonomyLevel::Paranoid);
        assert_eq!(policy.internet, InternetMode::Disabled);
        assert_eq!(policy.require_approval_above, RiskLevel::Safe);
        assert_eq!(policy.max_iterations, 10);
    }

    #[test]
    fn test_autonomous_policy() {
        let policy = AgentPolicy::autonomous();
        assert_eq!(policy.autonomy, AutonomyLevel::Autonomous);
        assert_eq!(policy.internet, InternetMode::FullAccess);
        assert!(policy.can_run_command("ls"));
        assert!(policy.can_use_mcp("any-server"));
    }

    #[test]
    fn test_can_access_path_with_allowed() {
        let policy = AgentPolicyBuilder::new()
            .allow_path("/home/user/project")
            .build();
        assert!(policy.can_access_path(Path::new("/home/user/project/src/main.rs")));
        assert!(!policy.can_access_path(Path::new("/etc/passwd")));
    }

    #[test]
    fn test_denied_path_overrides_allowed() {
        let policy = AgentPolicyBuilder::new()
            .allow_path("/home/user")
            .deny_path("/home/user/.ssh")
            .build();
        assert!(policy.can_access_path(Path::new("/home/user/project/file.txt")));
        assert!(!policy.can_access_path(Path::new("/home/user/.ssh/id_rsa")));
    }

    #[test]
    fn test_can_run_command() {
        let policy = AgentPolicyBuilder::new()
            .allow_command("cargo")
            .allow_command("git")
            .deny_command("rm")
            .build();
        assert!(policy.can_run_command("cargo build"));
        assert!(policy.can_run_command("git status"));
        assert!(!policy.can_run_command("rm -rf /tmp"));
        assert!(!policy.can_run_command("python script.py"));
    }

    #[test]
    fn test_empty_allowed_commands_denies_all() {
        let policy = AgentPolicy::default();
        assert!(!policy.can_run_command("ls"));
        assert!(!policy.can_run_command("echo hello"));
    }

    #[test]
    fn test_can_use_mcp() {
        let policy = AgentPolicyBuilder::new()
            .allow_mcp("filesystem")
            .allow_mcp("github")
            .build();
        assert!(policy.can_use_mcp("filesystem"));
        assert!(policy.can_use_mcp("github"));
        assert!(!policy.can_use_mcp("slack"));
    }

    #[test]
    fn test_internet_modes() {
        // Disabled
        let p = AgentPolicyBuilder::new()
            .internet(InternetMode::Disabled)
            .build();
        assert!(!p.can_access_internet("https://example.com"));

        // SearchOnly
        let p = AgentPolicy::default();
        assert!(!p.can_access_internet("https://example.com"));

        // FullAccess
        let p = AgentPolicyBuilder::new()
            .internet(InternetMode::FullAccess)
            .build();
        assert!(p.can_access_internet("https://example.com"));

        // AllowList
        let p = AgentPolicyBuilder::new()
            .internet(InternetMode::AllowList(vec![
                "github.com".to_string(),
                "api.openai.com".to_string(),
            ]))
            .build();
        assert!(p.can_access_internet("https://github.com/repo"));
        assert!(p.can_access_internet("https://api.openai.com/v1/chat"));
        assert!(!p.can_access_internet("https://evil.com"));
    }

    #[test]
    fn test_risk_assessment() {
        let policy = AgentPolicy::default();
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(ActionType::FileRead, "/tmp/a.txt")),
            RiskLevel::Safe
        );
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(ActionType::FileWrite, "/tmp/a.txt")),
            RiskLevel::Medium
        );
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(ActionType::FileDelete, "/tmp/a.txt")),
            RiskLevel::High
        );
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(ActionType::ShellExec, "rm -rf /")),
            RiskLevel::Critical
        );
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(ActionType::ShellExec, "ls")),
            RiskLevel::Medium
        );
        assert_eq!(
            policy.assess_risk(&ActionDescriptor::new(
                ActionType::ShellExec,
                "git push --force"
            )),
            RiskLevel::High
        );
    }

    #[test]
    fn test_needs_approval() {
        // Normal: approve Medium and above
        let normal = AgentPolicy::default();
        assert!(!normal.needs_approval(&ActionDescriptor::new(ActionType::FileRead, "f")));
        assert!(!normal.needs_approval(&ActionDescriptor::new(ActionType::ToolCall, "t")));
        assert!(normal.needs_approval(&ActionDescriptor::new(ActionType::FileWrite, "f")));
        assert!(normal.needs_approval(&ActionDescriptor::new(ActionType::ShellExec, "ls")));

        // Paranoid: approve everything
        let paranoid = AgentPolicy::paranoid();
        assert!(paranoid.needs_approval(&ActionDescriptor::new(ActionType::FileRead, "f")));

        // Autonomous: only Critical
        let auto = AgentPolicy::autonomous();
        assert!(!auto.needs_approval(&ActionDescriptor::new(ActionType::FileWrite, "f")));
        assert!(!auto.needs_approval(&ActionDescriptor::new(ActionType::FileDelete, "f")));
        assert!(auto.needs_approval(&ActionDescriptor::new(ActionType::ShellExec, "rm -rf /")));
    }

    #[test]
    fn test_validate_action_with_approval() {
        // Policy with a working directory so path checks pass
        let mut policy = AgentPolicy::default();
        policy.working_directory = Some(PathBuf::from("/tmp"));
        let handler: Arc<dyn ApprovalHandler> = Arc::new(AutoApproveAll);

        // FileRead is Safe → no approval needed → passes
        let action = ActionDescriptor::new(ActionType::FileRead, "/tmp/file.txt");
        assert!(policy.validate_action(&action, Some(&handler)).is_ok());

        // FileWrite is Medium → needs approval → AutoApproveAll approves
        let action = ActionDescriptor::new(ActionType::FileWrite, "/tmp/file.txt");
        assert!(policy.validate_action(&action, Some(&handler)).is_ok());

        // With deny handler
        let deny_handler: Arc<dyn ApprovalHandler> = Arc::new(AutoDenyAll);
        assert!(policy
            .validate_action(&action, Some(&deny_handler))
            .is_err());
    }

    #[test]
    fn test_validate_action_path_restrictions() {
        let policy = AgentPolicyBuilder::new()
            .allow_path("/home/user/project")
            .build();

        // Allowed path
        let action = ActionDescriptor::new(ActionType::FileRead, "/home/user/project/src/main.rs");
        assert!(policy.validate_action(&action, None).is_ok());

        // Denied path
        let action = ActionDescriptor::new(ActionType::FileRead, "/etc/passwd");
        assert!(policy.validate_action(&action, None).is_err());
    }

    #[test]
    fn test_builder_fluent() {
        let policy = AgentPolicyBuilder::new()
            .autonomy(AutonomyLevel::Autonomous)
            .internet(InternetMode::FullAccess)
            .allow_path("/home")
            .deny_path("/home/.secrets")
            .allow_command("cargo")
            .deny_command("rm")
            .allow_mcp("github")
            .max_iterations(100)
            .max_cost(5.0)
            .max_runtime(1800)
            .require_approval_above(RiskLevel::High)
            .allow_tool("read_file")
            .deny_tool("delete_file")
            .env_var("RUST_LOG", "debug")
            .working_directory("/home/user/project")
            .build();

        assert_eq!(policy.autonomy, AutonomyLevel::Autonomous);
        assert_eq!(policy.internet, InternetMode::FullAccess);
        assert_eq!(policy.allowed_paths.len(), 1);
        assert_eq!(policy.denied_paths.len(), 1);
        assert_eq!(policy.max_iterations, 100);
        assert_eq!(policy.max_cost_usd, 5.0);
        assert!(policy.can_use_tool("read_file"));
        assert!(!policy.can_use_tool("delete_file"));
        assert_eq!(policy.env_vars.get("RUST_LOG"), Some(&"debug".to_string()));
    }

    #[test]
    fn test_closure_approval_handler() {
        let handler = ClosureApprovalHandler::new(|_action, risk| risk < RiskLevel::High);
        assert!(handler.request_approval("safe action", RiskLevel::Low));
        assert!(handler.request_approval("medium action", RiskLevel::Medium));
        assert!(!handler.request_approval("high action", RiskLevel::High));
        assert!(!handler.request_approval("critical action", RiskLevel::Critical));
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://github.com/repo"), "github.com");
        assert_eq!(
            extract_domain("http://api.openai.com:8080/v1"),
            "api.openai.com"
        );
        assert_eq!(extract_domain("example.com/path"), "example.com");
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Safe < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }
}
