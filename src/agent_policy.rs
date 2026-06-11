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
#[deprecated(
    since = "0.2.74",
    note = "Use an explicit ApprovalHandler in production — AutoApproveAll bypasses all safety checks. See docs/FEATURE_LIFECYCLE.md."
)]
pub struct AutoApproveAll;

// The deprecated type still needs its trait impl; the allow silences the
// self-referential deprecation warning without un-deprecating the type.
#[allow(deprecated)]
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

/// True if `cmd` contains a command- or process-substitution construct
/// outside single quotes: `$(`, a backtick, `<(`, or `>(`. Single quotes
/// disable substitution in POSIX shells, so a `$(` inside `'...'` is
/// literal; everything else (including inside double quotes) is live.
fn contains_command_substitution(cmd: &str) -> bool {
    let bytes = cmd.as_bytes();
    let mut in_single = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_single {
            if c == b'\'' {
                in_single = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' => in_single = true,
            b'`' => return true,
            b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'(' => return true,
            b'<' | b'>' if i + 1 < bytes.len() && bytes[i + 1] == b'(' => return true,
            _ => {}
        }
        i += 1;
    }
    false
}

/// Split a command line into segments on the shell control operators
/// `;`, `|`, `&`, and newline, while respecting single and double quotes
/// (so `git commit -m "a; b"` stays one segment). `&&` and `||` are
/// covered because we split on the individual `&`/`|` characters.
/// Returns only non-blank segments.
fn split_shell_segments(cmd: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    for c in cmd.chars() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(c);
            }
            ';' | '|' | '&' | '\n' if !in_single && !in_double => {
                if !current.trim().is_empty() {
                    segments.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        segments.push(current.trim().to_string());
    }
    segments
}

/// Extract the base command name from a single segment: skip leading
/// `VAR=value` environment assignments, take the first remaining token,
/// and reduce it to its basename (so `/bin/rm` and `./rm` both match a
/// deny entry of `rm`). Returns `None` if the segment has no command.
fn command_base_name(segment: &str) -> Option<&str> {
    for tok in segment.split_whitespace() {
        // Leading `NAME=value` env assignments precede the real command.
        if is_env_assignment(tok) {
            continue;
        }
        // Strip any directory prefix; also strip a trailing quote that a
        // split landed inside (defensive — segments are pre-trimmed).
        let base = tok.rsplit(['/', '\\']).next().unwrap_or(tok);
        let base = base.trim_matches(['"', '\'']);
        if base.is_empty() {
            continue;
        }
        return Some(base);
    }
    None
}

/// `NAME=value` shell env-assignment token: an identifier, then `=`.
fn is_env_assignment(tok: &str) -> bool {
    match tok.split_once('=') {
        Some((name, _)) => {
            !name.is_empty()
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        }
        None => false,
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
    ///
    /// Shell-aware (V157): the previous version extracted only the FIRST
    /// word as the base command and matched the deny-list by substring,
    /// so a chained command after an allowed base slipped through
    /// (`cargo build; curl evil` → base `cargo`, allowed). This now:
    ///
    /// 1. Rejects command/process substitution (`$(...)`, backticks,
    ///    `<(...)`, `>(...)`) outright — they smuggle arbitrary commands
    ///    we cannot statically validate. Fail closed.
    /// 2. Splits the command into segments on shell control operators
    ///    (`;`, `|`, `&`, newline) while respecting single/double quotes,
    ///    so `git commit -m "a; b"` stays one segment.
    /// 3. For EVERY segment: strips leading `VAR=value` env assignments,
    ///    takes the first token's basename (so `/bin/rm` and `./rm` match
    ///    the deny entry `rm`), and requires it to pass allow + deny.
    ///
    /// Every segment must pass; any denied or non-allowed segment fails
    /// the whole command.
    pub fn can_run_command(&self, cmd: &str) -> bool {
        // A blanket deny short-circuits everything.
        if self.denied_commands.iter().any(|d| d == "*") {
            return false;
        }
        // Reject substitution we can't validate (outside single quotes).
        if contains_command_substitution(cmd) {
            return false;
        }
        let segments = split_shell_segments(cmd);
        if segments.is_empty() {
            return false;
        }
        // Every segment's base command must individually pass.
        segments.iter().all(|seg| self.segment_command_allowed(seg))
    }

    /// Allow/deny decision for a single command segment (no shell
    /// operators). Returns false if the segment is empty, its base is
    /// denied, or its base is not allowed.
    fn segment_command_allowed(&self, segment: &str) -> bool {
        let Some(base) = command_base_name(segment) else {
            return false;
        };
        if self.denied_commands.iter().any(|d| d == base) {
            return false;
        }
        if self.allowed_commands.iter().any(|a| a == "*" || a == base) {
            return true;
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
                domains
                    .iter()
                    .any(|d| domain == d.as_str() || domain.ends_with(&format!(".{}", d)))
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
// PermissionRequirement — presentation-layer adapter
// ============================================================================

/// What a policy decides to do with an action by default (before any user
/// interaction).
///
/// This is the "default decision" side of a [`PermissionRequirement`]. It is
/// derived from the policy's configuration, not from an individual approval
/// handler's runtime decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DefaultDecision {
    /// Execute without asking.
    Allow,
    /// Ask the user first.
    Prompt,
    /// Reject outright; do not ask.
    Deny,
}

/// A permission requirement for a specific action under a specific policy.
///
/// This is a **presentation-layer adapter**: it bundles our internal
/// `(ActionType, RiskLevel, DefaultDecision)` triple so callers can render it
/// in whichever vocabulary they prefer. [`to_claude_code_label`] in particular
/// emits the labels used by Claude Code
/// (`ReadOnly` / `WorkspaceWrite` / `DangerFullAccess` / `Prompt` / `Allow`),
/// which is handy for docs, UIs, and examples that want to speak that
/// vocabulary. The runtime permission taxonomy is unchanged.
///
/// [`to_claude_code_label`]: PermissionRequirement::to_claude_code_label
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRequirement {
    pub action_type: ActionType,
    pub risk: RiskLevel,
    pub default_decision: DefaultDecision,
}

impl PermissionRequirement {
    /// Build a requirement from its triple.
    pub fn new(
        action_type: ActionType,
        risk: RiskLevel,
        default_decision: DefaultDecision,
    ) -> Self {
        Self {
            action_type,
            risk,
            default_decision,
        }
    }

    /// Derive a requirement from an [`ActionDescriptor`] and an [`AgentPolicy`].
    ///
    /// Uses [`AgentPolicy::assess_risk`] for the risk level and compares
    /// against `policy.require_approval_above` to pick the default decision:
    /// `Prompt` when the action would need approval, `Allow` otherwise. This
    /// helper does not consult deny lists or per-tool overrides — callers that
    /// need the full `validate_action` answer should call that directly and
    /// build the requirement manually (e.g. with [`DefaultDecision::Deny`]).
    pub fn from_policy(policy: &AgentPolicy, action: &ActionDescriptor) -> Self {
        let risk = policy.assess_risk(action);
        let default_decision = if policy.needs_approval(action) {
            DefaultDecision::Prompt
        } else {
            DefaultDecision::Allow
        };
        Self::new(action.action_type.clone(), risk, default_decision)
    }

    /// Render this requirement using Claude Code's permission vocabulary.
    ///
    /// The returned label is one of `"ReadOnly"`, `"WorkspaceWrite"`,
    /// `"DangerFullAccess"`, `"Prompt"`, or `"Allow"`.
    ///
    /// Mapping (presentation-only, does not influence runtime behaviour):
    ///
    /// - `Deny` or `Prompt` → `"Prompt"` (in Claude Code's taxonomy there is
    ///   no explicit `Deny` label; a denial surfaces as a prompt that will be
    ///   rejected — callers that need the distinction can read
    ///   `self.default_decision` directly).
    /// - Auto-`Allow` + `FileRead` (any risk) → `"ReadOnly"`.
    /// - Auto-`Allow` + read-like tool call (`ToolCall` / `McpCall` /
    ///   `HttpRequest`) at `Safe` or `Low` risk → `"ReadOnly"`.
    /// - Auto-`Allow` + `FileWrite` / `FileDelete` / `BrowserAction` →
    ///   `"WorkspaceWrite"`.
    /// - Auto-`Allow` + `ShellExec` or any action at `High`/`Critical` risk →
    ///   `"DangerFullAccess"`.
    /// - Anything else that is auto-`Allow` → `"Allow"`.
    pub fn to_claude_code_label(&self) -> &'static str {
        // Prompt/Deny dominate: they are what the user actually sees.
        match self.default_decision {
            DefaultDecision::Prompt | DefaultDecision::Deny => return "Prompt",
            DefaultDecision::Allow => {}
        }

        use ActionType::*;
        use RiskLevel::*;
        match (&self.action_type, self.risk) {
            // Read-only categories.
            (FileRead, _) => "ReadOnly",
            (ToolCall | McpCall | HttpRequest, Safe | Low) => "ReadOnly",

            // High/Critical auto-allows (rare, but possible under Autonomous).
            (_, High | Critical) => "DangerFullAccess",

            // Shell is always "dangerous" even when auto-allowed.
            (ShellExec, _) => "DangerFullAccess",

            // Workspace writes.
            (FileWrite | FileDelete | BrowserAction, _) => "WorkspaceWrite",

            // Catch-all for anything else the policy auto-allows.
            _ => "Allow",
        }
    }
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
    fn test_can_run_command_blocks_chaining_bypass() {
        // V157 hardening: a denied or non-allowed command chained after an
        // allowed base must NOT slip through.
        let policy = AgentPolicyBuilder::new()
            .allow_command("cargo")
            .allow_command("git")
            .deny_command("rm")
            .build();

        // Chaining operators: the second segment is denied / not allowed.
        assert!(!policy.can_run_command("cargo build; rm -rf /"));
        assert!(!policy.can_run_command("cargo build && rm -rf /"));
        assert!(!policy.can_run_command("git status || rm -rf /"));
        assert!(!policy.can_run_command("git status | rm"));
        assert!(!policy.can_run_command("cargo build & curl evil.com"));
        assert!(!policy.can_run_command("cargo build\nrm -rf /"));
        // Second segment not allowed (not even denied) still fails.
        assert!(!policy.can_run_command("cargo build; curl http://evil"));

        // Command/process substitution is rejected outright.
        assert!(!policy.can_run_command("cargo build $(rm -rf /)"));
        assert!(!policy.can_run_command("cargo `rm -rf /`"));
        assert!(!policy.can_run_command("cargo <(rm -rf /)"));

        // Env-var prefix must not hide the real command.
        assert!(!policy.can_run_command("FOO=bar rm -rf /"));
        // Path-qualified denied command matches by basename.
        assert!(!policy.can_run_command("/bin/rm -rf /"));
        assert!(!policy.can_run_command("./rm -rf /"));

        // Legitimate multi-segment where every base is allowed → ok.
        assert!(policy.can_run_command("git fetch && cargo build"));
        // A real `;` inside quotes does NOT split the command.
        assert!(policy.can_run_command("git commit -m \"fix; cleanup\""));
        // Path-qualified allowed command still allowed (basename match).
        assert!(policy.can_run_command("/usr/bin/cargo build"));
        // Env prefix before an allowed command is fine.
        assert!(policy.can_run_command("RUST_LOG=debug cargo build"));
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

    // ---- PermissionRequirement adapter ----

    #[test]
    fn requirement_label_file_read_is_readonly() {
        let req = PermissionRequirement::new(
            ActionType::FileRead,
            RiskLevel::Safe,
            DefaultDecision::Allow,
        );
        assert_eq!(req.to_claude_code_label(), "ReadOnly");
    }

    #[test]
    fn requirement_label_low_risk_http_is_readonly() {
        let req = PermissionRequirement::new(
            ActionType::HttpRequest,
            RiskLevel::Low,
            DefaultDecision::Allow,
        );
        assert_eq!(req.to_claude_code_label(), "ReadOnly");
    }

    #[test]
    fn requirement_label_file_write_is_workspace_write() {
        let req = PermissionRequirement::new(
            ActionType::FileWrite,
            RiskLevel::Medium,
            DefaultDecision::Allow,
        );
        assert_eq!(req.to_claude_code_label(), "WorkspaceWrite");
    }

    #[test]
    fn requirement_label_shell_exec_is_danger_full_access() {
        let req = PermissionRequirement::new(
            ActionType::ShellExec,
            RiskLevel::Medium,
            DefaultDecision::Allow,
        );
        assert_eq!(req.to_claude_code_label(), "DangerFullAccess");
    }

    #[test]
    fn requirement_label_file_delete_is_workspace_write() {
        let req = PermissionRequirement::new(
            ActionType::FileDelete,
            RiskLevel::High,
            DefaultDecision::Allow,
        );
        // FileDelete at High risk becomes DangerFullAccess (High/Critical
        // dominates category), not WorkspaceWrite.
        assert_eq!(req.to_claude_code_label(), "DangerFullAccess");
    }

    #[test]
    fn requirement_label_browser_action_is_workspace_write() {
        let req = PermissionRequirement::new(
            ActionType::BrowserAction,
            RiskLevel::Medium,
            DefaultDecision::Allow,
        );
        assert_eq!(req.to_claude_code_label(), "WorkspaceWrite");
    }

    #[test]
    fn requirement_label_prompt_decision_overrides_category() {
        let req = PermissionRequirement::new(
            ActionType::FileRead,
            RiskLevel::Safe,
            DefaultDecision::Prompt,
        );
        assert_eq!(req.to_claude_code_label(), "Prompt");
    }

    #[test]
    fn requirement_label_deny_decision_reports_as_prompt() {
        let req = PermissionRequirement::new(
            ActionType::FileWrite,
            RiskLevel::Medium,
            DefaultDecision::Deny,
        );
        // Claude Code's label set has no "Deny" — we report as "Prompt"
        // (callers that need the distinction can read default_decision directly).
        assert_eq!(req.to_claude_code_label(), "Prompt");
    }

    #[test]
    fn requirement_from_policy_derives_risk_and_decision() {
        let policy = AgentPolicy::default(); // Normal: approval above Medium
        let read = ActionDescriptor::new(ActionType::FileRead, "/tmp/a.txt");
        let req = PermissionRequirement::from_policy(&policy, &read);
        assert_eq!(req.action_type, ActionType::FileRead);
        assert_eq!(req.risk, RiskLevel::Safe);
        assert_eq!(req.default_decision, DefaultDecision::Allow);
        assert_eq!(req.to_claude_code_label(), "ReadOnly");
    }

    #[test]
    fn requirement_from_policy_prompts_on_high_risk_under_default() {
        let policy = AgentPolicy::default(); // approve above Medium
        let delete = ActionDescriptor::new(ActionType::FileDelete, "/tmp/a.txt");
        let req = PermissionRequirement::from_policy(&policy, &delete);
        assert_eq!(req.risk, RiskLevel::High);
        assert_eq!(req.default_decision, DefaultDecision::Prompt);
        assert_eq!(req.to_claude_code_label(), "Prompt");
    }

    #[test]
    fn requirement_from_policy_paranoid_prompts_everything() {
        let policy = AgentPolicy::paranoid();
        let read = ActionDescriptor::new(ActionType::FileRead, "/tmp/a.txt");
        let req = PermissionRequirement::from_policy(&policy, &read);
        assert_eq!(req.default_decision, DefaultDecision::Prompt);
        assert_eq!(req.to_claude_code_label(), "Prompt");
    }

    #[test]
    fn requirement_from_policy_autonomous_allows_most_things() {
        let policy = AgentPolicy::autonomous(); // approve only above Critical
        let write = ActionDescriptor::new(ActionType::FileWrite, "/tmp/a.txt");
        let req = PermissionRequirement::from_policy(&policy, &write);
        assert_eq!(req.default_decision, DefaultDecision::Allow);
        assert_eq!(req.to_claude_code_label(), "WorkspaceWrite");
    }
}
