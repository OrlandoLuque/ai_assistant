//! Agent profiles — pre-made configuration profiles
//!
//! Provides reusable configuration profiles for agents, conversations,
//! and multi-step workflows. Includes built-in profiles for common use cases.

use crate::agent_policy::{AgentPolicy, AgentPolicyBuilder, AutonomyLevel, InternetMode, RiskLevel};
use crate::mode_manager::OperationMode;
use std::collections::HashMap;

// ============================================================================
// AgentProfile
// ============================================================================

/// A reusable configuration profile for an autonomous agent.
#[derive(Debug, Clone)]
pub struct AgentProfile {
    /// Unique profile name (e.g. "coding-assistant").
    pub name: String,
    /// Human-readable description of what this profile is for.
    pub description: String,
    /// Security and permission policy for the agent.
    pub policy: AgentPolicy,
    /// Preferred model identifier (e.g. "claude-sonnet-4-20250514").
    pub model: Option<String>,
    /// System prompt injected at the start of each conversation.
    pub system_prompt: Option<String>,
    /// Tool names the agent is expected to use.
    pub tools: Vec<String>,
    /// MCP server names the agent may connect to.
    pub mcp_servers: Vec<String>,
    /// Operation mode the agent runs in.
    pub mode: OperationMode,
    /// Tags for categorisation and search.
    pub tags: Vec<String>,
}

// ============================================================================
// ConversationProfile
// ============================================================================

/// A reusable configuration for conversation style and parameters.
#[derive(Debug, Clone)]
pub struct ConversationProfile {
    /// Unique profile name (e.g. "casual").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// System prompt that sets the conversation tone.
    pub system_prompt: String,
    /// Preferred model identifier.
    pub model: Option<String>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens per response.
    pub max_tokens: Option<usize>,
    /// Response format hint (e.g. "json", "markdown").
    pub response_format: Option<String>,
    /// Tags for categorisation and search.
    pub tags: Vec<String>,
}

// ============================================================================
// WorkflowProfile / WorkflowPhase
// ============================================================================

/// A single phase inside a workflow.
#[derive(Debug, Clone)]
pub struct WorkflowPhase {
    /// Phase name (e.g. "Analyze").
    pub name: String,
    /// Name of the agent profile to use for this phase.
    pub agent_profile: String,
    /// Template string describing the task for this phase.
    pub task_template: String,
    /// Names of phases that must complete before this one starts.
    pub depends_on: Vec<String>,
    /// Key under which this phase stores its output.
    pub output_key: String,
}

/// A multi-phase workflow built from agent profiles.
#[derive(Debug, Clone)]
pub struct WorkflowProfile {
    /// Unique profile name (e.g. "code-review-pipeline").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered list of phases.
    pub phases: Vec<WorkflowPhase>,
    /// Tags for categorisation and search.
    pub tags: Vec<String>,
}

// ============================================================================
// ProfileRegistry
// ============================================================================

/// Central registry that holds agent, conversation, and workflow profiles.
#[derive(Debug, Clone)]
pub struct ProfileRegistry {
    agent_profiles: HashMap<String, AgentProfile>,
    conversation_profiles: HashMap<String, ConversationProfile>,
    workflow_profiles: HashMap<String, WorkflowProfile>,
}

impl ProfileRegistry {
    /// Create an empty registry with no profiles loaded.
    pub fn new() -> Self {
        Self {
            agent_profiles: HashMap::new(),
            conversation_profiles: HashMap::new(),
            workflow_profiles: HashMap::new(),
        }
    }

    /// Create a registry pre-loaded with all built-in profiles.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        for p in builtin_agent_profiles() {
            registry.agent_profiles.insert(p.name.clone(), p);
        }
        for p in builtin_conversation_profiles() {
            registry.conversation_profiles.insert(p.name.clone(), p);
        }
        for p in builtin_workflow_profiles() {
            registry.workflow_profiles.insert(p.name.clone(), p);
        }
        registry
    }

    /// Register a custom agent profile (overwrites if name exists).
    pub fn register_agent_profile(&mut self, profile: AgentProfile) {
        self.agent_profiles.insert(profile.name.clone(), profile);
    }

    /// Register a custom conversation profile (overwrites if name exists).
    pub fn register_conversation_profile(&mut self, profile: ConversationProfile) {
        self.conversation_profiles.insert(profile.name.clone(), profile);
    }

    /// Register a custom workflow profile (overwrites if name exists).
    pub fn register_workflow_profile(&mut self, profile: WorkflowProfile) {
        self.workflow_profiles.insert(profile.name.clone(), profile);
    }

    /// Look up an agent profile by name.
    pub fn get_agent_profile(&self, name: &str) -> Option<&AgentProfile> {
        self.agent_profiles.get(name)
    }

    /// Look up a conversation profile by name.
    pub fn get_conversation_profile(&self, name: &str) -> Option<&ConversationProfile> {
        self.conversation_profiles.get(name)
    }

    /// Look up a workflow profile by name.
    pub fn get_workflow_profile(&self, name: &str) -> Option<&WorkflowProfile> {
        self.workflow_profiles.get(name)
    }

    /// List all registered agent profile names (sorted).
    pub fn list_agent_profiles(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.agent_profiles.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List all registered conversation profile names (sorted).
    pub fn list_conversation_profiles(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.conversation_profiles.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List all registered workflow profile names (sorted).
    pub fn list_workflow_profiles(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.workflow_profiles.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Find agent profiles whose tags contain the given tag.
    /// Returns a sorted list of matching profile names.
    pub fn profiles_by_tag(&self, tag: &str) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .agent_profiles
            .iter()
            .filter(|(_, p)| p.tags.iter().any(|t| t == tag))
            .map(|(name, _)| name.as_str())
            .collect();
        names.sort();
        names
    }
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in agent profiles
// ============================================================================

fn builtin_agent_profiles() -> Vec<AgentProfile> {
    vec![
        // 1. coding-assistant
        AgentProfile {
            name: "coding-assistant".to_string(),
            description: "Programming assistant with filesystem, shell, and git access".to_string(),
            policy: AgentPolicyBuilder::new()
                .allow_command("cargo")
                .allow_command("git")
                .allow_command("rustc")
                .allow_command("npm")
                .allow_command("node")
                .allow_path("/home")
                .max_iterations(100)
                .max_cost(2.0)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a coding assistant. Help the user write, debug, and refactor code. \
                 Use the filesystem and shell tools to explore and modify the project."
                    .to_string(),
            ),
            tools: vec![
                "read_file", "write_file", "run_command", "git_status", "git_diff", "git_log",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Programming,
            tags: vec!["coding".to_string(), "development".to_string()],
        },
        // 2. research-agent
        AgentProfile {
            name: "research-agent".to_string(),
            description: "Research agent with full internet access and read-only filesystem"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .internet(InternetMode::FullAccess)
                .allow_path("/home")
                .max_iterations(80)
                .max_cost(3.0)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a research agent. Search the web, read documents, and synthesize \
                 information into clear, well-sourced reports."
                    .to_string(),
            ),
            tools: vec!["web_search", "web_fetch", "read_file"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Assistant,
            tags: vec!["research".to_string(), "web".to_string()],
        },
        // 3. devops-agent
        AgentProfile {
            name: "devops-agent".to_string(),
            description: "DevOps agent with shell, git, and Docker access at high autonomy"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .autonomy(AutonomyLevel::Autonomous)
                .internet(InternetMode::SearchOnly)
                .allow_command("git")
                .allow_command("docker")
                .allow_command("kubectl")
                .allow_command("terraform")
                .allow_command("make")
                .allow_command("bash")
                .allow_path("/home")
                .allow_path("/var/log")
                .max_iterations(150)
                .max_cost(5.0)
                .max_runtime(1800)
                .require_approval_above(RiskLevel::High)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a DevOps agent. Manage infrastructure, CI/CD pipelines, and \
                 deployments. Exercise caution with destructive operations."
                    .to_string(),
            ),
            tools: vec![
                "run_command", "read_file", "write_file", "git_status", "git_diff",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Autonomous,
            tags: vec!["devops".to_string(), "infrastructure".to_string()],
        },
        // 4. data-analyst
        AgentProfile {
            name: "data-analyst".to_string(),
            description: "Data analyst with read access, Python execution, and analysis tools"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .allow_command("python")
                .allow_command("python3")
                .allow_command("pip")
                .allow_path("/home")
                .max_iterations(60)
                .max_cost(2.0)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a data analyst. Read data files, run Python scripts for analysis, \
                 and produce clear visualisations and summaries."
                    .to_string(),
            ),
            tools: vec!["read_file", "run_command", "write_file"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Programming,
            tags: vec!["data".to_string(), "analysis".to_string()],
        },
        // 5. content-writer
        AgentProfile {
            name: "content-writer".to_string(),
            description: "Content writer with internet search and file writing, no shell access"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .internet(InternetMode::SearchOnly)
                .allow_path("/home")
                .max_iterations(50)
                .max_cost(1.5)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a content writer. Research topics via web search and produce \
                 well-structured articles, blog posts, and documentation."
                    .to_string(),
            ),
            tools: vec!["web_search", "read_file", "write_file"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Assistant,
            tags: vec!["writing".to_string(), "content".to_string()],
        },
        // 6. code-reviewer
        AgentProfile {
            name: "code-reviewer".to_string(),
            description: "Code reviewer with read-only access and git diff tools".to_string(),
            policy: AgentPolicyBuilder::new()
                .allow_command("git")
                .allow_path("/home")
                .max_iterations(40)
                .max_cost(1.0)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a code reviewer. Read source files, analyse git diffs, and provide \
                 detailed, constructive feedback on code quality, correctness, and style."
                    .to_string(),
            ),
            tools: vec!["read_file", "git_diff", "git_log", "git_status"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Programming,
            tags: vec!["coding".to_string(), "review".to_string()],
        },
        // 7. sysadmin
        AgentProfile {
            name: "sysadmin".to_string(),
            description: "System administrator with full shell, network, and monitoring tools"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .autonomy(AutonomyLevel::Autonomous)
                .internet(InternetMode::FullAccess)
                .allow_command("systemctl")
                .allow_command("journalctl")
                .allow_command("top")
                .allow_command("htop")
                .allow_command("ps")
                .allow_command("netstat")
                .allow_command("ss")
                .allow_command("curl")
                .allow_command("ping")
                .allow_command("bash")
                .allow_path("/var/log")
                .allow_path("/etc")
                .allow_path("/home")
                .max_iterations(120)
                .max_cost(4.0)
                .max_runtime(3600)
                .require_approval_above(RiskLevel::High)
                .build(),
            model: None,
            system_prompt: Some(
                "You are a system administrator. Monitor servers, diagnose issues, \
                 and manage system configurations. Always explain changes before applying them."
                    .to_string(),
            ),
            tools: vec!["run_command", "read_file", "write_file"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Autonomous,
            tags: vec!["sysadmin".to_string(), "infrastructure".to_string()],
        },
        // 8. paranoid
        AgentProfile {
            name: "paranoid".to_string(),
            description: "Paranoid profile — approval required for every action, minimal tools"
                .to_string(),
            policy: AgentPolicyBuilder::new()
                .autonomy(AutonomyLevel::Paranoid)
                .internet(InternetMode::Disabled)
                .max_iterations(10)
                .max_cost(0.10)
                .max_runtime(120)
                .require_approval_above(RiskLevel::Safe)
                .build(),
            model: None,
            system_prompt: Some(
                "You operate under maximum restrictions. Every action requires explicit \
                 user approval. Use only the minimal tools available."
                    .to_string(),
            ),
            tools: vec!["read_file".to_string()],
            mcp_servers: Vec::new(),
            mode: OperationMode::Chat,
            tags: vec!["security".to_string(), "restricted".to_string()],
        },
    ]
}

// ============================================================================
// Built-in conversation profiles
// ============================================================================

fn builtin_conversation_profiles() -> Vec<ConversationProfile> {
    vec![
        // 1. casual
        ConversationProfile {
            name: "casual".to_string(),
            description: "Friendly, creative conversation style".to_string(),
            system_prompt: "You are a friendly, approachable assistant. Keep your tone \
                            casual and conversational. Feel free to be creative and playful."
                .to_string(),
            model: None,
            temperature: Some(0.9),
            max_tokens: None,
            response_format: None,
            tags: vec!["chat".to_string(), "creative".to_string()],
        },
        // 2. technical
        ConversationProfile {
            name: "technical".to_string(),
            description: "Precise, low-temperature technical answers".to_string(),
            system_prompt: "You are a precise technical assistant. Provide accurate, \
                            well-structured answers. Cite sources when possible. \
                            Avoid speculation."
                .to_string(),
            model: None,
            temperature: Some(0.3),
            max_tokens: None,
            response_format: Some("markdown".to_string()),
            tags: vec!["technical".to_string(), "precise".to_string()],
        },
        // 3. brainstorm
        ConversationProfile {
            name: "brainstorm".to_string(),
            description: "High temperature brainstorming — expansive and creative".to_string(),
            system_prompt: "You are a brainstorming partner. Generate many diverse ideas. \
                            Think outside the box, combine unexpected concepts, and explore \
                            unconventional approaches. Quantity over perfection."
                .to_string(),
            model: None,
            temperature: Some(1.1),
            max_tokens: None,
            response_format: None,
            tags: vec!["creative".to_string(), "brainstorm".to_string()],
        },
        // 4. interview
        ConversationProfile {
            name: "interview".to_string(),
            description: "Structured Q&A interview format".to_string(),
            system_prompt: "You are conducting a structured interview. Ask one clear \
                            question at a time. Listen carefully to answers and follow up \
                            with relevant probing questions. Summarise key points periodically."
                .to_string(),
            model: None,
            temperature: Some(0.5),
            max_tokens: None,
            response_format: None,
            tags: vec!["interview".to_string(), "structured".to_string()],
        },
    ]
}

// ============================================================================
// Built-in workflow profiles
// ============================================================================

fn builtin_workflow_profiles() -> Vec<WorkflowProfile> {
    vec![
        // 1. code-review-pipeline
        WorkflowProfile {
            name: "code-review-pipeline".to_string(),
            description: "Automated code review: Analyze, Review, Suggest, Report".to_string(),
            phases: vec![
                WorkflowPhase {
                    name: "Analyze".to_string(),
                    agent_profile: "code-reviewer".to_string(),
                    task_template: "Analyze the codebase structure and identify files changed in the latest commits.".to_string(),
                    depends_on: Vec::new(),
                    output_key: "analysis".to_string(),
                },
                WorkflowPhase {
                    name: "Review".to_string(),
                    agent_profile: "code-reviewer".to_string(),
                    task_template: "Review each changed file for bugs, style issues, and security concerns. Use {{analysis}} as context.".to_string(),
                    depends_on: vec!["Analyze".to_string()],
                    output_key: "review".to_string(),
                },
                WorkflowPhase {
                    name: "Suggest".to_string(),
                    agent_profile: "coding-assistant".to_string(),
                    task_template: "Based on {{review}}, draft concrete code suggestions and improvements.".to_string(),
                    depends_on: vec!["Review".to_string()],
                    output_key: "suggestions".to_string(),
                },
                WorkflowPhase {
                    name: "Report".to_string(),
                    agent_profile: "content-writer".to_string(),
                    task_template: "Compile {{analysis}}, {{review}}, and {{suggestions}} into a final review report.".to_string(),
                    depends_on: vec!["Suggest".to_string()],
                    output_key: "report".to_string(),
                },
            ],
            tags: vec!["coding".to_string(), "review".to_string(), "pipeline".to_string()],
        },
        // 2. research-report
        WorkflowProfile {
            name: "research-report".to_string(),
            description: "Research pipeline: Search, Analyze, Synthesize, Write".to_string(),
            phases: vec![
                WorkflowPhase {
                    name: "Search".to_string(),
                    agent_profile: "research-agent".to_string(),
                    task_template: "Search the web for information on the given topic. Collect diverse sources.".to_string(),
                    depends_on: Vec::new(),
                    output_key: "sources".to_string(),
                },
                WorkflowPhase {
                    name: "Analyze".to_string(),
                    agent_profile: "research-agent".to_string(),
                    task_template: "Analyze {{sources}} for key themes, contradictions, and gaps.".to_string(),
                    depends_on: vec!["Search".to_string()],
                    output_key: "analysis".to_string(),
                },
                WorkflowPhase {
                    name: "Synthesize".to_string(),
                    agent_profile: "data-analyst".to_string(),
                    task_template: "Synthesize {{analysis}} into a coherent narrative with supporting data.".to_string(),
                    depends_on: vec!["Analyze".to_string()],
                    output_key: "synthesis".to_string(),
                },
                WorkflowPhase {
                    name: "Write".to_string(),
                    agent_profile: "content-writer".to_string(),
                    task_template: "Write a polished report from {{synthesis}}. Include executive summary and references.".to_string(),
                    depends_on: vec!["Synthesize".to_string()],
                    output_key: "report".to_string(),
                },
            ],
            tags: vec!["research".to_string(), "writing".to_string(), "pipeline".to_string()],
        },
        // 3. bug-fix
        WorkflowProfile {
            name: "bug-fix".to_string(),
            description: "Bug fix pipeline: Reproduce, Diagnose, Fix, Test, Commit".to_string(),
            phases: vec![
                WorkflowPhase {
                    name: "Reproduce".to_string(),
                    agent_profile: "coding-assistant".to_string(),
                    task_template: "Reproduce the reported bug. Write a minimal test case that demonstrates the failure.".to_string(),
                    depends_on: Vec::new(),
                    output_key: "reproduction".to_string(),
                },
                WorkflowPhase {
                    name: "Diagnose".to_string(),
                    agent_profile: "code-reviewer".to_string(),
                    task_template: "Using {{reproduction}}, trace the root cause of the bug through the codebase.".to_string(),
                    depends_on: vec!["Reproduce".to_string()],
                    output_key: "diagnosis".to_string(),
                },
                WorkflowPhase {
                    name: "Fix".to_string(),
                    agent_profile: "coding-assistant".to_string(),
                    task_template: "Based on {{diagnosis}}, implement the fix. Modify only the necessary files.".to_string(),
                    depends_on: vec!["Diagnose".to_string()],
                    output_key: "fix".to_string(),
                },
                WorkflowPhase {
                    name: "Test".to_string(),
                    agent_profile: "coding-assistant".to_string(),
                    task_template: "Run the test suite to verify {{fix}} resolves the bug without regressions.".to_string(),
                    depends_on: vec!["Fix".to_string()],
                    output_key: "test_results".to_string(),
                },
                WorkflowPhase {
                    name: "Commit".to_string(),
                    agent_profile: "devops-agent".to_string(),
                    task_template: "If {{test_results}} pass, commit the fix with a descriptive message.".to_string(),
                    depends_on: vec!["Test".to_string()],
                    output_key: "commit".to_string(),
                },
            ],
            tags: vec!["coding".to_string(), "bugfix".to_string(), "pipeline".to_string()],
        },
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry() {
        let reg = ProfileRegistry::new();
        assert!(reg.list_agent_profiles().is_empty());
        assert!(reg.list_conversation_profiles().is_empty());
        assert!(reg.list_workflow_profiles().is_empty());
        assert!(reg.get_agent_profile("coding-assistant").is_none());
    }

    #[test]
    fn test_with_defaults_has_agent_profiles() {
        let reg = ProfileRegistry::with_defaults();
        let names = reg.list_agent_profiles();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"coding-assistant"));
        assert!(names.contains(&"research-agent"));
        assert!(names.contains(&"devops-agent"));
        assert!(names.contains(&"data-analyst"));
        assert!(names.contains(&"content-writer"));
        assert!(names.contains(&"code-reviewer"));
        assert!(names.contains(&"sysadmin"));
        assert!(names.contains(&"paranoid"));
    }

    #[test]
    fn test_with_defaults_has_conversation_profiles() {
        let reg = ProfileRegistry::with_defaults();
        let names = reg.list_conversation_profiles();
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"casual"));
        assert!(names.contains(&"technical"));
        assert!(names.contains(&"brainstorm"));
        assert!(names.contains(&"interview"));
    }

    #[test]
    fn test_with_defaults_has_workflow_profiles() {
        let reg = ProfileRegistry::with_defaults();
        let names = reg.list_workflow_profiles();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"code-review-pipeline"));
        assert!(names.contains(&"research-report"));
        assert!(names.contains(&"bug-fix"));
    }

    #[test]
    fn test_register_custom_agent_profile() {
        let mut reg = ProfileRegistry::new();
        assert!(reg.get_agent_profile("my-agent").is_none());

        reg.register_agent_profile(AgentProfile {
            name: "my-agent".to_string(),
            description: "Custom agent".to_string(),
            policy: AgentPolicyBuilder::new().max_iterations(5).build(),
            model: Some("custom-model".to_string()),
            system_prompt: None,
            tools: vec!["read_file".to_string()],
            mcp_servers: Vec::new(),
            mode: OperationMode::Chat,
            tags: vec!["custom".to_string()],
        });

        let profile = reg.get_agent_profile("my-agent").unwrap();
        assert_eq!(profile.description, "Custom agent");
        assert_eq!(profile.model, Some("custom-model".to_string()));
        assert_eq!(profile.policy.max_iterations, 5);
    }

    #[test]
    fn test_register_custom_conversation_profile() {
        let mut reg = ProfileRegistry::new();
        assert!(reg.get_conversation_profile("my-conv").is_none());

        reg.register_conversation_profile(ConversationProfile {
            name: "my-conv".to_string(),
            description: "Custom conversation".to_string(),
            system_prompt: "Be helpful.".to_string(),
            model: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            response_format: Some("json".to_string()),
            tags: vec!["custom".to_string()],
        });

        let profile = reg.get_conversation_profile("my-conv").unwrap();
        assert_eq!(profile.temperature, Some(0.7));
        assert_eq!(profile.max_tokens, Some(4096));
        assert_eq!(profile.response_format, Some("json".to_string()));
    }

    #[test]
    fn test_get_coding_assistant() {
        let reg = ProfileRegistry::with_defaults();
        let profile = reg.get_agent_profile("coding-assistant").unwrap();
        assert_eq!(profile.mode, OperationMode::Programming);
        assert!(profile.tools.contains(&"read_file".to_string()));
        assert!(profile.tools.contains(&"write_file".to_string()));
        assert!(profile.tools.contains(&"run_command".to_string()));
        assert!(profile.policy.can_run_command("cargo build"));
        assert!(profile.policy.can_run_command("git status"));
        assert!(!profile.policy.can_run_command("rm -rf /"));
    }

    #[test]
    fn test_get_research_agent() {
        let reg = ProfileRegistry::with_defaults();
        let profile = reg.get_agent_profile("research-agent").unwrap();
        assert_eq!(profile.mode, OperationMode::Assistant);
        assert_eq!(profile.policy.internet, InternetMode::FullAccess);
        assert!(profile.tools.contains(&"web_search".to_string()));
        assert!(profile.tools.contains(&"web_fetch".to_string()));
        // research-agent has no allowed commands, so shell is denied
        assert!(!profile.policy.can_run_command("bash"));
    }

    #[test]
    fn test_profiles_by_tag() {
        let reg = ProfileRegistry::with_defaults();

        let coding = reg.profiles_by_tag("coding");
        assert!(coding.contains(&"coding-assistant"));
        assert!(coding.contains(&"code-reviewer"));
        assert!(!coding.contains(&"research-agent"));

        let infra = reg.profiles_by_tag("infrastructure");
        assert!(infra.contains(&"devops-agent"));
        assert!(infra.contains(&"sysadmin"));

        let empty = reg.profiles_by_tag("nonexistent-tag");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_list_profiles() {
        let reg = ProfileRegistry::with_defaults();

        // Agent profiles are sorted alphabetically
        let agents = reg.list_agent_profiles();
        assert_eq!(agents[0], "code-reviewer");
        assert_eq!(agents[agents.len() - 1], "sysadmin");

        // Conversation profiles are sorted
        let convs = reg.list_conversation_profiles();
        assert_eq!(convs[0], "brainstorm");
        assert_eq!(convs[convs.len() - 1], "technical");

        // Workflow profiles are sorted
        let wfs = reg.list_workflow_profiles();
        assert_eq!(wfs[0], "bug-fix");
        assert_eq!(wfs[wfs.len() - 1], "research-report");
    }

    #[test]
    fn test_workflow_phases() {
        let reg = ProfileRegistry::with_defaults();
        let wf = reg.get_workflow_profile("bug-fix").unwrap();
        assert_eq!(wf.phases.len(), 5);

        // First phase has no dependencies
        assert!(wf.phases[0].depends_on.is_empty());
        assert_eq!(wf.phases[0].name, "Reproduce");
        assert_eq!(wf.phases[0].output_key, "reproduction");

        // Last phase depends on Test
        let commit = &wf.phases[4];
        assert_eq!(commit.name, "Commit");
        assert_eq!(commit.depends_on, vec!["Test".to_string()]);
        assert_eq!(commit.agent_profile, "devops-agent");

        // Verify chain: each phase depends on the previous
        for i in 1..wf.phases.len() {
            assert!(wf.phases[i]
                .depends_on
                .contains(&wf.phases[i - 1].name));
        }
    }

    #[test]
    fn test_paranoid_profile_policy() {
        let reg = ProfileRegistry::with_defaults();
        let profile = reg.get_agent_profile("paranoid").unwrap();

        assert_eq!(profile.policy.autonomy, AutonomyLevel::Paranoid);
        assert_eq!(profile.policy.internet, InternetMode::Disabled);
        assert_eq!(profile.policy.require_approval_above, RiskLevel::Safe);
        assert_eq!(profile.policy.max_iterations, 10);
        assert!(profile.policy.max_cost_usd <= 0.10 + f64::EPSILON);
        assert_eq!(profile.policy.max_runtime_secs, 120);
        assert_eq!(profile.mode, OperationMode::Chat);
        // No shell commands allowed
        assert!(!profile.policy.can_run_command("ls"));
        // No internet
        assert!(!profile.policy.can_access_internet("https://example.com"));
    }

    #[test]
    fn test_profiles_by_tag_custom() {
        let mut reg = ProfileRegistry::new();

        // Register profiles with specific tags
        reg.register_agent_profile(AgentProfile {
            name: "alpha".to_string(),
            description: "Alpha agent".to_string(),
            policy: AgentPolicyBuilder::new().build(),
            model: None,
            system_prompt: None,
            tools: Vec::new(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Chat,
            tags: vec!["fast".to_string(), "general".to_string()],
        });

        reg.register_agent_profile(AgentProfile {
            name: "beta".to_string(),
            description: "Beta agent".to_string(),
            policy: AgentPolicyBuilder::new().build(),
            model: None,
            system_prompt: None,
            tools: Vec::new(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Chat,
            tags: vec!["fast".to_string(), "specialized".to_string()],
        });

        reg.register_agent_profile(AgentProfile {
            name: "gamma".to_string(),
            description: "Gamma agent".to_string(),
            policy: AgentPolicyBuilder::new().build(),
            model: None,
            system_prompt: None,
            tools: Vec::new(),
            mcp_servers: Vec::new(),
            mode: OperationMode::Chat,
            tags: vec!["slow".to_string(), "specialized".to_string()],
        });

        // "fast" tag should return alpha and beta
        let fast = reg.profiles_by_tag("fast");
        assert_eq!(fast.len(), 2);
        assert!(fast.contains(&"alpha"));
        assert!(fast.contains(&"beta"));

        // "specialized" tag should return beta and gamma
        let specialized = reg.profiles_by_tag("specialized");
        assert_eq!(specialized.len(), 2);
        assert!(specialized.contains(&"beta"));
        assert!(specialized.contains(&"gamma"));

        // "nonexistent" tag should return empty
        assert!(reg.profiles_by_tag("nonexistent").is_empty());
    }

    #[test]
    fn test_workflow_phase_structure() {
        let workflow = WorkflowProfile {
            name: "test-pipeline".to_string(),
            description: "Test pipeline".to_string(),
            phases: vec![
                WorkflowPhase {
                    name: "Init".to_string(),
                    agent_profile: "coding-assistant".to_string(),
                    task_template: "Initialize the project".to_string(),
                    depends_on: Vec::new(),
                    output_key: "init_result".to_string(),
                },
                WorkflowPhase {
                    name: "Build".to_string(),
                    agent_profile: "devops-agent".to_string(),
                    task_template: "Build from {{init_result}}".to_string(),
                    depends_on: vec!["Init".to_string()],
                    output_key: "build_result".to_string(),
                },
                WorkflowPhase {
                    name: "Deploy".to_string(),
                    agent_profile: "devops-agent".to_string(),
                    task_template: "Deploy {{build_result}}".to_string(),
                    depends_on: vec!["Build".to_string(), "Init".to_string()],
                    output_key: "deploy_result".to_string(),
                },
            ],
            tags: vec!["ci".to_string()],
        };

        // Verify phase count
        assert_eq!(workflow.phases.len(), 3);

        // First phase has no dependencies
        assert!(workflow.phases[0].depends_on.is_empty());
        assert_eq!(workflow.phases[0].name, "Init");

        // Second phase depends on Init
        assert_eq!(workflow.phases[1].depends_on, vec!["Init".to_string()]);

        // Third phase depends on both Build and Init
        assert_eq!(workflow.phases[2].depends_on.len(), 2);
        assert!(workflow.phases[2].depends_on.contains(&"Build".to_string()));
        assert!(workflow.phases[2].depends_on.contains(&"Init".to_string()));
    }

    #[test]
    fn test_conversation_profile_defaults() {
        let reg = ProfileRegistry::with_defaults();

        // Verify casual profile has high temperature
        let casual = reg.get_conversation_profile("casual").unwrap();
        assert_eq!(casual.temperature, Some(0.9));
        assert!(!casual.system_prompt.is_empty());
        assert!(casual.tags.contains(&"chat".to_string()));

        // Verify technical profile has low temperature and markdown format
        let technical = reg.get_conversation_profile("technical").unwrap();
        assert_eq!(technical.temperature, Some(0.3));
        assert_eq!(technical.response_format, Some("markdown".to_string()));
        assert!(technical.tags.contains(&"technical".to_string()));

        // Verify brainstorm profile has highest temperature
        let brainstorm = reg.get_conversation_profile("brainstorm").unwrap();
        assert_eq!(brainstorm.temperature, Some(1.1));
        assert!(brainstorm.tags.contains(&"creative".to_string()));

        // Verify interview profile has moderate temperature
        let interview = reg.get_conversation_profile("interview").unwrap();
        assert_eq!(interview.temperature, Some(0.5));
        assert!(interview.tags.contains(&"structured".to_string()));
    }
}
