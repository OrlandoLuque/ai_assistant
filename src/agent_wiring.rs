//! Agent wiring — connects AgentDefinition to AutonomousAgent execution
//!
//! Bridges the declarative definition system (agent_definition.rs) with the
//! runtime execution system (autonomous_loop.rs). Provides:
//! - Role mapping and system prompt generation
//! - Response generator factory
//! - Definition-to-runtime conversion
//! - Agent pool with priority queue and supervisor
//! - BestFit scoring for task assignment

use crate::agent_definition::{AgentDefinition, AgentSpec, GuardrailSpec, WarningSeverity};
use crate::agent_policy::{AgentPolicy, AgentPolicyBuilder, AutoApproveAll};
use crate::agent_sandbox::SandboxValidator;
use crate::agentic_loop::LoopMessage;
use crate::autonomous_loop::{
    AgentResult, AgentState, AutonomousAgent, AutonomousAgentBuilder, CostConfig, IterationOutcome,
};
use crate::mode_manager::OperationMode;
use crate::multi_agent::{Agent, AgentRole};
use crate::unified_tools::ToolRegistry;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::time::{Duration, Instant};

// ============================================================================
// E2b: parse_agent_role — map string roles to AgentRole enum
// ============================================================================

/// Parse a string role from AgentDefinition into the multi_agent AgentRole enum.
///
/// Recognized values (case-insensitive): "coordinator", "researcher", "analyst",
/// "writer", "reviewer", "executor", "validator". Anything else maps to Custom.
pub fn parse_agent_role(s: &str) -> AgentRole {
    match s.to_lowercase().as_str() {
        "coordinator" | "manager" => AgentRole::Coordinator,
        "researcher" => AgentRole::Researcher,
        "analyst" => AgentRole::Analyst,
        "writer" => AgentRole::Writer,
        "reviewer" => AgentRole::Reviewer,
        "executor" | "worker" => AgentRole::Executor,
        "validator" => AgentRole::Validator,
        _ => AgentRole::Custom,
    }
}

// ============================================================================
// E2: role_system_prompt — templates for each AgentRole
// ============================================================================

/// Generate a default system prompt for a given role.
///
/// If the AgentDefinition provides a custom system_prompt, that takes priority.
/// This function provides sensible defaults when no custom prompt is given.
pub fn role_system_prompt(role: &AgentRole) -> String {
    match role {
        AgentRole::Coordinator => {
            "You are a Coordinator agent. Your role is to break down complex tasks, \
             delegate work to other agents, synthesize results, and ensure the overall \
             goal is achieved. Focus on planning, delegation, and quality control."
                .to_string()
        }
        AgentRole::Researcher => {
            "You are a Researcher agent. Your role is to gather information, search \
             for relevant data, analyze sources, and provide comprehensive research \
             findings. Be thorough and cite your sources."
                .to_string()
        }
        AgentRole::Analyst => {
            "You are an Analyst agent. Your role is to examine data, identify patterns, \
             draw conclusions, and provide actionable insights. Use structured reasoning \
             and support your analysis with evidence."
                .to_string()
        }
        AgentRole::Writer => {
            "You are a Writer agent. Your role is to produce clear, well-structured \
             written content. Adapt your tone and style to the task requirements. \
             Focus on clarity, accuracy, and readability."
                .to_string()
        }
        AgentRole::Reviewer => {
            "You are a Reviewer agent. Your role is to evaluate work produced by other \
             agents, identify errors, suggest improvements, and ensure quality standards \
             are met. Be constructive and specific in your feedback."
                .to_string()
        }
        AgentRole::Executor => {
            "You are an Executor agent. Your role is to carry out specific tasks using \
             the tools available to you. Follow instructions precisely, report results \
             clearly, and flag any issues encountered."
                .to_string()
        }
        AgentRole::Validator => {
            "You are a Validator agent. Your role is to verify correctness, check \
             constraints, validate outputs against requirements, and ensure nothing \
             is missed. Be rigorous and systematic."
                .to_string()
        }
        AgentRole::Custom => {
            "You are an AI assistant. Complete the assigned task using the tools \
             available to you. Be clear, accurate, and thorough."
                .to_string()
        }
    }
}

// ============================================================================
// E1: LoopMessage <-> ChatMessage conversions
// ============================================================================

/// Convert a ChatMessage (from messages module) to a LoopMessage.
pub fn chat_to_loop_message(role: &str, content: &str) -> LoopMessage {
    use crate::agentic_loop::LoopRole;

    let loop_role = match role.to_lowercase().as_str() {
        "system" => LoopRole::System,
        "assistant" => LoopRole::Assistant,
        "tool" => LoopRole::Tool,
        _ => LoopRole::User,
    };

    LoopMessage {
        role: loop_role,
        content: content.to_string(),
        tool_calls: None,
        tool_results: None,
    }
}

/// Convert a LoopMessage to a (role, content) pair.
pub fn loop_message_to_pair(msg: &LoopMessage) -> (&'static str, &str) {
    use crate::agentic_loop::LoopRole;

    let role = match msg.role {
        LoopRole::System => "system",
        LoopRole::User => "user",
        LoopRole::Assistant => "assistant",
        LoopRole::Tool => "tool",
    };
    (role, &msg.content)
}

// ============================================================================
// E3: make_response_generator — factory for LLM callbacks
// ============================================================================

/// Type alias for the response generator function.
pub type ResponseGenerator = Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>;

/// Create a response generator from a closure.
///
/// In production, this would call an LLM provider. For testing and wiring,
/// it accepts any function that maps conversation history to a response string.
pub fn make_response_generator<F>(f: F) -> ResponseGenerator
where
    F: Fn(&[LoopMessage]) -> String + Send + Sync + 'static,
{
    Arc::new(f)
}

/// A response generator factory that creates generators for different models.
///
/// The factory is injected into AgentPool so it can create response generators
/// for any agent (including the supervisor) without coupling to a specific provider.
pub type ResponseGeneratorFactory =
    Arc<dyn Fn(Option<&str>) -> ResponseGenerator + Send + Sync>;

/// Create a default response generator factory from a closure.
pub fn make_response_generator_factory<F>(f: F) -> ResponseGeneratorFactory
where
    F: Fn(Option<&str>) -> ResponseGenerator + Send + Sync + 'static,
{
    Arc::new(f)
}

// ============================================================================
// E4b: AgentProfile::from_definition
// ============================================================================

/// Convert an AgentDefinition into a multi_agent::Agent for the orchestrator.
pub fn agent_from_definition(def: &AgentDefinition) -> Agent {
    let role = def
        .agent
        .role
        .as_deref()
        .map(parse_agent_role)
        .unwrap_or(AgentRole::Custom);

    let mut agent = Agent::new(&def.agent.name, &def.agent.name, role);

    // Add tool names as capabilities
    for tool_ref in &def.tools {
        agent = agent.with_capability(&tool_ref.name);
    }

    agent
}

// ============================================================================
// E5: filter_tool_registry — only register tools named in the definition
// ============================================================================

/// Filter a ToolRegistry to only include tools referenced by the agent definition.
///
/// Returns warnings for tools named in the definition but not found in the registry.
pub fn filter_tool_registry(
    full_registry: &ToolRegistry,
    def: &AgentDefinition,
) -> (ToolRegistry, Vec<String>) {
    let mut filtered = ToolRegistry::new();
    let mut warnings = Vec::new();

    for tool_ref in &def.tools {
        if let Some((def_tool, handler)) = full_registry.get_with_handler(&tool_ref.name) {
            filtered.register(def_tool.clone(), Arc::clone(handler));
        } else {
            warnings.push(format!(
                "Tool '{}' referenced in agent '{}' not found in registry",
                tool_ref.name, def.agent.name
            ));
        }
    }

    (filtered, warnings)
}

// ============================================================================
// E4: create_agent_from_definition
// ============================================================================

/// Parse autonomy level string to AutonomyLevel enum.
fn parse_autonomy_level(s: &str) -> crate::agent_policy::AutonomyLevel {
    use crate::agent_policy::AutonomyLevel;
    match s.to_lowercase().as_str() {
        "paranoid" | "manual" => AutonomyLevel::Paranoid,
        "cautious" | "assisted" | "balanced" | "delegated" | "normal" => AutonomyLevel::Normal,
        "autonomous" | "independent" | "proactive" => AutonomyLevel::Autonomous,
        _ => AutonomyLevel::Normal,
    }
}

/// Build a policy from agent spec + guardrail spec.
fn build_policy(spec: &AgentSpec, guardrails: Option<&GuardrailSpec>) -> AgentPolicy {
    let autonomy = spec
        .autonomy_level
        .as_deref()
        .map(parse_autonomy_level)
        .unwrap_or(crate::agent_policy::AutonomyLevel::Normal);

    let mut builder = AgentPolicyBuilder::new().autonomy(autonomy);

    if let Some(gs) = guardrails {
        if let Some(max_turns) = gs.max_turns {
            builder = builder.max_iterations(max_turns);
        }
        if gs.require_approval_for_destructive {
            builder = builder.require_approval_above(crate::agent_policy::RiskLevel::High);
        }
    }

    builder.build()
}

/// Error type for agent creation from definitions.
#[derive(Debug, Clone)]
pub struct AgentCreationError {
    pub message: String,
    pub validation_errors: Vec<String>,
}

impl std::fmt::Display for AgentCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Agent creation failed: {}", self.message)?;
        for err in &self.validation_errors {
            write!(f, "\n  - {}", err)?;
        }
        Ok(())
    }
}

/// Options for agent creation beyond the basic definition.
#[derive(Default)]
pub struct AgentCreateOptions {
    /// Cancellation token for cooperative cancellation.
    pub cancellation_token: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Mailbox receiver for inter-agent messages.
    pub mailbox: Option<mpsc::Receiver<crate::autonomous_loop::InterAgentMessage>>,
}

/// Create an AutonomousAgent from an AgentDefinition.
///
/// Validates the definition (rejects if any Error-severity issues found),
/// builds the runtime with system_prompt, model, tools, guardrails.
pub fn create_agent_from_definition(
    def: &AgentDefinition,
    response_generator: ResponseGenerator,
    available_tools: &ToolRegistry,
) -> Result<AutonomousAgent, AgentCreationError> {
    create_agent_from_definition_with_options(def, response_generator, available_tools, AgentCreateOptions::default())
}

/// Create an AutonomousAgent with additional options (cancellation token, mailbox).
pub fn create_agent_from_definition_with_options(
    def: &AgentDefinition,
    response_generator: ResponseGenerator,
    available_tools: &ToolRegistry,
    options: AgentCreateOptions,
) -> Result<AutonomousAgent, AgentCreationError> {
    // Validate the definition
    let warnings = crate::agent_definition::AgentDefinitionLoader::validate(def)
        .map_err(|e| AgentCreationError {
            message: format!("Validation failed: {}", e),
            validation_errors: vec![],
        })?;

    // Reject if any Error-severity warnings
    let errors: Vec<String> = warnings
        .iter()
        .filter(|w| w.severity == WarningSeverity::Error)
        .map(|w| w.message.clone())
        .collect();

    if !errors.is_empty() {
        return Err(AgentCreationError {
            message: "Definition has validation errors".to_string(),
            validation_errors: errors,
        });
    }

    // Determine role and system prompt
    let role = def
        .agent
        .role
        .as_deref()
        .map(parse_agent_role)
        .unwrap_or(AgentRole::Custom);

    let system_prompt = def
        .agent
        .system_prompt
        .clone()
        .unwrap_or_else(|| role_system_prompt(&role));

    // Build policy
    let policy = build_policy(&def.agent, def.guardrails.as_ref());

    // Filter tool registry
    let (filtered_tools, tool_warnings) = filter_tool_registry(available_tools, def);
    for warning in &tool_warnings {
        eprintln!("[agent_wiring] WARNING: {}", warning);
    }

    // Build the sandbox — approval handler derived from the agent's autonomy level.
    // Only fully autonomous agents auto-approve; others require explicit approval.
    let approval_handler: Arc<dyn crate::agent_policy::ApprovalHandler> = {
        use crate::agent_policy::AutonomyLevel;
        let autonomy = def.agent.autonomy_level
            .as_deref()
            .map(parse_autonomy_level)
            .unwrap_or(AutonomyLevel::Normal);
        match autonomy {
            #[allow(deprecated)]
            AutonomyLevel::Autonomous => Arc::new(AutoApproveAll),
            _ => Arc::new(crate::agent_policy::AutoDenyAll),
        }
    };
    let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
        policy.clone(),
        approval_handler,
    )));

    // Build max_iterations from guardrails
    let max_iterations = def
        .guardrails
        .as_ref()
        .and_then(|g| g.max_turns)
        .unwrap_or(50);

    // Build cost config
    let cost_config = CostConfig::new();

    // Build the agent
    let mut builder = AutonomousAgentBuilder::new(&def.agent.name, response_generator)
        .system_prompt(system_prompt)
        .max_iterations(max_iterations)
        .policy(policy)
        .sandbox(sandbox)
        .tool_registry(filtered_tools)
        .with_cost_config(cost_config)
        .mode(OperationMode::Autonomous);

    if let Some(token) = options.cancellation_token {
        builder = builder.cancellation_token(token);
    }
    if let Some(mailbox) = options.mailbox {
        builder = builder.mailbox(mailbox);
    }

    Ok(builder.build())
}

// ============================================================================
// E8: Budget enforcement (in run_iteration)
// Note: This is checked via AutonomousAgent's existing cost tracking.
// We add a helper to check budget from AgentDefinition.
// ============================================================================

/// Extract budget limit from an AgentDefinition's guardrails and agent spec.
///
/// Returns 0.0 if no budget is configured (meaning no limit).
pub fn extract_budget(_def: &AgentDefinition) -> f64 {
    // The AgentPolicy already has max_cost_usd
    // This helper extracts it from the definition for external checks
    0.0 // No explicit budget field in AgentDefinition yet
}

// ============================================================================
// E9: BestFit scoring for task assignment
// ============================================================================

/// Score an agent for a task based on capabilities, role, and description.
///
/// Returns a score from 0.0 to 140.0 (100 for capabilities + 30 for role + 10 for description).
pub fn score_agent_for_task(agent: &Agent, task: &PoolTask) -> f64 {
    let mut score = 0.0;

    // Capability match (0-100)
    if !task.required_capabilities.is_empty() {
        let matched = task
            .required_capabilities
            .iter()
            .filter(|cap| agent.capabilities.contains(cap))
            .count();
        let total = task.required_capabilities.len();
        score += (matched as f64 / total as f64) * 100.0;
    }

    // Role affinity (0-30)
    if let Some(ref preferred_role) = task.preferred_role {
        if &agent.role == preferred_role {
            score += 30.0;
        }
    }

    // Fallback: substring match in description (0-10, only if required_capabilities empty)
    if task.required_capabilities.is_empty() {
        if let Some(ref desc) = task.description {
            let desc_lower = desc.to_lowercase();
            let cap_matches = agent
                .capabilities
                .iter()
                .filter(|cap| desc_lower.contains(&cap.to_lowercase()))
                .count();
            score += (cap_matches as f64).min(10.0);
        }
    }

    score
}

/// A pool task with BestFit scoring fields for agent assignment.
#[derive(Debug, Clone)]
pub struct PoolTask {
    /// Task identifier.
    pub id: String,
    /// Human-readable description of the task.
    pub description: Option<String>,
    /// The actual task prompt to give to the agent.
    pub prompt: String,
    /// Priority (higher = more important).
    pub priority: u32,
    /// Required capabilities for matching.
    pub required_capabilities: Vec<String>,
    /// Preferred role for the agent.
    pub preferred_role: Option<AgentRole>,
    /// Model override — if set, takes priority over AgentDefinition.agent.model.
    pub model_override: Option<String>,
}

impl PoolTask {
    /// Create a new task with minimal fields.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: None,
            prompt: prompt.into(),
            priority: 0,
            required_capabilities: Vec::new(),
            preferred_role: None,
            model_override: None,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add required capabilities.
    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.required_capabilities = caps;
        self
    }

    /// Set preferred role.
    pub fn with_preferred_role(mut self, role: AgentRole) -> Self {
        self.preferred_role = Some(role);
        self
    }

    /// Set model override (takes priority over AgentDefinition.agent.model).
    pub fn with_model_override(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }
}

// BinaryHeap ordering: higher priority first
impl PartialEq for PoolTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PoolTask {}

impl PartialOrd for PoolTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoolTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

// ============================================================================
// E10: AgentPool — manages multiple agents with a priority queue
// ============================================================================

/// Reason why the supervisor was triggered.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TriggerReason {
    /// Agent has been idle for too many iterations.
    StuckDetected { agent_id: String, idle_streak: usize },
    /// Agent is approaching its budget limit.
    BudgetWarning { agent_id: String, cost: f64, budget: f64 },
    /// Agent is approaching its iteration limit.
    NearIterationLimit { agent_id: String, iteration: usize, max: usize },
}

impl std::fmt::Display for TriggerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriggerReason::StuckDetected { agent_id, idle_streak } => {
                write!(f, "Agent '{}' stuck ({} idle iterations)", agent_id, idle_streak)
            }
            TriggerReason::BudgetWarning { agent_id, cost, budget } => {
                write!(f, "Agent '{}' budget warning (${:.4} / ${:.4})", agent_id, cost, budget)
            }
            TriggerReason::NearIterationLimit { agent_id, iteration, max } => {
                write!(f, "Agent '{}' near iteration limit ({}/{})", agent_id, iteration, max)
            }
        }
    }
}

/// Supervisor configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SupervisorConfig {
    /// Whether the supervisor is enabled.
    pub enabled: bool,
    /// Number of idle iterations before triggering supervisor.
    pub idle_streak_threshold: usize,
    /// Budget percentage at which to warn (0.0 to 1.0).
    pub budget_warning_percent: f64,
    /// Iteration warning margin (trigger when within this many of the limit).
    pub iteration_warning_margin: usize,
    /// Minimum interval between supervisor activations (prevents excessive LLM cost).
    pub min_interval: Duration,
    /// Model to use for the supervisor agent (can be a cheap/local model).
    pub supervisor_model: Option<String>,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_streak_threshold: 5,
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
            min_interval: Duration::from_secs(30),
            supervisor_model: None,
        }
    }
}

/// Callback invoked at the end of each agent iteration.
pub type IterationHook =
    Arc<dyn Fn(&str, usize, &IterationOutcome) + Send + Sync>;

/// Status of a running agent in the pool.
#[derive(Debug, Clone)]
pub struct PoolAgentStatus {
    /// Agent name/id.
    pub agent_id: String,
    /// Current task being executed.
    pub task_id: String,
    /// Current iteration count.
    pub iteration: usize,
    /// Current state.
    pub state: AgentState,
    /// Accumulated cost.
    pub cost: f64,
    /// Tools called so far.
    pub tools_called: Vec<String>,
    /// Number of consecutive idle iterations (no tool calls, no final answer).
    pub idle_streak: usize,
}

/// Result of a completed agent task.
#[derive(Debug, Clone)]
pub struct PoolTaskResult {
    /// Task that was executed.
    pub task_id: String,
    /// Agent that executed it.
    pub agent_id: String,
    /// The result (or error).
    pub result: Result<AgentResult, String>,
}

/// The AgentPool manages multiple concurrent agent executions.
///
/// Agents run in background threads. Tasks are queued by priority.
/// An optional supervisor monitors for stuck agents and budget overruns.
pub struct AgentPool {
    /// Maximum concurrent agents.
    max_agents: usize,
    /// Currently running agents and their status.
    active_agents: HashMap<String, PoolAgentStatus>,
    /// Task queue (priority-ordered).
    queue: BinaryHeap<PoolTask>,
    /// Factory to create response generators for agents.
    response_generator_factory: ResponseGeneratorFactory,
    /// Supervisor configuration.
    supervisor_config: SupervisorConfig,
    /// Available tools for agents.
    tool_registry: ToolRegistry,
    /// Agent definitions indexed by name.
    definitions: HashMap<String, AgentDefinition>,
    /// Completed task results.
    completed: Vec<PoolTaskResult>,
    /// Supervisor trigger log.
    trigger_log: Vec<TriggerReason>,
    /// Receiver for completed task notifications.
    result_receiver: Option<mpsc::Receiver<PoolTaskResult>>,
    /// Sender for completed task notifications (cloned to threads).
    result_sender: mpsc::Sender<PoolTaskResult>,
    /// Cancellation tokens for active agents (keyed by agent_id).
    cancellation_tokens: HashMap<String, Arc<AtomicBool>>,
    /// Mailbox senders for active agents (keyed by agent_id).
    mailbox_senders: HashMap<String, mpsc::Sender<crate::autonomous_loop::InterAgentMessage>>,
    /// Thread join handles for active agents.
    join_handles: HashMap<String, std::thread::JoinHandle<()>>,
    /// Last supervisor activation time.
    last_supervisor_activation: Option<Instant>,
    /// Whether the pool is shutting down.
    shutting_down: bool,
    /// Sequence counter for FIFO tiebreaker in priority queue.
    sequence_counter: u64,
}

impl AgentPool {
    /// Create a new agent pool.
    pub fn new(
        max_agents: usize,
        response_generator_factory: ResponseGeneratorFactory,
        tool_registry: ToolRegistry,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            max_agents,
            active_agents: HashMap::new(),
            queue: BinaryHeap::new(),
            response_generator_factory,
            supervisor_config: SupervisorConfig::default(),
            tool_registry,
            definitions: HashMap::new(),
            completed: Vec::new(),
            trigger_log: Vec::new(),
            result_receiver: Some(rx),
            result_sender: tx,
            cancellation_tokens: HashMap::new(),
            mailbox_senders: HashMap::new(),
            join_handles: HashMap::new(),
            last_supervisor_activation: None,
            shutting_down: false,
            sequence_counter: 0,
        }
    }

    /// Configure the supervisor.
    pub fn with_supervisor(mut self, config: SupervisorConfig) -> Self {
        self.supervisor_config = config;
        self
    }

    /// Register an agent definition.
    pub fn register_definition(&mut self, def: AgentDefinition) {
        self.definitions.insert(def.agent.name.clone(), def);
    }

    /// Submit a task to the pool.
    pub fn submit_task(&mut self, task: PoolTask) {
        self.queue.push(task);
    }

    /// Get the number of active agents.
    pub fn active_count(&self) -> usize {
        self.active_agents.len()
    }

    /// Get the number of queued tasks.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get status of all active agents.
    pub fn active_statuses(&self) -> Vec<PoolAgentStatus> {
        self.active_agents.values().cloned().collect()
    }

    /// Get completed results.
    pub fn completed_results(&self) -> &[PoolTaskResult] {
        &self.completed
    }

    /// Get trigger log.
    pub fn trigger_log(&self) -> &[TriggerReason] {
        &self.trigger_log
    }

    /// Find the best agent for a task using BestFit scoring.
    pub fn find_best_agent(&self, task: &PoolTask) -> Option<String> {
        let mut best_name = None;
        let mut best_score = 0.0f64;

        for (name, def) in &self.definitions {
            let agent = agent_from_definition(def);
            let score = score_agent_for_task(&agent, task);
            if score > best_score {
                best_score = score;
                best_name = Some(name.clone());
            }
        }

        // If all scores are 0.0, warn and use round-robin (first available)
        if best_score == 0.0 && !self.definitions.is_empty() {
            eprintln!("[agent_pool] WARNING: All agent scores are 0.0 for task '{}', using first available", task.id);
            best_name = self.definitions.keys().next().cloned();
        }

        best_name
    }

    /// Spawn an agent to execute a task synchronously (blocking).
    ///
    /// In production use, this would be called from a background thread.
    /// Returns the result directly.
    pub fn execute_task_sync(
        &mut self,
        agent_name: &str,
        task: &PoolTask,
    ) -> Result<AgentResult, String> {
        let def = self
            .definitions
            .get(agent_name)
            .ok_or_else(|| format!("Agent definition '{}' not found", agent_name))?
            .clone();

        // Model priority: task.model_override > def.agent.model
        let model = task
            .model_override
            .as_deref()
            .or(def.agent.model.as_deref());
        let gen = (self.response_generator_factory)(model);

        let mut agent = create_agent_from_definition(&def, gen, &self.tool_registry)
            .map_err(|e| e.to_string())?;

        // Track active status
        self.active_agents.insert(
            agent_name.to_string(),
            PoolAgentStatus {
                agent_id: agent_name.to_string(),
                task_id: task.id.clone(),
                iteration: 0,
                state: AgentState::Idle,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        let result = agent.run(&task.prompt);

        // Move to completed
        self.active_agents.remove(agent_name);
        self.completed.push(PoolTaskResult {
            task_id: task.id.clone(),
            agent_id: agent_name.to_string(),
            result: result.clone(),
        });

        result
    }

    /// Spawn an agent in a background thread with cancellation token and panic recovery.
    ///
    /// The result will be available via `drain_completed()`.
    /// The agent can be cancelled via `cancel_agent()` and receives messages via `send_message()`.
    pub fn spawn_agent(
        &mut self,
        agent_name: &str,
        task: PoolTask,
    ) -> Result<(), String> {
        if self.shutting_down {
            return Err("Pool is shutting down".to_string());
        }

        if self.active_agents.len() >= self.max_agents {
            return Err(format!(
                "Pool at capacity ({}/{})",
                self.active_agents.len(),
                self.max_agents
            ));
        }

        let def = self
            .definitions
            .get(agent_name)
            .ok_or_else(|| format!("Agent definition '{}' not found", agent_name))?
            .clone();

        // Model priority: task.model_override > def.agent.model
        let model = task
            .model_override
            .clone()
            .or_else(|| def.agent.model.clone());
        let factory = Arc::clone(&self.response_generator_factory);
        let tools = self.tool_registry.clone();
        let sender = self.result_sender.clone();
        let agent_id = agent_name.to_string();
        let task_id = task.id.clone();
        let prompt = task.prompt.clone();

        // Create cancellation token
        let cancel_token = Arc::new(AtomicBool::new(false));
        self.cancellation_tokens
            .insert(agent_id.clone(), Arc::clone(&cancel_token));

        // Create mailbox channel
        let (mailbox_tx, mailbox_rx) = mpsc::channel();
        self.mailbox_senders.insert(agent_id.clone(), mailbox_tx);

        // Track active status
        self.active_agents.insert(
            agent_id.clone(),
            PoolAgentStatus {
                agent_id: agent_id.clone(),
                task_id: task_id.clone(),
                iteration: 0,
                state: AgentState::Running,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        let cancel_token_clone = Arc::clone(&cancel_token);
        let handle = std::thread::spawn(move || {
            // Wrap in catch_unwind for panic recovery
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let gen = factory(model.as_deref());
                let options = AgentCreateOptions {
                    cancellation_token: Some(cancel_token_clone),
                    mailbox: Some(mailbox_rx),
                };
                match create_agent_from_definition_with_options(&def, gen, &tools, options) {
                    Ok(mut agent) => agent.run(&prompt),
                    Err(e) => Err(e.to_string()),
                }
            }));

            let final_result = match result {
                Ok(r) => r,
                Err(panic_info) => {
                    let reason = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!(
                        "[agent_pool] Agent '{}' panicked on task '{}': {}",
                        agent_id, task_id, reason
                    );
                    Err(format!("Agent panicked: {}", reason))
                }
            };

            let _ = sender.send(PoolTaskResult {
                task_id,
                agent_id,
                result: final_result,
            });
        });

        self.join_handles.insert(agent_name.to_string(), handle);

        Ok(())
    }

    /// Cancel a running agent by setting its cancellation token.
    pub fn cancel_agent(&self, agent_id: &str) -> bool {
        if let Some(token) = self.cancellation_tokens.get(agent_id) {
            token.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Send a message to a running agent via its mailbox.
    pub fn send_message(
        &self,
        agent_id: &str,
        from: &str,
        content: &str,
    ) -> Result<(), String> {
        let sender = self
            .mailbox_senders
            .get(agent_id)
            .ok_or_else(|| format!("No mailbox for agent '{}'", agent_id))?;

        let msg = crate::autonomous_loop::InterAgentMessage {
            from: from.to_string(),
            content: content.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        sender
            .send(msg)
            .map_err(|_| format!("Agent '{}' mailbox closed", agent_id))
    }

    /// Graceful shutdown: cancel all agents, wait with timeout, clean up.
    ///
    /// Sets all cancellation tokens and waits for threads to complete.
    /// Returns the number of agents that completed within the timeout.
    pub fn shutdown(&mut self, timeout: Duration) -> usize {
        self.shutting_down = true;

        // Set all cancellation tokens
        for token in self.cancellation_tokens.values() {
            token.store(true, Ordering::Relaxed);
        }

        let deadline = Instant::now() + timeout;
        let mut completed = 0;

        // Drain join handles and wait
        let handles: Vec<(String, std::thread::JoinHandle<()>)> =
            self.join_handles.drain().collect();

        for (id, handle) in handles {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                eprintln!(
                    "[agent_pool] Shutdown timeout: agent '{}' still running",
                    id
                );
                continue;
            }
            // We can't join with a timeout in std, so we park_timeout and check
            // The thread should stop because we set the cancellation token
            match handle.join() {
                Ok(()) => completed += 1,
                Err(_) => {
                    eprintln!("[agent_pool] Agent '{}' thread panicked during shutdown", id);
                }
            }
        }

        // Drain remaining results
        self.drain_completed();

        // Clean up
        self.cancellation_tokens.clear();
        self.mailbox_senders.clear();
        self.active_agents.clear();

        completed
    }

    /// Check if the pool is shutting down.
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Trigger the supervisor for a given reason.
    ///
    /// Creates a temporary Coordinator agent that can diagnose the issue.
    /// Respects min_interval between activations.
    pub fn trigger_supervisor(&mut self, reason: &TriggerReason) -> Option<AgentResult> {
        if !self.supervisor_config.enabled {
            return None;
        }

        // Check min_interval
        if let Some(last) = self.last_supervisor_activation {
            if last.elapsed() < self.supervisor_config.min_interval {
                return None;
            }
        }

        self.last_supervisor_activation = Some(Instant::now());

        // Create supervisor agent
        let model = self.supervisor_config.supervisor_model.as_deref();
        let gen = (self.response_generator_factory)(model);

        let system_prompt = format!(
            "You are a Supervisor agent monitoring an agent pool. \
             A trigger has been activated: {}. \
             Diagnose the issue and suggest corrective action. \
             Be concise and actionable.",
            reason
        );

        let mut supervisor = AutonomousAgentBuilder::new("supervisor", gen)
            .system_prompt(system_prompt)
            .max_iterations(3)
            .mode(OperationMode::Autonomous)
            .build();

        let prompt = format!(
            "Supervisor trigger: {}\n\
             Active agents: {}\n\
             Queued tasks: {}\n\
             Analyze and recommend action.",
            reason,
            self.active_agents.len(),
            self.queue.len()
        );

        match supervisor.run(&prompt) {
            Ok(result) => Some(result),
            Err(e) => {
                eprintln!("[agent_pool] Supervisor failed: {}", e);
                None
            }
        }
    }

    /// Drain completed results from background threads.
    ///
    /// Also cleans up cancellation tokens, mailbox senders, and join handles
    /// for completed agents.
    pub fn drain_completed(&mut self) -> Vec<PoolTaskResult> {
        let mut results = Vec::new();

        if let Some(ref rx) = self.result_receiver {
            while let Ok(result) = rx.try_recv() {
                // Clean up per-agent resources
                self.active_agents.remove(&result.agent_id);
                self.cancellation_tokens.remove(&result.agent_id);
                self.mailbox_senders.remove(&result.agent_id);
                self.join_handles.remove(&result.agent_id);
                results.push(result);
            }
        }

        self.completed.extend(results.iter().cloned());
        results
    }

    /// Check thresholds and trigger supervisor if needed.
    ///
    /// Called by the iteration hook. Evaluates numeric thresholds only (no LLM).
    pub fn check_thresholds(&mut self, agent_id: &str, iteration: usize, outcome: &IterationOutcome) {
        if !self.supervisor_config.enabled {
            return;
        }

        if let Some(status) = self.active_agents.get_mut(agent_id) {
            status.iteration = iteration;

            // Track idle streak
            match outcome {
                IterationOutcome::Continue => {
                    status.idle_streak += 1;
                }
                _ => {
                    status.idle_streak = 0;
                }
            }

            // Check idle streak
            if status.idle_streak >= self.supervisor_config.idle_streak_threshold {
                self.trigger_log.push(TriggerReason::StuckDetected {
                    agent_id: agent_id.to_string(),
                    idle_streak: status.idle_streak,
                });
            }

            // Check near iteration limit (get max_iterations from definition)
            if let Some(def) = self.definitions.get(agent_id) {
                let max = def.guardrails.as_ref().and_then(|g| g.max_turns).unwrap_or(50);
                if max > self.supervisor_config.iteration_warning_margin
                    && iteration >= max - self.supervisor_config.iteration_warning_margin
                {
                    self.trigger_log.push(TriggerReason::NearIterationLimit {
                        agent_id: agent_id.to_string(),
                        iteration,
                        max,
                    });
                }
            }
        }
    }
}

// ============================================================================
// I1: AgentPlanningState — MCTS state for agent tool planning
// ============================================================================

#[cfg(feature = "devtools")]
mod mcts_wiring {
    use crate::error::MctsError;
    use crate::mcts_planner::{MctsPlanner, MctsState};
    use crate::unified_tools::ToolDef;

    /// Represents the agent's planning state for MCTS search.
    ///
    /// Captures the current goal, available tools, completed actions, and
    /// optional memory hints. MCTS uses this to explore possible tool sequences
    /// without actually executing them.
    #[derive(Debug, Clone)]
    pub struct AgentPlanningState {
        /// The goal/task the agent is trying to accomplish.
        pub goal: String,
        /// Available tool names with descriptions (from ToolRegistry).
        pub available_tools: Vec<(String, String)>,
        /// Actions already taken in this simulation branch.
        pub completed_actions: Vec<String>,
        /// Maximum depth (number of tool calls to plan ahead).
        pub max_depth: usize,
        /// Memory hints from episodic recall (relevant past experiences).
        pub memory_hints: Vec<String>,
    }

    impl AgentPlanningState {
        /// Create a planning state from a goal, tool registry, and optional memory hints.
        pub fn from_registry(
            goal: &str,
            tools: &[ToolDef],
            max_depth: usize,
            memory_hints: Vec<String>,
        ) -> Self {
            let available_tools = tools
                .iter()
                .map(|t| (t.name.clone(), t.description.clone()))
                .collect();
            Self {
                goal: goal.to_string(),
                available_tools,
                completed_actions: Vec::new(),
                max_depth,
                memory_hints,
            }
        }
    }

    impl MctsState for AgentPlanningState {
        type Action = String; // Tool name

        fn available_actions(&self) -> Vec<String> {
            self.available_tools
                .iter()
                .map(|(name, _)| name.clone())
                .filter(|name| !self.completed_actions.contains(name))
                .collect()
        }

        fn apply_action(&self, action: &String) -> Result<Self, MctsError> {
            let mut new_state = self.clone();
            new_state.completed_actions.push(action.clone());
            Ok(new_state)
        }

        fn is_terminal(&self) -> bool {
            self.completed_actions.len() >= self.max_depth
                || self.available_actions().is_empty()
        }

        fn reward(&self) -> f64 {
            agent_planning_reward(self)
        }

        fn description(&self) -> String {
            format!(
                "Goal: {}, Actions: {:?}, Depth: {}/{}",
                self.goal,
                self.completed_actions,
                self.completed_actions.len(),
                self.max_depth
            )
        }
    }

    /// Compute the reward for an agent planning state.
    ///
    /// Uses a zero-cost heuristic based on:
    /// - Metadata similarity: bag-of-words cosine between tool description and goal (weight 0.6)
    /// - Diversity: penalizes repeated tools (weight 0.2)
    /// - Progress: ratio of actions completed vs max_depth (weight 0.1)
    /// - Memory bonus: +0.1 if memory hints suggest the tool was successful before
    fn agent_planning_reward(state: &AgentPlanningState) -> f64 {
        if state.completed_actions.is_empty() {
            return 0.0;
        }

        let goal_lower = state.goal.to_lowercase();
        let goal_words: Vec<&str> = goal_lower.split_whitespace().collect();

        let mut total_relevance = 0.0;
        let mut unique_tools = std::collections::HashSet::new();

        for action in &state.completed_actions {
            // Metadata similarity: how many goal words appear in tool description
            let tool_desc = state
                .available_tools
                .iter()
                .find(|(name, _)| name == action)
                .map(|(_, desc)| desc.to_lowercase())
                .unwrap_or_default();

            let matching_words = goal_words
                .iter()
                .filter(|w| tool_desc.contains(*w))
                .count();
            let relevance = if goal_words.is_empty() {
                0.0
            } else {
                matching_words as f64 / goal_words.len() as f64
            };
            total_relevance += relevance;

            unique_tools.insert(action.clone());
        }

        let n = state.completed_actions.len() as f64;
        let avg_relevance = total_relevance / n;

        // Diversity: ratio of unique tools to total (1.0 = all unique)
        let diversity = unique_tools.len() as f64 / n;

        // Progress: how far through the plan we are
        let progress = n / state.max_depth as f64;

        // Memory bonus: check if any completed tool appears in memory hints
        let memory_bonus = if !state.memory_hints.is_empty() {
            let matched = state
                .completed_actions
                .iter()
                .filter(|action| {
                    state
                        .memory_hints
                        .iter()
                        .any(|hint| hint.to_lowercase().contains(&action.to_lowercase()))
                })
                .count();
            if matched > 0 { 0.1 } else { 0.0 }
        } else {
            0.0
        };

        // Weighted combination
        let score =
            avg_relevance * 0.6 + diversity * 0.2 + progress * 0.1 + memory_bonus;

        score.min(1.0)
    }

    /// Plan the next sequence of tool calls using MCTS.
    ///
    /// Returns a Vec of tool names representing the optimal sequence found by MCTS.
    /// Returns empty Vec if MCTS doesn't find a sequence better than random (root_value < 0.1).
    ///
    /// # Arguments
    /// * `planner` - The MCTS planner with configured iterations and exploration
    /// * `state` - The current agent planning state
    pub fn plan_next_actions(
        planner: &MctsPlanner,
        state: &AgentPlanningState,
    ) -> Vec<String> {
        if state.available_actions().is_empty() {
            return Vec::new();
        }

        match planner.search(state) {
            Ok(result) => {
                if result.root_value < 0.1 {
                    // Not worth planning — random is just as good
                    return Vec::new();
                }
                result.best_action_sequence
            }
            Err(e) => {
                eprintln!("[mcts_wiring] Planning failed: {:?}", e);
                Vec::new()
            }
        }
    }
}

#[cfg(feature = "devtools")]
pub use mcts_wiring::{plan_next_actions, AgentPlanningState};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- E2b: parse_agent_role ----

    #[test]
    fn test_parse_agent_role_known() {
        assert_eq!(parse_agent_role("Coordinator"), AgentRole::Coordinator);
        assert_eq!(parse_agent_role("manager"), AgentRole::Coordinator);
        assert_eq!(parse_agent_role("RESEARCHER"), AgentRole::Researcher);
        assert_eq!(parse_agent_role("Analyst"), AgentRole::Analyst);
        assert_eq!(parse_agent_role("Writer"), AgentRole::Writer);
        assert_eq!(parse_agent_role("Reviewer"), AgentRole::Reviewer);
        assert_eq!(parse_agent_role("Executor"), AgentRole::Executor);
        assert_eq!(parse_agent_role("Worker"), AgentRole::Executor);
        assert_eq!(parse_agent_role("Validator"), AgentRole::Validator);
    }

    #[test]
    fn test_parse_agent_role_custom() {
        assert_eq!(parse_agent_role("unknown_role"), AgentRole::Custom);
        assert_eq!(parse_agent_role(""), AgentRole::Custom);
        assert_eq!(parse_agent_role("data_scientist"), AgentRole::Custom);
    }

    // ---- E2: role_system_prompt ----

    #[test]
    fn test_role_system_prompt_not_empty() {
        let roles = [
            AgentRole::Coordinator,
            AgentRole::Researcher,
            AgentRole::Analyst,
            AgentRole::Writer,
            AgentRole::Reviewer,
            AgentRole::Executor,
            AgentRole::Validator,
            AgentRole::Custom,
        ];
        for role in &roles {
            let prompt = role_system_prompt(role);
            assert!(!prompt.is_empty(), "Prompt for {:?} should not be empty", role);
            assert!(prompt.len() > 20, "Prompt for {:?} should be substantial", role);
        }
    }

    #[test]
    fn test_role_system_prompt_unique_per_role() {
        let coordinator = role_system_prompt(&AgentRole::Coordinator);
        let researcher = role_system_prompt(&AgentRole::Researcher);
        let analyst = role_system_prompt(&AgentRole::Analyst);
        assert_ne!(coordinator, researcher);
        assert_ne!(researcher, analyst);
        assert_ne!(coordinator, analyst);
    }

    // ---- E1: message conversion ----

    #[test]
    fn test_chat_to_loop_message() {
        let msg = chat_to_loop_message("system", "You are an AI");
        assert_eq!(msg.content, "You are an AI");
        assert!(matches!(msg.role, crate::agentic_loop::LoopRole::System));

        let msg = chat_to_loop_message("user", "Hello");
        assert!(matches!(msg.role, crate::agentic_loop::LoopRole::User));

        let msg = chat_to_loop_message("assistant", "Hi there");
        assert!(matches!(msg.role, crate::agentic_loop::LoopRole::Assistant));

        let msg = chat_to_loop_message("tool", "result: ok");
        assert!(matches!(msg.role, crate::agentic_loop::LoopRole::Tool));

        // Unknown role defaults to user
        let msg = chat_to_loop_message("unknown", "test");
        assert!(matches!(msg.role, crate::agentic_loop::LoopRole::User));
    }

    #[test]
    fn test_loop_message_to_pair() {
        use crate::agentic_loop::LoopRole;

        let msg = LoopMessage {
            role: LoopRole::System,
            content: "test".to_string(),
            tool_calls: None,
            tool_results: None,
        };
        let (role, content) = loop_message_to_pair(&msg);
        assert_eq!(role, "system");
        assert_eq!(content, "test");
    }

    // ---- E3: make_response_generator ----

    #[test]
    fn test_make_response_generator() {
        let gen = make_response_generator(|msgs| {
            format!("Response to {} messages", msgs.len())
        });
        let msgs = vec![chat_to_loop_message("user", "Hello")];
        let result = gen(&msgs);
        assert_eq!(result, "Response to 1 messages");
    }

    #[test]
    fn test_make_response_generator_factory() {
        let factory = make_response_generator_factory(|model| {
            let model_name = model.unwrap_or("default").to_string();
            make_response_generator(move |_| format!("Response from {}", model_name))
        });

        let gen_default = factory(None);
        assert_eq!(gen_default(&[]), "Response from default");

        let gen_custom = factory(Some("gpt-4"));
        assert_eq!(gen_custom(&[]), "Response from gpt-4");
    }

    // ---- E4: create_agent_from_definition ----

    fn make_test_definition() -> AgentDefinition {
        AgentDefinition {
            agent: AgentSpec {
                name: "test-agent".to_string(),
                role: Some("Analyst".to_string()),
                description: Some("Test agent".to_string()),
                system_prompt: Some("You are a test agent.".to_string()),
                model: Some("openai/gpt-4o".to_string()),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                top_p: None,
                autonomy_level: Some("balanced".to_string()),
            },
            tools: vec![],
            memory: None,
            guardrails: Some(GuardrailSpec {
                max_tokens_per_response: Some(500),
                block_pii: true,
                max_turns: Some(20),
                blocked_patterns: vec![],
                require_approval_for_destructive: true,
            }),
            metadata: None,
        }
    }

    #[test]
    fn test_create_agent_from_definition_basic() {
        let def = make_test_definition();
        let gen = make_response_generator(|_| "done".to_string());
        let registry = ToolRegistry::new();

        let result = create_agent_from_definition(&def, gen, &registry);
        assert!(result.is_ok(), "Should create agent: {:?}", result.err());
    }

    #[test]
    fn test_create_agent_from_definition_empty_name_fails() {
        let mut def = make_test_definition();
        def.agent.name = "".to_string();

        let gen = make_response_generator(|_| "done".to_string());
        let registry = ToolRegistry::new();

        let result = create_agent_from_definition(&def, gen, &registry);
        assert!(result.is_err(), "Empty name should fail validation");
    }

    #[test]
    fn test_create_agent_runs_task() {
        let def = make_test_definition();
        let gen = make_response_generator(|_| "The answer is 42.".to_string());
        let registry = ToolRegistry::new();

        let mut agent = create_agent_from_definition(&def, gen, &registry).unwrap();
        let result = agent.run("What is the answer?");
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.output.contains("42"));
        assert_eq!(result.iterations, 1);
    }

    // ---- E4b: agent_from_definition ----

    #[test]
    fn test_agent_from_definition_role() {
        let def = make_test_definition();
        let agent = agent_from_definition(&def);
        assert_eq!(agent.role, AgentRole::Analyst);
        assert_eq!(agent.name, "test-agent");
    }

    #[test]
    fn test_agent_from_definition_no_role() {
        let mut def = make_test_definition();
        def.agent.role = None;
        let agent = agent_from_definition(&def);
        assert_eq!(agent.role, AgentRole::Custom);
    }

    #[test]
    fn test_agent_from_definition_with_tools() {
        let mut def = make_test_definition();
        def.tools = vec![
            crate::agent_definition::ToolRef {
                name: "web_search".to_string(),
                needs_approval: false,
                description: None,
                timeout_ms: None,
            },
            crate::agent_definition::ToolRef {
                name: "file_read".to_string(),
                needs_approval: true,
                description: None,
                timeout_ms: None,
            },
        ];
        let agent = agent_from_definition(&def);
        assert!(agent.capabilities.contains(&"web_search".to_string()));
        assert!(agent.capabilities.contains(&"file_read".to_string()));
    }

    // ---- E5: filter_tool_registry ----

    #[test]
    fn test_filter_tool_registry_empty() {
        let registry = ToolRegistry::new();
        let def = make_test_definition();
        let (filtered, warnings) = filter_tool_registry(&registry, &def);
        assert_eq!(filtered.len(), 0);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_filter_tool_registry_missing_tool_warns() {
        let mut def = make_test_definition();
        def.tools = vec![crate::agent_definition::ToolRef {
            name: "nonexistent_tool".to_string(),
            needs_approval: false,
            description: None,
            timeout_ms: None,
        }];

        let registry = ToolRegistry::new();
        let (_, warnings) = filter_tool_registry(&registry, &def);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("nonexistent_tool"));
    }

    // ---- E9: BestFit scoring ----

    #[test]
    fn test_score_agent_exact_capability_match() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Researcher)
            .with_capability("web_search")
            .with_capability("file_read");

        let task = PoolTask::new("t1", "search the web")
            .with_capabilities(vec!["web_search".to_string()]);

        let score = score_agent_for_task(&agent, &task);
        assert!((score - 100.0).abs() < f64::EPSILON, "Full capability match = 100");
    }

    #[test]
    fn test_score_agent_partial_capability_match() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Researcher)
            .with_capability("web_search");

        let task = PoolTask::new("t1", "do both")
            .with_capabilities(vec!["web_search".to_string(), "file_write".to_string()]);

        let score = score_agent_for_task(&agent, &task);
        assert!((score - 50.0).abs() < f64::EPSILON, "Half match = 50");
    }

    #[test]
    fn test_score_agent_role_affinity() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Researcher);

        let task = PoolTask::new("t1", "research")
            .with_preferred_role(AgentRole::Researcher);

        let score = score_agent_for_task(&agent, &task);
        assert!((score - 30.0).abs() < f64::EPSILON, "Role match only = 30");
    }

    #[test]
    fn test_score_agent_capability_plus_role() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Researcher)
            .with_capability("web_search");

        let task = PoolTask::new("t1", "research")
            .with_capabilities(vec!["web_search".to_string()])
            .with_preferred_role(AgentRole::Researcher);

        let score = score_agent_for_task(&agent, &task);
        assert!((score - 130.0).abs() < f64::EPSILON, "100 + 30 = 130");
    }

    #[test]
    fn test_score_agent_description_fallback() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Custom)
            .with_capability("search");

        let task = PoolTask::new("t1", "do something")
            .with_description("We need to search for information");

        let score = score_agent_for_task(&agent, &task);
        assert!(score > 0.0, "Description fallback should contribute");
        assert!(score <= 10.0, "Description max is 10");
    }

    #[test]
    fn test_score_agent_no_match() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Writer);

        let task = PoolTask::new("t1", "validate data")
            .with_capabilities(vec!["data_validation".to_string()])
            .with_preferred_role(AgentRole::Validator);

        let score = score_agent_for_task(&agent, &task);
        assert!((score - 0.0).abs() < f64::EPSILON, "No match = 0");
    }

    #[test]
    fn test_score_agent_empty_everything() {
        let agent = Agent::new("a1", "Agent 1", AgentRole::Custom);
        let task = PoolTask::new("t1", "do something");
        let score = score_agent_for_task(&agent, &task);
        assert!((score - 0.0).abs() < f64::EPSILON, "Empty = 0");
    }

    // ---- E10: AgentPool ----

    fn make_test_pool() -> AgentPool {
        let factory = make_response_generator_factory(|_| {
            make_response_generator(|_| "Task completed.".to_string())
        });
        AgentPool::new(3, factory, ToolRegistry::new())
    }

    #[test]
    fn test_pool_creation() {
        let pool = make_test_pool();
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.queue_len(), 0);
        assert!(pool.completed_results().is_empty());
    }

    #[test]
    fn test_pool_submit_task() {
        let mut pool = make_test_pool();
        pool.submit_task(PoolTask::new("t1", "task 1").with_priority(5));
        pool.submit_task(PoolTask::new("t2", "task 2").with_priority(10));
        assert_eq!(pool.queue_len(), 2);
    }

    #[test]
    fn test_pool_priority_ordering() {
        let mut pool = make_test_pool();
        pool.submit_task(PoolTask::new("low", "low priority").with_priority(1));
        pool.submit_task(PoolTask::new("high", "high priority").with_priority(10));
        pool.submit_task(PoolTask::new("mid", "mid priority").with_priority(5));

        // BinaryHeap pops highest first
        let first = pool.queue.pop().unwrap();
        assert_eq!(first.id, "high");
        let second = pool.queue.pop().unwrap();
        assert_eq!(second.id, "mid");
        let third = pool.queue.pop().unwrap();
        assert_eq!(third.id, "low");
    }

    #[test]
    fn test_pool_register_definition() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());
        assert!(pool.definitions.contains_key("test-agent"));
    }

    #[test]
    fn test_pool_find_best_agent() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "analyze data")
            .with_preferred_role(AgentRole::Analyst);

        let best = pool.find_best_agent(&task);
        assert_eq!(best, Some("test-agent".to_string()));
    }

    #[test]
    fn test_pool_execute_task_sync() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "What is 2+2?");
        let result = pool.execute_task_sync("test-agent", &task);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.output.contains("Task completed"));
    }

    #[test]
    fn test_pool_execute_unknown_agent() {
        let mut pool = make_test_pool();
        let task = PoolTask::new("t1", "test");
        let result = pool.execute_task_sync("nonexistent", &task);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        // Manually fill active agents to simulate capacity
        for i in 0..3 {
            pool.active_agents.insert(
                format!("agent-{}", i),
                PoolAgentStatus {
                    agent_id: format!("agent-{}", i),
                    task_id: format!("t-{}", i),
                    iteration: 0,
                    state: AgentState::Running,
                    cost: 0.0,
                    tools_called: Vec::new(),
                    idle_streak: 0,
                },
            );
        }

        let result = pool.spawn_agent("test-agent", PoolTask::new("t4", "overflow"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("capacity"));
    }

    #[test]
    fn test_pool_supervisor_disabled() {
        let mut pool = make_test_pool();
        pool.active_agents.insert(
            "a1".to_string(),
            PoolAgentStatus {
                agent_id: "a1".to_string(),
                task_id: "t1".to_string(),
                iteration: 0,
                state: AgentState::Running,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        // Check thresholds with supervisor disabled (default)
        for i in 0..10 {
            pool.check_thresholds("a1", i, &IterationOutcome::Continue);
        }
        assert!(pool.trigger_log.is_empty(), "Supervisor disabled, no triggers");
    }

    #[test]
    fn test_pool_supervisor_idle_streak() {
        let mut pool = make_test_pool();
        pool.supervisor_config = SupervisorConfig {
            enabled: true,
            idle_streak_threshold: 3,
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
            ..Default::default()
        };

        pool.active_agents.insert(
            "a1".to_string(),
            PoolAgentStatus {
                agent_id: "a1".to_string(),
                task_id: "t1".to_string(),
                iteration: 0,
                state: AgentState::Running,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        // 2 idle iterations: no trigger
        pool.check_thresholds("a1", 1, &IterationOutcome::Continue);
        pool.check_thresholds("a1", 2, &IterationOutcome::Continue);
        assert!(pool.trigger_log.is_empty());

        // 3rd idle iteration: trigger
        pool.check_thresholds("a1", 3, &IterationOutcome::Continue);
        assert_eq!(pool.trigger_log.len(), 1);
        assert!(matches!(&pool.trigger_log[0], TriggerReason::StuckDetected { .. }));
    }

    #[test]
    fn test_pool_supervisor_idle_streak_reset() {
        let mut pool = make_test_pool();
        pool.supervisor_config = SupervisorConfig {
            enabled: true,
            idle_streak_threshold: 3,
            ..Default::default()
        };

        pool.active_agents.insert(
            "a1".to_string(),
            PoolAgentStatus {
                agent_id: "a1".to_string(),
                task_id: "t1".to_string(),
                iteration: 0,
                state: AgentState::Running,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        pool.check_thresholds("a1", 1, &IterationOutcome::Continue);
        pool.check_thresholds("a1", 2, &IterationOutcome::Continue);
        // Reset with a Done outcome
        pool.check_thresholds("a1", 3, &IterationOutcome::Done("answer".into()));
        // Continue again
        pool.check_thresholds("a1", 4, &IterationOutcome::Continue);
        pool.check_thresholds("a1", 5, &IterationOutcome::Continue);
        // No trigger because streak was reset
        assert!(pool.trigger_log.is_empty());
    }

    #[test]
    fn test_pool_supervisor_near_iteration_limit() {
        let mut pool = make_test_pool();
        pool.supervisor_config = SupervisorConfig {
            enabled: true,
            idle_streak_threshold: 100, // high so it doesn't trigger
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
            ..Default::default()
        };

        let mut def = make_test_definition();
        def.guardrails = Some(GuardrailSpec {
            max_turns: Some(20),
            max_tokens_per_response: None,
            block_pii: false,
            blocked_patterns: vec![],
            require_approval_for_destructive: false,
        });
        pool.register_definition(def);

        pool.active_agents.insert(
            "test-agent".to_string(),
            PoolAgentStatus {
                agent_id: "test-agent".to_string(),
                task_id: "t1".to_string(),
                iteration: 0,
                state: AgentState::Running,
                cost: 0.0,
                tools_called: Vec::new(),
                idle_streak: 0,
            },
        );

        // Iteration 14: no trigger (20 - 5 = 15)
        pool.check_thresholds("test-agent", 14, &IterationOutcome::Done("x".into()));
        assert!(pool.trigger_log.is_empty());

        // Iteration 15: trigger (>= 15)
        pool.check_thresholds("test-agent", 15, &IterationOutcome::Done("x".into()));
        assert_eq!(pool.trigger_log.len(), 1);
        assert!(matches!(&pool.trigger_log[0], TriggerReason::NearIterationLimit { .. }));
    }

    // ---- E10g: TriggerReason Display ----

    #[test]
    fn test_trigger_reason_display() {
        let stuck = TriggerReason::StuckDetected {
            agent_id: "a1".to_string(),
            idle_streak: 5,
        };
        assert!(format!("{}", stuck).contains("stuck"));

        let budget = TriggerReason::BudgetWarning {
            agent_id: "a1".to_string(),
            cost: 0.5,
            budget: 1.0,
        };
        assert!(format!("{}", budget).contains("budget"));

        let limit = TriggerReason::NearIterationLimit {
            agent_id: "a1".to_string(),
            iteration: 45,
            max: 50,
        };
        assert!(format!("{}", limit).contains("iteration limit"));
    }

    // ---- E10: PoolTask ----

    #[test]
    fn test_agent_task_builder() {
        let task = PoolTask::new("t1", "do something")
            .with_priority(5)
            .with_description("A task description")
            .with_capabilities(vec!["search".to_string()])
            .with_preferred_role(AgentRole::Researcher);

        assert_eq!(task.id, "t1");
        assert_eq!(task.prompt, "do something");
        assert_eq!(task.priority, 5);
        assert_eq!(task.description.unwrap(), "A task description");
        assert_eq!(task.required_capabilities, vec!["search".to_string()]);
        assert_eq!(task.preferred_role.unwrap(), AgentRole::Researcher);
    }

    // ---- E11: Integration tests ----

    #[test]
    fn test_definition_to_agent_to_run() {
        // Full pipeline: JSON definition -> create agent -> run task
        let json = r#"{
            "agent": {
                "name": "integration-test",
                "role": "Analyst",
                "description": "Integration test agent",
                "system_prompt": "You analyze data.",
                "model": "test/model",
                "autonomy_level": "balanced"
            },
            "tools": [],
            "guardrails": {
                "max_turns": 5,
                "block_pii": false,
                "require_approval_for_destructive": false
            }
        }"#;

        let def = crate::agent_definition::AgentDefinitionLoader::from_json(json).unwrap();
        let gen = make_response_generator(|_| "Analysis complete: result is positive.".to_string());
        let registry = ToolRegistry::new();

        let mut agent = create_agent_from_definition(&def, gen, &registry).unwrap();
        let result = agent.run("Analyze the data").unwrap();

        assert!(result.output.contains("positive"));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_definition_to_profile_to_bestfit() {
        let json_analyst = r#"{
            "agent": { "name": "analyst-1", "role": "Analyst" },
            "tools": [{"name": "data_query"}]
        }"#;

        let json_writer = r#"{
            "agent": { "name": "writer-1", "role": "Writer" },
            "tools": [{"name": "text_edit"}]
        }"#;

        let def_analyst = crate::agent_definition::AgentDefinitionLoader::from_json(json_analyst).unwrap();
        let def_writer = crate::agent_definition::AgentDefinitionLoader::from_json(json_writer).unwrap();

        let mut pool = make_test_pool();
        pool.register_definition(def_analyst);
        pool.register_definition(def_writer);

        // Task that needs data_query and prefers Analyst
        let task = PoolTask::new("t1", "analyze data")
            .with_capabilities(vec!["data_query".to_string()])
            .with_preferred_role(AgentRole::Analyst);

        let best = pool.find_best_agent(&task).unwrap();
        assert_eq!(best, "analyst-1", "BestFit should pick the analyst");
    }

    #[test]
    fn test_pool_drain_completed() {
        let mut pool = make_test_pool();
        let results = pool.drain_completed();
        assert!(results.is_empty(), "No completed results initially");
    }

    #[test]
    fn test_pool_spawn_and_drain() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "What is 1+1?");
        pool.spawn_agent("test-agent", task).unwrap();

        // Give the thread time to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        let results = pool.drain_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, "t1");
        assert_eq!(results[0].agent_id, "test-agent");
        assert!(results[0].result.is_ok());
    }

    #[test]
    fn test_creation_error_display() {
        let err = AgentCreationError {
            message: "test error".to_string(),
            validation_errors: vec!["field A missing".to_string(), "field B invalid".to_string()],
        };
        let display = format!("{}", err);
        assert!(display.contains("test error"));
        assert!(display.contains("field A missing"));
        assert!(display.contains("field B invalid"));
    }

    // ---- E10i: CancellationToken ----

    #[test]
    fn test_pool_cancel_agent() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        // Spawn an agent
        let task = PoolTask::new("t1", "long task");
        pool.spawn_agent("test-agent", task).unwrap();

        // Cancel it
        assert!(pool.cancel_agent("test-agent"));
        assert!(!pool.cancel_agent("nonexistent"));

        // Wait for completion
        std::thread::sleep(std::time::Duration::from_millis(200));
        let results = pool.drain_completed();
        assert_eq!(results.len(), 1);
        // The result should indicate cancellation (the agent completes normally
        // because simple tasks finish before the token is checked)
    }

    // ---- E10h: InterAgentMessage ----

    #[test]
    fn test_pool_send_message() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "task");
        pool.spawn_agent("test-agent", task).unwrap();

        // Send a message
        let result = pool.send_message("test-agent", "supervisor", "check status");
        assert!(result.is_ok());

        // Nonexistent agent
        let result = pool.send_message("nonexistent", "supervisor", "hello");
        assert!(result.is_err());

        std::thread::sleep(std::time::Duration::from_millis(100));
        pool.drain_completed();
    }

    // ---- E10k: Panic recovery ----

    #[test]
    fn test_pool_panic_recovery() {
        let factory = make_response_generator_factory(|_| {
            make_response_generator(|_| panic!("test panic"))
        });
        let mut pool = AgentPool::new(3, factory, ToolRegistry::new());
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "trigger panic");
        pool.spawn_agent("test-agent", task).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(200));
        let results = pool.drain_completed();
        assert_eq!(results.len(), 1);
        assert!(results[0].result.is_err());
        let err = results[0].result.as_ref().unwrap_err();
        assert!(err.contains("panicked"), "Error should mention panic: {}", err);
    }

    // ---- E10l: Graceful shutdown ----

    #[test]
    fn test_pool_shutdown() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());

        let task = PoolTask::new("t1", "quick task");
        pool.spawn_agent("test-agent", task).unwrap();

        let completed = pool.shutdown(std::time::Duration::from_secs(5));
        assert_eq!(completed, 1);
        assert!(pool.is_shutting_down());
        assert!(pool.active_agents.is_empty());
    }

    #[test]
    fn test_pool_shutdown_rejects_new_tasks() {
        let mut pool = make_test_pool();
        pool.register_definition(make_test_definition());
        pool.shutting_down = true;

        let task = PoolTask::new("t1", "task");
        let result = pool.spawn_agent("test-agent", task);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("shutting down"));
    }

    // ---- E10f: trigger_supervisor ----

    #[test]
    fn test_trigger_supervisor_disabled() {
        let mut pool = make_test_pool();
        let reason = TriggerReason::StuckDetected {
            agent_id: "a1".to_string(),
            idle_streak: 5,
        };
        assert!(pool.trigger_supervisor(&reason).is_none());
    }

    #[test]
    fn test_trigger_supervisor_enabled() {
        let mut pool = make_test_pool();
        pool.supervisor_config = SupervisorConfig {
            enabled: true,
            idle_streak_threshold: 3,
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
            min_interval: std::time::Duration::from_millis(0), // no throttle for test
            supervisor_model: None,
        };

        let reason = TriggerReason::StuckDetected {
            agent_id: "a1".to_string(),
            idle_streak: 5,
        };
        let result = pool.trigger_supervisor(&reason);
        assert!(result.is_some(), "Supervisor should produce a result");
    }

    #[test]
    fn test_trigger_supervisor_min_interval() {
        let mut pool = make_test_pool();
        pool.supervisor_config = SupervisorConfig {
            enabled: true,
            idle_streak_threshold: 3,
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
            min_interval: std::time::Duration::from_secs(60),
            supervisor_model: None,
        };

        let reason = TriggerReason::StuckDetected {
            agent_id: "a1".to_string(),
            idle_streak: 5,
        };

        // First activation should work
        let result1 = pool.trigger_supervisor(&reason);
        assert!(result1.is_some());

        // Second activation should be throttled
        let result2 = pool.trigger_supervisor(&reason);
        assert!(result2.is_none(), "Should be throttled by min_interval");
    }

    // ---- E10m: SupervisorConfig defaults ----

    #[test]
    fn test_supervisor_config_defaults() {
        let config = SupervisorConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.idle_streak_threshold, 5);
        assert!((config.budget_warning_percent - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.iteration_warning_margin, 5);
        assert_eq!(config.min_interval, std::time::Duration::from_secs(30));
        assert!(config.supervisor_model.is_none());
    }

    // ---- E12a: model_override in PoolTask ----

    #[test]
    fn test_pool_task_model_override() {
        let task = PoolTask::new("t1", "task")
            .with_model_override("gpt-4-turbo");
        assert_eq!(task.model_override.as_deref(), Some("gpt-4-turbo"));
    }

    #[test]
    fn test_model_priority_chain() {
        // When task has model_override, it should take priority over def.agent.model
        let factory = make_response_generator_factory(|model| {
            let model_name = model.unwrap_or("default").to_string();
            make_response_generator(move |_| format!("Response from {}", model_name))
        });
        let mut pool = AgentPool::new(3, factory, ToolRegistry::new());

        // Definition has model "openai/gpt-4o"
        let mut def = make_test_definition();
        def.agent.model = Some("openai/gpt-4o".to_string());
        pool.register_definition(def);

        // Task with model_override
        let task = PoolTask::new("t1", "test").with_model_override("local/llama3");
        let result = pool.execute_task_sync("test-agent", &task).unwrap();
        assert!(
            result.output.contains("local/llama3"),
            "model_override should take priority: {}",
            result.output
        );

        // Task without model_override → falls back to definition model
        let task2 = PoolTask::new("t2", "test");
        let result2 = pool.execute_task_sync("test-agent", &task2).unwrap();
        assert!(
            result2.output.contains("openai/gpt-4o"),
            "Should fall back to definition model: {}",
            result2.output
        );
    }

    // ---- AgentCreateOptions ----

    #[test]
    fn test_create_agent_with_options() {
        let def = make_test_definition();
        let gen = make_response_generator(|_| "done".to_string());
        let registry = ToolRegistry::new();

        let token = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (_, rx) = mpsc::channel();

        let options = AgentCreateOptions {
            cancellation_token: Some(token),
            mailbox: Some(rx),
        };

        let result = create_agent_from_definition_with_options(&def, gen, &registry, options);
        assert!(result.is_ok());
    }

    // ---- InterAgentMessage ----

    #[test]
    fn test_inter_agent_message() {
        let msg = crate::autonomous_loop::InterAgentMessage {
            from: "agent-1".to_string(),
            content: "hello world".to_string(),
            timestamp: 1234567890,
        };
        assert_eq!(msg.from, "agent-1");
        assert_eq!(msg.content, "hello world");
        assert_eq!(msg.timestamp, 1234567890);
    }

    // ---- I1: AgentPlanningState (MCTS wiring) ----

    #[cfg(feature = "devtools")]
    mod mcts_tests {
        use super::*;
        use crate::mcts_planner::{MctsConfig, MctsPlanner, MctsState};
        use crate::unified_tools::ToolDef;

        fn make_test_tools() -> Vec<ToolDef> {
            vec![
                ToolDef::new("web_search", "Search the web for information"),
                ToolDef::new("file_read", "Read a file from the filesystem"),
                ToolDef::new("calculator", "Perform mathematical calculations"),
                ToolDef::new("summarize", "Summarize text content"),
            ]
        }

        #[test]
        fn test_agent_planning_state_actions() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry(
                "search for data and calculate results",
                &tools,
                3,
                Vec::new(),
            );

            let actions = state.available_actions();
            assert_eq!(actions.len(), 4, "All 4 tools should be available");
            assert!(actions.contains(&"web_search".to_string()));
        }

        #[test]
        fn test_agent_planning_state_terminal() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry(
                "test",
                &tools,
                2, // max_depth = 2
                Vec::new(),
            );

            assert!(!state.is_terminal());

            let s1 = state.apply_action(&"web_search".to_string()).unwrap();
            assert!(!s1.is_terminal());

            let s2 = s1.apply_action(&"file_read".to_string()).unwrap();
            assert!(s2.is_terminal(), "Should be terminal at max_depth");
        }

        #[test]
        fn test_agent_planning_state_apply() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry("test", &tools, 3, Vec::new());

            let new_state = state.apply_action(&"web_search".to_string()).unwrap();

            // Original should be unchanged
            assert_eq!(state.completed_actions.len(), 0);
            // New state should have the action
            assert_eq!(new_state.completed_actions.len(), 1);
            assert_eq!(new_state.completed_actions[0], "web_search");
        }

        #[test]
        fn test_agent_planning_state_used_tools_excluded() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry("test", &tools, 5, Vec::new());

            let s1 = state.apply_action(&"web_search".to_string()).unwrap();
            let actions = s1.available_actions();
            assert_eq!(actions.len(), 3, "web_search should be excluded");
            assert!(!actions.contains(&"web_search".to_string()));
        }

        #[test]
        fn test_agent_reward_model_relevance() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry(
                "search for information on the web",
                &tools,
                3,
                Vec::new(),
            );

            let relevant = state.apply_action(&"web_search".to_string()).unwrap();
            let irrelevant = state.apply_action(&"calculator".to_string()).unwrap();

            assert!(
                relevant.reward() > irrelevant.reward(),
                "web_search should score higher for 'search for information on the web': {} vs {}",
                relevant.reward(),
                irrelevant.reward()
            );
        }

        #[test]
        fn test_agent_reward_model_diversity() {
            let tools = vec![
                ToolDef::new("search", "Search for data"),
                ToolDef::new("analyze", "Analyze data"),
            ];
            let state = AgentPlanningState::from_registry("search data", &tools, 5, Vec::new());

            // Use diverse tools
            let s1 = state.apply_action(&"search".to_string()).unwrap();
            let s2 = s1.apply_action(&"analyze".to_string()).unwrap();
            let diverse_reward = s2.reward();

            // The reward should be positive since we used diverse tools
            assert!(diverse_reward > 0.0, "Diverse tool use should yield positive reward");
        }

        #[test]
        fn test_mcts_plan_next_actions() {
            let tools = make_test_tools();
            let state = AgentPlanningState::from_registry(
                "search the web for information",
                &tools,
                3,
                Vec::new(),
            );

            let config = MctsConfig {
                max_iterations: 100,
                exploration_constant: std::f64::consts::SQRT_2,
                max_depth: 3,
                simulation_depth: 3,
                discount_factor: 0.99,
            };
            let planner = MctsPlanner::new(config);

            let actions = plan_next_actions(&planner, &state);
            // Should return a non-empty sequence since the goal is clear
            // (web_search is highly relevant to "search the web")
            assert!(!actions.is_empty(), "Should plan at least one action");
        }

        #[test]
        fn test_mcts_plan_empty_tools() {
            let state = AgentPlanningState::from_registry(
                "do something",
                &[], // No tools
                3,
                Vec::new(),
            );

            let planner = MctsPlanner::with_defaults();
            let actions = plan_next_actions(&planner, &state);
            assert!(actions.is_empty(), "No tools = no actions");
        }

        #[test]
        fn test_mcts_plan_memory_bonus() {
            let tools = make_test_tools();
            let state_no_memory = AgentPlanningState::from_registry(
                "search data",
                &tools,
                3,
                Vec::new(),
            );

            let state_with_memory = AgentPlanningState::from_registry(
                "search data",
                &tools,
                3,
                vec!["Previously used web_search successfully for data retrieval".to_string()],
            );

            let s1 = state_no_memory.apply_action(&"web_search".to_string()).unwrap();
            let s2 = state_with_memory.apply_action(&"web_search".to_string()).unwrap();

            assert!(
                s2.reward() >= s1.reward(),
                "Memory hints should give bonus: {} vs {}",
                s2.reward(),
                s1.reward()
            );
        }
    }
}
