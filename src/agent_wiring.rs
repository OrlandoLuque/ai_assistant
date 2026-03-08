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
use std::sync::{mpsc, Arc, RwLock};

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

/// Create an AutonomousAgent from an AgentDefinition.
///
/// Validates the definition (rejects if any Error-severity issues found),
/// builds the runtime with system_prompt, model, tools, guardrails.
pub fn create_agent_from_definition(
    def: &AgentDefinition,
    response_generator: ResponseGenerator,
    available_tools: &ToolRegistry,
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

    // Build the sandbox
    let sandbox = Arc::new(RwLock::new(SandboxValidator::with_approval(
        policy.clone(),
        Arc::new(AutoApproveAll),
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
    let agent = AutonomousAgentBuilder::new(&def.agent.name, response_generator)
        .system_prompt(system_prompt)
        .max_iterations(max_iterations)
        .policy(policy)
        .sandbox(sandbox)
        .tool_registry(filtered_tools)
        .with_cost_config(cost_config)
        .mode(OperationMode::Autonomous)
        .build();

    Ok(agent)
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
pub struct SupervisorConfig {
    /// Whether the supervisor is enabled.
    pub enabled: bool,
    /// Number of idle iterations before triggering supervisor.
    pub idle_streak_threshold: usize,
    /// Budget percentage at which to warn (0.0 to 1.0).
    pub budget_warning_percent: f64,
    /// Iteration warning margin (trigger when within this many of the limit).
    pub iteration_warning_margin: usize,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_streak_threshold: 5,
            budget_warning_percent: 0.8,
            iteration_warning_margin: 5,
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

        let model = def.agent.model.as_deref();
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

    /// Spawn an agent in a background thread.
    ///
    /// The result will be available via `drain_completed()`.
    pub fn spawn_agent(
        &mut self,
        agent_name: &str,
        task: PoolTask,
    ) -> Result<(), String> {
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

        let model = def.agent.model.as_deref().map(|s| s.to_string());
        let factory = Arc::clone(&self.response_generator_factory);
        let tools = self.tool_registry.clone();
        let sender = self.result_sender.clone();
        let agent_id = agent_name.to_string();
        let task_id = task.id.clone();
        let prompt = task.prompt.clone();

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

        std::thread::spawn(move || {
            let gen = factory(model.as_deref());
            let result = match create_agent_from_definition(&def, gen, &tools) {
                Ok(mut agent) => agent.run(&prompt),
                Err(e) => Err(e.to_string()),
            };

            let _ = sender.send(PoolTaskResult {
                task_id,
                agent_id,
                result,
            });
        });

        Ok(())
    }

    /// Drain completed results from background threads.
    pub fn drain_completed(&mut self) -> Vec<PoolTaskResult> {
        let mut results = Vec::new();

        if let Some(ref rx) = self.result_receiver {
            while let Ok(result) = rx.try_recv() {
                self.active_agents.remove(&result.agent_id);
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
}
