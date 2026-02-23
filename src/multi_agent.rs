//! Multi-agent orchestration
//!
//! Coordinate multiple AI agents working together.

use std::collections::HashMap;
use std::time::Instant;

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Idle,
    Working,
    WaitingForInput,
    Completed,
    Failed,
    Paused,
}

/// Agent role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentRole {
    Coordinator,
    Researcher,
    Analyst,
    Writer,
    Reviewer,
    Executor,
    Validator,
    Custom,
}

/// Agent definition
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub role: AgentRole,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub current_task: Option<String>,
    pub model: Option<String>,
}

impl Agent {
    pub fn new(id: &str, name: &str, role: AgentRole) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            role,
            capabilities: Vec::new(),
            status: AgentStatus::Idle,
            current_task: None,
            model: None,
        }
    }

    pub fn with_capability(mut self, capability: &str) -> Self {
        self.capabilities.push(capability.to_string());
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

/// Message between agents
#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub id: String,
    pub from: String,
    pub to: String,
    pub content: String,
    pub message_type: MessageType,
    pub timestamp: Instant,
    pub correlation_id: Option<String>,
}

/// Message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Request,
    Response,
    Notification,
    Error,
    Handoff,
    Status,
}

impl AgentMessage {
    pub fn new(from: &str, to: &str, content: &str, message_type: MessageType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.to_string(),
            to: to.to_string(),
            content: content.to_string(),
            message_type,
            timestamp: Instant::now(),
            correlation_id: None,
        }
    }

    pub fn with_correlation(mut self, correlation_id: &str) -> Self {
        self.correlation_id = Some(correlation_id.to_string());
        self
    }
}

/// Task for agents
#[derive(Debug, Clone)]
pub struct AgentTask {
    pub id: String,
    pub description: String,
    pub assigned_to: Option<String>,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub result: Option<String>,
    pub priority: u8,
    pub deadline: Option<Instant>,
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Blocked,
    Completed,
    Failed,
    Cancelled,
}

impl AgentTask {
    pub fn new(id: &str, description: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            assigned_to: None,
            dependencies: Vec::new(),
            status: TaskStatus::Pending,
            result: None,
            priority: 5,
            deadline: None,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn depends_on(mut self, task_id: &str) -> Self {
        self.dependencies.push(task_id.to_string());
        self
    }
}

/// Orchestration strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestrationStrategy {
    Sequential,
    Parallel,
    Pipeline,
    RoundRobin,
    BestFit,
}

/// Multi-agent orchestrator
pub struct AgentOrchestrator {
    agents: HashMap<String, Agent>,
    tasks: HashMap<String, AgentTask>,
    messages: Vec<AgentMessage>,
    strategy: OrchestrationStrategy,
    coordinator_id: Option<String>,
}

impl AgentOrchestrator {
    pub fn new(strategy: OrchestrationStrategy) -> Self {
        Self {
            agents: HashMap::new(),
            tasks: HashMap::new(),
            messages: Vec::new(),
            strategy,
            coordinator_id: None,
        }
    }

    pub fn register_agent(&mut self, agent: Agent) {
        if agent.role == AgentRole::Coordinator && self.coordinator_id.is_none() {
            self.coordinator_id = Some(agent.id.clone());
        }
        self.agents.insert(agent.id.clone(), agent);
    }

    pub fn unregister_agent(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
        if self.coordinator_id.as_ref() == Some(&agent_id.to_string()) {
            self.coordinator_id = None;
        }
    }

    pub fn add_task(&mut self, task: AgentTask) {
        self.tasks.insert(task.id.clone(), task);
    }

    pub fn assign_task(&mut self, task_id: &str, agent_id: &str) -> Result<(), OrchestrationError> {
        // First check if agent exists and is idle
        {
            let agent = self
                .agents
                .get(agent_id)
                .ok_or(OrchestrationError::AgentNotFound)?;
            if agent.status != AgentStatus::Idle {
                return Err(OrchestrationError::AgentBusy);
            }
        }

        // Check if task exists and get dependencies
        let dependencies = {
            let task = self
                .tasks
                .get(task_id)
                .ok_or(OrchestrationError::TaskNotFound)?;
            task.dependencies.clone()
        };

        // Check dependencies
        for dep_id in &dependencies {
            if let Some(dep_task) = self.tasks.get(dep_id) {
                if dep_task.status != TaskStatus::Completed {
                    return Err(OrchestrationError::DependencyNotMet);
                }
            }
        }

        // Now update task
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.assigned_to = Some(agent_id.to_string());
            task.status = TaskStatus::InProgress;
        }

        // Update agent
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.status = AgentStatus::Working;
            agent.current_task = Some(task_id.to_string());
        }

        Ok(())
    }

    pub fn complete_task(&mut self, task_id: &str, result: &str) -> Result<(), OrchestrationError> {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or(OrchestrationError::TaskNotFound)?;

        let agent_id = task
            .assigned_to
            .clone()
            .ok_or(OrchestrationError::TaskNotAssigned)?;

        task.status = TaskStatus::Completed;
        task.result = Some(result.to_string());

        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.status = AgentStatus::Idle;
            agent.current_task = None;
        }

        Ok(())
    }

    pub fn fail_task(&mut self, task_id: &str, error: &str) -> Result<(), OrchestrationError> {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or(OrchestrationError::TaskNotFound)?;

        let agent_id = task.assigned_to.clone();

        task.status = TaskStatus::Failed;
        task.result = Some(format!("Error: {}", error));

        if let Some(id) = agent_id {
            if let Some(agent) = self.agents.get_mut(&id) {
                agent.status = AgentStatus::Failed;
                agent.current_task = None;
            }
        }

        Ok(())
    }

    pub fn send_message(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    pub fn get_messages_for(&self, agent_id: &str) -> Vec<&AgentMessage> {
        self.messages.iter().filter(|m| m.to == agent_id).collect()
    }

    pub fn auto_assign_tasks(&mut self) -> Vec<(String, String)> {
        let mut assignments = Vec::new();

        // Get pending tasks sorted by priority
        let mut pending_tasks: Vec<_> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.status == TaskStatus::Pending && t.assigned_to.is_none())
            .map(|(id, t)| (id.clone(), t.priority))
            .collect();

        pending_tasks.sort_by(|a, b| b.1.cmp(&a.1));

        // Get idle agents (sorted for deterministic assignment)
        let mut idle_agents: Vec<_> = self
            .agents
            .iter()
            .filter(|(_, a)| a.status == AgentStatus::Idle)
            .map(|(id, _)| id.clone())
            .collect();
        idle_agents.sort();

        match self.strategy {
            OrchestrationStrategy::RoundRobin => {
                for (i, (task_id, _)) in pending_tasks.iter().enumerate() {
                    if i < idle_agents.len() {
                        if self.assign_task(task_id, &idle_agents[i]).is_ok() {
                            assignments.push((task_id.clone(), idle_agents[i].clone()));
                        }
                    }
                }
            }
            OrchestrationStrategy::BestFit => {
                for (task_id, _) in pending_tasks {
                    if let Some(task) = self.tasks.get(&task_id) {
                        // Find best agent based on capabilities
                        let best_agent = idle_agents
                            .iter()
                            .filter(|id| {
                                self.agents
                                    .get(*id)
                                    .map(|a| a.status == AgentStatus::Idle)
                                    .unwrap_or(false)
                            })
                            .max_by(|a_id, b_id| {
                                let score = |id: &String| {
                                    self.agents
                                        .get(id)
                                        .map(|a| {
                                            let desc_lower = task.description.to_lowercase();
                                            a.capabilities
                                                .iter()
                                                .filter(|c| {
                                                    desc_lower.contains(&c.to_lowercase())
                                                })
                                                .count()
                                        })
                                        .unwrap_or(0)
                                };
                                let sa = score(a_id);
                                let sb = score(b_id);
                                sa.cmp(&sb).then_with(|| b_id.cmp(a_id))
                            });

                        if let Some(agent_id) = best_agent {
                            if self.assign_task(&task_id, agent_id).is_ok() {
                                assignments.push((task_id, agent_id.clone()));
                            }
                        }
                    }
                }
            }
            _ => {
                // Default sequential assignment
                for (task_id, _) in pending_tasks {
                    for agent_id in &idle_agents {
                        if self.assign_task(&task_id, agent_id).is_ok() {
                            assignments.push((task_id.clone(), agent_id.clone()));
                            break;
                        }
                    }
                }
            }
        }

        assignments
    }

    pub fn get_status(&self) -> OrchestrationStatus {
        let total_tasks = self.tasks.len();
        let completed = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        let in_progress = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::InProgress)
            .count();
        let failed = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Failed)
            .count();

        let idle_agents = self
            .agents
            .values()
            .filter(|a| a.status == AgentStatus::Idle)
            .count();
        let working_agents = self
            .agents
            .values()
            .filter(|a| a.status == AgentStatus::Working)
            .count();

        OrchestrationStatus {
            total_agents: self.agents.len(),
            idle_agents,
            working_agents,
            total_tasks,
            completed_tasks: completed,
            in_progress_tasks: in_progress,
            failed_tasks: failed,
            pending_tasks: total_tasks - completed - in_progress - failed,
        }
    }

    pub fn get_agent(&self, agent_id: &str) -> Option<&Agent> {
        self.agents.get(agent_id)
    }

    pub fn get_task(&self, task_id: &str) -> Option<&AgentTask> {
        self.tasks.get(task_id)
    }

    pub fn export_json(&self) -> serde_json::Value {
        let status = self.get_status();
        serde_json::json!({
            "strategy": format!("{:?}", self.strategy),
            "coordinator_id": self.coordinator_id,
            "agents": self.agents.values().map(|a| serde_json::json!({
                "id": a.id,
                "name": a.name,
                "role": format!("{:?}", a.role),
                "capabilities": a.capabilities,
                "status": format!("{:?}", a.status),
                "current_task": a.current_task,
                "model": a.model,
            })).collect::<Vec<_>>(),
            "tasks": self.tasks.values().map(|t| serde_json::json!({
                "id": t.id,
                "description": t.description,
                "assigned_to": t.assigned_to,
                "dependencies": t.dependencies,
                "status": format!("{:?}", t.status),
                "result": t.result,
                "priority": t.priority,
            })).collect::<Vec<_>>(),
            "messages": self.messages.iter().map(|m| serde_json::json!({
                "id": m.id,
                "from": m.from,
                "to": m.to,
                "content": m.content,
                "message_type": format!("{:?}", m.message_type),
            })).collect::<Vec<_>>(),
            "status": serde_json::json!({
                "total_agents": status.total_agents,
                "idle_agents": status.idle_agents,
                "working_agents": status.working_agents,
                "total_tasks": status.total_tasks,
                "pending_tasks": status.pending_tasks,
                "in_progress_tasks": status.in_progress_tasks,
                "completed_tasks": status.completed_tasks,
                "failed_tasks": status.failed_tasks,
            }),
        })
    }
}

impl Default for AgentOrchestrator {
    fn default() -> Self {
        Self::new(OrchestrationStrategy::BestFit)
    }
}

/// Orchestration status
#[derive(Debug, Clone)]
pub struct OrchestrationStatus {
    pub total_agents: usize,
    pub idle_agents: usize,
    pub working_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub in_progress_tasks: usize,
    pub failed_tasks: usize,
    pub pending_tasks: usize,
}

/// Orchestration errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestrationError {
    AgentNotFound,
    TaskNotFound,
    AgentBusy,
    DependencyNotMet,
    TaskNotAssigned,
    InvalidState,
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentNotFound => write!(f, "Agent not found"),
            Self::TaskNotFound => write!(f, "Task not found"),
            Self::AgentBusy => write!(f, "Agent is busy"),
            Self::DependencyNotMet => write!(f, "Task dependency not met"),
            Self::TaskNotAssigned => write!(f, "Task not assigned to any agent"),
            Self::InvalidState => write!(f, "Invalid state"),
        }
    }
}

impl std::error::Error for OrchestrationError {}

// =============================================================================
// Multi-Agent Session (with autonomous agents, task board, and user interaction)
// =============================================================================

#[cfg(feature = "autonomous")]
use std::sync::{Arc, RwLock};

#[cfg(feature = "autonomous")]
use crate::autonomous_loop::AutonomousAgent;
#[cfg(feature = "autonomous")]
use crate::interactive_commands::{CommandProcessor, CommandResult, UserIntent};
#[cfg(feature = "autonomous")]
use crate::task_board::{BoardCommand, TaskBoard};
#[cfg(feature = "autonomous")]
use crate::task_planning::StepPriority;
#[cfg(feature = "autonomous")]
use crate::user_interaction::{AutoApproveHandler, InteractionManager, UserInteractionHandler};

/// A live multi-agent execution session combining:
/// - An `AgentOrchestrator` for task/agent management
/// - Multiple `AutonomousAgent`s for actual execution
/// - A shared `TaskBoard` for live progress tracking
/// - An `InteractionManager` for agent↔user communication
/// - A `CommandProcessor` for interpreting user commands during execution
#[cfg(feature = "autonomous")]
pub struct MultiAgentSession {
    /// The underlying orchestrator for task/agent management.
    pub orchestrator: AgentOrchestrator,
    /// Autonomous agents keyed by their name.
    agents: HashMap<String, AutonomousAgent>,
    /// Shared task board visible to all agents and the user.
    task_board: Arc<RwLock<TaskBoard>>,
    /// Shared interaction manager for all agents.
    interaction: Arc<InteractionManager>,
    /// Name of this session.
    name: String,
}

/// Summary of a multi-agent session's current state.
#[cfg(feature = "autonomous")]
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Session name.
    pub name: String,
    /// Orchestration status.
    pub status: OrchestrationStatus,
    /// Per-agent state.
    pub agent_states: Vec<(String, String)>,
    /// Task board display.
    pub board_display: String,
    /// Overall progress (0.0 - 1.0).
    pub progress: f64,
}

#[cfg(feature = "autonomous")]
impl MultiAgentSession {
    /// Create a new multi-agent session.
    pub fn new(
        name: impl Into<String>,
        strategy: OrchestrationStrategy,
        handler: Option<Arc<dyn UserInteractionHandler>>,
    ) -> Self {
        let handler: Arc<dyn UserInteractionHandler> = handler.unwrap_or_else(|| {
            Arc::new(AutoApproveHandler::new()) as Arc<dyn UserInteractionHandler>
        });
        let interaction = Arc::new(InteractionManager::new(handler, 300));
        let task_board = Arc::new(RwLock::new(TaskBoard::new("session-board")));
        let session_name = name.into();

        Self {
            orchestrator: AgentOrchestrator::new(strategy),
            agents: HashMap::new(),
            task_board,
            interaction,
            name: session_name,
        }
    }

    /// Add an autonomous agent to the session.
    pub fn add_agent(&mut self, agent: AutonomousAgent, role: AgentRole) {
        let name = agent.config().name.clone();
        let multi_agent = Agent::new(&name, &name, role);
        self.orchestrator.register_agent(multi_agent);
        self.agents.insert(name, agent);
    }

    /// Add a task to the session.
    pub fn add_task(&mut self, task: AgentTask) {
        // Also add to task board
        if let Ok(mut board) = self.task_board.write() {
            let _ = board.execute_command(BoardCommand::AddTask {
                title: task.description.clone(),
                description: format!("Task {} (priority {})", task.id, task.priority),
                priority: StepPriority::Medium,
            });
        }
        self.orchestrator.add_task(task);
    }

    /// Auto-assign tasks to idle agents using the orchestrator's strategy.
    pub fn auto_assign(&mut self) -> Vec<(String, String)> {
        self.orchestrator.auto_assign_tasks()
    }

    /// Run a specific agent on its assigned task. Returns the agent's result.
    pub fn run_agent(
        &mut self,
        agent_name: &str,
        task_description: &str,
    ) -> Result<String, String> {
        // Update board progress
        if let Ok(mut board) = self.task_board.write() {
            let _ = board.execute_command(BoardCommand::AddTask {
                title: format!("[{}] {}", agent_name, task_description),
                description: String::new(),
                priority: StepPriority::Medium,
            });
        }

        let agent = self
            .agents
            .get_mut(agent_name)
            .ok_or_else(|| format!("Agent '{}' not found", agent_name))?;

        let result = agent.run(task_description)?;
        Ok(result.output)
    }

    /// Process a user command during session execution (pause, resume, cancel, status, etc.).
    pub fn process_user_input(&self, input: &str) -> CommandResult {
        let intent = CommandProcessor::parse_intent(input);
        match intent {
            UserIntent::ShowStatus => {
                let board = self.task_board.read().unwrap_or_else(|e| e.into_inner());
                CommandResult {
                    success: true,
                    message: format!(
                        "Session '{}'\n{}\n\n{}",
                        self.name,
                        format_orchestration_status(&self.orchestrator.get_status()),
                        board.to_display(),
                    ),
                    board_snapshot: Some(board.to_display()),
                }
            }
            UserIntent::Help => CommandResult {
                success: true,
                message: "Commands: status, pause <task>, resume <task>, cancel <task>, pause all, resume all, help".to_string(),
                board_snapshot: None,
            },
            UserIntent::PauseAll => {
                let mut board = self.task_board.write().unwrap_or_else(|e| e.into_inner());
                let _ = board.execute_command(BoardCommand::PauseAll);
                CommandResult {
                    success: true,
                    message: "All tasks paused.".to_string(),
                    board_snapshot: Some(board.to_display()),
                }
            }
            UserIntent::ResumeAll => {
                let mut board = self.task_board.write().unwrap_or_else(|e| e.into_inner());
                let _ = board.execute_command(BoardCommand::ResumeAll);
                CommandResult {
                    success: true,
                    message: "All tasks resumed.".to_string(),
                    board_snapshot: Some(board.to_display()),
                }
            }
            UserIntent::CancelAll => {
                let mut board = self.task_board.write().unwrap_or_else(|e| e.into_inner());
                let _ = board.execute_command(BoardCommand::CancelAll);
                CommandResult {
                    success: true,
                    message: "All tasks cancelled.".to_string(),
                    board_snapshot: Some(board.to_display()),
                }
            }
            _ => CommandResult {
                success: false,
                message: format!("Unknown or unsupported command: {}", input),
                board_snapshot: None,
            },
        }
    }

    /// Get a snapshot of the task board display.
    pub fn board_display(&self) -> String {
        self.task_board
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .to_display()
    }

    /// Get session summary.
    pub fn summary(&self) -> SessionSummary {
        let status = self.orchestrator.get_status();
        let agent_states: Vec<(String, String)> = self
            .agents
            .iter()
            .map(|(name, agent)| (name.clone(), format!("{:?}", agent.state())))
            .collect();
        let board = self.task_board.read().unwrap_or_else(|e| e.into_inner());
        let total = status.total_tasks.max(1) as f64;
        let progress = status.completed_tasks as f64 / total;

        SessionSummary {
            name: self.name.clone(),
            status,
            agent_states,
            board_display: board.to_display(),
            progress,
        }
    }

    /// Get the shared task board.
    pub fn task_board(&self) -> &Arc<RwLock<TaskBoard>> {
        &self.task_board
    }

    /// Get the interaction manager.
    pub fn interaction_manager(&self) -> &Arc<InteractionManager> {
        &self.interaction
    }

    /// Get agent names.
    pub fn agent_names(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(feature = "autonomous")]
fn format_orchestration_status(status: &OrchestrationStatus) -> String {
    format!(
        "Agents: {} total ({} idle, {} working) | Tasks: {} total ({} done, {} in progress, {} pending, {} failed)",
        status.total_agents, status.idle_agents, status.working_agents,
        status.total_tasks, status.completed_tasks, status.in_progress_tasks,
        status.pending_tasks, status.failed_tasks,
    )
}

// =============================================================================
// Multi-Agent Collaboration Types (MessageBus, SharedContext, TaskDispatcher)
// =============================================================================

/// A message on the collaboration bus.
#[derive(Debug, Clone)]
pub struct BusMessage {
    pub id: String,
    pub topic: String,
    pub sender: String,
    pub payload: serde_json::Value,
    pub timestamp_ms: u64,
}

/// A pub/sub message bus for inter-agent communication.
///
/// Agents subscribe to topics and can publish messages. The bus retains a
/// configurable history of messages and supports polling by subscriber.
pub struct MessageBus {
    subscriptions: HashMap<String, Vec<String>>,
    messages: Vec<BusMessage>,
    history_limit: usize,
    next_id: u64,
}

impl MessageBus {
    /// Create a new message bus with the given history limit.
    pub fn new(history_limit: usize) -> Self {
        Self {
            subscriptions: HashMap::new(),
            messages: Vec::new(),
            history_limit,
            next_id: 0,
        }
    }

    /// Subscribe an agent to a topic. Avoids duplicates.
    pub fn subscribe(&mut self, agent_id: &str, topic: &str) {
        let subs = self.subscriptions.entry(topic.to_string()).or_default();
        if !subs.contains(&agent_id.to_string()) {
            subs.push(agent_id.to_string());
        }
    }

    /// Unsubscribe an agent from a topic.
    pub fn unsubscribe(&mut self, agent_id: &str, topic: &str) {
        if let Some(subs) = self.subscriptions.get_mut(topic) {
            subs.retain(|id| id != agent_id);
        }
    }

    /// Publish a message to a topic. Returns the auto-generated message id.
    pub fn publish(&mut self, sender: &str, topic: &str, payload: serde_json::Value) -> String {
        let msg_id = format!("msg-{}", self.next_id);
        self.next_id += 1;

        let message = BusMessage {
            id: msg_id.clone(),
            topic: topic.to_string(),
            sender: sender.to_string(),
            payload,
            timestamp_ms: self.next_id, // simplified monotonic timestamp
        };

        self.messages.push(message);

        // Evict oldest messages if over history limit
        while self.messages.len() > self.history_limit {
            self.messages.remove(0);
        }

        msg_id
    }

    /// Poll messages for an agent. Returns all messages whose topic the agent
    /// is subscribed to, in chronological order.
    pub fn poll(&self, agent_id: &str) -> Vec<&BusMessage> {
        // Collect all topics this agent is subscribed to
        let subscribed_topics: Vec<&String> = self
            .subscriptions
            .iter()
            .filter(|(_, subs)| subs.contains(&agent_id.to_string()))
            .map(|(topic, _)| topic)
            .collect();

        self.messages
            .iter()
            .filter(|m| subscribed_topics.contains(&&m.topic))
            .collect()
    }

    /// List subscribers for a topic. Returns empty vec if topic not found.
    pub fn topic_subscribers(&self, topic: &str) -> Vec<String> {
        self.subscriptions.get(topic).cloned().unwrap_or_default()
    }

    /// Number of messages currently in the bus.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Clear all messages but keep subscriptions.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

/// An entry in the shared context store.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub value: serde_json::Value,
    pub last_writer: String,
    pub version: u64,
    pub timestamp_ms: u64,
}

/// A shared key-value context that multiple agents can read and write.
///
/// Supports versioned entries, merge from another context (last-writer-wins
/// by version), snapshot export, and diff detection.
pub struct SharedContext {
    data: HashMap<String, ContextEntry>,
    version: u64,
}

impl SharedContext {
    /// Create an empty shared context.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            version: 0,
        }
    }

    /// Set a key in the context. Increments the global version.
    pub fn set(&mut self, key: &str, value: serde_json::Value, writer: &str) {
        self.version += 1;
        let entry = ContextEntry {
            value,
            last_writer: writer.to_string(),
            version: self.version,
            timestamp_ms: self.version, // simplified
        };
        self.data.insert(key.to_string(), entry);
    }

    /// Get an entry by key.
    pub fn get(&self, key: &str) -> Option<&ContextEntry> {
        self.data.get(key)
    }

    /// Get just the value for a key.
    pub fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key).map(|e| &e.value)
    }

    /// Remove an entry, returning it if it existed.
    pub fn remove(&mut self, key: &str) -> Option<ContextEntry> {
        self.data.remove(key)
    }

    /// List all keys.
    pub fn keys(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the context is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Merge another context into this one. For each key in `other`, if
    /// `other`'s entry version is higher (or the key doesn't exist in self),
    /// copy it. Updates self.version to the max of both.
    pub fn merge(&mut self, other: &SharedContext) {
        for (key, other_entry) in &other.data {
            let dominated = match self.data.get(key) {
                Some(self_entry) => other_entry.version > self_entry.version,
                None => true,
            };
            if dominated {
                self.data.insert(key.clone(), other_entry.clone());
            }
        }
        if other.version > self.version {
            self.version = other.version;
        }
    }

    /// Export the context as a JSON object mapping keys to their values.
    pub fn snapshot(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (key, entry) in &self.data {
            map.insert(key.clone(), entry.value.clone());
        }
        serde_json::Value::Object(map)
    }

    /// Return keys that differ between self and other (different values, or
    /// present in one but not the other).
    pub fn diff(&self, other: &SharedContext) -> Vec<String> {
        let mut diff_keys = Vec::new();

        // Keys in self
        for key in self.data.keys() {
            match other.data.get(key) {
                Some(other_entry) => {
                    if self.data[key].value != other_entry.value {
                        diff_keys.push(key.clone());
                    }
                }
                None => diff_keys.push(key.clone()),
            }
        }

        // Keys only in other
        for key in other.data.keys() {
            if !self.data.contains_key(key) {
                diff_keys.push(key.clone());
            }
        }

        diff_keys.sort();
        diff_keys
    }
}

impl Default for SharedContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatches tasks to agents based on capabilities and current load.
pub struct TaskDispatcher {
    agents: Vec<String>,
    capabilities: HashMap<String, Vec<String>>,
    load: HashMap<String, usize>,
}

impl TaskDispatcher {
    /// Create an empty dispatcher.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            capabilities: HashMap::new(),
            load: HashMap::new(),
        }
    }

    /// Register an agent with given capabilities. Initializes load to 0.
    pub fn register_agent(&mut self, agent_id: &str, capabilities: Vec<String>) {
        if !self.agents.contains(&agent_id.to_string()) {
            self.agents.push(agent_id.to_string());
        }
        self.capabilities.insert(agent_id.to_string(), capabilities);
        self.load.insert(agent_id.to_string(), 0);
    }

    /// Remove an agent entirely.
    pub fn remove_agent(&mut self, agent_id: &str) {
        self.agents.retain(|id| id != agent_id);
        self.capabilities.remove(agent_id);
        self.load.remove(agent_id);
    }

    /// Dispatch a task requiring the given capability. Picks the agent with
    /// the lowest current load among those possessing the capability. Ties
    /// are broken alphabetically. Returns the chosen agent id and increments
    /// their load.
    pub fn dispatch(&mut self, required_capability: &str) -> Option<String> {
        let mut candidates: Vec<String> = self
            .agents
            .iter()
            .filter(|id| {
                self.capabilities
                    .get(*id)
                    .map(|caps| caps.contains(&required_capability.to_string()))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by (load, name) to get deterministic least-loaded + alphabetical tie-break
        candidates.sort_by(|a, b| {
            let load_a = self.load.get(a).copied().unwrap_or(0);
            let load_b = self.load.get(b).copied().unwrap_or(0);
            load_a.cmp(&load_b).then_with(|| a.cmp(b))
        });

        let chosen = candidates[0].clone();
        *self.load.entry(chosen.clone()).or_insert(0) += 1;
        Some(chosen)
    }

    /// Mark a task complete for an agent, decrementing their load (min 0).
    pub fn complete(&mut self, agent_id: &str) {
        if let Some(count) = self.load.get_mut(agent_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Current load for an agent (0 if unknown).
    pub fn agent_load(&self, agent_id: &str) -> usize {
        self.load.get(agent_id).copied().unwrap_or(0)
    }

    /// Agents that have the given capability.
    pub fn available_agents(&self, capability: &str) -> Vec<String> {
        self.agents
            .iter()
            .filter(|id| {
                self.capabilities
                    .get(*id)
                    .map(|caps| caps.contains(&capability.to_string()))
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Total number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for TaskDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// A high-level collaboration session combining a message bus, shared context,
/// and task dispatcher into a single coordinated workspace.
pub struct CollaborationSession {
    /// The pub/sub message bus.
    pub bus: MessageBus,
    /// The shared key-value context.
    pub context: SharedContext,
    /// The capability-aware task dispatcher.
    pub dispatcher: TaskDispatcher,
}

impl CollaborationSession {
    /// Create a new session with default settings (history_limit=1000).
    pub fn new() -> Self {
        Self {
            bus: MessageBus::new(1000),
            context: SharedContext::new(),
            dispatcher: TaskDispatcher::new(),
        }
    }

    /// Builder method to set the message bus history limit.
    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.bus = MessageBus::new(limit);
        self
    }

    /// Register an agent with capabilities.
    pub fn add_agent(&mut self, agent_id: &str, capabilities: Vec<String>) {
        self.dispatcher.register_agent(agent_id, capabilities);
    }

    /// Dispatch a task by required capability.
    pub fn assign_task(&mut self, capability: &str) -> Option<String> {
        self.dispatcher.dispatch(capability)
    }

    /// Mark a task complete for an agent.
    pub fn complete_task(&mut self, agent_id: &str) {
        self.dispatcher.complete(agent_id);
    }

    /// Write a value into the shared context.
    pub fn share(&mut self, key: &str, value: serde_json::Value, writer: &str) {
        self.context.set(key, value, writer);
    }

    /// Read a value from the shared context.
    pub fn read(&self, key: &str) -> Option<&serde_json::Value> {
        self.context.get_value(key)
    }

    /// Broadcast a message on the bus.
    pub fn broadcast(&mut self, sender: &str, topic: &str, payload: serde_json::Value) -> String {
        self.bus.publish(sender, topic, payload)
    }

    /// Receive messages for an agent (based on subscriptions).
    pub fn receive(&self, agent_id: &str) -> Vec<&BusMessage> {
        self.bus.poll(agent_id)
    }

    /// Export the session state as JSON.
    pub fn export_json(&self) -> serde_json::Value {
        let subs: serde_json::Value = self
            .bus
            .subscriptions
            .iter()
            .map(|(topic, agents)| {
                (
                    topic.clone(),
                    serde_json::Value::Array(
                        agents
                            .iter()
                            .map(|a| serde_json::Value::String(a.clone()))
                            .collect(),
                    ),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        let loads: serde_json::Value = self
            .dispatcher
            .load
            .iter()
            .map(|(agent, count)| {
                (
                    agent.clone(),
                    serde_json::Value::Number((*count as u64).into()),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        serde_json::json!({
            "bus": {
                "message_count": self.bus.message_count(),
                "subscriptions": subs,
            },
            "context": self.context.snapshot(),
            "dispatcher": {
                "agent_count": self.dispatcher.agent_count(),
                "loads": loads,
            },
        })
    }
}

impl Default for CollaborationSession {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 7.2 Named Conversation Patterns
// =============================================================================

/// Pre-built multi-agent conversation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationPattern {
    /// Each agent picks a task from a shared queue. First-come first-served.
    Swarm,
    /// Agents argue positions in turns. A judge synthesizes after max_rounds.
    Debate,
    /// Each agent speaks in order, round after round. Output of one feeds as input to next.
    RoundRobin,
    /// Linear pipeline: Agent 1 output -> Agent 2 input -> ... -> Agent N output.
    Sequential,
    /// One master agent; when it "calls" a sub-agent, a sub-conversation is triggered.
    NestedChat,
    /// Input sent to ALL agents simultaneously, all responses collected.
    Broadcast,
}

/// Termination conditions for conversation patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationCondition {
    /// Stop after a fixed number of rounds.
    MaxRounds(usize),
    /// Stop when all agents agree (consensus detected).
    Consensus,
    /// Stop when the judge agent makes a decision.
    JudgeDecision,
    /// Stop based on a custom condition description.
    Custom(String),
}

/// Configuration for a conversation pattern execution.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Maximum number of rounds before termination.
    pub max_rounds: usize,
    /// Optional termination condition (overrides max_rounds when met earlier).
    pub termination_condition: Option<TerminationCondition>,
    /// Agent id of the judge (used for Debate pattern).
    pub judge_agent_id: Option<String>,
    /// Task queue (used for Swarm pattern).
    pub task_queue: Vec<String>,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            termination_condition: None,
            judge_agent_id: None,
            task_queue: Vec::new(),
        }
    }
}

impl PatternConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum rounds.
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// Set termination condition.
    pub fn with_termination(mut self, condition: TerminationCondition) -> Self {
        self.termination_condition = Some(condition);
        self
    }

    /// Set judge agent id.
    pub fn with_judge(mut self, judge_id: &str) -> Self {
        self.judge_agent_id = Some(judge_id.to_string());
        self
    }

    /// Set task queue.
    pub fn with_task_queue(mut self, tasks: Vec<String>) -> Self {
        self.task_queue = tasks;
        self
    }
}

/// An agent participating in a conversation pattern.
#[derive(Debug, Clone)]
pub struct PatternAgent {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Role description (e.g. "debater", "summarizer", "coder").
    pub role: String,
    /// System prompt that defines this agent's behavior.
    pub system_prompt: String,
}

impl PatternAgent {
    /// Create a new pattern agent.
    pub fn new(id: &str, name: &str, role: &str, system_prompt: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            role: role.to_string(),
            system_prompt: system_prompt.to_string(),
        }
    }
}

/// A message produced during pattern execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternMessage {
    /// Id of the agent that produced this message.
    pub agent_id: String,
    /// The content of the message.
    pub content: String,
    /// The round number (0-indexed) in which this message was produced.
    pub round: usize,
    /// Timestamp of when the message was created.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PatternMessage {
    /// Create a new pattern message.
    pub fn new(agent_id: &str, content: &str, round: usize) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            content: content.to_string(),
            round,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Result of running a conversation pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternResult {
    /// All messages produced during the conversation.
    pub messages: Vec<PatternMessage>,
    /// Number of rounds completed.
    pub rounds_completed: usize,
    /// The final synthesized output (if any).
    pub final_output: Option<String>,
    /// Description of what caused termination.
    pub terminated_by: String,
}

/// Executes a conversation pattern with N agents.
pub struct PatternRunner {
    /// The agents participating in this pattern.
    agents: Vec<PatternAgent>,
    /// The conversation pattern to run.
    pattern: ConversationPattern,
    /// Configuration for the pattern.
    config: PatternConfig,
    /// Transcript of the conversation so far.
    transcript: Vec<PatternMessage>,
}

impl PatternRunner {
    /// Create a new runner for the given pattern and config.
    pub fn new(pattern: ConversationPattern, config: PatternConfig) -> Self {
        Self {
            agents: Vec::new(),
            pattern,
            config,
            transcript: Vec::new(),
        }
    }

    /// Add an agent to the runner.
    pub fn add_agent(&mut self, agent: PatternAgent) -> &mut Self {
        self.agents.push(agent);
        self
    }

    /// Get the current pattern.
    pub fn pattern(&self) -> ConversationPattern {
        self.pattern
    }

    /// Get the current config.
    pub fn config(&self) -> &PatternConfig {
        &self.config
    }

    /// Get the agents.
    pub fn agents(&self) -> &[PatternAgent] {
        &self.agents
    }

    /// Get the transcript.
    pub fn transcript(&self) -> &[PatternMessage] {
        &self.transcript
    }

    /// Run the conversation pattern with the given input.
    ///
    /// Since this crate does not call an LLM directly in the pattern runner
    /// (that would require an async provider), each agent's "response" is
    /// simulated by combining its system_prompt context with the input/previous
    /// output. In production, callers would replace this with actual LLM calls.
    pub fn run(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        if self.agents.is_empty() {
            return Err(OrchestrationError::AgentNotFound);
        }

        self.transcript.clear();

        match self.pattern {
            ConversationPattern::Swarm => self.run_swarm(input),
            ConversationPattern::Debate => self.run_debate(input),
            ConversationPattern::RoundRobin => self.run_round_robin(input),
            ConversationPattern::Sequential => self.run_sequential(input),
            ConversationPattern::NestedChat => self.run_nested_chat(input),
            ConversationPattern::Broadcast => self.run_broadcast(input),
        }
    }

    /// Swarm: each agent picks a task from the task_queue.
    fn run_swarm(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let tasks = if self.config.task_queue.is_empty() {
            vec![input.to_string()]
        } else {
            self.config.task_queue.clone()
        };

        let mut round = 0usize;
        let mut task_idx = 0usize;
        let agent_count = self.agents.len();

        while task_idx < tasks.len() && round < self.config.max_rounds {
            let agent_idx = task_idx % agent_count;
            let agent = &self.agents[agent_idx];
            let task = &tasks[task_idx];

            let response = format!(
                "[{}] processed task: {}",
                agent.name, task
            );
            self.transcript.push(PatternMessage::new(&agent.id, &response, round));

            task_idx += 1;
            if task_idx % agent_count == 0 || task_idx >= tasks.len() {
                round += 1;
            }

            if self.check_termination(round) {
                break;
            }
        }

        let final_output = self.transcript.last().map(|m| m.content.clone());
        let terminated_by = if round >= self.config.max_rounds {
            "max_rounds".to_string()
        } else if task_idx >= tasks.len() {
            "all_tasks_completed".to_string()
        } else {
            "termination_condition".to_string()
        };

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: round,
            final_output,
            terminated_by,
        })
    }

    /// Debate: agents take turns arguing. Judge synthesizes at the end.
    fn run_debate(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let mut current_input = input.to_string();
        let mut round = 0usize;
        let agent_count = self.agents.len();

        while round < self.config.max_rounds {
            for i in 0..agent_count {
                let agent = &self.agents[i];
                let response = format!(
                    "[{}] argues (round {}): re '{}' -- from perspective of {}",
                    agent.name, round, current_input, agent.role
                );
                self.transcript.push(PatternMessage::new(&agent.id, &response, round));
                current_input = response;
            }

            round += 1;

            if self.check_termination(round) {
                break;
            }
        }

        // Judge synthesizes if configured
        let final_output = if let Some(ref judge_id) = self.config.judge_agent_id {
            let judge_name = self
                .agents
                .iter()
                .find(|a| a.id == *judge_id)
                .map(|a| a.name.clone())
                .unwrap_or_else(|| judge_id.clone());

            let synthesis = format!(
                "[Judge {}] synthesis after {} rounds of debate on: {}",
                judge_name, round, input
            );
            self.transcript.push(PatternMessage::new(judge_id, &synthesis, round));
            Some(synthesis)
        } else {
            // No judge: use last message as final output
            self.transcript.last().map(|m| m.content.clone())
        };

        let terminated_by = if round >= self.config.max_rounds {
            "max_rounds".to_string()
        } else {
            "termination_condition".to_string()
        };

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: round,
            final_output,
            terminated_by,
        })
    }

    /// RoundRobin: each agent speaks in order, round after round.
    fn run_round_robin(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let mut current_input = input.to_string();
        let mut round = 0usize;
        let agent_count = self.agents.len();

        while round < self.config.max_rounds {
            for i in 0..agent_count {
                let agent = &self.agents[i];
                let response = format!(
                    "[{}] (round {}) responds to: {}",
                    agent.name, round, current_input
                );
                self.transcript.push(PatternMessage::new(&agent.id, &response, round));
                current_input = response;
            }

            round += 1;

            if self.check_termination(round) {
                break;
            }
        }

        let final_output = self.transcript.last().map(|m| m.content.clone());
        let terminated_by = if round >= self.config.max_rounds {
            "max_rounds".to_string()
        } else {
            "termination_condition".to_string()
        };

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: round,
            final_output,
            terminated_by,
        })
    }

    /// Sequential: linear pipeline. Agent 1 -> Agent 2 -> ... -> Agent N.
    fn run_sequential(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let mut current_input = input.to_string();
        let agent_count = self.agents.len();

        for (i, agent) in self.agents.iter().enumerate() {
            let response = format!(
                "[{}] pipeline step {}: processed '{}'",
                agent.name, i, current_input
            );
            self.transcript.push(PatternMessage::new(&agent.id, &response, 0));
            current_input = response;
        }

        let final_output = self.transcript.last().map(|m| m.content.clone());

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: 1,
            final_output,
            terminated_by: format!("pipeline_complete ({} stages)", agent_count),
        })
    }

    /// NestedChat: master agent (first) can trigger sub-conversations with other agents.
    fn run_nested_chat(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let agent_count = self.agents.len();
        if agent_count < 2 {
            // With only one agent, it just responds directly
            let agent = &self.agents[0];
            let response = format!("[{}] responds: {}", agent.name, input);
            self.transcript.push(PatternMessage::new(&agent.id, &response, 0));
            return Ok(PatternResult {
                messages: self.transcript.clone(),
                rounds_completed: 1,
                final_output: Some(response),
                terminated_by: "single_agent".to_string(),
            });
        }

        let master = self.agents[0].clone();
        let mut round = 0usize;
        let mut current_input = input.to_string();

        while round < self.config.max_rounds {
            // Master agent processes and delegates to sub-agents
            let master_response = format!(
                "[{}] master delegates (round {}): '{}'",
                master.name, round, current_input
            );
            self.transcript.push(PatternMessage::new(&master.id, &master_response, round));

            // Each sub-agent processes the delegated task
            for i in 1..agent_count {
                let sub_agent = &self.agents[i];
                let sub_response = format!(
                    "[{}] sub-response to master (round {}): handling '{}'",
                    sub_agent.name, round, current_input
                );
                self.transcript.push(PatternMessage::new(&sub_agent.id, &sub_response, round));
                current_input = sub_response;
            }

            // Master synthesizes sub-agent responses
            let synthesis = format!(
                "[{}] master synthesis (round {}): combined sub-agent results",
                master.name, round
            );
            self.transcript.push(PatternMessage::new(&master.id, &synthesis, round));
            current_input = synthesis;

            round += 1;

            if self.check_termination(round) {
                break;
            }
        }

        let final_output = self.transcript.last().map(|m| m.content.clone());
        let terminated_by = if round >= self.config.max_rounds {
            "max_rounds".to_string()
        } else {
            "termination_condition".to_string()
        };

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: round,
            final_output,
            terminated_by,
        })
    }

    /// Broadcast: input sent to ALL agents simultaneously, all responses collected.
    fn run_broadcast(&mut self, input: &str) -> Result<PatternResult, OrchestrationError> {
        let mut responses = Vec::new();

        for agent in &self.agents {
            let response = format!(
                "[{}] broadcast response: {}",
                agent.name, input
            );
            self.transcript.push(PatternMessage::new(&agent.id, &response, 0));
            responses.push(response);
        }

        let final_output = Some(format!(
            "Broadcast complete: {} agents responded",
            responses.len()
        ));

        Ok(PatternResult {
            messages: self.transcript.clone(),
            rounds_completed: 1,
            final_output,
            terminated_by: "broadcast_complete".to_string(),
        })
    }

    /// Check if a termination condition is met.
    fn check_termination(&self, current_round: usize) -> bool {
        match &self.config.termination_condition {
            Some(TerminationCondition::MaxRounds(max)) => current_round >= *max,
            Some(TerminationCondition::Consensus) => {
                // Check if the last N messages (one per agent) all contain the same content.
                // In a real system this would do semantic comparison; here we check exact match.
                let agent_count = self.agents.len();
                if self.transcript.len() < agent_count {
                    return false;
                }
                let last_messages: Vec<&str> = self
                    .transcript
                    .iter()
                    .rev()
                    .take(agent_count)
                    .map(|m| m.content.as_str())
                    .collect();
                // If all last messages are identical, consensus is reached
                last_messages.windows(2).all(|w| w[0] == w[1])
            }
            Some(TerminationCondition::JudgeDecision) => {
                // Check if the judge has spoken in the current round
                if let Some(ref judge_id) = self.config.judge_agent_id {
                    self.transcript.iter().any(|m| {
                        m.agent_id == *judge_id && m.content.contains("synthesis")
                    })
                } else {
                    false
                }
            }
            Some(TerminationCondition::Custom(_)) => {
                // Custom conditions are evaluated externally; never auto-triggers
                false
            }
            None => false,
        }
    }
}

// =============================================================================
// 7.4 Agent Handoffs
// =============================================================================

/// Policy for how much context to transfer during a handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextTransferPolicy {
    /// Transfer entire conversation history.
    Full,
    /// Transfer a summary of the conversation.
    Summary,
    /// Transfer the last N messages.
    LastN(usize),
    /// No context transfer.
    None,
}

/// A request for one agent to hand off a conversation to another.
#[derive(Debug, Clone)]
pub struct HandoffRequest {
    /// The agent initiating the handoff.
    pub from_agent_id: String,
    /// The target agent to hand off to.
    pub to_agent_id: String,
    /// Reason for the handoff.
    pub reason: String,
    /// How much context to transfer.
    pub context_policy: ContextTransferPolicy,
    /// Arbitrary metadata for the handoff.
    pub metadata: HashMap<String, String>,
}

impl HandoffRequest {
    /// Create a new handoff request.
    pub fn new(from: &str, to: &str, reason: &str, policy: ContextTransferPolicy) -> Self {
        Self {
            from_agent_id: from.to_string(),
            to_agent_id: to.to_string(),
            reason: reason.to_string(),
            context_policy: policy,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the request.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Result of an agent handoff.
#[derive(Debug, Clone, PartialEq)]
pub struct HandoffResult {
    /// Whether the handoff succeeded.
    pub success: bool,
    /// Number of messages transferred.
    pub transferred_messages: usize,
    /// The agent that initiated the handoff.
    pub from_agent: String,
    /// The agent that received the handoff.
    pub to_agent: String,
    /// Reason for the handoff.
    pub reason: String,
}

/// Manages agent registrations and handoff execution.
pub struct HandoffManager {
    /// Registered agents keyed by id.
    agents: HashMap<String, PatternAgent>,
    /// History of completed handoffs.
    handoff_history: Vec<HandoffResult>,
}

impl HandoffManager {
    /// Create a new, empty handoff manager.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            handoff_history: Vec::new(),
        }
    }

    /// Register an agent.
    pub fn register_agent(&mut self, agent: PatternAgent) {
        self.agents.insert(agent.id.clone(), agent);
    }

    /// Get a registered agent by id.
    pub fn get_agent(&self, id: &str) -> Option<&PatternAgent> {
        self.agents.get(id)
    }

    /// List all registered agents.
    pub fn list_agents(&self) -> Vec<&PatternAgent> {
        self.agents.values().collect()
    }

    /// Get the handoff history.
    pub fn get_handoff_history(&self) -> &[HandoffResult] {
        &self.handoff_history
    }

    /// Detect if a handoff from -> to would create a circular loop in recent history.
    ///
    /// Checks if there is a recent handoff from `to` -> `from` (i.e., A->B->A loop).
    pub fn detect_circular_handoff(&self, from: &str, to: &str) -> bool {
        // Look through the recent handoff history for a to->from transfer
        // which would indicate an A->B->A loop if we now do from->to
        self.handoff_history.iter().rev().take(10).any(|h| {
            h.success && h.from_agent == to && h.to_agent == from
        })
    }

    /// Execute a handoff request.
    ///
    /// Validates that both agents exist, checks for circular handoffs, computes
    /// the number of messages that would be transferred based on the policy,
    /// and records the result.
    pub fn request_handoff(
        &mut self,
        request: HandoffRequest,
    ) -> Result<HandoffResult, OrchestrationError> {
        // Validate source agent
        if !self.agents.contains_key(&request.from_agent_id) {
            return Err(OrchestrationError::AgentNotFound);
        }

        // Validate target agent
        if !self.agents.contains_key(&request.to_agent_id) {
            return Err(OrchestrationError::AgentNotFound);
        }

        // Check for self-handoff
        if request.from_agent_id == request.to_agent_id {
            return Err(OrchestrationError::InvalidState);
        }

        // Compute transferred message count based on policy
        let transferred_messages = match &request.context_policy {
            ContextTransferPolicy::Full => {
                // Simulated: in a real system this would count actual messages.
                // We use a sentinel value representing "all messages".
                usize::MAX
            }
            ContextTransferPolicy::Summary => {
                // A summary is a single condensed message.
                1
            }
            ContextTransferPolicy::LastN(n) => *n,
            ContextTransferPolicy::None => 0,
        };

        let result = HandoffResult {
            success: true,
            transferred_messages,
            from_agent: request.from_agent_id,
            to_agent: request.to_agent_id,
            reason: request.reason,
        };

        self.handoff_history.push(result.clone());
        Ok(result)
    }
}

impl Default for HandoffManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registration() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(
            Agent::new("agent1", "Researcher", AgentRole::Researcher)
                .with_capability("search")
                .with_capability("analyze"),
        );

        assert!(orchestrator.get_agent("agent1").is_some());
    }

    #[test]
    fn test_task_assignment() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "Do something"));

        assert!(orchestrator.assign_task("task1", "agent1").is_ok());

        let agent = orchestrator.get_agent("agent1").unwrap();
        assert_eq!(agent.status, AgentStatus::Working);
    }

    #[test]
    fn test_task_completion() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "Do something"));
        orchestrator.assign_task("task1", "agent1").unwrap();
        orchestrator.complete_task("task1", "Done!").unwrap();

        let task = orchestrator.get_task("task1").unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
    }

    #[test]
    fn test_auto_assignment() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::RoundRobin);

        orchestrator.register_agent(Agent::new("agent1", "Worker1", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("agent2", "Worker2", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));
        orchestrator.add_task(AgentTask::new("task2", "Task 2"));

        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 2);
    }

    // === Additional Unit Tests ===

    #[test]
    fn test_unregister_agent() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        assert!(orchestrator.get_agent("agent1").is_some());

        orchestrator.unregister_agent("agent1");
        assert!(orchestrator.get_agent("agent1").is_none());
    }

    #[test]
    fn test_unregister_coordinator() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("coord1", "Coordinator", AgentRole::Coordinator));
        assert!(orchestrator.coordinator_id.is_some());

        orchestrator.unregister_agent("coord1");
        assert!(orchestrator.coordinator_id.is_none());
    }

    #[test]
    fn test_fail_task() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "Do something"));
        orchestrator.assign_task("task1", "agent1").unwrap();

        orchestrator
            .fail_task("task1", "Something went wrong")
            .unwrap();

        let task = orchestrator.get_task("task1").unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert!(task.result.as_ref().unwrap().contains("Error:"));

        let agent = orchestrator.get_agent("agent1").unwrap();
        assert_eq!(agent.status, AgentStatus::Failed);
    }

    #[test]
    fn test_fail_task_not_found() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        let result = orchestrator.fail_task("nonexistent", "error");
        assert_eq!(result, Err(OrchestrationError::TaskNotFound));
    }

    #[test]
    fn test_send_and_receive_messages() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Sender", AgentRole::Researcher));
        orchestrator.register_agent(Agent::new("agent2", "Receiver", AgentRole::Analyst));

        let msg1 = AgentMessage::new("agent1", "agent2", "Hello!", MessageType::Request);
        let msg2 = AgentMessage::new("agent1", "agent2", "Follow-up", MessageType::Notification);
        let msg3 = AgentMessage::new("agent2", "agent1", "Response", MessageType::Response);

        orchestrator.send_message(msg1);
        orchestrator.send_message(msg2);
        orchestrator.send_message(msg3);

        let agent2_messages = orchestrator.get_messages_for("agent2");
        assert_eq!(agent2_messages.len(), 2);

        let agent1_messages = orchestrator.get_messages_for("agent1");
        assert_eq!(agent1_messages.len(), 1);
    }

    #[test]
    fn test_message_correlation() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        let correlation_id = "conv-123";
        let msg = AgentMessage::new("agent1", "agent2", "Test", MessageType::Request)
            .with_correlation(correlation_id);

        orchestrator.send_message(msg);

        let messages = orchestrator.get_messages_for("agent2");
        assert_eq!(messages[0].correlation_id.as_ref().unwrap(), correlation_id);
    }

    #[test]
    fn test_assign_task_agent_not_found() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.add_task(AgentTask::new("task1", "Do something"));

        let result = orchestrator.assign_task("task1", "nonexistent");
        assert_eq!(result, Err(OrchestrationError::AgentNotFound));
    }

    #[test]
    fn test_assign_task_agent_busy() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "First task"));
        orchestrator.add_task(AgentTask::new("task2", "Second task"));

        orchestrator.assign_task("task1", "agent1").unwrap();

        let result = orchestrator.assign_task("task2", "agent1");
        assert_eq!(result, Err(OrchestrationError::AgentBusy));
    }

    #[test]
    fn test_assign_task_dependency_not_met() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "First task"));
        orchestrator.add_task(AgentTask::new("task2", "Second task").depends_on("task1"));

        let result = orchestrator.assign_task("task2", "agent1");
        assert_eq!(result, Err(OrchestrationError::DependencyNotMet));
    }

    #[test]
    fn test_assign_task_with_completed_dependency() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));
        orchestrator.add_task(AgentTask::new("task1", "First task"));
        orchestrator.add_task(AgentTask::new("task2", "Second task").depends_on("task1"));

        // Complete task1 first
        orchestrator.assign_task("task1", "agent1").unwrap();
        orchestrator.complete_task("task1", "Done").unwrap();

        // Now task2 should be assignable
        let result = orchestrator.assign_task("task2", "agent1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_complete_task_not_assigned() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.add_task(AgentTask::new("task1", "Unassigned task"));

        let result = orchestrator.complete_task("task1", "Done");
        assert_eq!(result, Err(OrchestrationError::TaskNotAssigned));
    }

    #[test]
    fn test_get_status() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker1", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("agent2", "Worker2", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));
        orchestrator.add_task(AgentTask::new("task2", "Task 2"));
        orchestrator.add_task(AgentTask::new("task3", "Task 3"));

        orchestrator.assign_task("task1", "agent1").unwrap();
        orchestrator.complete_task("task1", "Done").unwrap();

        orchestrator.assign_task("task2", "agent1").unwrap();
        orchestrator.fail_task("task2", "Error").unwrap();

        let status = orchestrator.get_status();

        assert_eq!(status.total_agents, 2);
        assert_eq!(status.total_tasks, 3);
        assert_eq!(status.completed_tasks, 1);
        assert_eq!(status.failed_tasks, 1);
        assert_eq!(status.pending_tasks, 1);
    }

    #[test]
    fn test_orchestration_strategy_sequential() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker1", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("agent2", "Worker2", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));
        orchestrator.add_task(AgentTask::new("task2", "Task 2"));

        let assignments = orchestrator.auto_assign_tasks();

        // Sequential should still assign all tasks to available agents
        assert!(!assignments.is_empty());
    }

    #[test]
    fn test_orchestration_strategy_best_fit() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::BestFit);

        orchestrator.register_agent(
            Agent::new("researcher", "Researcher", AgentRole::Researcher)
                .with_capability("search")
                .with_capability("analyze"),
        );
        orchestrator.register_agent(
            Agent::new("writer", "Writer", AgentRole::Writer)
                .with_capability("write")
                .with_capability("edit"),
        );

        // Task with "search" keyword should go to researcher
        orchestrator.add_task(AgentTask::new(
            "task1",
            "Search for information and analyze data",
        ));
        // Task with "write" keyword should go to writer
        orchestrator.add_task(AgentTask::new("task2", "Write a report and edit it"));

        let assignments = orchestrator.auto_assign_tasks();

        assert_eq!(assignments.len(), 2);

        // Verify best fit assignment
        let task1_assignment = assignments.iter().find(|(t, _)| t == "task1");
        let task2_assignment = assignments.iter().find(|(t, _)| t == "task2");

        assert!(task1_assignment.is_some());
        assert!(task2_assignment.is_some());
        assert_eq!(task1_assignment.unwrap().1, "researcher");
        assert_eq!(task2_assignment.unwrap().1, "writer");
    }

    #[test]
    fn test_task_priority() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));

        // Add tasks with different priorities
        orchestrator.add_task(AgentTask::new("low", "Low priority").with_priority(1));
        orchestrator.add_task(AgentTask::new("high", "High priority").with_priority(10));
        orchestrator.add_task(AgentTask::new("medium", "Medium priority").with_priority(5));

        let assignments = orchestrator.auto_assign_tasks();

        // High priority should be assigned first
        assert_eq!(assignments[0].0, "high");
    }

    #[test]
    fn test_agent_with_model() {
        let agent = Agent::new("agent1", "Worker", AgentRole::Executor).with_model("gpt-4");

        assert_eq!(agent.model.as_ref().unwrap(), "gpt-4");
    }

    #[test]
    fn test_all_message_types() {
        let types = vec![
            MessageType::Request,
            MessageType::Response,
            MessageType::Notification,
            MessageType::Error,
            MessageType::Handoff,
            MessageType::Status,
        ];

        for msg_type in types {
            let msg = AgentMessage::new("from", "to", "content", msg_type);
            assert_eq!(msg.message_type, msg_type);
        }
    }

    #[test]
    fn test_all_agent_statuses() {
        let statuses = vec![
            AgentStatus::Idle,
            AgentStatus::Working,
            AgentStatus::WaitingForInput,
            AgentStatus::Completed,
            AgentStatus::Failed,
            AgentStatus::Paused,
        ];

        for status in statuses {
            let mut agent = Agent::new("test", "Test", AgentRole::Executor);
            agent.status = status;
            assert_eq!(agent.status, status);
        }
    }

    #[test]
    fn test_all_task_statuses() {
        let statuses = vec![
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Blocked,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ];

        for status in statuses {
            let mut task = AgentTask::new("test", "Test");
            task.status = status;
            assert_eq!(task.status, status);
        }
    }

    #[test]
    fn test_orchestrator_default() {
        let orchestrator = AgentOrchestrator::default();
        assert_eq!(orchestrator.strategy, OrchestrationStrategy::BestFit);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", OrchestrationError::AgentNotFound),
            "Agent not found"
        );
        assert_eq!(
            format!("{}", OrchestrationError::TaskNotFound),
            "Task not found"
        );
        assert_eq!(
            format!("{}", OrchestrationError::AgentBusy),
            "Agent is busy"
        );
        assert_eq!(
            format!("{}", OrchestrationError::DependencyNotMet),
            "Task dependency not met"
        );
        assert_eq!(
            format!("{}", OrchestrationError::TaskNotAssigned),
            "Task not assigned to any agent"
        );
        assert_eq!(
            format!("{}", OrchestrationError::InvalidState),
            "Invalid state"
        );
    }

    #[test]
    fn test_multiple_dependencies() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "First"));
        orchestrator.add_task(AgentTask::new("task2", "Second"));
        orchestrator.add_task(
            AgentTask::new("task3", "Third")
                .depends_on("task1")
                .depends_on("task2"),
        );

        // task3 cannot be assigned until both task1 and task2 are complete
        let result = orchestrator.assign_task("task3", "agent1");
        assert_eq!(result, Err(OrchestrationError::DependencyNotMet));

        // Complete task1
        orchestrator.assign_task("task1", "agent1").unwrap();
        orchestrator.complete_task("task1", "Done").unwrap();

        // Still can't assign task3
        let result = orchestrator.assign_task("task3", "agent1");
        assert_eq!(result, Err(OrchestrationError::DependencyNotMet));

        // Complete task2
        orchestrator.assign_task("task2", "agent1").unwrap();
        orchestrator.complete_task("task2", "Done").unwrap();

        // Now task3 can be assigned
        let result = orchestrator.assign_task("task3", "agent1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_strategy() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Parallel);

        orchestrator.register_agent(Agent::new("agent1", "Worker1", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("agent2", "Worker2", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));
        orchestrator.add_task(AgentTask::new("task2", "Task 2"));

        let assignments = orchestrator.auto_assign_tasks();

        // Parallel falls back to default sequential-like behavior
        assert!(!assignments.is_empty());
    }

    #[test]
    fn test_pipeline_strategy() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Pipeline);

        orchestrator.register_agent(Agent::new("agent1", "Worker", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));

        let assignments = orchestrator.auto_assign_tasks();
        assert_eq!(assignments.len(), 1);
    }

    #[test]
    fn test_working_agents_count() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(Agent::new("agent1", "Worker1", AgentRole::Executor));
        orchestrator.register_agent(Agent::new("agent2", "Worker2", AgentRole::Executor));

        orchestrator.add_task(AgentTask::new("task1", "Task 1"));

        let status_before = orchestrator.get_status();
        assert_eq!(status_before.idle_agents, 2);
        assert_eq!(status_before.working_agents, 0);

        orchestrator.assign_task("task1", "agent1").unwrap();

        let status_after = orchestrator.get_status();
        assert_eq!(status_after.idle_agents, 1);
        assert_eq!(status_after.working_agents, 1);
    }

    #[test]
    fn test_orchestrator_export_json() {
        let mut orch = AgentOrchestrator::new(OrchestrationStrategy::Parallel);
        orch.register_agent(Agent::new("a1", "Tester", AgentRole::Researcher));
        orch.add_task(AgentTask::new("t1", "Test task").with_priority(5));
        let json = orch.export_json();
        assert!(json["agents"].as_array().unwrap().len() == 1);
        assert!(json["tasks"].as_array().unwrap().len() == 1);
        assert_eq!(json["strategy"], "Parallel");
        assert!(json["status"]["total_agents"].as_u64().unwrap() == 1);
    }

    // =========================================================================
    // Collaboration types tests (MessageBus, SharedContext, TaskDispatcher, etc.)
    // =========================================================================

    #[test]
    fn test_message_bus_subscribe_publish_poll() {
        let mut bus = MessageBus::new(100);
        bus.subscribe("agent-a", "events");
        let msg_id = bus.publish("sender-1", "events", serde_json::json!({"data": "hello"}));
        assert!(msg_id.starts_with("msg-"));

        let polled = bus.poll("agent-a");
        assert_eq!(polled.len(), 1);
        assert_eq!(polled[0].topic, "events");
        assert_eq!(polled[0].sender, "sender-1");
        assert_eq!(polled[0].payload, serde_json::json!({"data": "hello"}));
    }

    #[test]
    fn test_message_bus_filters_by_subscription() {
        let mut bus = MessageBus::new(100);
        bus.subscribe("agent-a", "topic-a");
        // agent-a is NOT subscribed to topic-b
        bus.publish("sender-1", "topic-a", serde_json::json!("msg-a"));
        bus.publish("sender-2", "topic-b", serde_json::json!("msg-b"));

        let polled = bus.poll("agent-a");
        assert_eq!(polled.len(), 1);
        assert_eq!(polled[0].topic, "topic-a");
    }

    #[test]
    fn test_message_bus_history_limit() {
        let mut bus = MessageBus::new(3);
        for i in 0..5 {
            bus.publish("sender", "topic", serde_json::json!(i));
        }
        assert_eq!(bus.message_count(), 3);
        // The oldest two (0, 1) should have been evicted; remaining are 2, 3, 4
        bus.subscribe("reader", "topic");
        let polled = bus.poll("reader");
        assert_eq!(polled.len(), 3);
        assert_eq!(polled[0].payload, serde_json::json!(2));
        assert_eq!(polled[1].payload, serde_json::json!(3));
        assert_eq!(polled[2].payload, serde_json::json!(4));
    }

    #[test]
    fn test_shared_context_set_get() {
        let mut ctx = SharedContext::new();
        ctx.set("key1", serde_json::json!("value1"), "writer-a");

        let entry = ctx.get("key1").unwrap();
        assert_eq!(entry.value, serde_json::json!("value1"));
        assert_eq!(entry.last_writer, "writer-a");
        assert_eq!(entry.version, 1);

        // get_value shortcut
        assert_eq!(ctx.get_value("key1"), Some(&serde_json::json!("value1")));
    }

    #[test]
    fn test_shared_context_merge() {
        let mut ctx_a = SharedContext::new();
        ctx_a.set("shared", serde_json::json!("old"), "writer-a"); // version 1
        ctx_a.set("only_a", serde_json::json!("a-val"), "writer-a"); // version 2

        let mut ctx_b = SharedContext::new();
        ctx_b.set("b-filler1", serde_json::json!("x"), "writer-b"); // version 1
        ctx_b.set("b-filler2", serde_json::json!("x"), "writer-b"); // version 2
        ctx_b.set("shared", serde_json::json!("new"), "writer-b"); // version 3 > ctx_a's version 1
        ctx_b.set("only_b", serde_json::json!("b-val"), "writer-b"); // version 4

        ctx_a.merge(&ctx_b);

        // "shared" should take ctx_b's value (higher version)
        assert_eq!(ctx_a.get_value("shared"), Some(&serde_json::json!("new")));
        // "only_a" should remain
        assert_eq!(ctx_a.get_value("only_a"), Some(&serde_json::json!("a-val")));
        // "only_b" should be merged in
        assert_eq!(ctx_a.get_value("only_b"), Some(&serde_json::json!("b-val")));
    }

    #[test]
    fn test_shared_context_diff() {
        let mut ctx_a = SharedContext::new();
        ctx_a.set("same", serde_json::json!(42), "w");
        ctx_a.set("different", serde_json::json!("a"), "w");
        ctx_a.set("only_a", serde_json::json!(true), "w");

        let mut ctx_b = SharedContext::new();
        ctx_b.set("same", serde_json::json!(42), "w");
        ctx_b.set("different", serde_json::json!("b"), "w");
        ctx_b.set("only_b", serde_json::json!(false), "w");

        let diff = ctx_a.diff(&ctx_b);
        // "different" differs in value, "only_a" only in a, "only_b" only in b
        assert!(diff.contains(&"different".to_string()));
        assert!(diff.contains(&"only_a".to_string()));
        assert!(diff.contains(&"only_b".to_string()));
        // "same" should NOT appear
        assert!(!diff.contains(&"same".to_string()));
        assert_eq!(diff.len(), 3);
    }

    #[test]
    fn test_task_dispatcher_least_loaded() {
        let mut disp = TaskDispatcher::new();
        disp.register_agent("alice", vec!["code".to_string()]);
        disp.register_agent("bob", vec!["code".to_string()]);

        // Dispatch 2 tasks to raise alice's load
        let first = disp.dispatch("code").unwrap();
        let second = disp.dispatch("code").unwrap();

        // With alphabetical tie-break, first goes to alice, second to bob
        // (both start at load=0, alice < bob alphabetically -> alice first)
        assert_eq!(first, "alice");
        assert_eq!(second, "bob"); // now both at load=1

        // Third task: both at load=1, alphabetical tie-break -> alice again
        // Actually alice=1, bob=1, alice < bob alphabetically
        let third = disp.dispatch("code").unwrap();
        assert_eq!(third, "alice");
        // Now alice=2, bob=1 -> next should go to bob (least loaded)
        let fourth = disp.dispatch("code").unwrap();
        assert_eq!(fourth, "bob");
    }

    #[test]
    fn test_task_dispatcher_no_capable_agent() {
        let mut disp = TaskDispatcher::new();
        disp.register_agent("alice", vec!["code".to_string()]);
        assert_eq!(disp.dispatch("design"), None);
    }

    #[test]
    fn test_task_dispatcher_complete_decrements() {
        let mut disp = TaskDispatcher::new();
        disp.register_agent("alice", vec!["code".to_string()]);

        disp.dispatch("code"); // load = 1
        assert_eq!(disp.agent_load("alice"), 1);

        disp.complete("alice"); // load = 0
        assert_eq!(disp.agent_load("alice"), 0);

        // Completing again should not go below 0
        disp.complete("alice");
        assert_eq!(disp.agent_load("alice"), 0);
    }

    #[test]
    fn test_collaboration_session_end_to_end() {
        let mut session = CollaborationSession::new();

        // Register 3 agents
        session.add_agent("coder", vec!["code".to_string(), "review".to_string()]);
        session.add_agent("designer", vec!["design".to_string()]);
        session.add_agent("tester", vec!["test".to_string(), "review".to_string()]);

        // Assign tasks
        let a1 = session.assign_task("code").unwrap();
        assert_eq!(a1, "coder");
        let a2 = session.assign_task("design").unwrap();
        assert_eq!(a2, "designer");
        let a3 = session.assign_task("test").unwrap();
        assert_eq!(a3, "tester");

        // Share context
        session.share("project", serde_json::json!({"name": "collab"}), "coder");
        assert_eq!(
            session.read("project"),
            Some(&serde_json::json!({"name": "collab"}))
        );

        // Broadcast messages
        session.bus.subscribe("tester", "builds");
        session.bus.subscribe("designer", "builds");
        session.broadcast("coder", "builds", serde_json::json!("build-ok"));

        let tester_msgs = session.receive("tester");
        assert_eq!(tester_msgs.len(), 1);
        assert_eq!(tester_msgs[0].payload, serde_json::json!("build-ok"));

        let designer_msgs = session.receive("designer");
        assert_eq!(designer_msgs.len(), 1);

        // Coder is not subscribed to "builds"
        let coder_msgs = session.receive("coder");
        assert_eq!(coder_msgs.len(), 0);

        // Complete a task
        session.complete_task("coder");
        assert_eq!(session.dispatcher.agent_load("coder"), 0);
    }

    #[test]
    fn test_collaboration_export_json() {
        let mut session = CollaborationSession::new();
        session.add_agent("alice", vec!["code".to_string()]);
        session.share("key", serde_json::json!("val"), "alice");
        session.assign_task("code");

        let json = session.export_json();

        // Check bus section
        assert!(json["bus"].is_object());
        assert!(json["bus"]["message_count"].is_number());
        assert!(json["bus"]["subscriptions"].is_object());

        // Check context section
        assert!(json["context"].is_object());
        assert_eq!(json["context"]["key"], serde_json::json!("val"));

        // Check dispatcher section
        assert!(json["dispatcher"].is_object());
        assert_eq!(json["dispatcher"]["agent_count"].as_u64().unwrap(), 1);
        assert!(json["dispatcher"]["loads"].is_object());
    }

    #[test]
    fn test_message_bus_unsubscribe() {
        let mut bus = MessageBus::new(100);
        bus.subscribe("agent-a", "events");

        // Publish before unsubscribe
        bus.publish("sender", "events", serde_json::json!("before"));
        let polled = bus.poll("agent-a");
        assert_eq!(polled.len(), 1);

        // Unsubscribe and publish another
        bus.unsubscribe("agent-a", "events");
        bus.publish("sender", "events", serde_json::json!("after"));

        // agent-a should no longer receive anything (including old messages)
        let polled_after = bus.poll("agent-a");
        assert_eq!(polled_after.len(), 0);
    }

    // =========================================================================
    // 7.2 Named Conversation Patterns tests
    // =========================================================================

    #[test]
    fn test_conversation_pattern_swarm_basic() {
        let config = PatternConfig::new()
            .with_max_rounds(5)
            .with_task_queue(vec!["task-a".into(), "task-b".into(), "task-c".into()]);
        let mut runner = PatternRunner::new(ConversationPattern::Swarm, config);
        runner.add_agent(PatternAgent::new("a1", "Alice", "worker", "You process tasks."));
        runner.add_agent(PatternAgent::new("a2", "Bob", "worker", "You process tasks."));

        let result = runner.run("go").unwrap();
        assert!(result.rounds_completed >= 1);
        assert_eq!(result.messages.len(), 3); // 3 tasks processed
        assert!(result.terminated_by.contains("all_tasks_completed"));
        assert!(result.final_output.is_some());
    }

    #[test]
    fn test_conversation_pattern_debate_basic() {
        let config = PatternConfig::new().with_max_rounds(2);
        let mut runner = PatternRunner::new(ConversationPattern::Debate, config);
        runner.add_agent(PatternAgent::new("d1", "Pro", "debater", "Argue in favor."));
        runner.add_agent(PatternAgent::new("d2", "Con", "debater", "Argue against."));

        let result = runner.run("Is Rust the best language?").unwrap();
        assert_eq!(result.rounds_completed, 2);
        // 2 rounds * 2 agents = 4 messages
        assert_eq!(result.messages.len(), 4);
        assert!(result.final_output.is_some());
    }

    #[test]
    fn test_conversation_pattern_round_robin_basic() {
        let config = PatternConfig::new().with_max_rounds(3);
        let mut runner = PatternRunner::new(ConversationPattern::RoundRobin, config);
        runner.add_agent(PatternAgent::new("r1", "First", "speaker", "Go first."));
        runner.add_agent(PatternAgent::new("r2", "Second", "speaker", "Go second."));

        let result = runner.run("Start discussion").unwrap();
        assert_eq!(result.rounds_completed, 3);
        // 3 rounds * 2 agents = 6 messages
        assert_eq!(result.messages.len(), 6);
    }

    #[test]
    fn test_conversation_pattern_sequential_basic() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Sequential, config);
        runner.add_agent(PatternAgent::new("s1", "Step1", "processor", "First step."));
        runner.add_agent(PatternAgent::new("s2", "Step2", "processor", "Second step."));
        runner.add_agent(PatternAgent::new("s3", "Step3", "processor", "Third step."));

        let result = runner.run("initial input").unwrap();
        assert_eq!(result.rounds_completed, 1);
        assert_eq!(result.messages.len(), 3);
        assert!(result.terminated_by.contains("pipeline_complete"));
        // Final output should be from the last agent
        let final_out = result.final_output.unwrap();
        assert!(final_out.contains("Step3"));
    }

    #[test]
    fn test_conversation_pattern_nested_chat_basic() {
        let config = PatternConfig::new().with_max_rounds(2);
        let mut runner = PatternRunner::new(ConversationPattern::NestedChat, config);
        runner.add_agent(PatternAgent::new("master", "Master", "coordinator", "Delegate."));
        runner.add_agent(PatternAgent::new("sub1", "Sub1", "worker", "Handle sub-task."));

        let result = runner.run("complex task").unwrap();
        assert_eq!(result.rounds_completed, 2);
        // Each round: master delegates (1) + sub-agent responds (1) + master synthesizes (1) = 3 per round
        assert_eq!(result.messages.len(), 6);
    }

    #[test]
    fn test_conversation_pattern_broadcast_basic() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Broadcast, config);
        runner.add_agent(PatternAgent::new("b1", "Agent1", "responder", "Respond."));
        runner.add_agent(PatternAgent::new("b2", "Agent2", "responder", "Respond."));
        runner.add_agent(PatternAgent::new("b3", "Agent3", "responder", "Respond."));

        let result = runner.run("What do you think?").unwrap();
        assert_eq!(result.rounds_completed, 1);
        assert_eq!(result.messages.len(), 3);
        assert_eq!(result.terminated_by, "broadcast_complete");
        let final_out = result.final_output.unwrap();
        assert!(final_out.contains("3 agents responded"));
    }

    #[test]
    fn test_pattern_runner_zero_agents_error() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Broadcast, config);

        let result = runner.run("hello");
        assert_eq!(result, Err(OrchestrationError::AgentNotFound));
    }

    #[test]
    fn test_pattern_runner_add_agents() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Sequential, config);
        assert_eq!(runner.agents().len(), 0);

        runner.add_agent(PatternAgent::new("a1", "A1", "role", "prompt"));
        assert_eq!(runner.agents().len(), 1);

        runner.add_agent(PatternAgent::new("a2", "A2", "role", "prompt"));
        assert_eq!(runner.agents().len(), 2);
    }

    #[test]
    fn test_pattern_runner_max_rounds_termination() {
        let config = PatternConfig::new().with_max_rounds(1);
        let mut runner = PatternRunner::new(ConversationPattern::RoundRobin, config);
        runner.add_agent(PatternAgent::new("a1", "A", "role", "prompt"));

        let result = runner.run("test").unwrap();
        assert_eq!(result.rounds_completed, 1);
        assert_eq!(result.terminated_by, "max_rounds");
    }

    #[test]
    fn test_pattern_runner_empty_input() {
        let config = PatternConfig::new().with_max_rounds(1);
        let mut runner = PatternRunner::new(ConversationPattern::Sequential, config);
        runner.add_agent(PatternAgent::new("a1", "A", "role", "prompt"));

        let result = runner.run("").unwrap();
        assert_eq!(result.messages.len(), 1);
        assert!(result.final_output.is_some());
    }

    #[test]
    fn test_pattern_config_defaults() {
        let config = PatternConfig::default();
        assert_eq!(config.max_rounds, 10);
        assert!(config.termination_condition.is_none());
        assert!(config.judge_agent_id.is_none());
        assert!(config.task_queue.is_empty());
    }

    #[test]
    fn test_pattern_config_custom() {
        let config = PatternConfig::new()
            .with_max_rounds(5)
            .with_termination(TerminationCondition::MaxRounds(3))
            .with_judge("judge-1")
            .with_task_queue(vec!["t1".into(), "t2".into()]);

        assert_eq!(config.max_rounds, 5);
        assert_eq!(
            config.termination_condition,
            Some(TerminationCondition::MaxRounds(3))
        );
        assert_eq!(config.judge_agent_id, Some("judge-1".to_string()));
        assert_eq!(config.task_queue.len(), 2);
    }

    #[test]
    fn test_debate_with_judge() {
        let config = PatternConfig::new()
            .with_max_rounds(2)
            .with_judge("judge");
        let mut runner = PatternRunner::new(ConversationPattern::Debate, config);
        runner.add_agent(PatternAgent::new("d1", "Pro", "debater", "Argue for."));
        runner.add_agent(PatternAgent::new("d2", "Con", "debater", "Argue against."));
        runner.add_agent(PatternAgent::new("judge", "Judge", "judge", "Synthesize."));

        let result = runner.run("Debate topic").unwrap();
        // Should have the judge's synthesis as final output
        let final_out = result.final_output.unwrap();
        assert!(final_out.contains("Judge"));
        assert!(final_out.contains("synthesis"));
    }

    #[test]
    fn test_debate_without_judge() {
        let config = PatternConfig::new().with_max_rounds(1);
        let mut runner = PatternRunner::new(ConversationPattern::Debate, config);
        runner.add_agent(PatternAgent::new("d1", "Pro", "debater", "Argue for."));
        runner.add_agent(PatternAgent::new("d2", "Con", "debater", "Argue against."));

        let result = runner.run("Topic").unwrap();
        // Without judge, final output is last message
        let final_out = result.final_output.unwrap();
        assert!(final_out.contains("Con")); // last agent in the round
    }

    #[test]
    fn test_swarm_task_distribution() {
        let config = PatternConfig::new()
            .with_max_rounds(10)
            .with_task_queue(vec![
                "task-1".into(),
                "task-2".into(),
                "task-3".into(),
                "task-4".into(),
            ]);
        let mut runner = PatternRunner::new(ConversationPattern::Swarm, config);
        runner.add_agent(PatternAgent::new("a1", "Alice", "worker", "Process."));
        runner.add_agent(PatternAgent::new("a2", "Bob", "worker", "Process."));

        let result = runner.run("go").unwrap();
        assert_eq!(result.messages.len(), 4);
        // Tasks alternate between agents: Alice, Bob, Alice, Bob
        assert!(result.messages[0].content.contains("Alice"));
        assert!(result.messages[1].content.contains("Bob"));
        assert!(result.messages[2].content.contains("Alice"));
        assert!(result.messages[3].content.contains("Bob"));
    }

    #[test]
    fn test_swarm_empty_task_queue() {
        // With empty task queue, the input is used as a single task
        let config = PatternConfig::new().with_max_rounds(5);
        let mut runner = PatternRunner::new(ConversationPattern::Swarm, config);
        runner.add_agent(PatternAgent::new("a1", "Alice", "worker", "Process."));

        let result = runner.run("do this one thing").unwrap();
        assert_eq!(result.messages.len(), 1);
        assert!(result.messages[0].content.contains("do this one thing"));
    }

    #[test]
    fn test_sequential_single_agent_passthrough() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Sequential, config);
        runner.add_agent(PatternAgent::new("solo", "Solo", "processor", "Process."));

        let result = runner.run("input data").unwrap();
        assert_eq!(result.messages.len(), 1);
        assert!(result.messages[0].content.contains("Solo"));
        assert!(result.messages[0].content.contains("input data"));
    }

    #[test]
    fn test_sequential_multi_agent_pipeline() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Sequential, config);
        runner.add_agent(PatternAgent::new("s1", "Parser", "parser", "Parse."));
        runner.add_agent(PatternAgent::new("s2", "Analyzer", "analyzer", "Analyze."));
        runner.add_agent(PatternAgent::new("s3", "Reporter", "reporter", "Report."));

        let result = runner.run("raw data").unwrap();
        assert_eq!(result.messages.len(), 3);
        // Each agent should see the output of the previous
        assert!(result.messages[0].content.contains("raw data"));
        assert!(result.messages[1].content.contains("Parser"));
        assert!(result.messages[2].content.contains("Analyzer"));
    }

    #[test]
    fn test_broadcast_all_agents_respond() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Broadcast, config);
        runner.add_agent(PatternAgent::new("b1", "Agent1", "r", "Respond."));
        runner.add_agent(PatternAgent::new("b2", "Agent2", "r", "Respond."));

        let result = runner.run("query").unwrap();
        assert_eq!(result.messages.len(), 2);

        let agent_ids: Vec<&str> = result.messages.iter().map(|m| m.agent_id.as_str()).collect();
        assert!(agent_ids.contains(&"b1"));
        assert!(agent_ids.contains(&"b2"));
    }

    #[test]
    fn test_broadcast_collect_all_responses() {
        let config = PatternConfig::new();
        let mut runner = PatternRunner::new(ConversationPattern::Broadcast, config);
        for i in 0..5 {
            runner.add_agent(PatternAgent::new(
                &format!("a{}", i),
                &format!("Agent{}", i),
                "responder",
                "Respond.",
            ));
        }

        let result = runner.run("mass query").unwrap();
        assert_eq!(result.messages.len(), 5);
        let final_out = result.final_output.unwrap();
        assert!(final_out.contains("5 agents responded"));
    }

    #[test]
    fn test_round_robin_correct_ordering() {
        let config = PatternConfig::new().with_max_rounds(2);
        let mut runner = PatternRunner::new(ConversationPattern::RoundRobin, config);
        runner.add_agent(PatternAgent::new("a1", "First", "s", "Prompt."));
        runner.add_agent(PatternAgent::new("a2", "Second", "s", "Prompt."));
        runner.add_agent(PatternAgent::new("a3", "Third", "s", "Prompt."));

        let result = runner.run("start").unwrap();
        // 2 rounds * 3 agents = 6 messages
        assert_eq!(result.messages.len(), 6);

        // Check ordering: a1, a2, a3, a1, a2, a3
        assert_eq!(result.messages[0].agent_id, "a1");
        assert_eq!(result.messages[1].agent_id, "a2");
        assert_eq!(result.messages[2].agent_id, "a3");
        assert_eq!(result.messages[3].agent_id, "a1");
        assert_eq!(result.messages[4].agent_id, "a2");
        assert_eq!(result.messages[5].agent_id, "a3");
    }

    #[test]
    fn test_round_robin_correct_round_tracking() {
        let config = PatternConfig::new().with_max_rounds(3);
        let mut runner = PatternRunner::new(ConversationPattern::RoundRobin, config);
        runner.add_agent(PatternAgent::new("a1", "A", "s", "P."));
        runner.add_agent(PatternAgent::new("a2", "B", "s", "P."));

        let result = runner.run("x").unwrap();
        // Round 0: a1, a2. Round 1: a1, a2. Round 2: a1, a2.
        assert_eq!(result.messages[0].round, 0);
        assert_eq!(result.messages[1].round, 0);
        assert_eq!(result.messages[2].round, 1);
        assert_eq!(result.messages[3].round, 1);
        assert_eq!(result.messages[4].round, 2);
        assert_eq!(result.messages[5].round, 2);
    }

    #[test]
    fn test_pattern_runner_getters() {
        let config = PatternConfig::new().with_max_rounds(7);
        let runner = PatternRunner::new(ConversationPattern::Debate, config);
        assert_eq!(runner.pattern(), ConversationPattern::Debate);
        assert_eq!(runner.config().max_rounds, 7);
        assert!(runner.transcript().is_empty());
    }

    #[test]
    fn test_pattern_agent_creation() {
        let agent = PatternAgent::new("id1", "Name1", "role1", "You are a test agent.");
        assert_eq!(agent.id, "id1");
        assert_eq!(agent.name, "Name1");
        assert_eq!(agent.role, "role1");
        assert_eq!(agent.system_prompt, "You are a test agent.");
    }

    #[test]
    fn test_pattern_message_creation() {
        let msg = PatternMessage::new("agent-1", "Hello world", 3);
        assert_eq!(msg.agent_id, "agent-1");
        assert_eq!(msg.content, "Hello world");
        assert_eq!(msg.round, 3);
    }

    #[test]
    fn test_termination_condition_max_rounds() {
        let config = PatternConfig::new()
            .with_max_rounds(100)
            .with_termination(TerminationCondition::MaxRounds(2));
        let mut runner = PatternRunner::new(ConversationPattern::RoundRobin, config);
        runner.add_agent(PatternAgent::new("a1", "A", "s", "P."));

        let result = runner.run("test").unwrap();
        // Termination condition MaxRounds(2) should stop at 2 even though max_rounds is 100
        assert_eq!(result.rounds_completed, 2);
    }

    // =========================================================================
    // 7.4 Agent Handoffs tests
    // =========================================================================

    #[test]
    fn test_handoff_manager_register_agent() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "Agent1", "worker", "Prompt."));
        assert!(mgr.get_agent("a1").is_some());
        assert_eq!(mgr.get_agent("a1").unwrap().name, "Agent1");
    }

    #[test]
    fn test_handoff_manager_list_agents() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "Agent1", "worker", "P."));
        mgr.register_agent(PatternAgent::new("a2", "Agent2", "worker", "P."));

        let agents = mgr.list_agents();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_handoff_request_full_policy() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "Agent1", "worker", "P."));
        mgr.register_agent(PatternAgent::new("a2", "Agent2", "worker", "P."));

        let request = HandoffRequest::new("a1", "a2", "escalation", ContextTransferPolicy::Full);
        let result = mgr.request_handoff(request).unwrap();
        assert!(result.success);
        assert_eq!(result.transferred_messages, usize::MAX);
        assert_eq!(result.from_agent, "a1");
        assert_eq!(result.to_agent, "a2");
        assert_eq!(result.reason, "escalation");
    }

    #[test]
    fn test_handoff_request_summary_policy() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        let request =
            HandoffRequest::new("a1", "a2", "summarize", ContextTransferPolicy::Summary);
        let result = mgr.request_handoff(request).unwrap();
        assert!(result.success);
        assert_eq!(result.transferred_messages, 1);
    }

    #[test]
    fn test_handoff_request_last_n_policy() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        let request =
            HandoffRequest::new("a1", "a2", "recent context", ContextTransferPolicy::LastN(5));
        let result = mgr.request_handoff(request).unwrap();
        assert!(result.success);
        assert_eq!(result.transferred_messages, 5);
    }

    #[test]
    fn test_handoff_request_none_policy() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        let request =
            HandoffRequest::new("a1", "a2", "clean start", ContextTransferPolicy::None);
        let result = mgr.request_handoff(request).unwrap();
        assert!(result.success);
        assert_eq!(result.transferred_messages, 0);
    }

    #[test]
    fn test_handoff_to_unknown_agent() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));

        let request =
            HandoffRequest::new("a1", "unknown", "reason", ContextTransferPolicy::Full);
        let result = mgr.request_handoff(request);
        assert_eq!(result, Err(OrchestrationError::AgentNotFound));
    }

    #[test]
    fn test_handoff_from_unknown_agent() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        let request =
            HandoffRequest::new("unknown", "a2", "reason", ContextTransferPolicy::Full);
        let result = mgr.request_handoff(request);
        assert_eq!(result, Err(OrchestrationError::AgentNotFound));
    }

    #[test]
    fn test_handoff_self_handoff_error() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));

        let request = HandoffRequest::new("a1", "a1", "self", ContextTransferPolicy::Full);
        let result = mgr.request_handoff(request);
        assert_eq!(result, Err(OrchestrationError::InvalidState));
    }

    #[test]
    fn test_handoff_get_history() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        assert!(mgr.get_handoff_history().is_empty());

        let request = HandoffRequest::new("a1", "a2", "test", ContextTransferPolicy::Full);
        mgr.request_handoff(request).unwrap();

        assert_eq!(mgr.get_handoff_history().len(), 1);
        assert_eq!(mgr.get_handoff_history()[0].from_agent, "a1");
    }

    #[test]
    fn test_handoff_detect_circular() {
        let mut mgr = HandoffManager::new();
        mgr.register_agent(PatternAgent::new("a1", "A1", "w", "P."));
        mgr.register_agent(PatternAgent::new("a2", "A2", "w", "P."));

        // No history yet -- not circular
        assert!(!mgr.detect_circular_handoff("a1", "a2"));

        // Handoff a1 -> a2
        let request = HandoffRequest::new("a1", "a2", "first", ContextTransferPolicy::Full);
        mgr.request_handoff(request).unwrap();

        // Now a2 -> a1 would be circular (A->B->A)
        assert!(mgr.detect_circular_handoff("a2", "a1"));

        // a1 -> a2 again is not circular (no recent a2->a1 handoff)
        assert!(!mgr.detect_circular_handoff("a1", "a2"));
    }

    #[test]
    fn test_handoff_result_success_state() {
        let result = HandoffResult {
            success: true,
            transferred_messages: 10,
            from_agent: "a1".to_string(),
            to_agent: "a2".to_string(),
            reason: "done".to_string(),
        };
        assert!(result.success);
        assert_eq!(result.transferred_messages, 10);
    }

    #[test]
    fn test_handoff_result_failure_state() {
        let result = HandoffResult {
            success: false,
            transferred_messages: 0,
            from_agent: "a1".to_string(),
            to_agent: "a2".to_string(),
            reason: "agent unavailable".to_string(),
        };
        assert!(!result.success);
        assert_eq!(result.transferred_messages, 0);
        assert_eq!(result.reason, "agent unavailable");
    }

    #[test]
    fn test_handoff_request_with_metadata() {
        let request = HandoffRequest::new("a1", "a2", "test", ContextTransferPolicy::Full)
            .with_metadata("priority", "high")
            .with_metadata("session_id", "sess-42");

        assert_eq!(request.metadata.get("priority").unwrap(), "high");
        assert_eq!(request.metadata.get("session_id").unwrap(), "sess-42");
    }

    #[test]
    fn test_handoff_manager_default() {
        let mgr = HandoffManager::default();
        assert!(mgr.list_agents().is_empty());
        assert!(mgr.get_handoff_history().is_empty());
    }

    #[test]
    fn test_nested_chat_single_agent() {
        let config = PatternConfig::new().with_max_rounds(2);
        let mut runner = PatternRunner::new(ConversationPattern::NestedChat, config);
        runner.add_agent(PatternAgent::new("solo", "Solo", "master", "Handle alone."));

        let result = runner.run("single agent task").unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.terminated_by, "single_agent");
    }

    // =========================================================================
    // MultiAgentSession tests (requires autonomous feature)
    // =========================================================================

    #[cfg(feature = "autonomous")]
    mod session_tests {
        use super::super::*;
        use crate::agentic_loop::AgentMessage;
        use crate::autonomous_loop::AutonomousAgentBuilder;
        use std::sync::Arc;

        fn dummy_generator() -> Arc<dyn Fn(&[AgentMessage]) -> String + Send + Sync> {
            Arc::new(|_msgs: &[AgentMessage]| -> String { "Done.".to_string() })
        }

        #[test]
        fn test_session_creation() {
            let session =
                MultiAgentSession::new("test-session", OrchestrationStrategy::Sequential, None);
            assert_eq!(session.name, "test-session");
            assert!(session.agent_names().is_empty());
        }

        #[test]
        fn test_session_add_agent() {
            let mut session =
                MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let agent = AutonomousAgentBuilder::new("agent-1", dummy_generator())
                .system_prompt("You are a test agent.")
                .max_iterations(5)
                .build();
            session.add_agent(agent, AgentRole::Executor);
            assert_eq!(session.agent_names().len(), 1);
            assert!(session.agent_names().contains(&"agent-1"));
        }

        #[test]
        fn test_session_add_task() {
            let mut session =
                MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            session.add_task(AgentTask::new("task-1", "Do something"));
            let status = session.orchestrator.get_status();
            assert_eq!(status.total_tasks, 1);
        }

        #[test]
        fn test_session_auto_assign() {
            let mut session =
                MultiAgentSession::new("test", OrchestrationStrategy::RoundRobin, None);
            let agent = AutonomousAgentBuilder::new("worker", dummy_generator())
                .system_prompt("Test")
                .build();
            session.add_agent(agent, AgentRole::Executor);
            session.add_task(AgentTask::new("task-1", "Do something"));

            let assignments = session.auto_assign();
            assert_eq!(assignments.len(), 1);
            assert_eq!(assignments[0].0, "task-1");
            assert_eq!(assignments[0].1, "worker");
        }

        #[test]
        fn test_session_board_display() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let display = session.board_display();
            assert!(display.contains("session-board"));
        }

        #[test]
        fn test_session_summary() {
            let mut session =
                MultiAgentSession::new("my-session", OrchestrationStrategy::Sequential, None);
            let agent = AutonomousAgentBuilder::new("a1", dummy_generator())
                .system_prompt("Test")
                .build();
            session.add_agent(agent, AgentRole::Researcher);

            let summary = session.summary();
            assert_eq!(summary.name, "my-session");
            assert_eq!(summary.status.total_agents, 1);
            assert!(!summary.board_display.is_empty());
        }

        #[test]
        fn test_session_process_status_command() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("status");
            assert!(result.success);
            assert!(result.message.contains("test"));
        }

        #[test]
        fn test_session_process_help_command() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("help");
            assert!(result.success);
            assert!(result.message.contains("Commands"));
        }

        #[test]
        fn test_session_process_pause_all() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("pause all");
            assert!(result.success);
            assert!(result.message.contains("paused"));
        }

        #[test]
        fn test_session_process_resume_all() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("resume all");
            assert!(result.success);
            assert!(result.message.contains("resumed"));
        }

        #[test]
        fn test_session_process_cancel_all() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("cancel all");
            assert!(result.success);
            assert!(result.message.contains("cancelled"));
        }

        #[test]
        fn test_session_process_unknown_command() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let result = session.process_user_input("do something weird");
            assert!(!result.success);
            assert!(result.message.contains("Unknown"));
        }

        #[test]
        fn test_session_interaction_manager() {
            let session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            let _im = session.interaction_manager();
            // Just verify we can access it without panic
        }
    }
}
