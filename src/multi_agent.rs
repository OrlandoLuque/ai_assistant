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
            let agent = self.agents.get(agent_id)
                .ok_or(OrchestrationError::AgentNotFound)?;
            if agent.status != AgentStatus::Idle {
                return Err(OrchestrationError::AgentBusy);
            }
        }

        // Check if task exists and get dependencies
        let dependencies = {
            let task = self.tasks.get(task_id)
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
        let task = self.tasks.get_mut(task_id)
            .ok_or(OrchestrationError::TaskNotFound)?;

        let agent_id = task.assigned_to.clone()
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
        let task = self.tasks.get_mut(task_id)
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
        self.messages.iter()
            .filter(|m| m.to == agent_id)
            .collect()
    }

    pub fn auto_assign_tasks(&mut self) -> Vec<(String, String)> {
        let mut assignments = Vec::new();

        // Get pending tasks sorted by priority
        let mut pending_tasks: Vec<_> = self.tasks.iter()
            .filter(|(_, t)| t.status == TaskStatus::Pending && t.assigned_to.is_none())
            .map(|(id, t)| (id.clone(), t.priority))
            .collect();

        pending_tasks.sort_by(|a, b| b.1.cmp(&a.1));

        // Get idle agents
        let idle_agents: Vec<_> = self.agents.iter()
            .filter(|(_, a)| a.status == AgentStatus::Idle)
            .map(|(id, _)| id.clone())
            .collect();

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
                        let best_agent = idle_agents.iter()
                            .filter(|id| {
                                self.agents.get(*id)
                                    .map(|a| a.status == AgentStatus::Idle)
                                    .unwrap_or(false)
                            })
                            .max_by_key(|id| {
                                self.agents.get(*id)
                                    .map(|a| {
                                        let desc_lower = task.description.to_lowercase();
                                        a.capabilities.iter()
                                            .filter(|c| desc_lower.contains(&c.to_lowercase()))
                                            .count()
                                    })
                                    .unwrap_or(0)
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
        let completed = self.tasks.values().filter(|t| t.status == TaskStatus::Completed).count();
        let in_progress = self.tasks.values().filter(|t| t.status == TaskStatus::InProgress).count();
        let failed = self.tasks.values().filter(|t| t.status == TaskStatus::Failed).count();

        let idle_agents = self.agents.values().filter(|a| a.status == AgentStatus::Idle).count();
        let working_agents = self.agents.values().filter(|a| a.status == AgentStatus::Working).count();

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
use crate::task_board::{TaskBoard, BoardCommand};
#[cfg(feature = "autonomous")]
use crate::task_planning::StepPriority;
#[cfg(feature = "autonomous")]
use crate::user_interaction::{InteractionManager, AutoApproveHandler, UserInteractionHandler};
#[cfg(feature = "autonomous")]
use crate::interactive_commands::{CommandProcessor, CommandResult, UserIntent};

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
    pub fn run_agent(&mut self, agent_name: &str, task_description: &str) -> Result<String, String> {
        // Update board progress
        if let Ok(mut board) = self.task_board.write() {
            let _ = board.execute_command(BoardCommand::AddTask {
                title: format!("[{}] {}", agent_name, task_description),
                description: String::new(),
                priority: StepPriority::Medium,
            });
        }

        let agent = self.agents.get_mut(agent_name)
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
        self.task_board.read().unwrap_or_else(|e| e.into_inner()).to_display()
    }

    /// Get session summary.
    pub fn summary(&self) -> SessionSummary {
        let status = self.orchestrator.get_status();
        let agent_states: Vec<(String, String)> = self.agents.iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registration() {
        let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Sequential);

        orchestrator.register_agent(
            Agent::new("agent1", "Researcher", AgentRole::Researcher)
                .with_capability("search")
                .with_capability("analyze")
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

        orchestrator.fail_task("task1", "Something went wrong").unwrap();

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
                .with_capability("analyze")
        );
        orchestrator.register_agent(
            Agent::new("writer", "Writer", AgentRole::Writer)
                .with_capability("write")
                .with_capability("edit")
        );

        // Task with "search" keyword should go to researcher
        orchestrator.add_task(AgentTask::new("task1", "Search for information and analyze data"));
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
        let agent = Agent::new("agent1", "Worker", AgentRole::Executor)
            .with_model("gpt-4");

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
        assert_eq!(format!("{}", OrchestrationError::AgentNotFound), "Agent not found");
        assert_eq!(format!("{}", OrchestrationError::TaskNotFound), "Task not found");
        assert_eq!(format!("{}", OrchestrationError::AgentBusy), "Agent is busy");
        assert_eq!(format!("{}", OrchestrationError::DependencyNotMet), "Task dependency not met");
        assert_eq!(format!("{}", OrchestrationError::TaskNotAssigned), "Task not assigned to any agent");
        assert_eq!(format!("{}", OrchestrationError::InvalidState), "Invalid state");
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
                .depends_on("task2")
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

    // =========================================================================
    // MultiAgentSession tests (requires autonomous feature)
    // =========================================================================

    #[cfg(feature = "autonomous")]
    mod session_tests {
        use super::super::*;
        use std::sync::Arc;
        use crate::autonomous_loop::AutonomousAgentBuilder;
        use crate::agentic_loop::AgentMessage;

        fn dummy_generator() -> Arc<dyn Fn(&[AgentMessage]) -> String + Send + Sync> {
            Arc::new(|_msgs: &[AgentMessage]| -> String {
                "Done.".to_string()
            })
        }

        #[test]
        fn test_session_creation() {
            let session = MultiAgentSession::new("test-session", OrchestrationStrategy::Sequential, None);
            assert_eq!(session.name, "test-session");
            assert!(session.agent_names().is_empty());
        }

        #[test]
        fn test_session_add_agent() {
            let mut session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
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
            let mut session = MultiAgentSession::new("test", OrchestrationStrategy::Sequential, None);
            session.add_task(AgentTask::new("task-1", "Do something"));
            let status = session.orchestrator.get_status();
            assert_eq!(status.total_tasks, 1);
        }

        #[test]
        fn test_session_auto_assign() {
            let mut session = MultiAgentSession::new("test", OrchestrationStrategy::RoundRobin, None);
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
            let mut session = MultiAgentSession::new("my-session", OrchestrationStrategy::Sequential, None);
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
