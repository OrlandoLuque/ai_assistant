//! Multi-agent orchestration
//!
//! Coordinate multiple AI agents working together.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
}
