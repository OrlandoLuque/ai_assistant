//! Task board — interactive live task board with real-time updates
//!
//! Wraps the existing [`TaskPlan`] with reactive capabilities: listeners,
//! undo/redo, agent assignments, execution state tracking, and progress reporting.

use crate::task_planning::{PlanStep, StepPriority, StepStatus, TaskPlan};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// TaskBoardListener
// ============================================================================

/// Trait for observing task board changes in real-time.
pub trait TaskBoardListener: Send + Sync {
    fn on_task_changed(&self, step_id: &str, old_status: StepStatus, new_status: StepStatus);
    fn on_task_added(&self, step_id: &str, title: &str);
    fn on_task_removed(&self, step_id: &str);
    fn on_progress_update(&self, step_id: &str, progress: f64, action: &str);
    fn on_board_reset(&self);
}

/// A listener that collects events for testing/inspection.
pub struct CollectingBoardListener {
    events: std::sync::Mutex<Vec<BoardEvent>>,
}

impl CollectingBoardListener {
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<BoardEvent> {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

impl Default for CollectingBoardListener {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskBoardListener for CollectingBoardListener {
    fn on_task_changed(&self, step_id: &str, old_status: StepStatus, new_status: StepStatus) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(BoardEvent::TaskChanged {
            step_id: step_id.to_string(),
            old_status,
            new_status,
        });
    }

    fn on_task_added(&self, step_id: &str, title: &str) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(BoardEvent::TaskAdded {
            step_id: step_id.to_string(),
            title: title.to_string(),
        });
    }

    fn on_task_removed(&self, step_id: &str) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(BoardEvent::TaskRemoved {
            step_id: step_id.to_string(),
        });
    }

    fn on_progress_update(&self, step_id: &str, progress: f64, action: &str) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(BoardEvent::ProgressUpdate {
            step_id: step_id.to_string(),
            progress,
            action: action.to_string(),
        });
    }

    fn on_board_reset(&self) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(BoardEvent::BoardReset);
    }
}

/// Events emitted by the task board.
#[derive(Debug, Clone)]
pub enum BoardEvent {
    TaskChanged {
        step_id: String,
        old_status: StepStatus,
        new_status: StepStatus,
    },
    TaskAdded {
        step_id: String,
        title: String,
    },
    TaskRemoved {
        step_id: String,
    },
    ProgressUpdate {
        step_id: String,
        progress: f64,
        action: String,
    },
    BoardReset,
}

// ============================================================================
// TaskExecutionState
// ============================================================================

/// Runtime state of a task being executed by an agent.
#[derive(Debug, Clone)]
pub struct TaskExecutionState {
    pub agent_name: Option<String>,
    pub started_at: Option<u64>,
    pub progress: f64,
    pub current_action: Option<String>,
    pub iterations: usize,
    pub cost_so_far: f64,
}

impl Default for TaskExecutionState {
    fn default() -> Self {
        Self {
            agent_name: None,
            started_at: None,
            progress: 0.0,
            current_action: None,
            iterations: 0,
            cost_so_far: 0.0,
        }
    }
}

// ============================================================================
// BoardCommand
// ============================================================================

/// Commands that can be issued to the task board.
#[derive(Debug, Clone)]
pub enum BoardCommand {
    AddTask {
        title: String,
        description: String,
        priority: StepPriority,
    },
    RemoveTask {
        id: String,
    },
    StartTask {
        id: String,
    },
    CompleteTask {
        id: String,
    },
    PauseTask {
        id: String,
    },
    ResumeTask {
        id: String,
    },
    CancelTask {
        id: String,
    },
    ReprioritizeTask {
        id: String,
        priority: StepPriority,
    },
    AssignToAgent {
        task_id: String,
        agent_name: String,
    },
    UnassignTask {
        task_id: String,
    },
    AddDependency {
        task_id: String,
        depends_on: String,
    },
    AddNote {
        task_id: String,
        note: String,
    },
    UpdateProgress {
        task_id: String,
        progress: f64,
        action: String,
    },
    PauseAll,
    ResumeAll,
    CancelAll,
}

// ============================================================================
// TaskBoardSummary
// ============================================================================

/// Summary statistics for the task board.
#[derive(Debug, Clone)]
pub struct TaskBoardSummary {
    pub name: String,
    pub total_tasks: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub blocked: usize,
    pub cancelled: usize,
    pub overall_progress: f64,
    pub agents_active: usize,
    pub total_cost: f64,
}

// ============================================================================
// TaskBoard
// ============================================================================

/// Interactive task board with real-time updates and undo support.
pub struct TaskBoard {
    pub name: String,
    plan: TaskPlan,
    listeners: Vec<Arc<dyn TaskBoardListener>>,
    command_history: Vec<(BoardCommand, u64)>,
    agent_assignments: HashMap<String, String>,
    execution_state: HashMap<String, TaskExecutionState>,
}

impl TaskBoard {
    /// Create a new empty task board.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let plan = TaskPlan::new(&name);
        Self {
            name,
            plan,
            listeners: Vec::new(),
            command_history: Vec::new(),
            agent_assignments: HashMap::new(),
            execution_state: HashMap::new(),
        }
    }

    /// Create a task board from an existing plan.
    pub fn from_plan(plan: TaskPlan) -> Self {
        let name = plan.name.clone();
        Self {
            name,
            plan,
            listeners: Vec::new(),
            command_history: Vec::new(),
            agent_assignments: HashMap::new(),
            execution_state: HashMap::new(),
        }
    }

    /// Add a listener for board changes.
    pub fn add_listener(&mut self, listener: Arc<dyn TaskBoardListener>) {
        self.listeners.push(listener);
    }

    /// Get a reference to the underlying plan.
    pub fn plan(&self) -> &TaskPlan {
        &self.plan
    }

    /// Get a mutable reference to the underlying plan.
    pub fn plan_mut(&mut self) -> &mut TaskPlan {
        &mut self.plan
    }

    /// Execute a command on the board.
    pub fn execute_command(&mut self, cmd: BoardCommand) -> Result<(), String> {
        self.command_history.push((cmd.clone(), now_millis()));

        match cmd {
            BoardCommand::AddTask {
                title,
                description,
                priority,
            } => {
                let mut step = PlanStep::new(&title).with_priority(priority);
                step.description = Some(description);
                let id = step.id.clone();
                self.plan.steps.push(step);
                self.notify_added(&id, &title);
                Ok(())
            }

            BoardCommand::RemoveTask { id } => {
                let existed = remove_step_recursive(&mut self.plan.steps, &id);
                if existed {
                    self.agent_assignments.remove(&id);
                    self.execution_state.remove(&id);
                    self.notify_removed(&id);
                    Ok(())
                } else {
                    Err(format!("Task not found: {}", id))
                }
            }

            BoardCommand::StartTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::InProgress;
                    self.execution_state
                        .entry(id.clone())
                        .or_default()
                        .started_at = Some(now_millis());
                    self.notify_changed(&id, old, StepStatus::InProgress);
                    Ok(())
                } else {
                    Err(format!("Task not found: {}", id))
                }
            }

            BoardCommand::CompleteTask { id } => {
                self.plan.complete_step(&id);
                let new_status = self.plan.find_step(&id).map(|s| s.status.clone()).unwrap_or(StepStatus::Done);
                if let Some(state) = self.execution_state.get_mut(&id) {
                    state.progress = 1.0;
                }
                self.notify_changed(&id, StepStatus::InProgress, new_status);
                Ok(())
            }

            BoardCommand::PauseTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::Blocked;
                    self.notify_changed(&id, old, StepStatus::Blocked);
                    Ok(())
                } else {
                    Err(format!("Task not found: {}", id))
                }
            }

            BoardCommand::ResumeTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::InProgress;
                    self.notify_changed(&id, old, StepStatus::InProgress);
                    Ok(())
                } else {
                    Err(format!("Task not found: {}", id))
                }
            }

            BoardCommand::CancelTask { id } => {
                self.plan.skip_step(&id);
                self.notify_changed(&id, StepStatus::Pending, StepStatus::Skipped);
                Ok(())
            }

            BoardCommand::ReprioritizeTask { id, priority } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    step.priority = priority;
                    Ok(())
                } else {
                    Err(format!("Task not found: {}", id))
                }
            }

            BoardCommand::AssignToAgent { task_id, agent_name } => {
                self.agent_assignments
                    .insert(task_id.clone(), agent_name.clone());
                self.execution_state
                    .entry(task_id)
                    .or_default()
                    .agent_name = Some(agent_name);
                Ok(())
            }

            BoardCommand::UnassignTask { task_id } => {
                self.agent_assignments.remove(&task_id);
                if let Some(state) = self.execution_state.get_mut(&task_id) {
                    state.agent_name = None;
                }
                Ok(())
            }

            BoardCommand::AddDependency {
                task_id,
                depends_on,
            } => {
                self.plan.block_step(&task_id, &depends_on);
                Ok(())
            }

            BoardCommand::AddNote { task_id, note } => {
                let sn = crate::task_planning::StepNote::new(&note);
                self.plan.add_note(&task_id, sn);
                Ok(())
            }

            BoardCommand::UpdateProgress {
                task_id,
                progress,
                action,
            } => {
                let state = self.execution_state.entry(task_id.clone()).or_default();
                state.progress = progress.clamp(0.0, 1.0);
                state.current_action = Some(action.clone());
                state.iterations += 1;
                self.notify_progress(&task_id, progress, &action);
                Ok(())
            }

            BoardCommand::PauseAll => {
                let ids: Vec<String> = self
                    .plan
                    .steps
                    .iter()
                    .filter(|s| s.status == StepStatus::InProgress)
                    .map(|s| s.id.clone())
                    .collect();
                for id in ids {
                    if let Some(step) = self.plan.find_step_mut(&id) {
                        step.status = StepStatus::Blocked;
                        self.notify_changed(&id, StepStatus::InProgress, StepStatus::Blocked);
                    }
                }
                Ok(())
            }

            BoardCommand::ResumeAll => {
                let ids: Vec<String> = self
                    .plan
                    .steps
                    .iter()
                    .filter(|s| s.status == StepStatus::Blocked)
                    .map(|s| s.id.clone())
                    .collect();
                for id in ids {
                    if let Some(step) = self.plan.find_step_mut(&id) {
                        step.status = StepStatus::InProgress;
                        self.notify_changed(&id, StepStatus::Blocked, StepStatus::InProgress);
                    }
                }
                Ok(())
            }

            BoardCommand::CancelAll => {
                let ids: Vec<String> = self
                    .plan
                    .steps
                    .iter()
                    .filter(|s| s.status != StepStatus::Done && s.status != StepStatus::Skipped)
                    .map(|s| s.id.clone())
                    .collect();
                for id in ids {
                    self.plan.skip_step(&id);
                    self.notify_changed(&id, StepStatus::Pending, StepStatus::Skipped);
                }
                Ok(())
            }
        }
    }

    /// Get all tasks that can be started now (pending, no unresolved deps).
    pub fn next_actionable(&self) -> Vec<&PlanStep> {
        self.plan
            .steps
            .iter()
            .filter(|s| {
                s.status == StepStatus::Pending
                    && s.blocked_by.iter().all(|dep_id| {
                        self.plan
                            .find_step(dep_id)
                            .map(|d| d.status == StepStatus::Done || d.status == StepStatus::Skipped)
                            .unwrap_or(true)
                    })
            })
            .collect()
    }

    /// Get tasks assigned to a specific agent.
    pub fn agent_tasks(&self, agent_name: &str) -> Vec<&PlanStep> {
        self.agent_assignments
            .iter()
            .filter(|(_, name)| name.as_str() == agent_name)
            .filter_map(|(task_id, _)| self.plan.find_step(task_id))
            .collect()
    }

    /// Get the execution state for a task.
    pub fn execution_state(&self, task_id: &str) -> Option<&TaskExecutionState> {
        self.execution_state.get(task_id)
    }

    /// Get overall progress (0.0 - 1.0).
    pub fn overall_progress(&self) -> f64 {
        self.plan.progress() as f64
    }

    /// Get a summary of the board state.
    pub fn summary(&self) -> TaskBoardSummary {
        let plan_summary = self.plan.summary();
        let agents_active = self
            .execution_state
            .values()
            .filter(|s| s.agent_name.is_some() && s.progress < 1.0)
            .count();
        let total_cost: f64 = self.execution_state.values().map(|s| s.cost_so_far).sum();

        TaskBoardSummary {
            name: self.name.clone(),
            total_tasks: plan_summary.total_steps,
            pending: plan_summary.pending,
            in_progress: plan_summary.in_progress,
            completed: plan_summary.done,
            blocked: plan_summary.blocked,
            cancelled: plan_summary.skipped,
            overall_progress: self.plan.progress() as f64,
            agents_active,
            total_cost,
        }
    }

    /// Find a task by partial title match (case-insensitive).
    pub fn find_by_title(&self, query: &str) -> Option<&PlanStep> {
        let lower = query.to_lowercase();
        self.plan.steps.iter().find(|s| s.title.to_lowercase().contains(&lower))
    }

    /// Undo the last command that can be reversed.
    ///
    /// Supports undoing: AddTask (removes it), RemoveTask (cannot reverse — data lost),
    /// StartTask/PauseTask/ResumeTask/CancelTask (restores previous status),
    /// ReprioritizeTask (cannot reverse — old priority not stored),
    /// PauseAll/ResumeAll/CancelAll (cannot reverse individual statuses).
    ///
    /// Returns a description of what was undone, or an error if nothing to undo.
    pub fn undo_last(&mut self) -> Result<String, String> {
        let (cmd, _ts) = self.command_history.pop()
            .ok_or_else(|| "Nothing to undo.".to_string())?;

        match cmd {
            BoardCommand::AddTask { title, .. } => {
                // Find and remove the task that was added (last matching title)
                if let Some(pos) = self.plan.steps.iter().rposition(|s| s.title == title) {
                    let id = self.plan.steps[pos].id.clone();
                    self.plan.steps.remove(pos);
                    self.agent_assignments.remove(&id);
                    self.execution_state.remove(&id);
                    self.notify_removed(&id);
                    Ok(format!("Undone: removed added task '{}'", title))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", title))
                }
            }

            BoardCommand::StartTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::Pending;
                    self.notify_changed(&id, old, StepStatus::Pending);
                    Ok(format!("Undone: task '{}' reverted to Pending", id))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", id))
                }
            }

            BoardCommand::PauseTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::InProgress;
                    self.notify_changed(&id, old, StepStatus::InProgress);
                    Ok(format!("Undone: task '{}' resumed from pause", id))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", id))
                }
            }

            BoardCommand::ResumeTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::Blocked;
                    self.notify_changed(&id, old, StepStatus::Blocked);
                    Ok(format!("Undone: task '{}' re-paused", id))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", id))
                }
            }

            BoardCommand::CancelTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::Pending;
                    self.notify_changed(&id, old, StepStatus::Pending);
                    Ok(format!("Undone: task '{}' restored from cancelled", id))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", id))
                }
            }

            BoardCommand::CompleteTask { id } => {
                if let Some(step) = self.plan.find_step_mut(&id) {
                    let old = step.status.clone();
                    step.status = StepStatus::InProgress;
                    self.notify_changed(&id, old, StepStatus::InProgress);
                    Ok(format!("Undone: task '{}' reverted to InProgress", id))
                } else {
                    Err(format!("Cannot undo: task '{}' not found", id))
                }
            }

            other => {
                // Re-push the command since we can't undo it
                self.command_history.push((other.clone(), now_millis()));
                Err(format!("Cannot undo {:?}: operation is not reversible", other))
            }
        }
    }

    /// Get command history.
    pub fn command_history(&self) -> &[(BoardCommand, u64)] {
        &self.command_history
    }

    /// Render the board as a formatted display string.
    pub fn to_display(&self) -> String {
        let mut lines = vec![format!("=== {} ===", self.name)];
        let summary = self.summary();
        lines.push(format!(
            "Progress: {:.0}% | {} total | {} pending | {} active | {} done | {} blocked",
            summary.overall_progress * 100.0,
            summary.total_tasks,
            summary.pending,
            summary.in_progress,
            summary.completed,
            summary.blocked,
        ));
        lines.push(String::new());

        for step in &self.plan.steps {
            let status_icon = match step.status {
                StepStatus::Pending => "[ ]",
                StepStatus::InProgress => "[~]",
                StepStatus::Done => "[x]",
                StepStatus::Blocked => "[!]",
                StepStatus::Skipped => "[-]",
            };
            let agent = self
                .agent_assignments
                .get(&step.id)
                .map(|a| format!(" @{}", a))
                .unwrap_or_default();
            let progress = self
                .execution_state
                .get(&step.id)
                .map(|s| {
                    let action = s
                        .current_action
                        .as_deref()
                        .unwrap_or("");
                    if s.progress > 0.0 && s.progress < 1.0 {
                        format!(" ({:.0}%{})", s.progress * 100.0, if action.is_empty() { String::new() } else { format!(": {}", action) })
                    } else {
                        String::new()
                    }
                })
                .unwrap_or_default();

            lines.push(format!("{} {}{}{}", status_icon, step.title, agent, progress));
        }

        lines.join("\n")
    }

    /// Export as markdown.
    pub fn to_markdown(&self) -> String {
        self.plan.to_markdown()
    }

    // --- Notification helpers ---

    fn notify_changed(&self, step_id: &str, old: StepStatus, new: StepStatus) {
        for listener in &self.listeners {
            listener.on_task_changed(step_id, old.clone(), new.clone());
        }
    }

    fn notify_added(&self, step_id: &str, title: &str) {
        for listener in &self.listeners {
            listener.on_task_added(step_id, title);
        }
    }

    fn notify_removed(&self, step_id: &str) {
        for listener in &self.listeners {
            listener.on_task_removed(step_id);
        }
    }

    fn notify_progress(&self, step_id: &str, progress: f64, action: &str) {
        for listener in &self.listeners {
            listener.on_progress_update(step_id, progress, action);
        }
    }
}

fn remove_step_recursive(steps: &mut Vec<PlanStep>, id: &str) -> bool {
    if let Some(pos) = steps.iter().position(|s| s.id == id) {
        steps.remove(pos);
        return true;
    }
    for step in steps.iter_mut() {
        if remove_step_recursive(&mut step.children, id) {
            return true;
        }
    }
    false
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

    fn sample_board() -> TaskBoard {
        let mut board = TaskBoard::new("Test Board");
        board
            .execute_command(BoardCommand::AddTask {
                title: "Task A".into(),
                description: "First task".into(),
                priority: StepPriority::High,
            })
            .unwrap();
        board
            .execute_command(BoardCommand::AddTask {
                title: "Task B".into(),
                description: "Second task".into(),
                priority: StepPriority::Medium,
            })
            .unwrap();
        board
            .execute_command(BoardCommand::AddTask {
                title: "Task C".into(),
                description: "Third task".into(),
                priority: StepPriority::Low,
            })
            .unwrap();
        board
    }

    #[test]
    fn test_add_and_summary() {
        let board = sample_board();
        let summary = board.summary();
        assert_eq!(summary.total_tasks, 3);
        assert_eq!(summary.pending, 3);
        assert_eq!(summary.in_progress, 0);
        assert_eq!(summary.completed, 0);
    }

    #[test]
    fn test_start_and_complete_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::InProgress);

        board.execute_command(BoardCommand::CompleteTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Done);
    }

    #[test]
    fn test_pause_and_resume() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        board.execute_command(BoardCommand::PauseTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Blocked);

        board.execute_command(BoardCommand::ResumeTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::InProgress);
    }

    #[test]
    fn test_cancel_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::CancelTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Skipped);
    }

    #[test]
    fn test_remove_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::RemoveTask { id }).unwrap();
        assert_eq!(board.plan().steps.len(), 2);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut board = sample_board();
        assert!(board.execute_command(BoardCommand::RemoveTask { id: "nonexistent".into() }).is_err());
    }

    #[test]
    fn test_assign_agent() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::AssignToAgent {
            task_id: id.clone(),
            agent_name: "researcher".into(),
        }).unwrap();

        let tasks = board.agent_tasks("researcher");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, id);
    }

    #[test]
    fn test_update_progress() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        board.execute_command(BoardCommand::UpdateProgress {
            task_id: id.clone(),
            progress: 0.5,
            action: "Reading files".into(),
        }).unwrap();

        let state = board.execution_state(&id).unwrap();
        assert!((state.progress - 0.5).abs() < 0.001);
        assert_eq!(state.current_action.as_deref(), Some("Reading files"));
    }

    #[test]
    fn test_listener_notifications() {
        let mut board = TaskBoard::new("Test");
        let listener = Arc::new(CollectingBoardListener::new());
        board.add_listener(Arc::clone(&listener) as Arc<dyn TaskBoardListener>);

        board.execute_command(BoardCommand::AddTask {
            title: "Test Task".into(),
            description: "".into(),
            priority: StepPriority::Medium,
        }).unwrap();

        let id = board.plan().steps[0].id.clone();
        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        board.execute_command(BoardCommand::CompleteTask { id }).unwrap();

        assert_eq!(listener.event_count(), 3); // added + changed + changed
    }

    #[test]
    fn test_next_actionable() {
        let mut board = sample_board();
        // All tasks are pending with no deps → all actionable
        assert_eq!(board.next_actionable().len(), 3);

        // Start one
        let id = board.plan().steps[0].id.clone();
        board.execute_command(BoardCommand::StartTask { id }).unwrap();
        // Now 2 are actionable (pending)
        assert_eq!(board.next_actionable().len(), 2);
    }

    #[test]
    fn test_to_display() {
        let board = sample_board();
        let display = board.to_display();
        assert!(display.contains("Test Board"));
        assert!(display.contains("[ ] Task A"));
        assert!(display.contains("[ ] Task B"));
        assert!(display.contains("[ ] Task C"));
    }

    #[test]
    fn test_find_by_title() {
        let board = sample_board();
        let found = board.find_by_title("task b");
        assert!(found.is_some());
        assert_eq!(found.unwrap().title, "Task B");
    }

    #[test]
    fn test_pause_all_resume_all() {
        let mut board = sample_board();
        let id_a = board.plan().steps[0].id.clone();
        let id_b = board.plan().steps[1].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id_a.clone() }).unwrap();
        board.execute_command(BoardCommand::StartTask { id: id_b.clone() }).unwrap();

        board.execute_command(BoardCommand::PauseAll).unwrap();
        assert_eq!(board.plan().find_step(&id_a).unwrap().status, StepStatus::Blocked);
        assert_eq!(board.plan().find_step(&id_b).unwrap().status, StepStatus::Blocked);

        board.execute_command(BoardCommand::ResumeAll).unwrap();
        assert_eq!(board.plan().find_step(&id_a).unwrap().status, StepStatus::InProgress);
        assert_eq!(board.plan().find_step(&id_b).unwrap().status, StepStatus::InProgress);
    }

    #[test]
    fn test_cancel_all() {
        let mut board = sample_board();
        board.execute_command(BoardCommand::CancelAll).unwrap();

        let summary = board.summary();
        assert_eq!(summary.cancelled, 3);
        assert_eq!(summary.pending, 0);
    }

    #[test]
    fn test_from_plan() {
        let mut plan = TaskPlan::new("Existing Plan");
        plan.steps.push(PlanStep::new("Step 1"));
        plan.steps.push(PlanStep::new("Step 2"));

        let board = TaskBoard::from_plan(plan);
        assert_eq!(board.plan().steps.len(), 2);
        assert_eq!(board.name, "Existing Plan");
    }

    #[test]
    fn test_undo_add_task() {
        let mut board = sample_board();
        assert_eq!(board.plan().steps.len(), 3);

        // Undo the last AddTask (Task C)
        let result = board.undo_last();
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Task C"));
        assert_eq!(board.plan().steps.len(), 2);
    }

    #[test]
    fn test_undo_start_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::InProgress);

        let result = board.undo_last();
        assert!(result.is_ok());
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Pending);
    }

    #[test]
    fn test_undo_pause_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        board.execute_command(BoardCommand::PauseTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Blocked);

        let result = board.undo_last();
        assert!(result.is_ok());
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::InProgress);
    }

    #[test]
    fn test_undo_cancel_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::CancelTask { id: id.clone() }).unwrap();
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Skipped);

        let result = board.undo_last();
        assert!(result.is_ok());
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::Pending);
    }

    #[test]
    fn test_undo_empty_history() {
        let mut board = TaskBoard::new("Empty");
        let result = board.undo_last();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Nothing to undo"));
    }

    #[test]
    fn test_undo_complete_task() {
        let mut board = sample_board();
        let id = board.plan().steps[0].id.clone();

        board.execute_command(BoardCommand::StartTask { id: id.clone() }).unwrap();
        board.execute_command(BoardCommand::CompleteTask { id: id.clone() }).unwrap();

        let result = board.undo_last();
        assert!(result.is_ok());
        assert_eq!(board.plan().find_step(&id).unwrap().status, StepStatus::InProgress);
    }
}
