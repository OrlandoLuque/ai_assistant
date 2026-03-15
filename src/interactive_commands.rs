//! Interactive commands — bilingual command processor for task management
//!
//! Interprets user prompts during execution to modify the task board,
//! pause/resume agents, query status, and manage tasks.
//! Supports both English and Spanish keywords for all operations.

use crate::task_board::{BoardCommand, TaskBoard};
use crate::task_planning::StepPriority;
use std::sync::{Arc, RwLock};

// ============================================================================
// UserIntent
// ============================================================================

/// All possible parsed intents from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UserIntent {
    AddTask {
        description: String,
    },
    RemoveTask {
        identifier: String,
    },
    PauseTask {
        identifier: String,
    },
    ResumeTask {
        identifier: String,
    },
    CancelTask {
        identifier: String,
    },
    ReprioritizeTask {
        identifier: String,
        priority: String,
    },
    PauseAll,
    ResumeAll,
    CancelAll,
    ShowStatus,
    ShowTaskDetail {
        identifier: String,
    },
    Undo,
    Help,
    Unknown(String),
}

// ============================================================================
// CommandResult
// ============================================================================

/// Result of executing a user command against the task board.
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub success: bool,
    pub message: String,
    pub board_snapshot: Option<String>,
}

impl CommandResult {
    fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            board_snapshot: None,
        }
    }

    fn ok_with_snapshot(message: impl Into<String>, snapshot: String) -> Self {
        Self {
            success: true,
            message: message.into(),
            board_snapshot: Some(snapshot),
        }
    }

    fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            board_snapshot: None,
        }
    }
}

// ============================================================================
// CommandProcessor
// ============================================================================

/// Processes bilingual (English/Spanish) user commands against a shared task board.
pub struct CommandProcessor {
    task_board: Arc<RwLock<TaskBoard>>,
}

impl CommandProcessor {
    /// Create a new command processor bound to the given task board.
    pub fn new(board: Arc<RwLock<TaskBoard>>) -> Self {
        Self { task_board: board }
    }

    /// Parse a raw user input string into a structured [`UserIntent`].
    ///
    /// Matching is case-insensitive and supports both English and Spanish keywords.
    /// "All" variants are checked before single-task variants so that
    /// "pause all" does not get parsed as PauseTask with identifier "all".
    pub fn parse_intent(input: &str) -> UserIntent {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return UserIntent::Unknown(String::new());
        }

        let lower = trimmed.to_lowercase();

        // --- "all" commands (checked first to avoid partial matches) ----------

        if lower == "pause all"
            || lower == "pausa todo"
            || lower == "stop all"
            || lower == "para todo"
        {
            return UserIntent::PauseAll;
        }
        if lower == "resume all"
            || lower == "continua todo"
            || lower == "sigue todo"
            || lower == "reanuda todo"
        {
            return UserIntent::ResumeAll;
        }
        if lower == "cancel all"
            || lower == "cancela todo"
            || lower == "abort all"
            || lower == "aborta todo"
        {
            return UserIntent::CancelAll;
        }

        // --- Undo / Help (no extra arguments) --------------------------------

        if lower == "undo" || lower == "deshacer" {
            return UserIntent::Undo;
        }
        if lower == "help" || lower == "ayuda" {
            return UserIntent::Help;
        }

        // --- Status (may be the entire input or start a phrase) ---------------

        if lower == "status"
            || lower == "estado"
            || lower.starts_with("how")
            || lower.starts_with("cómo va")
            || lower.starts_with("como va")
            || lower == "progress"
            || lower == "progreso"
        {
            return UserIntent::ShowStatus;
        }

        // --- Commands with a task identifier ---------------------------------

        // Add task
        if let Some(rest) = strip_keyword(&lower, &["add ", "añade ", "nueva tarea ", "new task "])
        {
            let desc = extract_rest(trimmed, rest.len());
            return UserIntent::AddTask { description: desc };
        }

        // Remove task
        if let Some(rest) = strip_keyword(
            &lower,
            &["remove ", "elimina ", "quita ", "delete ", "borra "],
        ) {
            let id = extract_rest(trimmed, rest.len());
            return UserIntent::RemoveTask { identifier: id };
        }

        // Pause task (single)
        if let Some(rest) = strip_keyword(&lower, &["pause ", "pausa ", "stop ", "para "]) {
            let id = extract_rest(trimmed, rest.len());
            return UserIntent::PauseTask { identifier: id };
        }

        // Resume task (single)
        if let Some(rest) = strip_keyword(&lower, &["resume ", "continua ", "sigue ", "reanuda "]) {
            let id = extract_rest(trimmed, rest.len());
            return UserIntent::ResumeTask { identifier: id };
        }

        // Cancel task (single)
        if let Some(rest) = strip_keyword(&lower, &["cancel ", "cancela ", "abort ", "aborta "]) {
            let id = extract_rest(trimmed, rest.len());
            return UserIntent::CancelTask { identifier: id };
        }

        // Reprioritize task
        if let Some(rest) =
            strip_keyword(&lower, &["priority ", "prioridad ", "urgente ", "urgent "])
        {
            let id = extract_rest(trimmed, rest.len());
            // Try to split "task_identifier high" — last word might be the priority
            let parts: Vec<&str> = id.rsplitn(2, ' ').collect();
            if parts.len() == 2 {
                return UserIntent::ReprioritizeTask {
                    identifier: parts[1].to_string(),
                    priority: parts[0].to_string(),
                };
            }
            return UserIntent::ReprioritizeTask {
                identifier: id,
                priority: "high".to_string(),
            };
        }

        // Show task detail
        if let Some(rest) = strip_keyword(&lower, &["detail ", "detalle ", "show ", "muestra "]) {
            let id = extract_rest(trimmed, rest.len());
            return UserIntent::ShowTaskDetail { identifier: id };
        }

        UserIntent::Unknown(trimmed.to_string())
    }

    /// Execute a parsed intent against the task board.
    pub fn execute(&self, intent: &UserIntent) -> CommandResult {
        match intent {
            UserIntent::ShowStatus => {
                let board = match self.task_board.read() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let display = board.to_display();
                CommandResult::ok_with_snapshot("Current board status:", display)
            }

            UserIntent::AddTask { description } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let cmd = BoardCommand::AddTask {
                    title: description.clone(),
                    description: description.clone(),
                    priority: StepPriority::Medium,
                };
                match board.execute_command(cmd) {
                    Ok(()) => CommandResult::ok(format!("Task added: {}", description)),
                    Err(e) => CommandResult::err(format!("Failed to add task: {}", e)),
                }
            }

            UserIntent::RemoveTask { identifier } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                match resolved {
                    Some(id) => {
                        let cmd = BoardCommand::RemoveTask { id: id.clone() };
                        match board.execute_command(cmd) {
                            Ok(()) => CommandResult::ok(format!("Task removed: {}", id)),
                            Err(e) => CommandResult::err(format!("Failed to remove task: {}", e)),
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::PauseTask { identifier } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                match resolved {
                    Some(id) => {
                        let cmd = BoardCommand::PauseTask { id: id.clone() };
                        match board.execute_command(cmd) {
                            Ok(()) => CommandResult::ok(format!("Task paused: {}", id)),
                            Err(e) => CommandResult::err(format!("Failed to pause task: {}", e)),
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::ResumeTask { identifier } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                match resolved {
                    Some(id) => {
                        let cmd = BoardCommand::ResumeTask { id: id.clone() };
                        match board.execute_command(cmd) {
                            Ok(()) => CommandResult::ok(format!("Task resumed: {}", id)),
                            Err(e) => CommandResult::err(format!("Failed to resume task: {}", e)),
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::CancelTask { identifier } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                match resolved {
                    Some(id) => {
                        let cmd = BoardCommand::CancelTask { id: id.clone() };
                        match board.execute_command(cmd) {
                            Ok(()) => CommandResult::ok(format!("Task cancelled: {}", id)),
                            Err(e) => CommandResult::err(format!("Failed to cancel task: {}", e)),
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::ReprioritizeTask {
                identifier,
                priority,
            } => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                let parsed_priority = parse_priority(priority);
                match resolved {
                    Some(id) => {
                        let cmd = BoardCommand::ReprioritizeTask {
                            id: id.clone(),
                            priority: parsed_priority,
                        };
                        match board.execute_command(cmd) {
                            Ok(()) => CommandResult::ok(format!(
                                "Task {} reprioritized to {:?}",
                                id, priority
                            )),
                            Err(e) => CommandResult::err(format!("Failed to reprioritize: {}", e)),
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::PauseAll => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                match board.execute_command(BoardCommand::PauseAll) {
                    Ok(()) => CommandResult::ok("All active tasks paused."),
                    Err(e) => CommandResult::err(format!("Failed to pause all: {}", e)),
                }
            }

            UserIntent::ResumeAll => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                match board.execute_command(BoardCommand::ResumeAll) {
                    Ok(()) => CommandResult::ok("All blocked tasks resumed."),
                    Err(e) => CommandResult::err(format!("Failed to resume all: {}", e)),
                }
            }

            UserIntent::CancelAll => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                match board.execute_command(BoardCommand::CancelAll) {
                    Ok(()) => CommandResult::ok("All remaining tasks cancelled."),
                    Err(e) => CommandResult::err(format!("Failed to cancel all: {}", e)),
                }
            }

            UserIntent::ShowTaskDetail { identifier } => {
                let board = match self.task_board.read() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                let resolved = resolve_task_id(identifier, &board);
                match resolved {
                    Some(id) => {
                        if let Some(step) = board.plan().find_step(&id) {
                            let desc = step.description.as_deref().unwrap_or("(no description)");
                            let detail = format!(
                                "Task: {}\nID: {}\nStatus: {:?}\nPriority: {:?}\nDescription: {}\nBlocked by: {:?}",
                                step.title, step.id, step.status, step.priority, desc, step.blocked_by
                            );
                            CommandResult::ok(detail)
                        } else {
                            CommandResult::err(format!("Task not found: {}", identifier))
                        }
                    }
                    None => CommandResult::err(format!("Task not found: {}", identifier)),
                }
            }

            UserIntent::Undo => {
                let mut board = match self.task_board.write() {
                    Ok(b) => b,
                    Err(e) => return CommandResult::err(format!("Lock error: {}", e)),
                };
                match board.undo_last() {
                    Ok(msg) => CommandResult::ok(msg),
                    Err(e) => CommandResult::err(e),
                }
            }

            UserIntent::Help => {
                let help_text = "\
Available commands (EN / ES):
  add <desc>           / añade <desc>          - Add a new task
  remove <task>        / elimina <task>        - Remove a task
  pause <task>         / pausa <task>          - Pause a task
  resume <task>        / continua <task>       - Resume a task
  cancel <task>        / cancela <task>        - Cancel a task
  priority <task> <p>  / prioridad <task> <p>  - Reprioritize a task
  pause all            / pausa todo            - Pause all active tasks
  resume all           / continua todo         - Resume all blocked tasks
  cancel all           / cancela todo          - Cancel all remaining tasks
  status               / estado                - Show board status
  detail <task>        / detalle <task>        - Show task details
  undo                 / deshacer              - Undo last command
  help                 / ayuda                 - Show this help

<task> can be a task ID or a partial title match.
<p> can be: critical, high, medium, low, optional.";
                CommandResult::ok(help_text)
            }

            UserIntent::Unknown(raw) => CommandResult::err(format!(
                "Command not recognized: '{}'. Type 'help' or 'ayuda' for available commands.",
                raw
            )),
        }
    }

    /// Convenience method: parse the input and execute in one step.
    pub fn process_input(&self, input: &str) -> CommandResult {
        let intent = Self::parse_intent(input);
        self.execute(&intent)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Try to strip one of the given keyword prefixes from the lowercased input.
/// Returns the remaining (lowercased) substring after the keyword if matched.
fn strip_keyword<'a>(lower: &'a str, keywords: &[&str]) -> Option<&'a str> {
    for kw in keywords {
        if lower.starts_with(kw) {
            return Some(&lower[kw.len()..]);
        }
    }
    None
}

/// Given the original (non-lowercased) trimmed input and the length of the
/// remaining portion in the lowercased version, extract the corresponding
/// original-cased rest string.  This preserves the user's casing for
/// task descriptions and identifiers.
fn extract_rest(original_trimmed: &str, rest_len: usize) -> String {
    let start = original_trimmed.len() - rest_len;
    original_trimmed[start..].trim().to_string()
}

/// Resolve a user-provided identifier to a concrete task ID.
///
/// Tries exact ID match first, then falls back to a partial title search
/// via `TaskBoard::find_by_title`.
fn resolve_task_id(identifier: &str, board: &TaskBoard) -> Option<String> {
    // Exact ID match
    if board.plan().find_step(identifier).is_some() {
        return Some(identifier.to_string());
    }
    // Partial title match
    if let Some(step) = board.find_by_title(identifier) {
        return Some(step.id.clone());
    }
    None
}

/// Parse a priority string (case-insensitive, bilingual) into a [`StepPriority`].
fn parse_priority(s: &str) -> StepPriority {
    match s.to_lowercase().as_str() {
        "critical" | "critica" | "crítica" => StepPriority::Critical,
        "high" | "alta" | "alto" => StepPriority::High,
        "medium" | "media" | "medio" | "normal" => StepPriority::Medium,
        "low" | "baja" | "bajo" => StepPriority::Low,
        "optional" | "opcional" => StepPriority::Optional,
        // "urgente"/"urgent" already handled at the keyword level; default to High
        _ => StepPriority::High,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_board::{BoardCommand, TaskBoard};
    use crate::task_planning::StepPriority;

    /// Create a board wrapped in Arc<RwLock<>> for the processor.
    fn make_board(name: &str) -> Arc<RwLock<TaskBoard>> {
        Arc::new(RwLock::new(TaskBoard::new(name)))
    }

    /// Create a board with a couple of pre-existing tasks and return
    /// the shared board plus the IDs of the added tasks.
    fn board_with_tasks() -> (Arc<RwLock<TaskBoard>>, Vec<String>) {
        let board = make_board("Test");
        let mut ids = Vec::new();
        {
            let mut b = board.write().unwrap();
            b.execute_command(BoardCommand::AddTask {
                title: "Setup database".into(),
                description: "Initialize the DB schema".into(),
                priority: StepPriority::High,
            })
            .unwrap();
            b.execute_command(BoardCommand::AddTask {
                title: "Write tests".into(),
                description: "Unit and integration tests".into(),
                priority: StepPriority::Medium,
            })
            .unwrap();
            for step in &b.plan().steps {
                ids.push(step.id.clone());
            }
        }
        (board, ids)
    }

    // --- Parsing tests -------------------------------------------------------

    #[test]
    fn test_parse_pause_en() {
        let intent = CommandProcessor::parse_intent("pause my-task-123");
        assert_eq!(
            intent,
            UserIntent::PauseTask {
                identifier: "my-task-123".to_string()
            }
        );
    }

    #[test]
    fn test_parse_pause_es() {
        let intent = CommandProcessor::parse_intent("pausa mi-tarea-456");
        assert_eq!(
            intent,
            UserIntent::PauseTask {
                identifier: "mi-tarea-456".to_string()
            }
        );
    }

    #[test]
    fn test_parse_resume() {
        let intent_en = CommandProcessor::parse_intent("resume task-1");
        assert_eq!(
            intent_en,
            UserIntent::ResumeTask {
                identifier: "task-1".to_string()
            }
        );

        let intent_es = CommandProcessor::parse_intent("continua tarea-1");
        assert_eq!(
            intent_es,
            UserIntent::ResumeTask {
                identifier: "tarea-1".to_string()
            }
        );
    }

    #[test]
    fn test_parse_add_task() {
        let intent = CommandProcessor::parse_intent("add Implement caching layer");
        assert_eq!(
            intent,
            UserIntent::AddTask {
                description: "Implement caching layer".to_string()
            }
        );

        let intent_es = CommandProcessor::parse_intent("añade Implementar capa de cache");
        assert_eq!(
            intent_es,
            UserIntent::AddTask {
                description: "Implementar capa de cache".to_string()
            }
        );
    }

    #[test]
    fn test_parse_status() {
        assert_eq!(
            CommandProcessor::parse_intent("status"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("estado"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("progress"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("how is it going"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("cómo va el proyecto"),
            UserIntent::ShowStatus
        );
    }

    #[test]
    fn test_parse_cancel_all() {
        assert_eq!(
            CommandProcessor::parse_intent("cancel all"),
            UserIntent::CancelAll
        );
        assert_eq!(
            CommandProcessor::parse_intent("cancela todo"),
            UserIntent::CancelAll
        );
    }

    #[test]
    fn test_parse_help() {
        assert_eq!(CommandProcessor::parse_intent("help"), UserIntent::Help);
        assert_eq!(CommandProcessor::parse_intent("ayuda"), UserIntent::Help);
    }

    #[test]
    fn test_parse_unknown() {
        let intent = CommandProcessor::parse_intent("do something weird");
        assert_eq!(
            intent,
            UserIntent::Unknown("do something weird".to_string())
        );
    }

    // --- Execution tests -----------------------------------------------------

    #[test]
    fn test_execute_show_status() {
        let (board, _ids) = board_with_tasks();
        let proc = CommandProcessor::new(board);
        let result = proc.execute(&UserIntent::ShowStatus);
        assert!(result.success);
        assert!(result.board_snapshot.is_some());
        let snap = result.board_snapshot.unwrap();
        assert!(snap.contains("Setup database"));
        assert!(snap.contains("Write tests"));
    }

    #[test]
    fn test_execute_add_task() {
        let board = make_board("Test");
        let proc = CommandProcessor::new(Arc::clone(&board));
        let result = proc.execute(&UserIntent::AddTask {
            description: "New shiny task".to_string(),
        });
        assert!(result.success);
        assert!(result.message.contains("New shiny task"));

        let b = board.read().unwrap();
        assert_eq!(b.plan().steps.len(), 1);
        assert_eq!(b.plan().steps[0].title, "New shiny task");
        assert_eq!(b.plan().steps[0].priority, StepPriority::Medium);
    }

    #[test]
    fn test_process_input_roundtrip() {
        let (board, _ids) = board_with_tasks();
        let proc = CommandProcessor::new(Arc::clone(&board));

        // Add a task via natural input
        let r1 = proc.process_input("add Deploy to staging");
        assert!(r1.success);

        // Query status
        let r2 = proc.process_input("status");
        assert!(r2.success);
        let snap = r2.board_snapshot.unwrap();
        assert!(snap.contains("Deploy to staging"));

        // Pause by partial title
        let r3 = proc.process_input("pause Deploy");
        assert!(r3.success);

        // Help
        let r4 = proc.process_input("ayuda");
        assert!(r4.success);
        assert!(r4.message.contains("Available commands"));

        // Unknown
        let r5 = proc.process_input("xyzzy");
        assert!(!r5.success);
        assert!(r5.message.contains("not recognized"));
    }

    // --- Additional parsing tests --------------------------------------------

    #[test]
    fn test_parse_bilingual() {
        // English: "pause task-1"
        let intent_en = CommandProcessor::parse_intent("pause task-1");
        // Spanish: "pausa task-1"
        let intent_es = CommandProcessor::parse_intent("pausa task-1");

        // Both should parse to the same PauseTask intent with the same identifier
        assert_eq!(
            intent_en,
            UserIntent::PauseTask {
                identifier: "task-1".to_string()
            }
        );
        assert_eq!(intent_en, intent_es);
    }

    #[test]
    fn test_unknown_command_passthrough() {
        let intent = CommandProcessor::parse_intent("tell me a joke");
        assert_eq!(intent, UserIntent::Unknown("tell me a joke".to_string()));
    }

    #[test]
    fn test_parse_status_queries() {
        assert_eq!(
            CommandProcessor::parse_intent("status"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("how are things"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("estado"),
            UserIntent::ShowStatus
        );
        assert_eq!(
            CommandProcessor::parse_intent("cómo va todo"),
            UserIntent::ShowStatus
        );
    }

    #[test]
    fn test_undo_command() {
        let (board, _ids) = board_with_tasks();
        let proc = CommandProcessor::new(Arc::clone(&board));

        // Add a task then undo it
        let r1 = proc.process_input("add Temporary task");
        assert!(r1.success);
        assert_eq!(board.read().unwrap().plan().steps.len(), 3);

        let r2 = proc.process_input("undo");
        assert!(r2.success, "undo should succeed: {}", r2.message);
        assert!(r2.message.contains("Undone"));
        assert_eq!(board.read().unwrap().plan().steps.len(), 2);
    }

    #[test]
    fn test_undo_empty_board() {
        let board = Arc::new(RwLock::new(TaskBoard::new("Empty")));
        let proc = CommandProcessor::new(board);

        let r = proc.process_input("deshacer");
        assert!(!r.success);
        assert!(r.message.contains("Nothing to undo"));
    }
}
