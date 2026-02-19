//! User interaction — agent ↔ user communication during execution
//!
//! Allows autonomous agents to pause, ask the user questions, and resume
//! with the response. Works in both single-agent and multi-agent scenarios.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Query / Response types
// ============================================================================

/// A question an agent wants to ask the user.
#[derive(Debug, Clone)]
pub enum UserQuery {
    /// Free-text question.
    FreeText { question: String },
    /// Multiple choice (single or multi-select).
    Choice {
        question: String,
        options: Vec<String>,
        multi_select: bool,
    },
    /// Yes/no confirmation.
    Confirmation { question: String },
    /// File selection prompt.
    FileSelection {
        prompt: String,
        filter: Option<String>,
    },
}

impl UserQuery {
    pub fn free_text(question: impl Into<String>) -> Self {
        Self::FreeText {
            question: question.into(),
        }
    }

    pub fn confirmation(question: impl Into<String>) -> Self {
        Self::Confirmation {
            question: question.into(),
        }
    }

    pub fn choice(
        question: impl Into<String>,
        options: Vec<String>,
        multi_select: bool,
    ) -> Self {
        Self::Choice {
            question: question.into(),
            options,
            multi_select,
        }
    }

    pub fn file_selection(prompt: impl Into<String>, filter: Option<String>) -> Self {
        Self::FileSelection {
            prompt: prompt.into(),
            filter,
        }
    }

    /// Get the question text regardless of variant.
    pub fn question_text(&self) -> &str {
        match self {
            Self::FreeText { question } => question,
            Self::Choice { question, .. } => question,
            Self::Confirmation { question } => question,
            Self::FileSelection { prompt, .. } => prompt,
        }
    }
}

/// The user's response to a query.
#[derive(Debug, Clone)]
pub enum UserResponse {
    /// Free text response.
    Text(String),
    /// Selected option indices.
    Choices(Vec<usize>),
    /// Yes/no answer.
    Confirmed(bool),
    /// Selected file path.
    File(PathBuf),
    /// User cancelled the query.
    Cancelled,
    /// Query timed out without response.
    Timeout,
}

impl UserResponse {
    /// Check if the response is a cancellation or timeout.
    pub fn is_negative(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Timeout)
    }

    /// Extract text from the response if possible.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }

    /// Extract confirmation value.
    pub fn as_confirmed(&self) -> Option<bool> {
        match self {
            Self::Confirmed(b) => Some(*b),
            _ => None,
        }
    }
}

/// Notification severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyLevel {
    Info,
    Warning,
    Error,
    Success,
}

// ============================================================================
// UserInteractionHandler trait
// ============================================================================

/// Trait for handling user interactions. Implement this to provide a UI backend
/// (CLI, web, desktop, etc.).
pub trait UserInteractionHandler: Send + Sync {
    /// Ask the user a question. Blocks until response or timeout.
    fn ask_user(&self, agent_name: &str, query: UserQuery) -> UserResponse;

    /// Send a notification to the user (non-blocking).
    fn notify_user(&self, agent_name: &str, message: &str, level: NotifyLevel);

    /// Show progress update (non-blocking).
    fn show_progress(&self, agent_name: &str, task: &str, progress: f64);
}

// ============================================================================
// Built-in handler implementations
// ============================================================================

/// Auto-approve handler: always responds with confirmation or default text.
/// Useful for tests and headless operation.
pub struct AutoApproveHandler {
    default_text: String,
    default_file_path: PathBuf,
}

impl AutoApproveHandler {
    pub fn new() -> Self {
        Self {
            default_text: "approved".to_string(),
            default_file_path: PathBuf::from("."),
        }
    }

    pub fn with_default_text(text: impl Into<String>) -> Self {
        Self {
            default_text: text.into(),
            default_file_path: PathBuf::from("."),
        }
    }

    /// Set the default path returned for FileSelection queries.
    pub fn with_default_file_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.default_file_path = path.into();
        self
    }
}

impl Default for AutoApproveHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl UserInteractionHandler for AutoApproveHandler {
    fn ask_user(&self, _agent_name: &str, query: UserQuery) -> UserResponse {
        match query {
            UserQuery::FreeText { .. } => UserResponse::Text(self.default_text.clone()),
            UserQuery::Choice { options, multi_select, .. } => {
                if multi_select {
                    UserResponse::Choices((0..options.len()).collect())
                } else if !options.is_empty() {
                    UserResponse::Choices(vec![0])
                } else {
                    UserResponse::Cancelled
                }
            }
            UserQuery::Confirmation { .. } => UserResponse::Confirmed(true),
            UserQuery::FileSelection { filter, .. } => {
                // If a filter was provided and looks like a path, use it; otherwise use default
                if let Some(ref f) = filter {
                    let p = PathBuf::from(f);
                    if p.extension().is_some() || p.is_absolute() {
                        return UserResponse::File(p);
                    }
                }
                UserResponse::File(self.default_file_path.clone())
            }
        }
    }

    fn notify_user(&self, _agent_name: &str, _message: &str, _level: NotifyLevel) {}

    fn show_progress(&self, _agent_name: &str, _task: &str, _progress: f64) {}
}

/// Callback-based handler: delegates to closures. Good for UI integration.
pub struct CallbackInteractionHandler {
    ask_fn: Box<dyn Fn(&str, UserQuery) -> UserResponse + Send + Sync>,
    notify_fn: Box<dyn Fn(&str, &str, NotifyLevel) + Send + Sync>,
    progress_fn: Box<dyn Fn(&str, &str, f64) + Send + Sync>,
}

impl CallbackInteractionHandler {
    pub fn new(
        ask_fn: impl Fn(&str, UserQuery) -> UserResponse + Send + Sync + 'static,
        notify_fn: impl Fn(&str, &str, NotifyLevel) + Send + Sync + 'static,
        progress_fn: impl Fn(&str, &str, f64) + Send + Sync + 'static,
    ) -> Self {
        Self {
            ask_fn: Box::new(ask_fn),
            notify_fn: Box::new(notify_fn),
            progress_fn: Box::new(progress_fn),
        }
    }
}

impl UserInteractionHandler for CallbackInteractionHandler {
    fn ask_user(&self, agent_name: &str, query: UserQuery) -> UserResponse {
        (self.ask_fn)(agent_name, query)
    }

    fn notify_user(&self, agent_name: &str, message: &str, level: NotifyLevel) {
        (self.notify_fn)(agent_name, message, level);
    }

    fn show_progress(&self, agent_name: &str, task: &str, progress: f64) {
        (self.progress_fn)(agent_name, task, progress);
    }
}

/// A stored notification from an agent.
#[derive(Debug, Clone)]
pub struct StoredNotification {
    pub agent_name: String,
    pub message: String,
    pub level: NotifyLevel,
    pub timestamp: u64,
}

/// A stored progress update from an agent.
#[derive(Debug, Clone)]
pub struct StoredProgress {
    pub agent_name: String,
    pub task: String,
    pub progress: f64,
    pub timestamp: u64,
}

/// Buffered handler: stores queries, notifications, and progress updates for
/// async retrieval. Good for web/API integrations.
pub struct BufferedInteractionHandler {
    queries: Arc<Mutex<Vec<PendingQuery>>>,
    notifications: Arc<Mutex<Vec<StoredNotification>>>,
    progress_updates: Arc<Mutex<HashMap<String, StoredProgress>>>,
    timeout_secs: u64,
}

impl BufferedInteractionHandler {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            queries: Arc::new(Mutex::new(Vec::new())),
            notifications: Arc::new(Mutex::new(Vec::new())),
            progress_updates: Arc::new(Mutex::new(HashMap::new())),
            timeout_secs,
        }
    }

    /// Get all pending (unanswered) queries.
    pub fn pending_queries(&self) -> Vec<PendingQuery> {
        let qs = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        qs.iter().filter(|q| q.response.is_none()).cloned().collect()
    }

    /// Respond to a pending query by ID.
    pub fn respond(&self, query_id: &str, response: UserResponse) -> bool {
        let mut qs = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(q) = qs.iter_mut().find(|q| q.id == query_id) {
            q.response = Some(response);
            true
        } else {
            false
        }
    }

    /// Get the query list handle (for sharing with other threads).
    pub fn queries_handle(&self) -> Arc<Mutex<Vec<PendingQuery>>> {
        Arc::clone(&self.queries)
    }

    /// Get all stored notifications.
    pub fn notifications(&self) -> Vec<StoredNotification> {
        self.notifications.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get the latest progress update for each agent+task combination.
    pub fn progress_updates(&self) -> HashMap<String, StoredProgress> {
        self.progress_updates.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Clear all stored notifications.
    pub fn clear_notifications(&self) {
        self.notifications.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Clear all stored progress updates.
    pub fn clear_progress(&self) {
        self.progress_updates.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

impl UserInteractionHandler for BufferedInteractionHandler {
    fn ask_user(&self, agent_name: &str, query: UserQuery) -> UserResponse {
        let id = format!(
            "q-{}-{}",
            agent_name,
            now_millis() % 1_000_000
        );
        let pending = PendingQuery {
            id: id.clone(),
            agent_name: agent_name.to_string(),
            query,
            created_at: now_millis(),
            response: None,
        };

        {
            let mut qs = self.queries.lock().unwrap_or_else(|e| e.into_inner());
            qs.push(pending);
        }

        // Poll for response (simplified — in real use, use condvar or channel)
        let deadline = now_millis() + self.timeout_secs * 1000;
        loop {
            {
                let qs = self.queries.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(q) = qs.iter().find(|q| q.id == id) {
                    if let Some(ref resp) = q.response {
                        return resp.clone();
                    }
                }
            }
            if now_millis() > deadline {
                return UserResponse::Timeout;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    fn notify_user(&self, agent_name: &str, message: &str, level: NotifyLevel) {
        let notification = StoredNotification {
            agent_name: agent_name.to_string(),
            message: message.to_string(),
            level,
            timestamp: now_millis(),
        };
        if let Ok(mut notifs) = self.notifications.lock() {
            notifs.push(notification);
        }
    }

    fn show_progress(&self, agent_name: &str, task: &str, progress: f64) {
        let key = format!("{}:{}", agent_name, task);
        let update = StoredProgress {
            agent_name: agent_name.to_string(),
            task: task.to_string(),
            progress,
            timestamp: now_millis(),
        };
        if let Ok(mut updates) = self.progress_updates.lock() {
            updates.insert(key, update);
        }
    }
}

// ============================================================================
// PendingQuery
// ============================================================================

/// A query that's waiting for the user to respond.
#[derive(Debug, Clone)]
pub struct PendingQuery {
    pub id: String,
    pub agent_name: String,
    pub query: UserQuery,
    pub created_at: u64,
    pub response: Option<UserResponse>,
}

// ============================================================================
// InteractionManager
// ============================================================================

/// Central manager for agent ↔ user interactions. Shared between agents.
pub struct InteractionManager {
    handler: Arc<dyn UserInteractionHandler>,
    query_log: Mutex<Vec<InteractionLogEntry>>,
    timeout_secs: u64,
    pending_queries: Arc<Mutex<HashMap<String, PendingQuery>>>,
}

/// A logged interaction.
#[derive(Debug, Clone)]
pub struct InteractionLogEntry {
    pub agent_name: String,
    pub query: UserQuery,
    pub response: UserResponse,
    pub timestamp: u64,
}

impl InteractionManager {
    pub fn new(handler: Arc<dyn UserInteractionHandler>, timeout_secs: u64) -> Self {
        Self {
            handler,
            query_log: Mutex::new(Vec::new()),
            timeout_secs,
            pending_queries: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Ask the user a question (blocking). The agent will pause until a response.
    pub fn ask(&self, agent_name: &str, query: UserQuery) -> UserResponse {
        let response = self.handler.ask_user(agent_name, query.clone());
        let mut log = self.query_log.lock().unwrap_or_else(|e| e.into_inner());
        log.push(InteractionLogEntry {
            agent_name: agent_name.to_string(),
            query,
            response: response.clone(),
            timestamp: now_millis(),
        });
        response
    }

    /// Submit a query without blocking. Returns query_id for later retrieval.
    pub fn ask_async(&self, agent_name: &str, query: UserQuery) -> String {
        let id = format!("iq-{}-{}", agent_name, now_millis() % 1_000_000);
        let pending = PendingQuery {
            id: id.clone(),
            agent_name: agent_name.to_string(),
            query,
            created_at: now_millis(),
            response: None,
        };
        let mut pq = self.pending_queries.lock().unwrap_or_else(|e| e.into_inner());
        pq.insert(id.clone(), pending);
        id
    }

    /// Respond to a previously submitted async query.
    pub fn respond(&self, query_id: &str, response: UserResponse) -> bool {
        let mut pq = self.pending_queries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pending) = pq.get_mut(query_id) {
            let query_clone = pending.query.clone();
            let agent_name = pending.agent_name.clone();
            pending.response = Some(response.clone());

            let mut log = self.query_log.lock().unwrap_or_else(|e| e.into_inner());
            log.push(InteractionLogEntry {
                agent_name,
                query: query_clone,
                response,
                timestamp: now_millis(),
            });
            true
        } else {
            false
        }
    }

    /// Get the response for an async query (None if not yet answered).
    pub fn get_response(&self, query_id: &str) -> Option<UserResponse> {
        let pq = self.pending_queries.lock().unwrap_or_else(|e| e.into_inner());
        pq.get(query_id).and_then(|p| p.response.clone())
    }

    /// Get all pending (unanswered) queries.
    pub fn pending_queries(&self) -> Vec<PendingQuery> {
        let pq = self.pending_queries.lock().unwrap_or_else(|e| e.into_inner());
        pq.values()
            .filter(|q| q.response.is_none())
            .cloned()
            .collect()
    }

    /// Send a notification to the user.
    pub fn notify(&self, agent_name: &str, message: &str, level: NotifyLevel) {
        self.handler.notify_user(agent_name, message, level);
    }

    /// Update progress display.
    pub fn show_progress(&self, agent_name: &str, task: &str, progress: f64) {
        self.handler.show_progress(agent_name, task, progress);
    }

    /// Get the full interaction log.
    pub fn interaction_log(&self) -> Vec<InteractionLogEntry> {
        self.query_log.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get the configured timeout.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Clear the interaction log.
    pub fn clear_log(&self) {
        self.query_log.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Clear all pending queries.
    pub fn clear_pending(&self) {
        self.pending_queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

// ============================================================================
// Helpers
// ============================================================================

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

    #[test]
    fn test_auto_approve_free_text() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user("agent-1", UserQuery::free_text("What file?"));
        assert_eq!(resp.as_text(), Some("approved"));
    }

    #[test]
    fn test_auto_approve_confirmation() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user("agent-1", UserQuery::confirmation("Delete file?"));
        assert_eq!(resp.as_confirmed(), Some(true));
    }

    #[test]
    fn test_auto_approve_choice() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user(
            "agent-1",
            UserQuery::choice("Pick one", vec!["A".into(), "B".into(), "C".into()], false),
        );
        if let UserResponse::Choices(indices) = resp {
            assert_eq!(indices, vec![0]);
        } else {
            panic!("Expected Choices");
        }
    }

    #[test]
    fn test_auto_approve_multi_choice() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user(
            "agent-1",
            UserQuery::choice("Pick many", vec!["A".into(), "B".into()], true),
        );
        if let UserResponse::Choices(indices) = resp {
            assert_eq!(indices, vec![0, 1]);
        } else {
            panic!("Expected Choices");
        }
    }

    #[test]
    fn test_auto_approve_custom_text() {
        let handler = AutoApproveHandler::with_default_text("yes please");
        let resp = handler.ask_user("agent-1", UserQuery::free_text("Anything?"));
        assert_eq!(resp.as_text(), Some("yes please"));
    }

    #[test]
    fn test_callback_handler() {
        let notifications: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let notif_clone = Arc::clone(&notifications);

        let handler = CallbackInteractionHandler::new(
            |_agent, _query| UserResponse::Text("callback response".into()),
            move |agent, msg, _level| {
                notif_clone.lock().unwrap().push(format!("{}: {}", agent, msg));
            },
            |_agent, _task, _progress| {},
        );

        let resp = handler.ask_user("test-agent", UserQuery::free_text("Q?"));
        assert_eq!(resp.as_text(), Some("callback response"));

        handler.notify_user("test-agent", "hello", NotifyLevel::Info);
        assert_eq!(notifications.lock().unwrap().len(), 1);
        assert_eq!(
            notifications.lock().unwrap()[0],
            "test-agent: hello"
        );
    }

    #[test]
    fn test_interaction_manager_ask() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 30);

        let resp = mgr.ask("agent-1", UserQuery::confirmation("Continue?"));
        assert_eq!(resp.as_confirmed(), Some(true));

        let log = mgr.interaction_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].agent_name, "agent-1");
    }

    #[test]
    fn test_interaction_manager_async_queries() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 30);

        let qid = mgr.ask_async("agent-1", UserQuery::free_text("What?"));
        assert!(!qid.is_empty());

        // Query is pending
        let pending = mgr.pending_queries();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].agent_name, "agent-1");

        // Respond
        assert!(mgr.respond(&qid, UserResponse::Text("answer".into())));
        assert_eq!(
            mgr.get_response(&qid).unwrap().as_text(),
            Some("answer")
        );

        // No longer pending
        let pending = mgr.pending_queries();
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn test_interaction_manager_respond_unknown_id() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 30);
        assert!(!mgr.respond("nonexistent", UserResponse::Text("x".into())));
    }

    #[test]
    fn test_user_response_helpers() {
        assert!(UserResponse::Cancelled.is_negative());
        assert!(UserResponse::Timeout.is_negative());
        assert!(!UserResponse::Text("hi".into()).is_negative());
        assert!(!UserResponse::Confirmed(true).is_negative());

        assert_eq!(UserResponse::Text("hi".into()).as_text(), Some("hi"));
        assert_eq!(UserResponse::Confirmed(false).as_confirmed(), Some(false));
        assert_eq!(UserResponse::Cancelled.as_text(), None);
    }

    #[test]
    fn test_user_query_question_text() {
        assert_eq!(
            UserQuery::free_text("What file?").question_text(),
            "What file?"
        );
        assert_eq!(
            UserQuery::confirmation("Sure?").question_text(),
            "Sure?"
        );
        assert_eq!(
            UserQuery::choice("Pick", vec![], false).question_text(),
            "Pick"
        );
        assert_eq!(
            UserQuery::file_selection("Select file", None).question_text(),
            "Select file"
        );
    }

    #[test]
    fn test_auto_approve_file_selection_default() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user("a", UserQuery::file_selection("Pick", None));
        if let UserResponse::File(p) = resp {
            assert_eq!(p, PathBuf::from("."));
        } else {
            panic!("Expected File");
        }
    }

    #[test]
    fn test_auto_approve_file_selection_with_filter() {
        let handler = AutoApproveHandler::new();
        let resp = handler.ask_user(
            "a",
            UserQuery::file_selection("Pick", Some("report.pdf".into())),
        );
        if let UserResponse::File(p) = resp {
            assert_eq!(p, PathBuf::from("report.pdf"));
        } else {
            panic!("Expected File");
        }
    }

    #[test]
    fn test_auto_approve_custom_file_path() {
        let handler = AutoApproveHandler::new()
            .with_default_file_path("/tmp/output");
        let resp = handler.ask_user("a", UserQuery::file_selection("Pick", None));
        if let UserResponse::File(p) = resp {
            assert_eq!(p, PathBuf::from("/tmp/output"));
        } else {
            panic!("Expected File");
        }
    }

    #[test]
    fn test_buffered_handler_notifications() {
        let handler = BufferedInteractionHandler::new(5);
        assert!(handler.notifications().is_empty());

        handler.notify_user("agent-1", "Starting task", NotifyLevel::Info);
        handler.notify_user("agent-2", "Warning: slow", NotifyLevel::Warning);

        let notifs = handler.notifications();
        assert_eq!(notifs.len(), 2);
        assert_eq!(notifs[0].agent_name, "agent-1");
        assert_eq!(notifs[0].message, "Starting task");
        assert_eq!(notifs[0].level, NotifyLevel::Info);
        assert_eq!(notifs[1].agent_name, "agent-2");
        assert_eq!(notifs[1].level, NotifyLevel::Warning);

        handler.clear_notifications();
        assert!(handler.notifications().is_empty());
    }

    #[test]
    fn test_buffered_handler_progress() {
        let handler = BufferedInteractionHandler::new(5);
        assert!(handler.progress_updates().is_empty());

        handler.show_progress("agent-1", "indexing", 0.5);
        handler.show_progress("agent-1", "indexing", 0.8);
        handler.show_progress("agent-2", "searching", 0.3);

        let updates = handler.progress_updates();
        assert_eq!(updates.len(), 2); // keyed by agent:task
        let idx = updates.get("agent-1:indexing").unwrap();
        assert!((idx.progress - 0.8).abs() < 1e-10); // latest update
        let search = updates.get("agent-2:searching").unwrap();
        assert!((search.progress - 0.3).abs() < 1e-10);

        handler.clear_progress();
        assert!(handler.progress_updates().is_empty());
    }

    #[test]
    fn test_interaction_manager_clear_log() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 30);

        mgr.ask("agent-1", UserQuery::confirmation("OK?"));
        assert_eq!(mgr.interaction_log().len(), 1);

        mgr.clear_log();
        assert!(mgr.interaction_log().is_empty());
    }

    #[test]
    fn test_interaction_manager_clear_pending() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 30);

        let _qid = mgr.ask_async("agent-1", UserQuery::free_text("What?"));
        assert_eq!(mgr.pending_queries().len(), 1);

        mgr.clear_pending();
        assert!(mgr.pending_queries().is_empty());
    }

    #[test]
    fn test_interaction_manager_timeout_secs() {
        let handler = Arc::new(AutoApproveHandler::new());
        let mgr = InteractionManager::new(handler, 42);
        assert_eq!(mgr.timeout_secs(), 42);
    }
}
