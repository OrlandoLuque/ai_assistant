//! MCP tools for user task management (personal TODO lists).
//!
//! Provides CRUD operations, full-text search, filtering, and soft-delete
//! with rollback. Tasks persist in SQLite (same unified.db file).

use crate::mcp_protocol::server::McpServer;
use crate::mcp_protocol::types::{McpTool, McpToolAnnotation};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

// ============================================================================
// Types
// ============================================================================

/// Priority levels for user tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl TaskPriority {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "critical" => Some(Self::Critical),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Status of a user task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TaskStatus {
    Pending,
    InProgress,
    Done,
}

impl TaskStatus {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "done" => Some(Self::Done),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Done => "done",
        }
    }
}

/// A user-managed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserTask {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub due_date: Option<String>,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Filters for listing tasks.
pub struct TaskFilters {
    pub status: Option<TaskStatus>,
    pub priority: Option<TaskPriority>,
    pub tag: Option<String>,
    pub sort_by: TaskSortField,
    pub sort_order: SortOrder,
    pub limit: usize,
}

impl Default for TaskFilters {
    fn default() -> Self {
        Self {
            status: None,
            priority: None,
            tag: None,
            sort_by: TaskSortField::Created,
            sort_order: SortOrder::Desc,
            limit: 50,
        }
    }
}

/// Sort field for task listing.
pub enum TaskSortField {
    DueDate,
    Priority,
    Created,
}

/// Sort order.
pub enum SortOrder {
    Asc,
    Desc,
}

/// Partial update fields.
pub struct TaskUpdates {
    pub title: Option<String>,
    pub description: Option<String>,
    pub status: Option<TaskStatus>,
    pub priority: Option<TaskPriority>,
    pub due_date: Option<Option<String>>,
    pub tags: Option<Vec<String>>,
}

// ============================================================================
// Validation
// ============================================================================

const MAX_TITLE_LEN: usize = 500;
const MAX_DESC_LEN: usize = 10_000;
const MAX_TAG_LEN: usize = 50;
const MAX_TAGS: usize = 20;
const MAX_RESULTS: usize = 200;
/// Soft-deleted tasks are purged after this many days.
const ROLLBACK_RETENTION_DAYS: u64 = 30;

fn validate_title(title: &str) -> Result<(), String> {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return Err("Title cannot be empty".into());
    }
    if trimmed.len() > MAX_TITLE_LEN {
        return Err(format!(
            "Title too long: {} chars (max {})",
            trimmed.len(),
            MAX_TITLE_LEN
        ));
    }
    Ok(())
}

fn validate_due_date(date: &str) -> Result<(), String> {
    chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|_| format!("Invalid date format '{}': expected YYYY-MM-DD", date))?;
    Ok(())
}

fn sanitize_tags(tags: &[serde_json::Value]) -> Result<Vec<String>, String> {
    if tags.len() > MAX_TAGS {
        return Err(format!("Too many tags: {} (max {})", tags.len(), MAX_TAGS));
    }
    let mut result: Vec<String> = Vec::new();
    for tag in tags {
        let s = tag.as_str().ok_or("Each tag must be a string")?;
        let trimmed = s.trim().to_lowercase();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.len() > MAX_TAG_LEN {
            return Err(format!(
                "Tag '{}' too long (max {} chars)",
                trimmed, MAX_TAG_LEN
            ));
        }
        // Reject tags with SQL/JSON special chars
        if trimmed.contains('"') || trimmed.contains('\'') || trimmed.contains(';') {
            return Err(format!("Tag '{}' contains invalid characters", trimmed));
        }
        if !result.contains(&trimmed) {
            result.push(trimmed);
        }
    }
    Ok(result)
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn new_uuid() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (d.as_nanos() & 0xFFFF_FFFF) as u32,
        ((d.as_nanos() >> 32) & 0xFFFF) as u16,
        ((d.as_nanos() >> 48) & 0x0FFF) as u16,
        (0x8000 | ((d.as_nanos() >> 60) & 0x3FFF)) as u16,
        (d.as_nanos() >> 74) ^ (d.subsec_nanos() as u128),
    )
}

// ============================================================================
// UserTaskStore
// ============================================================================

/// SQLite-backed store for user tasks with soft-delete and rollback.
pub struct UserTaskStore {
    conn: rusqlite::Connection,
}

impl UserTaskStore {
    /// Open or create the task store.
    pub fn open(db_path: &Path) -> Result<Self, String> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| format!("Failed to open task store: {}", e))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = ON;",
        )
        .map_err(|e| format!("Failed to set pragmas: {}", e))?;
        // Create table if migration hasn't run (standalone mode)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS user_tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'medium',
                due_date TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );",
        )
        .map_err(|e| format!("Failed to create table: {}", e))?;
        Ok(Self { conn })
    }

    /// Create a new task.
    pub fn create(&self, task: &UserTask) -> Result<(), String> {
        let tags_json = serde_json::to_string(&task.tags)
            .map_err(|e| format!("Failed to serialize tags: {}", e))?;
        self.conn
            .execute(
                "INSERT INTO user_tasks (id, title, description, status, priority, due_date, tags, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    task.id,
                    task.title,
                    task.description,
                    task.status.as_str(),
                    task.priority.as_str(),
                    task.due_date,
                    tags_json,
                    task.created_at,
                    task.updated_at,
                ],
            )
            .map_err(|e| format!("Failed to create task: {}", e))?;
        Ok(())
    }

    /// Get a task by ID (excludes soft-deleted).
    pub fn get(&self, id: &str) -> Result<Option<UserTask>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, title, description, status, priority, due_date, tags, created_at, updated_at FROM user_tasks WHERE id = ?1 AND deleted_at IS NULL")
            .map_err(|e| format!("Prepare error: {}", e))?;
        let mut rows = stmt
            .query_map(rusqlite::params![id], |row| Ok(row_to_task(row)))
            .map_err(|e| format!("Query error: {}", e))?;
        match rows.next() {
            Some(Ok(task)) => Ok(Some(task)),
            _ => Ok(None),
        }
    }

    /// List tasks with filters (excludes soft-deleted).
    pub fn list(&self, filters: &TaskFilters) -> Result<Vec<UserTask>, String> {
        let mut conditions = vec!["deleted_at IS NULL".to_string()];
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(ref status) = filters.status {
            params.push(Box::new(status.as_str().to_string()));
            conditions.push(format!("status = ?{}", params.len()));
        }
        if let Some(ref priority) = filters.priority {
            params.push(Box::new(priority.as_str().to_string()));
            conditions.push(format!("priority = ?{}", params.len()));
        }
        if let Some(ref tag) = filters.tag {
            let pattern = format!("%\"{}\"%", tag.to_lowercase());
            params.push(Box::new(pattern));
            conditions.push(format!("tags LIKE ?{}", params.len()));
        }

        let order_clause = match filters.sort_by {
            TaskSortField::DueDate => "due_date",
            TaskSortField::Priority => "CASE priority WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END",
            TaskSortField::Created => "created_at",
        };
        let dir = match filters.sort_order {
            SortOrder::Asc => "ASC",
            SortOrder::Desc => "DESC",
        };

        let limit = filters.limit.min(MAX_RESULTS);
        let sql = format!(
            "SELECT id, title, description, status, priority, due_date, tags, created_at, updated_at FROM user_tasks WHERE {} ORDER BY {} {} LIMIT {}",
            conditions.join(" AND "),
            order_clause,
            dir,
            limit,
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| format!("Prepare error: {}", e))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| Ok(row_to_task(row)))
            .map_err(|e| format!("Query error: {}", e))?;

        let mut tasks = Vec::new();
        for task in rows.flatten() {
            tasks.push(task);
        }
        Ok(tasks)
    }

    /// Update a task's fields.
    pub fn update(&self, id: &str, updates: &TaskUpdates) -> Result<UserTask, String> {
        // Check task exists and is not deleted
        if self.get(id)?.is_none() {
            return Err(format!("Task not found: {}", id));
        }

        let now = now_rfc3339();
        let mut sets = vec!["updated_at = ?1".to_string()];
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(now)];

        if let Some(ref title) = updates.title {
            validate_title(title)?;
            params.push(Box::new(title.trim().to_string()));
            sets.push(format!("title = ?{}", params.len()));
        }
        if let Some(ref desc) = updates.description {
            if desc.len() > MAX_DESC_LEN {
                return Err(format!("Description too long (max {} chars)", MAX_DESC_LEN));
            }
            params.push(Box::new(desc.clone()));
            sets.push(format!("description = ?{}", params.len()));
        }
        if let Some(ref status) = updates.status {
            params.push(Box::new(status.as_str().to_string()));
            sets.push(format!("status = ?{}", params.len()));
        }
        if let Some(ref priority) = updates.priority {
            params.push(Box::new(priority.as_str().to_string()));
            sets.push(format!("priority = ?{}", params.len()));
        }
        if let Some(ref due_date_opt) = updates.due_date {
            match due_date_opt {
                Some(d) => {
                    validate_due_date(d)?;
                    params.push(Box::new(d.clone()));
                }
                None => {
                    params.push(Box::new(rusqlite::types::Null));
                }
            }
            sets.push(format!("due_date = ?{}", params.len()));
        }
        if let Some(ref tags) = updates.tags {
            let tags_json = serde_json::to_string(tags)
                .map_err(|e| format!("Failed to serialize tags: {}", e))?;
            params.push(Box::new(tags_json));
            sets.push(format!("tags = ?{}", params.len()));
        }

        params.push(Box::new(id.to_string()));
        let sql = format!(
            "UPDATE user_tasks SET {} WHERE id = ?{} AND deleted_at IS NULL",
            sets.join(", "),
            params.len(),
        );

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        self.conn
            .execute(&sql, param_refs.as_slice())
            .map_err(|e| format!("Update error: {}", e))?;

        self.get(id)?
            .ok_or_else(|| format!("Task {} disappeared after update", id))
    }

    /// Soft-delete a task (can be rolled back within retention period).
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        if self.get(id)?.is_none() {
            return Err(format!("Task not found: {}", id));
        }
        let now = now_rfc3339();
        self.conn
            .execute(
                "UPDATE user_tasks SET deleted_at = ?1 WHERE id = ?2 AND deleted_at IS NULL",
                rusqlite::params![now, id],
            )
            .map_err(|e| format!("Delete error: {}", e))?;
        Ok(true)
    }

    /// Rollback (undelete) a soft-deleted task.
    pub fn rollback_delete(&self, id: &str) -> Result<Option<UserTask>, String> {
        let affected = self
            .conn
            .execute(
                "UPDATE user_tasks SET deleted_at = NULL, updated_at = ?1 WHERE id = ?2 AND deleted_at IS NOT NULL",
                rusqlite::params![now_rfc3339(), id],
            )
            .map_err(|e| format!("Rollback error: {}", e))?;
        if affected == 0 {
            return Err(format!("Task not found or not deleted: {}", id));
        }
        self.get(id)
    }

    /// Purge soft-deleted tasks older than the retention period.
    /// Call this periodically to prevent unbounded storage growth.
    pub fn purge_expired(&self) -> Result<usize, String> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(ROLLBACK_RETENTION_DAYS as i64);
        let cutoff_str = cutoff.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let count = self
            .conn
            .execute(
                "DELETE FROM user_tasks WHERE deleted_at IS NOT NULL AND deleted_at < ?1",
                rusqlite::params![cutoff_str],
            )
            .map_err(|e| format!("Purge error: {}", e))?;
        Ok(count)
    }

    /// Full-text search across title and description.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<UserTask>, String> {
        let limit = limit.min(MAX_RESULTS);
        // Wrap in double quotes for phrase search, escape internal quotes
        let safe_query = query.replace('"', "");
        if safe_query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = format!("\"{}\"", safe_query);

        // Check if FTS5 table exists (may not in standalone mode)
        let has_fts: bool = self
            .conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_tasks_fts'")
            .and_then(|mut stmt| stmt.exists(rusqlite::params![]))
            .unwrap_or(false);

        if has_fts {
            let sql = "SELECT ut.id, ut.title, ut.description, ut.status, ut.priority, ut.due_date, ut.tags, ut.created_at, ut.updated_at
                       FROM user_tasks_fts fts
                       JOIN user_tasks ut ON ut.rowid = fts.rowid
                       WHERE user_tasks_fts MATCH ?1 AND ut.deleted_at IS NULL
                       LIMIT ?2";
            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| format!("FTS prepare error: {}", e))?;
            let rows = stmt
                .query_map(rusqlite::params![fts_query, limit as i64], |row| {
                    Ok(row_to_task(row))
                })
                .map_err(|e| format!("FTS query error: {}", e))?;
            let mut tasks = Vec::new();
            for task in rows.flatten() {
                tasks.push(task);
            }
            Ok(tasks)
        } else {
            // Fallback: LIKE search
            let pattern = format!("%{}%", safe_query);
            let sql = "SELECT id, title, description, status, priority, due_date, tags, created_at, updated_at
                       FROM user_tasks
                       WHERE deleted_at IS NULL AND (title LIKE ?1 OR description LIKE ?1)
                       LIMIT ?2";
            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| format!("Search prepare error: {}", e))?;
            let rows = stmt
                .query_map(rusqlite::params![pattern, limit as i64], |row| {
                    Ok(row_to_task(row))
                })
                .map_err(|e| format!("Search error: {}", e))?;
            let mut tasks = Vec::new();
            for task in rows.flatten() {
                tasks.push(task);
            }
            Ok(tasks)
        }
    }
}

fn row_to_task(row: &rusqlite::Row) -> UserTask {
    let tags_json: String = row.get(6).unwrap_or_default();
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    let status_str: String = row.get(3).unwrap_or_default();
    let priority_str: String = row.get(4).unwrap_or_default();
    UserTask {
        id: row.get(0).unwrap_or_default(),
        title: row.get(1).unwrap_or_default(),
        description: row.get(2).unwrap_or_default(),
        status: TaskStatus::from_str(&status_str).unwrap_or(TaskStatus::Pending),
        priority: TaskPriority::from_str(&priority_str).unwrap_or(TaskPriority::Medium),
        due_date: row.get(5).ok(),
        tags,
        created_at: row.get(7).unwrap_or_default(),
        updated_at: row.get(8).unwrap_or_default(),
    }
}

// ============================================================================
// MCP Tool Registration
// ============================================================================

/// Register all 7 user task management MCP tools on the server.
pub fn register_task_tools(server: &mut McpServer, store: Arc<Mutex<UserTaskStore>>) {
    // --- task_create ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_create",
                "Create a new user task with optional priority, due date, and tags.",
            )
            .with_property("title", "string", "Task title (required)", true)
            .with_property("description", "string", "Task description", false)
            .with_property(
                "priority",
                "string",
                "Priority: low, medium, high, critical (default: medium)",
                false,
            )
            .with_property("due_date", "string", "Due date in YYYY-MM-DD format", false)
            .with_property("tags", "array", "List of tag strings", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Create Task".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(false),
                open_world_hint: Some(false),
            }),
            move |args| {
                let title = args
                    .get("title")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: title")?;
                validate_title(title)?;

                let description = args
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if description.len() > MAX_DESC_LEN {
                    return Err(format!("Description too long (max {} chars)", MAX_DESC_LEN));
                }

                let priority = match args.get("priority").and_then(|v| v.as_str()) {
                    Some(p) => TaskPriority::from_str(p).ok_or_else(|| {
                        format!(
                            "Invalid priority '{}': expected low/medium/high/critical",
                            p
                        )
                    })?,
                    None => TaskPriority::Medium,
                };

                let due_date = match args.get("due_date").and_then(|v| v.as_str()) {
                    Some(d) => {
                        validate_due_date(d)?;
                        Some(d.to_string())
                    }
                    None => None,
                };

                let tags = match args.get("tags").and_then(|v| v.as_array()) {
                    Some(arr) => sanitize_tags(arr)?,
                    None => Vec::new(),
                };

                let now = now_rfc3339();
                let task = UserTask {
                    id: new_uuid(),
                    title: title.trim().to_string(),
                    description: description.to_string(),
                    status: TaskStatus::Pending,
                    priority,
                    due_date,
                    tags,
                    created_at: now.clone(),
                    updated_at: now,
                };

                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.create(&task)?;

                // Auto-purge expired soft-deleted tasks
                let _ = guard.purge_expired();

                Ok(serde_json::json!({
                    "id": task.id,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "created_at": task.created_at,
                }))
            },
        );
    }

    // --- task_list ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_list",
                "List user tasks with optional filtering and sorting.",
            )
            .with_property(
                "status",
                "string",
                "Filter by status: pending, in_progress, done",
                false,
            )
            .with_property(
                "priority",
                "string",
                "Filter by priority: low, medium, high, critical",
                false,
            )
            .with_property("tag", "string", "Filter by tag", false)
            .with_property(
                "sort_by",
                "string",
                "Sort by: due_date, priority, created (default: created)",
                false,
            )
            .with_property(
                "sort_order",
                "string",
                "Sort order: asc, desc (default: desc)",
                false,
            )
            .with_property(
                "limit",
                "integer",
                "Max results (default: 50, max: 200)",
                false,
            )
            .with_annotations(McpToolAnnotation {
                title: Some("List Tasks".into()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let status = args
                    .get("status")
                    .and_then(|v| v.as_str())
                    .and_then(TaskStatus::from_str);
                let priority = args
                    .get("priority")
                    .and_then(|v| v.as_str())
                    .and_then(TaskPriority::from_str);
                let tag = args
                    .get("tag")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let sort_by = match args.get("sort_by").and_then(|v| v.as_str()) {
                    Some("due_date") => TaskSortField::DueDate,
                    Some("priority") => TaskSortField::Priority,
                    _ => TaskSortField::Created,
                };
                let sort_order = match args.get("sort_order").and_then(|v| v.as_str()) {
                    Some("asc") => SortOrder::Asc,
                    _ => SortOrder::Desc,
                };
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

                let filters = TaskFilters {
                    status,
                    priority,
                    tag,
                    sort_by,
                    sort_order,
                    limit,
                };
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                let tasks = guard.list(&filters)?;

                Ok(serde_json::json!({
                    "tasks": tasks,
                    "count": tasks.len(),
                }))
            },
        );
    }

    // --- task_get ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new("task_get", "Get a specific task by ID.")
                .with_property("id", "string", "Task ID (required)", true)
                .with_annotations(McpToolAnnotation {
                    title: Some("Get Task".into()),
                    read_only_hint: Some(true),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |args| {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: id")?;
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                match guard.get(id)? {
                    Some(task) => Ok(serde_json::to_value(&task).map_err(|e| e.to_string())?),
                    None => Err(format!("Task not found: {}", id)),
                }
            },
        );
    }

    // --- task_update ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_update",
                "Update a task's fields. Only provided fields are changed.",
            )
            .with_property("id", "string", "Task ID (required)", true)
            .with_property("title", "string", "New title", false)
            .with_property("description", "string", "New description", false)
            .with_property(
                "status",
                "string",
                "New status: pending, in_progress, done",
                false,
            )
            .with_property(
                "priority",
                "string",
                "New priority: low, medium, high, critical",
                false,
            )
            .with_property(
                "due_date",
                "string",
                "New due date (YYYY-MM-DD) or null to clear",
                false,
            )
            .with_property("tags", "array", "New tags list (replaces existing)", false)
            .with_annotations(McpToolAnnotation {
                title: Some("Update Task".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: id")?;

                let title = args
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let description = args
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let status = args
                    .get("status")
                    .and_then(|v| v.as_str())
                    .and_then(TaskStatus::from_str);
                let priority = args
                    .get("priority")
                    .and_then(|v| v.as_str())
                    .and_then(TaskPriority::from_str);

                let due_date = if args.get("due_date").map(|v| v.is_null()).unwrap_or(false) {
                    Some(None) // clear
                } else {
                    args.get("due_date")
                        .and_then(|v| v.as_str())
                        .map(|s| Some(s.to_string()))
                };

                let tags = match args.get("tags").and_then(|v| v.as_array()) {
                    Some(arr) => Some(sanitize_tags(arr)?),
                    None => None,
                };

                let updates = TaskUpdates {
                    title,
                    description,
                    status,
                    priority,
                    due_date,
                    tags,
                };
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                let task = guard.update(id, &updates)?;
                serde_json::to_value(&task).map_err(|e| e.to_string())
            },
        );
    }

    // --- task_complete ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new("task_complete", "Mark a task as done.")
                .with_property("id", "string", "Task ID (required)", true)
                .with_annotations(McpToolAnnotation {
                    title: Some("Complete Task".into()),
                    read_only_hint: Some(false),
                    destructive_hint: Some(false),
                    idempotent_hint: Some(true),
                    open_world_hint: Some(false),
                }),
            move |args| {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: id")?;
                let updates = TaskUpdates {
                    title: None,
                    description: None,
                    status: Some(TaskStatus::Done),
                    priority: None,
                    due_date: None,
                    tags: None,
                };
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                let task = guard.update(id, &updates)?;
                Ok(serde_json::json!({ "id": task.id, "status": "done" }))
            },
        );
    }

    // --- task_delete (soft-delete with rollback) ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_delete",
                "Soft-delete a task. Can be rolled back within 30 days via task_undelete.",
            )
            .with_property("id", "string", "Task ID (required)", true)
            .with_annotations(McpToolAnnotation {
                title: Some("Delete Task".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(true),
                idempotent_hint: Some(false),
                open_world_hint: Some(false),
            }),
            move |args| {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: id")?;
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                guard.delete(id)?;
                Ok(serde_json::json!({
                    "deleted": true,
                    "id": id,
                    "rollback_available": true,
                    "rollback_expires_days": ROLLBACK_RETENTION_DAYS,
                }))
            },
        );
    }

    // --- task_undelete (rollback) ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_undelete",
                "Restore a soft-deleted task (rollback delete).",
            )
            .with_property("id", "string", "Task ID (required)", true)
            .with_annotations(McpToolAnnotation {
                title: Some("Undelete Task".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: id")?;
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                match guard.rollback_delete(id)? {
                    Some(task) => Ok(serde_json::json!({
                        "restored": true,
                        "task": task,
                    })),
                    None => Err(format!("Task not found: {}", id)),
                }
            },
        );
    }

    // --- task_search ---
    {
        let store = store.clone();
        server.register_tool(
            McpTool::new(
                "task_search",
                "Full-text search across task titles and descriptions.",
            )
            .with_property("query", "string", "Search query (required)", true)
            .with_property(
                "limit",
                "integer",
                "Max results (default: 20, max: 200)",
                false,
            )
            .with_annotations(McpToolAnnotation {
                title: Some("Search Tasks".into()),
                read_only_hint: Some(true),
                destructive_hint: Some(false),
                idempotent_hint: Some(true),
                open_world_hint: Some(false),
            }),
            move |args| {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: query")?;
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let guard = store.lock().map_err(|e| format!("Lock error: {}", e))?;
                let tasks = guard.search(query, limit)?;
                Ok(serde_json::json!({
                    "tasks": tasks,
                    "count": tasks.len(),
                    "query": query,
                }))
            },
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> UserTaskStore {
        UserTaskStore::open(Path::new(":memory:")).expect("open in-memory")
    }

    fn sample_task(title: &str) -> UserTask {
        let now = now_rfc3339();
        UserTask {
            id: new_uuid(),
            title: title.into(),
            description: String::new(),
            status: TaskStatus::Pending,
            priority: TaskPriority::Medium,
            due_date: None,
            tags: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    #[test]
    fn test_create_and_get() {
        let store = test_store();
        let task = sample_task("Buy groceries");
        store.create(&task).expect("create");
        let fetched = store.get(&task.id).expect("get").expect("found");
        assert_eq!(fetched.title, "Buy groceries");
        assert_eq!(fetched.status, TaskStatus::Pending);
        assert_eq!(fetched.priority, TaskPriority::Medium);
    }

    #[test]
    fn test_create_missing_title_validation() {
        let result = validate_title("");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_list_empty() {
        let store = test_store();
        let tasks = store.list(&TaskFilters::default()).expect("list");
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_list_filter_by_status() {
        let store = test_store();
        let mut t1 = sample_task("Task A");
        t1.status = TaskStatus::Done;
        let t2 = sample_task("Task B"); // Pending
        store.create(&t1).expect("create");
        store.create(&t2).expect("create");

        let filters = TaskFilters {
            status: Some(TaskStatus::Pending),
            ..Default::default()
        };
        let tasks = store.list(&filters).expect("list");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].title, "Task B");
    }

    #[test]
    fn test_list_sort_by_priority() {
        let store = test_store();
        let mut t1 = sample_task("Low prio");
        t1.priority = TaskPriority::Low;
        let mut t2 = sample_task("Critical prio");
        t2.priority = TaskPriority::Critical;
        let mut t3 = sample_task("High prio");
        t3.priority = TaskPriority::High;
        store.create(&t1).expect("create");
        store.create(&t2).expect("create");
        store.create(&t3).expect("create");

        let filters = TaskFilters {
            sort_by: TaskSortField::Priority,
            sort_order: SortOrder::Desc,
            ..Default::default()
        };
        let tasks = store.list(&filters).expect("list");
        assert_eq!(tasks[0].title, "Critical prio");
        assert_eq!(tasks[1].title, "High prio");
        assert_eq!(tasks[2].title, "Low prio");
    }

    #[test]
    fn test_update_fields() {
        let store = test_store();
        let task = sample_task("Original");
        store.create(&task).expect("create");

        let updates = TaskUpdates {
            title: Some("Updated".into()),
            priority: Some(TaskPriority::High),
            description: None,
            status: None,
            due_date: None,
            tags: None,
        };
        let updated = store.update(&task.id, &updates).expect("update");
        assert_eq!(updated.title, "Updated");
        assert_eq!(updated.priority, TaskPriority::High);
        assert!(updated.updated_at >= task.updated_at);
    }

    #[test]
    fn test_complete_task() {
        let store = test_store();
        let task = sample_task("To complete");
        store.create(&task).expect("create");

        let updates = TaskUpdates {
            status: Some(TaskStatus::Done),
            title: None,
            description: None,
            priority: None,
            due_date: None,
            tags: None,
        };
        let completed = store.update(&task.id, &updates).expect("update");
        assert_eq!(completed.status, TaskStatus::Done);
    }

    #[test]
    fn test_soft_delete_and_rollback() {
        let store = test_store();
        let task = sample_task("Deletable");
        store.create(&task).expect("create");

        // Delete
        assert!(store.delete(&task.id).expect("delete"));
        assert!(store.get(&task.id).expect("get").is_none()); // Not visible

        // Rollback
        let restored = store
            .rollback_delete(&task.id)
            .expect("rollback")
            .expect("restored");
        assert_eq!(restored.title, "Deletable");
        assert!(store.get(&task.id).expect("get").is_some()); // Visible again
    }

    #[test]
    fn test_delete_not_found() {
        let store = test_store();
        let result = store.delete("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_search_like_fallback() {
        let store = test_store();
        let t1 = sample_task("Deploy the application");
        let t2 = sample_task("Review code changes");
        store.create(&t1).expect("create");
        store.create(&t2).expect("create");

        // FTS5 not available in :memory: without migration, falls back to LIKE
        let results = store.search("deploy", 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Deploy the application");
    }

    #[test]
    fn test_tag_sanitization() {
        let tags = vec![
            serde_json::json!("  Work  "),
            serde_json::json!("URGENT"),
            serde_json::json!("work"), // duplicate after lowercase
            serde_json::json!(""),     // empty
        ];
        let result = sanitize_tags(&tags).expect("sanitize");
        assert_eq!(result, vec!["work", "urgent"]);
    }

    #[test]
    fn test_date_validation() {
        assert!(validate_due_date("2026-04-15").is_ok());
        assert!(validate_due_date("not-a-date").is_err());
        assert!(validate_due_date("2026-13-01").is_err());
    }
}
