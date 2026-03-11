//! Google Agent-to-Agent (A2A) Protocol implementation
//!
//! This module implements the Google A2A protocol for inter-agent communication,
//! including agent discovery via AgentCards, task lifecycle management, JSON-RPC 2.0
//! transport, push notifications, and an agent directory for skill-based lookup.
//!
//! Gated behind the `a2a` feature flag.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{A2AError, AiError};

// =============================================================================
// Agent Card (discovery)
// =============================================================================

/// An Agent Card describes an agent's capabilities, skills, and endpoint,
/// following the Google A2A `.well-known/agent.json` convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub url: String,
    pub version: String,
    pub skills: Vec<AgentSkill>,
    pub auth_schemes: Vec<String>,
    pub capabilities: HashMap<String, bool>,
}

/// A skill that an agent advertises in its card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
}

impl AgentCard {
    /// Create a new AgentCard with the given name, description, and URL.
    /// Defaults to version "1.0.0", no skills, no auth, no capabilities.
    pub fn new(name: impl Into<String>, description: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            url: url.into(),
            version: "1.0.0".to_string(),
            skills: Vec::new(),
            auth_schemes: Vec::new(),
            capabilities: HashMap::new(),
        }
    }

    /// Add a skill to this agent card (builder pattern).
    pub fn with_skill(mut self, skill: AgentSkill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Set the version string (builder pattern).
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Add an authentication scheme (builder pattern).
    pub fn with_auth_scheme(mut self, scheme: impl Into<String>) -> Self {
        self.auth_schemes.push(scheme.into());
        self
    }

    /// Add a capability flag (builder pattern).
    pub fn with_capability(mut self, key: impl Into<String>, value: bool) -> Self {
        self.capabilities.insert(key.into(), value);
        self
    }
}

impl AgentSkill {
    /// Create a new AgentSkill with the given name, description, and tags.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        tags: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tags,
        }
    }
}

// =============================================================================
// Task Status lifecycle
// =============================================================================

/// Task status in the A2A lifecycle:
/// Submitted -> Working -> (InputRequired ->) Completed | Failed | Canceled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum A2ATaskStatus {
    Submitted,
    Working,
    InputRequired,
    Completed,
    Failed,
    Canceled,
}

impl fmt::Display for A2ATaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            A2ATaskStatus::Submitted => "submitted",
            A2ATaskStatus::Working => "working",
            A2ATaskStatus::InputRequired => "input-required",
            A2ATaskStatus::Completed => "completed",
            A2ATaskStatus::Failed => "failed",
            A2ATaskStatus::Canceled => "canceled",
        };
        write!(f, "{}", s)
    }
}

impl A2ATaskStatus {
    /// Returns true if the status is a terminal state (no further transitions allowed).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            A2ATaskStatus::Completed | A2ATaskStatus::Failed | A2ATaskStatus::Canceled
        )
    }

    /// Checks whether transitioning from `self` to `target` is valid.
    pub fn can_transition_to(self, target: A2ATaskStatus) -> bool {
        match self {
            A2ATaskStatus::Submitted => matches!(
                target,
                A2ATaskStatus::Working | A2ATaskStatus::Failed | A2ATaskStatus::Canceled
            ),
            A2ATaskStatus::Working => matches!(
                target,
                A2ATaskStatus::InputRequired
                    | A2ATaskStatus::Completed
                    | A2ATaskStatus::Failed
                    | A2ATaskStatus::Canceled
            ),
            A2ATaskStatus::InputRequired => matches!(
                target,
                A2ATaskStatus::Working | A2ATaskStatus::Failed | A2ATaskStatus::Canceled
            ),
            // Terminal states cannot transition further
            A2ATaskStatus::Completed | A2ATaskStatus::Failed | A2ATaskStatus::Canceled => false,
        }
    }
}

// =============================================================================
// A2A Task
// =============================================================================

/// Record of a status change in a task's lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusUpdate {
    pub status: A2ATaskStatus,
    pub message: Option<A2AMessage>,
    pub timestamp: u64,
}

/// An A2A Task represents a unit of work sent to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ATask {
    pub id: String,
    pub status: A2ATaskStatus,
    pub messages: Vec<A2AMessage>,
    pub artifacts: Vec<A2AArtifact>,
    pub history: Vec<TaskStatusUpdate>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl A2ATask {
    /// Create a new task with a random UUID and Submitted status.
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = current_timestamp();
        Self {
            id,
            status: A2ATaskStatus::Submitted,
            messages: Vec::new(),
            artifacts: Vec::new(),
            history: vec![TaskStatusUpdate {
                status: A2ATaskStatus::Submitted,
                message: None,
                timestamp: now,
            }],
            metadata: HashMap::new(),
        }
    }

    /// Attempt a state transition. Returns an error if the transition is invalid.
    pub fn transition(&mut self, new_status: A2ATaskStatus) -> Result<(), AiError> {
        if !self.status.can_transition_to(new_status) {
            return Err(A2AError::InvalidState {
                task_id: self.id.clone(),
                current: self.status.to_string(),
                attempted: new_status.to_string(),
            }
            .into());
        }
        self.status = new_status;
        self.history.push(TaskStatusUpdate {
            status: new_status,
            message: None,
            timestamp: current_timestamp(),
        });
        Ok(())
    }

    /// Add a message to the task.
    pub fn add_message(&mut self, msg: A2AMessage) {
        self.messages.push(msg);
    }

    /// Add an artifact to the task.
    pub fn add_artifact(&mut self, artifact: A2AArtifact) {
        self.artifacts.push(artifact);
    }
}

// =============================================================================
// Messages and Parts
// =============================================================================

/// A message in the A2A protocol, with role and typed parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AMessage {
    pub role: MessageRole,
    pub parts: Vec<A2APart>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Who sent the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Agent,
}

/// A typed part of an A2A message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum A2APart {
    Text(TextPart),
    File(FilePart),
    Data(DataPart),
}

/// Text content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

/// File reference part (inline data or URI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePart {
    pub name: String,
    pub mime_type: String,
    pub data: Option<String>,
    pub uri: Option<String>,
}

/// Structured data part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPart {
    pub mime_type: String,
    pub data: serde_json::Value,
}

impl A2AMessage {
    /// Create a text-only message.
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            parts: vec![A2APart::Text(TextPart { text: text.into() })],
            metadata: HashMap::new(),
        }
    }

    /// Create a message with the given role and parts.
    pub fn with_parts(role: MessageRole, parts: Vec<A2APart>) -> Self {
        Self {
            role,
            parts,
            metadata: HashMap::new(),
        }
    }
}

// =============================================================================
// Artifact
// =============================================================================

/// An artifact produced by an agent as output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AArtifact {
    pub name: String,
    pub description: Option<String>,
    pub parts: Vec<A2APart>,
    pub index: usize,
}

impl A2AArtifact {
    /// Create a new artifact with the given name and index.
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            description: None,
            parts: Vec::new(),
            index,
        }
    }

    /// Set the description (builder pattern).
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a part (builder pattern).
    pub fn with_part(mut self, part: A2APart) -> Self {
        self.parts.push(part);
        self
    }
}

// =============================================================================
// JSON-RPC 2.0
// =============================================================================

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<serde_json::Value>,
    pub id: serde_json::Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
    pub id: serde_json::Value,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

// Standard JSON-RPC 2.0 error codes
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

/// Task not found (application-level)
pub const TASK_NOT_FOUND: i32 = -32001;
/// Invalid task state transition (application-level)
pub const INVALID_STATE: i32 = -32002;
/// Sender agent not authorized (application-level)
pub const UNAUTHORIZED_AGENT: i32 = -32003;

impl JsonRpcRequest {
    /// Validate that the request is well-formed JSON-RPC 2.0.
    pub fn validate(&self) -> Result<(), JsonRpcError> {
        if self.jsonrpc != "2.0" {
            return Err(JsonRpcError {
                code: INVALID_REQUEST,
                message: "Invalid JSON-RPC version, expected \"2.0\"".to_string(),
                data: None,
            });
        }
        if self.method.is_empty() {
            return Err(JsonRpcError {
                code: INVALID_REQUEST,
                message: "Method must not be empty".to_string(),
                data: None,
            });
        }
        Ok(())
    }
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response.
    pub fn error(id: serde_json::Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

// =============================================================================
// Push Notifications
// =============================================================================

/// A push notification sent when a task's status changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushNotification {
    pub task_id: String,
    pub status: A2ATaskStatus,
    pub message: Option<A2AMessage>,
    pub timestamp: u64,
}

/// Configuration for push notification delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushNotificationConfig {
    pub callback_url: String,
    pub events: Vec<A2ATaskStatus>,
    pub auth_token: Option<String>,
}

// =============================================================================
// Task Handler callback
// =============================================================================

/// Callback that processes an incoming A2A message and returns a response message.
pub type TaskHandler = Box<dyn Fn(&A2AMessage) -> Result<A2AMessage, AiError> + Send + Sync>;

// =============================================================================
// A2A Server
// =============================================================================

/// An A2A-protocol server that hosts an agent and processes JSON-RPC requests.
pub struct A2AServer {
    card: AgentCard,
    handler: TaskHandler,
    pub tasks: Mutex<HashMap<String, A2ATask>>,
    push_configs: Mutex<HashMap<String, PushNotificationConfig>>,
    /// Allowlist of agent identifiers permitted to send tasks. If non-empty,
    /// only agents whose `sender_agent` param matches an entry are accepted.
    /// An empty set means all agents are allowed (open mode).
    allowed_agents: Mutex<HashSet<String>>,
}

impl fmt::Debug for A2AServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("A2AServer")
            .field("card", &self.card)
            .field("handler", &"<...>")
            .field("tasks", &self.tasks)
            .field("push_configs", &self.push_configs)
            .field("allowed_agents", &self.allowed_agents)
            .finish()
    }
}

impl A2AServer {
    /// Create a new A2AServer with the given agent card and task handler.
    pub fn new(card: AgentCard, handler: TaskHandler) -> Self {
        Self {
            card,
            handler,
            tasks: Mutex::new(HashMap::new()),
            push_configs: Mutex::new(HashMap::new()),
            allowed_agents: Mutex::new(HashSet::new()),
        }
    }

    /// Set the allowlist of agent identifiers permitted to send tasks.
    /// When non-empty, only `tasks/send` requests whose `sender_agent` param
    /// matches an entry in this set will be accepted; all others are rejected
    /// with `UNAUTHORIZED_AGENT`. An empty set disables the check (open mode).
    pub fn set_allowed_agents(&self, agents: HashSet<String>) {
        if let Ok(mut allowed) = self.allowed_agents.lock() {
            *allowed = agents;
        }
    }

    /// Add a single agent identifier to the allowlist.
    pub fn allow_agent(&self, agent_id: impl Into<String>) {
        if let Ok(mut allowed) = self.allowed_agents.lock() {
            allowed.insert(agent_id.into());
        }
    }

    /// Return a reference to the agent card.
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Register a push notification configuration for a task.
    pub fn register_push(
        &self,
        task_id: impl Into<String>,
        config: PushNotificationConfig,
    ) -> Result<(), AiError> {
        let task_id = task_id.into();
        // Verify the task exists
        {
            let tasks = self.tasks.lock().map_err(|e| {
                AiError::Other(format!("Failed to lock tasks: {}", e))
            })?;
            if !tasks.contains_key(&task_id) {
                return Err(A2AError::TaskNotFound {
                    task_id: task_id.clone(),
                }
                .into());
            }
        }
        let mut push = self.push_configs.lock().map_err(|e| {
            AiError::Other(format!("Failed to lock push_configs: {}", e))
        })?;
        push.insert(task_id, config);
        Ok(())
    }

    /// Get push notification config for a task (if registered).
    pub fn get_push_config(&self, task_id: &str) -> Option<PushNotificationConfig> {
        let push = self.push_configs.lock().ok()?;
        push.get(task_id).cloned()
    }

    /// Handle a raw JSON-RPC request string and return a JSON-RPC response string.
    pub fn handle_request(&self, json_str: &str) -> String {
        // Parse the JSON
        let request: JsonRpcRequest = match serde_json::from_str(json_str) {
            Ok(req) => req,
            Err(_) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    JsonRpcError {
                        code: PARSE_ERROR,
                        message: "Parse error: invalid JSON".to_string(),
                        data: None,
                    },
                );
                return serde_json::to_string(&resp).unwrap_or_default();
            }
        };

        // Validate JSON-RPC 2.0 envelope
        if let Err(err) = request.validate() {
            let resp = JsonRpcResponse::error(request.id.clone(), err);
            return serde_json::to_string(&resp).unwrap_or_default();
        }

        // Dispatch by method
        let response = match request.method.as_str() {
            "agent/card" => self.handle_agent_card(&request),
            "tasks/send" => self.handle_tasks_send(&request),
            "tasks/get" => self.handle_tasks_get(&request),
            "tasks/cancel" => self.handle_tasks_cancel(&request),
            _ => JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: METHOD_NOT_FOUND,
                    message: format!("Method '{}' not found", request.method),
                    data: None,
                },
            ),
        };

        serde_json::to_string(&response).unwrap_or_default()
    }

    fn handle_agent_card(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match serde_json::to_value(&self.card) {
            Ok(val) => JsonRpcResponse::success(request.id.clone(), val),
            Err(e) => JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: INTERNAL_ERROR,
                    message: format!("Serialization error: {}", e),
                    data: None,
                },
            ),
        }
    }

    fn handle_tasks_send(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        // --- Authentication: check sender against allowlist ---
        if let Ok(allowed) = self.allowed_agents.lock() {
            if !allowed.is_empty() {
                let sender = request
                    .params
                    .as_ref()
                    .and_then(|p| p.get("sender_agent"))
                    .and_then(|v| v.as_str());
                match sender {
                    Some(id) if allowed.contains(id) => { /* authorized */ }
                    _ => {
                        return JsonRpcResponse::error(
                            request.id.clone(),
                            JsonRpcError {
                                code: UNAUTHORIZED_AGENT,
                                message: "Sender agent not authorized".to_string(),
                                data: None,
                            },
                        );
                    }
                }
            }
        }

        // Extract the user message from params
        let message = match extract_message_from_params(&request.params) {
            Ok(msg) => msg,
            Err(err_msg) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INVALID_PARAMS,
                        message: err_msg,
                        data: None,
                    },
                );
            }
        };

        // Create a new task
        let mut task = A2ATask::new();
        task.add_message(message.clone());

        // Transition to Working
        if let Err(e) = task.transition(A2ATaskStatus::Working) {
            return JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: INTERNAL_ERROR,
                    message: format!("State transition error: {}", e),
                    data: None,
                },
            );
        }

        // Invoke the handler
        // TODO(security): Task results are not cryptographically signed. A malicious
        // intermediary could tamper with the response. Future work: add HMAC or
        // digital-signature envelope around `task_value` so consumers can verify
        // authenticity and integrity of task results.
        match (self.handler)(&message) {
            Ok(response_msg) => {
                task.add_message(response_msg);
                // Transition to Completed
                if let Err(e) = task.transition(A2ATaskStatus::Completed) {
                    return JsonRpcResponse::error(
                        request.id.clone(),
                        JsonRpcError {
                            code: INTERNAL_ERROR,
                            message: format!("State transition error: {}", e),
                            data: None,
                        },
                    );
                }
                // Store the task
                let task_id = task.id.clone();
                let task_value = match serde_json::to_value(&task) {
                    Ok(v) => v,
                    Err(e) => {
                        return JsonRpcResponse::error(
                            request.id.clone(),
                            JsonRpcError {
                                code: INTERNAL_ERROR,
                                message: format!("Serialization error: {}", e),
                                data: None,
                            },
                        );
                    }
                };
                if let Ok(mut tasks) = self.tasks.lock() {
                    tasks.insert(task_id, task);
                }
                JsonRpcResponse::success(request.id.clone(), task_value)
            }
            Err(e) => {
                // Handler failed — mark the task as Failed and store it
                let _ = task.transition(A2ATaskStatus::Failed);
                if let Ok(mut tasks) = self.tasks.lock() {
                    tasks.insert(task.id.clone(), task);
                }
                JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: format!("Handler error: {}", e),
                        data: None,
                    },
                )
            }
        }
    }

    fn handle_tasks_get(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let task_id = match extract_task_id_from_params(&request.params) {
            Ok(id) => id,
            Err(err_msg) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INVALID_PARAMS,
                        message: err_msg,
                        data: None,
                    },
                );
            }
        };

        let tasks = match self.tasks.lock() {
            Ok(t) => t,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: format!("Lock error: {}", e),
                        data: None,
                    },
                );
            }
        };

        match tasks.get(&task_id) {
            Some(task) => match serde_json::to_value(task) {
                Ok(val) => JsonRpcResponse::success(request.id.clone(), val),
                Err(e) => JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: format!("Serialization error: {}", e),
                        data: None,
                    },
                ),
            },
            None => JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: TASK_NOT_FOUND,
                    message: format!("Task '{}' not found", task_id),
                    data: None,
                },
            ),
        }
    }

    fn handle_tasks_cancel(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let task_id = match extract_task_id_from_params(&request.params) {
            Ok(id) => id,
            Err(err_msg) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INVALID_PARAMS,
                        message: err_msg,
                        data: None,
                    },
                );
            }
        };

        let mut tasks = match self.tasks.lock() {
            Ok(t) => t,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: format!("Lock error: {}", e),
                        data: None,
                    },
                );
            }
        };

        match tasks.get_mut(&task_id) {
            Some(task) => {
                if let Err(_) = task.transition(A2ATaskStatus::Canceled) {
                    return JsonRpcResponse::error(
                        request.id.clone(),
                        JsonRpcError {
                            code: INVALID_STATE,
                            message: format!(
                                "Cannot cancel task '{}' in state '{}'",
                                task_id, task.status
                            ),
                            data: None,
                        },
                    );
                }
                match serde_json::to_value(&*task) {
                    Ok(val) => JsonRpcResponse::success(request.id.clone(), val),
                    Err(e) => JsonRpcResponse::error(
                        request.id.clone(),
                        JsonRpcError {
                            code: INTERNAL_ERROR,
                            message: format!("Serialization error: {}", e),
                            data: None,
                        },
                    ),
                }
            }
            None => JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: TASK_NOT_FOUND,
                    message: format!("Task '{}' not found", task_id),
                    data: None,
                },
            ),
        }
    }
}

// =============================================================================
// A2A Client
// =============================================================================

/// A client for making A2A protocol requests to remote agents.
#[derive(Debug)]
pub struct A2AClient {
    pub base_url: String,
    pub auth_header: Option<(String, String)>,
}

impl A2AClient {
    /// Create a new client targeting the given base URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            base_url: url.into(),
            auth_header: None,
        }
    }

    /// Set an API key or other auth header (builder pattern).
    pub fn with_api_key(
        mut self,
        header: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.auth_header = Some((header.into(), value.into()));
        self
    }

    /// Build a JSON-RPC 2.0 request for the given method and params.
    pub fn build_request(
        &self,
        method: impl Into<String>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
            id: serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
        }
    }
}

// =============================================================================
// Agent Directory
// =============================================================================

/// A directory of known agents, indexed by URL, with skill-based lookup.
#[derive(Debug)]
pub struct AgentDirectory {
    agents: HashMap<String, AgentCard>,
}

impl AgentDirectory {
    /// Create a new empty directory.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register an agent card. Uses the card's URL as the key.
    pub fn register(&mut self, card: AgentCard) {
        self.agents.insert(card.url.clone(), card);
    }

    /// Remove an agent by URL.
    pub fn unregister(&mut self, url: &str) -> bool {
        self.agents.remove(url).is_some()
    }

    /// Find agents whose name contains the given substring (case-insensitive).
    pub fn find_by_name(&self, name: &str) -> Vec<&AgentCard> {
        let lower = name.to_lowercase();
        self.agents
            .values()
            .filter(|card| card.name.to_lowercase().contains(&lower))
            .collect()
    }

    /// Find agents that have at least one skill whose name or tags match the query.
    pub fn find_by_skill(&self, skill_query: &str) -> Vec<&AgentCard> {
        let lower = skill_query.to_lowercase();
        self.agents
            .values()
            .filter(|card| {
                card.skills.iter().any(|s| {
                    s.name.to_lowercase().contains(&lower)
                        || s.tags.iter().any(|t| t.to_lowercase().contains(&lower))
                })
            })
            .collect()
    }

    /// List all registered agent cards.
    pub fn list_all(&self) -> Vec<&AgentCard> {
        self.agents.values().collect()
    }

    /// Get a specific agent card by URL.
    pub fn get(&self, url: &str) -> Option<&AgentCard> {
        self.agents.get(url)
    }

    /// Number of registered agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the directory is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Get the current UNIX timestamp in seconds.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Extract a message from JSON-RPC params: expects `{ "message": { "role": ..., "parts": ..., ... } }`.
fn extract_message_from_params(
    params: &Option<serde_json::Value>,
) -> Result<A2AMessage, String> {
    let params = params
        .as_ref()
        .ok_or_else(|| "Missing params".to_string())?;
    let msg_val = params
        .get("message")
        .ok_or_else(|| "Missing 'message' in params".to_string())?;
    serde_json::from_value(msg_val.clone())
        .map_err(|e| format!("Invalid message: {}", e))
}

/// Extract a task_id from JSON-RPC params: expects `{ "task_id": "..." }`.
fn extract_task_id_from_params(
    params: &Option<serde_json::Value>,
) -> Result<String, String> {
    let params = params
        .as_ref()
        .ok_or_else(|| "Missing params".to_string())?;
    params
        .get("task_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "Missing or invalid 'task_id' in params".to_string())
}

// =============================================================================
// AGENTS.md Convention (v6 Phase 2.2)
// =============================================================================

/// An entry parsed from an AGENTS.md file describing an agent's identity,
/// supported protocols, endpoint, capabilities, and version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsMdEntry {
    pub name: String,
    pub description: String,
    pub protocols: Vec<String>,
    pub endpoint: Option<String>,
    pub capabilities: Vec<String>,
    pub version: Option<String>,
}

/// Parser for AGENTS.md markdown files.
///
/// The expected format is one or more agent entries, each starting with a
/// level-2 heading (`## Agent: <name>` or `## <name>`), followed by bullet
/// lines with field values:
///
/// ```markdown
/// ## Agent: MyAgent
/// - Description: Does useful things
/// - Protocols: A2A, MCP
/// - Endpoint: http://localhost:8080
/// - Capabilities: translate, summarize
/// - Version: 1.2.0
/// ```
#[derive(Debug)]
pub struct AgentsMdParser;

impl AgentsMdParser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        Self
    }

    /// Parse AGENTS.md markdown content into a list of [`AgentsMdEntry`] values.
    ///
    /// Each agent block starts with `## Agent: <name>` or `## <name>`.
    /// Fields are extracted from lines matching `- FieldName: value`.
    /// Missing optional fields (Endpoint, Version) default to `None`;
    /// missing list fields (Protocols, Capabilities) default to empty vectors.
    pub fn parse(&self, content: &str) -> Result<Vec<AgentsMdEntry>, AiError> {
        let mut entries: Vec<AgentsMdEntry> = Vec::new();
        let mut current_name: Option<String> = None;
        let mut description = String::new();
        let mut protocols: Vec<String> = Vec::new();
        let mut endpoint: Option<String> = None;
        let mut capabilities: Vec<String> = Vec::new();
        let mut version: Option<String> = None;

        for line in content.lines() {
            let trimmed = line.trim();

            // Detect heading: ## Agent: Name  or  ## Name
            if trimmed.starts_with("## ") {
                // Flush previous entry if any
                if let Some(name) = current_name.take() {
                    entries.push(AgentsMdEntry {
                        name,
                        description: description.clone(),
                        protocols: protocols.clone(),
                        endpoint: endpoint.take(),
                        capabilities: capabilities.clone(),
                        version: version.take(),
                    });
                    description.clear();
                    protocols.clear();
                    capabilities.clear();
                }

                let heading = trimmed.trim_start_matches("## ").trim();
                let name = if heading.starts_with("Agent:") || heading.starts_with("Agent :") {
                    heading
                        .trim_start_matches("Agent:")
                        .trim_start_matches("Agent :")
                        .trim()
                        .to_string()
                } else {
                    heading.to_string()
                };

                if name.is_empty() {
                    return Err(AiError::Other(
                        "AGENTS.md: empty agent name in heading".to_string(),
                    ));
                }
                current_name = Some(name);
                continue;
            }

            // Only process bullet fields if we are inside an agent block
            if current_name.is_some() && trimmed.starts_with("- ") {
                let field_line = trimmed.trim_start_matches("- ");
                if let Some(val) = strip_field_prefix(field_line, "Description:") {
                    description = val.to_string();
                } else if let Some(val) = strip_field_prefix(field_line, "Protocols:") {
                    protocols = split_comma_list(val);
                } else if let Some(val) = strip_field_prefix(field_line, "Endpoint:") {
                    let v = val.trim().to_string();
                    if !v.is_empty() {
                        endpoint = Some(v);
                    }
                } else if let Some(val) = strip_field_prefix(field_line, "Capabilities:") {
                    capabilities = split_comma_list(val);
                } else if let Some(val) = strip_field_prefix(field_line, "Version:") {
                    let v = val.trim().to_string();
                    if !v.is_empty() {
                        version = Some(v);
                    }
                }
            }
        }

        // Flush the last entry
        if let Some(name) = current_name.take() {
            entries.push(AgentsMdEntry {
                name,
                description,
                protocols,
                endpoint,
                capabilities,
                version,
            });
        }

        Ok(entries)
    }

    /// Serialize a slice of [`AgentsMdEntry`] back to AGENTS.md markdown format.
    pub fn to_markdown(entries: &[AgentsMdEntry]) -> String {
        let mut out = String::new();
        for (i, entry) in entries.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format!("## Agent: {}\n", entry.name));
            out.push_str(&format!("- Description: {}\n", entry.description));
            if !entry.protocols.is_empty() {
                out.push_str(&format!("- Protocols: {}\n", entry.protocols.join(", ")));
            }
            if let Some(ref ep) = entry.endpoint {
                out.push_str(&format!("- Endpoint: {}\n", ep));
            }
            if !entry.capabilities.is_empty() {
                out.push_str(&format!(
                    "- Capabilities: {}\n",
                    entry.capabilities.join(", ")
                ));
            }
            if let Some(ref ver) = entry.version {
                out.push_str(&format!("- Version: {}\n", ver));
            }
        }
        out
    }
}

/// Discovery service built on top of AGENTS.md parsed entries.
#[derive(Debug)]
pub struct AgentsMdDiscovery {
    entries: Vec<AgentsMdEntry>,
}

impl AgentsMdDiscovery {
    /// Create a new empty discovery instance.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Parse AGENTS.md content and add all entries. Returns the number of
    /// entries that were added.
    pub fn load_from_content(&mut self, content: &str) -> Result<usize, AiError> {
        let parser = AgentsMdParser::new();
        let parsed = parser.parse(content)?;
        let count = parsed.len();
        self.entries.extend(parsed);
        Ok(count)
    }

    /// Find all entries that list the given protocol (case-insensitive).
    pub fn find_by_protocol(&self, protocol: &str) -> Vec<&AgentsMdEntry> {
        let lower = protocol.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.protocols.iter().any(|p| p.to_lowercase() == lower))
            .collect()
    }

    /// Find an entry by exact name (case-insensitive).
    pub fn find_by_name(&self, name: &str) -> Option<&AgentsMdEntry> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .find(|e| e.name.to_lowercase() == lower)
    }

    /// Return a slice of all loaded entries.
    pub fn all_entries(&self) -> &[AgentsMdEntry] {
        &self.entries
    }

    /// Return the number of loaded entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Remove all loaded entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Helper: strip a case-insensitive field prefix and return the remainder.
fn strip_field_prefix<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let line_lower = line.to_lowercase();
    let prefix_lower = prefix.to_lowercase();
    if line_lower.starts_with(&prefix_lower) {
        Some(line[prefix.len()..].trim())
    } else {
        None
    }
}

/// Helper: split a comma-separated string into trimmed, non-empty tokens.
fn split_comma_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

// =============================================================================
// ACP Bridge (v6 Phase 2.3)
// =============================================================================

/// A message in the Agent Communication Protocol (ACP) format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpMessage {
    pub run_id: String,
    pub content_parts: Vec<AcpContentPart>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
}

/// A content part within an ACP message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcpContentPart {
    Text(String),
    Data { mime_type: String, data: String },
}

/// Descriptor for an agent in the ACP ecosystem, analogous to an [`AgentCard`]
/// in A2A but with ACP-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpAgentDescriptor {
    pub name: String,
    pub description: String,
    pub url: Option<String>,
    pub protocols: Vec<String>,
    pub capabilities: Vec<String>,
}

/// Bridge that translates between A2A and ACP message formats.
#[derive(Debug)]
pub struct AcpBridge;

impl AcpBridge {
    /// Create a new bridge instance.
    pub fn new() -> Self {
        Self
    }

    /// Convert an A2A message to an ACP message.
    ///
    /// Text parts become [`AcpContentPart::Text`]; File and Data parts become
    /// [`AcpContentPart::Data`] with their MIME type and serialized content.
    /// The `run_id` is generated as a new UUID; the timestamp is the current
    /// UNIX time. A2A metadata entries are converted to string values.
    pub fn a2a_to_acp(message: &A2AMessage) -> AcpMessage {
        let content_parts: Vec<AcpContentPart> = message
            .parts
            .iter()
            .map(|part| match part {
                A2APart::Text(tp) => AcpContentPart::Text(tp.text.clone()),
                A2APart::File(fp) => AcpContentPart::Data {
                    mime_type: fp.mime_type.clone(),
                    data: fp
                        .data
                        .clone()
                        .or_else(|| fp.uri.clone())
                        .unwrap_or_default(),
                },
                A2APart::Data(dp) => AcpContentPart::Data {
                    mime_type: dp.mime_type.clone(),
                    data: dp.data.to_string(),
                },
            })
            .collect();

        let metadata: HashMap<String, String> = message
            .metadata
            .iter()
            .map(|(k, v)| {
                let str_val = match v.as_str() {
                    Some(s) => s.to_string(),
                    None => v.to_string(),
                };
                (k.clone(), str_val)
            })
            .collect();

        AcpMessage {
            run_id: uuid::Uuid::new_v4().to_string(),
            content_parts,
            metadata,
            timestamp: current_timestamp(),
        }
    }

    /// Convert an ACP message to an A2A message.
    ///
    /// ACP Text parts map to [`A2APart::Text`]; ACP Data parts map to
    /// [`A2APart::Data`] with the data stored as a JSON string value.
    /// The resulting A2A message has role [`MessageRole::Agent`] and
    /// metadata converted from string values to JSON string values.
    pub fn acp_to_a2a(message: &AcpMessage) -> A2AMessage {
        let parts: Vec<A2APart> = message
            .content_parts
            .iter()
            .map(|cp| match cp {
                AcpContentPart::Text(text) => A2APart::Text(TextPart {
                    text: text.clone(),
                }),
                AcpContentPart::Data { mime_type, data } => A2APart::Data(DataPart {
                    mime_type: mime_type.clone(),
                    data: serde_json::Value::String(data.clone()),
                }),
            })
            .collect();

        let metadata: HashMap<String, serde_json::Value> = message
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();

        A2AMessage {
            role: MessageRole::Agent,
            parts,
            metadata,
        }
    }

    /// Translate an [`AgentCard`] into an [`AcpAgentDescriptor`].
    ///
    /// Capabilities are extracted from the card's capabilities map (keys where
    /// value is `true`) combined with skill names.
    pub fn translate_card_to_descriptor(card: &AgentCard) -> AcpAgentDescriptor {
        let mut capabilities: Vec<String> = card
            .capabilities
            .iter()
            .filter(|(_, v)| **v)
            .map(|(k, _)| k.clone())
            .collect();

        for skill in &card.skills {
            capabilities.push(skill.name.clone());
        }

        AcpAgentDescriptor {
            name: card.name.clone(),
            description: card.description.clone(),
            url: if card.url.is_empty() {
                None
            } else {
                Some(card.url.clone())
            },
            protocols: vec!["A2A".to_string()],
            capabilities,
        }
    }
}

/// Adapter that wraps an [`AcpAgentDescriptor`] and provides bidirectional
/// conversion to/from [`AgentCard`].
#[derive(Debug)]
pub struct AcpAgentAdapter {
    descriptor: AcpAgentDescriptor,
}

impl AcpAgentAdapter {
    /// Create an adapter from an existing [`AgentCard`].
    pub fn from_agent_card(card: &AgentCard) -> Self {
        Self {
            descriptor: AcpBridge::translate_card_to_descriptor(card),
        }
    }

    /// Return a reference to the underlying descriptor.
    pub fn descriptor(&self) -> &AcpAgentDescriptor {
        &self.descriptor
    }

    /// Convert the descriptor back to an [`AgentCard`].
    ///
    /// Skills are reconstructed from the descriptor's capabilities list.
    /// The URL defaults to an empty string if none is present.
    pub fn to_agent_card(&self) -> AgentCard {
        let skills: Vec<AgentSkill> = self
            .descriptor
            .capabilities
            .iter()
            .map(|cap| AgentSkill::new(cap.clone(), format!("{} capability", cap), vec![]))
            .collect();

        AgentCard {
            name: self.descriptor.name.clone(),
            description: self.descriptor.description.clone(),
            url: self.descriptor.url.clone().unwrap_or_default(),
            version: "1.0.0".to_string(),
            skills,
            auth_schemes: Vec::new(),
            capabilities: HashMap::new(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AgentCard tests ----

    #[test]
    fn test_agent_card_creation() {
        let card = AgentCard::new("TestAgent", "A test agent", "http://localhost:8080");
        assert_eq!(card.name, "TestAgent");
        assert_eq!(card.description, "A test agent");
        assert_eq!(card.url, "http://localhost:8080");
        assert_eq!(card.version, "1.0.0");
        assert!(card.skills.is_empty());
        assert!(card.auth_schemes.is_empty());
        assert!(card.capabilities.is_empty());
    }

    #[test]
    fn test_agent_card_with_skills() {
        let card = AgentCard::new("Translator", "Translation agent", "http://translate.local")
            .with_skill(AgentSkill::new(
                "translate",
                "Translate text",
                vec!["nlp".to_string(), "i18n".to_string()],
            ))
            .with_skill(AgentSkill::new(
                "detect-language",
                "Detect language",
                vec!["nlp".to_string()],
            ));
        assert_eq!(card.skills.len(), 2);
        assert_eq!(card.skills[0].name, "translate");
        assert_eq!(card.skills[1].tags, vec!["nlp".to_string()]);
    }

    #[test]
    fn test_agent_card_with_version() {
        let card = AgentCard::new("Agent", "desc", "http://a.local").with_version("2.1.0");
        assert_eq!(card.version, "2.1.0");
    }

    #[test]
    fn test_agent_card_with_auth_scheme() {
        let card = AgentCard::new("Agent", "desc", "http://a.local")
            .with_auth_scheme("bearer")
            .with_auth_scheme("api-key");
        assert_eq!(card.auth_schemes, vec!["bearer", "api-key"]);
    }

    #[test]
    fn test_agent_card_with_capabilities() {
        let card = AgentCard::new("Agent", "desc", "http://a.local")
            .with_capability("streaming", true)
            .with_capability("push_notifications", false);
        assert_eq!(card.capabilities.get("streaming"), Some(&true));
        assert_eq!(card.capabilities.get("push_notifications"), Some(&false));
    }

    #[test]
    fn test_agent_card_serialization() {
        let card = AgentCard::new("Agent", "desc", "http://a.local")
            .with_skill(AgentSkill::new("skill1", "desc1", vec![]));
        let json = serde_json::to_string(&card).expect("serialize AgentCard in test_agent_card_serialization");
        let parsed: AgentCard = serde_json::from_str(&json).expect("deserialize AgentCard in test_agent_card_serialization");
        assert_eq!(parsed.name, "Agent");
        assert_eq!(parsed.skills.len(), 1);
    }

    // ---- A2ATask tests ----

    #[test]
    fn test_task_new() {
        let task = A2ATask::new();
        assert!(!task.id.is_empty());
        assert_eq!(task.status, A2ATaskStatus::Submitted);
        assert!(task.messages.is_empty());
        assert!(task.artifacts.is_empty());
        assert_eq!(task.history.len(), 1);
        assert_eq!(task.history[0].status, A2ATaskStatus::Submitted);
    }

    #[test]
    fn test_task_transition_valid() {
        let mut task = A2ATask::new();
        assert!(task.transition(A2ATaskStatus::Working).is_ok());
        assert_eq!(task.status, A2ATaskStatus::Working);
        assert!(task.transition(A2ATaskStatus::Completed).is_ok());
        assert_eq!(task.status, A2ATaskStatus::Completed);
    }

    #[test]
    fn test_task_transition_invalid() {
        let mut task = A2ATask::new();
        // Cannot go directly from Submitted to Completed
        let result = task.transition(A2ATaskStatus::Completed);
        assert!(result.is_err());
        assert_eq!(task.status, A2ATaskStatus::Submitted);
    }

    #[test]
    fn test_task_transition_from_terminal() {
        let mut task = A2ATask::new();
        task.transition(A2ATaskStatus::Working).expect("transition to Working in test_task_transition_from_terminal");
        task.transition(A2ATaskStatus::Completed).expect("transition to Completed in test_task_transition_from_terminal");
        // Terminal state — no further transitions
        assert!(task.transition(A2ATaskStatus::Working).is_err());
        assert!(task.transition(A2ATaskStatus::Failed).is_err());
        assert!(task.transition(A2ATaskStatus::Canceled).is_err());
    }

    #[test]
    fn test_task_lifecycle_full() {
        let mut task = A2ATask::new();
        assert_eq!(task.status, A2ATaskStatus::Submitted);
        task.transition(A2ATaskStatus::Working).expect("transition to Working in test_task_lifecycle_full");
        assert_eq!(task.status, A2ATaskStatus::Working);
        task.transition(A2ATaskStatus::Completed).expect("transition to Completed in test_task_lifecycle_full");
        assert_eq!(task.status, A2ATaskStatus::Completed);
        assert_eq!(task.history.len(), 3); // Submitted, Working, Completed
    }

    #[test]
    fn test_task_lifecycle_with_input_required() {
        let mut task = A2ATask::new();
        task.transition(A2ATaskStatus::Working).expect("transition to Working in test_task_lifecycle_with_input_required");
        task.transition(A2ATaskStatus::InputRequired)
            .expect("transition to InputRequired in test_task_lifecycle_with_input_required");
        assert_eq!(task.status, A2ATaskStatus::InputRequired);
        task.transition(A2ATaskStatus::Working).expect("resume Working in test_task_lifecycle_with_input_required");
        task.transition(A2ATaskStatus::Completed).expect("transition to Completed in test_task_lifecycle_with_input_required");
        assert_eq!(task.history.len(), 5);
    }

    #[test]
    fn test_task_lifecycle_canceled() {
        let mut task = A2ATask::new();
        task.transition(A2ATaskStatus::Working).expect("transition to Working in test_task_lifecycle_canceled");
        task.transition(A2ATaskStatus::Canceled).expect("transition to Canceled in test_task_lifecycle_canceled");
        assert_eq!(task.status, A2ATaskStatus::Canceled);
        assert!(task.status.is_terminal());
    }

    #[test]
    fn test_task_lifecycle_failed() {
        let mut task = A2ATask::new();
        task.transition(A2ATaskStatus::Failed).expect("transition to Failed in test_task_lifecycle_failed");
        assert_eq!(task.status, A2ATaskStatus::Failed);
        assert!(task.status.is_terminal());
    }

    #[test]
    fn test_task_add_message() {
        let mut task = A2ATask::new();
        let msg = A2AMessage::text(MessageRole::User, "Hello");
        task.add_message(msg);
        assert_eq!(task.messages.len(), 1);
        assert_eq!(task.messages[0].role, MessageRole::User);
    }

    #[test]
    fn test_task_add_artifact() {
        let mut task = A2ATask::new();
        let artifact = A2AArtifact::new("result.txt", 0)
            .with_description("The result file")
            .with_part(A2APart::Text(TextPart {
                text: "contents".to_string(),
            }));
        task.add_artifact(artifact);
        assert_eq!(task.artifacts.len(), 1);
        assert_eq!(task.artifacts[0].name, "result.txt");
        assert_eq!(task.artifacts[0].description, Some("The result file".to_string()));
        assert_eq!(task.artifacts[0].parts.len(), 1);
    }

    #[test]
    fn test_task_history_timestamps() {
        let task = A2ATask::new();
        // The initial Submitted event should have a nonzero timestamp
        assert!(task.history[0].timestamp > 0);
    }

    #[test]
    fn test_task_metadata() {
        let mut task = A2ATask::new();
        task.metadata
            .insert("priority".to_string(), serde_json::json!("high"));
        assert_eq!(task.metadata.get("priority"), Some(&serde_json::json!("high")));
    }

    // ---- Message and Part tests ----

    #[test]
    fn test_message_text_part() {
        let msg = A2AMessage::text(MessageRole::User, "Hello world");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.parts.len(), 1);
        match &msg.parts[0] {
            A2APart::Text(tp) => assert_eq!(tp.text, "Hello world"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_message_file_part() {
        let part = A2APart::File(FilePart {
            name: "doc.pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            data: None,
            uri: Some("https://example.com/doc.pdf".to_string()),
        });
        let msg = A2AMessage::with_parts(MessageRole::Agent, vec![part]);
        assert_eq!(msg.parts.len(), 1);
        match &msg.parts[0] {
            A2APart::File(fp) => {
                assert_eq!(fp.name, "doc.pdf");
                assert_eq!(fp.mime_type, "application/pdf");
                assert!(fp.data.is_none());
                assert_eq!(fp.uri, Some("https://example.com/doc.pdf".to_string()));
            }
            _ => panic!("Expected file part"),
        }
    }

    #[test]
    fn test_message_data_part() {
        let part = A2APart::Data(DataPart {
            mime_type: "application/json".to_string(),
            data: serde_json::json!({"key": "value"}),
        });
        let msg = A2AMessage::with_parts(MessageRole::Agent, vec![part]);
        assert_eq!(msg.parts.len(), 1);
        match &msg.parts[0] {
            A2APart::Data(dp) => {
                assert_eq!(dp.mime_type, "application/json");
                assert_eq!(dp.data, serde_json::json!({"key": "value"}));
            }
            _ => panic!("Expected data part"),
        }
    }

    #[test]
    fn test_message_multiple_parts() {
        let parts = vec![
            A2APart::Text(TextPart {
                text: "see attached".to_string(),
            }),
            A2APart::File(FilePart {
                name: "img.png".to_string(),
                mime_type: "image/png".to_string(),
                data: Some("base64data".to_string()),
                uri: None,
            }),
        ];
        let msg = A2AMessage::with_parts(MessageRole::User, parts);
        assert_eq!(msg.parts.len(), 2);
    }

    #[test]
    fn test_message_serialization() {
        let msg = A2AMessage::text(MessageRole::Agent, "response");
        let json = serde_json::to_string(&msg).expect("serialize A2AMessage in test_message_serialization");
        let parsed: A2AMessage = serde_json::from_str(&json).expect("deserialize A2AMessage in test_message_serialization");
        assert_eq!(parsed.role, MessageRole::Agent);
        assert_eq!(parsed.parts.len(), 1);
    }

    #[test]
    fn test_message_with_metadata() {
        let mut msg = A2AMessage::text(MessageRole::User, "test");
        msg.metadata
            .insert("source".to_string(), serde_json::json!("api"));
        let json = serde_json::to_string(&msg).expect("serialize message with metadata");
        let parsed: A2AMessage = serde_json::from_str(&json).expect("deserialize message with metadata");
        assert_eq!(parsed.metadata.get("source"), Some(&serde_json::json!("api")));
    }

    // ---- Task status display ----

    #[test]
    fn test_task_status_display() {
        assert_eq!(format!("{}", A2ATaskStatus::Submitted), "submitted");
        assert_eq!(format!("{}", A2ATaskStatus::Working), "working");
        assert_eq!(format!("{}", A2ATaskStatus::InputRequired), "input-required");
        assert_eq!(format!("{}", A2ATaskStatus::Completed), "completed");
        assert_eq!(format!("{}", A2ATaskStatus::Failed), "failed");
        assert_eq!(format!("{}", A2ATaskStatus::Canceled), "canceled");
    }

    #[test]
    fn test_task_status_is_terminal() {
        assert!(!A2ATaskStatus::Submitted.is_terminal());
        assert!(!A2ATaskStatus::Working.is_terminal());
        assert!(!A2ATaskStatus::InputRequired.is_terminal());
        assert!(A2ATaskStatus::Completed.is_terminal());
        assert!(A2ATaskStatus::Failed.is_terminal());
        assert!(A2ATaskStatus::Canceled.is_terminal());
    }

    #[test]
    fn test_task_status_serialization() {
        let status = A2ATaskStatus::Working;
        let json = serde_json::to_string(&status).expect("serialize TaskStatus in test_task_status_serialization");
        let parsed: A2ATaskStatus = serde_json::from_str(&json).expect("deserialize TaskStatus in test_task_status_serialization");
        assert_eq!(parsed, A2ATaskStatus::Working);
    }

    // ---- JSON-RPC tests ----

    #[test]
    fn test_json_rpc_request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "tasks/send".to_string(),
            params: Some(serde_json::json!({"message": "hello"})),
            id: serde_json::json!(1),
        };
        let json = serde_json::to_string(&req).expect("serialize JsonRpcRequest in test_json_rpc_request_serialization");
        let parsed: JsonRpcRequest = serde_json::from_str(&json).expect("deserialize JsonRpcRequest in test_json_rpc_request_serialization");
        assert_eq!(parsed.jsonrpc, "2.0");
        assert_eq!(parsed.method, "tasks/send");
        assert_eq!(parsed.id, serde_json::json!(1));
    }

    #[test]
    fn test_json_rpc_response_success() {
        let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"ok": true}));
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        assert_eq!(resp.id, serde_json::json!(1));
    }

    #[test]
    fn test_json_rpc_response_error() {
        let resp = JsonRpcResponse::error(
            serde_json::json!(2),
            JsonRpcError {
                code: METHOD_NOT_FOUND,
                message: "Method not found".to_string(),
                data: None,
            },
        );
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        let err = resp.error.as_ref().expect("error in test_json_rpc_response_error");
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }

    #[test]
    fn test_json_rpc_response_serialization() {
        let resp = JsonRpcResponse::success(serde_json::json!("id-1"), serde_json::json!(42));
        let json = serde_json::to_string(&resp).expect("serialize JsonRpcResponse roundtrip");
        let parsed: JsonRpcResponse = serde_json::from_str(&json).expect("deserialize JsonRpcResponse roundtrip");
        assert_eq!(parsed.result, Some(serde_json::json!(42)));
    }

    #[test]
    fn test_json_rpc_error_codes() {
        assert_eq!(PARSE_ERROR, -32700);
        assert_eq!(INVALID_REQUEST, -32600);
        assert_eq!(METHOD_NOT_FOUND, -32601);
        assert_eq!(INVALID_PARAMS, -32602);
        assert_eq!(INTERNAL_ERROR, -32603);
        assert_eq!(TASK_NOT_FOUND, -32001);
        assert_eq!(INVALID_STATE, -32002);
    }

    #[test]
    fn test_json_rpc_request_validate_ok() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "agent/card".to_string(),
            params: None,
            id: serde_json::json!(1),
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_json_rpc_request_validate_bad_version() {
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            method: "agent/card".to_string(),
            params: None,
            id: serde_json::json!(1),
        };
        let err = req.validate().unwrap_err();
        assert_eq!(err.code, INVALID_REQUEST);
    }

    #[test]
    fn test_json_rpc_request_validate_empty_method() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "".to_string(),
            params: None,
            id: serde_json::json!(1),
        };
        let err = req.validate().unwrap_err();
        assert_eq!(err.code, INVALID_REQUEST);
    }

    // ---- Server tests ----

    fn make_echo_server() -> A2AServer {
        let card = AgentCard::new("EchoAgent", "Echoes messages", "http://echo.local");
        let handler: TaskHandler = Box::new(|msg: &A2AMessage| {
            // Echo back the first text part
            let text = msg
                .parts
                .iter()
                .find_map(|p| match p {
                    A2APart::Text(tp) => Some(tp.text.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "no text".to_string());
            Ok(A2AMessage::text(
                MessageRole::Agent,
                format!("echo: {}", text),
            ))
        });
        A2AServer::new(card, handler)
    }

    fn make_failing_server() -> A2AServer {
        let card = AgentCard::new("FailAgent", "Always fails", "http://fail.local");
        let handler: TaskHandler = Box::new(|_msg: &A2AMessage| {
            Err(AiError::Other("handler intentionally failed".to_string()))
        });
        A2AServer::new(card, handler)
    }

    fn rpc_request(method: &str, params: Option<serde_json::Value>) -> String {
        serde_json::to_string(&JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: serde_json::json!(1),
        })
        .expect("serialize rpc_request helper")
    }

    #[test]
    fn test_server_handle_agent_card() {
        let server = make_echo_server();
        let resp_str = server.handle_request(&rpc_request("agent/card", None));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse agent/card response");
        assert!(resp.error.is_none());
        let result = resp.result.expect("agent/card result present");
        assert_eq!(result.get("name").and_then(|v| v.as_str()), Some("EchoAgent"));
    }

    #[test]
    fn test_server_handle_tasks_send() {
        let server = make_echo_server();
        let msg = A2AMessage::text(MessageRole::User, "ping");
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).expect("serialize msg for tasks/send")
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse tasks/send response");
        assert!(resp.error.is_none(), "Unexpected error: {:?}", resp.error);
        let result = resp.result.expect("tasks/send result present");
        assert_eq!(
            result.get("status").and_then(|v| v.as_str()),
            Some("Completed")
        );
        // Should have 2 messages: user + agent echo
        let messages = result.get("messages").and_then(|v| v.as_array()).expect("messages array in tasks/send result");
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_server_handle_tasks_get() {
        let server = make_echo_server();
        // First send a task
        let msg = A2AMessage::text(MessageRole::User, "hello");
        let send_params = serde_json::json!({
            "message": serde_json::to_value(&msg).expect("serialize msg for handle_tasks_get setup")
        });
        let send_resp_str = server.handle_request(&rpc_request("tasks/send", Some(send_params)));
        let send_resp: JsonRpcResponse =
            serde_json::from_str(&send_resp_str).expect("parse send response in handle_tasks_get");
        let task_id = send_resp
            .result
            .as_ref()
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
            .expect("task id from send in handle_tasks_get");

        // Now get the task
        let get_params = serde_json::json!({ "task_id": task_id });
        let get_resp_str = server.handle_request(&rpc_request("tasks/get", Some(get_params)));
        let get_resp: JsonRpcResponse =
            serde_json::from_str(&get_resp_str).expect("parse tasks/get response");
        assert!(get_resp.error.is_none());
        let result = get_resp.result.expect("tasks/get result present");
        assert_eq!(
            result.get("id").and_then(|v| v.as_str()),
            Some(task_id)
        );
    }

    #[test]
    fn test_server_handle_tasks_get_not_found() {
        let server = make_echo_server();
        let params = serde_json::json!({ "task_id": "nonexistent-id" });
        let resp_str = server.handle_request(&rpc_request("tasks/get", Some(params)));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse response in tasks_get_not_found");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in tasks_get_not_found");
        assert_eq!(err.code, TASK_NOT_FOUND);
    }

    #[test]
    fn test_server_handle_tasks_cancel() {
        let server = make_echo_server();
        // Send a task first
        let msg = A2AMessage::text(MessageRole::User, "work");
        let send_params = serde_json::json!({
            "message": serde_json::to_value(&msg).expect("serialize msg for tasks_cancel setup")
        });
        let send_resp_str = server.handle_request(&rpc_request("tasks/send", Some(send_params)));
        let send_resp: JsonRpcResponse =
            serde_json::from_str(&send_resp_str).expect("parse send response in tasks_cancel");

        // The echo handler completes immediately, so task is Completed (terminal).
        // Cancel should fail on a completed task.
        let task_id = send_resp
            .result
            .as_ref()
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
            .expect("task id from send in tasks_cancel");

        let cancel_params = serde_json::json!({ "task_id": task_id });
        let cancel_resp_str =
            server.handle_request(&rpc_request("tasks/cancel", Some(cancel_params)));
        let cancel_resp: JsonRpcResponse =
            serde_json::from_str(&cancel_resp_str).expect("parse cancel response for completed task");
        // Cancel of a completed task should fail
        assert!(cancel_resp.error.is_some());
        let err = cancel_resp.error.expect("cancel error for completed task");
        assert_eq!(err.code, INVALID_STATE);
    }

    #[test]
    fn test_server_cancel_working_task() {
        // Manually insert a Working task, then cancel it
        let server = make_echo_server();
        let mut task = A2ATask::new();
        task.transition(A2ATaskStatus::Working).expect("transition to Working in cancel_working_task");
        let task_id = task.id.clone();
        server
            .tasks
            .lock()
            .expect("lock tasks mutex in cancel_working_task")
            .insert(task_id.clone(), task);

        let cancel_params = serde_json::json!({ "task_id": task_id });
        let resp_str = server.handle_request(&rpc_request("tasks/cancel", Some(cancel_params)));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse cancel response for working task");
        assert!(resp.error.is_none(), "Expected success: {:?}", resp.error);
        let result = resp.result.expect("cancel working task result");
        assert_eq!(
            result.get("status").and_then(|v| v.as_str()),
            Some("Canceled")
        );
    }

    #[test]
    fn test_server_handle_method_not_found() {
        let server = make_echo_server();
        let resp_str = server.handle_request(&rpc_request("nonexistent/method", None));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse method_not_found response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in method_not_found");
        assert_eq!(err.code, METHOD_NOT_FOUND);
        assert!(err.message.contains("nonexistent/method"));
    }

    #[test]
    fn test_server_handle_parse_error() {
        let server = make_echo_server();
        let resp_str = server.handle_request("this is not json");
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse parse_error response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in handle_parse_error");
        assert_eq!(err.code, PARSE_ERROR);
    }

    #[test]
    fn test_server_handle_invalid_jsonrpc() {
        let server = make_echo_server();
        let bad_req = serde_json::json!({
            "jsonrpc": "1.0",
            "method": "agent/card",
            "id": 1
        });
        let resp_str = server.handle_request(&serde_json::to_string(&bad_req).expect("serialize bad jsonrpc request"));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse invalid_jsonrpc response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in invalid_jsonrpc test");
        assert_eq!(err.code, INVALID_REQUEST);
    }

    #[test]
    fn test_server_tasks_send_missing_message() {
        let server = make_echo_server();
        let params = serde_json::json!({ "data": "no message field" });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse missing_message response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in tasks_send_missing_message");
        assert_eq!(err.code, INVALID_PARAMS);
    }

    #[test]
    fn test_server_tasks_send_no_params() {
        let server = make_echo_server();
        let resp_str = server.handle_request(&rpc_request("tasks/send", None));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse tasks_send_no_params response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in tasks_send_no_params");
        assert_eq!(err.code, INVALID_PARAMS);
    }

    #[test]
    fn test_server_tasks_get_no_params() {
        let server = make_echo_server();
        let resp_str = server.handle_request(&rpc_request("tasks/get", None));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse tasks_get_no_params response");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.expect("error in tasks_get_no_params").code, INVALID_PARAMS);
    }

    #[test]
    fn test_server_with_failing_handler() {
        let server = make_failing_server();
        let msg = A2AMessage::text(MessageRole::User, "trigger failure");
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).expect("serialize msg for failing_handler test")
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse =
            serde_json::from_str(&resp_str).expect("parse failing_handler response");
        assert!(resp.error.is_some());
        let err = resp.error.expect("error in failing_handler test");
        assert_eq!(err.code, INTERNAL_ERROR);
        assert!(err.message.contains("handler intentionally failed"));

        // The failed task should be stored
        let tasks = server.tasks.lock().expect("lock tasks mutex in failing_handler");
        assert_eq!(tasks.len(), 1);
        let task = tasks.values().next().expect("one task in failing_handler");
        assert_eq!(task.status, A2ATaskStatus::Failed);
    }

    // ---- Client tests ----

    #[test]
    fn test_client_creation() {
        let client = A2AClient::new("http://agent.local:8080");
        assert_eq!(client.base_url, "http://agent.local:8080");
        assert!(client.auth_header.is_none());
    }

    #[test]
    fn test_client_with_api_key() {
        let client = A2AClient::new("http://agent.local")
            .with_api_key("Authorization", "Bearer sk-123");
        assert_eq!(
            client.auth_header,
            Some(("Authorization".to_string(), "Bearer sk-123".to_string()))
        );
    }

    #[test]
    fn test_client_build_request() {
        let client = A2AClient::new("http://agent.local");
        let req = client.build_request("agent/card", None);
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "agent/card");
        assert!(req.params.is_none());
        // ID should be a valid UUID string
        assert!(req.id.is_string());
    }

    #[test]
    fn test_client_build_request_with_params() {
        let client = A2AClient::new("http://agent.local");
        let params = serde_json::json!({"task_id": "abc-123"});
        let req = client.build_request("tasks/get", Some(params.clone()));
        assert_eq!(req.method, "tasks/get");
        assert_eq!(req.params, Some(params));
    }

    #[test]
    fn test_client_build_request_unique_ids() {
        let client = A2AClient::new("http://agent.local");
        let req1 = client.build_request("m1", None);
        let req2 = client.build_request("m2", None);
        assert_ne!(req1.id, req2.id);
    }

    // ---- Directory tests ----

    #[test]
    fn test_directory_new() {
        let dir = AgentDirectory::new();
        assert!(dir.is_empty());
        assert_eq!(dir.len(), 0);
    }

    #[test]
    fn test_directory_register() {
        let mut dir = AgentDirectory::new();
        let card = AgentCard::new("Agent1", "First agent", "http://a1.local");
        dir.register(card);
        assert_eq!(dir.len(), 1);
        assert!(!dir.is_empty());
        assert!(dir.get("http://a1.local").is_some());
    }

    #[test]
    fn test_directory_register_overwrites() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("Agent1", "v1", "http://a1.local"));
        dir.register(AgentCard::new("Agent1-updated", "v2", "http://a1.local"));
        assert_eq!(dir.len(), 1);
        assert_eq!(dir.get("http://a1.local").expect("card after overwrite in register_overwrites").name, "Agent1-updated");
    }

    #[test]
    fn test_directory_unregister() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("Agent1", "desc", "http://a1.local"));
        assert!(dir.unregister("http://a1.local"));
        assert!(dir.is_empty());
    }

    #[test]
    fn test_directory_unregister_nonexistent() {
        let mut dir = AgentDirectory::new();
        assert!(!dir.unregister("http://nonexistent.local"));
    }

    #[test]
    fn test_directory_find_by_name() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("TranslatorBot", "translates", "http://t.local"));
        dir.register(AgentCard::new("SummarizerBot", "summarizes", "http://s.local"));
        dir.register(AgentCard::new("TranslateHelper", "also translates", "http://th.local"));

        let results = dir.find_by_name("translat");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_directory_find_by_name_case_insensitive() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("MyAgent", "desc", "http://a.local"));
        let results = dir.find_by_name("myagent");
        assert_eq!(results.len(), 1);
        let results2 = dir.find_by_name("MYAGENT");
        assert_eq!(results2.len(), 1);
    }

    #[test]
    fn test_directory_find_by_skill() {
        let mut dir = AgentDirectory::new();
        dir.register(
            AgentCard::new("NlpAgent", "NLP", "http://nlp.local")
                .with_skill(AgentSkill::new("translate", "translates", vec!["nlp".to_string()]))
                .with_skill(AgentSkill::new("sentiment", "analyze sentiment", vec!["nlp".to_string()])),
        );
        dir.register(
            AgentCard::new("CodeAgent", "Code", "http://code.local")
                .with_skill(AgentSkill::new("generate-code", "generates code", vec!["code".to_string()])),
        );

        let nlp_agents = dir.find_by_skill("nlp");
        assert_eq!(nlp_agents.len(), 1);
        assert_eq!(nlp_agents[0].name, "NlpAgent");

        let translate_agents = dir.find_by_skill("translate");
        assert_eq!(translate_agents.len(), 1);

        let code_agents = dir.find_by_skill("code");
        assert_eq!(code_agents.len(), 1);
        assert_eq!(code_agents[0].name, "CodeAgent");
    }

    #[test]
    fn test_directory_find_by_skill_no_match() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("Agent", "desc", "http://a.local"));
        let results = dir.find_by_skill("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_directory_list_all() {
        let mut dir = AgentDirectory::new();
        dir.register(AgentCard::new("A1", "d1", "http://a1.local"));
        dir.register(AgentCard::new("A2", "d2", "http://a2.local"));
        dir.register(AgentCard::new("A3", "d3", "http://a3.local"));
        let all = dir.list_all();
        assert_eq!(all.len(), 3);
    }

    // ---- Push notification tests ----

    #[test]
    fn test_push_notification_serialization() {
        let notif = PushNotification {
            task_id: "task-123".to_string(),
            status: A2ATaskStatus::Completed,
            message: Some(A2AMessage::text(MessageRole::Agent, "done")),
            timestamp: 1700000000,
        };
        let json = serde_json::to_string(&notif).expect("serialize PushNotification");
        let parsed: PushNotification = serde_json::from_str(&json).expect("deserialize PushNotification");
        assert_eq!(parsed.task_id, "task-123");
        assert_eq!(parsed.status, A2ATaskStatus::Completed);
        assert!(parsed.message.is_some());
        assert_eq!(parsed.timestamp, 1700000000);
    }

    #[test]
    fn test_push_config_serialization() {
        let config = PushNotificationConfig {
            callback_url: "https://my-app.local/webhooks/a2a".to_string(),
            events: vec![A2ATaskStatus::Completed, A2ATaskStatus::Failed],
            auth_token: Some("secret-token".to_string()),
        };
        let json = serde_json::to_string(&config).expect("serialize PushNotificationConfig with auth");
        let parsed: PushNotificationConfig = serde_json::from_str(&json).expect("deserialize PushNotificationConfig with auth");
        assert_eq!(parsed.callback_url, "https://my-app.local/webhooks/a2a");
        assert_eq!(parsed.events.len(), 2);
        assert_eq!(parsed.auth_token, Some("secret-token".to_string()));
    }

    #[test]
    fn test_push_config_no_auth() {
        let config = PushNotificationConfig {
            callback_url: "http://hook.local".to_string(),
            events: vec![A2ATaskStatus::Working],
            auth_token: None,
        };
        let json = serde_json::to_string(&config).expect("serialize PushNotificationConfig no auth");
        let parsed: PushNotificationConfig = serde_json::from_str(&json).expect("deserialize PushNotificationConfig no auth");
        assert!(parsed.auth_token.is_none());
    }

    #[test]
    fn test_server_register_push() {
        let server = make_echo_server();
        // Send a task first
        let msg = A2AMessage::text(MessageRole::User, "go");
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).expect("serialize msg for register_push setup")
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).expect("parse send response in register_push");
        let task_id = resp
            .result
            .as_ref()
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
            .expect("task id for register_push test")
            .to_string();

        // Register push for this task
        let config = PushNotificationConfig {
            callback_url: "http://hooks.local/notify".to_string(),
            events: vec![A2ATaskStatus::Completed, A2ATaskStatus::Failed],
            auth_token: Some("tok-123".to_string()),
        };
        server.register_push(&task_id, config).expect("register push config for task");

        // Verify it was stored
        let retrieved = server.get_push_config(&task_id).expect("push config should exist after register");
        assert_eq!(retrieved.callback_url, "http://hooks.local/notify");
        assert_eq!(retrieved.events.len(), 2);
    }

    #[test]
    fn test_server_register_push_nonexistent_task() {
        let server = make_echo_server();
        let config = PushNotificationConfig {
            callback_url: "http://hooks.local".to_string(),
            events: vec![],
            auth_token: None,
        };
        let result = server.register_push("nonexistent-task", config);
        assert!(result.is_err());
    }

    // ---- Allowed-agents authorization tests ----

    #[test]
    fn test_server_allowed_agents_rejects_unknown_sender() {
        let server = make_echo_server();
        server.allow_agent("trusted-agent-1");

        let msg = A2AMessage::text(MessageRole::User, "ping");
        // Send with an unknown sender_agent
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap(),
            "sender_agent": "malicious-agent"
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, UNAUTHORIZED_AGENT);
    }

    #[test]
    fn test_server_allowed_agents_rejects_missing_sender() {
        let server = make_echo_server();
        server.allow_agent("trusted-agent-1");

        let msg = A2AMessage::text(MessageRole::User, "ping");
        // Send without sender_agent field at all
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap()
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, UNAUTHORIZED_AGENT);
    }

    #[test]
    fn test_server_allowed_agents_accepts_trusted_sender() {
        let server = make_echo_server();
        server.allow_agent("trusted-agent-1");

        let msg = A2AMessage::text(MessageRole::User, "ping");
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap(),
            "sender_agent": "trusted-agent-1"
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none(), "Expected success but got: {:?}", resp.error);
        let result = resp.result.expect("result present");
        assert_eq!(result.get("status").and_then(|v| v.as_str()), Some("Completed"));
    }

    #[test]
    fn test_server_empty_allowlist_permits_all() {
        // Default (empty allowlist) should allow any sender or no sender
        let server = make_echo_server();

        let msg = A2AMessage::text(MessageRole::User, "ping");
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap()
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none(), "Empty allowlist should permit all senders");
    }

    #[test]
    fn test_server_set_allowed_agents_bulk() {
        let server = make_echo_server();
        let mut agents = HashSet::new();
        agents.insert("agent-a".to_string());
        agents.insert("agent-b".to_string());
        server.set_allowed_agents(agents);

        let msg = A2AMessage::text(MessageRole::User, "ping");

        // agent-a should be allowed
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap(),
            "sender_agent": "agent-a"
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());

        // agent-c should be rejected
        let params = serde_json::json!({
            "message": serde_json::to_value(&msg).unwrap(),
            "sender_agent": "agent-c"
        });
        let resp_str = server.handle_request(&rpc_request("tasks/send", Some(params)));
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, UNAUTHORIZED_AGENT);
    }

    // ---- State transition exhaustive tests ----

    #[test]
    fn test_transition_submitted_to_working() {
        assert!(A2ATaskStatus::Submitted.can_transition_to(A2ATaskStatus::Working));
    }

    #[test]
    fn test_transition_submitted_to_failed() {
        assert!(A2ATaskStatus::Submitted.can_transition_to(A2ATaskStatus::Failed));
    }

    #[test]
    fn test_transition_submitted_to_canceled() {
        assert!(A2ATaskStatus::Submitted.can_transition_to(A2ATaskStatus::Canceled));
    }

    #[test]
    fn test_transition_submitted_to_completed_invalid() {
        assert!(!A2ATaskStatus::Submitted.can_transition_to(A2ATaskStatus::Completed));
    }

    #[test]
    fn test_transition_submitted_to_input_required_invalid() {
        assert!(!A2ATaskStatus::Submitted.can_transition_to(A2ATaskStatus::InputRequired));
    }

    #[test]
    fn test_transition_working_to_input_required() {
        assert!(A2ATaskStatus::Working.can_transition_to(A2ATaskStatus::InputRequired));
    }

    #[test]
    fn test_transition_working_to_completed() {
        assert!(A2ATaskStatus::Working.can_transition_to(A2ATaskStatus::Completed));
    }

    #[test]
    fn test_transition_working_to_failed() {
        assert!(A2ATaskStatus::Working.can_transition_to(A2ATaskStatus::Failed));
    }

    #[test]
    fn test_transition_working_to_canceled() {
        assert!(A2ATaskStatus::Working.can_transition_to(A2ATaskStatus::Canceled));
    }

    #[test]
    fn test_transition_input_required_to_working() {
        assert!(A2ATaskStatus::InputRequired.can_transition_to(A2ATaskStatus::Working));
    }

    #[test]
    fn test_transition_input_required_to_failed() {
        assert!(A2ATaskStatus::InputRequired.can_transition_to(A2ATaskStatus::Failed));
    }

    #[test]
    fn test_transition_input_required_to_canceled() {
        assert!(A2ATaskStatus::InputRequired.can_transition_to(A2ATaskStatus::Canceled));
    }

    #[test]
    fn test_transition_input_required_to_completed_invalid() {
        // Cannot go from InputRequired to Completed directly; must go through Working
        assert!(!A2ATaskStatus::InputRequired.can_transition_to(A2ATaskStatus::Completed));
    }

    #[test]
    fn test_transition_completed_to_any_invalid() {
        for target in &[
            A2ATaskStatus::Submitted,
            A2ATaskStatus::Working,
            A2ATaskStatus::InputRequired,
            A2ATaskStatus::Failed,
            A2ATaskStatus::Canceled,
        ] {
            assert!(!A2ATaskStatus::Completed.can_transition_to(*target));
        }
    }

    #[test]
    fn test_transition_failed_to_any_invalid() {
        for target in &[
            A2ATaskStatus::Submitted,
            A2ATaskStatus::Working,
            A2ATaskStatus::InputRequired,
            A2ATaskStatus::Completed,
            A2ATaskStatus::Canceled,
        ] {
            assert!(!A2ATaskStatus::Failed.can_transition_to(*target));
        }
    }

    #[test]
    fn test_transition_canceled_to_any_invalid() {
        for target in &[
            A2ATaskStatus::Submitted,
            A2ATaskStatus::Working,
            A2ATaskStatus::InputRequired,
            A2ATaskStatus::Completed,
            A2ATaskStatus::Failed,
        ] {
            assert!(!A2ATaskStatus::Canceled.can_transition_to(*target));
        }
    }

    // ---- Artifact tests ----

    #[test]
    fn test_artifact_new() {
        let art = A2AArtifact::new("output.json", 0);
        assert_eq!(art.name, "output.json");
        assert_eq!(art.index, 0);
        assert!(art.description.is_none());
        assert!(art.parts.is_empty());
    }

    #[test]
    fn test_artifact_builder() {
        let art = A2AArtifact::new("report", 1)
            .with_description("A generated report")
            .with_part(A2APart::Text(TextPart {
                text: "Report contents".to_string(),
            }))
            .with_part(A2APart::Data(DataPart {
                mime_type: "application/json".to_string(),
                data: serde_json::json!({"summary": true}),
            }));
        assert_eq!(art.description, Some("A generated report".to_string()));
        assert_eq!(art.parts.len(), 2);
        assert_eq!(art.index, 1);
    }

    #[test]
    fn test_artifact_serialization() {
        let art = A2AArtifact::new("data.csv", 0)
            .with_description("CSV data")
            .with_part(A2APart::Text(TextPart {
                text: "a,b,c\n1,2,3".to_string(),
            }));
        let json = serde_json::to_string(&art).expect("serialize A2AArtifact");
        let parsed: A2AArtifact = serde_json::from_str(&json).expect("deserialize A2AArtifact");
        assert_eq!(parsed.name, "data.csv");
        assert_eq!(parsed.description, Some("CSV data".to_string()));
        assert_eq!(parsed.parts.len(), 1);
    }

    // ---- Task serialization tests ----

    #[test]
    fn test_task_serialization_roundtrip() {
        let mut task = A2ATask::new();
        task.add_message(A2AMessage::text(MessageRole::User, "hello"));
        task.transition(A2ATaskStatus::Working).expect("transition to Working in task_serialization_roundtrip");
        task.add_artifact(A2AArtifact::new("out", 0));
        task.transition(A2ATaskStatus::Completed).expect("transition to Completed in task_serialization_roundtrip");

        let json = serde_json::to_string(&task).expect("serialize A2ATask roundtrip");
        let parsed: A2ATask = serde_json::from_str(&json).expect("deserialize A2ATask roundtrip");
        assert_eq!(parsed.id, task.id);
        assert_eq!(parsed.status, A2ATaskStatus::Completed);
        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(parsed.artifacts.len(), 1);
        assert_eq!(parsed.history.len(), 3);
    }

    #[test]
    fn test_task_unique_ids() {
        let t1 = A2ATask::new();
        let t2 = A2ATask::new();
        assert_ne!(t1.id, t2.id);
    }

    // ---- Skill tests ----

    #[test]
    fn test_skill_creation() {
        let skill = AgentSkill::new("summarize", "Summarizes text", vec!["nlp".to_string(), "text".to_string()]);
        assert_eq!(skill.name, "summarize");
        assert_eq!(skill.description, "Summarizes text");
        assert_eq!(skill.tags.len(), 2);
    }

    #[test]
    fn test_skill_serialization() {
        let skill = AgentSkill::new("code", "Generate code", vec!["dev".to_string()]);
        let json = serde_json::to_string(&skill).expect("serialize AgentSkill");
        let parsed: AgentSkill = serde_json::from_str(&json).expect("deserialize AgentSkill");
        assert_eq!(parsed.name, "code");
        assert_eq!(parsed.tags, vec!["dev".to_string()]);
    }

    // ---- AGENTS.md Convention tests (v6 Phase 2.2) ----

    #[test]
    fn test_agents_md_entry_construction() {
        let entry = AgentsMdEntry {
            name: "TestAgent".to_string(),
            description: "A test agent".to_string(),
            protocols: vec!["A2A".to_string(), "MCP".to_string()],
            endpoint: Some("http://localhost:8080".to_string()),
            capabilities: vec!["translate".to_string()],
            version: Some("1.0.0".to_string()),
        };
        assert_eq!(entry.name, "TestAgent");
        assert_eq!(entry.description, "A test agent");
        assert_eq!(entry.protocols.len(), 2);
        assert_eq!(entry.endpoint, Some("http://localhost:8080".to_string()));
        assert_eq!(entry.capabilities, vec!["translate"]);
        assert_eq!(entry.version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_agents_md_parser_parse_valid() {
        let content = "\
## Agent: MyAgent
- Description: Does things
- Protocols: A2A, MCP
- Endpoint: http://localhost:9090
- Capabilities: summarize, translate
- Version: 2.0.0
";
        let parser = AgentsMdParser::new();
        let entries = parser.parse(content).expect("parse single agent entry");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "MyAgent");
        assert_eq!(entries[0].description, "Does things");
        assert_eq!(entries[0].protocols, vec!["A2A", "MCP"]);
        assert_eq!(entries[0].endpoint, Some("http://localhost:9090".to_string()));
        assert_eq!(entries[0].capabilities, vec!["summarize", "translate"]);
        assert_eq!(entries[0].version, Some("2.0.0".to_string()));
    }

    #[test]
    fn test_agents_md_parser_parse_multiple_entries() {
        let content = "\
## Agent: Alpha
- Description: First agent
- Protocols: A2A
- Endpoint: http://alpha.local
- Capabilities: search
- Version: 1.0.0

## Agent: Beta
- Description: Second agent
- Protocols: MCP, ACP
- Capabilities: code, review
";
        let parser = AgentsMdParser::new();
        let entries = parser.parse(content).expect("parse multiple agent entries");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "Alpha");
        assert_eq!(entries[0].protocols, vec!["A2A"]);
        assert_eq!(entries[0].endpoint, Some("http://alpha.local".to_string()));
        assert_eq!(entries[1].name, "Beta");
        assert_eq!(entries[1].protocols, vec!["MCP", "ACP"]);
        assert!(entries[1].endpoint.is_none());
        assert!(entries[1].version.is_none());
    }

    #[test]
    fn test_agents_md_parser_parse_empty_content() {
        let parser = AgentsMdParser::new();
        let entries = parser.parse("").expect("parse empty content");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_agents_md_parser_to_markdown_roundtrip() {
        let original_content = "\
## Agent: RoundTrip
- Description: Roundtrip test
- Protocols: A2A, MCP
- Endpoint: http://rt.local
- Capabilities: cap1, cap2
- Version: 3.0.0
";
        let parser = AgentsMdParser::new();
        let entries = parser.parse(original_content).expect("parse original for roundtrip");
        assert_eq!(entries.len(), 1);

        let markdown = AgentsMdParser::to_markdown(&entries);
        let re_parsed = parser.parse(&markdown).expect("re-parse markdown roundtrip");
        assert_eq!(re_parsed.len(), 1);
        assert_eq!(re_parsed[0].name, entries[0].name);
        assert_eq!(re_parsed[0].description, entries[0].description);
        assert_eq!(re_parsed[0].protocols, entries[0].protocols);
        assert_eq!(re_parsed[0].endpoint, entries[0].endpoint);
        assert_eq!(re_parsed[0].capabilities, entries[0].capabilities);
        assert_eq!(re_parsed[0].version, entries[0].version);
    }

    #[test]
    fn test_agents_md_discovery_load_from_content() {
        let content = "\
## Agent: Disco
- Description: Discovery agent
- Protocols: A2A
";
        let mut disc = AgentsMdDiscovery::new();
        let count = disc.load_from_content(content).expect("load single entry from content");
        assert_eq!(count, 1);
        assert_eq!(disc.entry_count(), 1);
    }

    #[test]
    fn test_agents_md_discovery_find_by_protocol() {
        let content = "\
## Agent: AlphaBot
- Description: Alpha
- Protocols: A2A, MCP

## Agent: BetaBot
- Description: Beta
- Protocols: ACP

## Agent: GammaBot
- Description: Gamma
- Protocols: A2A
";
        let mut disc = AgentsMdDiscovery::new();
        disc.load_from_content(content).expect("load entries for find_by_protocol test");
        let a2a = disc.find_by_protocol("A2A");
        assert_eq!(a2a.len(), 2);
        let acp = disc.find_by_protocol("ACP");
        assert_eq!(acp.len(), 1);
        assert_eq!(acp[0].name, "BetaBot");
        let empty = disc.find_by_protocol("UNKNOWN");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_agents_md_discovery_find_by_name() {
        let content = "\
## Agent: UniqueAgent
- Description: Unique
- Protocols: A2A
";
        let mut disc = AgentsMdDiscovery::new();
        disc.load_from_content(content).expect("load entry for find_by_name test");
        let found = disc.find_by_name("UniqueAgent");
        assert!(found.is_some());
        assert_eq!(found.expect("UniqueAgent should be found by name").description, "Unique");
        // Case insensitive
        let found_lower = disc.find_by_name("uniqueagent");
        assert!(found_lower.is_some());
        let not_found = disc.find_by_name("NonExistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_agents_md_discovery_entry_count() {
        let mut disc = AgentsMdDiscovery::new();
        assert_eq!(disc.entry_count(), 0);
        let content = "\
## A1
- Description: Agent 1
- Protocols: A2A

## A2
- Description: Agent 2
- Protocols: MCP
";
        disc.load_from_content(content).expect("load entries for entry_count test");
        assert_eq!(disc.entry_count(), 2);
        assert_eq!(disc.all_entries().len(), 2);
    }

    #[test]
    fn test_agents_md_discovery_clear() {
        let content = "\
## ClearMe
- Description: Will be cleared
- Protocols: A2A
";
        let mut disc = AgentsMdDiscovery::new();
        disc.load_from_content(content).expect("load entry for clear test");
        assert_eq!(disc.entry_count(), 1);
        disc.clear();
        assert_eq!(disc.entry_count(), 0);
        assert!(disc.all_entries().is_empty());
    }

    // ---- ACP Bridge tests (v6 Phase 2.3) ----

    #[test]
    fn test_acp_message_construction() {
        let msg = AcpMessage {
            run_id: "run-001".to_string(),
            content_parts: vec![AcpContentPart::Text("hello".to_string())],
            metadata: {
                let mut m = HashMap::new();
                m.insert("key".to_string(), "value".to_string());
                m
            },
            timestamp: 1700000000,
        };
        assert_eq!(msg.run_id, "run-001");
        assert_eq!(msg.content_parts.len(), 1);
        assert_eq!(msg.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(msg.timestamp, 1700000000);
    }

    #[test]
    fn test_acp_content_part_text_and_data() {
        let text_part = AcpContentPart::Text("hello world".to_string());
        match &text_part {
            AcpContentPart::Text(t) => assert_eq!(t, "hello world"),
            _ => panic!("Expected Text variant"),
        }

        let data_part = AcpContentPart::Data {
            mime_type: "application/json".to_string(),
            data: r#"{"key":"val"}"#.to_string(),
        };
        match &data_part {
            AcpContentPart::Data { mime_type, data } => {
                assert_eq!(mime_type, "application/json");
                assert_eq!(data, r#"{"key":"val"}"#);
            }
            _ => panic!("Expected Data variant"),
        }
    }

    #[test]
    fn test_acp_bridge_a2a_to_acp_preserves_content() {
        let a2a_msg = A2AMessage::text(MessageRole::User, "translate this");
        let acp_msg = AcpBridge::a2a_to_acp(&a2a_msg);

        assert!(!acp_msg.run_id.is_empty());
        assert_eq!(acp_msg.content_parts.len(), 1);
        match &acp_msg.content_parts[0] {
            AcpContentPart::Text(t) => assert_eq!(t, "translate this"),
            _ => panic!("Expected Text"),
        }
        assert!(acp_msg.timestamp > 0);
    }

    #[test]
    fn test_acp_bridge_acp_to_a2a_preserves_content() {
        let acp_msg = AcpMessage {
            run_id: "run-x".to_string(),
            content_parts: vec![
                AcpContentPart::Text("hello from ACP".to_string()),
                AcpContentPart::Data {
                    mime_type: "text/plain".to_string(),
                    data: "raw-data".to_string(),
                },
            ],
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".to_string(), "acp".to_string());
                m
            },
            timestamp: 12345,
        };

        let a2a_msg = AcpBridge::acp_to_a2a(&acp_msg);
        assert_eq!(a2a_msg.role, MessageRole::Agent);
        assert_eq!(a2a_msg.parts.len(), 2);

        match &a2a_msg.parts[0] {
            A2APart::Text(tp) => assert_eq!(tp.text, "hello from ACP"),
            _ => panic!("Expected Text part"),
        }
        match &a2a_msg.parts[1] {
            A2APart::Data(dp) => {
                assert_eq!(dp.mime_type, "text/plain");
                assert_eq!(dp.data, serde_json::Value::String("raw-data".to_string()));
            }
            _ => panic!("Expected Data part"),
        }
        assert_eq!(
            a2a_msg.metadata.get("source"),
            Some(&serde_json::Value::String("acp".to_string()))
        );
    }

    #[test]
    fn test_acp_bridge_translate_card_to_descriptor() {
        let card = AgentCard::new("TranslateBot", "Translates things", "http://t.local")
            .with_skill(AgentSkill::new(
                "translate",
                "translate text",
                vec!["nlp".to_string()],
            ))
            .with_capability("streaming", true)
            .with_capability("push", false);

        let desc = AcpBridge::translate_card_to_descriptor(&card);
        assert_eq!(desc.name, "TranslateBot");
        assert_eq!(desc.description, "Translates things");
        assert_eq!(desc.url, Some("http://t.local".to_string()));
        assert!(desc.protocols.contains(&"A2A".to_string()));
        // Should have "streaming" (cap=true) and "translate" (skill), but NOT "push" (cap=false)
        assert!(desc.capabilities.contains(&"streaming".to_string()));
        assert!(desc.capabilities.contains(&"translate".to_string()));
        assert!(!desc.capabilities.contains(&"push".to_string()));
    }

    #[test]
    fn test_acp_agent_descriptor_construction() {
        let desc = AcpAgentDescriptor {
            name: "Desc1".to_string(),
            description: "A descriptor".to_string(),
            url: Some("http://desc.local".to_string()),
            protocols: vec!["A2A".to_string(), "ACP".to_string()],
            capabilities: vec!["search".to_string()],
        };
        assert_eq!(desc.name, "Desc1");
        assert_eq!(desc.url, Some("http://desc.local".to_string()));
        assert_eq!(desc.protocols.len(), 2);
        assert_eq!(desc.capabilities, vec!["search"]);
    }

    #[test]
    fn test_acp_agent_adapter_from_agent_card() {
        let card = AgentCard::new("AdaptBot", "Adaptive agent", "http://adapt.local")
            .with_skill(AgentSkill::new("search", "search skill", vec![]));
        let adapter = AcpAgentAdapter::from_agent_card(&card);
        let desc = adapter.descriptor();
        assert_eq!(desc.name, "AdaptBot");
        assert_eq!(desc.description, "Adaptive agent");
        assert_eq!(desc.url, Some("http://adapt.local".to_string()));
        assert!(desc.capabilities.contains(&"search".to_string()));
    }

    #[test]
    fn test_acp_agent_adapter_to_agent_card_roundtrip() {
        let original = AgentCard::new("RoundBot", "Round-trip agent", "http://round.local")
            .with_skill(AgentSkill::new("analyze", "analyze data", vec!["ml".to_string()]))
            .with_capability("streaming", true);

        let adapter = AcpAgentAdapter::from_agent_card(&original);
        let converted = adapter.to_agent_card();

        assert_eq!(converted.name, "RoundBot");
        assert_eq!(converted.description, "Round-trip agent");
        assert_eq!(converted.url, "http://round.local");
        // Skills are reconstructed from capabilities; order may differ but
        // all original capability names should be present
        let skill_names: Vec<&str> = converted.skills.iter().map(|s| s.name.as_str()).collect();
        assert!(skill_names.contains(&"streaming"));
        assert!(skill_names.contains(&"analyze"));
    }

    #[test]
    fn test_acp_agent_adapter_descriptor() {
        let card = AgentCard::new("DescBot", "Descriptor bot", "http://d.local");
        let adapter = AcpAgentAdapter::from_agent_card(&card);
        let desc = adapter.descriptor();
        assert_eq!(desc.name, "DescBot");
        assert_eq!(desc.description, "Descriptor bot");
        assert_eq!(desc.url, Some("http://d.local".to_string()));
    }

    #[test]
    fn test_agents_md_parser_parse_missing_optional_fields() {
        let content = "\
## Agent: MinimalAgent
- Description: Minimal
- Protocols: A2A
";
        let parser = AgentsMdParser::new();
        let entries = parser.parse(content).expect("parse minimal agent missing optional fields");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "MinimalAgent");
        assert_eq!(entries[0].description, "Minimal");
        assert_eq!(entries[0].protocols, vec!["A2A"]);
        // Optional fields should be None / empty
        assert!(entries[0].endpoint.is_none());
        assert!(entries[0].version.is_none());
        assert!(entries[0].capabilities.is_empty());
    }
}
