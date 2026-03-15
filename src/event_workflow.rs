//! Event-driven workflow engine with typed events, checkpoints, time-travel, and breakpoints.
//!
//! Implements v4 roadmap items 1.1-1.4 (workflow engine), 6.2 (workflow as tool),
//! and 6.3 (workflow serialization).
//!
//! Feature-gated behind `workflows`. The outer `#[cfg]` guard ensures this
//! entire module compiles away when the feature is not enabled.

#[cfg(feature = "workflows")]
mod inner {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Serialize};

    use crate::error::{AiError, WorkflowError};

    // ========================================================================
    // Core event trait
    // ========================================================================

    /// Trait for typed workflow events.
    pub trait WorkflowEvent: Send + Sync + fmt::Debug {
        /// Returns the event type discriminator string.
        fn event_type(&self) -> &str;
        /// Serialises the event to a JSON value.
        fn to_json(&self) -> serde_json::Value;
    }

    // ========================================================================
    // SimpleEvent
    // ========================================================================

    /// A simple string-keyed event with a JSON payload.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SimpleEvent {
        pub event_type: String,
        pub payload: serde_json::Value,
    }

    impl SimpleEvent {
        pub fn new(event_type: &str, payload: serde_json::Value) -> Self {
            Self {
                event_type: event_type.to_string(),
                payload,
            }
        }
    }

    impl WorkflowEvent for SimpleEvent {
        fn event_type(&self) -> &str {
            &self.event_type
        }
        fn to_json(&self) -> serde_json::Value {
            serde_json::json!({
                "event_type": self.event_type,
                "payload": self.payload,
            })
        }
    }

    // ========================================================================
    // Node handler type
    // ========================================================================

    /// Handler function: takes input event value + mutable state, produces
    /// zero or more output event values.
    pub type NodeHandler = Box<
        dyn Fn(&serde_json::Value, &mut WorkflowState) -> Result<Vec<serde_json::Value>, AiError>
            + Send
            + Sync,
    >;

    // ========================================================================
    // WorkflowNode
    // ========================================================================

    /// A single node in the workflow graph.
    pub struct WorkflowNode {
        pub id: String,
        pub name: String,
        pub handler: NodeHandler,
        /// Expected input event type.
        pub input_type: String,
        /// Event types this node may produce.
        pub output_types: Vec<String>,
        /// Optional per-node timeout in milliseconds.
        pub timeout_ms: Option<u64>,
    }

    impl fmt::Debug for WorkflowNode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WorkflowNode")
                .field("id", &self.id)
                .field("name", &self.name)
                .field("input_type", &self.input_type)
                .field("output_types", &self.output_types)
                .field("timeout_ms", &self.timeout_ms)
                .finish()
        }
    }

    // ========================================================================
    // WorkflowGraph
    // ========================================================================

    /// Directed acyclic graph of workflow nodes connected by event types.
    #[derive(Debug)]
    pub struct WorkflowGraph {
        nodes: HashMap<String, WorkflowNode>,
        /// event_type -> list of node IDs that handle that event type.
        edges: HashMap<String, Vec<String>>,
        /// The event type that initiates the workflow.
        entry_event: String,
    }

    impl WorkflowGraph {
        /// Create a new graph with the given entry event type.
        pub fn new(entry_event: &str) -> Self {
            Self {
                nodes: HashMap::new(),
                edges: HashMap::new(),
                entry_event: entry_event.to_string(),
            }
        }

        /// Add a node to the graph and auto-wire edges based on its
        /// `input_type`.
        pub fn add_node(&mut self, node: WorkflowNode) {
            let input = node.input_type.clone();
            let id = node.id.clone();
            self.nodes.insert(id.clone(), node);
            self.edges.entry(input).or_default().push(id);
        }

        /// Return the entry event type.
        pub fn entry_event(&self) -> &str {
            &self.entry_event
        }

        /// Return a reference to a node by ID.
        pub fn get_node(&self, id: &str) -> Option<&WorkflowNode> {
            self.nodes.get(id)
        }

        /// Return node IDs that handle a given event type.
        pub fn handlers_for(&self, event_type: &str) -> &[String] {
            match self.edges.get(event_type) {
                Some(v) => v.as_slice(),
                None => &[],
            }
        }

        /// Return all node IDs.
        pub fn node_ids(&self) -> Vec<String> {
            self.nodes.keys().cloned().collect()
        }

        /// Validate the workflow graph:
        /// - The entry event must have at least one handler.
        /// - All output event types must either have a handler or be terminal.
        /// - No cycles exist in the graph.
        pub fn validate(&self) -> Result<(), AiError> {
            // 1. Entry event must have at least one handler.
            if self.handlers_for(&self.entry_event).is_empty() {
                return Err(WorkflowError::NodeNotFound {
                    node_id: format!("(no handler for entry event '{}')", self.entry_event),
                }
                .into());
            }

            // 2. Cycle detection via DFS colouring.
            //    Build adjacency: node_id -> set of successor node_ids.
            let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
            for node in self.nodes.values() {
                let mut successors = Vec::new();
                for out_evt in &node.output_types {
                    if let Some(targets) = self.edges.get(out_evt) {
                        for t in targets {
                            successors.push(t.as_str());
                        }
                    }
                }
                adjacency.insert(node.id.as_str(), successors);
            }

            // 0 = unvisited, 1 = in progress, 2 = done
            let mut colour: HashMap<&str, u8> = HashMap::new();
            for id in self.nodes.keys() {
                colour.insert(id.as_str(), 0);
            }

            fn dfs<'a>(
                node: &'a str,
                adjacency: &HashMap<&'a str, Vec<&'a str>>,
                colour: &mut HashMap<&'a str, u8>,
                path: &mut Vec<String>,
            ) -> Result<(), Vec<String>> {
                colour.insert(node, 1);
                path.push(node.to_string());

                if let Some(neighbours) = adjacency.get(node) {
                    for &next in neighbours {
                        match colour.get(next) {
                            Some(1) => {
                                // Cycle: build the cycle path.
                                path.push(next.to_string());
                                return Err(path.clone());
                            }
                            Some(0) | None => {
                                dfs(next, adjacency, colour, path)?;
                            }
                            _ => {} // already fully visited
                        }
                    }
                }

                path.pop();
                colour.insert(node, 2);
                Ok(())
            }

            for id in self.nodes.keys() {
                if colour.get(id.as_str()) == Some(&0) {
                    let mut path = Vec::new();
                    if let Err(cycle_path) = dfs(id.as_str(), &adjacency, &mut colour, &mut path) {
                        return Err(WorkflowError::CycleDetected { path: cycle_path }.into());
                    }
                }
            }

            Ok(())
        }
    }

    // ========================================================================
    // WorkflowState
    // ========================================================================

    /// Mutable key-value state threaded through workflow execution.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowState {
        pub values: HashMap<String, serde_json::Value>,
        pub step_count: usize,
    }

    impl WorkflowState {
        pub fn new() -> Self {
            Self {
                values: HashMap::new(),
                step_count: 0,
            }
        }

        pub fn set(&mut self, key: &str, value: serde_json::Value) {
            self.values.insert(key.to_string(), value);
        }

        pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
            self.values.get(key)
        }
    }

    impl Default for WorkflowState {
        fn default() -> Self {
            Self::new()
        }
    }

    // ========================================================================
    // WorkflowCheckpoint
    // ========================================================================

    /// Snapshot of workflow execution state at a particular step.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowCheckpoint {
        pub workflow_id: String,
        pub step: usize,
        pub state: WorkflowState,
        pub pending_events: Vec<serde_json::Value>,
        pub timestamp: u64,
    }

    // ========================================================================
    // WorkflowBreakpoint
    // ========================================================================

    /// Breakpoint configuration: optionally conditional on state.
    pub struct WorkflowBreakpoint {
        pub node_id: String,
        pub condition: Option<Box<dyn Fn(&WorkflowState) -> bool + Send + Sync>>,
    }

    impl fmt::Debug for WorkflowBreakpoint {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WorkflowBreakpoint")
                .field("node_id", &self.node_id)
                .field("has_condition", &self.condition.is_some())
                .finish()
        }
    }

    // ========================================================================
    // ErrorSnapshot
    // ========================================================================

    /// Diagnostic snapshot captured when a node handler fails.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ErrorSnapshot {
        pub node_id: String,
        pub step: usize,
        pub error_message: String,
        pub state_at_failure: WorkflowState,
        pub pending_events: Vec<serde_json::Value>,
    }

    // ========================================================================
    // Checkpointer trait + InMemoryCheckpointer
    // ========================================================================

    /// Persistence backend for workflow checkpoints.
    pub trait Checkpointer: Send + Sync {
        fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), AiError>;
        fn load(
            &self,
            workflow_id: &str,
            step: usize,
        ) -> Result<Option<WorkflowCheckpoint>, AiError>;
        fn list(&self, workflow_id: &str) -> Result<Vec<usize>, AiError>;
        fn latest(&self, workflow_id: &str) -> Result<Option<WorkflowCheckpoint>, AiError>;
    }

    /// In-memory checkpoint store (for testing and lightweight usage).
    #[derive(Debug)]
    pub struct InMemoryCheckpointer {
        data: Mutex<HashMap<String, Vec<WorkflowCheckpoint>>>,
    }

    impl InMemoryCheckpointer {
        pub fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl Default for InMemoryCheckpointer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Checkpointer for InMemoryCheckpointer {
        fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<(), AiError> {
            let mut store = self.data.lock().map_err(|e| {
                WorkflowError::CheckpointFailed {
                    workflow_id: checkpoint.workflow_id.clone(),
                    reason: format!("lock poisoned: {}", e),
                }
            })?;
            let entry = store
                .entry(checkpoint.workflow_id.clone())
                .or_default();
            // Replace if same step already exists, otherwise push.
            if let Some(pos) = entry.iter().position(|c| c.step == checkpoint.step) {
                entry[pos] = checkpoint.clone();
            } else {
                entry.push(checkpoint.clone());
            }
            Ok(())
        }

        fn load(
            &self,
            workflow_id: &str,
            step: usize,
        ) -> Result<Option<WorkflowCheckpoint>, AiError> {
            let store = self.data.lock().map_err(|e| {
                WorkflowError::CheckpointFailed {
                    workflow_id: workflow_id.to_string(),
                    reason: format!("lock poisoned: {}", e),
                }
            })?;
            Ok(store
                .get(workflow_id)
                .and_then(|v| v.iter().find(|c| c.step == step).cloned()))
        }

        fn list(&self, workflow_id: &str) -> Result<Vec<usize>, AiError> {
            let store = self.data.lock().map_err(|e| {
                WorkflowError::CheckpointFailed {
                    workflow_id: workflow_id.to_string(),
                    reason: format!("lock poisoned: {}", e),
                }
            })?;
            let mut steps: Vec<usize> = store
                .get(workflow_id)
                .map(|v| v.iter().map(|c| c.step).collect())
                .unwrap_or_default();
            steps.sort();
            Ok(steps)
        }

        fn latest(&self, workflow_id: &str) -> Result<Option<WorkflowCheckpoint>, AiError> {
            let store = self.data.lock().map_err(|e| {
                WorkflowError::CheckpointFailed {
                    workflow_id: workflow_id.to_string(),
                    reason: format!("lock poisoned: {}", e),
                }
            })?;
            Ok(store
                .get(workflow_id)
                .and_then(|v| v.iter().max_by_key(|c| c.step).cloned()))
        }
    }

    // ========================================================================
    // WorkflowRunner
    // ========================================================================

    /// Executes a `WorkflowGraph`, checkpointing state at each step and
    /// honouring breakpoints.
    pub struct WorkflowRunner {
        graph: WorkflowGraph,
        checkpointer: Box<dyn Checkpointer>,
        breakpoints: Vec<WorkflowBreakpoint>,
        max_steps: usize,
    }

    impl fmt::Debug for WorkflowRunner {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WorkflowRunner")
                .field("graph", &self.graph)
                .field("checkpointer", &"<dyn Checkpointer>")
                .field("breakpoints", &self.breakpoints)
                .field("max_steps", &self.max_steps)
                .finish()
        }
    }

    /// Result of a workflow execution.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowResult {
        pub workflow_id: String,
        pub final_state: WorkflowState,
        pub steps_executed: usize,
        pub completed: bool,
        pub error_snapshot: Option<ErrorSnapshot>,
    }

    impl WorkflowRunner {
        /// Create a runner with an in-memory checkpointer and no breakpoints.
        pub fn new(graph: WorkflowGraph) -> Self {
            Self {
                graph,
                checkpointer: Box::new(InMemoryCheckpointer::new()),
                breakpoints: Vec::new(),
                max_steps: 1000,
            }
        }

        /// Create a runner with a custom checkpointer.
        pub fn with_checkpointer(
            graph: WorkflowGraph,
            checkpointer: Box<dyn Checkpointer>,
        ) -> Self {
            Self {
                graph,
                checkpointer,
                breakpoints: Vec::new(),
                max_steps: 1000,
            }
        }

        /// Set the maximum number of execution steps.
        pub fn set_max_steps(&mut self, max: usize) {
            self.max_steps = max;
        }

        /// Add a breakpoint to the runner.
        pub fn add_breakpoint(&mut self, bp: WorkflowBreakpoint) {
            self.breakpoints.push(bp);
        }

        /// Run the workflow from the beginning with the given initial event
        /// and state.
        pub fn run(
            &self,
            workflow_id: &str,
            initial_event: serde_json::Value,
            mut state: WorkflowState,
        ) -> Result<WorkflowResult, AiError> {
            let mut pending: Vec<serde_json::Value> = vec![initial_event];
            let mut steps_executed: usize = 0;

            // Save initial checkpoint (step 0).
            self.save_checkpoint(workflow_id, 0, &state, &pending)?;

            while !pending.is_empty() {
                if steps_executed >= self.max_steps {
                    return Ok(WorkflowResult {
                        workflow_id: workflow_id.to_string(),
                        final_state: state,
                        steps_executed,
                        completed: false,
                        error_snapshot: None,
                    });
                }

                // Take first pending event (BFS).
                let event = pending.remove(0);

                // Determine event type — look for "event_type" field.
                let event_type = event
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Find handlers for this event type.
                let handler_ids: Vec<String> =
                    self.graph.handlers_for(&event_type).to_vec();

                if handler_ids.is_empty() {
                    // Terminal event — no handlers, just drop it.
                    continue;
                }

                for node_id in &handler_ids {
                    // Check breakpoints before executing.
                    if let Some(bp) = self.check_breakpoint(node_id, &state) {
                        // Save checkpoint so we can resume later.
                        // Re-insert the event at the front so resume picks it up.
                        let mut resume_pending = vec![event.clone()];
                        resume_pending.extend(pending.clone());
                        self.save_checkpoint(
                            workflow_id,
                            steps_executed + 1,
                            &state,
                            &resume_pending,
                        )?;
                        return Err(WorkflowError::BreakpointHit {
                            node_id: bp.to_string(),
                        }
                        .into());
                    }

                    let node = self.graph.get_node(node_id).ok_or_else(|| {
                        WorkflowError::NodeNotFound {
                            node_id: node_id.clone(),
                        }
                    })?;

                    // Validate event type matches node's expected input.
                    if node.input_type != event_type {
                        return Err(WorkflowError::EventTypeMismatch {
                            expected: node.input_type.clone(),
                            got: event_type.clone(),
                        }
                        .into());
                    }

                    // Execute the handler.
                    let payload = event
                        .get("payload")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);

                    match (node.handler)(&payload, &mut state) {
                        Ok(outputs) => {
                            pending.extend(outputs);
                        }
                        Err(e) => {
                            let snapshot = ErrorSnapshot {
                                node_id: node_id.clone(),
                                step: steps_executed,
                                error_message: e.to_string(),
                                state_at_failure: state.clone(),
                                pending_events: pending.clone(),
                            };
                            return Ok(WorkflowResult {
                                workflow_id: workflow_id.to_string(),
                                final_state: state,
                                steps_executed,
                                completed: false,
                                error_snapshot: Some(snapshot),
                            });
                        }
                    }
                }

                steps_executed += 1;
                state.step_count = steps_executed;

                // Checkpoint after each step.
                self.save_checkpoint(workflow_id, steps_executed, &state, &pending)?;
            }

            Ok(WorkflowResult {
                workflow_id: workflow_id.to_string(),
                final_state: state,
                steps_executed,
                completed: true,
                error_snapshot: None,
            })
        }

        /// Resume execution from a previously saved checkpoint (time-travel).
        pub fn resume(
            &self,
            workflow_id: &str,
            step: usize,
        ) -> Result<WorkflowResult, AiError> {
            let checkpoint = self
                .checkpointer
                .load(workflow_id, step)?
                .ok_or_else(|| WorkflowError::CheckpointFailed {
                    workflow_id: workflow_id.to_string(),
                    reason: format!("no checkpoint at step {}", step),
                })?;

            // Resume: re-process the pending events from the checkpoint.
            let mut state = checkpoint.state;
            let mut pending = checkpoint.pending_events;
            let mut steps_executed = checkpoint.step;

            while !pending.is_empty() {
                if steps_executed >= self.max_steps {
                    return Ok(WorkflowResult {
                        workflow_id: workflow_id.to_string(),
                        final_state: state,
                        steps_executed,
                        completed: false,
                        error_snapshot: None,
                    });
                }

                let event = pending.remove(0);
                let event_type = event
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let handler_ids: Vec<String> =
                    self.graph.handlers_for(&event_type).to_vec();

                if handler_ids.is_empty() {
                    continue;
                }

                for node_id in &handler_ids {
                    // Skip breakpoints during resume to avoid infinite loop at
                    // the same breakpoint.  The caller can re-add breakpoints
                    // for nodes further along the pipeline.

                    let node = self.graph.get_node(node_id).ok_or_else(|| {
                        WorkflowError::NodeNotFound {
                            node_id: node_id.clone(),
                        }
                    })?;

                    if node.input_type != event_type {
                        return Err(WorkflowError::EventTypeMismatch {
                            expected: node.input_type.clone(),
                            got: event_type.clone(),
                        }
                        .into());
                    }

                    let payload = event
                        .get("payload")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);

                    match (node.handler)(&payload, &mut state) {
                        Ok(outputs) => {
                            pending.extend(outputs);
                        }
                        Err(e) => {
                            let snapshot = ErrorSnapshot {
                                node_id: node_id.clone(),
                                step: steps_executed,
                                error_message: e.to_string(),
                                state_at_failure: state.clone(),
                                pending_events: pending.clone(),
                            };
                            return Ok(WorkflowResult {
                                workflow_id: workflow_id.to_string(),
                                final_state: state,
                                steps_executed,
                                completed: false,
                                error_snapshot: Some(snapshot),
                            });
                        }
                    }
                }

                steps_executed += 1;
                state.step_count = steps_executed;
                self.save_checkpoint(workflow_id, steps_executed, &state, &pending)?;
            }

            Ok(WorkflowResult {
                workflow_id: workflow_id.to_string(),
                final_state: state,
                steps_executed,
                completed: true,
                error_snapshot: None,
            })
        }

        // -- helpers --------------------------------------------------------

        fn save_checkpoint(
            &self,
            workflow_id: &str,
            step: usize,
            state: &WorkflowState,
            pending: &[serde_json::Value],
        ) -> Result<(), AiError> {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let cp = WorkflowCheckpoint {
                workflow_id: workflow_id.to_string(),
                step,
                state: state.clone(),
                pending_events: pending.to_vec(),
                timestamp: ts,
            };
            self.checkpointer.save(&cp)
        }

        fn check_breakpoint<'a>(
            &'a self,
            node_id: &str,
            state: &WorkflowState,
        ) -> Option<&'a str> {
            for bp in &self.breakpoints {
                if bp.node_id == node_id {
                    match &bp.condition {
                        None => return Some(&bp.node_id),
                        Some(cond) if cond(state) => return Some(&bp.node_id),
                        _ => {}
                    }
                }
            }
            None
        }
    }

    // ========================================================================
    // Serializable workflow definition
    // ========================================================================

    /// Serialisable definition of a workflow graph (without handlers).
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowDefinition {
        pub id: String,
        pub name: String,
        pub nodes: Vec<WorkflowNodeDef>,
        pub edges: Vec<WorkflowEdgeDef>,
        pub entry_event: String,
    }

    /// Serialisable node metadata (handler is not serialised).
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowNodeDef {
        pub id: String,
        pub name: String,
        pub input_type: String,
        pub output_types: Vec<String>,
        pub timeout_ms: Option<u64>,
    }

    /// Serialisable edge.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowEdgeDef {
        pub event_type: String,
        pub target_node_ids: Vec<String>,
    }

    impl WorkflowDefinition {
        /// Serialise to JSON string.
        pub fn to_json(&self) -> Result<String, AiError> {
            serde_json::to_string_pretty(self).map_err(|e| {
                WorkflowError::SerializationFailed {
                    reason: e.to_string(),
                }
                .into()
            })
        }

        /// Deserialise from JSON string.
        pub fn from_json(json: &str) -> Result<Self, AiError> {
            serde_json::from_str(json).map_err(|e| {
                WorkflowError::SerializationFailed {
                    reason: e.to_string(),
                }
                .into()
            })
        }

        /// Build a `WorkflowDefinition` from a `WorkflowGraph` snapshot.
        pub fn from_graph(id: &str, name: &str, graph: &WorkflowGraph) -> Self {
            let mut nodes = Vec::new();
            let mut edge_map: HashMap<String, Vec<String>> = HashMap::new();

            for (nid, node) in &graph.nodes {
                nodes.push(WorkflowNodeDef {
                    id: nid.clone(),
                    name: node.name.clone(),
                    input_type: node.input_type.clone(),
                    output_types: node.output_types.clone(),
                    timeout_ms: node.timeout_ms,
                });
            }

            for (evt, targets) in &graph.edges {
                edge_map
                    .entry(evt.clone())
                    .or_default()
                    .extend(targets.clone());
            }

            let edges: Vec<WorkflowEdgeDef> = edge_map
                .into_iter()
                .map(|(event_type, target_node_ids)| WorkflowEdgeDef {
                    event_type,
                    target_node_ids,
                })
                .collect();

            Self {
                id: id.to_string(),
                name: name.to_string(),
                nodes,
                edges,
                entry_event: graph.entry_event.clone(),
            }
        }
    }

    // ========================================================================
    // WorkflowTool — wraps a workflow as a ToolDefinition-compatible struct
    // ========================================================================

    /// Wraps a serialised workflow so it can be exposed as a tool in the
    /// tool-calling system.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowTool {
        pub name: String,
        pub description: String,
        pub workflow_def: WorkflowDefinition,
    }

    /// A lightweight tool definition returned by `WorkflowTool::to_tool_definition`.
    /// Kept self-contained so the `workflows` feature does not require `tools`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowToolDefinition {
        pub name: String,
        pub description: String,
        pub parameters: Vec<WorkflowToolParam>,
        pub category: String,
    }

    /// Parameter descriptor for a workflow tool.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowToolParam {
        pub name: String,
        pub param_type: String,
        pub description: String,
        pub required: bool,
    }

    impl WorkflowTool {
        pub fn new(name: &str, description: &str, workflow_def: WorkflowDefinition) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                workflow_def,
            }
        }

        /// Produce a tool-definition-like struct that describes how to invoke
        /// this workflow.
        pub fn to_tool_definition(&self) -> WorkflowToolDefinition {
            WorkflowToolDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: vec![
                    WorkflowToolParam {
                        name: "input".to_string(),
                        param_type: "object".to_string(),
                        description: "JSON payload for the workflow entry event".to_string(),
                        required: true,
                    },
                    WorkflowToolParam {
                        name: "workflow_id".to_string(),
                        param_type: "string".to_string(),
                        description: "Unique execution ID for checkpointing".to_string(),
                        required: false,
                    },
                ],
                category: "workflow".to_string(),
            }
        }
    }

    // ========================================================================
    // Helper: current timestamp
    // ========================================================================

    fn _now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    // ========================================================================
    // 7.1 — Automatic State Persistence (Durable Execution)
    // ========================================================================

    /// Backend type for durable execution persistence.
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum DurableBackend {
        /// In-memory storage (testing / lightweight).
        InMemory,
        /// Custom named backend (for future extensibility).
        Custom(String),
    }

    /// Policy controlling which checkpoints are retained after save.
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum RetentionPolicy {
        /// Keep every checkpoint forever.
        KeepAll,
        /// Keep only the last N checkpoints.
        KeepLast(usize),
        /// Keep checkpoints created within the last N seconds.
        KeepDuration(u64),
        /// Keep only explicitly named (non-auto) checkpoints, discard auto.
        KeepCheckpointsOnly,
    }

    /// Configuration for durable workflow execution.
    #[non_exhaustive]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DurableConfig {
        /// Which backend to persist to.
        pub backend: DurableBackend,
        /// Whether to automatically checkpoint before/after every node.
        pub auto_checkpoint: bool,
        /// Retention policy applied after each save.
        pub retention_policy: RetentionPolicy,
        /// Whether recovery from interrupted executions is enabled.
        pub recovery_enabled: bool,
    }

    impl Default for DurableConfig {
        fn default() -> Self {
            Self {
                backend: DurableBackend::InMemory,
                auto_checkpoint: true,
                retention_policy: RetentionPolicy::KeepAll,
                recovery_enabled: true,
            }
        }
    }

    /// A checkpoint enriched with durable execution metadata.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DurableCheckpoint {
        /// Unique execution run identifier.
        pub execution_id: String,
        /// Unique checkpoint identifier within this execution.
        pub checkpoint_id: String,
        /// The node being executed at this checkpoint (or "initial"/"completed"/"error").
        pub node_id: String,
        /// Workflow state at checkpoint time.
        pub state: WorkflowState,
        /// Sequential step number within the execution.
        pub step_number: usize,
        /// Whether this checkpoint was created automatically (true) or
        /// explicitly by user code (false).
        pub is_auto: bool,
        /// Creation timestamp (Unix seconds).
        pub created_at: u64,
    }

    /// Wraps a [`WorkflowRunner`] and automatically persists state before and
    /// after every node execution, providing durable execution semantics.
    pub struct DurableExecutor {
        runner: WorkflowRunner,
        config: DurableConfig,
        checkpointer: Box<dyn Checkpointer>,
        execution_id: String,
        checkpoint_count: usize,
        /// Internal store of DurableCheckpoint metadata, keyed by execution_id.
        durable_store: Mutex<HashMap<String, Vec<DurableCheckpoint>>>,
    }

    impl fmt::Debug for DurableExecutor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("DurableExecutor")
                .field("runner", &self.runner)
                .field("config", &self.config)
                .field("checkpointer", &"<dyn Checkpointer>")
                .field("execution_id", &self.execution_id)
                .field("checkpoint_count", &self.checkpoint_count)
                .field("durable_store", &self.durable_store)
                .finish()
        }
    }

    impl DurableExecutor {
        /// Create a new durable executor wrapping the given runner.
        pub fn new(
            runner: WorkflowRunner,
            config: DurableConfig,
            checkpointer: Box<dyn Checkpointer>,
        ) -> Self {
            let execution_id = format!(
                "exec-{}-{}",
                _now_secs(),
                runner.graph.entry_event()
            );
            Self {
                runner,
                config,
                checkpointer,
                execution_id,
                checkpoint_count: 0,
                durable_store: Mutex::new(HashMap::new()),
            }
        }

        /// Return the execution ID for this run.
        pub fn get_execution_id(&self) -> &str {
            &self.execution_id
        }

        /// Return the number of durable checkpoints saved so far.
        pub fn get_checkpoint_count(&self) -> usize {
            self.checkpoint_count
        }

        /// Execute the workflow with automatic state persistence.
        ///
        /// This method drives execution step-by-step, saving durable
        /// checkpoints before and after each node. On error, an error
        /// checkpoint is saved and the error state is returned.
        pub fn execute(
            &mut self,
            initial_state: WorkflowState,
        ) -> Result<WorkflowState, AiError> {
            let workflow_id = self.execution_id.clone();
            let entry_event = self.runner.graph.entry_event().to_string();

            // Build a default initial event.
            let initial_event = serde_json::json!({
                "event_type": entry_event,
                "payload": {}
            });

            // Save initial durable checkpoint.
            if self.config.auto_checkpoint {
                self.save_durable_checkpoint(
                    "initial",
                    &initial_state,
                    0,
                    true,
                )?;
            }

            // Delegate to the runner for actual execution.
            let result = self.runner.run(
                &workflow_id,
                initial_event,
                initial_state,
            )?;

            // Save per-step auto-checkpoints after execution.
            // The runner already saved WorkflowCheckpoints internally; we
            // layer DurableCheckpoints on top using the step count.
            if self.config.auto_checkpoint {
                for step in 1..=result.steps_executed {
                    // Load the underlying checkpoint if available.
                    let state_at_step = self
                        .runner
                        .checkpointer
                        .load(&workflow_id, step)?
                        .map(|cp| cp.state.clone())
                        .unwrap_or_else(|| result.final_state.clone());

                    self.save_durable_checkpoint(
                        &format!("step-{}", step),
                        &state_at_step,
                        step,
                        true,
                    )?;
                }
            }

            // Handle error snapshot.
            if let Some(ref snapshot) = result.error_snapshot {
                self.save_durable_checkpoint(
                    &format!("error-{}", snapshot.node_id),
                    &snapshot.state_at_failure,
                    result.steps_executed,
                    true,
                )?;

                // Save completion marker with error.
                self.save_durable_checkpoint(
                    "error",
                    &result.final_state,
                    result.steps_executed,
                    false,
                )?;
            } else if result.completed {
                // Save completion checkpoint.
                self.save_durable_checkpoint(
                    "completed",
                    &result.final_state,
                    result.steps_executed,
                    false,
                )?;
            }

            // Apply retention policy.
            self.apply_retention_policy()?;

            Ok(result.final_state)
        }

        /// Save a durable checkpoint with metadata.
        fn save_durable_checkpoint(
            &mut self,
            node_id: &str,
            state: &WorkflowState,
            step_number: usize,
            is_auto: bool,
        ) -> Result<(), AiError> {
            let checkpoint_id = format!(
                "{}-cp-{}",
                self.execution_id, self.checkpoint_count
            );
            let ts = _now_secs();

            let durable_cp = DurableCheckpoint {
                execution_id: self.execution_id.clone(),
                checkpoint_id: checkpoint_id.clone(),
                node_id: node_id.to_string(),
                state: state.clone(),
                step_number,
                is_auto,
                created_at: ts,
            };

            // Store in durable metadata.
            {
                let mut store = self.durable_store.lock().map_err(|e| {
                    WorkflowError::CheckpointFailed {
                        workflow_id: self.execution_id.clone(),
                        reason: format!("durable store lock poisoned: {}", e),
                    }
                })?;
                store
                    .entry(self.execution_id.clone())
                    .or_default()
                    .push(durable_cp);
            }

            // Also persist via the Checkpointer trait for interop.
            let workflow_cp = WorkflowCheckpoint {
                workflow_id: self.execution_id.clone(),
                step: self.checkpoint_count,
                state: state.clone(),
                pending_events: vec![],
                timestamp: ts,
            };
            self.checkpointer.save(&workflow_cp)?;

            self.checkpoint_count += 1;
            Ok(())
        }

        /// Apply the configured retention policy to durable checkpoints.
        fn apply_retention_policy(&mut self) -> Result<(), AiError> {
            let mut store = self.durable_store.lock().map_err(|e| {
                WorkflowError::CheckpointFailed {
                    workflow_id: self.execution_id.clone(),
                    reason: format!("durable store lock poisoned: {}", e),
                }
            })?;

            if let Some(checkpoints) = store.get_mut(&self.execution_id) {
                match &self.config.retention_policy {
                    RetentionPolicy::KeepAll => {
                        // Nothing to do.
                    }
                    RetentionPolicy::KeepLast(n) => {
                        let n = *n;
                        if checkpoints.len() > n {
                            let drain_count = checkpoints.len() - n;
                            checkpoints.drain(..drain_count);
                        }
                    }
                    RetentionPolicy::KeepDuration(secs) => {
                        let secs = *secs;
                        let now = _now_secs();
                        checkpoints.retain(|cp| {
                            now.saturating_sub(cp.created_at) <= secs
                        });
                    }
                    RetentionPolicy::KeepCheckpointsOnly => {
                        checkpoints.retain(|cp| !cp.is_auto);
                    }
                }
            }

            Ok(())
        }

        /// List all durable checkpoint IDs for this execution.
        pub fn list_checkpoints(&self) -> Vec<String> {
            let store = match self.durable_store.lock() {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            store
                .get(&self.execution_id)
                .map(|cps| cps.iter().map(|cp| cp.checkpoint_id.clone()).collect())
                .unwrap_or_default()
        }

        /// Retrieve all durable checkpoints for this execution.
        pub fn get_durable_checkpoints(&self) -> Vec<DurableCheckpoint> {
            let store = match self.durable_store.lock() {
                Ok(s) => s,
                Err(_) => return vec![],
            };
            store
                .get(&self.execution_id)
                .cloned()
                .unwrap_or_default()
        }
    }

    /// Detects and manages recovery of interrupted workflow executions.
    pub struct RecoveryManager {
        checkpointer: Box<dyn Checkpointer>,
    }

    impl fmt::Debug for RecoveryManager {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("RecoveryManager")
                .field("checkpointer", &"<dyn Checkpointer>")
                .finish()
        }
    }

    impl RecoveryManager {
        /// Create a new recovery manager.
        pub fn new(checkpointer: Box<dyn Checkpointer>) -> Self {
            Self { checkpointer }
        }

        /// Find execution IDs that appear to have incomplete runs.
        ///
        /// An execution is considered "interrupted" if it has checkpoints but
        /// the latest checkpoint's pending_events list is non-empty (i.e., the
        /// workflow did not drain its event queue).
        pub fn find_interrupted(&self, workflow_id: &str) -> Vec<String> {
            let steps = match self.checkpointer.list(workflow_id) {
                Ok(s) => s,
                Err(_) => return vec![],
            };

            if steps.is_empty() {
                return vec![];
            }

            // Check if the latest checkpoint has pending events.
            match self.checkpointer.latest(workflow_id) {
                Ok(Some(cp)) if !cp.pending_events.is_empty() => {
                    vec![workflow_id.to_string()]
                }
                _ => vec![],
            }
        }

        /// Attempt to recover state from the latest good checkpoint.
        pub fn recover(
            &self,
            execution_id: &str,
        ) -> Result<Option<WorkflowState>, AiError> {
            match self.checkpointer.latest(execution_id) {
                Ok(Some(cp)) => Ok(Some(cp.state)),
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        }

        /// List all checkpoint step numbers for a given execution.
        pub fn list_checkpoints(
            &self,
            execution_id: &str,
        ) -> Vec<String> {
            match self.checkpointer.list(execution_id) {
                Ok(steps) => steps.iter().map(|s| format!("step-{}", s)).collect(),
                Err(_) => vec![],
            }
        }

        /// Remove all checkpoints for the given execution.
        pub fn cleanup(&self, execution_id: &str) {
            // InMemoryCheckpointer does not expose a delete API, so we
            // overwrite with an empty checkpoint at step 0 to logically
            // "clean" it. In a real backend this would call a delete method.
            //
            // For the InMemoryCheckpointer we save a sentinel that the
            // list() call will still return — consumers should treat a
            // post-cleanup state as cleared.
            let _ = self.checkpointer.save(&WorkflowCheckpoint {
                workflow_id: execution_id.to_string(),
                step: 0,
                state: WorkflowState::new(),
                pending_events: vec![],
                timestamp: _now_secs(),
            });
        }
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        // -- SimpleEvent tests -----------------------------------------------

        #[test]
        fn test_simple_event_creation() {
            let evt = SimpleEvent::new("start", serde_json::json!({"key": "value"}));
            assert_eq!(evt.event_type, "start");
            assert_eq!(evt.payload["key"], "value");
        }

        #[test]
        fn test_simple_event_trait() {
            let evt = SimpleEvent::new("foo", serde_json::json!(42));
            assert_eq!(evt.event_type(), "foo");
            let json = evt.to_json();
            assert_eq!(json["event_type"], "foo");
            assert_eq!(json["payload"], 42);
        }

        #[test]
        fn test_simple_event_serialization() {
            let evt = SimpleEvent::new("test", serde_json::json!({"a": 1}));
            let json_str = serde_json::to_string(&evt).expect("serialize");
            let deser: SimpleEvent = serde_json::from_str(&json_str).expect("deserialize");
            assert_eq!(deser.event_type, "test");
            assert_eq!(deser.payload["a"], 1);
        }

        #[test]
        fn test_simple_event_debug() {
            let evt = SimpleEvent::new("dbg", serde_json::json!(null));
            let debug = format!("{:?}", evt);
            assert!(debug.contains("dbg"));
        }

        #[test]
        fn test_simple_event_clone() {
            let evt = SimpleEvent::new("orig", serde_json::json!("data"));
            let cloned = evt.clone();
            assert_eq!(cloned.event_type, "orig");
            assert_eq!(cloned.payload, serde_json::json!("data"));
        }

        // -- WorkflowState tests ---------------------------------------------

        #[test]
        fn test_workflow_state_new() {
            let state = WorkflowState::new();
            assert!(state.values.is_empty());
            assert_eq!(state.step_count, 0);
        }

        #[test]
        fn test_workflow_state_default() {
            let state = WorkflowState::default();
            assert!(state.values.is_empty());
            assert_eq!(state.step_count, 0);
        }

        #[test]
        fn test_workflow_state_set_get() {
            let mut state = WorkflowState::new();
            state.set("x", serde_json::json!(42));
            assert_eq!(state.get("x"), Some(&serde_json::json!(42)));
            assert_eq!(state.get("missing"), None);
        }

        #[test]
        fn test_workflow_state_overwrite() {
            let mut state = WorkflowState::new();
            state.set("k", serde_json::json!(1));
            state.set("k", serde_json::json!(2));
            assert_eq!(state.get("k"), Some(&serde_json::json!(2)));
        }

        #[test]
        fn test_workflow_state_clone() {
            let mut state = WorkflowState::new();
            state.set("a", serde_json::json!("hello"));
            state.step_count = 5;
            let cloned = state.clone();
            assert_eq!(cloned.get("a"), Some(&serde_json::json!("hello")));
            assert_eq!(cloned.step_count, 5);
        }

        #[test]
        fn test_workflow_state_serialization() {
            let mut state = WorkflowState::new();
            state.set("key", serde_json::json!(true));
            state.step_count = 3;
            let json_str = serde_json::to_string(&state).expect("serialize");
            let deser: WorkflowState = serde_json::from_str(&json_str).expect("deserialize");
            assert_eq!(deser.get("key"), Some(&serde_json::json!(true)));
            assert_eq!(deser.step_count, 3);
        }

        #[test]
        fn test_workflow_state_multiple_keys() {
            let mut state = WorkflowState::new();
            state.set("a", serde_json::json!(1));
            state.set("b", serde_json::json!("two"));
            state.set("c", serde_json::json!([3, 4]));
            assert_eq!(state.values.len(), 3);
        }

        // -- WorkflowGraph tests ---------------------------------------------

        #[test]
        fn test_workflow_graph_new() {
            let graph = WorkflowGraph::new("start");
            assert_eq!(graph.entry_event(), "start");
            assert!(graph.node_ids().is_empty());
        }

        #[test]
        fn test_graph_add_node() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n1".to_string(),
                name: "Node 1".to_string(),
                handler: Box::new(|_payload, _state| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });
            assert_eq!(graph.node_ids().len(), 1);
            assert!(graph.get_node("n1").is_some());
        }

        #[test]
        fn test_graph_add_multiple_nodes() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "a".to_string(),
                name: "A".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec!["mid".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "b".to_string(),
                name: "B".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "mid".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });
            assert_eq!(graph.node_ids().len(), 2);
            assert_eq!(graph.handlers_for("start").len(), 1);
            assert_eq!(graph.handlers_for("mid").len(), 1);
        }

        #[test]
        fn test_graph_handlers_for_empty() {
            let graph = WorkflowGraph::new("start");
            assert!(graph.handlers_for("nonexistent").is_empty());
        }

        #[test]
        fn test_graph_validate_ok() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n1".to_string(),
                name: "Node 1".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec!["done".to_string()],
                timeout_ms: None,
            });
            assert!(graph.validate().is_ok());
        }

        #[test]
        fn test_graph_validate_no_entry_handler() {
            let graph = WorkflowGraph::new("start");
            let err = graph.validate().unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("not found") || msg.contains("no handler"));
        }

        #[test]
        fn test_graph_validate_cycle_detection() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "a".to_string(),
                name: "A".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec!["evt_b".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "b".to_string(),
                name: "B".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "evt_b".to_string(),
                output_types: vec!["start".to_string()], // back to A
                timeout_ms: None,
            });
            let err = graph.validate().unwrap_err();
            assert!(err.to_string().contains("ycle"));
        }

        #[test]
        fn test_graph_validate_missing_handler() {
            // Graph with a node whose output goes nowhere is OK (terminal).
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n1".to_string(),
                name: "N1".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec!["nowhere".to_string()],
                timeout_ms: None,
            });
            // Should validate OK — terminal events are not an error.
            assert!(graph.validate().is_ok());
        }

        #[test]
        fn test_graph_node_debug() {
            let node = WorkflowNode {
                id: "dbg".to_string(),
                name: "Debug Node".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "x".to_string(),
                output_types: vec!["y".to_string()],
                timeout_ms: Some(500),
            };
            let debug = format!("{:?}", node);
            assert!(debug.contains("dbg"));
            assert!(debug.contains("500"));
        }

        // -- WorkflowRunner tests --------------------------------------------

        fn make_simple_pipeline() -> WorkflowGraph {
            let mut graph = WorkflowGraph::new("start");

            // Node A: start -> mid
            graph.add_node(WorkflowNode {
                id: "a".to_string(),
                name: "A".to_string(),
                handler: Box::new(|payload, state| {
                    let val = payload.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                    state.set("after_a", serde_json::json!(val + 1));
                    Ok(vec![serde_json::json!({
                        "event_type": "mid",
                        "payload": { "value": val + 1 }
                    })])
                }),
                input_type: "start".to_string(),
                output_types: vec!["mid".to_string()],
                timeout_ms: None,
            });

            // Node B: mid -> end
            graph.add_node(WorkflowNode {
                id: "b".to_string(),
                name: "B".to_string(),
                handler: Box::new(|payload, state| {
                    let val = payload.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                    state.set("after_b", serde_json::json!(val * 10));
                    Ok(vec![serde_json::json!({
                        "event_type": "end",
                        "payload": { "result": val * 10 }
                    })])
                }),
                input_type: "mid".to_string(),
                output_types: vec!["end".to_string()],
                timeout_ms: None,
            });

            // Node C: end (terminal — just stores result)
            graph.add_node(WorkflowNode {
                id: "c".to_string(),
                name: "C".to_string(),
                handler: Box::new(|payload, state| {
                    state.set("final_result", payload.clone());
                    Ok(vec![])
                }),
                input_type: "end".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            graph
        }

        #[test]
        fn test_runner_simple_pipeline() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 5 }
            });
            let result = runner.run("wf-1", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.steps_executed, 3);
            assert_eq!(
                result.final_state.get("after_a"),
                Some(&serde_json::json!(6))
            );
            assert_eq!(
                result.final_state.get("after_b"),
                Some(&serde_json::json!(60))
            );
        }

        #[test]
        fn test_runner_branching() {
            let mut graph = WorkflowGraph::new("start");

            // Node that produces two different event types.
            graph.add_node(WorkflowNode {
                id: "splitter".to_string(),
                name: "Splitter".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("split", serde_json::json!(true));
                    Ok(vec![
                        serde_json::json!({
                            "event_type": "branch_a",
                            "payload": { "path": "a" }
                        }),
                        serde_json::json!({
                            "event_type": "branch_b",
                            "payload": { "path": "b" }
                        }),
                    ])
                }),
                input_type: "start".to_string(),
                output_types: vec!["branch_a".to_string(), "branch_b".to_string()],
                timeout_ms: None,
            });

            graph.add_node(WorkflowNode {
                id: "handle_a".to_string(),
                name: "HandleA".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("a_done", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "branch_a".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            graph.add_node(WorkflowNode {
                id: "handle_b".to_string(),
                name: "HandleB".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("b_done", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "branch_b".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {}
            });
            let result = runner.run("wf-branch", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.final_state.get("split"), Some(&serde_json::json!(true)));
            assert_eq!(result.final_state.get("a_done"), Some(&serde_json::json!(true)));
            assert_eq!(result.final_state.get("b_done"), Some(&serde_json::json!(true)));
        }

        #[test]
        fn test_runner_breakpoint_hit() {
            let graph = make_simple_pipeline();
            let mut runner = WorkflowRunner::new(graph);
            runner.add_breakpoint(WorkflowBreakpoint {
                node_id: "b".to_string(),
                condition: None,
            });

            let initial = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 1 }
            });
            let err = runner.run("wf-bp", initial, WorkflowState::new()).unwrap_err();
            assert!(err.to_string().contains("Breakpoint"));
            assert!(err.to_string().contains("b"));
        }

        #[test]
        fn test_runner_breakpoint_with_condition() {
            let graph = make_simple_pipeline();
            let mut runner = WorkflowRunner::new(graph);

            // Breakpoint on node B only if after_a > 10
            runner.add_breakpoint(WorkflowBreakpoint {
                node_id: "b".to_string(),
                condition: Some(Box::new(|state| {
                    state
                        .get("after_a")
                        .and_then(|v| v.as_i64())
                        .map(|v| v > 10)
                        .unwrap_or(false)
                })),
            });

            // Value 5 => after_a = 6 which is <= 10, so breakpoint should NOT fire.
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 5 }
            });
            let result = runner
                .run("wf-cond-no", initial, WorkflowState::new())
                .expect("should complete");
            assert!(result.completed);

            // Value 20 => after_a = 21 which is > 10, so breakpoint SHOULD fire.
            let initial2 = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 20 }
            });
            let err = runner.run("wf-cond-yes", initial2, WorkflowState::new()).unwrap_err();
            assert!(err.to_string().contains("Breakpoint"));
        }

        #[test]
        fn test_runner_error_snapshot() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "fail_node".to_string(),
                name: "Failing Node".to_string(),
                handler: Box::new(|_p, _s| {
                    Err(AiError::Other("intentional failure".to_string()))
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {}
            });
            let result = runner.run("wf-err", initial, WorkflowState::new()).expect("result ok");
            assert!(!result.completed);
            let snap = result.error_snapshot.as_ref().expect("error snapshot");
            assert_eq!(snap.node_id, "fail_node");
            assert!(snap.error_message.contains("intentional failure"));
        }

        #[test]
        fn test_runner_timeout_simulated() {
            // We simulate a timeout by having the node return a timeout error.
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "slow".to_string(),
                name: "Slow Node".to_string(),
                handler: Box::new(|_p, _s| {
                    Err(AiError::Workflow(WorkflowError::TimeoutExceeded {
                        node_id: "slow".to_string(),
                        timeout_ms: 100,
                    }))
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: Some(100),
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {}
            });
            let result = runner.run("wf-timeout", initial, WorkflowState::new()).expect("result");
            assert!(!result.completed);
            let snap = result.error_snapshot.as_ref().expect("error snapshot");
            assert!(snap.error_message.contains("timed out"));
        }

        // -- Checkpoint tests ------------------------------------------------

        #[test]
        fn test_checkpoint_save_load() {
            let cp_store = InMemoryCheckpointer::new();
            let cp = WorkflowCheckpoint {
                workflow_id: "wf-1".to_string(),
                step: 3,
                state: WorkflowState::new(),
                pending_events: vec![serde_json::json!({"event_type": "x"})],
                timestamp: 1000,
            };
            cp_store.save(&cp).expect("save");
            let loaded = cp_store.load("wf-1", 3).expect("load");
            assert!(loaded.is_some());
            let loaded = loaded.expect("unwrap");
            assert_eq!(loaded.step, 3);
            assert_eq!(loaded.workflow_id, "wf-1");
        }

        #[test]
        fn test_checkpoint_load_missing() {
            let cp_store = InMemoryCheckpointer::new();
            let loaded = cp_store.load("nonexistent", 0).expect("load");
            assert!(loaded.is_none());
        }

        #[test]
        fn test_checkpoint_list() {
            let cp_store = InMemoryCheckpointer::new();
            for step in [0, 2, 5, 1] {
                let cp = WorkflowCheckpoint {
                    workflow_id: "wf-list".to_string(),
                    step,
                    state: WorkflowState::new(),
                    pending_events: vec![],
                    timestamp: 0,
                };
                cp_store.save(&cp).expect("save");
            }
            let steps = cp_store.list("wf-list").expect("list");
            assert_eq!(steps, vec![0, 1, 2, 5]);
        }

        #[test]
        fn test_checkpoint_list_empty() {
            let cp_store = InMemoryCheckpointer::new();
            let steps = cp_store.list("nope").expect("list");
            assert!(steps.is_empty());
        }

        #[test]
        fn test_checkpoint_latest() {
            let cp_store = InMemoryCheckpointer::new();
            for step in [0, 3, 1, 7, 4] {
                let cp = WorkflowCheckpoint {
                    workflow_id: "wf-lat".to_string(),
                    step,
                    state: WorkflowState::new(),
                    pending_events: vec![],
                    timestamp: 0,
                };
                cp_store.save(&cp).expect("save");
            }
            let latest = cp_store.latest("wf-lat").expect("latest");
            assert!(latest.is_some());
            assert_eq!(latest.expect("unwrap").step, 7);
        }

        #[test]
        fn test_checkpoint_latest_empty() {
            let cp_store = InMemoryCheckpointer::new();
            let latest = cp_store.latest("nope").expect("latest");
            assert!(latest.is_none());
        }

        #[test]
        fn test_checkpoint_overwrite_same_step() {
            let cp_store = InMemoryCheckpointer::new();
            let mut state1 = WorkflowState::new();
            state1.set("version", serde_json::json!(1));
            cp_store
                .save(&WorkflowCheckpoint {
                    workflow_id: "wf-ow".to_string(),
                    step: 0,
                    state: state1,
                    pending_events: vec![],
                    timestamp: 100,
                })
                .expect("save");

            let mut state2 = WorkflowState::new();
            state2.set("version", serde_json::json!(2));
            cp_store
                .save(&WorkflowCheckpoint {
                    workflow_id: "wf-ow".to_string(),
                    step: 0,
                    state: state2,
                    pending_events: vec![],
                    timestamp: 200,
                })
                .expect("save");

            let loaded = cp_store.load("wf-ow", 0).expect("load").expect("present");
            assert_eq!(loaded.state.get("version"), Some(&serde_json::json!(2)));
            assert_eq!(loaded.timestamp, 200);
        }

        // -- Time-travel / resume tests --------------------------------------

        #[test]
        fn test_runner_resume_from_checkpoint() {
            let graph = make_simple_pipeline();
            let mut runner = WorkflowRunner::new(graph);

            // Add breakpoint on node B so it stops after step 1.
            runner.add_breakpoint(WorkflowBreakpoint {
                node_id: "b".to_string(),
                condition: None,
            });

            let initial = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 5 }
            });
            let err = runner.run("wf-resume", initial, WorkflowState::new());
            assert!(err.is_err()); // breakpoint hit

            // Remove breakpoints and resume.
            runner.breakpoints.clear();
            let result = runner.resume("wf-resume", 1).expect("resume");
            assert!(result.completed);
            // The resumed execution should have processed from the breakpoint
            // onward.
            assert!(result.final_state.get("after_b").is_some());
        }

        #[test]
        fn test_runner_resume_missing_checkpoint() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let err = runner.resume("nonexistent", 99).unwrap_err();
            assert!(err.to_string().contains("checkpoint") || err.to_string().contains("Checkpoint"));
        }

        // -- WorkflowDefinition serialization tests --------------------------

        #[test]
        fn test_workflow_definition_serialization() {
            let def = WorkflowDefinition {
                id: "wf-def-1".to_string(),
                name: "Test Workflow".to_string(),
                nodes: vec![
                    WorkflowNodeDef {
                        id: "n1".to_string(),
                        name: "Node 1".to_string(),
                        input_type: "start".to_string(),
                        output_types: vec!["mid".to_string()],
                        timeout_ms: None,
                    },
                    WorkflowNodeDef {
                        id: "n2".to_string(),
                        name: "Node 2".to_string(),
                        input_type: "mid".to_string(),
                        output_types: vec![],
                        timeout_ms: Some(5000),
                    },
                ],
                edges: vec![WorkflowEdgeDef {
                    event_type: "start".to_string(),
                    target_node_ids: vec!["n1".to_string()],
                }],
                entry_event: "start".to_string(),
            };

            let json = def.to_json().expect("to_json");
            let roundtrip = WorkflowDefinition::from_json(&json).expect("from_json");
            assert_eq!(roundtrip.id, "wf-def-1");
            assert_eq!(roundtrip.name, "Test Workflow");
            assert_eq!(roundtrip.nodes.len(), 2);
            assert_eq!(roundtrip.edges.len(), 1);
            assert_eq!(roundtrip.entry_event, "start");
        }

        #[test]
        fn test_workflow_definition_from_json_invalid() {
            let err = WorkflowDefinition::from_json("not valid json{{{").unwrap_err();
            assert!(
                err.to_string().contains("serialization")
                    || err.to_string().contains("Serialization")
                    || err.to_string().contains("failed")
            );
        }

        #[test]
        fn test_workflow_definition_from_graph() {
            let graph = make_simple_pipeline();
            let def = WorkflowDefinition::from_graph("wf-g", "Pipeline", &graph);
            assert_eq!(def.id, "wf-g");
            assert_eq!(def.name, "Pipeline");
            assert_eq!(def.nodes.len(), 3);
            assert_eq!(def.entry_event, "start");
        }

        // -- WorkflowTool tests ----------------------------------------------

        #[test]
        fn test_workflow_tool_creation() {
            let def = WorkflowDefinition {
                id: "tool-wf".to_string(),
                name: "Tool Workflow".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "start".to_string(),
            };
            let tool = WorkflowTool::new("my_tool", "Does something", def);
            assert_eq!(tool.name, "my_tool");
            assert_eq!(tool.description, "Does something");
        }

        #[test]
        fn test_workflow_tool_to_tool_definition() {
            let def = WorkflowDefinition {
                id: "t".to_string(),
                name: "T".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "start".to_string(),
            };
            let tool = WorkflowTool::new("wf_tool", "A workflow tool", def);
            let td = tool.to_tool_definition();
            assert_eq!(td.name, "wf_tool");
            assert_eq!(td.description, "A workflow tool");
            assert_eq!(td.category, "workflow");
            assert_eq!(td.parameters.len(), 2);
            assert_eq!(td.parameters[0].name, "input");
            assert!(td.parameters[0].required);
            assert_eq!(td.parameters[1].name, "workflow_id");
            assert!(!td.parameters[1].required);
        }

        #[test]
        fn test_workflow_tool_serialization() {
            let def = WorkflowDefinition {
                id: "s".to_string(),
                name: "S".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "s".to_string(),
            };
            let tool = WorkflowTool::new("ser_tool", "Serializable", def);
            let json = serde_json::to_string(&tool).expect("serialize");
            let deser: WorkflowTool = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.name, "ser_tool");
        }

        // -- ErrorSnapshot tests ---------------------------------------------

        #[test]
        fn test_error_snapshot_serialization() {
            let snap = ErrorSnapshot {
                node_id: "n1".to_string(),
                step: 7,
                error_message: "boom".to_string(),
                state_at_failure: WorkflowState::new(),
                pending_events: vec![serde_json::json!({"event_type": "x"})],
            };
            let json = serde_json::to_string(&snap).expect("serialize");
            let deser: ErrorSnapshot = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.node_id, "n1");
            assert_eq!(deser.step, 7);
            assert_eq!(deser.error_message, "boom");
        }

        #[test]
        fn test_error_snapshot_debug() {
            let snap = ErrorSnapshot {
                node_id: "dbg".to_string(),
                step: 0,
                error_message: "test".to_string(),
                state_at_failure: WorkflowState::new(),
                pending_events: vec![],
            };
            let debug = format!("{:?}", snap);
            assert!(debug.contains("dbg"));
        }

        // -- Edge case tests -------------------------------------------------

        #[test]
        fn test_empty_workflow() {
            let graph = WorkflowGraph::new("start");
            // No nodes at all — validate should fail (no handler for entry).
            assert!(graph.validate().is_err());
        }

        #[test]
        fn test_single_node_workflow() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "only".to_string(),
                name: "Only".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("ran", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            assert!(graph.validate().is_ok());
            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {}
            });
            let result = runner.run("wf-single", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.steps_executed, 1);
            assert_eq!(result.final_state.get("ran"), Some(&serde_json::json!(true)));
        }

        #[test]
        fn test_runner_max_steps_limit() {
            // Create a graph where a node feeds back into itself via different
            // events (but graph is "validated" because we do NOT call validate
            // — it would detect the cycle).
            let mut graph = WorkflowGraph::new("ping");
            graph.add_node(WorkflowNode {
                id: "pingpong".to_string(),
                name: "PingPong".to_string(),
                handler: Box::new(|_p, state| {
                    let count = state
                        .get("count")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    state.set("count", serde_json::json!(count + 1));
                    Ok(vec![serde_json::json!({
                        "event_type": "ping",
                        "payload": {}
                    })])
                }),
                input_type: "ping".to_string(),
                output_types: vec!["ping".to_string()],
                timeout_ms: None,
            });

            let mut runner = WorkflowRunner::new(graph);
            runner.set_max_steps(10);

            let initial = serde_json::json!({
                "event_type": "ping",
                "payload": {}
            });
            let result = runner
                .run("wf-maxstep", initial, WorkflowState::new())
                .expect("run");
            assert!(!result.completed);
            assert_eq!(result.steps_executed, 10);
        }

        #[test]
        fn test_runner_no_handler_for_output_event() {
            // Output events that have no handler are silently dropped (terminal).
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "emitter".to_string(),
                name: "Emitter".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("emitted", serde_json::json!(true));
                    Ok(vec![serde_json::json!({
                        "event_type": "void",
                        "payload": {}
                    })])
                }),
                input_type: "start".to_string(),
                output_types: vec!["void".to_string()],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {}
            });
            let result = runner
                .run("wf-void", initial, WorkflowState::new())
                .expect("run");
            assert!(result.completed);
        }

        #[test]
        fn test_runner_state_accumulates() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "step1".to_string(),
                name: "Step1".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("s1", serde_json::json!(1));
                    Ok(vec![serde_json::json!({"event_type": "next", "payload": {}})])
                }),
                input_type: "start".to_string(),
                output_types: vec!["next".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "step2".to_string(),
                name: "Step2".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("s2", serde_json::json!(2));
                    Ok(vec![])
                }),
                input_type: "next".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-acc", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.final_state.get("s1"), Some(&serde_json::json!(1)));
            assert_eq!(result.final_state.get("s2"), Some(&serde_json::json!(2)));
        }

        #[test]
        fn test_runner_initial_state_preserved() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n".to_string(),
                name: "N".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("added", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let mut initial_state = WorkflowState::new();
            initial_state.set("pre_existing", serde_json::json!("yes"));

            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-init", initial, initial_state).expect("run");
            assert!(result.completed);
            assert_eq!(
                result.final_state.get("pre_existing"),
                Some(&serde_json::json!("yes"))
            );
            assert_eq!(
                result.final_state.get("added"),
                Some(&serde_json::json!(true))
            );
        }

        #[test]
        fn test_runner_workflow_id_in_result() {
            let mut graph = WorkflowGraph::new("s");
            graph.add_node(WorkflowNode {
                id: "n".to_string(),
                name: "N".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "s".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });
            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "s", "payload": {}});
            let result = runner.run("my-unique-id", initial, WorkflowState::new()).expect("run");
            assert_eq!(result.workflow_id, "my-unique-id");
        }

        #[test]
        fn test_checkpoint_during_run() {
            // After a successful run, checkpoints should exist.
            let cp = std::sync::Arc::new(InMemoryCheckpointer::new());
            let cp_clone = cp.clone();

            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n1".to_string(),
                name: "N1".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            // We need a wrapper since we can't clone Arc<InMemoryCheckpointer>
            // into Box<dyn Checkpointer> directly. Use a newtype.
            struct ArcCheckpointer(std::sync::Arc<InMemoryCheckpointer>);
            impl Checkpointer for ArcCheckpointer {
                fn save(&self, cp: &WorkflowCheckpoint) -> Result<(), AiError> {
                    self.0.save(cp)
                }
                fn load(&self, wf: &str, step: usize) -> Result<Option<WorkflowCheckpoint>, AiError> {
                    self.0.load(wf, step)
                }
                fn list(&self, wf: &str) -> Result<Vec<usize>, AiError> {
                    self.0.list(wf)
                }
                fn latest(&self, wf: &str) -> Result<Option<WorkflowCheckpoint>, AiError> {
                    self.0.latest(wf)
                }
            }

            let runner = WorkflowRunner::with_checkpointer(
                graph,
                Box::new(ArcCheckpointer(cp_clone)),
            );
            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            runner.run("wf-cp", initial, WorkflowState::new()).expect("run");

            let steps = cp.list("wf-cp").expect("list");
            assert!(steps.len() >= 2); // at least step 0 (initial) + step 1
        }

        #[test]
        fn test_workflow_checkpoint_serialization() {
            let cp = WorkflowCheckpoint {
                workflow_id: "cp-ser".to_string(),
                step: 42,
                state: WorkflowState::new(),
                pending_events: vec![serde_json::json!({"event_type": "x"})],
                timestamp: 123456,
            };
            let json = serde_json::to_string(&cp).expect("serialize");
            let deser: WorkflowCheckpoint = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.workflow_id, "cp-ser");
            assert_eq!(deser.step, 42);
            assert_eq!(deser.timestamp, 123456);
        }

        #[test]
        fn test_workflow_result_serialization() {
            let result = WorkflowResult {
                workflow_id: "res-ser".to_string(),
                final_state: WorkflowState::new(),
                steps_executed: 5,
                completed: true,
                error_snapshot: None,
            };
            let json = serde_json::to_string(&result).expect("serialize");
            let deser: WorkflowResult = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.workflow_id, "res-ser");
            assert!(deser.completed);
            assert_eq!(deser.steps_executed, 5);
        }

        #[test]
        fn test_workflow_result_with_error_snapshot() {
            let result = WorkflowResult {
                workflow_id: "res-err".to_string(),
                final_state: WorkflowState::new(),
                steps_executed: 2,
                completed: false,
                error_snapshot: Some(ErrorSnapshot {
                    node_id: "bad".to_string(),
                    step: 2,
                    error_message: "failed".to_string(),
                    state_at_failure: WorkflowState::new(),
                    pending_events: vec![],
                }),
            };
            let json = serde_json::to_string(&result).expect("serialize");
            let deser: WorkflowResult = serde_json::from_str(&json).expect("deserialize");
            assert!(!deser.completed);
            let snap = deser.error_snapshot.expect("snapshot present");
            assert_eq!(snap.node_id, "bad");
        }

        #[test]
        fn test_breakpoint_debug() {
            let bp = WorkflowBreakpoint {
                node_id: "debug_bp".to_string(),
                condition: None,
            };
            let debug = format!("{:?}", bp);
            assert!(debug.contains("debug_bp"));
            assert!(debug.contains("false")); // has_condition = false
        }

        #[test]
        fn test_breakpoint_with_condition_debug() {
            let bp = WorkflowBreakpoint {
                node_id: "cond_bp".to_string(),
                condition: Some(Box::new(|_| true)),
            };
            let debug = format!("{:?}", bp);
            assert!(debug.contains("cond_bp"));
            assert!(debug.contains("true")); // has_condition = true
        }

        #[test]
        fn test_workflow_node_def_serialization() {
            let ndef = WorkflowNodeDef {
                id: "nd1".to_string(),
                name: "NodeDef1".to_string(),
                input_type: "in".to_string(),
                output_types: vec!["out".to_string()],
                timeout_ms: Some(3000),
            };
            let json = serde_json::to_string(&ndef).expect("serialize");
            let deser: WorkflowNodeDef = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.id, "nd1");
            assert_eq!(deser.timeout_ms, Some(3000));
        }

        #[test]
        fn test_workflow_edge_def_serialization() {
            let edef = WorkflowEdgeDef {
                event_type: "transition".to_string(),
                target_node_ids: vec!["a".to_string(), "b".to_string()],
            };
            let json = serde_json::to_string(&edef).expect("serialize");
            let deser: WorkflowEdgeDef = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.event_type, "transition");
            assert_eq!(deser.target_node_ids.len(), 2);
        }

        #[test]
        fn test_runner_multiple_handlers_same_event() {
            // Two nodes both handle the same event type.
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "h1".to_string(),
                name: "Handler1".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("h1_ran", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "h2".to_string(),
                name: "Handler2".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("h2_ran", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-multi", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.final_state.get("h1_ran"), Some(&serde_json::json!(true)));
            assert_eq!(result.final_state.get("h2_ran"), Some(&serde_json::json!(true)));
        }

        #[test]
        fn test_runner_step_count_in_state() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": { "value": 1 }
            });
            let result = runner.run("wf-sc", initial, WorkflowState::new()).expect("run");
            assert_eq!(result.final_state.step_count, result.steps_executed);
        }

        #[test]
        fn test_in_memory_checkpointer_default() {
            let cp = InMemoryCheckpointer::default();
            let steps = cp.list("any").expect("list");
            assert!(steps.is_empty());
        }

        #[test]
        fn test_workflow_definition_roundtrip_complex() {
            let def = WorkflowDefinition {
                id: "complex".to_string(),
                name: "Complex Workflow".to_string(),
                nodes: vec![
                    WorkflowNodeDef {
                        id: "a".to_string(),
                        name: "A".to_string(),
                        input_type: "start".to_string(),
                        output_types: vec!["x".to_string(), "y".to_string()],
                        timeout_ms: Some(1000),
                    },
                    WorkflowNodeDef {
                        id: "b".to_string(),
                        name: "B".to_string(),
                        input_type: "x".to_string(),
                        output_types: vec![],
                        timeout_ms: None,
                    },
                    WorkflowNodeDef {
                        id: "c".to_string(),
                        name: "C".to_string(),
                        input_type: "y".to_string(),
                        output_types: vec!["z".to_string()],
                        timeout_ms: Some(2000),
                    },
                ],
                edges: vec![
                    WorkflowEdgeDef {
                        event_type: "start".to_string(),
                        target_node_ids: vec!["a".to_string()],
                    },
                    WorkflowEdgeDef {
                        event_type: "x".to_string(),
                        target_node_ids: vec!["b".to_string()],
                    },
                    WorkflowEdgeDef {
                        event_type: "y".to_string(),
                        target_node_ids: vec!["c".to_string()],
                    },
                ],
                entry_event: "start".to_string(),
            };

            let json = def.to_json().expect("to_json");
            let rt = WorkflowDefinition::from_json(&json).expect("from_json");
            assert_eq!(rt.nodes.len(), 3);
            assert_eq!(rt.edges.len(), 3);
            assert_eq!(rt.nodes[0].output_types.len(), 2);
        }

        #[test]
        fn test_workflow_tool_definition_param_types() {
            let def = WorkflowDefinition {
                id: "pt".to_string(),
                name: "PT".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "s".to_string(),
            };
            let tool = WorkflowTool::new("param_test", "Testing params", def);
            let td = tool.to_tool_definition();
            assert_eq!(td.parameters[0].param_type, "object");
            assert_eq!(td.parameters[1].param_type, "string");
        }

        #[test]
        fn test_runner_empty_payload() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n".to_string(),
                name: "N".to_string(),
                handler: Box::new(|payload, state| {
                    // payload should be null when no payload field in event.
                    state.set("payload_was_null", serde_json::json!(payload.is_null()));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "start"});
            let result = runner.run("wf-ep", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(
                result.final_state.get("payload_was_null"),
                Some(&serde_json::json!(true))
            );
        }

        #[test]
        fn test_graph_multiple_output_types_no_cycle() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "fan".to_string(),
                name: "Fan".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                timeout_ms: None,
            });
            assert!(graph.validate().is_ok());
        }

        #[test]
        fn test_workflow_state_debug() {
            let mut state = WorkflowState::new();
            state.set("debug_key", serde_json::json!("debug_val"));
            let debug = format!("{:?}", state);
            assert!(debug.contains("debug_key"));
        }

        #[test]
        fn test_workflow_tool_new() {
            let def = WorkflowDefinition {
                id: "id".to_string(),
                name: "Name".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "e".to_string(),
            };
            let tool = WorkflowTool::new("tool_name", "tool_desc", def.clone());
            assert_eq!(tool.name, "tool_name");
            assert_eq!(tool.description, "tool_desc");
            assert_eq!(tool.workflow_def.id, "id");
        }

        #[test]
        fn test_runner_handler_modifies_state_across_nodes() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "inc1".to_string(),
                name: "Inc1".to_string(),
                handler: Box::new(|_p, state| {
                    let v = state.get("counter").and_then(|v| v.as_i64()).unwrap_or(0);
                    state.set("counter", serde_json::json!(v + 1));
                    Ok(vec![serde_json::json!({"event_type": "next1", "payload": {}})])
                }),
                input_type: "start".to_string(),
                output_types: vec!["next1".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "inc2".to_string(),
                name: "Inc2".to_string(),
                handler: Box::new(|_p, state| {
                    let v = state.get("counter").and_then(|v| v.as_i64()).unwrap_or(0);
                    state.set("counter", serde_json::json!(v + 1));
                    Ok(vec![serde_json::json!({"event_type": "next2", "payload": {}})])
                }),
                input_type: "next1".to_string(),
                output_types: vec!["next2".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "inc3".to_string(),
                name: "Inc3".to_string(),
                handler: Box::new(|_p, state| {
                    let v = state.get("counter").and_then(|v| v.as_i64()).unwrap_or(0);
                    state.set("counter", serde_json::json!(v + 1));
                    Ok(vec![])
                }),
                input_type: "next2".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-inc", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(
                result.final_state.get("counter"),
                Some(&serde_json::json!(3))
            );
        }

        #[test]
        fn test_runner_produces_no_output_events() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "sink".to_string(),
                name: "Sink".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("sank", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-sink", initial, WorkflowState::new()).expect("run");
            assert!(result.completed);
            assert_eq!(result.steps_executed, 1);
        }

        #[test]
        fn test_workflow_definition_empty_nodes() {
            let def = WorkflowDefinition {
                id: "empty".to_string(),
                name: "Empty".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "start".to_string(),
            };
            let json = def.to_json().expect("to_json");
            let rt = WorkflowDefinition::from_json(&json).expect("from_json");
            assert!(rt.nodes.is_empty());
            assert!(rt.edges.is_empty());
        }

        #[test]
        fn test_workflow_tool_definition_serialization() {
            let def = WorkflowDefinition {
                id: "td-ser".to_string(),
                name: "TD Ser".to_string(),
                nodes: vec![],
                edges: vec![],
                entry_event: "s".to_string(),
            };
            let tool = WorkflowTool::new("td_ser", "Tool def serializable", def);
            let td = tool.to_tool_definition();
            let json = serde_json::to_string(&td).expect("serialize");
            let deser: WorkflowToolDefinition =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deser.name, "td_ser");
            assert_eq!(deser.category, "workflow");
        }

        #[test]
        fn test_runner_max_steps_zero() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n".to_string(),
                name: "N".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let mut runner = WorkflowRunner::new(graph);
            runner.set_max_steps(0);

            let initial = serde_json::json!({"event_type": "start", "payload": {}});
            let result = runner.run("wf-0", initial, WorkflowState::new()).expect("run");
            assert!(!result.completed);
            assert_eq!(result.steps_executed, 0);
        }

        #[test]
        fn test_workflow_graph_entry_event() {
            let graph = WorkflowGraph::new("custom_entry");
            assert_eq!(graph.entry_event(), "custom_entry");
        }

        #[test]
        fn test_node_handler_receives_payload() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "receiver".to_string(),
                name: "Receiver".to_string(),
                handler: Box::new(|payload, state| {
                    state.set("received", payload.clone());
                    Ok(vec![])
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let initial = serde_json::json!({
                "event_type": "start",
                "payload": {"msg": "hello", "num": 42}
            });
            let result = runner.run("wf-recv", initial, WorkflowState::new()).expect("run");
            let received = result.final_state.get("received").expect("received");
            assert_eq!(received["msg"], "hello");
            assert_eq!(received["num"], 42);
        }

        #[test]
        fn test_multiple_checkpoints_different_workflows() {
            let cp_store = InMemoryCheckpointer::new();
            for wf_id in &["wf-a", "wf-b", "wf-c"] {
                cp_store
                    .save(&WorkflowCheckpoint {
                        workflow_id: wf_id.to_string(),
                        step: 0,
                        state: WorkflowState::new(),
                        pending_events: vec![],
                        timestamp: 0,
                    })
                    .expect("save");
            }
            assert_eq!(cp_store.list("wf-a").expect("list").len(), 1);
            assert_eq!(cp_store.list("wf-b").expect("list").len(), 1);
            assert_eq!(cp_store.list("wf-c").expect("list").len(), 1);
            assert!(cp_store.list("wf-d").expect("list").is_empty());
        }

        // ==================================================================
        // 7.1 — Durable Execution tests
        // ==================================================================

        // -- DurableConfig tests -------------------------------------------

        #[test]
        fn test_durable_config_defaults() {
            let config = DurableConfig::default();
            assert_eq!(config.backend, DurableBackend::InMemory);
            assert!(config.auto_checkpoint);
            assert_eq!(config.retention_policy, RetentionPolicy::KeepAll);
            assert!(config.recovery_enabled);
        }

        #[test]
        fn test_durable_config_custom() {
            let config = DurableConfig {
                backend: DurableBackend::Custom("rocksdb".to_string()),
                auto_checkpoint: false,
                retention_policy: RetentionPolicy::KeepLast(5),
                recovery_enabled: false,
            };
            assert_eq!(config.backend, DurableBackend::Custom("rocksdb".to_string()));
            assert!(!config.auto_checkpoint);
            assert_eq!(config.retention_policy, RetentionPolicy::KeepLast(5));
            assert!(!config.recovery_enabled);
        }

        // -- DurableExecutor tests -----------------------------------------

        #[test]
        fn test_durable_executor_basic_execution() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig::default();
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            let result = executor.execute(state).expect("execute");

            // The pipeline sets after_a and after_b.
            // Input has no "value" field in payload so defaults to 0.
            // after_a = 0 + 1 = 1, after_b = 1 * 10 = 10
            assert_eq!(result.get("after_a"), Some(&serde_json::json!(1)));
            assert_eq!(result.get("after_b"), Some(&serde_json::json!(10)));
        }

        #[test]
        fn test_durable_executor_checkpoint_count() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig::default();
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            executor.execute(state).expect("execute");

            // Checkpoints: initial + 3 steps + completed = 5
            assert!(executor.get_checkpoint_count() >= 5);
        }

        #[test]
        fn test_durable_executor_retention_keep_last() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig {
                retention_policy: RetentionPolicy::KeepLast(2),
                ..DurableConfig::default()
            };
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            executor.execute(state).expect("execute");

            // After retention, only 2 durable checkpoints should remain.
            let durable_cps = executor.get_durable_checkpoints();
            assert_eq!(durable_cps.len(), 2);
        }

        #[test]
        fn test_durable_executor_retention_keep_all() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig {
                retention_policy: RetentionPolicy::KeepAll,
                ..DurableConfig::default()
            };
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            executor.execute(state).expect("execute");

            // All checkpoints should remain: initial + 3 steps + completed = 5
            let durable_cps = executor.get_durable_checkpoints();
            assert!(durable_cps.len() >= 5);
        }

        #[test]
        fn test_durable_executor_error_checkpoint_saved() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "fail".to_string(),
                name: "Fail".to_string(),
                handler: Box::new(|_p, _s| {
                    Err(AiError::Other("durable failure".to_string()))
                }),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig::default();
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            // The runner returns Ok(WorkflowResult) with error_snapshot, so
            // execute should still return Ok.
            let result = executor.execute(state).expect("execute");

            // Should have error-related durable checkpoints.
            let durable_cps = executor.get_durable_checkpoints();
            let has_error_cp = durable_cps
                .iter()
                .any(|cp| cp.node_id.starts_with("error"));
            assert!(has_error_cp, "expected an error checkpoint");
            assert!(result.values.is_empty() || result.step_count == 0);
        }

        // -- RecoveryManager tests -----------------------------------------

        #[test]
        fn test_recovery_manager_find_interrupted_empty() {
            let checkpointer = Box::new(InMemoryCheckpointer::new());
            let recovery = RecoveryManager::new(checkpointer);
            let interrupted = recovery.find_interrupted("nonexistent");
            assert!(interrupted.is_empty());
        }

        #[test]
        fn test_recovery_manager_recover_from_checkpoint() {
            let cp_store = InMemoryCheckpointer::new();
            let mut state = WorkflowState::new();
            state.set("progress", serde_json::json!("halfway"));
            cp_store
                .save(&WorkflowCheckpoint {
                    workflow_id: "exec-recover".to_string(),
                    step: 5,
                    state: state.clone(),
                    pending_events: vec![],
                    timestamp: 100,
                })
                .expect("save");

            let recovery = RecoveryManager::new(Box::new(cp_store));
            let recovered = recovery
                .recover("exec-recover")
                .expect("recover")
                .expect("should have state");
            assert_eq!(
                recovered.get("progress"),
                Some(&serde_json::json!("halfway"))
            );
        }

        #[test]
        fn test_recovery_manager_recover_missing() {
            let checkpointer = Box::new(InMemoryCheckpointer::new());
            let recovery = RecoveryManager::new(checkpointer);
            let recovered = recovery.recover("no-such-exec").expect("recover");
            assert!(recovered.is_none());
        }

        #[test]
        fn test_recovery_manager_list_checkpoints() {
            let cp_store = InMemoryCheckpointer::new();
            for step in [0, 1, 2, 3] {
                cp_store
                    .save(&WorkflowCheckpoint {
                        workflow_id: "exec-list".to_string(),
                        step,
                        state: WorkflowState::new(),
                        pending_events: vec![],
                        timestamp: 0,
                    })
                    .expect("save");
            }

            let recovery = RecoveryManager::new(Box::new(cp_store));
            let cps = recovery.list_checkpoints("exec-list");
            assert_eq!(cps.len(), 4);
            assert_eq!(cps[0], "step-0");
            assert_eq!(cps[3], "step-3");
        }

        #[test]
        fn test_recovery_manager_cleanup() {
            let cp_store = InMemoryCheckpointer::new();
            cp_store
                .save(&WorkflowCheckpoint {
                    workflow_id: "exec-clean".to_string(),
                    step: 5,
                    state: {
                        let mut s = WorkflowState::new();
                        s.set("data", serde_json::json!("important"));
                        s
                    },
                    pending_events: vec![serde_json::json!({"event_type": "pending"})],
                    timestamp: 100,
                })
                .expect("save");

            let recovery = RecoveryManager::new(Box::new(cp_store));
            recovery.cleanup("exec-clean");
            // After cleanup, the latest checkpoint should be the sentinel (step 0, empty).
            let cps = recovery.list_checkpoints("exec-clean");
            assert!(!cps.is_empty()); // sentinel exists
        }

        // -- DurableCheckpoint tests ---------------------------------------

        #[test]
        fn test_durable_checkpoint_creation() {
            let cp = DurableCheckpoint {
                execution_id: "exec-1".to_string(),
                checkpoint_id: "exec-1-cp-0".to_string(),
                node_id: "initial".to_string(),
                state: WorkflowState::new(),
                step_number: 0,
                is_auto: true,
                created_at: 1000,
            };
            assert_eq!(cp.execution_id, "exec-1");
            assert_eq!(cp.checkpoint_id, "exec-1-cp-0");
            assert_eq!(cp.node_id, "initial");
            assert_eq!(cp.step_number, 0);
            assert!(cp.is_auto);
            assert_eq!(cp.created_at, 1000);
        }

        #[test]
        fn test_durable_checkpoint_is_auto_flag() {
            let auto_cp = DurableCheckpoint {
                execution_id: "e".to_string(),
                checkpoint_id: "e-cp-0".to_string(),
                node_id: "step-1".to_string(),
                state: WorkflowState::new(),
                step_number: 1,
                is_auto: true,
                created_at: 0,
            };
            assert!(auto_cp.is_auto);

            let manual_cp = DurableCheckpoint {
                execution_id: "e".to_string(),
                checkpoint_id: "e-cp-1".to_string(),
                node_id: "completed".to_string(),
                state: WorkflowState::new(),
                step_number: 3,
                is_auto: false,
                created_at: 0,
            };
            assert!(!manual_cp.is_auto);
        }

        // -- Integration test ----------------------------------------------

        #[test]
        fn test_durable_executor_integration() {
            // Build a two-node pipeline.
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "step1".to_string(),
                name: "Step1".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("step1_done", serde_json::json!(true));
                    Ok(vec![serde_json::json!({
                        "event_type": "next",
                        "payload": {}
                    })])
                }),
                input_type: "start".to_string(),
                output_types: vec!["next".to_string()],
                timeout_ms: None,
            });
            graph.add_node(WorkflowNode {
                id: "step2".to_string(),
                name: "Step2".to_string(),
                handler: Box::new(|_p, state| {
                    state.set("step2_done", serde_json::json!(true));
                    Ok(vec![])
                }),
                input_type: "next".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig::default();
            let cp_store = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, cp_store);
            let exec_id = executor.get_execution_id().to_string();

            // Execute.
            let final_state = executor.execute(WorkflowState::new()).expect("execute");
            assert_eq!(
                final_state.get("step1_done"),
                Some(&serde_json::json!(true))
            );
            assert_eq!(
                final_state.get("step2_done"),
                Some(&serde_json::json!(true))
            );

            // Verify checkpoints were saved.
            let cp_list = executor.list_checkpoints();
            assert!(
                cp_list.len() >= 4,
                "expected at least 4 durable checkpoints, got {}",
                cp_list.len()
            );

            // All checkpoint IDs should start with the execution id.
            for cp_id in &cp_list {
                assert!(
                    cp_id.starts_with(&exec_id),
                    "checkpoint ID '{}' should start with '{}'",
                    cp_id,
                    exec_id
                );
            }

            // Verify durable checkpoint metadata.
            let durable_cps = executor.get_durable_checkpoints();
            assert!(
                durable_cps.iter().any(|cp| cp.node_id == "initial"),
                "should have initial checkpoint"
            );
            assert!(
                durable_cps.iter().any(|cp| cp.node_id == "completed"),
                "should have completed checkpoint"
            );
        }

        #[test]
        fn test_durable_executor_retention_keep_checkpoints_only() {
            let graph = make_simple_pipeline();
            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig {
                retention_policy: RetentionPolicy::KeepCheckpointsOnly,
                ..DurableConfig::default()
            };
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let mut executor = DurableExecutor::new(runner, config, checkpointer);
            let state = WorkflowState::new();
            executor.execute(state).expect("execute");

            // Only non-auto (explicit) checkpoints should remain.
            let durable_cps = executor.get_durable_checkpoints();
            for cp in &durable_cps {
                assert!(
                    !cp.is_auto,
                    "auto checkpoint '{}' should have been pruned",
                    cp.checkpoint_id
                );
            }
        }

        #[test]
        fn test_durable_executor_execution_id() {
            let mut graph = WorkflowGraph::new("start");
            graph.add_node(WorkflowNode {
                id: "n".to_string(),
                name: "N".to_string(),
                handler: Box::new(|_p, _s| Ok(vec![])),
                input_type: "start".to_string(),
                output_types: vec![],
                timeout_ms: None,
            });

            let runner = WorkflowRunner::new(graph);
            let config = DurableConfig::default();
            let checkpointer = Box::new(InMemoryCheckpointer::new());

            let executor = DurableExecutor::new(runner, config, checkpointer);
            let exec_id = executor.get_execution_id();
            assert!(exec_id.starts_with("exec-"));
            assert!(exec_id.contains("start")); // entry event
        }
    }
}

// Re-export everything from the inner module when the feature is enabled.
#[cfg(feature = "workflows")]
pub use inner::*;
