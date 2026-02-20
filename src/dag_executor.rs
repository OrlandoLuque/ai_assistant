//! DAG (Directed Acyclic Graph) workflow executor.
//!
//! A pure-Rust engine for defining, validating, and executing workflows modeled as
//! directed acyclic graphs. Supports conditional edges, retry logic, parallel execution
//! limits, critical path analysis, and progress tracking.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use serde_json::{self, Value};

// ---------------------------------------------------------------------------
// DagNodeId
// ---------------------------------------------------------------------------

/// Unique identifier for a node within a DAG.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DagNodeId(pub String);

impl DagNodeId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DagNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DagNodeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for DagNodeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// ---------------------------------------------------------------------------
// DagNodeStatus
// ---------------------------------------------------------------------------

/// Execution status of a single DAG node.
#[derive(Debug, Clone, PartialEq)]
pub enum DagNodeStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
    Skipped,
}

impl std::fmt::Display for DagNodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Pending => "Pending",
            Self::Ready => "Ready",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Skipped => "Skipped",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// DagNode
// ---------------------------------------------------------------------------

/// A single unit of work inside a DAG workflow.
#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: DagNodeId,
    pub name: String,
    pub action: String,
    pub status: DagNodeStatus,
    pub dependencies: Vec<DagNodeId>,
    pub result: Option<Value>,
    pub timeout_ms: Option<u64>,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl DagNode {
    pub fn new(id: &str, name: &str, action: &str) -> Self {
        Self {
            id: DagNodeId::new(id),
            name: name.to_string(),
            action: action.to_string(),
            status: DagNodeStatus::Pending,
            dependencies: Vec::new(),
            result: None,
            timeout_ms: None,
            retry_count: 0,
            max_retries: 0,
        }
    }

    pub fn with_dependency(mut self, dep: DagNodeId) -> Self {
        self.dependencies.push(dep);
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }
}

// ---------------------------------------------------------------------------
// EdgeCondition
// ---------------------------------------------------------------------------

/// Condition that must hold on the source node before the edge activates.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeCondition {
    /// The edge is always active once the source node finishes (any terminal status).
    Always,
    /// The source node must have completed successfully.
    OnSuccess,
    /// The source node must have failed.
    OnFailure,
    /// The source node must have completed with a result containing the given key.
    OnOutput(String),
}

impl std::fmt::Display for EdgeCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => write!(f, "Always"),
            Self::OnSuccess => write!(f, "OnSuccess"),
            Self::OnFailure => write!(f, "OnFailure"),
            Self::OnOutput(key) => write!(f, "OnOutput({})", key),
        }
    }
}

// ---------------------------------------------------------------------------
// DagEdge
// ---------------------------------------------------------------------------

/// A directed edge between two DAG nodes, optionally guarded by a condition.
#[derive(Debug, Clone)]
pub struct DagEdge {
    pub from: DagNodeId,
    pub to: DagNodeId,
    pub condition: EdgeCondition,
}

impl DagEdge {
    pub fn new(from: &str, to: &str) -> Self {
        Self {
            from: DagNodeId::new(from),
            to: DagNodeId::new(to),
            condition: EdgeCondition::Always,
        }
    }

    pub fn with_condition(mut self, condition: EdgeCondition) -> Self {
        self.condition = condition;
        self
    }
}

// ---------------------------------------------------------------------------
// DagError
// ---------------------------------------------------------------------------

/// Errors that can occur during DAG construction, validation, or execution.
#[derive(Debug, Clone, PartialEq)]
pub enum DagError {
    CycleDetected,
    NodeNotFound(String),
    InvalidTransition {
        node: String,
        from: String,
        to: String,
    },
    DependencyFailed(String),
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "Cycle detected in DAG"),
            Self::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            Self::InvalidTransition { node, from, to } => {
                write!(
                    f,
                    "Invalid transition for node '{}': {} -> {}",
                    node, from, to
                )
            }
            Self::DependencyFailed(id) => write!(f, "Dependency failed: {}", id),
        }
    }
}

impl std::error::Error for DagError {}

// ---------------------------------------------------------------------------
// DagDefinition
// ---------------------------------------------------------------------------

/// A declarative description of a DAG workflow (nodes + edges) that can be
/// validated and handed off to a [`DagExecutor`] for execution.
pub struct DagDefinition {
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
}

impl DagDefinition {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: DagNode) -> &mut Self {
        self.nodes.push(node);
        self
    }

    pub fn add_edge(&mut self, edge: DagEdge) -> &mut Self {
        self.edges.push(edge);
        self
    }

    /// Convenience: add an `Always` edge from `depends_on` to `node_id`.
    pub fn add_dependency(&mut self, node_id: &str, depends_on: &str) -> &mut Self {
        self.edges.push(DagEdge::new(depends_on, node_id));
        self
    }

    /// Validate the DAG contains no cycles using iterative DFS with three-color
    /// marking (white = unvisited, gray = in current path, black = fully explored).
    pub fn validate(&self) -> Result<(), DagError> {
        // Build adjacency list.
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut all_nodes: HashSet<&str> = HashSet::new();
        for node in &self.nodes {
            all_nodes.insert(node.id.as_str());
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from.as_str())
                .or_default()
                .push(edge.to.as_str());
        }

        const WHITE: u8 = 0;
        const GRAY: u8 = 1;
        const BLACK: u8 = 2;

        let mut color: HashMap<&str, u8> = HashMap::new();
        for &n in &all_nodes {
            color.insert(n, WHITE);
        }

        for &start in &all_nodes {
            if color[start] != WHITE {
                continue;
            }
            // Iterative DFS.
            let mut stack: Vec<(&str, bool)> = vec![(start, false)];
            while let Some((node, processed)) = stack.pop() {
                if processed {
                    color.insert(node, BLACK);
                    continue;
                }
                if color[node] == GRAY {
                    return Err(DagError::CycleDetected);
                }
                if color[node] == BLACK {
                    continue;
                }
                color.insert(node, GRAY);
                // Push a sentinel so we mark it BLACK after children.
                stack.push((node, true));
                if let Some(neighbors) = adj.get(node) {
                    for &nb in neighbors {
                        match color.get(nb).copied().unwrap_or(WHITE) {
                            GRAY => return Err(DagError::CycleDetected),
                            WHITE => stack.push((nb, false)),
                            _ => {} // BLACK — skip
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Produce a topological ordering using Kahn's algorithm.
    pub fn topological_sort(&self) -> Result<Vec<DagNodeId>, DagError> {
        // Build adjacency + in-degree.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(node.id.as_str()).or_insert(0);
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from.as_str())
                .or_default()
                .push(edge.to.as_str());
            *in_degree.entry(edge.to.as_str()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&node, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(node);
            }
        }

        let mut order: Vec<DagNodeId> = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(DagNodeId::new(node));
            if let Some(neighbors) = adj.get(node) {
                for &nb in neighbors {
                    if let Some(d) = in_degree.get_mut(nb) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(nb);
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(DagError::CycleDetected);
        }

        Ok(order)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// DagExecutor
// ---------------------------------------------------------------------------

/// Runtime executor that drives a DAG workflow through its lifecycle.
pub struct DagExecutor {
    pub nodes: HashMap<DagNodeId, DagNode>,
    pub edges: Vec<DagEdge>,
    pub max_parallel: usize,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
}

impl DagExecutor {
    /// Create a new executor from a validated [`DagDefinition`].
    pub fn new(definition: DagDefinition) -> Result<Self, DagError> {
        definition.validate()?;

        let mut nodes = HashMap::new();
        for node in &definition.nodes {
            nodes.insert(node.id.clone(), node.clone());
        }

        Ok(Self {
            nodes,
            edges: definition.edges,
            max_parallel: usize::MAX,
            started_at: None,
            completed_at: None,
        })
    }

    pub fn with_max_parallel(mut self, n: usize) -> Self {
        self.max_parallel = n;
        self
    }

    /// Return the IDs of nodes that are ready to run.
    ///
    /// A node is ready when it is `Pending` or `Ready` and **all** of its
    /// incoming edges are satisfied (see [`EdgeCondition`]).
    pub fn ready_nodes(&self) -> Vec<DagNodeId> {
        let mut ready = Vec::new();

        for (id, node) in &self.nodes {
            if node.status != DagNodeStatus::Pending && node.status != DagNodeStatus::Ready {
                continue;
            }

            // Collect all incoming edges for this node.
            let incoming: Vec<&DagEdge> = self.edges.iter().filter(|e| e.to == *id).collect();

            // If there are no incoming edges the node is immediately ready.
            if incoming.is_empty() {
                ready.push(id.clone());
                continue;
            }

            let all_satisfied = incoming.iter().all(|edge| {
                let dep = match self.nodes.get(&edge.from) {
                    Some(d) => d,
                    None => return false,
                };
                match &edge.condition {
                    EdgeCondition::Always => {
                        dep.status == DagNodeStatus::Completed
                            || dep.status == DagNodeStatus::Failed
                            || dep.status == DagNodeStatus::Skipped
                    }
                    EdgeCondition::OnSuccess => dep.status == DagNodeStatus::Completed,
                    EdgeCondition::OnFailure => dep.status == DagNodeStatus::Failed,
                    EdgeCondition::OnOutput(key) => {
                        dep.status == DagNodeStatus::Completed
                            && dep.result.as_ref().and_then(|v| v.get(key)).is_some()
                    }
                }
            });

            if all_satisfied {
                ready.push(id.clone());
            }
        }

        // Sort for deterministic output.
        ready.sort_by(|a, b| a.0.cmp(&b.0));
        ready
    }

    pub fn start_node(&mut self, id: &DagNodeId) -> Result<(), DagError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| DagError::NodeNotFound(id.to_string()))?;

        match node.status {
            DagNodeStatus::Pending | DagNodeStatus::Ready => {
                node.status = DagNodeStatus::Running;
                if self.started_at.is_none() {
                    self.started_at = Some(Instant::now());
                }
                Ok(())
            }
            _ => Err(DagError::InvalidTransition {
                node: id.to_string(),
                from: node.status.to_string(),
                to: "Running".to_string(),
            }),
        }
    }

    pub fn complete_node(&mut self, id: &DagNodeId, result: Value) -> Result<(), DagError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| DagError::NodeNotFound(id.to_string()))?;

        match node.status {
            DagNodeStatus::Running => {
                node.status = DagNodeStatus::Completed;
                node.result = Some(result);
                self.check_all_done();
                Ok(())
            }
            _ => Err(DagError::InvalidTransition {
                node: id.to_string(),
                from: node.status.to_string(),
                to: "Completed".to_string(),
            }),
        }
    }

    pub fn fail_node(&mut self, id: &DagNodeId, error: &str) -> Result<(), DagError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| DagError::NodeNotFound(id.to_string()))?;

        match node.status {
            DagNodeStatus::Running => {
                node.retry_count += 1;
                if node.retry_count <= node.max_retries {
                    // Still have retries left — go back to Pending.
                    node.status = DagNodeStatus::Pending;
                    node.result = Some(serde_json::json!({ "error": error }));
                } else {
                    node.status = DagNodeStatus::Failed;
                    node.result = Some(serde_json::json!({ "error": error }));
                    self.check_all_done();
                }
                Ok(())
            }
            _ => Err(DagError::InvalidTransition {
                node: id.to_string(),
                from: node.status.to_string(),
                to: "Failed".to_string(),
            }),
        }
    }

    pub fn skip_node(&mut self, id: &DagNodeId) -> Result<(), DagError> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| DagError::NodeNotFound(id.to_string()))?;

        node.status = DagNodeStatus::Skipped;
        self.check_all_done();
        Ok(())
    }

    /// Advance the DAG by one step: find ready nodes, mark up to `max_parallel`
    /// of them as `Running`, and return their IDs.
    pub fn step(&mut self) -> Vec<DagNodeId> {
        let ready = self.ready_nodes();
        let to_start: Vec<DagNodeId> = ready.into_iter().take(self.max_parallel).collect();

        for id in &to_start {
            let _ = self.start_node(id);
        }

        to_start
    }

    /// Synchronously run the DAG to completion using the provided handler.
    pub fn run_to_completion(&mut self, handler: &dyn Fn(&DagNode) -> Result<Value, String>) {
        loop {
            if self.is_complete() {
                break;
            }

            let started = self.step();
            if started.is_empty() {
                // No progress can be made — break to avoid infinite loop.
                break;
            }

            for id in started {
                // Snapshot the node for the handler (we need &DagNode without
                // borrowing self mutably).
                let node_snapshot = self.nodes.get(&id).cloned();
                if let Some(node) = node_snapshot {
                    match handler(&node) {
                        Ok(val) => {
                            let _ = self.complete_node(&id, val);
                        }
                        Err(err) => {
                            let _ = self.fail_node(&id, &err);
                        }
                    }
                }
            }
        }
    }

    /// Returns `true` when every node has reached a terminal state.
    pub fn is_complete(&self) -> bool {
        self.nodes.values().all(|n| {
            matches!(
                n.status,
                DagNodeStatus::Completed | DagNodeStatus::Failed | DagNodeStatus::Skipped
            )
        })
    }

    /// Fraction of nodes in a terminal state, in the range `[0.0, 1.0]`.
    pub fn progress(&self) -> f32 {
        if self.nodes.is_empty() {
            return 1.0;
        }
        let done = self
            .nodes
            .values()
            .filter(|n| {
                matches!(
                    n.status,
                    DagNodeStatus::Completed | DagNodeStatus::Failed | DagNodeStatus::Skipped
                )
            })
            .count();
        done as f32 / self.nodes.len() as f32
    }

    /// Return a topological ordering of the nodes by rebuilding a
    /// [`DagDefinition`] from the executor state.
    pub fn execution_order(&self) -> Result<Vec<DagNodeId>, DagError> {
        let mut def = DagDefinition::new();
        for node in self.nodes.values() {
            def.add_node(node.clone());
        }
        for edge in &self.edges {
            def.add_edge(edge.clone());
        }
        def.topological_sort()
    }

    /// Compute the critical path (longest chain of dependencies).
    ///
    /// For each node the depth is `1 + max(depth of predecessors)`. The chain
    /// ending at the deepest node is returned.
    pub fn critical_path(&self) -> Vec<DagNodeId> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Build predecessor map from edges (from -> to means `to` depends on `from`).
        let mut preds: HashMap<&DagNodeId, Vec<&DagNodeId>> = HashMap::new();
        for id in self.nodes.keys() {
            preds.entry(id).or_default();
        }
        for edge in &self.edges {
            if self.nodes.contains_key(&edge.from) && self.nodes.contains_key(&edge.to) {
                preds.entry(&edge.to).or_default().push(&edge.from);
            }
        }

        // Memoised depth computation.
        let mut depth_cache: HashMap<&DagNodeId, usize> = HashMap::new();
        let mut parent_cache: HashMap<&DagNodeId, Option<&DagNodeId>> = HashMap::new();

        fn compute_depth<'a>(
            node: &'a DagNodeId,
            preds: &HashMap<&'a DagNodeId, Vec<&'a DagNodeId>>,
            depth_cache: &mut HashMap<&'a DagNodeId, usize>,
            parent_cache: &mut HashMap<&'a DagNodeId, Option<&'a DagNodeId>>,
            visiting: &mut HashSet<&'a DagNodeId>,
        ) -> usize {
            if let Some(&d) = depth_cache.get(node) {
                return d;
            }
            if visiting.contains(node) {
                return 1; // cycle guard (should not happen in valid DAG)
            }
            visiting.insert(node);

            let predecessors = preds.get(node).cloned().unwrap_or_default();
            if predecessors.is_empty() {
                depth_cache.insert(node, 1);
                parent_cache.insert(node, None);
                visiting.remove(node);
                return 1;
            }

            let mut max_depth = 0usize;
            let mut best_parent: Option<&'a DagNodeId> = None;
            for pred in &predecessors {
                let d = compute_depth(pred, preds, depth_cache, parent_cache, visiting);
                if d > max_depth {
                    max_depth = d;
                    best_parent = Some(pred);
                }
            }

            let my_depth = 1 + max_depth;
            depth_cache.insert(node, my_depth);
            parent_cache.insert(node, best_parent);
            visiting.remove(node);
            my_depth
        }

        let mut visiting: HashSet<&DagNodeId> = HashSet::new();
        for id in self.nodes.keys() {
            compute_depth(
                id,
                &preds,
                &mut depth_cache,
                &mut parent_cache,
                &mut visiting,
            );
        }

        // Find the deepest node.
        let deepest = depth_cache
            .iter()
            .max_by_key(|(_, &d)| d)
            .map(|(id, _)| *id);

        let Some(mut current) = deepest else {
            return Vec::new();
        };

        // Walk back through parents to reconstruct the path.
        let mut path = Vec::new();
        loop {
            path.push(current.clone());
            match parent_cache.get(current).copied().flatten() {
                Some(p) => current = p,
                None => break,
            }
        }

        path.reverse();
        path
    }

    /// Serialise the current executor state to JSON.
    pub fn export_json(&self) -> Value {
        let nodes_json: Vec<Value> = {
            let mut items: Vec<_> = self.nodes.values().collect();
            items.sort_by(|a, b| a.id.0.cmp(&b.id.0));
            items
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "id": n.id.as_str(),
                        "name": &n.name,
                        "status": n.status.to_string(),
                        "result": n.result,
                        "dependencies": n.dependencies.iter().map(|d| d.as_str()).collect::<Vec<_>>(),
                    })
                })
                .collect()
        };

        let edges_json: Vec<Value> = self
            .edges
            .iter()
            .map(|e| {
                serde_json::json!({
                    "from": e.from.as_str(),
                    "to": e.to.as_str(),
                    "condition": e.condition.to_string(),
                })
            })
            .collect();

        serde_json::json!({
            "nodes": nodes_json,
            "edges": edges_json,
        })
    }

    pub fn node_status(&self, id: &DagNodeId) -> Option<&DagNodeStatus> {
        self.nodes.get(id).map(|n| &n.status)
    }

    pub fn node_result(&self, id: &DagNodeId) -> Option<&Value> {
        self.nodes.get(id).and_then(|n| n.result.as_ref())
    }

    // ------ internal helpers ------

    fn check_all_done(&mut self) {
        if self.is_complete() && self.completed_at.is_none() {
            self.completed_at = Some(Instant::now());
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple linear chain A -> B -> C.
    fn linear_chain() -> DagDefinition {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "Step A", "action_a"));
        def.add_node(DagNode::new("B", "Step B", "action_b"));
        def.add_node(DagNode::new("C", "Step C", "action_c"));
        def.add_dependency("B", "A");
        def.add_dependency("C", "B");
        def
    }

    /// Helper: build a diamond A -> {B, C} -> D.
    fn diamond_dag() -> DagDefinition {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "Step A", "action_a"));
        def.add_node(DagNode::new("B", "Step B", "action_b"));
        def.add_node(DagNode::new("C", "Step C", "action_c"));
        def.add_node(DagNode::new("D", "Step D", "action_d"));
        def.add_dependency("B", "A");
        def.add_dependency("C", "A");
        def.add_dependency("D", "B");
        def.add_dependency("D", "C");
        def
    }

    #[test]
    fn test_linear_chain_execution() {
        let def = linear_chain();
        let mut exec = DagExecutor::new(def).unwrap();

        exec.run_to_completion(&|node| Ok(serde_json::json!({ "done": node.id.as_str() })));

        assert!(exec.is_complete());
        assert_eq!(
            exec.node_status(&DagNodeId::new("A")),
            Some(&DagNodeStatus::Completed)
        );
        assert_eq!(
            exec.node_status(&DagNodeId::new("B")),
            Some(&DagNodeStatus::Completed)
        );
        assert_eq!(
            exec.node_status(&DagNodeId::new("C")),
            Some(&DagNodeStatus::Completed)
        );
    }

    #[test]
    fn test_diamond_dag() {
        let def = diamond_dag();
        let mut exec = DagExecutor::new(def).unwrap();

        // Step 1: only A should be ready.
        let s1 = exec.step();
        assert_eq!(s1, vec![DagNodeId::new("A")]);
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();

        // Step 2: B and C should be ready.
        let mut s2 = exec.step();
        s2.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(s2, vec![DagNodeId::new("B"), DagNodeId::new("C")]);
        exec.complete_node(&DagNodeId::new("B"), serde_json::json!("ok"))
            .unwrap();
        exec.complete_node(&DagNodeId::new("C"), serde_json::json!("ok"))
            .unwrap();

        // Step 3: D should be ready.
        let s3 = exec.step();
        assert_eq!(s3, vec![DagNodeId::new("D")]);
        exec.complete_node(&DagNodeId::new("D"), serde_json::json!("ok"))
            .unwrap();

        assert!(exec.is_complete());
    }

    #[test]
    fn test_cycle_detection() {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "A", "a"));
        def.add_node(DagNode::new("B", "B", "b"));
        def.add_node(DagNode::new("C", "C", "c"));
        def.add_dependency("B", "A");
        def.add_dependency("C", "B");
        def.add_dependency("A", "C"); // creates cycle A -> B -> C -> A

        assert_eq!(def.validate(), Err(DagError::CycleDetected));
    }

    #[test]
    fn test_topological_sort() {
        let def = linear_chain();
        let order = def.topological_sort().unwrap();
        let ids: Vec<&str> = order.iter().map(|id| id.as_str()).collect();
        let pos_a = ids.iter().position(|&x| x == "A").unwrap();
        let pos_b = ids.iter().position(|&x| x == "B").unwrap();
        let pos_c = ids.iter().position(|&x| x == "C").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_ready_nodes_partial() {
        let def = linear_chain();
        let mut exec = DagExecutor::new(def).unwrap();

        // Initially only A is ready.
        assert_eq!(exec.ready_nodes(), vec![DagNodeId::new("A")]);

        // Start and complete A.
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();

        // Now B is ready but C is still blocked.
        let ready = exec.ready_nodes();
        assert_eq!(ready, vec![DagNodeId::new("B")]);
        assert!(!ready.contains(&DagNodeId::new("C")));
    }

    #[test]
    fn test_fail_propagation() {
        let def = linear_chain();
        let mut exec = DagExecutor::new(def).unwrap();

        // Complete A, start B, then fail B.
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();
        exec.start_node(&DagNodeId::new("B")).unwrap();
        exec.fail_node(&DagNodeId::new("B"), "boom").unwrap();

        assert_eq!(
            exec.node_status(&DagNodeId::new("B")),
            Some(&DagNodeStatus::Failed)
        );

        // C depends on B via Always edge — B is Failed which satisfies Always.
        // So C is actually ready (Always means any terminal status).
        let ready = exec.ready_nodes();
        assert!(ready.contains(&DagNodeId::new("C")));
    }

    #[test]
    fn test_conditional_edge_on_success() {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "A", "a"));
        def.add_node(DagNode::new("B", "B", "b"));
        def.add_edge(DagEdge::new("A", "B").with_condition(EdgeCondition::OnSuccess));

        let mut exec = DagExecutor::new(def).unwrap();
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();

        let ready = exec.ready_nodes();
        assert!(ready.contains(&DagNodeId::new("B")));
    }

    #[test]
    fn test_conditional_edge_on_failure() {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "A", "a"));
        def.add_node(DagNode::new("B", "B", "b"));
        def.add_edge(DagEdge::new("A", "B").with_condition(EdgeCondition::OnFailure));

        let mut exec = DagExecutor::new(def).unwrap();

        // B should NOT be ready when A succeeds.
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();
        assert!(!exec.ready_nodes().contains(&DagNodeId::new("B")));

        // Reset: create fresh executor and fail A this time.
        let mut def2 = DagDefinition::new();
        def2.add_node(DagNode::new("A", "A", "a"));
        def2.add_node(DagNode::new("B", "B", "b"));
        def2.add_edge(DagEdge::new("A", "B").with_condition(EdgeCondition::OnFailure));

        let mut exec2 = DagExecutor::new(def2).unwrap();
        exec2.start_node(&DagNodeId::new("A")).unwrap();
        exec2.fail_node(&DagNodeId::new("A"), "error").unwrap();
        assert!(exec2.ready_nodes().contains(&DagNodeId::new("B")));
    }

    #[test]
    fn test_critical_path() {
        let def = diamond_dag();
        let exec = DagExecutor::new(def).unwrap();
        let path = exec.critical_path();

        // The critical path should have length 3 (A -> B -> D or A -> C -> D).
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], DagNodeId::new("A"));
        assert_eq!(path[2], DagNodeId::new("D"));
        // Middle node is either B or C.
        assert!(path[1] == DagNodeId::new("B") || path[1] == DagNodeId::new("C"));
    }

    #[test]
    fn test_progress_tracking() {
        let def = linear_chain(); // 3 nodes
        let mut exec = DagExecutor::new(def).unwrap();

        assert!((exec.progress() - 0.0).abs() < f32::EPSILON);

        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.complete_node(&DagNodeId::new("A"), serde_json::json!("ok"))
            .unwrap();
        let p1 = exec.progress();
        assert!((p1 - 1.0 / 3.0).abs() < 0.01);

        exec.start_node(&DagNodeId::new("B")).unwrap();
        exec.complete_node(&DagNodeId::new("B"), serde_json::json!("ok"))
            .unwrap();
        let p2 = exec.progress();
        assert!((p2 - 2.0 / 3.0).abs() < 0.01);

        exec.start_node(&DagNodeId::new("C")).unwrap();
        exec.complete_node(&DagNodeId::new("C"), serde_json::json!("ok"))
            .unwrap();
        assert!((exec.progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_dag() {
        let def = DagDefinition::new();
        let exec = DagExecutor::new(def).unwrap();
        assert!(exec.is_complete());
        assert!((exec.progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_export_json() {
        let def = linear_chain();
        let exec = DagExecutor::new(def).unwrap();
        let json = exec.export_json();

        assert!(json.get("nodes").unwrap().is_array());
        assert!(json.get("edges").unwrap().is_array());
        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(json["edges"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_max_retries() {
        let mut def = DagDefinition::new();
        def.add_node(DagNode::new("A", "A", "a").with_max_retries(2));

        let mut exec = DagExecutor::new(def).unwrap();

        // First failure — retry_count becomes 1, status -> Pending (1 <= 2).
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.fail_node(&DagNodeId::new("A"), "err1").unwrap();
        assert_eq!(
            exec.node_status(&DagNodeId::new("A")),
            Some(&DagNodeStatus::Pending)
        );

        // Second failure — retry_count becomes 2, status -> Pending (2 <= 2).
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.fail_node(&DagNodeId::new("A"), "err2").unwrap();
        assert_eq!(
            exec.node_status(&DagNodeId::new("A")),
            Some(&DagNodeStatus::Pending)
        );

        // Third failure — retry_count becomes 3, status -> Failed (3 > 2).
        exec.start_node(&DagNodeId::new("A")).unwrap();
        exec.fail_node(&DagNodeId::new("A"), "err3").unwrap();
        assert_eq!(
            exec.node_status(&DagNodeId::new("A")),
            Some(&DagNodeStatus::Failed)
        );
    }
}
