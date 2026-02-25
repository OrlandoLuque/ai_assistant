//! Programmatic agent graph export and execution trace recording.
//!
//! Provides data structures for modeling agent workflows as directed graphs,
//! exporting them to DOT/Mermaid/JSON formats, and recording execution traces
//! with analytics (critical path, bottlenecks, utilization).

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// EdgeType
// ---------------------------------------------------------------------------

/// The semantic type of an edge connecting two agents in a graph.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    DataFlow,
    Control,
    Delegation,
    Communication,
    Dependency,
}

impl EdgeType {
    /// Returns the edge type as a static snake_case string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DataFlow => "data_flow",
            Self::Control => "control",
            Self::Delegation => "delegation",
            Self::Communication => "communication",
            Self::Dependency => "dependency",
        }
    }
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// StepStatus
// ---------------------------------------------------------------------------

/// Execution status of a single trace step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Running,
    Completed,
    Failed,
    Skipped,
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Skipped => "Skipped",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// GraphError
// ---------------------------------------------------------------------------

/// Errors that can occur during agent graph construction or analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    CycleDetected,
    NodeNotFound(String),
    DuplicateNode(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "Cycle detected in agent graph"),
            Self::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            Self::DuplicateNode(id) => write!(f, "Duplicate node: {}", id),
        }
    }
}

impl std::error::Error for GraphError {}

// ---------------------------------------------------------------------------
// AgentNode
// ---------------------------------------------------------------------------

/// A node in the agent graph representing a single agent.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentNode {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl AgentNode {
    /// Create a new agent node with the given id, name, and type.
    pub fn new(id: &str, name: &str, agent_type: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            agent_type: agent_type.to_string(),
            capabilities: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Builder: add a capability.
    pub fn with_capability(mut self, cap: &str) -> Self {
        self.capabilities.push(cap.to_string());
        self
    }

    /// Builder: add a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// AgentEdge
// ---------------------------------------------------------------------------

/// A directed edge between two agents in the graph.
#[derive(Debug, Clone)]
pub struct AgentEdge {
    pub from: String,
    pub to: String,
    pub edge_type: EdgeType,
    pub label: Option<String>,
    pub weight: f64,
}

impl AgentEdge {
    /// Create a new edge with default weight 1.0 and no label.
    pub fn new(from: &str, to: &str, edge_type: EdgeType) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            edge_type,
            label: None,
            weight: 1.0,
        }
    }

    /// Builder: set a label on the edge.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Builder: set the edge weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

// ---------------------------------------------------------------------------
// AgentGraph
// ---------------------------------------------------------------------------

/// A directed graph of agents and their relationships.
#[derive(Debug, Clone)]
pub struct AgentGraph {
    pub nodes: Vec<AgentNode>,
    pub edges: Vec<AgentEdge>,
}

impl AgentGraph {
    /// Create an empty agent graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: AgentNode) {
        self.nodes.push(node);
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: AgentEdge) {
        self.edges.push(edge);
    }

    /// Remove a node and all edges referencing it.
    pub fn remove_node(&mut self, id: &str) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a node by id.
    pub fn get_node(&self, id: &str) -> Option<&AgentNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Return all nodes reachable via outgoing edges from the given node id.
    pub fn neighbors(&self, id: &str) -> Vec<&AgentNode> {
        let target_ids: Vec<&str> = self
            .edges
            .iter()
            .filter(|e| e.from == id)
            .map(|e| e.to.as_str())
            .collect();
        self.nodes
            .iter()
            .filter(|n| target_ids.contains(&n.id.as_str()))
            .collect()
    }

    /// Topological sort using Kahn's algorithm.
    ///
    /// Returns nodes in dependency order, or `Err(CycleDetected)` if the graph
    /// contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<&AgentNode>, GraphError> {
        let node_ids: Vec<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();

        // Build in-degree map
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in &node_ids {
            in_degree.insert(id, 0);
        }
        for edge in &self.edges {
            if let Some(deg) = in_degree.get_mut(edge.to.as_str()) {
                *deg += 1;
            }
        }

        // Seed queue with zero in-degree nodes
        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut result: Vec<&str> = Vec::new();
        while let Some(id) = queue.pop_front() {
            result.push(id);
            for edge in &self.edges {
                if edge.from == id {
                    if let Some(deg) = in_degree.get_mut(edge.to.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(edge.to.as_str());
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        // Map back to node references
        Ok(result
            .into_iter()
            .filter_map(|id| self.get_node(id))
            .collect())
    }

    /// Export the graph in Graphviz DOT format.
    pub fn export_dot(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph AgentGraph {\n");
        out.push_str("  rankdir=LR;\n");
        for node in &self.nodes {
            out.push_str(&format!(
                "  \"{}\" [label=\"{}\\n({})\"];\n",
                node.id, node.name, node.agent_type
            ));
        }
        for edge in &self.edges {
            out.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                edge.from, edge.to, edge.edge_type
            ));
        }
        out.push_str("}\n");
        out
    }

    /// Export the graph in Mermaid flowchart format.
    pub fn export_mermaid(&self) -> String {
        let mut out = String::new();
        out.push_str("graph LR\n");
        for node in &self.nodes {
            out.push_str(&format!("  {}[\"{}\"]\n", node.id, node.name));
        }
        for edge in &self.edges {
            out.push_str(&format!(
                "  {} --> |{}| {}\n",
                edge.from, edge.edge_type, edge.to
            ));
        }
        out
    }

    /// Export the graph as a JSON string.
    pub fn export_json(&self) -> String {
        let nodes: Vec<serde_json::Value> = self
            .nodes
            .iter()
            .map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "name": n.name,
                    "type": n.agent_type,
                })
            })
            .collect();
        let edges: Vec<serde_json::Value> = self
            .edges
            .iter()
            .map(|e| {
                serde_json::json!({
                    "from": e.from,
                    "to": e.to,
                    "type": e.edge_type.to_string(),
                })
            })
            .collect();
        let json = serde_json::json!({
            "nodes": nodes,
            "edges": edges,
        });
        json.to_string()
    }

    /// Construct an `AgentGraph` from a [`DagDefinition`](crate::dag_executor::DagDefinition).
    pub fn from_dag(dag: &crate::dag_executor::DagDefinition) -> Self {
        use crate::dag_executor::EdgeCondition;

        let mut graph = Self::new();

        for node in &dag.nodes {
            graph.add_node(AgentNode::new(
                node.id.as_str(),
                &node.name,
                &node.action,
            ));
        }

        for edge in &dag.edges {
            let edge_type = match &edge.condition {
                EdgeCondition::Always | EdgeCondition::OnSuccess => EdgeType::DataFlow,
                EdgeCondition::OnFailure => EdgeType::Control,
                EdgeCondition::OnOutput(_) => EdgeType::Communication,
            };
            let label = match &edge.condition {
                EdgeCondition::Always => None,
                other => Some(other.to_string()),
            };
            let mut agent_edge =
                AgentEdge::new(edge.from.as_str(), edge.to.as_str(), edge_type);
            if let Some(lbl) = label {
                agent_edge = agent_edge.with_label(&lbl);
            }
            graph.add_edge(agent_edge);
        }

        graph
    }
}

// ---------------------------------------------------------------------------
// TraceStep
// ---------------------------------------------------------------------------

/// A single recorded step in an execution trace.
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub agent_id: String,
    pub action: String,
    pub input_summary: String,
    pub output_summary: String,
    pub timestamp: u64,
    pub duration_ms: u64,
    pub status: StepStatus,
    pub metadata: HashMap<String, String>,
}

impl TraceStep {
    /// Create a new trace step with the current timestamp.
    pub fn new(agent_id: &str, action: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            agent_id: agent_id.to_string(),
            action: action.to_string(),
            input_summary: String::new(),
            output_summary: String::new(),
            timestamp,
            duration_ms: 0,
            status: StepStatus::Running,
            metadata: HashMap::new(),
        }
    }

    /// Builder: set the input summary.
    pub fn with_input(mut self, summary: &str) -> Self {
        self.input_summary = summary.to_string();
        self
    }

    /// Builder: set the output summary.
    pub fn with_output(mut self, summary: &str) -> Self {
        self.output_summary = summary.to_string();
        self
    }

    /// Builder: set the duration in milliseconds.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Builder: set the step status.
    pub fn with_status(mut self, status: StepStatus) -> Self {
        self.status = status;
        self
    }
}

// ---------------------------------------------------------------------------
// ExecutionTrace
// ---------------------------------------------------------------------------

/// Records a sequence of execution steps for later analysis.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub steps: Vec<TraceStep>,
    pub started_at: Option<u64>,
}

impl ExecutionTrace {
    /// Create an empty execution trace.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            started_at: None,
        }
    }

    /// Record a step in the trace. Sets `started_at` on the first call.
    pub fn record(&mut self, step: TraceStep) {
        if self.started_at.is_none() {
            self.started_at = Some(step.timestamp);
        }
        self.steps.push(step);
    }

    /// Number of recorded steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Total duration from first step start to last step end.
    pub fn duration(&self) -> Duration {
        if self.steps.is_empty() {
            return Duration::ZERO;
        }
        let first_ts = self.steps.first().expect("non-empty: checked above").timestamp;
        let last = self.steps.last().expect("non-empty: checked above");
        let end = last.timestamp + last.duration_ms;
        Duration::from_millis(end.saturating_sub(first_ts))
    }

    /// Filter steps by agent id.
    pub fn filter_by_agent(&self, agent_id: &str) -> Vec<&TraceStep> {
        self.steps
            .iter()
            .filter(|s| s.agent_id == agent_id)
            .collect()
    }

    /// Export steps as a JSON timeline array.
    pub fn export_timeline_json(&self) -> String {
        let items: Vec<serde_json::Value> = self
            .steps
            .iter()
            .map(|s| {
                serde_json::json!({
                    "agent_id": s.agent_id,
                    "action": s.action,
                    "timestamp": s.timestamp,
                    "duration_ms": s.duration_ms,
                    "status": s.status.to_string(),
                })
            })
            .collect();
        serde_json::Value::Array(items).to_string()
    }
}

// ---------------------------------------------------------------------------
// GraphAnalytics
// ---------------------------------------------------------------------------

/// Analytics utilities for agent graphs and execution traces.
pub struct GraphAnalytics;

impl GraphAnalytics {
    /// Compute the critical path through the graph using trace durations.
    ///
    /// For each node in topological order, the longest path to it is computed by
    /// summing `duration_ms` from matching trace steps. If the trace is empty,
    /// the longest path by edge count is used instead.
    pub fn critical_path(graph: &AgentGraph, trace: &ExecutionTrace) -> Vec<String> {
        let sorted = match graph.topological_sort() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        if sorted.is_empty() {
            return Vec::new();
        }

        let use_duration = !trace.steps.is_empty();

        // Build a map of agent_id -> total duration_ms from trace
        let mut duration_map: HashMap<&str, u64> = HashMap::new();
        if use_duration {
            for step in &trace.steps {
                let entry = duration_map.entry(step.agent_id.as_str()).or_insert(0);
                *entry += step.duration_ms;
            }
        }

        // Build adjacency: for each node, which nodes lead into it (predecessors)
        let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &sorted {
            predecessors.insert(node.id.as_str(), Vec::new());
        }
        for edge in &graph.edges {
            if let Some(preds) = predecessors.get_mut(edge.to.as_str()) {
                preds.push(edge.from.as_str());
            }
        }

        // dist[node] = longest path weight ending at node
        let mut dist: HashMap<&str, u64> = HashMap::new();
        // prev[node] = predecessor on the longest path
        let mut prev: HashMap<&str, Option<&str>> = HashMap::new();

        for node in &sorted {
            let id = node.id.as_str();
            let node_weight = if use_duration {
                *duration_map.get(id).unwrap_or(&0)
            } else {
                1
            };

            let preds = predecessors.get(id).cloned().unwrap_or_default();
            if preds.is_empty() {
                dist.insert(id, node_weight);
                prev.insert(id, None);
            } else {
                let mut best_dist = 0u64;
                let mut best_pred: Option<&str> = None;
                for p in &preds {
                    let pd = *dist.get(p).unwrap_or(&0);
                    if pd > best_dist || best_pred.is_none() {
                        best_dist = pd;
                        best_pred = Some(p);
                    }
                }
                dist.insert(id, best_dist + node_weight);
                prev.insert(id, best_pred);
            }
        }

        // Find the node with the largest dist
        let mut end_node: Option<&str> = None;
        let mut max_dist = 0u64;
        for (&id, &d) in &dist {
            if d > max_dist || end_node.is_none() {
                max_dist = d;
                end_node = Some(id);
            }
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = end_node;
        while let Some(id) = current {
            path.push(id.to_string());
            current = *prev.get(id).unwrap_or(&None);
        }
        path.reverse();
        path
    }

    /// Return trace steps whose duration exceeds the given threshold.
    pub fn bottlenecks(trace: &ExecutionTrace, threshold_ms: u64) -> Vec<&TraceStep> {
        trace
            .steps
            .iter()
            .filter(|s| s.duration_ms > threshold_ms)
            .collect()
    }

    /// Compute utilization fraction for each agent (duration / total trace duration).
    pub fn agent_utilization(trace: &ExecutionTrace) -> HashMap<String, f64> {
        let total_ms = trace.duration().as_millis() as f64;
        if total_ms == 0.0 {
            return HashMap::new();
        }

        let mut agent_durations: HashMap<String, u64> = HashMap::new();
        for step in &trace.steps {
            let entry = agent_durations
                .entry(step.agent_id.clone())
                .or_insert(0);
            *entry += step.duration_ms;
        }

        agent_durations
            .into_iter()
            .map(|(id, ms)| (id, ms as f64 / total_ms))
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_type_display() {
        assert_eq!(EdgeType::DataFlow.to_string(), "data_flow");
        assert_eq!(EdgeType::Control.to_string(), "control");
        assert_eq!(EdgeType::Delegation.to_string(), "delegation");
        assert_eq!(EdgeType::Communication.to_string(), "communication");
        assert_eq!(EdgeType::Dependency.to_string(), "dependency");
    }

    #[test]
    fn test_step_status_display() {
        assert_eq!(StepStatus::Running.to_string(), "Running");
        assert_eq!(StepStatus::Completed.to_string(), "Completed");
        assert_eq!(StepStatus::Failed.to_string(), "Failed");
        assert_eq!(StepStatus::Skipped.to_string(), "Skipped");
    }

    #[test]
    fn test_graph_error_display() {
        assert_eq!(
            GraphError::CycleDetected.to_string(),
            "Cycle detected in agent graph"
        );
        assert_eq!(
            GraphError::NodeNotFound("x".to_string()).to_string(),
            "Node not found: x"
        );
        assert_eq!(
            GraphError::DuplicateNode("y".to_string()).to_string(),
            "Duplicate node: y"
        );
        // Verify it implements std::error::Error
        let err: &dyn std::error::Error = &GraphError::CycleDetected;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_agent_node_new() {
        let node = AgentNode::new("a1", "Researcher", "research");
        assert_eq!(node.id, "a1");
        assert_eq!(node.name, "Researcher");
        assert_eq!(node.agent_type, "research");
        assert!(node.capabilities.is_empty());
        assert!(node.metadata.is_empty());
    }

    #[test]
    fn test_agent_node_builder() {
        let node = AgentNode::new("a1", "Researcher", "research")
            .with_capability("web_search")
            .with_capability("summarize")
            .with_metadata("priority", "high");
        assert_eq!(node.capabilities, vec!["web_search", "summarize"]);
        assert_eq!(node.metadata.get("priority").unwrap(), "high");
    }

    #[test]
    fn test_graph_add_remove_nodes() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a1", "A", "type_a"));
        graph.add_node(AgentNode::new("a2", "B", "type_b"));
        graph.add_node(AgentNode::new("a3", "C", "type_c"));
        graph.add_edge(AgentEdge::new("a1", "a2", EdgeType::DataFlow));
        graph.add_edge(AgentEdge::new("a2", "a3", EdgeType::Control));
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);

        graph.remove_node("a2");
        assert_eq!(graph.node_count(), 2);
        // Both edges referenced a2, so both should be removed
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.get_node("a2").is_none());
        assert!(graph.get_node("a1").is_some());
    }

    #[test]
    fn test_graph_neighbors() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a1", "A", "t"));
        graph.add_node(AgentNode::new("a2", "B", "t"));
        graph.add_node(AgentNode::new("a3", "C", "t"));
        graph.add_edge(AgentEdge::new("a1", "a2", EdgeType::DataFlow));
        graph.add_edge(AgentEdge::new("a1", "a3", EdgeType::Delegation));

        let nbrs = graph.neighbors("a1");
        let ids: Vec<&str> = nbrs.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"a2"));
        assert!(ids.contains(&"a3"));
        assert_eq!(ids.len(), 2);

        // a3 has no outgoing edges
        assert!(graph.neighbors("a3").is_empty());
    }

    #[test]
    fn test_graph_topological_sort() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a", "A", "t"));
        graph.add_node(AgentNode::new("b", "B", "t"));
        graph.add_node(AgentNode::new("c", "C", "t"));
        graph.add_edge(AgentEdge::new("a", "b", EdgeType::DataFlow));
        graph.add_edge(AgentEdge::new("b", "c", EdgeType::DataFlow));

        let sorted = graph.topological_sort().unwrap();
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_graph_topological_sort_cycle() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a", "A", "t"));
        graph.add_node(AgentNode::new("b", "B", "t"));
        graph.add_edge(AgentEdge::new("a", "b", EdgeType::DataFlow));
        graph.add_edge(AgentEdge::new("b", "a", EdgeType::DataFlow));

        let result = graph.topological_sort();
        assert_eq!(result, Err(GraphError::CycleDetected));
    }

    #[test]
    fn test_export_dot() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a1", "Researcher", "research"));
        graph.add_node(AgentNode::new("a2", "Writer", "writing"));
        graph.add_edge(AgentEdge::new("a1", "a2", EdgeType::DataFlow));

        let dot = graph.export_dot();
        assert!(dot.contains("digraph AgentGraph"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("\"a1\""));
        assert!(dot.contains("Researcher"));
        assert!(dot.contains("\"a1\" -> \"a2\""));
        assert!(dot.contains("data_flow"));
    }

    #[test]
    fn test_export_mermaid() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a1", "Researcher", "research"));
        graph.add_node(AgentNode::new("a2", "Writer", "writing"));
        graph.add_edge(AgentEdge::new("a1", "a2", EdgeType::DataFlow));

        let mermaid = graph.export_mermaid();
        assert!(mermaid.contains("graph LR"));
        assert!(mermaid.contains("a1[\"Researcher\"]"));
        assert!(mermaid.contains("a2[\"Writer\"]"));
        assert!(mermaid.contains("a1 --> |data_flow| a2"));
    }

    #[test]
    fn test_export_json() {
        let mut graph = AgentGraph::new();
        graph.add_node(AgentNode::new("a1", "Researcher", "research"));
        graph.add_node(AgentNode::new("a2", "Writer", "writing"));
        graph.add_edge(AgentEdge::new("a1", "a2", EdgeType::DataFlow));

        let json_str = graph.export_json();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed["nodes"].is_array());
        assert!(parsed["edges"].is_array());
        assert_eq!(parsed["nodes"][0]["id"], "a1");
        assert_eq!(parsed["edges"][0]["type"], "data_flow");
    }

    #[test]
    fn test_from_dag() {
        use crate::dag_executor::{DagDefinition, DagEdge as DE, DagNode as DN, EdgeCondition};

        let mut dag = DagDefinition::new();
        dag.add_node(DN::new("n1", "Fetch", "fetch_data"));
        dag.add_node(DN::new("n2", "Process", "transform"));
        dag.add_edge(DE::new("n1", "n2").with_condition(EdgeCondition::OnSuccess));

        let graph = AgentGraph::from_dag(&dag);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.get_node("n1").unwrap().name, "Fetch");
        assert_eq!(graph.get_node("n1").unwrap().agent_type, "fetch_data");
        assert_eq!(graph.edges[0].edge_type, EdgeType::DataFlow);
        assert_eq!(graph.edges[0].label.as_deref(), Some("OnSuccess"));
    }

    #[test]
    fn test_execution_trace_record() {
        let mut trace = ExecutionTrace::new();
        assert_eq!(trace.step_count(), 0);

        let s1 = TraceStep::new("a1", "search")
            .with_input("query")
            .with_duration(100)
            .with_status(StepStatus::Completed);
        let s2 = TraceStep::new("a2", "write")
            .with_output("article")
            .with_duration(200)
            .with_status(StepStatus::Completed);
        let s3 = TraceStep::new("a1", "verify")
            .with_duration(50)
            .with_status(StepStatus::Failed);

        trace.record(s1);
        trace.record(s2);
        trace.record(s3);

        assert_eq!(trace.step_count(), 3);
        assert!(trace.started_at.is_some());

        let a1_steps = trace.filter_by_agent("a1");
        assert_eq!(a1_steps.len(), 2);
        assert_eq!(a1_steps[0].action, "search");
        assert_eq!(a1_steps[1].action, "verify");
    }

    #[test]
    fn test_execution_trace_timeline_json() {
        let mut trace = ExecutionTrace::new();
        let s1 = TraceStep::new("a1", "search")
            .with_duration(100)
            .with_status(StepStatus::Completed);
        trace.record(s1);

        let json_str = trace.export_timeline_json();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed[0]["agent_id"], "a1");
        assert_eq!(parsed[0]["action"], "search");
        assert_eq!(parsed[0]["status"], "Completed");
    }

    #[test]
    fn test_bottlenecks() {
        let mut trace = ExecutionTrace::new();
        let base_ts = 1000u64;
        let mut s1 = TraceStep::new("a1", "fast");
        s1.timestamp = base_ts;
        s1.duration_ms = 50;
        s1.status = StepStatus::Completed;

        let mut s2 = TraceStep::new("a2", "slow");
        s2.timestamp = base_ts + 50;
        s2.duration_ms = 500;
        s2.status = StepStatus::Completed;

        let mut s3 = TraceStep::new("a3", "medium");
        s3.timestamp = base_ts + 550;
        s3.duration_ms = 150;
        s3.status = StepStatus::Completed;

        trace.record(s1);
        trace.record(s2);
        trace.record(s3);

        let bottlenecks = GraphAnalytics::bottlenecks(&trace, 100);
        assert_eq!(bottlenecks.len(), 2);
        assert_eq!(bottlenecks[0].action, "slow");
        assert_eq!(bottlenecks[1].action, "medium");
    }

    #[test]
    fn test_agent_utilization() {
        let mut trace = ExecutionTrace::new();
        let base_ts = 1000u64;

        let mut s1 = TraceStep::new("a1", "work");
        s1.timestamp = base_ts;
        s1.duration_ms = 300;
        s1.status = StepStatus::Completed;

        let mut s2 = TraceStep::new("a2", "work");
        s2.timestamp = base_ts + 300;
        s2.duration_ms = 700;
        s2.status = StepStatus::Completed;

        trace.record(s1);
        trace.record(s2);

        let util = GraphAnalytics::agent_utilization(&trace);
        // Total duration = (300 + 700) - 0 = 1000 from timestamp perspective:
        // first timestamp = 1000, last end = 1000 + 300 + 700 = but actually
        // last step: ts=1300, dur=700 => end=2000. first ts=1000. total=1000ms
        let a1 = util.get("a1").unwrap();
        let a2 = util.get("a2").unwrap();
        // a1: 300/1000 = 0.3, a2: 700/1000 = 0.7
        assert!((a1 - 0.3).abs() < 0.01);
        assert!((a2 - 0.7).abs() < 0.01);
        assert!(((a1 + a2) - 1.0).abs() < 0.01);
    }
}
