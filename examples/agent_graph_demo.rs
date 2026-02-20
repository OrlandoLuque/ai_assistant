//! Example: Agent Graph Visualization
//!
//! Demonstrates building agent graphs, exporting to DOT/Mermaid/JSON,
//! recording execution traces, and computing analytics.
//!
//! Run with: cargo run --example agent_graph_demo

use ai_assistant::agent_graph::{
    AgentEdge, AgentGraph, AgentNode, EdgeType, ExecutionTrace, GraphAnalytics, StepStatus,
    TraceStep,
};
use ai_assistant::{DagDefinition, DagNode, DagEdge};

fn main() {
    println!("=== Agent Graph Visualization Demo ===\n");

    // 1. Build a graph manually
    println!("--- Manual Graph Construction ---");
    let mut graph = AgentGraph::new();
    graph.add_node(
        AgentNode::new("researcher", "Researcher", "research")
            .with_capability("web_search")
            .with_capability("summarize"),
    );
    graph.add_node(
        AgentNode::new("writer", "Writer", "writing")
            .with_capability("draft")
            .with_capability("edit"),
    );
    graph.add_node(
        AgentNode::new("reviewer", "Reviewer", "review")
            .with_capability("critique"),
    );
    graph.add_edge(AgentEdge::new("researcher", "writer", EdgeType::DataFlow));
    graph.add_edge(
        AgentEdge::new("writer", "reviewer", EdgeType::DataFlow)
            .with_label("draft"),
    );
    graph.add_edge(
        AgentEdge::new("reviewer", "writer", EdgeType::Control)
            .with_label("revision"),
    );

    println!("  Nodes: {}, Edges: {}", graph.node_count(), graph.edge_count());

    // 2. Export formats
    println!("\n--- DOT Export ---");
    println!("{}", graph.export_dot());

    println!("--- Mermaid Export ---");
    println!("{}", graph.export_mermaid());

    println!("--- JSON Export ---");
    println!("{}\n", graph.export_json());

    // 3. Build from DAG
    println!("--- From DAG Definition ---");
    let mut dag = DagDefinition::new();
    dag.add_node(DagNode::new("fetch", "Fetch Data", "http_get"));
    dag.add_node(DagNode::new("process", "Process Data", "transform"));
    dag.add_node(DagNode::new("store", "Store Results", "db_write"));
    dag.add_edge(DagEdge::new("fetch", "process"));
    dag.add_edge(DagEdge::new("process", "store"));

    let dag_graph = AgentGraph::from_dag(&dag);
    println!("  DAG graph: {} nodes, {} edges", dag_graph.node_count(), dag_graph.edge_count());
    println!("{}", dag_graph.export_mermaid());

    // 4. Execution trace
    println!("--- Execution Trace ---");
    let mut trace = ExecutionTrace::new();
    trace.record(
        TraceStep::new("researcher", "web_search")
            .with_input("query: Rust AI frameworks")
            .with_output("Found 5 results")
            .with_duration(1200)
            .with_status(StepStatus::Completed),
    );
    trace.record(
        TraceStep::new("writer", "draft")
            .with_input("5 research results")
            .with_output("Draft: 500 words")
            .with_duration(3500)
            .with_status(StepStatus::Completed),
    );
    trace.record(
        TraceStep::new("reviewer", "critique")
            .with_input("Draft: 500 words")
            .with_output("2 revision suggestions")
            .with_duration(800)
            .with_status(StepStatus::Completed),
    );
    trace.record(
        TraceStep::new("writer", "revise")
            .with_input("2 revision suggestions")
            .with_output("Final: 550 words")
            .with_duration(2000)
            .with_status(StepStatus::Completed),
    );

    println!("  Steps: {}", trace.step_count());
    println!("  Duration: {:?}", trace.duration());
    println!("  Writer steps: {}", trace.filter_by_agent("writer").len());

    // 5. Analytics
    println!("\n--- Graph Analytics ---");
    let bottlenecks = GraphAnalytics::bottlenecks(&trace, 2000);
    println!("  Bottlenecks (>2000ms): {}", bottlenecks.len());
    for step in &bottlenecks {
        println!("    - {} / {} ({}ms)", step.agent_id, step.action, step.duration_ms);
    }

    let utilization = GraphAnalytics::agent_utilization(&trace);
    println!("  Agent utilization:");
    for (agent, pct) in &utilization {
        println!("    - {}: {:.1}%", agent, pct * 100.0);
    }

    let critical = GraphAnalytics::critical_path(&graph, &trace);
    println!("  Critical path: {:?}", critical);

    println!("\nDone!");
}
