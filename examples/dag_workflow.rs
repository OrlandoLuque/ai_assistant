//! DAG workflow execution example.
//!
//! Run with: cargo run --example dag_workflow
//!
//! Demonstrates building a directed acyclic graph (DAG) workflow,
//! validating it, executing nodes with a handler, and inspecting
//! progress, critical path, and topological ordering.
//! No feature flags required.

use ai_assistant::{DagDefinition, DagEdge, DagExecutor, DagNode, EdgeCondition};

fn main() {
    // 1. Define a data-processing workflow as a DAG
    //
    //    fetch_data ──> parse_data ──> analyze ──> generate_report
    //                                    │
    //                              validate_data
    //
    let mut dag = DagDefinition::new();

    dag.add_node(DagNode::new("fetch", "Fetch Data", "http_get"));
    dag.add_node(DagNode::new("parse", "Parse Data", "json_parse"));
    dag.add_node(DagNode::new("validate", "Validate Data", "schema_check").with_max_retries(2));
    dag.add_node(DagNode::new("analyze", "Analyze Data", "statistics"));
    dag.add_node(DagNode::new("report", "Generate Report", "render_pdf").with_timeout(5000));

    // Edges: fetch -> parse -> validate -> analyze -> report
    dag.add_edge(DagEdge::new("fetch", "parse"));
    dag.add_edge(DagEdge::new("parse", "validate"));
    dag.add_edge(DagEdge::new("validate", "analyze").with_condition(EdgeCondition::OnSuccess));
    dag.add_edge(DagEdge::new("analyze", "report").with_condition(EdgeCondition::OnSuccess));

    // 2. Validate (no cycles)
    dag.validate().expect("DAG validation failed");
    println!(
        "DAG validated: {} nodes, {} edges",
        dag.node_count(),
        dag.edges.len()
    );

    // 3. Topological sort
    let topo = dag.topological_sort().expect("topological sort failed");
    println!(
        "Topological order: {}",
        topo.iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>()
            .join(" -> ")
    );

    // 4. Create an executor and inspect ready nodes
    let mut executor = DagExecutor::new(dag)
        .expect("executor creation failed")
        .with_max_parallel(2);

    let ready = executor.ready_nodes();
    println!(
        "\nInitially ready: {:?}",
        ready.iter().map(|id| id.as_str()).collect::<Vec<_>>()
    );

    // 5. Critical path analysis
    let critical = executor.critical_path();
    println!(
        "Critical path:   {}",
        critical
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>()
            .join(" -> ")
    );

    // 6. Run to completion with a handler
    println!("\n--- Executing workflow ---");
    executor.run_to_completion(&|node: &DagNode| {
        println!(
            "  Running '{}' (action: {}, retries: {}/{})",
            node.name, node.action, node.retry_count, node.max_retries
        );
        // Simulate work: return a JSON result
        Ok(serde_json::json!({
            "node": node.id.as_str(),
            "status": "ok",
            "output_size": 1024
        }))
    });

    // 7. Check completion state
    println!("\nWorkflow complete: {}", executor.is_complete());
    println!("Progress: {:.0}%", executor.progress() * 100.0);

    // 8. Inspect final node states and results
    println!("\nNode results:");
    if let Ok(order) = executor.execution_order() {
        for id in &order {
            if let Some(node) = executor.nodes.get(id) {
                let result_summary = node
                    .result
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(none)".to_string());
                println!(
                    "  {:<20} {:>10}  result: {}",
                    node.name,
                    node.status.to_string(),
                    if result_summary.len() > 60 {
                        format!("{}...", &result_summary[..57])
                    } else {
                        result_summary
                    }
                );
            }
        }
    }
}
