//! Event-driven workflow engine demo.
//!
//! Run with: cargo run --example workflow_demo --features workflows
//!
//! Demonstrates workflow graph construction, node handlers,
//! checkpointing, and execution with state.

use ai_assistant::{
    SimpleEvent, WorkflowGraph, WorkflowNode, WorkflowRunner, WorkflowState,
};

fn main() {
    println!("=== Event-Driven Workflow Demo ===\n");

    // 1. Create a workflow graph with an entry event
    let mut graph = WorkflowGraph::new("order.placed");

    // 2. Add nodes with handlers
    graph.add_node(WorkflowNode {
        id: "validate".to_string(),
        name: "Validate Order".to_string(),
        handler: Box::new(|input, state| {
            println!("  [validate] Processing input...");
            state.set("validated", serde_json::json!(true));
            Ok(vec![serde_json::json!({
                "event_type": "order.validated",
                "order_id": input["order_id"]
            })])
        }),
        input_type: "order.placed".to_string(),
        output_types: vec!["order.validated".to_string()],
        timeout_ms: None,
    });

    graph.add_node(WorkflowNode {
        id: "fulfill".to_string(),
        name: "Fulfill Order".to_string(),
        handler: Box::new(|_input, state| {
            println!("  [fulfill] Fulfilling order...");
            state.set("fulfilled", serde_json::json!(true));
            Ok(vec![]) // Terminal node — no further events
        }),
        input_type: "order.validated".to_string(),
        output_types: vec![],
        timeout_ms: Some(5000),
    });

    println!("Graph entry event: {}", graph.entry_event());
    println!("Nodes: validate, fulfill");

    // 3. Create a runner (uses in-memory checkpointer by default)
    let runner = WorkflowRunner::new(graph);

    // 4. Create the initial event
    let event = SimpleEvent::new(
        "order.placed",
        serde_json::json!({"order_id": "ORD-001", "total": 99.99}),
    );
    println!("\nInitial event type: {}", event.event_type);

    // 5. Run the workflow
    let state = WorkflowState::new();
    let result = runner.run("wf-001", event.payload.clone(), state);

    match result {
        Ok(res) => {
            println!("\nWorkflow completed:");
            println!("  Steps executed: {}", res.steps_executed);
            println!("  Completed: {}", res.completed);
            println!("  Final state: {:?}", res.final_state.values);
        }
        Err(e) => println!("\nWorkflow error: {}", e),
    }

    println!("\n=== Done ===");
}
