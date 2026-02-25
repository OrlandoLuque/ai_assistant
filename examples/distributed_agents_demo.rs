//! Distributed agents demo — multi-node agent task management.
//!
//! Shows how to create a `DistributedAgentManager`, submit tasks,
//! register worker nodes, claim and complete tasks, and run map-reduce
//! agent jobs.
//!
//! Run with: cargo run --example distributed_agents_demo --features "distributed-agents"

use ai_assistant::{
    AgentNodeInfo, DistributedAgentManager, MapReduceStatus, NodeId,
    TaskDistributionStatus,
};

fn main() {
    println!("=== Distributed Agents Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a manager for the local node
    // -----------------------------------------------------------------------
    let local_id = NodeId::from_string("coordinator-node");
    let mut mgr = DistributedAgentManager::new(local_id);
    println!("Created manager for local node: {:?}", mgr.local_node_id());

    // -----------------------------------------------------------------------
    // 2. Register worker nodes
    // -----------------------------------------------------------------------
    println!("\n--- Registering worker nodes ---");
    let worker_a = NodeId::from_string("worker-alpha");
    let worker_b = NodeId::from_string("worker-beta");

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    mgr.register_node(AgentNodeInfo {
        node_id: worker_a,
        agent_profile: "researcher".to_string(),
        status: "idle".to_string(),
        current_task: None,
        load: 0.1,
        last_heartbeat: now_ms,
    });
    mgr.register_node(AgentNodeInfo {
        node_id: worker_b,
        agent_profile: "coder".to_string(),
        status: "idle".to_string(),
        current_task: None,
        load: 0.3,
        last_heartbeat: now_ms,
    });
    println!("Registered {} nodes", mgr.node_count());
    println!("Active nodes: {}", mgr.active_nodes().len());

    // -----------------------------------------------------------------------
    // 3. Submit tasks with varying priorities
    // -----------------------------------------------------------------------
    println!("\n--- Submitting tasks ---");
    let _t1 = mgr.submit_task("Summarize the quarterly report", "researcher", 5);
    let t2 = mgr.submit_task("Fix authentication bug", "coder", 10);
    let t3 = mgr.submit_task("Update README with new API docs", "writer", 3);

    println!("Submitted {} tasks (pending: {})", mgr.task_count(), mgr.pending_tasks().len());
    for task in mgr.list_tasks() {
        println!("  [{}] prio={} desc=\"{}\"", task.id, task.priority, task.description);
    }

    // -----------------------------------------------------------------------
    // 4. Worker claims a task (gets highest priority first)
    // -----------------------------------------------------------------------
    println!("\n--- Worker alpha claims a task ---");
    if let Some(claimed) = mgr.claim_task(&worker_a) {
        println!(
            "Claimed: {} (prio={}, desc=\"{}\")",
            claimed.id, claimed.priority, claimed.description
        );
        match &claimed.status {
            TaskDistributionStatus::Assigned => println!("  Status: Assigned"),
            _ => println!("  Status: other"),
        }
    }
    println!("Active tasks: {}", mgr.active_task_count());

    // -----------------------------------------------------------------------
    // 5. Complete the claimed task
    // -----------------------------------------------------------------------
    println!("\n--- Completing task {} ---", t2);
    let completed = mgr.complete_task(&t2, "Bug fixed: session token was not refreshed");
    println!("Completed: {}", completed);
    if let Some(task) = mgr.get_task(&t2) {
        println!("  Result: {:?}", task.result);
    }

    // -----------------------------------------------------------------------
    // 6. Cancel a low-priority task
    // -----------------------------------------------------------------------
    println!("\n--- Cancelling task {} ---", t3);
    let cancelled = mgr.cancel_task(&t3);
    println!("Cancelled: {}", cancelled);

    // -----------------------------------------------------------------------
    // 7. Map-Reduce job
    // -----------------------------------------------------------------------
    println!("\n--- Map-Reduce job ---");
    let mr_id = mgr.submit_map_reduce(
        vec![
            "Analyze chapter 1".to_string(),
            "Analyze chapter 2".to_string(),
            "Analyze chapter 3".to_string(),
        ],
        "researcher",
        "Combine all chapter analyses into a single summary",
        "summarizer",
    );
    println!("Created MR job: {}", mr_id);

    // Record map results
    mgr.record_map_result(&mr_id, "Analyze chapter 1", "Chapter 1: introduces the problem space");
    mgr.record_map_result(&mr_id, "Analyze chapter 2", "Chapter 2: proposes solutions");
    mgr.record_map_result(&mr_id, "Analyze chapter 3", "Chapter 3: evaluates results");

    println!("Map phase complete: {}", mgr.is_map_phase_complete(&mr_id));

    // Complete the reduce phase
    mgr.complete_reduce(&mr_id, "Full summary: Problem -> Solution -> Evaluation");

    if let Some(job) = mgr.get_mr_job(&mr_id) {
        let status_str = match &job.status {
            MapReduceStatus::Pending => "Pending",
            MapReduceStatus::Mapping => "Mapping",
            MapReduceStatus::Reducing => "Reducing",
            MapReduceStatus::Completed => "Completed",
            MapReduceStatus::Failed(e) => e.as_str(),
        };
        println!("MR job status: {}", status_str);
        println!("Reduce result: {:?}", job.reduce_result);
    }

    // -----------------------------------------------------------------------
    // 8. Stats summary
    // -----------------------------------------------------------------------
    println!("\n--- Final stats ---");
    println!("Total tasks: {}", mgr.task_count());
    println!("Active tasks: {}", mgr.active_task_count());
    println!("Completed tasks: {}", mgr.completed_tasks().len());
    println!("Pending tasks: {}", mgr.pending_tasks().len());
    println!("Nodes: {}", mgr.node_count());

    println!("\nDistributed agents demo complete.");
}
