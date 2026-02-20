//! Multi-agent orchestration example.
//!
//! Run with: cargo run --example multi_agent --features multi-agent
//!
//! Demonstrates creating agents, assigning tasks, and tracking execution.

use ai_assistant::{Agent, AgentOrchestrator, AgentRole, AgentTask, OrchestrationStrategy};

fn main() {
    // Create an orchestrator with parallel execution strategy
    let mut orchestrator = AgentOrchestrator::new(OrchestrationStrategy::Parallel);

    // Register agents with different roles (using builder pattern)
    let researcher = Agent::new("agent-1", "Researcher", AgentRole::Researcher)
        .with_capability("web_search")
        .with_capability("document_analysis")
        .with_model("llama3");

    let analyst = Agent::new("agent-2", "Analyst", AgentRole::Analyst)
        .with_capability("data_analysis")
        .with_capability("summarization")
        .with_model("llama3");

    let writer = Agent::new("agent-3", "Writer", AgentRole::Writer)
        .with_capability("content_generation")
        .with_capability("editing")
        .with_model("llama3");

    orchestrator.register_agent(researcher);
    orchestrator.register_agent(analyst);
    orchestrator.register_agent(writer);

    // Create tasks with dependencies (using builder pattern)
    let research_task = AgentTask::new("task-1", "Research recent advances in Rust async runtime")
        .with_priority(10);

    let analysis_task = AgentTask::new("task-2", "Analyze findings and identify key patterns")
        .depends_on("task-1") // Depends on research
        .with_priority(8);

    let writing_task = AgentTask::new("task-3", "Write summary report with recommendations")
        .depends_on("task-2") // Depends on analysis
        .with_priority(6);

    orchestrator.add_task(research_task);
    orchestrator.add_task(analysis_task);
    orchestrator.add_task(writing_task);

    // Check orchestration status
    let status = orchestrator.get_status();
    println!("Orchestration Status:");
    println!(
        "  Agents: {} total ({} idle)",
        status.total_agents, status.idle_agents
    );
    println!(
        "  Tasks:  {} total ({} pending, {} in-progress)",
        status.total_tasks, status.pending_tasks, status.in_progress_tasks
    );

    // Auto-assign available tasks to best-fit agents
    let assignments = orchestrator.auto_assign_tasks();
    println!("\nAuto-assignments:");
    for (task_id, agent_id) in &assignments {
        println!("  {} -> {}", task_id, agent_id);
    }
}
