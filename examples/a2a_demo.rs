//! Agent-to-Agent (A2A) protocol demo.
//!
//! Run with: cargo run --example a2a_demo --features a2a
//!
//! Demonstrates the Google A2A protocol: agent cards,
//! task lifecycle, JSON-RPC 2.0 messaging, and agent directory.

use ai_assistant::{
    AgentCard, AgentSkill, A2ATask, A2ATaskStatus, JsonRpcRequest, JsonRpcResponse,
    AgentDirectory,
};

fn main() {
    println!("=== Agent-to-Agent (A2A) Protocol Demo ===\n");

    // 1. Create an agent card
    let card = AgentCard::new(
        "CodeAssistant",
        "An AI agent that helps with code review and generation",
        "https://agents.example.com/code-assistant",
    )
    .with_skill(AgentSkill {
        name: "Code Review".to_string(),
        description: "Review code for bugs and improvements".to_string(),
        tags: vec!["code".to_string(), "review".to_string()],
    })
    .with_skill(AgentSkill {
        name: "Code Generation".to_string(),
        description: "Generate code from natural language".to_string(),
        tags: vec!["code".to_string(), "generation".to_string()],
    });

    println!("Agent: {}", card.name);
    println!("URL: {}", card.url);
    println!("Version: {}", card.version);
    println!("Skills: {}", card.skills.len());
    for skill in &card.skills {
        println!("  - {} (tags: {:?})", skill.name, skill.tags);
    }

    // 2. Create a task
    let task = A2ATask::new();
    println!("\nTask ID: {}", task.id);
    println!("Status: {:?}", task.status);

    // 3. Simulate task lifecycle
    let mut task = task;
    task.status = A2ATaskStatus::Working;
    println!("After starting: {:?}", task.status);
    task.status = A2ATaskStatus::Completed;
    println!("After completing: {:?}", task.status);

    // 4. JSON-RPC 2.0 request
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(1),
        method: "tasks/send".to_string(),
        params: Some(serde_json::json!({
            "message": "Review this code",
            "skill": "Code Review"
        })),
    };

    println!("\nJSON-RPC Request:");
    println!("  Method: {}", request.method);
    println!("  Params: {:?}", request.params);

    // 5. JSON-RPC response
    let response = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(1),
        result: Some(serde_json::json!({"task_id": task.id, "status": "completed"})),
        error: None,
    };
    println!("  Response result: {:?}", response.result);

    // 6. Agent directory
    let mut directory = AgentDirectory::new();
    directory.register(card);
    println!("\nAgent directory: {} agents registered", directory.len());

    println!("\n=== Done ===");
}
