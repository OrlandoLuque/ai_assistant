//! Autonomous agent example with tool execution loop.
//!
//! Run with: cargo run --example autonomous_agent --features "autonomous,tools"
//!
//! Demonstrates building an AgentRuntime with a response generator,
//! tool registry, policy, and sandbox, then running it on a task.

use std::sync::Arc;

use ai_assistant::agentic_loop::AgentMessage;
use ai_assistant::{
    AgentPolicyBuilder, AgentRuntime, AutonomyLevel, OperationMode, ToolBuilder,
    UnifiedToolHandler, UnifiedToolOutput, UnifiedToolRegistry,
};

fn main() {
    // 1. Create a mock response generator.
    //    In a real application this would call an LLM (e.g. via Ollama).
    //    The first call returns a tool invocation; the second returns a plain answer.
    let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    let response_generator: Arc<dyn Fn(&[AgentMessage]) -> String + Send + Sync> =
        Arc::new(move |_conversation| {
            let n = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                // First iteration: invoke the "get_time" tool
                "<tool>get_time</tool>".to_string()
            } else {
                // Second iteration: produce a final answer (no tool call)
                "The current time has been retrieved. Task complete.".to_string()
            }
        });

    // 2. Build a tool registry with a simple "get_time" tool.
    let mut registry = UnifiedToolRegistry::new();
    let tool_def = ToolBuilder::new("get_time", "Returns the current timestamp").build();
    let handler: UnifiedToolHandler = Arc::new(|_call| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(UnifiedToolOutput::text(format!(
            "Current timestamp: {}",
            now
        )))
    });
    registry.register(tool_def, handler);

    // 3. Configure the agent policy (full autonomy for this demo).
    let policy = AgentPolicyBuilder::new()
        .autonomy(AutonomyLevel::Autonomous)
        .build();

    // 4. Build and run the agent.
    let mut agent = AgentRuntime::builder("demo-agent", response_generator)
        .max_iterations(5)
        .system_prompt("You are a helpful assistant with access to tools.")
        .policy(policy)
        .tool_registry(registry)
        .mode(OperationMode::Autonomous)
        .build();

    println!("Agent built. Running task...\n");

    match agent.run("What time is it?") {
        Ok(result) => {
            println!("Agent completed successfully:");
            println!("  Output:     {}", result.output);
            println!("  Iterations: {}", result.iterations);
            println!("  Tools used: {:?}", result.tools_called);
            println!("  Cost:       ${:.4}", result.cost);
            println!("  Duration:   {} ms", result.duration_ms);
        }
        Err(e) => {
            println!("Agent error: {}", e);
        }
    }
}
