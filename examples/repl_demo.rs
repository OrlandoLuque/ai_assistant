//! Example: Using the REPL engine programmatically
//!
//! Demonstrates creating a REPL engine, processing commands and messages,
//! managing conversation history, and saving/loading sessions — all without
//! interactive stdin.
//!
//! Run with: cargo run --example repl_demo --features full

use ai_assistant::{ReplAction, ReplCommand, ReplConfig, ReplEngine};

fn main() {
    println!("=== REPL Engine Demo ===\n");

    // 1. Create a REPL engine with custom config
    let config = ReplConfig {
        prompt_string: "ai> ".to_string(),
        max_history: 100,
        show_metrics: true,
        show_timestamps: true,
    };
    let mut engine = ReplEngine::new(config);
    println!("Created REPL engine (prompt: {:?})\n", engine.config().prompt_string);

    // 2. Process a /help command
    match engine.process_input("/help") {
        ReplAction::ExecuteCommand(ReplCommand::Help) => {
            println!("{}\n", ReplEngine::format_help());
        }
        other => println!("Unexpected action: {:?}", other),
    }

    // 3. Switch model via /model command
    match engine.process_input("/model llama3:8b") {
        ReplAction::ExecuteCommand(ReplCommand::Model(name)) => {
            engine.set_model(&name);
            println!("Switched model to: {}", engine.current_model());
        }
        _ => {}
    }

    // 4. Process regular messages (simulating a conversation)
    let messages = [
        ("user", "What is Rust?"),
        ("assistant", "Rust is a systems programming language focused on safety and performance."),
        ("user", "How does ownership work?"),
        ("assistant", "Ownership is Rust's core memory management model with three rules: each value has one owner, only one owner at a time, and the value is dropped when the owner goes out of scope."),
    ];

    for (role, content) in &messages {
        if *role == "user" {
            let action = engine.process_input(content);
            assert!(matches!(action, ReplAction::SendMessage(_)));
        }
        engine.add_message(role, content);
    }
    println!("\nConversation has {} messages.", engine.history().len());

    // 5. Show config
    println!("\n{}\n", engine.format_config());

    // 6. Save and reload the session
    let tmp = std::env::temp_dir().join("repl_demo_session.json");
    let path = tmp.to_string_lossy().to_string();
    engine.save_session(&path).expect("save should succeed");
    println!("Session saved to: {}", path);

    let mut engine2 = ReplEngine::new(ReplConfig::default());
    engine2.load_session(&path).expect("load should succeed");
    println!(
        "Session loaded: {} messages, model={}",
        engine2.history().len(),
        engine2.current_model()
    );

    // 7. Show command history
    println!("\nCommand history ({} entries):", engine.command_history().len());
    for (i, entry) in engine.command_history().iter().enumerate() {
        println!("  [{}] {}", i + 1, entry);
    }

    // 8. Exit command
    assert_eq!(engine.process_input("/exit"), ReplAction::Exit);
    println!("\n/exit received — done.");

    // Cleanup
    let _ = std::fs::remove_file(&tmp);
}
