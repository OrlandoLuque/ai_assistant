//! Basic chat example with a local LLM provider.
//!
//! Run with: cargo run --example basic_chat
//!
//! Requires a running Ollama instance at localhost:11434.

use ai_assistant::{AiAssistant, AiConfig, AiProvider};

fn main() {
    // Configure for Ollama (default local LLM provider)
    let mut config = AiConfig::default();
    config.provider = AiProvider::Ollama;
    config.selected_model = "llama3".to_string();
    config.ollama_url = "http://localhost:11434".to_string();

    // Create assistant instance
    let mut assistant = AiAssistant::new();
    assistant.load_config(config);

    // Discover available models
    assistant.fetch_models();
    println!("Available models:");
    for model in &assistant.available_models {
        println!("  - {} ({})", model.name, model.provider.display_name());
    }

    // Set a system prompt
    assistant.set_system_prompt("You are a helpful coding assistant. Be concise.");

    // Send a message (requires running LLM)
    assistant.send_message_simple("What is the Rust ownership model in one sentence?".to_string());

    // Access the conversation history
    for msg in &assistant.conversation {
        println!("[{}] {}", msg.role, msg.content);
    }
}
