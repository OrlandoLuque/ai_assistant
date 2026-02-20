//! Example: UI Framework Hooks
//!
//! Demonstrates ChatHooks event subscription, StreamAdapter for converting
//! raw chunks to typed events, and ChatSession state management.
//!
//! Run with: cargo run --example ui_hooks_demo

use ai_assistant::{
    ChatHooks, ChatStatus, ChatStreamEvent, StreamAdapter, UsageInfo,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() {
    println!("=== UI Framework Hooks Demo ===\n");

    // 1. Event subscription
    println!("--- ChatHooks: Event Subscription ---");
    let mut hooks = ChatHooks::new();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    hooks.on_event(Box::new(move |event| {
        counter_clone.fetch_add(1, Ordering::SeqCst);
        println!("  Event: {} -> {}", event.event_type(), event.to_json());
    }));

    hooks.emit(ChatStreamEvent::StatusChange {
        status: ChatStatus::Thinking,
    });
    hooks.emit(ChatStreamEvent::MessageStart {
        id: "msg-1".to_string(),
        role: "assistant".to_string(),
    });
    hooks.emit(ChatStreamEvent::MessageDelta {
        id: "msg-1".to_string(),
        content_chunk: "Hello, ".to_string(),
    });
    hooks.emit(ChatStreamEvent::MessageDelta {
        id: "msg-1".to_string(),
        content_chunk: "world!".to_string(),
    });
    hooks.emit(ChatStreamEvent::MessageEnd {
        id: "msg-1".to_string(),
        finish_reason: "stop".to_string(),
        usage: Some(UsageInfo::new(100, 50)),
    });
    hooks.emit(ChatStreamEvent::StatusChange {
        status: ChatStatus::Idle,
    });
    println!("  Total events received: {}\n", counter.load(Ordering::SeqCst));

    // 2. StreamAdapter
    println!("--- StreamAdapter: Converting Chunks ---");
    let chunks = vec!["Hello", " from", " streaming!"];
    let events = StreamAdapter::from_chunks(&chunks);
    for event in &events {
        println!("  {} -> {}", event.event_type(), event.to_json());
    }

    println!("\n--- StreamAdapter: Tool Call ---");
    let tool_events = StreamAdapter::from_tool_call(
        "web_search",
        r#"{"query": "Rust programming"}"#,
        r#"{"results": ["rust-lang.org"]}"#,
    );
    for event in &tool_events {
        println!("  {}", event.event_type());
    }

    // 3. ChatSession state management
    println!("\n--- ChatSession: State Management ---");
    let mut session = ai_assistant::ui_hooks::ChatSession::new();
    session.set_status(ChatStatus::Idle);
    session.add_message("user", "What is Rust?");
    session.add_message("assistant", "Rust is a systems programming language.");
    session.add_message("user", "Tell me more.");
    session.add_message("assistant", "It focuses on safety and performance.");

    println!("  Messages: {}", session.message_count());
    println!("  Status: {:?}", session.status);
    println!("  Last assistant: {:?}", session.last_assistant_message());
    println!("  Session JSON (first 200 chars): {}...", &session.to_json()[..200.min(session.to_json().len())]);

    println!("\nDone!");
}
