//! UI framework integration hooks for frontend frameworks (React, Vue, Svelte, WASM).
//!
//! Provides typed event streaming infrastructure that bridges the AI assistant
//! backend with frontend UI frameworks. Includes chat status tracking, stream
//! event generation, subscriber-based hooks, and session state management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// ChatStatus
// ---------------------------------------------------------------------------

/// Represents the current status of a chat interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChatStatus {
    Idle,
    Thinking,
    Streaming,
    ToolCalling,
    Error,
}

impl fmt::Display for ChatStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ChatStatus::Idle => "idle",
            ChatStatus::Thinking => "thinking",
            ChatStatus::Streaming => "streaming",
            ChatStatus::ToolCalling => "toolcalling",
            ChatStatus::Error => "error",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// UsageInfo
// ---------------------------------------------------------------------------

/// Token usage information for a completed message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: Option<f64>,
}

impl UsageInfo {
    /// Creates a new `UsageInfo` with the given input and output token counts.
    /// `total_tokens` is computed as `input + output`.
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cost_usd: None,
        }
    }

    /// Builder-style method to attach a USD cost estimate.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_usd = Some(cost);
        self
    }
}

// ---------------------------------------------------------------------------
// ChatStreamEvent
// ---------------------------------------------------------------------------

/// Events emitted during a streaming chat interaction.
#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    MessageStart { id: String, role: String },
    MessageDelta { id: String, content_chunk: String },
    MessageEnd { id: String, finish_reason: String, usage: Option<UsageInfo> },
    ToolCallStart { id: String, name: String },
    ToolCallDelta { id: String, args_chunk: String },
    ToolCallEnd { id: String, result: String },
    Error { message: String },
    StatusChange { status: ChatStatus },
}

impl ChatStreamEvent {
    /// Returns a static string identifying the event type.
    pub fn event_type(&self) -> &'static str {
        match self {
            ChatStreamEvent::MessageStart { .. } => "message_start",
            ChatStreamEvent::MessageDelta { .. } => "message_delta",
            ChatStreamEvent::MessageEnd { .. } => "message_end",
            ChatStreamEvent::ToolCallStart { .. } => "tool_call_start",
            ChatStreamEvent::ToolCallDelta { .. } => "tool_call_delta",
            ChatStreamEvent::ToolCallEnd { .. } => "tool_call_end",
            ChatStreamEvent::Error { .. } => "error",
            ChatStreamEvent::StatusChange { .. } => "status_change",
        }
    }

    /// Serializes the event to a JSON string with `{"type": "...", "data": {...}}` shape.
    pub fn to_json(&self) -> String {
        let event_type = self.event_type();
        let data = match self {
            ChatStreamEvent::MessageStart { id, role } => {
                serde_json::json!({ "id": id, "role": role })
            }
            ChatStreamEvent::MessageDelta { id, content_chunk } => {
                serde_json::json!({ "id": id, "content_chunk": content_chunk })
            }
            ChatStreamEvent::MessageEnd { id, finish_reason, usage } => {
                serde_json::json!({ "id": id, "finish_reason": finish_reason, "usage": usage })
            }
            ChatStreamEvent::ToolCallStart { id, name } => {
                serde_json::json!({ "id": id, "name": name })
            }
            ChatStreamEvent::ToolCallDelta { id, args_chunk } => {
                serde_json::json!({ "id": id, "args_chunk": args_chunk })
            }
            ChatStreamEvent::ToolCallEnd { id, result } => {
                serde_json::json!({ "id": id, "result": result })
            }
            ChatStreamEvent::Error { message } => {
                serde_json::json!({ "message": message })
            }
            ChatStreamEvent::StatusChange { status } => {
                serde_json::json!({ "status": status.to_string() })
            }
        };
        serde_json::json!({ "type": event_type, "data": data }).to_string()
    }
}

// ---------------------------------------------------------------------------
// ChatHooks
// ---------------------------------------------------------------------------

/// Subscriber-based event hook system for streaming chat events to UI layers.
pub struct ChatHooks {
    subscribers: Vec<Box<dyn Fn(&ChatStreamEvent) + Send + Sync>>,
    event_count: usize,
}

impl ChatHooks {
    /// Creates a new `ChatHooks` with no subscribers.
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
            event_count: 0,
        }
    }

    /// Registers a callback that will be invoked for every emitted event.
    pub fn on_event(&mut self, callback: Box<dyn Fn(&ChatStreamEvent) + Send + Sync>) {
        self.subscribers.push(callback);
    }

    /// Broadcasts an event to all subscribers and increments the event count.
    pub fn emit(&mut self, event: ChatStreamEvent) {
        for subscriber in &self.subscribers {
            subscriber(&event);
        }
        self.event_count += 1;
    }

    /// Returns the total number of events emitted so far.
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Returns the current number of registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Removes all subscribers and resets the event count to zero.
    pub fn clear(&mut self) {
        self.subscribers.clear();
        self.event_count = 0;
    }
}

// ---------------------------------------------------------------------------
// StreamAdapter
// ---------------------------------------------------------------------------

/// Utility for converting raw data into sequences of [`ChatStreamEvent`]s.
pub struct StreamAdapter;

impl StreamAdapter {
    /// Converts a slice of text chunks into a complete message event sequence:
    /// `MessageStart` + N x `MessageDelta` + `MessageEnd`.
    ///
    /// For empty input, still produces `MessageStart` + `MessageEnd`.
    pub fn from_chunks(chunks: &[&str]) -> Vec<ChatStreamEvent> {
        let mut events = Vec::new();
        events.push(ChatStreamEvent::MessageStart {
            id: "msg-1".to_string(),
            role: "assistant".to_string(),
        });
        for chunk in chunks {
            events.push(ChatStreamEvent::MessageDelta {
                id: "msg-1".to_string(),
                content_chunk: chunk.to_string(),
            });
        }
        events.push(ChatStreamEvent::MessageEnd {
            id: "msg-1".to_string(),
            finish_reason: "stop".to_string(),
            usage: None,
        });
        events
    }

    /// Converts a tool call specification into a three-event sequence:
    /// `ToolCallStart` + `ToolCallDelta` + `ToolCallEnd`.
    pub fn from_tool_call(name: &str, args: &str, result: &str) -> Vec<ChatStreamEvent> {
        vec![
            ChatStreamEvent::ToolCallStart {
                id: "tc-1".to_string(),
                name: name.to_string(),
            },
            ChatStreamEvent::ToolCallDelta {
                id: "tc-1".to_string(),
                args_chunk: args.to_string(),
            },
            ChatStreamEvent::ToolCallEnd {
                id: "tc-1".to_string(),
                result: result.to_string(),
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single chat message with metadata, suitable for UI rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

impl ChatMessage {
    /// Creates a new message with an auto-generated id (`msg-{timestamp}`) and
    /// the current Unix timestamp in seconds.
    pub fn new(role: &str, content: &str) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: format!("msg-{}", ts),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: ts,
            metadata: HashMap::new(),
        }
    }

    /// Builder-style method to add a metadata key-value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------

/// Tracks the state of an ongoing chat session for UI consumption.
pub struct ChatSession {
    pub messages: Vec<ChatMessage>,
    pub status: ChatStatus,
    pub is_loading: bool,
}

impl ChatSession {
    /// Creates a new empty session in `Idle` state.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            status: ChatStatus::Idle,
            is_loading: false,
        }
    }

    /// Adds a new message to the session.
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage::new(role, content));
    }

    /// Returns the content of the last message with role `"assistant"`, if any.
    pub fn last_assistant_message(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.as_str())
    }

    /// Returns the number of messages in the session.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Updates the chat status. Also sets `is_loading` to `true` when the
    /// status is `Thinking`, `Streaming`, or `ToolCalling`, and `false`
    /// otherwise.
    pub fn set_status(&mut self, status: ChatStatus) {
        self.is_loading = matches!(
            status,
            ChatStatus::Thinking | ChatStatus::Streaming | ChatStatus::ToolCalling
        );
        self.status = status;
    }

    /// Serializes the session to a JSON string.
    pub fn to_json(&self) -> String {
        let messages_json: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                })
            })
            .collect();
        serde_json::json!({
            "messages": messages_json,
            "status": self.status.to_string(),
            "is_loading": self.is_loading,
        })
        .to_string()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_chat_status_display() {
        assert_eq!(ChatStatus::Idle.to_string(), "idle");
        assert_eq!(ChatStatus::Thinking.to_string(), "thinking");
        assert_eq!(ChatStatus::Streaming.to_string(), "streaming");
        assert_eq!(ChatStatus::ToolCalling.to_string(), "toolcalling");
        assert_eq!(ChatStatus::Error.to_string(), "error");
    }

    #[test]
    fn test_usage_info_new() {
        let usage = UsageInfo::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert!(usage.cost_usd.is_none());
    }

    #[test]
    fn test_usage_info_with_cost() {
        let usage = UsageInfo::new(200, 100).with_cost(0.005);
        assert_eq!(usage.total_tokens, 300);
        assert_eq!(usage.cost_usd, Some(0.005));
    }

    #[test]
    fn test_chat_stream_event_type() {
        assert_eq!(
            ChatStreamEvent::MessageStart { id: String::new(), role: String::new() }.event_type(),
            "message_start"
        );
        assert_eq!(
            ChatStreamEvent::MessageDelta { id: String::new(), content_chunk: String::new() }.event_type(),
            "message_delta"
        );
        assert_eq!(
            ChatStreamEvent::MessageEnd { id: String::new(), finish_reason: String::new(), usage: None }.event_type(),
            "message_end"
        );
        assert_eq!(
            ChatStreamEvent::ToolCallStart { id: String::new(), name: String::new() }.event_type(),
            "tool_call_start"
        );
        assert_eq!(
            ChatStreamEvent::ToolCallDelta { id: String::new(), args_chunk: String::new() }.event_type(),
            "tool_call_delta"
        );
        assert_eq!(
            ChatStreamEvent::ToolCallEnd { id: String::new(), result: String::new() }.event_type(),
            "tool_call_end"
        );
        assert_eq!(
            ChatStreamEvent::Error { message: String::new() }.event_type(),
            "error"
        );
        assert_eq!(
            ChatStreamEvent::StatusChange { status: ChatStatus::Idle }.event_type(),
            "status_change"
        );
    }

    #[test]
    fn test_chat_stream_event_to_json() {
        let event = ChatStreamEvent::MessageStart {
            id: "m1".to_string(),
            role: "assistant".to_string(),
        };
        let json = event.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "message_start");
        assert_eq!(parsed["data"]["id"], "m1");
        assert_eq!(parsed["data"]["role"], "assistant");
    }

    #[test]
    fn test_chat_hooks_emit_and_count() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut hooks = ChatHooks::new();
        hooks.on_event(Box::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        hooks.emit(ChatStreamEvent::StatusChange { status: ChatStatus::Thinking });
        hooks.emit(ChatStreamEvent::StatusChange { status: ChatStatus::Streaming });
        hooks.emit(ChatStreamEvent::StatusChange { status: ChatStatus::Idle });

        assert_eq!(hooks.event_count(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_chat_hooks_multiple_subscribers() {
        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));
        let ca = counter_a.clone();
        let cb = counter_b.clone();

        let mut hooks = ChatHooks::new();
        hooks.on_event(Box::new(move |_| { ca.fetch_add(1, Ordering::SeqCst); }));
        hooks.on_event(Box::new(move |_| { cb.fetch_add(1, Ordering::SeqCst); }));

        hooks.emit(ChatStreamEvent::Error { message: "test".to_string() });

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_chat_hooks_clear() {
        let mut hooks = ChatHooks::new();
        hooks.on_event(Box::new(|_| {}));
        hooks.on_event(Box::new(|_| {}));
        assert_eq!(hooks.subscriber_count(), 2);

        hooks.emit(ChatStreamEvent::StatusChange { status: ChatStatus::Idle });
        assert_eq!(hooks.event_count(), 1);

        hooks.clear();
        assert_eq!(hooks.subscriber_count(), 0);
        assert_eq!(hooks.event_count(), 0);
    }

    #[test]
    fn test_stream_adapter_from_chunks() {
        let events = StreamAdapter::from_chunks(&["Hello", " world"]);
        assert_eq!(events.len(), 4); // Start + 2 Deltas + End
        assert_eq!(events[0].event_type(), "message_start");
        assert_eq!(events[1].event_type(), "message_delta");
        assert_eq!(events[2].event_type(), "message_delta");
        assert_eq!(events[3].event_type(), "message_end");

        // Verify delta content
        if let ChatStreamEvent::MessageDelta { content_chunk, .. } = &events[1] {
            assert_eq!(content_chunk, "Hello");
        } else {
            panic!("Expected MessageDelta");
        }
    }

    #[test]
    fn test_stream_adapter_from_chunks_empty() {
        let events = StreamAdapter::from_chunks(&[]);
        assert_eq!(events.len(), 2); // Start + End
        assert_eq!(events[0].event_type(), "message_start");
        assert_eq!(events[1].event_type(), "message_end");
    }

    #[test]
    fn test_stream_adapter_from_tool_call() {
        let events = StreamAdapter::from_tool_call("search", "{\"q\":\"rust\"}", "found 10 results");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type(), "tool_call_start");
        assert_eq!(events[1].event_type(), "tool_call_delta");
        assert_eq!(events[2].event_type(), "tool_call_end");

        if let ChatStreamEvent::ToolCallStart { name, .. } = &events[0] {
            assert_eq!(name, "search");
        } else {
            panic!("Expected ToolCallStart");
        }
    }

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new("user", "hello");
        assert!(msg.id.starts_with("msg-"));
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "hello");
        assert!(msg.timestamp > 0);
        assert!(msg.metadata.is_empty());
    }

    #[test]
    fn test_chat_session_add_message() {
        let mut session = ChatSession::new();
        session.add_message("user", "hi");
        session.add_message("assistant", "hello");
        assert_eq!(session.message_count(), 2);
    }

    #[test]
    fn test_chat_session_last_assistant_message() {
        let mut session = ChatSession::new();
        session.add_message("user", "question 1");
        session.add_message("assistant", "answer 1");
        session.add_message("user", "question 2");
        session.add_message("assistant", "answer 2");

        assert_eq!(session.last_assistant_message(), Some("answer 2"));
    }

    #[test]
    fn test_chat_session_to_json() {
        let mut session = ChatSession::new();
        session.add_message("user", "test");
        let json = session.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["messages"].is_array());
        assert_eq!(parsed["status"], "idle");
        assert_eq!(parsed["is_loading"], false);
    }

    #[test]
    fn test_chat_session_status_transitions() {
        let mut session = ChatSession::new();
        assert_eq!(session.status, ChatStatus::Idle);
        assert!(!session.is_loading);

        session.set_status(ChatStatus::Thinking);
        assert_eq!(session.status, ChatStatus::Thinking);
        assert!(session.is_loading);

        session.set_status(ChatStatus::Streaming);
        assert!(session.is_loading);

        session.set_status(ChatStatus::ToolCalling);
        assert!(session.is_loading);

        session.set_status(ChatStatus::Idle);
        assert_eq!(session.status, ChatStatus::Idle);
        assert!(!session.is_loading);

        session.set_status(ChatStatus::Error);
        assert!(!session.is_loading);
    }
}
