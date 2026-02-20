//! React-style hooks for WASM UI frameworks.
//!
//! Provides `useChat`, `useCompletion`, and `useAgent` hook patterns for building
//! interactive AI-powered UIs in WebAssembly. These hooks follow the React hooks
//! convention of encapsulating stateful logic with a clean API surface.
//!
//! # Three-Variant CFG Pattern
//!
//! This module uses the same three-variant cfg pattern as [`crate::wasm`]:
//!
//! 1. **wasm32 + `wasm` feature**: Full `wasm_bindgen` exports with JS interop callbacks
//! 2. **wasm32 without feature**: Lightweight stubs (no-ops)
//! 3. **Native (non-wasm32)**: Full implementation for testing, no wasm_bindgen
//!
//! # Hooks
//!
//! - [`UseChatHook`] — Streaming chat with an LLM (multi-turn conversation)
//! - [`UseCompletionHook`] — Single prompt-to-response completion
//! - [`UseAgentHook`] — Agentic loop with tool calling and human-in-the-loop approval
//!
//! # Example (native)
//!
//! ```rust
//! use ai_assistant::wasm_hooks::*;
//!
//! let mut chat = UseChatHook::new(ChatConfig::new().with_model("llama3"));
//! chat.send_message("Hello!");
//! assert_eq!(chat.message_count(), 2); // user + assistant stub
//! ```

// ---------------------------------------------------------------------------
// Shared types (all platforms)
// ---------------------------------------------------------------------------

/// Chat message with role and content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of the message sender (e.g. "user", "assistant", "system").
    pub role: String,
    /// The text content of the message.
    pub content: String,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
}

/// Configuration for chat hooks.
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// API endpoint URL.
    pub api_url: String,
    /// Model identifier.
    pub model: String,
    /// System prompt prepended to conversations.
    pub system_prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: f64,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:11434/api/generate".to_string(),
            model: "llama3".to_string(),
            system_prompt: String::new(),
            max_tokens: 2048,
            temperature: 0.7,
        }
    }
}

impl ChatConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the API URL.
    pub fn with_api_url(mut self, url: &str) -> Self {
        self.api_url = url.to_string();
        self
    }

    /// Set the model identifier.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    /// Set the maximum token count.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }
}

/// Agent state machine states.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    /// Agent is idle, no task running.
    Idle,
    /// Agent is reasoning about the next step.
    Thinking,
    /// Agent is executing a tool call.
    ExecutingTool,
    /// Agent is waiting for human approval of a tool call.
    WaitingApproval,
    /// Agent has completed its task.
    Done,
    /// Agent encountered an error.
    Error,
}

/// Information about a tool call made by the agent.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    /// Tool name.
    pub name: String,
    /// JSON-encoded arguments.
    pub arguments: String,
    /// Result of the tool execution, if completed.
    pub result: Option<String>,
    /// Whether the tool call was approved (`Some(true)`), denied (`Some(false)`),
    /// or pending (`None`).
    pub approved: Option<bool>,
}

/// Configuration for agent hooks.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Underlying chat configuration.
    pub chat_config: ChatConfig,
    /// Whether tool calls are auto-approved without human review.
    pub auto_approve: bool,
    /// Maximum number of agent iterations before stopping.
    pub max_iterations: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            chat_config: ChatConfig::default(),
            auto_approve: false,
            max_iterations: 10,
        }
    }
}

impl AgentConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the underlying chat configuration.
    pub fn with_chat_config(mut self, config: ChatConfig) -> Self {
        self.chat_config = config;
        self
    }

    /// Set whether tool calls are auto-approved.
    pub fn with_auto_approve(mut self, auto: bool) -> Self {
        self.auto_approve = auto;
        self
    }

    /// Set the maximum number of agent iterations.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }
}

// ===========================================================================
// Variant 1: wasm32 + "wasm" feature — real wasm_bindgen exports
// ===========================================================================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm_impl {
    use super::*;
    use js_sys;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct UseChatHook {
        config: ChatConfig,
        messages: Vec<ChatMessage>,
        is_loading: bool,
        error: Option<String>,
        on_chunk: Option<js_sys::Function>,
        on_complete: Option<js_sys::Function>,
        on_error: Option<js_sys::Function>,
    }

    #[wasm_bindgen]
    impl UseChatHook {
        #[wasm_bindgen(constructor)]
        pub fn new(api_url: &str, model: &str) -> Self {
            let config = ChatConfig::new().with_api_url(api_url).with_model(model);
            Self {
                config,
                messages: Vec::new(),
                is_loading: false,
                error: None,
                on_chunk: None,
                on_complete: None,
                on_error: None,
            }
        }

        #[wasm_bindgen(js_name = "sendMessage")]
        pub fn send_message(&mut self, content: &str) {
            self.messages.push(ChatMessage {
                role: "user".to_string(),
                content: content.to_string(),
                timestamp: js_sys::Date::now() as u64 / 1000,
            });
            self.is_loading = true;
            // In real implementation, would use web_sys::Request + fetch API
            // and invoke on_chunk / on_complete / on_error callbacks.
        }

        #[wasm_bindgen(js_name = "getMessages")]
        pub fn get_messages_json(&self) -> String {
            let items: Vec<String> = self
                .messages
                .iter()
                .map(|m| {
                    format!(
                        "{{\"role\":\"{}\",\"content\":\"{}\",\"timestamp\":{}}}",
                        m.role,
                        m.content.replace('\"', "\\\""),
                        m.timestamp
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }

        #[wasm_bindgen(js_name = "messageCount")]
        pub fn message_count(&self) -> usize {
            self.messages.len()
        }

        #[wasm_bindgen(js_name = "isLoading")]
        pub fn is_loading(&self) -> bool {
            self.is_loading
        }

        #[wasm_bindgen(js_name = "clearMessages")]
        pub fn clear_messages(&mut self) {
            self.messages.clear();
        }

        #[wasm_bindgen(js_name = "setOnChunk")]
        pub fn set_on_chunk(&mut self, callback: js_sys::Function) {
            self.on_chunk = Some(callback);
        }

        #[wasm_bindgen(js_name = "setOnComplete")]
        pub fn set_on_complete(&mut self, callback: js_sys::Function) {
            self.on_complete = Some(callback);
        }

        #[wasm_bindgen(js_name = "setOnError")]
        pub fn set_on_error(&mut self, callback: js_sys::Function) {
            self.on_error = Some(callback);
        }
    }

    #[wasm_bindgen]
    pub struct UseCompletionHook {
        config: ChatConfig,
        result: Option<String>,
        is_loading: bool,
        error: Option<String>,
    }

    #[wasm_bindgen]
    impl UseCompletionHook {
        #[wasm_bindgen(constructor)]
        pub fn new(api_url: &str, model: &str) -> Self {
            let config = ChatConfig::new().with_api_url(api_url).with_model(model);
            Self {
                config,
                result: None,
                is_loading: false,
                error: None,
            }
        }

        pub fn complete(&mut self, prompt: &str) {
            self.is_loading = true;
            self.error = None;
            // In real implementation, would use web_sys fetch API
            let _ = prompt;
        }

        #[wasm_bindgen(js_name = "getResult")]
        pub fn get_result(&self) -> Option<String> {
            self.result.clone()
        }

        #[wasm_bindgen(js_name = "isLoading")]
        pub fn is_loading(&self) -> bool {
            self.is_loading
        }
    }

    #[wasm_bindgen]
    pub struct UseAgentHook {
        config: AgentConfig,
        state: String,
        messages: Vec<ChatMessage>,
        tool_calls: Vec<ToolCallInfo>,
        iterations: u32,
        error: Option<String>,
    }

    #[wasm_bindgen]
    impl UseAgentHook {
        #[wasm_bindgen(constructor)]
        pub fn new(api_url: &str, model: &str, auto_approve: bool) -> Self {
            let chat_config = ChatConfig::new().with_api_url(api_url).with_model(model);
            let config = AgentConfig::new()
                .with_chat_config(chat_config)
                .with_auto_approve(auto_approve);
            Self {
                config,
                state: "idle".to_string(),
                messages: Vec::new(),
                tool_calls: Vec::new(),
                iterations: 0,
                error: None,
            }
        }

        pub fn start(&mut self, prompt: &str) {
            self.state = "thinking".to_string();
            self.iterations = 0;
            self.messages.push(ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                timestamp: js_sys::Date::now() as u64 / 1000,
            });
        }

        #[wasm_bindgen(js_name = "approveToolCall")]
        pub fn approve_tool_call(&mut self, index: usize) {
            if let Some(tc) = self.tool_calls.get_mut(index) {
                tc.approved = Some(true);
            }
        }

        #[wasm_bindgen(js_name = "denyToolCall")]
        pub fn deny_tool_call(&mut self, index: usize) {
            if let Some(tc) = self.tool_calls.get_mut(index) {
                tc.approved = Some(false);
            }
            self.state = "error".to_string();
        }

        #[wasm_bindgen(js_name = "getState")]
        pub fn get_state(&self) -> String {
            self.state.clone()
        }

        #[wasm_bindgen(js_name = "getMessages")]
        pub fn get_messages_json(&self) -> String {
            let items: Vec<String> = self
                .messages
                .iter()
                .map(|m| {
                    format!(
                        "{{\"role\":\"{}\",\"content\":\"{}\",\"timestamp\":{}}}",
                        m.role,
                        m.content.replace('\"', "\\\""),
                        m.timestamp
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }

        pub fn iterations(&self) -> u32 {
            self.iterations
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm_impl::*;

// ===========================================================================
// Variant 2: wasm32 WITHOUT "wasm" feature — lightweight stubs
// ===========================================================================

#[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
mod wasm_stub {
    use super::*;

    /// Stub chat hook (wasm32 without `wasm` feature).
    pub struct UseChatHook {
        config: ChatConfig,
    }

    impl UseChatHook {
        pub fn new(config: ChatConfig) -> Self {
            Self { config }
        }

        pub fn send_message(&mut self, _content: &str) {}

        pub fn get_messages(&self) -> &[ChatMessage] {
            &[]
        }

        pub fn message_count(&self) -> usize {
            0
        }

        pub fn is_loading(&self) -> bool {
            false
        }

        pub fn error(&self) -> Option<&str> {
            None
        }

        pub fn clear_messages(&mut self) {}

        pub fn config(&self) -> &ChatConfig {
            &self.config
        }
    }

    /// Stub completion hook (wasm32 without `wasm` feature).
    pub struct UseCompletionHook {
        config: ChatConfig,
    }

    impl UseCompletionHook {
        pub fn new(config: ChatConfig) -> Self {
            Self { config }
        }

        pub fn complete(&mut self, _prompt: &str) {}

        pub fn get_result(&self) -> Option<&str> {
            None
        }

        pub fn is_loading(&self) -> bool {
            false
        }

        pub fn error(&self) -> Option<&str> {
            None
        }

        pub fn config(&self) -> &ChatConfig {
            &self.config
        }
    }

    /// Stub agent hook (wasm32 without `wasm` feature).
    pub struct UseAgentHook {
        config: AgentConfig,
    }

    impl UseAgentHook {
        pub fn new(config: AgentConfig) -> Self {
            Self { config }
        }

        pub fn start(&mut self, _prompt: &str) {}

        pub fn approve_tool_call(&mut self, _index: usize) {}

        pub fn deny_tool_call(&mut self, _index: usize) {}

        pub fn get_state(&self) -> &AgentState {
            &AgentState::Idle
        }

        pub fn get_messages(&self) -> &[ChatMessage] {
            &[]
        }

        pub fn get_tool_calls(&self) -> &[ToolCallInfo] {
            &[]
        }

        pub fn iterations(&self) -> u32 {
            0
        }

        pub fn error(&self) -> Option<&str> {
            None
        }

        pub fn config(&self) -> &AgentConfig {
            &self.config
        }
    }
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
pub use wasm_stub::*;

// ===========================================================================
// Variant 3: Native (non-wasm32) — full implementation for testing
// ===========================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;

    /// React-style `useChat` hook for streaming multi-turn chat with an LLM.
    ///
    /// Maintains a conversation history and provides methods to send messages,
    /// check loading state, and retrieve the message list.
    pub struct UseChatHook {
        config: ChatConfig,
        messages: Vec<ChatMessage>,
        is_loading: bool,
        error: Option<String>,
    }

    impl UseChatHook {
        /// Create a new chat hook with the given configuration.
        pub fn new(config: ChatConfig) -> Self {
            Self {
                config,
                messages: Vec::new(),
                is_loading: false,
                error: None,
            }
        }

        /// Send a user message and receive a stub assistant response.
        ///
        /// In the WASM variant, this would initiate a streaming fetch() call.
        /// The native variant simulates an immediate response for testing.
        pub fn send_message(&mut self, content: &str) {
            self.messages.push(ChatMessage {
                role: "user".to_string(),
                content: content.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            });
            self.is_loading = true;

            // Native stub: simulate an immediate response
            self.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: format!("[Native stub] Response to: {}", content),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            });
            self.is_loading = false;
        }

        /// Get a slice of all messages in the conversation.
        pub fn get_messages(&self) -> &[ChatMessage] {
            &self.messages
        }

        /// Get the number of messages in the conversation.
        pub fn message_count(&self) -> usize {
            self.messages.len()
        }

        /// Whether a request is currently in flight.
        pub fn is_loading(&self) -> bool {
            self.is_loading
        }

        /// Get the current error, if any.
        pub fn error(&self) -> Option<&str> {
            self.error.as_deref()
        }

        /// Clear all messages from the conversation.
        pub fn clear_messages(&mut self) {
            self.messages.clear();
        }

        /// Get a reference to the chat configuration.
        pub fn config(&self) -> &ChatConfig {
            &self.config
        }
    }

    /// React-style `useCompletion` hook for single prompt-to-response completion.
    ///
    /// Unlike [`UseChatHook`], this does not maintain a conversation history.
    /// Each call to [`complete`](UseCompletionHook::complete) replaces the previous result.
    pub struct UseCompletionHook {
        config: ChatConfig,
        result: Option<String>,
        is_loading: bool,
        error: Option<String>,
    }

    impl UseCompletionHook {
        /// Create a new completion hook with the given configuration.
        pub fn new(config: ChatConfig) -> Self {
            Self {
                config,
                result: None,
                is_loading: false,
                error: None,
            }
        }

        /// Run a completion for the given prompt.
        ///
        /// In the WASM variant, this would make a fetch() call.
        /// The native variant simulates an immediate result for testing.
        pub fn complete(&mut self, prompt: &str) {
            self.is_loading = true;
            self.error = None;
            // Native stub: simulate completion
            self.result = Some(format!("[Native stub] Completion for: {}", prompt));
            self.is_loading = false;
        }

        /// Get the completion result, if available.
        pub fn get_result(&self) -> Option<&str> {
            self.result.as_deref()
        }

        /// Whether a request is currently in flight.
        pub fn is_loading(&self) -> bool {
            self.is_loading
        }

        /// Get the current error, if any.
        pub fn error(&self) -> Option<&str> {
            self.error.as_deref()
        }

        /// Get a reference to the chat configuration.
        pub fn config(&self) -> &ChatConfig {
            &self.config
        }
    }

    /// React-style `useAgent` hook with tool calling and human-in-the-loop approval.
    ///
    /// Models an agentic loop where the LLM can request tool calls, and the user
    /// can approve or deny them before execution. Tracks state transitions through
    /// [`AgentState`].
    pub struct UseAgentHook {
        config: AgentConfig,
        state: AgentState,
        messages: Vec<ChatMessage>,
        tool_calls: Vec<ToolCallInfo>,
        iterations: u32,
        error: Option<String>,
    }

    impl UseAgentHook {
        /// Create a new agent hook with the given configuration.
        pub fn new(config: AgentConfig) -> Self {
            Self {
                config,
                state: AgentState::Idle,
                messages: Vec::new(),
                tool_calls: Vec::new(),
                iterations: 0,
                error: None,
            }
        }

        /// Start the agent with a user prompt.
        ///
        /// The agent transitions through Thinking -> ExecutingTool, then either
        /// completes (if `auto_approve` is true) or waits for approval.
        pub fn start(&mut self, prompt: &str) {
            self.state = AgentState::Thinking;
            self.iterations = 0;

            self.messages.push(ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            });

            // Simulate a tool call
            self.state = AgentState::ExecutingTool;
            self.iterations += 1;

            self.tool_calls.push(ToolCallInfo {
                name: "stub_tool".to_string(),
                arguments: format!("{{\"query\": \"{}\"}}", prompt),
                result: None,
                approved: if self.config.auto_approve {
                    Some(true)
                } else {
                    None
                },
            });

            if self.config.auto_approve {
                // Auto-approve: execute and proceed to done
                if let Some(tc) = self.tool_calls.last_mut() {
                    tc.result = Some("[Native stub] Tool result".to_string());
                }
                self.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: format!("[Native stub] Agent response for: {}", prompt),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                });
                self.state = AgentState::Done;
            } else {
                self.state = AgentState::WaitingApproval;
            }
        }

        /// Approve a pending tool call by index.
        pub fn approve_tool_call(&mut self, index: usize) {
            if let Some(tc) = self.tool_calls.get_mut(index) {
                tc.approved = Some(true);
                tc.result = Some("[Native stub] Approved tool result".to_string());
            }
            self.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: "[Native stub] Agent completed after approval".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            });
            self.state = AgentState::Done;
        }

        /// Deny a pending tool call by index, transitioning to Error state.
        pub fn deny_tool_call(&mut self, index: usize) {
            if let Some(tc) = self.tool_calls.get_mut(index) {
                tc.approved = Some(false);
            }
            self.error = Some("Tool call denied by user".to_string());
            self.state = AgentState::Error;
        }

        /// Get the current agent state.
        pub fn get_state(&self) -> &AgentState {
            &self.state
        }

        /// Get a slice of all messages in the agent conversation.
        pub fn get_messages(&self) -> &[ChatMessage] {
            &self.messages
        }

        /// Get a slice of all tool calls made by the agent.
        pub fn get_tool_calls(&self) -> &[ToolCallInfo] {
            &self.tool_calls
        }

        /// Get the number of agent iterations completed.
        pub fn iterations(&self) -> u32 {
            self.iterations
        }

        /// Get the current error, if any.
        pub fn error(&self) -> Option<&str> {
            self.error.as_deref()
        }

        /// Get a reference to the agent configuration.
        pub fn config(&self) -> &AgentConfig {
            &self.config
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

// ===========================================================================
// Tests (native only)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_config_builder() {
        let config = ChatConfig::new()
            .with_api_url("http://example.com")
            .with_model("gpt-4")
            .with_temperature(0.5)
            .with_max_tokens(1024);
        assert_eq!(config.api_url, "http://example.com");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_chat_config_defaults() {
        let config = ChatConfig::default();
        assert_eq!(config.api_url, "http://localhost:11434/api/generate");
        assert_eq!(config.model, "llama3");
        assert!(config.system_prompt.is_empty());
        assert_eq!(config.max_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_config_with_system_prompt() {
        let config = ChatConfig::new().with_system_prompt("You are a helpful assistant.");
        assert_eq!(config.system_prompt, "You are a helpful assistant.");
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfig::new()
            .with_auto_approve(true)
            .with_max_iterations(5);
        assert!(config.auto_approve);
        assert_eq!(config.max_iterations, 5);
    }

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig::default();
        assert!(!config.auto_approve);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.chat_config.model, "llama3");
    }

    #[test]
    fn test_agent_config_with_chat_config() {
        let chat = ChatConfig::new().with_model("gpt-4");
        let config = AgentConfig::new().with_chat_config(chat);
        assert_eq!(config.chat_config.model, "gpt-4");
    }

    #[test]
    fn test_use_chat_hook_send_message() {
        let config = ChatConfig::new();
        let mut hook = UseChatHook::new(config);
        assert_eq!(hook.message_count(), 0);
        hook.send_message("Hello");
        assert_eq!(hook.message_count(), 2); // user + stub response
        assert_eq!(hook.get_messages()[0].role, "user");
        assert_eq!(hook.get_messages()[0].content, "Hello");
        assert_eq!(hook.get_messages()[1].role, "assistant");
        assert!(hook.get_messages()[1].content.contains("Hello"));
    }

    #[test]
    fn test_use_chat_hook_clear() {
        let mut hook = UseChatHook::new(ChatConfig::new());
        hook.send_message("test");
        assert!(hook.message_count() > 0);
        hook.clear_messages();
        assert_eq!(hook.message_count(), 0);
    }

    #[test]
    fn test_use_chat_hook_loading_state() {
        let hook = UseChatHook::new(ChatConfig::new());
        // Native stub completes synchronously, so loading is always false when observed
        assert!(!hook.is_loading());
    }

    #[test]
    fn test_use_chat_hook_error() {
        let hook = UseChatHook::new(ChatConfig::new());
        assert!(hook.error().is_none());
    }

    #[test]
    fn test_use_chat_hook_config() {
        let config = ChatConfig::new().with_model("test-model");
        let hook = UseChatHook::new(config);
        assert_eq!(hook.config().model, "test-model");
    }

    #[test]
    fn test_use_completion_hook() {
        let mut hook = UseCompletionHook::new(ChatConfig::new());
        assert!(hook.get_result().is_none());
        hook.complete("Tell me a joke");
        assert!(hook.get_result().is_some());
        assert!(hook.get_result().unwrap().contains("Tell me a joke"));
    }

    #[test]
    fn test_use_completion_hook_replaces_result() {
        let mut hook = UseCompletionHook::new(ChatConfig::new());
        hook.complete("First prompt");
        assert!(hook.get_result().unwrap().contains("First prompt"));
        hook.complete("Second prompt");
        assert!(hook.get_result().unwrap().contains("Second prompt"));
        assert!(!hook.get_result().unwrap().contains("First prompt"));
    }

    #[test]
    fn test_use_completion_hook_loading_and_error() {
        let hook = UseCompletionHook::new(ChatConfig::new());
        assert!(!hook.is_loading());
        assert!(hook.error().is_none());
    }

    #[test]
    fn test_use_completion_hook_config() {
        let config = ChatConfig::new().with_api_url("http://custom.api");
        let hook = UseCompletionHook::new(config);
        assert_eq!(hook.config().api_url, "http://custom.api");
    }

    #[test]
    fn test_use_agent_hook_auto_approve() {
        let config = AgentConfig::new().with_auto_approve(true);
        let mut hook = UseAgentHook::new(config);
        assert_eq!(*hook.get_state(), AgentState::Idle);
        hook.start("Search for info");
        assert_eq!(*hook.get_state(), AgentState::Done);
        assert_eq!(hook.iterations(), 1);
        assert!(!hook.get_tool_calls().is_empty());
        assert_eq!(hook.get_tool_calls()[0].approved, Some(true));
        assert!(hook.get_tool_calls()[0].result.is_some());
    }

    #[test]
    fn test_use_agent_hook_manual_approve() {
        let config = AgentConfig::new().with_auto_approve(false);
        let mut hook = UseAgentHook::new(config);
        hook.start("Do something");
        assert_eq!(*hook.get_state(), AgentState::WaitingApproval);
        assert_eq!(hook.get_tool_calls()[0].approved, None);
        hook.approve_tool_call(0);
        assert_eq!(*hook.get_state(), AgentState::Done);
        assert_eq!(hook.get_tool_calls()[0].approved, Some(true));
    }

    #[test]
    fn test_use_agent_hook_deny() {
        let config = AgentConfig::new().with_auto_approve(false);
        let mut hook = UseAgentHook::new(config);
        hook.start("Do something");
        hook.deny_tool_call(0);
        assert_eq!(*hook.get_state(), AgentState::Error);
        assert!(hook.error().is_some());
        assert_eq!(hook.error().unwrap(), "Tool call denied by user");
        assert_eq!(hook.get_tool_calls()[0].approved, Some(false));
    }

    #[test]
    fn test_agent_state_variants() {
        assert_eq!(AgentState::Idle, AgentState::Idle);
        assert_ne!(AgentState::Idle, AgentState::Thinking);
        assert_ne!(AgentState::ExecutingTool, AgentState::WaitingApproval);
        assert_ne!(AgentState::Done, AgentState::Error);
    }

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            timestamp: 1234567890,
        };
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
        assert_eq!(msg.timestamp, 1234567890);
    }

    #[test]
    fn test_chat_message_clone() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: "World".to_string(),
            timestamp: 999,
        };
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
        assert_eq!(cloned.content, msg.content);
        assert_eq!(cloned.timestamp, msg.timestamp);
    }

    #[test]
    fn test_tool_call_info_creation() {
        let tc = ToolCallInfo {
            name: "search".to_string(),
            arguments: "{\"q\": \"test\"}".to_string(),
            result: Some("found it".to_string()),
            approved: Some(true),
        };
        assert_eq!(tc.name, "search");
        assert_eq!(tc.approved, Some(true));
        assert!(tc.result.is_some());
    }

    #[test]
    fn test_agent_hook_messages_include_user_prompt() {
        let config = AgentConfig::new().with_auto_approve(true);
        let mut hook = UseAgentHook::new(config);
        hook.start("My prompt");
        assert!(hook
            .get_messages()
            .iter()
            .any(|m| m.role == "user" && m.content == "My prompt"));
    }

    #[test]
    fn test_agent_hook_config_accessor() {
        let config = AgentConfig::new()
            .with_auto_approve(true)
            .with_max_iterations(3);
        let hook = UseAgentHook::new(config);
        assert!(hook.config().auto_approve);
        assert_eq!(hook.config().max_iterations, 3);
    }

    #[test]
    fn test_agent_hook_tool_call_arguments() {
        let config = AgentConfig::new().with_auto_approve(false);
        let mut hook = UseAgentHook::new(config);
        hook.start("test query");
        let tc = &hook.get_tool_calls()[0];
        assert_eq!(tc.name, "stub_tool");
        assert!(tc.arguments.contains("test query"));
    }
}
