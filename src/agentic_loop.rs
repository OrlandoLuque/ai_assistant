//! Agentic loop system
//!
//! Implements an autonomous agent loop that can use tools, search the web,
//! and complete complex tasks iteratively.

use crate::tool_calling::{ParameterType, Tool, ToolCall, ToolParameter, ToolRegistry, ToolResult};
use crate::web_search::{SearchConfig, WebSearchEngine};
use std::sync::Arc;

/// Agent configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoopConfig {
    /// Maximum number of iterations before stopping
    pub max_iterations: usize,
    /// Whether to automatically use web search for questions needing current info
    pub auto_search: bool,
    /// Keywords that trigger automatic web search
    pub search_triggers: Vec<String>,
    /// System prompt for the agent
    pub system_prompt: String,
    /// Whether to include tool results in conversation history
    pub include_tool_history: bool,
    /// Maximum tokens for context
    pub max_context_tokens: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            auto_search: true,
            search_triggers: vec![
                "current".to_string(),
                "latest".to_string(),
                "today".to_string(),
                "now".to_string(),
                "recent".to_string(),
                "2024".to_string(),
                "2025".to_string(),
                "news".to_string(),
                "price".to_string(),
                "weather".to_string(),
            ],
            system_prompt: String::new(),
            include_tool_history: true,
            max_context_tokens: 4096,
        }
    }
}

/// Agent state during execution
#[derive(Debug, Clone)]
pub struct LoopState {
    pub iteration: usize,
    pub status: LoopStatus,
    pub thought: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub final_answer: Option<String>,
}

/// Agent execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoopStatus {
    Thinking,
    CallingTool,
    WaitingForToolResult,
    Finished,
    Error,
    MaxIterationsReached,
    /// The agent stopped because the in-crate stall heuristic (see
    /// [`crate::stall_detection`]) detected repeated tool calls or user
    /// frustration. Produced only when the `stall-detection` feature is
    /// enabled; the variant is always present so existing matches stay
    /// exhaustive regardless of feature selection.
    UserStalled,
}

/// Message in the agent conversation
#[derive(Debug, Clone)]
pub struct LoopMessage {
    pub role: LoopRole,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_results: Option<Vec<ToolResult>>,
    /// Optional image attachments. Empty by default; populated by
    /// `AgenticLoop::process_with_images` and propagated to vision-capable
    /// providers via [`crate::vision::agent_bridge::loop_messages_to_vision`].
    #[cfg(feature = "vision")]
    pub images: Vec<crate::vision::ImageInput>,
}

/// Role in agent conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoopRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Agentic loop that can reason, use tools, and search
pub struct AgenticLoop {
    config: LoopConfig,
    tools: ToolRegistry,
    search_manager: Option<WebSearchEngine>,
    conversation: Vec<LoopMessage>,
    state: LoopState,
    response_generator: Option<Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>>,
    /// Vision configuration. When set, [`process_with_images`] uses it to
    /// validate the loaded model is vision-capable before dispatching.
    #[cfg(feature = "vision")]
    vision_config: Option<crate::vision::VisionLimits>,
    #[cfg(feature = "stall-detection")]
    stall_heuristic: Option<Box<dyn crate::stall_detection::StallHeuristic>>,
}

impl AgenticLoop {
    pub fn new(config: LoopConfig) -> Self {
        let mut tools = ToolRegistry::new();

        // Register built-in tools
        tools.register(crate::tool_calling::CommonTools::calculator());
        tools.register(crate::tool_calling::CommonTools::datetime());
        tools.register(crate::tool_calling::CommonTools::text_length());

        Self {
            config,
            tools,
            search_manager: None,
            conversation: Vec::new(),
            state: LoopState {
                iteration: 0,
                status: LoopStatus::Thinking,
                thought: None,
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                final_answer: None,
            },
            response_generator: None,
            #[cfg(feature = "vision")]
            vision_config: None,
            #[cfg(feature = "stall-detection")]
            stall_heuristic: None,
        }
    }

    /// Attach an in-crate [`crate::stall_detection::StallHeuristic`] so the
    /// loop automatically observes user messages and tool calls, and halts
    /// with [`LoopStatus::UserStalled`] when the heuristic fires.
    ///
    /// The heuristic is consumed by value (boxed) so any trait object works —
    /// including [`crate::stall_detection::KeywordStallDetector`] and the
    /// feature-gated `LlmAssistedStallDetector`.
    #[cfg(feature = "stall-detection")]
    pub fn with_stall_heuristic(
        mut self,
        heuristic: Box<dyn crate::stall_detection::StallHeuristic>,
    ) -> Self {
        self.stall_heuristic = Some(heuristic);
        self
    }

    /// Access the attached stall heuristic, if any. Useful for inspecting
    /// state after the loop returns (e.g. reading `last_emotion`).
    #[cfg(feature = "stall-detection")]
    pub fn stall_heuristic(&self) -> Option<&dyn crate::stall_detection::StallHeuristic> {
        self.stall_heuristic.as_deref()
    }

    /// Mutable access to the attached stall heuristic, if any. Needed to call
    /// [`crate::stall_detection::StallHeuristic::check`] from outside the loop
    /// (the trait method takes `&mut self` because some heuristics advance
    /// internal cooldowns).
    #[cfg(feature = "stall-detection")]
    pub fn stall_heuristic_mut(
        &mut self,
    ) -> Option<&mut (dyn crate::stall_detection::StallHeuristic + 'static)> {
        self.stall_heuristic.as_deref_mut()
    }

    /// Enable web search capability
    pub fn with_web_search(mut self, search_config: SearchConfig) -> Self {
        self.search_manager = Some(WebSearchEngine::new(search_config));

        // Register web search tool
        let search_tool =
            Tool::new("web_search", "Search the web for current information").with_parameter(
                ToolParameter::new("query", "The search query", ParameterType::String),
            );

        self.tools.register(search_tool);
        self
    }

    /// Register a custom tool
    pub fn register_tool(&mut self, tool: Tool) {
        self.tools.register(tool);
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.config.system_prompt = prompt.to_string();
        self
    }

    /// Configure vision limits used by [`process_with_images`].
    /// Calling this is what opts the loop into vision dispatch — without
    /// it, [`process_with_images`] uses default limits.
    #[cfg(feature = "vision")]
    pub fn with_vision_config(mut self, limits: crate::vision::VisionLimits) -> Self {
        self.vision_config = Some(limits);
        self
    }

    /// Process a user message with image attachments. Validates each image
    /// against the configured (or default) [`crate::vision::VisionLimits`],
    /// rejects oversize/animated/decompression-bomb payloads, and pushes a
    /// `LoopMessage` carrying the validated images. The downstream
    /// `response_generator` receives the conversation including images via
    /// `LoopMessage.images`; vision-aware generators should use
    /// [`crate::vision::agent_bridge::loop_messages_to_vision`] to translate
    /// before dispatching.
    #[cfg(feature = "vision")]
    pub fn process_with_images(
        &mut self,
        user_message: &str,
        images: Vec<crate::vision::ImageInput>,
    ) -> Result<AgentLoopResult, String> {
        let limits = self.vision_config.unwrap_or_default();
        let preprocessor = crate::vision::ImagePreprocessor::high_detail().with_limits(limits);

        // Validate each base64-encoded image up-front. URL-only references
        // are fetched (and validated) by the provider layer. Reject the whole
        // batch on first failure so the loop never sees a partial set.
        for img in &images {
            if let crate::vision::ImageData::Base64(b64) = &img.data {
                let bytes = crate::vision::base64_decode(b64)
                    .map_err(|e| format!("vision image base64 decode failed: {e}"))?;
                preprocessor
                    .validate_bytes(&bytes)
                    .map_err(|e| format!("vision image validation failed: {e}"))?;
            }
        }

        // Reset state for new query
        self.state = LoopState {
            iteration: 0,
            status: LoopStatus::Thinking,
            thought: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            final_answer: None,
        };

        if !self.config.system_prompt.is_empty() && self.conversation.is_empty() {
            self.conversation.push(LoopMessage {
                role: LoopRole::System,
                content: self.config.system_prompt.clone(),
                tool_calls: None,
                tool_results: None,
                images: Vec::new(),
            });
        }

        self.conversation.push(LoopMessage {
            role: LoopRole::User,
            content: user_message.to_string(),
            tool_calls: None,
            tool_results: None,
            images,
        });

        // Run the standard iteration loop. Vision-aware response_generator
        // is responsible for inspecting `LoopMessage.images`; non-vision
        // generators see images as empty (transparent fallback).
        let mut iterations = Vec::new();
        while self.state.iteration < self.config.max_iterations {
            self.state.iteration += 1;
            let it = self.run_iteration(false);
            iterations.push(it);
            if matches!(
                self.state.status,
                LoopStatus::Finished | LoopStatus::Error | LoopStatus::UserStalled
            ) {
                break;
            }
        }
        if self.state.iteration >= self.config.max_iterations
            && self.state.status != LoopStatus::Finished
        {
            self.state.status = LoopStatus::MaxIterationsReached;
        }
        Ok(AgentLoopResult {
            final_answer: self.state.final_answer.clone(),
            iterations,
            total_iterations: self.state.iteration,
            status: self.state.status,
            tool_calls_made: self.state.tool_calls.clone(),
            tool_results: self.state.tool_results.clone(),
        })
    }

    /// Process a user message and run the agent loop
    pub fn process(&mut self, user_message: &str) -> AgentLoopResult {
        // Reset state for new query
        self.state = LoopState {
            iteration: 0,
            status: LoopStatus::Thinking,
            thought: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            final_answer: None,
        };

        // Add system prompt if configured
        if !self.config.system_prompt.is_empty() && self.conversation.is_empty() {
            self.conversation.push(LoopMessage {
                role: LoopRole::System,
                content: self.config.system_prompt.clone(),
                tool_calls: None,
                tool_results: None,
                #[cfg(feature = "vision")]
                images: Vec::new(),
            });
        }

        // Add user message
        self.conversation.push(LoopMessage {
            role: LoopRole::User,
            content: user_message.to_string(),
            tool_calls: None,
            tool_results: None,
            #[cfg(feature = "vision")]
            images: Vec::new(),
        });

        // Feed the user message to the stall heuristic, if one is attached.
        #[cfg(feature = "stall-detection")]
        if let Some(h) = self.stall_heuristic.as_mut() {
            h.observe_user_message(user_message);
        }

        // Check if we should auto-search
        let needs_search = self.config.auto_search && self.needs_web_search(user_message);

        // Run the agent loop
        let mut iterations = Vec::new();

        while self.state.iteration < self.config.max_iterations {
            self.state.iteration += 1;

            #[cfg(feature = "stall-detection")]
            let tool_calls_before = self.state.tool_calls.len();

            let iteration_result = self.run_iteration(needs_search && self.state.iteration == 1);
            iterations.push(iteration_result.clone());

            // After the iteration, let the stall heuristic observe any tool
            // calls that fired, then evaluate. If stalled, stop the loop.
            #[cfg(feature = "stall-detection")]
            {
                let new_calls = self.state.tool_calls[tool_calls_before..].to_vec();
                if let Some(h) = self.stall_heuristic.as_mut() {
                    for call in &new_calls {
                        let args_bytes = serde_json::to_vec(&call.arguments).unwrap_or_default();
                        let hash = crate::stall_detection::hash_tool_call(&call.name, &args_bytes);
                        h.observe_tool_call(&call.name, hash);
                    }
                    if let crate::stall_detection::StallDecision::Stalled(_) = h.check() {
                        self.state.status = LoopStatus::UserStalled;
                    }
                }
            }

            if matches!(
                self.state.status,
                LoopStatus::Finished | LoopStatus::Error | LoopStatus::UserStalled
            ) {
                break;
            }
        }

        if self.state.iteration >= self.config.max_iterations
            && self.state.status != LoopStatus::Finished
        {
            self.state.status = LoopStatus::MaxIterationsReached;
        }

        AgentLoopResult {
            final_answer: self.state.final_answer.clone(),
            iterations,
            total_iterations: self.state.iteration,
            status: self.state.status,
            tool_calls_made: self.state.tool_calls.clone(),
            tool_results: self.state.tool_results.clone(),
        }
    }

    /// Check if the query needs web search
    fn needs_web_search(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.config
            .search_triggers
            .iter()
            .any(|trigger| query_lower.contains(trigger))
    }

    /// Run a single iteration of the agent loop
    fn run_iteration(&mut self, force_search: bool) -> IterationResult {
        let mut result = IterationResult {
            iteration: self.state.iteration,
            thought: None,
            action: None,
            observation: None,
        };

        // If forcing search, create search tool call
        if force_search {
            if let Some(user_msg) = self
                .conversation
                .iter()
                .rev()
                .find(|m| m.role == LoopRole::User)
            {
                let query = self.extract_search_query(&user_msg.content);

                let call = ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: "web_search".to_string(),
                    arguments: [("query".to_string(), serde_json::Value::String(query))]
                        .into_iter()
                        .collect(),
                };

                result.action = Some(format!(
                    "Searching web: {}",
                    call.arguments
                        .get("query")
                        .unwrap_or(&serde_json::Value::Null)
                ));
                let tool_result = self.execute_tool_call(&call);
                result.observation = Some(tool_result.output.clone());

                self.state.tool_calls.push(call);

                // Add tool result to conversation
                if self.config.include_tool_history {
                    let output_content = tool_result.output.clone();
                    self.conversation.push(LoopMessage {
                        role: LoopRole::Tool,
                        content: output_content,
                        tool_calls: None,
                        tool_results: Some(vec![tool_result.clone()]),
                        #[cfg(feature = "vision")]
                        images: tool_result.images.clone(),
                    });
                }

                self.state.tool_results.push(tool_result);

                return result;
            }
        }

        // Generate context for model
        let context = self.build_context();
        result.thought = Some(format!(
            "Processing with {} messages in context",
            context.len()
        ));

        // Generate response: use callback if configured, otherwise build from context
        self.state.status = LoopStatus::Finished;
        self.state.final_answer = Some(self.generate_response_with_context());

        result
    }

    /// Extract a search query from user message
    fn extract_search_query(&self, message: &str) -> String {
        // Remove common question words and clean up
        let mut query = message.to_string();
        for word in &[
            "what is",
            "what are",
            "how to",
            "where is",
            "when is",
            "who is",
            "tell me about",
        ] {
            query = query.to_lowercase().replace(word, "").trim().to_string();
        }

        // Limit length
        if query.len() > 100 {
            query.truncate(100);
        }

        query
    }

    /// Execute a tool call
    fn execute_tool_call(&mut self, call: &ToolCall) -> ToolResult {
        // Special handling for web search
        if call.name == "web_search" {
            if let Some(ref mut search_manager) = self.search_manager {
                let query = call
                    .arguments
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let max_chars = call
                    .arguments
                    .get("num_results")
                    .and_then(|v| v.as_u64())
                    .map(|n| (n as usize) * 500) // Approx 500 chars per result
                    .unwrap_or(2500);

                match search_manager.search_for_context(query, max_chars) {
                    Ok(formatted) => {
                        return ToolResult {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            success: true,
                            output: formatted,
                            error: None,
                            #[cfg(feature = "vision")]
                            images: Vec::new(),
                        };
                    }
                    Err(e) => {
                        return ToolResult {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(e.to_string()),
                            #[cfg(feature = "vision")]
                            images: Vec::new(),
                        };
                    }
                }
            }
        }

        // Execute other tools through registry
        self.tools.execute(call)
    }

    /// Build context for the model
    fn build_context(&self) -> Vec<&LoopMessage> {
        self.conversation.iter().collect()
    }

    /// Set a custom response generator callback.
    ///
    /// When set, this function is called with the conversation history to produce
    /// the agent's response. Without a callback, the agent synthesizes a response
    /// from tool results and context.
    pub fn set_response_generator<F>(&mut self, f: F)
    where
        F: Fn(&[LoopMessage]) -> String + Send + Sync + 'static,
    {
        self.response_generator = Some(Arc::new(f));
    }

    /// Generate response using available context
    fn generate_response_with_context(&self) -> String {
        // If a response generator callback is configured, delegate to it
        if let Some(ref generator) = self.response_generator {
            return generator(&self.conversation);
        }

        // Build response from tool results and conversation context
        let mut parts = Vec::new();

        // Include successful tool results
        for result in &self.state.tool_results {
            if result.success && !result.output.is_empty() {
                parts.push(format!("[{}] {}", result.name, result.output));
            }
        }

        if parts.is_empty() {
            "Response generated based on available context.".to_string()
        } else {
            format!("Based on tool results:\n{}", parts.join("\n---\n"))
        }
    }

    /// Get current conversation
    pub fn get_conversation(&self) -> &[LoopMessage] {
        &self.conversation
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation.clear();
        self.state = LoopState {
            iteration: 0,
            status: LoopStatus::Thinking,
            thought: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            final_answer: None,
        };
    }

    /// Get available tools
    pub fn get_tools(&self) -> Vec<&Tool> {
        self.tools.list()
    }

    /// Get tools as JSON schema for API calls
    pub fn get_tools_schema(&self) -> Vec<serde_json::Value> {
        self.tools.to_json_schema()
    }
}

impl Default for AgenticLoop {
    fn default() -> Self {
        Self::new(LoopConfig::default())
    }
}

/// Result of a single iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iteration: usize,
    pub thought: Option<String>,
    pub action: Option<String>,
    pub observation: Option<String>,
}

/// Result of the full agent loop
#[derive(Debug, Clone)]
pub struct AgentLoopResult {
    pub final_answer: Option<String>,
    pub iterations: Vec<IterationResult>,
    pub total_iterations: usize,
    pub status: LoopStatus,
    pub tool_calls_made: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

/// Builder for creating configured agents
pub struct AgentBuilder {
    config: LoopConfig,
    tools: Vec<Tool>,
    search_config: Option<SearchConfig>,
    response_generator: Option<Arc<dyn Fn(&[LoopMessage]) -> String + Send + Sync>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            config: LoopConfig::default(),
            tools: Vec::new(),
            search_config: None,
            response_generator: None,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    pub fn with_auto_search(mut self, enabled: bool) -> Self {
        self.config.auto_search = enabled;
        self
    }

    pub fn with_search_triggers(mut self, triggers: Vec<String>) -> Self {
        self.config.search_triggers = triggers;
        self
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.config.system_prompt = prompt.to_string();
        self
    }

    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_web_search(mut self, config: SearchConfig) -> Self {
        self.search_config = Some(config);
        self
    }

    pub fn with_response_generator<F>(mut self, f: F) -> Self
    where
        F: Fn(&[LoopMessage]) -> String + Send + Sync + 'static,
    {
        self.response_generator = Some(Arc::new(f));
        self
    }

    pub fn build(self) -> AgenticLoop {
        let mut agent = AgenticLoop::new(self.config);

        for tool in self.tools {
            agent.register_tool(tool);
        }

        if let Some(search_config) = self.search_config {
            agent = agent.with_web_search(search_config);
        }

        agent.response_generator = self.response_generator;
        agent
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = AgenticLoop::new(LoopConfig::default());
        assert!(agent.get_tools().len() >= 3); // calculator, datetime, text_length
    }

    #[test]
    fn test_needs_web_search() {
        let agent = AgenticLoop::new(LoopConfig::default());
        assert!(agent.needs_web_search("What is the current price of Bitcoin?"));
        assert!(agent.needs_web_search("Latest news about AI"));
        assert!(!agent.needs_web_search("What is 2 + 2?"));
    }

    #[test]
    fn test_agent_builder() {
        let agent = AgentBuilder::new()
            .with_max_iterations(5)
            .with_auto_search(true)
            .with_system_prompt("You are a helpful assistant.")
            .build();

        assert_eq!(agent.config.max_iterations, 5);
        assert!(agent.config.auto_search);
    }

    #[test]
    fn test_agent_process() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("What is 2 + 2?");
        assert!(matches!(
            result.status,
            LoopStatus::Finished | LoopStatus::MaxIterationsReached
        ));
    }

    #[test]
    fn test_extract_search_query() {
        let agent = AgenticLoop::new(LoopConfig::default());
        let query = agent.extract_search_query("What is the current weather in Madrid?");
        assert!(!query.contains("what is"));
    }

    // === Additional Unit Tests ===

    #[test]
    fn test_agent_config_default() {
        let config = LoopConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.auto_search);
        assert!(config.search_triggers.contains(&"current".to_string()));
        assert!(config.search_triggers.contains(&"latest".to_string()));
        assert!(config.include_tool_history);
        assert_eq!(config.max_context_tokens, 4096);
    }

    #[test]
    fn test_agentic_loop_default() {
        let agent = AgenticLoop::default();
        assert!(agent.get_tools().len() >= 3);
    }

    #[test]
    fn test_clear_history() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        agent.process("First message");
        assert!(!agent.get_conversation().is_empty());

        agent.clear_history();
        assert!(agent.get_conversation().is_empty());
    }

    #[test]
    fn test_with_system_prompt() {
        let agent =
            AgenticLoop::new(LoopConfig::default()).with_system_prompt("You are a test assistant.");

        assert_eq!(agent.config.system_prompt, "You are a test assistant.");
    }

    #[test]
    fn test_register_custom_tool() {
        let mut agent = AgenticLoop::new(LoopConfig::default());
        let initial_count = agent.get_tools().len();

        let custom_tool = Tool::new("custom_tool", "A custom tool for testing").with_parameter(
            ToolParameter::new("input", "The input parameter", ParameterType::String),
        );

        agent.register_tool(custom_tool);

        assert_eq!(agent.get_tools().len(), initial_count + 1);
        assert!(agent.get_tools().iter().any(|t| t.name == "custom_tool"));
    }

    #[test]
    fn test_get_tools_schema() {
        let agent = AgenticLoop::new(LoopConfig::default());
        let schema = agent.get_tools_schema();

        assert!(!schema.is_empty());
        // Each tool should have a function with a name in the schema (OpenAI format)
        for tool_schema in &schema {
            assert_eq!(
                tool_schema.get("type").and_then(|v| v.as_str()),
                Some("function")
            );
            assert!(tool_schema.get("function").is_some());
            assert!(tool_schema["function"].get("name").is_some());
        }
    }

    #[test]
    fn test_search_trigger_detection_all_keywords() {
        let agent = AgenticLoop::new(LoopConfig::default());

        let triggers = vec![
            "current price of Bitcoin",
            "latest news",
            "today's weather",
            "what is happening now",
            "recent developments",
            "2024 predictions",
            "2025 outlook",
            "breaking news",
            "Bitcoin price today",
            "weather forecast",
        ];

        for query in triggers {
            assert!(agent.needs_web_search(query), "Failed for: {}", query);
        }
    }

    #[test]
    fn test_no_search_trigger() {
        let agent = AgenticLoop::new(LoopConfig::default());

        let no_triggers = vec![
            "What is 2 + 2?",
            "Explain recursion",
            "How does a computer work?",
            "Define photosynthesis",
        ];

        for query in no_triggers {
            assert!(
                !agent.needs_web_search(query),
                "False positive for: {}",
                query
            );
        }
    }

    #[test]
    fn test_custom_search_triggers() {
        let agent = AgenticLoop::new(LoopConfig {
            search_triggers: vec!["custom_trigger".to_string()],
            ..Default::default()
        });

        assert!(agent.needs_web_search("This has custom_trigger"));
        assert!(!agent.needs_web_search("This has current")); // Default trigger removed
    }

    #[test]
    fn test_auto_search_disabled() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("What is the current weather?");

        // Should still process but not trigger search behavior
        assert!(matches!(
            result.status,
            LoopStatus::Finished | LoopStatus::MaxIterationsReached
        ));
    }

    #[test]
    fn test_extract_search_query_variants() {
        let agent = AgenticLoop::new(LoopConfig::default());

        // Test various query formats
        let queries = vec![
            ("what is the weather?", "the weather?"),
            ("how to cook pasta?", "cook pasta?"),
            ("where is Tokyo?", "tokyo?"),
            ("tell me about Rust", "rust"),
        ];

        for (input, _expected_contains) in queries {
            let extracted = agent.extract_search_query(input);
            assert!(!extracted.to_lowercase().starts_with("what is"));
            assert!(!extracted.to_lowercase().starts_with("how to"));
            assert!(!extracted.to_lowercase().starts_with("where is"));
            assert!(!extracted.to_lowercase().starts_with("tell me about"));
        }
    }

    #[test]
    fn test_extract_search_query_length_limit() {
        let agent = AgenticLoop::new(LoopConfig::default());

        let long_query = "A".repeat(200);
        let extracted = agent.extract_search_query(&long_query);

        assert!(extracted.len() <= 100);
    }

    #[test]
    fn test_max_iterations_reached() {
        let mut agent = AgenticLoop::new(LoopConfig {
            max_iterations: 1,
            auto_search: false,
            ..Default::default()
        });

        // Force multiple iterations by using a config that doesn't auto-finish
        let result = agent.process("Test query");

        // Should complete within max iterations
        assert!(result.total_iterations <= 1);
    }

    #[test]
    fn test_conversation_history() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            system_prompt: "Test system prompt".to_string(),
            ..Default::default()
        });

        agent.process("First message");

        let conversation = agent.get_conversation();

        // Should have system prompt and user message
        assert!(conversation.len() >= 2);
        assert!(conversation.iter().any(|m| m.role == LoopRole::System));
        assert!(conversation.iter().any(|m| m.role == LoopRole::User));
    }

    #[test]
    fn test_agent_state_initial() {
        let agent = AgenticLoop::new(LoopConfig::default());

        assert_eq!(agent.state.iteration, 0);
        assert_eq!(agent.state.status, LoopStatus::Thinking);
        assert!(agent.state.tool_calls.is_empty());
        assert!(agent.state.tool_results.is_empty());
    }

    #[test]
    fn test_iteration_result_structure() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("Simple query");

        // Should have at least one iteration
        assert!(!result.iterations.is_empty());

        // First iteration should have an iteration number
        assert_eq!(result.iterations[0].iteration, 1);
    }

    #[test]
    fn test_agent_loop_result_structure() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("Test");

        // Result should have all expected fields
        assert!(result.total_iterations > 0);
        assert!(result.final_answer.is_some() || result.status == LoopStatus::MaxIterationsReached);
    }

    #[test]
    fn test_agent_builder_chaining() {
        let agent = AgentBuilder::new()
            .with_max_iterations(20)
            .with_auto_search(false)
            .with_search_triggers(vec!["custom".to_string()])
            .with_system_prompt("Custom prompt")
            .build();

        assert_eq!(agent.config.max_iterations, 20);
        assert!(!agent.config.auto_search);
        assert_eq!(agent.config.search_triggers, vec!["custom".to_string()]);
        assert_eq!(agent.config.system_prompt, "Custom prompt");
    }

    #[test]
    fn test_agent_builder_with_custom_tool() {
        let custom_tool = Tool::new("test_tool", "Test description").with_parameter(
            ToolParameter::new("param", "A parameter", ParameterType::String),
        );

        let agent = AgentBuilder::new().with_tool(custom_tool).build();

        assert!(agent.get_tools().iter().any(|t| t.name == "test_tool"));
    }

    #[test]
    fn test_agent_builder_default() {
        let builder = AgentBuilder::default();
        let agent = builder.build();

        // Should have default config
        assert_eq!(agent.config.max_iterations, 10);
    }

    #[test]
    fn test_all_agent_statuses() {
        let statuses = vec![
            LoopStatus::Thinking,
            LoopStatus::CallingTool,
            LoopStatus::WaitingForToolResult,
            LoopStatus::Finished,
            LoopStatus::Error,
            LoopStatus::MaxIterationsReached,
            LoopStatus::UserStalled,
        ];

        for status in statuses {
            // Just verify they can be created and compared
            assert_eq!(status, status);
        }
    }

    #[test]
    fn test_all_agent_roles() {
        let roles = vec![
            LoopRole::System,
            LoopRole::User,
            LoopRole::Assistant,
            LoopRole::Tool,
        ];

        for role in roles {
            let msg = LoopMessage {
                role,
                content: "test".to_string(),
                tool_calls: None,
                tool_results: None,
                #[cfg(feature = "vision")]
                images: Vec::new(),
            };
            assert_eq!(msg.role, role);
        }
    }

    #[test]
    fn test_builtin_tools_present() {
        let agent = AgenticLoop::new(LoopConfig::default());
        let tools = agent.get_tools();

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"datetime"));
        assert!(names.contains(&"text_length"));
    }

    #[test]
    fn test_process_resets_state() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        // First process
        agent.process("First query");

        // State should be reset for second query
        let result = agent.process("Second query");

        // Iteration count should start fresh
        assert!(result.total_iterations <= agent.config.max_iterations);
    }

    #[test]
    fn test_build_context() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            system_prompt: "System".to_string(),
            ..Default::default()
        });

        agent.process("User message");

        let context = agent.build_context();

        // Context should include system and user messages
        assert!(context.len() >= 2);
    }

    #[test]
    fn test_include_tool_history_setting() {
        let agent_with_history = AgenticLoop::new(LoopConfig {
            include_tool_history: true,
            ..Default::default()
        });
        assert!(agent_with_history.config.include_tool_history);

        let agent_without_history = AgenticLoop::new(LoopConfig {
            include_tool_history: false,
            ..Default::default()
        });
        assert!(!agent_without_history.config.include_tool_history);
    }

    #[test]
    fn test_multiple_sequential_processes() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        // Process multiple queries without clearing history
        let result1 = agent.process("Query 1");
        let result2 = agent.process("Query 2");
        let result3 = agent.process("Query 3");

        // Each should complete successfully
        assert!(matches!(
            result1.status,
            LoopStatus::Finished | LoopStatus::MaxIterationsReached
        ));
        assert!(matches!(
            result2.status,
            LoopStatus::Finished | LoopStatus::MaxIterationsReached
        ));
        assert!(matches!(
            result3.status,
            LoopStatus::Finished | LoopStatus::MaxIterationsReached
        ));

        // Conversation should grow (at least user messages are added)
        let conversation = agent.get_conversation();
        assert!(
            conversation.len() >= 3,
            "Expected at least 3 messages, got {}",
            conversation.len()
        );
    }

    #[test]
    fn test_response_generator_callback() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        agent.set_response_generator(|conversation| {
            let user_msgs: Vec<_> = conversation
                .iter()
                .filter(|m| m.role == LoopRole::User)
                .collect();
            format!("Custom response for {} user messages", user_msgs.len())
        });

        let result = agent.process("Hello");
        assert_eq!(result.status, LoopStatus::Finished);
        assert!(result
            .final_answer
            .as_ref()
            .unwrap()
            .contains("Custom response for 1 user messages"));
    }

    #[test]
    fn test_builder_with_response_generator() {
        let agent = AgentBuilder::new()
            .with_max_iterations(5)
            .with_response_generator(|_| "Builder callback response".to_string())
            .build();

        assert!(agent.response_generator.is_some());
    }

    #[test]
    fn test_default_response_without_callback() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("test query");
        // Without callback, should produce default context-based response
        assert!(result.final_answer.is_some());
        assert!(result
            .final_answer
            .unwrap()
            .contains("based on available context"));
    }

    #[cfg(feature = "stall-detection")]
    #[test]
    fn test_stall_heuristic_builder_and_accessor() {
        use crate::stall_detection::KeywordStallDetector;
        let agent = AgenticLoop::new(LoopConfig::default())
            .with_stall_heuristic(Box::new(KeywordStallDetector::new()));
        assert!(agent.stall_heuristic().is_some());
    }

    #[cfg(feature = "stall-detection")]
    #[test]
    fn test_stall_heuristic_observes_user_message() {
        use crate::stall_detection::{KeywordStallDetector, StallDecision, StallHeuristic};
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            ..Default::default()
        })
        .with_stall_heuristic(Box::new(KeywordStallDetector::new()));
        let _ = agent.process("I'm so frustrated with this, it's useless!");
        // Heuristic has absorbed the frustrated user message; check() should fire.
        let h = agent.stall_heuristic_mut().expect("heuristic attached");
        assert!(matches!(h.check(), StallDecision::Stalled(_)));
    }

    // === Batch 2 — vision wiring ===

    #[cfg(feature = "vision")]
    #[test]
    fn test_loop_message_carries_images() {
        use crate::vision::ImageInput;
        let img = ImageInput::from_url("https://example.com/x.png");
        let msg = LoopMessage {
            role: LoopRole::User,
            content: "look".to_string(),
            tool_calls: None,
            tool_results: None,
            images: vec![img],
        };
        assert_eq!(msg.images.len(), 1);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_tool_result_with_image_builder() {
        use crate::tool_calling::ToolResult;
        use crate::vision::ImageInput;
        let img = ImageInput::from_url("https://example.com/screenshot.png");
        let tr = ToolResult {
            call_id: "1".to_string(),
            name: "screenshot".to_string(),
            success: true,
            output: "captured".to_string(),
            error: None,
            images: Vec::new(),
        }
        .with_image(img);
        assert!(tr.has_images());
        assert_eq!(tr.images.len(), 1);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_with_vision_config_sets_limits() {
        use crate::vision::VisionLimits;
        let agent = AgenticLoop::new(LoopConfig::default()).with_vision_config(VisionLimits {
            max_decoded_pixels: 1024 * 1024,
            max_encoded_bytes: 1024,
            max_dimension: 1024,
        });
        assert!(agent.vision_config.is_some());
        let cfg = agent.vision_config.unwrap();
        assert_eq!(cfg.max_dimension, 1024);
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_process_with_images_rejects_oversize_payload() {
        use crate::vision::{ImageInput, VisionLimits};
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            max_iterations: 1,
            ..Default::default()
        })
        .with_vision_config(VisionLimits {
            max_decoded_pixels: 100,
            max_encoded_bytes: 8,
            max_dimension: 100,
        });
        // Build a fake JPEG header big enough to fail the encoded-bytes cap.
        let big_payload = vec![0xFFu8; 1024];
        let b64 = crate::vision::base64_encode(&big_payload);
        let img = ImageInput::from_base64(&b64, "image/jpeg");
        let res = agent.process_with_images("describe", vec![img]);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("validation failed"));
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_process_with_images_url_only_skips_validation() {
        let mut agent = AgenticLoop::new(LoopConfig {
            auto_search: false,
            max_iterations: 1,
            ..Default::default()
        });
        agent.set_response_generator(|_| "ok".to_string());
        let img = crate::vision::ImageInput::from_url("https://example.com/x.png");
        let res = agent.process_with_images("look", vec![img]);
        assert!(res.is_ok());
        let r = res.unwrap();
        // The conversation now carries the image-bearing user message.
        let user_msg = agent
            .conversation
            .iter()
            .find(|m| m.role == LoopRole::User)
            .unwrap();
        assert_eq!(user_msg.images.len(), 1);
        assert_eq!(r.status, LoopStatus::Finished);
    }
}
