//! Agentic loop system
//!
//! Implements an autonomous agent loop that can use tools, search the web,
//! and complete complex tasks iteratively.

use crate::tool_calling::{ToolRegistry, ToolCall, ToolResult, Tool, ToolParameter, ParameterType};
use crate::web_search::{WebSearchManager, SearchConfig};

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
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

impl Default for AgentConfig {
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
pub struct AgentState {
    pub iteration: usize,
    pub status: AgentStatus,
    pub thought: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub final_answer: Option<String>,
}

/// Agent execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Thinking,
    CallingTool,
    WaitingForToolResult,
    Finished,
    Error,
    MaxIterationsReached,
}

/// Message in the agent conversation
#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub role: AgentRole,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_results: Option<Vec<ToolResult>>,
}

/// Role in agent conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Agentic loop that can reason, use tools, and search
pub struct AgenticLoop {
    config: AgentConfig,
    tools: ToolRegistry,
    search_manager: Option<WebSearchManager>,
    conversation: Vec<AgentMessage>,
    state: AgentState,
}

impl AgenticLoop {
    pub fn new(config: AgentConfig) -> Self {
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
            state: AgentState {
                iteration: 0,
                status: AgentStatus::Thinking,
                thought: None,
                tool_calls: Vec::new(),
                tool_results: Vec::new(),
                final_answer: None,
            },
        }
    }

    /// Enable web search capability
    pub fn with_web_search(mut self, search_config: SearchConfig) -> Self {
        self.search_manager = Some(WebSearchManager::new(search_config));

        // Register web search tool
        let search_tool = Tool::new("web_search", "Search the web for current information")
            .with_parameter(ToolParameter::new(
                "query",
                "The search query",
                ParameterType::String,
            ));

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

    /// Process a user message and run the agent loop
    pub fn process(&mut self, user_message: &str) -> AgentLoopResult {
        // Reset state for new query
        self.state = AgentState {
            iteration: 0,
            status: AgentStatus::Thinking,
            thought: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            final_answer: None,
        };

        // Add system prompt if configured
        if !self.config.system_prompt.is_empty() && self.conversation.is_empty() {
            self.conversation.push(AgentMessage {
                role: AgentRole::System,
                content: self.config.system_prompt.clone(),
                tool_calls: None,
                tool_results: None,
            });
        }

        // Add user message
        self.conversation.push(AgentMessage {
            role: AgentRole::User,
            content: user_message.to_string(),
            tool_calls: None,
            tool_results: None,
        });

        // Check if we should auto-search
        let needs_search = self.config.auto_search && self.needs_web_search(user_message);

        // Run the agent loop
        let mut iterations = Vec::new();

        while self.state.iteration < self.config.max_iterations {
            self.state.iteration += 1;

            let iteration_result = self.run_iteration(needs_search && self.state.iteration == 1);
            iterations.push(iteration_result.clone());

            if matches!(self.state.status, AgentStatus::Finished | AgentStatus::Error) {
                break;
            }
        }

        if self.state.iteration >= self.config.max_iterations && self.state.status != AgentStatus::Finished {
            self.state.status = AgentStatus::MaxIterationsReached;
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
        self.config.search_triggers.iter().any(|trigger| query_lower.contains(trigger))
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
            if let Some(user_msg) = self.conversation.iter().rev().find(|m| m.role == AgentRole::User) {
                let query = self.extract_search_query(&user_msg.content);

                let call = ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    tool_name: "web_search".to_string(),
                    arguments: [("query".to_string(), serde_json::Value::String(query))]
                        .into_iter().collect(),
                };

                result.action = Some(format!("Searching web: {}", call.arguments.get("query").unwrap()));
                let tool_result = self.execute_tool_call(&call);
                result.observation = Some(tool_result.output.clone());

                self.state.tool_calls.push(call);

                // Add tool result to conversation
                if self.config.include_tool_history {
                    let output_content = tool_result.output.clone();
                    self.conversation.push(AgentMessage {
                        role: AgentRole::Tool,
                        content: output_content,
                        tool_calls: None,
                        tool_results: Some(vec![tool_result.clone()]),
                    });
                }

                self.state.tool_results.push(tool_result);

                return result;
            }
        }

        // Generate context for model (this would normally call the LLM)
        let context = self.build_context();
        result.thought = Some(format!("Processing with {} messages in context", context.len()));

        // In a real implementation, this would call the LLM
        // For now, we simulate a finish state
        self.state.status = AgentStatus::Finished;
        self.state.final_answer = Some(self.generate_response_with_context());

        result
    }

    /// Extract a search query from user message
    fn extract_search_query(&self, message: &str) -> String {
        // Remove common question words and clean up
        let mut query = message.to_string();
        for word in &["what is", "what are", "how to", "where is", "when is", "who is", "tell me about"] {
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
        if call.tool_name == "web_search" {
            if let Some(ref mut search_manager) = self.search_manager {
                let query = call.arguments.get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let max_chars = call.arguments.get("num_results")
                    .and_then(|v| v.as_u64())
                    .map(|n| (n as usize) * 500) // Approx 500 chars per result
                    .unwrap_or(2500);

                match search_manager.search_for_context(query, max_chars) {
                    Ok(formatted) => {
                        return ToolResult {
                            call_id: call.id.clone(),
                            tool_name: call.tool_name.clone(),
                            success: true,
                            output: formatted,
                            error: None,
                        };
                    }
                    Err(e) => {
                        return ToolResult {
                            call_id: call.id.clone(),
                            tool_name: call.tool_name.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(e.to_string()),
                        };
                    }
                }
            }
        }

        // Execute other tools through registry
        self.tools.execute(call)
    }

    /// Build context for the model
    fn build_context(&self) -> Vec<&AgentMessage> {
        self.conversation.iter().collect()
    }

    /// Generate response using available context
    fn generate_response_with_context(&self) -> String {
        // In real implementation, this calls the LLM
        // For now, return summary of available info
        let mut response = String::new();

        // Include tool results if available
        for result in &self.state.tool_results {
            if result.success {
                response.push_str(&format!("Based on search results:\n{}\n", result.output));
            }
        }

        if response.is_empty() {
            response = "Response generated based on available context.".to_string();
        }

        response
    }

    /// Get current conversation
    pub fn get_conversation(&self) -> &[AgentMessage] {
        &self.conversation
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation.clear();
        self.state = AgentState {
            iteration: 0,
            status: AgentStatus::Thinking,
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
        Self::new(AgentConfig::default())
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
    pub status: AgentStatus,
    pub tool_calls_made: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

/// Builder for creating configured agents
pub struct AgentBuilder {
    config: AgentConfig,
    tools: Vec<Tool>,
    search_config: Option<SearchConfig>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            tools: Vec::new(),
            search_config: None,
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

    pub fn build(self) -> AgenticLoop {
        let mut agent = AgenticLoop::new(self.config);

        for tool in self.tools {
            agent.register_tool(tool);
        }

        if let Some(search_config) = self.search_config {
            agent = agent.with_web_search(search_config);
        }

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
        let agent = AgenticLoop::new(AgentConfig::default());
        assert!(agent.get_tools().len() >= 3); // calculator, datetime, text_length
    }

    #[test]
    fn test_needs_web_search() {
        let agent = AgenticLoop::new(AgentConfig::default());
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
        let mut agent = AgenticLoop::new(AgentConfig {
            auto_search: false,
            ..Default::default()
        });

        let result = agent.process("What is 2 + 2?");
        assert!(matches!(result.status, AgentStatus::Finished | AgentStatus::MaxIterationsReached));
    }

    #[test]
    fn test_extract_search_query() {
        let agent = AgenticLoop::new(AgentConfig::default());
        let query = agent.extract_search_query("What is the current weather in Madrid?");
        assert!(!query.contains("what is"));
    }
}
