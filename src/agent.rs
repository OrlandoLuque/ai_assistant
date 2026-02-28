//! Agent Framework - Basic agent system for multi-step tasks
//!
//! This module provides a simple agent framework for executing
//! multi-step tasks with tool calling capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Agent state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is idle
    Idle,
    /// Agent is thinking
    Thinking,
    /// Agent is executing a tool
    ExecutingTool(String),
    /// Agent is waiting for user input
    WaitingForInput,
    /// Agent completed successfully
    Completed,
    /// Agent failed with error
    Failed(String),
}

/// A step in the agent's execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// Step number
    pub step_number: usize,
    /// Thought process
    pub thought: String,
    /// Action taken (tool name or "final_answer")
    pub action: String,
    /// Action input
    pub action_input: String,
    /// Observation/result
    pub observation: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Tool definition for agents
#[derive(Clone)]
pub struct AgentTool {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Parameter descriptions
    pub parameters: HashMap<String, String>,
    /// The tool handler function
    handler: Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
}

impl fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("handler", &"<Arc<dyn Fn>>")
            .finish()
    }
}

impl AgentTool {
    /// Create a new agent tool
    pub fn new<F>(name: impl Into<String>, description: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: HashMap::new(),
            handler: Arc::new(handler),
        }
    }

    /// Add a parameter description
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters.insert(name.into(), description.into());
        self
    }

    /// Execute the tool
    pub fn execute(&self, input: &str) -> Result<String, String> {
        (self.handler)(input)
    }
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum steps before stopping
    pub max_steps: usize,
    /// Maximum retries per step
    pub max_retries: usize,
    /// Verbose logging
    pub verbose: bool,
    /// Stop on first error
    pub stop_on_error: bool,
    /// System prompt
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            max_retries: 3,
            verbose: false,
            stop_on_error: false,
            system_prompt: None,
        }
    }
}

/// Agent execution context
#[derive(Debug, Clone)]
pub struct AgentContext {
    /// Variables available to the agent
    pub variables: HashMap<String, String>,
    /// Accumulated observations
    pub observations: Vec<String>,
    /// Current step number
    pub current_step: usize,
}

impl AgentContext {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            observations: Vec::new(),
            current_step: 0,
        }
    }

    /// Set a variable
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(key.into(), value.into());
    }

    /// Get a variable
    pub fn get(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Add an observation
    pub fn add_observation(&mut self, obs: impl Into<String>) {
        self.observations.push(obs.into());
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Whether execution completed successfully
    pub success: bool,
    /// Final answer (if successful)
    pub answer: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// All steps taken
    pub steps: Vec<AgentStep>,
    /// Total execution time in ms
    pub execution_time_ms: u64,
    /// Total tokens used (if tracked)
    pub total_tokens: Option<usize>,
}

/// React-style agent (Reasoning + Acting)
#[derive(Debug)]
pub struct ReactAgent {
    /// Agent configuration
    config: AgentConfig,
    /// Available tools
    tools: HashMap<String, AgentTool>,
    /// Execution history
    steps: Vec<AgentStep>,
    /// Current state
    state: AgentState,
    /// Context
    context: AgentContext,
}

impl ReactAgent {
    /// Create a new React agent
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            tools: HashMap::new(),
            steps: Vec::new(),
            state: AgentState::Idle,
            context: AgentContext::new(),
        }
    }

    /// Add a tool
    pub fn add_tool(&mut self, tool: AgentTool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Get current state
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Get all steps
    pub fn steps(&self) -> &[AgentStep] {
        &self.steps
    }

    /// Get context
    pub fn context(&self) -> &AgentContext {
        &self.context
    }

    /// Get mutable context
    pub fn context_mut(&mut self) -> &mut AgentContext {
        &mut self.context
    }

    /// Build the system prompt for the agent
    pub fn build_system_prompt(&self) -> String {
        let mut prompt = self.config.system_prompt.clone().unwrap_or_else(|| {
            "You are a helpful AI assistant that can use tools to accomplish tasks. \
             Think step by step and use the available tools when needed."
                .to_string()
        });

        prompt.push_str("\n\nAvailable tools:\n");
        for tool in self.tools.values() {
            prompt.push_str(&format!("\n- {}: {}", tool.name, tool.description));
            if !tool.parameters.is_empty() {
                prompt.push_str("\n  Parameters:");
                for (param, desc) in &tool.parameters {
                    prompt.push_str(&format!("\n    - {}: {}", param, desc));
                }
            }
        }

        prompt.push_str("\n\nRespond in this format:\n");
        prompt.push_str("Thought: <your reasoning>\n");
        prompt.push_str("Action: <tool name or 'final_answer'>\n");
        prompt.push_str("Action Input: <input for the tool or the final answer>\n");

        prompt
    }

    /// Parse the agent's response
    pub fn parse_response(&self, response: &str) -> Option<(String, String, String)> {
        let mut thought = String::new();
        let mut action = String::new();
        let mut action_input = String::new();

        for line in response.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Thought:") {
                thought = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("Action:") {
                action = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("Action Input:") {
                action_input = rest.trim().to_string();
            }
        }

        if !thought.is_empty() && !action.is_empty() {
            Some((thought, action, action_input))
        } else {
            None
        }
    }

    /// Execute a single step
    pub fn execute_step(
        &mut self,
        thought: String,
        action: String,
        action_input: String,
    ) -> Result<String, String> {
        self.context.current_step += 1;
        let step_num = self.context.current_step;

        self.state = AgentState::ExecutingTool(action.clone());

        // Check if this is the final answer
        if action.to_lowercase() == "final_answer" || action.to_lowercase() == "finish" {
            let step = AgentStep {
                step_number: step_num,
                thought,
                action: "final_answer".to_string(),
                action_input: action_input.clone(),
                observation: None,
                timestamp: current_timestamp(),
            };
            self.steps.push(step);
            self.state = AgentState::Completed;
            return Ok(action_input);
        }

        // Execute the tool
        let observation = if let Some(tool) = self.tools.get(&action) {
            match tool.execute(&action_input) {
                Ok(result) => {
                    self.context.add_observation(result.clone());
                    result
                }
                Err(e) => {
                    if self.config.stop_on_error {
                        self.state = AgentState::Failed(e.clone());
                        return Err(e);
                    }
                    format!("Error: {}", e)
                }
            }
        } else {
            let err = format!("Unknown tool: {}", action);
            if self.config.stop_on_error {
                self.state = AgentState::Failed(err.clone());
                return Err(err);
            }
            err
        };

        let step = AgentStep {
            step_number: step_num,
            thought,
            action,
            action_input,
            observation: Some(observation.clone()),
            timestamp: current_timestamp(),
        };
        self.steps.push(step);

        self.state = AgentState::Thinking;
        Ok(observation)
    }

    /// Build a prompt with history
    pub fn build_prompt_with_history(&self, task: &str) -> String {
        let mut prompt = format!("Task: {}\n", task);

        for step in &self.steps {
            prompt.push_str(&format!("\nThought: {}\n", step.thought));
            prompt.push_str(&format!("Action: {}\n", step.action));
            prompt.push_str(&format!("Action Input: {}\n", step.action_input));
            if let Some(ref obs) = step.observation {
                prompt.push_str(&format!("Observation: {}\n", obs));
            }
        }

        prompt.push_str("\nContinue:\n");
        prompt
    }

    /// Check if agent should stop
    pub fn should_stop(&self) -> bool {
        match &self.state {
            AgentState::Completed | AgentState::Failed(_) => true,
            _ => self.context.current_step >= self.config.max_steps,
        }
    }

    /// Reset the agent
    pub fn reset(&mut self) {
        self.steps.clear();
        self.state = AgentState::Idle;
        self.context = AgentContext::new();
    }
}

/// Simple planning agent
#[derive(Debug)]
pub struct PlanningAgent {
    /// Agent configuration
    config: AgentConfig,
    /// The plan steps
    plan: Vec<PlanStep>,
    /// Execution results
    results: HashMap<usize, String>,
}

/// A step in the plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step number
    pub step: usize,
    /// Description
    pub description: String,
    /// Dependencies (other step numbers)
    pub dependencies: Vec<usize>,
    /// Tool to use (optional)
    pub tool: Option<String>,
    /// Status
    pub status: PlanStepStatus,
}

/// Plan step status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
    Skipped,
}

impl PlanningAgent {
    /// Create a new planning agent
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            plan: Vec::new(),
            results: HashMap::new(),
        }
    }

    /// Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Add a plan step
    pub fn add_step(&mut self, description: impl Into<String>) -> &mut PlanStep {
        let step_num = self.plan.len() + 1;
        self.plan.push(PlanStep {
            step: step_num,
            description: description.into(),
            dependencies: Vec::new(),
            tool: None,
            status: PlanStepStatus::Pending,
        });
        self.plan.last_mut().expect("plan step was just pushed")
    }

    /// Get the plan
    pub fn plan(&self) -> &[PlanStep] {
        &self.plan
    }

    /// Get next executable step
    pub fn next_step(&self) -> Option<&PlanStep> {
        self.plan.iter().find(|step| {
            step.status == PlanStepStatus::Pending
                && step.dependencies.iter().all(|dep| {
                    self.plan
                        .iter()
                        .find(|s| s.step == *dep)
                        .map(|s| s.status == PlanStepStatus::Completed)
                        .unwrap_or(false)
                })
        })
    }

    /// Mark a step as completed
    pub fn complete_step(&mut self, step_num: usize, result: String) {
        if let Some(step) = self.plan.iter_mut().find(|s| s.step == step_num) {
            step.status = PlanStepStatus::Completed;
            self.results.insert(step_num, result);
        }
    }

    /// Mark a step as failed
    pub fn fail_step(&mut self, step_num: usize, error: String) {
        if let Some(step) = self.plan.iter_mut().find(|s| s.step == step_num) {
            step.status = PlanStepStatus::Failed(error);
        }
    }

    /// Get result for a step
    pub fn get_result(&self, step_num: usize) -> Option<&String> {
        self.results.get(&step_num)
    }

    /// Check if plan is complete
    pub fn is_complete(&self) -> bool {
        self.plan.iter().all(|step| {
            matches!(
                step.status,
                PlanStepStatus::Completed | PlanStepStatus::Skipped
            )
        })
    }

    /// Generate plan summary
    pub fn summary(&self) -> String {
        let mut summary = String::from("Plan Status:\n");
        for step in &self.plan {
            let status = match &step.status {
                PlanStepStatus::Pending => "⏳",
                PlanStepStatus::InProgress => "🔄",
                PlanStepStatus::Completed => "✅",
                PlanStepStatus::Failed(_) => "❌",
                PlanStepStatus::Skipped => "⏭️",
            };
            summary.push_str(&format!("{} {}. {}\n", status, step.step, step.description));
        }
        summary
    }
}

impl PlanStep {
    /// Add a dependency
    pub fn depends_on(&mut self, step: usize) -> &mut Self {
        self.dependencies.push(step);
        self
    }

    /// Set the tool
    pub fn with_tool(&mut self, tool: impl Into<String>) -> &mut Self {
        self.tool = Some(tool.into());
        self
    }
}

/// Agent executor for running agents
#[derive(Debug)]
pub struct AgentExecutor {
    /// Maximum concurrent agents
    max_concurrent: usize,
}

impl AgentExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self { max_concurrent: 1 }
    }

    /// Set max concurrent agents
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }
}

impl Default for AgentExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent callback trait
pub trait AgentCallback: Send + Sync {
    /// Called when agent starts thinking
    fn on_thinking(&self, _agent_id: &str) {}

    /// Called when agent executes a tool
    fn on_tool_execution(&self, _agent_id: &str, _tool: &str, _input: &str) {}

    /// Called when tool returns result
    fn on_tool_result(&self, _agent_id: &str, _tool: &str, _result: &str) {}

    /// Called when agent completes
    fn on_complete(&self, _agent_id: &str, _result: &AgentResult) {}

    /// Called on error
    fn on_error(&self, _agent_id: &str, _error: &str) {}
}

/// Logging callback implementation
#[derive(Debug)]
pub struct LoggingCallback {
    /// Whether verbose
    pub verbose: bool,
}

impl AgentCallback for LoggingCallback {
    fn on_thinking(&self, agent_id: &str) {
        if self.verbose {
            println!("[{}] Thinking...", agent_id);
        }
    }

    fn on_tool_execution(&self, agent_id: &str, tool: &str, input: &str) {
        if self.verbose {
            println!(
                "[{}] Executing tool '{}' with input: {}",
                agent_id, tool, input
            );
        }
    }

    fn on_tool_result(&self, agent_id: &str, tool: &str, result: &str) {
        if self.verbose {
            println!("[{}] Tool '{}' returned: {}", agent_id, tool, result);
        }
    }

    fn on_complete(&self, agent_id: &str, result: &AgentResult) {
        println!("[{}] Completed. Success: {}", agent_id, result.success);
    }

    fn on_error(&self, agent_id: &str, error: &str) {
        log::error!("[{}] Error: {}", agent_id, error);
    }
}

/// Create built-in tools for agents
pub fn create_builtin_agent_tools() -> Vec<AgentTool> {
    vec![
        // Calculator tool
        AgentTool::new(
            "calculator",
            "Perform basic arithmetic calculations",
            |input| {
                // Simple expression evaluator
                let result = evaluate_simple_math(input)?;
                Ok(format!("{}", result))
            },
        )
        .with_parameter("expression", "Mathematical expression like '2 + 3 * 4'"),
        // String manipulation tool
        AgentTool::new("string_tool", "Manipulate strings", |input| {
            let parts: Vec<&str> = input.splitn(2, ':').collect();
            if parts.len() != 2 {
                return Err("Format: operation:text".to_string());
            }
            let (op, text) = (parts[0].trim(), parts[1].trim());
            match op {
                "upper" => Ok(text.to_uppercase()),
                "lower" => Ok(text.to_lowercase()),
                "reverse" => Ok(text.chars().rev().collect()),
                "length" => Ok(text.len().to_string()),
                "trim" => Ok(text.trim().to_string()),
                _ => Err(format!("Unknown operation: {}", op)),
            }
        })
        .with_parameter(
            "input",
            "Format: operation:text (operations: upper, lower, reverse, length, trim)",
        ),
    ]
}

/// Simple math expression evaluator
fn evaluate_simple_math(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();

    // Handle parentheses first
    if expr.contains('(') {
        let start = match expr.rfind('(') {
            Some(pos) => pos,
            None => return Err("Mismatched parentheses".to_string()),
        };
        let end = expr[start..]
            .find(')')
            .map(|i| start + i)
            .ok_or("Mismatched parentheses")?;
        let inner = &expr[start + 1..end];
        let inner_result = evaluate_simple_math(inner)?;
        let new_expr = format!("{}{}{}", &expr[..start], inner_result, &expr[end + 1..]);
        return evaluate_simple_math(&new_expr);
    }

    // Find operator with lowest precedence
    let mut paren_depth = 0;
    let mut op_pos = None;
    let mut op_char = ' ';

    for (i, c) in expr.chars().enumerate() {
        match c {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '+' | '-' if paren_depth == 0 && i > 0 => {
                op_pos = Some(i);
                op_char = c;
            }
            '*' | '/' if paren_depth == 0 && op_pos.is_none() => {
                op_pos = Some(i);
                op_char = c;
            }
            _ => {}
        }
    }

    if let Some(pos) = op_pos {
        let left = evaluate_simple_math(&expr[..pos])?;
        let right = evaluate_simple_math(&expr[pos + 1..])?;
        match op_char {
            '+' => Ok(left + right),
            '-' => Ok(left - right),
            '*' => Ok(left * right),
            '/' => {
                if right == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(left / right)
                }
            }
            _ => Err(format!("Unknown operator: {}", op_char)),
        }
    } else {
        expr.parse::<f64>()
            .map_err(|_| format!("Cannot parse: {}", expr))
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_tool() {
        let tool = AgentTool::new("echo", "Echoes input", |input| Ok(input.to_string()))
            .with_parameter("text", "Text to echo");

        assert_eq!(tool.name, "echo");
        assert!(tool.parameters.contains_key("text"));
        assert_eq!(tool.execute("hello").unwrap(), "hello");
    }

    #[test]
    fn test_react_agent_parse_response() {
        let agent = ReactAgent::new(AgentConfig::default());

        let response =
            "Thought: I need to calculate something\nAction: calculator\nAction Input: 2 + 2";
        let parsed = agent.parse_response(response);

        assert!(parsed.is_some());
        let (thought, action, input) = parsed.unwrap();
        assert_eq!(thought, "I need to calculate something");
        assert_eq!(action, "calculator");
        assert_eq!(input, "2 + 2");
    }

    #[test]
    fn test_react_agent_execute_step() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("echo", "Echoes input", |input| {
            Ok(input.to_string())
        }));

        let result = agent.execute_step(
            "Testing".to_string(),
            "echo".to_string(),
            "hello".to_string(),
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello");
        assert_eq!(agent.steps().len(), 1);
    }

    #[test]
    fn test_react_agent_final_answer() {
        let mut agent = ReactAgent::new(AgentConfig::default());

        let result = agent.execute_step(
            "I have the answer".to_string(),
            "final_answer".to_string(),
            "42".to_string(),
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42");
        assert_eq!(*agent.state(), AgentState::Completed);
    }

    #[test]
    fn test_planning_agent() {
        let mut agent = PlanningAgent::new(AgentConfig::default());

        agent.add_step("First step");
        agent.add_step("Second step").depends_on(1);
        agent.add_step("Third step").depends_on(2);

        // First step should be next
        let next = agent.next_step();
        assert!(next.is_some());
        assert_eq!(next.unwrap().step, 1);

        // Complete first step
        agent.complete_step(1, "Done".to_string());

        // Second step should be next
        let next = agent.next_step();
        assert!(next.is_some());
        assert_eq!(next.unwrap().step, 2);
    }

    #[test]
    fn test_simple_math_evaluator() {
        assert!((evaluate_simple_math("2 + 3").unwrap() - 5.0).abs() < 0.001);
        assert!((evaluate_simple_math("10 - 4").unwrap() - 6.0).abs() < 0.001);
        assert!((evaluate_simple_math("3 * 4").unwrap() - 12.0).abs() < 0.001);
        assert!((evaluate_simple_math("15 / 3").unwrap() - 5.0).abs() < 0.001);
        assert!((evaluate_simple_math("2 + 3 * 4").unwrap() - 14.0).abs() < 0.001);
    }

    #[test]
    fn test_builtin_tools() {
        let tools = create_builtin_agent_tools();
        assert!(tools.len() >= 2);

        // Test calculator
        let calc = tools.iter().find(|t| t.name == "calculator").unwrap();
        assert_eq!(calc.execute("2 + 2").unwrap(), "4");

        // Test string tool
        let str_tool = tools.iter().find(|t| t.name == "string_tool").unwrap();
        assert_eq!(str_tool.execute("upper: hello").unwrap(), "HELLO");
        assert_eq!(str_tool.execute("length: hello").unwrap(), "5");
    }

    #[test]
    fn test_agent_context() {
        let mut ctx = AgentContext::new();
        ctx.set("key1", "value1");
        ctx.add_observation("obs1");

        assert_eq!(ctx.get("key1"), Some(&"value1".to_string()));
        assert_eq!(ctx.observations.len(), 1);
    }

    #[test]
    fn test_agent_reset() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("test", "Test", |_| Ok("ok".to_string())));

        let _ = agent.execute_step("Think".to_string(), "test".to_string(), "input".to_string());
        assert!(!agent.steps().is_empty());

        agent.reset();
        assert!(agent.steps().is_empty());
        assert_eq!(*agent.state(), AgentState::Idle);
    }

    // === Additional Unit Tests ===

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_steps, 10);
        assert_eq!(config.max_retries, 3);
        assert!(!config.verbose);
        assert!(!config.stop_on_error);
        assert!(config.system_prompt.is_none());
    }

    #[test]
    fn test_agent_config_custom() {
        let config = AgentConfig {
            max_steps: 20,
            max_retries: 5,
            verbose: true,
            stop_on_error: true,
            system_prompt: Some("Custom prompt".to_string()),
        };

        assert_eq!(config.max_steps, 20);
        assert_eq!(config.max_retries, 5);
        assert!(config.verbose);
        assert!(config.stop_on_error);
        assert_eq!(config.system_prompt.as_ref().unwrap(), "Custom prompt");
    }

    #[test]
    fn test_all_agent_states() {
        let states = vec![
            AgentState::Idle,
            AgentState::Thinking,
            AgentState::ExecutingTool("test_tool".to_string()),
            AgentState::WaitingForInput,
            AgentState::Completed,
            AgentState::Failed("error message".to_string()),
        ];

        for state in states {
            match state {
                AgentState::Idle => assert_eq!(state, AgentState::Idle),
                AgentState::Thinking => assert_eq!(state, AgentState::Thinking),
                AgentState::ExecutingTool(ref name) => assert_eq!(name, "test_tool"),
                AgentState::WaitingForInput => assert_eq!(state, AgentState::WaitingForInput),
                AgentState::Completed => assert_eq!(state, AgentState::Completed),
                AgentState::Failed(ref msg) => assert_eq!(msg, "error message"),
            }
        }
    }

    #[test]
    fn test_agent_tool_error() {
        let tool = AgentTool::new("failing_tool", "Always fails", |_| {
            Err("Intentional failure".to_string())
        });

        let result = tool.execute("input");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Intentional failure");
    }

    #[test]
    fn test_execute_step_unknown_tool() {
        let mut agent = ReactAgent::new(AgentConfig {
            stop_on_error: false,
            ..Default::default()
        });

        let result = agent.execute_step(
            "Thinking".to_string(),
            "nonexistent_tool".to_string(),
            "input".to_string(),
        );

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Unknown tool"));
    }

    #[test]
    fn test_execute_step_unknown_tool_stop_on_error() {
        let mut agent = ReactAgent::new(AgentConfig {
            stop_on_error: true,
            ..Default::default()
        });

        let result = agent.execute_step(
            "Thinking".to_string(),
            "nonexistent_tool".to_string(),
            "input".to_string(),
        );

        assert!(result.is_err());
        assert!(matches!(*agent.state(), AgentState::Failed(_)));
    }

    #[test]
    fn test_execute_step_tool_error_with_stop() {
        let mut agent = ReactAgent::new(AgentConfig {
            stop_on_error: true,
            ..Default::default()
        });
        agent.add_tool(AgentTool::new("failing", "Fails", |_| {
            Err("Error!".to_string())
        }));

        let result = agent.execute_step(
            "Thinking".to_string(),
            "failing".to_string(),
            "input".to_string(),
        );

        assert!(result.is_err());
        assert!(matches!(*agent.state(), AgentState::Failed(_)));
    }

    #[test]
    fn test_execute_step_tool_error_continue() {
        let mut agent = ReactAgent::new(AgentConfig {
            stop_on_error: false,
            ..Default::default()
        });
        agent.add_tool(AgentTool::new("failing", "Fails", |_| {
            Err("Error!".to_string())
        }));

        let result = agent.execute_step(
            "Thinking".to_string(),
            "failing".to_string(),
            "input".to_string(),
        );

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Error:"));
        assert_eq!(*agent.state(), AgentState::Thinking);
    }

    #[test]
    fn test_should_stop_conditions() {
        let mut agent = ReactAgent::new(AgentConfig {
            max_steps: 5,
            ..Default::default()
        });

        // Initially should not stop
        assert!(!agent.should_stop());

        // After completing, should stop
        let _ = agent.execute_step(
            "Done".to_string(),
            "final_answer".to_string(),
            "The answer".to_string(),
        );
        assert!(agent.should_stop());
    }

    #[test]
    fn test_should_stop_max_steps() {
        let mut agent = ReactAgent::new(AgentConfig {
            max_steps: 2,
            ..Default::default()
        });
        agent.add_tool(AgentTool::new("tool", "Tool", |_| Ok("ok".to_string())));

        // Execute steps up to max
        let _ = agent.execute_step("Step 1".to_string(), "tool".to_string(), "".to_string());
        assert!(!agent.should_stop());

        let _ = agent.execute_step("Step 2".to_string(), "tool".to_string(), "".to_string());
        assert!(agent.should_stop());
    }

    #[test]
    fn test_build_system_prompt() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(
            AgentTool::new(
                "calculator",
                "Perform calculations",
                |_| Ok("0".to_string()),
            )
            .with_parameter("expression", "Math expression"),
        );

        let prompt = agent.build_system_prompt();

        assert!(prompt.contains("calculator"));
        assert!(prompt.contains("Perform calculations"));
        assert!(prompt.contains("expression"));
        assert!(prompt.contains("Thought:"));
        assert!(prompt.contains("Action:"));
        assert!(prompt.contains("Action Input:"));
    }

    #[test]
    fn test_build_system_prompt_custom() {
        let agent = ReactAgent::new(AgentConfig {
            system_prompt: Some("Custom system prompt".to_string()),
            ..Default::default()
        });

        let prompt = agent.build_system_prompt();
        assert!(prompt.contains("Custom system prompt"));
    }

    #[test]
    fn test_parse_response_invalid() {
        let agent = ReactAgent::new(AgentConfig::default());

        // Missing thought
        let response1 = "Action: test\nAction Input: input";
        assert!(agent.parse_response(response1).is_none());

        // Missing action
        let response2 = "Thought: thinking\nAction Input: input";
        assert!(agent.parse_response(response2).is_none());

        // Empty response
        assert!(agent.parse_response("").is_none());

        // Random text
        assert!(agent.parse_response("Just some random text").is_none());
    }

    #[test]
    fn test_parse_response_empty_action_input() {
        let agent = ReactAgent::new(AgentConfig::default());

        // Action input can be empty
        let response = "Thought: thinking\nAction: test\nAction Input:";
        let parsed = agent.parse_response(response);
        assert!(parsed.is_some());
        let (thought, action, input) = parsed.unwrap();
        assert_eq!(thought, "thinking");
        assert_eq!(action, "test");
        assert_eq!(input, "");
    }

    #[test]
    fn test_build_prompt_with_history() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("tool", "Test", |_| Ok("result".to_string())));

        // Execute some steps
        let _ = agent.execute_step(
            "First thought".to_string(),
            "tool".to_string(),
            "input1".to_string(),
        );
        let _ = agent.execute_step(
            "Second thought".to_string(),
            "tool".to_string(),
            "input2".to_string(),
        );

        let prompt = agent.build_prompt_with_history("Test task");

        assert!(prompt.contains("Task: Test task"));
        assert!(prompt.contains("First thought"));
        assert!(prompt.contains("Second thought"));
        assert!(prompt.contains("Observation:"));
        assert!(prompt.contains("Continue:"));
    }

    #[test]
    fn test_finish_action_variants() {
        for action in &[
            "final_answer",
            "Final_Answer",
            "FINAL_ANSWER",
            "finish",
            "Finish",
            "FINISH",
        ] {
            let mut agent = ReactAgent::new(AgentConfig::default());
            let result =
                agent.execute_step("Done".to_string(), action.to_string(), "Answer".to_string());

            assert!(result.is_ok());
            assert_eq!(*agent.state(), AgentState::Completed);
        }
    }

    #[test]
    fn test_agent_step_timestamp() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("tool", "Test", |_| Ok("ok".to_string())));

        let _ = agent.execute_step(
            "Thought".to_string(),
            "tool".to_string(),
            "input".to_string(),
        );

        let step = &agent.steps()[0];
        assert!(step.timestamp > 0);
    }

    #[test]
    fn test_agent_context_default() {
        let ctx = AgentContext::default();
        assert!(ctx.variables.is_empty());
        assert!(ctx.observations.is_empty());
        assert_eq!(ctx.current_step, 0);
    }

    #[test]
    fn test_planning_agent_fail_step() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Step 1");
        agent.add_step("Step 2");

        agent.fail_step(1, "Something went wrong".to_string());

        let step1 = agent.plan().iter().find(|s| s.step == 1).unwrap();
        assert!(matches!(step1.status, PlanStepStatus::Failed(_)));
    }

    #[test]
    fn test_planning_agent_get_result() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Step 1");

        agent.complete_step(1, "Result of step 1".to_string());

        let result = agent.get_result(1);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "Result of step 1");

        // Non-existent step
        assert!(agent.get_result(999).is_none());
    }

    #[test]
    fn test_planning_agent_is_complete() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Step 1");
        agent.add_step("Step 2");

        assert!(!agent.is_complete());

        agent.complete_step(1, "Done".to_string());
        assert!(!agent.is_complete());

        agent.complete_step(2, "Done".to_string());
        assert!(agent.is_complete());
    }

    #[test]
    fn test_planning_agent_summary() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("First step");
        agent.add_step("Second step");

        agent.complete_step(1, "Done".to_string());

        let summary = agent.summary();

        assert!(summary.contains("Plan Status:"));
        assert!(summary.contains("First step"));
        assert!(summary.contains("Second step"));
    }

    #[test]
    fn test_plan_step_with_tool() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Use calculator").with_tool("calculator");

        let step = &agent.plan()[0];
        assert_eq!(step.tool.as_ref().unwrap(), "calculator");
    }

    #[test]
    fn test_plan_step_multiple_dependencies() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Step 1");
        agent.add_step("Step 2");
        agent.add_step("Step 3").depends_on(1).depends_on(2);

        let step3 = agent.plan().iter().find(|s| s.step == 3).unwrap();
        assert_eq!(step3.dependencies.len(), 2);
        assert!(step3.dependencies.contains(&1));
        assert!(step3.dependencies.contains(&2));
    }

    #[test]
    fn test_planning_agent_next_step_with_dependencies() {
        let mut agent = PlanningAgent::new(AgentConfig::default());
        agent.add_step("Step 1");
        agent.add_step("Step 2").depends_on(1);
        agent.add_step("Step 3").depends_on(1).depends_on(2);

        // Step 1 should be first
        let next = agent.next_step();
        assert_eq!(next.unwrap().step, 1);

        // Complete step 1
        agent.complete_step(1, "Done".to_string());

        // Step 2 should be next
        let next = agent.next_step();
        assert_eq!(next.unwrap().step, 2);

        // Complete step 2
        agent.complete_step(2, "Done".to_string());

        // Step 3 should be next (both dependencies met)
        let next = agent.next_step();
        assert_eq!(next.unwrap().step, 3);
    }

    #[test]
    fn test_all_plan_step_statuses() {
        let statuses = vec![
            PlanStepStatus::Pending,
            PlanStepStatus::InProgress,
            PlanStepStatus::Completed,
            PlanStepStatus::Failed("error".to_string()),
            PlanStepStatus::Skipped,
        ];

        for status in statuses {
            match status {
                PlanStepStatus::Pending => assert_eq!(status, PlanStepStatus::Pending),
                PlanStepStatus::InProgress => assert_eq!(status, PlanStepStatus::InProgress),
                PlanStepStatus::Completed => assert_eq!(status, PlanStepStatus::Completed),
                PlanStepStatus::Failed(ref msg) => assert_eq!(msg, "error"),
                PlanStepStatus::Skipped => assert_eq!(status, PlanStepStatus::Skipped),
            }
        }
    }

    #[test]
    fn test_agent_executor_new() {
        let executor = AgentExecutor::new();
        assert_eq!(executor.max_concurrent, 1);
    }

    #[test]
    fn test_agent_executor_with_max_concurrent() {
        let executor = AgentExecutor::new().with_max_concurrent(4);
        assert_eq!(executor.max_concurrent, 4);
    }

    #[test]
    fn test_agent_executor_default() {
        let executor = AgentExecutor::default();
        assert_eq!(executor.max_concurrent, 1);
    }

    #[test]
    fn test_logging_callback() {
        let callback = LoggingCallback { verbose: true };

        // These should not panic
        callback.on_thinking("test_agent");
        callback.on_tool_execution("test_agent", "calculator", "2+2");
        callback.on_tool_result("test_agent", "calculator", "4");
        callback.on_error("test_agent", "test error");

        let result = AgentResult {
            success: true,
            answer: Some("42".to_string()),
            error: None,
            steps: vec![],
            execution_time_ms: 100,
            total_tokens: Some(50),
        };
        callback.on_complete("test_agent", &result);
    }

    #[test]
    fn test_simple_math_evaluator_division_by_zero() {
        let result = evaluate_simple_math("10 / 0");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Division by zero"));
    }

    #[test]
    fn test_simple_math_evaluator_parentheses() {
        let result = evaluate_simple_math("(2 + 3) * 4").unwrap();
        assert!((result - 20.0).abs() < 0.001);

        let result = evaluate_simple_math("2 * (3 + 4)").unwrap();
        assert!((result - 14.0).abs() < 0.001);

        let result = evaluate_simple_math("((1 + 2) * (3 + 4))").unwrap();
        assert!((result - 21.0).abs() < 0.001);
    }

    #[test]
    fn test_simple_math_evaluator_invalid() {
        let result = evaluate_simple_math("abc");
        assert!(result.is_err());

        let result = evaluate_simple_math("(2 + 3");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_math_evaluator_negative() {
        let result = evaluate_simple_math("5 - 10").unwrap();
        assert!((result - (-5.0)).abs() < 0.001);
    }

    #[test]
    fn test_simple_math_evaluator_decimals() {
        let result = evaluate_simple_math("3.14 * 2").unwrap();
        assert!((result - 6.28).abs() < 0.001);
    }

    #[test]
    fn test_string_tool_operations() {
        let tools = create_builtin_agent_tools();
        let str_tool = tools.iter().find(|t| t.name == "string_tool").unwrap();

        // Test all operations
        assert_eq!(str_tool.execute("upper: hello").unwrap(), "HELLO");
        assert_eq!(str_tool.execute("lower: HELLO").unwrap(), "hello");
        assert_eq!(str_tool.execute("reverse: hello").unwrap(), "olleh");
        assert_eq!(str_tool.execute("length: hello").unwrap(), "5");
        assert_eq!(str_tool.execute("trim:   hello   ").unwrap(), "hello");
    }

    #[test]
    fn test_string_tool_invalid_operation() {
        let tools = create_builtin_agent_tools();
        let str_tool = tools.iter().find(|t| t.name == "string_tool").unwrap();

        let result = str_tool.execute("invalid: text");
        assert!(result.is_err());
    }

    #[test]
    fn test_string_tool_invalid_format() {
        let tools = create_builtin_agent_tools();
        let str_tool = tools.iter().find(|t| t.name == "string_tool").unwrap();

        let result = str_tool.execute("no colon here");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Format"));
    }

    #[test]
    fn test_agent_result_structure() {
        let result = AgentResult {
            success: true,
            answer: Some("The answer is 42".to_string()),
            error: None,
            steps: vec![AgentStep {
                step_number: 1,
                thought: "I need to calculate".to_string(),
                action: "calculator".to_string(),
                action_input: "6 * 7".to_string(),
                observation: Some("42".to_string()),
                timestamp: 1234567890,
            }],
            execution_time_ms: 150,
            total_tokens: Some(100),
        };

        assert!(result.success);
        assert_eq!(result.answer.unwrap(), "The answer is 42");
        assert!(result.error.is_none());
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.execution_time_ms, 150);
        assert_eq!(result.total_tokens.unwrap(), 100);
    }

    #[test]
    fn test_react_agent_context_access() {
        let mut agent = ReactAgent::new(AgentConfig::default());

        // Test context_mut
        agent.context_mut().set("key", "value");
        agent.context_mut().add_observation("obs1");

        // Test context
        assert_eq!(agent.context().get("key"), Some(&"value".to_string()));
        assert_eq!(agent.context().observations.len(), 1);
    }

    #[test]
    fn test_observations_accumulate() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("tool", "Test", |input| {
            Ok(format!("Result: {}", input))
        }));

        let _ = agent.execute_step("Step 1".to_string(), "tool".to_string(), "a".to_string());
        let _ = agent.execute_step("Step 2".to_string(), "tool".to_string(), "b".to_string());

        assert_eq!(agent.context().observations.len(), 2);
        assert!(agent.context().observations[0].contains("Result: a"));
        assert!(agent.context().observations[1].contains("Result: b"));
    }

    #[test]
    fn test_planning_agent_empty() {
        let agent = PlanningAgent::new(AgentConfig::default());

        assert!(agent.plan().is_empty());
        assert!(agent.next_step().is_none());
        assert!(agent.is_complete()); // Empty plan is complete
    }

    #[test]
    fn test_react_agent_state_transitions() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("tool", "Test", |_| Ok("ok".to_string())));

        // Initial state
        assert_eq!(*agent.state(), AgentState::Idle);

        // After starting execution
        let _ = agent.execute_step("Think".to_string(), "tool".to_string(), "input".to_string());
        assert_eq!(*agent.state(), AgentState::Thinking);

        // After final answer
        let _ = agent.execute_step(
            "Done".to_string(),
            "final_answer".to_string(),
            "answer".to_string(),
        );
        assert_eq!(*agent.state(), AgentState::Completed);
    }

    #[test]
    fn test_current_step_increments() {
        let mut agent = ReactAgent::new(AgentConfig::default());
        agent.add_tool(AgentTool::new("tool", "Test", |_| Ok("ok".to_string())));

        assert_eq!(agent.context().current_step, 0);

        let _ = agent.execute_step("1".to_string(), "tool".to_string(), "".to_string());
        assert_eq!(agent.context().current_step, 1);

        let _ = agent.execute_step("2".to_string(), "tool".to_string(), "".to_string());
        assert_eq!(agent.context().current_step, 2);
    }
}
